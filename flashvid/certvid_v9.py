"""CertVID V9: V3-preserving state completion and trustworthy fusion."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from .certvid import CertVidPlan, _cfg_float, _cfg_int, apply_certvid_plan
from .certvid_v3 import certvid_v3_compression


@dataclass
class _V3Analysis:
    metric_flat: torch.Tensor
    design: torch.Tensor
    demand_weight: torch.Tensor
    attention: torch.Tensor
    query_score: torch.Tensor
    query_relevance: torch.Tensor
    query_confidence: float
    component_ids: torch.Tensor
    frame_ids: torch.Tensor
    temporal_ids: torch.Tensor
    candidate_indices: torch.Tensor
    ridge: float


@dataclass(frozen=True)
class _RepairCandidate:
    token: int
    priority: int
    provenance: str
    score: float


_RELATION_PATTERNS = (
    r"\bbefore\b",
    r"\bafter\b",
    r"\blater\b",
    r"\bearlier\b",
    r"\bfirst\b",
    r"\bfinally\b",
    r"\bthen\b",
    r"\bsubsequently\b",
    r"\bchange(?:d|s|ing)?\b",
    r"\bdifferent\b",
    r"\bbeginning\b",
    r"\bend of\b",
    r"\border\b",
    r"\bsequence\b",
    r"\breappear",
)


def _cfg_bool(config: Any, name: str, default: bool) -> bool:
    value = getattr(config, name, default)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


def _analysis_from_sink(sink: Dict[str, Any]) -> _V3Analysis:
    names = (
        "metric_flat",
        "design",
        "demand_weight",
        "attention",
        "query_score",
        "query_relevance",
        "query_confidence",
        "component_ids",
        "frame_ids",
        "temporal_ids",
        "candidate_indices",
        "ridge",
    )
    missing = [name for name in names if name not in sink]
    if missing:
        raise ValueError(f"V3 analysis is missing: {', '.join(missing)}")
    return _V3Analysis(**{name: sink[name] for name in names})


def _frame_times(
    config: Any,
    frame_count: int,
    device: torch.device,
) -> tuple[torch.Tensor, bool, str]:
    raw = getattr(config, "_certvid_frame_times_sec", None)
    source = str(getattr(config, "_certvid_frame_times_source", "missing"))
    if raw is None:
        return torch.arange(frame_count, device=device).float(), False, "frame_index"
    times = torch.as_tensor(raw, dtype=torch.float32, device=device).flatten()
    valid = (
        times.numel() == frame_count
        and bool(torch.isfinite(times).all())
        and (times.numel() <= 1 or bool(torch.all(times[1:] > times[:-1])))
    )
    if not valid:
        return torch.arange(frame_count, device=device).float(), False, "frame_index"
    return times, True, source


def _event_segments(
    metric_flat: torch.Tensor,
    frame_times: torch.Tensor,
    has_real_times: bool,
    frame_count: int,
    tokens_per_frame: int,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    frames = metric_flat.view(frame_count, tokens_per_frame, -1)
    frame_mean = F.normalize(frames.mean(dim=1), p=2, dim=-1, eps=1e-6)
    semantic_gap = torch.zeros(frame_count, dtype=torch.float32, device=metric_flat.device)
    if frame_count > 1:
        semantic_gap[1:] = (
            1.0 - torch.sum(frame_mean[1:] * frame_mean[:-1], dim=-1)
        ).clamp_min(0.0)
    positive = semantic_gap[1:]
    quantile = min(1.0, max(0.0, _cfg_float(config, "certv9_event_quantile", 0.85)))
    floor = max(0.0, _cfg_float(config, "certv9_event_floor", 0.10))
    threshold = (
        max(floor, float(torch.quantile(positive, quantile).item()))
        if positive.numel() > 0
        else floor
    )
    boundary_score = semantic_gap.clone()
    boundary = semantic_gap >= threshold
    boundary[0] = False

    if has_real_times and frame_count > 1:
        time_gap = frame_times[1:] - frame_times[:-1]
        median_gap = float(time_gap.median().item())
        time_threshold = max(
            _cfg_float(config, "certv9_cross_segment_max_seconds", 8.0),
            2.5 * median_gap,
        )
        time_boundary = time_gap > time_threshold
        boundary[1:] |= time_boundary
        boundary_score[1:] += time_boundary.float()

    boundary_indices = torch.where(boundary)[0]
    if boundary_indices.numel() > 11:
        order = torch.argsort(
            boundary_score[boundary_indices],
            descending=True,
            stable=True,
        )[:11]
        keep = boundary_indices[order]
        boundary.zero_()
        boundary[keep] = True

    frame_segments = torch.cumsum(boundary.long(), dim=0)
    token_segments = frame_segments.repeat_interleave(tokens_per_frame)
    return frame_segments, token_segments, semantic_gap, threshold


def _coverage_state(
    metric_flat: torch.Tensor,
    selected: torch.Tensor,
    token_segments: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_similarity = metric_flat.float() @ metric_flat[selected].float().transpose(0, 1)
    valid = token_segments.unsqueeze(1) == token_segments[selected].unsqueeze(0)
    similarity = raw_similarity.masked_fill(~valid, -2.0)
    topk = min(2, int(selected.numel()))
    values, owners = torch.topk(similarity, k=topk, dim=1, largest=True)
    if topk == 1:
        values = torch.cat([values, values], dim=1)
        owners = torch.cat([owners, owners], dim=1)
    return values[:, 0], owners[:, 0], values[:, 1], raw_similarity


def _state_pair_candidates(
    analysis: _V3Analysis,
    selected: torch.Tensor,
    risk: torch.Tensor,
    config: Any,
) -> tuple[list[_RepairCandidate], list[dict[str, Any]]]:
    if not _cfg_bool(config, "certv9_state_pair_enabled", True):
        return [], []
    threshold = max(0.0, _cfg_float(config, "certv9_state_distance_threshold", 0.15))
    min_span = max(1, _cfg_int(config, "certv9_state_min_bin_span", 2))
    selected_mask = torch.zeros(
        analysis.metric_flat.shape[0],
        dtype=torch.bool,
        device=selected.device,
    )
    selected_mask[selected] = True
    candidates: list[_RepairCandidate] = []
    records: list[dict[str, Any]] = []

    for component in torch.unique(analysis.component_ids).detach().cpu().tolist():
        members = torch.where(analysis.component_ids == int(component))[0]
        if members.numel() < 2:
            continue
        temporal = analysis.temporal_ids[members]
        span = int((temporal.max() - temporal.min()).item())
        if span < min_span:
            continue
        anchors = members[selected_mask[members]]
        covered_pair = False
        if anchors.numel() >= 2:
            pair_similarity = (
                analysis.metric_flat[anchors].float()
                @ analysis.metric_flat[anchors].float().transpose(0, 1)
            )
            covered_pair = float((1.0 - pair_similarity).max().item()) >= threshold
        if covered_pair:
            continue

        if anchors.numel() > 0:
            similarity = (
                analysis.metric_flat[members].float()
                @ analysis.metric_flat[anchors].float().transpose(0, 1)
            ).amax(dim=1)
            state_distance = 1.0 - similarity
        else:
            center = F.normalize(
                analysis.metric_flat[members].float().mean(dim=0),
                p=2,
                dim=0,
                eps=1e-6,
            )
            state_distance = 1.0 - analysis.metric_flat[members].float() @ center
        state_distance[selected_mask[members]] = -1.0
        score = state_distance + 0.25 * risk[members]
        local = int(torch.argmax(score).item())
        token = int(members[local].item())
        distance = float(state_distance[local].item())
        if distance < threshold:
            continue
        candidates.append(
            _RepairCandidate(
                token=token,
                priority=2,
                provenance="state_endpoint",
                score=distance + float(risk[token].item()),
            )
        )
        records.append(
            {
                "component": int(component),
                "token": token,
                "temporal_span": span,
                "state_distance": distance,
            }
        )
    return candidates, records


def _state_pair_protected_anchors(
    analysis: _V3Analysis,
    selected: torch.Tensor,
    config: Any,
) -> set[int]:
    if not _cfg_bool(config, "certv9_state_pair_enabled", True):
        return set()
    threshold = max(0.0, _cfg_float(config, "certv9_state_distance_threshold", 0.15))
    min_span = max(1, _cfg_int(config, "certv9_state_min_bin_span", 2))
    selected_mask = torch.zeros(
        analysis.metric_flat.shape[0],
        dtype=torch.bool,
        device=selected.device,
    )
    selected_mask[selected] = True
    protected: set[int] = set()
    for component in torch.unique(analysis.component_ids).detach().cpu().tolist():
        members = torch.where(analysis.component_ids == int(component))[0]
        if members.numel() < 2:
            continue
        temporal = analysis.temporal_ids[members]
        if int((temporal.max() - temporal.min()).item()) < min_span:
            continue
        anchors = members[selected_mask[members]]
        if anchors.numel() < 2:
            continue
        pair_distance = 1.0 - (
            analysis.metric_flat[anchors].float()
            @ analysis.metric_flat[anchors].float().transpose(0, 1)
        )
        flat_index = int(torch.argmax(pair_distance).item())
        distance = float(pair_distance.flatten()[flat_index].item())
        if distance < threshold:
            continue
        row = flat_index // int(anchors.numel())
        column = flat_index % int(anchors.numel())
        protected.add(int(anchors[row].item()))
        protected.add(int(anchors[column].item()))
    return protected


def _query_peak_candidates(
    analysis: _V3Analysis,
    selected: torch.Tensor,
    coverage: torch.Tensor,
    frame_count: int,
    tokens_per_frame: int,
    config: Any,
) -> tuple[list[_RepairCandidate], list[dict[str, Any]], bool]:
    if (
        not _cfg_bool(config, "certv9_multi_peak_enabled", True)
        or analysis.query_relevance.numel() == 0
    ):
        return [], [], False
    text = str(getattr(config, "_certvid_query_text", "") or "").lower()
    relational = any(re.search(pattern, text) is not None for pattern in _RELATION_PATTERNS)
    confidence_threshold = _cfg_float(config, "certv3_query_threshold", 0.10)
    if analysis.query_confidence < confidence_threshold:
        return [], [], relational

    max_peaks = max(1, min(3, _cfg_int(config, "certv9_query_max_peaks", 3)))
    target_peaks = min(max_peaks, 2 if relational else 1)
    separation = max(1, _cfg_int(config, "certv9_query_peak_separation", 2))
    temporal_per_frame = analysis.temporal_ids.view(frame_count, tokens_per_frame)[:, 0]
    selected_mask = torch.zeros(
        analysis.metric_flat.shape[0],
        dtype=torch.bool,
        device=selected.device,
    )
    selected_mask[selected] = True
    candidates: list[_RepairCandidate] = []
    records: list[dict[str, Any]] = []
    top_count = max(1, int(math.ceil(tokens_per_frame * 0.05)))

    for atom_index, atom_scores in enumerate(analysis.query_relevance):
        frame_scores = atom_scores.view(frame_count, tokens_per_frame)
        curve = torch.topk(frame_scores, k=top_count, dim=1).values.mean(dim=1)
        median = curve.median()
        spread = curve.std(unbiased=False)
        significant = median + 0.50 * spread
        chosen_frames: list[int] = []
        for frame in torch.argsort(curve, descending=True, stable=True).detach().cpu().tolist():
            frame = int(frame)
            if float(curve[frame].item()) < float(significant.item()) and chosen_frames:
                continue
            temporal = int(temporal_per_frame[frame].item())
            if any(
                abs(temporal - int(temporal_per_frame[other].item())) < separation
                for other in chosen_frames
            ):
                continue
            chosen_frames.append(frame)
            if len(chosen_frames) >= target_peaks:
                break
        if relational and max_peaks >= 3 and len(chosen_frames) >= 2:
            for frame in torch.argsort(curve, descending=True, stable=True).detach().cpu().tolist():
                frame = int(frame)
                temporal = int(temporal_per_frame[frame].item())
                if float(curve[frame].item()) < float(significant.item()):
                    continue
                if all(
                    abs(temporal - int(temporal_per_frame[other].item())) >= separation
                    for other in chosen_frames
                ):
                    chosen_frames.append(frame)
                    break

        for peak_rank, frame in enumerate(chosen_frames):
            members = torch.arange(
                frame * tokens_per_frame,
                (frame + 1) * tokens_per_frame,
                device=selected.device,
            )
            token = int(members[torch.argmax(atom_scores[members])].item())
            covered = bool(selected_mask[token]) or float(coverage[token].item()) >= _cfg_float(
                config,
                "certv9_merge_threshold",
                0.80,
            )
            records.append(
                {
                    "atom": int(atom_index),
                    "rank": int(peak_rank),
                    "frame": int(frame),
                    "token": token,
                    "score": float(curve[frame].item()),
                    "covered": covered,
                }
            )
            if not covered:
                candidates.append(
                    _RepairCandidate(
                        token=token,
                        priority=3,
                        provenance="query_peak",
                        score=float(curve[frame].item()),
                    )
                )
    return candidates, records, relational


def _logdet_and_inverse(
    design: torch.Tensor,
    selected: torch.Tensor,
    ridge: float,
) -> tuple[torch.Tensor, float]:
    rows = design[selected].float()
    dimension = int(rows.shape[1])
    identity = torch.eye(dimension, dtype=torch.float32, device=rows.device)
    information = max(1e-4, float(ridge)) * identity + rows.transpose(0, 1) @ rows
    sign, logabsdet = torch.linalg.slogdet(information)
    if float(sign.item()) <= 0.0:
        raise RuntimeError("V9 information matrix is not positive definite")
    return torch.linalg.inv(information), float(logabsdet.item())


def _d_efficiency(
    design: torch.Tensor,
    selected: torch.Tensor,
    ridge: float,
    base_logdet: float,
) -> float:
    _, logdet = _logdet_and_inverse(design, selected, ridge)
    dimension = max(1, int(design.shape[1]))
    exponent = max(-50.0, min(50.0, (logdet - base_logdet) / dimension))
    return float(math.exp(exponent))


def _batch_repair(
    analysis: _V3Analysis,
    selected: torch.Tensor,
    locked: set[int],
    candidates: list[_RepairCandidate],
    token_segments: torch.Tensor,
    risk: torch.Tensor,
    coverage: torch.Tensor,
    owner: torch.Tensor,
    second: torch.Tensor,
    config: Any,
) -> tuple[torch.Tensor, list[dict[str, Any]], float]:
    if not candidates or not _cfg_bool(config, "certv9_full_pool_repair_enabled", True):
        return selected, [], 1.0
    budget = int(selected.numel())
    swap_limit = min(
        budget,
        max(0, int(math.ceil(budget * _cfg_float(config, "certv9_max_swap_ratio", 0.15)))),
    )
    if swap_limit <= 0:
        return selected, [], 1.0

    base_inverse, base_logdet = _logdet_and_inverse(
        analysis.design,
        selected,
        analysis.ridge,
    )
    selected_rows = analysis.design[selected].float()
    leverage = torch.sum((selected_rows @ base_inverse) * selected_rows, dim=1)
    removal_loss = -torch.log1p(-leverage.clamp(max=1.0 - 1e-5))
    frame_counts = torch.bincount(
        analysis.frame_ids[selected],
        minlength=int(analysis.frame_ids.max().item()) + 1,
    )
    segment_counts = torch.bincount(
        token_segments[selected],
        minlength=int(token_segments.max().item()) + 1,
    )
    removable = [
        position
        for position, token in enumerate(selected.detach().cpu().tolist())
        if int(token) not in locked
        and int(frame_counts[analysis.frame_ids[token]].item()) > 1
        and int(segment_counts[token_segments[token]].item()) > 1
    ]
    removable.sort(
        key=lambda position: (
            float(removal_loss[position].item()),
            float(risk[int(selected[position].item())].item()),
            int(selected[position].item()),
        )
    )
    removable = removable[: max(1, _cfg_int(config, "certv9_remove_pool", 64))]
    if not removable:
        return selected, [], 1.0

    by_token: dict[int, _RepairCandidate] = {}
    selected_set = set(int(token) for token in selected.detach().cpu().tolist())
    for candidate in candidates:
        if candidate.token in selected_set:
            continue
        previous = by_token.get(candidate.token)
        if previous is None or (candidate.priority, candidate.score) > (
            previous.priority,
            previous.score,
        ):
            by_token[candidate.token] = candidate
    ordered = sorted(
        by_token.values(),
        key=lambda item: (-item.priority, -item.score, item.token),
    )
    ordered = ordered[: max(1, _cfg_int(config, "certv9_repair_pool", 128))]

    min_gain = _cfg_float(config, "certv9_min_objective_gain", 1e-4)
    efficiency_floor = _cfg_float(config, "certv9_d_efficiency_floor", 0.98)
    used_removals: set[int] = set()
    proposals: list[dict[str, Any]] = []
    metric = analysis.metric_flat.float()

    for candidate in ordered:
        if len(proposals) >= swap_limit:
            break
        token = int(candidate.token)
        add_row = analysis.design[token].float()
        add_similarity = metric @ metric[token]
        add_similarity = add_similarity.masked_fill(
            token_segments != token_segments[token],
            -2.0,
        )
        base_add_leverage = torch.dot(add_row, base_inverse @ add_row)
        best: Optional[dict[str, Any]] = None

        for position in removable:
            remove_token = int(selected[position].item())
            if remove_token in used_removals:
                continue
            frame = int(analysis.frame_ids[remove_token].item())
            segment = int(token_segments[remove_token].item())
            if int(frame_counts[frame].item()) <= 1 or int(segment_counts[segment].item()) <= 1:
                continue

            without = torch.where(owner == position, second, coverage)
            final_coverage = torch.maximum(without, add_similarity)
            gain = float(
                torch.sum(analysis.demand_weight * (final_coverage - coverage)).item()
            )
            if gain < min_gain:
                continue

            removed = selected_rows[position]
            direction = base_inverse @ removed
            denominator = 1.0 - torch.dot(removed, direction)
            if float(denominator.item()) <= 1e-5:
                continue
            cross = torch.dot(add_row, direction)
            add_leverage = base_add_leverage + cross.square() / denominator
            delta = torch.log(denominator) + torch.log1p(add_leverage.clamp_min(0.0))
            one_swap_efficiency = math.exp(
                max(
                    -50.0,
                    min(50.0, float(delta.item()) / max(1, int(analysis.design.shape[1]))),
                )
            )
            if one_swap_efficiency < efficiency_floor:
                continue
            score = gain + 1e-3 * candidate.priority - 1e-4 * float(removal_loss[position].item())
            record = {
                "add": token,
                "remove": remove_token,
                "position": int(position),
                "provenance": candidate.provenance,
                "priority": int(candidate.priority),
                "coverage_gain": gain,
                "one_swap_d_efficiency": one_swap_efficiency,
                "score": score,
            }
            if best is None or (score, -remove_token) > (best["score"], -best["remove"]):
                best = record

        if best is None:
            continue
        position = int(best["position"])
        remove_token = int(best["remove"])
        used_removals.add(remove_token)
        frame_counts[analysis.frame_ids[remove_token]] -= 1
        segment_counts[token_segments[remove_token]] -= 1
        frame_counts[analysis.frame_ids[token]] += 1
        segment_counts[token_segments[token]] += 1
        proposals.append(best)

    if not proposals:
        return selected, [], 1.0

    # Validate the joint design, then trim the weakest proposals until it is safe.
    accepted = list(proposals)
    final = selected.clone()
    while accepted:
        final = selected.clone()
        for proposal in accepted:
            final[int(proposal["position"])] = int(proposal["add"])
        final = torch.sort(final).values
        efficiency = _d_efficiency(
            analysis.design,
            final,
            analysis.ridge,
            base_logdet,
        )
        if efficiency >= efficiency_floor:
            break
        accepted.pop()
    if not accepted:
        return selected, [], 1.0
    return final, accepted, efficiency


def _build_trustworthy_plan(
    selected: torch.Tensor,
    v3_indices: torch.Tensor,
    v3_plan: CertVidPlan,
    promoted: set[int],
    locked: set[int],
    analysis: _V3Analysis,
    token_segments: torch.Tensor,
    frame_times: torch.Tensor,
    has_real_times: bool,
    config: Any,
) -> tuple[CertVidPlan, dict[str, Any]]:
    total_tokens = int(analysis.metric_flat.shape[0])
    budget = int(selected.numel())
    metric = analysis.metric_flat.float()
    raw_similarity = metric @ metric[selected].transpose(0, 1)
    source_segment = token_segments.unsqueeze(1)
    anchor_segment = token_segments[selected].unsqueeze(0)
    same_segment = source_segment == anchor_segment
    adjacent_segment = (source_segment - anchor_segment).abs() == 1
    cross_similarity = _cfg_float(config, "certv9_cross_segment_similarity", 0.92)
    cross_valid = adjacent_segment & (raw_similarity >= cross_similarity)
    if has_real_times:
        source_frame = analysis.frame_ids.unsqueeze(1)
        anchor_frame = analysis.frame_ids[selected].unsqueeze(0)
        time_gap = torch.abs(frame_times[source_frame] - frame_times[anchor_frame])
        cross_valid &= time_gap <= _cfg_float(
            config,
            "certv9_cross_segment_max_seconds",
            8.0,
        )
    if _cfg_bool(config, "certv9_event_mask_enabled", True):
        valid = same_segment | cross_valid
    else:
        valid = (
            analysis.temporal_ids.unsqueeze(1)
            - analysis.temporal_ids[selected].unsqueeze(0)
        ).abs() <= 1
    if not bool(valid.any(dim=1).all()):
        raise RuntimeError("V9 plan has a token without a reachable anchor")

    score = raw_similarity.masked_fill(~valid, -2.0)
    same_component = (
        analysis.component_ids.unsqueeze(1)
        == analysis.component_ids[selected].unsqueeze(0)
    )
    score += 0.08 * same_component.float()
    topk = min(2, budget)
    values, assignment = torch.topk(score, k=topk, dim=1, largest=True)
    chosen_valid = torch.gather(valid, 1, assignment)
    raw_values = torch.gather(raw_similarity, 1, assignment)
    temperature = max(1e-4, _cfg_float(config, "certv3_assignment_temperature", 0.07))
    weights = torch.softmax(values.float() / temperature, dim=1) * chosen_valid.float()
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

    threshold = _cfg_float(config, "certv9_merge_threshold", 0.80)
    best_legal = raw_similarity.masked_fill(~valid, -2.0).amax(dim=1)
    rejected = best_legal < threshold
    if _cfg_bool(config, "certv9_merge_rejection_enabled", True):
        weights[rejected] = 0.0

    anchor_positions = torch.arange(budget, dtype=torch.long, device=selected.device)
    assignment[selected, 0] = anchor_positions
    weights[selected] = 0.0
    weights[selected, 0] = 1.0
    rejected[selected] = False

    source_mass = (0.5 + 0.5 * analysis.demand_weight * total_tokens).clamp(0.25, 2.0)
    alpha = torch.zeros(budget, dtype=torch.float32, device=selected.device)
    v3_position = {
        int(token): position
        for position, token in enumerate(v3_indices.detach().cpu().tolist())
    }
    for position, token in enumerate(selected.detach().cpu().tolist()):
        old_position = v3_position.get(int(token))
        if old_position is not None:
            alpha[position] = v3_plan.fusion_alpha[old_position]

    target_mass = torch.zeros(budget, dtype=torch.float32, device=selected.device)
    target_similarity = torch.zeros(budget, dtype=torch.float32, device=selected.device)
    for neighbor in range(topk):
        mass = weights[:, neighbor] * source_mass
        target = assignment[:, neighbor]
        target_mass.index_add_(0, target, mass)
        target_similarity.index_add_(0, target, mass * raw_values[:, neighbor])
    mean_similarity = target_similarity / target_mass.clamp_min(1e-6)
    confidence = ((mean_similarity - threshold) / max(1e-6, 1.0 - threshold)).clamp(0.0, 1.0)
    alpha *= confidence

    protected = locked | promoted
    if protected:
        protected_tensor = torch.tensor(
            sorted(protected),
            dtype=torch.long,
            device=selected.device,
        )
        alpha[torch.isin(selected, protected_tensor)] = 0.0

    first_assignment = assignment[:, 0]
    cross_segment = (
        token_segments != token_segments[selected[first_assignment]]
    ) & (weights[:, 0] > 0.0)
    residual_mask = torch.ones(total_tokens, dtype=torch.bool, device=selected.device)
    residual_mask[selected] = False
    residual_best = best_legal[residual_mask]
    if residual_best.numel() > 0:
        quantiles = torch.quantile(
            residual_best.float(),
            torch.tensor([0.0, 0.05, 0.50], device=selected.device),
        )
        similarity_summary = {
            "min": float(quantiles[0].item()),
            "p05": float(quantiles[1].item()),
            "median": float(quantiles[2].item()),
        }
    else:
        similarity_summary = {"min": 1.0, "p05": 1.0, "median": 1.0}

    plan = CertVidPlan(
        anchor_indices=selected,
        assignment_indices=assignment,
        assignment_weights=weights,
        source_mass=source_mass,
        fusion_alpha=alpha,
        raw_token_count=total_tokens,
    )
    stats = {
        "assignment_similarity": similarity_summary,
        "low_similarity_assignment_count": int((residual_best < threshold).sum().item()),
        "low_similarity_assignment_mass": float(
            analysis.demand_weight[
                residual_mask & (best_legal < threshold)
            ].sum().item()
        ),
        "rejected_residual_count": int((rejected & residual_mask).sum().item()),
        "rejected_residual_mass": float(
            analysis.demand_weight[rejected & residual_mask].sum().item()
        ),
        "cross_segment_assignment_rate": float(
            cross_segment[residual_mask].float().mean().item()
        )
        if bool(residual_mask.any())
        else 0.0,
    }
    return plan, stats


def _temporal_entropy(counts: torch.Tensor) -> float:
    probability = counts.float() / counts.sum().clamp_min(1.0)
    nonzero = probability > 0
    if int(nonzero.sum().item()) <= 1:
        return 0.0
    entropy = -(probability[nonzero] * probability[nonzero].log()).sum()
    return float((entropy / math.log(int(nonzero.sum().item()))).item())


def _json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _store_diagnostics(config: Any, diagnostics: Dict[str, Any]) -> None:
    config.last_certv9_diagnostics = diagnostics
    config.last_certv9_fallback_reason = diagnostics.get("fallback_reason")
    config.last_certv9_swap_count = int(diagnostics.get("swap_count", 0))
    config.last_certv9_v3_overlap_ratio = float(
        diagnostics.get("v3_overlap_ratio", 1.0)
    )
    config.last_certv9_d_efficiency = float(diagnostics.get("d_efficiency", 1.0))
    template = os.environ.get("CERTV9_DIAGNOSTICS_JSONL", "").strip()
    if template:
        rank = os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))
        path = template.replace("{rank}", rank).replace("{pid}", str(os.getpid()))
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        record = dict(diagnostics)
        record["sample_id"] = str(getattr(config, "_debug_sample_id", "unknown"))
        record["question"] = str(getattr(config, "_certvid_query_text", "") or "")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")
    if _cfg_bool(config, "certv9_debug", False):
        print(
            "[certvid-v9] "
            f"sample={getattr(config, '_debug_sample_id', 'unknown')} "
            f"fallback={diagnostics.get('fallback_reason')} "
            f"gate={diagnostics.get('gate_reasons', [])} "
            f"swaps={diagnostics.get('swap_count', 0)} "
            f"rejected={diagnostics.get('rejected_residual_count', 0)} "
            f"D-eff={diagnostics.get('d_efficiency', 1.0):.4f}"
        )


def _fallback(
    config: Any,
    diagnostics: Dict[str, Any],
    reason: str,
    output: torch.Tensor,
    indices: torch.Tensor,
    plan: CertVidPlan,
) -> tuple[torch.Tensor, torch.Tensor]:
    diagnostics["fallback_reason"] = reason
    diagnostics.setdefault("swap_count", 0)
    diagnostics.setdefault("v3_overlap_ratio", 1.0)
    diagnostics.setdefault("d_efficiency", 1.0)
    config._certvid_plan = plan
    config.vision_token_length = int(output.shape[0])
    config.visual_token_length = int(output.shape[0])
    config.llm_token_length = None
    config.last_adapter_variant = "certvid_v9"
    _store_diagnostics(config, diagnostics)
    return output, indices


def certvid_v9_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    config: Any,
    question_features: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repair V3 state/query deficits without changing its fixed token budget."""
    sink: Dict[str, Any] = {}
    v3_output, v3_indices = certvid_v3_compression(
        video_features,
        cls_attention,
        config,
        question_features,
        analysis_sink=sink,
    )
    v3_plan = getattr(config, "_certvid_plan", None)
    if v3_plan is None:
        raise RuntimeError("CertVID V3 did not publish an aggregation plan")

    frame_count, tokens_per_frame, _ = video_features.shape
    diagnostics: Dict[str, Any] = {
        "fallback_reason": None,
        "raw_token_count": int(frame_count * tokens_per_frame),
        "budget": int(v3_indices.numel()),
        "gate_reasons": [],
        "swap_count": 0,
        "v3_overlap_ratio": 1.0,
        "d_efficiency": 1.0,
    }
    if not _cfg_bool(config, "certv9_enabled", True):
        return _fallback(config, diagnostics, "disabled", v3_output, v3_indices, v3_plan)
    if sink.get("identity", False):
        return _fallback(config, diagnostics, "identity_budget", v3_output, v3_indices, v3_plan)

    try:
        analysis = _analysis_from_sink(sink)
        frame_times, has_real_times, timestamp_source = _frame_times(
            config,
            frame_count,
            video_features.device,
        )
        frame_segments, token_segments, semantic_gap, event_threshold = _event_segments(
            analysis.metric_flat,
            frame_times,
            has_real_times,
            frame_count,
            tokens_per_frame,
            config,
        )
        coverage, owner, second, _ = _coverage_state(
            analysis.metric_flat,
            v3_indices,
            token_segments,
        )
        selected_mask = torch.zeros(
            analysis.metric_flat.shape[0],
            dtype=torch.bool,
            device=v3_indices.device,
        )
        selected_mask[v3_indices] = True
        state_score = torch.zeros_like(coverage)
        risk = analysis.demand_weight * (1.0 - coverage).clamp_min(0.0) * (
            1.0 + 0.35 * analysis.query_score + 0.45 * state_score
        )

        state_candidates, state_records = _state_pair_candidates(
            analysis,
            v3_indices,
            risk,
            config,
        )
        state_distance_by_token = {
            int(record["token"]): float(record["state_distance"])
            for record in state_records
        }
        for candidate in state_candidates:
            state_score[candidate.token] = torch.maximum(
                state_score[candidate.token],
                state_score.new_tensor(state_distance_by_token[candidate.token]),
            )
        risk = analysis.demand_weight * (1.0 - coverage).clamp_min(0.0) * (
            1.0 + 0.35 * analysis.query_score + 0.45 * state_score.clamp(0.0, 1.0)
        )
        query_candidates, query_records, relational = _query_peak_candidates(
            analysis,
            v3_indices,
            coverage,
            frame_count,
            tokens_per_frame,
            config,
        )
        threshold = _cfg_float(config, "certv9_merge_threshold", 0.80)
        unsafe = (~selected_mask) & (coverage < threshold)
        uncovered_mass = float(analysis.demand_weight[unsafe].sum().item())
        gate_reasons: list[str] = []
        if uncovered_mass >= _cfg_float(
            config,
            "certv9_uncovered_mass_threshold",
            0.05,
        ):
            gate_reasons.append("uncovered_mass")
        if state_candidates:
            gate_reasons.append("state_pair")
        if relational and query_candidates:
            gate_reasons.append("query_peak")

        v3_frame_counts = torch.bincount(
            analysis.frame_ids[v3_indices],
            minlength=frame_count,
        )
        segment_count = int(frame_segments.max().item()) + 1
        v3_segment_counts = torch.bincount(
            token_segments[v3_indices],
            minlength=segment_count,
        )
        diagnostics.update(
            {
                "timestamp_source": timestamp_source,
                "has_real_timestamps": has_real_times,
                "event_threshold": event_threshold,
                "semantic_gap": semantic_gap,
                "segment_count": segment_count,
                "frame_segments": frame_segments,
                "gate_reasons": gate_reasons,
                "gate_triggered": bool(gate_reasons),
                "uncovered_mass": uncovered_mass,
                "state_pair_deficits_before": state_records,
                "state_pair_deficit_count_before": len(state_candidates),
                "query_peaks_before": query_records,
                "query_peaks_detected_before": len(query_records),
                "query_peaks_covered_before": sum(
                    bool(record["covered"]) for record in query_records
                ),
                "query_is_relational": relational,
                "v3_frame_counts": v3_frame_counts,
                "v3_segment_counts": v3_segment_counts,
                "v3_temporal_entropy": _temporal_entropy(v3_frame_counts),
            }
        )
        if not gate_reasons:
            return _fallback(config, diagnostics, "risk_gate_closed", v3_output, v3_indices, v3_plan)

        generic_candidates: list[_RepairCandidate] = []
        if _cfg_bool(config, "certv9_full_pool_repair_enabled", True):
            order = torch.argsort(risk, descending=True, stable=True)
            generic_limit = max(
                64,
                4
                * int(
                    math.ceil(
                        int(v3_indices.numel())
                        * _cfg_float(config, "certv9_max_swap_ratio", 0.15)
                    )
                ),
            )
            for token in order.detach().cpu().tolist():
                token = int(token)
                if selected_mask[token]:
                    continue
                generic_candidates.append(
                    _RepairCandidate(
                        token=token,
                        priority=1,
                        provenance="full_pool_risk",
                        score=float(risk[token].item()),
                    )
                )
                if len(generic_candidates) >= generic_limit:
                    break

        locked = {
            int(token)
            for token in v3_indices[
                v3_plan.fusion_alpha <= 1e-12
            ].detach().cpu().tolist()
        }
        locked.update(_state_pair_protected_anchors(analysis, v3_indices, config))
        repaired, swaps, efficiency = _batch_repair(
            analysis,
            v3_indices,
            locked,
            query_candidates + state_candidates + generic_candidates,
            token_segments,
            risk,
            coverage,
            owner,
            second,
            config,
        )
        promoted = {int(record["add"]) for record in swaps}
        must_rebuild = bool(swaps) or (
            "uncovered_mass" in gate_reasons
            and _cfg_bool(config, "certv9_merge_rejection_enabled", True)
        )
        if not must_rebuild:
            return _fallback(config, diagnostics, "no_safe_repair", v3_output, v3_indices, v3_plan)

        final_coverage, _, _, _ = _coverage_state(
            analysis.metric_flat,
            repaired,
            token_segments,
        )
        final_coverage_gain = float(
            torch.sum(
                analysis.demand_weight * (final_coverage - coverage)
            ).item()
        )
        if swaps and final_coverage_gain < _cfg_float(
            config,
            "certv9_min_objective_gain",
            1e-4,
        ):
            diagnostics["joint_coverage_gain"] = final_coverage_gain
            return _fallback(
                config,
                diagnostics,
                "joint_coverage_not_improved",
                v3_output,
                v3_indices,
                v3_plan,
            )

        final_state_candidates, final_state_records = _state_pair_candidates(
            analysis,
            repaired,
            risk,
            config,
        )
        _, final_query_records, _ = _query_peak_candidates(
            analysis,
            repaired,
            final_coverage,
            frame_count,
            tokens_per_frame,
            config,
        )
        plan, plan_stats = _build_trustworthy_plan(
            repaired,
            v3_indices,
            v3_plan,
            promoted,
            locked,
            analysis,
            token_segments,
            frame_times,
            has_real_times,
            config,
        )
        output = apply_certvid_plan(video_features.reshape(-1, video_features.shape[-1]), plan)
        if output.shape[0] != v3_output.shape[0] or not bool(torch.isfinite(output).all()):
            raise RuntimeError("V9 output failed budget or finite-value validation")
        final_frame_counts = torch.bincount(
            analysis.frame_ids[repaired],
            minlength=frame_count,
        )
        final_segment_counts = torch.bincount(
            token_segments[repaired],
            minlength=segment_count,
        )
        candidate_set = set(
            int(token) for token in analysis.candidate_indices.detach().cpu().tolist()
        )
        added_records = []
        for record in swaps:
            item = dict(record)
            token = int(item["add"])
            removed = int(item["remove"])
            item.update(
                {
                    "add_frame": int(analysis.frame_ids[token].item()),
                    "add_segment": int(token_segments[token].item()),
                    "add_component": int(analysis.component_ids[token].item()),
                    "add_risk": float(risk[token].item()),
                    "remove_frame": int(analysis.frame_ids[removed].item()),
                    "remove_segment": int(token_segments[removed].item()),
                    "remove_component": int(analysis.component_ids[removed].item()),
                    "remove_risk": float(risk[removed].item()),
                    "outside_v3_candidate_pool": token not in candidate_set,
                }
            )
            added_records.append(item)
        diagnostics.update(
            {
                "swap_count": len(swaps),
                "swaps": added_records,
                "promoted_outside_v3_candidate_pool": sum(
                    bool(record["outside_v3_candidate_pool"]) for record in added_records
                ),
                "v3_overlap_ratio": float(
                    torch.isin(repaired, v3_indices).float().mean().item()
                ),
                "d_efficiency": efficiency,
                "joint_coverage_gain": final_coverage_gain,
                "state_pair_deficits_after": final_state_records,
                "state_pair_deficit_count_after": len(final_state_candidates),
                "query_peaks_after": final_query_records,
                "query_peaks_detected_after": len(final_query_records),
                "query_peaks_covered_after": sum(
                    bool(record["covered"]) for record in final_query_records
                ),
                "final_frame_counts": final_frame_counts,
                "final_segment_counts": final_segment_counts,
                "final_temporal_entropy": _temporal_entropy(final_frame_counts),
                **plan_stats,
            }
        )
        config._certvid_plan = plan
        config.vision_token_length = int(output.shape[0])
        config.visual_token_length = int(output.shape[0])
        config.llm_token_length = None
        config.last_adapter_variant = "certvid_v9"
        config.last_adapter_raw_tokens = float(frame_count * tokens_per_frame)
        config.last_adapter_output_tokens = float(output.shape[0])
        _store_diagnostics(config, diagnostics)
        return output, repaired
    except (RuntimeError, ValueError, IndexError) as error:
        diagnostics["error"] = f"{type(error).__name__}: {error}"
        return _fallback(config, diagnostics, "repair_error", v3_output, v3_indices, v3_plan)
