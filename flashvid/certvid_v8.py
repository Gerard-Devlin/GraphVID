"""CertVID V8: agreement-calibrated semantic and local evidence coresets.

The unchanged V3 selector supplies a global semantic coreset. An independent
local evidence view favors temporally unpredictable and spatially distinctive
tokens under a balanced frame allocation. V8 preserves their consensus and V3
certificates, then admits complementary local evidence only when joint
coverage and V3 design efficiency remain safe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .certvid import CertVidPlan, _cfg_float, _cfg_int, _minmax, apply_certvid_plan
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
    ridge: float


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
) -> Tuple[torch.Tensor, bool, str]:
    raw = getattr(config, "_certvid_frame_times_sec", None)
    source = str(getattr(config, "_certvid_frame_times_source", "missing"))
    if raw is None:
        return (
            torch.arange(frame_count, dtype=torch.float32, device=device),
            False,
            "frame_index",
        )
    times = torch.as_tensor(raw, dtype=torch.float32, device=device).flatten()
    valid = (
        times.numel() == frame_count
        and bool(torch.isfinite(times).all())
        and (
            times.numel() <= 1
            or bool(torch.all(times[1:] > times[:-1]))
        )
    )
    if not valid:
        return (
            torch.arange(frame_count, dtype=torch.float32, device=device),
            False,
            "frame_index",
        )
    return times, True, source


def _safe_attention(
    cls_attention: torch.Tensor,
    frame_count: int,
    tokens_per_frame: int,
) -> torch.Tensor:
    attention = torch.nan_to_num(
        cls_attention.float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if attention.numel() != frame_count * tokens_per_frame:
        raise ValueError(
            "cls_attention shape mismatch: "
            f"expected {frame_count * tokens_per_frame} values, "
            f"got {attention.numel()}"
        )
    attention = attention.reshape(frame_count, tokens_per_frame)
    spread = attention.amax(dim=1, keepdim=True) - attention.amin(dim=1, keepdim=True)
    normalized = _minmax(attention, dim=1)
    return torch.where(spread > 1e-6, normalized, torch.zeros_like(normalized))


def _temporal_unpredictability(
    metric_frames: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    frame_count, tokens_per_frame, _ = metric_frames.shape
    novelty = torch.zeros(
        (frame_count, tokens_per_frame),
        dtype=torch.float32,
        device=metric_frames.device,
    )
    event = torch.zeros(frame_count, dtype=torch.float32, device=metric_frames.device)
    if frame_count <= 1:
        return novelty, event

    contributions = torch.zeros_like(novelty)
    counts = torch.zeros_like(novelty)
    frame_means = F.normalize(metric_frames.mean(dim=1), p=2, dim=-1, eps=1e-6)
    event[1:] = (1.0 - (frame_means[1:] * frame_means[:-1]).sum(dim=1)).clamp(0.0, 2.0)
    event[:-1] = torch.maximum(event[:-1], event[1:])

    for frame in range(frame_count - 1):
        similarity = metric_frames[frame] @ metric_frames[frame + 1].transpose(0, 1)
        left = (1.0 - similarity.amax(dim=1)).clamp(0.0, 2.0)
        right = (1.0 - similarity.amax(dim=0)).clamp(0.0, 2.0)
        contributions[frame] += left
        contributions[frame + 1] += right
        counts[frame] += 1.0
        counts[frame + 1] += 1.0
    novelty = contributions / counts.clamp_min(1.0)
    return _minmax(novelty, dim=1), _minmax(event, dim=0)


def _structural_scores(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    analysis: _V3Analysis,
    config: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    frame_count, tokens_per_frame, _ = video_features.shape
    metric_frames = analysis.metric_flat.view(frame_count, tokens_per_frame, -1)
    attention = _safe_attention(cls_attention, frame_count, tokens_per_frame)
    novelty, event = _temporal_unpredictability(metric_frames)
    centroid = F.normalize(metric_frames.mean(dim=1), p=2, dim=-1, eps=1e-6)
    detail = (
        1.0 - (metric_frames * centroid.unsqueeze(1)).sum(dim=-1)
    ).clamp(0.0, 2.0)
    detail = _minmax(detail, dim=1)
    query = analysis.query_score.view(frame_count, tokens_per_frame).clamp(0.0, 1.0)

    attention_weight = min(
        0.60,
        max(0.0, _cfg_float(config, "certv8_structural_attention_weight", 0.25)),
    )
    novelty_weight = min(
        0.60,
        max(0.0, _cfg_float(config, "certv8_structural_novelty_weight", 0.35)),
    )
    detail_weight = min(
        0.60,
        max(0.0, _cfg_float(config, "certv8_structural_detail_weight", 0.25)),
    )
    query_weight = min(
        0.40,
        max(0.0, _cfg_float(config, "certv8_query_weight", 0.15))
        * float(analysis.query_confidence),
    )
    event_weight = max(
        0.0,
        1.0 - attention_weight - novelty_weight - detail_weight - query_weight,
    )
    normalizer = max(
        1e-6,
        attention_weight
        + novelty_weight
        + detail_weight
        + query_weight
        + event_weight,
    )
    score = (
        attention_weight * attention
        + novelty_weight * novelty
        + detail_weight * detail
        + query_weight * query
        + event_weight * event.unsqueeze(1)
    ) / normalizer
    return _minmax(score, dim=1).reshape(-1), event


def _allocate_counts(
    budget: int,
    desired: torch.Tensor,
    lower: torch.Tensor,
    capacity: torch.Tensor,
) -> torch.Tensor:
    desired = torch.nan_to_num(desired.float(), nan=0.0, posinf=0.0, neginf=0.0)
    lower = lower.long().clamp_min(0)
    capacity = torch.maximum(capacity.long(), lower)
    if int(lower.sum().item()) > budget:
        raise RuntimeError("mandatory V8 anchors exceed the token budget")
    if int(capacity.sum().item()) < budget:
        raise RuntimeError("V8 candidate union cannot fill the token budget")

    if float(desired.sum().item()) <= 1e-8:
        desired = torch.ones_like(desired)
    target_float = desired / desired.sum() * float(budget)
    counts = lower.clone()
    remaining = budget - int(counts.sum().item())
    frame_ids = torch.arange(counts.numel(), device=counts.device)
    while remaining > 0:
        eligible = counts < capacity
        if not bool(eligible.any()):
            raise RuntimeError("V8 count allocation exhausted candidate capacity")
        priority = target_float - counts.float()
        priority = priority - frame_ids.float() * 1e-8
        priority = priority.masked_fill(~eligible, -1e9)
        frame = int(torch.argmax(priority).item())
        counts[frame] += 1
        remaining -= 1
    return counts


def _greedy_frame_selection(
    candidate_indices: torch.Tensor,
    metric_flat: torch.Tensor,
    quality: torch.Tensor,
    count: int,
    diversity_weight: float,
) -> torch.Tensor:
    if count <= 0:
        return candidate_indices[:0]
    if candidate_indices.numel() <= count:
        return torch.sort(candidate_indices).values
    candidate_indices = torch.sort(candidate_indices).values
    features = F.normalize(
        metric_flat[candidate_indices].float(),
        p=2,
        dim=-1,
        eps=1e-6,
    )
    similarity = features @ features.transpose(0, 1)
    available = torch.ones(
        candidate_indices.numel(),
        dtype=torch.bool,
        device=candidate_indices.device,
    )
    max_similarity = torch.full(
        (candidate_indices.numel(),),
        -1.0,
        dtype=torch.float32,
        device=candidate_indices.device,
    )
    selected: list[int] = []
    base = quality[candidate_indices].float()
    for _ in range(count):
        novelty = 0.5 * (1.0 - max_similarity).clamp(0.0, 2.0)
        score = (1.0 - diversity_weight) * base + diversity_weight * novelty
        score = score.masked_fill(~available, -1e9)
        local = int(torch.argmax(score).item())
        selected.append(int(candidate_indices[local].item()))
        available[local] = False
        max_similarity = torch.maximum(max_similarity, similarity[:, local])
    return torch.tensor(
        sorted(selected),
        dtype=torch.long,
        device=candidate_indices.device,
    )


def _local_evidence_coreset(
    analysis: _V3Analysis,
    local_score: torch.Tensor,
    frame_event: torch.Tensor,
    frame_count: int,
    tokens_per_frame: int,
    budget: int,
    config: Any,
) -> torch.Tensor:
    uniformity = min(
        1.0,
        max(0.0, _cfg_float(config, "certv8_local_uniformity", 0.80)),
    )
    topk = max(1, min(tokens_per_frame, int(math.ceil(0.15 * tokens_per_frame))))
    frame_quality = (
        local_score.view(frame_count, tokens_per_frame)
        .topk(topk, dim=1, largest=True)
        .values.mean(dim=1)
    )
    importance = _minmax(0.70 * frame_quality + 0.30 * frame_event, dim=0)
    desired = uniformity * torch.ones_like(importance) + (1.0 - uniformity) * (
        0.25 + importance
    )
    counts = _allocate_counts(
        budget=budget,
        desired=desired,
        lower=torch.zeros(
            frame_count,
            dtype=torch.long,
            device=importance.device,
        ),
        capacity=torch.full(
            (frame_count,),
            tokens_per_frame,
            dtype=torch.long,
            device=importance.device,
        ),
    )
    diversity_weight = min(
        0.60,
        max(0.0, _cfg_float(config, "certv8_local_diversity_weight", 0.28)),
    )
    selected = []
    for frame in range(frame_count):
        start = frame * tokens_per_frame
        candidates = torch.arange(
            start,
            start + tokens_per_frame,
            dtype=torch.long,
            device=importance.device,
        )
        selected.append(
            _greedy_frame_selection(
                candidates,
                analysis.metric_flat,
                local_score,
                int(counts[frame].item()),
                diversity_weight,
            )
        )
    return torch.sort(torch.cat(selected)).values


def _nearest_opposite_distance(
    candidates: torch.Tensor,
    opposite: torch.Tensor,
    metric_flat: torch.Tensor,
    tokens_per_frame: int,
) -> torch.Tensor:
    result = torch.ones(
        candidates.numel(),
        dtype=torch.float32,
        device=candidates.device,
    )
    if candidates.numel() == 0 or opposite.numel() == 0:
        return result
    for frame in torch.unique(candidates // tokens_per_frame).tolist():
        candidate_mask = candidates // tokens_per_frame == int(frame)
        opposite_frame = opposite[opposite // tokens_per_frame == int(frame)]
        if opposite_frame.numel() == 0:
            continue
        similarity = (
            metric_flat[candidates[candidate_mask]].float()
            @ metric_flat[opposite_frame].float().transpose(0, 1)
        )
        result[candidate_mask] = (
            0.5 * (1.0 - similarity.amax(dim=1))
        ).clamp(0.0, 1.0)
    return result


def _removal_cost(
    v3_indices: torch.Tensor,
    analysis: _V3Analysis,
    tokens_per_frame: int,
) -> torch.Tensor:
    design = _minmax(
        analysis.design[v3_indices].float().square().sum(dim=1),
        dim=0,
    )
    demand = _minmax(analysis.demand_weight[v3_indices].float(), dim=0)
    query = analysis.query_score[v3_indices].float().clamp(0.0, 1.0)
    uniqueness = torch.ones_like(demand)
    for frame in torch.unique(v3_indices // tokens_per_frame).tolist():
        mask = v3_indices // tokens_per_frame == int(frame)
        local = analysis.metric_flat[v3_indices[mask]].float()
        if local.shape[0] <= 1:
            continue
        similarity = local @ local.transpose(0, 1)
        similarity.fill_diagonal_(-1.0)
        uniqueness[mask] = (
            0.5 * (1.0 - similarity.amax(dim=1))
        ).clamp(0.0, 1.0)
    return 0.34 * design + 0.28 * demand + 0.20 * query + 0.18 * uniqueness


def _addition_utility(
    structural_only: torch.Tensor,
    v3_indices: torch.Tensor,
    analysis: _V3Analysis,
    structural_score: torch.Tensor,
    tokens_per_frame: int,
    config: Any,
) -> torch.Tensor:
    complementarity = _nearest_opposite_distance(
        structural_only,
        v3_indices,
        analysis.metric_flat,
        tokens_per_frame,
    )
    structural = structural_score[structural_only]
    demand = _minmax(analysis.demand_weight[structural_only].float(), dim=0)
    query = analysis.query_score[structural_only].float().clamp(0.0, 1.0)
    complement_weight = min(
        0.60,
        max(0.0, _cfg_float(config, "certv8_complementarity_weight", 0.30)),
    )
    structural_weight = min(
        0.60,
        max(0.0, _cfg_float(config, "certv8_local_quality_weight", 0.35)),
    )
    query_weight = min(
        0.30,
        max(0.0, _cfg_float(config, "certv8_query_weight", 0.15)),
    )
    demand_weight = max(
        0.0,
        1.0 - complement_weight - structural_weight - query_weight,
    )
    normalizer = max(
        1e-6,
        complement_weight + structural_weight + query_weight + demand_weight,
    )
    return (
        complement_weight * complementarity
        + structural_weight * structural
        + query_weight * query
        + demand_weight * demand
    ) / normalizer


def _propose_swaps(
    v3_indices: torch.Tensor,
    structural_indices: torch.Tensor,
    protected: set[int],
    target_counts: torch.Tensor,
    analysis: _V3Analysis,
    structural_score: torch.Tensor,
    frame_count: int,
    tokens_per_frame: int,
    limit: int,
    config: Any,
) -> Tuple[list[int], list[int]]:
    v3_set = set(int(value) for value in v3_indices.detach().cpu().tolist())
    structural_set = set(
        int(value) for value in structural_indices.detach().cpu().tolist()
    )
    additions_tensor = torch.tensor(
        sorted(structural_set - v3_set),
        dtype=torch.long,
        device=v3_indices.device,
    )
    removable_tensor = torch.tensor(
        sorted(v3_set - structural_set - protected),
        dtype=torch.long,
        device=v3_indices.device,
    )
    if additions_tensor.numel() == 0 or removable_tensor.numel() == 0 or limit <= 0:
        return [], []

    add_utility = _addition_utility(
        additions_tensor,
        v3_indices,
        analysis,
        structural_score,
        tokens_per_frame,
        config,
    )
    remove_cost_all = _removal_cost(v3_indices, analysis, tokens_per_frame)
    cost_by_token = {
        int(token): float(cost)
        for token, cost in zip(
            v3_indices.detach().cpu().tolist(),
            remove_cost_all.detach().cpu().tolist(),
        )
    }
    utility_by_token = {
        int(token): float(utility)
        for token, utility in zip(
            additions_tensor.detach().cpu().tolist(),
            add_utility.detach().cpu().tolist(),
        )
    }
    additions_by_frame: list[list[int]] = [[] for _ in range(frame_count)]
    removals_by_frame: list[list[int]] = [[] for _ in range(frame_count)]
    for token in additions_tensor.detach().cpu().tolist():
        additions_by_frame[int(token) // tokens_per_frame].append(int(token))
    for token in removable_tensor.detach().cpu().tolist():
        removals_by_frame[int(token) // tokens_per_frame].append(int(token))
    for frame in range(frame_count):
        additions_by_frame[frame].sort(
            key=lambda token: (-utility_by_token[token], token)
        )
        removals_by_frame[frame].sort(
            key=lambda token: (cost_by_token[token], token)
        )

    counts = torch.bincount(
        v3_indices // tokens_per_frame,
        minlength=frame_count,
    ).clone()
    additions: list[int] = []
    removals: list[int] = []

    while len(additions) < limit:
        under = [
            frame
            for frame in range(frame_count)
            if int(counts[frame].item()) < int(target_counts[frame].item())
            and additions_by_frame[frame]
        ]
        over = [
            frame
            for frame in range(frame_count)
            if int(counts[frame].item()) > int(target_counts[frame].item())
            and removals_by_frame[frame]
        ]
        if not under or not over:
            break
        best_pair = None
        best_key = None
        for add_frame in under:
            addition = additions_by_frame[add_frame][0]
            need = int(target_counts[add_frame].item() - counts[add_frame].item())
            for remove_frame in over:
                removal = removals_by_frame[remove_frame][0]
                excess = int(counts[remove_frame].item() - target_counts[remove_frame].item())
                score = (
                    utility_by_token[addition]
                    - cost_by_token[removal]
                    + 0.20 * float(need + excess)
                )
                key = (score, -addition, -removal)
                if best_key is None or key > best_key:
                    best_key = key
                    best_pair = (add_frame, addition, remove_frame, removal)
        if best_pair is None:
            break
        add_frame, addition, remove_frame, removal = best_pair
        additions_by_frame[add_frame].pop(0)
        removals_by_frame[remove_frame].pop(0)
        additions.append(addition)
        removals.append(removal)
        counts[add_frame] += 1
        counts[remove_frame] -= 1

    margin = _cfg_float(config, "certv8_swap_margin", -0.02)
    while len(additions) < limit:
        choices = []
        for frame in range(frame_count):
            if not additions_by_frame[frame] or not removals_by_frame[frame]:
                continue
            addition = additions_by_frame[frame][0]
            removal = removals_by_frame[frame][0]
            gain = utility_by_token[addition] - cost_by_token[removal]
            choices.append((gain, -addition, -removal, frame, addition, removal))
        if not choices:
            break
        gain, _, _, frame, addition, removal = max(choices)
        if gain + 1e-12 < margin:
            break
        additions_by_frame[frame].pop(0)
        removals_by_frame[frame].pop(0)
        additions.append(addition)
        removals.append(removal)
    return additions, removals


def _trial_selection(
    baseline: torch.Tensor,
    additions: Sequence[int],
    removals: Sequence[int],
    count: int,
) -> torch.Tensor:
    selected = set(int(value) for value in baseline.detach().cpu().tolist())
    for addition, removal in zip(additions[:count], removals[:count]):
        selected.remove(int(removal))
        selected.add(int(addition))
    return torch.tensor(
        sorted(selected),
        dtype=torch.long,
        device=baseline.device,
    )


def _logdet_efficiency(
    design: torch.Tensor,
    baseline: torch.Tensor,
    selected: torch.Tensor,
    ridge: float,
) -> float:
    dimension = int(design.shape[1])
    identity = torch.eye(dimension, dtype=torch.float32, device=design.device)

    def value(indices: torch.Tensor) -> torch.Tensor:
        rows = design[indices].float()
        information = ridge * identity + rows.transpose(0, 1) @ rows
        sign, logdet = torch.linalg.slogdet(information)
        if float(sign.item()) <= 0.0:
            raise RuntimeError("non-positive V8 design information matrix")
        return logdet

    delta = (value(selected) - value(baseline)) / max(1, dimension)
    return float(torch.exp(delta.clamp(min=-20.0, max=20.0)).item())


def _reference_coverage(
    reference: torch.Tensor,
    selected: torch.Tensor,
    metric_flat: torch.Tensor,
    tokens_per_frame: int,
) -> float:
    values = []
    for frame in torch.unique(reference // tokens_per_frame).tolist():
        ref = reference[reference // tokens_per_frame == int(frame)]
        anchors = selected[selected // tokens_per_frame == int(frame)]
        if anchors.numel() == 0:
            values.append(torch.zeros(ref.numel(), device=reference.device))
            continue
        similarity = metric_flat[ref].float() @ metric_flat[anchors].float().transpose(0, 1)
        values.append((0.5 * (similarity.amax(dim=1) + 1.0)).clamp(0.0, 1.0))
    if not values:
        return 0.0
    return float(torch.cat(values).mean().item())


def _joint_coverage(
    v3_indices: torch.Tensor,
    structural_indices: torch.Tensor,
    selected: torch.Tensor,
    metric_flat: torch.Tensor,
    tokens_per_frame: int,
    v3_weight: float,
) -> Tuple[float, float, float]:
    v3_coverage = _reference_coverage(
        v3_indices,
        selected,
        metric_flat,
        tokens_per_frame,
    )
    structural_coverage = _reference_coverage(
        structural_indices,
        selected,
        metric_flat,
        tokens_per_frame,
    )
    joint = v3_weight * v3_coverage + (1.0 - v3_weight) * structural_coverage
    return joint, v3_coverage, structural_coverage


def _build_local_plan(
    selected: torch.Tensor,
    new_anchors: set[int],
    protected_v3: set[int],
    analysis: _V3Analysis,
    frame_times: torch.Tensor,
    has_real_times: bool,
    frame_count: int,
    tokens_per_frame: int,
    config: Any,
) -> CertVidPlan:
    total_tokens = int(analysis.metric_flat.shape[0])
    budget = int(selected.numel())
    similarity = (
        analysis.metric_flat.float()
        @ analysis.metric_flat[selected].float().transpose(0, 1)
    )
    source_frame = torch.arange(
        frame_count,
        device=selected.device,
    ).repeat_interleave(tokens_per_frame)
    anchor_frame = source_frame[selected]
    frame_delta = (source_frame.unsqueeze(1) - anchor_frame.unsqueeze(0)).abs()
    valid = frame_delta == 0
    adjacent = frame_delta == 1
    cross_similarity = _cfg_float(config, "certv8_cross_frame_similarity", 0.88)
    cross_valid = adjacent & (similarity >= cross_similarity)
    if has_real_times:
        time_gap = torch.abs(
            frame_times[source_frame].unsqueeze(1) - frame_times[anchor_frame].unsqueeze(0)
        )
        cross_valid &= time_gap <= _cfg_float(
            config,
            "certv8_cross_frame_max_seconds",
            8.0,
        )
    valid |= cross_valid
    if not bool(valid.any(dim=1).all()):
        raise RuntimeError("V8 local plan has a frame without a reachable anchor")
    similarity = similarity.masked_fill(~valid, -2.0)
    same_component = (
        analysis.component_ids.unsqueeze(1)
        == analysis.component_ids[selected].unsqueeze(0)
    )
    similarity = similarity + _cfg_float(
        config,
        "certv8_component_bonus",
        0.08,
    ) * same_component.float()

    topk = min(
        max(1, _cfg_int(config, "certv8_assignment_topk", 2)),
        budget,
    )
    values, assignment = torch.topk(similarity, k=topk, dim=1, largest=True)
    chosen_valid = torch.gather(valid, 1, assignment)
    best_valid = torch.argmax(similarity, dim=1, keepdim=True).expand_as(assignment)
    assignment = torch.where(chosen_valid, assignment, best_valid)
    values = torch.gather(similarity, 1, assignment)
    weights = torch.softmax(
        values.float()
        / max(1e-4, _cfg_float(config, "certv8_assignment_temperature", 0.07)),
        dim=1,
    )
    anchor_positions = torch.arange(
        budget,
        dtype=torch.long,
        device=selected.device,
    )
    assignment[selected, 0] = anchor_positions
    weights[selected] = 0.0
    weights[selected, 0] = 1.0

    source_mass = (
        0.5 + 0.5 * analysis.demand_weight * total_tokens
    ).clamp(0.25, 2.0)
    alpha_value = min(
        0.25,
        max(0.0, _cfg_float(config, "certv8_fusion_alpha", 0.10)),
    )
    alpha = torch.full(
        (budget,),
        alpha_value,
        dtype=torch.float32,
        device=selected.device,
    )
    protected = protected_v3 | new_anchors
    if protected:
        protected_tensor = torch.tensor(
            sorted(protected),
            dtype=torch.long,
            device=selected.device,
        )
        alpha[torch.isin(selected, protected_tensor)] = 0.0
    return CertVidPlan(
        anchor_indices=selected,
        assignment_indices=assignment,
        assignment_weights=weights,
        source_mass=source_mass,
        fusion_alpha=alpha,
        raw_token_count=total_tokens,
    )


def _store_diagnostics(config: Any, diagnostics: Dict[str, Any]) -> None:
    config.last_certv8_diagnostics = diagnostics
    config.last_certv8_fallback_reason = diagnostics.get("fallback_reason")
    config.last_certv8_swap_count = int(diagnostics.get("swap_count", 0))
    config.last_certv8_modified_ratio = float(diagnostics.get("modified_ratio", 0.0))
    config.last_certv8_v3_overlap_ratio = float(
        diagnostics.get("v3_overlap_ratio", 1.0)
    )
    config.last_certv8_structural_overlap_ratio = float(
        diagnostics.get("structural_overlap_ratio", 0.0)
    )
    config.last_certv8_local_overlap_ratio = float(
        diagnostics.get("local_overlap_ratio", 0.0)
    )
    config.last_certv8_expert_agreement_ratio = float(
        diagnostics.get("expert_agreement_ratio", 0.0)
    )
    config.last_certv8_base_joint_coverage = float(
        diagnostics.get("base_joint_coverage", 0.0)
    )
    config.last_certv8_final_joint_coverage = float(
        diagnostics.get("final_joint_coverage", 0.0)
    )
    config.last_certv8_d_efficiency = float(diagnostics.get("d_efficiency", 1.0))
    config.last_certv8_unsafe_assignment_count = int(
        diagnostics.get("unsafe_assignment_count", 0)
    )
    if bool(getattr(config, "certv8_debug", False)):
        print(
            "[certvid-v8] "
            f"sample={getattr(config, '_debug_sample_id', 'unknown')} "
            f"fallback={diagnostics.get('fallback_reason')} "
            f"duration={diagnostics.get('duration_seconds', 0.0):.1f}s "
            f"max_gap={diagnostics.get('max_frame_gap_seconds', 0.0):.1f}s "
            f"agreement={diagnostics.get('expert_agreement_ratio', 0.0):.3f} "
            f"swaps={diagnostics.get('swap_count', 0)} "
            f"v3_overlap={diagnostics.get('v3_overlap_ratio', 1.0):.3f} "
            f"local_overlap={diagnostics.get('local_overlap_ratio', 0.0):.3f} "
            f"joint={diagnostics.get('base_joint_coverage', 0.0):.4f}->"
            f"{diagnostics.get('final_joint_coverage', 0.0):.4f} "
            f"D-eff={diagnostics.get('d_efficiency', 1.0):.4f} "
            f"v3_frames={diagnostics.get('v3_frame_counts', [])} "
            f"local_frames={diagnostics.get('local_frame_counts', [])} "
            f"target_frames={diagnostics.get('target_frame_counts', [])} "
            f"final_frames={diagnostics.get('final_frame_counts', [])}"
        )


def _fallback(
    config: Any,
    diagnostics: Dict[str, Any],
    reason: str,
    output: torch.Tensor,
    indices: torch.Tensor,
    plan: Optional[CertVidPlan],
) -> Tuple[torch.Tensor, torch.Tensor]:
    diagnostics["fallback_reason"] = reason
    diagnostics.setdefault("swap_count", 0)
    diagnostics.setdefault("modified_ratio", 0.0)
    diagnostics.setdefault("v3_overlap_ratio", 1.0)
    diagnostics.setdefault("d_efficiency", 1.0)
    diagnostics.setdefault(
        "final_joint_coverage",
        diagnostics.get("base_joint_coverage", 0.0),
    )
    diagnostics.setdefault(
        "final_frame_counts",
        diagnostics.get("v3_frame_counts", []),
    )
    if plan is not None:
        config._certvid_plan = plan
    config.last_adapter_variant = "certvid_v8"
    _store_diagnostics(config, diagnostics)
    return output, indices


def certvid_v8_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    config: Any,
    question_features: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse complementary V3 and structural evidence under one fixed budget."""
    if video_features.ndim != 3:
        raise ValueError(
            f"expected video_features [T, HW, D], got {tuple(video_features.shape)}"
        )
    frame_count, tokens_per_frame, _ = video_features.shape
    sink: Dict[str, Any] = {}
    v3_output, v3_indices = certvid_v3_compression(
        video_features,
        cls_attention,
        config,
        question_features,
        analysis_sink=sink,
    )
    v3_plan = getattr(config, "_certvid_plan", None)
    v3_counts = torch.bincount(
        v3_indices // max(1, tokens_per_frame),
        minlength=frame_count,
    )
    diagnostics: Dict[str, Any] = {
        "fallback_reason": None,
        "v3_frame_counts": [
            int(value) for value in v3_counts.detach().cpu().tolist()
        ],
    }
    if not bool(getattr(config, "certv8_enabled", True)):
        return _fallback(
            config,
            diagnostics,
            "disabled",
            v3_output,
            v3_indices,
            v3_plan,
        )
    if v3_plan is None:
        return _fallback(
            config,
            diagnostics,
            "missing_v3_plan",
            v3_output,
            v3_indices,
            v3_plan,
        )
    if bool(sink.get("identity", False)) or not sink:
        return _fallback(
            config,
            diagnostics,
            "identity_budget",
            v3_output,
            v3_indices,
            v3_plan,
        )

    try:
        analysis = _analysis_from_sink(sink)
        frame_times, has_real_times, timestamp_source = _frame_times(
            config,
            frame_count,
            video_features.device,
        )
        duration = (
            float(frame_times[-1].item() - frame_times[0].item())
            if has_real_times and frame_count > 1
            else 0.0
        )
        max_gap = (
            float((frame_times[1:] - frame_times[:-1]).amax().item())
            if has_real_times and frame_count > 1
            else 0.0
        )
        diagnostics.update(
            {
                "timestamp_source": timestamp_source,
                "duration_seconds": duration,
                "max_frame_gap_seconds": max_gap,
            }
        )
        min_gap = max(
            0.0,
            _cfg_float(config, "certv8_min_horizon_gap_seconds", 4.0),
        )
        if not has_real_times:
            return _fallback(
                config,
                diagnostics,
                "missing_real_timestamps",
                v3_output,
                v3_indices,
                v3_plan,
            )
        if max_gap + 1e-12 < min_gap:
            return _fallback(
                config,
                diagnostics,
                "short_horizon",
                v3_output,
                v3_indices,
                v3_plan,
            )

        budget = int(v3_indices.numel())
        local_score, frame_event = _structural_scores(
            video_features,
            cls_attention,
            analysis,
            config,
        )
        local_indices = _local_evidence_coreset(
            analysis,
            local_score,
            frame_event,
            frame_count,
            tokens_per_frame,
            budget,
            config,
        )
        local_counts = torch.bincount(
            local_indices // tokens_per_frame,
            minlength=frame_count,
        )
        agreement = torch.isin(v3_indices, local_indices)
        agreement_count = int(agreement.sum().item())
        agreement_ratio = agreement_count / max(
            1,
            min(budget, int(local_indices.numel())),
        )
        diagnostics.update(
            {
                "expert_agreement_count": agreement_count,
                "expert_agreement_ratio": agreement_ratio,
                "local_coreset_tokens": int(local_indices.numel()),
                "local_frame_counts": [
                    int(value)
                    for value in local_counts.detach().cpu().tolist()
                ],
            }
        )
        min_disagreement = min(
            1.0,
            max(0.0, _cfg_float(config, "certv8_min_disagreement_ratio", 0.08)),
        )
        if 1.0 - agreement_ratio < min_disagreement:
            return _fallback(
                config,
                diagnostics,
                "experts_already_agree",
                v3_output,
                v3_indices,
                v3_plan,
            )

        protected = {
            int(token)
            for token in v3_indices[
                v3_plan.fusion_alpha <= 1e-12
            ].detach().cpu().tolist()
        }
        protect_ratio = min(
            0.40,
            max(0.0, _cfg_float(config, "certv8_design_protect_ratio", 0.08)),
        )
        protect_count = min(
            budget,
            int(math.ceil(protect_ratio * budget)),
        )
        if protect_count > 0:
            leverage = analysis.design[v3_indices].float().square().sum(dim=1)
            local = torch.topk(leverage, k=protect_count, largest=True).indices
            protected.update(
                int(token)
                for token in v3_indices[local].detach().cpu().tolist()
            )

        union = torch.unique(torch.cat([v3_indices, local_indices]), sorted=True)
        union_counts = torch.bincount(
            union // tokens_per_frame,
            minlength=frame_count,
        )
        protected_tensor = torch.tensor(
            sorted(protected),
            dtype=torch.long,
            device=v3_indices.device,
        )
        protected_counts = torch.bincount(
            protected_tensor // tokens_per_frame,
            minlength=frame_count,
        )
        local_mix = min(
            1.0,
            max(0.0, _cfg_float(config, "certv8_local_mix", 0.55)),
        )
        desired = (
            (1.0 - local_mix) * v3_counts.float()
            + local_mix * local_counts.float()
        )
        target_counts = _allocate_counts(
            budget,
            desired,
            protected_counts,
            union_counts,
        )
        diagnostics["target_frame_counts"] = [
            int(value) for value in target_counts.detach().cpu().tolist()
        ]

        max_ratio = min(
            0.50,
            max(0.0, _cfg_float(config, "certv8_long_max_swap_ratio", 0.20)),
        )
        swap_limit = min(
            budget,
            int(math.ceil(max_ratio * budget)),
        )
        additions, removals = _propose_swaps(
            v3_indices,
            local_indices,
            protected,
            target_counts,
            analysis,
            local_score,
            frame_count,
            tokens_per_frame,
            swap_limit,
            config,
        )
        diagnostics["swap_limit"] = swap_limit
        if not additions:
            return _fallback(
                config,
                diagnostics,
                "no_complementary_swaps",
                v3_output,
                v3_indices,
                v3_plan,
            )

        v3_weight = min(
            0.90,
            max(0.10, _cfg_float(config, "certv8_v3_coverage_weight", 0.58)),
        )
        base_joint, base_v3_coverage, base_structural_coverage = _joint_coverage(
            v3_indices,
            local_indices,
            v3_indices,
            analysis.metric_flat,
            tokens_per_frame,
            v3_weight,
        )
        diagnostics.update(
            {
                "base_joint_coverage": base_joint,
                "base_v3_coverage": base_v3_coverage,
                "base_structural_coverage": base_structural_coverage,
            }
        )
        efficiency_floor = min(
            1.0,
            max(0.0, _cfg_float(config, "certv8_long_d_efficiency_floor", 0.95)),
        )
        min_joint_gain = max(
            0.0,
            _cfg_float(config, "certv8_min_joint_gain", 0.001),
        )
        count = min(len(additions), len(removals), swap_limit)
        selected: Optional[torch.Tensor] = None
        final_joint = base_joint
        final_v3_coverage = base_v3_coverage
        final_structural_coverage = base_structural_coverage
        d_efficiency = 1.0
        while count > 0:
            trial = _trial_selection(v3_indices, additions, removals, count)
            trial_efficiency = _logdet_efficiency(
                analysis.design,
                v3_indices,
                trial,
                analysis.ridge,
            )
            joint, v3_coverage, structural_coverage = _joint_coverage(
                v3_indices,
                local_indices,
                trial,
                analysis.metric_flat,
                tokens_per_frame,
                v3_weight,
            )
            if (
                trial_efficiency + 1e-12 >= efficiency_floor
                and joint + 1e-12 >= base_joint + min_joint_gain
            ):
                selected = trial
                final_joint = joint
                final_v3_coverage = v3_coverage
                final_structural_coverage = structural_coverage
                d_efficiency = trial_efficiency
                break
            next_count = int(math.floor(0.75 * count))
            count = next_count if next_count < count else count - 1
        if selected is None:
            return _fallback(
                config,
                diagnostics,
                "joint_or_design_guard",
                v3_output,
                v3_indices,
                v3_plan,
            )

        accepted_additions = set(int(value) for value in additions[:count])
        plan = _build_local_plan(
            selected,
            accepted_additions,
            protected,
            analysis,
            frame_times,
            has_real_times,
            frame_count,
            tokens_per_frame,
            config,
        )
        source_frames = torch.arange(
            frame_count,
            device=selected.device,
        ).repeat_interleave(tokens_per_frame)
        assigned_frames = source_frames[selected][plan.assignment_indices]
        unsafe = int(
            ((source_frames.unsqueeze(1) - assigned_frames).abs() > 1)
            .any(dim=1)
            .sum()
            .item()
        )
        if unsafe:
            return _fallback(
                config,
                diagnostics,
                "unsafe_assignment",
                v3_output,
                v3_indices,
                v3_plan,
            )

        output = apply_certvid_plan(
            video_features.reshape(frame_count * tokens_per_frame, -1),
            plan,
        )
        final_counts = torch.bincount(
            selected // tokens_per_frame,
            minlength=frame_count,
        )
        v3_overlap = float(torch.isin(selected, v3_indices).float().mean().item())
        local_overlap = float(
            torch.isin(selected, local_indices).float().mean().item()
        )
        diagnostics.update(
            {
                "swap_count": count,
                "modified_ratio": count / max(1, budget),
                "v3_overlap_ratio": v3_overlap,
                "structural_overlap_ratio": local_overlap,
                "local_overlap_ratio": local_overlap,
                "final_joint_coverage": final_joint,
                "final_v3_coverage": final_v3_coverage,
                "final_structural_coverage": final_structural_coverage,
                "d_efficiency": d_efficiency,
                "unsafe_assignment_count": unsafe,
                "final_frame_counts": [
                    int(value)
                    for value in final_counts.detach().cpu().tolist()
                ],
            }
        )
        config._certvid_plan = plan
        config.vision_token_length = int(output.shape[0])
        config.visual_token_length = int(output.shape[0])
        config.llm_token_length = None
        config.last_adapter_variant = "certvid_v8"
        config.last_adapter_raw_tokens = float(frame_count * tokens_per_frame)
        config.last_adapter_output_tokens = float(output.shape[0])
        _store_diagnostics(config, diagnostics)
        return output, selected
    except Exception as error:
        diagnostics["optimization_error"] = f"{type(error).__name__}: {error}"
        return _fallback(
            config,
            diagnostics,
            "optimization_error",
            v3_output,
            v3_indices,
            v3_plan,
        )
