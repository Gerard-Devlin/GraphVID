"""CertVID V7: causal multi-scale transition repair over exact V3 anchors.

V7 keeps V3 as the static evidence selector and exact fallback.  For reliable
long-horizon inputs it learns forward transition operators at several temporal
scales, then exchanges a bounded set of anchors only when the compressed video
better preserves their causal residuals without collapsing V3's D-optimal
information volume.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .certvid import CertVidPlan, _cfg_float, _cfg_int, _minmax, apply_certvid_plan
from .certvid_hr import _normalize_frame_times
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


@dataclass
class _Edges:
    source: torch.Tensor
    target: torch.Tensor
    weight: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.source.numel())


@dataclass
class _TransitionFamily:
    edges: _Edges
    operator: torch.Tensor


def _empty_edges(device: torch.device) -> _Edges:
    return _Edges(
        source=torch.empty(0, dtype=torch.long, device=device),
        target=torch.empty(0, dtype=torch.long, device=device),
        weight=torch.empty(0, dtype=torch.float32, device=device),
    )


def _analysis_from_sink(sink: Dict[str, Any]) -> _V3Analysis:
    required = {
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
    }
    missing = sorted(required.difference(sink))
    if missing:
        raise RuntimeError(f"CertVID V7 missing captured V3 tensors: {missing}")
    return _V3Analysis(**{name: sink[name] for name in required})


def _semantic_segments(metric_frames: torch.Tensor) -> tuple[torch.Tensor, float]:
    frame_repr = F.normalize(metric_frames.float().mean(dim=1), dim=-1, eps=1e-6)
    if frame_repr.shape[0] <= 1:
        return torch.zeros(1, dtype=torch.long, device=metric_frames.device), 1.0
    gaps = 1.0 - (frame_repr[1:] * frame_repr[:-1]).sum(dim=-1)
    threshold = max(0.10, float(torch.quantile(gaps.float(), 0.85).item()))
    boundaries = gaps >= threshold
    segments = torch.zeros(frame_repr.shape[0], dtype=torch.long, device=metric_frames.device)
    segments[1:] = torch.cumsum(boundaries.long(), dim=0)
    return segments, threshold


def _matched_edges(
    metric_frames: torch.Tensor,
    frame_times: torch.Tensor,
    scale: int,
    component_frames: torch.Tensor,
    *,
    similarity_floor: float,
    spatial_radius: float,
) -> _Edges:
    frame_count, tokens_per_frame, _ = metric_frames.shape
    if scale <= 0 or frame_count <= scale:
        return _empty_edges(metric_frames.device)
    sources = []
    targets = []
    weights = []
    positive_gaps = frame_times[1:] - frame_times[:-1]
    positive_gaps = positive_gaps[positive_gaps > 0]
    median_gap = float(torch.median(positive_gaps).item()) if positive_gaps.numel() else 1.0
    height = max(1, int(math.sqrt(tokens_per_frame)))
    while height > 1 and tokens_per_frame % height != 0:
        height -= 1
    width = max(1, tokens_per_frame // height)
    token_ids = torch.arange(tokens_per_frame, device=metric_frames.device)
    rows = torch.div(token_ids, width, rounding_mode="floor").float()
    cols = torch.remainder(token_ids, width).float()
    coords = torch.stack(
        [rows / max(1, height - 1), cols / max(1, width - 1)],
        dim=-1,
    )
    for frame in range(frame_count - scale):
        similarity = metric_frames[frame] @ metric_frames[frame + scale].T
        match_similarity, match = similarity.max(dim=1)
        reverse = similarity.argmax(dim=0)
        local_ids = torch.arange(tokens_per_frame, device=metric_frames.device)
        mutual = reverse.index_select(0, match) == local_ids
        spatial_distance = (coords - coords.index_select(0, match)).norm(dim=-1)
        same_component = (
            component_frames[frame]
            == component_frames[frame + scale].index_select(0, match)
        )
        reliable = match_similarity >= float(similarity_floor)
        reliable &= spatial_distance <= float(max(0.0, spatial_radius))
        if scale > 1:
            reliable &= same_component
        valid = mutual & reliable
        minimum = max(1, tokens_per_frame // 8)
        if int(valid.sum().item()) < minimum:
            reliability = match_similarity - 0.10 * spatial_distance
            if scale > 1:
                reliability = reliability + 0.10 * same_component.float()
            eligible = torch.where(reliable)[0]
            if eligible.numel():
                count = min(minimum, int(eligible.numel()))
                fallback_order = torch.argsort(
                    reliability.index_select(0, eligible),
                    descending=True,
                    stable=True,
                )[:count]
                valid[eligible.index_select(0, fallback_order)] = True
        local_ids = local_ids[valid]
        match = match[valid]
        match_similarity = match_similarity[valid]
        source = local_ids + frame * tokens_per_frame
        target = match + (frame + scale) * tokens_per_frame
        elapsed = max(0.0, float(frame_times[frame + scale].item() - frame_times[frame].item()))
        time_ratio = max(0.0, elapsed / max(1e-6, median_gap))
        horizon_bonus = 1.0 + 0.25 * math.log1p(time_ratio)
        edge_weight = ((match_similarity + 1.0) * 0.5).clamp_min(0.05) * horizon_bonus
        sources.append(source)
        targets.append(target)
        weights.append(edge_weight.float())
    return _Edges(
        source=torch.cat(sources),
        target=torch.cat(targets),
        weight=torch.cat(weights),
    )


def _fit_transition(
    metric: torch.Tensor,
    edges: _Edges,
    ridge: float,
) -> _TransitionFamily:
    dim = int(metric.shape[1])
    if edges.count == 0:
        return _TransitionFamily(edges, torch.eye(dim, dtype=torch.float32, device=metric.device))
    source = metric.index_select(0, edges.source).float()
    target = metric.index_select(0, edges.target).float()
    root_weight = edges.weight.float().clamp_min(1e-6).sqrt().unsqueeze(1)
    source_weighted = source * root_weight
    target_weighted = target * root_weight
    gram = source_weighted.T @ source_weighted
    gram.diagonal().add_(max(1e-6, float(ridge)))
    cross = source_weighted.T @ target_weighted
    try:
        operator = torch.linalg.solve(gram, cross)
    except RuntimeError:
        operator = torch.linalg.pinv(gram) @ cross
    if not torch.isfinite(operator).all():
        raise RuntimeError("CertVID V7 learned a non-finite transition operator")
    return _TransitionFamily(edges, operator)


def _query_edges(
    query_relevance: torch.Tensor,
    query_confidence: float,
    frame_count: int,
    tokens_per_frame: int,
    frame_times: torch.Tensor,
    *,
    max_peaks: int,
    min_frame_gap: int,
) -> _Edges:
    device = frame_times.device
    if query_relevance.numel() == 0 or query_confidence < 0.10 or max_peaks < 2:
        return _empty_edges(device)
    sources = []
    targets = []
    weights = []
    relevance = query_relevance.reshape(query_relevance.shape[0], frame_count, tokens_per_frame)
    for atom in range(relevance.shape[0]):
        frame_score, frame_token = relevance[atom].max(dim=1)
        peak_threshold = max(
            0.75,
            float(torch.quantile(frame_score.float(), 0.90).item()),
        )
        order = torch.argsort(frame_score, descending=True, stable=True)
        peaks = []
        for candidate in order.tolist():
            if float(frame_score[candidate].item()) + 1e-12 < peak_threshold:
                continue
            if all(abs(candidate - existing) >= max(1, min_frame_gap) for existing in peaks):
                peaks.append(candidate)
            if len(peaks) >= max_peaks:
                break
        if len(peaks) < 2:
            continue
        peaks.sort()
        for left, right in zip(peaks[:-1], peaks[1:]):
            source = left * tokens_per_frame + int(frame_token[left].item())
            target = right * tokens_per_frame + int(frame_token[right].item())
            score = math.sqrt(
                max(0.0, float(frame_score[left].item()))
                * max(0.0, float(frame_score[right].item()))
            )
            elapsed = max(1e-6, float(frame_times[right].item() - frame_times[left].item()))
            duration_bonus = 1.0 + math.log1p(elapsed) / 10.0
            sources.append(source)
            targets.append(target)
            weights.append(max(1e-4, score * query_confidence * duration_bonus))
    if not sources:
        return _empty_edges(device)
    return _Edges(
        source=torch.tensor(sources, dtype=torch.long, device=device),
        target=torch.tensor(targets, dtype=torch.long, device=device),
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
    )


def _reconstruct(metric: torch.Tensor, plan: CertVidPlan) -> torch.Tensor:
    anchors = metric.index_select(0, plan.anchor_indices.long())
    assignment = plan.assignment_indices.long()
    weights = plan.assignment_weights.float()
    reconstructed = torch.zeros_like(metric.float())
    for neighbor in range(assignment.shape[1]):
        reconstructed += anchors.index_select(0, assignment[:, neighbor]) * weights[:, neighbor].unsqueeze(1)
    return F.normalize(reconstructed, dim=-1, eps=1e-6)


def _causal_distortion(
    metric: torch.Tensor,
    reconstructed: torch.Tensor,
    family: _TransitionFamily,
) -> torch.Tensor:
    edges = family.edges
    if edges.count == 0:
        return torch.empty(0, dtype=torch.float32, device=metric.device)
    source = metric.index_select(0, edges.source).float()
    target = metric.index_select(0, edges.target).float()
    source_hat = reconstructed.index_select(0, edges.source).float()
    target_hat = reconstructed.index_select(0, edges.target).float()
    original_residual = target - source @ family.operator
    compressed_residual = target_hat - source_hat @ family.operator
    error = (original_residual - compressed_residual).pow(2).mean(dim=-1)
    raw_transition = (target - source).pow(2).mean(dim=-1)
    scale = original_residual.pow(2).mean(dim=-1) + 0.10 * raw_transition + 1e-4
    return (error / scale).clamp(0.0, 4.0)


def _reconstruct_scalar(values: torch.Tensor, plan: CertVidPlan) -> torch.Tensor:
    anchors = values.index_select(0, plan.anchor_indices.long()).float()
    reconstructed = torch.zeros_like(values, dtype=torch.float32)
    for neighbor in range(plan.assignment_indices.shape[1]):
        reconstructed += anchors.index_select(
            0,
            plan.assignment_indices[:, neighbor].long(),
        ) * plan.assignment_weights[:, neighbor].float()
    return reconstructed


def _query_pair_distortion(
    query_score: torch.Tensor,
    plan: CertVidPlan,
    edges: _Edges,
) -> torch.Tensor:
    if edges.count == 0:
        return torch.empty(0, dtype=torch.float32, device=query_score.device)
    reconstructed = _reconstruct_scalar(query_score, plan)
    source = query_score.index_select(0, edges.source).float()
    target = query_score.index_select(0, edges.target).float()
    source_hat = reconstructed.index_select(0, edges.source)
    target_hat = reconstructed.index_select(0, edges.target)
    original_pair = torch.stack([source, target, target - source], dim=-1)
    compressed_pair = torch.stack(
        [source_hat, target_hat, target_hat - source_hat],
        dim=-1,
    )
    error = (original_pair - compressed_pair).pow(2).mean(dim=-1)
    scale = original_pair.pow(2).mean(dim=-1) + 1e-4
    return (error / scale).clamp(0.0, 4.0)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=weights.device)
    return (values.float() * weights.float()).sum() / weights.float().sum().clamp_min(1e-6)


def _family_statistics(
    metric: torch.Tensor,
    reconstructed: torch.Tensor,
    families: list[_TransitionFamily],
) -> tuple[torch.Tensor, list[tuple[_Edges, torch.Tensor]]]:
    records: list[tuple[_Edges, torch.Tensor]] = []
    family_losses = []
    for family in families:
        distortion = _causal_distortion(metric, reconstructed, family)
        if distortion.numel() == 0:
            continue
        records.append((family.edges, distortion))
        family_losses.append(_weighted_mean(distortion, family.edges.weight))
    if not family_losses:
        return torch.zeros((), dtype=torch.float32, device=metric.device), records
    return torch.stack(family_losses).mean(), records


def _path_objective(
    analysis: _V3Analysis,
    plan: CertVidPlan,
    local_families: list[_TransitionFamily],
    skip_families: list[_TransitionFamily],
    query_family: Optional[_TransitionFamily],
    config: Any,
) -> tuple[float, Dict[str, float], torch.Tensor]:
    reconstructed = _reconstruct(analysis.metric_flat, plan)
    node_residual = 1.0 - (analysis.metric_flat * reconstructed).sum(dim=-1).clamp(-1.0, 1.0)
    local_loss, local_records = _family_statistics(
        analysis.metric_flat,
        reconstructed,
        local_families,
    )
    skip_loss, skip_records = _family_statistics(
        analysis.metric_flat,
        reconstructed,
        skip_families,
    )
    query_records: list[tuple[_Edges, torch.Tensor]] = []
    query_loss = torch.zeros((), dtype=torch.float32, device=analysis.metric_flat.device)
    if query_family is not None and query_family.edges.count:
        causal_query = _causal_distortion(
            analysis.metric_flat,
            reconstructed,
            query_family,
        )
        pair_query = _query_pair_distortion(
            analysis.query_score,
            plan,
            query_family.edges,
        )
        query_distortion = 0.70 * causal_query + 0.30 * pair_query
        query_loss = _weighted_mean(query_distortion, query_family.edges.weight)
        query_records.append((query_family.edges, query_distortion))
    components = {
        "node": float((node_residual * analysis.demand_weight).sum().item()),
        "local": float(local_loss.item()),
        "skip": float(skip_loss.item()),
        "query": float(query_loss.item()),
    }
    configured = {
        "node": max(0.0, _cfg_float(config, "certv7_node_weight", 0.20)),
        "local": max(0.0, _cfg_float(config, "certv7_local_edge_weight", 0.35)),
        "skip": max(0.0, _cfg_float(config, "certv7_skip_edge_weight", 0.25)),
        "query": max(0.0, _cfg_float(config, "certv7_query_edge_weight", 0.20)),
    }
    active = {
        "node": configured["node"],
        "local": configured["local"] if local_records else 0.0,
        "skip": configured["skip"] if skip_records else 0.0,
        "query": configured["query"] if query_records else 0.0,
    }
    denominator = max(1e-8, sum(active.values()))
    objective = sum(active[name] * components[name] for name in active) / denominator

    incident = node_residual * analysis.demand_weight * float(analysis.metric_flat.shape[0])
    incident_weight = torch.ones_like(incident)
    for records, scale in (
        (local_records, active["local"]),
        (skip_records, active["skip"]),
        (query_records, active["query"]),
    ):
        if scale <= 0.0:
            continue
        for edges, distortion in records:
            normalized_weight = (
                edges.weight.float()
                / edges.weight.float().sum().clamp_min(1e-6)
                * float(analysis.metric_flat.shape[0])
            )
            contribution = distortion * normalized_weight * scale
            weight = normalized_weight * scale
            incident.index_add_(0, edges.source, contribution)
            incident.index_add_(0, edges.target, contribution)
            incident_weight.index_add_(0, edges.source, weight)
            incident_weight.index_add_(0, edges.target, weight)
    incident = incident / incident_weight.clamp_min(1e-6)
    return float(objective), components, incident


def _build_path_plan(
    selected: torch.Tensor,
    analysis: _V3Analysis,
    frame_times: torch.Tensor,
    frame_segments: torch.Tensor,
    locked_tokens: torch.Tensor,
    added_tokens: torch.Tensor,
    config: Any,
) -> CertVidPlan:
    metric = analysis.metric_flat
    total_tokens = int(metric.shape[0])
    budget = int(selected.numel())
    similarity = metric @ metric.index_select(0, selected).T
    source_frames = analysis.frame_ids.long()
    anchor_frames = source_frames.index_select(0, selected)
    source_times = frame_times.index_select(0, source_frames)
    anchor_times = frame_times.index_select(0, anchor_frames)
    time_distance = (source_times.unsqueeze(1) - anchor_times.unsqueeze(0)).abs()
    positive_gaps = frame_times[1:] - frame_times[:-1]
    positive_gaps = positive_gaps[positive_gaps > 0]
    median_gap = float(torch.median(positive_gaps).item()) if positive_gaps.numel() else 0.0
    radius = max(
        max(0.0, _cfg_float(config, "certv7_assignment_max_seconds", 12.0)),
        1.5 * median_gap,
    )
    source_segments = frame_segments.index_select(0, source_frames)
    anchor_segments = frame_segments.index_select(0, anchor_frames)
    same_frame = source_frames.unsqueeze(1) == anchor_frames.unsqueeze(0)
    cross_valid = (
        (source_segments.unsqueeze(1) == anchor_segments.unsqueeze(0))
        & (time_distance <= radius)
        & (similarity >= _cfg_float(config, "certv7_cross_time_similarity", 0.90))
    )
    valid = same_frame | cross_valid
    no_target = ~valid.any(dim=1)
    if bool(no_target.any()):
        nearest = time_distance.argmin(dim=1)
        valid[no_target, nearest[no_target]] = True
    same_component = analysis.component_ids.unsqueeze(1) == analysis.component_ids.index_select(0, selected).unsqueeze(0)
    scores = similarity + 0.08 * same_component.float()
    scores = scores.masked_fill(~valid, -2.0)
    topk = min(max(1, _cfg_int(config, "certv7_assignment_topk", 2)), budget)
    values, assignment = torch.topk(scores, k=topk, dim=1, largest=True)
    valid_top = values > -1.5
    weights = torch.softmax(
        values.float() / max(1e-4, _cfg_float(config, "certv7_assignment_temperature", 0.07)),
        dim=1,
    )
    weights = weights * valid_top.float()
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

    anchor_positions = torch.arange(budget, dtype=torch.long, device=selected.device)
    assignment[selected, 0] = anchor_positions
    weights[selected] = 0.0
    weights[selected, 0] = 1.0
    source_mass = (0.5 + 0.5 * analysis.demand_weight * float(total_tokens)).clamp(0.25, 2.0)
    protection = torch.maximum(analysis.attention[selected], analysis.query_score[selected])
    alpha = torch.full(
        (budget,),
        min(max(_cfg_float(config, "certv3_fusion_alpha", 0.12), 0.0), 0.75),
        dtype=torch.float32,
        device=selected.device,
    )
    alpha *= 1.0 - 0.65 * protection.clamp(0.0, 1.0)
    protected_count = min(budget, max(1, int(math.ceil(0.15 * budget))))
    alpha[torch.topk(protection, k=protected_count).indices] = 0.0
    if locked_tokens.numel():
        alpha[torch.isin(selected, locked_tokens)] = 0.0
    if added_tokens.numel():
        alpha[torch.isin(selected, added_tokens)] = 0.0
    return CertVidPlan(
        anchor_indices=selected,
        assignment_indices=assignment,
        assignment_weights=weights,
        source_mass=source_mass,
        fusion_alpha=alpha,
        raw_token_count=total_tokens,
    )


def _information_logdet(design: torch.Tensor, selected: torch.Tensor, ridge: float) -> float:
    rows = design.index_select(0, selected).float()
    information = rows.T @ rows
    information.diagonal().add_(max(1e-6, float(ridge)))
    sign, logdet = torch.linalg.slogdet(information)
    if float(sign.item()) <= 0.0 or not torch.isfinite(logdet):
        raise RuntimeError("CertVID V7 information matrix is not positive definite")
    return float(logdet.item())


def _anchor_load(plan: CertVidPlan, demand: torch.Tensor) -> torch.Tensor:
    load = torch.zeros(plan.anchor_indices.numel(), dtype=torch.float32, device=demand.device)
    for neighbor in range(plan.assignment_indices.shape[1]):
        load.index_add_(
            0,
            plan.assignment_indices[:, neighbor].long(),
            demand * plan.assignment_weights[:, neighbor].float(),
        )
    return load


def _exchange_proposals(
    analysis: _V3Analysis,
    baseline_plan: CertVidPlan,
    incident: torch.Tensor,
    max_swaps: int,
    add_pool: int,
    remove_pool: int,
) -> list[tuple[int, int]]:
    selected = baseline_plan.anchor_indices.long()
    total_tokens = int(analysis.metric_flat.shape[0])
    selected_mask = torch.zeros(total_tokens, dtype=torch.bool, device=selected.device)
    selected_mask[selected] = True
    locked_mask = baseline_plan.fusion_alpha.float() <= 1e-12

    similarity = analysis.metric_flat @ analysis.metric_flat.index_select(0, selected).T
    diversity = 1.0 - similarity.max(dim=1).values
    add_score = 0.65 * _minmax(incident, dim=0) + 0.20 * _minmax(diversity, dim=0)
    add_score += 0.15 * _minmax(analysis.query_score, dim=0)
    add_score[selected_mask] = float("-inf")
    additions = torch.argsort(add_score, descending=True, stable=True)[: max(1, add_pool)]

    load = _anchor_load(baseline_plan, analysis.demand_weight)
    rows = analysis.design.index_select(0, selected).float()
    information = rows.T @ rows
    information.diagonal().add_(max(1e-6, analysis.ridge))
    inverse = torch.linalg.pinv(information)
    leverage = torch.sum((rows @ inverse) * rows, dim=1)
    remove_keep = (
        0.50 * _minmax(incident.index_select(0, selected), dim=0)
        + 0.30 * _minmax(load, dim=0)
        + 0.20 * _minmax(leverage, dim=0)
    )
    eligible_positions = torch.where(~locked_mask)[0]
    if eligible_positions.numel() == 0:
        return []
    eligible_keep = remove_keep.index_select(0, eligible_positions)
    removal_order = torch.argsort(eligible_keep, descending=False, stable=True)
    removals = eligible_positions.index_select(
        0,
        removal_order[: min(max(1, remove_pool), int(eligible_positions.numel()))],
    )

    frame_counts = torch.bincount(
        analysis.frame_ids.index_select(0, selected).long(),
        minlength=int(analysis.frame_ids.max().item()) + 1,
    ).tolist()
    used_removals = set()
    proposals: list[tuple[int, int]] = []
    for add_tensor in additions:
        add = int(add_tensor.item())
        add_frame = int(analysis.frame_ids[add].item())
        frame_counts[add_frame] += 1
        chosen_position = None
        for remove_position_tensor in removals:
            remove_position = int(remove_position_tensor.item())
            if remove_position in used_removals or bool(locked_mask[remove_position].item()):
                continue
            remove = int(selected[remove_position].item())
            remove_frame = int(analysis.frame_ids[remove].item())
            if frame_counts[remove_frame] <= 1:
                continue
            chosen_position = remove_position
            frame_counts[remove_frame] -= 1
            break
        if chosen_position is None:
            frame_counts[add_frame] -= 1
            continue
        used_removals.add(chosen_position)
        proposals.append((add, int(selected[chosen_position].item())))
        if len(proposals) >= max_swaps:
            break
    return proposals


def _store_diagnostics(config: Any, diagnostics: Dict[str, Any]) -> None:
    config.last_certv7_diagnostics = diagnostics
    config.last_certv7_timestamp_source = diagnostics.get("timestamp_source", "missing")
    config.last_certv7_fallback_reason = diagnostics.get("fallback_reason")
    config.last_certv7_duration_seconds = float(diagnostics.get("duration_seconds", 0.0))
    config.last_certv7_local_edge_count = int(diagnostics.get("local_edge_count", 0))
    config.last_certv7_skip_edge_count = int(diagnostics.get("skip_edge_count", 0))
    config.last_certv7_query_edge_count = int(diagnostics.get("query_edge_count", 0))
    config.last_certv7_transition_scale_count = int(
        diagnostics.get("transition_scale_count", 0)
    )
    config.last_certv7_swap_count = int(diagnostics.get("swap_count", 0))
    config.last_certv7_modified_ratio = float(diagnostics.get("modified_ratio", 0.0))
    config.last_certv7_d_efficiency = float(diagnostics.get("d_efficiency", 1.0))
    config.last_certv7_base_path_loss = float(diagnostics.get("base_path_loss", 0.0))
    config.last_certv7_final_path_loss = float(diagnostics.get("final_path_loss", 0.0))
    if bool(getattr(config, "certv7_debug", False)):
        print(
            "[CertVID-V7] "
            f"fallback={diagnostics.get('fallback_reason')} "
            f"duration={diagnostics.get('duration_seconds', 0.0):.1f}s "
            f"edges={diagnostics.get('local_edge_count', 0)}/"
            f"{diagnostics.get('skip_edge_count', 0)}/"
            f"{diagnostics.get('query_edge_count', 0)} "
            f"swaps={diagnostics.get('swap_count', 0)} "
            f"path={diagnostics.get('base_path_loss', 0.0):.5f}->"
            f"{diagnostics.get('final_path_loss', 0.0):.5f}"
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
    diagnostics.setdefault("d_efficiency", 1.0)
    diagnostics.setdefault("final_path_loss", diagnostics.get("base_path_loss", 0.0))
    if plan is not None:
        config._certvid_plan = plan
    config.last_adapter_variant = "certvid_v7"
    _store_diagnostics(config, diagnostics)
    return output, indices


def certvid_v7_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: Any,
    question_features: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Repair V3 anchors against causal multi-scale transition distortion."""

    if video_features.ndim != 3:
        raise ValueError(f"expected video_features [T, HW, D], got {tuple(video_features.shape)}")
    config = flashvid_config
    frame_count, tokens_per_frame, _ = video_features.shape
    diagnostics: Dict[str, Any] = {
        "timestamp_source": str(getattr(config, "_certvid_frame_times_source", "missing")),
        "fallback_reason": None,
        "duration_seconds": 0.0,
        "local_edge_count": 0,
        "skip_edge_count": 0,
        "query_edge_count": 0,
        "swap_count": 0,
        "modified_ratio": 0.0,
        "d_efficiency": 1.0,
    }
    raw_times = getattr(config, "_certvid_frame_times_sec", None)
    frame_times, timestamp_error = _normalize_frame_times(
        raw_times,
        frame_count,
        device=video_features.device,
    )
    duration = (
        float(frame_times[-1].item() - frame_times[0].item())
        if frame_times is not None and frame_times.numel() > 1
        else 0.0
    )
    diagnostics["duration_seconds"] = duration
    can_analyze = frame_times is not None and duration >= _cfg_float(config, "certv7_min_duration_seconds", 120.0)
    sink: Optional[Dict[str, Any]] = {} if can_analyze else None
    v3_output, v3_indices = certvid_v3_compression(
        video_features=video_features,
        cls_attention=cls_attention,
        flashvid_config=config,
        question_features=question_features,
        analysis_sink=sink,
    )
    v3_plan = getattr(config, "_certvid_plan", None)
    if v3_plan is None:
        return _fallback(config, diagnostics, "missing_v3_plan", v3_output, v3_indices, v3_plan)
    if frame_times is None:
        return _fallback(
            config,
            diagnostics,
            timestamp_error or "invalid_timestamps",
            v3_output,
            v3_indices,
            v3_plan,
        )
    if not can_analyze:
        return _fallback(config, diagnostics, "short_horizon", v3_output, v3_indices, v3_plan)
    if sink is None or bool(sink.get("identity", False)):
        return _fallback(config, diagnostics, "identity_budget", v3_output, v3_indices, v3_plan)

    try:
        analysis = _analysis_from_sink(sink)
        metric_frames = analysis.metric_flat.reshape(frame_count, tokens_per_frame, -1)
        component_frames = analysis.component_ids.reshape(frame_count, tokens_per_frame)
        frame_segments, segment_threshold = _semantic_segments(metric_frames)
        transition_ridge = max(
            1e-6,
            _cfg_float(config, "certv7_transition_ridge", 0.10),
        )
        similarity_floor = _cfg_float(config, "certv7_match_similarity", 0.30)
        spatial_radius = max(
            0.0,
            _cfg_float(config, "certv7_match_spatial_radius", 0.60),
        )
        local_edges = _matched_edges(
            metric_frames,
            frame_times,
            1,
            component_frames,
            similarity_floor=similarity_floor,
            spatial_radius=spatial_radius,
        )
        local_families = [_fit_transition(analysis.metric_flat, local_edges, transition_ridge)]
        skip_families: list[_TransitionFamily] = []
        max_skip = max(2, _cfg_int(config, "certv7_max_skip_units", 8))
        scale = 2
        while scale <= max_skip:
            edges = _matched_edges(
                metric_frames,
                frame_times,
                scale,
                component_frames,
                similarity_floor=similarity_floor,
                spatial_radius=spatial_radius,
            )
            if edges.count:
                skip_families.append(
                    _fit_transition(analysis.metric_flat, edges, transition_ridge)
                )
            scale *= 2
        query_edges = _query_edges(
            analysis.query_relevance,
            analysis.query_confidence,
            frame_count,
            tokens_per_frame,
            frame_times,
            max_peaks=max(2, _cfg_int(config, "certv7_query_peaks_per_atom", 3)),
            min_frame_gap=max(1, _cfg_int(config, "certv7_query_min_frame_gap", 2)),
        )
        query_family = (
            _fit_transition(analysis.metric_flat, query_edges, transition_ridge)
            if query_edges.count
            else None
        )
        diagnostics.update(
            {
                "segment_count": int(frame_segments.max().item()) + 1,
                "segment_threshold": segment_threshold,
                "transition_ridge": transition_ridge,
                "local_edge_count": local_edges.count,
                "skip_edge_count": sum(family.edges.count for family in skip_families),
                "transition_scale_count": len(local_families) + len(skip_families),
                "query_edge_count": query_edges.count,
            }
        )
        base_loss, base_components, incident = _path_objective(
            analysis,
            v3_plan,
            local_families,
            skip_families,
            query_family,
            config,
        )
        diagnostics["base_path_loss"] = base_loss
        diagnostics["base_path_components"] = base_components
        if base_loss < _cfg_float(config, "certv7_min_path_residual", 0.02):
            return _fallback(config, diagnostics, "path_already_faithful", v3_output, v3_indices, v3_plan)

        budget = int(v3_indices.numel())
        max_swaps = min(
            budget,
            max(1, int(math.ceil(_cfg_float(config, "certv7_max_swap_ratio", 0.30) * budget))),
        )
        proposals = _exchange_proposals(
            analysis,
            v3_plan,
            incident,
            max_swaps=max_swaps,
            add_pool=max(max_swaps, _cfg_int(config, "certv7_add_pool", 96)),
            remove_pool=max(max_swaps, _cfg_int(config, "certv7_remove_pool", 96)),
        )
        if not proposals:
            return _fallback(config, diagnostics, "no_exchange_candidates", v3_output, v3_indices, v3_plan)

        baseline_logdet = _information_logdet(analysis.design, v3_indices, analysis.ridge)
        d_floor = _cfg_float(config, "certv7_d_efficiency_floor", 0.97)
        path_margin = _cfg_float(config, "certv7_path_margin", 1e-4)
        locked = v3_indices[v3_plan.fusion_alpha.float() <= 1e-12]
        prefix_sizes = []
        size = len(proposals)
        while size >= 1:
            if size not in prefix_sizes:
                prefix_sizes.append(size)
            size //= 2
        best = None
        for prefix in prefix_sizes:
            selected_set = {int(token) for token in v3_indices.tolist()}
            added = []
            for add, remove in proposals[:prefix]:
                selected_set.remove(remove)
                selected_set.add(add)
                added.append(add)
            selected = torch.tensor(sorted(selected_set), dtype=torch.long, device=v3_indices.device)
            if selected.numel() != budget or selected.unique().numel() != budget:
                continue
            logdet = _information_logdet(analysis.design, selected, analysis.ridge)
            d_efficiency = math.exp((logdet - baseline_logdet) / max(1, analysis.design.shape[1]))
            if d_efficiency + 1e-12 < d_floor:
                continue
            added_tensor = torch.tensor(added, dtype=torch.long, device=selected.device)
            plan = _build_path_plan(
                selected,
                analysis,
                frame_times,
                frame_segments,
                locked,
                added_tensor,
                config,
            )
            loss, components, _ = _path_objective(
                analysis,
                plan,
                local_families,
                skip_families,
                query_family,
                config,
            )
            if base_loss - loss < path_margin:
                continue
            if best is None or loss < best[0]:
                best = (loss, components, selected, plan, d_efficiency, prefix)
        if best is None:
            return _fallback(config, diagnostics, "no_safe_path_improvement", v3_output, v3_indices, v3_plan)

        final_loss, final_components, selected, plan, d_efficiency, swaps = best
        if not set(locked.tolist()).issubset(set(selected.tolist())):
            raise RuntimeError("CertVID V7 removed a locked V3 anchor")
        output = apply_certvid_plan(video_features.reshape(-1, video_features.shape[-1]), plan)
        config._certvid_plan = plan
        config.vision_token_length = int(output.shape[0])
        config.visual_token_length = int(output.shape[0])
        config.llm_token_length = None
        config.last_adapter_variant = "certvid_v7"
        config.last_adapter_raw_tokens = float(video_features.shape[0] * video_features.shape[1])
        config.last_adapter_output_tokens = float(output.shape[0])
        diagnostics.update(
            {
                "fallback_reason": None,
                "final_path_loss": final_loss,
                "final_path_components": final_components,
                "path_improvement": base_loss - final_loss,
                "swap_count": int(swaps),
                "modified_ratio": float(swaps / max(1, budget)),
                "d_efficiency": float(d_efficiency),
            }
        )
        _store_diagnostics(config, diagnostics)
        return output, selected
    except (RuntimeError, ValueError, IndexError) as error:
        config._certvid_plan = v3_plan
        diagnostics["error"] = f"{type(error).__name__}: {error}"
        return _fallback(config, diagnostics, "optimization_error", v3_output, v3_indices, v3_plan)
