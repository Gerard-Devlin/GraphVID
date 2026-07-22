"""CertVID-LH: long-horizon budget allocation over the CertVID V3 selector.

Short videos take the unmodified CertVID V3 path.  Long videos keep the same
global token budget, allocate it across temporal event groups, reserve a small
relay bank for evidence chains, and constrain residual fusion across groups.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch
import torch.nn.functional as F

from .certvid import (
    CertVidPlan,
    _build_components,
    _cfg_float,
    _cfg_int,
    _grid_hw,
    _local_detail,
    _metric_features,
    _minmax,
    _question_atoms,
    _question_relevance,
    _rank_normalize,
    _spatial_layout,
    apply_certvid_plan,
)
from .certvid_v2 import _component_support, _trajectory_signals
from .certvid_v3 import (
    _candidate_pool,
    _d_optimal_greedy,
    _design_features,
    _effective_ratio,
    _hard_certificates,
    _swap_refine,
    certvid_v3_compression,
)
from .configuration_flashvid import FlashVidConfig


@dataclass
class _LHAnalysis:
    metric_flat: torch.Tensor
    metric_frames: torch.Tensor
    design: torch.Tensor
    demand_weight: torch.Tensor
    quality: torch.Tensor
    event_score: torch.Tensor
    attention: torch.Tensor
    query_score: torch.Tensor
    query_relevance: torch.Tensor
    atom_weights: torch.Tensor
    query_confidence: float
    component_ids: torch.Tensor
    frame_ids: torch.Tensor
    spatial_ids: torch.Tensor
    novelty: torch.Tensor
    curvature: torch.Tensor
    detail: torch.Tensor
    frame_event: torch.Tensor
    ridge: float


def _safe_quantile(values: torch.Tensor, quantile: float, default: float = 0.0) -> float:
    values = values.float().reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return float(default)
    return float(torch.quantile(values, min(1.0, max(0.0, float(quantile)))).item())


def _normalize_frame_times(
    raw_times: Any,
    frame_count: int,
    *,
    device: torch.device,
) -> tuple[Optional[torch.Tensor], Optional[str]]:
    if raw_times is None:
        return None, "missing_timestamps"
    try:
        times = torch.as_tensor(raw_times, dtype=torch.float64, device=device).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        return None, "invalid_timestamp_type"
    if times.numel() == 0 or frame_count <= 0:
        return None, "empty_timestamps"
    if not bool(torch.isfinite(times).all()):
        return None, "nonfinite_timestamps"
    if times.numel() == frame_count:
        mapped = times
    elif times.numel() > frame_count and times.numel() % frame_count == 0:
        mapped = times.reshape(frame_count, -1).mean(dim=1)
    else:
        return None, "timestamp_length_mismatch"
    if mapped.numel() > 1 and bool(torch.any(mapped[1:] < mapped[:-1])):
        return None, "nonmonotonic_timestamps"
    return mapped.float(), None


def _frame_semantic_gap(metric_frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    frame_repr = F.normalize(metric_frames.float().mean(dim=1), p=2, dim=-1, eps=1e-6)
    gap = torch.zeros(frame_repr.shape[0], dtype=torch.float32, device=metric_frames.device)
    if frame_repr.shape[0] > 1:
        gap[1:] = (1.0 - torch.sum(frame_repr[1:] * frame_repr[:-1], dim=-1)).clamp(0.0, 2.0)
    return frame_repr, gap


def _query_peak_signal(
    query_relevance: torch.Tensor,
    frame_count: int,
    tokens_per_frame: int,
    *,
    peak_quantile: float,
    peak_floor: float,
    min_frame_distance: int,
) -> float:
    if query_relevance.numel() == 0 or frame_count < 2:
        return 0.0
    for atom_scores in query_relevance:
        frame_scores = atom_scores.view(frame_count, tokens_per_frame).amax(dim=1)
        threshold = max(peak_floor, _safe_quantile(frame_scores, peak_quantile, peak_floor))
        order = torch.argsort(frame_scores, descending=True, stable=True).tolist()
        peaks: list[int] = []
        for frame in order:
            if float(frame_scores[frame].item()) < threshold:
                break
            if all(abs(int(frame) - previous) >= min_frame_distance for previous in peaks):
                peaks.append(int(frame))
            if len(peaks) >= 2:
                return 1.0
    return 0.0


def _gate_metrics(
    *,
    metric_frames: torch.Tensor,
    cls_attention: torch.Tensor,
    query_relevance: torch.Tensor,
    frame_times: torch.Tensor,
    config: FlashVidConfig,
) -> dict[str, float | bool]:
    frame_count, tokens_per_frame, _ = metric_frames.shape
    _, semantic_gap = _frame_semantic_gap(metric_frames)
    span = float((frame_times[-1] - frame_times[0]).item()) if frame_count > 1 else 0.0
    max_gap = float((frame_times[1:] - frame_times[:-1]).max().item()) if frame_count > 1 else 0.0
    event_floor = _cfg_float(config, "certlh_event_floor", 0.08)
    event_threshold = max(
        event_floor,
        _safe_quantile(semantic_gap[1:], _cfg_float(config, "certlh_event_quantile", 0.80), event_floor),
    )
    event_count = int(torch.sum(semantic_gap[1:] >= event_threshold).item()) if frame_count > 1 else 0

    raw_attention = torch.nan_to_num(cls_attention.float().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    concentration = 0.0
    if raw_attention.numel() == frame_count * tokens_per_frame:
        frame_attention = raw_attention.view(frame_count, tokens_per_frame).mean(dim=1)
        spread = float((frame_attention.max() - frame_attention.min()).item())
        if spread > 1e-6:
            weights = torch.softmax((frame_attention - frame_attention.mean()) / frame_attention.std().clamp_min(1e-6), dim=0)
            top_count = max(1, int(math.ceil(0.25 * frame_count)))
            raw_concentration = float(torch.topk(weights, k=top_count).values.sum().item())
            concentration = min(1.0, max(0.0, (raw_concentration - 0.25) / 0.75))

    multi_peak = _query_peak_signal(
        query_relevance,
        frame_count,
        tokens_per_frame,
        peak_quantile=_cfg_float(config, "certlh_query_peak_quantile", 0.90),
        peak_floor=_cfg_float(config, "certlh_query_peak_floor", 0.75),
        min_frame_distance=max(2, frame_count // 4),
    )
    min_duration = max(1.0, _cfg_float(config, "certlh_min_duration_seconds", 120.0))
    duration_score = min(1.0, span / min_duration)
    event_score = min(1.0, event_count / float(max(1, _cfg_int(config, "certlh_min_groups", 4) - 1)))
    horizon_score = 0.45 * duration_score + 0.25 * event_score + 0.15 * concentration + 0.15 * multi_peak
    eligible = span >= min_duration and (
        max_gap >= _cfg_float(config, "certlh_horizon_gap_seconds", 4.0) or event_count > 0
    )
    enabled = eligible and horizon_score >= _cfg_float(config, "certlh_gate_threshold", 0.55)
    return {
        "enabled": bool(enabled),
        "duration_seconds": span,
        "max_timestamp_gap": max_gap,
        "semantic_threshold": event_threshold,
        "event_count": float(event_count),
        "attention_concentration": concentration,
        "multi_peak_signal": multi_peak,
        "horizon_score": horizon_score,
    }


def _build_analysis(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    question_features: Optional[torch.Tensor],
    config: FlashVidConfig,
    *,
    precomputed_metric_flat: Optional[torch.Tensor] = None,
    precomputed_query: Optional[tuple[torch.Tensor, torch.Tensor, float]] = None,
) -> _LHAnalysis:
    frame_count, tokens_per_frame, _ = video_features.shape
    total_tokens = frame_count * tokens_per_frame
    device = video_features.device
    metric_dim = max(32, _cfg_int(config, "certv3_metric_dim", 96))
    metric_flat = (
        _metric_features(video_features, metric_dim)
        if precomputed_metric_flat is None
        else precomputed_metric_flat
    )
    metric_frames = metric_flat.view(frame_count, tokens_per_frame, -1)
    height, width = _grid_hw(tokens_per_frame, config)
    spatial_bins = max(1, _cfg_int(config, "certv3_spatial_bins", 3))
    coords, frame_spatial_ids = _spatial_layout(tokens_per_frame, height, width, spatial_bins, device)
    frame_event, _, novelty_2d, curvature_2d, matches = _trajectory_signals(
        metric_frames,
        coords,
        _cfg_float(config, "certv3_spatial_penalty", 0.08),
    )
    component_ids_cpu, component_sizes_cpu = _build_components(
        frame_count,
        tokens_per_frame,
        frame_event,
        matches,
        _cfg_float(config, "certv3_track_threshold", 0.82),
    )
    component_ids = component_ids_cpu.to(device)
    component_sizes = component_sizes_cpu.to(device)
    frame_ids = torch.arange(frame_count, device=device).repeat_interleave(tokens_per_frame)
    component_value = _component_support(
        metric_flat,
        component_ids,
        component_sizes,
        frame_ids,
        frame_count,
    )
    temporal_count = min(frame_count, max(1, _cfg_int(config, "certv3_temporal_bins", 12)))
    temporal_ids = torch.div(frame_ids * temporal_count, max(1, frame_count), rounding_mode="floor").clamp_max(
        temporal_count - 1
    )
    spatial_ids = frame_spatial_ids.repeat(frame_count)
    spatial_count = spatial_bins * spatial_bins

    attention = _rank_normalize(cls_attention.float()).reshape(-1)
    novelty = novelty_2d.reshape(-1)
    curvature = curvature_2d.reshape(-1)
    detail = _local_detail(video_features, height, width).reshape(-1)
    event = frame_event.repeat_interleave(tokens_per_frame)
    if precomputed_query is None:
        atoms = _question_atoms(
            question_features,
            max(0, _cfg_int(config, "certv3_query_atoms", 8)),
            metric_dim,
        ).to(device)
        query_relevance, atom_weights, query_confidence = _question_relevance(atoms, metric_flat)
    else:
        query_relevance, atom_weights, query_confidence = precomputed_query
    query_score = (
        (query_relevance * atom_weights.unsqueeze(1)).sum(dim=0)
        if query_relevance.numel() > 0
        else torch.zeros(total_tokens, dtype=torch.float32, device=device)
    )
    query_weight = min(0.30, max(0.0, _cfg_float(config, "certv3_query_weight", 0.18) * query_confidence))
    visual_quality = _minmax(
        0.28 * attention
        + 0.20 * novelty
        + 0.14 * curvature
        + 0.12 * event
        + 0.12 * detail
        + 0.14 * component_value,
        dim=0,
    )
    quality = _minmax((1.0 - query_weight) * visual_quality + query_weight * query_score, dim=0)
    event_score = _minmax(
        0.34 * novelty + 0.28 * curvature + 0.18 * event + 0.10 * detail + 0.10 * query_score,
        dim=0,
    )
    demand_weight = 0.20 + 0.42 * quality + 0.20 * event_score + 0.18 * component_value
    demand_weight = demand_weight / demand_weight.sum().clamp_min(1e-6)
    design = _design_features(
        metric_features=metric_flat,
        quality=quality,
        temporal_ids=temporal_ids,
        spatial_ids=spatial_ids,
        attention=attention,
        novelty=novelty,
        curvature=curvature,
        event=event,
        detail=detail,
        component_support=component_value,
        query_relevance=query_relevance,
        atom_weights=atom_weights,
        query_confidence=query_confidence,
        temporal_count=temporal_count,
        spatial_count=spatial_count,
        structural_weight=_cfg_float(config, "certv3_structural_weight", 0.32),
        whitening_strength=_cfg_float(config, "certv3_whitening_strength", 0.50),
        quality_floor=_cfg_float(config, "certv3_quality_floor", 0.15),
    )
    return _LHAnalysis(
        metric_flat=metric_flat,
        metric_frames=metric_frames,
        design=design,
        demand_weight=demand_weight,
        quality=quality,
        event_score=event_score,
        attention=attention,
        query_score=query_score,
        query_relevance=query_relevance,
        atom_weights=atom_weights,
        query_confidence=float(query_confidence),
        component_ids=component_ids,
        frame_ids=frame_ids,
        spatial_ids=spatial_ids,
        novelty=novelty,
        curvature=curvature,
        detail=detail,
        frame_event=frame_event,
        ridge=_cfg_float(config, "certv3_ridge", 0.50),
    )


def _optimal_boundaries(
    semantic_gap: torch.Tensor,
    group_count: int,
    min_units: int,
    max_units: int,
) -> list[int]:
    """Dynamic programming chooses high-change boundaries under size constraints."""
    frame_count = int(semantic_gap.numel())
    negative = float("-inf")
    dp = [[negative] * (frame_count + 1) for _ in range(group_count + 1)]
    parent = [[-1] * (frame_count + 1) for _ in range(group_count + 1)]
    dp[0][0] = 0.0
    gap_values = semantic_gap.detach().float().cpu().tolist()
    for groups in range(1, group_count + 1):
        for end in range(1, frame_count + 1):
            for length in range(min_units, max_units + 1):
                start = end - length
                if start < 0 or not math.isfinite(dp[groups - 1][start]):
                    continue
                boundary_value = 0.0 if start == 0 else float(gap_values[start])
                candidate = dp[groups - 1][start] + boundary_value
                if candidate > dp[groups][end]:
                    dp[groups][end] = candidate
                    parent[groups][end] = start
    if not math.isfinite(dp[group_count][frame_count]):
        return [int(round(index * frame_count / group_count)) for index in range(group_count + 1)]
    boundaries = [frame_count]
    end = frame_count
    for groups in range(group_count, 0, -1):
        end = parent[groups][end]
        boundaries.append(end)
    return sorted(boundaries)


def _build_groups(
    semantic_gap: torch.Tensor,
    config: FlashVidConfig,
) -> tuple[torch.Tensor, list[tuple[int, int]], float]:
    frame_count = int(semantic_gap.numel())
    min_units = max(1, _cfg_int(config, "certlh_min_group_units", 2))
    max_units = max(min_units, _cfg_int(config, "certlh_max_group_units", 8))
    max_possible = max(1, frame_count // min_units)
    min_required = max(1, int(math.ceil(frame_count / float(max_units))))
    min_groups = min(max_possible, max(min_required, _cfg_int(config, "certlh_min_groups", 4)))
    max_groups = min(max_possible, max(min_groups, _cfg_int(config, "certlh_max_groups", 8)))
    event_floor = _cfg_float(config, "certlh_event_floor", 0.08)
    threshold = max(
        event_floor,
        _safe_quantile(semantic_gap[1:], _cfg_float(config, "certlh_event_quantile", 0.80), event_floor),
    )
    event_count = int(torch.sum(semantic_gap[1:] >= threshold).item()) if frame_count > 1 else 0
    group_count = min(max_groups, max(min_groups, event_count + 1))
    while group_count * min_units > frame_count:
        group_count -= 1
    while group_count * max_units < frame_count:
        group_count += 1
    group_count = min(max_groups, max(1, group_count))
    boundaries = _optimal_boundaries(semantic_gap, group_count, min_units, max_units)
    groups = [(boundaries[index], boundaries[index + 1]) for index in range(len(boundaries) - 1)]
    frame_groups = torch.empty(frame_count, dtype=torch.long, device=semantic_gap.device)
    for group, (start, end) in enumerate(groups):
        frame_groups[start:end] = group
    return frame_groups, groups, threshold


def _largest_remainder(total: int, weights: Iterable[float]) -> list[int]:
    weights = [max(0.0, float(value)) for value in weights]
    denominator = sum(weights)
    if denominator <= 1e-12:
        weights = [1.0] * len(weights)
        denominator = float(len(weights))
    raw = [total * value / denominator for value in weights]
    counts = [int(math.floor(value)) for value in raw]
    order = sorted(range(len(raw)), key=lambda index: (-(raw[index] - counts[index]), index))
    for index in order[: total - sum(counts)]:
        counts[index] += 1
    return counts


def _group_scores(
    analysis: _LHAnalysis,
    token_group_ids: torch.Tensor,
    group_count: int,
    config: FlashVidConfig,
) -> list[float]:
    prototypes = []
    for group in range(group_count):
        members = torch.where(token_group_ids == group)[0]
        prototypes.append(F.normalize(analysis.metric_flat[members].mean(dim=0), dim=0, eps=1e-6))
    prototype_matrix = torch.stack(prototypes, dim=0)
    scores: list[float] = []
    query_alpha = min(
        0.50,
        max(0.0, _cfg_float(config, "certlh_query_weight", 0.35) * analysis.query_confidence),
    )
    for group in range(group_count):
        members = torch.where(token_group_ids == group)[0]
        other = torch.cat([prototype_matrix[:group], prototype_matrix[group + 1 :]], dim=0)
        if other.numel() == 0:
            uniqueness = 1.0
        else:
            similarity = analysis.metric_flat[members] @ other.transpose(0, 1)
            novelty = 1.0 - similarity.max(dim=1).values
            topk = max(1, int(math.ceil(0.10 * members.numel())))
            uniqueness = float(torch.topk(novelty, k=topk).values.mean().item())
        query = float(analysis.query_score[members].max().item()) if analysis.query_score.numel() else 0.0
        scores.append((1.0 - query_alpha) * uniqueness + query_alpha * query)
    return scores


def _allocate_group_budgets(
    total: int,
    capacities: list[int],
    frame_counts: list[int],
    scores: list[float],
    config: FlashVidConfig,
) -> list[int]:
    group_count = len(capacities)
    equal = total / float(max(1, group_count))
    floor_ratio = min(1.0, max(0.0, _cfg_float(config, "certlh_group_floor_ratio", 0.50)))
    budgets = [
        min(capacity, max(1, frame_count, int(math.floor(equal * floor_ratio))))
        for capacity, frame_count in zip(capacities, frame_counts)
    ]
    while sum(budgets) > total:
        removable = [index for index, value in enumerate(budgets) if value > 1]
        if not removable:
            break
        index = max(removable, key=lambda item: (budgets[item], -item))
        budgets[index] -= 1

    remaining = total - sum(budgets)
    temperature = max(1e-4, _cfg_float(config, "certlh_budget_temperature", 0.25))
    score_tensor = torch.tensor(scores, dtype=torch.float64)
    weights = torch.softmax((score_tensor - score_tensor.mean()) / temperature, dim=0).tolist()
    while remaining > 0:
        active = [index for index in range(group_count) if budgets[index] < capacities[index]]
        if not active:
            break
        allocation = _largest_remainder(remaining, [weights[index] for index in active])
        progress = 0
        for local, index in enumerate(active):
            add = min(allocation[local], capacities[index] - budgets[index])
            budgets[index] += add
            progress += add
        remaining -= progress
        if progress == 0:
            index = max(active, key=lambda item: (weights[item], -item))
            budgets[index] += 1
            remaining -= 1
    if sum(budgets) != total:
        raise RuntimeError(f"CertVID-LH allocated {sum(budgets)} local tokens for budget {total}")
    return budgets


def _select_group(
    analysis: _LHAnalysis,
    members: torch.Tensor,
    group_budget: int,
    group_start: int,
    group_frames: int,
    config: FlashVidConfig,
    swap_steps: int,
) -> tuple[torch.Tensor, set[int]]:
    if group_budget >= members.numel():
        values = torch.sort(members).values
        return values, set(values.detach().cpu().tolist())
    local_frame_ids = analysis.frame_ids[members] - group_start
    temporal_count = min(group_frames, max(1, _cfg_int(config, "certv3_temporal_bins", 12)))
    temporal_ids = torch.div(
        local_frame_ids * temporal_count,
        max(1, group_frames),
        rounding_mode="floor",
    ).clamp_max(temporal_count - 1)
    spatial_count = int(analysis.spatial_ids.max().item()) + 1
    local_query = analysis.query_relevance[:, members] if analysis.query_relevance.numel() else analysis.query_relevance
    mandatory_local, _ = _hard_certificates(
        budget=group_budget,
        quality=analysis.quality[members],
        event_score=analysis.event_score[members],
        frame_ids=local_frame_ids,
        temporal_ids=temporal_ids,
        spatial_ids=analysis.spatial_ids[members],
        query_relevance=local_query,
        atom_weights=analysis.atom_weights,
        query_confidence=analysis.query_confidence,
        frame_count=group_frames,
        temporal_count=temporal_count,
        spatial_count=spatial_count,
        frame_coverage_ratio=_cfg_float(config, "certv3_frame_coverage_ratio", 1.0),
        cell_coverage_ratio=_cfg_float(config, "certv3_cell_coverage_ratio", 0.50),
        query_threshold=_cfg_float(config, "certv3_query_threshold", 0.10),
        query_per_atom=_cfg_int(config, "certv3_query_per_atom", 1),
    )
    candidates_local = _candidate_pool(
        budget=group_budget,
        quality=analysis.quality[members],
        component_ids=analysis.component_ids[members],
        temporal_ids=temporal_ids,
        spatial_ids=analysis.spatial_ids[members],
        query_relevance=local_query,
        mandatory=mandatory_local,
        multiplier=_cfg_float(config, "certv3_candidate_multiplier", 2.5),
    )
    selected_local = _d_optimal_greedy(
        design=analysis.design[members],
        candidates=candidates_local,
        mandatory=mandatory_local,
        budget=group_budget,
        ridge=analysis.ridge,
    )
    selected_local, _, _ = _swap_refine(
        selected=selected_local,
        candidates=candidates_local,
        design=analysis.design[members],
        mandatory=mandatory_local,
        ridge=analysis.ridge,
        steps=swap_steps,
        pool_size=_cfg_int(config, "certv3_swap_pool", 24),
        margin=_cfg_float(config, "certv3_swap_margin", 1e-4),
    )
    selected = torch.sort(members[selected_local]).values
    mandatory = {int(members[index].item()) for index in mandatory_local}
    return selected, mandatory


def _balanced_order(score: torch.Tensor, group_ids: torch.Tensor, group_count: int) -> list[int]:
    score_cpu = score.detach().float().cpu().tolist()
    queues: list[list[int]] = []
    for group in range(group_count):
        members = torch.where(group_ids == group)[0]
        local = torch.argsort(score[members], descending=True, stable=True)
        queues.append(members[local].detach().cpu().tolist())
    output: list[int] = []
    max_length = max((len(queue) for queue in queues), default=0)
    for rank in range(max_length):
        offers = [
            (-float(score_cpu[queue[rank]]), group, int(queue[rank]))
            for group, queue in enumerate(queues)
            if rank < len(queue)
        ]
        offers.sort()
        output.extend(token for _, _, token in offers)
    return output


def _multi_peak_order(
    analysis: _LHAnalysis,
    token_group_ids: torch.Tensor,
    group_count: int,
    config: FlashVidConfig,
) -> list[int]:
    if analysis.query_relevance.numel() == 0:
        return []
    max_peaks = max(1, _cfg_int(config, "certlh_query_peaks_per_atom", 2))
    min_distance = max(1, _cfg_int(config, "certlh_query_min_group_distance", 2))
    offers: list[tuple[float, int, int]] = []
    for atom, atom_scores in enumerate(analysis.query_relevance):
        threshold = max(
            _cfg_float(config, "certlh_query_peak_floor", 0.75),
            _safe_quantile(
                atom_scores,
                _cfg_float(config, "certlh_query_peak_quantile", 0.90),
                _cfg_float(config, "certlh_query_peak_floor", 0.75),
            ),
        )
        group_peaks: list[tuple[float, int, int]] = []
        for group in range(group_count):
            members = torch.where(token_group_ids == group)[0]
            local = int(torch.argmax(atom_scores[members]).item())
            token = int(members[local].item())
            score = float(atom_scores[token].item())
            if score >= threshold:
                group_peaks.append((score, group, token))
        group_peaks.sort(key=lambda item: (-item[0], item[1], item[2]))
        chosen: list[tuple[float, int, int]] = []
        for candidate in group_peaks:
            if all(abs(candidate[1] - previous[1]) >= min_distance for previous in chosen):
                chosen.append(candidate)
            if len(chosen) >= max_peaks:
                break
        if len(chosen) >= 2:
            atom_weight = float(analysis.atom_weights[atom].item())
            offers.extend((score * atom_weight, group, token) for score, group, token in chosen)
    offers.sort(key=lambda item: (-item[0], item[1], item[2]))
    return list(dict.fromkeys(token for _, _, token in offers))


def _fill_unique(
    selected: set[int],
    output: list[int],
    order: Iterable[int],
    target: int,
) -> None:
    for token in order:
        if len(output) >= target:
            return
        token = int(token)
        if token in selected:
            continue
        selected.add(token)
        output.append(token)


def _select_relays(
    analysis: _LHAnalysis,
    local_selected: torch.Tensor,
    token_group_ids: torch.Tensor,
    groups: list[tuple[int, int]],
    semantic_gap: torch.Tensor,
    relay_budget: int,
    config: FlashVidConfig,
) -> tuple[torch.Tensor, dict[str, int]]:
    if relay_budget <= 0:
        return torch.empty(0, dtype=torch.long, device=analysis.metric_flat.device), {
            "query": 0,
            "boundary": 0,
            "transition": 0,
            "context": 0,
            "fill": 0,
        }
    selected = set(local_selected.detach().cpu().tolist())
    relay: list[int] = []
    group_count = len(groups)
    query_order = _multi_peak_order(analysis, token_group_ids, group_count, config)
    _fill_unique(selected, relay, query_order, relay_budget)
    query_count = len(relay)

    remaining = relay_budget - len(relay)
    boundary_quota, transition_quota, context_quota = _largest_remainder(remaining, (0.30, 0.45, 0.25))
    transition_score = _minmax(0.55 * analysis.novelty + 0.30 * analysis.curvature + 0.15 * analysis.detail, dim=0)
    frame_boundary = torch.zeros_like(semantic_gap)
    for start, _ in groups[1:]:
        frame_boundary[start - 1] = torch.maximum(frame_boundary[start - 1], semantic_gap[start])
        frame_boundary[start] = torch.maximum(frame_boundary[start], semantic_gap[start])
    boundary_score = frame_boundary.index_select(0, analysis.frame_ids) * (
        0.55 * transition_score + 0.45 * analysis.quality
    )
    context_score = _minmax(
        0.65 * analysis.quality + 0.35 * _minmax(analysis.demand_weight, dim=0),
        dim=0,
    )

    before = len(relay)
    _fill_unique(
        selected,
        relay,
        _balanced_order(boundary_score, token_group_ids, group_count),
        before + boundary_quota,
    )
    boundary_count = len(relay) - before
    before = len(relay)
    _fill_unique(
        selected,
        relay,
        _balanced_order(transition_score, token_group_ids, group_count),
        before + transition_quota,
    )
    transition_count = len(relay) - before
    before = len(relay)
    _fill_unique(
        selected,
        relay,
        _balanced_order(context_score, token_group_ids, group_count),
        before + context_quota,
    )
    context_count = len(relay) - before

    combined = (
        _balanced_order(boundary_score, token_group_ids, group_count)
        + _balanced_order(transition_score, token_group_ids, group_count)
        + _balanced_order(context_score, token_group_ids, group_count)
        + torch.argsort(analysis.quality, descending=True, stable=True).detach().cpu().tolist()
    )
    before = len(relay)
    _fill_unique(selected, relay, combined, relay_budget)
    fill_count = len(relay) - before
    if len(relay) != relay_budget:
        raise RuntimeError(f"CertVID-LH selected {len(relay)} relay tokens for budget {relay_budget}")
    return torch.tensor(relay, dtype=torch.long, device=analysis.metric_flat.device), {
        "query": query_count,
        "boundary": boundary_count,
        "transition": transition_count,
        "context": context_count,
        "fill": fill_count,
    }


def _build_constrained_plan(
    *,
    selected: torch.Tensor,
    analysis: _LHAnalysis,
    token_group_ids: torch.Tensor,
    frame_times: torch.Tensor,
    relay_tokens: torch.Tensor,
    mandatory_tokens: set[int],
    config: FlashVidConfig,
) -> tuple[CertVidPlan, int]:
    total_tokens = int(analysis.metric_flat.shape[0])
    budget = int(selected.numel())
    similarity = analysis.metric_flat @ analysis.metric_flat[selected].transpose(0, 1)
    source_group = token_group_ids.unsqueeze(1)
    anchor_group = token_group_ids[selected].unsqueeze(0)
    group_distance = (source_group - anchor_group).abs()
    same_group = group_distance == 0
    adjacent = group_distance == 1
    source_time = frame_times.index_select(0, analysis.frame_ids).unsqueeze(1)
    anchor_time = frame_times.index_select(0, analysis.frame_ids[selected]).unsqueeze(0)
    time_valid = (source_time - anchor_time).abs() <= _cfg_float(config, "certlh_cross_group_max_seconds", 8.0)
    similarity_valid = similarity >= _cfg_float(config, "certlh_cross_group_similarity", 0.90)
    valid = same_group | (adjacent & time_valid & similarity_valid)
    similarity = similarity.masked_fill(~valid, -1e4)
    same_component = analysis.component_ids.unsqueeze(1) == analysis.component_ids[selected].unsqueeze(0)
    similarity = similarity + 0.08 * same_component.float()
    topk = min(2, budget)
    values, assignment = torch.topk(similarity, k=topk, dim=1, largest=True)
    weights = torch.softmax(values.float() / max(1e-4, _cfg_float(config, "certv3_assignment_temperature", 0.07)), dim=1)

    anchor_positions = torch.arange(budget, dtype=torch.long, device=selected.device)
    assignment[selected, 0] = anchor_positions
    weights[selected] = 0.0
    weights[selected, 0] = 1.0
    source_mass = (0.5 + 0.5 * analysis.demand_weight * float(total_tokens)).clamp(0.25, 2.0)
    protection = torch.maximum(analysis.attention[selected], analysis.query_score[selected])
    protected_count = min(budget, max(1, int(math.ceil(0.15 * budget))))
    protected = torch.zeros(budget, dtype=torch.bool, device=selected.device)
    protected[torch.topk(protection, k=protected_count, largest=True).indices] = True
    alpha = torch.full(
        (budget,),
        min(max(_cfg_float(config, "certv3_fusion_alpha", 0.12), 0.0), 0.75),
        dtype=torch.float32,
        device=selected.device,
    )
    alpha = alpha * (1.0 - 0.65 * protection.clamp(0.0, 1.0))
    alpha[protected] = 0.0
    locked = set(mandatory_tokens)
    locked.update(relay_tokens.detach().cpu().tolist())
    if locked:
        locked_tensor = torch.tensor(sorted(locked), dtype=torch.long, device=selected.device)
        alpha[torch.isin(selected, locked_tensor)] = 0.0

    assigned_anchor_groups = token_group_ids[selected][assignment]
    cross_edges = int(torch.sum((assigned_anchor_groups != token_group_ids.unsqueeze(1)) & (weights > 1e-6)).item())
    return CertVidPlan(
        anchor_indices=selected,
        assignment_indices=assignment,
        assignment_weights=weights,
        source_mass=source_mass,
        fusion_alpha=alpha,
        raw_token_count=total_tokens,
    ), cross_edges


def _store_diagnostics(config: FlashVidConfig, diagnostics: dict[str, Any]) -> None:
    config.last_certlh_diagnostics = diagnostics
    for name in (
        "target_tokens",
        "local_tokens",
        "relay_tokens",
        "group_count",
        "duration_seconds",
        "max_timestamp_gap",
        "horizon_score",
        "cross_group_edges",
    ):
        setattr(config, f"last_certlh_{name}", float(diagnostics.get(name, 0.0)))
    if bool(getattr(config, "certlh_debug", False)):
        print(
            "[CertVID-LH] "
            f"mode={diagnostics.get('mode', 'fallback')} "
            f"duration={float(diagnostics.get('duration_seconds', 0.0)):.1f}s "
            f"groups={int(diagnostics.get('group_count', 0))} "
            f"budget={int(diagnostics.get('target_tokens', 0))} "
            f"local/relay={int(diagnostics.get('local_tokens', 0))}/"
            f"{int(diagnostics.get('relay_tokens', 0))}"
        )


def _v3_fallback(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    config: FlashVidConfig,
    question_features: Optional[torch.Tensor],
    diagnostics: dict[str, Any],
    reason: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    output, indices = certvid_v3_compression(video_features, cls_attention, config, question_features)
    diagnostics.update(
        {
            "mode": "v3",
            "fallback_reason": reason,
            "target_tokens": int(indices.numel()),
            "local_tokens": int(indices.numel()),
            "relay_tokens": 0,
            "group_count": 0,
            "cross_group_edges": 0,
        }
    )
    config.last_adapter_variant = "certvid_lh"
    _store_diagnostics(config, diagnostics)
    return output, indices


def certvid_lh_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
    question_features: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply exact V3 to short videos and long-horizon allocation otherwise."""
    if video_features.ndim != 3:
        raise ValueError(f"expected video_features [T, HW, D], got {tuple(video_features.shape)}")
    frame_count, tokens_per_frame, _ = video_features.shape
    total_tokens = frame_count * tokens_per_frame
    frame_times, timing_error = _normalize_frame_times(
        getattr(flashvid_config, "_certvid_frame_times_sec", None),
        frame_count,
        device=video_features.device,
    )
    diagnostics: dict[str, Any] = {
        "timestamp_source": str(getattr(flashvid_config, "_certvid_frame_times_source", "missing")),
        "raw_tokens": total_tokens,
    }
    if frame_times is None:
        return _v3_fallback(
            video_features,
            cls_attention,
            flashvid_config,
            question_features,
            diagnostics,
            timing_error or "invalid_timestamps",
        )

    duration_seconds = float((frame_times[-1] - frame_times[0]).item()) if frame_count > 1 else 0.0
    max_timestamp_gap = (
        float((frame_times[1:] - frame_times[:-1]).max().item()) if frame_count > 1 else 0.0
    )
    diagnostics.update(
        {
            "duration_seconds": duration_seconds,
            "max_timestamp_gap": max_timestamp_gap,
        }
    )
    if duration_seconds < max(1.0, _cfg_float(flashvid_config, "certlh_min_duration_seconds", 120.0)):
        return _v3_fallback(
            video_features,
            cls_attention,
            flashvid_config,
            question_features,
            diagnostics,
            "short_duration",
        )

    metric_dim = max(32, _cfg_int(flashvid_config, "certv3_metric_dim", 96))
    gate_metric_flat = _metric_features(video_features, metric_dim)
    gate_metric = gate_metric_flat.view(frame_count, tokens_per_frame, -1)
    gate_atoms = _question_atoms(
        question_features,
        max(0, _cfg_int(flashvid_config, "certv3_query_atoms", 8)),
        metric_dim,
    ).to(video_features.device)
    gate_query_state = _question_relevance(gate_atoms, gate_metric_flat)
    gate_query, _, _ = gate_query_state
    gate = _gate_metrics(
        metric_frames=gate_metric,
        cls_attention=cls_attention,
        query_relevance=gate_query,
        frame_times=frame_times,
        config=flashvid_config,
    )
    diagnostics.update(gate)
    if not bool(gate["enabled"]):
        return _v3_fallback(
            video_features,
            cls_attention,
            flashvid_config,
            question_features,
            diagnostics,
            "short_or_low_horizon",
        )

    budget = max(1, min(total_tokens, int(round(total_tokens * _effective_ratio(flashvid_config)))))
    if budget >= total_tokens:
        return _v3_fallback(
            video_features,
            cls_attention,
            flashvid_config,
            question_features,
            diagnostics,
            "identity_budget",
        )

    try:
        analysis = _build_analysis(
            video_features,
            cls_attention,
            question_features,
            flashvid_config,
            precomputed_metric_flat=gate_metric_flat,
            precomputed_query=gate_query_state,
        )
        _, semantic_gap = _frame_semantic_gap(analysis.metric_frames)
        frame_groups, groups, semantic_threshold = _build_groups(semantic_gap, flashvid_config)
        group_count = len(groups)
        token_group_ids = frame_groups.repeat_interleave(tokens_per_frame)
        if budget < group_count:
            raise RuntimeError(
                f"token budget {budget} cannot preserve one anchor in each of {group_count} temporal groups"
            )
        relay_ratio = min(0.40, max(0.0, _cfg_float(flashvid_config, "certlh_relay_ratio", 0.10)))
        relay_budget = max(0, min(budget - group_count, int(round(budget * relay_ratio))))
        local_budget = budget - relay_budget
        capacities = [(end - start) * tokens_per_frame for start, end in groups]
        frame_counts = [end - start for start, end in groups]
        scores = _group_scores(analysis, token_group_ids, group_count, flashvid_config)
        group_budgets = _allocate_group_budgets(
            local_budget,
            capacities,
            frame_counts,
            scores,
            flashvid_config,
        )
        swap_budgets = _largest_remainder(
            max(0, _cfg_int(flashvid_config, "certv3_swap_steps", 6)),
            group_budgets,
        )
        local_parts: list[torch.Tensor] = []
        mandatory_tokens: set[int] = set()
        for group, ((start, end), group_budget) in enumerate(zip(groups, group_budgets)):
            members = torch.where(token_group_ids == group)[0]
            chosen, mandatory = _select_group(
                analysis,
                members,
                group_budget,
                start,
                end - start,
                flashvid_config,
                swap_budgets[group],
            )
            local_parts.append(chosen)
            mandatory_tokens.update(mandatory)
        local_selected = torch.sort(torch.cat(local_parts, dim=0)).values
        relay_tokens, relay_counts = _select_relays(
            analysis,
            local_selected,
            token_group_ids,
            groups,
            semantic_gap,
            relay_budget,
            flashvid_config,
        )
        selected = torch.sort(torch.cat([local_selected, relay_tokens], dim=0)).values
        if selected.numel() != budget or selected.unique().numel() != budget:
            raise RuntimeError("CertVID-LH did not produce an exact unique budget")
        plan, cross_edges = _build_constrained_plan(
            selected=selected,
            analysis=analysis,
            token_group_ids=token_group_ids,
            frame_times=frame_times,
            relay_tokens=relay_tokens,
            mandatory_tokens=mandatory_tokens,
            config=flashvid_config,
        )
        output = apply_certvid_plan(video_features.reshape(total_tokens, -1), plan)
    except (RuntimeError, ValueError) as error:
        diagnostics["long_mode_error"] = str(error)
        return _v3_fallback(
            video_features,
            cls_attention,
            flashvid_config,
            question_features,
            diagnostics,
            "long_mode_safe_fallback",
        )

    diagnostics.update(
        {
            "mode": "long_horizon",
            "fallback_reason": None,
            "target_tokens": budget,
            "local_tokens": local_budget,
            "relay_tokens": relay_budget,
            "relay_query_tokens": relay_counts["query"],
            "relay_boundary_tokens": relay_counts["boundary"],
            "relay_transition_tokens": relay_counts["transition"],
            "relay_context_tokens": relay_counts["context"],
            "relay_fill_tokens": relay_counts["fill"],
            "group_count": group_count,
            "group_boundaries": groups,
            "group_budgets": group_budgets,
            "group_scores": scores,
            "semantic_threshold": semantic_threshold,
            "cross_group_edges": cross_edges,
            "modified_ratio": 1.0,
        }
    )
    flashvid_config._certvid_plan = plan
    flashvid_config.vision_token_length = int(output.shape[0])
    flashvid_config.visual_token_length = int(output.shape[0])
    flashvid_config.llm_token_length = None
    flashvid_config.last_adapter_variant = "certvid_lh"
    flashvid_config.last_adapter_raw_tokens = float(total_tokens)
    flashvid_config.last_adapter_output_tokens = float(output.shape[0])
    _store_diagnostics(flashvid_config, diagnostics)
    return output, plan.anchor_indices


__all__ = ["certvid_lh_compression"]
