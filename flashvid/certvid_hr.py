"""CertVID-HR: conservative long-horizon repair on top of CertVID V3.

The V3 selector and residual plan remain the source of truth.  HR only swaps a
small number of unprotected anchors when reliable wall-clock timestamps expose
a concrete segment-coverage or multi-hop query deficit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from .certvid import (
    _build_components,
    _build_plan,
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
from .certvid_v3 import _design_features, certvid_v3_compression


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


def _safe_quantile(values: torch.Tensor, quantile: float, default: float = 0.0) -> float:
    values = values.float().reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return float(default)
    quantile = min(1.0, max(0.0, float(quantile)))
    return float(torch.quantile(values, quantile).item())


def _normalize_frame_times(
    raw_times: Any,
    frame_count: int,
    *,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    """Validate timestamps and map contiguous input-frame groups to visual units."""

    if raw_times is None:
        return None, "missing_timestamps"
    try:
        times = torch.as_tensor(raw_times, dtype=torch.float64, device=device).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        return None, "invalid_timestamp_type"
    if times.numel() == 0 or frame_count <= 0:
        return None, "empty_timestamps"
    if not torch.isfinite(times).all():
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


def _build_horizon_chunks(
    metric_frames: torch.Tensor,
    frame_times: torch.Tensor,
    *,
    max_seconds: float,
    max_units: int,
    semantic_quantile: float,
    semantic_floor: float,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Split visual units at semantic boundaries or explicit horizon limits."""

    frame_repr = F.normalize(metric_frames.float().mean(dim=1), dim=-1, eps=1e-6)
    semantic_gap = torch.zeros(frame_repr.shape[0], device=frame_repr.device)
    if frame_repr.shape[0] > 1:
        semantic_gap[1:] = 1.0 - (frame_repr[1:] * frame_repr[:-1]).sum(dim=-1)
    semantic_threshold = max(
        float(semantic_floor),
        _safe_quantile(semantic_gap[1:], semantic_quantile, semantic_floor),
    )

    chunk_ids = torch.zeros(frame_repr.shape[0], dtype=torch.long, device=frame_repr.device)
    chunk = 0
    chunk_start = 0
    for frame in range(1, frame_repr.shape[0]):
        duration = float(frame_times[frame].item() - frame_times[chunk_start].item())
        units = frame - chunk_start
        split = (
            float(semantic_gap[frame].item()) >= semantic_threshold
            or duration >= float(max_seconds)
            or units >= max(1, int(max_units))
        )
        if split:
            chunk += 1
            chunk_start = frame
        chunk_ids[frame] = chunk
    return chunk_ids, semantic_gap, semantic_threshold


def _chunk_coverage(
    metric_flat: torch.Tensor,
    selected: torch.Tensor,
    token_chunk_ids: torch.Tensor,
    chunk_count: int,
) -> torch.Tensor:
    coverage = torch.zeros(chunk_count, dtype=torch.float32, device=metric_flat.device)
    for chunk in range(chunk_count):
        members = torch.nonzero(token_chunk_ids == chunk, as_tuple=False).reshape(-1)
        anchors = selected[token_chunk_ids.index_select(0, selected) == chunk]
        if members.numel() == 0 or anchors.numel() == 0:
            continue
        similarity = metric_flat.index_select(0, members) @ metric_flat.index_select(0, anchors).T
        coverage[chunk] = similarity.max(dim=1).values.mean()
    return coverage


def _query_requirements(
    query_relevance: torch.Tensor,
    query_confidence: float,
    token_chunk_ids: torch.Tensor,
    selected: torch.Tensor,
    *,
    confidence_threshold: float,
    peak_quantile: float,
    peak_floor: float,
) -> Tuple[List[Tuple[int, int, float]], List[Tuple[int, int, float]]]:
    """Return multi-hop requirements and the currently missing subset."""

    if query_relevance.numel() == 0 or query_confidence < float(confidence_threshold):
        return [], []
    chunk_count = int(token_chunk_ids.max().item()) + 1
    requirements: List[Tuple[int, int, float]] = []
    missing: List[Tuple[int, int, float]] = []
    selected_chunks = token_chunk_ids.index_select(0, selected)
    for atom in range(query_relevance.shape[0]):
        relevance = query_relevance[atom]
        threshold = max(float(peak_floor), _safe_quantile(relevance, peak_quantile, peak_floor))
        active: List[int] = []
        for chunk in range(chunk_count):
            members = torch.nonzero(token_chunk_ids == chunk, as_tuple=False).reshape(-1)
            if members.numel() and float(relevance.index_select(0, members).max().item()) >= threshold:
                active.append(chunk)
        if len(active) < 2 or max(active) - min(active) < 2:
            continue
        for chunk in active:
            requirement = (atom, chunk, threshold)
            requirements.append(requirement)
            chunk_selected = selected[selected_chunks == chunk]
            covered = (
                chunk_selected.numel() > 0
                and float(relevance.index_select(0, chunk_selected).max().item()) >= threshold
            )
            if not covered:
                missing.append(requirement)
    return requirements, missing


def _v3_analysis(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    question_features: Optional[torch.Tensor],
    config: Any,
) -> _V3Analysis:
    """Rebuild V3's analysis tensors without changing V3 implementation."""

    frames, tokens_per_frame, _ = video_features.shape
    total_tokens = frames * tokens_per_frame
    device = video_features.device
    metric_dim = max(32, int(getattr(config, "certv3_metric_dim", 96)))
    metric_flat = _metric_features(video_features, metric_dim)
    metric_frames = metric_flat.view(frames, tokens_per_frame, -1)
    grid_h, grid_w = _grid_hw(tokens_per_frame, config)
    spatial_bins = max(1, int(getattr(config, "certv3_spatial_bins", 3)))
    coords, spatial_ids_frame = _spatial_layout(
        tokens_per_frame,
        grid_h,
        grid_w,
        spatial_bins,
        device,
    )
    spatial_ids = spatial_ids_frame.repeat(frames)

    frame_event, _, novelty_2d, curvature_2d, matches = _trajectory_signals(
        metric_frames,
        coords,
        float(getattr(config, "certv3_spatial_penalty", 0.08)),
    )
    component_ids_cpu, component_sizes_cpu = _build_components(
        frames,
        tokens_per_frame,
        frame_event,
        matches,
        float(getattr(config, "certv3_track_threshold", 0.82)),
    )
    component_ids = component_ids_cpu.to(device)
    component_sizes = component_sizes_cpu.to(device)
    frame_ids = torch.arange(frames, device=device).repeat_interleave(tokens_per_frame)
    component_score = _component_support(
        metric_flat,
        component_ids,
        component_sizes,
        frame_ids,
        frames,
    )
    temporal_bins = max(1, min(int(getattr(config, "certv3_temporal_bins", 12)), frames))
    temporal_ids = torch.div(frame_ids * temporal_bins, max(1, frames), rounding_mode="floor").clamp_max(
        temporal_bins - 1
    )

    attention = _rank_normalize(cls_attention.float()).reshape(-1)
    detail = _local_detail(video_features, grid_h, grid_w).reshape(-1)
    novelty = novelty_2d.reshape(-1)
    curvature = curvature_2d.reshape(-1)
    event = frame_event.index_select(0, frame_ids)
    query_atoms = _question_atoms(
        question_features,
        int(getattr(config, "certv3_query_atoms", 8)),
        metric_dim,
    ).to(device)
    query_relevance, atom_weights, query_confidence = _question_relevance(query_atoms, metric_flat)
    query_score = (
        (query_relevance * atom_weights.unsqueeze(1)).sum(dim=0)
        if query_relevance.numel() > 0
        else torch.zeros(total_tokens, dtype=torch.float32, device=device)
    )

    query_weight = min(
        0.30,
        max(0.0, float(getattr(config, "certv3_query_weight", 0.18)) * query_confidence),
    )
    visual_quality = _minmax(
        0.28 * attention
        + 0.20 * novelty
        + 0.14 * curvature
        + 0.12 * event
        + 0.12 * detail
        + 0.14 * component_score,
        dim=0,
    )
    quality = _minmax((1.0 - query_weight) * visual_quality + query_weight * query_score, dim=0)
    event_score = _minmax(
        0.34 * novelty + 0.28 * curvature + 0.18 * event + 0.10 * detail + 0.10 * query_score,
        dim=0,
    )
    demand_weight = 0.20 + 0.42 * quality + 0.20 * event_score + 0.18 * component_score
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
        component_support=component_score,
        query_relevance=query_relevance,
        atom_weights=atom_weights,
        query_confidence=query_confidence,
        temporal_count=temporal_bins,
        spatial_count=spatial_bins * spatial_bins,
        structural_weight=float(getattr(config, "certv3_structural_weight", 0.32)),
        whitening_strength=float(getattr(config, "certv3_whitening_strength", 0.50)),
        quality_floor=float(getattr(config, "certv3_quality_floor", 0.15)),
    )
    return _V3Analysis(
        metric_flat=metric_flat,
        design=design,
        demand_weight=demand_weight,
        attention=attention,
        query_score=query_score,
        query_relevance=query_relevance,
        query_confidence=float(query_confidence),
        component_ids=component_ids,
        frame_ids=frame_ids,
        temporal_ids=temporal_ids,
        ridge=float(getattr(config, "certv3_ridge", 0.50)),
    )


def _analysis_from_sink(sink: Dict[str, Any]) -> _V3Analysis:
    """Build the HR view over intermediates captured by the exact V3 pass."""

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
        raise RuntimeError(f"CertVID-HR missing captured V3 tensors: {missing}")
    return _V3Analysis(**{name: sink[name] for name in required})


def _information_state(
    design: torch.Tensor,
    selected: torch.Tensor,
    ridge: float,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    dim = design.shape[1]
    information = torch.eye(dim, dtype=torch.float32, device=design.device) * float(ridge)
    rows = design.index_select(0, selected).float()
    information = information + rows.T @ rows
    chol, info = torch.linalg.cholesky_ex(information)
    if int(info.item()) != 0:
        raise RuntimeError("CertVID-HR information matrix is not positive definite")
    inverse = torch.cholesky_solve(torch.eye(dim, device=design.device), chol)
    logdet = float((2.0 * torch.log(torch.diagonal(chol))).sum().item())
    return information, inverse, logdet


def _post_removal_coverage(
    metric_flat: torch.Tensor,
    selected: torch.Tensor,
    token_chunk_ids: torch.Tensor,
    chunk_count: int,
) -> torch.Tensor:
    """Compute every single-anchor removal score with one matrix per chunk."""

    scores = torch.full(
        (metric_flat.shape[0],),
        float("-inf"),
        dtype=torch.float32,
        device=metric_flat.device,
    )
    selected_chunks = token_chunk_ids.index_select(0, selected)
    for chunk in range(chunk_count):
        members = torch.nonzero(token_chunk_ids == chunk, as_tuple=False).reshape(-1)
        anchors = selected[selected_chunks == chunk]
        if members.numel() == 0 or anchors.numel() <= 1:
            continue
        similarity = metric_flat.index_select(0, members) @ metric_flat.index_select(0, anchors).T
        top = torch.topk(similarity, k=2, dim=1, largest=True)
        winners = top.indices[:, 0]
        best = top.values[:, 0]
        second = top.values[:, 1]
        positions = torch.arange(anchors.numel(), device=anchors.device)
        post = torch.where(
            winners.unsqueeze(1) == positions.unsqueeze(0),
            second.unsqueeze(1),
            best.unsqueeze(1),
        ).mean(dim=0)
        scores.index_copy_(0, anchors, post)
    return scores


def _missing_query_requirements(
    requirements: List[Tuple[int, int, float]],
    query_relevance: torch.Tensor,
    selected: torch.Tensor,
    token_chunk_ids: torch.Tensor,
) -> List[Tuple[int, int, float]]:
    return [
        requirement
        for requirement in requirements
        if not _requirement_covered(requirement, query_relevance, selected, token_chunk_ids)
    ]


def _query_critical_tokens(
    requirements: List[Tuple[int, int, float]],
    query_relevance: torch.Tensor,
    selected: torch.Tensor,
    selected_chunks: torch.Tensor,
) -> Set[int]:
    """Return anchors whose removal would break a currently met query certificate."""

    critical: Set[int] = set()
    for atom, chunk, threshold in requirements:
        anchors = selected[selected_chunks == chunk]
        if anchors.numel() == 0:
            continue
        qualified = anchors[query_relevance[atom].index_select(0, anchors) >= float(threshold)]
        if qualified.numel() == 1:
            critical.add(int(qualified[0].item()))
    return critical


def _addition_pool(
    analysis: _V3Analysis,
    selected: torch.Tensor,
    members: torch.Tensor,
    current_similarity: torch.Tensor,
    current_coverage: torch.Tensor,
    target_requirement: Optional[Tuple[int, int, float]],
    pool_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score every addition with one batched similarity matrix."""

    candidates = members[~torch.isin(members, selected)]
    if candidates.numel() == 0:
        empty_float = torch.empty(0, dtype=torch.float32, device=members.device)
        return candidates, empty_float, empty_float, empty_float

    member_features = analysis.metric_flat.index_select(0, members)
    candidate_features = analysis.metric_flat.index_select(0, candidates)
    candidate_similarity = member_features @ candidate_features.T
    post_coverage = torch.maximum(current_similarity.unsqueeze(1), candidate_similarity).mean(dim=0)
    coverage_gain = post_coverage - current_coverage
    query_gain = torch.zeros_like(coverage_gain)
    if target_requirement is not None:
        atom, _, threshold = target_requirement
        query_gain = (
            analysis.query_relevance[atom].index_select(0, candidates) - float(threshold)
        ).clamp_min(0.0)
    gain = coverage_gain + 0.5 * query_gain

    # candidates inherit ascending global indices from members; stable sorting
    # therefore preserves the original gain-then-index tie break.
    order = torch.argsort(gain, descending=True, stable=True)[: max(1, int(pool_size))]
    return (
        candidates.index_select(0, order),
        gain.index_select(0, order),
        coverage_gain.index_select(0, order),
        query_gain.index_select(0, order),
    )


def _removal_pool(
    analysis: _V3Analysis,
    selected: torch.Tensor,
    selected_chunks: torch.Tensor,
    protected: Set[int],
    requirements: List[Tuple[int, int, float]],
    token_chunk_ids: torch.Tensor,
    target_chunk: int,
    coverage_target: float,
    deficit_threshold: float,
    information_inverse: torch.Tensor,
    pool_size: int,
    chunk_count: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Filter and rank removable anchors without per-anchor GPU syncs."""

    post_coverage = _post_removal_coverage(
        analysis.metric_flat,
        selected,
        token_chunk_ids,
        chunk_count,
    ).index_select(0, selected)
    chunk_sizes = torch.bincount(selected_chunks, minlength=chunk_count)
    eligible = selected_chunks != int(target_chunk)
    eligible &= chunk_sizes.index_select(0, selected_chunks) > 1
    eligible &= post_coverage >= float(coverage_target - deficit_threshold)

    locked = set(protected)
    locked.update(
        _query_critical_tokens(
            requirements,
            analysis.query_relevance,
            selected,
            selected_chunks,
        )
    )
    if locked:
        locked_tensor = torch.tensor(sorted(locked), dtype=torch.long, device=selected.device)
        eligible &= ~torch.isin(selected, locked_tensor)

    rows = analysis.design.index_select(0, selected).float()
    leverage = torch.sum((rows @ information_inverse) * rows, dim=1)
    denominator = 1.0 - leverage
    eligible &= denominator > 1e-6
    positions = torch.nonzero(eligible, as_tuple=False).reshape(-1)
    if positions.numel() == 0:
        return selected.new_empty(0), rows.new_empty(0)

    tokens = selected.index_select(0, positions)
    losses = -torch.log(denominator.index_select(0, positions))
    # selected is globally sorted, so stable loss sorting retains token order.
    order = torch.argsort(losses, descending=False, stable=True)[: max(1, int(pool_size))]
    return tokens.index_select(0, order), losses.index_select(0, order)


def _best_swap(
    analysis: _V3Analysis,
    additions: torch.Tensor,
    gains: torch.Tensor,
    coverage_gains: torch.Tensor,
    query_gains: torch.Tensor,
    removals: torch.Tensor,
    information_inverse: torch.Tensor,
    current_logdet: float,
    baseline_logdet: float,
    d_floor: float,
    *,
    require_positive_gain: bool,
) -> Optional[Tuple[int, int, float, float, float]]:
    """Evaluate the Cartesian swap pool in one tensor operation."""

    if additions.numel() == 0 or removals.numel() == 0:
        return None
    add_rows = analysis.design.index_select(0, additions).float()
    remove_rows = analysis.design.index_select(0, removals).float()
    inverse_remove = remove_rows @ information_inverse
    remove_denominator = 1.0 - torch.sum(inverse_remove * remove_rows, dim=1)
    valid_remove = remove_denominator > 1e-6

    base_add = torch.sum((add_rows @ information_inverse) * add_rows, dim=1)
    cross = add_rows @ inverse_remove.T
    leverage_without = (
        base_add.unsqueeze(1)
        + cross.square() / remove_denominator.clamp_min(1e-6).unsqueeze(0)
    )
    candidate_logdet = (
        float(current_logdet)
        + torch.log(remove_denominator.clamp_min(1e-6)).unsqueeze(0)
        + torch.log1p(leverage_without.clamp_min(0.0))
    )
    design_dim = max(1, analysis.design.shape[1])
    d_efficiency = torch.exp((candidate_logdet - float(baseline_logdet)) / design_dim)
    valid = valid_remove.unsqueeze(0) & (d_efficiency + 1e-12 >= float(d_floor))
    if require_positive_gain:
        valid &= gains.unsqueeze(1) > 1e-8
    valid_pairs = torch.nonzero(valid, as_tuple=False)
    if valid_pairs.numel() == 0:
        return None

    # One host transfer replaces hundreds of scalar .item() synchronizations.
    pair_rows = valid_pairs[:, 0]
    pair_cols = valid_pairs[:, 1]
    gain_values = gains.index_select(0, pair_rows).detach().cpu().tolist()
    logdet_values = candidate_logdet[pair_rows, pair_cols].detach().cpu().tolist()
    add_values = additions.index_select(0, pair_rows).detach().cpu().tolist()
    remove_values = removals.index_select(0, pair_cols).detach().cpu().tolist()
    efficiency_values = d_efficiency[pair_rows, pair_cols].detach().cpu().tolist()

    best_position = max(
        range(len(gain_values)),
        key=lambda index: (
            float(gain_values[index]),
            float(logdet_values[index]),
            -int(add_values[index]),
            -int(remove_values[index]),
        ),
    )
    add_position = int(pair_rows[best_position].item())
    return (
        int(add_values[best_position]),
        int(remove_values[best_position]),
        float(coverage_gains[add_position].item()),
        float(query_gains[add_position].item()),
        float(efficiency_values[best_position]),
    )


def _requirement_covered(
    requirement: Tuple[int, int, float],
    query_relevance: torch.Tensor,
    selected: torch.Tensor,
    token_chunk_ids: torch.Tensor,
) -> bool:
    atom, chunk, threshold = requirement
    anchors = selected[token_chunk_ids.index_select(0, selected) == chunk]
    return bool(
        anchors.numel()
        and float(query_relevance[atom].index_select(0, anchors).max().item()) >= float(threshold)
    )


def _repair_selection(
    analysis: _V3Analysis,
    original_selected: torch.Tensor,
    protected: Set[int],
    token_chunk_ids: torch.Tensor,
    config: Any,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Apply deterministic, D-efficient swaps to unresolved horizon chunks."""

    selected = original_selected.clone()
    chunk_count = int(token_chunk_ids.max().item()) + 1
    max_swap_ratio = min(1.0, max(0.0, float(getattr(config, "certhr_max_swap_ratio", 0.05))))
    max_swaps = int(math.ceil(max_swap_ratio * selected.numel()))
    coverage_floor = float(getattr(config, "certhr_coverage_floor", 0.70))
    deficit_threshold = float(getattr(config, "certhr_deficit_threshold", 0.05))
    add_pool = max(1, int(getattr(config, "certhr_add_pool", 32)))
    remove_pool = max(1, int(getattr(config, "certhr_remove_pool", 24)))
    d_floor = float(getattr(config, "certhr_d_efficiency_floor", 0.995))
    query_threshold = float(getattr(config, "certv3_query_threshold", 0.10))
    _, baseline_inverse, baseline_logdet = _information_state(
        analysis.design,
        original_selected,
        analysis.ridge,
    )
    requirements, _ = _query_requirements(
        analysis.query_relevance,
        analysis.query_confidence,
        token_chunk_ids,
        selected,
        confidence_threshold=query_threshold,
        peak_quantile=float(getattr(config, "certhr_query_peak_quantile", 0.90)),
        peak_floor=float(getattr(config, "certhr_query_peak_floor", 0.75)),
    )
    swaps: List[Dict[str, Any]] = []

    for _ in range(max_swaps):
        coverage = _chunk_coverage(analysis.metric_flat, selected, token_chunk_ids, chunk_count)
        coverage_target = max(coverage_floor, float(torch.median(coverage).item()) - 0.05)
        segment_deficits = [
            chunk
            for chunk in range(chunk_count)
            if coverage_target - float(coverage[chunk].item()) > deficit_threshold
        ]
        missing = _missing_query_requirements(
            requirements,
            analysis.query_relevance,
            selected,
            token_chunk_ids,
        )
        if not segment_deficits and not missing:
            break

        target_requirement = sorted(missing, key=lambda item: (item[1], item[0]))[0] if missing else None
        if target_requirement is not None:
            target_chunk = target_requirement[1]
        else:
            target_chunk = min(segment_deficits, key=lambda chunk: (float(coverage[chunk].item()), chunk))

        members = torch.nonzero(token_chunk_ids == target_chunk, as_tuple=False).reshape(-1)
        existing = selected[token_chunk_ids.index_select(0, selected) == target_chunk]
        current_similarity = (
            (analysis.metric_flat.index_select(0, members) @ analysis.metric_flat.index_select(0, existing).T)
            .max(dim=1)
            .values
            if existing.numel()
            else torch.zeros(members.numel(), device=analysis.metric_flat.device)
        )
        additions, gains, coverage_gains, query_gains = _addition_pool(
            analysis,
            selected,
            members,
            current_similarity,
            coverage[target_chunk],
            target_requirement,
            add_pool,
        )
        if additions.numel() == 0:
            break

        if not swaps:
            information_inverse = baseline_inverse
            current_logdet = baseline_logdet
        else:
            _, information_inverse, current_logdet = _information_state(
                analysis.design,
                selected,
                analysis.ridge,
            )
        selected_chunks = token_chunk_ids.index_select(0, selected)
        removals, _ = _removal_pool(
            analysis,
            selected,
            selected_chunks,
            protected,
            requirements,
            token_chunk_ids,
            target_chunk,
            coverage_target,
            deficit_threshold,
            information_inverse,
            remove_pool,
            chunk_count,
        )
        if removals.numel() == 0:
            break

        best = _best_swap(
            analysis,
            additions,
            gains,
            coverage_gains,
            query_gains,
            removals,
            information_inverse,
            current_logdet,
            baseline_logdet,
            d_floor,
            require_positive_gain=target_requirement is None,
        )
        if best is None:
            break

        add_token, remove_token, coverage_gain, query_gain, d_efficiency = best
        selected[selected == remove_token] = add_token
        selected = torch.sort(selected).values
        swaps.append(
            {
                "added": add_token,
                "removed": remove_token,
                "target_chunk": target_chunk,
                "coverage_gain": coverage_gain,
                "query_gain": query_gain,
                "d_efficiency": d_efficiency,
            }
        )

    final_coverage = _chunk_coverage(analysis.metric_flat, selected, token_chunk_ids, chunk_count)
    final_target = max(coverage_floor, float(torch.median(final_coverage).item()) - 0.05)
    final_segment_deficits = [
        chunk
        for chunk in range(chunk_count)
        if final_target - float(final_coverage[chunk].item()) > deficit_threshold
    ]
    final_missing = _missing_query_requirements(
        requirements,
        analysis.query_relevance,
        selected,
        token_chunk_ids,
    )
    _, _, final_logdet = _information_state(analysis.design, selected, analysis.ridge)
    final_d_efficiency = math.exp(
        (final_logdet - baseline_logdet) / max(1, analysis.design.shape[1])
    )
    if selected.numel() != original_selected.numel() or selected.unique().numel() != selected.numel():
        raise RuntimeError("CertVID-HR repair changed the anchor budget or introduced duplicates")
    if not protected.issubset({int(token) for token in selected.tolist()}):
        raise RuntimeError("CertVID-HR repair removed a protected V3 anchor")
    if not math.isfinite(final_d_efficiency) or final_d_efficiency + 1e-12 < d_floor:
        raise RuntimeError("CertVID-HR repair violated the D-efficiency floor")
    return selected, {
        "swaps": swaps,
        "swap_count": len(swaps),
        "max_swap_count": max_swaps,
        "protected_anchor_count": len(protected),
        "final_chunk_coverage": [float(value) for value in final_coverage.tolist()],
        "segment_deficits_after": final_segment_deficits,
        "query_deficit_after": len(final_missing),
        "d_efficiency": final_d_efficiency,
    }


def _store_diagnostics(config: Any, diagnostics: Dict[str, Any]) -> None:
    config.last_certhr_diagnostics = diagnostics
    config.last_certhr_timestamp_source = diagnostics.get("timestamp_source", "missing")
    config.last_certhr_fallback_reason = diagnostics.get("fallback_reason")
    config.last_certhr_max_timestamp_gap = float(diagnostics.get("max_timestamp_gap", 0.0))
    config.last_certhr_chunk_count = int(diagnostics.get("chunk_count", 0))
    config.last_certhr_query_deficit_before = int(diagnostics.get("query_deficit_before", 0))
    config.last_certhr_query_deficit_after = int(diagnostics.get("query_deficit_after", 0))
    config.last_certhr_swap_count = int(diagnostics.get("swap_count", 0))
    config.last_certhr_d_efficiency = float(diagnostics.get("d_efficiency", 1.0))
    config.last_certhr_modified_ratio = float(diagnostics.get("modified_ratio", 0.0))
    if bool(getattr(config, "certhr_debug", False)):
        print(
            "[CertVID-HR] "
            f"source={diagnostics.get('timestamp_source', 'missing')} "
            f"mapping={diagnostics.get('timestamp_mapping', 'none')} "
            f"fallback={diagnostics.get('fallback_reason')} "
            f"gap={float(diagnostics.get('max_timestamp_gap', 0.0)):.3f}s "
            f"chunks={int(diagnostics.get('chunk_count', 0))} "
            f"swaps={int(diagnostics.get('swap_count', 0))} "
            f"d_eff={float(diagnostics.get('d_efficiency', 1.0)):.6f}"
        )


def _fallback(
    config: Any,
    diagnostics: Dict[str, Any],
    reason: str,
    output: torch.Tensor,
    indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    diagnostics["fallback_reason"] = reason
    diagnostics.setdefault("swap_count", 0)
    diagnostics.setdefault("modified_ratio", 0.0)
    diagnostics.setdefault("d_efficiency", 1.0)
    _store_diagnostics(config, diagnostics)
    config.last_adapter_variant = "certvid_hr"
    return output, indices


def certvid_hr_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: Any,
    question_features: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run V3 exactly, then repair only verified long-horizon evidence deficits."""

    config = flashvid_config
    diagnostics: Dict[str, Any] = {
        "timestamp_source": str(getattr(config, "_certvid_frame_times_source", "missing")),
        "fallback_reason": None,
        "swap_count": 0,
        "modified_ratio": 0.0,
        "d_efficiency": 1.0,
    }
    raw_frame_times = getattr(config, "_certvid_frame_times_sec", None)
    frame_times, timestamp_error = _normalize_frame_times(
        raw_frame_times,
        video_features.shape[0],
        device=video_features.device,
    )
    gaps = frame_times[1:] - frame_times[:-1] if frame_times is not None else None
    max_gap = float(gaps.max().item()) if gaps is not None and gaps.numel() else 0.0
    diagnostics["max_timestamp_gap"] = max_gap
    horizon_threshold = float(getattr(config, "certhr_horizon_gap_seconds", 4.0))

    # Capture V3 intermediates only when HR can actually enter its repair path.
    # This removes the former second full analysis pass without retaining any
    # tensors on the persistent model configuration.
    analysis_sink: Optional[Dict[str, Any]] = (
        {} if frame_times is not None and max_gap > horizon_threshold else None
    )
    v3_output, v3_indices = certvid_v3_compression(
        video_features=video_features,
        cls_attention=cls_attention,
        flashvid_config=config,
        question_features=question_features,
        analysis_sink=analysis_sink,
    )
    v3_plan = getattr(config, "_certvid_plan", None)
    if v3_plan is None or v3_indices.numel() == 0:
        return _fallback(config, diagnostics, "missing_v3_plan", v3_output, v3_indices)
    if frame_times is None:
        return _fallback(config, diagnostics, timestamp_error or "invalid_timestamps", v3_output, v3_indices)
    raw_timestamp_count = int(torch.as_tensor(raw_frame_times).numel())
    diagnostics["raw_timestamp_count"] = raw_timestamp_count
    diagnostics["visual_time_units"] = int(video_features.shape[0])
    diagnostics["timestamp_mapping"] = (
        "direct" if raw_timestamp_count == int(video_features.shape[0]) else "contiguous_group_mean"
    )
    if max_gap <= horizon_threshold:
        return _fallback(config, diagnostics, "short_horizon", v3_output, v3_indices)
    if analysis_sink is not None and bool(analysis_sink.get("identity", False)):
        return _fallback(config, diagnostics, "identity_budget", v3_output, v3_indices)

    try:
        analysis = _analysis_from_sink(analysis_sink or {})
        diagnostics["analysis_source"] = "v3_capture"
        metric_frames = analysis.metric_flat.reshape(video_features.shape[0], video_features.shape[1], -1)
        frame_chunk_ids, semantic_gap, semantic_threshold = _build_horizon_chunks(
            metric_frames,
            frame_times,
            max_seconds=float(getattr(config, "certhr_chunk_max_seconds", 60.0)),
            max_units=int(getattr(config, "certhr_chunk_max_units", 4)),
            semantic_quantile=float(getattr(config, "certhr_semantic_quantile", 0.85)),
            semantic_floor=float(getattr(config, "certhr_semantic_floor", 0.10)),
        )
        token_chunk_ids = frame_chunk_ids.repeat_interleave(video_features.shape[1])
        chunk_count = int(frame_chunk_ids.max().item()) + 1
        initial_coverage = _chunk_coverage(analysis.metric_flat, v3_indices, token_chunk_ids, chunk_count)
        coverage_target = max(
            float(getattr(config, "certhr_coverage_floor", 0.70)),
            float(torch.median(initial_coverage).item()) - 0.05,
        )
        deficit_threshold = float(getattr(config, "certhr_deficit_threshold", 0.05))
        segment_deficits = [
            chunk
            for chunk in range(chunk_count)
            if coverage_target - float(initial_coverage[chunk].item()) > deficit_threshold
        ]
        requirements, query_missing = _query_requirements(
            analysis.query_relevance,
            analysis.query_confidence,
            token_chunk_ids,
            v3_indices,
            confidence_threshold=float(getattr(config, "certv3_query_threshold", 0.10)),
            peak_quantile=float(getattr(config, "certhr_query_peak_quantile", 0.90)),
            peak_floor=float(getattr(config, "certhr_query_peak_floor", 0.75)),
        )
        diagnostics.update(
            {
                "chunk_count": chunk_count,
                "semantic_threshold": semantic_threshold,
                "semantic_gap": [float(value) for value in semantic_gap.tolist()],
                "initial_chunk_coverage": [float(value) for value in initial_coverage.tolist()],
                "coverage_target": coverage_target,
                "segment_deficits_before": segment_deficits,
                "query_requirements": len(requirements),
                "query_deficit_before": len(query_missing),
            }
        )
        if not segment_deficits and not query_missing:
            diagnostics["final_chunk_coverage"] = diagnostics["initial_chunk_coverage"]
            diagnostics["segment_deficits_after"] = []
            diagnostics["query_deficit_after"] = 0
            return _fallback(config, diagnostics, "coverage_sufficient", v3_output, v3_indices)

        fusion_alpha = v3_plan.fusion_alpha.reshape(-1)
        protected = {
            int(token)
            for token, alpha in zip(v3_indices.tolist(), fusion_alpha.tolist())
            if float(alpha) <= 1e-12
        }
        repaired_indices, repair_diagnostics = _repair_selection(
            analysis,
            v3_indices,
            protected,
            token_chunk_ids,
            config,
        )
        diagnostics.update(repair_diagnostics)
        if int(repair_diagnostics["swap_count"]) == 0:
            return _fallback(config, diagnostics, "no_safe_swap", v3_output, v3_indices)

        new_plan = _build_plan(
            selected=repaired_indices,
            metric_features=analysis.metric_flat,
            demand_weight=analysis.demand_weight,
            attention=analysis.attention,
            query_score=analysis.query_score,
            temporal_ids=analysis.temporal_ids,
            component_ids=analysis.component_ids,
            fusion_alpha=float(getattr(config, "certv3_fusion_alpha", 0.12)),
            temperature=float(getattr(config, "certv3_assignment_temperature", 0.07)),
        )
        v3_index_set = {int(token) for token in v3_indices.tolist()}
        protected_mask = torch.tensor(
            [int(token) in protected or int(token) not in v3_index_set for token in repaired_indices.tolist()],
            dtype=torch.bool,
            device=video_features.device,
        )
        new_plan.fusion_alpha[protected_mask] = 0.0
        output = apply_certvid_plan(video_features.reshape(-1, video_features.shape[-1]), new_plan)
        config._certvid_plan = new_plan
        diagnostics["fallback_reason"] = None
        diagnostics["modified_ratio"] = float(repair_diagnostics["swap_count"]) / max(1, v3_indices.numel())
        _store_diagnostics(config, diagnostics)
        config.last_adapter_variant = "certvid_hr"
        config.last_raw_visual_tokens = int(video_features.shape[0] * video_features.shape[1])
        config.last_output_visual_tokens = int(output.shape[0])
        config.last_vision_output_tokens = int(output.shape[0])
        config.last_output_indices = repaired_indices.detach()
        return output, repaired_indices
    except (RuntimeError, ValueError, IndexError, ArithmeticError) as error:
        config._certvid_plan = v3_plan
        diagnostics["repair_error"] = f"{type(error).__name__}: {error}"
        return _fallback(config, diagnostics, "repair_failed", v3_output, v3_indices)


__all__ = ["certvid_hr_compression"]
