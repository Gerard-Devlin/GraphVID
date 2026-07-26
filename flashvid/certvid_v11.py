"""CertVID V11: correspondence-lifted spatiotemporal D-optimal repair.

V3 remains the appearance-evidence selector. V11 only repairs transition
endpoints supported by reliable bidirectional correspondences, so spatial
geometry validates temporal evidence instead of owning an independent quota.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from .certvid import (
    CertVidPlan,
    _cfg_float,
    _cfg_int,
    _grid_hw,
    _spatial_layout,
    apply_certvid_plan,
)
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


@dataclass
class _CorrespondenceGraph:
    source: torch.Tensor
    target: torch.Tensor
    confidence: torch.Tensor
    similarity: torch.Tensor
    margin: torch.Tensor
    displacement: torch.Tensor
    state_change: torch.Tensor
    frame_pair: torch.Tensor
    scene_boundary: torch.Tensor
    frame_similarity: torch.Tensor
    valid_edge_rate: float
    cycle_consistency_rate: float
    reliability: float


@dataclass
class _TransitionState:
    rows: torch.Tensor
    token_relation: torch.Tensor
    token_state: torch.Tensor
    edge_importance: torch.Tensor


@dataclass
class _RepairResult:
    selected: torch.Tensor
    swaps: list[dict[str, Any]]
    node_efficiency: float
    edge_coverage_before: float
    edge_coverage_after: float
    joint_before: float
    joint_after: float
    objective_gain: float
    transition_weight: float


_DIAGNOSTIC_HANDLES: dict[str, Any] = {}


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
    analysis = _V3Analysis(**{name: sink[name] for name in names})
    total_tokens = int(analysis.metric_flat.shape[0])
    if analysis.metric_flat.ndim != 2 or analysis.design.ndim != 2:
        raise ValueError("V3 metric and design tensors must be rank two")
    for name in (
        "design",
        "demand_weight",
        "attention",
        "query_score",
        "frame_ids",
        "temporal_ids",
        "component_ids",
    ):
        value = getattr(analysis, name)
        if int(value.shape[0]) != total_tokens:
            raise ValueError(f"V3 analysis tensor {name} has the wrong length")
    for name in (
        "metric_flat",
        "design",
        "demand_weight",
        "attention",
        "query_score",
    ):
        if not bool(torch.isfinite(getattr(analysis, name)).all()):
            raise ValueError(f"V3 analysis tensor {name} contains NaN or Inf")
    return analysis


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
    if times.numel() != frame_count and (
        frame_count > 0 and times.numel() % frame_count == 0
    ):
        reduction = times.numel() // frame_count
        times = times.reshape(frame_count, reduction).mean(dim=1)
        source = f"{source}_mean_x{reduction}"
    valid = times.numel() == frame_count and bool(torch.isfinite(times).all())
    valid = valid and (
        times.numel() <= 1 or bool(torch.all(times[1:] > times[:-1]))
    )
    if not valid:
        return torch.arange(frame_count, device=device).float(), False, "frame_index"
    return times, True, source


def _empty_graph(
    frame_count: int,
    device: torch.device,
) -> _CorrespondenceGraph:
    return _CorrespondenceGraph(
        source=torch.empty(0, dtype=torch.long, device=device),
        target=torch.empty(0, dtype=torch.long, device=device),
        confidence=torch.empty(0, dtype=torch.float32, device=device),
        similarity=torch.empty(0, dtype=torch.float32, device=device),
        margin=torch.empty(0, dtype=torch.float32, device=device),
        displacement=torch.empty((0, 2), dtype=torch.float32, device=device),
        state_change=torch.empty(0, dtype=torch.float32, device=device),
        frame_pair=torch.empty(0, dtype=torch.long, device=device),
        scene_boundary=torch.zeros(
            max(0, frame_count - 1),
            dtype=torch.bool,
            device=device,
        ),
        frame_similarity=torch.ones(
            max(0, frame_count - 1),
            dtype=torch.float32,
            device=device,
        ),
        valid_edge_rate=0.0,
        cycle_consistency_rate=0.0,
        reliability=0.0,
    )


def _build_correspondence_graph(
    analysis: _V3Analysis,
    frame_count: int,
    tokens_per_frame: int,
    frame_times: torch.Tensor,
    has_real_times: bool,
    config: Any,
) -> _CorrespondenceGraph:
    device = analysis.metric_flat.device
    if frame_count <= 1 or tokens_per_frame <= 1:
        return _empty_graph(frame_count, device)

    height, width = _grid_hw(tokens_per_frame, config)
    coords, _ = _spatial_layout(
        tokens_per_frame,
        height,
        width,
        max(1, _cfg_int(config, "certv3_spatial_bins", 3)),
        device,
    )
    rows = torch.div(
        torch.arange(tokens_per_frame, device=device),
        width,
        rounding_mode="floor",
    )
    cols = torch.remainder(torch.arange(tokens_per_frame, device=device), width)
    spatial_distance = torch.cdist(coords.float(), coords.float(), p=2)
    frames = analysis.metric_flat.view(frame_count, tokens_per_frame, -1)
    frame_representatives = F.normalize(
        frames.mean(dim=1),
        p=2,
        dim=-1,
        eps=1e-6,
    )
    frame_similarity = torch.sum(
        frame_representatives[:-1] * frame_representatives[1:],
        dim=-1,
    ).clamp(-1.0, 1.0)

    similarity_threshold = _cfg_float(
        config,
        "certv11_match_similarity",
        0.72,
    )
    margin_threshold = _cfg_float(config, "certv11_match_margin", 0.015)
    cycle_radius = max(0, _cfg_int(config, "certv11_cycle_radius", 1))
    max_jump = max(
        1e-4,
        _cfg_float(config, "certv11_max_spatial_jump", 0.60),
    )
    scene_threshold = _cfg_float(
        config,
        "certv11_scene_similarity",
        0.50,
    )
    spatial_weight = max(
        0.0,
        _cfg_float(config, "certv11_spatial_match_weight", 0.04),
    )
    time_scale_seconds = max(
        1e-3,
        _cfg_float(config, "certv11_time_confidence_seconds", 30.0),
    )

    all_source: list[torch.Tensor] = []
    all_target: list[torch.Tensor] = []
    all_confidence: list[torch.Tensor] = []
    all_similarity: list[torch.Tensor] = []
    all_margin: list[torch.Tensor] = []
    all_displacement: list[torch.Tensor] = []
    all_state_change: list[torch.Tensor] = []
    all_pair: list[torch.Tensor] = []
    scene_boundary = frame_similarity < scene_threshold
    cycle_numerator = torch.zeros(
        (),
        dtype=torch.long,
        device=device,
    )
    cycle_denominator = torch.zeros_like(cycle_numerator)
    if has_real_times:
        frame_time_confidence = torch.exp(
            -(
                frame_times[1:] - frame_times[:-1]
            ).clamp_min(0.0)
            / time_scale_seconds
        ).clamp_min(0.25)
    else:
        frame_time_confidence = torch.ones(
            frame_count - 1,
            dtype=torch.float32,
            device=device,
        )

    local_ids = torch.arange(tokens_per_frame, device=device)
    for frame_idx in range(frame_count - 1):
        current = frames[frame_idx]
        following = frames[frame_idx + 1]
        raw_similarity = current @ following.transpose(0, 1)
        matching_score = raw_similarity - spatial_weight * spatial_distance.square()
        values, targets = torch.topk(
            matching_score,
            k=min(2, tokens_per_frame),
            dim=1,
            largest=True,
        )
        best_target = targets[:, 0]
        if values.shape[1] > 1:
            margin = values[:, 0] - values[:, 1]
        else:
            margin = torch.ones_like(values[:, 0])
        backward = torch.argmax(matching_score, dim=0)
        cycle_source = backward[best_target]
        cycle_valid = (
            (rows[cycle_source] - rows).abs() <= cycle_radius
        ) & ((cols[cycle_source] - cols).abs() <= cycle_radius)
        matched_similarity = raw_similarity[local_ids, best_target]
        displacement = coords[best_target] - coords
        jump = torch.linalg.vector_norm(displacement, dim=-1)

        semantic_candidates = matched_similarity >= similarity_threshold
        cycle_numerator += (cycle_valid & semantic_candidates).sum()
        cycle_denominator += semantic_candidates.sum()
        valid = (
            semantic_candidates
            & (margin >= margin_threshold)
            & cycle_valid
            & (jump <= max_jump)
            & (~scene_boundary[frame_idx])
        )
        camera_displacement = torch.nanmedian(
            displacement.masked_fill(
                ~valid.unsqueeze(1),
                float("nan"),
            ),
            dim=0,
        ).values
        camera_displacement = torch.nan_to_num(
            camera_displacement,
            nan=0.0,
        )
        centered_displacement = displacement - camera_displacement.unsqueeze(0)
        centered_jump = torch.linalg.vector_norm(centered_displacement, dim=-1)
        sim_confidence = (
            (matched_similarity - similarity_threshold)
            / max(1e-6, 1.0 - similarity_threshold)
        ).clamp(0.0, 1.0)
        margin_confidence = (
            margin / max(1e-6, 4.0 * margin_threshold)
        ).clamp(0.0, 1.0)
        geometry_confidence = torch.exp(
            -centered_jump.square() / max(1e-6, 0.5 * max_jump * max_jump)
        )
        scene_confidence = (
            (frame_similarity[frame_idx] - scene_threshold)
            / max(1e-6, 1.0 - scene_threshold)
        ).clamp(0.10, 1.0)
        confidence = (
            0.45 * sim_confidence
            + 0.20 * margin_confidence
            + 0.20 * geometry_confidence
            + 0.15 * scene_confidence
        ) * torch.sqrt(frame_time_confidence[frame_idx])
        confidence = confidence.clamp(0.0, 1.0)

        kept = torch.where(valid)[0]
        all_source.append(frame_idx * tokens_per_frame + kept)
        all_target.append(
            (frame_idx + 1) * tokens_per_frame + best_target[kept]
        )
        all_confidence.append(confidence[kept])
        all_similarity.append(matched_similarity[kept])
        all_margin.append(margin[kept])
        all_displacement.append(centered_displacement[kept])
        all_state_change.append((1.0 - matched_similarity[kept]).clamp(0.0, 1.0))
        all_pair.append(
            torch.full_like(kept, frame_idx, dtype=torch.long)
        )

    cycle_numerator_value = int(cycle_numerator.item())
    cycle_denominator_value = int(cycle_denominator.item())
    source = torch.cat(all_source)
    if source.numel() == 0:
        graph = _empty_graph(frame_count, device)
        graph.scene_boundary = scene_boundary
        graph.frame_similarity = frame_similarity
        graph.cycle_consistency_rate = (
            float(cycle_numerator_value / cycle_denominator_value)
            if cycle_denominator_value > 0
            else 0.0
        )
        return graph

    target = torch.cat(all_target)
    confidence = torch.cat(all_confidence)
    similarity = torch.cat(all_similarity)
    margin = torch.cat(all_margin)
    displacement = torch.cat(all_displacement)
    state_change = torch.cat(all_state_change)
    frame_pair = torch.cat(all_pair)
    possible_edges = max(1, (frame_count - 1) * tokens_per_frame)
    valid_edge_rate = float(source.numel() / possible_edges)
    cycle_rate = (
        float(cycle_numerator_value / cycle_denominator_value)
        if cycle_denominator_value > 0
        else 0.0
    )
    reliability = valid_edge_rate * float(confidence.mean().item())
    return _CorrespondenceGraph(
        source=source,
        target=target,
        confidence=confidence,
        similarity=similarity,
        margin=margin,
        displacement=displacement,
        state_change=state_change,
        frame_pair=frame_pair,
        scene_boundary=scene_boundary,
        frame_similarity=frame_similarity,
        valid_edge_rate=valid_edge_rate,
        cycle_consistency_rate=cycle_rate,
        reliability=reliability,
    )


def _transition_state(
    analysis: _V3Analysis,
    graph: _CorrespondenceGraph,
    config: Any,
) -> _TransitionState:
    total_tokens = int(analysis.metric_flat.shape[0])
    transition_dim = max(
        8,
        _cfg_int(config, "certv11_transition_dim", 32),
    )
    device = analysis.metric_flat.device
    if graph.source.numel() == 0:
        return _TransitionState(
            rows=torch.zeros(
                (total_tokens, transition_dim),
                dtype=torch.float32,
                device=device,
            ),
            token_relation=torch.zeros(
                total_tokens,
                dtype=torch.float32,
                device=device,
            ),
            token_state=torch.zeros(
                total_tokens,
                dtype=torch.float32,
                device=device,
            ),
            edge_importance=torch.empty(
                0,
                dtype=torch.float32,
                device=device,
            ),
        )

    delta = (
        analysis.metric_flat[graph.target]
        - analysis.metric_flat[graph.source]
    )
    projected_dim = max(4, transition_dim - 7)
    if delta.shape[1] != projected_dim:
        delta = F.adaptive_avg_pool1d(
            delta.unsqueeze(1),
            projected_dim,
        ).squeeze(1)
    displacement_norm = torch.linalg.vector_norm(
        graph.displacement,
        dim=-1,
        keepdim=True,
    )
    descriptor = torch.cat(
        [
            delta,
            graph.displacement,
            displacement_norm,
            graph.state_change.unsqueeze(1),
            graph.confidence.unsqueeze(1),
            graph.margin.unsqueeze(1),
            torch.ones_like(graph.confidence).unsqueeze(1),
        ],
        dim=1,
    )
    if descriptor.shape[1] != transition_dim:
        descriptor = F.adaptive_avg_pool1d(
            descriptor.unsqueeze(1),
            transition_dim,
        ).squeeze(1)
    state_scale = (
        graph.state_change
        / max(
            1e-4,
            _cfg_float(config, "certv11_state_scale", 0.20),
        )
    ).clamp(0.0, 1.0)
    edge_scale = torch.sqrt(
        graph.confidence.clamp_min(1e-6)
        * (0.20 + 0.80 * state_scale)
    )
    descriptor = F.normalize(
        descriptor,
        p=2,
        dim=-1,
        eps=1e-6,
    ) * edge_scale.unsqueeze(1)

    outgoing = torch.zeros(
        (total_tokens, transition_dim),
        dtype=torch.float32,
        device=device,
    )
    incoming = torch.zeros_like(outgoing)
    outgoing_count = torch.zeros(
        total_tokens,
        dtype=torch.float32,
        device=device,
    )
    incoming_count = torch.zeros_like(outgoing_count)
    outgoing.index_add_(0, graph.source, descriptor)
    incoming.index_add_(0, graph.target, descriptor)
    outgoing_count.index_add_(
        0,
        graph.source,
        torch.ones_like(graph.confidence),
    )
    incoming_count.index_add_(
        0,
        graph.target,
        torch.ones_like(graph.confidence),
    )
    outgoing = outgoing / outgoing_count.clamp_min(1.0).unsqueeze(1)
    incoming = incoming / incoming_count.clamp_min(1.0).unsqueeze(1)
    rows = torch.cat(
        [outgoing, incoming, (outgoing - incoming).abs()],
        dim=1,
    )
    rows = F.adaptive_avg_pool1d(
        rows.unsqueeze(1),
        transition_dim,
    ).squeeze(1)
    nonzero = rows.norm(dim=1) > 1e-8
    rows[nonzero] = F.normalize(
        rows[nonzero],
        p=2,
        dim=-1,
        eps=1e-6,
    )

    edge_query = 0.5 * (
        analysis.query_score[graph.source]
        + analysis.query_score[graph.target]
    )
    edge_demand = 0.5 * (
        analysis.demand_weight[graph.source]
        + analysis.demand_weight[graph.target]
    )
    edge_demand = edge_demand / edge_demand.mean().clamp_min(1e-6)
    edge_importance = (
        graph.confidence
        * (0.25 + 0.75 * state_scale)
        * (0.75 + 0.25 * edge_query.clamp(0.0, 1.0))
        * edge_demand.clamp(0.25, 2.0)
    )
    token_relation = torch.zeros(
        total_tokens,
        dtype=torch.float32,
        device=device,
    )
    token_state = torch.zeros_like(token_relation)
    token_relation.index_add_(0, graph.source, 0.5 * edge_importance)
    token_relation.index_add_(0, graph.target, 0.5 * edge_importance)
    token_state.scatter_reduce_(
        0,
        graph.source,
        graph.state_change,
        reduce="amax",
        include_self=True,
    )
    token_state.scatter_reduce_(
        0,
        graph.target,
        graph.state_change,
        reduce="amax",
        include_self=True,
    )
    if bool(token_relation.max() > token_relation.min()):
        token_relation = (
            token_relation - token_relation.min()
        ) / (token_relation.max() - token_relation.min() + 1e-6)
    if bool(token_state.max() > token_state.min()):
        token_state = (
            token_state - token_state.min()
        ) / (token_state.max() - token_state.min() + 1e-6)
    return _TransitionState(
        rows=rows,
        token_relation=token_relation,
        token_state=token_state,
        edge_importance=edge_importance,
    )


def _same_frame_coverage(
    analysis: _V3Analysis,
    selected: torch.Tensor,
) -> torch.Tensor:
    total_tokens = int(analysis.metric_flat.shape[0])
    frame_count = int(analysis.frame_ids.max().item()) + 1
    if frame_count <= 0 or total_tokens % frame_count != 0:
        raise ValueError("V11 cannot group visual tokens by frame")
    tokens_per_frame = total_tokens // frame_count
    expected_frames = torch.arange(
        frame_count,
        dtype=torch.long,
        device=selected.device,
    ).repeat_interleave(tokens_per_frame)
    if not bool(torch.equal(analysis.frame_ids, expected_frames)):
        raise ValueError("V11 requires frame-major visual token ordering")

    selected_frames = analysis.frame_ids[selected]
    counts = torch.bincount(selected_frames, minlength=frame_count)
    max_count = max(1, int(counts.max().item()))
    order = torch.argsort(selected_frames, stable=True)
    sorted_frames = selected_frames[order]
    starts = torch.cumsum(counts, dim=0) - counts
    local_positions = torch.arange(
        selected.numel(),
        dtype=torch.long,
        device=selected.device,
    ) - torch.repeat_interleave(starts, counts)
    padded = torch.full(
        (frame_count, max_count),
        0,
        dtype=torch.long,
        device=selected.device,
    )
    valid = torch.zeros(
        (frame_count, max_count),
        dtype=torch.bool,
        device=selected.device,
    )
    padded[sorted_frames, local_positions] = selected[order]
    valid[sorted_frames, local_positions] = True

    metric_frames = analysis.metric_flat.float().reshape(
        frame_count,
        tokens_per_frame,
        -1,
    )
    anchors = analysis.metric_flat[padded].float()
    similarity = torch.bmm(
        metric_frames,
        anchors.transpose(1, 2),
    ).masked_fill(~valid.unsqueeze(1), -2.0)
    return similarity.amax(dim=-1).reshape(-1).clamp(0.0, 1.0)


def _preserves_group_support(
    analysis: _V3Analysis,
    base_selected: torch.Tensor,
    proposed: torch.Tensor,
) -> bool:
    for ids in (analysis.frame_ids, analysis.temporal_ids):
        group_count = int(ids.max().item()) + 1
        base_counts = torch.bincount(
            ids[base_selected],
            minlength=group_count,
        )
        proposed_counts = torch.bincount(
            ids[proposed],
            minlength=group_count,
        )
        if bool(((base_counts > 0) & (proposed_counts <= 0)).any()):
            return False
    return True


def _edge_coverage(
    graph: _CorrespondenceGraph,
    token_coverage: torch.Tensor,
    edge_importance: torch.Tensor,
) -> tuple[float, float, torch.Tensor]:
    if graph.source.numel() == 0:
        return 1.0, 0.0, torch.empty_like(graph.confidence)
    coverage = torch.minimum(
        token_coverage[graph.source],
        token_coverage[graph.target],
    )
    weights = edge_importance.clamp_min(1e-8)
    weighted = float(
        (coverage * weights).sum().item() / weights.sum().item()
    )
    deficit = max(0.0, 1.0 - weighted)
    return weighted, deficit, coverage


def _temporal_entropy(counts: torch.Tensor) -> float:
    probability = counts.float() / counts.sum().clamp_min(1.0)
    nonzero = probability > 0
    if int(nonzero.sum().item()) <= 1:
        return 0.0
    entropy = -(probability[nonzero] * probability[nonzero].log()).sum()
    return float((entropy / math.log(int(nonzero.sum().item()))).item())


def _cholesky(
    matrix: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    identity = torch.eye(
        matrix.shape[0],
        dtype=matrix.dtype,
        device=matrix.device,
    )
    for jitter in (0.0, 1e-6, 1e-5, 1e-4):
        candidate = matrix if jitter == 0.0 else matrix + jitter * identity
        factor, info = torch.linalg.cholesky_ex(candidate)
        if int(info.max().item()) == 0:
            logdet = float(
                (2.0 * torch.log(torch.diagonal(factor))).sum().item()
            )
            return factor, logdet
    raise RuntimeError("V11 information matrix is not positive definite")


def _information(
    design: torch.Tensor,
    selected: torch.Tensor,
    ridge: float,
) -> tuple[torch.Tensor, float]:
    dimension = int(design.shape[1])
    information = (
        max(1e-6, float(ridge))
        * torch.eye(
            dimension,
            dtype=torch.float32,
            device=design.device,
        )
        + design[selected].float().transpose(0, 1)
        @ design[selected].float()
    )
    return _cholesky(information)


def _leverage(
    rows: torch.Tensor,
    factor: torch.Tensor,
) -> torch.Tensor:
    if rows.numel() == 0:
        return torch.empty(
            rows.shape[0],
            dtype=torch.float32,
            device=rows.device,
        )
    solution = torch.cholesky_solve(
        rows.float().transpose(0, 1),
        factor,
    ).transpose(0, 1)
    return torch.sum(rows.float() * solution, dim=1).clamp_min(0.0)


def _joint_design(
    analysis: _V3Analysis,
    transition: _TransitionState,
    graph: _CorrespondenceGraph,
    deficit: float,
    config: Any,
) -> tuple[torch.Tensor, float]:
    minimum = min(
        0.95,
        max(
            0.0,
            _cfg_float(
                config,
                "certv11_transition_weight_min",
                0.08,
            ),
        ),
    )
    maximum = min(
        0.95,
        max(
            minimum,
            _cfg_float(
                config,
                "certv11_transition_weight_max",
                0.24,
            ),
        ),
    )
    reliability_scale = min(
        1.0,
        graph.reliability
        / max(
            1e-6,
            _cfg_float(
                config,
                "certv11_reliability_target",
                0.12,
            ),
        ),
    )
    deficit_scale = min(
        1.0,
        deficit
        / max(
            1e-6,
            _cfg_float(
                config,
                "certv11_deficit_scale",
                0.20,
            ),
        ),
    )
    weight = minimum + (maximum - minimum) * reliability_scale * deficit_scale
    if not bool((transition.rows.norm(dim=1) > 1e-8).any()):
        weight = 0.0
    joint = torch.cat(
        [
            analysis.design.float() * math.sqrt(max(1e-6, 1.0 - weight)),
            transition.rows.float() * math.sqrt(max(0.0, weight)),
        ],
        dim=1,
    )
    return joint, float(weight)


def _protected_anchors(
    analysis: _V3Analysis,
    selected: torch.Tensor,
    base_plan: CertVidPlan,
    transition: _TransitionState,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    budget = int(selected.numel())
    protected = base_plan.fusion_alpha <= 1e-12
    protected = protected.to(device=selected.device, dtype=torch.bool).clone()

    frame_count = int(analysis.frame_ids.max().item()) + 1
    frame_counts = torch.bincount(
        analysis.frame_ids[selected],
        minlength=frame_count,
    )
    temporal_count = int(analysis.temporal_ids.max().item()) + 1
    temporal_counts = torch.bincount(
        analysis.temporal_ids[selected],
        minlength=temporal_count,
    )
    protected |= frame_counts[analysis.frame_ids[selected]] <= 1
    protected |= temporal_counts[analysis.temporal_ids[selected]] <= 1

    factor, _ = _information(analysis.design, selected, analysis.ridge)
    node_leverage = _leverage(analysis.design[selected], factor)
    protect_ratio = min(
        0.50,
        max(
            0.0,
            _cfg_float(config, "certv11_v3_protect_ratio", 0.10),
        ),
    )
    protect_count = min(
        budget,
        int(math.ceil(budget * protect_ratio)),
    )
    if protect_count > 0:
        protected[
            torch.topk(
                node_leverage,
                k=protect_count,
                largest=True,
            ).indices
        ] = True

    relation_count = min(budget, max(1, int(math.ceil(0.05 * budget))))
    if relation_count > 0 and bool(
        transition.token_relation[selected].max() > 0
    ):
        protected[
            torch.topk(
                transition.token_relation[selected],
                k=relation_count,
                largest=True,
            ).indices
        ] = True
    return protected, node_leverage


def _repair_candidates(
    analysis: _V3Analysis,
    selected: torch.Tensor,
    graph: _CorrespondenceGraph,
    transition: _TransitionState,
    token_coverage: torch.Tensor,
    edge_coverage: torch.Tensor,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, str]]:
    total_tokens = int(analysis.metric_flat.shape[0])
    selected_mask = torch.zeros(
        total_tokens,
        dtype=torch.bool,
        device=selected.device,
    )
    selected_mask[selected] = True
    endpoint_risk = torch.zeros(
        total_tokens,
        dtype=torch.float32,
        device=selected.device,
    )
    if graph.source.numel() > 0:
        edge_risk = transition.edge_importance * (1.0 - edge_coverage)
        endpoint_risk.index_add_(0, graph.source, 0.5 * edge_risk)
        endpoint_risk.index_add_(0, graph.target, 0.5 * edge_risk)
    if bool(endpoint_risk.max() > endpoint_risk.min()):
        endpoint_risk = (
            endpoint_risk - endpoint_risk.min()
        ) / (endpoint_risk.max() - endpoint_risk.min() + 1e-6)

    coverage_risk = (
        analysis.demand_weight
        * (1.0 - token_coverage).clamp_min(0.0)
    )
    if bool(coverage_risk.max() > coverage_risk.min()):
        coverage_risk = (
            coverage_risk - coverage_risk.min()
        ) / (coverage_risk.max() - coverage_risk.min() + 1e-6)
    transition_norm = transition.rows.norm(dim=1)
    if bool(transition_norm.max() > transition_norm.min()):
        transition_norm = (
            transition_norm - transition_norm.min()
        ) / (transition_norm.max() - transition_norm.min() + 1e-6)
    score = (
        0.42 * endpoint_risk
        + 0.23 * transition.token_relation
        + 0.17 * transition.token_state
        + 0.13 * coverage_risk
        + 0.05 * analysis.query_score.clamp(0.0, 1.0)
    )
    score = score.masked_fill(selected_mask, -1.0)

    pool_size = max(
        1,
        min(
            total_tokens - int(selected.numel()),
            _cfg_int(config, "certv11_add_pool", 160),
        ),
    )
    frame_count = int(analysis.frame_ids.max().item()) + 1
    per_frame_cap = max(
        2,
        int(math.ceil(2.5 * pool_size / max(1, frame_count))),
    )
    frame_offered = [0] * frame_count
    candidates: list[int] = []
    provenance: dict[int, str] = {}
    order = torch.argsort(score, descending=True, stable=True)
    state_cut = (
        float(torch.quantile(transition.token_state, 0.80).item())
        if transition.token_state.numel() > 0
        else 1.0
    )
    relation_cut = (
        float(torch.quantile(transition.token_relation, 0.75).item())
        if transition.token_relation.numel() > 0
        else 1.0
    )
    query_cut = (
        float(torch.quantile(analysis.query_score, 0.90).item())
        if analysis.query_score.numel() > 0
        else 1.0
    )
    score_cpu = score.detach().cpu()
    frame_ids_cpu = analysis.frame_ids.detach().cpu()
    token_state_cpu = transition.token_state.detach().cpu()
    endpoint_risk_cpu = endpoint_risk.detach().cpu()
    query_score_cpu = analysis.query_score.detach().cpu()
    token_relation_cpu = transition.token_relation.detach().cpu()
    for token_value in order.detach().cpu().tolist():
        token = int(token_value)
        if float(score_cpu[token]) < 0.0:
            break
        frame = int(frame_ids_cpu[token])
        if frame_offered[frame] >= per_frame_cap:
            continue
        if (
            float(token_state_cpu[token]) >= state_cut
            and float(endpoint_risk_cpu[token]) > 0.0
        ):
            source = "state_endpoint"
        elif (
            float(query_score_cpu[token]) >= query_cut
            and float(endpoint_risk_cpu[token]) > 0.0
        ):
            source = "query_transition"
        elif float(token_relation_cpu[token]) >= relation_cut:
            source = "transition_endpoint"
        else:
            source = "full_pool_coverage"
        candidates.append(token)
        provenance[token] = source
        frame_offered[frame] += 1
        if len(candidates) >= pool_size:
            break
    if not candidates:
        return (
            torch.empty(0, dtype=torch.long, device=selected.device),
            score,
            provenance,
        )
    return (
        torch.tensor(candidates, dtype=torch.long, device=selected.device),
        score,
        provenance,
    )


def _removal_pool(
    analysis: _V3Analysis,
    selected: torch.Tensor,
    protected: torch.Tensor,
    node_leverage: torch.Tensor,
    transition: _TransitionState,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    budget = int(selected.numel())
    dimension = max(1, int(analysis.design.shape[1]))
    node_loss = (
        -torch.log1p(-node_leverage.clamp(max=1.0 - 1e-6))
    ) / dimension
    selected_metric = analysis.metric_flat[selected]
    if budget > 1:
        redundancy = selected_metric @ selected_metric.transpose(0, 1)
        redundancy.fill_diagonal_(-2.0)
        redundancy = redundancy.amax(dim=1).clamp(0.0, 1.0)
    else:
        redundancy = torch.zeros_like(node_loss)

    frame_count = int(analysis.frame_ids.max().item()) + 1
    frame_counts = torch.bincount(
        analysis.frame_ids[selected],
        minlength=frame_count,
    ).float()
    frame_mean = float(budget / max(1, frame_count))
    frame_surplus = (
        frame_counts[analysis.frame_ids[selected]] / max(1.0, frame_mean) - 1.0
    ).clamp_min(0.0)
    cost = (
        _cfg_float(config, "certv11_node_loss_weight", 0.35) * node_loss
        + 0.22 * transition.token_relation[selected]
        + 0.08 * (1.0 - redundancy)
        - 0.05 * frame_surplus
    )
    cost = cost.masked_fill(protected, float("inf"))
    order = torch.argsort(cost, descending=False, stable=True)
    finite = order[torch.isfinite(cost[order])]
    pool_size = min(
        int(finite.numel()),
        max(1, _cfg_int(config, "certv11_remove_pool", 64)),
    )
    return finite[:pool_size], cost


def _candidate_utility(
    analysis: _V3Analysis,
    selected: torch.Tensor,
    candidates: torch.Tensor,
    candidate_score: torch.Tensor,
    joint: torch.Tensor,
    transition: _TransitionState,
    config: Any,
) -> torch.Tensor:
    if candidates.numel() == 0:
        return torch.empty(
            0,
            dtype=torch.float32,
            device=selected.device,
        )
    node_factor, _ = _information(analysis.design, selected, analysis.ridge)
    joint_factor, _ = _information(joint, selected, analysis.ridge)
    node_gain = torch.log1p(
        _leverage(analysis.design[candidates], node_factor)
    ) / max(1, int(analysis.design.shape[1]))
    joint_gain = torch.log1p(
        _leverage(joint[candidates], joint_factor)
    ) / max(1, int(joint.shape[1]))

    frame_count = int(analysis.frame_ids.max().item()) + 1
    counts = torch.bincount(
        analysis.frame_ids[selected],
        minlength=frame_count,
    ).float()
    mean_count = float(selected.numel() / max(1, frame_count))
    balance_bonus = (
        (mean_count - counts[analysis.frame_ids[candidates]])
        / max(1.0, mean_count)
    ).clamp(-1.0, 1.0)
    return (
        joint_gain
        + 0.30 * candidate_score[candidates].clamp_min(0.0)
        + 0.16 * transition.token_relation[candidates]
        + 0.08 * transition.token_state[candidates]
        + _cfg_float(config, "certv11_frame_balance_weight", 0.08)
        * balance_bonus
        + 0.05 * node_gain
    )


def _evaluate_selection(
    design: torch.Tensor,
    selected: torch.Tensor,
    ridge: float,
) -> float:
    _, logdet = _information(design, selected, ridge)
    return logdet


def _exchange_repair(
    analysis: _V3Analysis,
    base_selected: torch.Tensor,
    candidates: torch.Tensor,
    candidate_score: torch.Tensor,
    provenance: dict[int, str],
    removal_positions: torch.Tensor,
    removal_cost: torch.Tensor,
    joint: torch.Tensor,
    graph: _CorrespondenceGraph,
    transition: _TransitionState,
    edge_coverage_before: float,
    deficit: float,
    transition_weight: float,
    config: Any,
) -> _RepairResult:
    budget = int(base_selected.numel())
    base_node_logdet = _evaluate_selection(
        analysis.design,
        base_selected,
        analysis.ridge,
    )
    base_joint_logdet = _evaluate_selection(
        joint,
        base_selected,
        analysis.ridge,
    )
    if candidates.numel() == 0 or removal_positions.numel() == 0:
        return _RepairResult(
            selected=base_selected,
            swaps=[],
            node_efficiency=1.0,
            edge_coverage_before=edge_coverage_before,
            edge_coverage_after=edge_coverage_before,
            joint_before=base_joint_logdet,
            joint_after=base_joint_logdet,
            objective_gain=0.0,
            transition_weight=transition_weight,
        )

    utility = _candidate_utility(
        analysis,
        base_selected,
        candidates,
        candidate_score,
        joint,
        transition,
        config,
    )
    add_order = torch.argsort(utility, descending=True, stable=True)
    candidates = candidates[add_order]
    utility = utility[add_order]
    removal_positions = removal_positions[
        torch.argsort(
            removal_cost[removal_positions],
            descending=False,
            stable=True,
        )
    ]

    minimum_ratio = min(
        0.50,
        max(
            0.0,
            _cfg_float(config, "certv11_min_swap_ratio", 0.04),
        ),
    )
    maximum_ratio = min(
        0.50,
        max(
            minimum_ratio,
            _cfg_float(config, "certv11_max_swap_ratio", 0.12),
        ),
    )
    deficit_scale = min(
        1.0,
        deficit
        / max(
            1e-6,
            _cfg_float(config, "certv11_deficit_scale", 0.20),
        ),
    )
    ratio = minimum_ratio + (maximum_ratio - minimum_ratio) * deficit_scale
    target = min(
        int(candidates.numel()),
        int(removal_positions.numel()),
        max(1, int(math.ceil(budget * ratio))),
    )
    if target <= 0:
        return _RepairResult(
            selected=base_selected,
            swaps=[],
            node_efficiency=1.0,
            edge_coverage_before=edge_coverage_before,
            edge_coverage_after=edge_coverage_before,
            joint_before=base_joint_logdet,
            joint_after=base_joint_logdet,
            objective_gain=0.0,
            transition_weight=transition_weight,
        )

    candidates = candidates[:target]
    removal_positions = removal_positions[:target]
    frame_count = int(analysis.frame_ids.max().item()) + 1
    base_counts = torch.bincount(
        analysis.frame_ids[base_selected],
        minlength=frame_count,
    )
    base_entropy = _temporal_entropy(base_counts)

    trial_count = min(10, target)
    sizes = {
        1,
        target,
        max(1, int(math.ceil(budget * minimum_ratio))),
    }
    for index in range(1, trial_count + 1):
        sizes.add(max(1, int(round(target * index / trial_count))))
    floor = min(
        1.0,
        max(
            0.0,
            _cfg_float(config, "certv11_node_efficiency_floor", 0.95),
        ),
    )
    node_loss_weight = max(
        0.0,
        _cfg_float(config, "certv11_node_loss_weight", 0.35),
    )
    relation_weight = max(
        0.0,
        _cfg_float(config, "certv11_edge_coverage_weight", 0.30),
    )
    balance_weight = max(
        0.0,
        _cfg_float(config, "certv11_frame_balance_weight", 0.08),
    )
    margin = _cfg_float(config, "certv11_swap_margin", 0.0)

    best_selected = base_selected
    best_size = 0
    best_efficiency = 1.0
    best_edge_coverage = edge_coverage_before
    best_joint_logdet = base_joint_logdet
    best_gain = 0.0
    for size in sorted(size for size in sizes if 0 < size <= target):
        keep = torch.ones(
            budget,
            dtype=torch.bool,
            device=base_selected.device,
        )
        keep[removal_positions[:size]] = False
        proposed = torch.sort(
            torch.cat(
                [base_selected[keep], candidates[:size]],
                dim=0,
            )
        ).values
        if (
            proposed.numel() != budget
            or torch.unique(proposed).numel() != budget
        ):
            continue
        if not _preserves_group_support(
            analysis,
            base_selected,
            proposed,
        ):
            continue
        node_logdet = _evaluate_selection(
            analysis.design,
            proposed,
            analysis.ridge,
        )
        efficiency = math.exp(
            (node_logdet - base_node_logdet)
            / max(1, int(analysis.design.shape[1]))
        )
        if efficiency + 1e-8 < floor:
            continue
        joint_logdet = _evaluate_selection(
            joint,
            proposed,
            analysis.ridge,
        )
        joint_gain = (
            joint_logdet - base_joint_logdet
        ) / max(1, int(joint.shape[1]))
        proposed_coverage = _same_frame_coverage(
            analysis,
            proposed,
        )
        edge_coverage_after, _, _ = _edge_coverage(
            graph,
            proposed_coverage,
            transition.edge_importance,
        )
        edge_gain = edge_coverage_after - edge_coverage_before
        if edge_gain <= 1e-6:
            continue
        counts = torch.bincount(
            analysis.frame_ids[proposed],
            minlength=frame_count,
        )
        entropy_gain = _temporal_entropy(counts) - base_entropy
        objective_gain = (
            joint_gain
            + relation_weight * edge_gain
            + balance_weight * entropy_gain
            - node_loss_weight * max(0.0, 1.0 - efficiency)
        )
        if objective_gain > best_gain + margin:
            best_selected = proposed
            best_size = size
            best_efficiency = efficiency
            best_edge_coverage = edge_coverage_after
            best_joint_logdet = joint_logdet
            best_gain = objective_gain

    if best_size <= 0:
        return _RepairResult(
            selected=base_selected,
            swaps=[],
            node_efficiency=1.0,
            edge_coverage_before=edge_coverage_before,
            edge_coverage_after=edge_coverage_before,
            joint_before=base_joint_logdet,
            joint_after=base_joint_logdet,
            objective_gain=0.0,
            transition_weight=transition_weight,
        )

    swaps: list[dict[str, Any]] = []
    candidate_set = set(
        int(token)
        for token in analysis.candidate_indices.detach().cpu().tolist()
    )
    add_tokens_cpu = candidates[:best_size].detach().cpu().tolist()
    remove_positions_cpu = (
        removal_positions[:best_size].detach().cpu().tolist()
    )
    frame_ids_cpu = analysis.frame_ids.detach().cpu()
    component_ids_cpu = analysis.component_ids.detach().cpu()
    relation_cpu = transition.token_relation.detach().cpu()
    state_cpu = transition.token_state.detach().cpu()
    base_selected_cpu = base_selected.detach().cpu()
    for add_token, remove_position in zip(
        add_tokens_cpu,
        remove_positions_cpu,
    ):
        add_token = int(add_token)
        remove_token = int(base_selected_cpu[int(remove_position)])
        swaps.append(
            {
                "add": add_token,
                "remove": remove_token,
                "provenance": provenance.get(
                    add_token,
                    "transition_endpoint",
                ),
                "add_frame": int(frame_ids_cpu[add_token]),
                "remove_frame": int(frame_ids_cpu[remove_token]),
                "add_component": int(component_ids_cpu[add_token]),
                "remove_component": int(component_ids_cpu[remove_token]),
                "add_relation": float(relation_cpu[add_token]),
                "add_state": float(state_cpu[add_token]),
                "remove_relation": float(relation_cpu[remove_token]),
                "outside_v3_candidate_pool": add_token not in candidate_set,
            }
        )
    return _RepairResult(
        selected=best_selected,
        swaps=swaps,
        node_efficiency=best_efficiency,
        edge_coverage_before=edge_coverage_before,
        edge_coverage_after=best_edge_coverage,
        joint_before=base_joint_logdet,
        joint_after=best_joint_logdet,
        objective_gain=best_gain,
        transition_weight=transition_weight,
    )


def _build_correspondence_plan(
    selected: torch.Tensor,
    base_selected: torch.Tensor,
    base_plan: CertVidPlan,
    promoted: set[int],
    locked: set[int],
    analysis: _V3Analysis,
    graph: _CorrespondenceGraph,
    frame_times: torch.Tensor,
    has_real_times: bool,
    config: Any,
) -> tuple[CertVidPlan, dict[str, Any]]:
    total_tokens = int(analysis.metric_flat.shape[0])
    budget = int(selected.numel())
    if not _preserves_group_support(
        analysis,
        base_selected,
        selected,
    ):
        raise RuntimeError("V11 repair removed the last frame/bin anchor")

    raw_similarity = (
        analysis.metric_flat.float()
        @ analysis.metric_flat[selected].float().transpose(0, 1)
    )
    source_frame = analysis.frame_ids.unsqueeze(1)
    anchor_frame = analysis.frame_ids[selected].unsqueeze(0)
    same_frame = source_frame == anchor_frame
    valid = same_frame.clone()

    selected_position = torch.full(
        (total_tokens,),
        -1,
        dtype=torch.long,
        device=selected.device,
    )
    selected_position[selected] = torch.arange(
        budget,
        dtype=torch.long,
        device=selected.device,
    )
    target_positions = selected_position[graph.target]
    source_positions = selected_position[graph.source]
    target_is_anchor = target_positions >= 0
    source_is_anchor = source_positions >= 0
    direct_edge_valid = torch.ones(
        graph.source.numel(),
        dtype=torch.bool,
        device=selected.device,
    )
    cross_frame_max_seconds = _cfg_float(
        config,
        "certv11_cross_frame_max_seconds",
        12.0,
    )
    if has_real_times and graph.source.numel() > 0:
        edge_time_gap = torch.abs(
            frame_times[analysis.frame_ids[graph.source]]
            - frame_times[analysis.frame_ids[graph.target]]
        )
        direct_edge_valid &= edge_time_gap <= cross_frame_max_seconds
    direct_time_rejected = int((~direct_edge_valid).sum().item())

    forward_valid = target_is_anchor & direct_edge_valid
    valid[
        graph.source[forward_valid],
        target_positions[forward_valid],
    ] = True
    reverse_valid = source_is_anchor & direct_edge_valid
    valid[
        graph.target[reverse_valid],
        source_positions[reverse_valid],
    ] = True

    adjacent = (source_frame - anchor_frame).abs() == 1
    no_scene_cut = torch.zeros_like(adjacent)
    if graph.scene_boundary.numel() > 0:
        lower_frame = torch.minimum(source_frame, anchor_frame)
        pair_ids = lower_frame.clamp(
            min=0,
            max=max(0, graph.scene_boundary.numel() - 1),
        )
        no_scene_cut[adjacent] = ~graph.scene_boundary[pair_ids[adjacent]]
    high_similarity_fallback = (
        adjacent
        & no_scene_cut
        & (
            raw_similarity
            >= _cfg_float(
                config,
                "certv11_cross_frame_similarity",
                0.92,
            )
        )
    )
    if has_real_times:
        time_gap = torch.abs(
            frame_times[source_frame] - frame_times[anchor_frame]
        )
        high_similarity_fallback &= time_gap <= cross_frame_max_seconds
    valid |= high_similarity_fallback
    if not bool(valid.any(dim=1).all()):
        raise RuntimeError("V11 plan has a token without a reachable anchor")

    score = raw_similarity.masked_fill(~valid, -2.0)
    same_component = (
        analysis.component_ids.unsqueeze(1)
        == analysis.component_ids[selected].unsqueeze(0)
    )
    score += 0.05 * same_component.float()

    base_assignment = base_plan.assignment_indices.to(
        device=selected.device,
        dtype=torch.long,
    )
    base_weights = base_plan.assignment_weights.to(
        device=selected.device,
        dtype=torch.float32,
    )
    if (
        base_assignment.ndim != 2
        or base_weights.shape != base_assignment.shape
        or int(base_assignment.shape[0]) != total_tokens
    ):
        raise ValueError("V3 assignment plan has an invalid shape")
    old_target_tokens = base_selected[base_assignment]
    mapped_base = selected_position[old_target_tokens]
    mapped_safe = mapped_base.clamp_min(0)
    preserved = mapped_base >= 0
    preserved &= torch.gather(valid, 1, mapped_safe)
    preserved_weights = base_weights * preserved.float()
    base_similarity = torch.gather(raw_similarity, 1, mapped_safe)

    # Retained V3 edges keep their exact mass. Only missing or structurally
    # invalid mass is routed through correspondence-constrained candidates.
    repair_score = score.clone()
    for neighbor in range(base_assignment.shape[1]):
        rows = torch.nonzero(
            preserved[:, neighbor],
            as_tuple=False,
        ).flatten()
        if rows.numel() > 0:
            repair_score[
                rows,
                mapped_safe[rows, neighbor],
            ] = -2.0

    repair_topk = min(
        budget,
        max(1, _cfg_int(config, "certv11_assignment_topk", 2)),
    )
    values, repair_assignment = torch.topk(
        repair_score,
        k=repair_topk,
        dim=1,
        largest=True,
    )
    repair_valid = torch.gather(valid, 1, repair_assignment)
    repair_similarity = torch.gather(
        raw_similarity,
        1,
        repair_assignment,
    )
    fusion_floor = _cfg_float(
        config,
        "certv11_fusion_similarity_floor",
        0.70,
    )
    repair_valid &= repair_similarity >= fusion_floor
    temperature = max(
        1e-4,
        _cfg_float(config, "certv11_assignment_temperature", 0.07),
    )
    safe_values = (values.float() / temperature).masked_fill(
        ~repair_valid,
        -1e4,
    )
    repair_weights = torch.softmax(safe_values, dim=1)
    repair_weights *= repair_valid.float()
    repair_weights /= repair_weights.sum(
        dim=1,
        keepdim=True,
    ).clamp_min(1e-6)
    missing_mass = (
        1.0 - preserved_weights.sum(dim=1, keepdim=True)
    ).clamp(0.0, 1.0)
    repair_weights *= missing_mass

    assignment = torch.cat([mapped_safe, repair_assignment], dim=1)
    weights = torch.cat([preserved_weights, repair_weights], dim=1)
    chosen_similarity = torch.cat(
        [base_similarity, repair_similarity],
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
    chosen_similarity[selected] = 0.0
    chosen_similarity[selected, 0] = 1.0

    source_mass = base_plan.source_mass.to(
        device=selected.device,
        dtype=torch.float32,
    ).clone()
    if source_mass.ndim != 1 or int(source_mass.numel()) != total_tokens:
        raise ValueError("V3 source mass has an invalid shape")
    alpha = torch.zeros(
        budget,
        dtype=torch.float32,
        device=selected.device,
    )
    old_position_by_token = torch.full(
        (total_tokens,),
        -1,
        dtype=torch.long,
        device=selected.device,
    )
    old_position_by_token[base_selected] = torch.arange(
        base_selected.numel(),
        dtype=torch.long,
        device=selected.device,
    )
    old_positions = old_position_by_token[selected]
    retained = old_positions >= 0
    base_alpha = base_plan.fusion_alpha.to(
        device=selected.device,
        dtype=torch.float32,
    )
    alpha[retained] = base_alpha[old_positions[retained]]

    target_mass = torch.zeros_like(alpha)
    target_similarity = torch.zeros_like(alpha)
    for neighbor in range(assignment.shape[1]):
        mass = weights[:, neighbor] * source_mass
        target = assignment[:, neighbor]
        target_mass.index_add_(0, target, mass)
        target_similarity.index_add_(
            0,
            target,
            mass * chosen_similarity[:, neighbor],
        )
    mean_similarity = target_similarity / target_mass.clamp_min(1e-6)
    confidence = (
        (mean_similarity - fusion_floor)
        / max(1e-6, 1.0 - fusion_floor)
    ).clamp(0.0, 1.0)
    affected_anchors = torch.zeros(
        budget,
        dtype=torch.bool,
        device=selected.device,
    )
    for neighbor in range(base_assignment.shape[1]):
        lost = (
            (mapped_base[:, neighbor] >= 0)
            & (~preserved[:, neighbor])
            & (base_weights[:, neighbor] > 1e-8)
        )
        affected_anchors[mapped_safe[lost, neighbor]] = True
    for neighbor in range(repair_assignment.shape[1]):
        routed = repair_weights[:, neighbor] > 1e-8
        affected_anchors[repair_assignment[routed, neighbor]] = True
    alpha[affected_anchors] *= confidence[affected_anchors]
    protected = locked | promoted
    if protected:
        protected_tokens = torch.tensor(
            sorted(protected),
            dtype=torch.long,
            device=selected.device,
        )
        alpha[torch.isin(selected, protected_tokens)] = 0.0

    plan = CertVidPlan(
        anchor_indices=selected,
        assignment_indices=assignment,
        assignment_weights=weights,
        source_mass=source_mass,
        fusion_alpha=alpha,
        raw_token_count=total_tokens,
    )
    residual = torch.ones(
        total_tokens,
        dtype=torch.bool,
        device=selected.device,
    )
    residual[selected] = False
    active_similarity = chosen_similarity.masked_fill(weights <= 0.0, -1.0)
    best_similarity = active_similarity.amax(dim=1)
    residual_similarity = best_similarity[residual]
    assigned_similarity = residual_similarity[residual_similarity >= 0.0]
    if assigned_similarity.numel() > 0:
        quantiles = torch.quantile(
            assigned_similarity,
            torch.tensor(
                [0.0, 0.05, 0.50],
                dtype=torch.float32,
                device=selected.device,
            ),
        )
        summary = {
            "min": float(quantiles[0].item()),
            "p05": float(quantiles[1].item()),
            "median": float(quantiles[2].item()),
        }
    else:
        summary = {"min": 1.0, "p05": 1.0, "median": 1.0}

    best_column = weights.argmax(dim=1, keepdim=True)
    first_target = selected[
        torch.gather(assignment, 1, best_column).squeeze(1)
    ]
    cross_frame = (
        analysis.frame_ids != analysis.frame_ids[first_target]
    ) & residual
    residual_count = residual.sum().clamp_min(1).float()
    preserved_mass_ratio = float(
        (preserved_weights[residual].sum() / residual_count).item()
    )
    rerouted = residual & (missing_mass.squeeze(1) > 1e-6)
    zero_assignment = residual & (weights.sum(dim=1) <= 1e-8)
    stats = {
        "assignment_similarity": summary,
        "v3_assignment_mass_preserved": preserved_mass_ratio,
        "rerouted_source_rate": float(
            rerouted[residual].float().mean().item()
        )
        if bool(residual.any())
        else 0.0,
        "zero_assignment_source_rate": float(
            zero_assignment[residual].float().mean().item()
        )
        if bool(residual.any())
        else 0.0,
        "direct_edge_time_rejected_count": direct_time_rejected,
        "affected_fusion_anchor_count": int(
            affected_anchors.sum().item()
        ),
        "same_frame_assignment_rate": float(
            (
                analysis.frame_ids[residual]
                == analysis.frame_ids[first_target[residual]]
            ).float().mean().item()
        )
        if bool(residual.any())
        else 1.0,
        "cross_frame_assignment_rate": float(
            cross_frame[residual].float().mean().item()
        )
        if bool(residual.any())
        else 0.0,
        "low_confidence_source_rate": float(
            (
                (residual_similarity < fusion_floor)
                & (residual_similarity >= 0.0)
            ).float().mean().item()
        )
        if residual_similarity.numel() > 0
        else 0.0,
        "fusion_suppressed_anchor_count": int(
            (affected_anchors & (confidence <= 1e-6)).sum().item()
        ),
        "mean_fusion_alpha": float(alpha.mean().item()),
    }
    return plan, stats

def _json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _store_diagnostics(
    config: Any,
    diagnostics: Dict[str, Any],
) -> None:
    config.last_certv11_diagnostics = diagnostics
    config.last_certv11_fallback_reason = diagnostics.get("fallback_reason")
    config.last_certv11_swap_count = int(diagnostics.get("swap_count", 0))
    config.last_certv11_v3_overlap_ratio = float(
        diagnostics.get("v3_overlap_ratio", 1.0)
    )
    config.last_certv11_node_efficiency = float(
        diagnostics.get("node_efficiency", 1.0)
    )
    config.last_certv11_structural_deficit = float(
        diagnostics.get("structural_deficit", 0.0)
    )
    template = os.environ.get(
        "CERTV11_DIAGNOSTICS_JSONL",
        "",
    ).strip()
    if template:
        rank = os.environ.get(
            "LOCAL_RANK",
            os.environ.get("RANK", "0"),
        )
        world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
        path = template.replace("{rank}", rank).replace(
            "{pid}",
            str(os.getpid()),
        )
        if (
            world_size > 1
            and "{rank}" not in template
            and "{pid}" not in template
        ):
            root, extension = os.path.splitext(path)
            path = f"{root}.rank{rank}{extension or '.jsonl'}"
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        record = dict(diagnostics)
        record["sample_id"] = str(
            getattr(config, "_debug_sample_id", "unknown")
        )
        record["question"] = str(
            getattr(config, "_certvid_query_text", "") or ""
        )
        handle = _DIAGNOSTIC_HANDLES.get(path)
        if handle is None or handle.closed:
            handle = open(
                path,
                "a",
                encoding="utf-8",
                buffering=1,
            )
            _DIAGNOSTIC_HANDLES[path] = handle
        handle.write(
            json.dumps(
                _json_safe(record),
                sort_keys=True,
            )
            + "\n"
        )
    if _cfg_bool(config, "certv11_debug", False):
        print(
            "[certvid-v11] "
            f"sample={getattr(config, '_debug_sample_id', 'unknown')} "
            f"fallback={diagnostics.get('fallback_reason')} "
            f"edges={diagnostics.get('graph_valid_edge_count', 0)} "
            f"deficit={diagnostics.get('structural_deficit', 0.0):.4f} "
            f"swaps={diagnostics.get('swap_count', 0)} "
            f"overlap={diagnostics.get('v3_overlap_ratio', 1.0):.4f} "
            f"node_eff={diagnostics.get('node_efficiency', 1.0):.4f}"
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
    diagnostics.setdefault("node_efficiency", 1.0)
    config._certvid_plan = plan
    config.vision_token_length = int(output.shape[0])
    config.visual_token_length = int(output.shape[0])
    config.llm_token_length = None
    config.last_adapter_variant = "certvid_v11"
    config.last_adapter_raw_tokens = float(plan.raw_token_count)
    config.last_adapter_output_tokens = float(output.shape[0])
    _store_diagnostics(config, diagnostics)
    return output, indices


def certvid_v11_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    config: Any,
    question_features: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repair V3 with reliable appearance-motion transition endpoints."""
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

    frame_count, tokens_per_frame, feature_dim = video_features.shape
    diagnostics: Dict[str, Any] = {
        "fallback_reason": None,
        "raw_token_count": int(frame_count * tokens_per_frame),
        "budget": int(v3_indices.numel()),
        "swap_count": 0,
        "v3_overlap_ratio": 1.0,
        "node_efficiency": 1.0,
        "structural_deficit": 0.0,
    }
    if not _cfg_bool(config, "certv11_enabled", True):
        return _fallback(
            config,
            diagnostics,
            "disabled",
            v3_output,
            v3_indices,
            v3_plan,
        )
    if sink.get("identity", False):
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
        graph = _build_correspondence_graph(
            analysis,
            frame_count,
            tokens_per_frame,
            frame_times,
            has_real_times,
            config,
        )
        diagnostics.update(
            {
                "timestamp_source": timestamp_source,
                "has_real_timestamps": has_real_times,
                "graph_valid_edge_count": int(graph.source.numel()),
                "graph_valid_edge_rate": graph.valid_edge_rate,
                "cycle_consistency_rate": graph.cycle_consistency_rate,
                "graph_reliability": graph.reliability,
                "scene_boundary_count": int(
                    graph.scene_boundary.sum().item()
                ),
            }
        )
        if graph.source.numel() == 0:
            return _fallback(
                config,
                diagnostics,
                "no_reliable_correspondence",
                v3_output,
                v3_indices,
                v3_plan,
            )
        if graph.reliability < _cfg_float(
            config,
            "certv11_reliability_floor",
            0.015,
        ):
            return _fallback(
                config,
                diagnostics,
                "graph_reliability_below_floor",
                v3_output,
                v3_indices,
                v3_plan,
            )

        transition = _transition_state(analysis, graph, config)
        base_coverage = _same_frame_coverage(analysis, v3_indices)
        edge_coverage_before, deficit, edge_coverage_tensor = _edge_coverage(
            graph,
            base_coverage,
            transition.edge_importance,
        )
        diagnostics.update(
            {
                "structural_deficit": deficit,
                "edge_coverage_before": edge_coverage_before,
                "match_similarity_p05": float(
                    torch.quantile(graph.similarity, 0.05).item()
                ),
                "match_similarity_median": float(
                    torch.quantile(graph.similarity, 0.50).item()
                ),
                "match_similarity_p95": float(
                    torch.quantile(graph.similarity, 0.95).item()
                ),
                "match_margin_median": float(
                    torch.quantile(graph.margin, 0.50).item()
                ),
                "spatial_displacement_median": float(
                    torch.quantile(
                        torch.linalg.vector_norm(
                            graph.displacement,
                            dim=-1,
                        ),
                        0.50,
                    ).item()
                ),
                "state_change_median": float(
                    torch.quantile(graph.state_change, 0.50).item()
                ),
            }
        )
        if deficit < _cfg_float(
            config,
            "certv11_deficit_threshold",
            0.025,
        ):
            return _fallback(
                config,
                diagnostics,
                "transition_coverage_sufficient",
                v3_output,
                v3_indices,
                v3_plan,
            )

        joint, transition_weight = _joint_design(
            analysis,
            transition,
            graph,
            deficit,
            config,
        )
        protected, node_leverage = _protected_anchors(
            analysis,
            v3_indices,
            v3_plan,
            transition,
            config,
        )
        candidates, candidate_score, provenance = _repair_candidates(
            analysis,
            v3_indices,
            graph,
            transition,
            base_coverage,
            edge_coverage_tensor,
            config,
        )
        removal_positions, removal_cost = _removal_pool(
            analysis,
            v3_indices,
            protected,
            node_leverage,
            transition,
            config,
        )
        repair = _exchange_repair(
            analysis,
            v3_indices,
            candidates,
            candidate_score,
            provenance,
            removal_positions,
            removal_cost,
            joint,
            graph,
            transition,
            edge_coverage_before,
            deficit,
            transition_weight,
            config,
        )
        diagnostics.update(
            {
                "transition_weight": repair.transition_weight,
                "target_add_pool": int(candidates.numel()),
                "removal_pool": int(removal_positions.numel()),
                "protected_anchor_count": int(protected.sum().item()),
                "joint_objective_before": repair.joint_before,
                "joint_objective_after": repair.joint_after,
                "joint_objective_gain": repair.objective_gain,
                "node_efficiency": repair.node_efficiency,
                "repair_edge_coverage_before": (
                    repair.edge_coverage_before
                ),
                "repair_edge_coverage_after": (
                    repair.edge_coverage_after
                ),
            }
        )
        if not repair.swaps:
            return _fallback(
                config,
                diagnostics,
                "no_positive_joint_repair",
                v3_output,
                v3_indices,
                v3_plan,
            )

        promoted = {int(record["add"]) for record in repair.swaps}
        locked = {
            int(token)
            for token in v3_indices[
                v3_plan.fusion_alpha <= 1e-12
            ].detach().cpu().tolist()
        }
        plan, plan_stats = _build_correspondence_plan(
            repair.selected,
            v3_indices,
            v3_plan,
            promoted,
            locked,
            analysis,
            graph,
            frame_times,
            has_real_times,
            config,
        )
        output = apply_certvid_plan(
            video_features.reshape(-1, feature_dim),
            plan,
        )
        if (
            output.shape[0] != v3_output.shape[0]
            or repair.selected.numel() != v3_indices.numel()
            or torch.unique(repair.selected).numel()
            != repair.selected.numel()
            or not bool(torch.isfinite(output).all())
        ):
            raise RuntimeError(
                "V11 output failed budget, uniqueness, or finite validation"
            )

        final_coverage = _same_frame_coverage(
            analysis,
            repair.selected,
        )
        edge_coverage_after, _, _ = _edge_coverage(
            graph,
            final_coverage,
            transition.edge_importance,
        )
        if edge_coverage_after <= edge_coverage_before + 1e-6:
            raise RuntimeError(
                "V11 repair did not improve transition endpoint coverage"
            )
        v3_frame_counts = torch.bincount(
            analysis.frame_ids[v3_indices],
            minlength=frame_count,
        )
        final_frame_counts = torch.bincount(
            analysis.frame_ids[repair.selected],
            minlength=frame_count,
        )
        diagnostics.update(
            {
                "fallback_reason": None,
                "swap_count": len(repair.swaps),
                "swaps": repair.swaps,
                "v3_overlap_ratio": float(
                    torch.isin(
                        repair.selected,
                        v3_indices,
                    ).float().mean().item()
                ),
                "promoted_outside_v3_candidate_count": sum(
                    bool(record["outside_v3_candidate_pool"])
                    for record in repair.swaps
                ),
                "edge_coverage_after": edge_coverage_after,
                "edge_coverage_gain": (
                    edge_coverage_after - edge_coverage_before
                ),
                "v3_frame_counts": v3_frame_counts,
                "final_frame_counts": final_frame_counts,
                "v3_temporal_entropy": _temporal_entropy(
                    v3_frame_counts
                ),
                "final_temporal_entropy": _temporal_entropy(
                    final_frame_counts
                ),
                **plan_stats,
            }
        )
        config._certvid_plan = plan
        config.vision_token_length = int(output.shape[0])
        config.visual_token_length = int(output.shape[0])
        config.llm_token_length = None
        config.last_adapter_variant = "certvid_v11"
        config.last_adapter_raw_tokens = float(
            frame_count * tokens_per_frame
        )
        config.last_adapter_output_tokens = float(output.shape[0])
        _store_diagnostics(config, diagnostics)
        return output, repair.selected
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            raise
        diagnostics["error"] = f"{type(error).__name__}: {error}"
        return _fallback(
            config,
            diagnostics,
            "repair_error",
            v3_output,
            v3_indices,
            v3_plan,
        )
    except (ValueError, IndexError) as error:
        diagnostics["error"] = f"{type(error).__name__}: {error}"
        return _fallback(
            config,
            diagnostics,
            "repair_error",
            v3_output,
            v3_indices,
            v3_plan,
        )
