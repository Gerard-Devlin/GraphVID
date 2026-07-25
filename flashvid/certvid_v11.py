"""CertVID V11: graph-structured spatio-temporal D-optimal selection."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Optional

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
from .certvid_v3 import _design_features


@dataclass
class _Analysis:
    metric_flat: torch.Tensor
    metric_frames: torch.Tensor
    design: torch.Tensor
    quality: torch.Tensor
    event_score: torch.Tensor
    demand_weight: torch.Tensor
    attention: torch.Tensor
    query_score: torch.Tensor
    query_relevance: torch.Tensor
    query_confidence: float
    atom_weights: torch.Tensor
    novelty: torch.Tensor
    curvature: torch.Tensor
    event: torch.Tensor
    detail: torch.Tensor
    frame_ids: torch.Tensor
    temporal_ids: torch.Tensor
    spatial_ids: torch.Tensor
    coords: torch.Tensor
    ridge: float


@dataclass
class _GSTM:
    parent: torch.Tensor
    tree_ids: torch.Tensor
    branch_ids: torch.Tensor
    segment_ids: torch.Tensor
    depth: torch.Tensor
    edge_similarity: torch.Tensor
    motion_vector: torch.Tensor
    spatial_jump: torch.Tensor
    state_change: torch.Tensor
    turn: torch.Tensor
    child_count: torch.Tensor
    endpoint: torch.Tensor
    node_score: torch.Tensor
    tree_score: torch.Tensor
    tree_size: torch.Tensor
    tree_span: torch.Tensor
    tree_root: torch.Tensor
    tree_leaf: torch.Tensor
    tree_turn: torch.Tensor
    tree_change: torch.Tensor
    frame_score: torch.Tensor
    reliability: float
    branch_count: int


def _cfg_bool(config: Any, name: str, default: bool) -> bool:
    value = getattr(config, name, default)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


def _effective_ratio(config: Any) -> float:
    ratio = _cfg_float(config, "retention_ratio", 0.10)
    if _cfg_bool(config, "certv3_budget_uses_expansion", True):
        ratio *= _cfg_float(config, "expansion", 1.0)
    return min(1.0, max(0.0, ratio))


def _identity_plan(total_tokens: int, device: torch.device) -> CertVidPlan:
    indices = torch.arange(total_tokens, dtype=torch.long, device=device)
    return CertVidPlan(
        anchor_indices=indices,
        assignment_indices=indices.unsqueeze(1),
        assignment_weights=torch.ones(
            (total_tokens, 1), dtype=torch.float32, device=device
        ),
        source_mass=torch.ones(total_tokens, dtype=torch.float32, device=device),
        fusion_alpha=torch.zeros(total_tokens, dtype=torch.float32, device=device),
        raw_token_count=total_tokens,
    )


def _build_analysis(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    question_features: Optional[torch.Tensor],
    config: Any,
) -> _Analysis:
    frame_count, tokens_per_frame, _ = video_features.shape
    total_tokens = frame_count * tokens_per_frame
    metric_dim = max(32, _cfg_int(config, "certv3_metric_dim", 96))
    metric_flat = _metric_features(video_features, metric_dim)
    metric_frames = metric_flat.view(frame_count, tokens_per_frame, -1)
    height, width = _grid_hw(tokens_per_frame, config)
    spatial_bins = max(1, _cfg_int(config, "certv3_spatial_bins", 3))
    coords, frame_spatial_ids = _spatial_layout(
        tokens_per_frame,
        height,
        width,
        spatial_bins,
        video_features.device,
    )
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
    component_ids = component_ids_cpu.to(video_features.device)
    component_sizes = component_sizes_cpu.to(video_features.device)
    frame_ids = torch.arange(
        frame_count, device=video_features.device
    ).repeat_interleave(tokens_per_frame)
    component_value = _component_support(
        metric_flat,
        component_ids,
        component_sizes,
        frame_ids,
        frame_count,
    )
    temporal_count = min(
        frame_count, max(1, _cfg_int(config, "certv3_temporal_bins", 12))
    )
    temporal_ids = torch.div(
        frame_ids * temporal_count,
        max(1, frame_count),
        rounding_mode="floor",
    ).clamp_max(temporal_count - 1)
    spatial_ids = frame_spatial_ids.repeat(frame_count)
    spatial_count = spatial_bins * spatial_bins

    if cls_attention.numel() != total_tokens:
        raise ValueError(
            f"expected {total_tokens} attention scores, got {cls_attention.numel()}"
        )
    attention = _rank_normalize(cls_attention.float()).reshape(-1)
    novelty = novelty_2d.reshape(-1)
    curvature = curvature_2d.reshape(-1)
    detail = _local_detail(video_features, height, width).reshape(-1)
    event = frame_event.repeat_interleave(tokens_per_frame)
    atoms = _question_atoms(
        question_features,
        max(0, _cfg_int(config, "certv3_query_atoms", 8)),
        metric_dim,
    ).to(video_features.device)
    query_relevance, atom_weights, query_confidence = _question_relevance(
        atoms, metric_flat
    )
    query_score = (
        (query_relevance * atom_weights.unsqueeze(1)).sum(dim=0)
        if query_relevance.numel() > 0
        else torch.zeros(
            total_tokens, dtype=torch.float32, device=video_features.device
        )
    )
    query_weight = min(
        0.30,
        max(
            0.0,
            _cfg_float(config, "certv3_query_weight", 0.18)
            * query_confidence,
        ),
    )
    visual_quality = _minmax(
        0.28 * attention
        + 0.20 * novelty
        + 0.14 * curvature
        + 0.12 * event
        + 0.12 * detail
        + 0.14 * component_value,
        dim=0,
    )
    quality = _minmax(
        (1.0 - query_weight) * visual_quality + query_weight * query_score,
        dim=0,
    )
    event_score = _minmax(
        0.34 * novelty
        + 0.28 * curvature
        + 0.18 * event
        + 0.10 * detail
        + 0.10 * query_score,
        dim=0,
    )
    demand_weight = (
        0.20 + 0.42 * quality + 0.20 * event_score + 0.18 * component_value
    )
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
        whitening_strength=_cfg_float(
            config, "certv3_whitening_strength", 0.50
        ),
        quality_floor=_cfg_float(config, "certv3_quality_floor", 0.15),
    )
    return _Analysis(
        metric_flat=metric_flat,
        metric_frames=metric_frames,
        design=design,
        quality=quality,
        event_score=event_score,
        demand_weight=demand_weight,
        attention=attention,
        query_score=query_score,
        query_relevance=query_relevance,
        query_confidence=float(query_confidence),
        atom_weights=atom_weights,
        novelty=novelty,
        curvature=curvature,
        event=event,
        detail=detail,
        frame_ids=frame_ids,
        temporal_ids=temporal_ids,
        spatial_ids=spatial_ids,
        coords=coords,
        ridge=max(1e-4, _cfg_float(config, "certv3_ridge", 0.50)),
    )


def _argmax_by_group(
    values: torch.Tensor,
    groups: torch.Tensor,
    group_count: int,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    output = torch.full(
        (group_count,), -1, dtype=torch.long, device=values.device
    )
    valid = (
        torch.ones_like(groups, dtype=torch.bool)
        if mask is None
        else mask.to(device=groups.device, dtype=torch.bool)
    )
    token_ids = torch.where(valid)[0].detach().cpu().tolist()
    values_cpu = values.detach().float().cpu().tolist()
    groups_cpu = groups.detach().cpu().tolist()
    best_cpu = [float("-inf")] * group_count
    output_cpu = [-1] * group_count
    for token in token_ids:
        group = int(groups_cpu[token])
        score = float(values_cpu[token])
        if score > best_cpu[group] or (
            score == best_cpu[group]
            and (output_cpu[group] < 0 or token < output_cpu[group])
        ):
            best_cpu[group] = score
            output_cpu[group] = token
    output.copy_(torch.tensor(output_cpu, dtype=torch.long, device=values.device))
    return output


def _build_gstm(analysis: _Analysis, config: Any) -> _GSTM:
    frame_count, tokens_per_frame, _ = analysis.metric_frames.shape
    total_tokens = frame_count * tokens_per_frame
    device = analysis.metric_flat.device
    frame_reps = F.normalize(
        analysis.metric_frames.mean(dim=1), p=2, dim=-1, eps=1e-6
    )
    frame_gap = torch.zeros(frame_count, dtype=torch.float32, device=device)
    if frame_count > 1:
        frame_gap[1:] = (
            1.0 - torch.sum(frame_reps[1:] * frame_reps[:-1], dim=-1)
        ).clamp(0.0, 2.0) * 0.5
    maximum_segments = min(
        frame_count, max(1, _cfg_int(config, "certv11_max_segments", 8))
    )
    similarity_threshold = _cfg_float(
        config, "certv11_segment_similarity", 0.82
    )
    candidate_boundaries = torch.where(
        frame_gap[1:] > (1.0 - similarity_threshold) * 0.5
    )[0] + 1
    if candidate_boundaries.numel() > maximum_segments - 1:
        strengths = frame_gap[candidate_boundaries]
        keep = torch.topk(
            strengths, k=maximum_segments - 1, largest=True
        ).indices
        candidate_boundaries = candidate_boundaries[keep]
    boundaries = torch.zeros(frame_count, dtype=torch.bool, device=device)
    boundaries[candidate_boundaries] = True
    segment_per_frame = torch.cumsum(boundaries.long(), dim=0)

    parent = torch.full(
        (total_tokens,), -1, dtype=torch.long, device=device
    )
    edge_similarity = torch.zeros(
        total_tokens, dtype=torch.float32, device=device
    )
    motion_vector = torch.zeros(
        (total_tokens, 2), dtype=torch.float32, device=device
    )
    spatial_jump = torch.zeros(
        total_tokens, dtype=torch.float32, device=device
    )
    threshold = _cfg_float(config, "certv11_tree_threshold", 0.78)
    spatial_penalty = max(
        0.0, _cfg_float(config, "certv11_spatial_penalty", 0.04)
    )
    max_spatial_jump = max(
        0.0, _cfg_float(config, "certv11_max_spatial_jump", 0.50)
    )
    spatial_distance = torch.cdist(
        analysis.coords.float(), analysis.coords.float(), p=2
    )
    for frame in range(1, frame_count):
        if int(segment_per_frame[frame].item()) != int(
            segment_per_frame[frame - 1].item()
        ):
            continue
        current = analysis.metric_frames[frame]
        previous = analysis.metric_frames[frame - 1]
        raw_similarity = current @ previous.transpose(0, 1)
        score = raw_similarity - spatial_penalty * spatial_distance
        best_previous = score.argmax(dim=1)
        token_ids = torch.arange(tokens_per_frame, device=device)
        matched = raw_similarity[token_ids, best_previous]
        displacement = (
            analysis.coords[token_ids] - analysis.coords[best_previous]
        )
        jump = displacement.norm(dim=-1)
        valid = (matched >= threshold) & (jump <= max_spatial_jump)
        current_global = frame * tokens_per_frame + token_ids
        parent[current_global[valid]] = (
            (frame - 1) * tokens_per_frame + best_previous[valid]
        )
        edge_similarity[current_global[valid]] = matched[valid]
        motion_vector[current_global[valid]] = displacement[valid]
        spatial_jump[current_global[valid]] = jump[valid]

    parent_cpu = parent.detach().cpu().tolist()
    roots = [0] * total_tokens
    depths = [0] * total_tokens
    for token in range(total_tokens):
        previous = int(parent_cpu[token])
        if previous < 0:
            roots[token] = token
            depths[token] = 0
        else:
            roots[token] = roots[previous]
            depths[token] = depths[previous] + 1
    root_to_tree: dict[int, int] = {}
    tree_ids_cpu: list[int] = []
    for root in roots:
        if root not in root_to_tree:
            root_to_tree[root] = len(root_to_tree)
        tree_ids_cpu.append(root_to_tree[root])
    tree_count = len(root_to_tree)
    tree_ids = torch.tensor(tree_ids_cpu, dtype=torch.long, device=device)
    depth = torch.tensor(depths, dtype=torch.long, device=device)

    child_count = torch.zeros(
        total_tokens, dtype=torch.long, device=device
    )
    children = torch.where(parent >= 0)[0]
    if children.numel() > 0:
        child_count.index_add_(
            0, parent[children], torch.ones_like(children, dtype=torch.long)
        )
    child_count_cpu = child_count.detach().cpu().tolist()
    branch_ids_cpu = [-1] * total_tokens
    branch_count = 0
    for token in range(total_tokens):
        previous = int(parent_cpu[token])
        if previous < 0 or int(child_count_cpu[previous]) > 1:
            branch_ids_cpu[token] = branch_count
            branch_count += 1
        else:
            branch_ids_cpu[token] = branch_ids_cpu[previous]
    branch_ids = torch.tensor(
        branch_ids_cpu, dtype=torch.long, device=device
    )

    state_change = torch.zeros(
        total_tokens, dtype=torch.float32, device=device
    )
    valid_edges = parent >= 0
    if valid_edges.any():
        state_change[valid_edges] = (
            1.0
            - torch.sum(
                analysis.metric_flat[valid_edges]
                * analysis.metric_flat[parent[valid_edges]],
                dim=-1,
            )
        ).clamp(0.0, 2.0) * 0.5
    turn = torch.zeros_like(state_change)
    grandparent = torch.full_like(parent, -1)
    parent_nodes = torch.where(valid_edges)[0]
    grandparent[parent_nodes] = parent[parent[parent_nodes]]
    turn_nodes = torch.where(grandparent >= 0)[0]
    if turn_nodes.numel() > 0:
        previous = parent[turn_nodes]
        grand_previous = grandparent[turn_nodes]
        current_local = torch.remainder(turn_nodes, tokens_per_frame)
        previous_local = torch.remainder(previous, tokens_per_frame)
        grand_local = torch.remainder(grand_previous, tokens_per_frame)
        incoming = (
            analysis.coords[previous_local] - analysis.coords[grand_local]
        )
        outgoing = (
            analysis.coords[current_local] - analysis.coords[previous_local]
        )
        direction = (
            1.0
            - F.cosine_similarity(
                incoming, outgoing, dim=-1, eps=1e-6
            )
        ).clamp(0.0, 2.0) * 0.5
        motion_gate = torch.sqrt(
            incoming.norm(dim=-1) * outgoing.norm(dim=-1)
        ).clamp(0.0, 1.0)
        turn[turn_nodes] = direction * motion_gate

    tree_size = torch.bincount(tree_ids, minlength=tree_count)
    first = torch.full(
        (tree_count,), frame_count, dtype=torch.long, device=device
    )
    last = torch.full(
        (tree_count,), -1, dtype=torch.long, device=device
    )
    for frame in range(frame_count):
        present = torch.unique(
            tree_ids[
                frame * tokens_per_frame : (frame + 1) * tokens_per_frame
            ]
        )
        first[present] = torch.minimum(
            first[present], torch.full_like(present, frame)
        )
        last[present] = torch.maximum(
            last[present], torch.full_like(present, frame)
        )
    tree_span = (last - first).clamp_min(0)
    endpoint = ((parent < 0) | (child_count == 0)) & (
        tree_span[tree_ids] > 0
    )
    branch_strength = _minmax(
        torch.log1p(child_count.float()), dim=0
    )
    node_score = _minmax(
        0.26 * _minmax(state_change, dim=0)
        + 0.24 * _minmax(turn, dim=0)
        + 0.16 * branch_strength
        + 0.12 * endpoint.float()
        + 0.10 * analysis.novelty
        + 0.12 * _minmax(spatial_jump, dim=0),
        dim=0,
    )
    tree_node_max = torch.zeros(
        tree_count, dtype=torch.float32, device=device
    )
    tree_node_max.scatter_reduce_(
        0, tree_ids, node_score, reduce="amax", include_self=True
    )
    tree_quality_max = torch.zeros_like(tree_node_max)
    tree_quality_max.scatter_reduce_(
        0, tree_ids, analysis.quality, reduce="amax", include_self=True
    )
    tree_query_max = torch.zeros_like(tree_node_max)
    tree_query_max.scatter_reduce_(
        0, tree_ids, analysis.query_score, reduce="amax", include_self=True
    )
    span_score = tree_span.float() / max(1, frame_count - 1)
    mass_score = torch.log1p(tree_size.float())
    mass_score = mass_score / mass_score.max().clamp_min(1e-6)
    tree_score = _minmax(
        0.32 * tree_node_max
        + 0.24 * span_score
        + 0.16 * mass_score
        + 0.18 * tree_quality_max
        + 0.10 * tree_query_max,
        dim=0,
    )
    tree_root = _argmax_by_group(
        analysis.quality + 0.25 * node_score,
        tree_ids,
        tree_count,
        parent < 0,
    )
    tree_leaf = _argmax_by_group(
        analysis.quality + 0.45 * state_change,
        tree_ids,
        tree_count,
        child_count == 0,
    )
    tree_turn = _argmax_by_group(turn, tree_ids, tree_count)
    tree_change = _argmax_by_group(
        state_change, tree_ids, tree_count
    )
    frame_score = torch.zeros(
        frame_count, dtype=torch.float32, device=device
    )
    for frame in range(frame_count):
        members = torch.where(analysis.frame_ids == frame)[0]
        keep = min(
            int(members.numel()),
            max(1, int(math.ceil(tokens_per_frame * 0.10))),
        )
        frame_score[frame] = (
            0.55 * torch.topk(node_score[members], k=keep).values.mean()
            + 0.25 * analysis.event[members[0]]
            + 0.20 * torch.topk(
                analysis.quality[members], k=keep
            ).values.mean()
        )
    segment_ids = segment_per_frame.repeat_interleave(tokens_per_frame)
    reliability = float(valid_edges.sum().item()) / max(
        1, (frame_count - 1) * tokens_per_frame
    )
    return _GSTM(
        parent=parent,
        tree_ids=tree_ids,
        branch_ids=branch_ids,
        segment_ids=segment_ids,
        depth=depth,
        edge_similarity=edge_similarity,
        motion_vector=motion_vector,
        spatial_jump=spatial_jump,
        state_change=_minmax(state_change, dim=0),
        turn=_minmax(turn, dim=0),
        child_count=child_count,
        endpoint=endpoint.float(),
        node_score=node_score,
        tree_score=tree_score,
        tree_size=tree_size,
        tree_span=tree_span,
        tree_root=tree_root,
        tree_leaf=tree_leaf,
        tree_turn=tree_turn,
        tree_change=tree_change,
        frame_score=_minmax(frame_score, dim=0),
        reliability=reliability,
        branch_count=branch_count,
    )


def _allocate_integer(
    weights: torch.Tensor,
    total: int,
    minimum: int,
    maximum: int,
) -> torch.Tensor:
    count = int(weights.numel())
    minimum = max(0, min(minimum, maximum))
    if count * minimum > total:
        minimum = 0
    output = [minimum] * count
    remaining = total - sum(output)
    weights_cpu = weights.detach().float().clamp_min(0.0).cpu().tolist()
    weight_sum = sum(weights_cpu)
    if weight_sum <= 1e-8:
        weights_cpu = [1.0] * count
        weight_sum = float(count)
    weights_cpu = [weight / weight_sum for weight in weights_cpu]
    while remaining > 0:
        eligible = [
            index for index, value in enumerate(output)
            if value < maximum
        ]
        if not eligible:
            break
        index = max(
            eligible,
            key=lambda item: weights_cpu[item] / (output[item] + 1.0),
        )
        output[index] += 1
        remaining -= 1
    return torch.tensor(
        output, dtype=torch.long, device=weights.device
    )


def _frame_targets(
    graph: _GSTM,
    budget: int,
    tokens_per_frame: int,
    config: Any,
) -> torch.Tensor:
    frame_count = int(graph.frame_score.numel())
    average = float(budget) / max(1, frame_count)
    minimum = max(
        1,
        int(
            math.floor(
                average
                * min(
                    1.0,
                    max(
                        0.0,
                        _cfg_float(
                            config, "certv11_frame_floor_ratio", 0.65
                        ),
                    ),
                )
            )
        ),
    )
    maximum = min(
        tokens_per_frame,
        max(
            minimum,
            int(
                math.ceil(
                    average
                    * max(
                        1.0,
                        _cfg_float(
                            config, "certv11_frame_cap_ratio", 1.65
                        ),
                    )
                )
            ),
        ),
    )
    temperature = max(
        1e-3, _cfg_float(config, "certv11_frame_temperature", 0.60)
    )
    weights = torch.softmax(graph.frame_score / temperature, dim=0)
    return _allocate_integer(
        weights, budget, minimum=minimum, maximum=maximum
    )


def _initial_mandatory(
    analysis: _Analysis,
    graph: _GSTM,
    frame_count: int,
    config: Any,
) -> list[int]:
    mandatory: list[int] = []
    for frame in range(frame_count):
        members = torch.where(analysis.frame_ids == frame)[0]
        score = (
            0.48 * analysis.quality[members]
            + 0.34 * graph.node_score[members]
            + 0.18 * analysis.event_score[members]
        )
        mandatory.append(int(members[torch.argmax(score)].item()))
    query_threshold = _cfg_float(
        config, "certv11_query_threshold", 0.12
    )
    if (
        analysis.query_relevance.numel() > 0
        and analysis.query_confidence >= query_threshold
    ):
        for atom_score in analysis.query_relevance:
            mandatory.append(int(torch.argmax(atom_score).item()))
    return list(dict.fromkeys(mandatory))


def _tree_quotas(
    graph: _GSTM,
    mandatory: list[int],
    budget: int,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    tree_count = int(graph.tree_score.numel())
    mandatory_tensor = torch.tensor(
        mandatory, dtype=torch.long, device=graph.tree_ids.device
    )
    mandatory_counts = torch.bincount(
        graph.tree_ids[mandatory_tensor], minlength=tree_count
    )
    active_ratio = min(
        0.80,
        max(0.05, _cfg_float(config, "certv11_active_tree_ratio", 0.30)),
    )
    active_target = min(
        tree_count,
        budget,
        max(
            int(graph.frame_score.numel()),
            int(math.ceil(budget * active_ratio)),
        ),
    )
    mandatory_trees = torch.where(mandatory_counts > 0)[0]
    ranked = torch.argsort(
        graph.tree_score, descending=True, stable=True
    )
    ranked_cpu = ranked.detach().cpu().tolist()
    tree_size_cpu = graph.tree_size.detach().cpu().tolist()
    active_set = set(
        int(tree) for tree in mandatory_trees.detach().cpu().tolist()
    )
    for tree in ranked_cpu:
        if len(active_set) >= active_target:
            break
        active_set.add(int(tree))
    active_capacity = sum(int(tree_size_cpu[tree]) for tree in active_set)
    if active_capacity < budget:
        # active_tree_ratio is a diversity target, not a hard capacity limit.
        # Small high-scoring trees may not contain enough tokens to satisfy the
        # fixed budget, so deterministically admit more trees as needed.
        for tree in ranked_cpu:
            tree = int(tree)
            if tree in active_set:
                continue
            active_set.add(tree)
            active_capacity += int(tree_size_cpu[tree])
            if active_capacity >= budget:
                break
    if active_capacity < budget:
        raise RuntimeError(
            "GSTM tree capacity is smaller than the requested token budget: "
            f"capacity={active_capacity}, budget={budget}"
        )
    active = torch.tensor(
        sorted(active_set), dtype=torch.long, device=graph.tree_ids.device
    )
    active_cpu = active.detach().cpu().tolist()
    mandatory_counts_cpu = mandatory_counts.detach().cpu().tolist()
    quotas_cpu = [0] * tree_count
    for tree in active_cpu:
        quotas_cpu[tree] = max(1, int(mandatory_counts_cpu[tree]))
    remaining = budget - sum(quotas_cpu)
    average = float(budget) / max(1, int(active.numel()))
    cap_ratio = max(
        1.0, _cfg_float(config, "certv11_tree_cap_ratio", 3.0)
    )
    soft_cap = max(1, int(math.ceil(average * cap_ratio)))
    temperature = max(
        1e-3, _cfg_float(config, "certv11_tree_temperature", 0.55)
    )
    weights_cpu = (
        torch.softmax(graph.tree_score[active] / temperature, dim=0)
        .detach()
        .cpu()
        .tolist()
    )
    while remaining > 0:
        eligible = [
            local
            for local, tree in enumerate(active_cpu)
            if quotas_cpu[tree] < min(int(tree_size_cpu[tree]), soft_cap)
        ]
        if not eligible:
            eligible = [
                local
                for local, tree in enumerate(active_cpu)
                if quotas_cpu[tree] < int(tree_size_cpu[tree])
            ]
        if not eligible:
            break
        local = max(
            eligible,
            key=lambda item: (
                weights_cpu[item]
                / (quotas_cpu[active_cpu[item]] + 1.0)
            ),
        )
        quotas_cpu[active_cpu[local]] += 1
        remaining -= 1
    if remaining != 0:
        raise RuntimeError(
            f"GSTM tree quotas left {remaining} unallocated tokens"
        )
    quotas = torch.tensor(
        quotas_cpu, dtype=torch.long, device=graph.tree_ids.device
    )
    return quotas, active


def _structure_seeds(
    graph: _GSTM,
    quotas: torch.Tensor,
    active: torch.Tensor,
    mandatory: list[int],
    budget: int,
    config: Any,
) -> list[int]:
    seed_set = set(int(token) for token in mandatory)
    tree_ids_cpu = graph.tree_ids.detach().cpu().tolist()
    quotas_cpu = quotas.detach().cpu().tolist()
    counts = [0] * int(quotas.numel())
    for token in seed_set:
        counts[int(tree_ids_cpu[token])] += 1
    structure_ratio = min(
        0.60,
        max(
            0.05,
            _cfg_float(config, "certv11_structure_ratio", 0.35),
        ),
    )
    target = min(budget, max(len(seed_set), int(math.ceil(budget * structure_ratio))))
    ranked_trees = (
        active[
            torch.argsort(
                graph.tree_score[active], descending=True, stable=True
            )
        ]
        .detach()
        .cpu()
        .tolist()
    )
    tree_span_cpu = graph.tree_span.detach().cpu().tolist()
    tree_root_cpu = graph.tree_root.detach().cpu().tolist()
    tree_leaf_cpu = graph.tree_leaf.detach().cpu().tolist()

    # Preserve both states of the strongest cross-frame trajectories before
    # spending structural slots on isolated roots or local extrema.
    for tree in ranked_trees:
        if len(seed_set) >= target:
            break
        tree = int(tree)
        capacity = int(quotas_cpu[tree]) - int(counts[tree])
        if capacity <= 0 or int(tree_span_cpu[tree]) <= 0:
            continue
        pair = [
            int(tree_root_cpu[tree]),
            int(tree_leaf_cpu[tree]),
        ]
        missing = [
            token for token in dict.fromkeys(pair)
            if token >= 0 and token not in seed_set
        ]
        if (
            not missing
            or len(missing) > capacity
            or len(seed_set) + len(missing) > target
        ):
            continue
        for token in missing:
            seed_set.add(token)
            counts[tree] += 1

    choices = tuple(
        choice.detach().cpu().tolist()
        for choice in (
            graph.tree_turn,
            graph.tree_change,
            graph.tree_root,
            graph.tree_leaf,
        )
    )
    for choice in choices:
        for tree in ranked_trees:
            if len(seed_set) >= target:
                break
            tree = int(tree)
            if int(counts[tree]) >= int(quotas_cpu[tree]):
                continue
            token = int(choice[tree])
            if token < 0 or token in seed_set:
                continue
            seed_set.add(token)
            counts[tree] += 1
        if len(seed_set) >= target:
            break
    return sorted(seed_set)


def _candidate_pool(
    analysis: _Analysis,
    graph: _GSTM,
    quotas: torch.Tensor,
    active: torch.Tensor,
    mandatory: list[int],
    budget: int,
    config: Any,
) -> torch.Tensor:
    multiplier = max(
        1.0, _cfg_float(config, "certv11_candidate_multiplier", 3.0)
    )
    limit = min(
        int(analysis.quality.numel()),
        max(budget, int(math.ceil(budget * multiplier))),
    )
    required = set(int(token) for token in mandatory)
    extras: dict[int, float] = {}
    combined = (
        0.42 * analysis.quality
        + 0.36 * graph.node_score
        + 0.12 * analysis.query_score
        + 0.10 * analysis.event_score
    )
    combined_cpu = combined.detach().cpu().tolist()
    tree_ids_cpu = graph.tree_ids.detach().cpu().tolist()
    branch_ids_cpu = graph.branch_ids.detach().cpu().tolist()
    quotas_cpu = quotas.detach().cpu().tolist()
    members_by_tree: list[list[int]] = [
        [] for _ in range(int(quotas.numel()))
    ]
    for token, tree in enumerate(tree_ids_cpu):
        members_by_tree[int(tree)].append(token)
    for tree in active.detach().cpu().tolist():
        tree = int(tree)
        members = members_by_tree[tree]
        quota = int(quotas_cpu[tree])
        if quota <= 0 or not members:
            continue
        ordered_members = sorted(
            members, key=lambda token: (-combined_cpu[token], token)
        )
        core_count = min(len(members), quota)
        required.update(ordered_members[:core_count])
        offer_count = min(
            len(members),
            max(core_count, int(math.ceil(quota * multiplier))),
        )
        for token in ordered_members[:offer_count]:
            extras[token] = max(
                extras.get(token, float("-inf")),
                float(combined_cpu[token]),
            )
        branch_best: dict[int, int] = {}
        for token in ordered_members:
            branch_best.setdefault(int(branch_ids_cpu[token]), token)
        branch_offers = [
            (float(combined_cpu[token]), token)
            for token in branch_best.values()
        ]
        branch_offers.sort(key=lambda item: (-item[0], item[1]))
        for score, token in branch_offers[: max(1, quota)]:
            extras[token] = max(extras.get(token, float("-inf")), score + 0.08)
    candidates = set(required)
    ordered = sorted(extras.items(), key=lambda item: (-item[1], item[0]))
    for token, _ in ordered:
        if len(candidates) >= limit:
            break
        candidates.add(token)
    if len(candidates) < budget:
        global_order = sorted(
            range(len(combined_cpu)),
            key=lambda token: (-combined_cpu[token], token),
        )
        for token in global_order:
            if int(quotas_cpu[int(tree_ids_cpu[token])]) <= 0:
                continue
            candidates.add(int(token))
            if len(candidates) >= budget:
                break
    return torch.tensor(
        sorted(candidates), dtype=torch.long, device=analysis.quality.device
    )


def _structured_d_optimal(
    analysis: _Analysis,
    graph: _GSTM,
    candidates: torch.Tensor,
    mandatory: list[int],
    quotas: torch.Tensor,
    frame_targets: torch.Tensor,
    budget: int,
    config: Any,
) -> tuple[torch.Tensor, float]:
    spatial_design_weight = max(
        0.0,
        _cfg_float(config, "certv11_spatial_design_weight", 0.24),
    )
    motion = graph.motion_vector.float()
    spatial_scale = max(
        1e-6,
        _cfg_float(config, "certv11_max_spatial_jump", 0.50),
    )
    dy = motion[:, 0] / spatial_scale
    dx = motion[:, 1] / spatial_scale
    motion_axes = torch.stack(
        [
            F.relu(dy),
            F.relu(-dy),
            F.relu(dx),
            F.relu(-dx),
            (graph.spatial_jump / spatial_scale).clamp(0.0, 1.0),
            graph.turn,
        ],
        dim=1,
    )
    motion_axes = (
        math.sqrt(spatial_design_weight)
        * F.normalize(motion_axes, p=2, dim=1, eps=1e-6)
    )
    all_rows = torch.cat(
        [analysis.design.float(), motion_axes], dim=1
    )
    rows = all_rows[candidates]
    candidate_count, dimension = rows.shape
    if candidate_count < budget:
        raise RuntimeError(
            f"GSTM candidate pool has {candidate_count} tokens for budget {budget}"
        )
    inverse = torch.eye(
        dimension, dtype=torch.float32, device=rows.device
    ) / analysis.ridge
    leverage = rows.square().sum(dim=1) / analysis.ridge
    active_mask = torch.ones(
        candidate_count, dtype=torch.bool, device=rows.device
    )
    token_to_column = {
        int(token): column
        for column, token in enumerate(candidates.detach().cpu().tolist())
    }
    selected_columns: list[int] = []
    selected_tokens: list[int] = []
    tree_counts = torch.zeros_like(quotas)
    frame_counts = torch.zeros_like(frame_targets)
    branch_selected = torch.zeros(
        max(1, graph.branch_count),
        dtype=torch.bool,
        device=rows.device,
    )
    spatial_count = max(
        1, int(analysis.spatial_ids.max().item()) + 1
    )
    frame_count = int(frame_targets.numel())
    tree_count = int(quotas.numel())
    selected_frame_cells = torch.zeros(
        frame_count * spatial_count,
        dtype=torch.bool,
        device=rows.device,
    )
    selected_tree_cells = torch.zeros(
        tree_count * spatial_count,
        dtype=torch.bool,
        device=rows.device,
    )

    def add(column: int) -> None:
        nonlocal inverse, leverage
        if not bool(active_mask[column]):
            return
        token = int(candidates[column].item())
        row = rows[column]
        direction = inverse @ row
        denominator = (1.0 + torch.dot(row, direction)).clamp_min(1e-6)
        projection = rows @ direction
        leverage = (
            leverage - projection.square() / denominator
        ).clamp_min(0.0)
        inverse = inverse - torch.outer(direction, direction) / denominator
        inverse = 0.5 * (inverse + inverse.transpose(0, 1))
        active_mask[column] = False
        leverage[column] = -1.0
        selected_columns.append(column)
        selected_tokens.append(token)
        tree_counts[graph.tree_ids[token]] += 1
        frame_counts[analysis.frame_ids[token]] += 1
        branch_selected[graph.branch_ids[token]] = True
        spatial_id = analysis.spatial_ids[token]
        selected_frame_cells[
            analysis.frame_ids[token] * spatial_count + spatial_id
        ] = True
        selected_tree_cells[
            graph.tree_ids[token] * spatial_count + spatial_id
        ] = True

    for token in mandatory:
        column = token_to_column.get(int(token))
        if column is not None and len(selected_columns) < budget:
            add(column)

    structure_weight = max(
        0.0, _cfg_float(config, "certv11_structure_weight", 0.42)
    )
    tree_fill_weight = max(
        0.0, _cfg_float(config, "certv11_tree_fill_weight", 0.28)
    )
    frame_fill_weight = max(
        0.0, _cfg_float(config, "certv11_frame_fill_weight", 0.24)
    )
    branch_bonus = max(
        0.0, _cfg_float(config, "certv11_branch_bonus", 0.18)
    )
    spatial_coverage_weight = max(
        0.0,
        _cfg_float(config, "certv11_spatial_coverage_weight", 0.22),
    )
    candidate_trees = graph.tree_ids[candidates]
    candidate_frames = analysis.frame_ids[candidates]
    candidate_branches = graph.branch_ids[candidates]
    candidate_spatial = analysis.spatial_ids[candidates]
    candidate_frame_cells = (
        candidate_frames * spatial_count + candidate_spatial
    )
    candidate_tree_cells = (
        candidate_trees * spatial_count + candidate_spatial
    )
    while len(selected_columns) < budget:
        under_tree = tree_counts[candidate_trees] < quotas[candidate_trees]
        eligible = active_mask & under_tree
        if not bool(eligible.any()):
            raise RuntimeError(
                "GSTM constrained selector exhausted tree quotas early"
            )
        tree_need = (
            quotas[candidate_trees] - tree_counts[candidate_trees]
        ).float() / quotas[candidate_trees].clamp_min(1).float()
        frame_need = (
            frame_targets[candidate_frames] - frame_counts[candidate_frames]
        ).clamp_min(0).float() / frame_targets[candidate_frames].clamp_min(1).float()
        unseen_branch = (~branch_selected[candidate_branches]).float()
        unseen_frame_cell = (
            ~selected_frame_cells[candidate_frame_cells]
        ).float()
        unseen_tree_cell = (
            ~selected_tree_cells[candidate_tree_cells]
        ).float()
        spatial_coverage = (
            0.55 * unseen_frame_cell + 0.45 * unseen_tree_cell
        )
        score = (
            torch.log1p(leverage.clamp_min(0.0))
            + structure_weight * graph.node_score[candidates]
            + tree_fill_weight * tree_need
            + frame_fill_weight * frame_need
            + branch_bonus * unseen_branch
            + spatial_coverage_weight * spatial_coverage
        )
        score = score.masked_fill(~eligible, float("-inf"))
        column = int(torch.argmax(score).item())
        if not math.isfinite(float(score[column].item())):
            raise RuntimeError("GSTM D-optimal score became non-finite")
        add(column)

    selected = torch.tensor(
        selected_tokens, dtype=torch.long, device=candidates.device
    )
    information = (
        analysis.ridge
        * torch.eye(dimension, dtype=torch.float32, device=rows.device)
        + all_rows[selected].T @ all_rows[selected]
    )
    sign, logdet = torch.linalg.slogdet(information)
    if float(sign.item()) <= 0.0:
        raise RuntimeError("GSTM information matrix is not positive definite")
    return torch.sort(selected).values, float(logdet.item())


def _build_plan(
    analysis: _Analysis,
    graph: _GSTM,
    selected: torch.Tensor,
    mandatory: list[int],
    config: Any,
) -> tuple[CertVidPlan, dict[str, Any]]:
    total_tokens = int(analysis.metric_flat.shape[0])
    budget = int(selected.numel())
    raw_similarity = (
        analysis.metric_flat.float()
        @ analysis.metric_flat[selected].float().T
    )
    source_frame = analysis.frame_ids.unsqueeze(1)
    anchor_frame = analysis.frame_ids[selected].unsqueeze(0)
    frame_distance = (source_frame - anchor_frame).abs()
    same_tree = (
        graph.tree_ids.unsqueeze(1)
        == graph.tree_ids[selected].unsqueeze(0)
    )
    same_segment = (
        graph.segment_ids.unsqueeze(1)
        == graph.segment_ids[selected].unsqueeze(0)
    )
    source_local = torch.remainder(
        torch.arange(total_tokens, device=selected.device),
        analysis.coords.shape[0],
    )
    anchor_local = torch.remainder(
        selected, analysis.coords.shape[0]
    )
    spatial_distance = torch.linalg.vector_norm(
        analysis.coords[source_local].unsqueeze(1)
        - analysis.coords[anchor_local].unsqueeze(0),
        dim=-1,
    )
    radius = max(
        1, _cfg_int(config, "certv11_tree_assignment_radius", 3)
    )
    assignment_spatial_radius = max(
        0.0,
        _cfg_float(
            config, "certv11_assignment_spatial_radius", 0.65
        ),
    )
    per_edge_spatial_radius = max(
        0.0,
        _cfg_float(config, "certv11_max_spatial_jump", 0.50),
    )
    same_tree_spatial = spatial_distance <= (
        assignment_spatial_radius
        + frame_distance.float() * per_edge_spatial_radius
    )
    cross_similarity = _cfg_float(
        config, "certv11_cross_tree_similarity", 0.90
    )
    valid = (
        (
            same_tree
            & (frame_distance <= radius)
            & same_tree_spatial
        )
        | (
            (frame_distance == 0)
            & (spatial_distance <= assignment_spatial_radius)
        )
        | (
            same_segment
            & (frame_distance <= 1)
            & (raw_similarity >= cross_similarity)
            & (
                spatial_distance
                <= assignment_spatial_radius
                + per_edge_spatial_radius
            )
        )
    )
    score = raw_similarity + 0.08 * same_tree.float()
    score = score.masked_fill(~valid, -2.0)
    topk = min(
        budget,
        max(1, _cfg_int(config, "certv11_assignment_topk", 2)),
    )
    _, assignment = torch.topk(score, k=topk, dim=1, largest=True)
    values = torch.gather(raw_similarity, 1, assignment)
    temperature = max(
        1e-4,
        _cfg_float(config, "certv11_assignment_temperature", 0.07),
    )
    weights = torch.softmax(values / temperature, dim=1)
    merge_threshold = _cfg_float(
        config, "certv11_merge_threshold", 0.78
    )
    rejected = values[:, 0] < merge_threshold
    weights[rejected] = 0.0
    anchor_positions = torch.arange(
        budget, dtype=torch.long, device=selected.device
    )
    assignment[selected, 0] = anchor_positions
    weights[selected] = 0.0
    weights[selected, 0] = 1.0

    source_mass = (
        0.5 + 0.5 * analysis.demand_weight * float(total_tokens)
    ).clamp(0.25, 2.0)
    base_alpha = min(
        1.0, max(0.0, _cfg_float(config, "certv11_fusion_alpha", 0.10))
    )
    alpha = torch.full(
        (budget,), base_alpha, dtype=torch.float32, device=selected.device
    )
    confidence_sum = torch.zeros_like(alpha)
    confidence_mass = torch.zeros_like(alpha)
    best_confidence = values[:, 0].clamp(0.0, 1.0)
    target = assignment[:, 0]
    live_mass = source_mass * (~rejected).float()
    confidence_sum.index_add_(
        0, target, best_confidence * live_mass
    )
    confidence_mass.index_add_(0, target, live_mass)
    confidence = confidence_sum / confidence_mass.clamp_min(1e-6)
    alpha *= (
        (confidence - merge_threshold)
        / max(1e-6, 1.0 - merge_threshold)
    ).clamp(0.0, 1.0)
    protected = torch.tensor(
        mandatory, dtype=torch.long, device=selected.device
    )
    alpha[torch.isin(selected, protected)] = 0.0
    structural_protect = _cfg_float(
        config, "certv11_structure_protect_threshold", 0.65
    )
    alpha[graph.node_score[selected] >= structural_protect] = 0.0
    plan = CertVidPlan(
        anchor_indices=selected,
        assignment_indices=assignment,
        assignment_weights=weights,
        source_mass=source_mass,
        fusion_alpha=alpha,
        raw_token_count=total_tokens,
    )
    residual = ~torch.isin(
        torch.arange(total_tokens, device=selected.device), selected
    )
    assigned_cross_frame = frame_distance.gather(
        1, assignment[:, :1]
    ).squeeze(1) > 0
    assigned_spatial_distance = spatial_distance.gather(
        1, assignment[:, :1]
    ).squeeze(1)
    return plan, {
        "rejected_residual_count": int(
            (rejected & residual).sum().item()
        ),
        "rejected_residual_ratio": float(
            (rejected & residual).float().sum().item()
            / max(1, int(residual.sum().item()))
        ),
        "assignment_similarity_p05": float(
            torch.quantile(values[:, 0].float(), 0.05).item()
        ),
        "assignment_similarity_median": float(
            values[:, 0].median().item()
        ),
        "cross_frame_assignment_rate": float(
            assigned_cross_frame.float().mean().item()
        ),
        "assignment_spatial_distance_median": float(
            assigned_spatial_distance.median().item()
        ),
        "assignment_spatial_distance_p95": float(
            torch.quantile(
                assigned_spatial_distance.float(), 0.95
            ).item()
        ),
    }


def _temporal_entropy(
    selected: torch.Tensor,
    frame_ids: torch.Tensor,
    frame_count: int,
) -> float:
    counts = torch.bincount(
        frame_ids[selected], minlength=frame_count
    ).float()
    probability = counts / counts.sum().clamp_min(1.0)
    entropy = -torch.sum(
        probability * torch.log(probability.clamp_min(1e-8))
    )
    return float(
        (entropy / max(1e-8, math.log(max(2, frame_count)))).item()
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _store_diagnostics(config: Any, diagnostics: dict[str, Any]) -> None:
    config.last_certv11_diagnostics = diagnostics
    config.last_certv11_tree_count = int(
        diagnostics.get("tree_count", 0)
    )
    config.last_certv11_active_tree_count = int(
        diagnostics.get("active_tree_count", 0)
    )
    config.last_certv11_temporal_entropy = float(
        diagnostics.get("temporal_entropy", 0.0)
    )
    template = os.environ.get(
        "CERTV11_DIAGNOSTICS_JSONL", ""
    ).strip()
    if template:
        rank = os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))
        path = template.replace("{rank}", rank).replace(
            "{pid}", str(os.getpid())
        )
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        record = dict(diagnostics)
        record["sample_id"] = str(
            getattr(config, "_debug_sample_id", "unknown")
        )
        record["question"] = str(
            getattr(config, "_certvid_query_text", "") or ""
        )
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(_json_safe(record), sort_keys=True) + "\n"
            )
    if _cfg_bool(config, "certv11_debug", False):
        print(
            "[certvid-v11] "
            f"sample={getattr(config, '_debug_sample_id', 'unknown')} "
            f"trees={diagnostics.get('active_tree_count', 0)}/"
            f"{diagnostics.get('tree_count', 0)} "
            f"branches={diagnostics.get('selected_branch_count', 0)}/"
            f"{diagnostics.get('branch_count', 0)} "
            f"entropy={diagnostics.get('temporal_entropy', 0.0):.3f}"
        )


def certvid_v11_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    config: Any,
    question_features: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select a structure-preserving D-optimal coreset under a fixed budget."""
    if video_features.ndim != 3:
        raise ValueError(
            f"expected video_features [T, HW, D], got {tuple(video_features.shape)}"
        )
    frame_count, tokens_per_frame, feature_dim = video_features.shape
    total_tokens = frame_count * tokens_per_frame
    budget = max(
        1,
        min(
            total_tokens,
            int(round(total_tokens * _effective_ratio(config))),
        ),
    )
    flat_features = video_features.reshape(total_tokens, feature_dim)
    if budget >= total_tokens:
        indices = torch.arange(
            total_tokens, dtype=torch.long, device=video_features.device
        )
        plan = _identity_plan(total_tokens, video_features.device)
        output = flat_features
        diagnostics = {
            "identity": True,
            "budget": budget,
            "raw_token_count": total_tokens,
        }
    else:
        analysis = _build_analysis(
            video_features, cls_attention, question_features, config
        )
        graph = _build_gstm(analysis, config)
        mandatory = _initial_mandatory(
            analysis, graph, frame_count, config
        )
        quotas, active = _tree_quotas(
            graph, mandatory, budget, config
        )
        mandatory = _structure_seeds(
            graph, quotas, active, mandatory, budget, config
        )
        candidates = _candidate_pool(
            analysis,
            graph,
            quotas,
            active,
            mandatory,
            budget,
            config,
        )
        frame_targets = _frame_targets(
            graph, budget, tokens_per_frame, config
        )
        indices, logdet = _structured_d_optimal(
            analysis,
            graph,
            candidates,
            mandatory,
            quotas,
            frame_targets,
            budget,
            config,
        )
        plan, plan_stats = _build_plan(
            analysis, graph, indices, mandatory, config
        )
        output = apply_certvid_plan(flat_features, plan)
        selected_trees = torch.unique(graph.tree_ids[indices])
        selected_branches = torch.unique(graph.branch_ids[indices])
        selected_mask = torch.zeros(
            total_tokens, dtype=torch.bool, device=indices.device
        )
        selected_mask[indices] = True
        state_pairs = (
            (graph.tree_root >= 0)
            & (graph.tree_leaf >= 0)
            & (graph.tree_root != graph.tree_leaf)
        )
        paired_trees = (
            state_pairs
            & selected_mask[graph.tree_root.clamp_min(0)]
            & selected_mask[graph.tree_leaf.clamp_min(0)]
        )
        spatial_count = max(
            1, int(analysis.spatial_ids.max().item()) + 1
        )
        all_frame_cells = (
            analysis.frame_ids * spatial_count + analysis.spatial_ids
        )
        selected_frame_cells = (
            analysis.frame_ids[indices] * spatial_count
            + analysis.spatial_ids[indices]
        )
        all_tree_cells = (
            graph.tree_ids * spatial_count + analysis.spatial_ids
        )
        selected_tree_cells = (
            graph.tree_ids[indices] * spatial_count
            + analysis.spatial_ids[indices]
        )
        linked = graph.parent >= 0
        linked_jump = graph.spatial_jump[linked]
        diagnostics = {
            "identity": False,
            "budget": budget,
            "raw_token_count": total_tokens,
            "candidate_count": int(candidates.numel()),
            "mandatory_count": len(mandatory),
            "tree_count": int(graph.tree_score.numel()),
            "active_tree_count": int(active.numel()),
            "selected_tree_count": int(selected_trees.numel()),
            "branch_count": int(graph.branch_count),
            "selected_branch_count": int(selected_branches.numel()),
            "available_state_pair_count": int(state_pairs.sum().item()),
            "selected_state_pair_count": int(paired_trees.sum().item()),
            "selected_endpoint_count": int(
                graph.endpoint[indices].bool().sum().item()
            ),
            "selected_turn_count": int(
                (graph.turn[indices] > 0.25).sum().item()
            ),
            "selected_change_count": int(
                (graph.state_change[indices] > 0.25).sum().item()
            ),
            "selected_motion_node_count": int(
                (graph.spatial_jump[indices] > 1e-6).sum().item()
            ),
            "frame_spatial_cell_coverage": float(
                torch.unique(selected_frame_cells).numel()
                / max(1, torch.unique(all_frame_cells).numel())
            ),
            "tree_spatial_cell_coverage": float(
                torch.unique(selected_tree_cells).numel()
                / max(1, torch.unique(all_tree_cells).numel())
            ),
            "linked_spatial_jump_median": (
                float(linked_jump.median().item())
                if linked_jump.numel() > 0
                else 0.0
            ),
            "linked_spatial_jump_p95": (
                float(torch.quantile(linked_jump.float(), 0.95).item())
                if linked_jump.numel() > 0
                else 0.0
            ),
            "edge_reliability": graph.reliability,
            "tree_quota_nonzero": int((quotas > 0).sum().item()),
            "tree_quota_max": int(quotas.max().item()),
            "frame_targets": frame_targets,
            "frame_counts": torch.bincount(
                analysis.frame_ids[indices], minlength=frame_count
            ),
            "temporal_entropy": _temporal_entropy(
                indices, analysis.frame_ids, frame_count
            ),
            "logdet": logdet,
            **plan_stats,
        }
    if output.shape[0] != budget:
        raise RuntimeError(
            f"CertVID V11 produced {output.shape[0]} tokens for budget {budget}"
        )
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("CertVID V11 produced NaN or Inf")
    config._certvid_plan = plan
    config.vision_token_length = int(output.shape[0])
    config.visual_token_length = int(output.shape[0])
    config.llm_token_length = None
    config.last_adapter_variant = "certvid_v11"
    config.last_adapter_raw_tokens = float(total_tokens)
    config.last_adapter_output_tokens = float(output.shape[0])
    _store_diagnostics(config, diagnostics)
    return output, indices
