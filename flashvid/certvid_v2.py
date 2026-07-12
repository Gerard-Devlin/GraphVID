from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from .certvid import (
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
    _temporal_signals,
)
from .configuration_flashvid import FlashVidConfig


def _effective_ratio(config: FlashVidConfig) -> float:
    ratio = _cfg_float(config, "retention_ratio", 0.10)
    if bool(getattr(config, "certv2_budget_uses_expansion", True)):
        ratio *= _cfg_float(config, "expansion", 1.0)
    return min(1.0, max(0.0, ratio))


def _density_peaks(metric_frames: torch.Tensor, neighbors: int) -> torch.Tensor:
    """FastVID-style density peaks computed independently inside each frame."""
    frame_count, tokens_per_frame, _ = metric_frames.shape
    if tokens_per_frame <= 1:
        return torch.ones((frame_count, tokens_per_frame), device=metric_frames.device)
    k = min(max(1, int(neighbors)), tokens_per_frame - 1)
    output = torch.zeros((frame_count, tokens_per_frame), dtype=torch.float32, device=metric_frames.device)
    for frame_idx in range(frame_count):
        similarity = metric_frames[frame_idx] @ metric_frames[frame_idx].transpose(0, 1)
        distance = (1.0 - similarity).clamp_min(0.0)
        distance.fill_diagonal_(float("inf"))
        local_distance = torch.topk(distance, k=k, dim=-1, largest=False).values
        density = torch.exp(-torch.mean(local_distance.square(), dim=-1))
        order = torch.argsort(density, descending=True, stable=True)
        delta = torch.zeros_like(density)
        finite_distance = distance.masked_fill(torch.isinf(distance), 0.0)
        delta[order[0]] = finite_distance[order[0]].max()
        for rank in range(1, tokens_per_frame):
            token_idx = order[rank]
            delta[token_idx] = distance[token_idx, order[:rank]].min()
        output[frame_idx] = density * delta
    return _minmax(output, dim=-1)


def _trajectory_signals(
    metric_frames: torch.Tensor,
    coords: torch.Tensor,
    spatial_penalty: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """Return frame events, bidirectional novelty, and second-order track curvature."""
    frame_event, forward_novelty, matches = _temporal_signals(
        metric_frames,
        coords,
        spatial_penalty,
    )
    frame_count, tokens_per_frame, _ = metric_frames.shape
    backward_novelty = torch.zeros_like(forward_novelty)
    curvature = torch.zeros_like(forward_novelty)
    next_matches: list[torch.Tensor] = []
    spatial_distance = torch.cdist(coords.float(), coords.float(), p=2)

    for frame_idx in range(frame_count - 1):
        current = metric_frames[frame_idx]
        following = metric_frames[frame_idx + 1]
        raw_similarity = current @ following.transpose(0, 1)
        match_score = raw_similarity - float(max(0.0, spatial_penalty)) * spatial_distance
        best_following = match_score.argmax(dim=1)
        token_ids = torch.arange(tokens_per_frame, device=metric_frames.device)
        matched_similarity = raw_similarity[token_ids, best_following]
        backward_novelty[frame_idx] = (1.0 - matched_similarity).clamp(0.0, 2.0) * 0.5
        next_matches.append(best_following)

    if frame_count > 1:
        backward_novelty[-1] = forward_novelty[-1]
    for frame_idx in range(1, frame_count - 1):
        previous_indices = matches[frame_idx - 1][0]
        following_indices = next_matches[frame_idx]
        current = metric_frames[frame_idx]
        incoming = current - metric_frames[frame_idx - 1][previous_indices]
        outgoing = metric_frames[frame_idx + 1][following_indices] - current
        incoming_norm = incoming.norm(dim=-1)
        outgoing_norm = outgoing.norm(dim=-1)
        direction_change = 1.0 - F.cosine_similarity(incoming, outgoing, dim=-1, eps=1e-6)
        motion_gate = torch.sqrt(incoming_norm * outgoing_norm).clamp(0.0, 1.0)
        curvature[frame_idx] = direction_change.clamp(0.0, 2.0) * 0.5 * motion_gate

    novelty = _minmax(0.5 * forward_novelty + 0.5 * backward_novelty, dim=-1)
    curvature = _minmax(curvature, dim=-1)
    return frame_event, novelty, curvature, matches


def _component_support(
    metric_flat: torch.Tensor,
    component_ids: torch.Tensor,
    component_sizes: torch.Tensor,
    frame_ids: torch.Tensor,
    frame_count: int,
) -> torch.Tensor:
    """Score persistent trajectories without replacing their representative token."""
    component_count = int(component_sizes.numel())
    feature_dim = int(metric_flat.shape[-1])
    sums = torch.zeros((component_count, feature_dim), dtype=torch.float32, device=metric_flat.device)
    sums.index_add_(0, component_ids, metric_flat.float())
    means = F.normalize(
        sums / component_sizes.float().clamp_min(1.0).unsqueeze(1),
        p=2,
        dim=-1,
        eps=1e-6,
    )
    variation_token = (1.0 - torch.sum(metric_flat * means[component_ids], dim=-1)).clamp(0.0, 2.0) * 0.5
    variation_sum = torch.zeros(component_count, dtype=torch.float32, device=metric_flat.device)
    variation_sum.index_add_(0, component_ids, variation_token)
    variation = variation_sum / component_sizes.float().clamp_min(1.0)

    first = torch.full((component_count,), frame_count, dtype=torch.long, device=metric_flat.device)
    last = torch.full((component_count,), -1, dtype=torch.long, device=metric_flat.device)
    for frame_idx in range(frame_count):
        present = torch.unique(component_ids[frame_ids == frame_idx])
        first[present] = torch.minimum(first[present], torch.full_like(present, frame_idx))
        last[present] = torch.maximum(last[present], torch.full_like(present, frame_idx))
    span = (last - first + 1).clamp_min(1).float() / max(1, frame_count)
    mass = torch.log1p(component_sizes.float())
    mass = mass / mass.max().clamp_min(1e-6)
    component_score = _minmax(0.40 * span + 0.30 * mass + 0.30 * variation, dim=0)
    return component_score[component_ids]


def _add_best(
    selected: list[int],
    selected_set: set[int],
    members: torch.Tensor,
    score: torch.Tensor,
    count: int,
    budget: int,
) -> None:
    if count <= 0 or members.numel() == 0 or len(selected) >= budget:
        return
    ordered = members[torch.argsort(score[members], descending=True, stable=True)]
    for token_idx in ordered[:count].detach().cpu().tolist():
        token_idx = int(token_idx)
        if token_idx in selected_set:
            continue
        selected.append(token_idx)
        selected_set.add(token_idx)
        if len(selected) >= budget:
            return


def _mandatory_anchors(
    *,
    budget: int,
    quality: torch.Tensor,
    appearance: torch.Tensor,
    change: torch.Tensor,
    frame_ids: torch.Tensor,
    temporal_ids: torch.Tensor,
    query_relevance: torch.Tensor,
    query_confidence: float,
    floor_ratio: float,
) -> list[int]:
    frame_count = int(frame_ids.max().item()) + 1
    selected: list[int] = []
    selected_set: set[int] = set()
    floor_budget = min(budget, int(round(budget * min(max(floor_ratio, 0.0), 0.60))))
    base_floor = floor_budget // max(1, frame_count)
    remainder = floor_budget - base_floor * frame_count
    frame_importance = torch.zeros(frame_count, dtype=torch.float32, device=quality.device)
    for frame_idx in range(frame_count):
        members = torch.where(frame_ids == frame_idx)[0]
        frame_importance[frame_idx] = quality[members].topk(k=min(4, members.numel())).values.mean()
    extra_frames = set(
        torch.argsort(frame_importance, descending=True, stable=True)[:remainder].detach().cpu().tolist()
    )
    for frame_idx in range(frame_count):
        members = torch.where(frame_ids == frame_idx)[0]
        _add_best(
            selected,
            selected_set,
            members,
            quality,
            base_floor + int(frame_idx in extra_frames),
            budget,
        )

    # Each temporal interval receives one appearance pole and one change pole.
    for temporal_id in torch.unique(temporal_ids).detach().cpu().tolist():
        members = torch.where(temporal_ids == int(temporal_id))[0]
        _add_best(selected, selected_set, members, appearance, 1, budget)
        _add_best(selected, selected_set, members, change, 1, budget)

    if query_relevance.numel() > 0 and query_confidence >= 0.12:
        for atom_score in query_relevance:
            per_bin: list[tuple[float, int]] = []
            for temporal_id in torch.unique(temporal_ids).detach().cpu().tolist():
                members = torch.where(temporal_ids == int(temporal_id))[0]
                best = int(members[torch.argmax(atom_score[members])].item())
                per_bin.append((float(atom_score[best].item()), best))
            per_bin.sort(reverse=True)
            for _, token_idx in per_bin[:2]:
                if token_idx not in selected_set and len(selected) < budget:
                    selected.append(token_idx)
                    selected_set.add(token_idx)
    return selected


def _candidate_pool(
    *,
    budget: int,
    quality: torch.Tensor,
    density: torch.Tensor,
    component_ids: torch.Tensor,
    cell_ids: torch.Tensor,
    mandatory: list[int],
    multiplier: float,
) -> torch.Tensor:
    total_tokens = int(quality.numel())
    limit = min(total_tokens, max(budget, int(math.ceil(budget * max(1.0, multiplier)))))
    candidate_set = set(int(idx) for idx in mandatory)

    component_representatives: list[int] = []
    for component_id in torch.unique(component_ids).detach().cpu().tolist():
        members = torch.where(component_ids == int(component_id))[0]
        component_representatives.append(int(members[torch.argmax(quality[members])].item()))
    component_representatives.sort(key=lambda idx: float(quality[idx].item()), reverse=True)
    candidate_set.update(component_representatives[:limit])

    for cell_id in torch.unique(cell_ids).detach().cpu().tolist():
        members = torch.where(cell_ids == int(cell_id))[0]
        candidate_set.add(int(members[torch.argmax(quality[members])].item()))
        candidate_set.add(int(members[torch.argmax(density[members])].item()))

    for token_idx in torch.argsort(quality, descending=True, stable=True).detach().cpu().tolist():
        candidate_set.add(int(token_idx))
        if len(candidate_set) >= limit:
            break
    if len(candidate_set) > limit:
        mandatory_set = set(mandatory)
        others = sorted(
            (idx for idx in candidate_set if idx not in mandatory_set),
            key=lambda idx: float(quality[idx].item()),
            reverse=True,
        )
        candidate_set = mandatory_set | set(others[: max(0, limit - len(mandatory_set))])
    return torch.tensor(sorted(candidate_set), dtype=torch.long, device=quality.device)


def _virtual_support(
    *,
    metric_flat: torch.Tensor,
    candidates: torch.Tensor,
    demand: torch.Tensor,
    frame_ids: torch.Tensor,
    temporal_ids: torch.Tensor,
) -> torch.Tensor:
    """Assign demand virtually; output embeddings remain untouched original anchors."""
    metric_dtype = torch.float16 if metric_flat.device.type == "cuda" else torch.float32
    similarity = metric_flat.to(metric_dtype) @ metric_flat[candidates].to(metric_dtype).transpose(0, 1)
    temporal_valid = (temporal_ids.unsqueeze(1) - temporal_ids[candidates].unsqueeze(0)).abs() <= 1
    similarity = similarity.masked_fill(~temporal_valid, -2.0)
    best_similarity, assignment = similarity.max(dim=1)
    candidate_count = int(candidates.numel())
    mass = torch.zeros(candidate_count, dtype=torch.float32, device=metric_flat.device)
    residual = torch.zeros_like(mass)
    mass.index_add_(0, assignment, demand)
    residual.index_add_(0, assignment, demand * (1.0 - best_similarity.float()).clamp(0.0, 2.0))
    frame_hits = torch.zeros_like(mass)
    for frame_idx in torch.unique(frame_ids).detach().cpu().tolist():
        assigned = torch.unique(assignment[frame_ids == int(frame_idx)])
        frame_hits[assigned] += 1.0
    frame_hits /= max(1, int(frame_ids.max().item()) + 1)
    return _minmax(0.55 * _minmax(mass, dim=0) + 0.25 * _minmax(residual, dim=0) + 0.20 * frame_hits, dim=0)


def _global_arbitration(
    *,
    budget: int,
    candidates: torch.Tensor,
    mandatory: list[int],
    metric_flat: torch.Tensor,
    base_score: torch.Tensor,
    temporal_ids: torch.Tensor,
    cell_ids: torch.Tensor,
    component_ids: torch.Tensor,
    frame_ids: torch.Tensor,
    diversity_weight: float,
    coverage_weight: float,
) -> list[int]:
    candidate_count = int(candidates.numel())
    candidate_lookup = {int(token): idx for idx, token in enumerate(candidates.detach().cpu().tolist())}
    selected_columns: list[int] = []
    selected_mask = torch.zeros(candidate_count, dtype=torch.bool, device=candidates.device)
    max_similarity = torch.full((candidate_count,), -1.0, dtype=torch.float32, device=candidates.device)
    temporal_covered = torch.zeros(
        int(temporal_ids.max().item()) + 1,
        dtype=torch.bool,
        device=candidates.device,
    )
    cell_covered = torch.zeros(
        int(cell_ids.max().item()) + 1,
        dtype=torch.bool,
        device=candidates.device,
    )
    component_covered = torch.zeros(
        int(component_ids.max().item()) + 1,
        dtype=torch.bool,
        device=candidates.device,
    )
    frame_counts = torch.zeros(int(frame_ids.max().item()) + 1, dtype=torch.float32, device=candidates.device)
    candidate_features = metric_flat[candidates]

    def select_column(column: int) -> None:
        if bool(selected_mask[column]):
            return
        selected_columns.append(column)
        selected_mask[column] = True
        similarity = candidate_features @ candidate_features[column]
        max_similarity.copy_(torch.maximum(max_similarity, similarity.float()))
        token_idx = int(candidates[column].item())
        temporal_covered[temporal_ids[token_idx]] = True
        cell_covered[cell_ids[token_idx]] = True
        component_covered[component_ids[token_idx]] = True
        frame_counts[int(frame_ids[token_idx].item())] += 1.0

    for token_idx in mandatory:
        column = candidate_lookup.get(int(token_idx))
        if column is not None and len(selected_columns) < budget:
            select_column(column)

    candidate_temporal = temporal_ids[candidates]
    candidate_cells = cell_ids[candidates]
    candidate_components = component_ids[candidates]
    candidate_frames = frame_ids[candidates]
    avg_frame_budget = max(1.0, budget / max(1, frame_counts.numel()))
    while len(selected_columns) < budget:
        new_temporal = (~temporal_covered[candidate_temporal]).float()
        new_cell = (~cell_covered[candidate_cells]).float()
        new_component = (~component_covered[candidate_components]).float()
        coverage = 0.45 * new_temporal + 0.30 * new_cell + 0.25 * new_component
        crowding = frame_counts[candidate_frames] / avg_frame_budget
        score = (
            base_score
            + float(diversity_weight) * (1.0 - max_similarity).clamp(0.0, 2.0) * 0.5
            + float(coverage_weight) * coverage
            - 0.035 * crowding
        )
        score = score.masked_fill(selected_mask, -float("inf"))
        column = int(torch.argmax(score).item())
        if not torch.isfinite(score[column]):
            break
        select_column(column)

    return [int(candidates[column].item()) for column in selected_columns[:budget]]


def certvid_v2_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
    question_features: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-pruning trajectory coreset with deterministic global arbitration."""
    if video_features.ndim != 3:
        raise ValueError(f"expected video_features [T, HW, D], got {tuple(video_features.shape)}")
    frame_count, tokens_per_frame, _ = video_features.shape
    total_tokens = int(frame_count * tokens_per_frame)
    ratio = _effective_ratio(flashvid_config)
    budget = max(1, min(total_tokens, int(round(total_tokens * ratio))))
    flat_features = video_features.reshape(total_tokens, -1)
    if budget >= total_tokens:
        indices = torch.arange(total_tokens, dtype=torch.long, device=video_features.device)
        return flat_features, indices

    metric_dim = max(32, _cfg_int(flashvid_config, "certv2_metric_dim", 256))
    metric_flat = _metric_features(video_features, metric_dim)
    metric_frames = metric_flat.view(frame_count, tokens_per_frame, -1)
    height, width = _grid_hw(tokens_per_frame, flashvid_config)
    spatial_bins = max(1, _cfg_int(flashvid_config, "certv2_spatial_bins", 4))
    coords, spatial_cells = _spatial_layout(
        tokens_per_frame,
        height,
        width,
        spatial_bins,
        video_features.device,
    )
    frame_event, novelty_2d, curvature_2d, matches = _trajectory_signals(
        metric_frames,
        coords,
        _cfg_float(flashvid_config, "certv2_spatial_penalty", 0.06),
    )
    density_2d = _density_peaks(
        metric_frames,
        _cfg_int(flashvid_config, "certv2_density_neighbors", 4),
    )
    component_ids_cpu, component_sizes_cpu = _build_components(
        frame_count,
        tokens_per_frame,
        frame_event,
        matches,
        _cfg_float(flashvid_config, "certv2_track_threshold", 0.80),
    )
    component_ids = component_ids_cpu.to(video_features.device)
    component_sizes = component_sizes_cpu.to(video_features.device)
    frame_ids = torch.arange(frame_count, device=video_features.device).repeat_interleave(tokens_per_frame)
    component_support = _component_support(
        metric_flat,
        component_ids,
        component_sizes,
        frame_ids,
        frame_count,
    )

    temporal_bins = min(frame_count, max(1, _cfg_int(flashvid_config, "certv2_temporal_bins", 12)))
    temporal_ids = torch.div(
        frame_ids * temporal_bins,
        max(1, frame_count),
        rounding_mode="floor",
    ).clamp_max(temporal_bins - 1)
    cell_ids = temporal_ids * (spatial_bins * spatial_bins) + spatial_cells.repeat(frame_count)
    attention = _rank_normalize(cls_attention.float()).reshape(-1)
    novelty = novelty_2d.reshape(-1)
    curvature = curvature_2d.reshape(-1)
    density = density_2d.reshape(-1)
    detail = _local_detail(video_features, height, width).reshape(-1)
    event = frame_event.repeat_interleave(tokens_per_frame)

    atoms = _question_atoms(
        question_features,
        max(0, _cfg_int(flashvid_config, "certv2_query_atoms", 6)),
        metric_dim,
    ).to(video_features.device)
    query_relevance, atom_weights, query_confidence = _question_relevance(atoms, metric_flat)
    query_score = (
        (query_relevance * atom_weights.unsqueeze(1)).sum(dim=0)
        if query_relevance.numel() > 0
        else torch.zeros(total_tokens, dtype=torch.float32, device=video_features.device)
    )

    # Higher budgets can afford more detail/context; low budgets keep CertVID's
    # strong attention bias. The interpolation depends only on the budget.
    budget_phase = min(1.0, max(0.0, (ratio - 0.125) / 0.20))
    base_quality = (
        (0.30 - 0.05 * budget_phase) * attention
        + 0.19 * novelty
        + 0.13 * curvature
        + (0.14 + 0.04 * budget_phase) * detail
        + 0.10 * density
        + 0.08 * event
        + (0.06 + 0.01 * budget_phase) * component_support
    )
    query_weight = min(
        0.35,
        max(0.0, _cfg_float(flashvid_config, "certv2_query_weight", 0.16) * query_confidence),
    )
    quality = _minmax((1.0 - query_weight) * base_quality + query_weight * query_score, dim=0)
    appearance = _minmax(0.45 * attention + 0.25 * density + 0.20 * detail + 0.10 * query_score, dim=0)
    change = _minmax(0.38 * novelty + 0.32 * curvature + 0.20 * event + 0.10 * query_score, dim=0)
    demand = 0.35 + 0.23 * attention + 0.20 * novelty + 0.14 * detail + 0.08 * component_support
    demand = demand / demand.sum().clamp_min(1e-6)

    mandatory = _mandatory_anchors(
        budget=budget,
        quality=quality,
        appearance=appearance,
        change=change,
        frame_ids=frame_ids,
        temporal_ids=temporal_ids,
        query_relevance=query_relevance,
        query_confidence=query_confidence,
        floor_ratio=_cfg_float(flashvid_config, "certv2_frame_floor_ratio", 0.28),
    )
    candidates = _candidate_pool(
        budget=budget,
        quality=quality,
        density=density,
        component_ids=component_ids,
        cell_ids=cell_ids,
        mandatory=mandatory,
        multiplier=_cfg_float(flashvid_config, "certv2_candidate_multiplier", 3.0),
    )
    support = _virtual_support(
        metric_flat=metric_flat,
        candidates=candidates,
        demand=demand,
        frame_ids=frame_ids,
        temporal_ids=temporal_ids,
    )
    candidate_score = 0.82 * quality[candidates] + 0.18 * support
    selected_list = _global_arbitration(
        budget=budget,
        candidates=candidates,
        mandatory=mandatory,
        metric_flat=metric_flat,
        base_score=candidate_score,
        temporal_ids=temporal_ids,
        cell_ids=cell_ids,
        component_ids=component_ids,
        frame_ids=frame_ids,
        diversity_weight=_cfg_float(flashvid_config, "certv2_diversity_weight", 0.24),
        coverage_weight=_cfg_float(flashvid_config, "certv2_coverage_weight", 0.12),
    )
    if len(selected_list) < budget:
        selected_set = set(selected_list)
        for token_idx in torch.argsort(quality, descending=True, stable=True).detach().cpu().tolist():
            if int(token_idx) not in selected_set:
                selected_list.append(int(token_idx))
                selected_set.add(int(token_idx))
            if len(selected_list) >= budget:
                break
    selected = torch.tensor(sorted(selected_list[:budget]), dtype=torch.long, device=video_features.device)
    output = flat_features.index_select(0, selected)

    flashvid_config.vision_token_length = int(output.shape[0])
    flashvid_config.visual_token_length = int(output.shape[0])
    flashvid_config.llm_token_length = None
    setattr(flashvid_config, "last_adapter_variant", "certvid_v2")
    setattr(flashvid_config, "last_adapter_raw_tokens", float(total_tokens))
    setattr(flashvid_config, "last_adapter_output_tokens", float(output.shape[0]))
    setattr(flashvid_config, "last_certv2_target_tokens", float(budget))
    setattr(flashvid_config, "last_certv2_candidate_tokens", float(candidates.numel()))
    setattr(flashvid_config, "last_certv2_component_count", float(component_sizes.numel()))
    setattr(flashvid_config, "last_certv2_query_confidence", float(query_confidence))
    setattr(flashvid_config, "last_certv2_mandatory_tokens", float(len(mandatory)))
    return output, selected
