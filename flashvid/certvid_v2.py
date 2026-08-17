from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn.functional as F

from .certvid import (
    CertVidPlan,
    _build_components,
    _build_plan,
    _candidate_pool as _certvid_candidate_pool,
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
    _stochastic_coreset as _certvid_stochastic_coreset,
    _temporal_signals,
    apply_certvid_plan,
)
from .configuration_flashvid import FlashVidConfig


_EXACT_CUDA_GRAPH_ENV = "CERTV3_USE_EXACT_CUDA_GRAPHS"
_TRAJECTORY_GRAPH_CACHE: dict[tuple[object, ...], dict[str, object]] = {}


def _exact_cuda_graph_enabled(tensor: torch.Tensor) -> bool:
    value = os.environ.get(_EXACT_CUDA_GRAPH_ENV, "").strip().lower()
    return tensor.is_cuda and value in {"1", "true", "yes", "on"}


def _effective_ratio(config: FlashVidConfig) -> float:
    ratio = _cfg_float(config, "retention_ratio", 0.10)
    if bool(getattr(config, "certv2_budget_uses_expansion", True)):
        ratio *= _cfg_float(config, "expansion", 1.0)
    return min(1.0, max(0.0, ratio))


def _identity_plan(total_tokens: int, device: torch.device) -> CertVidPlan:
    indices = torch.arange(total_tokens, dtype=torch.long, device=device)
    return CertVidPlan(
        anchor_indices=indices,
        assignment_indices=indices.unsqueeze(1),
        assignment_weights=torch.ones((total_tokens, 1), dtype=torch.float32, device=device),
        source_mass=torch.ones(total_tokens, dtype=torch.float32, device=device),
        fusion_alpha=torch.zeros(total_tokens, dtype=torch.float32, device=device),
        raw_token_count=total_tokens,
    )


def _density_peaks(metric_frames: torch.Tensor, neighbors: int) -> torch.Tensor:
    """Local density peaks used only as a repair signal, never as a hard selector."""
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


def _trajectory_signals_eager(
    metric_frames: torch.Tensor,
    coords: torch.Tensor,
    spatial_penalty: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
]:
    """Measure bidirectional novelty and second-order motion without forcing coverage."""
    spatial_distance = torch.cdist(coords.float(), coords.float(), p=2)
    frame_event, forward_novelty, matches = _temporal_signals(
        metric_frames,
        coords,
        spatial_penalty,
        spatial_distance,
    )
    frame_count, tokens_per_frame, _ = metric_frames.shape
    backward_novelty = torch.zeros_like(forward_novelty)
    curvature = torch.zeros_like(forward_novelty)
    next_matches: list[torch.Tensor] = []
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
        direction_change = 1.0 - F.cosine_similarity(incoming, outgoing, dim=-1, eps=1e-6)
        motion_gate = torch.sqrt(incoming.norm(dim=-1) * outgoing.norm(dim=-1)).clamp(0.0, 1.0)
        curvature[frame_idx] = direction_change.clamp(0.0, 2.0) * 0.5 * motion_gate

    novelty = _minmax(0.5 * forward_novelty + 0.5 * backward_novelty, dim=-1)
    curvature = _minmax(curvature, dim=-1)
    return frame_event, forward_novelty, novelty, curvature, matches


def _trajectory_signals(
    metric_frames: torch.Tensor,
    coords: torch.Tensor,
    spatial_penalty: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
]:
    """Run the unchanged trajectory program eagerly or from an exact CUDA graph."""
    if not _exact_cuda_graph_enabled(metric_frames):
        return _trajectory_signals_eager(metric_frames, coords, spatial_penalty)

    device_index = metric_frames.device.index
    key = (
        device_index,
        tuple(metric_frames.shape),
        metric_frames.dtype,
        tuple(coords.shape),
        coords.dtype,
        float(spatial_penalty),
    )
    cached = _TRAJECTORY_GRAPH_CACHE.get(key)
    if cached is None:
        static_metric = metric_frames.detach().clone()
        static_coords = coords.detach().clone()
        try:
            warmup_stream = torch.cuda.Stream(device=metric_frames.device)
            warmup_stream.wait_stream(torch.cuda.current_stream(metric_frames.device))
            with torch.cuda.stream(warmup_stream):
                _trajectory_signals_eager(
                    static_metric,
                    static_coords,
                    spatial_penalty,
                )
            torch.cuda.current_stream(metric_frames.device).wait_stream(warmup_stream)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_output = _trajectory_signals_eager(
                    static_metric,
                    static_coords,
                    spatial_penalty,
                )
            cached = {
                "graph": graph,
                "metric": static_metric,
                "output": static_output,
            }
            print(
                "[CertVID V3] captured trajectory CUDA graph "
                f"for shape={tuple(metric_frames.shape)}"
            )
        except RuntimeError as error:
            # Unsupported CUDA/runtime combinations retain the original path.
            cached = {"disabled": True}
            print(
                "[CertVID V3] trajectory CUDA graph unavailable; "
                f"using eager path: {error}"
            )
        _TRAJECTORY_GRAPH_CACHE[key] = cached

    if bool(cached.get("disabled", False)):
        return _trajectory_signals_eager(metric_frames, coords, spatial_penalty)

    static_metric = cached["metric"]
    assert isinstance(static_metric, torch.Tensor)
    static_metric.copy_(metric_frames)
    graph = cached["graph"]
    assert isinstance(graph, torch.cuda.CUDAGraph)
    graph.replay()
    return cached["output"]  # type: ignore[return-value]


def _component_support(
    metric_flat: torch.Tensor,
    component_ids: torch.Tensor,
    component_sizes: torch.Tensor,
    frame_ids: torch.Tensor,
    frame_count: int,
) -> torch.Tensor:
    """Reward persistent but non-static trajectories."""
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
    first.scatter_reduce_(
        0,
        component_ids,
        frame_ids,
        reduce="amin",
        include_self=True,
    )
    last.scatter_reduce_(
        0,
        component_ids,
        frame_ids,
        reduce="amax",
        include_self=True,
    )
    span = (last - first + 1).clamp_min(1).float() / max(1, frame_count)
    mass = torch.log1p(component_sizes.float())
    mass = mass / mass.max().clamp_min(1e-6)
    component_score = _minmax(0.42 * span + 0.28 * mass + 0.30 * variation, dim=0)
    return component_score[component_ids]


def _attention_concentration(attention: torch.Tensor) -> float:
    values = _minmax(attention.reshape(-1).float(), dim=0)
    count = int(values.numel())
    if count <= 1:
        return 1.0
    probability = (values + 1e-6) / (values.sum() + 1e-6 * count)
    top_count = max(1, int(math.ceil(0.10 * count)))
    top_mass = float(torch.topk(probability, k=top_count).values.sum().item())
    return min(1.0, max(0.0, (top_mass - 0.10) / 0.40))


def _top_mean(values: torch.Tensor, fraction: float) -> float:
    flat = values.reshape(-1).float()
    if flat.numel() == 0:
        return 0.0
    count = min(flat.numel(), max(1, int(math.ceil(flat.numel() * fraction))))
    return float(torch.topk(flat, k=count).values.mean().item())


def _trajectory_complexity(metric_frames: torch.Tensor) -> float:
    """Estimate motion on the original cosine scale instead of min-max ranks."""
    frame_count = int(metric_frames.shape[0])
    if frame_count <= 1:
        return 0.0
    representatives = F.normalize(metric_frames.mean(dim=1), p=2, dim=-1, eps=1e-6)
    change = (1.0 - torch.sum(representatives[1:] * representatives[:-1], dim=-1)).clamp(0.0, 2.0) * 0.5
    turn = change.new_zeros(max(0, frame_count - 2))
    if frame_count > 2:
        incoming = representatives[1:-1] - representatives[:-2]
        outgoing = representatives[2:] - representatives[1:-1]
        direction = (1.0 - F.cosine_similarity(incoming, outgoing, dim=-1, eps=1e-6)).clamp(0.0, 2.0) * 0.5
        gate = torch.sqrt(incoming.norm(dim=-1) * outgoing.norm(dim=-1)).clamp(0.0, 1.0)
        turn = direction * gate
    value = 0.55 * float(change.mean().item()) + 0.25 * _top_mean(change, 0.25)
    if turn.numel() > 0:
        value += 0.20 * float(turn.mean().item())
    return min(1.0, max(0.0, value))


def _repair_fraction(
    config: FlashVidConfig,
    ratio: float,
    trajectory_complexity: float,
    attention_concentration: float,
    query_confidence: float,
) -> float:
    low = min(max(_cfg_float(config, "certv2_repair_ratio", 0.05), 0.0), 0.25)
    high = min(max(_cfg_float(config, "certv2_repair_ratio_high", 0.13), low), 0.30)
    phase = min(1.0, max(0.0, (ratio - 0.125) / 0.1875))
    nominal = low + phase * (high - low)
    route = (
        0.58 * trajectory_complexity
        + 0.27 * (1.0 - attention_concentration)
        + 0.15 * (1.0 - query_confidence)
    )
    strength = min(max(_cfg_float(config, "certv2_router_strength", 0.65), 0.0), 1.0)
    scale = (1.0 - strength) + strength * (0.45 + 1.10 * route)
    return min(0.18, max(0.0, nominal * scale))


def _repair_backbone(
    *,
    selected: torch.Tensor,
    metric_flat: torch.Tensor,
    evidence: torch.Tensor,
    repair_score: torch.Tensor,
    query_score: torch.Tensor,
    frame_ids: torch.Tensor,
    temporal_ids: torch.Tensor,
    cell_ids: torch.Tensor,
    component_ids: torch.Tensor,
    max_swaps: int,
    protect_ratio: float,
    frame_floor_ratio: float,
    candidate_multiplier: float,
    diversity_weight: float,
    coverage_weight: float,
    swap_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replace only redundant, low-value V1 anchors with high-gain event evidence."""
    total_tokens = int(metric_flat.shape[0])
    budget = int(selected.numel())
    device = selected.device
    if max_swaps <= 0 or budget <= 1 or budget >= total_tokens:
        return selected.sort().values, selected[:0], selected[:0]

    selected = selected.to(device=device, dtype=torch.long).clone()
    selected_mask = torch.zeros(total_tokens, dtype=torch.bool, device=device)
    selected_mask[selected] = True

    protection_score = 0.62 * evidence[selected] + 0.23 * query_score[selected] + 0.15 * repair_score[selected]
    protect_count = min(budget, max(1, int(math.ceil(budget * min(max(protect_ratio, 0.0), 0.80)))))
    protected_mask = torch.zeros(total_tokens, dtype=torch.bool, device=device)
    protected_mask[selected[torch.topk(protection_score, k=protect_count).indices]] = True
    for temporal_id in torch.unique(temporal_ids).detach().cpu().tolist():
        members = selected[temporal_ids[selected] == int(temporal_id)]
        if members.numel() > 0:
            pole_score = 0.60 * evidence[members] + 0.40 * repair_score[members]
            protected_mask[members[torch.argmax(pole_score)]] = True

    outside = torch.where(~selected_mask)[0]
    if outside.numel() == 0:
        return selected.sort().values, selected[:0], selected[protected_mask[selected]]
    pool_limit = min(
        int(outside.numel()),
        max(max_swaps * 12, int(math.ceil(budget * max(1.0, candidate_multiplier)))),
    )
    candidate_set: set[int] = set()
    ranked_outside = outside[torch.argsort(repair_score[outside], descending=True, stable=True)]
    candidate_set.update(int(idx) for idx in ranked_outside[:pool_limit].detach().cpu().tolist())
    for frame_idx in torch.unique(frame_ids).detach().cpu().tolist():
        members = outside[frame_ids[outside] == int(frame_idx)]
        if members.numel() > 0:
            candidate_set.add(int(members[torch.argmax(repair_score[members])].item()))
    for temporal_id in torch.unique(temporal_ids).detach().cpu().tolist():
        members = outside[temporal_ids[outside] == int(temporal_id)]
        if members.numel() > 0:
            candidate_set.add(int(members[torch.argmax(repair_score[members])].item()))
    for cell_id in torch.unique(cell_ids).detach().cpu().tolist():
        members = outside[cell_ids[outside] == int(cell_id)]
        if members.numel() > 0:
            candidate_set.add(int(members[torch.argmax(repair_score[members])].item()))
    pool = torch.tensor(sorted(candidate_set), dtype=torch.long, device=device)
    if pool.numel() > pool_limit:
        pool = pool[torch.argsort(repair_score[pool], descending=True, stable=True)[:pool_limit]]

    metric_dtype = torch.float16 if device.type == "cuda" else torch.float32
    pool_similarity = metric_flat[pool].to(metric_dtype) @ metric_flat[selected].to(metric_dtype).transpose(0, 1)
    pool_max_similarity = pool_similarity.max(dim=1).values.float()
    diversity = ((1.0 - pool_max_similarity).clamp(0.0, 2.0) * 0.5)

    selected_similarity = metric_flat[selected].to(metric_dtype) @ metric_flat[selected].to(metric_dtype).transpose(0, 1)
    selected_similarity.fill_diagonal_(-2.0)
    selected_max_similarity = selected_similarity.max(dim=1).values.float()
    uniqueness = ((1.0 - selected_max_similarity).clamp(0.0, 2.0) * 0.5)
    removal_base = (
        0.54 * evidence[selected]
        + 0.16 * repair_score[selected]
        + 0.14 * query_score[selected]
        + 0.16 * uniqueness
    )

    frame_count = int(frame_ids.max().item()) + 1
    temporal_count = int(temporal_ids.max().item()) + 1
    cell_count = int(cell_ids.max().item()) + 1
    component_count = int(component_ids.max().item()) + 1
    frame_counts = torch.bincount(frame_ids[selected], minlength=frame_count).float()
    temporal_counts = torch.bincount(temporal_ids[selected], minlength=temporal_count).float()
    cell_counts = torch.bincount(cell_ids[selected], minlength=cell_count).float()
    component_counts = torch.bincount(component_ids[selected], minlength=component_count).float()
    desired_frames = min(
        frame_count,
        max(1, int(round(budget * min(max(frame_floor_ratio, 0.0), 0.25)))),
    )

    available_pool = torch.ones(pool.numel(), dtype=torch.bool, device=device)
    removable = ~protected_mask[selected]
    added: list[int] = []
    removed: list[int] = []
    for _ in range(min(max_swaps, int(removable.sum().item()), int(pool.numel()))):
        pool_frames = frame_ids[pool]
        pool_temporal = temporal_ids[pool]
        pool_cells = cell_ids[pool]
        pool_components = component_ids[pool]
        covered_frames = int((frame_counts > 0).sum().item())
        need_frames = float(covered_frames < desired_frames)
        coverage = (
            0.45 * (frame_counts[pool_frames] == 0).float() * need_frames
            + 0.25 * (temporal_counts[pool_temporal] == 0).float()
            + 0.20 * (cell_counts[pool_cells] == 0).float()
            + 0.10 * (component_counts[pool_components] == 0).float()
        )
        add_gain = (
            0.58 * repair_score[pool]
            + 0.18 * evidence[pool]
            + float(max(0.0, diversity_weight)) * diversity
            + float(max(0.0, coverage_weight)) * coverage
        )
        add_gain = add_gain.masked_fill(~available_pool, -float("inf"))
        add_column = int(torch.argmax(add_gain).item())
        if not torch.isfinite(add_gain[add_column]):
            break

        remove_frames = frame_ids[selected]
        remove_temporal = temporal_ids[selected]
        remove_cells = cell_ids[selected]
        remove_components = component_ids[selected]
        frame_critical = (frame_counts[remove_frames] <= 1).float() * float(covered_frames <= desired_frames)
        temporal_critical = (temporal_counts[remove_temporal] <= 1).float()
        cell_critical = (cell_counts[remove_cells] <= 1).float()
        component_critical = (component_counts[remove_components] <= 1).float()
        removal_cost = (
            removal_base
            + 0.60 * frame_critical
            + 0.25 * temporal_critical
            + 0.05 * cell_critical
            + 0.10 * component_critical
        ).masked_fill(~removable, float("inf"))
        remove_column = int(torch.argmin(removal_cost).item())
        if not torch.isfinite(removal_cost[remove_column]):
            break
        if float(add_gain[add_column].item()) <= float(removal_cost[remove_column].item()) + float(swap_margin):
            break

        add_token = int(pool[add_column].item())
        remove_token = int(selected[remove_column].item())
        frame_counts[frame_ids[remove_token]] -= 1.0
        temporal_counts[temporal_ids[remove_token]] -= 1.0
        cell_counts[cell_ids[remove_token]] -= 1.0
        component_counts[component_ids[remove_token]] -= 1.0
        frame_counts[frame_ids[add_token]] += 1.0
        temporal_counts[temporal_ids[add_token]] += 1.0
        cell_counts[cell_ids[add_token]] += 1.0
        component_counts[component_ids[add_token]] += 1.0
        selected_mask[remove_token] = False
        selected_mask[add_token] = True
        selected[remove_column] = add_token
        removable[remove_column] = False
        available_pool[add_column] = False
        added.append(add_token)
        removed.append(remove_token)

        similarity_to_added = metric_flat[pool].to(metric_dtype) @ metric_flat[add_token].to(metric_dtype)
        pool_max_similarity = torch.maximum(pool_max_similarity, similarity_to_added.float())
        diversity = ((1.0 - pool_max_similarity).clamp(0.0, 2.0) * 0.5)

    final = selected.sort().values
    added_tensor = torch.tensor(sorted(added), dtype=torch.long, device=device)
    protected_tensor = torch.where(protected_mask & selected_mask)[0]
    return final, added_tensor, protected_tensor


def certvid_v2_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
    question_features: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evidence-first compression with gated trajectory repair and shared fusion."""
    if video_features.ndim != 3:
        raise ValueError(f"expected video_features [T, HW, D], got {tuple(video_features.shape)}")
    frame_count, tokens_per_frame, _ = video_features.shape
    total_tokens = int(frame_count * tokens_per_frame)
    ratio = _effective_ratio(flashvid_config)
    budget = max(1, min(total_tokens, int(round(total_tokens * ratio))))
    flat_features = video_features.reshape(total_tokens, -1)
    if budget >= total_tokens:
        indices = torch.arange(total_tokens, dtype=torch.long, device=video_features.device)
        plan = _identity_plan(total_tokens, video_features.device)
        setattr(flashvid_config, "_certvid_plan", plan)
        output = flat_features
        repair_fraction = 0.0
        trajectory_complexity = 0.0
        concentration = 1.0
        query_confidence = 0.0
        added = indices[:0]
        candidates = total_tokens
        components = total_tokens
    else:
        metric_dim = max(32, _cfg_int(flashvid_config, "certv2_metric_dim", 256))
        metric_flat = _metric_features(video_features, metric_dim)
        metric_frames = metric_flat.view(frame_count, tokens_per_frame, -1)
        height, width = _grid_hw(tokens_per_frame, flashvid_config)
        spatial_bins = max(1, _cfg_int(flashvid_config, "certv2_spatial_bins", 3))
        coords, spatial_cells = _spatial_layout(
            tokens_per_frame,
            height,
            width,
            spatial_bins,
            video_features.device,
        )
        frame_event, forward_novelty_2d, novelty_2d, curvature_2d, matches = _trajectory_signals(
            metric_frames,
            coords,
            _cfg_float(flashvid_config, "certv2_spatial_penalty", 0.08),
        )
        density = _density_peaks(
            metric_frames,
            _cfg_int(flashvid_config, "certv2_density_neighbors", 4),
        ).reshape(-1)
        component_ids_cpu, component_sizes_cpu = _build_components(
            frame_count,
            tokens_per_frame,
            frame_event,
            matches,
            _cfg_float(flashvid_config, "certv2_track_threshold", 0.82),
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
        temporal_bins = min(frame_count, max(1, _cfg_int(flashvid_config, "certv2_temporal_bins", 8)))
        temporal_ids = torch.div(
            frame_ids * temporal_bins,
            max(1, frame_count),
            rounding_mode="floor",
        ).clamp_max(temporal_bins - 1)
        cell_ids = temporal_ids * (spatial_bins * spatial_bins) + spatial_cells.repeat(frame_count)

        attention = _rank_normalize(cls_attention.float()).reshape(-1)
        novelty = novelty_2d.reshape(-1)
        curvature = curvature_2d.reshape(-1)
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
        query_weight = min(
            0.30,
            max(0.0, _cfg_float(flashvid_config, "certv2_query_weight", 0.18) * query_confidence),
        )
        visual_evidence = _minmax(
            0.34 * attention + 0.24 * novelty + 0.18 * detail + 0.12 * event + 0.12 * query_score,
            dim=0,
        )
        evidence = _minmax((1.0 - query_weight) * visual_evidence + query_weight * query_score, dim=0)
        repair_score = _minmax(
            0.27 * novelty
            + 0.24 * curvature
            + 0.15 * event
            + 0.12 * density
            + 0.12 * component_support
            + 0.10 * query_score,
            dim=0,
        )

        # Keep the proven CertVID V1 coreset as the full-budget backbone. The
        # V2 branch may only replace anchors after this selection is complete.
        backbone_novelty = forward_novelty_2d.reshape(-1)
        demand_weight = (
            0.40 + 0.22 * attention + 0.18 * backbone_novelty + 0.12 * detail + 0.08 * event
        )
        demand_weight = demand_weight / demand_weight.sum().clamp_min(1e-6)
        backbone_quality = _minmax(
            0.28 * attention
            + 0.24 * backbone_novelty
            + 0.20 * detail
            + 0.14 * event
            + 0.14 * query_score,
            dim=0,
        )
        cell_value = _minmax(0.55 * detail + 0.30 * attention + 0.15 * backbone_novelty, dim=0)
        temporal_value = _minmax(0.55 * event + 0.30 * backbone_novelty + 0.15 * query_score, dim=0)
        base_candidates, mandatory, _ = _certvid_candidate_pool(
            budget,
            backbone_quality,
            component_ids,
            temporal_ids,
            cell_ids,
            query_relevance,
            query_confidence,
            _cfg_float(flashvid_config, "cert_candidate_multiplier", 3.0),
        )
        backbone_repair_ratio = min(
            max(_cfg_float(flashvid_config, "cert_repair_ratio", 0.20), 0.0),
            0.50,
        )
        main_budget = max(len(mandatory), budget - int(round(budget * backbone_repair_ratio)))
        base_selected_list = _certvid_stochastic_coreset(
            budget=budget,
            main_budget=main_budget,
            metric_features=metric_flat,
            candidates=base_candidates,
            mandatory=mandatory,
            demand_weight=demand_weight,
            query_relevance=query_relevance,
            atom_weights=atom_weights,
            temporal_ids=temporal_ids,
            temporal_value=temporal_value,
            cell_ids=cell_ids,
            cell_value=cell_value,
            component_ids=component_ids,
            component_sizes=component_sizes,
            query_weight=_cfg_float(flashvid_config, "cert_query_weight", 0.20) * query_confidence,
            temporal_weight=_cfg_float(flashvid_config, "cert_temporal_weight", 0.20),
            detail_weight=_cfg_float(flashvid_config, "cert_detail_weight", 0.10),
        )
        base_selected = torch.tensor(
            sorted(base_selected_list),
            dtype=torch.long,
            device=video_features.device,
        )
        if int(base_selected.numel()) != budget:
            raise RuntimeError(
                f"CertVID V1 backbone produced {int(base_selected.numel())} tokens for V2 budget {budget}"
            )

        trajectory_complexity = _trajectory_complexity(metric_frames)
        concentration = _attention_concentration(cls_attention)
        repair_fraction = _repair_fraction(
            flashvid_config,
            ratio,
            trajectory_complexity,
            concentration,
            query_confidence,
        )
        max_swaps = min(budget, int(round(budget * repair_fraction)))
        selected, added, protected = _repair_backbone(
            selected=base_selected,
            metric_flat=metric_flat,
            evidence=evidence,
            repair_score=repair_score,
            query_score=query_score,
            frame_ids=frame_ids,
            temporal_ids=temporal_ids,
            cell_ids=cell_ids,
            component_ids=component_ids,
            max_swaps=max_swaps,
            protect_ratio=_cfg_float(flashvid_config, "certv2_protect_ratio", 0.30),
            frame_floor_ratio=_cfg_float(flashvid_config, "certv2_frame_floor_ratio", 0.08),
            candidate_multiplier=_cfg_float(flashvid_config, "certv2_candidate_multiplier", 3.0),
            diversity_weight=_cfg_float(flashvid_config, "certv2_diversity_weight", 0.12),
            coverage_weight=_cfg_float(flashvid_config, "certv2_coverage_weight", 0.10),
            swap_margin=_cfg_float(flashvid_config, "certv2_swap_margin", 0.02),
        )
        plan = _build_plan(
            selected=selected,
            metric_features=metric_flat,
            demand_weight=demand_weight,
            attention=attention,
            query_score=query_score,
            temporal_ids=temporal_ids,
            component_ids=component_ids,
            fusion_alpha=_cfg_float(flashvid_config, "certv2_fusion_alpha", 0.25),
            temperature=_cfg_float(flashvid_config, "certv2_assignment_temperature", 0.07),
        )
        if protected.numel() > 0:
            protected_mask = torch.zeros(total_tokens, dtype=torch.bool, device=selected.device)
            protected_mask[protected] = True
            plan.fusion_alpha[protected_mask[selected]] = 0.0
        if added.numel() > 0:
            added_mask = torch.zeros(total_tokens, dtype=torch.bool, device=selected.device)
            added_mask[added] = True
            repair_alpha = min(max(_cfg_float(flashvid_config, "certv2_repair_fusion_alpha", 0.08), 0.0), 0.50)
            plan.fusion_alpha[added_mask[selected]] = torch.clamp(
                plan.fusion_alpha[added_mask[selected]],
                max=repair_alpha,
            )
        output = apply_certvid_plan(flat_features, plan)
        setattr(flashvid_config, "_certvid_plan", plan)
        candidates = int(base_candidates.numel())
        components = int(component_sizes.numel())

    flashvid_config.vision_token_length = int(output.shape[0])
    flashvid_config.visual_token_length = int(output.shape[0])
    flashvid_config.llm_token_length = None
    setattr(flashvid_config, "last_adapter_variant", "certvid_v2")
    setattr(flashvid_config, "last_adapter_raw_tokens", float(total_tokens))
    setattr(flashvid_config, "last_adapter_output_tokens", float(output.shape[0]))
    setattr(flashvid_config, "last_certv2_target_tokens", float(budget))
    setattr(flashvid_config, "last_certv2_candidate_tokens", float(candidates))
    setattr(flashvid_config, "last_certv2_component_count", float(components))
    setattr(flashvid_config, "last_certv2_query_confidence", float(query_confidence))
    setattr(flashvid_config, "last_certv2_repair_fraction", float(repair_fraction))
    setattr(flashvid_config, "last_certv2_repair_tokens", float(added.numel()))
    setattr(flashvid_config, "last_certv2_trajectory_complexity", float(trajectory_complexity))
    setattr(flashvid_config, "last_certv2_attention_concentration", float(concentration))
    return output, plan.anchor_indices
