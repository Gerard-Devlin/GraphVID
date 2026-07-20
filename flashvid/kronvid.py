from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from .configuration_flashvid import FlashVidConfig


def _cfg_float(config: FlashVidConfig, name: str, default: float) -> float:
    try:
        return float(getattr(config, name, default))
    except (TypeError, ValueError):
        return float(default)


def _cfg_int(config: FlashVidConfig, name: str, default: int) -> int:
    try:
        return int(getattr(config, name, default))
    except (TypeError, ValueError):
        return int(default)


def _grid_hw(tokens_per_frame: int, config: FlashVidConfig) -> tuple[int, int]:
    height = _cfg_int(config, "H", 0)
    width = _cfg_int(config, "W", 0)
    if height > 0 and width > 0 and height * width == tokens_per_frame:
        return height, width
    height = max(1, int(math.sqrt(tokens_per_frame)))
    while height > 1 and tokens_per_frame % height != 0:
        height -= 1
    return height, max(1, tokens_per_frame // height)


@dataclass
class KronSegmentPlan:
    """Galerkin operator for one contiguous temporal block."""

    source_indices: torch.Tensor
    anchor_indices: torch.Tensor
    output_positions: torch.Tensor
    prolongation: torch.Tensor
    reduced_laplacian: torch.Tensor
    node_weights: torch.Tensor
    system_cholesky: Optional[torch.Tensor]
    identity_rho: float
    pure_pruning: bool


@dataclass
class KronVidPlan:
    """Reusable graph-coarsening plan shared by base and DeepStack features."""

    anchor_indices: torch.Tensor
    segments: tuple[KronSegmentPlan, ...]
    raw_token_count: int


def _resolve_budget(
    config: FlashVidConfig,
    total_tokens: int,
) -> tuple[int, dict[str, object]]:
    mode = str(getattr(config, "kron_budget_mode", "layer_average")).strip().lower()
    if mode not in {"layer_average", "outer_only"}:
        raise ValueError(f"unsupported kron_budget_mode={mode!r}")

    nominal = _cfg_float(config, "retention_ratio", 0.10)
    expansion = _cfg_float(config, "expansion", 1.0)
    pruning_layer = _cfg_int(config, "pruning_layer", 0)
    inner_retention = _cfg_float(config, "llm_retention_ratio", 1.0)
    layers = _cfg_int(config, "kron_num_hidden_layers", 0)
    tolerance = 1e-4
    if not (0.0 < nominal <= 1.0):
        raise ValueError(f"retention_ratio must be in (0, 1], got {nominal}")

    if mode == "outer_only":
        if abs(expansion - 1.0) > tolerance or abs(inner_retention - 1.0) > tolerance:
            raise ValueError(
                "kronvid outer_only requires expansion=1 and llm_retention_ratio=1; "
                f"got expansion={expansion}, llm_retention_ratio={inner_retention}"
            )
        outer_retention = nominal
        post_inner_retention = nominal
        layer_multiplier = 1.0
    else:
        if layers <= 1:
            raise ValueError("kronvid layer_average requires kron_num_hidden_layers")
        if not (0 < pruning_layer < layers):
            raise ValueError(
                f"pruning_layer must satisfy 0 < K < L, got K={pruning_layer}, L={layers}"
            )
        if not (0.0 < inner_retention < 1.0):
            raise ValueError(
                "kronvid layer_average requires 0 < llm_retention_ratio < 1, "
                f"got {inner_retention}"
            )
        if not bool(getattr(config, "kron_inner_hook_enabled", False)):
            raise ValueError("kronvid layer_average requires an installed inner-pruning hook")
        if nominal * expansion > 1.0 + tolerance:
            raise ValueError(
                f"outer retention R*E must not exceed 1, got {nominal * expansion:.8f}"
            )
        layer_multiplier = expansion * (
            pruning_layer + (layers - pruning_layer) * inner_retention
        ) / float(layers)
        if abs(layer_multiplier - 1.0) > tolerance:
            raise ValueError(
                "kronvid layer_average budget is not aligned: "
                f"E*(K+(L-K)*r)/L={layer_multiplier:.8f}, expected 1 within {tolerance}"
            )
        outer_retention = nominal * expansion
        post_inner_retention = outer_retention * inner_retention

    budget = max(1, int(round(total_tokens * outer_retention)))
    if budget > total_tokens:
        raise ValueError(f"kronvid outer budget {budget} exceeds raw token count {total_tokens}")
    post_inner_tokens = (
        max(1, int(round(budget * inner_retention))) if mode == "layer_average" else budget
    )
    average_layer_tokens = (
        (pruning_layer * budget + (layers - pruning_layer) * post_inner_tokens) / float(layers)
        if mode == "layer_average"
        else float(budget)
    )
    diagnostics: dict[str, object] = {
        "mode": mode,
        "nominal_retention": nominal,
        "outer_retention": outer_retention,
        "post_inner_retention": post_inner_retention,
        "average_layer_multiplier": layer_multiplier,
        "raw_tokens": total_tokens,
        "target_tokens": budget,
        "post_inner_tokens": post_inner_tokens,
        "average_layer_tokens": average_layer_tokens,
    }
    return budget, diagnostics


def _fixed_projection(
    features: torch.Tensor,
    output_dim: int,
    seed: int,
) -> torch.Tensor:
    flat = features.float()
    centered = flat - flat.mean(dim=0, keepdim=True)
    output_dim = max(8, min(int(output_dim), int(centered.shape[1])))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    projection = torch.randn(
        (centered.shape[1], output_dim),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ) / math.sqrt(float(output_dim))
    projected = centered @ projection.to(centered.device)
    return F.normalize(
        torch.nan_to_num(projected, nan=0.0, posinf=0.0, neginf=0.0),
        p=2,
        dim=-1,
        eps=1e-6,
    )


def _token_coordinates(
    frame_count: int,
    tokens_per_frame: int,
    height: int,
    width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    frame_ids = torch.arange(frame_count, device=device).repeat_interleave(tokens_per_frame)
    patch_ids = torch.arange(tokens_per_frame, device=device).repeat(frame_count)
    rows = torch.div(patch_ids, width, rounding_mode="floor").clamp_max(height - 1)
    cols = torch.remainder(patch_ids, width).clamp_max(width - 1)
    coordinates = torch.stack(
        [
            frame_ids.float() / max(1, frame_count - 1),
            rows.float() / max(1, height - 1),
            cols.float() / max(1, width - 1),
        ],
        dim=1,
    )
    return coordinates, frame_ids.long(), rows.long(), cols.long()


def _position_fourier(coordinates: torch.Tensor, frequencies: int) -> torch.Tensor:
    frequencies = max(1, int(frequencies))
    parts: list[torch.Tensor] = []
    centered = coordinates.float() * 2.0 - 1.0
    for frequency in range(frequencies):
        phase = centered * (math.pi * float(2**frequency))
        parts.extend([torch.sin(phase), torch.cos(phase)])
    return F.normalize(torch.cat(parts, dim=1), p=2, dim=-1, eps=1e-6)


def _augmented_features(
    video_features: torch.Tensor,
    coordinates: torch.Tensor,
    config: FlashVidConfig,
) -> torch.Tensor:
    visual = _fixed_projection(
        video_features.reshape(-1, video_features.shape[-1]),
        _cfg_int(config, "kron_metric_dim", 64),
        _cfg_int(config, "kron_projection_seed", 17),
    )
    position = _position_fourier(
        coordinates,
        _cfg_int(config, "kron_position_frequencies", 3),
    )
    position_weight = min(0.75, max(0.0, _cfg_float(config, "kron_position_weight", 0.20)))
    visual_weight = max(1e-6, 1.0 - position_weight)
    return F.normalize(
        torch.cat(
            [
                visual * math.sqrt(visual_weight),
                position * math.sqrt(max(position_weight, 1e-6)),
            ],
            dim=1,
        ),
        p=2,
        dim=-1,
        eps=1e-6,
    )


def _segment_ranges(frame_count: int, segment_count: int) -> list[tuple[int, int]]:
    segment_count = max(1, min(int(segment_count), frame_count))
    boundaries = torch.linspace(0, frame_count, steps=segment_count + 1).round().long().tolist()
    ranges: list[tuple[int, int]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end > start:
            ranges.append((int(start), int(end)))
    return ranges


def _ridge_scale(features: torch.Tensor, relative_ridge: float) -> tuple[torch.Tensor, float]:
    gram = features.transpose(0, 1) @ features
    mean_eigenvalue = float(torch.trace(gram).item()) / float(max(1, gram.shape[0]))
    ridge = max(1e-6, float(relative_ridge) * max(mean_eigenvalue, 1e-6))
    return gram, ridge


def _effective_dimension(features: torch.Tensor, relative_ridge: float) -> float:
    singular_values = torch.linalg.svdvals(features.float())
    eigenvalues = singular_values.square()
    if eigenvalues.numel() == 0:
        return 0.0
    ridge = max(
        1e-6,
        float(relative_ridge) * float(eigenvalues.sum().item()) / float(max(1, features.shape[1])),
    )
    return float((eigenvalues / (eigenvalues + ridge)).sum().item())


def _allocate_budget(
    total_budget: int,
    capacities: Sequence[int],
    effective_dimensions: Sequence[float],
    floor_ratio: float,
) -> list[int]:
    count = len(capacities)
    if count == 0:
        return []
    floor_ratio = min(1.0, max(0.0, float(floor_ratio)))
    base = max(1, int(math.floor((total_budget / float(count)) * floor_ratio)))
    allocation = [min(int(capacity), base) for capacity in capacities]
    while sum(allocation) > total_budget:
        candidates = [idx for idx, value in enumerate(allocation) if value > 0]
        idx = min(candidates, key=lambda item: (effective_dimensions[item], -item))
        allocation[idx] -= 1

    remaining = total_budget - sum(allocation)
    while remaining > 0:
        active = [idx for idx, cap in enumerate(capacities) if allocation[idx] < int(cap)]
        if not active:
            break
        weights = torch.tensor(
            [max(1e-6, float(effective_dimensions[idx])) for idx in active],
            dtype=torch.float64,
        )
        raw = weights / weights.sum() * float(remaining)
        additions = torch.floor(raw).long().tolist()
        progressed = 0
        for position, idx in enumerate(active):
            add = min(int(additions[position]), int(capacities[idx]) - allocation[idx])
            if add > 0:
                allocation[idx] += add
                remaining -= add
                progressed += add
        if remaining <= 0:
            break
        fractions = (raw - torch.floor(raw)).tolist()
        order = sorted(
            range(len(active)),
            key=lambda position: (-fractions[position], -effective_dimensions[active[position]], active[position]),
        )
        for position in order:
            idx = active[position]
            if allocation[idx] >= int(capacities[idx]):
                continue
            allocation[idx] += 1
            remaining -= 1
            progressed += 1
            if remaining <= 0:
                break
        if progressed == 0:
            break
    if sum(allocation) != total_budget:
        raise RuntimeError(
            f"KronVID budget allocation produced {sum(allocation)} tokens, expected {total_budget}"
        )
    return allocation


def _ridge_leverage_scores(features: torch.Tensor, relative_ridge: float) -> torch.Tensor:
    gram, ridge = _ridge_scale(features.float(), relative_ridge)
    eye = torch.eye(gram.shape[0], dtype=torch.float32, device=features.device)
    system = gram + ridge * eye
    solution = torch.linalg.solve(system, features.transpose(0, 1).float()).transpose(0, 1)
    return torch.nan_to_num((features.float() * solution).sum(dim=1), nan=0.0).clamp_min_(0.0)


def _select_anchors(
    leverage: torch.Tensor,
    frame_ids: torch.Tensor,
    budget: int,
    frame_floor: bool,
) -> torch.Tensor:
    selected: list[int] = []
    selected_set: set[int] = set()
    unique_frames = torch.unique(frame_ids, sorted=True)
    if frame_floor and budget > 0:
        if budget < int(unique_frames.numel()):
            positions = torch.round(
                torch.linspace(0, unique_frames.numel() - 1, steps=budget, device=frame_ids.device)
            ).long().unique()
            unique_frames = unique_frames.index_select(0, positions)
        for frame_id in unique_frames.tolist():
            members = torch.where(frame_ids == int(frame_id))[0]
            if members.numel() == 0:
                continue
            ranked = torch.argsort(leverage[members], descending=True, stable=True)
            token = int(members[ranked[0]].item())
            selected.append(token)
            selected_set.add(token)
            if len(selected) >= budget:
                break

    ranked = torch.argsort(leverage, descending=True, stable=True)
    for token in ranked.tolist():
        token = int(token)
        if token in selected_set:
            continue
        selected.append(token)
        selected_set.add(token)
        if len(selected) >= budget:
            break
    return torch.tensor(sorted(selected), dtype=torch.long, device=leverage.device)


def _offer_edges(
    weights: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    similarity: torch.Tensor,
    position_distance: torch.Tensor,
    feature_temperature: float,
    position_temperature: float,
) -> None:
    if source.numel() == 0:
        return
    edge_weights = torch.exp(
        (similarity.float() - 1.0) / max(1e-4, float(feature_temperature))
        - position_distance.float() / max(1e-4, float(position_temperature))
    ).clamp_(1e-8, 1.0)
    flat_indices = source.long() * int(weights.shape[1]) + target.long()
    weights.view(-1).scatter_reduce_(
        0,
        flat_indices,
        edge_weights,
        reduce="amax",
        include_self=True,
    )


def _topk_edges(
    weights: torch.Tensor,
    scores: torch.Tensor,
    similarities: torch.Tensor,
    distances: torch.Tensor,
    source_indices: torch.Tensor,
    target_indices: torch.Tensor,
    topk: int,
    feature_temperature: float,
    position_temperature: float,
) -> None:
    if topk <= 0 or scores.numel() == 0:
        return
    topk = min(int(topk), int(scores.shape[1]))
    positions = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :topk]
    values = torch.gather(scores, 1, positions)
    valid = torch.isfinite(values)
    if not bool(valid.any()):
        return
    row_ids = torch.arange(scores.shape[0], device=scores.device).unsqueeze(1).expand_as(positions)
    source = source_indices[row_ids[valid]]
    target = target_indices[positions[valid]]
    _offer_edges(
        weights,
        source,
        target,
        similarities[row_ids[valid], positions[valid]],
        distances[row_ids[valid], positions[valid]],
        feature_temperature,
        position_temperature,
    )


def _build_sparse_graph(
    metric_features: torch.Tensor,
    coordinates: torch.Tensor,
    frame_ids: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    config: FlashVidConfig,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    count = int(metric_features.shape[0])
    weights = torch.zeros((count, count), dtype=torch.float32, device=metric_features.device)
    feature_temperature = _cfg_float(config, "kron_feature_temperature", 0.20)
    position_temperature = _cfg_float(config, "kron_position_temperature", 0.50)
    spatial_radius = max(1, _cfg_int(config, "kron_spatial_radius", 1))
    spatial_topk = max(0, _cfg_int(config, "kron_spatial_topk", 4))
    temporal_radius = max(0, _cfg_int(config, "kron_temporal_radius", 1))
    temporal_topk = max(0, _cfg_int(config, "kron_temporal_topk", 2))
    semantic_topk = max(0, _cfg_int(config, "kron_semantic_topk", 2))

    unique_frames = torch.unique(frame_ids, sorted=True)
    for frame_id in unique_frames.tolist():
        members = torch.where(frame_ids == int(frame_id))[0]
        if members.numel() <= 1:
            continue
        similarity = metric_features[members] @ metric_features[members].transpose(0, 1)
        dr = (rows[members].unsqueeze(1) - rows[members].unsqueeze(0)).abs()
        dc = (cols[members].unsqueeze(1) - cols[members].unsqueeze(0)).abs()
        local = (dr <= spatial_radius) & (dc <= spatial_radius) & ((dr + dc) > 0)
        distance = torch.cdist(coordinates[members], coordinates[members], p=2).square()
        scores = similarity.masked_fill(~local, -torch.inf)
        _topk_edges(
            weights,
            scores,
            similarity,
            distance,
            members,
            members,
            spatial_topk,
            feature_temperature,
            position_temperature,
        )

    if temporal_radius > 0 and temporal_topk > 0:
        frame_values = [int(value) for value in unique_frames.tolist()]
        for left_pos, left_frame in enumerate(frame_values):
            left = torch.where(frame_ids == left_frame)[0]
            for right_frame in frame_values[left_pos + 1 :]:
                if right_frame - left_frame > temporal_radius:
                    break
                right = torch.where(frame_ids == right_frame)[0]
                similarity = metric_features[left] @ metric_features[right].transpose(0, 1)
                distance = torch.cdist(coordinates[left], coordinates[right], p=2).square()
                scores = similarity
                _topk_edges(
                    weights,
                    scores,
                    similarity,
                    distance,
                    left,
                    right,
                    temporal_topk,
                    feature_temperature,
                    position_temperature,
                )
                _topk_edges(
                    weights,
                    scores.transpose(0, 1),
                    similarity.transpose(0, 1),
                    distance.transpose(0, 1),
                    right,
                    left,
                    temporal_topk,
                    feature_temperature,
                    position_temperature,
                )

    if semantic_topk > 0 and count > 1:
        similarity = metric_features @ metric_features.transpose(0, 1)
        distance = torch.cdist(coordinates, coordinates, p=2).square()
        scores = similarity.clone()
        scores.fill_diagonal_(-torch.inf)
        ids = torch.arange(count, device=metric_features.device)
        _topk_edges(
            weights,
            scores,
            similarity,
            distance,
            ids,
            ids,
            semantic_topk,
            feature_temperature,
            position_temperature,
        )

    weights = torch.maximum(weights, weights.transpose(0, 1))
    weights.fill_diagonal_(0.0)
    degree = weights.sum(dim=1).clamp_min(1e-6)
    laplacian = torch.diag(degree) - weights
    edge_count = int(torch.count_nonzero(torch.triu(weights, diagonal=1)).item())
    return laplacian, degree, edge_count


def _cholesky_with_jitter(matrix: torch.Tensor) -> Optional[torch.Tensor]:
    eye = torch.eye(matrix.shape[0], dtype=torch.float32, device=matrix.device)
    for jitter in (0.0, 1e-6, 1e-5, 1e-4, 1e-3):
        cholesky, info = torch.linalg.cholesky_ex(matrix.float() + jitter * eye)
        if int(info.max().item()) == 0 and torch.isfinite(cholesky).all():
            return cholesky
    return None


def _nearest_anchor_coordinates(
    metric_features: torch.Tensor,
    anchors: torch.Tensor,
) -> torch.Tensor:
    similarity = metric_features @ metric_features[anchors].transpose(0, 1)
    nearest = torch.argmax(similarity, dim=1)
    output = torch.zeros(
        (metric_features.shape[0], anchors.numel()),
        dtype=torch.float32,
        device=metric_features.device,
    )
    output.scatter_(1, nearest.unsqueeze(1), 1.0)
    output[anchors] = 0.0
    output[anchors, torch.arange(anchors.numel(), device=anchors.device)] = 1.0
    return output


def _harmonic_prolongation(
    laplacian: torch.Tensor,
    metric_features: torch.Tensor,
    anchors: torch.Tensor,
    harmonic_mu: float,
) -> tuple[torch.Tensor, bool]:
    count = int(laplacian.shape[0])
    budget = int(anchors.numel())
    prolongation = torch.zeros((count, budget), dtype=torch.float32, device=laplacian.device)
    prolongation[anchors, torch.arange(budget, device=anchors.device)] = 1.0
    residual_mask = torch.ones(count, dtype=torch.bool, device=laplacian.device)
    residual_mask[anchors] = False
    residual = torch.where(residual_mask)[0]
    if residual.numel() == 0:
        return prolongation, False

    l_rr = laplacian.index_select(0, residual).index_select(1, residual)
    l_rs = laplacian.index_select(0, residual).index_select(1, anchors)
    mean_degree = float(torch.diagonal(laplacian).mean().item())
    regularizer = max(1e-7, float(harmonic_mu) * max(mean_degree, 1e-6))
    system = l_rr + regularizer * torch.eye(
        l_rr.shape[0], dtype=torch.float32, device=laplacian.device
    )
    cholesky = _cholesky_with_jitter(system)
    if cholesky is None:
        return _nearest_anchor_coordinates(metric_features, anchors), True
    harmonic = torch.cholesky_solve(-l_rs.float(), cholesky)
    harmonic = torch.nan_to_num(harmonic, nan=0.0, posinf=0.0, neginf=0.0).clamp_min_(0.0)
    row_sum = harmonic.sum(dim=1, keepdim=True)
    invalid = row_sum.squeeze(1) <= 1e-8
    harmonic = harmonic / row_sum.clamp_min(1e-8)
    if bool(invalid.any()):
        nearest = _nearest_anchor_coordinates(metric_features, anchors)
        harmonic[invalid] = nearest[residual[invalid]]
    prolongation[residual] = harmonic
    return prolongation, False


def _build_segment_plan(
    *,
    source_indices: torch.Tensor,
    anchor_indices: torch.Tensor,
    output_positions: torch.Tensor,
    metric_features: torch.Tensor,
    coordinates: torch.Tensor,
    frame_ids: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    config: FlashVidConfig,
) -> tuple[KronSegmentPlan, dict[str, float]]:
    local_metric = metric_features[source_indices]
    local_coordinates = coordinates[source_indices]
    local_frames = frame_ids[source_indices]
    local_rows = rows[source_indices]
    local_cols = cols[source_indices]
    anchor_lookup = {int(token): position for position, token in enumerate(source_indices.tolist())}
    local_anchors = torch.tensor(
        [anchor_lookup[int(token)] for token in anchor_indices.tolist()],
        dtype=torch.long,
        device=source_indices.device,
    )
    laplacian, degree, edge_count = _build_sparse_graph(
        local_metric,
        local_coordinates,
        local_frames,
        local_rows,
        local_cols,
        config,
    )
    prolongation, harmonic_fallback = _harmonic_prolongation(
        laplacian,
        local_metric,
        local_anchors,
        _cfg_float(config, "kron_harmonic_mu", 0.01),
    )
    reduced_laplacian = prolongation.transpose(0, 1) @ laplacian @ prolongation
    reduced_laplacian = 0.5 * (reduced_laplacian + reduced_laplacian.transpose(0, 1))

    merge_mode = str(getattr(config, "kron_merge_mode", "galerkin")).strip().lower()
    if merge_mode not in {"galerkin", "prune"}:
        raise ValueError(f"unsupported kron_merge_mode={merge_mode!r}")
    pure_pruning = merge_mode == "prune"
    identity_rho = max(0.0, _cfg_float(config, "kron_identity_rho", 4.0))
    node_weights = degree / degree.sum().clamp_min(1e-8) * float(anchor_indices.numel())
    system_cholesky: Optional[torch.Tensor] = None
    galerkin_fallback = False
    if not pure_pruning:
        weighted_p = prolongation * node_weights.unsqueeze(1)
        system = prolongation.transpose(0, 1) @ weighted_p
        system = system + identity_rho * torch.eye(
            anchor_indices.numel(), dtype=torch.float32, device=source_indices.device
        )
        system_cholesky = _cholesky_with_jitter(system)
        if system_cholesky is None:
            pure_pruning = True
            galerkin_fallback = True

    plan = KronSegmentPlan(
        source_indices=source_indices,
        anchor_indices=anchor_indices,
        output_positions=output_positions,
        prolongation=prolongation,
        reduced_laplacian=reduced_laplacian,
        node_weights=node_weights,
        system_cholesky=system_cholesky,
        identity_rho=identity_rho,
        pure_pruning=pure_pruning,
    )
    diagnostics = {
        "edge_count": float(edge_count),
        "mean_degree": float(degree.mean().item()),
        "reduced_laplacian_trace": float(torch.trace(reduced_laplacian).item()),
        "harmonic_fallback": float(harmonic_fallback),
        "galerkin_fallback": float(galerkin_fallback),
    }
    return plan, diagnostics


def apply_kronvid_plan(flat_features: torch.Tensor, plan: KronVidPlan) -> torch.Tensor:
    if flat_features.ndim != 2:
        raise ValueError(f"expected flat features [N, D], got {tuple(flat_features.shape)}")
    if int(flat_features.shape[0]) != int(plan.raw_token_count):
        raise ValueError(
            f"KronVID plan expects {plan.raw_token_count} tokens, got {flat_features.shape[0]}"
        )
    output = torch.empty(
        (plan.anchor_indices.numel(), flat_features.shape[1]),
        dtype=torch.float32,
        device=flat_features.device,
    )
    source = flat_features.float()
    for segment in plan.segments:
        source_indices = segment.source_indices.to(flat_features.device)
        anchor_indices = segment.anchor_indices.to(flat_features.device)
        output_positions = segment.output_positions.to(flat_features.device)
        anchors = source[anchor_indices]
        if segment.pure_pruning or segment.system_cholesky is None:
            compressed = anchors
        else:
            prolongation = segment.prolongation.to(flat_features.device, torch.float32)
            node_weights = segment.node_weights.to(flat_features.device, torch.float32)
            local = source[source_indices]
            rhs = prolongation.transpose(0, 1) @ (local * node_weights.unsqueeze(1))
            rhs = rhs + float(segment.identity_rho) * anchors
            compressed = torch.cholesky_solve(
                rhs,
                segment.system_cholesky.to(flat_features.device, torch.float32),
            )
        output[output_positions] = compressed
    return torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).to(flat_features.dtype)


def compress_kronvid_deepstack(
    deepstack_video_embeds: Sequence[torch.Tensor],
    plan: KronVidPlan,
) -> list[torch.Tensor]:
    compressed: list[torch.Tensor] = []
    for layer_index, features in enumerate(deepstack_video_embeds):
        if features.ndim != 2:
            raise ValueError(
                f"KronVID DeepStack layer {layer_index} must be [N, D], got {tuple(features.shape)}"
            )
        compressed.append(apply_kronvid_plan(features, plan))
    return compressed


def kronvid_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
    question_features: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    del cls_attention, question_features
    if video_features.ndim != 3:
        raise ValueError(f"expected video_features [T, HW, D], got {tuple(video_features.shape)}")
    frame_count, tokens_per_frame, _ = video_features.shape
    total_tokens = int(frame_count * tokens_per_frame)
    budget, budget_diagnostics = _resolve_budget(flashvid_config, total_tokens)
    flat_features = video_features.reshape(total_tokens, -1)
    if budget >= total_tokens:
        indices = torch.arange(total_tokens, dtype=torch.long, device=video_features.device)
        segment = KronSegmentPlan(
            source_indices=indices,
            anchor_indices=indices,
            output_positions=indices,
            prolongation=torch.eye(total_tokens, dtype=torch.float32, device=video_features.device),
            reduced_laplacian=torch.empty(
                (0, 0), dtype=torch.float32, device=video_features.device
            ),
            node_weights=torch.ones(total_tokens, dtype=torch.float32, device=video_features.device),
            system_cholesky=None,
            identity_rho=0.0,
            pure_pruning=True,
        )
        plan = KronVidPlan(indices, (segment,), total_tokens)
        output = flat_features
        diagnostics = {
            **budget_diagnostics,
            "segment_count": 1.0,
            "effective_dimension_sum": float(total_tokens),
            "graph_edges": 0.0,
            "reduced_laplacian_trace": 0.0,
            "harmonic_fallback_count": 0.0,
            "galerkin_fallback_count": 0.0,
        }
    else:
        height, width = _grid_hw(tokens_per_frame, flashvid_config)
        coordinates, frame_ids, rows, cols = _token_coordinates(
            frame_count,
            tokens_per_frame,
            height,
            width,
            video_features.device,
        )
        augmented = _augmented_features(video_features, coordinates, flashvid_config)
        requested_segments = _cfg_int(flashvid_config, "kron_temporal_segments", 8)
        ranges = _segment_ranges(frame_count, min(requested_segments, budget))
        segment_sources: list[torch.Tensor] = []
        effective_dimensions: list[float] = []
        for frame_start, frame_end in ranges:
            token_start = frame_start * tokens_per_frame
            token_end = frame_end * tokens_per_frame
            source_indices = torch.arange(
                token_start, token_end, dtype=torch.long, device=video_features.device
            )
            segment_sources.append(source_indices)
            effective_dimensions.append(
                _effective_dimension(
                    augmented[source_indices],
                    _cfg_float(flashvid_config, "kron_effective_dim_ridge", 0.10),
                )
            )
        allocations = _allocate_budget(
            budget,
            [int(source.numel()) for source in segment_sources],
            effective_dimensions,
            _cfg_float(flashvid_config, "kron_segment_floor_ratio", 0.35),
        )

        segment_anchors: list[torch.Tensor] = []
        for source_indices, segment_budget in zip(segment_sources, allocations):
            leverage = _ridge_leverage_scores(
                augmented[source_indices],
                _cfg_float(flashvid_config, "kron_leverage_ridge", 0.10),
            )
            local_selected = _select_anchors(
                leverage,
                frame_ids[source_indices],
                segment_budget,
                bool(getattr(flashvid_config, "kron_frame_floor", True)),
            )
            segment_anchors.append(source_indices[local_selected])
        anchor_indices = torch.sort(torch.cat(segment_anchors, dim=0)).values
        if int(anchor_indices.numel()) != budget or int(torch.unique(anchor_indices).numel()) != budget:
            raise RuntimeError("KronVID anchor selection violated the exact unique-token budget")
        output_lookup = {int(token): position for position, token in enumerate(anchor_indices.tolist())}

        plans: list[KronSegmentPlan] = []
        graph_edges = 0.0
        degree_sum = 0.0
        reduced_trace = 0.0
        harmonic_fallbacks = 0.0
        galerkin_fallbacks = 0.0
        for source_indices, anchors in zip(segment_sources, segment_anchors):
            output_positions = torch.tensor(
                [output_lookup[int(token)] for token in anchors.tolist()],
                dtype=torch.long,
                device=video_features.device,
            )
            segment_plan, segment_diagnostics = _build_segment_plan(
                source_indices=source_indices,
                anchor_indices=anchors,
                output_positions=output_positions,
                metric_features=augmented,
                coordinates=coordinates,
                frame_ids=frame_ids,
                rows=rows,
                cols=cols,
                config=flashvid_config,
            )
            plans.append(segment_plan)
            graph_edges += segment_diagnostics["edge_count"]
            degree_sum += segment_diagnostics["mean_degree"]
            reduced_trace += segment_diagnostics["reduced_laplacian_trace"]
            harmonic_fallbacks += segment_diagnostics["harmonic_fallback"]
            galerkin_fallbacks += segment_diagnostics["galerkin_fallback"]
        plan = KronVidPlan(anchor_indices, tuple(plans), total_tokens)
        output = apply_kronvid_plan(flat_features, plan)
        anchors = flat_features[anchor_indices].float()
        cosine = F.cosine_similarity(output.float(), anchors, dim=1, eps=1e-6)
        displacement = (output.float() - anchors).norm(dim=1) / anchors.norm(dim=1).clamp_min(1e-6)
        diagnostics = {
            **budget_diagnostics,
            "segment_count": float(len(plans)),
            "effective_dimension_sum": float(sum(effective_dimensions)),
            "effective_dimension_min": float(min(effective_dimensions)),
            "effective_dimension_max": float(max(effective_dimensions)),
            "segment_budget_min": float(min(allocations)),
            "segment_budget_max": float(max(allocations)),
            "graph_edges": float(graph_edges),
            "mean_segment_degree": float(degree_sum / max(1, len(plans))),
            "reduced_laplacian_trace": float(reduced_trace),
            "harmonic_fallback_count": float(harmonic_fallbacks),
            "galerkin_fallback_count": float(galerkin_fallbacks),
            "mean_anchor_cosine": float(cosine.mean().item()),
            "max_relative_displacement": float(displacement.max().item()),
        }

    setattr(flashvid_config, "_kronvid_plan", plan)
    flashvid_config.vision_token_length = int(output.shape[0])
    flashvid_config.visual_token_length = int(output.shape[0])
    flashvid_config.llm_token_length = None
    setattr(flashvid_config, "last_adapter_variant", "kronvid")
    setattr(flashvid_config, "last_adapter_raw_tokens", float(total_tokens))
    setattr(flashvid_config, "last_adapter_output_tokens", float(output.shape[0]))
    setattr(flashvid_config, "last_kron_diagnostics", diagnostics)
    for name, value in diagnostics.items():
        if isinstance(value, (int, float)):
            setattr(flashvid_config, f"last_kron_{name}", float(value))
    if bool(getattr(flashvid_config, "kron_debug", False)):
        print(f"[KronVID] {diagnostics}")
    return output, plan.anchor_indices


__all__ = [
    "KronSegmentPlan",
    "KronVidPlan",
    "apply_kronvid_plan",
    "compress_kronvid_deepstack",
    "kronvid_compression",
]
