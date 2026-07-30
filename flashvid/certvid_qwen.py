from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from .certvid import (
    CertVidPlan,
    _build_components,
    _cfg_float,
    _cfg_int,
    _grid_hw,
    _local_detail,
    _minmax,
    _question_atoms,
    _question_relevance,
    _spatial_layout,
    apply_certvid_plan,
)
from .certvid_v2 import _component_support, _trajectory_signals
from .certvid_v3 import _candidate_pool, _hard_certificates
from .configuration_flashvid import FlashVidConfig


def _effective_ratio(config: FlashVidConfig) -> float:
    ratio = _cfg_float(config, "retention_ratio", 0.10)
    if bool(getattr(config, "qcert_budget_uses_expansion", True)):
        ratio *= _cfg_float(config, "expansion", 1.0)
    return min(1.0, max(0.0, ratio))


def _full_metric_features(
    features: torch.Tensor,
    whitening_strength: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Preserve every hidden channel with diagonal shrinkage whitening."""
    flat = features.reshape(-1, features.shape[-1]).float()
    center = flat.mean(dim=0, keepdim=True)
    centered = flat - center
    strength = min(1.0, max(0.0, float(whitening_strength)))
    variance = centered.square().mean(dim=0, keepdim=True).clamp_min(1e-6)
    scale = variance.pow(-0.5 * strength)
    metric = F.normalize(centered * scale, p=2, dim=-1, eps=1e-6)
    return metric, center, scale


def _tie_safe_attention(values: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    tensor = values.float()
    if tuple(tensor.shape) == () or tensor.shape[-1] <= 1:
        return torch.zeros_like(tensor)
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("Qwen visual attention contains NaN or Inf")
    rows = tensor.reshape(-1, tensor.shape[-1])
    normalized = torch.zeros_like(rows)
    for row_idx, row in enumerate(rows):
        if float((row.max() - row.min()).item()) < eps:
            continue
        if float(row.std(unbiased=False).item()) < eps:
            continue
        _, inverse, counts = torch.unique(
            row,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        counts_float = counts.float()
        starts = torch.cumsum(counts_float, dim=0) - counts_float
        mid_ranks = starts + 0.5 * (counts_float - 1.0)
        normalized[row_idx] = mid_ranks[inverse] / float(row.numel() - 1)
    return normalized.reshape_as(tensor)


def _mrope_coordinates(
    config: FlashVidConfig,
    frame_count: int,
    tokens_per_frame: int,
    height: int,
    width: int,
    device: torch.device,
) -> tuple[torch.Tensor, str]:
    total_tokens = frame_count * tokens_per_frame
    runtime = getattr(config, "_qwen_mrope_coords", None)
    if runtime is not None:
        coords = torch.as_tensor(runtime, device=device).float()
        if coords.shape == (total_tokens, 3) and bool(torch.isfinite(coords).all()):
            return coords, "model_position_ids"

    token_ids = torch.arange(tokens_per_frame, device=device)
    rows = torch.div(token_ids, width, rounding_mode="floor").clamp_max(height - 1)
    cols = torch.remainder(token_ids, width).clamp_max(width - 1)
    frame = torch.arange(frame_count, device=device).repeat_interleave(tokens_per_frame)
    return (
        torch.stack(
            [
                frame.float(),
                rows.repeat(frame_count).float(),
                cols.repeat(frame_count).float(),
            ],
            dim=1,
        ),
        "grid_fallback",
    )


def _normalize_coordinates(coords: torch.Tensor) -> torch.Tensor:
    lo = coords.amin(dim=0, keepdim=True)
    hi = coords.amax(dim=0, keepdim=True)
    return (coords - lo) / (hi - lo).clamp_min(1.0)


def _phase_features(coords: torch.Tensor, levels: int) -> torch.Tensor:
    normalized = _normalize_coordinates(coords)
    parts = [normalized]
    for level in range(max(1, int(levels))):
        angle = normalized * (math.pi * float(2**level))
        parts.extend([torch.sin(angle), torch.cos(angle)])
    return F.normalize(torch.cat(parts, dim=1), p=2, dim=-1, eps=1e-6)


def _dynamic_temporal_bins(frame_count: int, config: FlashVidConfig) -> int:
    configured = _cfg_int(config, "qcert_temporal_bins", 0)
    if configured > 0:
        return min(frame_count, configured)
    return min(frame_count, max(4, int(round(2.0 * math.sqrt(frame_count)))))


def _dynamic_spatial_bins(tokens_per_frame: int, config: FlashVidConfig) -> int:
    configured = _cfg_int(config, "qcert_spatial_bins", 0)
    if configured > 0:
        return configured
    return 4 if tokens_per_frame >= 400 else 3


def _kernel_design_factors(
    *,
    metric_features: torch.Tensor,
    phase_features: torch.Tensor,
    temporal_ids: torch.Tensor,
    spatial_ids: torch.Tensor,
    signals: torch.Tensor,
    query_relevance: torch.Tensor,
    atom_weights: torch.Tensor,
    query_confidence: float,
    quality: torch.Tensor,
    temporal_count: int,
    spatial_count: int,
    config: FlashVidConfig,
) -> torch.Tensor:
    semantic_weight = max(0.0, _cfg_float(config, "qcert_semantic_weight", 0.68))
    phase_weight = max(0.0, _cfg_float(config, "qcert_phase_weight", 0.14))
    temporal_weight = max(0.0, _cfg_float(config, "qcert_temporal_weight", 0.06))
    spatial_weight = max(0.0, _cfg_float(config, "qcert_spatial_weight", 0.04))
    signal_weight = max(0.0, _cfg_float(config, "qcert_signal_weight", 0.04))
    query_weight = max(0.0, _cfg_float(config, "qcert_design_query_weight", 0.04))
    query_weight *= min(1.0, max(0.0, float(query_confidence)))

    temporal = F.one_hot(temporal_ids, num_classes=temporal_count).float()
    spatial = F.one_hot(spatial_ids, num_classes=spatial_count).float()
    parts = [
        metric_features * math.sqrt(semantic_weight),
        phase_features * math.sqrt(phase_weight),
        temporal * math.sqrt(temporal_weight),
        spatial * math.sqrt(spatial_weight),
        F.normalize(signals.float(), p=2, dim=-1, eps=1e-6)
        * math.sqrt(signal_weight),
    ]
    if query_relevance.numel() > 0 and query_weight > 0.0:
        query_axes = query_relevance.transpose(0, 1) * torch.sqrt(
            atom_weights.clamp_min(1e-6)
        ).unsqueeze(0)
        parts.append(query_axes * math.sqrt(query_weight))

    factors = F.normalize(torch.cat(parts, dim=1), p=2, dim=-1, eps=1e-6)
    quality_floor = min(
        1.0,
        max(1e-4, _cfg_float(config, "qcert_quality_floor", 0.15)),
    )
    row_mass = quality_floor + (1.0 - quality_floor) * quality.clamp(0.0, 1.0)
    return factors * torch.sqrt(row_mass).unsqueeze(1)


def _kernel_d_optimal_greedy(
    *,
    factors: torch.Tensor,
    candidates: torch.Tensor,
    mandatory: list[int],
    quality: torch.Tensor,
    budget: int,
    ridge: float,
    tolerance: float,
    max_kernel_pivots: int,
) -> tuple[torch.Tensor, int, float]:
    """Exact full-feature D-optimal pivots using the dual Gram identity."""
    rows = factors[candidates].float()
    candidate_count = int(rows.shape[0])
    if candidate_count < budget:
        raise RuntimeError(
            f"kernel D-optimal pool has {candidate_count} candidates for budget {budget}"
        )

    ridge = max(1e-4, float(ridge))
    token_to_column = {
        int(token): column
        for column, token in enumerate(candidates.detach().cpu().tolist())
    }
    mandatory_columns = [
        column
        for token in mandatory
        if (column := token_to_column.get(int(token))) is not None
    ]
    configured_pivots = int(max_kernel_pivots)
    pivot_limit = min(
        budget,
        candidate_count,
        max(
            len(mandatory_columns),
            budget if configured_pivots <= 0 else configured_pivots,
        ),
    )
    projections = torch.zeros(
        (candidate_count, pivot_limit),
        dtype=torch.float32,
        device=rows.device,
    )
    gram = (rows @ rows.transpose(0, 1)) / ridge
    residual = 1.0 + torch.diagonal(gram)
    active = torch.ones(candidate_count, dtype=torch.bool, device=rows.device)
    selected_columns: list[int] = []
    pivot_logdet = 0.0

    def add(column: int) -> bool:
        nonlocal pivot_logdet
        if not bool(active[column]):
            return False
        step = len(selected_columns)
        if step >= pivot_limit:
            return False
        kernel_column = gram[:, column].clone()
        kernel_column[column] += 1.0
        if step:
            kernel_column = kernel_column - (
                projections[:, :step] @ projections[column, :step]
            )
        pivot = kernel_column[column].clamp_min(1e-6)
        projections[:, step] = kernel_column / torch.sqrt(pivot)
        residual.sub_(projections[:, step].square()).clamp_(min=0.0)
        active[column] = False
        residual[column] = float("-inf")
        selected_columns.append(column)
        pivot_logdet += math.log(max(1e-6, float(pivot.item())))
        return True

    for token in mandatory:
        column = token_to_column.get(int(token))
        if column is not None and len(selected_columns) < budget:
            add(column)

    kernel_pivots = len(selected_columns)
    threshold = 1.0 + max(0.0, float(tolerance))
    while len(selected_columns) < min(budget, pivot_limit):
        column = int(torch.argmax(residual).item())
        best = float(residual[column].item())
        if not math.isfinite(best) or best <= threshold:
            break
        if not add(column):
            break
        kernel_pivots += 1

    if len(selected_columns) < budget:
        remaining = torch.where(active)[0]
        remaining_quality = quality[candidates[remaining]]
        order = torch.argsort(remaining_quality, descending=True, stable=True)
        needed = budget - len(selected_columns)
        selected_columns.extend(remaining[order[:needed]].detach().cpu().tolist())

    selected = candidates[
        torch.tensor(selected_columns[:budget], dtype=torch.long, device=candidates.device)
    ]
    selected = torch.sort(selected).values
    if int(selected.numel()) != budget:
        raise RuntimeError(
            f"kernel D-optimal selected {int(selected.numel())} tokens for budget {budget}"
        )
    if int(torch.unique(selected).numel()) != budget:
        raise RuntimeError("kernel D-optimal returned duplicate token indices")
    return selected, kernel_pivots, pivot_logdet


def _coordinate_step(coords: torch.Tensor, dimension: int) -> torch.Tensor:
    values = torch.unique(coords[:, dimension]).sort().values
    if values.numel() <= 1:
        return coords.new_tensor(1.0)
    gaps = torch.diff(values)
    gaps = gaps[gaps > 0]
    return gaps.median().clamp_min(1.0) if gaps.numel() else coords.new_tensor(1.0)


def _build_qwen_plan(
    *,
    selected: torch.Tensor,
    metric_features: torch.Tensor,
    demand_weight: torch.Tensor,
    attention: torch.Tensor,
    query_score: torch.Tensor,
    component_ids: torch.Tensor,
    mrope_coords: torch.Tensor,
    mandatory: list[int],
    config: FlashVidConfig,
) -> tuple[CertVidPlan, int, float]:
    total_tokens = int(metric_features.shape[0])
    budget = int(selected.numel())
    similarity = metric_features @ metric_features[selected].transpose(0, 1)

    time_step = _coordinate_step(mrope_coords, 0)
    height_step = _coordinate_step(mrope_coords, 1)
    width_step = _coordinate_step(mrope_coords, 2)
    delta = (mrope_coords.unsqueeze(1) - mrope_coords[selected].unsqueeze(0)).abs()
    time_units = delta[..., 0] / time_step
    height_units = delta[..., 1] / height_step
    width_units = delta[..., 2] / width_step
    spatial_units = torch.sqrt(height_units.square() + width_units.square())

    temporal_radius = max(
        0.0,
        _cfg_float(config, "qcert_fusion_temporal_radius", 1.0),
    )
    spatial_radius = max(
        0.0,
        _cfg_float(config, "qcert_fusion_spatial_radius", 2.0),
    )
    threshold = _cfg_float(config, "qcert_fusion_similarity", 0.82)
    local = (time_units <= temporal_radius) & (spatial_units <= spatial_radius)
    same_component = component_ids.unsqueeze(1) == component_ids[selected].unsqueeze(0)
    valid = local & ((similarity >= threshold) | same_component)
    scored_similarity = similarity + 0.08 * same_component.float()
    scored_similarity = scored_similarity.masked_fill(~valid, -1e4)

    topk = min(2, budget)
    values, assignment = torch.topk(scored_similarity, k=topk, dim=1, largest=True)
    valid_rows = valid.any(dim=1)
    weights = torch.zeros_like(values, dtype=torch.float32)
    if bool(valid_rows.any()):
        weights[valid_rows] = torch.softmax(
            values[valid_rows].float()
            / max(
                1e-4,
                _cfg_float(config, "qcert_assignment_temperature", 0.07),
            ),
            dim=1,
        )

    anchor_positions = torch.arange(budget, dtype=torch.long, device=selected.device)
    assignment[selected, 0] = anchor_positions
    weights[selected] = 0.0
    weights[selected, 0] = 1.0

    source_mass = (0.5 + 0.5 * demand_weight * float(total_tokens)).clamp(0.25, 2.0)
    protection = torch.maximum(attention[selected], query_score[selected])
    protected_count = min(budget, max(1, int(math.ceil(0.15 * budget))))
    protected = torch.zeros(budget, dtype=torch.bool, device=selected.device)
    protected[torch.topk(protection, k=protected_count, largest=True).indices] = True

    alpha = torch.full(
        (budget,),
        min(
            0.75,
            max(0.0, _cfg_float(config, "qcert_fusion_alpha", 0.08)),
        ),
        dtype=torch.float32,
        device=selected.device,
    )
    alpha *= 1.0 - 0.65 * protection.clamp(0.0, 1.0)
    alpha[protected] = 0.0
    if mandatory:
        mandatory_tensor = torch.tensor(
            mandatory,
            dtype=torch.long,
            device=selected.device,
        )
        alpha[torch.isin(selected, mandatory_tensor)] = 0.0

    assigned_similarity = torch.zeros(budget, dtype=torch.float32, device=selected.device)
    assigned_mass = torch.zeros_like(assigned_similarity)
    for neighbor in range(topk):
        target = assignment[:, neighbor]
        weight = weights[:, neighbor]
        assigned_similarity.scatter_add_(
            0,
            target,
            similarity.gather(1, target.unsqueeze(1)).squeeze(1) * weight,
        )
        assigned_mass.scatter_add_(0, target, weight)
    confidence = assigned_similarity / assigned_mass.clamp_min(1e-6)
    confidence = ((confidence - threshold) / max(1e-6, 1.0 - threshold)).clamp(0.0, 1.0)
    alpha *= confidence

    rejected = int((~valid_rows).sum().item())
    rejected_mass = float(source_mass[~valid_rows].sum().item())
    return (
        CertVidPlan(
            anchor_indices=selected,
            assignment_indices=assignment,
            assignment_weights=weights,
            source_mass=source_mass,
            fusion_alpha=alpha,
            raw_token_count=total_tokens,
        ),
        rejected,
        rejected_mass,
    )


def certvid_qwen_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
    question_features: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Qwen-aware full-feature kernel D-optimal visual compression."""
    if video_features.ndim != 3:
        raise ValueError(
            f"expected video_features [T, HW, D], got {tuple(video_features.shape)}"
        )
    frame_count, tokens_per_frame, _ = video_features.shape
    total_tokens = int(frame_count * tokens_per_frame)
    expected_attention_shape = (frame_count, tokens_per_frame)
    if tuple(cls_attention.shape) != expected_attention_shape:
        raise ValueError(
            "Qwen CertVID attention shape mismatch: "
            f"expected {expected_attention_shape}, got {tuple(cls_attention.shape)}"
        )
    if not bool(torch.isfinite(cls_attention).all()):
        raise ValueError("Qwen CertVID attention contains NaN or Inf")
    budget = max(
        1,
        min(total_tokens, int(round(total_tokens * _effective_ratio(flashvid_config)))),
    )
    flat_features = video_features.reshape(total_tokens, -1)
    if budget >= total_tokens:
        selected = torch.arange(total_tokens, dtype=torch.long, device=video_features.device)
        plan = CertVidPlan(
            anchor_indices=selected,
            assignment_indices=selected.unsqueeze(1),
            assignment_weights=torch.ones(
                (total_tokens, 1),
                dtype=torch.float32,
                device=video_features.device,
            ),
            source_mass=torch.ones(total_tokens, dtype=torch.float32, device=video_features.device),
            fusion_alpha=torch.zeros(total_tokens, dtype=torch.float32, device=video_features.device),
            raw_token_count=total_tokens,
        )
        setattr(flashvid_config, "_certvid_plan", plan)
        setattr(flashvid_config, "last_adapter_variant", "certvid_qwen")
        setattr(flashvid_config, "last_adapter_raw_tokens", float(total_tokens))
        setattr(flashvid_config, "last_adapter_output_tokens", float(total_tokens))
        setattr(flashvid_config, "last_qcert_target_tokens", float(total_tokens))
        setattr(flashvid_config, "last_qcert_coordinate_source", "identity")
        return flat_features, selected

    whitening = _cfg_float(flashvid_config, "qcert_whitening_strength", 0.25)
    metric_flat, center, scale = _full_metric_features(video_features, whitening)
    metric_frames = metric_flat.view(frame_count, tokens_per_frame, -1)
    height, width = _grid_hw(tokens_per_frame, flashvid_config)
    spatial_bins = _dynamic_spatial_bins(tokens_per_frame, flashvid_config)
    coords_2d, frame_spatial_ids = _spatial_layout(
        tokens_per_frame,
        height,
        width,
        spatial_bins,
        video_features.device,
    )
    frame_event, _, novelty_2d, curvature_2d, matches = _trajectory_signals(
        metric_frames,
        coords_2d,
        _cfg_float(flashvid_config, "qcert_track_spatial_penalty", 0.08),
    )
    component_ids_cpu, component_sizes_cpu = _build_components(
        frame_count,
        tokens_per_frame,
        frame_event,
        matches,
        _cfg_float(flashvid_config, "qcert_track_threshold", 0.82),
    )
    component_ids = component_ids_cpu.to(video_features.device)
    component_sizes = component_sizes_cpu.to(video_features.device)
    frame_ids = torch.arange(
        frame_count,
        device=video_features.device,
    ).repeat_interleave(tokens_per_frame)
    component_value = _component_support(
        metric_flat,
        component_ids,
        component_sizes,
        frame_ids,
        frame_count,
    )

    temporal_count = _dynamic_temporal_bins(frame_count, flashvid_config)
    temporal_ids = torch.div(
        frame_ids * temporal_count,
        max(1, frame_count),
        rounding_mode="floor",
    ).clamp_max(temporal_count - 1)
    spatial_ids = frame_spatial_ids.repeat(frame_count)
    spatial_count = spatial_bins * spatial_bins
    mrope_coords, coordinate_source = _mrope_coordinates(
        flashvid_config,
        frame_count,
        tokens_per_frame,
        height,
        width,
        video_features.device,
    )
    phase = _phase_features(
        mrope_coords,
        _cfg_int(flashvid_config, "qcert_phase_levels", 4),
    )

    attention = _tie_safe_attention(cls_attention.float()).reshape(-1)
    novelty = novelty_2d.reshape(-1)
    curvature = curvature_2d.reshape(-1)
    detail = _local_detail(video_features, height, width).reshape(-1)
    event = frame_event.repeat_interleave(tokens_per_frame)

    transformed_question = None
    if question_features is not None and question_features.numel() > 0:
        if question_features.shape[-1] != flat_features.shape[-1]:
            raise ValueError(
                "Qwen CertVID requires question and visual features in the same hidden space"
            )
        transformed_question = F.normalize(
            (question_features.float() - center) * scale,
            p=2,
            dim=-1,
            eps=1e-6,
        )
    atoms = _question_atoms(
        transformed_question,
        max(0, _cfg_int(flashvid_config, "qcert_query_atoms", 8)),
        int(metric_flat.shape[-1]),
    ).to(video_features.device)
    query_relevance, atom_weights, query_confidence = _question_relevance(
        atoms,
        metric_flat,
    )
    query_score = (
        (query_relevance * atom_weights.unsqueeze(1)).sum(dim=0)
        if query_relevance.numel() > 0
        else torch.zeros(total_tokens, dtype=torch.float32, device=video_features.device)
    )

    query_weight = min(
        0.30,
        max(
            0.0,
            _cfg_float(flashvid_config, "qcert_quality_query_weight", 0.12)
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
    demand_weight = 0.20 + 0.42 * quality + 0.20 * event_score + 0.18 * component_value
    demand_weight = demand_weight / demand_weight.sum().clamp_min(1e-6)

    mandatory, query_seeds = _hard_certificates(
        budget=budget,
        quality=quality,
        event_score=event_score,
        frame_ids=frame_ids,
        temporal_ids=temporal_ids,
        spatial_ids=spatial_ids,
        query_relevance=query_relevance,
        atom_weights=atom_weights,
        query_confidence=query_confidence,
        frame_count=frame_count,
        temporal_count=temporal_count,
        spatial_count=spatial_count,
        frame_coverage_ratio=_cfg_float(
            flashvid_config,
            "qcert_frame_coverage_ratio",
            1.0,
        ),
        cell_coverage_ratio=_cfg_float(
            flashvid_config,
            "qcert_cell_coverage_ratio",
            0.35,
        ),
        query_threshold=_cfg_float(
            flashvid_config,
            "qcert_query_threshold",
            0.10,
        ),
        query_per_atom=_cfg_int(flashvid_config, "qcert_query_per_atom", 1),
    )
    candidates = _candidate_pool(
        budget=budget,
        quality=quality,
        component_ids=component_ids,
        temporal_ids=temporal_ids,
        spatial_ids=spatial_ids,
        query_relevance=query_relevance,
        mandatory=mandatory,
        multiplier=_cfg_float(
            flashvid_config,
            "qcert_candidate_multiplier",
            2.5,
        ),
    )
    signals = torch.stack(
        [attention, novelty, curvature, event, detail, component_value],
        dim=1,
    )
    factors = _kernel_design_factors(
        metric_features=metric_flat,
        phase_features=phase,
        temporal_ids=temporal_ids,
        spatial_ids=spatial_ids,
        signals=signals,
        query_relevance=query_relevance,
        atom_weights=atom_weights,
        query_confidence=query_confidence,
        quality=quality,
        temporal_count=temporal_count,
        spatial_count=spatial_count,
        config=flashvid_config,
    )
    selected, kernel_pivots, logdet = _kernel_d_optimal_greedy(
        factors=factors,
        candidates=candidates,
        mandatory=mandatory,
        quality=quality,
        budget=budget,
        ridge=_cfg_float(flashvid_config, "qcert_ridge", 0.50),
        tolerance=_cfg_float(
            flashvid_config,
            "qcert_kernel_tolerance",
            1e-4,
        ),
        max_kernel_pivots=_cfg_int(
            flashvid_config,
            "qcert_max_kernel_pivots",
            1024,
        ),
    )
    plan, rejected, rejected_mass = _build_qwen_plan(
        selected=selected,
        metric_features=metric_flat,
        demand_weight=demand_weight,
        attention=attention,
        query_score=query_score,
        component_ids=component_ids,
        mrope_coords=mrope_coords,
        mandatory=mandatory,
        config=flashvid_config,
    )
    output = apply_certvid_plan(flat_features, plan)
    if int(output.shape[0]) != budget or int(selected.numel()) != budget:
        raise RuntimeError(
            "Qwen CertVID budget invariant failed: "
            f"output={int(output.shape[0])}, indices={int(selected.numel())}, budget={budget}"
        )
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("Qwen CertVID produced NaN or Inf output features")
    if selected.numel() > 1 and not bool((selected[1:] > selected[:-1]).all()):
        raise RuntimeError("Qwen CertVID indices must be unique and strictly increasing")
    if not torch.equal(plan.anchor_indices, selected):
        raise RuntimeError("Qwen CertVID plan anchors are not aligned with output indices")
    setattr(flashvid_config, "_certvid_plan", plan)
    setattr(flashvid_config, "last_adapter_variant", "certvid_qwen")
    setattr(flashvid_config, "last_adapter_raw_tokens", float(total_tokens))
    setattr(flashvid_config, "last_adapter_output_tokens", float(output.shape[0]))
    setattr(flashvid_config, "last_qcert_target_tokens", float(budget))
    setattr(flashvid_config, "last_qcert_candidate_tokens", float(candidates.numel()))
    setattr(flashvid_config, "last_qcert_kernel_pivots", float(kernel_pivots))
    setattr(flashvid_config, "last_qcert_kernel_logdet", float(logdet))
    setattr(flashvid_config, "last_qcert_component_count", float(component_sizes.numel()))
    setattr(flashvid_config, "last_qcert_certificate_count", float(len(mandatory)))
    setattr(flashvid_config, "last_qcert_query_seed_count", float(len(query_seeds)))
    setattr(flashvid_config, "last_qcert_query_confidence", float(query_confidence))
    setattr(flashvid_config, "last_qcert_rejected_residuals", float(rejected))
    setattr(flashvid_config, "last_qcert_rejected_mass", float(rejected_mass))
    setattr(flashvid_config, "last_qcert_coordinate_source", coordinate_source)
    return output, selected
