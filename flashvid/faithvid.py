from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .certvid import CertVidPlan, _cfg_float, _cfg_int, _grid_hw, _spatial_layout
from .certvid_v3 import certvid_v3_compression
from .configuration_flashvid import FlashVidConfig


@dataclass
class FaithVidPlan(CertVidPlan):
    """CertVID-compatible aggregation plan with explicit attention mass."""

    group_mass: torch.Tensor
    group_variance: torch.Tensor
    attention_log_mass: torch.Tensor
    temporal_centroid: torch.Tensor
    spatial_centroid: torch.Tensor
    phase_radius: torch.Tensor


def _identity_plan(
    video_features: torch.Tensor,
    frame_ids: torch.Tensor,
    spatial_coords: torch.Tensor,
) -> FaithVidPlan:
    total_tokens = int(video_features.shape[0] * video_features.shape[1])
    device = video_features.device
    indices = torch.arange(total_tokens, dtype=torch.long, device=device)
    ones = torch.ones(total_tokens, dtype=torch.float32, device=device)
    zeros = torch.zeros(total_tokens, dtype=torch.float32, device=device)
    return FaithVidPlan(
        anchor_indices=indices,
        assignment_indices=indices.unsqueeze(1),
        assignment_weights=ones.unsqueeze(1),
        source_mass=ones,
        fusion_alpha=zeros,
        raw_token_count=total_tokens,
        group_mass=ones,
        group_variance=zeros,
        attention_log_mass=zeros,
        temporal_centroid=frame_ids.float(),
        spatial_centroid=spatial_coords.float(),
        phase_radius=zeros,
    )


def _weighted_group_sum(
    values: torch.Tensor,
    assignment_indices: torch.Tensor,
    assignment_weights: torch.Tensor,
    group_count: int,
) -> torch.Tensor:
    values = values.float()
    output_shape = (group_count, *values.shape[1:])
    output = torch.zeros(output_shape, dtype=torch.float32, device=values.device)
    for neighbor in range(int(assignment_indices.shape[1])):
        target = assignment_indices[:, neighbor]
        weight = assignment_weights[:, neighbor]
        weighted = values * weight.view(-1, *([1] * (values.ndim - 1)))
        output.index_add_(0, target, weighted)
    return output


def _weighted_assignment_sum(
    values: torch.Tensor,
    assignment_indices: torch.Tensor,
    assignment_weights: torch.Tensor,
    group_count: int,
) -> torch.Tensor:
    """Reduce one scalar per source-assignment edge into anchor groups."""
    if values.shape != assignment_weights.shape:
        raise ValueError(
            f"edge values must match assignment weights, got {tuple(values.shape)} and "
            f"{tuple(assignment_weights.shape)}"
        )
    output = torch.zeros(group_count, dtype=torch.float32, device=values.device)
    for neighbor in range(int(assignment_indices.shape[1])):
        output.index_add_(
            0,
            assignment_indices[:, neighbor],
            values[:, neighbor].float() * assignment_weights[:, neighbor],
        )
    return output


def _phase_constrained_plan(
    *,
    video_features: torch.Tensor,
    selected: torch.Tensor,
    metric_features: torch.Tensor,
    component_ids: torch.Tensor,
    flashvid_config: FlashVidConfig,
) -> tuple[FaithVidPlan, torch.Tensor, dict[str, float]]:
    frame_count, tokens_per_frame, _ = video_features.shape
    total_tokens = int(frame_count * tokens_per_frame)
    budget = int(selected.numel())
    device = video_features.device
    flat_features = video_features.reshape(total_tokens, -1)

    height, width = _grid_hw(tokens_per_frame, flashvid_config)
    coords, _ = _spatial_layout(tokens_per_frame, height, width, 1, device)
    spatial_coords = coords.repeat(frame_count, 1)
    frame_ids = torch.arange(frame_count, device=device).repeat_interleave(tokens_per_frame)

    similarity = metric_features @ metric_features[selected].transpose(0, 1)
    source_frames = frame_ids.float().unsqueeze(1)
    anchor_frames = frame_ids[selected].float().unsqueeze(0)
    temporal_distance = (source_frames - anchor_frames).abs()
    spatial_distance = torch.cdist(spatial_coords.float(), spatial_coords[selected].float(), p=2)

    temporal_radius = max(0, _cfg_int(flashvid_config, "faith_temporal_radius", 1))
    spatial_radius = max(0.0, _cfg_float(flashvid_config, "faith_spatial_radius", 0.75))
    valid = temporal_distance <= float(temporal_radius)
    if spatial_radius > 0.0:
        valid &= spatial_distance <= spatial_radius

    # Extreme budgets may not place an anchor inside every phase ball. In that
    # case choose the minimum phase-distance anchor rather than dropping mass.
    missing = ~valid.any(dim=1)
    if bool(missing.any()):
        phase_cost = temporal_distance / float(max(1, temporal_radius))
        if spatial_radius > 0.0:
            phase_cost = phase_cost + spatial_distance / spatial_radius
        fallback = phase_cost.argmin(dim=1)
        valid[missing, fallback[missing]] = True

    same_component = component_ids.unsqueeze(1) == component_ids[selected].unsqueeze(0)
    score = (
        similarity
        + _cfg_float(flashvid_config, "faith_component_bonus", 0.08) * same_component.float()
        - _cfg_float(flashvid_config, "faith_temporal_penalty", 0.04) * temporal_distance
        - _cfg_float(flashvid_config, "faith_spatial_penalty", 0.04) * spatial_distance
    )
    score = score.masked_fill(~valid, -torch.inf)

    topk = min(max(1, _cfg_int(flashvid_config, "faith_assignment_topk", 2)), budget)
    values, assignment = torch.topk(score, k=topk, dim=1, largest=True)
    finite = torch.isfinite(values)
    safe_values = values.masked_fill(~finite, -1e4)
    temperature = max(1e-4, _cfg_float(flashvid_config, "faith_assignment_temperature", 0.07))
    weights = torch.softmax(safe_values.float() / temperature, dim=1) * finite.float()
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)

    # Selected evidence always keeps an exact identity route.
    anchor_positions = torch.arange(budget, dtype=torch.long, device=device)
    assignment[selected, 0] = anchor_positions
    weights[selected] = 0.0
    weights[selected, 0] = 1.0

    unit_mass = torch.ones(total_tokens, dtype=torch.float32, device=device)
    group_mass = _weighted_group_sum(
        unit_mass,
        assignment,
        weights,
        budget,
    ).clamp_min(1e-6)
    pooled = _weighted_group_sum(
        flat_features,
        assignment,
        weights,
        budget,
    ) / group_mass.unsqueeze(1)

    temporal_centroid = _weighted_group_sum(
        frame_ids.float(), assignment, weights, budget
    ) / group_mass
    spatial_centroid = _weighted_group_sum(
        spatial_coords, assignment, weights, budget
    ) / group_mass.unsqueeze(1)

    anchor_metric = metric_features[selected]
    assigned_anchor_metric = anchor_metric[assignment]
    cosine_dispersion = (
        1.0 - (metric_features.unsqueeze(1) * assigned_anchor_metric).sum(dim=-1)
    ).clamp(0.0, 2.0)
    group_variance = _weighted_assignment_sum(
        cosine_dispersion, assignment, weights, budget
    ) / group_mass

    temporal_delta = frame_ids.float().unsqueeze(1) - temporal_centroid[assignment]
    spatial_delta = spatial_coords.unsqueeze(1) - spatial_centroid[assignment]
    phase_sq = temporal_delta.square() / float(max(1, frame_count - 1) ** 2)
    phase_sq = phase_sq + spatial_delta.square().sum(dim=-1)
    phase_radius = torch.sqrt(
        _weighted_assignment_sum(phase_sq, assignment, weights, budget) / group_mass
    )

    mass_strength = max(0.0, _cfg_float(flashvid_config, "faith_mass_strength", 1.0))
    variance_strength = max(0.0, _cfg_float(flashvid_config, "faith_variance_strength", 0.50))
    max_log_bias = max(0.0, _cfg_float(flashvid_config, "faith_max_log_bias", 20.0))
    attention_log_mass = (
        mass_strength * torch.log(group_mass.clamp_min(1.0))
        + 0.5 * variance_strength * group_variance
    ).clamp(0.0, max_log_bias)

    merge_alpha = min(1.0, max(0.0, _cfg_float(flashvid_config, "faith_merge_alpha", 1.0)))
    fusion_alpha = torch.full((budget,), merge_alpha, dtype=torch.float32, device=device)
    plan = FaithVidPlan(
        anchor_indices=selected,
        assignment_indices=assignment,
        assignment_weights=weights,
        # Unit source mass is deliberate: group_mass is token multiplicity,
        # whereas CertVID source_mass is an importance/demand reweighting.
        source_mass=unit_mass,
        fusion_alpha=fusion_alpha,
        raw_token_count=total_tokens,
        group_mass=group_mass,
        group_variance=group_variance,
        attention_log_mass=attention_log_mass,
        temporal_centroid=temporal_centroid,
        spatial_centroid=spatial_centroid,
        phase_radius=phase_radius,
    )
    output = flat_features[selected] + merge_alpha * (pooled - flat_features[selected])
    diagnostics = {
        "raw_tokens": float(total_tokens),
        "output_tokens": float(budget),
        "mass_sum": float(group_mass.sum().item()),
        "mass_conservation_error": float(abs(group_mass.sum().item() - total_tokens)),
        "mean_group_mass": float(group_mass.mean().item()),
        "max_group_mass": float(group_mass.max().item()),
        "mean_log_mass": float(attention_log_mass.mean().item()),
        "mean_group_variance": float(group_variance.mean().item()),
        "mean_phase_radius": float(phase_radius.mean().item()),
        "phase_fallback_sources": float(missing.sum().item()),
        "faithfulness_bound": float(
            (group_mass * (group_variance + phase_radius)).sum().item() / max(1, total_tokens)
        ),
    }
    return plan, output.to(dtype=flat_features.dtype), diagnostics


def publish_faithvid_runtime(flashvid_config: FlashVidConfig, plan: FaithVidPlan) -> None:
    """Publish lightweight key-mass metadata after the GPU-heavy plan is consumed."""
    group_mass = plan.group_mass.detach()
    variance = plan.group_variance.detach()
    log_mass = plan.attention_log_mass.detach()
    flashvid_config._faithvid_outer_group_mass = group_mass
    flashvid_config._faithvid_outer_variance = variance
    flashvid_config._faithvid_outer_log_mass = log_mass
    flashvid_config._faithvid_inner_group_mass = group_mass
    flashvid_config._faithvid_inner_variance = variance
    flashvid_config._faithvid_inner_log_mass = log_mass
    flashvid_config._faithvid_validated_metadata = None
    flashvid_config.last_faithvid_attention_mass = float(group_mass.sum().item())


def apply_faithvid_position_centroids(
    flashvid_config: FlashVidConfig,
    position_ids: torch.Tensor,
    visual_token_indexes: torch.Tensor,
) -> torch.Tensor:
    """Place Qwen M-RoPE visual anchors at quantized group centroids.

    Qwen rotary embeddings accept integer position IDs, so the continuous
    assignment centroid is rounded to the nearest valid temporal/spatial
    lattice point. LLaVA's one-dimensional sequence positions remain managed
    by its multimodal input builder.
    """
    if str(getattr(flashvid_config, "compression_variant", "")).strip().lower() != "faithvid":
        return position_ids
    plan = getattr(flashvid_config, "_certvid_plan", None)
    if not isinstance(plan, FaithVidPlan):
        raise RuntimeError("FaithVID position centroids require the active FaithVidPlan")
    if position_ids.ndim != 3 or int(position_ids.shape[1]) != 1:
        raise ValueError(
            "FaithVID M-RoPE centroid placement currently requires batch size 1, "
            f"got position_ids={tuple(position_ids.shape)}"
        )

    visual_token_indexes = visual_token_indexes.to(device=position_ids.device, dtype=torch.long)
    if int(visual_token_indexes.numel()) != int(plan.raw_token_count):
        raise ValueError(
            "FaithVID visual position count does not match its compression plan: "
            f"positions={int(visual_token_indexes.numel())}, raw={int(plan.raw_token_count)}"
        )

    raw_positions = position_ids.index_select(-1, visual_token_indexes).float()
    source_positions = raw_positions.permute(2, 0, 1).reshape(plan.raw_token_count, -1)
    group_count = int(plan.anchor_indices.numel())
    centroid = _weighted_group_sum(
        source_positions,
        plan.assignment_indices.to(position_ids.device),
        plan.assignment_weights.to(position_ids.device),
        group_count,
    ) / plan.group_mass.to(position_ids.device).clamp_min(1e-6).unsqueeze(1)
    centroid = centroid.reshape(group_count, position_ids.shape[0], 1).permute(1, 2, 0)
    centroid = centroid.round().to(dtype=position_ids.dtype)

    anchor_positions = visual_token_indexes.index_select(
        0, plan.anchor_indices.to(device=visual_token_indexes.device, dtype=torch.long)
    )
    updated = position_ids.clone()
    updated[:, :, anchor_positions] = centroid
    flashvid_config.last_faithvid_position_mode = "quantized_mrope_centroid"
    flashvid_config.last_faithvid_position_mean_shift = float(
        (centroid.float() - raw_positions.index_select(-1, plan.anchor_indices.to(raw_positions.device)).float())
        .abs()
        .mean()
        .item()
    )
    return updated


def expand_faithvid_frame_newlines(
    flashvid_config: FlashVidConfig,
    keep_visual_indices: torch.Tensor,
    frame_count: int,
    tokens_per_frame: int,
) -> None:
    """Insert neutral mass entries for LLaVA's per-frame newline tokens."""
    if str(getattr(flashvid_config, "compression_variant", "")).lower() != "faithvid":
        return
    mass = getattr(flashvid_config, "_faithvid_outer_group_mass", None)
    variance = getattr(flashvid_config, "_faithvid_outer_variance", None)
    log_mass = getattr(flashvid_config, "_faithvid_outer_log_mass", None)
    if not all(isinstance(item, torch.Tensor) for item in (mass, variance, log_mass)):
        return
    order: list[int] = []
    for frame_idx in range(int(frame_count)):
        members = torch.where(
            (keep_visual_indices >= frame_idx * tokens_per_frame)
            & (keep_visual_indices < (frame_idx + 1) * tokens_per_frame)
        )[0]
        order.extend(int(item) for item in members.tolist())
        order.append(-1)

    def expand(values: torch.Tensor, neutral: float) -> torch.Tensor:
        parts = [
            values[idx : idx + 1] if idx >= 0 else values.new_full((1,), neutral)
            for idx in order
        ]
        return torch.cat(parts, dim=0)

    expanded_mass = expand(mass, 1.0)
    expanded_variance = expand(variance, 0.0)
    expanded_log_mass = expand(log_mass, 0.0)
    flashvid_config._faithvid_outer_group_mass = expanded_mass
    flashvid_config._faithvid_outer_variance = expanded_variance
    flashvid_config._faithvid_outer_log_mass = expanded_log_mass
    flashvid_config._faithvid_inner_group_mass = expanded_mass
    flashvid_config._faithvid_inner_variance = expanded_variance
    flashvid_config._faithvid_inner_log_mass = expanded_log_mass
    flashvid_config._faithvid_validated_metadata = None
    flashvid_config.last_faithvid_attention_mass = float(expanded_mass.sum().item())


def append_faithvid_neutral_tokens(
    flashvid_config: FlashVidConfig,
    count: int,
) -> None:
    """Append synthetic LLaVA tokens with unit mass and zero correction."""
    if str(getattr(flashvid_config, "compression_variant", "")).strip().lower() != "faithvid":
        return
    count = max(0, int(count))
    if count == 0:
        return
    mass = getattr(flashvid_config, "_faithvid_outer_group_mass", None)
    variance = getattr(flashvid_config, "_faithvid_outer_variance", None)
    log_mass = getattr(flashvid_config, "_faithvid_outer_log_mass", None)
    if not all(isinstance(item, torch.Tensor) for item in (mass, variance, log_mass)):
        raise RuntimeError("FaithVID cannot append LLaVA tokens without mass metadata")
    mass = torch.cat((mass, mass.new_ones(count)), dim=0)
    variance = torch.cat((variance, variance.new_zeros(count)), dim=0)
    log_mass = torch.cat((log_mass, log_mass.new_zeros(count)), dim=0)
    flashvid_config._faithvid_outer_group_mass = mass
    flashvid_config._faithvid_outer_variance = variance
    flashvid_config._faithvid_outer_log_mass = log_mass
    flashvid_config._faithvid_inner_group_mass = mass
    flashvid_config._faithvid_inner_variance = variance
    flashvid_config._faithvid_inner_log_mass = log_mass
    flashvid_config._faithvid_validated_metadata = None
    flashvid_config.last_faithvid_attention_mass = float(mass.sum().item())


def pack_faithvid_frame_newlines(
    flashvid_config: FlashVidConfig,
    compressed_tokens: torch.Tensor,
    keep_visual_indices: torch.Tensor,
    frame_count: int,
    tokens_per_frame: int,
    newline_token: torch.Tensor,
) -> torch.Tensor:
    """Pack an irregular FaithVID coreset frame-wise for LLaVA.

    A selected coreset is no longer a rectangular spatial grid, so LLaVA's
    grid-newline reshape is undefined. Frame-wise packing preserves temporal
    order and gives every synthetic newline neutral attention mass.
    """
    pieces: list[torch.Tensor] = []
    newline_token = newline_token.reshape(1, -1).to(
        device=compressed_tokens.device,
        dtype=compressed_tokens.dtype,
    )
    for frame_idx in range(int(frame_count)):
        start = frame_idx * int(tokens_per_frame)
        members = torch.where(
            (keep_visual_indices >= start)
            & (keep_visual_indices < start + int(tokens_per_frame))
        )[0]
        pieces.append(torch.cat((compressed_tokens[members], newline_token), dim=0))
    packed = torch.cat(pieces, dim=0)
    expand_faithvid_frame_newlines(
        flashvid_config,
        keep_visual_indices,
        frame_count,
        tokens_per_frame,
    )
    flashvid_config.vision_token_length = int(packed.shape[0])
    flashvid_config.visual_token_length = int(packed.shape[0])
    flashvid_config.llm_token_length = None
    return packed


def faithvid_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
    question_features: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a fixed-budget, mass-preserving visual coreset without training."""
    if video_features.ndim != 3:
        raise ValueError(f"expected video_features [T, HW, D], got {tuple(video_features.shape)}")
    frame_count, tokens_per_frame, _ = video_features.shape
    total_tokens = int(frame_count * tokens_per_frame)

    analysis: dict[str, object] = {}
    previous_budget_mode = bool(getattr(flashvid_config, "certv3_budget_uses_expansion", True))
    flashvid_config.certv3_budget_uses_expansion = bool(
        getattr(flashvid_config, "faith_budget_uses_expansion", True)
    )
    try:
        _, selected = certvid_v3_compression(
            video_features=video_features,
            cls_attention=cls_attention,
            flashvid_config=flashvid_config,
            question_features=question_features,
            analysis_sink=analysis,
        )
    finally:
        flashvid_config.certv3_budget_uses_expansion = previous_budget_mode
    selected = torch.sort(selected.to(device=video_features.device, dtype=torch.long)).values

    height, width = _grid_hw(tokens_per_frame, flashvid_config)
    coords, _ = _spatial_layout(tokens_per_frame, height, width, 1, video_features.device)
    spatial_coords = coords.repeat(frame_count, 1)
    frame_ids = torch.arange(frame_count, device=video_features.device).repeat_interleave(tokens_per_frame)

    if int(selected.numel()) >= total_tokens:
        plan = _identity_plan(video_features, frame_ids, spatial_coords)
        output = video_features.reshape(total_tokens, -1)
        diagnostics = {
            "raw_tokens": float(total_tokens),
            "output_tokens": float(total_tokens),
            "mass_sum": float(total_tokens),
            "mass_conservation_error": 0.0,
            "mean_group_mass": 1.0,
            "max_group_mass": 1.0,
            "mean_log_mass": 0.0,
            "mean_group_variance": 0.0,
            "mean_phase_radius": 0.0,
            "phase_fallback_sources": 0.0,
            "faithfulness_bound": 0.0,
        }
    else:
        metric_features = analysis.get("metric_flat")
        component_ids = analysis.get("component_ids")
        if not isinstance(metric_features, torch.Tensor) or not isinstance(component_ids, torch.Tensor):
            raise RuntimeError("FaithVID could not obtain the V3 optimization state")
        plan, output, diagnostics = _phase_constrained_plan(
            video_features=video_features,
            selected=selected,
            metric_features=metric_features,
            component_ids=component_ids,
            flashvid_config=flashvid_config,
        )

    setattr(flashvid_config, "_certvid_plan", plan)
    publish_faithvid_runtime(flashvid_config, plan)
    flashvid_config.vision_token_length = int(output.shape[0])
    flashvid_config.visual_token_length = int(output.shape[0])
    flashvid_config.llm_token_length = None
    flashvid_config.last_adapter_variant = "faithvid"
    flashvid_config.last_adapter_raw_tokens = float(total_tokens)
    flashvid_config.last_adapter_output_tokens = float(output.shape[0])
    flashvid_config.last_faithvid_diagnostics = diagnostics
    for name, value in diagnostics.items():
        setattr(flashvid_config, f"last_faithvid_{name}", value)
    return output, plan.anchor_indices
