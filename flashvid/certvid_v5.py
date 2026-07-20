from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from .certvid import CertVidPlan, _cfg_float, _cfg_int, _metric_features, apply_certvid_plan
from .certvid_v3 import certvid_v3_compression
from .configuration_flashvid import FlashVidConfig


def _resolve_budget(
    config: FlashVidConfig,
    total_tokens: int,
) -> tuple[int, dict[str, object]]:
    """Validate the layer-average contract used by the V3 anchor selector."""
    mode = str(getattr(config, "certv5_budget_mode", "layer_average")).strip().lower()
    if mode not in {"layer_average", "outer_only"}:
        raise ValueError(f"unsupported certv5_budget_mode={mode!r}")

    nominal = _cfg_float(config, "retention_ratio", 0.10)
    expansion = _cfg_float(config, "expansion", 1.0)
    pruning_layer = _cfg_int(config, "pruning_layer", 0)
    inner_retention = _cfg_float(config, "llm_retention_ratio", 1.0)
    layers = _cfg_int(config, "certv5_num_hidden_layers", 0)
    tolerance = 1e-4
    if not (0.0 < nominal <= 1.0):
        raise ValueError(f"retention_ratio must be in (0, 1], got {nominal}")

    if mode == "outer_only":
        if abs(expansion - 1.0) > tolerance or abs(inner_retention - 1.0) > tolerance:
            raise ValueError(
                "certv5 outer_only requires expansion=1 and llm_retention_ratio=1; "
                f"got expansion={expansion}, llm_retention_ratio={inner_retention}"
            )
        outer_retention = nominal
        post_inner_retention = nominal
        layer_multiplier = 1.0
        average_retention = nominal
    else:
        if layers <= 1:
            raise ValueError("certv5 layer_average requires certv5_num_hidden_layers")
        if not (0 < pruning_layer < layers):
            raise ValueError(
                f"pruning_layer must satisfy 0 < K < L, got K={pruning_layer}, L={layers}"
            )
        if not (0.0 < inner_retention < 1.0):
            raise ValueError(
                "certv5 layer_average requires 0 < llm_retention_ratio < 1, "
                f"got {inner_retention}"
            )
        if not bool(getattr(config, "certv5_inner_hook_enabled", False)):
            raise ValueError("certv5 layer_average requires an installed inner-pruning hook")
        if nominal * expansion > 1.0 + tolerance:
            raise ValueError(
                f"outer retention R*E must not exceed 1, got {nominal * expansion:.8f}"
            )
        layer_multiplier = expansion * (
            pruning_layer + (layers - pruning_layer) * inner_retention
        ) / float(layers)
        if abs(layer_multiplier - 1.0) > tolerance:
            raise ValueError(
                "certv5 layer_average budget is not aligned: "
                f"E*(K+(L-K)*r)/L={layer_multiplier:.8f}, expected 1 within {tolerance}"
            )
        outer_retention = nominal * expansion
        post_inner_retention = outer_retention * inner_retention
        average_retention = nominal * layer_multiplier

    budget = max(1, int(round(total_tokens * outer_retention)))
    if budget > total_tokens:
        raise ValueError(f"certv5 outer budget {budget} exceeds raw token count {total_tokens}")
    post_inner_tokens = (
        max(1, int(round(budget * inner_retention)))
        if mode == "layer_average"
        else budget
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
        "average_retention": average_retention,
        "average_layer_multiplier": layer_multiplier,
        "expansion": expansion,
        "pruning_layer": pruning_layer,
        "inner_retention": inner_retention,
        "num_hidden_layers": layers,
        "raw_tokens": total_tokens,
        "target_tokens": budget,
        "post_inner_tokens": post_inner_tokens,
        "average_layer_tokens": average_layer_tokens,
    }
    return budget, diagnostics


def _clone_plan(plan: CertVidPlan) -> CertVidPlan:
    return CertVidPlan(
        anchor_indices=plan.anchor_indices.clone(),
        assignment_indices=plan.assignment_indices.clone(),
        assignment_weights=plan.assignment_weights.clone(),
        source_mass=plan.source_mass.clone(),
        fusion_alpha=plan.fusion_alpha.clone(),
        raw_token_count=int(plan.raw_token_count),
    )


def _scatter_load(
    assignment: torch.Tensor,
    weights: torch.Tensor,
    source_mass: torch.Tensor,
    anchor_count: int,
    source_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    load = torch.zeros(anchor_count, dtype=torch.float32, device=assignment.device)
    mass = source_mass.float()
    if source_mask is not None:
        mass = mass * source_mask.float()
    for neighbor in range(assignment.shape[1]):
        contribution = weights[:, neighbor].float() * mass
        load.index_add_(0, assignment[:, neighbor], contribution)
    return load


def _distribution_kl(load: torch.Tensor, target: torch.Tensor) -> float:
    if load.numel() == 0 or float(load.sum().item()) <= 1e-8:
        return 0.0
    p = load.float().clamp_min(1e-8)
    q = target.float().clamp_min(1e-8)
    p = p / p.sum()
    q = q / q.sum()
    return float((p * (p.log() - q.log())).sum().item())


def _coefficient_of_variation(values: torch.Tensor) -> float:
    values = values.float()
    if values.numel() <= 1:
        return 0.0
    mean = values.mean()
    if float(mean.item()) <= 1e-8:
        return 0.0
    return float((values.std(unbiased=False) / mean).item())


def _sparse_transport(
    *,
    costs: torch.Tensor,
    candidate_positions: torch.Tensor,
    valid_edges: torch.Tensor,
    source_transport_mass: torch.Tensor,
    target_mass: torch.Tensor,
    temperature: float,
    capacity_tau: float,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve row-balanced, column-soft entropic transport on sparse edges."""
    temperature = max(1e-4, float(temperature))
    capacity_tau = max(0.0, float(capacity_tau))
    exponent = capacity_tau / (capacity_tau + temperature) if capacity_tau > 0.0 else 0.0
    log_scale = torch.zeros_like(target_mass, dtype=torch.float32)
    safe_costs = torch.nan_to_num(costs.float(), nan=1e4, posinf=1e4, neginf=-1e4)
    coupling = torch.zeros_like(safe_costs)
    load = torch.zeros_like(target_mass, dtype=torch.float32)

    for _ in range(max(1, int(steps))):
        logits = -safe_costs / temperature + log_scale[candidate_positions]
        logits = logits.masked_fill(~valid_edges, -torch.inf)
        row_has_edge = valid_edges.any(dim=1) & (source_transport_mass > 0.0)
        probabilities = torch.zeros_like(logits)
        if bool(row_has_edge.any()):
            probabilities[row_has_edge] = torch.softmax(logits[row_has_edge], dim=1)
        coupling = probabilities * source_transport_mass.unsqueeze(1)
        load.zero_()
        for neighbor in range(candidate_positions.shape[1]):
            load.index_add_(0, candidate_positions[:, neighbor], coupling[:, neighbor])
        if exponent > 0.0:
            # Generalized Sinkhorn column update. The current column load
            # already contains v, so recover K^T u by subtracting log(v).
            log_scale = exponent * (
                log_scale
                + target_mass.clamp_min(1e-8).log()
                - load.clamp_min(1e-8).log()
            )
            log_scale.clamp_(-20.0, 20.0)

    logits = -safe_costs / temperature + log_scale[candidate_positions]
    logits = logits.masked_fill(~valid_edges, -torch.inf)
    row_has_edge = valid_edges.any(dim=1) & (source_transport_mass > 0.0)
    probabilities = torch.zeros_like(logits)
    if bool(row_has_edge.any()):
        probabilities[row_has_edge] = torch.softmax(logits[row_has_edge], dim=1)
    coupling = probabilities * source_transport_mass.unsqueeze(1)
    load.zero_()
    for neighbor in range(candidate_positions.shape[1]):
        load.index_add_(0, candidate_positions[:, neighbor], coupling[:, neighbor])
    return coupling, load


def _pooled_features(flat_features: torch.Tensor, plan: CertVidPlan) -> torch.Tensor:
    budget = int(plan.anchor_indices.numel())
    feature_dim = int(flat_features.shape[-1])
    source = flat_features.float()
    accumulation = torch.zeros((budget, feature_dim), dtype=torch.float32, device=source.device)
    mass = torch.zeros(budget, dtype=torch.float32, device=source.device)
    for neighbor in range(plan.assignment_indices.shape[1]):
        target = plan.assignment_indices[:, neighbor]
        weight = plan.assignment_weights[:, neighbor].float() * plan.source_mass.float()
        accumulation.index_add_(0, target, source * weight.unsqueeze(1))
        mass.index_add_(0, target, weight)
    return accumulation / mass.clamp_min(1e-8).unsqueeze(1)


def _apply_trust_region(
    flat_features: torch.Tensor,
    plan: CertVidPlan,
    baseline_alpha: torch.Tensor,
    max_displacement: float,
    min_cosine: float,
) -> tuple[int, float, float]:
    max_displacement = max(0.0, float(max_displacement))
    min_cosine = min(1.0, max(-1.0, float(min_cosine)))
    anchors = flat_features.float()[plan.anchor_indices]
    pooled = _pooled_features(flat_features, plan)
    delta = pooled - anchors
    alpha = torch.minimum(plan.fusion_alpha.float(), baseline_alpha.float()).clamp_min_(0.0)
    scale = torch.ones_like(alpha)

    relative = delta.norm(dim=1) * alpha / anchors.norm(dim=1).clamp_min(1e-6)
    if max_displacement <= 0.0:
        scale.zero_()
    else:
        scale = torch.minimum(scale, max_displacement / relative.clamp_min(1e-8))
    scale.clamp_(0.0, 1.0)

    low = torch.zeros_like(scale)
    high = scale.clone()
    for _ in range(12):
        middle = (low + high) * 0.5
        candidate = anchors + (alpha * middle).unsqueeze(1) * delta
        cosine = F.cosine_similarity(candidate, anchors, dim=1, eps=1e-6)
        valid = cosine >= min_cosine
        low = torch.where(valid, middle, low)
        high = torch.where(valid, high, middle)
    scale = low
    final_alpha = alpha * scale
    final_alpha[baseline_alpha <= 0.0] = 0.0
    plan.fusion_alpha = final_alpha.to(dtype=plan.fusion_alpha.dtype)

    output = anchors + final_alpha.unsqueeze(1) * delta
    final_relative = (output - anchors).norm(dim=1) / anchors.norm(dim=1).clamp_min(1e-6)
    final_cosine = F.cosine_similarity(output, anchors, dim=1, eps=1e-6)
    clipped = int((final_alpha < baseline_alpha.float() - 1e-7).sum().item())
    return clipped, float(final_relative.max().item()), float(final_cosine.min().item())


def _recover_residual_plan(
    *,
    video_features: torch.Tensor,
    baseline: CertVidPlan,
    config: FlashVidConfig,
) -> tuple[CertVidPlan, dict[str, object]]:
    flat_features = video_features.reshape(-1, video_features.shape[-1])
    total_tokens = int(flat_features.shape[0])
    anchor_count = int(baseline.anchor_indices.numel())
    baseline_alpha = baseline.fusion_alpha.float()
    active_mask = baseline_alpha > 1e-8
    active_positions = torch.nonzero(active_mask, as_tuple=False).flatten()
    locked_positions = torch.nonzero(~active_mask, as_tuple=False).flatten()
    diagnostics: dict[str, object] = {
        "active_anchor_count": int(active_positions.numel()),
        "locked_anchor_count": int(locked_positions.numel()),
        "fallback": False,
        "fallback_reason": "",
    }
    if active_positions.numel() == 0 or anchor_count >= total_tokens:
        diagnostics.update(fallback=True, fallback_reason="no_active_or_residual_anchors")
        return _clone_plan(baseline), diagnostics

    assignment = baseline.assignment_indices.long()
    baseline_weights = baseline.assignment_weights.float()
    source_mass = baseline.source_mass.float()
    target_is_active = active_mask[assignment]
    live_fraction = (baseline_weights * target_is_active.float()).sum(dim=1)
    dead_fraction = (baseline_weights * (~target_is_active).float()).sum(dim=1)

    source_is_anchor = torch.zeros(total_tokens, dtype=torch.bool, device=flat_features.device)
    source_is_anchor[baseline.anchor_indices] = True
    residual_mask = ~source_is_anchor
    dead_mass_before = float((dead_fraction * source_mass * residual_mask.float()).sum().item())

    metric_dim = max(32, _cfg_int(config, "certv3_metric_dim", 96))
    metric = _metric_features(video_features, metric_dim)
    active_global = baseline.anchor_indices[active_positions]
    similarity = metric @ metric[active_global].transpose(0, 1)

    frame_count, tokens_per_frame = int(video_features.shape[0]), int(video_features.shape[1])
    frame_ids = torch.arange(frame_count, device=flat_features.device).repeat_interleave(tokens_per_frame)
    temporal_count = min(frame_count, max(1, _cfg_int(config, "certv3_temporal_bins", 12)))
    temporal_ids = torch.div(
        frame_ids * temporal_count,
        max(1, frame_count),
        rounding_mode="floor",
    ).clamp_max(temporal_count - 1)
    temporal_distance = (
        temporal_ids.unsqueeze(1) - temporal_ids[active_global].unsqueeze(0)
    ).abs()
    temporal_valid = temporal_distance <= 1
    temporal_penalty = max(0.0, _cfg_float(config, "certv5_ot_temporal_penalty", 0.04))
    costs = 1.0 - similarity + temporal_penalty * temporal_distance.float()
    costs = costs.masked_fill(~temporal_valid, torch.inf)

    requested_topk = max(1, _cfg_int(config, "certv5_ot_topk", 4))
    topk = min(requested_topk, int(active_positions.numel()))
    candidate_costs, candidate_local = torch.topk(costs, k=topk, dim=1, largest=False)
    candidate_valid = torch.isfinite(candidate_costs)

    baseline_global = baseline.anchor_indices[assignment]
    baseline_similarity = (metric.unsqueeze(1) * metric[baseline_global]).sum(dim=-1)
    baseline_temporal = (
        temporal_ids.unsqueeze(1) - temporal_ids[baseline_global]
    ).abs().float()
    baseline_costs = 1.0 - baseline_similarity + temporal_penalty * baseline_temporal
    baseline_costs = baseline_costs.masked_fill(baseline_weights <= 0.0, torch.inf)
    baseline_best = baseline_costs.amin(dim=1)
    cost_slack = max(0.0, _cfg_float(config, "certv5_ot_cost_slack", 0.05))
    candidate_valid &= candidate_costs <= baseline_best.unsqueeze(1) + cost_slack

    safe_row = candidate_valid.any(dim=1) & residual_mask
    live_move = min(1.0, max(0.0, _cfg_float(config, "certv5_ot_live_fraction", 0.25)))
    dead_transport_fraction = dead_fraction * safe_row.float()
    live_transport_fraction = live_move * live_fraction * safe_row.float()
    transport_fraction = dead_transport_fraction + live_transport_fraction
    source_transport_mass = source_mass * transport_fraction
    total_transport_mass = source_transport_mass.sum()
    if float(total_transport_mass.item()) <= 1e-8:
        diagnostics.update(
            fallback=True,
            fallback_reason="no_safe_transport_edges",
            dead_mass_before=dead_mass_before,
            dead_mass_after=dead_mass_before,
            rerouted_mass=0.0,
        )
        return _clone_plan(baseline), diagnostics

    fixed_weights = baseline_weights.clone()
    fixed_weights = torch.where(
        target_is_active,
        fixed_weights * (1.0 - live_move * safe_row.float().unsqueeze(1)),
        fixed_weights * (~safe_row).float().unsqueeze(1),
    )
    fixed_active_load = _scatter_load(
        assignment,
        fixed_weights,
        source_mass,
        anchor_count,
        residual_mask,
    )[active_positions]
    baseline_active_load = _scatter_load(
        assignment,
        baseline_weights * target_is_active.float(),
        source_mass,
        anchor_count,
        residual_mask,
    )[active_positions]

    prior_shrink = min(1.0, max(0.0, _cfg_float(config, "certv5_ot_prior_shrink", 0.10)))
    prior = baseline_active_load.clamp_min(1e-8)
    prior = prior / prior.sum()
    prior = (1.0 - prior_shrink) * prior + prior_shrink / float(active_positions.numel())
    desired_total = fixed_active_load.sum() + total_transport_mass
    target_mass = (prior * desired_total - fixed_active_load).clamp_min(1e-8)
    target_mass = target_mass / target_mass.sum() * total_transport_mass

    _, naive_transport_load = _sparse_transport(
        costs=candidate_costs,
        candidate_positions=candidate_local,
        valid_edges=candidate_valid,
        source_transport_mass=source_transport_mass,
        target_mass=target_mass,
        temperature=_cfg_float(config, "certv5_ot_temperature", 0.07),
        capacity_tau=0.0,
        steps=1,
    )
    coupling, transported_load = _sparse_transport(
        costs=candidate_costs,
        candidate_positions=candidate_local,
        valid_edges=candidate_valid,
        source_transport_mass=source_transport_mass,
        target_mass=target_mass,
        temperature=_cfg_float(config, "certv5_ot_temperature", 0.07),
        capacity_tau=_cfg_float(config, "certv5_ot_capacity_tau", 0.10),
        steps=_cfg_int(config, "certv5_ot_steps", 6),
    )
    if not bool(torch.isfinite(coupling).all()):
        diagnostics.update(fallback=True, fallback_reason="non_finite_transport")
        return _clone_plan(baseline), diagnostics
    row_mass_error = (coupling.sum(dim=1) - source_transport_mass).abs().max()
    if float(row_mass_error.item()) > 1e-5:
        diagnostics.update(
            fallback=True,
            fallback_reason=f"source_mass_error:{float(row_mass_error.item()):.3e}",
        )
        return _clone_plan(baseline), diagnostics

    ot_weights = coupling / source_mass.clamp_min(1e-8).unsqueeze(1)
    candidate_positions = active_positions[candidate_local]
    new_assignment = torch.cat([assignment, candidate_positions], dim=1)
    new_weights = torch.cat([fixed_weights, ot_weights], dim=1)
    row_sum = new_weights.sum(dim=1, keepdim=True)
    new_weights = new_weights / row_sum.clamp_min(1e-8)

    anchor_rows = baseline.anchor_indices
    new_weights[anchor_rows] = 0.0
    new_assignment[anchor_rows, 0] = torch.arange(
        anchor_count,
        dtype=torch.long,
        device=flat_features.device,
    )
    new_weights[anchor_rows, 0] = 1.0

    plan = CertVidPlan(
        anchor_indices=baseline.anchor_indices.clone(),
        assignment_indices=new_assignment,
        assignment_weights=new_weights,
        source_mass=baseline.source_mass.clone(),
        fusion_alpha=baseline.fusion_alpha.clone(),
        raw_token_count=int(baseline.raw_token_count),
    )
    clipped, max_relative_displacement, min_output_cosine = _apply_trust_region(
        flat_features,
        plan,
        baseline_alpha,
        _cfg_float(config, "certv5_ot_max_displacement", 0.12),
        _cfg_float(config, "certv5_ot_min_cosine", 0.98),
    )

    after_locked_load = _scatter_load(
        plan.assignment_indices,
        plan.assignment_weights * (~active_mask[plan.assignment_indices]).float(),
        source_mass,
        anchor_count,
        residual_mask,
    )
    dead_mass_after = float(after_locked_load.sum().item())
    final_active_load = _scatter_load(
        plan.assignment_indices,
        plan.assignment_weights * active_mask[plan.assignment_indices].float(),
        source_mass,
        anchor_count,
        residual_mask,
    )[active_positions]
    baseline_distribution = prior * baseline_active_load.sum().clamp_min(1e-8)
    v3_capacity_kl = _distribution_kl(baseline_active_load, baseline_distribution)
    capacity_kl_before = _distribution_kl(fixed_active_load + naive_transport_load, prior)
    capacity_kl_after = _distribution_kl(final_active_load, prior)
    transport_cost = float(
        (coupling * candidate_costs.masked_fill(~candidate_valid, 0.0)).sum().item()
        / max(1e-8, float(total_transport_mass.item()))
    )
    transported_edges = coupling > 1e-8
    max_cost_excess = float(
        (
            candidate_costs - baseline_best.unsqueeze(1)
        ).masked_fill(~transported_edges, -torch.inf).amax().item()
    )
    diagnostics.update(
        dead_mass_before=dead_mass_before,
        dead_mass_after=dead_mass_after,
        rerouted_mass=max(0.0, dead_mass_before - dead_mass_after),
        transported_mass=float(total_transport_mass.item()),
        dead_transport_mass=float((source_mass * dead_transport_fraction).sum().item()),
        live_transport_mass=float((source_mass * live_transport_fraction).sum().item()),
        live_transport_fraction=live_move,
        transported_source_count=int((source_transport_mass > 0.0).sum().item()),
        safe_source_count=int(safe_row.sum().item()),
        transport_cost=transport_cost,
        max_cost_excess=max_cost_excess,
        row_mass_error=float(row_mass_error.item()),
        v3_capacity_kl=v3_capacity_kl,
        capacity_kl_before=capacity_kl_before,
        capacity_kl_after=capacity_kl_after,
        load_cv_before=_coefficient_of_variation(baseline_active_load),
        load_cv_after=_coefficient_of_variation(final_active_load),
        trust_region_clipped_count=clipped,
        max_relative_displacement=max_relative_displacement,
        min_output_anchor_cosine=min_output_cosine,
        active_target_mass=float(transported_load.sum().item()),
    )
    fallback_reason = ""
    if dead_mass_after > dead_mass_before + 1e-5:
        fallback_reason = "dead_mass_increased"
    elif capacity_kl_after > capacity_kl_before + 1e-5:
        fallback_reason = "capacity_kl_increased"
    elif max_cost_excess > cost_slack + 1e-5:
        fallback_reason = "transport_cost_slack_exceeded"
    elif max_relative_displacement > _cfg_float(config, "certv5_ot_max_displacement", 0.12) + 1e-5:
        fallback_reason = "trust_displacement_exceeded"
    elif min_output_cosine < _cfg_float(config, "certv5_ot_min_cosine", 0.98) - 1e-5:
        fallback_reason = "trust_cosine_violated"
    if fallback_reason:
        diagnostics.update(fallback=True, fallback_reason=fallback_reason)
        return _clone_plan(baseline), diagnostics
    return plan, diagnostics


def _publish_diagnostics(
    config: FlashVidConfig,
    diagnostics: dict[str, object],
) -> None:
    setattr(config, "last_certv5_diagnostics", diagnostics)
    budget = diagnostics["budget"]
    transport = diagnostics["transport"]
    scalar_values = {
        "v3_anchor_match": int(bool(diagnostics.get("v3_anchor_match", False))),
        "target_tokens": budget["target_tokens"],
        "nominal_retention": budget["nominal_retention"],
        "outer_retention": budget["outer_retention"],
        "post_inner_retention": budget["post_inner_retention"],
        "average_layer_multiplier": budget["average_layer_multiplier"],
        "post_inner_tokens": budget["post_inner_tokens"],
        "average_layer_tokens": budget["average_layer_tokens"],
        "dead_mass_before": transport.get("dead_mass_before", 0.0),
        "dead_mass_after": transport.get("dead_mass_after", 0.0),
        "rerouted_mass": transport.get("rerouted_mass", 0.0),
        "transported_mass": transport.get("transported_mass", 0.0),
        "transport_cost": transport.get("transport_cost", 0.0),
        "max_cost_excess": transport.get("max_cost_excess", 0.0),
        "v3_capacity_kl": transport.get("v3_capacity_kl", 0.0),
        "capacity_kl_before": transport.get("capacity_kl_before", 0.0),
        "capacity_kl_after": transport.get("capacity_kl_after", 0.0),
        "load_cv_before": transport.get("load_cv_before", 0.0),
        "load_cv_after": transport.get("load_cv_after", 0.0),
        "trust_region_clipped_count": transport.get("trust_region_clipped_count", 0),
        "max_relative_displacement": transport.get("max_relative_displacement", 0.0),
        "min_output_anchor_cosine": transport.get("min_output_anchor_cosine", 1.0),
        "fallback_count": int(bool(transport.get("fallback", False))),
    }
    for name, value in scalar_values.items():
        setattr(config, f"last_certv5_{name}", float(value))
    if bool(getattr(config, "certv5_debug", False)):
        print(f"[CertVID-V5] {diagnostics}")


def certvid_v5_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
    question_features: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reuse V3 anchors exactly and recover residual evidence with conservative OT."""
    if video_features.ndim != 3:
        raise ValueError(f"expected video_features [T, HW, D], got {tuple(video_features.shape)}")
    total_tokens = int(video_features.shape[0] * video_features.shape[1])
    budget, budget_diagnostics = _resolve_budget(flashvid_config, total_tokens)

    mode = str(getattr(flashvid_config, "certv5_budget_mode", "layer_average")).strip().lower()
    previous_v3_budget = bool(getattr(flashvid_config, "certv3_budget_uses_expansion", True))
    flashvid_config.certv3_budget_uses_expansion = mode == "layer_average"
    try:
        v3_output, v3_indices = certvid_v3_compression(
            video_features,
            cls_attention,
            flashvid_config,
            question_features,
        )
    finally:
        flashvid_config.certv3_budget_uses_expansion = previous_v3_budget

    baseline_plan = getattr(flashvid_config, "_certvid_plan", None)
    if not isinstance(baseline_plan, CertVidPlan):
        raise RuntimeError("CertVID V3 did not publish a valid residual plan")
    if int(v3_indices.numel()) != budget:
        raise RuntimeError(
            f"CertVID V3 selected {int(v3_indices.numel())} anchors, expected V5 budget {budget}"
        )

    ot_enabled = bool(getattr(flashvid_config, "certv5_ot_enabled", True))
    if ot_enabled:
        plan, transport_diagnostics = _recover_residual_plan(
            video_features=video_features,
            baseline=baseline_plan,
            config=flashvid_config,
        )
    else:
        plan = baseline_plan
        transport_diagnostics = {
            "fallback": False,
            "fallback_reason": "ot_disabled_exact_v3",
            "dead_mass_before": 0.0,
            "dead_mass_after": 0.0,
            "rerouted_mass": 0.0,
        }

    if ot_enabled and not bool(transport_diagnostics.get("fallback", False)):
        output = apply_certvid_plan(video_features.reshape(total_tokens, -1), plan)
        if not bool(torch.isfinite(output).all()):
            plan = baseline_plan
            output = v3_output
            transport_diagnostics["fallback"] = True
            transport_diagnostics["fallback_reason"] = "non_finite_output"
    else:
        plan = baseline_plan
        output = v3_output

    anchor_match = bool(torch.equal(plan.anchor_indices, v3_indices))
    if not anchor_match:
        raise RuntimeError("CertVID V5 changed the V3 anchor indices")
    if torch.any(plan.fusion_alpha > baseline_plan.fusion_alpha + 1e-7):
        raise RuntimeError("CertVID V5 fusion alpha exceeded the V3 plan")
    if torch.any(plan.fusion_alpha[baseline_plan.fusion_alpha <= 0.0] != 0.0):
        raise RuntimeError("CertVID V5 modified a locked V3 anchor")

    setattr(flashvid_config, "_certvid_plan", plan)
    flashvid_config.vision_token_length = int(output.shape[0])
    flashvid_config.visual_token_length = int(output.shape[0])
    flashvid_config.llm_token_length = None
    setattr(flashvid_config, "last_adapter_variant", "certvid_v5")
    setattr(flashvid_config, "last_adapter_raw_tokens", float(total_tokens))
    setattr(flashvid_config, "last_adapter_output_tokens", float(output.shape[0]))
    diagnostics: dict[str, object] = {
        "algorithm": "v3_anchors_residual_recovery_ot",
        "budget": budget_diagnostics,
        "v3_anchor_match": anchor_match,
        "v3_anchor_count": int(v3_indices.numel()),
        "ot_enabled": ot_enabled,
        "transport": transport_diagnostics,
    }
    _publish_diagnostics(flashvid_config, diagnostics)
    return output, plan.anchor_indices
