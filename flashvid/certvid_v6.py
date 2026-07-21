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
    _identity_plan,
    _swap_refine,
)
from .configuration_flashvid import FlashVidConfig


def _scene_ids(
    video_features: torch.Tensor,
    config: FlashVidConfig,
) -> tuple[torch.Tensor, int, float]:
    """Scene ids per frame from DySeg cuts, plus the continuity coefficient.

    Continuity is the median adjacent-frame cosine similarity: near 1.0 for
    densely sampled clips with real motion, and low for long videos where
    uniform sampling turns neighbouring frames into unrelated scenes.
    """
    from .utils import segment

    frame_count = int(video_features.shape[0])
    frame_means = video_features.mean(dim=1).float()
    if frame_count <= 1:
        ids = torch.zeros(frame_count, dtype=torch.long, device=video_features.device)
        return ids, max(1, frame_count), 1.0

    normed = F.normalize(frame_means, p=2, dim=-1, eps=1e-6)
    transition = torch.sum(normed[:-1] * normed[1:], dim=-1)
    continuity = float(transition.median().clamp(-1.0, 1.0).item())

    if bool(getattr(config, "certv6_scene_temporal", True)):
        lengths = segment(
            video_features=frame_means,
            segment_threshold=_cfg_float(config, "segment_threshold", 0.9),
            min_segment_num=_cfg_int(config, "min_segment_num", 8),
            complementary_segment=bool(getattr(config, "complementary_segment", True)),
        ).to(video_features.device)
        if (
            lengths.ndim != 1
            or lengths.numel() == 0
            or bool((lengths <= 0).any())
            or int(lengths.sum().item()) != frame_count
        ):
            raise RuntimeError(
                "CertVID V6 segmentation produced invalid frame coverage: "
                f"lengths={lengths.tolist()}, frame_count={frame_count}"
            )
        ids = torch.repeat_interleave(
            torch.arange(int(lengths.numel()), dtype=torch.long, device=video_features.device),
            lengths,
        )
        return ids, int(lengths.numel()), continuity

    bins = min(frame_count, max(1, _cfg_int(config, "certv3_temporal_bins", 12)))
    frame_ids = torch.arange(frame_count, dtype=torch.long, device=video_features.device)
    ids = torch.div(frame_ids * bins, max(1, frame_count), rounding_mode="floor").clamp_max(bins - 1)
    return ids, bins, continuity


def _continuity_gate(continuity: float, config: FlashVidConfig) -> float:
    """Map continuity to a [0, 1] trust weight for motion-derived signals."""
    if not bool(getattr(config, "certv6_gate_enabled", True)):
        return 1.0
    low = _cfg_float(config, "certv6_continuity_low", 0.55)
    high = _cfg_float(config, "certv6_continuity_high", 0.80)
    if high <= low:
        return 1.0 if continuity >= high else 0.0
    return min(1.0, max(0.0, (continuity - low) / (high - low)))


def _build_plan_v6(
    *,
    selected: torch.Tensor,
    metric_features: torch.Tensor,
    demand_weight: torch.Tensor,
    attention: torch.Tensor,
    query_score: torch.Tensor,
    scene_ids: torch.Tensor,
    component_ids: torch.Tensor,
    fusion_alpha: float,
    temperature: float,
) -> CertVidPlan:
    """V3 anchor plan with scene-bounded assignment instead of uniform-bin windows."""
    total_tokens = int(metric_features.shape[0])
    budget = int(selected.numel())
    similarity = metric_features @ metric_features[selected].transpose(0, 1)
    same_scene = scene_ids.unsqueeze(1) == scene_ids[selected].unsqueeze(0)
    masked = similarity.masked_fill(~same_scene, -2.0)
    same_component = component_ids.unsqueeze(1) == component_ids[selected].unsqueeze(0)
    masked = masked + 0.08 * same_component.float()

    topk = min(2, budget)
    values, assignment = torch.topk(masked, k=topk, dim=1, largest=True)
    # A scene without anchors leaves every entry masked; fall back to the
    # unrestricted neighbours instead of averaging arbitrary anchors.
    orphan = values[:, 0] <= -1.5
    if bool(orphan.any()):
        fallback_values, fallback_assignment = torch.topk(
            similarity[orphan], k=topk, dim=1, largest=True
        )
        values[orphan] = fallback_values
        assignment[orphan] = fallback_assignment
    weights = torch.softmax(values.float() / max(1e-4, float(temperature)), dim=1)

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
        min(max(float(fusion_alpha), 0.0), 0.75),
        dtype=torch.float32,
        device=selected.device,
    )
    alpha = alpha * (1.0 - 0.65 * protection.clamp(0.0, 1.0))
    alpha[protected] = 0.0
    return CertVidPlan(
        anchor_indices=selected,
        assignment_indices=assignment,
        assignment_weights=weights,
        source_mass=source_mass,
        fusion_alpha=alpha,
        raw_token_count=total_tokens,
    )


def certvid_v6_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
    question_features: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Certified D-optimal coreset with continuity-gated signals and scene structure.

    Keeps the V3 pipeline (certificates -> candidate pool -> D-optimal greedy ->
    Fedorov swaps -> anchored soft assignment) and changes only what breaks on
    sparsely sampled long videos:
    1. motion-derived quality signals are trusted proportionally to measured
       frame-to-frame continuity instead of unconditionally;
    2. temporal structure (certificates, design axes, assignment mask) follows
       DySeg scene boundaries instead of uniform index bins;
    3. the query evidence quota grows as continuity drops, since locating the
       relevant scene dominates long-video questions.
    """
    if video_features.ndim != 3:
        raise ValueError(f"expected video_features [T, HW, D], got {tuple(video_features.shape)}")
    frame_count, tokens_per_frame, _ = video_features.shape
    total_tokens = int(frame_count * tokens_per_frame)
    ratio = _effective_ratio(flashvid_config)
    budget = max(1, min(total_tokens, int(round(total_tokens * ratio))))
    flat_features = video_features.reshape(total_tokens, -1)

    if budget >= total_tokens:
        plan = _identity_plan(total_tokens, video_features.device)
        output = flat_features
        candidates = total_tokens
        components = total_tokens
        mandatory: list[int] = list(range(total_tokens))
        query_seeds: list[int] = []
        query_confidence = 0.0
        swaps = 0
        logdet = 0.0
        scene_count = 1
        continuity = 1.0
        gate = 1.0
    else:
        metric_dim = max(32, _cfg_int(flashvid_config, "certv3_metric_dim", 96))
        metric_flat = _metric_features(video_features, metric_dim)
        metric_frames = metric_flat.view(frame_count, tokens_per_frame, -1)
        height, width = _grid_hw(tokens_per_frame, flashvid_config)
        spatial_bins = max(1, _cfg_int(flashvid_config, "certv3_spatial_bins", 3))
        coords, frame_spatial_ids = _spatial_layout(
            tokens_per_frame,
            height,
            width,
            spatial_bins,
            video_features.device,
        )
        frame_scene_ids, scene_count, continuity = _scene_ids(video_features, flashvid_config)
        gate = _continuity_gate(continuity, flashvid_config)

        frame_event, _, novelty_2d, curvature_2d, matches = _trajectory_signals(
            metric_frames,
            coords,
            _cfg_float(flashvid_config, "certv3_spatial_penalty", 0.08),
        )
        component_ids_cpu, component_sizes_cpu = _build_components(
            frame_count,
            tokens_per_frame,
            frame_event,
            matches,
            _cfg_float(flashvid_config, "certv3_track_threshold", 0.82),
        )
        component_ids = component_ids_cpu.to(video_features.device)
        component_sizes = component_sizes_cpu.to(video_features.device)
        frame_ids = torch.arange(frame_count, device=video_features.device).repeat_interleave(tokens_per_frame)
        component_value = _component_support(
            metric_flat,
            component_ids,
            component_sizes,
            frame_ids,
            frame_count,
        )

        temporal_count = int(scene_count)
        temporal_ids = frame_scene_ids.repeat_interleave(tokens_per_frame)
        spatial_ids = frame_spatial_ids.repeat(frame_count)
        spatial_count = spatial_bins * spatial_bins

        attention = _rank_normalize(cls_attention.float()).reshape(-1)
        novelty = novelty_2d.reshape(-1)
        curvature = curvature_2d.reshape(-1)
        detail = _local_detail(video_features, height, width).reshape(-1)
        event = frame_event.repeat_interleave(tokens_per_frame)
        atoms = _question_atoms(
            question_features,
            max(0, _cfg_int(flashvid_config, "certv3_query_atoms", 8)),
            metric_dim,
        ).to(video_features.device)
        query_relevance, atom_weights, query_confidence = _question_relevance(atoms, metric_flat)
        query_score = (
            (query_relevance * atom_weights.unsqueeze(1)).sum(dim=0)
            if query_relevance.numel() > 0
            else torch.zeros(total_tokens, dtype=torch.float32, device=video_features.device)
        )

        # Motion-derived terms carry 0.60 of the V3 quality mass; scale them by
        # the continuity gate and hand the freed mass to signals that stay valid
        # under sparse sampling (attention, local detail, query relevance).
        freed = 0.60 * (1.0 - gate)
        query_cap = 0.30 + 0.15 * (1.0 - gate)
        query_weight = min(
            query_cap,
            max(0.0, _cfg_float(flashvid_config, "certv3_query_weight", 0.18) * query_confidence)
            + 0.20 * freed * query_confidence,
        )
        visual_quality = _minmax(
            (0.28 + 0.55 * freed) * attention
            + gate * (0.20 * novelty + 0.14 * curvature + 0.12 * event)
            + (0.12 + 0.25 * freed) * detail
            + gate * 0.14 * component_value,
            dim=0,
        )
        quality = _minmax((1.0 - query_weight) * visual_quality + query_weight * query_score, dim=0)
        event_score = _minmax(
            gate * (0.34 * novelty + 0.28 * curvature + 0.18 * event)
            + (1.0 - gate) * 0.80 * (0.50 * attention + 0.25 * detail + 0.25 * query_score)
            + 0.10 * detail
            + 0.10 * query_score,
            dim=0,
        )
        demand_component = gate * component_value + (1.0 - gate) * attention
        demand_weight = 0.20 + 0.42 * quality + 0.20 * event_score + 0.18 * demand_component
        demand_weight = demand_weight / demand_weight.sum().clamp_min(1e-6)

        per_atom_base = max(1, _cfg_int(flashvid_config, "certv3_query_per_atom", 1))
        per_atom_max = max(per_atom_base, _cfg_int(flashvid_config, "certv6_query_per_atom_max", 3))
        query_per_atom = per_atom_base + int(round((per_atom_max - per_atom_base) * (1.0 - gate)))

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
            frame_coverage_ratio=_cfg_float(flashvid_config, "certv3_frame_coverage_ratio", 1.0),
            cell_coverage_ratio=_cfg_float(flashvid_config, "certv3_cell_coverage_ratio", 0.50),
            query_threshold=_cfg_float(flashvid_config, "certv3_query_threshold", 0.10),
            query_per_atom=query_per_atom,
        )
        candidate_indices = _candidate_pool(
            budget=budget,
            quality=quality,
            component_ids=component_ids,
            temporal_ids=temporal_ids,
            spatial_ids=spatial_ids,
            query_relevance=query_relevance,
            mandatory=mandatory,
            multiplier=_cfg_float(flashvid_config, "certv3_candidate_multiplier", 2.5),
        )
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
            structural_weight=_cfg_float(flashvid_config, "certv3_structural_weight", 0.32),
            whitening_strength=_cfg_float(flashvid_config, "certv3_whitening_strength", 0.50),
            quality_floor=_cfg_float(flashvid_config, "certv3_quality_floor", 0.15),
        )
        ridge = _cfg_float(flashvid_config, "certv3_ridge", 0.50)
        selected = _d_optimal_greedy(
            design=design,
            candidates=candidate_indices,
            mandatory=mandatory,
            budget=budget,
            ridge=ridge,
        )
        selected, swaps, logdet = _swap_refine(
            selected=selected,
            candidates=candidate_indices,
            design=design,
            mandatory=mandatory,
            ridge=ridge,
            steps=_cfg_int(flashvid_config, "certv3_swap_steps", 6),
            pool_size=_cfg_int(flashvid_config, "certv3_swap_pool", 24),
            margin=_cfg_float(flashvid_config, "certv3_swap_margin", 1e-4),
        )
        selected = torch.sort(selected).values
        plan = _build_plan_v6(
            selected=selected,
            metric_features=metric_flat,
            demand_weight=demand_weight,
            attention=attention,
            query_score=query_score,
            scene_ids=temporal_ids,
            component_ids=component_ids,
            fusion_alpha=_cfg_float(flashvid_config, "certv3_fusion_alpha", 0.12),
            temperature=_cfg_float(flashvid_config, "certv3_assignment_temperature", 0.07),
        )
        if mandatory:
            certificate_indices = torch.tensor(mandatory, dtype=torch.long, device=selected.device)
            plan.fusion_alpha[torch.isin(selected, certificate_indices)] = 0.0
        output = apply_certvid_plan(flat_features, plan)
        candidates = int(candidate_indices.numel())
        components = int(component_sizes.numel())

    setattr(flashvid_config, "_certvid_plan", plan)
    flashvid_config.vision_token_length = int(output.shape[0])
    flashvid_config.visual_token_length = int(output.shape[0])
    flashvid_config.llm_token_length = None
    setattr(flashvid_config, "last_adapter_variant", "certvid_v6")
    setattr(flashvid_config, "last_adapter_raw_tokens", float(total_tokens))
    setattr(flashvid_config, "last_adapter_output_tokens", float(output.shape[0]))
    setattr(flashvid_config, "last_certv6_target_tokens", float(budget))
    setattr(flashvid_config, "last_certv6_candidate_tokens", float(candidates))
    setattr(flashvid_config, "last_certv6_component_count", float(components))
    setattr(flashvid_config, "last_certv6_certificate_count", float(len(mandatory)))
    setattr(flashvid_config, "last_certv6_query_seed_count", float(len(query_seeds)))
    setattr(flashvid_config, "last_certv6_query_confidence", float(query_confidence))
    setattr(flashvid_config, "last_certv6_swap_count", float(swaps))
    setattr(flashvid_config, "last_certv6_logdet", float(logdet))
    setattr(flashvid_config, "last_certv6_scene_count", float(scene_count))
    setattr(flashvid_config, "last_certv6_continuity", float(continuity))
    setattr(flashvid_config, "last_certv6_gate", float(gate))
    return output, plan.anchor_indices
