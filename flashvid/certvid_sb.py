from __future__ import annotations

import math
from typing import Any, Iterable, Optional

import torch

from .certvid import (
    CertVidPlan,
    _build_plan,
    _cfg_float,
    _cfg_int,
    _grid_hw,
    _local_detail,
    _minmax,
    apply_certvid_plan,
)
from .certvid_v3 import certvid_v3_compression
from .configuration_flashvid import FlashVidConfig


def _schedule_shares(retention_ratio: float) -> tuple[float, float, float]:
    """Interpolate semantic, temporal, and coverage shares across budgets."""
    points = (
        (0.10, (0.55, 0.25, 0.20)),
        (0.15, (0.60, 0.25, 0.15)),
        (0.20, (0.65, 0.20, 0.15)),
        (0.25, (0.70, 0.15, 0.15)),
    )
    ratio = float(retention_ratio)
    if ratio <= points[0][0]:
        return points[0][1]
    if ratio >= points[-1][0]:
        return points[-1][1]
    for (left_x, left), (right_x, right) in zip(points[:-1], points[1:]):
        if left_x <= ratio <= right_x:
            weight = (ratio - left_x) / max(1e-8, right_x - left_x)
            return tuple(
                (1.0 - weight) * left_value + weight * right_value
                for left_value, right_value in zip(left, right)
            )
    return points[-1][1]


def _configured_shares(config: FlashVidConfig) -> tuple[float, float, float]:
    explicit = (
        _cfg_float(config, "certsb_semantic_ratio", -1.0),
        _cfg_float(config, "certsb_temporal_ratio", -1.0),
        _cfg_float(config, "certsb_coverage_ratio", -1.0),
    )
    if all(value >= 0.0 for value in explicit) and sum(explicit) > 1e-8:
        total = sum(explicit)
        return tuple(value / total for value in explicit)
    return _schedule_shares(_cfg_float(config, "retention_ratio", 0.10))


def _largest_remainder(total: int, weights: Iterable[float]) -> list[int]:
    values = [max(0.0, float(value)) for value in weights]
    denominator = sum(values)
    if denominator <= 1e-8:
        values = [1.0] * len(values)
        denominator = float(len(values))
    raw = [total * value / denominator for value in values]
    counts = [int(math.floor(value)) for value in raw]
    remainder = total - sum(counts)
    order = sorted(range(len(raw)), key=lambda idx: (-(raw[idx] - counts[idx]), idx))
    for idx in order[:remainder]:
        counts[idx] += 1
    return counts


def _bank_quotas(
    budget: int,
    shares: tuple[float, float, float],
    locked_count: int,
) -> tuple[int, int, int]:
    semantic, temporal, coverage = _largest_remainder(budget, shares)
    if locked_count <= semantic:
        return semantic, temporal, coverage
    semantic = min(budget, locked_count)
    remaining = budget - semantic
    temporal, coverage = _largest_remainder(remaining, shares[1:])
    return semantic, temporal, coverage


def _local_pair_change(
    current: torch.Tensor,
    reference: torch.Tensor,
    height: int,
    width: int,
    radius: int,
) -> torch.Tensor:
    """Measure change against the best nearby token in an adjacent frame."""
    pair_count, tokens_per_frame, feature_dim = current.shape
    if height * width != tokens_per_frame or radius <= 0:
        return (1.0 - torch.sum(current * reference, dim=-1)).clamp(0.0, 2.0) * 0.5

    current_grid = current.view(pair_count, height, width, feature_dim)
    reference_grid = reference.view(pair_count, height, width, feature_dim)
    rows = torch.arange(height, device=current.device).view(height, 1)
    cols = torch.arange(width, device=current.device).view(1, width)
    best = torch.full(
        (pair_count, height, width),
        -1.0,
        dtype=torch.float32,
        device=current.device,
    )
    for row_shift in range(-radius, radius + 1):
        for col_shift in range(-radius, radius + 1):
            shifted = torch.roll(
                reference_grid,
                shifts=(row_shift, col_shift),
                dims=(1, 2),
            )
            source_rows = rows - row_shift
            source_cols = cols - col_shift
            valid = (
                (source_rows >= 0)
                & (source_rows < height)
                & (source_cols >= 0)
                & (source_cols < width)
            )
            similarity = torch.sum(current_grid * shifted, dim=-1)
            similarity = similarity.masked_fill(~valid.unsqueeze(0), -1.0)
            best = torch.maximum(best, similarity)
    return ((1.0 - best).clamp(0.0, 2.0) * 0.5).view(pair_count, tokens_per_frame)


def _temporal_change_score(
    *,
    metric_frames: torch.Tensor,
    detail: torch.Tensor,
    query_score: torch.Tensor,
    height: int,
    width: int,
    config: FlashVidConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    frame_count, tokens_per_frame, _ = metric_frames.shape
    if frame_count <= 1:
        score = _minmax(detail.reshape(-1), dim=0)
        return score, {
            "change_q50": 0.0,
            "change_q75": 0.0,
            "change_q90": 0.0,
            "suppressed_scene_frames": 0.0,
        }

    direct_pair = (
        1.0 - torch.sum(metric_frames[1:] * metric_frames[:-1], dim=-1)
    ).clamp(0.0, 2.0) * 0.5
    radius = max(0, _cfg_int(config, "certsb_local_radius", 1))
    forward = _local_pair_change(
        metric_frames[1:], metric_frames[:-1], height, width, radius
    )
    backward = _local_pair_change(
        metric_frames[:-1], metric_frames[1:], height, width, radius
    )

    direct = torch.zeros(
        (frame_count, tokens_per_frame), dtype=torch.float32, device=metric_frames.device
    )
    local = torch.zeros_like(direct)
    direct[1:] = torch.maximum(direct[1:], direct_pair)
    direct[:-1] = torch.maximum(direct[:-1], direct_pair)
    local[1:] = torch.maximum(local[1:], forward)
    local[:-1] = torch.maximum(local[:-1], backward)
    raw = torch.nan_to_num(0.35 * direct + 0.65 * local, nan=0.0, posinf=0.0, neginf=0.0)

    flat = raw.reshape(-1)
    q50 = torch.quantile(flat, min(1.0, max(0.0, _cfg_float(config, "certsb_change_low_quantile", 0.50))))
    q75 = torch.quantile(flat, min(1.0, max(0.0, _cfg_float(config, "certsb_change_peak_quantile", 0.75))))
    q90 = torch.quantile(flat, min(1.0, max(0.0, _cfg_float(config, "certsb_change_high_quantile", 0.90))))
    lower_scale = (q75 - q50).abs().clamp_min(1e-4)
    upper_scale = (q90 - q75).abs().clamp_min(1e-4)
    scale = torch.where(flat <= q75, lower_scale, upper_scale)
    band = torch.exp(-0.5 * ((flat - q75) / scale).square())
    band = band * (flat >= q50).float()

    amplitude = _minmax(flat, dim=0)
    temporal = amplitude * band
    frame_change = raw.mean(dim=1)
    scene_quantile = min(1.0, max(0.0, _cfg_float(config, "certsb_scene_quantile", 0.90)))
    scene_threshold = torch.quantile(frame_change, scene_quantile)
    scene_frames = frame_change > scene_threshold
    suppression = min(1.0, max(0.0, _cfg_float(config, "certsb_scene_suppression", 0.65)))
    scene_weight = torch.ones(frame_count, dtype=torch.float32, device=metric_frames.device)
    scene_weight[scene_frames] = 1.0 - suppression

    query_weight = min(0.5, max(0.0, _cfg_float(config, "certsb_temporal_query_weight", 0.15)))
    detail_flat = _minmax(detail.reshape(-1), dim=0)
    query_flat = _minmax(query_score.reshape(-1), dim=0)
    temporal = temporal * scene_weight.repeat_interleave(tokens_per_frame)
    temporal = temporal * (
        0.75 + 0.10 * detail_flat + query_weight * query_flat
    )
    temporal = _minmax(temporal, dim=0)
    return temporal, {
        "change_q50": float(q50.item()),
        "change_q75": float(q75.item()),
        "change_q90": float(q90.item()),
        "suppressed_scene_frames": float(scene_frames.sum().item()),
    }


def _balanced_order(score: torch.Tensor, group_ids: torch.Tensor, group_count: int) -> list[int]:
    score_values = score.detach().float().cpu().tolist()
    group_orders: list[list[int]] = []
    for group in range(group_count):
        members = torch.where(group_ids == group)[0]
        if members.numel() == 0:
            group_orders.append([])
            continue
        local = torch.argsort(score[members], descending=True, stable=True)
        group_orders.append(members[local].detach().cpu().tolist())

    output: list[int] = []
    max_length = max((len(tokens) for tokens in group_orders), default=0)
    for rank in range(max_length):
        offers: list[tuple[float, int, int]] = []
        for group, tokens in enumerate(group_orders):
            if rank < len(tokens):
                token = int(tokens[rank])
                offers.append((-float(score_values[token]), group, token))
        offers.sort()
        output.extend(token for _, _, token in offers)
    return output


def _coverage_order(
    semantic_score: torch.Tensor,
    temporal_score: torch.Tensor,
    detail_score: torch.Tensor,
    group_ids: torch.Tensor,
    group_count: int,
) -> list[int]:
    orders = [
        _balanced_order(semantic_score, group_ids, group_count),
        _balanced_order(temporal_score, group_ids, group_count),
        _balanced_order(detail_score, group_ids, group_count),
    ]
    pointers = [0, 0, 0]
    pattern = (0, 1, 2, 0)  # semantic : change : detail = 2 : 1 : 1
    output: list[int] = []
    offered: set[int] = set()
    while len(offered) < semantic_score.numel():
        progress = False
        for source in pattern:
            order = orders[source]
            while pointers[source] < len(order) and order[pointers[source]] in offered:
                pointers[source] += 1
            if pointers[source] < len(order):
                token = int(order[pointers[source]])
                pointers[source] += 1
                offered.add(token)
                output.append(token)
                progress = True
        if not progress:
            break
    return output


def _fill_bank(
    selected: set[int],
    bank: list[int],
    order: Iterable[int],
    target: int,
) -> None:
    for token in order:
        if len(bank) >= target:
            return
        token = int(token)
        if token in selected:
            continue
        selected.add(token)
        bank.append(token)


def _store_diagnostics(config: FlashVidConfig, diagnostics: dict[str, Any]) -> None:
    setattr(config, "last_certsb_diagnostics", diagnostics)
    for name in (
        "target_tokens",
        "semantic_tokens",
        "temporal_tokens",
        "coverage_tokens",
        "locked_tokens",
        "v3_anchor_overlap",
        "temporal_bin_coverage",
        "protected_structured_tokens",
    ):
        setattr(config, f"last_certsb_{name}", float(diagnostics.get(name, 0.0)))
    if bool(getattr(config, "certsb_debug", False)):
        print(
            "[CertVID-SB] "
            f"budget={int(diagnostics.get('target_tokens', 0))} "
            f"banks={int(diagnostics.get('semantic_tokens', 0))}/"
            f"{int(diagnostics.get('temporal_tokens', 0))}/"
            f"{int(diagnostics.get('coverage_tokens', 0))} "
            f"locked={int(diagnostics.get('locked_tokens', 0))} "
            f"v3_overlap={float(diagnostics.get('v3_anchor_overlap', 0.0)):.3f}"
        )


def certvid_sb_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
    question_features: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate one exact budget across semantic, temporal, and coverage evidence."""
    analysis: dict[str, Any] = {}
    v3_output, v3_indices = certvid_v3_compression(
        video_features,
        cls_attention,
        flashvid_config,
        question_features,
        analysis_sink=analysis,
    )
    v3_plan = getattr(flashvid_config, "_certvid_plan", None)
    if not isinstance(v3_plan, CertVidPlan) or bool(analysis.get("identity", False)):
        diagnostics = {
            "target_tokens": int(v3_indices.numel()),
            "semantic_tokens": int(v3_indices.numel()),
            "temporal_tokens": 0,
            "coverage_tokens": 0,
            "locked_tokens": int(v3_indices.numel()),
            "v3_anchor_overlap": 1.0,
            "temporal_bin_coverage": 1.0,
            "protected_structured_tokens": 0,
            "fallback_reason": "identity_or_missing_v3_plan",
        }
        setattr(flashvid_config, "last_adapter_variant", "certvid_sb")
        _store_diagnostics(flashvid_config, diagnostics)
        return v3_output, v3_indices

    required = (
        "metric_flat",
        "demand_weight",
        "attention",
        "query_score",
        "component_ids",
        "frame_ids",
        "temporal_ids",
    )
    missing = [name for name in required if name not in analysis]
    if missing:
        raise RuntimeError(f"CertVID-SB missing V3 analysis tensors: {missing}")

    frame_count, tokens_per_frame, _ = video_features.shape
    total_tokens = frame_count * tokens_per_frame
    budget = int(v3_indices.numel())
    metric_flat = analysis["metric_flat"]
    metric_frames = metric_flat.view(frame_count, tokens_per_frame, -1)
    demand_weight = analysis["demand_weight"]
    attention = analysis["attention"]
    query_score = analysis["query_score"]
    component_ids = analysis["component_ids"]
    frame_ids = analysis["frame_ids"]
    temporal_ids = analysis["temporal_ids"]

    height, width = _grid_hw(tokens_per_frame, flashvid_config)
    detail = _local_detail(video_features, height, width).reshape(-1)
    temporal_score, temporal_diagnostics = _temporal_change_score(
        metric_frames=metric_frames,
        detail=detail,
        query_score=query_score,
        height=height,
        width=width,
        config=flashvid_config,
    )
    semantic_score = _minmax(
        0.60 * _minmax(demand_weight, dim=0)
        + 0.25 * _minmax(query_score, dim=0)
        + 0.15 * _minmax(attention, dim=0),
        dim=0,
    )
    detail_score = _minmax(detail, dim=0)

    group_count = min(
        frame_count,
        max(1, _cfg_int(flashvid_config, "certsb_temporal_bins", 16)),
    )
    group_ids = torch.div(
        frame_ids * group_count,
        max(1, frame_count),
        rounding_mode="floor",
    ).clamp_max(group_count - 1)

    v3_selected = set(int(token) for token in v3_indices.detach().cpu().tolist())
    locked_mask = v3_plan.fusion_alpha <= 1e-8
    locked = sorted(
        int(token)
        for token in v3_plan.anchor_indices[locked_mask].detach().cpu().tolist()
    )
    shares = _configured_shares(flashvid_config)
    semantic_quota, temporal_quota, coverage_quota = _bank_quotas(
        budget, shares, len(locked)
    )

    selected: set[int] = set(locked)
    semantic_bank = list(locked)
    temporal_bank: list[int] = []
    coverage_bank: list[int] = []

    v3_selected_tensor = torch.tensor(
        sorted(v3_selected), dtype=torch.long, device=video_features.device
    )
    v3_semantic_order = v3_selected_tensor[
        torch.argsort(
            semantic_score[v3_selected_tensor], descending=True, stable=True
        )
    ].detach().cpu().tolist()
    global_semantic_order = torch.argsort(
        semantic_score, descending=True, stable=True
    ).detach().cpu().tolist()
    _fill_bank(selected, semantic_bank, v3_semantic_order, semantic_quota)
    _fill_bank(selected, semantic_bank, global_semantic_order, semantic_quota)

    temporal_order = _balanced_order(temporal_score, group_ids, group_count)
    _fill_bank(selected, temporal_bank, temporal_order, temporal_quota)

    coverage_candidates = _coverage_order(
        semantic_score,
        temporal_score,
        detail_score,
        group_ids,
        group_count,
    )
    _fill_bank(selected, coverage_bank, coverage_candidates, coverage_quota)

    # Degenerate inputs can exhaust one bank through overlap; semantic quality
    # provides a deterministic final fallback without changing the exact budget.
    if len(selected) < budget:
        fallback: list[int] = []
        _fill_bank(selected, fallback, global_semantic_order, budget - len(selected))
        semantic_bank.extend(fallback)
    if len(selected) != budget:
        raise RuntimeError(f"CertVID-SB selected {len(selected)} tokens for budget {budget}")

    selected_tensor = torch.tensor(
        sorted(selected), dtype=torch.long, device=video_features.device
    )
    plan = _build_plan(
        selected=selected_tensor,
        metric_features=metric_flat,
        demand_weight=demand_weight,
        attention=attention,
        query_score=query_score,
        temporal_ids=temporal_ids,
        component_ids=component_ids,
        fusion_alpha=_cfg_float(flashvid_config, "certv3_fusion_alpha", 0.12),
        temperature=_cfg_float(flashvid_config, "certv3_assignment_temperature", 0.07),
    )
    structured_tokens = sorted(set(temporal_bank) | set(coverage_bank) | set(locked))
    if bool(getattr(flashvid_config, "certsb_protect_structured", True)) and structured_tokens:
        protected = torch.tensor(
            structured_tokens, dtype=torch.long, device=selected_tensor.device
        )
        plan.fusion_alpha[torch.isin(selected_tensor, protected)] = 0.0

    flat_features = video_features.reshape(total_tokens, -1)
    output = apply_certvid_plan(flat_features, plan)
    selected_groups = torch.unique(group_ids[selected_tensor]).numel()
    diagnostics: dict[str, Any] = {
        "target_tokens": budget,
        "semantic_tokens": len(semantic_bank),
        "temporal_tokens": len(temporal_bank),
        "coverage_tokens": len(coverage_bank),
        "semantic_share": shares[0],
        "temporal_share": shares[1],
        "coverage_share": shares[2],
        "locked_tokens": len(locked),
        "v3_anchor_overlap": len(selected & v3_selected) / float(max(1, budget)),
        "temporal_bin_coverage": float(selected_groups) / float(max(1, group_count)),
        "protected_structured_tokens": len(structured_tokens),
        "temporal_group_count": group_count,
        **temporal_diagnostics,
    }

    setattr(flashvid_config, "_certvid_plan", plan)
    flashvid_config.vision_token_length = int(output.shape[0])
    flashvid_config.visual_token_length = int(output.shape[0])
    flashvid_config.llm_token_length = None
    setattr(flashvid_config, "last_adapter_variant", "certvid_sb")
    setattr(flashvid_config, "last_adapter_raw_tokens", float(total_tokens))
    setattr(flashvid_config, "last_adapter_output_tokens", float(output.shape[0]))
    _store_diagnostics(flashvid_config, diagnostics)
    return output, plan.anchor_indices


__all__ = ["certvid_sb_compression"]
