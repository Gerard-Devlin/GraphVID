"""CertVID V8: relation-witness repair over a stable V3 evidence core.

V8 first executes CertVID V3 without modification. It then asks whether the
V3 coreset misses reliable state-transition witnesses. Only uncovered
relations are allowed to replace low-contribution V3 anchors, and every
replacement is guarded by the V3 design efficiency and per-frame coverage.
Static or already-covered inputs return the exact V3 output and plan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .certvid import CertVidPlan, _cfg_float, _cfg_int, _grid_hw, _minmax, apply_certvid_plan
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
    ridge: float


@dataclass
class _Relations:
    left: torch.Tensor
    right: torch.Tensor
    score: torch.Tensor
    lag: torch.Tensor
    descriptor: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.score.numel())


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
        "ridge",
    )
    missing = [name for name in names if name not in sink]
    if missing:
        raise ValueError(f"V3 analysis is missing: {', '.join(missing)}")
    return _V3Analysis(**{name: sink[name] for name in names})


def _parse_lags(value: Any, frame_count: int) -> Tuple[int, ...]:
    if isinstance(value, str):
        raw: Sequence[Any] = value.replace(":", ",").replace(";", ",").split(",")
    elif isinstance(value, Sequence):
        raw = value
    else:
        raw = (1, 2, 4)
    lags = set()
    for item in raw:
        text = str(item).strip()
        if not text:
            continue
        try:
            lag = int(text)
        except ValueError:
            continue
        if 0 < lag < max(1, frame_count):
            lags.add(lag)
    lags = sorted(lags)
    return tuple(lags or [1])


def _frame_times(
    config: Any,
    frame_count: int,
    device: torch.device,
) -> Tuple[torch.Tensor, bool, str]:
    raw = getattr(config, "_certvid_frame_times_sec", None)
    source = str(getattr(config, "_certvid_frame_times_source", "missing"))
    if raw is None:
        return torch.arange(frame_count, dtype=torch.float32, device=device), False, "frame_index"
    times = torch.as_tensor(raw, dtype=torch.float32, device=device).flatten()
    if (
        times.numel() != frame_count
        or not bool(torch.isfinite(times).all())
        or (times.numel() > 1 and not bool(torch.all(times[1:] > times[:-1])))
    ):
        return torch.arange(frame_count, dtype=torch.float32, device=device), False, "frame_index"
    return times, True, source


def _spatial_coordinates(tokens_per_frame: int, config: Any, device: torch.device) -> torch.Tensor:
    height, width = _grid_hw(tokens_per_frame, config)
    rows = torch.arange(tokens_per_frame, device=device) // max(1, width)
    cols = torch.arange(tokens_per_frame, device=device) % max(1, width)
    return torch.stack(
        (
            rows.float() / max(1, height - 1),
            cols.float() / max(1, width - 1),
        ),
        dim=1,
    )


def _empty_relations(device: torch.device, dimension: int) -> _Relations:
    return _Relations(
        left=torch.empty(0, dtype=torch.long, device=device),
        right=torch.empty(0, dtype=torch.long, device=device),
        score=torch.empty(0, dtype=torch.float32, device=device),
        lag=torch.empty(0, dtype=torch.long, device=device),
        descriptor=torch.empty((0, dimension), dtype=torch.float32, device=device),
    )


def _build_relations(
    analysis: _V3Analysis,
    frame_count: int,
    tokens_per_frame: int,
    config: Any,
) -> Tuple[_Relations, torch.Tensor]:
    device = analysis.metric_flat.device
    metric = F.normalize(
        analysis.metric_flat.float().reshape(frame_count, tokens_per_frame, -1),
        dim=-1,
    )
    coordinates = _spatial_coordinates(tokens_per_frame, config, device)
    spatial_distance = torch.cdist(coordinates, coordinates, p=2)
    spatial_penalty = max(0.0, _cfg_float(config, "certv8_match_spatial_penalty", 0.08))
    minimum_similarity = min(
        0.95,
        max(-1.0, _cfg_float(config, "certv8_match_min_similarity", 0.25)),
    )
    pairs_per_boundary = max(1, _cfg_int(config, "certv8_pairs_per_boundary", 4))
    query_weight = min(0.40, max(0.0, _cfg_float(config, "certv8_query_weight", 0.15)))
    demand = _minmax(analysis.demand_weight.float(), dim=0)
    query = analysis.query_score.float().clamp(0.0, 1.0)
    lags = _parse_lags(getattr(config, "certv8_relation_lags", "1,2,4"), frame_count)

    left_parts: list[torch.Tensor] = []
    right_parts: list[torch.Tensor] = []
    score_parts: list[torch.Tensor] = []
    lag_parts: list[torch.Tensor] = []
    descriptor_parts: list[torch.Tensor] = []
    adjacent_change = torch.zeros(max(0, frame_count - 1), dtype=torch.float32, device=device)

    for lag in lags:
        lag_scale = 1.0 / math.sqrt(float(lag))
        for current_frame in range(lag, frame_count):
            previous_frame = current_frame - lag
            previous = metric[previous_frame]
            current = metric[current_frame]
            raw_similarity = current @ previous.transpose(0, 1)
            matched_similarity = raw_similarity - spatial_penalty * spatial_distance
            predecessor = torch.argmax(matched_similarity, dim=1)
            token_ids = torch.arange(tokens_per_frame, device=device)
            cosine = raw_similarity[token_ids, predecessor]
            if lag == 1:
                adjacent_change[previous_frame] = (1.0 - cosine).clamp(0.0, 2.0).mean()

            valid = cosine >= minimum_similarity
            if not bool(valid.any()):
                continue
            change = (0.5 * (1.0 - cosine)).clamp(0.0, 1.0)
            reliability = ((cosine - minimum_similarity) / max(1e-6, 1.0 - minimum_similarity)).clamp(
                0.0,
                1.0,
            )
            spatial_confidence = (
                1.0 - spatial_distance[token_ids, predecessor] / math.sqrt(2.0)
            ).clamp(0.0, 1.0)
            current_global = current_frame * tokens_per_frame + token_ids
            previous_global = previous_frame * tokens_per_frame + predecessor
            endpoint_demand = 0.5 * (demand[current_global] + demand[previous_global])
            endpoint_query = 0.5 * (query[current_global] + query[previous_global])
            evidence_support = (
                0.55
                + 0.20 * endpoint_demand
                + query_weight * endpoint_query
                + max(0.0, 0.25 - query_weight) * spatial_confidence
            )
            score = change * evidence_support * (0.35 + 0.65 * reliability) * lag_scale
            score = score.masked_fill(~valid, -1.0)
            keep = min(pairs_per_boundary, int(valid.sum().item()))
            order = torch.argsort(score, descending=True, stable=True)[:keep]
            selected_score = score[order]
            selected_previous = previous_global[order]
            selected_current = current_global[order]
            relation_descriptor = F.normalize(
                current[order] - previous[predecessor[order]],
                dim=-1,
            )
            left_parts.append(selected_previous)
            right_parts.append(selected_current)
            score_parts.append(selected_score)
            lag_parts.append(torch.full((keep,), lag, dtype=torch.long, device=device))
            descriptor_parts.append(relation_descriptor)

    if not score_parts:
        return _empty_relations(device, int(metric.shape[-1])), adjacent_change

    relations = _Relations(
        left=torch.cat(left_parts),
        right=torch.cat(right_parts),
        score=torch.cat(score_parts).clamp_min(0.0),
        lag=torch.cat(lag_parts),
        descriptor=torch.cat(descriptor_parts),
    )
    order = torch.argsort(relations.score, descending=True, stable=True)
    relations = _Relations(
        left=relations.left[order],
        right=relations.right[order],
        score=relations.score[order],
        lag=relations.lag[order],
        descriptor=relations.descriptor[order],
    )
    return relations, adjacent_change


def _same_frame_coverage(
    metric_flat: torch.Tensor,
    selected: torch.Tensor,
    frame_count: int,
    tokens_per_frame: int,
) -> torch.Tensor:
    metric = F.normalize(metric_flat.float(), dim=-1).reshape(frame_count, tokens_per_frame, -1)
    coverage = torch.zeros(frame_count * tokens_per_frame, dtype=torch.float32, device=metric.device)
    selected_frames = selected // tokens_per_frame
    for frame in range(frame_count):
        local = selected[selected_frames == frame] - frame * tokens_per_frame
        if local.numel() == 0:
            continue
        similarity = metric[frame] @ metric[frame, local].transpose(0, 1)
        coverage[frame * tokens_per_frame : (frame + 1) * tokens_per_frame] = (
            0.5 * (similarity.amax(dim=1) + 1.0)
        ).clamp(0.0, 1.0)
    return coverage


def _relation_objective(
    relations: _Relations,
    coverage: torch.Tensor,
) -> float:
    if relations.count == 0:
        return 0.0
    edge_coverage = torch.minimum(coverage[relations.left], coverage[relations.right])
    weights = relations.score / relations.score.sum().clamp_min(1e-8)
    return float(torch.sum(weights * edge_coverage).item())


def _relation_deficit(
    relations: _Relations,
    coverage: torch.Tensor,
    threshold: float,
) -> float:
    if relations.count == 0:
        return 0.0
    edge_coverage = torch.minimum(coverage[relations.left], coverage[relations.right])
    deficit = (threshold - edge_coverage).clamp_min(0.0) / max(1e-6, threshold)
    weights = relations.score / relations.score.sum().clamp_min(1e-8)
    return float(torch.sum(weights * deficit).item())


def _effective_relation_rank(descriptor: torch.Tensor) -> float:
    if descriptor.shape[0] < 2:
        return 0.0
    singular = torch.linalg.svdvals(descriptor.float())
    energy = singular.square()
    rank = energy.sum().square() / energy.square().sum().clamp_min(1e-8)
    return float((rank / max(1, min(descriptor.shape))).clamp(0.0, 1.0).item())


def _router_score(
    relations: _Relations,
    adjacent_change: torch.Tensor,
    analysis: _V3Analysis,
    frame_count: int,
    tokens_per_frame: int,
    duration_seconds: float,
) -> Tuple[float, Dict[str, float]]:
    if relations.count == 0:
        return 0.0, {
            "transition_signal": 0.0,
            "frame_change_signal": 0.0,
            "relation_rank": 0.0,
            "query_spread": 0.0,
            "duration_signal": 0.0,
        }
    top_count = max(1, int(math.ceil(0.25 * relations.count)))
    transition_signal = float(
        (relations.score[:top_count].mean() / 0.35).clamp(0.0, 1.0).item()
    )
    if adjacent_change.numel():
        change_count = max(1, int(math.ceil(0.25 * adjacent_change.numel())))
        frame_change_signal = float(
            (
                torch.topk(adjacent_change, k=change_count, largest=True).values.mean() / 0.18
            )
            .clamp(0.0, 1.0)
            .item()
        )
    else:
        frame_change_signal = 0.0
    relation_rank = _effective_relation_rank(relations.descriptor)

    query_frames = analysis.query_score.reshape(frame_count, tokens_per_frame).amax(dim=1)
    query_weights = torch.softmax(6.0 * query_frames.float(), dim=0)
    query_entropy = -torch.sum(query_weights * torch.log(query_weights.clamp_min(1e-8)))
    query_spread = float(
        (
            query_entropy / max(1e-8, math.log(max(2, frame_count)))
            * min(1.0, max(0.0, float(analysis.query_confidence)))
        ).item()
    )
    duration_signal = min(1.0, math.log1p(max(0.0, duration_seconds) / 120.0) / math.log(6.0))
    router = (
        0.42 * transition_signal
        + 0.28 * frame_change_signal
        + 0.14 * relation_rank
        + 0.10 * query_spread
        + 0.06 * duration_signal
    )
    signals = {
        "transition_signal": transition_signal,
        "frame_change_signal": frame_change_signal,
        "relation_rank": relation_rank,
        "query_spread": query_spread,
        "duration_signal": duration_signal,
    }
    return min(1.0, max(0.0, router)), signals


def _logdet_efficiency(
    design: torch.Tensor,
    baseline: torch.Tensor,
    selected: torch.Tensor,
    ridge: float,
) -> float:
    dimension = int(design.shape[1])
    identity = torch.eye(dimension, dtype=torch.float32, device=design.device)
    ridge = max(1e-4, float(ridge))

    def value(indices: torch.Tensor) -> torch.Tensor:
        rows = design[indices].float()
        information = ridge * identity + rows.transpose(0, 1) @ rows
        sign, logdet = torch.linalg.slogdet(information)
        if float(sign.item()) <= 0.0:
            raise RuntimeError("non-positive V8 design information matrix")
        return logdet

    delta = (value(selected) - value(baseline)) / max(1, dimension)
    return float(torch.exp(delta.clamp(min=-20.0, max=20.0)).item())


def _candidate_order(
    relations: _Relations,
    coverage: torch.Tensor,
    selected: torch.Tensor,
    analysis: _V3Analysis,
    frame_count: int,
    tokens_per_frame: int,
    limit: int,
    target_counts: Optional[torch.Tensor] = None,
) -> list[int]:
    if relations.count == 0 or limit <= 0:
        return []
    total_tokens = int(coverage.numel())
    edge_coverage = torch.minimum(coverage[relations.left], coverage[relations.right])
    edge_deficit = (1.0 - edge_coverage).clamp_min(0.0)
    token_score = torch.zeros(total_tokens, dtype=torch.float32, device=coverage.device)
    endpoint_weight = relations.score * edge_deficit
    token_score.index_add_(0, relations.left, endpoint_weight)
    token_score.index_add_(0, relations.right, endpoint_weight)
    token_score = (
        token_score
        + 0.10 * analysis.query_score
        + 0.08 * _minmax(analysis.demand_weight, dim=0)
        + 0.12 * (1.0 - coverage).clamp(0.0, 1.0)
    )
    token_score[selected] = -1.0

    frame_counts = torch.bincount(
        selected // tokens_per_frame,
        minlength=frame_count,
    ).float()
    token_frames = torch.arange(total_tokens, device=coverage.device) // tokens_per_frame
    if target_counts is not None:
        target = target_counts.to(device=coverage.device, dtype=torch.float32)
        frame_need = ((target - frame_counts).clamp_min(0.0) / target.clamp_min(1.0))
        token_score = token_score * (1.0 + 1.50 * frame_need[token_frames])
        eligible = frame_counts[token_frames] < target[token_frames]
    else:
        average = max(1.0, float(selected.numel()) / max(1, frame_count))
        frame_need = ((average - frame_counts).clamp_min(0.0) / average)
        token_score = token_score * (1.0 + 0.20 * frame_need[token_frames])
        eligible = torch.ones(total_tokens, dtype=torch.bool, device=coverage.device)

    candidate_ids = torch.where((token_score > 0.0) & eligible)[0]
    if candidate_ids.numel() == 0:
        return []
    candidate_ids = candidate_ids[
        torch.argsort(token_score[candidate_ids], descending=True, stable=True)
    ]
    pool_size = min(
        int(candidate_ids.numel()),
        max(128, int(limit) * 6),
    )
    candidate_ids = candidate_ids[:pool_size]
    base_score = token_score[candidate_ids]

    direction_sum = torch.zeros(
        (total_tokens, int(relations.descriptor.shape[1])),
        dtype=torch.float32,
        device=coverage.device,
    )
    weighted_direction = relations.descriptor.float() * endpoint_weight.unsqueeze(1)
    direction_sum.index_add_(0, relations.left, weighted_direction)
    direction_sum.index_add_(0, relations.right, weighted_direction)
    fallback_direction = F.normalize(analysis.metric_flat.float(), dim=-1)
    relation_norm = direction_sum.norm(dim=1, keepdim=True)
    all_directions = torch.where(
        relation_norm > 1e-8,
        direction_sum / relation_norm.clamp_min(1e-8),
        fallback_direction,
    )
    directions = all_directions[candidate_ids]
    candidate_frames = candidate_ids // tokens_per_frame
    available = torch.ones(pool_size, dtype=torch.bool, device=coverage.device)
    max_similarity = torch.zeros(pool_size, dtype=torch.float32, device=coverage.device)
    selected_tokens: list[int] = []

    for _ in range(min(limit, pool_size)):
        if target_counts is not None:
            target = target_counts.to(device=coverage.device, dtype=torch.float32)
            dynamic_need = (
                (target - frame_counts).clamp_min(0.0) / target.clamp_min(1.0)
            )
            need_multiplier = 1.0 + 1.50 * dynamic_need[candidate_frames]
            available &= frame_counts[candidate_frames] < target[candidate_frames]
        else:
            average = max(1.0, float(selected.numel()) / max(1, frame_count))
            dynamic_need = (average - frame_counts).clamp_min(0.0) / average
            need_multiplier = 1.0 + 0.20 * dynamic_need[candidate_frames]
        score = base_score * need_multiplier * (1.0 - 0.35 * max_similarity)
        score = score.masked_fill(~available, -1.0)
        best_local = int(torch.argmax(score).item())
        if float(score[best_local].item()) <= 0.0:
            break
        token = int(candidate_ids[best_local].item())
        selected_tokens.append(token)
        available[best_local] = False
        frame_counts[token // tokens_per_frame] += 1.0
        similarity = torch.abs(directions @ directions[best_local])
        max_similarity = torch.maximum(max_similarity, similarity)
    return selected_tokens


def _protected_v3_anchors(
    v3_indices: torch.Tensor,
    v3_plan: CertVidPlan,
    analysis: _V3Analysis,
    relations: _Relations,
    config: Any,
) -> set[int]:
    protected = {
        int(token)
        for token in v3_indices[v3_plan.fusion_alpha <= 1e-12].detach().cpu().tolist()
    }
    ratio = min(0.50, max(0.0, _cfg_float(config, "certv8_design_protect_ratio", 0.10)))
    count = min(int(v3_indices.numel()), int(math.ceil(ratio * int(v3_indices.numel()))))
    if count > 0:
        leverage = analysis.design[v3_indices].float().square().sum(dim=1)
        local = torch.topk(leverage, k=count, largest=True).indices
        protected.update(int(token) for token in v3_indices[local].detach().cpu().tolist())

    relation_ratio = min(
        0.25,
        max(0.0, _cfg_float(config, "certv8_relation_protect_ratio", 0.05)),
    )
    relation_count = min(
        int(v3_indices.numel()),
        int(math.ceil(relation_ratio * int(v3_indices.numel()))),
    )
    if relation_count > 0 and relations.count:
        token_score = torch.zeros(
            int(analysis.metric_flat.shape[0]),
            dtype=torch.float32,
            device=v3_indices.device,
        )
        token_score.index_add_(0, relations.left, relations.score)
        token_score.index_add_(0, relations.right, relations.score)
        selected_score = token_score[v3_indices]
        local = torch.topk(selected_score, k=relation_count, largest=True).indices
        protected.update(int(token) for token in v3_indices[local].detach().cpu().tolist())
    return protected


def _removal_costs(
    v3_indices: torch.Tensor,
    analysis: _V3Analysis,
) -> Dict[int, float]:
    design = _minmax(analysis.design[v3_indices].float().square().sum(dim=1), dim=0)
    demand = _minmax(analysis.demand_weight[v3_indices].float(), dim=0)
    query = analysis.query_score[v3_indices].float().clamp(0.0, 1.0)
    selected_metric = F.normalize(analysis.metric_flat[v3_indices].float(), dim=-1)
    if v3_indices.numel() > 1:
        similarity = selected_metric @ selected_metric.transpose(0, 1)
        similarity.fill_diagonal_(-1.0)
        redundancy = (0.5 * (similarity.amax(dim=1) + 1.0)).clamp(0.0, 1.0)
    else:
        redundancy = torch.zeros_like(design)
    cost = 0.48 * design + 0.22 * demand + 0.20 * query + 0.10 * (1.0 - redundancy)
    return {
        int(token): float(value)
        for token, value in zip(v3_indices.detach().cpu().tolist(), cost.detach().cpu().tolist())
    }


def _frame_target_counts(
    v3_indices: torch.Tensor,
    protected: set[int],
    relations: _Relations,
    coverage: torch.Tensor,
    adjacent_change: torch.Tensor,
    analysis: _V3Analysis,
    frame_count: int,
    tokens_per_frame: int,
    router: float,
    config: Any,
) -> torch.Tensor:
    """Project the global V3 budget onto a near-uniform, event-aware frame shape."""
    device = v3_indices.device
    budget = int(v3_indices.numel())
    average = float(budget) / max(1, frame_count)
    floor = min(
        tokens_per_frame,
        max(
            0 if budget < frame_count else 1,
            int(math.floor(average * _cfg_float(config, "certv8_frame_floor_ratio", 0.88))),
        ),
    )
    cap = min(
        tokens_per_frame,
        max(
            floor + 1,
            int(math.ceil(average * _cfg_float(config, "certv8_frame_cap_ratio", 1.18))),
        ),
    )

    edge_coverage = torch.minimum(coverage[relations.left], coverage[relations.right])
    edge_need = relations.score * (1.0 - edge_coverage).clamp(0.0, 1.0)
    relation_demand = torch.zeros(frame_count, dtype=torch.float32, device=device)
    relation_demand.index_add_(0, relations.left // tokens_per_frame, edge_need)
    relation_demand.index_add_(0, relations.right // tokens_per_frame, edge_need)
    relation_demand = _minmax(relation_demand, dim=0)

    change_demand = torch.zeros(frame_count, dtype=torch.float32, device=device)
    if adjacent_change.numel():
        change_demand[:-1] += adjacent_change
        change_demand[1:] += adjacent_change
    change_demand = _minmax(change_demand, dim=0)

    query_demand = analysis.query_score.reshape(frame_count, tokens_per_frame).amax(dim=1)
    query_demand = _minmax(query_demand.float(), dim=0) * min(
        1.0,
        max(0.0, float(analysis.query_confidence)),
    )
    visual_demand = _minmax(
        analysis.demand_weight.reshape(frame_count, tokens_per_frame).mean(dim=1).float(),
        dim=0,
    )
    frame_demand = (
        0.50 * relation_demand
        + 0.25 * change_demand
        + 0.15 * query_demand
        + 0.10 * visual_demand
    )
    frame_demand = _minmax(frame_demand, dim=0)

    # FlashVID's robust long-video shape is close to a per-frame budget with
    # modest event-boundary bumps. Match that geometry without borrowing its
    # token scorer or changing V3's global token count.
    modulation = 0.18 + 0.08 * min(1.0, max(0.0, float(router)))
    weights = 1.0 + modulation * (frame_demand - frame_demand.mean())
    weights = weights.clamp(0.80, 1.25)
    raw_target = float(budget) * weights / weights.sum().clamp_min(1e-8)

    protected_counts = torch.zeros(frame_count, dtype=torch.long, device=device)
    if protected:
        protected_indices = torch.tensor(sorted(protected), dtype=torch.long, device=device)
        protected_counts = torch.bincount(
            protected_indices // tokens_per_frame,
            minlength=frame_count,
        )
    lower = torch.maximum(
        protected_counts,
        torch.full((frame_count,), floor, dtype=torch.long, device=device),
    )
    if int(lower.sum().item()) > budget:
        lower = protected_counts.clone()
    upper = torch.maximum(
        lower,
        torch.full((frame_count,), cap, dtype=torch.long, device=device),
    ).clamp_max(tokens_per_frame)

    target = lower.clone()
    remaining = budget - int(target.sum().item())
    while remaining > 0:
        eligible = target < upper
        if not bool(eligible.any()):
            eligible = target < tokens_per_frame
        if not bool(eligible.any()):
            break
        residual = raw_target - target.float()
        residual = residual.masked_fill(~eligible, -float("inf"))
        frame = int(torch.argmax(residual).item())
        target[frame] += 1
        remaining -= 1
    if int(target.sum().item()) != budget:
        raise RuntimeError("V8 frame target could not preserve the global budget")
    return target


def _frame_shape_error(
    selected: torch.Tensor,
    target_counts: torch.Tensor,
    tokens_per_frame: int,
) -> float:
    counts = torch.bincount(
        selected // tokens_per_frame,
        minlength=int(target_counts.numel()),
    )
    # One swap fixes one surplus and one deficit, hence the factor of two.
    return float(
        (
            (counts.float() - target_counts.float()).abs().sum()
            / max(1.0, 2.0 * float(selected.numel()))
        ).item()
    )


def _propose_swaps(
    v3_indices: torch.Tensor,
    additions: Sequence[int],
    protected: set[int],
    removal_cost: Dict[int, float],
    frame_count: int,
    tokens_per_frame: int,
    config: Any,
    target_counts: Optional[torch.Tensor] = None,
) -> Tuple[list[int], list[int]]:
    budget = int(v3_indices.numel())
    average = float(budget) / max(1, frame_count)
    floor = min(
        tokens_per_frame,
        max(
            0 if budget < frame_count else 1,
            int(math.floor(average * _cfg_float(config, "certv8_frame_floor_ratio", 0.70))),
        ),
    )
    cap = min(
        tokens_per_frame,
        max(floor + 1, int(math.ceil(average * _cfg_float(config, "certv8_frame_cap_ratio", 1.35)))),
    )
    current = {int(token) for token in v3_indices.detach().cpu().tolist()}
    counts = torch.bincount(v3_indices // tokens_per_frame, minlength=frame_count).cpu().tolist()
    accepted_additions: list[int] = []
    accepted_removals: list[int] = []

    for addition in additions:
        if addition in current:
            continue
        add_frame = addition // tokens_per_frame
        if target_counts is not None and counts[add_frame] >= int(target_counts[add_frame]):
            continue
        if counts[add_frame] >= cap:
            continue
        counts[add_frame] += 1
        candidates = []
        for token in current:
            if token in protected:
                continue
            if token not in removal_cost:
                continue
            frame = token // tokens_per_frame
            if counts[frame] - 1 < floor:
                continue
            if target_counts is not None:
                surplus = counts[frame] - int(target_counts[frame])
                if surplus <= 0:
                    continue
                overrepresentation = float(surplus) / max(
                    1.0,
                    float(target_counts[frame]),
                )
                adjusted_cost = removal_cost.get(token, 1.0) - 0.30 * overrepresentation
            else:
                overrepresentation = max(0.0, (counts[frame] - average) / max(1.0, average))
                adjusted_cost = removal_cost.get(token, 1.0) - 0.08 * overrepresentation
            candidates.append((adjusted_cost, token))
        if not candidates:
            counts[add_frame] -= 1
            continue
        candidates.sort(key=lambda item: (item[0], item[1]))
        removal = candidates[0][1]
        current.remove(removal)
        current.add(addition)
        counts[removal // tokens_per_frame] -= 1
        accepted_additions.append(addition)
        accepted_removals.append(removal)
    return accepted_additions, accepted_removals


def _trial_selection(
    baseline: torch.Tensor,
    additions: Sequence[int],
    removals: Sequence[int],
    count: int,
) -> torch.Tensor:
    selected = {int(token) for token in baseline.detach().cpu().tolist()}
    for addition, removal in zip(additions[:count], removals[:count]):
        selected.remove(removal)
        selected.add(addition)
    return torch.tensor(sorted(selected), dtype=torch.long, device=baseline.device)


def _build_local_plan(
    selected: torch.Tensor,
    new_anchors: set[int],
    protected_v3: set[int],
    analysis: _V3Analysis,
    frame_times: torch.Tensor,
    has_real_times: bool,
    frame_count: int,
    tokens_per_frame: int,
    config: Any,
) -> CertVidPlan:
    total_tokens = int(analysis.metric_flat.shape[0])
    budget = int(selected.numel())
    similarity = analysis.metric_flat.float() @ analysis.metric_flat[selected].float().transpose(0, 1)
    source_frame = torch.arange(frame_count, device=selected.device).repeat_interleave(tokens_per_frame)
    anchor_frame = source_frame[selected]
    frame_delta = (source_frame.unsqueeze(1) - anchor_frame.unsqueeze(0)).abs()
    valid = frame_delta == 0
    adjacent = frame_delta == 1
    cross_similarity = _cfg_float(config, "certv8_cross_frame_similarity", 0.88)
    cross_valid = adjacent & (similarity >= cross_similarity)
    if has_real_times:
        time_gap = torch.abs(frame_times[source_frame].unsqueeze(1) - frame_times[anchor_frame].unsqueeze(0))
        cross_valid &= time_gap <= _cfg_float(config, "certv8_cross_frame_max_seconds", 8.0)
    valid |= cross_valid
    if not bool(valid.any(dim=1).all()):
        raise RuntimeError("V8 local plan has a frame without a reachable anchor")
    similarity = similarity.masked_fill(~valid, -2.0)
    same_component = analysis.component_ids.unsqueeze(1) == analysis.component_ids[selected].unsqueeze(0)
    similarity = similarity + _cfg_float(config, "certv8_component_bonus", 0.08) * same_component.float()

    topk = min(max(1, _cfg_int(config, "certv8_assignment_topk", 2)), budget)
    values, assignment = torch.topk(similarity, k=topk, dim=1, largest=True)
    chosen_valid = torch.gather(valid, 1, assignment)
    best_valid = torch.argmax(similarity, dim=1, keepdim=True).expand_as(assignment)
    assignment = torch.where(chosen_valid, assignment, best_valid)
    values = torch.gather(similarity, 1, assignment)
    weights = torch.softmax(
        values.float() / max(1e-4, _cfg_float(config, "certv8_assignment_temperature", 0.07)),
        dim=1,
    )
    anchor_positions = torch.arange(budget, dtype=torch.long, device=selected.device)
    assignment[selected, 0] = anchor_positions
    weights[selected] = 0.0
    weights[selected, 0] = 1.0

    source_mass = (0.5 + 0.5 * analysis.demand_weight * total_tokens).clamp(0.25, 2.0)
    alpha_value = min(0.25, max(0.0, _cfg_float(config, "certv8_fusion_alpha", 0.10)))
    alpha = torch.full((budget,), alpha_value, dtype=torch.float32, device=selected.device)
    protected = protected_v3 | new_anchors
    if protected:
        protected_tensor = torch.tensor(sorted(protected), dtype=torch.long, device=selected.device)
        alpha[torch.isin(selected, protected_tensor)] = 0.0
    return CertVidPlan(
        anchor_indices=selected,
        assignment_indices=assignment,
        assignment_weights=weights,
        source_mass=source_mass,
        fusion_alpha=alpha,
        raw_token_count=total_tokens,
    )


def _store_diagnostics(config: Any, diagnostics: Dict[str, Any]) -> None:
    config.last_certv8_diagnostics = diagnostics
    config.last_certv8_fallback_reason = diagnostics.get("fallback_reason")
    config.last_certv8_router_score = float(diagnostics.get("router_score", 0.0))
    config.last_certv8_relation_count = int(diagnostics.get("relation_count", 0))
    config.last_certv8_relation_deficit = float(diagnostics.get("relation_deficit", 0.0))
    config.last_certv8_base_relation_coverage = float(
        diagnostics.get("base_relation_coverage", 0.0)
    )
    config.last_certv8_final_relation_coverage = float(
        diagnostics.get("final_relation_coverage", 0.0)
    )
    config.last_certv8_swap_count = int(diagnostics.get("swap_count", 0))
    config.last_certv8_modified_ratio = float(diagnostics.get("modified_ratio", 0.0))
    config.last_certv8_v3_overlap_ratio = float(diagnostics.get("v3_overlap_ratio", 1.0))
    config.last_certv8_d_efficiency = float(diagnostics.get("d_efficiency", 1.0))
    config.last_certv8_unsafe_assignment_count = int(
        diagnostics.get("unsafe_assignment_count", 0)
    )
    if bool(getattr(config, "certv8_debug", False)):
        print(
            "[certvid-v8] "
            f"sample={getattr(config, '_debug_sample_id', 'unknown')} "
            f"fallback={diagnostics.get('fallback_reason')} "
            f"duration={diagnostics.get('duration_seconds', 0.0):.1f}s "
            f"router={diagnostics.get('router_score', 0.0):.3f} "
            f"deficit={diagnostics.get('relation_deficit', 0.0):.3f} "
            f"relations={diagnostics.get('relation_count', 0)} "
            f"limit={diagnostics.get('swap_limit', 0)} "
            f"swaps={diagnostics.get('swap_count', 0)} "
            f"overlap={diagnostics.get('v3_overlap_ratio', 1.0):.3f} "
            f"relation={diagnostics.get('base_relation_coverage', 0.0):.3f}->"
            f"{diagnostics.get('final_relation_coverage', 0.0):.3f} "
            f"D-eff={diagnostics.get('d_efficiency', 1.0):.4f} "
            f"shape={diagnostics.get('base_frame_shape_error', 0.0):.3f}->"
            f"{diagnostics.get('final_frame_shape_error', 0.0):.3f} "
            f"base_frames={diagnostics.get('base_frame_counts', [])} "
            f"target_frames={diagnostics.get('target_frame_counts', [])} "
            f"final_frames={diagnostics.get('final_frame_counts', [])}"
        )


def _fallback(
    config: Any,
    diagnostics: Dict[str, Any],
    reason: str,
    output: torch.Tensor,
    indices: torch.Tensor,
    plan: Optional[CertVidPlan],
) -> Tuple[torch.Tensor, torch.Tensor]:
    diagnostics["fallback_reason"] = reason
    diagnostics.setdefault("swap_count", 0)
    diagnostics.setdefault("modified_ratio", 0.0)
    diagnostics.setdefault("v3_overlap_ratio", 1.0)
    diagnostics.setdefault("d_efficiency", 1.0)
    diagnostics.setdefault(
        "final_relation_coverage",
        diagnostics.get("base_relation_coverage", 0.0),
    )
    diagnostics.setdefault(
        "final_frame_counts",
        diagnostics.get("base_frame_counts", []),
    )
    diagnostics.setdefault(
        "final_frame_shape_error",
        diagnostics.get("base_frame_shape_error", 0.0),
    )
    if plan is not None:
        config._certvid_plan = plan
    config.last_adapter_variant = "certvid_v8"
    _store_diagnostics(config, diagnostics)
    return output, indices


def certvid_v8_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: Any,
    question_features: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Repair uncovered temporal relations while preserving the V3 evidence core."""
    if video_features.ndim != 3:
        raise ValueError(f"expected video_features [T, HW, D], got {tuple(video_features.shape)}")
    config = flashvid_config
    frame_count, tokens_per_frame, _ = video_features.shape
    sink: Dict[str, Any] = {}
    v3_output, v3_indices = certvid_v3_compression(
        video_features,
        cls_attention,
        config,
        question_features,
        analysis_sink=sink,
    )
    v3_plan = getattr(config, "_certvid_plan", None)
    diagnostics: Dict[str, Any] = {
        "fallback_reason": None,
        "router_score": 0.0,
        "relation_count": 0,
        "relation_deficit": 0.0,
        "base_relation_coverage": 0.0,
        "final_relation_coverage": 0.0,
        "base_frame_counts": [
            int(value)
            for value in torch.bincount(
                v3_indices // max(1, tokens_per_frame),
                minlength=frame_count,
            )
            .detach()
            .cpu()
            .tolist()
        ],
    }
    if not bool(getattr(config, "certv8_enabled", True)):
        return _fallback(config, diagnostics, "disabled", v3_output, v3_indices, v3_plan)
    if v3_plan is None:
        return _fallback(config, diagnostics, "missing_v3_plan", v3_output, v3_indices, v3_plan)
    if bool(sink.get("identity", False)) or not sink:
        return _fallback(config, diagnostics, "identity_budget", v3_output, v3_indices, v3_plan)
    if frame_count < 2 or v3_indices.numel() < 2:
        return _fallback(config, diagnostics, "insufficient_temporal_input", v3_output, v3_indices, v3_plan)

    try:
        analysis = _analysis_from_sink(sink)
        frame_times, has_real_times, timestamp_source = _frame_times(
            config,
            frame_count,
            video_features.device,
        )
        duration = (
            float(frame_times[-1].item() - frame_times[0].item())
            if has_real_times and frame_count > 1
            else 0.0
        )
        relations, adjacent_change = _build_relations(
            analysis,
            frame_count,
            tokens_per_frame,
            config,
        )
        diagnostics.update(
            {
                "timestamp_source": timestamp_source,
                "duration_seconds": duration,
                "relation_count": relations.count,
            }
        )
        if relations.count == 0:
            return _fallback(config, diagnostics, "no_reliable_relations", v3_output, v3_indices, v3_plan)

        base_coverage = _same_frame_coverage(
            analysis.metric_flat,
            v3_indices,
            frame_count,
            tokens_per_frame,
        )
        coverage_threshold = min(
            0.999,
            max(0.50, _cfg_float(config, "certv8_relation_coverage_threshold", 0.88)),
        )
        base_objective = _relation_objective(relations, base_coverage)
        deficit = _relation_deficit(relations, base_coverage, coverage_threshold)
        router, router_signals = _router_score(
            relations,
            adjacent_change,
            analysis,
            frame_count,
            tokens_per_frame,
            duration,
        )
        diagnostics.update(
            {
                "router_score": router,
                "relation_deficit": deficit,
                "base_relation_coverage": base_objective,
                **router_signals,
            }
        )
        gate = min(1.0, max(0.0, _cfg_float(config, "certv8_gate_threshold", 0.18)))
        minimum_deficit = min(
            1.0,
            max(0.0, _cfg_float(config, "certv8_min_relation_deficit", 0.04)),
        )
        if router < gate:
            return _fallback(config, diagnostics, "weak_relation_signal", v3_output, v3_indices, v3_plan)
        if deficit < minimum_deficit:
            return _fallback(config, diagnostics, "v3_relation_sufficient", v3_output, v3_indices, v3_plan)

        long_threshold = max(
            0.0,
            _cfg_float(config, "certv8_long_duration_seconds", 120.0),
        )
        has_long_timeline = duration >= long_threshold
        protected = _protected_v3_anchors(v3_indices, v3_plan, analysis, relations, config)
        candidate_target_counts = (
            _frame_target_counts(
                v3_indices,
                protected,
                relations,
                base_coverage,
                adjacent_change,
                analysis,
                frame_count,
                tokens_per_frame,
                router,
                config,
            )
            if has_long_timeline
            else None
        )
        base_shape_error = (
            _frame_shape_error(
                v3_indices,
                candidate_target_counts,
                tokens_per_frame,
            )
            if candidate_target_counts is not None
            else 0.0
        )
        shape_gate = min(
            1.0,
            max(0.0, _cfg_float(config, "certv8_frame_shape_gate", 0.08)),
        )
        shape_mode = has_long_timeline and base_shape_error + 1e-12 >= shape_gate
        target_counts = candidate_target_counts if shape_mode else None
        maximum_ratio = _cfg_float(
            config,
            "certv8_long_max_swap_ratio" if shape_mode else "certv8_short_max_swap_ratio",
            0.06 if shape_mode else 0.12,
        )
        maximum_ratio = min(0.35, max(0.0, maximum_ratio))
        if shape_mode and target_counts is not None:
            current_counts = torch.bincount(
                v3_indices // tokens_per_frame,
                minlength=frame_count,
            )
            shape_required = int(
                (target_counts - current_counts).clamp_min(0).sum().item()
            )
            swap_limit = min(
                int(math.ceil(int(v3_indices.numel()) * maximum_ratio)),
                shape_required,
            )
        else:
            adaptive_ratio = maximum_ratio * deficit * (0.55 + 0.45 * router)
            swap_limit = max(
                1,
                int(math.ceil(int(v3_indices.numel()) * adaptive_ratio)),
            )
        swap_limit = min(int(v3_indices.numel()), max(1, swap_limit))
        diagnostics.update(
            {
                "long_horizon": has_long_timeline,
                "frame_shape_mode": shape_mode,
                "frame_shape_gate": shape_gate,
                "swap_limit": swap_limit,
                "base_frame_shape_error": base_shape_error,
                "candidate_target_frame_counts": (
                    [
                        int(value)
                        for value in candidate_target_counts.detach().cpu().tolist()
                    ]
                    if candidate_target_counts is not None
                    else []
                ),
                "target_frame_counts": (
                    [int(value) for value in target_counts.detach().cpu().tolist()]
                    if target_counts is not None
                    else []
                ),
            }
        )
        additions = _candidate_order(
            relations,
            base_coverage,
            v3_indices,
            analysis,
            frame_count,
            tokens_per_frame,
            swap_limit * 3,
            target_counts=target_counts,
        )
        removal_cost = _removal_costs(v3_indices, analysis)
        additions, removals = _propose_swaps(
            v3_indices,
            additions[: swap_limit * 2],
            protected,
            removal_cost,
            frame_count,
            tokens_per_frame,
            config,
            target_counts=target_counts,
        )
        if not additions:
            return _fallback(config, diagnostics, "no_safe_swaps", v3_output, v3_indices, v3_plan)

        count = min(swap_limit, len(additions))
        efficiency_floor_key = (
            "certv8_long_d_efficiency_floor"
            if shape_mode
            else "certv8_d_efficiency_floor"
        )
        efficiency_floor = min(
            1.0,
            max(
                0.0,
                _cfg_float(
                    config,
                    efficiency_floor_key,
                    0.98,
                ),
            ),
        )
        diagnostics["d_efficiency_floor"] = efficiency_floor
        minimum_gain = max(0.0, _cfg_float(config, "certv8_min_relation_gain", 0.002))
        selected: Optional[torch.Tensor] = None
        final_coverage: Optional[torch.Tensor] = None
        final_objective = base_objective
        d_efficiency = 1.0
        while count > 0:
            trial = _trial_selection(v3_indices, additions, removals, count)
            trial_efficiency = _logdet_efficiency(
                analysis.design,
                v3_indices,
                trial,
                analysis.ridge,
            )
            trial_coverage = _same_frame_coverage(
                analysis.metric_flat,
                trial,
                frame_count,
                tokens_per_frame,
            )
            trial_objective = _relation_objective(relations, trial_coverage)
            trial_shape_error = (
                _frame_shape_error(trial, target_counts, tokens_per_frame)
                if target_counts is not None
                else 0.0
            )
            if (
                trial_efficiency + 1e-12 >= efficiency_floor
                and trial_objective >= base_objective + minimum_gain
                and (
                    target_counts is None
                    or trial_shape_error + 1e-12 < base_shape_error
                )
            ):
                selected = trial
                final_coverage = trial_coverage
                final_objective = trial_objective
                d_efficiency = trial_efficiency
                final_shape_error = trial_shape_error
                break
            next_count = int(math.floor(0.75 * count))
            count = next_count if next_count < count else count - 1
        if selected is None or final_coverage is None:
            return _fallback(config, diagnostics, "design_or_gain_guard", v3_output, v3_indices, v3_plan)

        accepted_additions = set(additions[:count])
        plan = _build_local_plan(
            selected,
            accepted_additions,
            protected,
            analysis,
            frame_times,
            has_real_times,
            frame_count,
            tokens_per_frame,
            config,
        )
        source_frames = torch.arange(frame_count, device=selected.device).repeat_interleave(
            tokens_per_frame
        )
        anchor_frames = source_frames[selected]
        assigned_frames = anchor_frames[plan.assignment_indices]
        frame_delta = (source_frames.unsqueeze(1) - assigned_frames).abs()
        unsafe_assignments = int((frame_delta > 1).sum().item())
        if unsafe_assignments:
            raise RuntimeError("V8 generated a non-local residual assignment")

        output = apply_certvid_plan(
            video_features.reshape(-1, video_features.shape[-1]),
            plan,
        )
        diagnostics.update(
            {
                "fallback_reason": None,
                "swap_count": count,
                "modified_ratio": float(count / max(1, int(v3_indices.numel()))),
                "v3_overlap_ratio": float(
                    (int(v3_indices.numel()) - count) / max(1, int(v3_indices.numel()))
                ),
                "d_efficiency": d_efficiency,
                "final_relation_coverage": final_objective,
                "relation_gain": final_objective - base_objective,
                "final_frame_shape_error": final_shape_error,
                "unsafe_assignment_count": unsafe_assignments,
                "max_assignment_frame_delta": int(frame_delta.max().item()),
                "frame_budget_min": int(
                    torch.bincount(selected // tokens_per_frame, minlength=frame_count).min().item()
                ),
                "frame_budget_max": int(
                    torch.bincount(selected // tokens_per_frame, minlength=frame_count).max().item()
                ),
                "final_frame_counts": [
                    int(value)
                    for value in torch.bincount(
                        selected // tokens_per_frame,
                        minlength=frame_count,
                    )
                    .detach()
                    .cpu()
                    .tolist()
                ],
            }
        )
        config._certvid_plan = plan
        config.vision_token_length = int(output.shape[0])
        config.visual_token_length = int(output.shape[0])
        config.llm_token_length = None
        config.last_adapter_variant = "certvid_v8"
        config.last_adapter_raw_tokens = float(frame_count * tokens_per_frame)
        config.last_adapter_output_tokens = float(output.shape[0])
        _store_diagnostics(config, diagnostics)
        return output, selected
    except (RuntimeError, ValueError, IndexError) as error:
        diagnostics["error"] = f"{type(error).__name__}: {error}"
        return _fallback(config, diagnostics, "optimization_error", v3_output, v3_indices, v3_plan)
