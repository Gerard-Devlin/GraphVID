"""CertVID V7: relation-preserving compression for long-horizon video.

Short or untimed inputs return the exact CertVID V3 result. For reliable
long-horizon inputs, V7 keeps only a small V3 certificate subset and spends a
fixed in-budget relation reserve on query context, transition endpoint pairs,
and persistent trajectory chains. The remaining tokens are selected by
temporally balanced per-frame facility location. Residual fusion is restricted
to the same frame or a demonstrably safe adjacent-frame transition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .certvid import (
    CertVidPlan,
    _cfg_float,
    _cfg_int,
    _grid_hw,
    _minmax,
    apply_certvid_plan,
)
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


@dataclass(frozen=True)
class _RelayGroup:
    score: float
    tokens: Tuple[int, ...]
    kind: str


def _analysis_from_sink(sink: Dict[str, Any]) -> _V3Analysis:
    required = (
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
    missing = [name for name in required if name not in sink]
    if missing:
        raise ValueError(f"V3 analysis is missing: {', '.join(missing)}")
    return _V3Analysis(**{name: sink[name] for name in required})


def _normalize_frame_times(
    raw_times: Any,
    frame_count: int,
    *,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    if raw_times is None:
        return None, "missing_timestamps"
    try:
        times = torch.as_tensor(raw_times, dtype=torch.float64, device=device).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        return None, "invalid_timestamp_type"
    if times.numel() == 0 or frame_count <= 0:
        return None, "empty_timestamps"
    if not bool(torch.isfinite(times).all()):
        return None, "nonfinite_timestamps"
    if times.numel() == frame_count:
        mapped = times
    elif times.numel() > frame_count and times.numel() % frame_count == 0:
        mapped = times.reshape(frame_count, -1).mean(dim=1)
    else:
        return None, "timestamp_length_mismatch"
    if mapped.numel() > 1 and bool(torch.any(mapped[1:] < mapped[:-1])):
        return None, "nonmonotonic_timestamps"
    return mapped.float(), None


def _effective_ratio(config: Any) -> float:
    ratio = _cfg_float(config, "retention_ratio", 0.10)
    if bool(getattr(config, "certv3_budget_uses_expansion", True)):
        ratio *= _cfg_float(config, "expansion", 1.0)
    return min(1.0, max(0.0, ratio))


def _long_horizon_budget(
    frame_count: int,
    tokens_per_frame: int,
    config: Any,
) -> Tuple[int, int, str]:
    """Return target, native V3 budget, and the applied rounding policy."""

    total_tokens = int(frame_count * tokens_per_frame)
    ratio = _effective_ratio(config)
    native_budget = max(1, min(total_tokens, int(round(total_tokens * ratio))))
    mode = str(getattr(config, "certv7_budget_rounding", "per_frame_ceil")).strip().lower()
    if mode == "global_round":
        return native_budget, native_budget, mode
    if mode != "per_frame_ceil":
        raise ValueError(
            f"unsupported certv7_budget_rounding={mode!r}; expected per_frame_ceil|global_round"
        )
    per_frame = max(1, min(tokens_per_frame, int(math.ceil(tokens_per_frame * ratio))))
    return frame_count * per_frame, native_budget, mode


def _v3_with_target_budget(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    config: Any,
    question_features: Optional[torch.Tensor],
    analysis_sink: Optional[Dict[str, Any]],
    target_budget: int,
    native_budget: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if target_budget == native_budget:
        return certvid_v3_compression(
            video_features,
            cls_attention,
            config,
            question_features,
            analysis_sink=analysis_sink,
        )

    total_tokens = int(video_features.shape[0] * video_features.shape[1])
    expansion = (
        _cfg_float(config, "expansion", 1.0)
        if bool(getattr(config, "certv3_budget_uses_expansion", True))
        else 1.0
    )
    if expansion <= 0.0:
        raise ValueError("expansion must be positive when matching a per-frame token budget")
    original_ratio = getattr(config, "retention_ratio", 0.10)
    config.retention_ratio = (target_budget / max(1, total_tokens)) / expansion
    try:
        return certvid_v3_compression(
            video_features,
            cls_attention,
            config,
            question_features,
            analysis_sink=analysis_sink,
        )
    finally:
        config.retention_ratio = original_ratio


def _spatial_prototypes(
    metric_frames: torch.Tensor,
    demand_frames: torch.Tensor,
    height: int,
    width: int,
    bins: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool each frame into demand-weighted spatial transport prototypes."""

    frame_count, tokens_per_frame, _ = metric_frames.shape
    row_bins = min(max(1, int(bins)), height)
    col_bins = min(max(1, int(bins)), width)
    token_ids = torch.arange(tokens_per_frame, device=metric_frames.device)
    rows = torch.div(token_ids, width, rounding_mode="floor").clamp_max(height - 1)
    cols = torch.remainder(token_ids, width).clamp_max(width - 1)
    row_ids = torch.div(rows * row_bins, max(1, height), rounding_mode="floor").clamp_max(row_bins - 1)
    col_ids = torch.div(cols * col_bins, max(1, width), rounding_mode="floor").clamp_max(col_bins - 1)
    cell_ids = row_ids * col_bins + col_ids
    cell_count = row_bins * col_bins

    prototypes = []
    masses = []
    coordinates = []
    for cell in range(cell_count):
        members = torch.where(cell_ids == cell)[0]
        if members.numel() == 0:
            continue
        weights = demand_frames[:, members].float().clamp_min(1e-8)
        mass = weights.sum(dim=1)
        pooled = torch.sum(metric_frames[:, members].float() * weights.unsqueeze(-1), dim=1)
        pooled = pooled / mass.unsqueeze(1).clamp_min(1e-8)
        prototypes.append(F.normalize(pooled, p=2, dim=-1, eps=1e-6))
        masses.append(mass)
        coordinates.append(
            torch.tensor(
                [
                    (float(rows[members].float().mean().item()) + 0.5) / max(1, height),
                    (float(cols[members].float().mean().item()) + 0.5) / max(1, width),
                ],
                dtype=torch.float32,
                device=metric_frames.device,
            )
        )
    if not prototypes:
        raise RuntimeError("transport prototype construction produced no cells")
    prototype_tensor = torch.stack(prototypes, dim=1)
    mass_tensor = torch.stack(masses, dim=1)
    mass_tensor = mass_tensor / mass_tensor.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return prototype_tensor, mass_tensor, torch.stack(coordinates, dim=0)


def _sinkhorn_transport_costs(
    prototypes: torch.Tensor,
    masses: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    epsilon: float,
    steps: int,
    spatial_weight: float,
) -> torch.Tensor:
    """Return one entropic transport cost for every adjacent frame pair."""

    if prototypes.shape[0] <= 1:
        return prototypes.new_empty((0,), dtype=torch.float32)
    left = prototypes[:-1].float()
    right = prototypes[1:].float()
    feature_cost = 0.5 * (
        1.0 - torch.bmm(left, right.transpose(1, 2)).clamp(-1.0, 1.0)
    )
    spatial = torch.cdist(coordinates.float(), coordinates.float(), p=2)
    spatial = spatial / math.sqrt(2.0)
    frame_gap = 1.0 - torch.sum(left.mean(dim=1) * right.mean(dim=1), dim=-1).clamp(-1.0, 1.0)
    semantic_weight = (0.60 + 0.30 * (0.5 * frame_gap).clamp(0.0, 1.0)).view(-1, 1, 1)
    spatial_share = min(0.45, max(0.0, float(spatial_weight)))
    semantic_weight = torch.maximum(semantic_weight, feature_cost.new_tensor(1.0 - spatial_share))
    cost = semantic_weight * feature_cost + (1.0 - semantic_weight) * spatial.unsqueeze(0)

    source = masses[:-1].float()
    target = masses[1:].float()
    epsilon = max(1e-3, float(epsilon))
    log_kernel = -cost / epsilon
    log_source = source.clamp_min(1e-20).log()
    log_target = target.clamp_min(1e-20).log()
    log_u = torch.zeros_like(log_source)
    log_v = torch.zeros_like(log_target)
    for _ in range(max(1, int(steps))):
        log_v = log_target - torch.logsumexp(log_kernel + log_u.unsqueeze(2), dim=1)
        log_u = log_source - torch.logsumexp(log_kernel + log_v.unsqueeze(1), dim=2)
    transport = torch.exp(log_u.unsqueeze(2) + log_kernel + log_v.unsqueeze(1))
    values = torch.sum(transport * cost, dim=(1, 2))
    if not bool(torch.isfinite(values).all()):
        raise RuntimeError("non-finite temporal transport cost")
    return values


def _temporal_signals(
    metric_frames: torch.Tensor,
    pair_costs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Frame uniqueness, transition stress, event strength, token novelty."""

    frame_count, tokens_per_frame, _ = metric_frames.shape
    frame_reps = F.normalize(metric_frames.float().mean(dim=1), p=2, dim=-1, eps=1e-6)
    if frame_count <= 1:
        zeros = metric_frames.new_zeros((frame_count,), dtype=torch.float32)
        return zeros, zeros, zeros, metric_frames.new_zeros((frame_count, tokens_per_frame))

    global_similarity = 0.5 * (frame_reps @ frame_reps.transpose(0, 1) + 1.0)
    global_similarity.fill_diagonal_(-1.0)
    neighbors = min(3, frame_count - 1)
    uniqueness = 1.0 - torch.topk(global_similarity, k=neighbors, dim=1).values.mean(dim=1)
    uniqueness = _minmax(uniqueness, dim=0)

    adjacent_gap = 0.5 * (
        1.0 - torch.sum(frame_reps[:-1] * frame_reps[1:], dim=-1).clamp(-1.0, 1.0)
    )
    event = torch.zeros(frame_count, dtype=torch.float32, device=metric_frames.device)
    event[:-1] = torch.maximum(event[:-1], adjacent_gap)
    event[1:] = torch.maximum(event[1:], adjacent_gap)
    if frame_count > 2:
        incoming = frame_reps[1:-1] - frame_reps[:-2]
        outgoing = frame_reps[2:] - frame_reps[1:-1]
        curvature = 0.5 * (
            1.0 - F.cosine_similarity(incoming, outgoing, dim=-1, eps=1e-6)
        )
        event[1:-1] = torch.maximum(event[1:-1], curvature)
    event = _minmax(event, dim=0)

    pair_stress = _minmax(pair_costs, dim=0) if pair_costs.numel() else pair_costs
    transport = torch.zeros(frame_count, dtype=torch.float32, device=metric_frames.device)
    if pair_stress.numel():
        transport[:-1] = torch.maximum(transport[:-1], pair_stress)
        transport[1:] = torch.maximum(transport[1:], pair_stress)

    novelty = torch.zeros((frame_count, tokens_per_frame), dtype=torch.float32, device=metric_frames.device)
    for frame in range(frame_count - 1):
        similarity = metric_frames[frame].float() @ metric_frames[frame + 1].float().transpose(0, 1)
        forward = 0.5 * (1.0 - similarity.max(dim=1).values.clamp(-1.0, 1.0))
        backward = 0.5 * (1.0 - similarity.max(dim=0).values.clamp(-1.0, 1.0))
        novelty[frame] = torch.maximum(novelty[frame], forward)
        novelty[frame + 1] = torch.maximum(novelty[frame + 1], backward)
    novelty = _minmax(novelty, dim=1)
    return uniqueness, transport, event, novelty


def _frame_query_signal(analysis: _V3Analysis, frame_count: int, tokens_per_frame: int) -> torch.Tensor:
    if analysis.query_relevance.numel() == 0:
        return analysis.metric_flat.new_zeros((frame_count,), dtype=torch.float32)
    relevance = analysis.query_relevance.reshape(-1, frame_count, tokens_per_frame)
    return _minmax(relevance.amax(dim=(0, 2)), dim=0)


def _query_relay_groups(
    analysis: _V3Analysis,
    metric_frames: torch.Tensor,
    frame_count: int,
    tokens_per_frame: int,
    *,
    peaks_per_atom: int,
    min_frame_gap: int,
    threshold: float,
    context_radius: int,
) -> list[_RelayGroup]:
    if analysis.query_relevance.numel() == 0 or analysis.query_confidence <= 1e-6:
        return []
    groups: list[_RelayGroup] = []
    relevance = analysis.query_relevance.reshape(-1, frame_count, tokens_per_frame)
    for atom_scores in relevance:
        frame_scores, local_tokens = atom_scores.max(dim=1)
        chosen_frames: list[int] = []
        order = torch.argsort(frame_scores, descending=True, stable=True).tolist()
        for frame in order:
            if float(frame_scores[frame].item()) < float(threshold):
                continue
            if any(abs(frame - previous) < max(1, int(min_frame_gap)) for previous in chosen_frames):
                continue
            local = int(local_tokens[frame].item())
            tokens = [frame * tokens_per_frame + local]
            radius = max(0, int(context_radius))
            for offset in range(-radius, radius + 1):
                neighbor = frame + offset
                if offset == 0 or not 0 <= neighbor < frame_count:
                    continue
                similarity = metric_frames[neighbor].float() @ metric_frames[frame, local].float()
                neighbor_local = int(torch.argmax(similarity).item())
                tokens.append(neighbor * tokens_per_frame + neighbor_local)
            groups.append(
                _RelayGroup(
                    score=float(frame_scores[frame].item()),
                    tokens=tuple(sorted(set(tokens))),
                    kind="query",
                )
            )
            chosen_frames.append(frame)
            if len(chosen_frames) >= max(1, int(peaks_per_atom)):
                break
    groups.sort(key=lambda group: (-group.score, group.tokens))
    deduplicated: list[_RelayGroup] = []
    seen: set[Tuple[int, ...]] = set()
    for group in groups:
        if group.tokens not in seen:
            deduplicated.append(group)
            seen.add(group.tokens)
    return deduplicated


def _transition_relay_groups(
    metric_frames: torch.Tensor,
    demand_frames: torch.Tensor,
    novelty: torch.Tensor,
    pair_costs: torch.Tensor,
    *,
    pairs_per_boundary: int,
    min_similarity: float,
) -> list[_RelayGroup]:
    frame_count, tokens_per_frame = novelty.shape
    if frame_count <= 1:
        return []
    stress = _minmax(pair_costs, dim=0) if pair_costs.numel() else novelty.new_zeros((frame_count - 1,))
    groups: list[_RelayGroup] = []
    for pair in range(frame_count - 1):
        similarity = metric_frames[pair].float() @ metric_frames[pair + 1].float().transpose(0, 1)
        right_similarity, right_index = similarity.max(dim=1)
        left_index = similarity.argmax(dim=0)
        mutual = left_index[right_index] == torch.arange(tokens_per_frame, device=similarity.device)
        right_novelty = novelty[pair + 1, right_index]
        right_demand = _minmax(demand_frames[pair + 1], dim=0)[right_index]
        state_change = 0.5 * (1.0 - right_similarity.clamp(-1.0, 1.0))
        score = (
            0.30 * novelty[pair]
            + 0.30 * right_novelty
            + 0.20 * state_change
            + 0.10 * _minmax(demand_frames[pair], dim=0)
            + 0.10 * right_demand
            + 0.08 * mutual.float()
        )
        score *= 0.35 + 0.65 * stress[pair]
        score = score.masked_fill(right_similarity < float(min_similarity), float("-inf"))
        count = min(max(1, int(pairs_per_boundary)), tokens_per_frame)
        values, left_tokens = torch.topk(score, k=count, largest=True, sorted=True)
        for value, left_local in zip(values.tolist(), left_tokens.tolist()):
            if not math.isfinite(float(value)):
                continue
            right_local = int(right_index[left_local].item())
            groups.append(
                _RelayGroup(
                    score=float(value),
                    tokens=(
                        pair * tokens_per_frame + int(left_local),
                        (pair + 1) * tokens_per_frame + right_local,
                    ),
                    kind="transition",
                )
            )
    groups.sort(key=lambda group: (-group.score, group.tokens))
    return groups


def _trajectory_relay_groups(
    analysis: _V3Analysis,
    novelty: torch.Tensor,
    frame_count: int,
    tokens_per_frame: int,
    *,
    min_span: int,
    points_per_track: int,
) -> list[_RelayGroup]:
    component_ids = analysis.component_ids.detach().cpu().tolist()
    metric_cpu = analysis.metric_flat.detach().float().cpu()
    demand_cpu = analysis.demand_weight.detach().float().cpu().tolist()
    query_cpu = analysis.query_score.detach().float().cpu().tolist()
    novelty_cpu = novelty.reshape(-1).detach().float().cpu().tolist()
    buckets: dict[int, list[int]] = {}
    for token, component in enumerate(component_ids):
        buckets.setdefault(int(component), []).append(token)

    groups: list[_RelayGroup] = []
    required_span = max(2, int(min_span))
    point_limit = max(2, int(points_per_track))
    for members in buckets.values():
        by_frame: dict[int, list[int]] = {}
        for token in members:
            by_frame.setdefault(token // tokens_per_frame, []).append(token)
        ordered_frames = sorted(by_frame)
        if len(ordered_frames) < 2 or ordered_frames[-1] - ordered_frames[0] + 1 < required_span:
            continue

        representatives: list[int] = []
        for frame in ordered_frames:
            representative = max(
                by_frame[frame],
                key=lambda token: (
                    0.45 * demand_cpu[token]
                    + 0.30 * novelty_cpu[token]
                    + 0.25 * query_cpu[token],
                    -token,
                ),
            )
            representatives.append(representative)

        chosen = [representatives[0], representatives[-1]]
        if point_limit > 2 and len(representatives) > 2:
            interior = representatives[1:-1]
            interior.sort(
                key=lambda token: (
                    -(0.45 * novelty_cpu[token] + 0.30 * query_cpu[token] + 0.25 * demand_cpu[token]),
                    token,
                )
            )
            chosen.extend(interior[: point_limit - 2])
        chosen = sorted(set(chosen))
        first, last = chosen[0], chosen[-1]
        endpoint_change = 0.5 * (
            1.0 - float(torch.sum(metric_cpu[first] * metric_cpu[last]).clamp(-1.0, 1.0).item())
        )
        span = (last // tokens_per_frame - first // tokens_per_frame) / max(1, frame_count - 1)
        evidence = max(0.45 * novelty_cpu[token] + 0.30 * query_cpu[token] + 0.25 * demand_cpu[token] for token in chosen)
        groups.append(
            _RelayGroup(
                score=0.45 * float(span) + 0.30 * endpoint_change + 0.25 * evidence,
                tokens=tuple(chosen),
                kind="trajectory",
            )
        )
    groups.sort(key=lambda group: (-group.score, group.tokens))
    return groups


def _v3_certificate_candidates(
    v3_plan: CertVidPlan,
    analysis: _V3Analysis,
) -> list[int]:
    protected = v3_plan.anchor_indices[v3_plan.fusion_alpha <= 1e-12]
    if protected.numel() == 0:
        return []
    leverage = _minmax(analysis.design[protected].float().square().sum(dim=1), dim=0)
    score = (
        0.45 * leverage
        + 0.25 * _minmax(analysis.demand_weight[protected], dim=0)
        + 0.18 * analysis.query_score[protected]
        + 0.12 * analysis.attention[protected]
    )
    order = torch.argsort(score, descending=True, stable=True)
    return [int(token) for token in protected[order].tolist()]


def _compose_mandatory(
    v3_plan: CertVidPlan,
    analysis: _V3Analysis,
    query_groups: list[_RelayGroup],
    transition_groups: list[_RelayGroup],
    trajectory_groups: list[_RelayGroup],
    *,
    budget: int,
    frame_count: int,
    tokens_per_frame: int,
    frame_capacity: int,
    v3_certificate_ratio: float,
    relay_ratio: float,
    query_share: float,
    transition_share: float,
) -> Tuple[list[int], Dict[str, int]]:
    certificate_ratio = min(0.25, max(0.0, float(v3_certificate_ratio)))
    certificate_limit = min(budget, max(0, int(round(budget * certificate_ratio))))
    frame_capacity = min(tokens_per_frame, max(1, int(frame_capacity)))
    frame_load = [0 for _ in range(frame_count)]
    certificates: list[int] = []
    for token in _v3_certificate_candidates(v3_plan, analysis):
        frame = token // tokens_per_frame
        if not 0 <= frame < frame_count or frame_load[frame] >= frame_capacity:
            continue
        certificates.append(token)
        frame_load[frame] += 1
        if len(certificates) >= certificate_limit:
            break
    mandatory = list(certificates)
    mandatory_set = set(mandatory)
    relay_limit = min(
        max(0, budget - len(mandatory)),
        max(0, int(round(budget * min(0.60, max(0.0, float(relay_ratio)))))),
    )
    query_share = min(1.0, max(0.0, float(query_share)))
    transition_share = min(1.0 - query_share, max(0.0, float(transition_share)))
    trajectory_share = max(0.0, 1.0 - query_share - transition_share)
    categories = {
        "query": query_groups,
        "transition": transition_groups,
        "trajectory": trajectory_groups,
    }
    quotas = {
        "query": int(round(relay_limit * query_share)),
        "transition": int(round(relay_limit * transition_share)),
    }
    quotas["trajectory"] = max(0, relay_limit - quotas["query"] - quotas["transition"])
    if trajectory_share <= 0.0:
        quotas["transition"] += quotas["trajectory"]
        quotas["trajectory"] = 0

    admitted = {name: 0 for name in categories}
    cursors = {name: 0 for name in categories}

    def admit(group: _RelayGroup, room: int) -> int:
        fresh = list(dict.fromkeys(token for token in group.tokens if token not in mandatory_set))
        if not fresh or len(fresh) > room or len(mandatory) + len(fresh) > budget:
            return 0
        increments: Dict[int, int] = {}
        for token in fresh:
            frame = token // tokens_per_frame
            if not 0 <= frame < frame_count:
                return 0
            increments[frame] = increments.get(frame, 0) + 1
        if any(frame_load[frame] + count > frame_capacity for frame, count in increments.items()):
            return 0
        mandatory.extend(fresh)
        mandatory_set.update(fresh)
        for frame, count in increments.items():
            frame_load[frame] += count
        return len(fresh)

    for name, groups in categories.items():
        room = quotas[name]
        while cursors[name] < len(groups) and room > 0:
            group = groups[cursors[name]]
            cursors[name] += 1
            used = admit(group, room)
            room -= used
            admitted[name] += used

    remaining = relay_limit - sum(admitted.values())
    while remaining > 0:
        progressed = False
        for name, groups in categories.items():
            while cursors[name] < len(groups):
                group = groups[cursors[name]]
                cursors[name] += 1
                used = admit(group, remaining)
                if used > 0:
                    admitted[name] += used
                    remaining -= used
                    progressed = True
                    break
            if remaining <= 0:
                break
        if not progressed:
            break

    diagnostics = {
        "v3_certificate_count": len(certificates),
        "query_relay_count": admitted["query"],
        "transition_relay_count": admitted["transition"],
        "trajectory_relay_count": admitted["trajectory"],
        "query_group_offered": len(query_groups),
        "transition_group_offered": len(transition_groups),
        "trajectory_group_offered": len(trajectory_groups),
        "mandatory_frame_cap": frame_capacity,
        "mandatory_frame_max": max(frame_load, default=0),
    }
    return mandatory, diagnostics


def _allocate_frame_budget(
    budget: int,
    tokens_per_frame: int,
    mandatory: list[int],
    importance: torch.Tensor,
    *,
    floor_ratio: float,
    cap_ratio: float,
    temperature: float,
) -> torch.Tensor:
    frame_count = int(importance.numel())
    mandatory_counts = torch.zeros(frame_count, dtype=torch.long, device=importance.device)
    if mandatory:
        ids = torch.tensor(mandatory, dtype=torch.long, device=importance.device) // tokens_per_frame
        mandatory_counts = torch.bincount(ids, minlength=frame_count)
    average = budget / max(1, frame_count)
    floor = max(1, int(math.floor(average * min(1.0, max(0.0, float(floor_ratio))))))
    quotas = torch.maximum(mandatory_counts, torch.full_like(mandatory_counts, floor))
    if int(quotas.sum().item()) > budget:
        quotas = mandatory_counts.clone()
        for frame in torch.argsort(importance, descending=True, stable=True).tolist():
            if int(quotas.sum().item()) >= budget:
                break
            if quotas[frame] == 0:
                quotas[frame] = 1
    if int(quotas.sum().item()) > budget:
        raise RuntimeError("mandatory relay set exceeds the token budget")

    remaining = budget - int(quotas.sum().item())
    frame_cap = min(
        tokens_per_frame,
        max(floor, int(math.ceil(average * max(1.0, float(cap_ratio))))),
    )
    capacity = torch.maximum(
        mandatory_counts,
        torch.full_like(quotas, frame_cap),
    ).clamp_max(tokens_per_frame)
    temperature = max(1e-3, float(temperature))
    while remaining > 0:
        active = quotas < capacity
        if not bool(active.any()):
            break
        active_ids = torch.where(active)[0]
        weights = torch.softmax(importance[active] / temperature, dim=0)
        proposal = weights * remaining
        additions = torch.floor(proposal).long()
        room = capacity[active] - quotas[active]
        additions = torch.minimum(additions, room)
        if int(additions.sum().item()) == 0:
            fractional = proposal - torch.floor(proposal)
            order = torch.argsort(fractional, descending=True, stable=True)
            take = min(remaining, int(order.numel()))
            additions[order[:take]] = 1
            additions = torch.minimum(additions, room)
        quotas[active_ids] += additions
        used = int(additions.sum().item())
        if used <= 0:
            break
        remaining -= used
    if int(quotas.sum().item()) != budget:
        raise RuntimeError(f"frame allocator produced {int(quotas.sum().item())}/{budget} tokens")
    return quotas


def _facility_select_frame(
    metric: torch.Tensor,
    demand: torch.Tensor,
    priority: torch.Tensor,
    mandatory_local: list[int],
    quota: int,
    *,
    quality_mix: float,
) -> list[int]:
    token_count = int(metric.shape[0])
    quota = min(token_count, max(len(mandatory_local), int(quota)))
    if quota >= token_count:
        return list(range(token_count))
    similarity = 0.5 * (metric.float() @ metric.float().transpose(0, 1) + 1.0)
    weights = demand.float().clamp_min(1e-8)
    weights = weights / weights.sum().clamp_min(1e-8)
    selected = list(dict.fromkeys(int(token) for token in mandatory_local))
    selected_mask = torch.zeros(token_count, dtype=torch.bool, device=metric.device)
    if selected:
        selected_tensor = torch.tensor(selected, dtype=torch.long, device=metric.device)
        selected_mask[selected_tensor] = True
        coverage = similarity[:, selected_tensor].amax(dim=1)
    else:
        coverage = torch.zeros(token_count, dtype=torch.float32, device=metric.device)
    mix = min(0.50, max(0.0, float(quality_mix)))
    priority = _minmax(priority.float(), dim=0)
    while len(selected) < quota:
        gain = torch.sum(
            (similarity - coverage.unsqueeze(1)).clamp_min(0.0) * weights.unsqueeze(1),
            dim=0,
        )
        score = (1.0 - mix) * gain + mix * priority
        score = score.masked_fill(selected_mask, float("-inf"))
        token = int(torch.argmax(score).item())
        if not math.isfinite(float(score[token].item())):
            token = int(torch.where(~selected_mask)[0][0].item())
        selected.append(token)
        selected_mask[token] = True
        coverage = torch.maximum(coverage, similarity[:, token])
    return selected


def _select_anchors(
    analysis: _V3Analysis,
    mandatory: list[int],
    quotas: torch.Tensor,
    novelty: torch.Tensor,
    frame_query: torch.Tensor,
    *,
    quality_mix: float,
) -> torch.Tensor:
    frame_count = int(quotas.numel())
    tokens_per_frame = int(analysis.metric_flat.shape[0] // frame_count)
    metric_frames = analysis.metric_flat.reshape(frame_count, tokens_per_frame, -1)
    demand_frames = analysis.demand_weight.reshape(frame_count, tokens_per_frame)
    attention_frames = analysis.attention.reshape(frame_count, tokens_per_frame)
    query_frames = analysis.query_score.reshape(frame_count, tokens_per_frame)
    mandatory_by_frame: list[list[int]] = [[] for _ in range(frame_count)]
    for token in mandatory:
        mandatory_by_frame[token // tokens_per_frame].append(token % tokens_per_frame)

    selected: list[int] = []
    for frame in range(frame_count):
        local_priority = (
            0.34 * _minmax(demand_frames[frame], dim=0)
            + 0.18 * attention_frames[frame]
            + 0.30 * novelty[frame]
            + 0.18 * query_frames[frame] * frame_query[frame]
        )
        local = _facility_select_frame(
            metric_frames[frame],
            demand_frames[frame],
            local_priority,
            mandatory_by_frame[frame],
            int(quotas[frame].item()),
            quality_mix=quality_mix,
        )
        selected.extend(frame * tokens_per_frame + token for token in local)
    result = torch.tensor(sorted(selected), dtype=torch.long, device=analysis.metric_flat.device)
    if result.numel() != int(quotas.sum().item()) or result.unique().numel() != result.numel():
        raise RuntimeError("facility selector violated the fixed unique-token budget")
    return result


def _logdet_efficiency(
    design: torch.Tensor,
    baseline: torch.Tensor,
    selected: torch.Tensor,
    ridge: float,
) -> float:
    dimension = int(design.shape[1])
    identity = torch.eye(dimension, dtype=torch.float32, device=design.device)
    ridge = max(1e-4, float(ridge))

    def logdet(indices: torch.Tensor) -> torch.Tensor:
        rows = design[indices].float()
        information = ridge * identity + rows.transpose(0, 1) @ rows
        sign, value = torch.linalg.slogdet(information)
        if float(sign.item()) <= 0.0:
            raise RuntimeError("non-positive design information matrix")
        return value

    difference = (logdet(selected) - logdet(baseline)) / max(1, dimension)
    return float(torch.exp(difference.clamp(max=20.0)).item())


def _coverage_score(metric: torch.Tensor, selected: torch.Tensor, demand: torch.Tensor) -> float:
    similarity = 0.5 * (metric.float() @ metric[selected].float().transpose(0, 1) + 1.0)
    coverage = similarity.amax(dim=1)
    weights = demand.float() / demand.float().sum().clamp_min(1e-8)
    return float(torch.sum(coverage * weights).item())


def _build_local_plan(
    selected: torch.Tensor,
    mandatory: list[int],
    analysis: _V3Analysis,
    frame_times: torch.Tensor,
    pair_costs: torch.Tensor,
    *,
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

    if pair_costs.numel() and frame_count > 1:
        normalized_cost = _minmax(pair_costs, dim=0)
        safe_quantile = min(1.0, max(0.0, _cfg_float(config, "certv7_cross_frame_cost_quantile", 0.45)))
        cost_limit = float(torch.quantile(normalized_cost, safe_quantile).item())
        adjacent = frame_delta == 1
        earlier = torch.minimum(source_frame.unsqueeze(1), anchor_frame.unsqueeze(0)).clamp_max(frame_count - 2)
        pair_safe = normalized_cost[earlier] <= cost_limit + 1e-8
        time_gap = torch.abs(frame_times[source_frame].unsqueeze(1) - frame_times[anchor_frame].unsqueeze(0))
        cross_safe = (
            adjacent
            & pair_safe
            & (time_gap <= _cfg_float(config, "certv7_cross_frame_max_seconds", 12.0))
            & (similarity >= _cfg_float(config, "certv7_cross_frame_similarity", 0.82))
        )
        valid = valid | cross_safe
    similarity = similarity.masked_fill(~valid, -2.0)
    same_component = analysis.component_ids.unsqueeze(1) == analysis.component_ids[selected].unsqueeze(0)
    similarity = similarity + _cfg_float(config, "certv7_component_bonus", 0.08) * same_component.float()

    topk = min(max(1, _cfg_int(config, "certv7_assignment_topk", 2)), budget)
    values, assignment = torch.topk(similarity, k=topk, dim=1, largest=True)
    weights = torch.softmax(
        values.float() / max(1e-4, _cfg_float(config, "certv7_assignment_temperature", 0.07)),
        dim=1,
    )
    anchor_positions = torch.arange(budget, dtype=torch.long, device=selected.device)
    assignment[selected, 0] = anchor_positions
    weights[selected] = 0.0
    weights[selected, 0] = 1.0

    source_mass = (0.5 + 0.5 * analysis.demand_weight * total_tokens).clamp(0.25, 2.0)
    alpha_value = min(0.25, max(0.0, _cfg_float(config, "certv7_long_fusion_alpha", 0.04)))
    alpha = torch.full((budget,), alpha_value, dtype=torch.float32, device=selected.device)
    protected = torch.zeros(budget, dtype=torch.bool, device=selected.device)
    if mandatory:
        mandatory_tensor = torch.tensor(mandatory, dtype=torch.long, device=selected.device)
        protected |= torch.isin(selected, mandatory_tensor)
    protect_ratio = min(0.50, max(0.0, _cfg_float(config, "certv7_design_protect_ratio", 0.15)))
    protect_count = min(budget, int(math.ceil(protect_ratio * budget)))
    if protect_count > 0:
        leverage_proxy = analysis.design[selected].float().square().sum(dim=1)
        protected[torch.topk(leverage_proxy, k=protect_count, largest=True).indices] = True
    alpha[protected] = 0.0
    return CertVidPlan(
        anchor_indices=selected,
        assignment_indices=assignment,
        assignment_weights=weights,
        source_mass=source_mass,
        fusion_alpha=alpha,
        raw_token_count=total_tokens,
    )


def _store_diagnostics(config: Any, diagnostics: Dict[str, Any]) -> None:
    config.last_certv7_diagnostics = diagnostics
    config.last_certv7_timestamp_source = diagnostics.get("timestamp_source", "missing")
    config.last_certv7_fallback_reason = diagnostics.get("fallback_reason")
    config.last_certv7_duration_seconds = float(diagnostics.get("duration_seconds", 0.0))
    config.last_certv7_pair_cost_mean = float(diagnostics.get("pair_cost_mean", 0.0))
    config.last_certv7_pair_cost_max = float(diagnostics.get("pair_cost_max", 0.0))
    config.last_certv7_frame_budget_min = int(diagnostics.get("frame_budget_min", 0))
    config.last_certv7_frame_budget_max = int(diagnostics.get("frame_budget_max", 0))
    config.last_certv7_frame_budget_sum = int(diagnostics.get("frame_budget_sum", 0))
    config.last_certv7_budget_rounding = diagnostics.get("budget_rounding", "global_round")
    config.last_certv7_native_v3_tokens = int(diagnostics.get("native_v3_budget", 0))
    config.last_certv7_target_tokens = int(diagnostics.get("target_budget", 0))
    config.last_certv7_protected_count = int(diagnostics.get("v3_certificate_count", 0))
    config.last_certv7_v3_certificate_count = int(diagnostics.get("v3_certificate_count", 0))
    config.last_certv7_query_relay_count = int(diagnostics.get("query_relay_count", 0))
    config.last_certv7_transition_relay_count = int(diagnostics.get("transition_relay_count", 0))
    config.last_certv7_trajectory_relay_count = int(diagnostics.get("trajectory_relay_count", 0))
    config.last_certv7_boundary_relay_count = int(diagnostics.get("transition_relay_count", 0))
    config.last_certv7_v3_anchor_overlap_ratio = float(diagnostics.get("v3_anchor_overlap_ratio", 1.0))
    config.last_certv7_selection_change_ratio = float(diagnostics.get("selection_change_ratio", 0.0))
    config.last_certv7_d_efficiency = float(diagnostics.get("d_efficiency", 1.0))
    config.last_certv7_base_coverage = float(diagnostics.get("base_coverage", 0.0))
    config.last_certv7_final_coverage = float(diagnostics.get("final_coverage", 0.0))
    config.last_certv7_unsafe_assignment_count = int(diagnostics.get("unsafe_assignment_count", 0))
    # Compatibility with older log collectors that displayed V7 as a repair.
    config.last_certv7_swap_count = int(diagnostics.get("changed_anchor_count", 0))
    config.last_certv7_modified_ratio = float(diagnostics.get("selection_change_ratio", 0.0))
    config.last_certv7_base_path_loss = 0.0
    config.last_certv7_final_path_loss = 0.0
    if bool(getattr(config, "certv7_debug", False)):
        print(
            "[certvid-v7] "
            f"fallback={diagnostics.get('fallback_reason')} "
            f"duration={diagnostics.get('duration_seconds', 0.0):.1f}s "
            f"budget={diagnostics.get('frame_budget_min', 0)}-"
            f"{diagnostics.get('frame_budget_max', 0)} "
            f"relay=q{diagnostics.get('query_relay_count', 0)}/"
            f"t{diagnostics.get('transition_relay_count', 0)}/"
            f"r{diagnostics.get('trajectory_relay_count', 0)} "
            f"v3-overlap={diagnostics.get('v3_anchor_overlap_ratio', 1.0):.3f} "
            f"coverage={diagnostics.get('base_coverage', 0.0):.4f}->"
            f"{diagnostics.get('final_coverage', 0.0):.4f}"
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
    diagnostics.setdefault("d_efficiency", 1.0)
    diagnostics.setdefault("selection_change_ratio", 0.0)
    diagnostics.setdefault("changed_anchor_count", 0)
    if plan is not None:
        config._certvid_plan = plan
    config.last_adapter_variant = "certvid_v7"
    _store_diagnostics(config, diagnostics)
    return output, indices


def certvid_v7_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: Any,
    question_features: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compress long videos with relation evidence and balanced local coverage."""

    if video_features.ndim != 3:
        raise ValueError(f"expected video_features [T, HW, D], got {tuple(video_features.shape)}")
    config = flashvid_config
    frame_count, tokens_per_frame, _ = video_features.shape
    diagnostics: Dict[str, Any] = {
        "timestamp_source": str(getattr(config, "_certvid_frame_times_source", "missing")),
        "fallback_reason": None,
        "duration_seconds": 0.0,
    }
    frame_times, timestamp_error = _normalize_frame_times(
        getattr(config, "_certvid_frame_times_sec", None),
        frame_count,
        device=video_features.device,
    )
    duration = (
        float(frame_times[-1].item() - frame_times[0].item())
        if frame_times is not None and frame_times.numel() > 1
        else 0.0
    )
    diagnostics["duration_seconds"] = duration
    can_analyze = (
        frame_times is not None
        and duration >= _cfg_float(config, "certv7_min_duration_seconds", 120.0)
        and frame_count > 1
    )
    target_budget, native_budget, budget_rounding = _long_horizon_budget(
        frame_count,
        tokens_per_frame,
        config,
    )
    if not can_analyze:
        target_budget = native_budget
        budget_rounding = "global_round"
    diagnostics.update(
        {
            "budget_rounding": budget_rounding,
            "native_v3_budget": native_budget,
            "target_budget": target_budget,
        }
    )
    sink: Optional[Dict[str, Any]] = {} if can_analyze else None
    v3_output, v3_indices = _v3_with_target_budget(
        video_features,
        cls_attention,
        config,
        question_features,
        sink,
        target_budget,
        native_budget,
    )
    v3_plan = getattr(config, "_certvid_plan", None)
    if v3_plan is None:
        return _fallback(config, diagnostics, "missing_v3_plan", v3_output, v3_indices, v3_plan)
    if frame_times is None:
        return _fallback(
            config,
            diagnostics,
            timestamp_error or "invalid_timestamps",
            v3_output,
            v3_indices,
            v3_plan,
        )
    if not can_analyze:
        return _fallback(config, diagnostics, "short_horizon", v3_output, v3_indices, v3_plan)
    if sink is None or bool(sink.get("identity", False)):
        return _fallback(config, diagnostics, "identity_budget", v3_output, v3_indices, v3_plan)

    try:
        analysis = _analysis_from_sink(sink)
        budget = int(v3_indices.numel())
        metric_frames = analysis.metric_flat.reshape(frame_count, tokens_per_frame, -1)
        demand_frames = analysis.demand_weight.reshape(frame_count, tokens_per_frame)
        height, width = _grid_hw(tokens_per_frame, config)
        prototypes, masses, coordinates = _spatial_prototypes(
            metric_frames,
            demand_frames,
            height,
            width,
            _cfg_int(config, "certv7_transport_spatial_bins", 4),
        )
        pair_costs = _sinkhorn_transport_costs(
            prototypes,
            masses,
            coordinates,
            epsilon=_cfg_float(config, "certv7_transport_epsilon", 0.08),
            steps=_cfg_int(config, "certv7_transport_steps", 8),
            spatial_weight=_cfg_float(config, "certv7_transport_spatial_weight", 0.20),
        )
        uniqueness, transport, event, novelty = _temporal_signals(metric_frames, pair_costs)
        frame_query = _frame_query_signal(analysis, frame_count, tokens_per_frame)
        query_gate = min(1.0, max(0.0, float(analysis.query_confidence)))
        weights = torch.tensor(
            [
                max(0.0, _cfg_float(config, "certv7_uniqueness_weight", 0.25)),
                max(0.0, _cfg_float(config, "certv7_transport_weight", 0.35)),
                max(0.0, _cfg_float(config, "certv7_event_weight", 0.20)),
                max(0.0, _cfg_float(config, "certv7_query_weight", 0.20)) * query_gate,
            ],
            dtype=torch.float32,
            device=video_features.device,
        )
        weights = weights / weights.sum().clamp_min(1e-8)
        importance = _minmax(
            weights[0] * uniqueness
            + weights[1] * transport
            + weights[2] * event
            + weights[3] * frame_query,
            dim=0,
        )

        query_groups = _query_relay_groups(
            analysis,
            metric_frames,
            frame_count,
            tokens_per_frame,
            peaks_per_atom=_cfg_int(config, "certv7_query_peaks_per_atom", 2),
            min_frame_gap=_cfg_int(config, "certv7_query_min_frame_gap", 3),
            threshold=_cfg_float(config, "certv7_query_peak_threshold", 0.70),
            context_radius=_cfg_int(config, "certv7_query_context_radius", 1),
        )
        transition_groups = _transition_relay_groups(
            metric_frames,
            demand_frames,
            novelty,
            pair_costs,
            pairs_per_boundary=_cfg_int(config, "certv7_transition_pairs_per_boundary", 2),
            min_similarity=_cfg_float(config, "certv7_transition_min_similarity", 0.30),
        )
        trajectory_groups = _trajectory_relay_groups(
            analysis,
            novelty,
            frame_count,
            tokens_per_frame,
            min_span=_cfg_int(config, "certv7_trajectory_min_span", 3),
            points_per_track=_cfg_int(config, "certv7_trajectory_points", 3),
        )
        mandatory, relation_diagnostics = _compose_mandatory(
            v3_plan,
            analysis,
            query_groups,
            transition_groups,
            trajectory_groups,
            budget=budget,
            frame_count=frame_count,
            tokens_per_frame=tokens_per_frame,
            frame_capacity=min(
                tokens_per_frame,
                max(
                    1,
                    int(
                        math.ceil(
                            (budget / max(1, frame_count))
                            * max(1.0, _cfg_float(config, "certv7_frame_cap_ratio", 1.0))
                        )
                    ),
                ),
            ),
            v3_certificate_ratio=_cfg_float(config, "certv7_v3_certificate_ratio", 0.05),
            relay_ratio=_cfg_float(config, "certv7_relay_ratio", 0.25),
            query_share=_cfg_float(config, "certv7_relay_query_share", 0.25),
            transition_share=_cfg_float(config, "certv7_transition_relay_share", 0.45),
        )
        quotas = _allocate_frame_budget(
            budget,
            tokens_per_frame,
            mandatory,
            importance,
            floor_ratio=_cfg_float(config, "certv7_frame_floor_ratio", 1.0),
            cap_ratio=_cfg_float(config, "certv7_frame_cap_ratio", 1.0),
            temperature=_cfg_float(config, "certv7_budget_temperature", 0.50),
        )
        v3_counts = torch.bincount(v3_indices // tokens_per_frame, minlength=frame_count)
        quota_shift = 0.5 * torch.abs(quotas - v3_counts).sum().float() / max(1, budget)
        certificate_count = relation_diagnostics["v3_certificate_count"]
        relay_tensor = torch.tensor(mandatory[certificate_count:], dtype=torch.long, device=v3_indices.device)
        relay_deficit = (
            float((~torch.isin(relay_tensor, v3_indices)).sum().item()) / max(1, budget)
            if relay_tensor.numel()
            else 0.0
        )
        diagnostics.update(
            {
                "pair_cost_mean": float(pair_costs.mean().item()) if pair_costs.numel() else 0.0,
                "pair_cost_max": float(pair_costs.max().item()) if pair_costs.numel() else 0.0,
                "frame_budget_min": int(quotas.min().item()),
                "frame_budget_max": int(quotas.max().item()),
                "frame_budget_sum": int(quotas.sum().item()),
                "frame_budgets": quotas.detach().cpu().tolist(),
                "v3_frame_budgets": v3_counts.detach().cpu().tolist(),
                "quota_shift_ratio": float(quota_shift.item()),
                "relay_deficit_ratio": relay_deficit,
                **relation_diagnostics,
            }
        )
        min_change = _cfg_float(config, "certv7_min_reallocation_ratio", 0.02)
        if max(float(quota_shift.item()), relay_deficit) < min_change:
            return _fallback(config, diagnostics, "v3_temporally_sufficient", v3_output, v3_indices, v3_plan)

        selected = _select_anchors(
            analysis,
            mandatory,
            quotas,
            novelty,
            frame_query,
            quality_mix=_cfg_float(config, "certv7_facility_quality_mix", 0.18),
        )
        d_efficiency = _logdet_efficiency(
            analysis.design,
            v3_indices,
            selected,
            analysis.ridge,
        )
        diagnostics["d_efficiency"] = d_efficiency
        if d_efficiency + 1e-12 < _cfg_float(config, "certv7_d_efficiency_floor", 0.80):
            return _fallback(config, diagnostics, "design_guard", v3_output, v3_indices, v3_plan)

        base_coverage = _coverage_score(analysis.metric_flat, v3_indices, analysis.demand_weight)
        final_coverage = _coverage_score(analysis.metric_flat, selected, analysis.demand_weight)
        plan = _build_local_plan(
            selected,
            mandatory,
            analysis,
            frame_times,
            pair_costs,
            frame_count=frame_count,
            tokens_per_frame=tokens_per_frame,
            config=config,
        )
        source_frames = torch.arange(frame_count, device=selected.device).repeat_interleave(tokens_per_frame)
        anchor_frames = source_frames[selected]
        assigned_frames = anchor_frames[plan.assignment_indices]
        assignment_frame_delta = (source_frames.unsqueeze(1) - assigned_frames).abs()
        diagnostics.update(
            {
                "cross_frame_assignment_count": int((assignment_frame_delta > 0).sum().item()),
                "unsafe_assignment_count": int((assignment_frame_delta > 1).sum().item()),
                "max_assignment_frame_delta": int(assignment_frame_delta.max().item()),
            }
        )
        output = apply_certvid_plan(video_features.reshape(-1, video_features.shape[-1]), plan)
        changed = int((~torch.isin(selected, v3_indices)).sum().item())
        overlap = budget - changed
        diagnostics.update(
            {
                "fallback_reason": None,
                "base_coverage": base_coverage,
                "final_coverage": final_coverage,
                "changed_anchor_count": changed,
                "selection_change_ratio": float(changed / max(1, budget)),
                "v3_anchor_overlap_count": overlap,
                "v3_anchor_overlap_ratio": float(overlap / max(1, budget)),
            }
        )
        config._certvid_plan = plan
        config.vision_token_length = int(output.shape[0])
        config.visual_token_length = int(output.shape[0])
        config.llm_token_length = None
        config.last_adapter_variant = "certvid_v7"
        config.last_adapter_raw_tokens = float(frame_count * tokens_per_frame)
        config.last_adapter_output_tokens = float(output.shape[0])
        _store_diagnostics(config, diagnostics)
        return output, selected
    except (RuntimeError, ValueError, IndexError) as error:
        config._certvid_plan = v3_plan
        diagnostics["error"] = f"{type(error).__name__}: {error}"
        return _fallback(config, diagnostics, "optimization_error", v3_output, v3_indices, v3_plan)
