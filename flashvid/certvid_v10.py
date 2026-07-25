"""CertVID V10: trajectory-balanced evidence design over a V3 backbone."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from .certvid import CertVidPlan, _cfg_float, _cfg_int, apply_certvid_plan
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
    candidate_indices: torch.Tensor
    ridge: float


@dataclass
class _MotionGraph:
    trajectory_ids: torch.Tensor
    speed: torch.Tensor
    appearance_change: torch.Tensor
    turn: torch.Tensor
    endpoint: torch.Tensor
    token_score: torch.Tensor
    frame_curvature: torch.Tensor
    frame_score: torch.Tensor
    valid_incoming: torch.Tensor
    reliability: float
    track_count: int
    long_track_count: int
    track_records: list[dict[str, Any]]


@dataclass(frozen=True)
class _Candidate:
    token: int
    score: float
    provenance: str
    trajectory: int


_MOTION_QUERY = re.compile(
    r"\b(move|moving|direction|left|right|up|down|toward|away|"
    r"count|many|times|repeat|again|order|sequence|before|after|"
    r"first|next|then|finally|turn|change|enter|leave|approach|pass)\b",
    flags=re.IGNORECASE,
)


def _cfg_bool(config: Any, name: str, default: bool) -> bool:
    value = getattr(config, name, default)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


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
        "candidate_indices",
        "ridge",
    )
    missing = [name for name in names if name not in sink]
    if missing:
        raise ValueError(f"V3 analysis is missing: {', '.join(missing)}")
    return _V3Analysis(**{name: sink[name] for name in names})


def _normalize(values: torch.Tensor) -> torch.Tensor:
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
    minimum = values.min()
    span = values.max() - minimum
    if float(span.item()) <= 1e-8:
        return torch.zeros_like(values)
    return (values - minimum) / span


def _grid_coords(tokens_per_frame: int, device: torch.device) -> torch.Tensor:
    height = max(1, int(math.sqrt(tokens_per_frame)))
    while height > 1 and tokens_per_frame % height != 0:
        height -= 1
    width = max(1, tokens_per_frame // height)
    y = torch.linspace(-1.0, 1.0, steps=height, device=device)
    x = torch.linspace(-1.0, 1.0, steps=width, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([yy.flatten(), xx.flatten()], dim=-1)[:tokens_per_frame]


def _build_motion_graph(
    metric_flat: torch.Tensor,
    frame_count: int,
    tokens_per_frame: int,
    config: Any,
) -> _MotionGraph:
    device = metric_flat.device
    metric = metric_flat.view(frame_count, tokens_per_frame, -1).float()
    coords = _grid_coords(tokens_per_frame, device)
    spatial_distance = torch.cdist(coords.float(), coords.float(), p=2)
    threshold = _cfg_float(config, "certv10_track_similarity", 0.72)
    spatial_penalty = max(0.0, _cfg_float(config, "certv10_spatial_penalty", 0.03))

    previous = torch.full(
        (frame_count, tokens_per_frame),
        -1,
        dtype=torch.long,
        device=device,
    )
    following = torch.full_like(previous, -1)
    incoming_similarity = torch.zeros(
        (frame_count, tokens_per_frame), dtype=torch.float32, device=device
    )
    valid_incoming = torch.zeros(
        (frame_count, tokens_per_frame), dtype=torch.bool, device=device
    )
    speed = torch.zeros_like(incoming_similarity)
    appearance = torch.zeros_like(incoming_similarity)

    parent = list(range(frame_count * tokens_per_frame))
    size = [1] * len(parent)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            return
        if size[root_left] < size[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        size[root_left] += size[root_right]

    for frame in range(1, frame_count):
        current = metric[frame]
        prior = metric[frame - 1]
        raw_similarity = current @ prior.transpose(0, 1)
        score = raw_similarity - spatial_penalty * spatial_distance
        best_prior = score.argmax(dim=1)
        best_current = score.argmax(dim=0)
        current_ids = torch.arange(tokens_per_frame, device=device)
        mutual = best_current[best_prior] == current_ids
        matched_similarity = raw_similarity[current_ids, best_prior]
        valid = mutual & (matched_similarity >= threshold)
        previous[frame, valid] = best_prior[valid]
        incoming_similarity[frame, valid] = matched_similarity[valid]
        valid_incoming[frame] = valid
        displacement = coords - coords[best_prior]
        speed[frame, valid] = displacement.norm(dim=-1)[valid] / math.sqrt(8.0)
        appearance[frame] = (1.0 - matched_similarity).clamp(0.0, 2.0) * 0.5

        current_tokens = torch.where(valid)[0]
        following[frame - 1, best_prior[current_tokens]] = current_tokens
        matched_pairs = torch.stack(
            [current_tokens, best_prior[current_tokens]], dim=1
        ).detach().cpu().tolist()
        for current_token, prior_token in matched_pairs:
            union(
                frame * tokens_per_frame + int(current_token),
                (frame - 1) * tokens_per_frame + prior_token,
            )

    turn = torch.zeros_like(speed)
    for frame in range(1, frame_count - 1):
        has_previous = previous[frame] >= 0
        has_following = following[frame] >= 0
        valid = has_previous & has_following
        if not bool(valid.any()):
            continue
        token_ids = torch.where(valid)[0]
        prior_ids = previous[frame, token_ids]
        next_ids = following[frame, token_ids]
        incoming = coords[token_ids] - coords[prior_ids]
        outgoing = coords[next_ids] - coords[token_ids]
        direction_change = (
            1.0 - F.cosine_similarity(incoming, outgoing, dim=-1, eps=1e-6)
        ).clamp(0.0, 2.0) * 0.5
        motion_gate = torch.sqrt(
            incoming.norm(dim=-1) * outgoing.norm(dim=-1)
        ).clamp(0.0, 1.0)
        turn[frame, token_ids] = direction_change * motion_gate

    roots = [find(node) for node in range(frame_count * tokens_per_frame)]
    root_to_track: dict[int, int] = {}
    trajectory_ids: list[int] = []
    members_by_track: list[list[int]] = []
    for token, root in enumerate(roots):
        if root not in root_to_track:
            root_to_track[root] = len(root_to_track)
            members_by_track.append([])
        track_id = root_to_track[root]
        trajectory_ids.append(track_id)
        members_by_track[track_id].append(token)
    trajectory_ids_tensor = torch.tensor(
        trajectory_ids, dtype=torch.long, device=device
    ).view(frame_count, tokens_per_frame)

    endpoint = torch.zeros_like(speed)
    records: list[dict[str, Any]] = []
    minimum_span = max(1, _cfg_int(config, "certv10_track_min_span", 2))
    long_track_count = 0
    appearance_cpu = appearance.flatten().detach().cpu().tolist()
    turn_cpu = turn.flatten().detach().cpu().tolist()
    speed_cpu = speed.flatten().detach().cpu().tolist()
    endpoint_tokens: list[int] = []
    for track_id, members in enumerate(members_by_track):
        if len(members) < 2:
            continue
        first_frame = members[0] // tokens_per_frame
        last_frame = members[-1] // tokens_per_frame
        span = last_frame - first_frame
        if span < minimum_span:
            continue
        long_track_count += 1
        first_members = [
            token
            for token in members
            if token // tokens_per_frame == first_frame
        ]
        last_members = [
            token
            for token in members
            if token // tokens_per_frame == last_frame
        ]
        start = max(first_members, key=lambda token: appearance_cpu[token])
        end = max(last_members, key=lambda token: appearance_cpu[token])
        endpoint_tokens.extend((start, end))
        turn_token = max(members, key=lambda token: turn_cpu[token])
        speed_token = max(members, key=lambda token: speed_cpu[token])
        records.append(
            {
                "track": int(track_id),
                "span": int(span),
                "size": len(members),
                "start": start,
                "end": end,
                "turn": turn_token,
                "speed": speed_token,
            }
        )
    if endpoint_tokens:
        endpoint.flatten()[
            torch.tensor(endpoint_tokens, dtype=torch.long, device=device)
        ] = 1.0

    frame_representatives = F.normalize(
        metric.mean(dim=1), p=2, dim=-1, eps=1e-6
    )
    frame_curvature = torch.zeros(frame_count, dtype=torch.float32, device=device)
    if frame_count > 2:
        incoming = frame_representatives[1:-1] - frame_representatives[:-2]
        outgoing = frame_representatives[2:] - frame_representatives[1:-1]
        frame_curvature[1:-1] = (
            1.0 - F.cosine_similarity(incoming, outgoing, dim=-1, eps=1e-6)
        ).clamp(0.0, 2.0) * 0.5
    if frame_count > 1:
        boundary = (
            1.0
            - torch.sum(
                frame_representatives[1:] * frame_representatives[:-1], dim=-1
            )
        ).clamp(0.0, 2.0) * 0.5
        frame_curvature[0] = boundary[0]
        frame_curvature[-1] = boundary[-1]

    speed_n = _normalize(speed)
    appearance_n = _normalize(appearance)
    turn_n = _normalize(turn)
    token_score = (
        0.24 * speed_n
        + 0.28 * appearance_n
        + 0.30 * turn_n
        + 0.18 * endpoint
    )
    top_count = max(1, int(math.ceil(tokens_per_frame * 0.10)))
    frame_score = (
        0.35 * _normalize(frame_curvature)
        + 0.30 * torch.topk(appearance_n, k=top_count, dim=1).values.mean(dim=1)
        + 0.20 * torch.topk(speed_n, k=top_count, dim=1).values.mean(dim=1)
        + 0.15 * torch.topk(turn_n, k=top_count, dim=1).values.mean(dim=1)
    )
    possible_edges = max(1, (frame_count - 1) * tokens_per_frame)
    reliability = float(valid_incoming[1:].sum().item()) / float(possible_edges)
    return _MotionGraph(
        trajectory_ids=trajectory_ids_tensor.flatten(),
        speed=speed_n.flatten(),
        appearance_change=appearance_n.flatten(),
        turn=turn_n.flatten(),
        endpoint=endpoint.flatten(),
        token_score=token_score.flatten(),
        frame_curvature=_normalize(frame_curvature),
        frame_score=_normalize(frame_score),
        valid_incoming=valid_incoming.flatten(),
        reliability=reliability,
        track_count=len(root_to_track),
        long_track_count=long_track_count,
        track_records=records,
    )


def _integer_budget(
    weights: torch.Tensor,
    total: int,
    minimum: int,
    maximum: int,
) -> torch.Tensor:
    count = int(weights.numel())
    minimum = max(0, min(int(minimum), int(maximum)))
    if total < count * minimum:
        minimum = 0
    allocation = torch.full(
        (count,), minimum, dtype=torch.long, device=weights.device
    )
    remaining = int(total - allocation.sum().item())
    if remaining <= 0:
        return allocation
    probability = weights.float().clamp_min(0.0)
    if float(probability.sum().item()) <= 1e-8:
        probability = torch.ones_like(probability)
    probability = probability / probability.sum()
    raw = probability * float(remaining)
    extra = torch.floor(raw).long()
    extra = torch.minimum(
        extra, torch.full_like(extra, max(0, maximum - minimum))
    )
    allocation += extra
    remaining = int(total - allocation.sum().item())
    fractions = raw - torch.floor(raw)
    while remaining > 0:
        eligible = allocation < maximum
        if not bool(eligible.any()):
            break
        score = fractions.masked_fill(~eligible, -1.0)
        index = int(torch.argmax(score).item())
        allocation[index] += 1
        fractions[index] = -1.0
        remaining -= 1
    return allocation


def _target_frame_budget(
    graph: _MotionGraph,
    v3_counts: torch.Tensor,
    budget: int,
    router: float,
    tokens_per_frame: int,
    config: Any,
) -> torch.Tensor:
    frame_count = int(v3_counts.numel())
    average = float(budget) / max(1, frame_count)
    floor_ratio = min(
        1.0, max(0.0, _cfg_float(config, "certv10_frame_floor_ratio", 0.55))
    )
    cap_ratio = max(
        1.0, _cfg_float(config, "certv10_frame_cap_ratio", 1.80)
    )
    minimum = max(1, int(math.floor(average * floor_ratio)))
    maximum = min(tokens_per_frame, max(minimum, int(math.ceil(average * cap_ratio))))
    temperature = max(1e-3, _cfg_float(config, "certv10_budget_temperature", 0.55))
    weights = torch.softmax(graph.frame_score / temperature, dim=0)
    motion_allocation = _integer_budget(weights, budget, minimum, maximum)
    strength = min(
        1.0,
        max(0.0, _cfg_float(config, "certv10_allocation_strength", 0.75) * router),
    )
    mixed = (1.0 - strength) * v3_counts.float() + strength * motion_allocation.float()
    base = torch.floor(mixed).long().clamp(min=1, max=tokens_per_frame)
    remaining = int(budget - base.sum().item())
    fractions = mixed - torch.floor(mixed)
    while remaining > 0:
        eligible = base < tokens_per_frame
        score = fractions.masked_fill(~eligible, -1.0)
        index = int(torch.argmax(score).item())
        base[index] += 1
        fractions[index] = -1.0
        remaining -= 1
    while remaining < 0:
        eligible = base > 1
        score = fractions.masked_fill(~eligible, 2.0)
        index = int(torch.argmin(score).item())
        base[index] -= 1
        fractions[index] = 2.0
        remaining += 1
    return base


def _motion_router(graph: _MotionGraph, config: Any) -> tuple[float, bool]:
    question = str(getattr(config, "_certvid_query_text", "") or "")
    motion_query = bool(_MOTION_QUERY.search(question))
    mean_frame = float(graph.frame_score.mean().item())
    top_frames = min(4, int(graph.frame_score.numel()))
    peak_frame = float(
        torch.topk(graph.frame_score, k=max(1, top_frames)).values.mean().item()
    )
    track_density = min(
        1.0,
        float(graph.long_track_count)
        / max(1.0, 0.25 * float(graph.frame_score.numel())),
    )
    reliability_target = max(
        1e-4, _cfg_float(config, "certv10_reliability_target", 0.18)
    )
    reliability = min(1.0, graph.reliability / reliability_target)
    route = (
        0.30 * mean_frame
        + 0.25 * peak_frame
        + 0.25 * track_density
        + 0.20 * reliability
        + (0.18 if motion_query else 0.0)
    )
    return min(1.0, max(0.0, route)), motion_query


def _candidate_pool(
    graph: _MotionGraph,
    analysis: _V3Analysis,
    selected: torch.Tensor,
    target_counts: torch.Tensor,
    config: Any,
) -> list[_Candidate]:
    selected_set = set(int(token) for token in selected.detach().cpu().tolist())
    candidates: dict[int, _Candidate] = {}
    token_score_cpu = graph.token_score.detach().cpu().tolist()

    def offer(token: int, score: float, provenance: str, trajectory: int) -> None:
        if token in selected_set:
            return
        item = _Candidate(token, float(score), provenance, trajectory)
        previous = candidates.get(token)
        if previous is None or (item.score, item.provenance) > (
            previous.score,
            previous.provenance,
        ):
            candidates[token] = item

    for record in graph.track_records:
        track = int(record["track"])
        span_gain = min(1.0, float(record["span"]) / 6.0)
        for provenance, bonus in (
            ("start", 0.20),
            ("end", 0.28),
            ("turn", 0.34),
            ("speed", 0.26),
        ):
            token = int(record[provenance])
            score = float(token_score_cpu[token]) + bonus + 0.12 * span_gain
            offer(token, score, f"track_{provenance}", track)

    frame_count = int(target_counts.numel())
    peak_count = min(
        frame_count,
        max(2, _cfg_int(config, "certv10_motion_peak_frames", 8)),
    )
    blocked = torch.zeros(frame_count, dtype=torch.bool, device=target_counts.device)
    frame_scores = graph.frame_score.clone()
    for _ in range(peak_count):
        score = frame_scores.masked_fill(blocked, -1.0)
        frame = int(torch.argmax(score).item())
        if float(score[frame].item()) < 0.0:
            break
        members = torch.where(analysis.frame_ids == frame)[0]
        if members.numel() > 0:
            combined = (
                0.72 * graph.token_score[members]
                + 0.18 * analysis.query_score[members]
                + 0.10 * _normalize(analysis.demand_weight)[members]
            )
            token = int(members[torch.argmax(combined)].item())
            trajectory = int(graph.trajectory_ids[token].item())
            offer(
                token,
                float(combined.max().item()) + 0.20,
                "motion_peak",
                trajectory,
            )
        blocked[max(0, frame - 1) : min(frame_count, frame + 2)] = True

    v3_counts = torch.bincount(
        analysis.frame_ids[selected], minlength=frame_count
    ).detach().cpu().tolist()
    target_counts_cpu = target_counts.detach().cpu().tolist()
    for frame in range(frame_count):
        deficit = max(
            0, int(target_counts_cpu[frame]) - int(v3_counts[frame])
        )
        if deficit <= 0:
            continue
        members = torch.where(analysis.frame_ids == frame)[0]
        score = (
            0.60 * graph.token_score[members]
            + 0.22 * analysis.query_score[members]
            + 0.18 * _normalize(analysis.demand_weight)[members]
        )
        keep = min(int(members.numel()), max(2, deficit * 3))
        order = torch.topk(score, k=keep, largest=True).indices
        chosen = members[order]
        chosen_tokens = chosen.detach().cpu().tolist()
        chosen_scores = score[order].detach().cpu().tolist()
        chosen_trajectories = (
            graph.trajectory_ids[chosen].detach().cpu().tolist()
        )
        for token, token_score, trajectory in zip(
            chosen_tokens, chosen_scores, chosen_trajectories
        ):
            offer(
                int(token),
                float(token_score) + 0.08 * deficit,
                "frame_deficit",
                int(trajectory),
            )

    limit = max(64, _cfg_int(config, "certv10_candidate_pool", 384))
    return sorted(
        candidates.values(),
        key=lambda item: (-item.score, item.token),
    )[:limit]


def _design_costs(analysis: _V3Analysis, selected: torch.Tensor) -> torch.Tensor:
    rows = analysis.design[selected].float()
    dimension = int(rows.shape[1])
    identity = torch.eye(dimension, dtype=torch.float32, device=rows.device)
    information = max(1e-4, float(analysis.ridge)) * identity + rows.T @ rows
    inverse = torch.linalg.pinv(information)
    leverage = torch.sum((rows @ inverse) * rows, dim=1).clamp(0.0, 1.0 - 1e-5)
    return _normalize(-torch.log1p(-leverage))


def _d_efficiency(
    analysis: _V3Analysis,
    baseline: torch.Tensor,
    selected: torch.Tensor,
) -> float:
    dimension = int(analysis.design.shape[1])
    identity = torch.eye(
        dimension, dtype=torch.float32, device=analysis.design.device
    )
    ridge = max(1e-4, float(analysis.ridge))
    old = ridge * identity + analysis.design[baseline].float().T @ analysis.design[baseline].float()
    new = ridge * identity + analysis.design[selected].float().T @ analysis.design[selected].float()
    old_sign, old_logdet = torch.linalg.slogdet(old)
    new_sign, new_logdet = torch.linalg.slogdet(new)
    if float(old_sign.item()) <= 0.0 or float(new_sign.item()) <= 0.0:
        return 0.0
    return float(torch.exp((new_logdet - old_logdet) / max(1, dimension)).item())


def _trajectory_reallocate(
    graph: _MotionGraph,
    analysis: _V3Analysis,
    v3_selected: torch.Tensor,
    target_counts: torch.Tensor,
    router: float,
    config: Any,
) -> tuple[torch.Tensor, list[dict[str, Any]], int, int]:
    budget = int(v3_selected.numel())
    frame_count = int(target_counts.numel())
    min_ratio = min(0.40, max(0.0, _cfg_float(config, "certv10_min_swap_ratio", 0.08)))
    max_ratio = min(
        0.45,
        max(min_ratio, _cfg_float(config, "certv10_max_swap_ratio", 0.25)),
    )
    swap_ratio = min_ratio + (max_ratio - min_ratio) * router
    max_swaps = max(1, int(math.ceil(budget * swap_ratio)))
    minimum_swaps = min(max_swaps, max(1, int(math.ceil(budget * min_ratio))))
    candidates = _candidate_pool(graph, analysis, v3_selected, target_counts, config)
    if not candidates:
        return v3_selected, [], minimum_swaps, max_swaps

    selected = v3_selected.clone()
    base_tokens_cpu = v3_selected.detach().cpu().tolist()
    selected_set = set(int(token) for token in base_tokens_cpu)
    frame_counts = torch.bincount(analysis.frame_ids[selected], minlength=frame_count)
    frame_counts_cpu = frame_counts.detach().cpu().tolist()
    target_counts_cpu = target_counts.detach().cpu().tolist()
    frame_ids_cpu = analysis.frame_ids.detach().cpu().tolist()
    query_score_cpu = analysis.query_score.detach().cpu().tolist()
    demand = _normalize(analysis.demand_weight)
    demand_cpu = demand.detach().cpu().tolist()
    d_cost = _design_costs(analysis, v3_selected)

    metric = analysis.metric_flat[v3_selected].float()
    similarity = metric @ metric.transpose(0, 1)
    similarity.fill_diagonal_(-2.0)
    uniqueness = _normalize(1.0 - similarity.max(dim=1).values)
    static_cost = (
        0.28 * demand[v3_selected]
        + 0.24 * analysis.attention[v3_selected]
        + 0.18 * analysis.query_score[v3_selected]
        + 0.16 * uniqueness
        + 0.14 * d_cost
    )

    protected: set[int] = set()
    for frame in range(frame_count):
        positions = torch.where(analysis.frame_ids[v3_selected] == frame)[0]
        if positions.numel() == 0:
            continue
        score = static_cost[positions]
        protected.add(int(v3_selected[positions[torch.argmax(score)]].item()))
    protect_ratio = min(
        0.30, max(0.0, _cfg_float(config, "certv10_v3_protect_ratio", 0.10))
    )
    protect_count = min(
        budget, max(1, int(math.ceil(budget * protect_ratio)))
    )
    protect_positions = (
        torch.topk(static_cost, k=protect_count)
        .indices.detach()
        .cpu()
        .tolist()
    )
    protected.update(int(base_tokens_cpu[index]) for index in protect_positions)

    base_frames = analysis.frame_ids[v3_selected]
    protected_tensor = torch.tensor(
        sorted(protected), dtype=torch.long, device=v3_selected.device
    )
    removable = ~torch.isin(v3_selected, protected_tensor)
    swaps: list[dict[str, Any]] = []
    minimum_gain = _cfg_float(config, "certv10_min_swap_gain", -0.04)
    d_weight = max(0.0, _cfg_float(config, "certv10_d_soft_weight", 0.10))
    average = float(budget) / max(1, frame_count)

    for candidate in candidates:
        if len(swaps) >= max_swaps or candidate.token in selected_set:
            continue
        add_frame = int(frame_ids_cpu[candidate.token])
        deficit = max(
            0.0,
            float(
                target_counts_cpu[add_frame] - frame_counts_cpu[add_frame]
            ),
        )
        add_value = (
            0.62 * candidate.score
            + 0.18 * query_score_cpu[candidate.token]
            + 0.12 * demand_cpu[candidate.token]
            + 0.18 * min(1.0, deficit / max(1.0, average))
        )

        frame_count_at_position = frame_counts[base_frames]
        eligible = removable & (frame_count_at_position > 1)
        if not bool(eligible.any()):
            continue
        surplus = (
            frame_count_at_position - target_counts[base_frames]
        ).clamp_min(0).float()
        remove_cost = (
            static_cost
            + d_weight * d_cost
            - 0.32 * (surplus / max(1.0, average)).clamp_max(1.0)
        )
        add_before = abs(
            float(
                frame_counts_cpu[add_frame] - target_counts_cpu[add_frame]
            )
        )
        add_after = abs(
            float(
                frame_counts_cpu[add_frame]
                + 1
                - target_counts_cpu[add_frame]
            )
        )
        add_allocation_gain = (add_before - add_after) / max(1.0, average)
        remove_before = (
            frame_count_at_position.float() - target_counts[base_frames].float()
        ).abs()
        remove_after = (
            frame_count_at_position.float()
            - 1.0
            - target_counts[base_frames].float()
        ).abs()
        remove_allocation_gain = (
            remove_before - remove_after
        ) / max(1.0, average)
        different_frame = base_frames != add_frame
        allocation_gain = torch.where(
            different_frame,
            remove_allocation_gain + add_allocation_gain,
            torch.zeros_like(remove_allocation_gain),
        )
        gains = add_value - remove_cost + 0.24 * allocation_gain
        gains = gains.masked_fill(~eligible, float("-inf"))
        best_position = int(torch.argmax(gains).item())
        best_gain = float(gains[best_position].item())
        best_remove_cost = float(remove_cost[best_position].item())
        if not math.isfinite(best_gain):
            continue
        # The reliable-motion branch is deliberately active rather than a
        # conservative V3 repair. Strong trajectory candidates fill the
        # minimum reallocation quota before the regular gain gate applies.
        if len(swaps) >= minimum_swaps and best_gain < minimum_gain:
            continue
        removed = int(base_tokens_cpu[best_position])
        remove_frame = int(frame_ids_cpu[removed])
        selected[best_position] = int(candidate.token)
        selected_set.remove(removed)
        selected_set.add(candidate.token)
        removable[best_position] = False
        frame_counts[remove_frame] -= 1
        frame_counts[add_frame] += 1
        frame_counts_cpu[remove_frame] -= 1
        frame_counts_cpu[add_frame] += 1
        swaps.append(
            {
                "add": candidate.token,
                "remove": removed,
                "provenance": candidate.provenance,
                "trajectory": candidate.trajectory,
                "gain": float(best_gain),
                "add_score": float(add_value),
                "remove_cost": float(best_remove_cost),
                "add_frame": add_frame,
                "remove_frame": remove_frame,
            }
        )

    return torch.sort(selected).values, swaps, minimum_swaps, max_swaps


def _frame_times(config: Any, frame_count: int, device: torch.device) -> tuple[torch.Tensor, bool]:
    raw = getattr(config, "_certvid_frame_times_sec", None)
    if raw is None:
        return torch.arange(frame_count, device=device).float(), False
    times = torch.as_tensor(raw, dtype=torch.float32, device=device).flatten()
    valid = (
        times.numel() == frame_count
        and bool(torch.isfinite(times).all())
        and (frame_count <= 1 or bool(torch.all(times[1:] > times[:-1])))
    )
    if not valid:
        return torch.arange(frame_count, device=device).float(), False
    return times, True


def _build_plan(
    selected: torch.Tensor,
    v3_selected: torch.Tensor,
    v3_plan: CertVidPlan,
    graph: _MotionGraph,
    analysis: _V3Analysis,
    frame_times: torch.Tensor,
    has_real_times: bool,
    swaps: list[dict[str, Any]],
    config: Any,
) -> tuple[CertVidPlan, dict[str, Any]]:
    total_tokens = int(analysis.metric_flat.shape[0])
    budget = int(selected.numel())
    similarity = analysis.metric_flat.float() @ analysis.metric_flat[selected].float().T
    source_frame = analysis.frame_ids.unsqueeze(1)
    anchor_frame = analysis.frame_ids[selected].unsqueeze(0)
    frame_distance = (source_frame - anchor_frame).abs()
    same_track = graph.trajectory_ids.unsqueeze(1) == graph.trajectory_ids[selected].unsqueeze(0)
    track_radius = max(1, _cfg_int(config, "certv10_track_assignment_radius", 2))
    valid = (frame_distance <= 1) | (same_track & (frame_distance <= track_radius))
    if has_real_times:
        maximum_seconds = max(
            0.0, _cfg_float(config, "certv10_cross_frame_max_seconds", 12.0)
        )
        time_distance = (
            frame_times[analysis.frame_ids].unsqueeze(1)
            - frame_times[analysis.frame_ids[selected]].unsqueeze(0)
        ).abs()
        valid &= (frame_distance == 0) | (time_distance <= maximum_seconds)
    similarity = similarity + 0.12 * same_track.float()
    similarity = similarity.masked_fill(~valid, -2.0)
    topk = min(max(1, _cfg_int(config, "certv10_assignment_topk", 2)), budget)
    values, assignment = torch.topk(similarity, k=topk, dim=1, largest=True)
    temperature = max(1e-4, _cfg_float(config, "certv10_assignment_temperature", 0.07))
    weights = torch.softmax(values / temperature, dim=1)
    merge_threshold = _cfg_float(config, "certv10_merge_threshold", 0.76)
    rejected = values[:, 0] < merge_threshold
    weights[rejected] = 0.0

    anchor_positions = torch.arange(budget, dtype=torch.long, device=selected.device)
    assignment[selected, 0] = anchor_positions
    weights[selected] = 0.0
    weights[selected, 0] = 1.0

    alpha = torch.zeros(budget, dtype=torch.float32, device=selected.device)
    old_positions = torch.searchsorted(v3_selected, selected)
    old_positions_clamped = old_positions.clamp_max(v3_selected.numel() - 1)
    retained_from_v3 = (
        (old_positions < v3_selected.numel())
        & (v3_selected[old_positions_clamped] == selected)
    )
    alpha[retained_from_v3] = v3_plan.fusion_alpha[
        old_positions_clamped[retained_from_v3]
    ]
    trajectory_protect = _cfg_float(config, "certv10_trajectory_fusion_scale", 0.25)
    alpha *= 1.0 - (1.0 - trajectory_protect) * graph.token_score[selected]
    promoted = {int(record["add"]) for record in swaps}
    if promoted:
        promoted_tensor = torch.tensor(
            sorted(promoted), dtype=torch.long, device=selected.device
        )
        alpha[torch.isin(selected, promoted_tensor)] = 0.0

    plan = CertVidPlan(
        anchor_indices=selected,
        assignment_indices=assignment,
        assignment_weights=weights,
        source_mass=v3_plan.source_mass,
        fusion_alpha=alpha.clamp(0.0, float(v3_plan.fusion_alpha.max().item())),
        raw_token_count=total_tokens,
    )
    return plan, {
        "rejected_residual_count": int((rejected & ~torch.isin(torch.arange(total_tokens, device=selected.device), selected)).sum().item()),
        "assignment_similarity_min": float(values[:, 0].min().item()),
        "assignment_similarity_median": float(values[:, 0].median().item()),
        "cross_frame_assignment_rate": float(
            (frame_distance.gather(1, assignment[:, :1]) > 0).float().mean().item()
        ),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _store_diagnostics(config: Any, diagnostics: Dict[str, Any]) -> None:
    config.last_certv10_diagnostics = diagnostics
    config.last_certv10_swap_count = int(diagnostics.get("swap_count", 0))
    config.last_certv10_router = float(diagnostics.get("motion_router", 0.0))
    config.last_certv10_v3_overlap_ratio = float(
        diagnostics.get("v3_overlap_ratio", 1.0)
    )
    template = os.environ.get("CERTV10_DIAGNOSTICS_JSONL", "").strip()
    if template:
        rank = os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))
        path = template.replace("{rank}", rank).replace("{pid}", str(os.getpid()))
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        record = dict(diagnostics)
        record["sample_id"] = str(getattr(config, "_debug_sample_id", "unknown"))
        record["question"] = str(getattr(config, "_certvid_query_text", "") or "")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")
    if _cfg_bool(config, "certv10_debug", False):
        print(
            "[certvid-v10] "
            f"sample={getattr(config, '_debug_sample_id', 'unknown')} "
            f"router={diagnostics.get('motion_router', 0.0):.3f} "
            f"tracks={diagnostics.get('long_track_count', 0)} "
            f"swaps={diagnostics.get('swap_count', 0)}/"
            f"{diagnostics.get('max_swaps', 0)} "
            f"overlap={diagnostics.get('v3_overlap_ratio', 1.0):.3f}"
        )


def _return_v3(
    config: Any,
    output: torch.Tensor,
    indices: torch.Tensor,
    plan: CertVidPlan,
    diagnostics: Dict[str, Any],
    reason: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    diagnostics["fallback_reason"] = reason
    diagnostics.setdefault("swap_count", 0)
    diagnostics.setdefault("v3_overlap_ratio", 1.0)
    config._certvid_plan = plan
    config.vision_token_length = int(output.shape[0])
    config.visual_token_length = int(output.shape[0])
    config.llm_token_length = None
    config.last_adapter_variant = "certvid_v10"
    _store_diagnostics(config, diagnostics)
    return output, indices


def certvid_v10_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    config: Any,
    question_features: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reallocate V3 anchors toward reliable motion chains under the same budget."""
    sink: Dict[str, Any] = {}
    v3_output, v3_indices = certvid_v3_compression(
        video_features,
        cls_attention,
        config,
        question_features,
        analysis_sink=sink,
    )
    v3_plan = getattr(config, "_certvid_plan", None)
    if v3_plan is None:
        raise RuntimeError("CertVID V3 did not publish an aggregation plan")

    frame_count, tokens_per_frame, feature_dim = video_features.shape
    diagnostics: Dict[str, Any] = {
        "fallback_reason": None,
        "raw_token_count": int(frame_count * tokens_per_frame),
        "budget": int(v3_indices.numel()),
        "swap_count": 0,
        "v3_overlap_ratio": 1.0,
    }
    if not _cfg_bool(config, "certv10_enabled", True):
        return _return_v3(
            config, v3_output, v3_indices, v3_plan, diagnostics, "disabled"
        )
    if sink.get("identity", False):
        return _return_v3(
            config, v3_output, v3_indices, v3_plan, diagnostics, "identity_budget"
        )

    try:
        analysis = _analysis_from_sink(sink)
        graph = _build_motion_graph(
            analysis.metric_flat,
            frame_count,
            tokens_per_frame,
            config,
        )
        router, motion_query = _motion_router(graph, config)
        reliability_floor = max(
            0.0, _cfg_float(config, "certv10_reliability_floor", 0.025)
        )
        diagnostics.update(
            {
                "motion_router": router,
                "motion_query": motion_query,
                "edge_reliability": graph.reliability,
                "track_count": graph.track_count,
                "long_track_count": graph.long_track_count,
                "frame_motion_score": graph.frame_score,
            }
        )
        if graph.reliability < reliability_floor or graph.long_track_count <= 0:
            return _return_v3(
                config,
                v3_output,
                v3_indices,
                v3_plan,
                diagnostics,
                "unreliable_trajectory_graph",
            )

        v3_frame_counts = torch.bincount(
            analysis.frame_ids[v3_indices], minlength=frame_count
        )
        target_counts = _target_frame_budget(
            graph,
            v3_frame_counts,
            int(v3_indices.numel()),
            router,
            tokens_per_frame,
            config,
        )
        selected, swaps, minimum_swaps, max_swaps = _trajectory_reallocate(
            graph,
            analysis,
            v3_indices,
            target_counts,
            router,
            config,
        )
        diagnostics.update(
            {
                "v3_frame_counts": v3_frame_counts,
                "target_frame_counts": target_counts,
                "minimum_swaps": minimum_swaps,
                "max_swaps": max_swaps,
                "swap_count": len(swaps),
                "swaps": swaps,
                "v3_overlap_ratio": float(
                    torch.isin(selected, v3_indices).float().mean().item()
                ),
            }
        )
        if not swaps:
            return _return_v3(
                config,
                v3_output,
                v3_indices,
                v3_plan,
                diagnostics,
                "no_trajectory_exchange",
            )

        frame_times, has_real_times = _frame_times(
            config, frame_count, video_features.device
        )
        plan, plan_stats = _build_plan(
            selected,
            v3_indices,
            v3_plan,
            graph,
            analysis,
            frame_times,
            has_real_times,
            swaps,
            config,
        )
        output = apply_certvid_plan(video_features.reshape(-1, feature_dim), plan)
        if output.shape != v3_output.shape or not bool(torch.isfinite(output).all()):
            raise RuntimeError("V10 output failed shape or finite-value validation")
        diagnostics.update(plan_stats)
        diagnostics["final_frame_counts"] = torch.bincount(
            analysis.frame_ids[selected], minlength=frame_count
        )
        diagnostics["d_efficiency"] = _d_efficiency(
            analysis, v3_indices, selected
        )

        config._certvid_plan = plan
        config.vision_token_length = int(output.shape[0])
        config.visual_token_length = int(output.shape[0])
        config.llm_token_length = None
        config.last_adapter_variant = "certvid_v10"
        config.last_adapter_raw_tokens = float(frame_count * tokens_per_frame)
        config.last_adapter_output_tokens = float(output.shape[0])
        _store_diagnostics(config, diagnostics)
        return output, selected
    except Exception as exc:
        diagnostics["exception"] = f"{type(exc).__name__}: {exc}"
        return _return_v3(
            config,
            v3_output,
            v3_indices,
            v3_plan,
            diagnostics,
            "validation_failure",
        )
