"""CertVID V8: V3-preserving long-context evidence repair.

V8 always runs the unchanged CertVID V3 selector first. It only edits the V3
anchor set when real timestamps expose a long sampling horizon and the V3
selection leaves a measurable temporal or query-evidence deficit. The repair
uses the same fixed token budget, protects V3 certificates, and falls back to
the exact V3 output whenever its guards are not satisfied.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .certvid import CertVidPlan, _cfg_float, _cfg_int, _minmax, apply_certvid_plan
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
class _IntentRoute:
    name: str
    repair_strength: float
    peak_count: int
    floor_ratio: float
    cap_ratio: float
    max_swap_ratio: float
    query_weight: float
    event_weight: float
    balance_weight: float
    d_efficiency_floor: float


_SEQUENCE_PATTERNS = (
    r"\bcorrect (?:temporal |chronological )?order\b",
    r"\bchronological\b",
    r"\bin what order\b",
    r"\bsequence of\b",
    r"\bfirst\b.*\bthen\b",
)
_CHANGE_PATTERNS = (
    r"\bwhat change",
    r"\bhow (?:did|does|has).+chang",
    r"\battribute change",
    r"\bdifferent (?:between|from)\b",
    r"\bfrom .+ to .+\b",
    r"\bbefore and after\b",
)
_TRACKING_PATTERNS = (
    r"\btrack",
    r"\bappear(?:s|ed|ing)? again\b",
    r"\breappear",
    r"\bsame (?:person|object|vehicle|animal)\b",
    r"\bwhere (?:is|was|did).+(?:go|appear|move)\b",
)
_TEMPORAL_PATTERNS = (
    r"\bbefore\b",
    r"\bafter\b",
    r"\bearlier\b",
    r"\blater\b",
    r"\bsubsequently\b",
    r"\bprior to\b",
    r"\bfollowing\b",
    r"\bat the beginning\b",
    r"\bat the end\b",
)
_RETRIEVAL_PATTERNS = (
    r"\bwhen\b",
    r"\bwhile\b",
    r"\bduring\b",
    r"\bwhere\b",
    r"\bwhich (?:scene|moment|object|person)\b",
    r"\bwhat (?:object|event|action|attribute|color)\b",
)


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


def _frame_times(
    config: Any,
    frame_count: int,
    device: torch.device,
) -> Tuple[torch.Tensor, bool, str]:
    raw = getattr(config, "_certvid_frame_times_sec", None)
    source = str(getattr(config, "_certvid_frame_times_source", "missing"))
    if raw is None:
        return torch.arange(frame_count, device=device).float(), False, "frame_index"
    times = torch.as_tensor(raw, dtype=torch.float32, device=device).flatten()
    valid = (
        times.numel() == frame_count
        and bool(torch.isfinite(times).all())
        and (times.numel() <= 1 or bool(torch.all(times[1:] > times[:-1])))
    )
    if not valid:
        return torch.arange(frame_count, device=device).float(), False, "frame_index"
    return times, True, source


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) is not None for pattern in patterns)


def _blend(base: float, target: float, strength: float) -> float:
    return (1.0 - strength) * base + strength * target


def _query_route(config: Any, analysis: _V3Analysis) -> _IntentRoute:
    text = str(getattr(config, "_certvid_query_text", "") or "").lower()
    # Route from the question stem only. Choice text can contain incidental
    # temporal words ("before", "after", "change") that do not describe the
    # reasoning operation requested by the question.
    text = re.split(r"\n\s*[a-g][\.\)]\s+", text, maxsplit=1)[0]
    text = re.sub(r"\s+", " ", text)
    router_enabled = bool(getattr(config, "certv8_intent_router", True))
    strength = min(
        1.0,
        max(0.0, _cfg_float(config, "certv8_intent_strength", 0.75)),
    )
    if not router_enabled:
        strength = 0.0

    base_floor = min(
        0.95,
        max(0.0, _cfg_float(config, "certv8_frame_floor_ratio", 0.35)),
    )
    base_cap = max(
        1.0,
        _cfg_float(config, "certv8_frame_cap_ratio", 2.20),
    )
    base_swap = min(
        0.50,
        max(0.0, _cfg_float(config, "certv8_max_swap_ratio", 0.12)),
    )
    base_query = min(
        0.50,
        max(0.0, _cfg_float(config, "certv8_query_weight", 0.30)),
    )
    base_event = min(
        0.50,
        max(0.0, _cfg_float(config, "certv8_event_weight", 0.25)),
    )
    base_balance = min(
        0.60,
        max(0.0, _cfg_float(config, "certv8_balance_weight", 0.22)),
    )
    base_floor_eff = min(
        1.0,
        max(0.0, _cfg_float(config, "certv8_d_efficiency_floor", 0.97)),
    )
    base_peaks = max(1, _cfg_int(config, "certv8_query_peak_count", 2))

    if _matches_any(text, _SEQUENCE_PATTERNS):
        name = "sequence"
        target = (0.62, 3, 0.38, 2.10, 0.10, 0.18, 0.32, 0.24, 0.98)
    elif _matches_any(text, _CHANGE_PATTERNS):
        name = "attribute_change"
        target = (0.70, 2, 0.46, 1.95, 0.12, 0.32, 0.30, 0.28, 0.965)
    elif _matches_any(text, _TRACKING_PATTERNS):
        name = "object_tracking"
        target = (0.62, 2, 0.42, 2.05, 0.11, 0.34, 0.24, 0.24, 0.975)
    elif _matches_any(text, _TEMPORAL_PATTERNS):
        name = "temporal_relation"
        target = (0.55, 2, 0.36, 2.15, 0.09, 0.28, 0.32, 0.22, 0.98)
    elif _matches_any(text, _RETRIEVAL_PATTERNS):
        name = "referred_retrieval"
        target = (0.52, 1, 0.36, 2.20, 0.12, 0.36, 0.20, 0.21, 0.975)
    elif text:
        name = "generic_question"
        target = (0.38, 1, 0.30, 2.40, 0.10, 0.24, 0.22, 0.16, 0.98)
    else:
        name = "embedding_only"
        inferred = 0.35 + 0.30 * min(1.0, float(analysis.query_confidence))
        target = (inferred, 1, 0.30, 2.40, 0.10, 0.24, 0.22, 0.16, 0.98)

    (
        repair_strength,
        peak_count,
        floor_ratio,
        cap_ratio,
        swap_ratio,
        query_weight,
        event_weight,
        balance_weight,
        d_efficiency,
    ) = target
    effective_peak_count = max(
        1,
        int(round(_blend(float(base_peaks), float(peak_count), strength))),
    )
    if float(analysis.query_confidence) < 0.10:
        effective_peak_count = 1
    return _IntentRoute(
        name=name,
        repair_strength=_blend(0.35, repair_strength, strength),
        peak_count=effective_peak_count,
        floor_ratio=_blend(base_floor, floor_ratio, strength),
        cap_ratio=_blend(base_cap, cap_ratio, strength),
        max_swap_ratio=_blend(base_swap, swap_ratio, strength),
        query_weight=_blend(base_query, query_weight, strength),
        event_weight=_blend(base_event, event_weight, strength),
        balance_weight=_blend(base_balance, balance_weight, strength),
        d_efficiency_floor=_blend(base_floor_eff, d_efficiency, strength),
    )


def _safe_attention(
    cls_attention: torch.Tensor,
    frame_count: int,
    tokens_per_frame: int,
) -> torch.Tensor:
    attention = torch.nan_to_num(
        cls_attention.float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    expected = frame_count * tokens_per_frame
    if attention.numel() != expected:
        raise ValueError(
            f"cls_attention shape mismatch: expected {expected}, got {attention.numel()}"
        )
    attention = attention.reshape(frame_count, tokens_per_frame)
    spread = attention.amax(dim=1, keepdim=True) - attention.amin(dim=1, keepdim=True)
    ranked = _minmax(attention, dim=1)
    return torch.where(spread > 1e-6, ranked, torch.zeros_like(ranked))


def _temporal_signals(
    metric_frames: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    frame_count, tokens_per_frame, _ = metric_frames.shape
    novelty = torch.zeros(
        (frame_count, tokens_per_frame),
        dtype=torch.float32,
        device=metric_frames.device,
    )
    event = torch.zeros(frame_count, dtype=torch.float32, device=metric_frames.device)
    if frame_count <= 1:
        return novelty, event

    contributions = torch.zeros_like(novelty)
    counts = torch.zeros_like(novelty)
    frame_means = F.normalize(metric_frames.mean(dim=1), p=2, dim=-1, eps=1e-6)
    gaps = (1.0 - (frame_means[1:] * frame_means[:-1]).sum(dim=1)).clamp(0.0, 2.0)
    event[1:] = gaps
    event[:-1] = torch.maximum(event[:-1], gaps)
    for frame in range(frame_count - 1):
        similarity = metric_frames[frame] @ metric_frames[frame + 1].transpose(0, 1)
        contributions[frame] += (1.0 - similarity.amax(dim=1)).clamp(0.0, 2.0)
        contributions[frame + 1] += (1.0 - similarity.amax(dim=0)).clamp(0.0, 2.0)
        counts[frame] += 1.0
        counts[frame + 1] += 1.0
    return _minmax(contributions / counts.clamp_min(1.0), dim=1), _minmax(event, dim=0)


def _frame_query_demand(
    analysis: _V3Analysis,
    frame_count: int,
    tokens_per_frame: int,
    route: _IntentRoute,
    config: Any,
) -> Tuple[torch.Tensor, list[int]]:
    if analysis.query_relevance.numel() == 0:
        return (
            torch.zeros(frame_count, device=analysis.metric_flat.device),
            [],
        )
    relevance = analysis.query_relevance.view(
        analysis.query_relevance.shape[0],
        frame_count,
        tokens_per_frame,
    ).amax(dim=2)
    relevance = _minmax(relevance, dim=1)
    separation = max(1, _cfg_int(config, "certv8_query_peak_separation", 2))
    boost = torch.zeros(frame_count, device=relevance.device)
    peak_frames: list[int] = []
    frame_ids = torch.arange(frame_count, device=relevance.device)
    for atom_scores in relevance:
        available = torch.ones(frame_count, dtype=torch.bool, device=relevance.device)
        for _ in range(route.peak_count):
            scores = atom_scores.masked_fill(~available, -1.0)
            frame = int(torch.argmax(scores).item())
            value = float(scores[frame].item())
            if value <= 0.0:
                break
            boost[frame] = torch.maximum(boost[frame], scores[frame])
            peak_frames.append(frame)
            available &= (frame_ids - frame).abs() >= separation
    if peak_frames:
        smoothed = boost.clone()
        smoothed[1:] = torch.maximum(smoothed[1:], 0.45 * boost[:-1])
        smoothed[:-1] = torch.maximum(smoothed[:-1], 0.45 * boost[1:])
        boost = _minmax(smoothed, dim=0)
    return boost, sorted(set(peak_frames))


def _token_scores(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    analysis: _V3Analysis,
    route: _IntentRoute,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frame_count, tokens_per_frame, _ = video_features.shape
    metric_frames = analysis.metric_flat.view(frame_count, tokens_per_frame, -1)
    attention = _safe_attention(cls_attention, frame_count, tokens_per_frame)
    novelty, event = _temporal_signals(metric_frames)
    centroid = F.normalize(metric_frames.mean(dim=1), p=2, dim=-1, eps=1e-6)
    detail = (
        1.0 - (metric_frames * centroid.unsqueeze(1)).sum(dim=-1)
    ).clamp(0.0, 2.0)
    detail = _minmax(detail, dim=1)
    query = analysis.query_score.view(frame_count, tokens_per_frame).clamp(0.0, 1.0)
    query_weight = route.query_weight * float(analysis.query_confidence)
    event_weight = route.event_weight
    fixed = 0.18 + 0.25 + 0.17 + query_weight + event_weight
    normalizer = max(1e-6, fixed)
    score = (
        0.18 * attention
        + 0.25 * novelty
        + 0.17 * detail
        + query_weight * query
        + event_weight * event.unsqueeze(1)
    ) / normalizer
    topk = max(1, min(tokens_per_frame, int(math.ceil(0.15 * tokens_per_frame))))
    frame_quality = score.topk(topk, dim=1).values.mean(dim=1)
    return _minmax(score, dim=1).reshape(-1), _minmax(frame_quality, dim=0), event


def _allocate_counts(
    budget: int,
    desired: torch.Tensor,
    lower: torch.Tensor,
    capacity: torch.Tensor,
) -> torch.Tensor:
    desired = torch.nan_to_num(desired.float(), nan=0.0, posinf=0.0, neginf=0.0)
    lower = lower.long().clamp_min(0)
    capacity = torch.maximum(capacity.long(), lower)
    if int(lower.sum().item()) > budget:
        raise RuntimeError("V8 lower frame budget exceeds total budget")
    if int(capacity.sum().item()) < budget:
        raise RuntimeError("V8 frame capacity cannot fill total budget")
    if float(desired.sum().item()) <= 1e-8:
        desired = torch.ones_like(desired)
    target_float = desired / desired.sum() * float(budget)
    counts = lower.clone()
    remaining = budget - int(counts.sum().item())
    frame_ids = torch.arange(counts.numel(), device=counts.device)
    while remaining > 0:
        eligible = counts < capacity
        if not bool(eligible.any()):
            raise RuntimeError("V8 frame allocation exhausted capacity")
        priority = target_float - counts.float() - frame_ids.float() * 1e-8
        priority = priority.masked_fill(~eligible, -1e9)
        counts[int(torch.argmax(priority).item())] += 1
        remaining -= 1
    return counts


def _target_frame_counts(
    base_counts: torch.Tensor,
    protected_counts: torch.Tensor,
    frame_quality: torch.Tensor,
    frame_event: torch.Tensor,
    query_demand: torch.Tensor,
    tokens_per_frame: int,
    route: _IntentRoute,
) -> torch.Tensor:
    frame_count = int(base_counts.numel())
    budget = int(base_counts.sum().item())
    mean_count = float(budget) / max(1, frame_count)
    floor_count = max(1, int(math.floor(mean_count * route.floor_ratio)))
    cap_count = max(floor_count, int(math.ceil(mean_count * route.cap_ratio)))
    lower = torch.maximum(
        protected_counts,
        torch.full_like(base_counts, floor_count),
    )
    capacity = torch.maximum(
        lower,
        torch.full_like(base_counts, min(tokens_per_frame, cap_count)),
    )
    if int(lower.sum().item()) > budget:
        lower = protected_counts.clone()
    if int(capacity.sum().item()) < budget:
        capacity = torch.full_like(base_counts, tokens_per_frame)

    # Query confidence already controls atom construction in the V3 analysis.
    # Multiplying it here again systematically underweights otherwise valid
    # long-range evidence.
    query_weight = min(0.60, route.query_weight)
    event_weight = min(0.50, route.event_weight)
    visual_weight = max(0.0, 1.0 - query_weight - event_weight)
    evidence = (
        visual_weight * (0.25 + frame_quality)
        + event_weight * (0.25 + frame_event)
        + query_weight * (0.25 + query_demand)
    )
    structured = evidence / evidence.sum().clamp_min(1e-6) * float(budget)
    desired = (
        (1.0 - route.repair_strength) * base_counts.float()
        + route.repair_strength * structured
    )
    return _allocate_counts(budget, desired, lower, capacity)


def _removal_cost(
    v3_indices: torch.Tensor,
    analysis: _V3Analysis,
    tokens_per_frame: int,
) -> torch.Tensor:
    design = _minmax(analysis.design[v3_indices].float().square().sum(dim=1), dim=0)
    demand = _minmax(analysis.demand_weight[v3_indices].float(), dim=0)
    query = analysis.query_score[v3_indices].float().clamp(0.0, 1.0)
    uniqueness = torch.ones_like(demand)
    for frame in torch.unique(v3_indices // tokens_per_frame).tolist():
        mask = v3_indices // tokens_per_frame == int(frame)
        local = analysis.metric_flat[v3_indices[mask]].float()
        if local.shape[0] <= 1:
            continue
        similarity = local @ local.transpose(0, 1)
        similarity.fill_diagonal_(-1.0)
        uniqueness[mask] = (
            0.5 * (1.0 - similarity.amax(dim=1))
        ).clamp(0.0, 1.0)
    return 0.36 * design + 0.28 * demand + 0.22 * query + 0.14 * uniqueness


def _addition_scores(
    candidates: torch.Tensor,
    selected: torch.Tensor,
    token_score: torch.Tensor,
    analysis: _V3Analysis,
    tokens_per_frame: int,
) -> torch.Tensor:
    score = 0.70 * token_score[candidates] + 0.30 * _minmax(
        analysis.demand_weight[candidates].float(),
        dim=0,
    )
    for frame in torch.unique(candidates // tokens_per_frame).tolist():
        mask = candidates // tokens_per_frame == int(frame)
        anchors = selected[selected // tokens_per_frame == int(frame)]
        if anchors.numel() == 0:
            continue
        similarity = (
            analysis.metric_flat[candidates[mask]].float()
            @ analysis.metric_flat[anchors].float().transpose(0, 1)
        )
        diversity = (
            0.5 * (1.0 - similarity.amax(dim=1))
        ).clamp(0.0, 1.0)
        score[mask] = 0.75 * score[mask] + 0.25 * diversity
    return score


def _propose_swaps(
    v3_indices: torch.Tensor,
    protected: set[int],
    target_counts: torch.Tensor,
    token_score: torch.Tensor,
    analysis: _V3Analysis,
    frame_count: int,
    tokens_per_frame: int,
    limit: int,
) -> Tuple[list[int], list[int]]:
    selected_set = set(int(value) for value in v3_indices.detach().cpu().tolist())
    removal_costs = _removal_cost(v3_indices, analysis, tokens_per_frame)
    removal_map = {
        int(token): float(cost)
        for token, cost in zip(
            v3_indices.detach().cpu().tolist(),
            removal_costs.detach().cpu().tolist(),
        )
    }
    counts = torch.bincount(
        v3_indices // tokens_per_frame,
        minlength=frame_count,
    ).clone()
    additions_by_frame: list[list[int]] = [[] for _ in range(frame_count)]
    removals_by_frame: list[list[int]] = [[] for _ in range(frame_count)]

    for frame in range(frame_count):
        start = frame * tokens_per_frame
        candidates = torch.tensor(
            [
                token
                for token in range(start, start + tokens_per_frame)
                if token not in selected_set
            ],
            dtype=torch.long,
            device=v3_indices.device,
        )
        if candidates.numel() > 0:
            values = _addition_scores(
                candidates,
                v3_indices,
                token_score,
                analysis,
                tokens_per_frame,
            )
            order = sorted(
                zip(
                    candidates.detach().cpu().tolist(),
                    values.detach().cpu().tolist(),
                ),
                key=lambda item: (-float(item[1]), int(item[0])),
            )
            additions_by_frame[frame] = [int(item[0]) for item in order]
        removals_by_frame[frame] = sorted(
            [
                int(token)
                for token in v3_indices[
                    v3_indices // tokens_per_frame == frame
                ].detach().cpu().tolist()
                if int(token) not in protected
            ],
            key=lambda token: (removal_map[token], token),
        )

    additions: list[int] = []
    removals: list[int] = []
    while len(additions) < limit:
        under = [
            frame
            for frame in range(frame_count)
            if int(counts[frame].item()) < int(target_counts[frame].item())
            and additions_by_frame[frame]
        ]
        over = [
            frame
            for frame in range(frame_count)
            if int(counts[frame].item()) > int(target_counts[frame].item())
            and removals_by_frame[frame]
        ]
        if not under or not over:
            break
        add_frame = max(
            under,
            key=lambda frame: (
                int(target_counts[frame].item() - counts[frame].item()),
                -frame,
            ),
        )
        remove_frame = max(
            over,
            key=lambda frame: (
                int(counts[frame].item() - target_counts[frame].item()),
                -removal_map[removals_by_frame[frame][0]],
                -frame,
            ),
        )
        addition = additions_by_frame[add_frame].pop(0)
        removal = removals_by_frame[remove_frame].pop(0)
        additions.append(addition)
        removals.append(removal)
        counts[add_frame] += 1
        counts[remove_frame] -= 1
    return additions, removals


def _trial_selection(
    baseline: torch.Tensor,
    additions: list[int],
    removals: list[int],
    count: int,
) -> torch.Tensor:
    selected = set(int(value) for value in baseline.detach().cpu().tolist())
    for token in removals[:count]:
        selected.remove(int(token))
    selected.update(int(token) for token in additions[:count])
    return torch.tensor(
        sorted(selected),
        dtype=torch.long,
        device=baseline.device,
    )


def _logdet_efficiency(
    design: torch.Tensor,
    baseline: torch.Tensor,
    trial: torch.Tensor,
    ridge: float,
) -> float:
    dimension = int(design.shape[1])
    eye = torch.eye(dimension, dtype=torch.float32, device=design.device)

    def objective(indices: torch.Tensor) -> torch.Tensor:
        rows = design[indices].float()
        information = rows.transpose(0, 1) @ rows + max(1e-6, ridge) * eye
        sign, value = torch.linalg.slogdet(information)
        if float(sign.item()) <= 0.0 or not bool(torch.isfinite(value)):
            raise RuntimeError("V8 information matrix is not positive definite")
        return value

    base = objective(baseline)
    current = objective(trial)
    return float(torch.exp((current - base) / max(1, dimension)).item())


def _query_coverage(
    selected: torch.Tensor,
    analysis: _V3Analysis,
) -> float:
    if analysis.query_relevance.numel() == 0:
        return 1.0
    full = analysis.query_relevance.float().amax(dim=1).clamp_min(1e-6)
    kept = analysis.query_relevance[:, selected].float().amax(dim=1)
    weights = full / full.sum().clamp_min(1e-6)
    return float(((kept / full).clamp(0.0, 1.0) * weights).sum().item())


def _v3_coverage(
    v3_indices: torch.Tensor,
    selected: torch.Tensor,
    metric_flat: torch.Tensor,
    tokens_per_frame: int,
) -> float:
    scores = []
    for frame in torch.unique(v3_indices // tokens_per_frame).tolist():
        source = v3_indices[v3_indices // tokens_per_frame == int(frame)]
        anchors = selected[selected // tokens_per_frame == int(frame)]
        if anchors.numel() == 0:
            return 0.0
        similarity = metric_flat[source].float() @ metric_flat[anchors].float().T
        scores.append(similarity.amax(dim=1).mean())
    if not scores:
        return 0.0
    value = torch.stack(scores).mean()
    return float((0.5 * (value + 1.0)).clamp(0.0, 1.0).item())


def _temporal_alignment(counts: torch.Tensor, target: torch.Tensor) -> float:
    budget = max(1, int(counts.sum().item()))
    distance = float((counts.float() - target.float()).abs().sum().item())
    return max(0.0, 1.0 - distance / (2.0 * budget))


def _selection_objective(
    v3_indices: torch.Tensor,
    selected: torch.Tensor,
    counts: torch.Tensor,
    target_counts: torch.Tensor,
    token_score: torch.Tensor,
    analysis: _V3Analysis,
    tokens_per_frame: int,
    route: _IntentRoute,
) -> Tuple[float, Dict[str, float]]:
    temporal = _temporal_alignment(counts, target_counts)
    query = _query_coverage(selected, analysis)
    evidence = float(token_score[selected].mean().item())
    v3 = _v3_coverage(v3_indices, selected, analysis.metric_flat, tokens_per_frame)
    balance_weight = min(0.60, route.balance_weight)
    query_weight = min(0.45, route.query_weight)
    evidence_weight = min(0.25, route.event_weight)
    total = balance_weight + query_weight + evidence_weight
    if total > 0.85:
        scale = 0.85 / total
        balance_weight *= scale
        query_weight *= scale
        evidence_weight *= scale
    v3_weight = 1.0 - balance_weight - query_weight - evidence_weight
    objective = (
        balance_weight * temporal
        + query_weight * query
        + evidence_weight * evidence
        + v3_weight * v3
    )
    return objective, {
        "temporal_alignment": temporal,
        "query_coverage": query,
        "evidence_score": evidence,
        "v3_coverage": v3,
    }


def _build_plan(
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
    similarity = analysis.metric_flat.float() @ analysis.metric_flat[selected].float().T
    source_frame = torch.arange(
        frame_count,
        device=selected.device,
    ).repeat_interleave(tokens_per_frame)
    anchor_frame = source_frame[selected]
    frame_delta = (source_frame.unsqueeze(1) - anchor_frame.unsqueeze(0)).abs()
    valid = frame_delta == 0
    adjacent = frame_delta == 1
    cross_valid = adjacent & (
        similarity >= _cfg_float(config, "certv8_cross_frame_similarity", 0.88)
    )
    if has_real_times:
        time_gap = torch.abs(
            frame_times[source_frame].unsqueeze(1) - frame_times[anchor_frame].unsqueeze(0)
        )
        cross_valid &= time_gap <= _cfg_float(
            config,
            "certv8_cross_frame_max_seconds",
            8.0,
        )
    valid |= cross_valid
    if not bool(valid.any(dim=1).all()):
        raise RuntimeError("V8 plan has a frame without a reachable anchor")
    similarity = similarity.masked_fill(~valid, -2.0)
    same_component = (
        analysis.component_ids.unsqueeze(1)
        == analysis.component_ids[selected].unsqueeze(0)
    )
    similarity += 0.08 * same_component.float()

    topk = min(2, budget)
    values, assignment = torch.topk(similarity, k=topk, dim=1, largest=True)
    chosen_valid = torch.gather(valid, 1, assignment)
    best_valid = torch.argmax(similarity, dim=1, keepdim=True).expand_as(assignment)
    assignment = torch.where(chosen_valid, assignment, best_valid)
    values = torch.gather(similarity, 1, assignment)
    weights = torch.softmax(
        values.float()
        / max(1e-4, _cfg_float(config, "certv3_assignment_temperature", 0.07)),
        dim=1,
    )
    anchor_positions = torch.arange(budget, dtype=torch.long, device=selected.device)
    assignment[selected, 0] = anchor_positions
    weights[selected] = 0.0
    weights[selected, 0] = 1.0

    source_mass = (0.5 + 0.5 * analysis.demand_weight * total_tokens).clamp(0.25, 2.0)
    protection = torch.maximum(
        analysis.attention[selected],
        analysis.query_score[selected],
    )
    protected_count = min(budget, max(1, int(math.ceil(0.15 * budget))))
    v3_protected = torch.topk(
        protection,
        k=protected_count,
        largest=True,
    ).indices
    alpha = torch.full(
        (budget,),
        min(
            0.75,
            max(0.0, _cfg_float(config, "certv3_fusion_alpha", 0.12)),
        ),
        dtype=torch.float32,
        device=selected.device,
    )
    alpha *= 1.0 - 0.65 * protection.clamp(0.0, 1.0)
    alpha[v3_protected] = 0.0
    protected = protected_v3 | new_anchors
    if protected:
        protected_tensor = torch.tensor(
            sorted(protected),
            dtype=torch.long,
            device=selected.device,
        )
        alpha[torch.isin(selected, protected_tensor)] = 0.0
    return CertVidPlan(
        anchor_indices=selected,
        assignment_indices=assignment,
        assignment_weights=weights,
        source_mass=source_mass,
        fusion_alpha=alpha,
        raw_token_count=total_tokens,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _vector_summary(values: torch.Tensor) -> Dict[str, float]:
    values = values.detach().float().flatten()
    if values.numel() == 0:
        return {"mean": 0.0, "min": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    quantiles = torch.quantile(values, torch.tensor([0.5, 0.9], device=values.device))
    return {
        "mean": float(values.mean().item()),
        "min": float(values.min().item()),
        "p50": float(quantiles[0].item()),
        "p90": float(quantiles[1].item()),
        "max": float(values.max().item()),
    }


def _frame_count_summary(counts: torch.Tensor) -> Dict[str, float]:
    counts = counts.detach().float().flatten()
    total = float(counts.sum().item())
    if counts.numel() == 0 or total <= 0.0:
        return {
            "active_frames": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "cv": 0.0,
            "normalized_entropy": 0.0,
        }
    probabilities = counts / total
    entropy = -(
        probabilities.clamp_min(1e-12) * probabilities.clamp_min(1e-12).log()
    ).sum()
    normalizer = math.log(max(2, int(counts.numel())))
    mean = counts.mean()
    std = counts.std(unbiased=False)
    return {
        "active_frames": int((counts > 0).sum().item()),
        "min": float(counts.min().item()),
        "max": float(counts.max().item()),
        "mean": float(mean.item()),
        "std": float(std.item()),
        "cv": float((std / mean.clamp_min(1e-6)).item()),
        "normalized_entropy": float((entropy / normalizer).item()),
    }


def _selection_profile(
    selected: torch.Tensor,
    counts: torch.Tensor,
    token_score: torch.Tensor,
    analysis: _V3Analysis,
) -> Dict[str, Any]:
    return {
        "frame_distribution": _frame_count_summary(counts),
        "evidence": _vector_summary(token_score[selected]),
        "query": _vector_summary(analysis.query_score[selected]),
        "attention": _vector_summary(analysis.attention[selected]),
        "demand": _vector_summary(analysis.demand_weight[selected]),
        "component_count": int(torch.unique(analysis.component_ids[selected]).numel()),
    }


def _token_record(
    token: int,
    token_score: torch.Tensor,
    analysis: _V3Analysis,
    frame_times: torch.Tensor,
    tokens_per_frame: int,
) -> Dict[str, Any]:
    frame = int(token // tokens_per_frame)
    return {
        "global_index": int(token),
        "frame_index": frame,
        "local_index": int(token % tokens_per_frame),
        "time_seconds": float(frame_times[frame].item()),
        "evidence_score": float(token_score[token].item()),
        "query_score": float(analysis.query_score[token].item()),
        "attention_score": float(analysis.attention[token].item()),
        "demand_weight": float(analysis.demand_weight[token].item()),
        "component_id": int(analysis.component_ids[token].item()),
    }


def _write_diagnostics(config: Any, diagnostics: Dict[str, Any]) -> None:
    template = os.environ.get("CERTV8_DIAGNOSTICS_JSONL", "").strip()
    if not template:
        return
    rank = os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))
    path = template.replace("{rank}", rank).replace("{pid}", str(os.getpid()))
    if "{rank}" not in template and "{pid}" not in template:
        root, extension = os.path.splitext(path)
        path = f"{root}.rank{rank}{extension or '.jsonl'}"
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    record = dict(diagnostics)
    record["sample_id"] = str(getattr(config, "_debug_sample_id", "unknown"))
    record["task"] = getattr(config, "_certvid_task_name", None)
    record["question"] = str(getattr(config, "_certvid_query_text", "") or "")
    category = getattr(config, "_certvid_eval_category", None)
    if category is not None:
        record["eval_category"] = str(category)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")


def _store_diagnostics(config: Any, diagnostics: Dict[str, Any]) -> None:
    config.last_certv8_diagnostics = diagnostics
    config.last_certv8_fallback_reason = diagnostics.get("fallback_reason")
    config.last_certv8_swap_count = int(diagnostics.get("swap_count", 0))
    config.last_certv8_modified_ratio = float(diagnostics.get("modified_ratio", 0.0))
    config.last_certv8_v3_overlap_ratio = float(
        diagnostics.get("v3_overlap_ratio", 1.0)
    )
    config.last_certv8_d_efficiency = float(diagnostics.get("d_efficiency", 1.0))
    _write_diagnostics(config, diagnostics)
    if bool(getattr(config, "certv8_debug", False)):
        print(
            "[certvid-v8] "
            f"sample={getattr(config, '_debug_sample_id', 'unknown')} "
            f"intent={diagnostics.get('query_intent', 'unknown')} "
            f"fallback={diagnostics.get('fallback_reason')} "
            f"tokens={diagnostics.get('budget', 0)}/"
            f"{diagnostics.get('raw_token_count', 0)} "
            f"duration={diagnostics.get('duration_seconds', 0.0):.1f}s "
            f"max_gap={diagnostics.get('max_frame_gap_seconds', 0.0):.1f}s "
            f"deficit={diagnostics.get('base_deficit', 0.0):.4f}->"
            f"{diagnostics.get('final_deficit', 0.0):.4f} "
            f"query={diagnostics.get('base_query_coverage', 1.0):.4f}->"
            f"{diagnostics.get('final_query_coverage', 1.0):.4f} "
            f"swaps={diagnostics.get('swap_count', 0)} "
            f"swap_cap={diagnostics.get('hard_swap_cap', 0.0):.3f} "
            f"protected={diagnostics.get('protected_anchor_count', 0)} "
            f"v3_overlap={diagnostics.get('v3_overlap_ratio', 1.0):.3f} "
            f"D-eff={diagnostics.get('d_efficiency', 1.0):.4f} "
            f"qconf={diagnostics.get('query_confidence', 0.0):.4f} "
            f"minCV={diagnostics.get('minimum_trial_frame_cv', 0.0):.4f} "
            f"v3_frames={diagnostics.get('v3_frame_counts', [])} "
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
    diagnostics.setdefault("final_deficit", diagnostics.get("base_deficit", 0.0))
    diagnostics.setdefault(
        "final_query_coverage",
        diagnostics.get("base_query_coverage", 1.0),
    )
    diagnostics.setdefault(
        "final_frame_counts",
        diagnostics.get("v3_frame_counts", []),
    )
    diagnostics.setdefault(
        "final_frame_distribution",
        diagnostics.get("v3_frame_distribution", {}),
    )
    if plan is not None:
        config._certvid_plan = plan
    config.last_adapter_variant = "certvid_v8"
    _store_diagnostics(config, diagnostics)
    return output, indices


def certvid_v8_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    config: Any,
    question_features: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Repair V3 temporal/query deficits without changing its fixed budget."""
    if video_features.ndim != 3:
        raise ValueError(
            f"expected video_features [T, HW, D], got {tuple(video_features.shape)}"
        )
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
    v3_counts = torch.bincount(
        v3_indices // max(1, tokens_per_frame),
        minlength=frame_count,
    )
    diagnostics: Dict[str, Any] = {
        "fallback_reason": None,
        "raw_token_count": int(frame_count * tokens_per_frame),
        "budget": int(v3_indices.numel()),
        "nominal_retention_ratio": float(getattr(config, "retention_ratio", 0.0)),
        "expansion": float(getattr(config, "expansion", 1.0)),
        "outer_retention_ratio": float(
            v3_indices.numel() / max(1, frame_count * tokens_per_frame)
        ),
        "frame_count": int(frame_count),
        "tokens_per_frame": int(tokens_per_frame),
        "v3_frame_counts": [
            int(value) for value in v3_counts.detach().cpu().tolist()
        ],
        "v3_frame_distribution": _frame_count_summary(v3_counts),
    }
    if not bool(getattr(config, "certv8_enabled", True)):
        return _fallback(
            config,
            diagnostics,
            "disabled",
            v3_output,
            v3_indices,
            v3_plan,
        )
    if v3_plan is None:
        return _fallback(
            config,
            diagnostics,
            "missing_v3_plan",
            v3_output,
            v3_indices,
            v3_plan,
        )
    if bool(sink.get("identity", False)) or not sink:
        return _fallback(
            config,
            diagnostics,
            "identity_budget",
            v3_output,
            v3_indices,
            v3_plan,
        )

    try:
        analysis = _analysis_from_sink(sink)
        route = _query_route(config, analysis)
        diagnostics["query_intent"] = route.name
        diagnostics["route"] = {
            "repair_strength": route.repair_strength,
            "peak_count": route.peak_count,
            "floor_ratio": route.floor_ratio,
            "cap_ratio": route.cap_ratio,
            "max_swap_ratio": route.max_swap_ratio,
            "query_weight": route.query_weight,
            "event_weight": route.event_weight,
            "balance_weight": route.balance_weight,
            "d_efficiency_floor": route.d_efficiency_floor,
        }

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
        max_gap = (
            float((frame_times[1:] - frame_times[:-1]).amax().item())
            if has_real_times and frame_count > 1
            else 0.0
        )
        diagnostics.update(
            {
                "timestamp_source": timestamp_source,
                "duration_seconds": duration,
                "max_frame_gap_seconds": max_gap,
            }
        )
        if not has_real_times:
            return _fallback(
                config,
                diagnostics,
                "missing_real_timestamps",
                v3_output,
                v3_indices,
                v3_plan,
            )
        if max_gap + 1e-12 < max(
            0.0,
            _cfg_float(config, "certv8_min_horizon_gap_seconds", 4.0),
        ):
            return _fallback(
                config,
                diagnostics,
                "short_horizon",
                v3_output,
                v3_indices,
                v3_plan,
            )

        budget = int(v3_indices.numel())
        token_score, frame_quality, frame_event = _token_scores(
            video_features,
            cls_attention,
            analysis,
            route,
        )
        query_demand, query_peaks = _frame_query_demand(
            analysis,
            frame_count,
            tokens_per_frame,
            route,
            config,
        )
        diagnostics["query_peak_frames"] = query_peaks
        diagnostics["frame_quality"] = [
            float(value) for value in frame_quality.detach().cpu().tolist()
        ]
        diagnostics["frame_event"] = [
            float(value) for value in frame_event.detach().cpu().tolist()
        ]
        diagnostics["frame_query_demand"] = [
            float(value) for value in query_demand.detach().cpu().tolist()
        ]
        diagnostics["v3_profile"] = _selection_profile(
            v3_indices,
            v3_counts,
            token_score,
            analysis,
        )

        protected = {
            int(token)
            for token in v3_indices[
                v3_plan.fusion_alpha <= 1e-12
            ].detach().cpu().tolist()
        }
        design_ratio = min(
            0.40,
            max(0.0, _cfg_float(config, "certv8_design_protect_ratio", 0.08)),
        )
        design_count = min(budget, int(math.ceil(design_ratio * budget)))
        if design_count > 0:
            leverage = analysis.design[v3_indices].float().square().sum(dim=1)
            positions = torch.topk(leverage, k=design_count, largest=True).indices
            protected.update(
                int(token)
                for token in v3_indices[positions].detach().cpu().tolist()
            )
        query_protect_ratio = min(
            0.20,
            max(0.0, _cfg_float(config, "certv8_query_protect_ratio", 0.05)),
        )
        query_protect = min(budget, int(math.ceil(query_protect_ratio * budget)))
        if query_protect > 0 and analysis.query_relevance.numel() > 0:
            positions = torch.topk(
                analysis.query_score[v3_indices],
                k=query_protect,
                largest=True,
            ).indices
            protected.update(
                int(token)
                for token in v3_indices[positions].detach().cpu().tolist()
            )
        protected_tensor = torch.tensor(
            sorted(protected),
            dtype=torch.long,
            device=v3_indices.device,
        )
        protected_counts = torch.bincount(
            protected_tensor // tokens_per_frame,
            minlength=frame_count,
        )
        diagnostics["protected_anchor_count"] = int(len(protected))
        diagnostics["protected_frame_counts"] = [
            int(value) for value in protected_counts.detach().cpu().tolist()
        ]
        target_counts = _target_frame_counts(
            v3_counts,
            protected_counts,
            frame_quality,
            frame_event,
            query_demand,
            tokens_per_frame,
            route,
        )
        diagnostics["target_frame_counts"] = [
            int(value) for value in target_counts.detach().cpu().tolist()
        ]

        base_deficit = 1.0 - _temporal_alignment(v3_counts, target_counts)
        base_query = _query_coverage(v3_indices, analysis)
        diagnostics["base_deficit"] = base_deficit
        diagnostics["base_query_coverage"] = base_query
        query_deficit = max(0.0, 1.0 - base_query)
        min_deficit = max(
            0.0,
            _cfg_float(config, "certv8_min_deficit", 0.04),
        )
        if max(base_deficit, query_deficit) + 1e-12 < min_deficit:
            return _fallback(
                config,
                diagnostics,
                "no_temporal_or_query_deficit",
                v3_output,
                v3_indices,
                v3_plan,
            )

        hard_swap_cap = min(
            0.50,
            max(0.0, _cfg_float(config, "certv8_max_swap_ratio", 0.12)),
        )
        # Grow the trust region with measured deficit, but saturate before
        # large, uncertain deficits trigger destructive wholesale rewrites.
        deficit_swap_cap = min(
            hard_swap_cap,
            0.07 + 0.40 * min(0.125, max(0.0, base_deficit)),
        )
        swap_limit = min(
            budget,
            int(
                math.ceil(
                    min(route.max_swap_ratio, hard_swap_cap, deficit_swap_cap)
                    * budget
                )
            ),
        )
        additions, removals = _propose_swaps(
            v3_indices,
            protected,
            target_counts,
            token_score,
            analysis,
            frame_count,
            tokens_per_frame,
            swap_limit,
        )
        diagnostics["swap_limit"] = swap_limit
        diagnostics["proposed_swap_count"] = min(len(additions), len(removals))
        if not additions:
            return _fallback(
                config,
                diagnostics,
                "no_budget_repair",
                v3_output,
                v3_indices,
                v3_plan,
            )

        base_objective, base_metrics = _selection_objective(
            v3_indices,
            v3_indices,
            v3_counts,
            target_counts,
            token_score,
            analysis,
            tokens_per_frame,
            route,
        )
        diagnostics.update({f"base_{key}": value for key, value in base_metrics.items()})
        diagnostics["base_objective"] = base_objective

        min_gain = max(0.0, _cfg_float(config, "certv8_min_objective_gain", 0.001))
        count = min(len(additions), len(removals), swap_limit)
        selected: Optional[torch.Tensor] = None
        accepted_metrics: Dict[str, float] = {}
        accepted_objective = base_objective
        accepted_efficiency = 1.0
        concentration_preserve = min(
            1.0,
            max(
                0.0,
                _cfg_float(
                    config,
                    "certv8_concentration_preserve_ratio",
                    0.55,
                ),
            ),
        )
        base_distribution = _frame_count_summary(v3_counts)
        minimum_trial_cv = (
            concentration_preserve * float(base_distribution["cv"])
        )
        hard_efficiency_floor = max(
            route.d_efficiency_floor,
            min(
                1.0,
                max(
                    0.0,
                    _cfg_float(config, "certv8_d_efficiency_floor", 0.97),
                ),
            ),
        )
        diagnostics["query_confidence"] = float(analysis.query_confidence)
        diagnostics["hard_swap_cap"] = hard_swap_cap
        diagnostics["deficit_swap_cap"] = deficit_swap_cap
        diagnostics["hard_d_efficiency_floor"] = hard_efficiency_floor
        diagnostics["minimum_trial_frame_cv"] = minimum_trial_cv
        while count > 0:
            trial = _trial_selection(v3_indices, additions, removals, count)
            trial_counts = torch.bincount(
                trial // tokens_per_frame,
                minlength=frame_count,
            )
            efficiency = _logdet_efficiency(
                analysis.design,
                v3_indices,
                trial,
                analysis.ridge,
            )
            objective, metrics = _selection_objective(
                v3_indices,
                trial,
                trial_counts,
                target_counts,
                token_score,
                analysis,
                tokens_per_frame,
                route,
            )
            query_safe = metrics["query_coverage"] + 1e-8 >= base_query
            deficit_improved = (
                1.0 - metrics["temporal_alignment"] + 1e-8 < base_deficit
            )
            trial_cv = float(_frame_count_summary(trial_counts)["cv"])
            concentration_safe = trial_cv + 1e-8 >= minimum_trial_cv
            if (
                efficiency + 1e-12 >= hard_efficiency_floor
                and objective + 1e-12 >= base_objective + min_gain
                and query_safe
                and deficit_improved
                and concentration_safe
            ):
                selected = trial
                accepted_metrics = metrics
                accepted_objective = objective
                accepted_efficiency = efficiency
                break
            next_count = int(math.floor(0.75 * count))
            count = next_count if next_count < count else count - 1
        if selected is None:
            return _fallback(
                config,
                diagnostics,
                "repair_guard",
                v3_output,
                v3_indices,
                v3_plan,
            )

        accepted_additions = set(int(value) for value in additions[:count])
        accepted_removals = set(int(value) for value in removals[:count])
        plan = _build_plan(
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
        source_frames = torch.arange(
            frame_count,
            device=selected.device,
        ).repeat_interleave(tokens_per_frame)
        assigned_frames = source_frames[selected][plan.assignment_indices]
        unsafe = int(
            ((source_frames.unsqueeze(1) - assigned_frames).abs() > 1)
            .any(dim=1)
            .sum()
            .item()
        )
        if unsafe:
            return _fallback(
                config,
                diagnostics,
                "unsafe_assignment",
                v3_output,
                v3_indices,
                v3_plan,
            )

        output = apply_certvid_plan(
            video_features.reshape(frame_count * tokens_per_frame, -1),
            plan,
        )
        final_counts = torch.bincount(
            selected // tokens_per_frame,
            minlength=frame_count,
        )
        diagnostics.update(
            {
                "swap_count": count,
                "modified_ratio": count / max(1, budget),
                "v3_overlap_ratio": float(
                    torch.isin(selected, v3_indices).float().mean().item()
                ),
                "d_efficiency": accepted_efficiency,
                "final_objective": accepted_objective,
                "objective_gain": accepted_objective - base_objective,
                "final_deficit": 1.0 - accepted_metrics["temporal_alignment"],
                "final_query_coverage": accepted_metrics["query_coverage"],
                "final_v3_coverage": accepted_metrics["v3_coverage"],
                "final_evidence_score": accepted_metrics["evidence_score"],
                "unsafe_assignment_count": unsafe,
                "final_frame_counts": [
                    int(value)
                    for value in final_counts.detach().cpu().tolist()
                ],
                "final_frame_distribution": _frame_count_summary(final_counts),
                "v3_profile": _selection_profile(
                    v3_indices,
                    v3_counts,
                    token_score,
                    analysis,
                ),
                "final_profile": _selection_profile(
                    selected,
                    final_counts,
                    token_score,
                    analysis,
                ),
                "added_tokens": [
                    _token_record(
                        token,
                        token_score,
                        analysis,
                        frame_times,
                        tokens_per_frame,
                    )
                    for token in sorted(accepted_additions)
                ],
                "removed_tokens": [
                    _token_record(
                        token,
                        token_score,
                        analysis,
                        frame_times,
                        tokens_per_frame,
                    )
                    for token in sorted(accepted_removals)
                ],
            }
        )
        if os.environ.get("CERTV8_DIAGNOSTICS_DETAIL", "summary").strip().lower() in {
            "tokens",
            "full",
        }:
            diagnostics["v3_selected_indices"] = [
                int(value) for value in v3_indices.detach().cpu().tolist()
            ]
            diagnostics["final_selected_indices"] = [
                int(value) for value in selected.detach().cpu().tolist()
            ]
        config._certvid_plan = plan
        config.vision_token_length = int(output.shape[0])
        config.visual_token_length = int(output.shape[0])
        config.llm_token_length = None
        config.last_adapter_variant = "certvid_v8"
        config.last_adapter_raw_tokens = float(frame_count * tokens_per_frame)
        config.last_adapter_output_tokens = float(output.shape[0])
        _store_diagnostics(config, diagnostics)
        return output, selected
    except Exception as error:
        diagnostics["optimization_error"] = f"{type(error).__name__}: {error}"
        return _fallback(
            config,
            diagnostics,
            "optimization_error",
            v3_output,
            v3_indices,
            v3_plan,
        )
