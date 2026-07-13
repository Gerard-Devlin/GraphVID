from __future__ import annotations

import math
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


def _effective_ratio(config: FlashVidConfig) -> float:
    ratio = _cfg_float(config, "retention_ratio", 0.10)
    if bool(getattr(config, "prism_budget_uses_expansion", True)):
        ratio *= _cfg_float(config, "expansion", 1.0)
    return min(1.0, max(0.0, ratio))


def _minmax(values: torch.Tensor, dim: int = -1) -> torch.Tensor:
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
    lo = values.amin(dim=dim, keepdim=True)
    hi = values.amax(dim=dim, keepdim=True)
    return ((values - lo) / (hi - lo + 1e-6)).clamp_(0.0, 1.0)


def _stable_descending(values: torch.Tensor) -> torch.Tensor:
    values = torch.nan_to_num(values.float(), nan=-1e9, posinf=1e9, neginf=-1e9)
    return torch.argsort(values, descending=True, stable=True)


def _rank01(values: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if values.shape[dim] <= 1:
        return torch.ones_like(values, dtype=torch.float32)
    clean = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
    order = torch.argsort(clean, dim=dim, stable=True)
    ranks = torch.argsort(order, dim=dim, stable=True).float()
    ranks = ranks / float(values.shape[dim] - 1)
    spread = clean.amax(dim=dim, keepdim=True) - clean.amin(dim=dim, keepdim=True)
    return torch.where(spread > 1e-8, ranks, torch.full_like(ranks, 0.5))


def _signed_hash_projection(flat: torch.Tensor, metric_dim: int) -> torch.Tensor:
    if metric_dim <= 0 or flat.shape[-1] <= metric_dim:
        return flat.float()
    feature_dim = flat.shape[-1]
    device = flat.device
    dimensions = torch.arange(feature_dim, device=device, dtype=torch.long)
    signs = (((dimensions * 1103515245 + 12345) >> 16) & 1).float().mul_(2.0).sub_(1.0)

    full_length = (feature_dim // metric_dim) * metric_dim
    grouped = (flat[:, :full_length].float() * signs[:full_length]).reshape(
        flat.shape[0], -1, metric_dim
    )
    projected = grouped.sum(dim=1)
    if full_length < feature_dim:
        remainder = flat[:, full_length:].float() * signs[full_length:]
        projected[:, : remainder.shape[-1]] += remainder
    return projected / math.sqrt(max(1.0, math.ceil(feature_dim / metric_dim)))


def _project_metric(features: torch.Tensor, metric_dim: int) -> torch.Tensor:
    flat = features.reshape(-1, features.shape[-1])
    flat = _signed_hash_projection(flat, metric_dim)
    return F.normalize(flat, p=2, dim=-1, eps=1e-6)


def _reshape_deepstack_level(
    level: torch.Tensor,
    frame_count: int,
    tokens_per_frame: int,
) -> Optional[torch.Tensor]:
    if not isinstance(level, torch.Tensor) or level.numel() == 0:
        return None
    total_tokens = frame_count * tokens_per_frame
    if level.ndim < 2 or level.shape[-1] <= 0:
        return None
    if level.numel() != total_tokens * level.shape[-1]:
        return None
    return level.reshape(frame_count, tokens_per_frame, level.shape[-1])


def _prepare_levels(
    video_features: torch.Tensor,
    deepstack_features: Optional[Sequence[torch.Tensor]],
    metric_dim: int,
) -> list[torch.Tensor]:
    frame_count, tokens_per_frame, _ = video_features.shape
    levels: list[torch.Tensor] = []
    if deepstack_features is not None:
        for level in deepstack_features:
            reshaped = _reshape_deepstack_level(level, frame_count, tokens_per_frame)
            if reshaped is not None:
                projected = _project_metric(reshaped, metric_dim)
                levels.append(projected.to(device=video_features.device))
    levels.append(_project_metric(video_features, metric_dim))
    return levels


def _debiased_attention(cls_attention: torch.Tensor) -> torch.Tensor:
    attention = torch.nan_to_num(cls_attention.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-8)
    within_frame = _rank01(attention, dim=-1)

    # Remove persistent spatial preference before ranking attention evidence.
    log_attention = attention.log()
    spatial_baseline = log_attention.median(dim=0, keepdim=True).values
    residual = log_attention - spatial_baseline
    residual_scale = (residual - residual.median(dim=-1, keepdim=True).values).abs().median(
        dim=-1, keepdim=True
    ).values
    residual_score = torch.sigmoid(residual / (1.4826 * residual_scale + 1e-5))
    return (0.62 * within_frame + 0.38 * residual_score).clamp_(0.0, 1.0)


def _question_atoms(
    question_features: Optional[torch.Tensor],
    max_atoms: int,
    metric_dim: int,
    device: torch.device,
) -> torch.Tensor:
    if question_features is None or question_features.numel() == 0 or max_atoms <= 0:
        return torch.empty((0, max(1, metric_dim)), dtype=torch.float32, device=device)
    question = question_features.reshape(-1, question_features.shape[-1]).to(device=device)
    question = _signed_hash_projection(question, metric_dim)
    question = F.normalize(question, p=2, dim=-1, eps=1e-6)

    atom_count = min(max_atoms, int(question.shape[0]))
    center = F.normalize(question.mean(dim=0), p=2, dim=-1, eps=1e-6)
    selected = [int(torch.argmin(question @ center).item())]
    min_distance = 1.0 - question @ question[selected[0]]
    for _ in range(1, atom_count):
        min_distance[selected] = -1.0
        next_index = int(torch.argmax(min_distance).item())
        if float(min_distance[next_index].item()) < 0.01:
            break
        selected.append(next_index)
        min_distance = torch.minimum(min_distance, 1.0 - question @ question[next_index])
    index = torch.tensor(selected, dtype=torch.long, device=device)
    return question[index]


def _question_signal(
    atoms: torch.Tensor,
    visual_metric: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    token_count = visual_metric.shape[0]
    if atoms.numel() == 0:
        return visual_metric.new_zeros(token_count, dtype=torch.float32), 0.0

    similarities = atoms @ visual_metric.transpose(0, 1)
    raw_relevance = similarities.max(dim=0).values
    relevance = 0.55 * _rank01(raw_relevance, dim=0) + 0.45 * _minmax(raw_relevance, dim=0)

    top_count = max(1, int(math.ceil(token_count * 0.05)))
    top_mean = torch.topk(raw_relevance, k=top_count).values.mean()
    median = raw_relevance.median()
    spread = raw_relevance.std(unbiased=False).clamp_min(1e-6)
    contrast = (top_mean - median) / spread

    scaled = (raw_relevance - median) / spread
    probabilities = torch.softmax(scaled, dim=0)
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum()
    entropy /= math.log(max(2, token_count))
    concentration = (1.0 - entropy).clamp(0.0, 1.0)
    contrast_confidence = ((contrast - 1.15) / 1.50).clamp(0.0, 1.0)
    confidence = float((0.75 * contrast_confidence + 0.25 * (4.0 * concentration).clamp(0.0, 1.0)).item())
    return relevance.clamp_(0.0, 1.0), confidence


def _multi_scale_events(
    levels: Sequence[torch.Tensor],
    frame_count: int,
    tokens_per_frame: int,
) -> torch.Tensor:
    device = levels[-1].device
    token_event = torch.zeros((frame_count, tokens_per_frame), dtype=torch.float32, device=device)
    frame_event = torch.zeros(frame_count, dtype=torch.float32, device=device)

    for flat_level in levels:
        level = flat_level.view(frame_count, tokens_per_frame, -1)
        frame_representatives = F.normalize(level.mean(dim=1), p=2, dim=-1, eps=1e-6)
        for lag in (1, 2, 4):
            if lag >= frame_count:
                continue
            token_novelty = (1.0 - (level[lag:] * level[:-lag]).sum(dim=-1)).clamp(0.0, 2.0) * 0.5
            token_event[lag:] = torch.maximum(token_event[lag:], token_novelty)
            token_event[:-lag] = torch.maximum(token_event[:-lag], 0.5 * token_novelty)

            frame_novelty = (
                1.0 - (frame_representatives[lag:] * frame_representatives[:-lag]).sum(dim=-1)
            ).clamp(0.0, 2.0) * 0.5
            frame_event[lag:] = torch.maximum(frame_event[lag:], frame_novelty)
            frame_event[:-lag] = torch.maximum(frame_event[:-lag], 0.5 * frame_novelty)

        if frame_count > 2:
            incoming = frame_representatives[1:-1] - frame_representatives[:-2]
            outgoing = frame_representatives[2:] - frame_representatives[1:-1]
            curvature = (1.0 - F.cosine_similarity(incoming, outgoing, dim=-1, eps=1e-6)).clamp(0.0, 2.0) * 0.5
            frame_event[1:-1] = torch.maximum(frame_event[1:-1], curvature)

    token_flat = token_event.flatten()
    frame_flat = frame_event[:, None].expand(-1, tokens_per_frame).flatten()
    combined = 0.72 * token_flat + 0.28 * frame_flat
    return (0.60 * _rank01(combined, dim=0) + 0.40 * _minmax(combined, dim=0)).clamp_(0.0, 1.0)


def _cross_level_disagreement(levels: Sequence[torch.Tensor]) -> torch.Tensor:
    token_count = levels[-1].shape[0]
    if len(levels) <= 1:
        return levels[-1].new_zeros(token_count, dtype=torch.float32)
    stack = torch.stack(list(levels), dim=0)
    consensus = F.normalize(stack.mean(dim=0), p=2, dim=-1, eps=1e-6)
    disagreement = (1.0 - (stack * consensus.unsqueeze(0)).sum(dim=-1)).clamp(0.0, 2.0).mean(dim=0) * 0.5
    return (0.60 * _rank01(disagreement, dim=0) + 0.40 * _minmax(disagreement, dim=0)).clamp_(0.0, 1.0)


def _detail_signal(final_level: torch.Tensor, frame_count: int, tokens_per_frame: int) -> torch.Tensor:
    level = final_level.view(frame_count, tokens_per_frame, -1)
    frame_center = F.normalize(level.mean(dim=1), p=2, dim=-1, eps=1e-6)
    detail = (1.0 - (level * frame_center[:, None, :]).sum(dim=-1)).clamp(0.0, 2.0).flatten() * 0.5
    return _rank01(detail, dim=0)


def _router_weights(
    attention: torch.Tensor,
    events: torch.Tensor,
    disagreement: torch.Tensor,
    query_confidence: float,
    config: FlashVidConfig,
) -> tuple[float, float, float, float, float]:
    attention_weight = max(0.0, _cfg_float(config, "prism_attention_weight", 0.30))
    event_weight = max(0.0, _cfg_float(config, "prism_event_weight", 0.24))
    query_weight = max(0.0, _cfg_float(config, "prism_query_weight", 0.16)) * query_confidence
    disagreement_weight = max(0.0, _cfg_float(config, "prism_disagreement_weight", 0.16))
    detail_weight = 0.10
    strength = min(1.0, max(0.0, _cfg_float(config, "prism_router_strength", 0.50)))

    token_count = attention.numel()
    top_count = max(1, int(math.ceil(token_count * 0.10)))
    attention_concentration = float(torch.topk(attention, k=top_count).values.mean().item())
    event_strength = float(torch.topk(events, k=top_count).values.mean().item())
    disagreement_strength = float(torch.topk(disagreement, k=top_count).values.mean().item())

    attention_weight *= 1.0 + strength * (attention_concentration - 0.50)
    event_weight *= 1.0 + strength * (event_strength - 0.50)
    disagreement_weight *= 1.0 + strength * (disagreement_strength - 0.50)
    total = attention_weight + event_weight + query_weight + disagreement_weight + detail_weight
    if total <= 1e-8:
        return 0.30, 0.24, 0.0, 0.16, 0.10
    return tuple(
        value / total
        for value in (attention_weight, event_weight, query_weight, disagreement_weight, detail_weight)
    )


def _frame_floor(
    quality: torch.Tensor,
    frame_count: int,
    tokens_per_frame: int,
    budget: int,
    floor_ratio: float,
) -> torch.Tensor:
    device = quality.device
    floor_total = min(budget, max(1, int(round(budget * max(0.0, min(1.0, floor_ratio))))))
    quality_frames = quality.view(frame_count, tokens_per_frame)

    selected: list[torch.Tensor] = []
    if floor_total < frame_count:
        boundaries = torch.linspace(0, frame_count, floor_total + 1, device=device)
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            start = min(frame_count - 1, int(math.floor(float(left.item()))))
            stop = min(frame_count, max(start + 1, int(math.ceil(float(right.item())))))
            region = quality_frames[start:stop].reshape(-1)
            local = int(_stable_descending(region)[0].item())
            frame_offset, token_offset = divmod(local, tokens_per_frame)
            selected.append(torch.tensor((start + frame_offset) * tokens_per_frame + token_offset, device=device))
    else:
        base, remainder = divmod(floor_total, frame_count)
        frame_priority = _stable_descending(quality_frames.amax(dim=1))
        extra_frames = set(frame_priority[:remainder].detach().cpu().tolist())
        for frame_index in range(frame_count):
            keep = min(tokens_per_frame, base + (1 if frame_index in extra_frames else 0))
            local = _stable_descending(quality_frames[frame_index])[:keep]
            selected.extend(frame_index * tokens_per_frame + token for token in local.unbind())

    if not selected:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.unique(torch.stack(selected).long(), sorted=True)[:budget]


def _stratified_indices(token_count: int, count: int, device: torch.device) -> torch.Tensor:
    if count <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if count >= token_count:
        return torch.arange(token_count, dtype=torch.long, device=device)
    centers = (torch.arange(count, device=device, dtype=torch.float32) + 0.5) * token_count / count
    return centers.floor().long().clamp_max(token_count - 1)


def _protected_union(primary: torch.Tensor, secondary: torch.Tensor, limit: int) -> torch.Tensor:
    if limit <= 0:
        return primary.new_empty(0, dtype=torch.long)
    primary = torch.unique(primary.long(), sorted=True)
    if primary.numel() > limit:
        positions = _stratified_indices(int(primary.numel()), limit, primary.device)
        return primary[positions]
    secondary = torch.unique(secondary.long(), sorted=True)
    secondary = secondary[~torch.isin(secondary, primary)]
    room = limit - int(primary.numel())
    return torch.unique(torch.cat([primary, secondary[:room]]).long(), sorted=True)


def _candidate_pool(
    budget: int,
    mandatory: torch.Tensor,
    quality: torch.Tensor,
    attention: torch.Tensor,
    events: torch.Tensor,
    disagreement: torch.Tensor,
    query: torch.Tensor,
    multiplier: float,
) -> torch.Tensor:
    token_count = quality.numel()
    target = min(token_count, max(budget, int(math.ceil(budget * max(1.0, multiplier)))))
    device = quality.device
    protected_uniform = _stratified_indices(token_count, max(1, int(round(target * 0.12))), device)

    parts = [mandatory, protected_uniform]
    for signal, fraction in (
        (quality, 0.45),
        (events, 0.24),
        (disagreement, 0.20),
        (query, 0.18),
        (attention, 0.18),
    ):
        if float((signal.amax() - signal.amin()).item()) <= 1e-6:
            continue
        count = min(token_count, max(1, int(math.ceil(target * fraction))))
        parts.append(_stable_descending(signal)[:count])
    pool = torch.unique(torch.cat(parts).long(), sorted=True)

    protected = _protected_union(mandatory, protected_uniform, target)
    if pool.numel() > target:
        is_protected = torch.isin(pool, protected)
        optional = pool[~is_protected]
        priority = torch.maximum(quality[optional], torch.maximum(events[optional], disagreement[optional]))
        priority = torch.maximum(priority, query[optional])
        room = max(0, target - int(protected.numel()))
        optional = optional[_stable_descending(priority)[:room]]
        pool = torch.unique(torch.cat([protected[:target], optional]).long(), sorted=True)

    if pool.numel() < target:
        remaining = _stable_descending(quality)
        remaining = remaining[~torch.isin(remaining, pool)]
        pool = torch.unique(torch.cat([pool, remaining[: target - pool.numel()]]).long(), sorted=True)
    return pool


def _probe_set(
    quality: torch.Tensor,
    events: torch.Tensor,
    disagreement: torch.Tensor,
    query: torch.Tensor,
    mandatory: torch.Tensor,
    max_probes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_count = quality.numel()
    count = min(token_count, max(1, max_probes))
    device = quality.device
    uniform = _stratified_indices(token_count, max(1, count // 3), device)
    quota = max(1, count // 5)
    parts = [mandatory, uniform, _stable_descending(quality)[:quota]]
    for signal in (events, disagreement, query):
        if float((signal.amax() - signal.amin()).item()) > 1e-6:
            parts.append(_stable_descending(signal)[:quota])
    probes = torch.unique(torch.cat(parts).long(), sorted=True)
    if probes.numel() > count:
        protected = _protected_union(mandatory, uniform, count)
        optional = probes[~torch.isin(probes, protected)]
        room = max(0, count - int(protected.numel()))
        optional = optional[_stable_descending(quality[optional])[:room]]
        probes = torch.unique(torch.cat([protected, optional]).long(), sorted=True)
    elif probes.numel() < count:
        remaining = _stable_descending(quality)
        remaining = remaining[~torch.isin(remaining, probes)]
        probes = torch.unique(torch.cat([probes, remaining[: count - probes.numel()]]).long(), sorted=True)

    weights = 0.35 + 0.65 * quality[probes]
    weights = weights / weights.sum().clamp_min(1e-6)
    return probes, weights


def _select_coreset(
    levels: Sequence[torch.Tensor],
    candidates: torch.Tensor,
    probes: torch.Tensor,
    probe_weights: torch.Tensor,
    mandatory: torch.Tensor,
    quality: torch.Tensor,
    frame_count: int,
    tokens_per_frame: int,
    budget: int,
    config: FlashVidConfig,
) -> torch.Tensor:
    if budget >= quality.numel():
        return torch.arange(quality.numel(), dtype=torch.long, device=quality.device)

    similarity_matrices = [
        ((level[probes] @ level[candidates].transpose(0, 1)).clamp(-1.0, 1.0) + 1.0) * 0.5
        for level in levels
    ]
    level_weights = torch.linspace(1.0, 2.0, len(levels), device=quality.device)
    level_weights /= level_weights.sum()

    selected_mask = torch.isin(candidates, mandatory)
    if int(selected_mask.sum().item()) > budget:
        selected_mask[:] = False
        mandatory_priority = _stable_descending(quality[mandatory])[:budget]
        selected_mask = torch.isin(candidates, mandatory[mandatory_priority])

    best_coverage: list[torch.Tensor] = []
    for similarity in similarity_matrices:
        if bool(selected_mask.any()):
            best_coverage.append(similarity[:, selected_mask].amax(dim=1))
        else:
            best_coverage.append(similarity.new_zeros(similarity.shape[0]))

    selected_count = int(selected_mask.sum().item())
    frame_ids = torch.div(candidates, tokens_per_frame, rounding_mode="floor")
    frame_counts = torch.bincount(frame_ids[selected_mask], minlength=frame_count).float()
    batch_size = max(1, _cfg_int(config, "prism_batch_size", 8))
    coverage_weight = min(1.0, max(0.0, _cfg_float(config, "prism_coverage_weight", 0.68)))
    pareto_weight = min(1.0, max(0.0, _cfg_float(config, "prism_pareto_weight", 0.20)))
    final_candidate_metric = levels[-1][candidates]

    while selected_count < budget:
        available = ~selected_mask
        if not bool(available.any()):
            break
        marginal_levels = []
        for similarity, best in zip(similarity_matrices, best_coverage):
            improvement = (similarity - best[:, None]).clamp_min(0.0)
            marginal_levels.append((improvement * probe_weights[:, None]).sum(dim=0))
        marginal_stack = torch.stack(marginal_levels, dim=0)
        weighted_marginal = (marginal_stack * level_weights[:, None]).sum(dim=0)
        pareto_marginal = marginal_stack.amin(dim=0)
        coverage = weighted_marginal + pareto_weight * pareto_marginal
        coverage = coverage / coverage.amax().clamp_min(1e-6)

        temporal_balance = 1.0 / (1.0 + frame_counts[frame_ids])
        temporal_balance = temporal_balance / temporal_balance.amax().clamp_min(1e-6)
        score = coverage_weight * coverage + (1.0 - coverage_weight) * quality[candidates] + 0.08 * temporal_balance
        score[~available] = -torch.inf

        remaining = budget - selected_count
        take = min(batch_size, remaining, int(available.sum().item()))
        proposal_count = min(int(available.sum().item()), max(take, take * 6))
        proposals = _stable_descending(score)[:proposal_count]
        chosen: list[int] = []
        proposal_available = torch.ones(proposals.numel(), dtype=torch.bool, device=quality.device)
        for _ in range(take):
            proposal_score = score[proposals].clone()
            proposal_score[~proposal_available] = -torch.inf
            if chosen:
                chosen_tensor = torch.tensor(chosen, dtype=torch.long, device=quality.device)
                redundancy = (
                    final_candidate_metric[proposals]
                    @ final_candidate_metric[chosen_tensor].transpose(0, 1)
                ).amax(dim=1).clamp_min(0.0)
                proposal_score -= 0.10 * redundancy
            local_choice = int(torch.argmax(proposal_score).item())
            if not bool(torch.isfinite(proposal_score[local_choice])):
                break
            chosen.append(int(proposals[local_choice].item()))
            proposal_available[local_choice] = False

        if not chosen:
            break
        chosen_tensor = torch.tensor(chosen, dtype=torch.long, device=quality.device)
        selected_mask[chosen_tensor] = True
        selected_count += int(chosen_tensor.numel())
        frame_counts += torch.bincount(frame_ids[chosen_tensor], minlength=frame_count).float()
        for level_index, similarity in enumerate(similarity_matrices):
            batch_coverage = similarity[:, chosen_tensor].amax(dim=1)
            best_coverage[level_index] = torch.maximum(best_coverage[level_index], batch_coverage)

    selected = candidates[selected_mask]
    if selected.numel() < budget:
        remaining = _stable_descending(quality)
        remaining = remaining[~torch.isin(remaining, selected)]
        selected = torch.cat([selected, remaining[: budget - selected.numel()]])
    elif selected.numel() > budget:
        order = _stable_descending(quality[selected])[:budget]
        selected = selected[order]
    return torch.unique(selected.long(), sorted=True)


@torch.no_grad()
def prismvid_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
    question_features: Optional[torch.Tensor] = None,
    deepstack_features: Optional[Sequence[torch.Tensor]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select an exact Qwen3-aware multi-level visual coreset."""
    if video_features.ndim != 3:
        raise ValueError(f"video_features must have shape [T, P, D], got {tuple(video_features.shape)}")
    frame_count, tokens_per_frame, feature_dim = video_features.shape
    if frame_count <= 0 or tokens_per_frame <= 0 or feature_dim <= 0:
        raise ValueError("video_features must be non-empty")
    if cls_attention.shape != (frame_count, tokens_per_frame):
        raise ValueError(
            "cls_attention must match video_features[:2], "
            f"got {tuple(cls_attention.shape)} versus {(frame_count, tokens_per_frame)}"
        )

    flat_features = video_features.reshape(-1, feature_dim)
    token_count = flat_features.shape[0]
    per_frame_budget = int(math.ceil(tokens_per_frame * _effective_ratio(flashvid_config)))
    budget = max(1, min(token_count, frame_count * per_frame_budget))
    if budget >= token_count:
        indices = torch.arange(token_count, dtype=torch.long, device=video_features.device)
        output = flat_features
        query_confidence = 0.0
        level_count = 1
    else:
        metric_dim = max(16, _cfg_int(flashvid_config, "prism_metric_dim", 256))
        levels = _prepare_levels(video_features, deepstack_features, metric_dim)
        level_count = len(levels)

        attention = _debiased_attention(cls_attention.to(video_features.device)).flatten()
        events = _multi_scale_events(levels, frame_count, tokens_per_frame)
        disagreement = _cross_level_disagreement(levels)
        detail = _detail_signal(levels[-1], frame_count, tokens_per_frame)
        atoms = _question_atoms(
            question_features,
            max_atoms=max(0, _cfg_int(flashvid_config, "prism_query_atoms", 6)),
            metric_dim=metric_dim,
            device=video_features.device,
        )
        query, query_confidence = _question_signal(atoms, levels[-1])
        weights = _router_weights(attention, events, disagreement, query_confidence, flashvid_config)
        quality = (
            weights[0] * attention
            + weights[1] * events
            + weights[2] * query
            + weights[3] * disagreement
            + weights[4] * detail
        ).clamp_(0.0, 1.0)

        mandatory = _frame_floor(
            quality,
            frame_count,
            tokens_per_frame,
            budget,
            _cfg_float(flashvid_config, "prism_frame_floor_ratio", 0.20),
        )
        candidates = _candidate_pool(
            budget,
            mandatory,
            quality,
            attention,
            events,
            disagreement,
            query,
            _cfg_float(flashvid_config, "prism_candidate_multiplier", 2.25),
        )
        probes, probe_weights = _probe_set(
            quality,
            events,
            disagreement,
            query,
            mandatory,
            max_probes=max(32, _cfg_int(flashvid_config, "prism_probe_tokens", 512)),
        )
        indices = _select_coreset(
            levels,
            candidates,
            probes,
            probe_weights,
            mandatory,
            quality,
            frame_count,
            tokens_per_frame,
            budget,
            flashvid_config,
        )
        output = flat_features[indices]

    if output.shape[0] != indices.shape[0]:
        raise RuntimeError("PrismVID output/index cardinality mismatch")
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("PrismVID produced non-finite output features")

    flashvid_config.last_adapter_variant = "prismvid"
    flashvid_config.last_adapter_raw_tokens = int(token_count)
    flashvid_config.last_adapter_output_tokens = int(output.shape[0])
    flashvid_config.last_prism_budget = int(budget)
    flashvid_config.last_prism_per_frame_budget = int(per_frame_budget)
    flashvid_config.last_prism_levels = int(level_count)
    flashvid_config.last_prism_query_confidence = float(query_confidence)
    return output, indices


def compress_prism_deepstack(
    deepstack_video_embeds: Sequence[torch.Tensor],
    kept_video_indices: torch.Tensor,
) -> list[torch.Tensor]:
    """Apply exact PrismVID indices to every Qwen3 DeepStack level."""
    compressed: list[torch.Tensor] = []
    for layer_index, layer_features in enumerate(deepstack_video_embeds):
        if layer_features.ndim != 2:
            raise ValueError(
                f"PrismVID DeepStack layer {layer_index} must be [N, D], "
                f"got {tuple(layer_features.shape)}"
            )
        indices = kept_video_indices.to(device=layer_features.device, dtype=torch.long)
        compressed.append(layer_features.index_select(0, indices))
    return compressed


def merge_prism_visual_deepstack(
    *,
    deepstack_image_embeds: Sequence[torch.Tensor],
    compressed_video_embeds: Sequence[torch.Tensor],
    image_mask: torch.Tensor,
    video_mask: torch.Tensor,
    kept_video_indices: torch.Tensor,
) -> list[torch.Tensor]:
    """Rebuild mixed image/video DeepStack tensors in retained prompt order."""
    if len(deepstack_image_embeds) != len(compressed_video_embeds):
        raise ValueError(
            "PrismVID image/video DeepStack depth mismatch: "
            f"{len(deepstack_image_embeds)} != {len(compressed_video_embeds)}"
        )
    if image_mask.ndim != 2 or video_mask.ndim != 2 or image_mask.shape[0] != 1:
        raise ValueError(
            "PrismVID mixed visual inputs require batch size 1 masks, "
            f"got image={tuple(image_mask.shape)}, video={tuple(video_mask.shape)}"
        )

    image_positions = torch.where(image_mask[0])[0]
    video_positions = torch.where(video_mask[0])[0]
    video_indices = kept_video_indices.to(device=video_positions.device, dtype=torch.long)
    kept_video_positions = video_positions.index_select(0, video_indices)
    joint_positions = torch.cat([image_positions, kept_video_positions], dim=0)
    order = torch.argsort(joint_positions, stable=True)

    merged: list[torch.Tensor] = []
    for layer_index, (image_features, video_features) in enumerate(
        zip(deepstack_image_embeds, compressed_video_embeds)
    ):
        if int(image_features.shape[0]) != int(image_positions.numel()):
            raise ValueError(
                f"PrismVID image DeepStack layer {layer_index} has {image_features.shape[0]} "
                f"features for {image_positions.numel()} placeholders"
            )
        if int(video_features.shape[0]) != int(kept_video_positions.numel()):
            raise ValueError(
                f"PrismVID video DeepStack layer {layer_index} has {video_features.shape[0]} "
                f"features for {kept_video_positions.numel()} retained placeholders"
            )
        joint = torch.cat(
            [image_features, video_features.to(image_features.device, image_features.dtype)],
            dim=0,
        )
        merged.append(joint.index_select(0, order.to(joint.device)))
    return merged


__all__ = [
    "compress_prism_deepstack",
    "merge_prism_visual_deepstack",
    "prismvid_compression",
]
