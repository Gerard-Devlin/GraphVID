from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from .configuration_flashvid import FlashVidConfig


def _cfg_float(config: FlashVidConfig, name: str, default: float) -> float:
    try:
        value = getattr(config, name, default)
        return float(default if value is None else value)
    except Exception:
        return float(default)


def _cfg_int(config: FlashVidConfig, name: str, default: int) -> int:
    try:
        value = getattr(config, name, default)
        return int(default if value is None else value)
    except Exception:
        return int(default)


def _effective_ratio(config: FlashVidConfig) -> float:
    ratio = _cfg_float(config, "retention_ratio", 0.10)
    if bool(getattr(config, "ridge_budget_uses_expansion", True)):
        ratio *= _cfg_float(config, "expansion", 1.0)
    return max(0.0, min(1.0, ratio))


def _normalize(values: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    values = values.float()
    valid = values.reshape(-1) if mask is None else values[mask]
    if valid.numel() == 0:
        return torch.zeros_like(values, dtype=torch.float32)
    lo = valid.min()
    hi = valid.max()
    return ((values - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)


def _grid_hw(num_tokens: int, config: FlashVidConfig) -> tuple[int, int]:
    h = int(getattr(config, "H", 0) or 0)
    w = int(getattr(config, "W", 0) or 0)
    if h > 0 and w > 0 and h * w == num_tokens:
        return h, w
    h = int(math.sqrt(num_tokens))
    while h > 1 and num_tokens % h != 0:
        h -= 1
    return max(1, h), max(1, num_tokens // max(1, h))


def _cell_ids(num_tokens: int, h: int, w: int, bins: int, device: torch.device) -> torch.Tensor:
    bins = max(1, int(bins))
    ids = torch.zeros((num_tokens,), dtype=torch.long, device=device)
    for idx in range(num_tokens):
        r, c = divmod(idx, w)
        rb = min(bins - 1, int(r * bins / max(1, h)))
        cb = min(bins - 1, int(c * bins / max(1, w)))
        ids[idx] = rb * bins + cb
    return ids


def _attention_map(cls_attention: torch.Tensor, frame_count: int, tokens_per_frame: int, device: torch.device) -> torch.Tensor:
    if cls_attention is None:
        return torch.zeros((frame_count, tokens_per_frame), dtype=torch.float32, device=device)
    attn = cls_attention.detach().float().to(device)
    if attn.ndim > 2:
        attn = attn.reshape(attn.shape[0], -1)
    if attn.ndim == 1:
        attn = attn.unsqueeze(0)
    if attn.shape[0] != frame_count:
        attn = attn.reshape(-1, tokens_per_frame) if attn.numel() >= frame_count * tokens_per_frame else None
    if attn is None:
        return torch.zeros((frame_count, tokens_per_frame), dtype=torch.float32, device=device)
    if attn.shape[-1] < tokens_per_frame:
        pad = torch.zeros((attn.shape[0], tokens_per_frame - attn.shape[-1]), dtype=attn.dtype, device=device)
        attn = torch.cat([attn, pad], dim=-1)
    attn = attn[:frame_count, :tokens_per_frame]
    return _normalize(torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0))


def _temporal_frame_signal(frame_reps: torch.Tensor) -> torch.Tensor:
    frame_count = int(frame_reps.shape[0])
    if frame_count <= 1:
        return torch.ones((frame_count,), dtype=torch.float32, device=frame_reps.device)

    prev_sim = torch.ones((frame_count,), dtype=torch.float32, device=frame_reps.device)
    next_sim = torch.ones((frame_count,), dtype=torch.float32, device=frame_reps.device)
    pair_sim = (frame_reps[:-1] * frame_reps[1:]).sum(dim=-1).clamp(-1.0, 1.0)
    prev_sim[1:] = pair_sim
    next_sim[:-1] = pair_sim
    motion = 1.0 - torch.minimum(prev_sim, next_sim)

    bend = torch.zeros_like(motion)
    if frame_count > 2:
        vin = frame_reps[1:-1] - frame_reps[:-2]
        vout = frame_reps[2:] - frame_reps[1:-1]
        bend[1:-1] = 1.0 - F.cosine_similarity(vin, vout, dim=-1, eps=1e-6).clamp(-1.0, 1.0)
    bend[0] = bend[-1] = motion.max().clamp_min(0.1)
    return _normalize(0.55 * motion + 0.45 * bend)


def _token_temporal_novelty(normed: torch.Tensor) -> torch.Tensor:
    frame_count, tokens_per_frame, _ = normed.shape
    novelty = torch.ones((frame_count, tokens_per_frame), dtype=torch.float32, device=normed.device)
    if frame_count <= 1:
        return novelty

    sims = (normed[:-1] * normed[1:]).sum(dim=-1).clamp(-1.0, 1.0)
    step = (1.0 - sims).clamp(0.0, 2.0) * 0.5
    novelty[:-1] = torch.minimum(novelty[:-1], step)
    novelty[1:] = torch.minimum(novelty[1:], step)
    return _normalize(novelty)


def _local_contrast(normed: torch.Tensor, h: int, w: int) -> torch.Tensor:
    frame_count, tokens_per_frame, dim = normed.shape
    if h * w != tokens_per_frame or h <= 1 or w <= 1:
        center = F.normalize(normed.mean(dim=1), dim=-1, eps=1e-6).unsqueeze(1)
        return _normalize(1.0 - (normed * center).sum(dim=-1).clamp(-1.0, 1.0))

    image = normed.reshape(frame_count, h, w, dim).permute(0, 3, 1, 2)
    pooled = F.avg_pool2d(image, kernel_size=3, stride=1, padding=1)
    pooled = F.normalize(pooled.permute(0, 2, 3, 1).reshape(frame_count, tokens_per_frame, dim), dim=-1, eps=1e-6)
    return _normalize(1.0 - (normed * pooled).sum(dim=-1).clamp(-1.0, 1.0))


def _question_relevance(video_features: torch.Tensor, question_features: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    frame_count, tokens_per_frame, _ = video_features.shape
    device = video_features.device
    zeros = torch.zeros((frame_count, tokens_per_frame), dtype=torch.float32, device=device)
    frame_zeros = torch.zeros((frame_count,), dtype=torch.float32, device=device)
    if question_features is None or question_features.numel() == 0:
        return zeros, frame_zeros
    try:
        q = question_features.float().to(device)
        if q.ndim == 1:
            q = q.unsqueeze(0)
        q_center = F.normalize(q.mean(dim=0), dim=-1, eps=1e-6)
        token_rel = (F.normalize(video_features.float(), dim=-1, eps=1e-6) * q_center).sum(dim=-1)
        frame_rel = token_rel.mean(dim=1)
        return _normalize(token_rel), _normalize(frame_rel)
    except Exception:
        return zeros, frame_zeros


def _allocate_frame_budget(
    frame_score: torch.Tensor,
    total_budget: int,
    tokens_per_frame: int,
    *,
    frame_floor_ratio: float,
    min_per_frame: int,
    temporal_bins: int,
    strata_strength: float,
    temperature: float,
) -> torch.Tensor:
    frame_count = int(frame_score.shape[0])
    total_budget = int(max(1, min(total_budget, frame_count * tokens_per_frame)))
    avg_budget = total_budget / max(1, frame_count)
    floor = int(math.floor(avg_budget * max(0.0, min(1.0, frame_floor_ratio))))
    floor = max(int(min_per_frame), floor)
    floor = min(tokens_per_frame, floor)
    if floor * frame_count > total_budget:
        floor = max(0, total_budget // max(1, frame_count))

    alloc = torch.full((frame_count,), floor, dtype=torch.long, device=frame_score.device)
    remaining = int(total_budget - int(alloc.sum().item()))
    if remaining <= 0:
        return alloc

    saliency = torch.softmax(frame_score.float() / max(1e-6, float(temperature)), dim=0)
    bins = max(1, min(int(temporal_bins), frame_count))
    strata = torch.zeros_like(saliency)
    for bid in range(bins):
        start = int(round(bid * frame_count / bins))
        end = int(round((bid + 1) * frame_count / bins))
        end = max(start + 1, min(frame_count, end))
        strata[start:end] = 1.0 / float(bins * (end - start))
    mix = max(0.0, min(1.0, float(strata_strength)))
    weights = (1.0 - mix) * saliency + mix * strata
    weights = weights / weights.sum().clamp_min(1e-6)

    raw = weights * float(remaining)
    extra = torch.floor(raw).long()
    extra = torch.minimum(extra, torch.full_like(extra, max(0, tokens_per_frame - floor)))
    alloc += extra

    remaining = int(total_budget - int(alloc.sum().item()))
    if remaining > 0:
        frac = raw - torch.floor(raw)
        frac = frac.masked_fill(alloc >= tokens_per_frame, -1.0)
        for _ in range(remaining):
            idx = torch.argmax(frac)
            if float(frac[idx].item()) < 0.0:
                break
            alloc[idx] += 1
            if alloc[idx] >= tokens_per_frame:
                frac[idx] = -1.0
    elif remaining < 0:
        over = -remaining
        order = torch.argsort(weights, descending=False)
        for idx in order:
            if over <= 0:
                break
            while over > 0 and int(alloc[idx].item()) > floor:
                alloc[idx] -= 1
                over -= 1
    return alloc


def _select_frame_tokens(
    frame_normed: torch.Tensor,
    score: torch.Tensor,
    k: int,
    cell_id: torch.Tensor,
    *,
    coverage_ratio: float,
    mmr_lambda: float,
) -> torch.Tensor:
    tokens_per_frame = int(score.shape[0])
    k = max(0, min(int(k), tokens_per_frame))
    if k <= 0:
        return torch.zeros((0,), dtype=torch.long, device=score.device)
    if k >= tokens_per_frame:
        return torch.arange(tokens_per_frame, dtype=torch.long, device=score.device)

    score = _normalize(score)
    selected: list[int] = []
    used = torch.zeros((tokens_per_frame,), dtype=torch.bool, device=score.device)
    covered_cells: set[int] = set()

    reserve = min(k, int(math.ceil(k * max(0.0, min(1.0, coverage_ratio)))))
    if reserve > 0:
        cell_values: list[tuple[float, int, int]] = []
        for cell in torch.unique(cell_id).tolist():
            idx = torch.where(cell_id == int(cell))[0]
            if idx.numel() == 0:
                continue
            best_local = idx[torch.argmax(score[idx])]
            cell_values.append((float(score[best_local].item()), int(cell), int(best_local.item())))
        for _, cell, idx in sorted(cell_values, reverse=True)[:reserve]:
            if not bool(used[idx].item()):
                selected.append(idx)
                used[idx] = True
                covered_cells.add(cell)

    lam = max(0.0, min(1.0, float(mmr_lambda)))
    while len(selected) < k:
        candidates = torch.where(~used)[0]
        if candidates.numel() == 0:
            break
        if selected:
            chosen = torch.tensor(selected, dtype=torch.long, device=score.device)
            sim = torch.matmul(frame_normed[candidates], frame_normed[chosen].T).max(dim=1).values.clamp(-1.0, 1.0)
            penalty = (sim + 1.0) * 0.5
        else:
            penalty = torch.zeros((candidates.numel(),), dtype=torch.float32, device=score.device)
        cell_bonus = torch.tensor(
            [0.06 if int(cell_id[int(idx)].item()) not in covered_cells else 0.0 for idx in candidates.tolist()],
            dtype=torch.float32,
            device=score.device,
        )
        mmr = lam * score[candidates] - (1.0 - lam) * penalty + cell_bonus
        best = int(candidates[torch.argmax(mmr)].item())
        selected.append(best)
        used[best] = True
        covered_cells.add(int(cell_id[best].item()))

    out = torch.tensor(selected, dtype=torch.long, device=score.device)
    return torch.sort(out).values


def ridgevid_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
    question_features: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Temporal-ridge ledger compression with coverage-aware raw token selection."""
    if video_features.ndim != 3:
        raise ValueError(f"expected video_features [T, HW, D], got shape={tuple(video_features.shape)}")

    frame_count, tokens_per_frame, dim = video_features.shape
    device = video_features.device
    raw_tokens = int(frame_count * tokens_per_frame)
    if raw_tokens <= 0:
        empty_idx = torch.zeros((0,), dtype=torch.long, device=device)
        return video_features.new_empty((0, dim)), empty_idx

    ratio = _effective_ratio(flashvid_config)
    total_budget = max(1, min(raw_tokens, int(round(raw_tokens * ratio))))
    h, w = _grid_hw(tokens_per_frame, flashvid_config)
    spatial_bins = max(1, _cfg_int(flashvid_config, "ridge_spatial_bins", 3))
    cell_id = _cell_ids(tokens_per_frame, h, w, spatial_bins, device)

    normed = F.normalize(video_features.float(), dim=-1, eps=1e-6)
    frame_reps = F.normalize(normed.mean(dim=1), dim=-1, eps=1e-6)
    attn = _attention_map(cls_attention, frame_count, tokens_per_frame, device)
    frame_attn = _normalize(attn.mean(dim=1))
    frame_motion = _temporal_frame_signal(frame_reps)
    frame_dispersion = _normalize((1.0 - (normed * frame_reps.unsqueeze(1)).sum(dim=-1).clamp(-1.0, 1.0)).mean(dim=1))
    q_token, q_frame = _question_relevance(video_features, question_features)

    frame_score = (
        0.38 * frame_motion
        + 0.24 * frame_attn
        + 0.24 * frame_dispersion
        + 0.14 * q_frame
    )

    temporal_novelty = _token_temporal_novelty(normed)
    contrast = _local_contrast(normed, h, w)
    energy = _normalize(video_features.float().norm(dim=-1))
    token_score = (
        _cfg_float(flashvid_config, "ridge_attention_weight", 0.34) * attn
        + _cfg_float(flashvid_config, "ridge_motion_weight", 0.24) * temporal_novelty
        + _cfg_float(flashvid_config, "ridge_contrast_weight", 0.24) * contrast
        + _cfg_float(flashvid_config, "ridge_question_weight", 0.12) * q_token
        + 0.06 * energy
    )

    per_frame_budget = _allocate_frame_budget(
        frame_score,
        total_budget,
        tokens_per_frame,
        frame_floor_ratio=_cfg_float(flashvid_config, "ridge_frame_floor_ratio", 0.42),
        min_per_frame=_cfg_int(flashvid_config, "ridge_min_per_frame", 1),
        temporal_bins=_cfg_int(flashvid_config, "ridge_temporal_bins", 4),
        strata_strength=_cfg_float(flashvid_config, "ridge_strata_strength", 0.35),
        temperature=_cfg_float(flashvid_config, "ridge_budget_temperature", 0.75),
    )

    selected_parts: list[torch.Tensor] = []
    token_parts: list[torch.Tensor] = []
    for frame_idx in range(frame_count):
        k = int(per_frame_budget[frame_idx].item())
        local_idx = _select_frame_tokens(
            normed[frame_idx],
            token_score[frame_idx],
            k,
            cell_id,
            coverage_ratio=_cfg_float(flashvid_config, "ridge_coverage_ratio", 0.30),
            mmr_lambda=_cfg_float(flashvid_config, "ridge_mmr_lambda", 0.78),
        )
        if local_idx.numel() == 0:
            continue
        selected_parts.append(local_idx + frame_idx * tokens_per_frame)
        token_parts.append(video_features[frame_idx].index_select(0, local_idx))

    if not selected_parts:
        selected = torch.zeros((1,), dtype=torch.long, device=device)
        hidden_states = video_features.reshape(-1, dim)[:1]
    else:
        selected = torch.cat(selected_parts, dim=0).long()
        hidden_states = torch.cat(token_parts, dim=0)
        order = torch.argsort(selected)
        selected = selected[order]
        hidden_states = hidden_states[order]

    hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=0.0, neginf=0.0).to(dtype=video_features.dtype)
    out_tokens = int(hidden_states.shape[0])
    flashvid_config.vision_token_length = out_tokens
    flashvid_config.llm_token_length = None
    flashvid_config.visual_token_length = out_tokens
    setattr(flashvid_config, "last_adapter_variant", "ridgevid")
    setattr(flashvid_config, "last_adapter_output_tokens", float(out_tokens))
    setattr(flashvid_config, "last_adapter_raw_tokens", float(raw_tokens))
    setattr(flashvid_config, "last_ridge_target_tokens", float(total_budget))
    setattr(flashvid_config, "last_ridge_output_tokens", float(out_tokens))
    setattr(flashvid_config, "last_ridge_frame_floor_ratio", _cfg_float(flashvid_config, "ridge_frame_floor_ratio", 0.42))
    return hidden_states, selected
