from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from torch.nn import functional as F

from .configuration_flashvid import FlashVidConfig


def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    scores = scores.float()
    if scores.numel() == 0:
        return scores
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)


def _safe_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y")
    return bool(value)


def _resolve_grid_hw(num_visual_tokens: int, config: FlashVidConfig) -> Tuple[int, int]:
    h = int(getattr(config, "H", 0) or 0)
    w = int(getattr(config, "W", 0) or 0)
    if h > 0 and w > 0 and h * w >= num_visual_tokens:
        return h, w
    h = max(1, int(round(math.sqrt(max(1, num_visual_tokens)))))
    w = max(1, int(math.ceil(num_visual_tokens / h)))
    return h, w


def _build_local_neighbors(num_visual_tokens: int, config: FlashVidConfig, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    _, grid_w = _resolve_grid_hw(num_visual_tokens, config)
    radius = max(0, int(getattr(config, "talon_core_neighbor_radius", 1)))
    token_ids = torch.arange(num_visual_tokens, device=device)
    y = token_ids // max(1, grid_w)
    x = token_ids % max(1, grid_w)
    offsets = [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
    if not offsets:
        offsets = [(0, 0)]

    neighbor_idx = torch.empty((num_visual_tokens, len(offsets)), dtype=torch.long, device=device)
    valid = torch.zeros((num_visual_tokens, len(offsets)), dtype=torch.bool, device=device)
    for col, (dy, dx) in enumerate(offsets):
        ny = y + dy
        nx = x + dx
        cand = ny * max(1, grid_w) + nx
        ok = (ny >= 0) & (nx >= 0) & (nx < max(1, grid_w)) & (cand >= 0) & (cand < num_visual_tokens)
        neighbor_idx[:, col] = torch.where(ok, cand, token_ids)
        valid[:, col] = ok
    valid[:, 0] = True
    return neighbor_idx, valid


def _question_scores(
    flat_features: torch.Tensor,
    question_features: Optional[torch.Tensor],
    config: FlashVidConfig,
) -> Tuple[torch.Tensor, bool]:
    if not _safe_bool(getattr(config, "question_aware_reweighting", False)):
        return flat_features.new_zeros((flat_features.shape[0],), dtype=torch.float32), False
    if question_features is None or question_features.numel() == 0:
        return flat_features.new_zeros((flat_features.shape[0],), dtype=torch.float32), False
    token_features = F.normalize(flat_features.float(), p=2, dim=-1, eps=1e-6)
    question_proto = F.normalize(question_features.float().mean(dim=0), p=2, dim=-1, eps=1e-6)
    return _normalize_scores(torch.matmul(token_features, question_proto)), True


def _temporal_residual_scores(video_features: torch.Tensor, config: FlashVidConfig) -> torch.Tensor:
    num_frames, num_visual_tokens, _ = video_features.shape
    device = video_features.device
    residual = torch.zeros((num_frames, num_visual_tokens), dtype=torch.float32, device=device)
    if num_frames <= 1:
        return residual.reshape(-1)

    neighbor_idx, neighbor_valid = _build_local_neighbors(num_visual_tokens, config, device)
    topk = max(1, int(getattr(config, "talon_core_topk_neighbors", 4)))
    temperature = max(1e-4, float(getattr(config, "talon_core_temperature", 0.07)))

    for t in range(1, num_frames):
        prev = video_features[t - 1]
        cur = video_features[t]
        prev_norm = F.normalize(prev.float(), p=2, dim=-1, eps=1e-6)
        cur_norm = F.normalize(cur.float(), p=2, dim=-1, eps=1e-6)
        local_prev = prev_norm[neighbor_idx]
        sims = torch.sum(cur_norm.unsqueeze(1) * local_prev, dim=-1).masked_fill(~neighbor_valid, -1e9)
        k = min(topk, sims.shape[1])
        vals, pos = torch.topk(sims, k=k, dim=1)
        src_idx = neighbor_idx.gather(1, pos)
        weights = torch.softmax(vals / temperature, dim=1).to(cur.dtype)
        recon = torch.sum(prev[src_idx] * weights.unsqueeze(-1), dim=1)
        residual[t] = torch.mean((cur.float() - recon.float()) ** 2, dim=-1)

    residual[0] = residual[1]
    return residual.reshape(-1)


def _lowrank_residual_scores(video_features: torch.Tensor, rank: int) -> torch.Tensor:
    num_frames, num_visual_tokens, _ = video_features.shape
    rank = max(0, min(int(rank), num_visual_tokens))
    if rank <= 0:
        return video_features.new_zeros((num_frames * num_visual_tokens,), dtype=torch.float32)

    matrix = video_features.float().permute(1, 0, 2).reshape(num_visual_tokens, -1)
    cov = torch.matmul(matrix, matrix.transpose(0, 1))
    eigvals, eigvecs = torch.linalg.eigh(cov)
    top = torch.argsort(eigvals, descending=True)[:rank]
    basis = eigvecs[:, top].to(dtype=video_features.dtype)

    residuals = []
    for frame in video_features:
        coeff = torch.matmul(basis.transpose(0, 1), frame)
        recon = torch.matmul(basis, coeff)
        residuals.append(torch.mean((frame.float() - recon.float()) ** 2, dim=-1))
    return torch.stack(residuals, dim=0).reshape(-1)


def _frame_importance(cls_attention: torch.Tensor, innovation_scores: torch.Tensor) -> torch.Tensor:
    num_frames, num_visual_tokens = cls_attention.shape
    attention = _normalize_scores(cls_attention.float().mean(dim=1))
    innovation = _normalize_scores(innovation_scores.view(num_frames, num_visual_tokens).mean(dim=1))
    return _normalize_scores(0.65 * attention + 0.35 * innovation)


def _allocate_frame_budget(total_budget: int, frame_importance: torch.Tensor, mode: str, min_keep_per_frame: int) -> List[int]:
    num_frames = int(frame_importance.shape[0])
    if num_frames <= 0:
        return []
    total_budget = max(0, int(total_budget))
    min_keep = max(0, int(min_keep_per_frame))
    min_keep = min(min_keep, max(0, total_budget // max(1, num_frames)))
    budgets = [min_keep for _ in range(num_frames)]
    remaining = total_budget - min_keep * num_frames
    if remaining <= 0:
        return budgets

    if str(mode).strip().lower() == "attention":
        weights = frame_importance.float().clamp_min(0.0)
        if float(weights.sum().item()) <= 1e-8:
            weights = torch.ones_like(weights)
        weights = weights / weights.sum()
        raw = weights * float(remaining)
        base = torch.floor(raw).to(dtype=torch.long)
        for i in range(num_frames):
            budgets[i] += int(base[i].item())
        leftover = int(remaining - int(base.sum().item()))
        if leftover > 0:
            order = torch.argsort(raw - base.float(), descending=True)
            for idx in order[:leftover]:
                budgets[int(idx.item())] += 1
        return budgets

    base = remaining // num_frames
    leftover = remaining - base * num_frames
    budgets = [b + int(base) for b in budgets]
    for i in range(int(leftover)):
        budgets[i] += 1
    return budgets


def _select_raw_tokens(
    relevance_scores: torch.Tensor,
    innovation_scores: torch.Tensor,
    combined_scores: torch.Tensor,
    total_budget: int,
    num_frames: int,
    num_visual_tokens: int,
    frame_importance: torch.Tensor,
    config: FlashVidConfig,
) -> Tuple[torch.Tensor, List[int]]:
    total_budget = min(max(1, int(total_budget)), int(combined_scores.numel()))
    min_keep = max(0, int(getattr(config, "talon_core_min_keep_per_frame", 1)))
    budgets = _allocate_frame_budget(
        total_budget=total_budget,
        frame_importance=frame_importance,
        mode=str(getattr(config, "talon_core_frame_budget_mode", "attention") or "attention"),
        min_keep_per_frame=min_keep,
    )
    anchor_ratio = min(max(float(getattr(config, "talon_core_anchor_ratio", 0.35)), 0.0), 0.90)
    rel_grid = relevance_scores.view(num_frames, num_visual_tokens)
    inn_grid = innovation_scores.view(num_frames, num_visual_tokens)
    cmb_grid = combined_scores.view(num_frames, num_visual_tokens)
    chosen_parts: List[torch.Tensor] = []

    for t in range(num_frames):
        budget_t = min(max(0, int(budgets[t])), num_visual_tokens)
        if budget_t <= 0:
            continue
        selected = torch.zeros((num_visual_tokens,), dtype=torch.bool, device=combined_scores.device)
        anchor_k = min(budget_t, max(1, int(round(budget_t * anchor_ratio)))) if budget_t > 1 else budget_t
        if anchor_k > 0:
            anchor_idx = torch.topk(rel_grid[t], k=anchor_k, dim=0).indices
            selected[anchor_idx] = True
        remaining = budget_t - int(selected.sum().item())
        if remaining > 0:
            innovation_local = inn_grid[t].masked_fill(selected, -1e9)
            valid = int((innovation_local > -1e8).sum().item())
            if valid > 0:
                innovation_idx = torch.topk(innovation_local, k=min(remaining, valid), dim=0).indices
                selected[innovation_idx] = True
        remaining = budget_t - int(selected.sum().item())
        if remaining > 0:
            combined_local = cmb_grid[t].masked_fill(selected, -1e9)
            valid = int((combined_local > -1e8).sum().item())
            if valid > 0:
                fill_idx = torch.topk(combined_local, k=min(remaining, valid), dim=0).indices
                selected[fill_idx] = True
        local = torch.where(selected)[0]
        chosen_parts.append(t * num_visual_tokens + local)

    chosen = torch.cat(chosen_parts, dim=0).unique() if chosen_parts else torch.empty((0,), dtype=torch.long, device=combined_scores.device)
    if int(chosen.numel()) < total_budget:
        occupied = torch.zeros((combined_scores.numel(),), dtype=torch.bool, device=combined_scores.device)
        if chosen.numel() > 0:
            occupied[chosen] = True
        fill_scores = combined_scores.masked_fill(occupied, -1e9)
        fill_k = min(total_budget - int(chosen.numel()), int((fill_scores > -1e8).sum().item()))
        if fill_k > 0:
            chosen = torch.cat([chosen, torch.topk(fill_scores, k=fill_k, dim=0).indices], dim=0).unique()
    if int(chosen.numel()) > total_budget:
        chosen = chosen[torch.topk(combined_scores[chosen], k=total_budget, dim=0).indices]
    return torch.sort(chosen.to(dtype=torch.long)).values, budgets


def _resolve_target_budget(num_frames: int, num_visual_tokens: int, config: FlashVidConfig) -> int:
    per_frame = int(getattr(config, "talon_core_target_tokens_per_frame", 0) or 0)
    if per_frame <= 0:
        per_frame = int(getattr(config, "talon_target_tokens_per_frame", 0) or 0)
    if per_frame > 0:
        per_frame = max(1, min(per_frame, num_visual_tokens))
        return max(1, min(num_frames * per_frame, num_frames * num_visual_tokens))
    ratio = min(max(float(getattr(config, "retention_ratio", 0.10)), 0.01), 1.0)
    expansion = max(0.01, float(getattr(config, "expansion", 1.0)))
    return max(1, min(num_frames * num_visual_tokens, int(math.ceil(num_frames * num_visual_tokens * ratio * expansion))))


def talon_core_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
    question_features: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_frames, num_visual_tokens, _ = video_features.shape
    num_tokens = num_frames * num_visual_tokens
    flat_features = video_features.reshape(num_tokens, -1)
    flat_indices = torch.arange(num_tokens, dtype=torch.long, device=video_features.device)
    attention_norm = _normalize_scores(cls_attention.reshape(num_tokens).float())
    question_norm, question_active = _question_scores(flat_features, question_features, flashvid_config)

    temporal_residual = _temporal_residual_scores(video_features, flashvid_config)
    rank = int(getattr(flashvid_config, "talon_core_rank", 4) or 0)
    lowrank_residual = _lowrank_residual_scores(video_features, rank=rank)
    temporal_norm = _normalize_scores(temporal_residual)
    lowrank_norm = _normalize_scores(lowrank_residual)
    relevance = _normalize_scores(0.72 * attention_norm + 0.28 * question_norm) if question_active else attention_norm
    innovation = _normalize_scores(0.60 * temporal_norm + 0.40 * lowrank_norm)

    rel_w = max(0.0, float(getattr(flashvid_config, "talon_core_relevance_weight", 0.42)))
    temporal_w = max(0.0, float(getattr(flashvid_config, "talon_core_temporal_weight", 0.33)))
    lowrank_w = max(0.0, float(getattr(flashvid_config, "talon_core_lowrank_weight", 0.25)))
    denom = max(1e-6, rel_w + temporal_w + lowrank_w)
    combined = _normalize_scores((rel_w / denom) * relevance + (temporal_w / denom) * temporal_norm + (lowrank_w / denom) * lowrank_norm)

    target_budget = _resolve_target_budget(num_frames, num_visual_tokens, flashvid_config)
    frame_importance = _frame_importance(cls_attention, innovation)
    chosen, frame_budgets = _select_raw_tokens(
        relevance_scores=relevance,
        innovation_scores=innovation,
        combined_scores=combined,
        total_budget=target_budget,
        num_frames=num_frames,
        num_visual_tokens=num_visual_tokens,
        frame_importance=frame_importance,
        config=flashvid_config,
    )

    chosen_mask = torch.zeros((num_tokens,), dtype=torch.bool, device=video_features.device)
    if chosen.numel() > 0:
        chosen_mask[chosen] = True
    semantic_tokens = int((chosen_mask & (relevance >= innovation)).sum().item())
    innovation_tokens = int(chosen_mask.sum().item()) - semantic_tokens

    flashvid_config.num_attn_div_tokens = None
    flashvid_config.num_sttm_tokens = None
    flashvid_config.vision_token_length = int(chosen.numel())
    flashvid_config.llm_token_length = None
    flashvid_config.visual_token_length = int(chosen.numel())
    flashvid_config.last_talon_core_target_budget = int(target_budget)
    flashvid_config.last_talon_core_residual_mean = float((0.55 * temporal_residual + 0.45 * lowrank_residual).mean().item())
    flashvid_config.last_talon_core_semantic_tokens = semantic_tokens
    flashvid_config.last_talon_core_innovation_tokens = innovation_tokens
    flashvid_config.last_talon_core_duplicate_index_count = int(chosen.numel()) - int(chosen.unique().numel())
    flashvid_config.last_talon_core_question_aware_active = bool(question_active)
    flashvid_config.last_talon_core_budget_min = min(frame_budgets) if frame_budgets else None
    flashvid_config.last_talon_core_budget_max = max(frame_budgets) if frame_budgets else None
    grid_h, grid_w = _resolve_grid_hw(num_visual_tokens, flashvid_config)
    flashvid_config.last_talon_core_grid_h = int(grid_h)
    flashvid_config.last_talon_core_grid_w = int(grid_w)
    return flat_features[chosen], flat_indices[chosen]
