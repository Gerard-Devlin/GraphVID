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
    radius = max(0, int(getattr(config, "echo_neighbor_radius", 1)))
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


def _frame_importance(cls_attention: torch.Tensor, residual_scores: torch.Tensor) -> torch.Tensor:
    num_frames, num_visual_tokens = cls_attention.shape
    attention = _normalize_scores(cls_attention.float().mean(dim=1))
    novelty = _normalize_scores(residual_scores.view(num_frames, num_visual_tokens).mean(dim=1))
    return _normalize_scores(0.65 * attention + 0.35 * novelty)


def _allocate_frame_budget(
    total_budget: int,
    frame_importance: torch.Tensor,
    mode: str,
    min_keep_per_frame: int,
) -> List[int]:
    num_frames = int(frame_importance.shape[0])
    if num_frames <= 0:
        return []
    total_budget = max(0, int(total_budget))
    min_keep = max(0, int(min_keep_per_frame))
    if total_budget <= 0:
        return [0 for _ in range(num_frames)]
    if min_keep > 0:
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


def _hybrid_select(
    scores: torch.Tensor,
    total_budget: int,
    num_frames: int,
    num_visual_tokens: int,
    frame_importance: torch.Tensor,
    config: FlashVidConfig,
) -> torch.Tensor:
    total_budget = min(max(1, int(total_budget)), int(scores.numel()))
    global_ratio = min(max(float(getattr(config, "echo_global_topk_ratio", 0.70)), 0.0), 1.0)
    min_per_frame = max(0, int(getattr(config, "echo_min_keep_per_frame", 1)))
    mode = str(getattr(config, "echo_frame_budget_mode", "attention") or "attention")

    selected = []
    occupied = torch.zeros((scores.numel(),), dtype=torch.bool, device=scores.device)

    if min_per_frame > 0 and total_budget >= num_frames:
        score_grid = scores.view(num_frames, num_visual_tokens)
        for t in range(num_frames):
            keep_t = min(min_per_frame, int((~occupied.view(num_frames, num_visual_tokens)[t]).sum().item()))
            if keep_t <= 0 or int(occupied.sum().item()) >= total_budget:
                continue
            local_top = torch.topk(score_grid[t], k=keep_t, dim=0).indices
            global_idx = t * num_visual_tokens + local_top
            selected.append(global_idx)
            occupied[global_idx] = True

    used = sum(int(x.numel()) for x in selected)
    global_k = min(max(0, total_budget - used), max(0, int(round(total_budget * global_ratio))))
    if global_k > 0:
        top_global = torch.topk(scores.masked_fill(occupied, -1e9), k=global_k, dim=0).indices
        selected.append(top_global)
        occupied[top_global] = True

    remaining = total_budget - sum(int(x.numel()) for x in selected)
    if remaining > 0:
        frame_budgets = _allocate_frame_budget(remaining, frame_importance, mode, 0)
        score_grid = scores.view(num_frames, num_visual_tokens)
        occupied_grid = occupied.view(num_frames, num_visual_tokens)
        frame_picks = []
        for t in range(num_frames):
            keep_t = min(int(frame_budgets[t]), int((~occupied_grid[t]).sum().item()))
            if keep_t <= 0:
                continue
            local_scores = score_grid[t].masked_fill(occupied_grid[t], -1e9)
            local_top = torch.topk(local_scores, k=keep_t, dim=0).indices
            global_idx = t * num_visual_tokens + local_top
            frame_picks.append(global_idx)
            occupied[global_idx] = True
        if frame_picks:
            selected.append(torch.cat(frame_picks, dim=0))

    chosen = torch.cat(selected, dim=0).unique() if selected else torch.empty((0,), dtype=torch.long, device=scores.device)
    if int(chosen.numel()) < total_budget:
        occupied = torch.zeros((scores.numel(),), dtype=torch.bool, device=scores.device)
        if chosen.numel() > 0:
            occupied[chosen] = True
        fill_scores = scores.masked_fill(occupied, -1e9)
        fill_k = min(total_budget - int(chosen.numel()), int((fill_scores > -1e8).sum().item()))
        if fill_k > 0:
            chosen = torch.cat([chosen, torch.topk(fill_scores, k=fill_k, dim=0).indices], dim=0).unique()
    if int(chosen.numel()) > total_budget:
        chosen = chosen[torch.topk(scores[chosen], k=total_budget, dim=0).indices]
    return torch.sort(chosen.to(dtype=torch.long)).values


def _echo_residual_and_continuation(video_features: torch.Tensor, config: FlashVidConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    num_frames, num_visual_tokens, _ = video_features.shape
    device = video_features.device
    residual = torch.zeros((num_frames, num_visual_tokens), dtype=torch.float32, device=device)
    matched_prev = torch.zeros((num_frames, num_visual_tokens), dtype=torch.long, device=device)
    if num_frames <= 1:
        return residual.reshape(-1), residual.reshape(-1)

    neighbor_idx, neighbor_valid = _build_local_neighbors(num_visual_tokens, config, device)
    topk = max(1, int(getattr(config, "echo_topk_neighbors", 4)))
    temperature = max(1e-4, float(getattr(config, "echo_temperature", 0.07)))

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
        matched_prev[t] = src_idx[:, 0]
        weights = torch.softmax(vals / temperature, dim=1).to(cur.dtype)
        recon = torch.sum(prev[src_idx] * weights.unsqueeze(-1), dim=1)
        residual[t] = torch.mean((cur.float() - recon.float()) ** 2, dim=-1)

    if num_frames > 1:
        residual[0] = residual[1]

    continuation = torch.zeros_like(residual)
    for t in range(1, num_frames):
        continuation[t] = residual[t - 1, matched_prev[t]]
    continuation[0] = residual[0]
    return residual.reshape(-1), continuation.reshape(-1)


def _echo_residual_scores(video_features: torch.Tensor, config: FlashVidConfig) -> torch.Tensor:
    residual, _ = _echo_residual_and_continuation(video_features, config)
    return residual


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


def _quota_select(
    relevance_scores: torch.Tensor,
    novelty_scores: torch.Tensor,
    combined_scores: torch.Tensor,
    total_budget: int,
    num_frames: int,
    num_visual_tokens: int,
    frame_importance: torch.Tensor,
    config: FlashVidConfig,
    anchor_ratio: float,
) -> torch.Tensor:
    total_budget = min(max(1, int(total_budget)), int(combined_scores.numel()))
    budgets = _allocate_frame_budget(
        total_budget=total_budget,
        frame_importance=frame_importance,
        mode=str(getattr(config, "echo_frame_budget_mode", "attention") or "attention"),
        min_keep_per_frame=max(0, int(getattr(config, "echo_min_keep_per_frame", 1))),
    )
    anchor_ratio = min(max(float(anchor_ratio), 0.0), 0.90)

    rel_grid = relevance_scores.view(num_frames, num_visual_tokens)
    nov_grid = novelty_scores.view(num_frames, num_visual_tokens)
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
            novelty_local = nov_grid[t].masked_fill(selected, -1e9)
            valid = int((novelty_local > -1e8).sum().item())
            if valid > 0:
                novelty_idx = torch.topk(novelty_local, k=min(remaining, valid), dim=0).indices
                selected[novelty_idx] = True

        remaining = budget_t - int(selected.sum().item())
        if remaining > 0:
            combined_local = cmb_grid[t].masked_fill(selected, -1e9)
            valid = int((combined_local > -1e8).sum().item())
            if valid > 0:
                fill_idx = torch.topk(combined_local, k=min(remaining, valid), dim=0).indices
                selected[fill_idx] = True

        local = torch.where(selected)[0]
        if local.numel() > budget_t:
            local = local[torch.topk(cmb_grid[t, local], k=budget_t, dim=0).indices]
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
    return torch.sort(chosen.to(dtype=torch.long)).values


def _resolve_target_budget(
    num_frames: int,
    num_visual_tokens: int,
    config: FlashVidConfig,
) -> int:
    per_frame = int(getattr(config, "echo_target_tokens_per_frame", 0) or 0)
    if per_frame <= 0:
        per_frame = int(getattr(config, "talon_target_tokens_per_frame", 0) or 0)
    if per_frame > 0:
        per_frame = max(1, min(per_frame, num_visual_tokens))
        return max(1, min(num_frames * per_frame, num_frames * num_visual_tokens))

    ratio = min(max(float(getattr(config, "retention_ratio", 0.10)), 0.01), 1.0)
    expansion = max(0.01, float(getattr(config, "expansion", 1.0)))
    return max(1, min(num_frames * num_visual_tokens, int(math.ceil(num_frames * num_visual_tokens * ratio * expansion))))


def _finish_raw_selection(
    flat_features: torch.Tensor,
    flat_indices: torch.Tensor,
    chosen: torch.Tensor,
    target_budget: int,
    residual_scores: torch.Tensor,
    relevance_scores: torch.Tensor,
    question_active: bool,
    config: FlashVidConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    chosen_mask = torch.zeros((flat_indices.numel(),), dtype=torch.bool, device=flat_indices.device)
    if chosen.numel() > 0:
        chosen_mask[chosen] = True
    semantic_tokens = int((chosen_mask & (relevance_scores >= _normalize_scores(residual_scores))).sum().item())
    novelty_tokens = int(chosen_mask.sum().item()) - semantic_tokens

    config.num_attn_div_tokens = None
    config.num_sttm_tokens = None
    config.vision_token_length = int(chosen.numel())
    config.llm_token_length = None
    config.visual_token_length = int(chosen.numel())
    config.last_echo_target_budget = int(target_budget)
    config.last_echo_residual_mean = float(residual_scores.mean().item()) if residual_scores.numel() > 0 else 0.0
    config.last_echo_semantic_tokens = semantic_tokens
    config.last_echo_novelty_tokens = novelty_tokens
    config.last_echo_duplicate_index_count = int(chosen.numel()) - int(chosen.unique().numel())
    config.last_echo_question_aware_active = bool(question_active)
    return flat_features[chosen], flat_indices[chosen]


def echoprune_compression(
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

    echo_residual, echo_continuation = _echo_residual_and_continuation(video_features, flashvid_config)
    residual_norm = _normalize_scores(echo_residual)
    continuation_norm = _normalize_scores(echo_continuation)
    relevance = _normalize_scores(0.70 * attention_norm + 0.30 * question_norm) if question_active else attention_norm
    novelty = _normalize_scores(0.78 * residual_norm + 0.22 * continuation_norm)

    rel_w = max(0.0, float(getattr(flashvid_config, "echoprune_relevance_weight", 0.45)))
    echo_w = max(0.0, float(getattr(flashvid_config, "echoprune_echo_weight", 0.45)))
    cont_w = max(0.0, float(getattr(flashvid_config, "echoprune_continuation_weight", 0.10)))
    denom = max(1e-6, rel_w + echo_w + cont_w)
    combined = _normalize_scores((rel_w / denom) * relevance + (echo_w / denom) * residual_norm + (cont_w / denom) * continuation_norm)

    target_budget = _resolve_target_budget(num_frames, num_visual_tokens, flashvid_config)
    frame_importance = _frame_importance(cls_attention, novelty)
    chosen = _quota_select(
        relevance_scores=relevance,
        novelty_scores=novelty,
        combined_scores=combined,
        total_budget=target_budget,
        num_frames=num_frames,
        num_visual_tokens=num_visual_tokens,
        frame_importance=frame_importance,
        config=flashvid_config,
        anchor_ratio=float(getattr(flashvid_config, "echoprune_anchor_ratio", 0.35)),
    )
    return _finish_raw_selection(
        flat_features=flat_features,
        flat_indices=flat_indices,
        chosen=chosen,
        target_budget=target_budget,
        residual_scores=echo_residual,
        relevance_scores=relevance,
        question_active=question_active,
        config=flashvid_config,
    )


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

    echo_residual, echo_continuation = _echo_residual_and_continuation(video_features, flashvid_config)
    rank = int(getattr(flashvid_config, "talon_core_rank", 4) or 0)
    lowrank_residual = _lowrank_residual_scores(video_features, rank=rank)
    echo_norm = _normalize_scores(echo_residual)
    lowrank_norm = _normalize_scores(lowrank_residual)
    continuation_norm = _normalize_scores(echo_continuation)
    relevance = _normalize_scores(0.72 * attention_norm + 0.28 * question_norm) if question_active else attention_norm
    novelty = _normalize_scores(0.52 * echo_norm + 0.33 * lowrank_norm + 0.15 * continuation_norm)

    rel_w = max(0.0, float(getattr(flashvid_config, "talon_core_relevance_weight", 0.40)))
    echo_w = max(0.0, float(getattr(flashvid_config, "talon_core_echo_weight", 0.35)))
    lr_w = max(0.0, float(getattr(flashvid_config, "talon_core_lowrank_weight", 0.25)))
    denom = max(1e-6, rel_w + echo_w + lr_w)
    combined = _normalize_scores((rel_w / denom) * relevance + (echo_w / denom) * echo_norm + (lr_w / denom) * lowrank_norm)

    target_budget = _resolve_target_budget(num_frames, num_visual_tokens, flashvid_config)
    frame_importance = _frame_importance(cls_attention, novelty)
    chosen = _quota_select(
        relevance_scores=relevance,
        novelty_scores=novelty,
        combined_scores=combined,
        total_budget=target_budget,
        num_frames=num_frames,
        num_visual_tokens=num_visual_tokens,
        frame_importance=frame_importance,
        config=flashvid_config,
        anchor_ratio=float(getattr(flashvid_config, "talon_core_anchor_ratio", 0.35)),
    )
    return _finish_raw_selection(
        flat_features=flat_features,
        flat_indices=flat_indices,
        chosen=chosen,
        target_budget=target_budget,
        residual_scores=0.55 * echo_residual + 0.45 * lowrank_residual,
        relevance_scores=relevance,
        question_active=question_active,
        config=flashvid_config,
    )
