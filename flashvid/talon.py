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
        return value.strip().lower() in ("1", "true", "yes", "y", "on", "enabled")
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
    radius = max(0, int(getattr(config, "talon_transport_radius", 1)))
    token_ids = torch.arange(num_visual_tokens, device=device)
    y = token_ids // max(1, grid_w)
    x = token_ids % max(1, grid_w)
    offsets = [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)] or [(0, 0)]

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


def _question_aware_scores(
    flat_features: torch.Tensor,
    flat_attention: torch.Tensor,
    question_features: Optional[torch.Tensor],
    config: FlashVidConfig,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    attention_scores = _normalize_scores(flat_attention)
    if not _safe_bool(getattr(config, "question_aware_reweighting", False)):
        return attention_scores, None
    if question_features is None or question_features.numel() == 0:
        return attention_scores, None

    token_features = F.normalize(flat_features.float(), p=2, dim=-1, eps=1e-6)
    question_proto = F.normalize(question_features.float().mean(dim=0), p=2, dim=-1, eps=1e-6)
    question_scores = _normalize_scores(torch.matmul(token_features, question_proto))
    beta = min(max(float(getattr(config, "question_reweight_beta", 0.35)), 0.0), 1.0)
    fused = _normalize_scores((1.0 - beta) * attention_scores + beta * question_scores)
    return fused, question_scores


def _estimate_video_complexity(video_features: torch.Tensor) -> float:
    num_frames = int(video_features.shape[0])
    normed = F.normalize(video_features.float(), p=2, dim=-1, eps=1e-6)
    frame_centers = F.normalize(normed.mean(dim=1), p=2, dim=-1, eps=1e-6)
    if num_frames > 1:
        temporal = 0.5 * (1.0 - torch.sum(frame_centers[:-1] * frame_centers[1:], dim=-1)).clamp(0.0, 2.0)
        temporal_score = temporal.mean()
    else:
        temporal_score = torch.tensor(0.0, device=video_features.device)
    spatial = 0.5 * (1.0 - torch.sum(normed * frame_centers.unsqueeze(1), dim=-1)).clamp(0.0, 2.0)
    return float((0.6 * temporal_score + 0.4 * spatial.mean()).item())


def _estimate_question_difficulty(question_features: Optional[torch.Tensor]) -> float:
    if question_features is None or question_features.numel() == 0:
        return 0.5
    q = F.normalize(question_features.float(), p=2, dim=-1, eps=1e-6)
    center = F.normalize(q.mean(dim=0), p=2, dim=-1, eps=1e-6)
    dispersion = 0.5 * (1.0 - torch.matmul(q, center)).clamp(0.0, 2.0)
    return float(0.5 * dispersion.mean().item() + 0.5 * min(1.0, q.shape[0] / 32.0))


def _resolve_target_per_frame(
    video_features: torch.Tensor,
    question_features: Optional[torch.Tensor],
    config: FlashVidConfig,
) -> int:
    num_visual_tokens = int(video_features.shape[1])
    target = int(getattr(config, "talon_target_tokens_per_frame", 0) or 0)
    if target <= 0:
        config.last_talon_target_tokens_per_frame = None
        config.last_talon_complexity_score = None
        return 0
    target = max(1, min(target, num_visual_tokens))

    if _safe_bool(getattr(config, "talon_force_fixed_target", False)) or not _safe_bool(
        getattr(config, "talon_adaptive_target_enabled", False)
    ):
        config.last_talon_target_tokens_per_frame = target
        config.last_talon_complexity_score = None
        return target

    low = int(getattr(config, "talon_adaptive_target_low", 0) or max(1, target - 2))
    mid = int(getattr(config, "talon_adaptive_target_mid", 0) or max(1, target - 1))
    high = int(getattr(config, "talon_adaptive_target_high", 0) or target)
    low, mid, high = sorted([max(1, min(low, num_visual_tokens)), max(1, min(mid, num_visual_tokens)), max(1, min(high, num_visual_tokens))])

    score = 0.7 * _estimate_video_complexity(video_features) + 0.3 * _estimate_question_difficulty(question_features)
    score = max(0.0, min(1.0, score))
    config.last_talon_complexity_score = score
    floor = float(getattr(config, "talon_complexity_floor", 0.20))
    ceil = float(getattr(config, "talon_complexity_ceil", 0.40))
    norm = score if ceil <= floor else (score - floor) / (ceil - floor)
    norm = max(0.0, min(1.0, norm)) ** max(1e-6, float(getattr(config, "talon_adaptive_gamma", 1.0)))
    if norm <= 0.5:
        chosen = int(round((1.0 - norm / 0.5) * low + (norm / 0.5) * mid))
    else:
        alpha = (norm - 0.5) / 0.5
        chosen = int(round((1.0 - alpha) * mid + alpha * high))

    mean_cap = float(getattr(config, "talon_target_mean_cap", 0.0) or 0.0)
    if mean_cap > 0:
        running_sum = float(getattr(config, "talon_running_target_sum", 0.0) or 0.0)
        running_count = int(getattr(config, "talon_running_target_count", 0) or 0)
        while chosen > low and (running_sum + float(chosen)) / float(running_count + 1) > mean_cap + 1e-8:
            chosen -= 1
        setattr(config, "talon_running_target_sum", running_sum + float(chosen))
        setattr(config, "talon_running_target_count", running_count + 1)

    config.last_talon_target_tokens_per_frame = chosen
    return chosen


def _resolve_total_budget(
    num_frames: int,
    num_visual_tokens: int,
    config: FlashVidConfig,
    target_per_frame: int,
) -> int:
    num_tokens = num_frames * num_visual_tokens
    if target_per_frame > 0:
        budget = num_frames * max(1, min(target_per_frame, num_visual_tokens))
    else:
        scale = min(max(float(getattr(config, "talon_budget_scale", 0.60)), 0.01), 2.0)
        ratio = min(max(float(getattr(config, "retention_ratio", 0.10)), 0.001), 1.0)
        expansion = max(0.01, float(getattr(config, "expansion", 1.0)))
        budget = int(math.ceil(num_tokens * min(1.0, ratio * expansion * scale)))
    min_total = max(1, int(getattr(config, "talon_min_total_tokens", 1) or 1))
    return max(min_total, min(num_tokens, int(budget)))


def _transport_align(segment_features: torch.Tensor, config: FlashVidConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    num_frames, num_visual_tokens, feat_dim = segment_features.shape
    device = segment_features.device
    identity = torch.arange(num_visual_tokens, dtype=torch.long, device=device)
    source_to_slot = identity.unsqueeze(0).repeat(num_frames, 1)
    if num_frames <= 1:
        return segment_features, source_to_slot

    neighbor_idx, neighbor_valid = _build_local_neighbors(num_visual_tokens, config, device)
    mode = str(getattr(config, "talon_transport_mode", "hard") or "hard").strip().lower()
    temperature = max(1e-4, float(getattr(config, "talon_transport_temperature", 0.07)))
    aligned = segment_features.clone()

    for t in range(1, num_frames):
        prev = aligned[t - 1]
        cur = segment_features[t]
        prev_norm = F.normalize(prev.float(), p=2, dim=-1, eps=1e-6)
        cur_norm = F.normalize(cur.float(), p=2, dim=-1, eps=1e-6)
        sims = torch.sum(cur_norm.unsqueeze(1) * prev_norm[neighbor_idx], dim=-1).masked_fill(~neighbor_valid, -1e9)
        best_pos = torch.argmax(sims, dim=1)
        best_slot = neighbor_idx[identity, best_pos]
        source_to_slot[t] = best_slot

        aligned_t = torch.zeros((num_visual_tokens, feat_dim), dtype=cur.dtype, device=device)
        counts = torch.zeros((num_visual_tokens, 1), dtype=cur.dtype, device=device)
        if mode in ("soft", "entropy", "entropic"):
            weights = torch.softmax(sims / temperature, dim=1).to(cur.dtype)
            flat_slot = neighbor_idx.reshape(-1)
            flat_valid = neighbor_valid.reshape(-1)
            flat_weights = weights.reshape(-1)[flat_valid]
            flat_slot = flat_slot[flat_valid]
            flat_src = cur.unsqueeze(1).expand(-1, neighbor_idx.shape[1], -1).reshape(-1, feat_dim)[flat_valid]
            aligned_t.scatter_add_(0, flat_slot.unsqueeze(-1).expand(-1, feat_dim), flat_src * flat_weights.unsqueeze(-1))
            counts.scatter_add_(0, flat_slot.unsqueeze(-1), flat_weights.unsqueeze(-1))
            aligned_t = aligned_t / counts.clamp_min(1e-6)
        else:
            aligned_t.scatter_add_(0, best_slot.unsqueeze(-1).expand(-1, feat_dim), cur)
            counts.scatter_add_(0, best_slot.unsqueeze(-1), torch.ones((num_visual_tokens, 1), dtype=cur.dtype, device=device))
            aligned_t = aligned_t / counts.clamp_min(1.0)

        empty = counts.squeeze(-1) <= 0
        if empty.any():
            aligned_t[empty] = prev[empty]
        aligned[t] = aligned_t
    return aligned, source_to_slot


def _lowrank_basis(aligned: torch.Tensor, rank: int, config: FlashVidConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    _, num_visual_tokens, _ = aligned.shape
    rank = max(0, min(int(rank), num_visual_tokens))
    if rank <= 0:
        return aligned.new_zeros((num_visual_tokens, 0)), aligned.new_zeros((0,), dtype=torch.float32)

    cov = torch.zeros((num_visual_tokens, num_visual_tokens), dtype=torch.float32, device=aligned.device)
    for frame in aligned:
        f = frame.float()
        cov = cov + torch.matmul(f, f.transpose(0, 1))
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    return eigvecs[:, :rank].to(dtype=aligned.dtype), eigvals.float()


def _reconstruct_source(aligned: torch.Tensor, source_to_slot: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    num_frames, _, _ = aligned.shape
    if basis.numel() == 0:
        return torch.zeros_like(aligned)
    recon = torch.zeros_like(aligned)
    for t in range(num_frames):
        coeff = torch.matmul(basis.transpose(0, 1), aligned[t])
        bg = torch.matmul(basis, coeff)
        recon[t] = bg[source_to_slot[t]]
    return recon


def _frame_importance(
    cls_attention: torch.Tensor,
    residual_scores: torch.Tensor,
    fused_scores: torch.Tensor,
    config: FlashVidConfig,
) -> torch.Tensor:
    num_frames, num_visual_tokens = cls_attention.shape
    attn = _normalize_scores(cls_attention.float().mean(dim=1))
    resid = _normalize_scores(residual_scores.view(num_frames, num_visual_tokens).mean(dim=1))
    fused = _normalize_scores(fused_scores.view(num_frames, num_visual_tokens).mean(dim=1))
    boundary = torch.zeros_like(attn)
    if num_frames > 0:
        boundary[0] = 1.0
        boundary[-1] = 1.0
    motion_w = max(0.0, float(getattr(config, "talon_motion_importance_weight", 0.35)))
    boundary_w = max(0.0, float(getattr(config, "talon_boundary_importance_weight", 0.10)))
    question_w = max(0.0, float(getattr(config, "talon_question_frame_weight", 0.20)))
    attn_w = 1.0
    total = max(1e-6, attn_w + motion_w + boundary_w + question_w)
    return _normalize_scores((attn_w * attn + motion_w * resid + boundary_w * boundary + question_w * fused) / total)


def _allocate_frame_budget(total_budget: int, frame_importance: torch.Tensor, config: FlashVidConfig) -> List[int]:
    num_frames = int(frame_importance.shape[0])
    if num_frames <= 0:
        return []
    min_anchor = max(0, int(getattr(config, "talon_min_anchor_per_frame", 2)))
    avg_budget = float(total_budget) / float(max(1, num_frames))
    # Video QA is very sensitive to temporal coverage. In the low-budget regime,
    # do not let attention-weighted allocation starve "boring" frames that may
    # contain the evidence for a later question.
    coverage_floor_ratio = min(max(float(getattr(config, "talon_frame_coverage_floor_ratio", 0.65)), 0.0), 1.0)
    coverage_floor = int(math.floor(avg_budget * coverage_floor_ratio))
    min_each = min(max(min_anchor, coverage_floor), max(0, total_budget // max(1, num_frames)))
    budgets = [min_each for _ in range(num_frames)]
    remaining = int(total_budget - min_each * num_frames)
    if remaining <= 0:
        return budgets

    mode = str(getattr(config, "talon_budget_mode", "attention") or "attention").strip().lower()
    if mode == "attention":
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
    for i in range(leftover):
        budgets[i] += 1
    return budgets


def _diverse_topk(
    frame_features: torch.Tensor,
    scores: torch.Tensor,
    k: int,
    config: FlashVidConfig,
) -> torch.Tensor:
    """Select high-scoring anchors while avoiding same-region feature collapse."""
    k = min(max(0, int(k)), int(scores.numel()))
    if k <= 0:
        return torch.empty((0,), dtype=torch.long, device=scores.device)

    diversity_weight = min(max(float(getattr(config, "talon_anchor_diversity_weight", 0.0)), 0.0), 0.80)
    candidate_multiplier = max(1.0, float(getattr(config, "talon_anchor_candidate_multiplier", 4.0)))
    candidate_count = min(int(scores.numel()), max(k, int(math.ceil(k * candidate_multiplier))))
    candidates = torch.topk(scores.float(), k=candidate_count, dim=0).indices
    if diversity_weight <= 1e-8 or k >= candidate_count:
        return candidates[:k]

    candidate_scores = _normalize_scores(scores[candidates])
    candidate_features = F.normalize(frame_features[candidates].float(), p=2, dim=-1, eps=1e-6)
    selected_positions: List[int] = []
    selected_mask = torch.zeros((candidate_count,), dtype=torch.bool, device=scores.device)

    for _ in range(k):
        if not selected_positions:
            pos = int(torch.argmax(candidate_scores).item())
        else:
            selected_feat = candidate_features[torch.tensor(selected_positions, dtype=torch.long, device=scores.device)]
            nearest_sim = torch.matmul(candidate_features, selected_feat.transpose(0, 1)).max(dim=1).values
            novelty = _normalize_scores((1.0 - nearest_sim).clamp(min=0.0, max=2.0))
            greedy_score = (1.0 - diversity_weight) * candidate_scores + diversity_weight * novelty
            greedy_score = greedy_score.masked_fill(selected_mask, -1e9)
            pos = int(torch.argmax(greedy_score).item())
        selected_positions.append(pos)
        selected_mask[pos] = True

    selected_pos = torch.tensor(selected_positions, dtype=torch.long, device=scores.device)
    return candidates[selected_pos]


def _select_tokens(
    frame_features: torch.Tensor,
    fused_scores: torch.Tensor,
    residual_scores: torch.Tensor,
    combined_scores: torch.Tensor,
    total_budget: int,
    num_frames: int,
    num_visual_tokens: int,
    frame_importance: torch.Tensor,
    config: FlashVidConfig,
) -> Tuple[torch.Tensor, List[int], torch.Tensor, torch.Tensor]:
    total_budget = min(max(1, int(total_budget)), int(combined_scores.numel()))
    budgets = _allocate_frame_budget(total_budget, frame_importance, config)
    anchor_ratio = min(max(float(getattr(config, "talon_anchor_safety_ratio", 0.28)), 0.0), 0.85)
    global_ratio = min(max(float(getattr(config, "talon_global_topk_ratio", 0.70)), 0.0), 1.0)

    fused_grid = fused_scores.view(num_frames, num_visual_tokens)
    residual_grid = residual_scores.view(num_frames, num_visual_tokens)
    combined_grid = combined_scores.view(num_frames, num_visual_tokens)
    selected_mask = torch.zeros((num_frames * num_visual_tokens,), dtype=torch.bool, device=combined_scores.device)
    anchor_mask = torch.zeros_like(selected_mask)
    event_mask = torch.zeros_like(selected_mask)
    chosen_parts: List[torch.Tensor] = []

    for t in range(num_frames):
        budget_t = min(max(0, int(budgets[t])), num_visual_tokens)
        if budget_t <= 0:
            continue
        local_selected = torch.zeros((num_visual_tokens,), dtype=torch.bool, device=combined_scores.device)
        # TALON's residual branch is useful for innovation recall, but it should
        # not dominate raw-token selection in frozen VLMs. Keep semantic anchors
        # as the backbone, especially when target/frame is only ~18-22.
        if budget_t <= 24:
            anchor_ratio_eff = max(anchor_ratio, 0.70)
        else:
            anchor_ratio_eff = anchor_ratio
        anchor_k = min(budget_t, max(1, int(round(budget_t * anchor_ratio_eff)))) if budget_t > 1 else budget_t
        if anchor_k > 0:
            idx = _diverse_topk(frame_features[t], fused_grid[t], anchor_k, config)
            local_selected[idx] = True
            anchor_mask[t * num_visual_tokens + idx] = True
        remain = budget_t - int(local_selected.sum().item())
        if remain > 0:
            event_ratio = min(max(float(getattr(config, "talon_event_budget_ratio", 0.30)), 0.0), 1.0)
            event_k = min(remain, max(0, int(round(budget_t * event_ratio))))
            resid = residual_grid[t].masked_fill(local_selected, -1e9)
            valid = int((resid > -1e8).sum().item())
            if event_k > 0 and valid > 0:
                idx = torch.topk(resid, k=min(event_k, valid), dim=0).indices
                local_selected[idx] = True
                event_mask[t * num_visual_tokens + idx] = True
        remain = budget_t - int(local_selected.sum().item())
        if remain > 0:
            cmb = combined_grid[t].masked_fill(local_selected, -1e9)
            valid = int((cmb > -1e8).sum().item())
            if valid > 0:
                idx = torch.topk(cmb, k=min(remain, valid), dim=0).indices
                local_selected[idx] = True
        local = torch.where(local_selected)[0]
        chosen_parts.append(t * num_visual_tokens + local)

    if chosen_parts:
        chosen = torch.cat(chosen_parts, dim=0).unique()
    else:
        chosen = torch.empty((0,), dtype=torch.long, device=combined_scores.device)
    if chosen.numel() > 0:
        selected_mask[chosen] = True

    target_global = max(0, int(round(total_budget * global_ratio)))
    current_global = int(chosen.numel())
    if target_global > current_global:
        scores = combined_scores.masked_fill(selected_mask, -1e9)
        fill_k = min(target_global - current_global, int((scores > -1e8).sum().item()))
        if fill_k > 0:
            fill = torch.topk(scores, k=fill_k, dim=0).indices
            chosen = torch.cat([chosen, fill], dim=0).unique()
            selected_mask[fill] = True

    if int(chosen.numel()) < total_budget:
        scores = combined_scores.masked_fill(selected_mask, -1e9)
        fill_k = min(total_budget - int(chosen.numel()), int((scores > -1e8).sum().item()))
        if fill_k > 0:
            chosen = torch.cat([chosen, torch.topk(scores, k=fill_k, dim=0).indices], dim=0).unique()

    if int(chosen.numel()) > total_budget:
        chosen = chosen[torch.topk(combined_scores[chosen], k=total_budget, dim=0).indices]
    chosen = torch.sort(chosen.to(dtype=torch.long)).values
    return chosen, budgets, anchor_mask, event_mask


def talon_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
    question_features: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Clean TALON: transport-aligned low-rank background + sparse raw-token innovation.

    This implementation intentionally emits only original visual tokens. Low-rank
    background is used to score innovation, not to create synthetic coefficient or
    memory tokens, which avoids distribution shift in frozen VLMs.
    """
    num_frames, num_visual_tokens, feat_dim = video_features.shape
    num_tokens = num_frames * num_visual_tokens
    device = video_features.device
    flat_features = video_features.reshape(num_tokens, feat_dim)
    flat_indices = torch.arange(num_tokens, dtype=torch.long, device=device)
    flat_attention = cls_attention.reshape(num_tokens).float()

    target_per_frame = _resolve_target_per_frame(video_features, question_features, flashvid_config)
    total_budget = _resolve_total_budget(num_frames, num_visual_tokens, flashvid_config, target_per_frame)

    fused_scores, question_scores = _question_aware_scores(flat_features, flat_attention, question_features, flashvid_config)
    aligned, source_to_slot = _transport_align(video_features, flashvid_config)

    max_rank = min(
        max(0, int(getattr(flashvid_config, "talon_rank_max", 32))),
        num_visual_tokens,
        max(0, int(round((total_budget / max(1, num_frames)) * float(getattr(flashvid_config, "talon_background_max_ratio", 0.45))))),
    )
    min_rank = max(0, int(getattr(flashvid_config, "talon_rank_min", 2)))
    if max_rank > 0 and max_rank < min_rank:
        max_rank = min(max_rank, num_visual_tokens)
    rank = max_rank
    basis, eigvals = _lowrank_basis(aligned, rank=rank, config=flashvid_config)
    reconstruction = _reconstruct_source(aligned, source_to_slot, basis)
    residual_scores = torch.mean((video_features.float() - reconstruction.float()) ** 2, dim=-1).reshape(num_tokens)
    residual_norm = _normalize_scores(residual_scores)

    innovation_attention_weight = min(max(float(getattr(flashvid_config, "talon_innovation_attention_weight", 0.45)), 0.0), 1.0)
    innovation_scores = _normalize_scores((1.0 - innovation_attention_weight) * residual_norm + innovation_attention_weight * fused_scores)
    if question_scores is not None and _safe_bool(getattr(flashvid_config, "talon_use_question_innovation", True)):
        q_weight = min(max(float(getattr(flashvid_config, "talon_innovation_qweight", 0.25)), 0.0), 1.0)
        innovation_scores = _normalize_scores((1.0 - q_weight) * innovation_scores + q_weight * question_scores)

    final_fused_weight = min(max(float(getattr(flashvid_config, "talon_final_fused_weight", 0.70)), 0.0), 1.0)
    final_residual_weight = min(max(float(getattr(flashvid_config, "talon_final_residual_weight", 0.20)), 0.0), 1.0)
    final_frame_weight = min(max(float(getattr(flashvid_config, "talon_final_frame_weight", 0.10)), 0.0), 1.0)
    frame_importance = _frame_importance(cls_attention, residual_scores, fused_scores, flashvid_config)
    frame_scores = frame_importance.repeat_interleave(num_visual_tokens)
    denom = max(1e-6, final_fused_weight + final_residual_weight + final_frame_weight)
    combined_scores = _normalize_scores(
        (final_fused_weight / denom) * fused_scores
        + (final_residual_weight / denom) * innovation_scores
        + (final_frame_weight / denom) * frame_scores
    )

    chosen, budgets, anchor_mask, event_mask = _select_tokens(
        frame_features=video_features,
        fused_scores=fused_scores,
        residual_scores=innovation_scores,
        combined_scores=combined_scores,
        total_budget=total_budget,
        num_frames=num_frames,
        num_visual_tokens=num_visual_tokens,
        frame_importance=frame_importance,
        config=flashvid_config,
    )
    if chosen.numel() == 0:
        chosen = torch.topk(combined_scores, k=1, dim=0).indices.sort().values

    chosen_mask = torch.zeros((num_tokens,), dtype=torch.bool, device=device)
    chosen_mask[chosen] = True
    anchor_tokens = int((chosen_mask & anchor_mask).sum().item())
    event_tokens = int((chosen_mask & event_mask).sum().item())
    # Residual top-k that entered via the final combined/global fill is still an event.
    if event_tokens < int(chosen.numel()) - anchor_tokens:
        event_threshold_k = max(0, int(chosen.numel()) - anchor_tokens)
        residual_pick = chosen[torch.topk(innovation_scores[chosen], k=event_threshold_k, dim=0).indices]
        event_tokens = int(residual_pick.numel())

    flashvid_config.num_attn_div_tokens = None
    flashvid_config.num_sttm_tokens = None
    flashvid_config.vision_token_length = int(chosen.numel())
    flashvid_config.llm_token_length = None
    flashvid_config.visual_token_length = int(chosen.numel())
    flashvid_config.last_talon_target_tokens_per_frame = target_per_frame if target_per_frame > 0 else None
    flashvid_config.last_talon_target_budget = int(total_budget)
    flashvid_config.last_talon_anchor_tokens = int(anchor_tokens)
    flashvid_config.last_talon_rank_tokens = 0
    flashvid_config.last_talon_event_tokens = int(max(0, min(int(chosen.numel()) - anchor_tokens, event_tokens)))
    flashvid_config.last_talon_recall_tokens = 0
    flashvid_config.last_talon_memory_tokens = 0
    flashvid_config.last_talon_segment_count = 1
    flashvid_config.last_talon_rank_cap = int(max_rank)
    flashvid_config.last_talon_chosen_rank = int(rank)
    flashvid_config.last_talon_duplicate_index_count = int(chosen.numel()) - int(chosen.unique().numel())
    flashvid_config.last_talon_question_aware_active = question_scores is not None

    return flat_features[chosen], flat_indices[chosen]
