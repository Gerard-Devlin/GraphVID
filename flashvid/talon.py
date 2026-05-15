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
    question_tokens = F.normalize(question_features.float(), p=2, dim=-1, eps=1e-6)
    pooling = str(getattr(config, "talon_question_pooling", "mean") or "mean").strip().lower()
    if pooling in ("max", "topk", "token_max", "token_topk"):
        token_question_sim = torch.matmul(token_features, question_tokens.transpose(0, 1))
        if pooling in ("topk", "token_topk"):
            topk = max(1, int(getattr(config, "talon_question_pooling_topk", 4) or 4))
            topk = min(topk, int(token_question_sim.shape[-1]))
            question_scores = _normalize_scores(torch.topk(token_question_sim, k=topk, dim=-1).values.mean(dim=-1))
        else:
            question_scores = _normalize_scores(token_question_sim.max(dim=-1).values)
    else:
        question_proto = F.normalize(question_tokens.mean(dim=0), p=2, dim=-1, eps=1e-6)
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


def _echo_residual(segment_features: torch.Tensor, config: FlashVidConfig) -> torch.Tensor:
    """Temporal echo residual: high when a token is poorly explained by previous-frame neighbors."""
    num_frames, num_visual_tokens, feat_dim = segment_features.shape
    residual = torch.zeros((num_frames, num_visual_tokens), dtype=torch.float32, device=segment_features.device)
    if num_frames <= 1:
        return residual

    neighbor_idx, neighbor_valid = _build_local_neighbors(num_visual_tokens, config, segment_features.device)
    temperature = max(1e-4, float(getattr(config, "talon_echo_temperature", 0.07)))
    topk = max(1, int(getattr(config, "talon_echo_topk_neighbors", 4) or 4))
    topk = min(topk, int(neighbor_idx.shape[1]))

    for t in range(1, num_frames):
        prev = segment_features[t - 1]
        cur = segment_features[t]
        prev_norm = F.normalize(prev.float(), p=2, dim=-1, eps=1e-6)
        cur_norm = F.normalize(cur.float(), p=2, dim=-1, eps=1e-6)
        sims = torch.sum(cur_norm.unsqueeze(1) * prev_norm[neighbor_idx], dim=-1).masked_fill(~neighbor_valid, -1e9)
        top_vals, top_pos = torch.topk(sims, k=topk, dim=1)
        top_idx = torch.gather(neighbor_idx, dim=1, index=top_pos)
        weights = torch.softmax(top_vals / temperature, dim=1).to(dtype=cur.dtype)
        pred = torch.sum(prev[top_idx] * weights.unsqueeze(-1), dim=1)
        residual[t] = torch.mean((cur.float() - pred.float()) ** 2, dim=-1)
    residual[0] = residual[1] if num_frames > 1 else residual[0]
    return residual


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
    duration = str(getattr(config, "current_video_duration", "") or "").strip().lower()

    def pool_frame_score(scores: torch.Tensor) -> torch.Tensor:
        grid = scores.view(num_frames, num_visual_tokens).float()
        pooling = str(getattr(config, "talon_frame_importance_pooling", "mean") or "mean").strip().lower()
        if duration == "short" and not _safe_bool(getattr(config, "talon_frame_importance_apply_to_short", False)):
            pooling = "mean"
        if pooling in ("topk", "max", "evidence"):
            if pooling == "max":
                return grid.max(dim=1).values
            topk = max(1, int(getattr(config, "talon_frame_importance_topk", 6) or 6))
            topk = min(topk, num_visual_tokens)
            return torch.topk(grid, k=topk, dim=1).values.mean(dim=1)
        return grid.mean(dim=1)

    attn = _normalize_scores(pool_frame_score(cls_attention))
    resid = _normalize_scores(pool_frame_score(residual_scores))
    fused = _normalize_scores(pool_frame_score(fused_scores))
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
    coverage_floor_ratio = float(getattr(config, "talon_frame_coverage_floor_ratio", 0.65))
    duration = str(getattr(config, "current_video_duration", "") or "").strip().lower()
    if duration == "medium":
        medium_floor = float(getattr(config, "talon_medium_frame_coverage_floor_ratio", -1.0))
        if medium_floor >= 0.0:
            coverage_floor_ratio = medium_floor
    elif duration == "long":
        long_floor = float(getattr(config, "talon_long_frame_coverage_floor_ratio", -1.0))
        if long_floor >= 0.0:
            coverage_floor_ratio = long_floor
    coverage_floor_ratio = min(max(coverage_floor_ratio, 0.0), 1.0)
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


def _spatial_anchor_topk(
    scores: torch.Tensor,
    k: int,
    local_selected: torch.Tensor,
    num_visual_tokens: int,
    config: FlashVidConfig,
) -> torch.Tensor:
    k = min(max(0, int(k)), int(scores.numel()))
    if k <= 0:
        return torch.empty((0,), dtype=torch.long, device=scores.device)
    duration = str(getattr(config, "current_video_duration", "") or "").strip().lower()
    if duration == "short" and not _safe_bool(getattr(config, "talon_spatial_anchor_apply_to_short", False)):
        return torch.empty((0,), dtype=torch.long, device=scores.device)
    if not _safe_bool(getattr(config, "talon_spatial_anchor_coverage", False)):
        return torch.empty((0,), dtype=torch.long, device=scores.device)

    rows = max(1, int(getattr(config, "talon_spatial_anchor_rows", 3) or 3))
    cols = max(1, int(getattr(config, "talon_spatial_anchor_cols", 3) or 3))
    grid_h, grid_w = _resolve_grid_hw(num_visual_tokens, config)
    token_ids = torch.arange(num_visual_tokens, device=scores.device, dtype=torch.long)
    y = token_ids // max(1, grid_w)
    x = token_ids % max(1, grid_w)
    bin_y = torch.clamp((y * rows) // max(1, grid_h), 0, rows - 1)
    bin_x = torch.clamp((x * cols) // max(1, grid_w), 0, cols - 1)
    bin_ids = bin_y * cols + bin_x

    picks: List[torch.Tensor] = []
    bin_scores: List[Tuple[float, int, int]] = []
    masked_scores = scores.float().masked_fill(local_selected, -1e9)
    for bin_id in range(rows * cols):
        members = torch.where(bin_ids == bin_id)[0]
        if members.numel() == 0:
            continue
        vals = masked_scores[members]
        if int((vals > -1e8).sum().item()) <= 0:
            continue
        pos = int(torch.argmax(vals).item())
        token = int(members[pos].item())
        bin_scores.append((float(vals[pos].item()), bin_id, token))
    if not bin_scores:
        return torch.empty((0,), dtype=torch.long, device=scores.device)
    bin_scores.sort(key=lambda x: x[0], reverse=True)
    for _, _, token in bin_scores[:k]:
        picks.append(torch.tensor([token], dtype=torch.long, device=scores.device))
    return torch.cat(picks, dim=0) if picks else torch.empty((0,), dtype=torch.long, device=scores.device)


def _select_tokens(
    frame_features: torch.Tensor,
    fused_scores: torch.Tensor,
    question_scores: Optional[torch.Tensor],
    residual_scores: torch.Tensor,
    combined_scores: torch.Tensor,
    total_budget: int,
    num_frames: int,
    num_visual_tokens: int,
    frame_importance: torch.Tensor,
    config: FlashVidConfig,
) -> Tuple[torch.Tensor, List[int], torch.Tensor, torch.Tensor]:
    total_budget = min(max(1, int(total_budget)), int(combined_scores.numel()))
    local_budget_ratio = min(max(float(getattr(config, "talon_frame_local_budget_ratio", 1.0)), 0.10), 1.0)
    local_budget = min(total_budget, max(1, int(round(float(total_budget) * local_budget_ratio))))
    budgets = _allocate_frame_budget(local_budget, frame_importance, config)
    anchor_ratio = min(max(float(getattr(config, "talon_anchor_safety_ratio", 0.28)), 0.0), 0.85)
    global_ratio = min(max(float(getattr(config, "talon_global_topk_ratio", 0.70)), 0.0), 1.0)
    event_ratio_cfg = min(max(float(getattr(config, "talon_event_budget_ratio", 0.30)), 0.0), 1.0)
    duration = str(getattr(config, "current_video_duration", "") or "").strip().lower()
    task_category = str(getattr(config, "current_task_category", "") or "").strip().lower()
    category = str(getattr(config, "current_category", "") or "").strip().lower()
    recall_ratio_override: Optional[float] = None
    router_ratios = _adaptive_router_ratios(
        fused_scores=fused_scores,
        question_scores=question_scores,
        residual_scores=residual_scores,
        frame_importance=frame_importance,
        config=config,
        duration=duration,
    )
    if router_ratios is not None:
        router_anchor, router_event, router_recall = router_ratios
        anchor_ratio = min(0.90, max(0.0, router_anchor))
        event_ratio_cfg = min(1.0, max(0.0, router_event))
        recall_ratio_override = max(0.0, router_recall)

    strong_visual_task = (
        _safe_bool(getattr(config, "talon_visual_task_balance", False))
        and router_ratios is None
        and duration in ("medium", "long")
        and category not in ("knowledge", "life record")
        and (
            "object" in task_category
            or "action" in task_category
            or "counting" in task_category
        )
    )
    if strong_visual_task:
        anchor_target = float(getattr(config, "talon_visual_task_anchor_ratio", 0.84))
        event_target = float(getattr(config, "talon_visual_task_event_ratio", 0.12))
        recall_target = float(getattr(config, "talon_visual_task_recall_ratio", 0.02))
        anchor_ratio = min(0.90, max(anchor_ratio, anchor_target))
        event_ratio_cfg = min(event_ratio_cfg, max(0.0, event_target))
        recall_ratio_override = max(0.0, recall_target)
    if _safe_bool(getattr(config, "talon_duration_aware", False)):
        if duration == "medium":
            anchor_ratio = min(
                0.85,
                max(anchor_ratio, float(getattr(config, "talon_medium_anchor_safety_ratio", 0.78))),
            )
            event_ratio_cfg = min(
                event_ratio_cfg,
                max(0.0, float(getattr(config, "talon_medium_event_budget_ratio", 0.18))),
            )
            global_ratio = max(
                global_ratio,
                min(1.0, max(0.0, float(getattr(config, "talon_medium_global_topk_ratio", 0.80)))),
            )
        elif duration == "long":
            anchor_ratio = min(
                0.85,
                max(anchor_ratio, float(getattr(config, "talon_long_anchor_safety_ratio", 0.80))),
            )
            event_ratio_cfg = min(
                event_ratio_cfg,
                max(0.0, float(getattr(config, "talon_long_event_budget_ratio", 0.14))),
            )
            global_ratio = max(
                global_ratio,
                min(1.0, max(0.0, float(getattr(config, "talon_long_global_topk_ratio", 0.85)))),
            )

    fused_grid = fused_scores.view(num_frames, num_visual_tokens)
    question_grid = question_scores.view(num_frames, num_visual_tokens) if question_scores is not None else None
    residual_grid = residual_scores.view(num_frames, num_visual_tokens)
    combined_grid = combined_scores.view(num_frames, num_visual_tokens)
    spatial_score_mode = str(getattr(config, "talon_spatial_anchor_score", "fused") or "fused").strip().lower()
    selected_mask = torch.zeros((num_frames * num_visual_tokens,), dtype=torch.bool, device=combined_scores.device)
    anchor_mask = torch.zeros_like(selected_mask)
    event_mask = torch.zeros_like(selected_mask)
    recall_mask = torch.zeros_like(selected_mask)
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
            if spatial_score_mode == "combined":
                spatial_scores = combined_grid[t]
            elif spatial_score_mode == "question" and question_grid is not None:
                spatial_scores = question_grid[t]
            elif spatial_score_mode == "event":
                spatial_scores = residual_grid[t]
            else:
                spatial_scores = fused_grid[t]
            spatial_ratio = min(max(float(getattr(config, "talon_spatial_anchor_ratio", 0.35)), 0.0), 1.0)
            spatial_k = min(anchor_k, int(round(anchor_k * spatial_ratio)))
            spatial_idx = _spatial_anchor_topk(spatial_scores, spatial_k, local_selected, num_visual_tokens, config)
            if spatial_idx.numel() > 0:
                local_selected[spatial_idx] = True
                anchor_mask[t * num_visual_tokens + spatial_idx] = True
            remain_anchor = anchor_k - int(local_selected.sum().item())
            if remain_anchor > 0:
                anchor_scores = fused_grid[t].masked_fill(local_selected, -1e9)
                valid = int((anchor_scores > -1e8).sum().item())
                if valid > 0:
                    idx = _diverse_topk(frame_features[t], anchor_scores, min(remain_anchor, valid), config)
                    local_selected[idx] = True
                    anchor_mask[t * num_visual_tokens + idx] = True
        remain = budget_t - int(local_selected.sum().item())
        if remain > 0 and question_grid is not None:
            recall_ratio = min(max(float(getattr(config, "talon_question_recall_ratio", 0.06)), 0.0), 0.60)
            if recall_ratio_override is not None:
                recall_ratio = min(recall_ratio, recall_ratio_override)
            recall_k = min(remain, max(0, int(round(budget_t * recall_ratio))))
            recall_q_weight = min(max(float(getattr(config, "talon_question_recall_qweight", 0.65)), 0.0), 1.0)
            recall_score = _normalize_scores(
                recall_q_weight * question_grid[t] + (1.0 - recall_q_weight) * fused_grid[t]
            )
            qscore = recall_score.masked_fill(local_selected, -1e9)
            valid = int((qscore > -1e8).sum().item())
            if recall_k > 0 and valid > 0:
                idx = torch.topk(qscore, k=min(recall_k, valid), dim=0).indices
                local_selected[idx] = True
                recall_mask[t * num_visual_tokens + idx] = True
        remain = budget_t - int(local_selected.sum().item())
        if remain > 0:
            event_k = min(remain, max(0, int(round(budget_t * event_ratio_cfg))))
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
    return chosen, budgets, anchor_mask, event_mask, recall_mask


def _apply_temporal_chunk_coverage(
    chosen: torch.Tensor,
    combined_scores: torch.Tensor,
    fused_scores: torch.Tensor,
    question_scores: Optional[torch.Tensor],
    innovation_scores: torch.Tensor,
    num_frames: int,
    num_visual_tokens: int,
    total_budget: int,
    config: FlashVidConfig,
) -> torch.Tensor:
    if not _safe_bool(getattr(config, "talon_temporal_chunk_aware", False)):
        return chosen
    duration = str(getattr(config, "current_video_duration", "") or "").strip().lower()
    if duration not in ("medium", "long"):
        return chosen
    if chosen.numel() == 0 or total_budget <= 0 or num_frames <= 1:
        return chosen

    num_chunks = max(1, min(num_frames, int(getattr(config, "talon_temporal_num_chunks", 4) or 4)))
    chunk_ratio = min(max(float(getattr(config, "talon_temporal_chunk_min_ratio", 0.18)), 0.0), 0.80)
    min_per_chunk = max(1, int(round((float(total_budget) / float(num_chunks)) * chunk_ratio)))
    score_mode = str(getattr(config, "talon_temporal_chunk_score", "combined") or "combined").strip().lower()
    if score_mode == "fused":
        score = fused_scores
    elif score_mode == "question" and question_scores is not None:
        score = question_scores
    elif score_mode == "event":
        score = innovation_scores
    else:
        score = combined_scores

    chosen = torch.sort(chosen.to(dtype=torch.long).unique()).values
    chosen_mask = torch.zeros((num_frames * num_visual_tokens,), dtype=torch.bool, device=chosen.device)
    chosen_mask[chosen] = True
    protected_mask = torch.zeros_like(chosen_mask)

    chunk_edges = torch.linspace(0, num_frames, steps=num_chunks + 1, device=chosen.device)
    for chunk_idx in range(num_chunks):
        start_f = int(torch.floor(chunk_edges[chunk_idx]).item())
        end_f = int(torch.floor(chunk_edges[chunk_idx + 1]).item())
        end_f = max(end_f, start_f + 1)
        start_f = min(start_f, num_frames)
        end_f = min(end_f, num_frames)
        if start_f >= end_f:
            continue
        start = start_f * num_visual_tokens
        end = end_f * num_visual_tokens
        chunk_slice = torch.arange(start, end, dtype=torch.long, device=chosen.device)
        existing = chunk_slice[chosen_mask[chunk_slice]]
        if existing.numel() >= min_per_chunk:
            protected_mask[existing] = True
            continue
        need = min_per_chunk - int(existing.numel())
        candidates = chunk_slice[~chosen_mask[chunk_slice]]
        if candidates.numel() == 0:
            protected_mask[existing] = True
            continue
        add = candidates[torch.topk(score[candidates], k=min(need, int(candidates.numel())), dim=0).indices]
        protected_mask[existing] = True
        protected_mask[add] = True
        chosen = torch.cat([chosen, add], dim=0).unique()
        chosen_mask[add] = True

    if chosen.numel() > total_budget:
        removable = chosen[~protected_mask[chosen]]
        overflow = int(chosen.numel()) - int(total_budget)
        if removable.numel() > 0 and overflow > 0:
            remove_k = min(overflow, int(removable.numel()))
            remove = removable[torch.topk(-combined_scores[removable], k=remove_k, dim=0).indices]
            keep_mask = torch.ones((chosen.numel(),), dtype=torch.bool, device=chosen.device)
            remove_mask = torch.zeros((num_frames * num_visual_tokens,), dtype=torch.bool, device=chosen.device)
            remove_mask[remove] = True
            keep_mask &= ~remove_mask[chosen]
            chosen = chosen[keep_mask]
    if chosen.numel() > total_budget:
        chosen = chosen[torch.topk(combined_scores[chosen], k=total_budget, dim=0).indices]
    return torch.sort(chosen.to(dtype=torch.long).unique()).values


def _score_by_mode(
    mode: str,
    combined_scores: torch.Tensor,
    fused_scores: torch.Tensor,
    question_scores: Optional[torch.Tensor],
    innovation_scores: torch.Tensor,
) -> torch.Tensor:
    mode = str(mode or "combined").strip().lower()
    if mode == "fused":
        return fused_scores
    if mode == "question" and question_scores is not None:
        return question_scores
    if mode == "event":
        return innovation_scores
    return combined_scores


def _score_concentration(scores: torch.Tensor, top_ratio: float = 0.10) -> float:
    if scores.numel() == 0:
        return 0.0
    vals = _normalize_scores(scores.float()).clamp_min(0.0)
    total = float(vals.sum().item())
    if total <= 1e-8:
        return 0.0
    k = max(1, int(math.ceil(float(vals.numel()) * min(max(top_ratio, 0.01), 1.0))))
    return float(torch.topk(vals, k=k, dim=0).values.sum().item() / total)


def _normalized_entropy(weights: torch.Tensor) -> float:
    if weights.numel() <= 1:
        return 0.0
    vals = weights.float().clamp_min(0.0)
    total = float(vals.sum().item())
    if total <= 1e-8:
        return 1.0
    prob = vals / total
    entropy = -torch.sum(prob * torch.log(prob.clamp_min(1e-8)))
    return float((entropy / math.log(float(vals.numel()))).item())


def _adaptive_router_ratios(
    fused_scores: torch.Tensor,
    question_scores: Optional[torch.Tensor],
    residual_scores: torch.Tensor,
    frame_importance: torch.Tensor,
    config: FlashVidConfig,
    duration: str,
) -> Optional[Tuple[float, float, float]]:
    # Numeric mode codes are easier to aggregate in jsonl/summary:
    # 0=disabled, 1=visual-anchor, 2=temporal-context, 3=balanced.
    config.last_talon_router_mode_code = 0
    config.last_talon_router_fused_concentration = None
    config.last_talon_router_residual_concentration = None
    config.last_talon_router_question_concentration = None
    config.last_talon_router_frame_entropy = None
    if not _safe_bool(getattr(config, "talon_adaptive_router", False)):
        return None
    if duration == "short" and not _safe_bool(getattr(config, "talon_router_apply_to_short", False)):
        return None

    fused_conc = _score_concentration(fused_scores, top_ratio=0.10)
    residual_conc = _score_concentration(residual_scores, top_ratio=0.10)
    question_conc = _score_concentration(question_scores, top_ratio=0.10) if question_scores is not None else 0.0
    frame_entropy = _normalized_entropy(frame_importance)
    config.last_talon_router_fused_concentration = fused_conc
    config.last_talon_router_residual_concentration = residual_conc
    config.last_talon_router_question_concentration = question_conc
    config.last_talon_router_frame_entropy = frame_entropy

    # Intrinsic routing:
    # - visual-anchor: event residual is flat/noisy, so preserve stable semantic evidence.
    # - temporal-context: only when residual has real peaks and evidence is frame-distributed.
    # - balanced: default safe mode.
    #
    # The previous rule used high frame entropy alone as a temporal signal. On
    # VideoMME medium videos that over-routed to event tokens even though the
    # residual distribution was flat, which is exactly the train-free failure
    # mode we want to avoid.
    visual_evidence = max(fused_conc, question_conc)
    visual_threshold = float(getattr(config, "talon_router_visual_concentration_threshold", 0.28))
    low_residual_threshold = float(getattr(config, "talon_router_low_residual_threshold", 0.30))
    temporal_entropy_threshold = float(getattr(config, "talon_router_temporal_entropy_threshold", 0.95))
    temporal_residual_threshold = float(getattr(config, "talon_router_temporal_residual_threshold", 0.36))
    if residual_conc <= low_residual_threshold or visual_evidence >= visual_threshold:
        config.last_talon_router_mode_code = 1
        return (
            float(getattr(config, "talon_router_visual_anchor_ratio", 0.84)),
            float(getattr(config, "talon_router_visual_event_ratio", 0.12)),
            float(getattr(config, "talon_router_visual_recall_ratio", 0.02)),
        )
    if frame_entropy >= temporal_entropy_threshold and residual_conc >= temporal_residual_threshold:
        config.last_talon_router_mode_code = 2
        return (
            float(getattr(config, "talon_router_temporal_anchor_ratio", 0.66)),
            float(getattr(config, "talon_router_temporal_event_ratio", 0.34)),
            float(getattr(config, "talon_router_temporal_recall_ratio", 0.08)),
        )
    config.last_talon_router_mode_code = 3
    return (
        float(getattr(config, "talon_router_balanced_anchor_ratio", 0.72)),
        float(getattr(config, "talon_router_balanced_event_ratio", 0.30)),
        float(getattr(config, "talon_router_balanced_recall_ratio", 0.08)),
    )


def _apply_track_coverage(
    chosen: torch.Tensor,
    source_to_slot: torch.Tensor,
    combined_scores: torch.Tensor,
    fused_scores: torch.Tensor,
    question_scores: Optional[torch.Tensor],
    innovation_scores: torch.Tensor,
    num_frames: int,
    num_visual_tokens: int,
    total_budget: int,
    config: FlashVidConfig,
) -> torch.Tensor:
    if not _safe_bool(getattr(config, "talon_track_aware", False)):
        return chosen
    duration = str(getattr(config, "current_video_duration", "") or "").strip().lower()
    if duration not in ("medium", "long"):
        return chosen
    if chosen.numel() == 0 or total_budget <= 0 or source_to_slot.numel() == 0:
        return chosen

    score = _score_by_mode(
        getattr(config, "talon_track_score", "combined"),
        combined_scores,
        fused_scores,
        question_scores,
        innovation_scores,
    )
    track_ratio = min(max(float(getattr(config, "talon_track_budget_ratio", 0.12)), 0.0), 0.60)
    tokens_per_slot = max(1, int(getattr(config, "talon_track_tokens_per_slot", 1) or 1))
    track_budget = max(1, int(round(float(total_budget) * track_ratio)))
    num_slots = min(num_visual_tokens, max(1, int(math.ceil(track_budget / float(tokens_per_slot)))))

    flat_slots = source_to_slot.reshape(num_frames * num_visual_tokens).to(dtype=torch.long).clamp(0, num_visual_tokens - 1)
    slot_scores = torch.full((num_visual_tokens,), -1e9, dtype=torch.float32, device=chosen.device)
    try:
        slot_scores.scatter_reduce_(0, flat_slots, score.float(), reduce="amax", include_self=True)
    except Exception:
        for slot in range(num_visual_tokens):
            vals = score[flat_slots == slot]
            if vals.numel() > 0:
                slot_scores[slot] = vals.max()

    top_slots = torch.topk(slot_scores, k=min(num_slots, num_visual_tokens), dim=0).indices
    chosen = torch.sort(chosen.to(dtype=torch.long).unique()).values
    chosen_mask = torch.zeros((num_frames * num_visual_tokens,), dtype=torch.bool, device=chosen.device)
    chosen_mask[chosen] = True
    track_picks: List[torch.Tensor] = []
    for slot in top_slots:
        members = torch.where(flat_slots == int(slot.item()))[0]
        if members.numel() == 0:
            continue
        members = members[~chosen_mask[members]]
        if members.numel() == 0:
            continue
        take = members[torch.topk(score[members], k=min(tokens_per_slot, int(members.numel())), dim=0).indices]
        track_picks.append(take)
        chosen_mask[take] = True

    if not track_picks:
        return chosen
    track_tokens = torch.cat(track_picks, dim=0).unique()
    chosen = torch.cat([chosen, track_tokens], dim=0).unique()
    if chosen.numel() > total_budget:
        protected = torch.zeros((num_frames * num_visual_tokens,), dtype=torch.bool, device=chosen.device)
        protected[track_tokens] = True
        removable = chosen[~protected[chosen]]
        overflow = int(chosen.numel()) - int(total_budget)
        if removable.numel() > 0 and overflow > 0:
            remove = removable[torch.topk(-combined_scores[removable], k=min(overflow, int(removable.numel())), dim=0).indices]
            remove_mask = torch.zeros((num_frames * num_visual_tokens,), dtype=torch.bool, device=chosen.device)
            remove_mask[remove] = True
            chosen = chosen[~remove_mask[chosen]]
    if chosen.numel() > total_budget:
        chosen = chosen[torch.topk(combined_scores[chosen], k=total_budget, dim=0).indices]
    return torch.sort(chosen.to(dtype=torch.long).unique()).values


def _apply_summary_raw_swap(
    chosen: torch.Tensor,
    combined_scores: torch.Tensor,
    fused_scores: torch.Tensor,
    question_scores: Optional[torch.Tensor],
    innovation_scores: torch.Tensor,
    num_frames: int,
    num_visual_tokens: int,
    total_budget: int,
    config: FlashVidConfig,
) -> torch.Tensor:
    if not _safe_bool(getattr(config, "talon_summary_raw_swap", False)):
        return chosen
    duration = str(getattr(config, "current_video_duration", "") or "").strip().lower()
    if duration not in ("medium", "long"):
        return chosen
    if chosen.numel() == 0 or total_budget <= 0:
        return chosen

    ratio = min(max(float(getattr(config, "talon_summary_ratio", 0.08)), 0.0), 0.50)
    if ratio <= 0.0:
        return chosen
    score = _score_by_mode(
        getattr(config, "talon_summary_score", "combined"),
        combined_scores,
        fused_scores,
        question_scores,
        innovation_scores,
    )
    num_chunks = max(1, min(num_frames, int(getattr(config, "talon_summary_num_chunks", 8) or 8)))
    total_swaps = max(1, int(round(float(chosen.numel()) * ratio)))
    per_chunk = max(1, int(math.ceil(total_swaps / float(num_chunks))))

    chosen = torch.sort(chosen.to(dtype=torch.long).unique()).values
    selected_mask = torch.zeros((num_frames * num_visual_tokens,), dtype=torch.bool, device=chosen.device)
    selected_mask[chosen] = True
    chunk_edges = torch.linspace(0, num_frames, steps=num_chunks + 1, device=chosen.device)
    used = 0

    for chunk_idx in range(num_chunks):
        if used >= total_swaps:
            break
        start_f = int(torch.floor(chunk_edges[chunk_idx]).item())
        end_f = int(torch.floor(chunk_edges[chunk_idx + 1]).item())
        end_f = max(end_f, start_f + 1)
        start_f = min(start_f, num_frames)
        end_f = min(end_f, num_frames)
        if start_f >= end_f:
            continue
        start = start_f * num_visual_tokens
        end = end_f * num_visual_tokens
        chunk_flat = torch.arange(start, end, dtype=torch.long, device=chosen.device)
        selected = chunk_flat[selected_mask[chunk_flat]]
        dropped = chunk_flat[~selected_mask[chunk_flat]]
        if selected.numel() == 0 or dropped.numel() == 0:
            continue

        take = min(per_chunk, total_swaps - used, int(selected.numel()), int(dropped.numel()))
        add = dropped[torch.topk(score[dropped], k=take, dim=0).indices]
        remove = selected[torch.topk(-score[selected], k=take, dim=0).indices]
        # Only perform beneficial swaps under the same scoring view. This keeps
        # the operation as a safe raw-token coverage repair rather than random churn.
        keep_pair = score[add] > score[remove]
        if int(keep_pair.sum().item()) <= 0:
            continue
        add = add[keep_pair]
        remove = remove[keep_pair]
        selected_mask[remove] = False
        selected_mask[add] = True
        used += int(add.numel())

    out = torch.where(selected_mask)[0]
    if out.numel() > total_budget:
        out = out[torch.topk(combined_scores[out], k=total_budget, dim=0).indices]
    elif out.numel() < min(total_budget, num_frames * num_visual_tokens):
        fill_scores = combined_scores.masked_fill(selected_mask, -1e9)
        fill_k = min(total_budget - int(out.numel()), int((fill_scores > -1e8).sum().item()))
        if fill_k > 0:
            out = torch.cat([out, torch.topk(fill_scores, k=fill_k, dim=0).indices], dim=0).unique()
    return torch.sort(out.to(dtype=torch.long).unique()).values


def _absorb_dropped_tokens(
    flat_features: torch.Tensor,
    chosen: torch.Tensor,
    combined_scores: torch.Tensor,
    fused_scores: torch.Tensor,
    question_scores: Optional[torch.Tensor],
    innovation_scores: torch.Tensor,
    num_frames: int,
    num_visual_tokens: int,
    config: FlashVidConfig,
) -> torch.Tensor:
    out = flat_features[chosen].clone()
    if not _safe_bool(getattr(config, "talon_absorb_dropped_tokens", False)):
        return out
    duration = str(getattr(config, "current_video_duration", "") or "").strip().lower()
    if duration not in ("medium", "long"):
        return out
    absorb_alpha = min(max(float(getattr(config, "talon_absorb_alpha", 0.25)), 0.0), 1.0)
    absorb_ratio = min(max(float(getattr(config, "talon_absorb_ratio", 0.35)), 0.0), 1.0)
    if absorb_alpha <= 0.0 or absorb_ratio <= 0.0:
        return out

    score = _score_by_mode(
        getattr(config, "talon_absorb_score", "combined"),
        combined_scores,
        fused_scores,
        question_scores,
        innovation_scores,
    )
    chosen_pos = torch.full((num_frames * num_visual_tokens,), -1, dtype=torch.long, device=chosen.device)
    chosen_pos[chosen] = torch.arange(chosen.numel(), dtype=torch.long, device=chosen.device)
    selected_mask = chosen_pos >= 0

    for t in range(num_frames):
        start = t * num_visual_tokens
        end = start + num_visual_tokens
        frame_idx = torch.arange(start, end, dtype=torch.long, device=chosen.device)
        selected = frame_idx[selected_mask[frame_idx]]
        dropped = frame_idx[~selected_mask[frame_idx]]
        if selected.numel() == 0 or dropped.numel() == 0:
            continue
        keep_dropped = max(1, int(round(float(dropped.numel()) * absorb_ratio)))
        dropped = dropped[torch.topk(score[dropped], k=min(keep_dropped, int(dropped.numel())), dim=0).indices]

        sel_feat = F.normalize(flat_features[selected].float(), p=2, dim=-1, eps=1e-6)
        drop_feat = F.normalize(flat_features[dropped].float(), p=2, dim=-1, eps=1e-6)
        assign = torch.argmax(torch.matmul(drop_feat, sel_feat.transpose(0, 1)), dim=1)

        for local_sel in torch.unique(assign):
            assigned = dropped[assign == local_sel]
            if assigned.numel() == 0:
                continue
            target_global = selected[int(local_sel.item())]
            target_out = chosen_pos[target_global]
            weights = _normalize_scores(score[assigned]).to(dtype=flat_features.dtype).clamp_min(1e-4)
            merged = torch.sum(flat_features[assigned] * weights.unsqueeze(-1), dim=0) / weights.sum()
            out[target_out] = (1.0 - absorb_alpha) * out[target_out] + absorb_alpha * merged.to(dtype=out.dtype)
    return out


def _apply_summary_replacement(
    output_features: torch.Tensor,
    flat_features: torch.Tensor,
    chosen: torch.Tensor,
    combined_scores: torch.Tensor,
    fused_scores: torch.Tensor,
    question_scores: Optional[torch.Tensor],
    innovation_scores: torch.Tensor,
    num_frames: int,
    num_visual_tokens: int,
    config: FlashVidConfig,
) -> torch.Tensor:
    if not _safe_bool(getattr(config, "talon_summary_replacement", False)):
        return output_features
    duration = str(getattr(config, "current_video_duration", "") or "").strip().lower()
    if duration not in ("medium", "long"):
        return output_features
    ratio = min(max(float(getattr(config, "talon_summary_ratio", 0.08)), 0.0), 0.50)
    alpha = min(max(float(getattr(config, "talon_summary_alpha", 0.55)), 0.0), 1.0)
    if ratio <= 0.0 or alpha <= 0.0 or chosen.numel() == 0:
        return output_features

    score = _score_by_mode(
        getattr(config, "talon_summary_score", "combined"),
        combined_scores,
        fused_scores,
        question_scores,
        innovation_scores,
    )
    num_chunks = max(1, int(getattr(config, "talon_summary_num_chunks", 8) or 8))
    num_chunks = min(num_chunks, num_frames)
    total_summary = max(1, int(round(float(chosen.numel()) * ratio)))
    pool_topk = max(1, int(getattr(config, "talon_summary_pool_topk", 12) or 12))

    chosen_pos = torch.full((num_frames * num_visual_tokens,), -1, dtype=torch.long, device=chosen.device)
    chosen_pos[chosen] = torch.arange(chosen.numel(), dtype=torch.long, device=chosen.device)
    chunk_edges = torch.linspace(0, num_frames, steps=num_chunks + 1, device=chosen.device)
    per_chunk = max(1, int(math.ceil(total_summary / float(num_chunks))))
    used = 0
    out = output_features

    for chunk_idx in range(num_chunks):
        if used >= total_summary:
            break
        start_f = int(torch.floor(chunk_edges[chunk_idx]).item())
        end_f = int(torch.floor(chunk_edges[chunk_idx + 1]).item())
        end_f = max(end_f, start_f + 1)
        start_f = min(start_f, num_frames)
        end_f = min(end_f, num_frames)
        if start_f >= end_f:
            continue
        start = start_f * num_visual_tokens
        end = end_f * num_visual_tokens
        chunk_idx_flat = torch.arange(start, end, dtype=torch.long, device=chosen.device)
        selected = chunk_idx_flat[chosen_pos[chunk_idx_flat] >= 0]
        if selected.numel() == 0:
            continue

        take = min(per_chunk, total_summary - used, int(selected.numel()))
        # Keep the strongest raw evidence untouched. Summary tokens should
        # occupy the weakest selected slots, otherwise this branch corrupts the
        # very anchors that made the frozen VLM stable.
        reps = selected[torch.topk(-score[selected], k=take, dim=0).indices]

        selected_mask = chosen_pos[chunk_idx_flat] >= 0
        dropped = chunk_idx_flat[~selected_mask]
        pool_source = dropped if dropped.numel() > 0 else chunk_idx_flat
        pool_k = min(pool_topk * take, int(pool_source.numel()))
        pool = pool_source[torch.topk(score[pool_source], k=pool_k, dim=0).indices]
        pool_scores = score[pool].float()
        weights = torch.softmax(pool_scores / 0.07, dim=0).to(dtype=flat_features.dtype)
        summary = torch.sum(flat_features[pool] * weights.unsqueeze(-1), dim=0)

        for rep in reps:
            out_pos = int(chosen_pos[rep].item())
            out[out_pos] = (1.0 - alpha) * out[out_pos] + alpha * summary.to(dtype=out.dtype)
            used += 1
            if used >= total_summary:
                break
    return out


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
    lowrank_residual = torch.mean((video_features.float() - reconstruction.float()) ** 2, dim=-1)
    echo_residual = _echo_residual(video_features, flashvid_config)
    echo_weight = min(max(float(getattr(flashvid_config, "talon_echo_residual_weight", 0.0)), 0.0), 1.0)
    residual_scores = _normalize_scores(
        (1.0 - echo_weight) * _normalize_scores(lowrank_residual.reshape(num_tokens))
        + echo_weight * _normalize_scores(echo_residual.reshape(num_tokens))
    )
    residual_norm = residual_scores

    innovation_attention_weight = min(max(float(getattr(flashvid_config, "talon_innovation_attention_weight", 0.45)), 0.0), 1.0)
    duration = str(getattr(flashvid_config, "current_video_duration", "") or "").strip().lower()
    task_category = str(getattr(flashvid_config, "current_task_category", "") or "").strip().lower()
    category = str(getattr(flashvid_config, "current_category", "") or "").strip().lower()
    task_aware_event = (
        _safe_bool(getattr(flashvid_config, "talon_task_aware_event", False))
        and duration in ("medium", "long")
        and (
            "object" in task_category
            or "action" in task_category
            or "attribute" in task_category
            or "temporal" in task_category
            or category in ("sports competition", "film & television", "artistic performance")
        )
    )
    if task_aware_event:
        innovation_attention_weight = max(
            innovation_attention_weight,
            min(max(float(getattr(flashvid_config, "talon_task_event_attention_weight", 0.82)), 0.0), 1.0),
        )
    innovation_scores = _normalize_scores((1.0 - innovation_attention_weight) * residual_norm + innovation_attention_weight * fused_scores)
    if question_scores is not None and _safe_bool(getattr(flashvid_config, "talon_use_question_innovation", True)):
        q_weight = min(max(float(getattr(flashvid_config, "talon_innovation_qweight", 0.25)), 0.0), 1.0)
        if task_aware_event:
            q_weight = max(
                q_weight,
                min(max(float(getattr(flashvid_config, "talon_task_event_qweight", 0.30)), 0.0), 1.0),
            )
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

    monotonic_base_tpf = max(0, int(getattr(flashvid_config, "talon_monotonic_base_tokens_per_frame", 20) or 0))
    use_monotonic_expansion = (
        monotonic_base_tpf > 0
        and target_per_frame > monotonic_base_tpf
        and total_budget > monotonic_base_tpf * num_frames
    )
    if use_monotonic_expansion:
        # Keep higher budgets as a strict extension of the stable low-budget set.
        # This avoids the t=21 path reshuffling anchors/events and accidentally
        # dropping tokens that made the t=20 path work.
        base_budget = _resolve_total_budget(num_frames, num_visual_tokens, flashvid_config, monotonic_base_tpf)
        chosen, budgets, anchor_mask, event_mask, recall_mask = _select_tokens(
            frame_features=video_features,
            fused_scores=fused_scores,
            question_scores=question_scores,
            residual_scores=innovation_scores,
            combined_scores=combined_scores,
            total_budget=base_budget,
            num_frames=num_frames,
            num_visual_tokens=num_visual_tokens,
            frame_importance=frame_importance,
            config=flashvid_config,
        )
        if int(chosen.numel()) < total_budget:
            selected_mask = torch.zeros((num_tokens,), dtype=torch.bool, device=device)
            selected_mask[chosen] = True
            # Extra budget should be conservative: the stable t=20 path already
            # captures innovation/recall. Additional tokens are safer as semantic
            # anchors than as residual-heavy event picks, which can distract a
            # frozen VLM even when the token count increases.
            extra_scores = fused_scores.masked_fill(selected_mask, -1e9)
            fill_k = min(total_budget - int(chosen.numel()), int((extra_scores > -1e8).sum().item()))
            if fill_k > 0:
                extra = torch.topk(extra_scores, k=fill_k, dim=0).indices
                chosen = torch.cat([chosen, extra], dim=0).unique()
                anchor_mask[extra] = True
        if int(chosen.numel()) > total_budget:
            chosen = chosen[torch.topk(fused_scores[chosen], k=total_budget, dim=0).indices]
        chosen = torch.sort(chosen.to(dtype=torch.long)).values
        budgets = _allocate_frame_budget(total_budget, frame_importance, flashvid_config)
    else:
        chosen, budgets, anchor_mask, event_mask, recall_mask = _select_tokens(
            frame_features=video_features,
            fused_scores=fused_scores,
            question_scores=question_scores,
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
    chosen = _apply_temporal_chunk_coverage(
        chosen=chosen,
        combined_scores=combined_scores,
        fused_scores=fused_scores,
        question_scores=question_scores,
        innovation_scores=innovation_scores,
        num_frames=num_frames,
        num_visual_tokens=num_visual_tokens,
        total_budget=total_budget,
        config=flashvid_config,
    )
    chosen = _apply_track_coverage(
        chosen=chosen,
        source_to_slot=source_to_slot,
        combined_scores=combined_scores,
        fused_scores=fused_scores,
        question_scores=question_scores,
        innovation_scores=innovation_scores,
        num_frames=num_frames,
        num_visual_tokens=num_visual_tokens,
        total_budget=total_budget,
        config=flashvid_config,
    )
    chosen = _apply_summary_raw_swap(
        chosen=chosen,
        combined_scores=combined_scores,
        fused_scores=fused_scores,
        question_scores=question_scores,
        innovation_scores=innovation_scores,
        num_frames=num_frames,
        num_visual_tokens=num_visual_tokens,
        total_budget=total_budget,
        config=flashvid_config,
    )

    chosen_mask = torch.zeros((num_tokens,), dtype=torch.bool, device=device)
    chosen_mask[chosen] = True
    anchor_tokens = int((chosen_mask & anchor_mask).sum().item())
    recall_tokens = int((chosen_mask & recall_mask).sum().item())
    event_tokens = int((chosen_mask & event_mask).sum().item())
    # Residual top-k that entered via the final combined/global fill is still an event.
    if event_tokens < int(chosen.numel()) - anchor_tokens - recall_tokens:
        event_threshold_k = max(0, int(chosen.numel()) - anchor_tokens - recall_tokens)
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
    flashvid_config.last_talon_recall_tokens = int(recall_tokens)
    flashvid_config.last_talon_memory_tokens = 0
    flashvid_config.last_talon_segment_count = 1
    flashvid_config.last_talon_rank_cap = int(max_rank)
    flashvid_config.last_talon_chosen_rank = int(rank)
    flashvid_config.last_talon_duplicate_index_count = int(chosen.numel()) - int(chosen.unique().numel())
    flashvid_config.last_talon_question_aware_active = question_scores is not None

    output_features = _absorb_dropped_tokens(
        flat_features=flat_features,
        chosen=chosen,
        combined_scores=combined_scores,
        fused_scores=fused_scores,
        question_scores=question_scores,
        innovation_scores=innovation_scores,
        num_frames=num_frames,
        num_visual_tokens=num_visual_tokens,
        config=flashvid_config,
    )
    output_features = _apply_summary_replacement(
        output_features=output_features,
        flat_features=flat_features,
        chosen=chosen,
        combined_scores=combined_scores,
        fused_scores=fused_scores,
        question_scores=question_scores,
        innovation_scores=innovation_scores,
        num_frames=num_frames,
        num_visual_tokens=num_visual_tokens,
        config=flashvid_config,
    )
    return output_features, flat_indices[chosen]
