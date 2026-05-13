from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch
from torch.nn import functional as F

from .configuration_flashvid import FlashVidConfig


@dataclass
class _TransportState:
    aligned: torch.Tensor
    source_to_aligned: torch.Tensor


@dataclass
class _RankPlan:
    rank: int
    residual_scores: torch.Tensor
    reconstruction_source: torch.Tensor


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


def _question_aware_scores(
    flat_features: torch.Tensor,
    flat_attention: torch.Tensor,
    question_features: Optional[torch.Tensor],
    config: FlashVidConfig,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    visual_scores = _normalize_scores(flat_attention)
    if not _safe_bool(getattr(config, "question_aware_reweighting", False)):
        return visual_scores, None
    if question_features is None or question_features.numel() == 0:
        return visual_scores, None

    token_features = F.normalize(flat_features.float(), p=2, dim=-1, eps=1e-6)
    question_proto = F.normalize(question_features.float().mean(dim=0), p=2, dim=-1, eps=1e-6)
    question_scores = _normalize_scores(torch.matmul(token_features, question_proto))
    beta = min(max(float(getattr(config, "question_reweight_beta", 0.35)), 0.0), 1.0)
    return (1.0 - beta) * visual_scores + beta * question_scores, question_scores


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
    length_score = min(1.0, q.shape[0] / 32.0)
    return float(0.5 * dispersion.mean().item() + 0.5 * length_score)


def _resolve_retention_ratio(
    video_features: torch.Tensor,
    question_features: Optional[torch.Tensor],
    config: FlashVidConfig,
) -> float:
    base = float(config.retention_ratio)
    if not _safe_bool(getattr(config, "adaptive_token_budget", False)):
        config.last_adaptive_retention_ratio = base
        return base
    candidates = sorted(
        [
            max(0.01, min(1.0, float(getattr(config, "adaptive_budget_low", 0.10)))),
            max(0.01, min(1.0, float(getattr(config, "adaptive_budget_mid", 0.15)))),
            max(0.01, min(1.0, float(getattr(config, "adaptive_budget_high", 0.20)))),
        ]
    )
    score = 0.7 * _estimate_video_complexity(video_features) + 0.3 * _estimate_question_difficulty(question_features)
    idx = min(len(candidates) - 1, int(score * len(candidates)))
    config.last_adaptive_retention_ratio = candidates[idx]
    return candidates[idx]


def _resolve_talon_target_tokens_per_frame(
    video_features: torch.Tensor,
    question_features: Optional[torch.Tensor],
    config: FlashVidConfig,
) -> int:
    base_target = int(getattr(config, "talon_target_tokens_per_frame", 0) or 0)
    if base_target <= 0:
        config.last_talon_target_tokens_per_frame = None
        return 0

    base_target = max(1, min(base_target, int(video_features.shape[1])))
    if not _safe_bool(getattr(config, "adaptive_token_budget", False)):
        config.last_talon_target_tokens_per_frame = base_target
        return base_target

    low = int(getattr(config, "talon_adaptive_target_low", 0) or 0)
    mid = int(getattr(config, "talon_adaptive_target_mid", 0) or 0)
    high = int(getattr(config, "talon_adaptive_target_high", 0) or 0)
    if low <= 0:
        low = max(1, int(round(base_target * 0.90)))
    if mid <= 0:
        mid = base_target
    if high <= 0:
        high = min(int(video_features.shape[1]), max(mid, int(round(base_target * 1.20))))

    candidates = sorted([
        max(1, min(low, int(video_features.shape[1]))),
        max(1, min(mid, int(video_features.shape[1]))),
        max(1, min(high, int(video_features.shape[1]))),
    ])
    score = 0.7 * _estimate_video_complexity(video_features) + 0.3 * _estimate_question_difficulty(question_features)
    score = max(0.0, min(1.0, float(score)))
    config.last_talon_complexity_score = score

    low_t, mid_t, high_t = candidates
    if score <= 0.5:
        alpha = score / 0.5
        target = int(round((1.0 - alpha) * low_t + alpha * mid_t))
    else:
        alpha = (score - 0.5) / 0.5
        target = int(round((1.0 - alpha) * mid_t + alpha * high_t))
    config.last_talon_target_tokens_per_frame = target
    return target


def _segment_lengths(video_features: torch.Tensor, config: FlashVidConfig) -> torch.Tensor:
    num_frames = int(video_features.shape[0])
    if num_frames <= 1 or not _safe_bool(getattr(config, "do_segment", True)):
        return torch.tensor([num_frames], dtype=torch.long, device=video_features.device)
    if not _safe_bool(getattr(config, "talon_use_segmentation", False)):
        return torch.tensor([num_frames], dtype=torch.long, device=video_features.device)

    min_segments = int(getattr(config, "min_segment_num", 8))
    complementary = _safe_bool(getattr(config, "complementary_segment", True))
    if _safe_bool(getattr(config, "talon_disable_oversegmentation", True)):
        min_segments = min(min_segments, max(1, int(getattr(config, "talon_max_segments", 4))), num_frames)
        complementary = False

    frame_features = F.normalize(video_features.mean(dim=1).float(), p=2, dim=-1, eps=1e-6)
    sims = torch.sum(frame_features[:-1] * frame_features[1:], dim=-1)
    threshold = float(getattr(config, "segment_threshold", 0.9))
    cut_indices = torch.where(sims < threshold)[0]

    num_segments = int(cut_indices.numel()) + 1
    if complementary and num_segments < min_segments and sims.numel() > 0:
        remaining = min_segments - num_segments
        masked = sims.clone()
        masked[masked < threshold] = 1.0
        extra = torch.topk(masked, k=min(remaining, masked.shape[0]), largest=False).indices
        cut_indices = torch.cat([cut_indices, extra]).sort().values

    padded = F.pad(cut_indices, (1, 1), value=0)
    padded[0] = -1
    padded[-1] = num_frames - 1
    return torch.diff(padded, n=1, dim=0).to(dtype=torch.long)


def _build_local_neighbors(num_visual_tokens: int, config: FlashVidConfig, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    _, grid_w = _resolve_grid_hw(num_visual_tokens, config)
    radius = max(0, int(getattr(config, "talon_transport_radius", 1)))
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


def _transport_align(segment_features: torch.Tensor, config: FlashVidConfig) -> _TransportState:
    num_frames, num_visual_tokens, feat_dim = segment_features.shape
    device = segment_features.device
    identity = torch.arange(num_visual_tokens, dtype=torch.long, device=device)
    if num_frames <= 1:
        mapping = identity.unsqueeze(0).repeat(num_frames, 1)
        return _TransportState(segment_features, mapping)

    neighbor_idx, neighbor_valid = _build_local_neighbors(num_visual_tokens, config, device)
    mode = str(getattr(config, "talon_transport_mode", "hard") or "hard").strip().lower()
    temperature = max(1e-4, float(getattr(config, "talon_transport_temperature", 0.07)))

    aligned = segment_features.clone()
    source_to_aligned = torch.empty((num_frames, num_visual_tokens), dtype=torch.long, device=device)
    source_to_aligned[0] = identity

    for t in range(1, num_frames):
        prev = aligned[t - 1]
        cur = segment_features[t]
        prev_norm = F.normalize(prev.float(), p=2, dim=-1, eps=1e-6)
        cur_norm = F.normalize(cur.float(), p=2, dim=-1, eps=1e-6)
        local_prev = prev_norm[neighbor_idx]
        sims = torch.sum(cur_norm.unsqueeze(1) * local_prev, dim=-1)
        sims = sims.masked_fill(~neighbor_valid, -1e9)
        best_pos = torch.argmax(sims, dim=1)
        best_aligned = neighbor_idx[identity, best_pos]

        if mode in ("soft", "entropy", "entropic"):
            weights = torch.softmax(sims / temperature, dim=1).to(cur.dtype)
            aligned_t = torch.zeros((num_visual_tokens, feat_dim), dtype=cur.dtype, device=device)
            counts = torch.zeros((num_visual_tokens, 1), dtype=cur.dtype, device=device)
            flat_aligned = neighbor_idx.reshape(-1)
            flat_weights = weights.reshape(-1)
            flat_valid = neighbor_valid.reshape(-1)
            flat_sources = cur.unsqueeze(1).expand(-1, neighbor_idx.shape[1], -1).reshape(-1, feat_dim)
            flat_aligned = flat_aligned[flat_valid]
            flat_weights = flat_weights[flat_valid]
            flat_sources = flat_sources[flat_valid]
            aligned_t.scatter_add_(0, flat_aligned.unsqueeze(-1).expand(-1, feat_dim), flat_sources * flat_weights.unsqueeze(-1))
            counts.scatter_add_(0, flat_aligned.unsqueeze(-1), flat_weights.unsqueeze(-1))
            aligned_t = aligned_t / counts.clamp_min(1e-6)
        else:
            aligned_t = torch.zeros((num_visual_tokens, feat_dim), dtype=cur.dtype, device=device)
            counts = torch.zeros((num_visual_tokens, 1), dtype=cur.dtype, device=device)
            aligned_t.scatter_add_(0, best_aligned.unsqueeze(-1).expand(-1, feat_dim), cur)
            counts.scatter_add_(0, best_aligned.unsqueeze(-1), torch.ones((num_visual_tokens, 1), dtype=cur.dtype, device=device))
            aligned_t = aligned_t / counts.clamp_min(1.0)

        empty = torch.isnan(aligned_t).any(dim=-1) | (aligned_t.abs().sum(dim=-1) <= 0)
        if empty.any():
            aligned_t[empty] = cur[empty]
        aligned[t] = aligned_t
        source_to_aligned[t] = best_aligned

    return _TransportState(aligned=aligned, source_to_aligned=source_to_aligned)


def _lowrank_basis(aligned_features: torch.Tensor, max_rank: int, config: FlashVidConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    _, num_visual_tokens, _ = aligned_features.shape
    max_rank = max(0, min(int(max_rank), int(num_visual_tokens)))
    if max_rank <= 0:
        return aligned_features.new_zeros((num_visual_tokens, 0)), aligned_features.new_zeros((0,), dtype=torch.float32)

    method = str(getattr(config, "talon_basis_method", "randomized") or "randomized").strip().lower()
    if method in ("randomized", "random", "sketch"):
        oversample = max(0, int(getattr(config, "talon_basis_oversample", 4)))
        sketch_rank = min(num_visual_tokens, max_rank + oversample)
        # Treat aligned video as A in R^{P x (T*d)} and approximate the top
        # left singular vectors without materializing the P x P covariance/eigh path.
        matrix = aligned_features.float().permute(1, 0, 2).reshape(num_visual_tokens, -1)
        generator = torch.Generator(device=matrix.device)
        generator.manual_seed(0)
        omega = torch.randn(
            (matrix.shape[1], sketch_rank),
            dtype=matrix.dtype,
            device=matrix.device,
            generator=generator,
        )
        q_basis, _ = torch.linalg.qr(torch.matmul(matrix, omega), mode="reduced")
        small = torch.matmul(q_basis.transpose(0, 1), matrix)
        small_cov = torch.matmul(small, small.transpose(0, 1))
        eigvals_small, eigvecs_small = torch.linalg.eigh(small_cov)
        order = torch.argsort(eigvals_small, descending=True)
        eigvals = eigvals_small[order][:max_rank].float()
        basis = torch.matmul(q_basis, eigvecs_small[:, order[:max_rank]])
        return basis.to(dtype=aligned_features.dtype), eigvals

    cov = torch.zeros((num_visual_tokens, num_visual_tokens), dtype=torch.float32, device=aligned_features.device)
    for frame in aligned_features:
        frame_f = frame.float()
        cov = cov + torch.matmul(frame_f, frame_f.transpose(0, 1))
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    return eigvecs[:, :max_rank].to(dtype=aligned_features.dtype), eigvals[:max_rank].float()


def _resolve_memory_budget(num_tokens: int, target_budget: int, config: FlashVidConfig) -> int:
    dropped = max(0, num_tokens - target_budget)
    if dropped == 0:
        return 0
    ratio = max(0.0, float(getattr(config, "memory_token_ratio", 0.10)))
    memory_min = max(0, int(getattr(config, "memory_token_min", 1)))
    memory_max = max(memory_min, int(getattr(config, "memory_token_max", 16)))
    budget = int(round(target_budget * ratio))
    budget = min(max(memory_min, budget), memory_max, dropped, max(0, target_budget - 1))
    return max(0, budget)


def _allocate_frame_budget(total_budget: int, frame_importance: torch.Tensor, mode: str) -> List[int]:
    num_frames = int(frame_importance.shape[0])
    if num_frames <= 0:
        return []
    if total_budget <= 0:
        return [0 for _ in range(num_frames)]

    mode = str(mode).strip().lower()
    if mode == "attention":
        weights = frame_importance.float().clamp_min(0.0)
        if float(weights.sum().item()) <= 1e-8:
            weights = torch.ones_like(weights)
        weights = weights / weights.sum()
        raw = weights * float(total_budget)
        base = torch.floor(raw).to(dtype=torch.long)
        budgets = base.clone()
        remaining = int(total_budget - int(base.sum().item()))
        if remaining > 0:
            fractional = raw - base.float()
            order = torch.argsort(fractional, descending=True)
            budgets[order[:remaining]] += 1
        return [int(x.item()) for x in budgets]

    base = int(total_budget // num_frames)
    budgets = [base for _ in range(num_frames)]
    remaining = int(total_budget - base * num_frames)
    if remaining <= 0:
        return budgets
    order = torch.arange(num_frames, device=frame_importance.device)
    for idx in order[:remaining]:
        budgets[int(idx.item())] += 1
    return budgets


def _hybrid_global_frame_select(
    scores: torch.Tensor,
    selected_mask: torch.Tensor,
    budget: int,
    num_frames: int,
    num_visual_tokens: int,
    frame_importance: torch.Tensor,
    budget_mode: str,
    global_ratio: float = 0.70,
    min_per_frame: int = 0,
) -> torch.Tensor:
    budget = min(max(0, int(budget)), int((~selected_mask).sum().item()))
    if budget <= 0:
        return torch.empty((0,), dtype=torch.long, device=scores.device)

    global_ratio = min(max(float(global_ratio), 0.0), 1.0)
    selected = []
    occupied = selected_mask.clone()

    # Stage-1: global top-k to maximize question/attention utility.
    global_k = min(budget, max(0, int(round(budget * global_ratio))))
    if global_k > 0:
        global_scores = scores.masked_fill(occupied, -1e9)
        valid_global = int((global_scores > -1e8).sum().item())
        if valid_global > 0:
            gk = min(global_k, valid_global)
            top_global = torch.topk(global_scores, k=gk, dim=0).indices
            selected.append(top_global)
            occupied[top_global] = True

    # Stage-2: frame-aware refill to prevent over-collapse on few frames.
    remaining = budget - sum(int(x.numel()) for x in selected)
    if remaining <= 0:
        return torch.cat(selected, dim=0).to(dtype=torch.long) if selected else torch.empty((0,), dtype=torch.long, device=scores.device)

    frame_budgets = _allocate_frame_budget(remaining, frame_importance, budget_mode)
    score_grid = scores.view(num_frames, num_visual_tokens)
    occupied_grid = occupied.view(num_frames, num_visual_tokens)
    frame_pick = []
    for t in range(num_frames):
        required = max(0, int(min_per_frame))
        frame_budget = max(required, int(frame_budgets[t]))
        available = int((~occupied_grid[t]).sum().item())
        keep_t = min(frame_budget, available)
        if keep_t <= 0:
            continue
        local_scores = score_grid[t].masked_fill(occupied_grid[t], -1e9)
        local_top = torch.topk(local_scores, k=keep_t, dim=0).indices
        global_idx = t * num_visual_tokens + local_top
        frame_pick.append(global_idx)
        occupied[global_idx] = True

    if frame_pick:
        selected.append(torch.cat(frame_pick, dim=0))

    # Stage-3: final global refill for any leftover slots.
    picked = sum(int(x.numel()) for x in selected)
    leftover = budget - picked
    if leftover > 0:
        refill_scores = scores.masked_fill(occupied, -1e9)
        valid_refill = int((refill_scores > -1e8).sum().item())
        if valid_refill > 0:
            rk = min(leftover, valid_refill)
            refill = torch.topk(refill_scores, k=rk, dim=0).indices
            selected.append(refill)

    if not selected:
        return torch.empty((0,), dtype=torch.long, device=scores.device)
    return torch.cat(selected, dim=0).unique().to(dtype=torch.long)


def _frame_importance(
    segment_features: torch.Tensor,
    cls_attention: torch.Tensor,
    question_features: Optional[torch.Tensor],
    config: FlashVidConfig,
) -> torch.Tensor:
    num_frames = int(segment_features.shape[0])
    attention = _normalize_scores(cls_attention.float().mean(dim=1))
    if num_frames <= 1:
        return attention

    centers = F.normalize(segment_features.float().mean(dim=1), p=2, dim=-1, eps=1e-6)
    transition = torch.zeros((num_frames,), dtype=torch.float32, device=segment_features.device)
    delta = 0.5 * (1.0 - torch.sum(centers[:-1] * centers[1:], dim=-1)).clamp(0.0, 2.0)
    transition[:-1] = torch.maximum(transition[:-1], delta)
    transition[1:] = torch.maximum(transition[1:], delta)
    transition = _normalize_scores(transition)

    boundary = torch.zeros_like(transition)
    boundary[0] = 1.0
    boundary[-1] = 1.0

    motion_weight = min(max(float(getattr(config, "talon_motion_importance_weight", 0.35)), 0.0), 1.0)
    boundary_weight = min(max(float(getattr(config, "talon_boundary_importance_weight", 0.10)), 0.0), 1.0)
    question_weight = 0.0
    question_frame = torch.zeros_like(attention)
    if _safe_bool(getattr(config, "question_aware_reweighting", False)) and question_features is not None and question_features.numel() > 0:
        q = F.normalize(question_features.float(), p=2, dim=-1, eps=1e-6)
        q_proto = F.normalize(q.mean(dim=0), p=2, dim=-1, eps=1e-6)
        question_frame = _normalize_scores(torch.matmul(centers, q_proto))
        question_weight = min(max(float(getattr(config, "talon_question_frame_weight", 0.20)), 0.0), 1.0)

    attention_weight = max(0.0, 1.0 - motion_weight - boundary_weight - question_weight)
    return _normalize_scores(
        attention_weight * attention
        + motion_weight * transition
        + boundary_weight * boundary
        + question_weight * question_frame
    )


def _build_memory_tokens(
    flat_features: torch.Tensor,
    dropped_indices: torch.Tensor,
    residual_scores: torch.Tensor,
    memory_budget: int,
    num_frames: Optional[int] = None,
    num_visual_tokens: Optional[int] = None,
    frame_importance: Optional[torch.Tensor] = None,
    budget_mode: str = "uniform",
    frame_balanced: bool = False,
    memory_mode: str = "raw",
) -> Tuple[torch.Tensor, torch.Tensor]:
    feat_dim = flat_features.shape[-1]
    if memory_budget <= 0 or dropped_indices.numel() == 0:
        return flat_features.new_zeros((0, feat_dim)), torch.empty((0,), dtype=torch.long, device=flat_features.device)
    raw_mode = str(memory_mode or "raw").strip().lower() in ("raw", "anchor", "anchors", "select")

    if frame_balanced and num_frames is not None and num_visual_tokens is not None and frame_importance is not None:
        priorities_all = residual_scores.float()
        frame_budgets = _allocate_frame_budget(memory_budget, frame_importance, budget_mode)
        tokens = []
        indices = []
        for t, budget_t in enumerate(frame_budgets):
            if budget_t <= 0:
                continue
            start = t * int(num_visual_tokens)
            end = start + int(num_visual_tokens)
            frame_mask = (dropped_indices >= start) & (dropped_indices < end)
            frame_dropped = dropped_indices[frame_mask]
            if frame_dropped.numel() == 0:
                continue
            priorities = priorities_all[frame_dropped].clamp_min(1e-6)
            order = torch.argsort(priorities, descending=True)
            frame_dropped = frame_dropped[order]
            priorities = priorities[order]
            if raw_mode:
                keep = min(int(budget_t), int(frame_dropped.numel()))
                tokens.append(flat_features[frame_dropped[:keep]])
                indices.append(frame_dropped[:keep])
                continue
            chunks = torch.chunk(
                torch.arange(frame_dropped.numel(), device=flat_features.device),
                min(int(budget_t), int(frame_dropped.numel())),
            )
            for chunk in chunks:
                if chunk.numel() == 0:
                    continue
                idx = frame_dropped[chunk]
                weights = priorities[chunk].to(flat_features.dtype)
                merged = torch.sum(flat_features[idx] * weights.unsqueeze(-1), dim=0) / weights.sum()
                tokens.append(merged)
                indices.append(idx[0])
        if tokens:
            if raw_mode:
                return torch.cat(tokens, dim=0), torch.cat(indices, dim=0)
            return torch.stack(tokens, dim=0), torch.stack(indices, dim=0)

    priorities = residual_scores[dropped_indices].float()
    order = torch.argsort(priorities, descending=True)
    ordered = dropped_indices[order]
    ordered_priorities = priorities[order].clamp_min(1e-6)
    num_chunks = min(memory_budget, int(ordered.numel()))
    if raw_mode:
        ordered = ordered[:num_chunks]
        return flat_features[ordered], ordered
    chunks = torch.chunk(torch.arange(ordered.numel(), device=flat_features.device), num_chunks)

    tokens = []
    indices = []
    for chunk in chunks:
        if chunk.numel() == 0:
            continue
        idx = ordered[chunk]
        weights = ordered_priorities[chunk].to(flat_features.dtype)
        merged = torch.sum(flat_features[idx] * weights.unsqueeze(-1), dim=0) / weights.sum()
        tokens.append(merged)
        indices.append(idx[0])
    if not tokens:
        return flat_features.new_zeros((0, feat_dim)), torch.empty((0,), dtype=torch.long, device=flat_features.device)
    return torch.stack(tokens, dim=0), torch.stack(indices, dim=0)


def _emit_rescue_tokens(
    flat_features: torch.Tensor,
    flat_indices: torch.Tensor,
    fused_scores: torch.Tensor,
    residual_scores: torch.Tensor,
    frame_importance: torch.Tensor,
    selected_mask: torch.Tensor,
    budget: int,
    num_frames: int,
    num_visual_tokens: int,
    budget_mode: str,
    config: FlashVidConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    budget = min(max(0, int(budget)), int((~selected_mask).sum().item()))
    if budget <= 0:
        feat_dim = flat_features.shape[-1]
        return flat_features.new_zeros((0, feat_dim)), torch.empty((0,), dtype=torch.long, device=flat_features.device)

    fused_w = float(getattr(config, "talon_rescue_fused_weight", 0.55))
    residual_w = float(getattr(config, "talon_rescue_residual_weight", 0.35))
    frame_w = float(getattr(config, "talon_rescue_frame_weight", 0.10))
    fused_w = max(0.0, fused_w)
    residual_w = max(0.0, residual_w)
    frame_w = max(0.0, frame_w)
    denom = max(1e-6, fused_w + residual_w + frame_w)
    fused_w, residual_w, frame_w = fused_w / denom, residual_w / denom, frame_w / denom

    frame_prior = frame_importance.float().repeat_interleave(num_visual_tokens)
    scores = (
        fused_w * _normalize_scores(fused_scores.float())
        + residual_w * _normalize_scores(residual_scores.float())
        + frame_w * _normalize_scores(frame_prior)
    )
    scores = scores.masked_fill(selected_mask, -1e9)
    if _safe_bool(getattr(config, "talon_frame_balanced_selection", True)):
        top = _hybrid_global_frame_select(
            scores=scores,
            selected_mask=selected_mask,
            budget=budget,
            num_frames=num_frames,
            num_visual_tokens=num_visual_tokens,
            frame_importance=frame_importance,
            budget_mode=budget_mode,
            global_ratio=float(getattr(config, "talon_rescue_global_ratio", 0.85)),
            min_per_frame=0,
        )
    else:
        valid = int((scores > -1e8).sum().item())
        if valid <= 0:
            feat_dim = flat_features.shape[-1]
            return flat_features.new_zeros((0, feat_dim)), torch.empty((0,), dtype=torch.long, device=flat_features.device)
        top = torch.topk(scores, k=min(budget, valid), dim=0).indices
    if top.numel() == 0:
        feat_dim = flat_features.shape[-1]
        return flat_features.new_zeros((0, feat_dim)), torch.empty((0,), dtype=torch.long, device=flat_features.device)
    selected_mask[top] = True
    return flat_features[top], flat_indices[top]


class TalonCompressor:
    def __init__(self, config: FlashVidConfig):
        self.config = config

    def compress(
        self,
        video_features: torch.Tensor,
        cls_attention: torch.Tensor,
        question_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_frames, num_visual_tokens, _ = video_features.shape
        ratio = _resolve_retention_ratio(video_features, question_features, self.config)
        resolved_per_frame_target = _resolve_talon_target_tokens_per_frame(video_features, question_features, self.config)
        segment_lengths = _segment_lengths(video_features, self.config)
        global_indices = torch.arange(num_frames * num_visual_tokens, dtype=torch.long, device=video_features.device)
        global_grid = global_indices.view(num_frames, num_visual_tokens)

        all_tokens: List[torch.Tensor] = []
        all_indices: List[torch.Tensor] = []
        offset = 0
        for seg_len_tensor in segment_lengths:
            seg_len = int(seg_len_tensor.item())
            seg_tokens, seg_indices = self._compress_segment(
                segment_features=video_features[offset : offset + seg_len],
                segment_global_indices=global_grid[offset : offset + seg_len],
                cls_attention=cls_attention[offset : offset + seg_len],
                retention_ratio=ratio,
                question_features=question_features,
                resolved_per_frame_target=resolved_per_frame_target,
            )
            all_tokens.append(seg_tokens)
            all_indices.append(seg_indices)
            offset += seg_len

        final_tokens = torch.cat(all_tokens, dim=0) if all_tokens else video_features.new_zeros((0, video_features.shape[-1]))
        final_indices = torch.cat(all_indices, dim=0) if all_indices else torch.empty((0,), dtype=torch.long, device=video_features.device)
        if final_indices.numel() > 0:
            order = torch.argsort(final_indices)
            final_tokens = final_tokens[order]
            final_indices = final_indices[order]

        self.config.num_attn_div_tokens = None
        self.config.num_sttm_tokens = None
        self.config.vision_token_length = int(final_tokens.shape[0])
        self.config.llm_token_length = None
        self.config.visual_token_length = int(final_tokens.shape[0])
        return final_tokens, final_indices

    def _compress_segment(
        self,
        segment_features: torch.Tensor,
        segment_global_indices: torch.Tensor,
        cls_attention: torch.Tensor,
        retention_ratio: float,
        question_features: Optional[torch.Tensor],
        resolved_per_frame_target: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_frames, num_visual_tokens, feat_dim = segment_features.shape
        num_tokens = num_frames * num_visual_tokens
        device = segment_features.device
        if num_tokens == 0:
            return segment_features.new_zeros((0, feat_dim)), torch.empty((0,), dtype=torch.long, device=device)

        flat_features = segment_features.reshape(num_tokens, feat_dim)
        flat_attention = cls_attention.reshape(num_tokens).float()
        flat_indices = segment_global_indices.reshape(num_tokens)
        fused_scores, _ = _question_aware_scores(flat_features, flat_attention, question_features, self.config)
        fused_grid = fused_scores.view(num_frames, num_visual_tokens)

        target_budget = self._resolve_target_budget(
            num_frames=num_frames,
            num_visual_tokens=num_visual_tokens,
            num_tokens=num_tokens,
            retention_ratio=retention_ratio,
            resolved_per_frame_target=resolved_per_frame_target,
        )
        memory_budget = _resolve_memory_budget(num_tokens, target_budget, self.config)
        core_budget = max(1, target_budget - memory_budget)

        selected = torch.zeros((num_tokens,), dtype=torch.bool, device=device)
        core_tokens: List[torch.Tensor] = []
        core_indices: List[torch.Tensor] = []

        frame_importance = _frame_importance(segment_features, cls_attention, question_features, self.config)
        budget_mode = str(getattr(self.config, "talon_budget_mode", "uniform") or "uniform")

        passthrough_budget = self._resolve_passthrough_budget(core_budget)
        min_anchor_per_frame = max(0, int(getattr(self.config, "talon_min_anchor_per_frame", 2)))
        anchor_safety_ratio = min(max(float(getattr(self.config, "talon_anchor_safety_ratio", 0.20)), 0.0), 0.80)
        anchor_safety_budget = int(round(core_budget * anchor_safety_ratio))
        anchor_floor_budget = min(num_tokens, min_anchor_per_frame * num_frames)
        passthrough_budget = min(
            max(0, core_budget - 1),
            max(passthrough_budget, anchor_safety_budget, anchor_floor_budget),
        )
        if passthrough_budget > 0:
            passthrough_idx = self._select_passthrough_indices(
                fused_scores=fused_scores,
                raw_attention=flat_attention,
                frame_importance=frame_importance,
                budget=passthrough_budget,
                num_frames=num_frames,
                num_visual_tokens=num_visual_tokens,
                budget_mode=budget_mode,
            )
            selected[passthrough_idx] = True
            core_tokens.append(flat_features[passthrough_idx])
            core_indices.append(flat_indices[passthrough_idx])
        else:
            passthrough_idx = torch.empty((0,), dtype=torch.long, device=device)

        talon_budget = max(0, core_budget - int(passthrough_idx.numel()))
        per_frame_talon_budget = talon_budget / max(1, num_frames)
        background_max_ratio = min(max(float(getattr(self.config, "talon_background_max_ratio", 0.45)), 0.0), 1.0)
        background_rank_cap = int(math.floor(per_frame_talon_budget * background_max_ratio))
        max_rank_by_budget = max(0, min(talon_budget // max(1, num_frames), background_rank_cap))
        rank_cap = min(
            max_rank_by_budget,
            int(getattr(self.config, "talon_rank_max", 32)),
            num_visual_tokens,
        )

        transport = _transport_align(segment_features, self.config)
        basis, eigvals = _lowrank_basis(transport.aligned, rank_cap, self.config)
        rank_plan = self._select_rank_plan(
            segment_features=segment_features,
            transport=transport,
            basis=basis,
            eigvals=eigvals,
            selected=selected,
            talon_budget=talon_budget,
        )

        rank = rank_plan.rank
        reconstruction_source = rank_plan.reconstruction_source
        residual_scores = rank_plan.residual_scores
        output_mode = str(getattr(self.config, "talon_output_mode", "manifold") or "manifold").strip().lower()
        coefficient_output = output_mode in ("coefficient", "coeff", "strict")

        if rank > 0:
            bg_tokens, bg_indices = self._emit_background_tokens(
                segment_features=segment_features,
                segment_global_indices=segment_global_indices,
                transport=transport,
                basis=basis[:, :rank],
                reconstruction_source=reconstruction_source,
                fused_grid=fused_grid,
                selected=selected,
                coefficient_output=coefficient_output,
            )
            if bg_tokens.numel() > 0:
                core_tokens.append(bg_tokens)
                core_indices.append(bg_indices)

        used_after_background = int(selected.sum().item()) - int(passthrough_idx.numel())
        remaining_innovation = max(0, talon_budget - used_after_background)
        if remaining_innovation > 0:
            innovation_tokens, innovation_indices = self._emit_innovations(
                segment_features=segment_features,
                segment_global_indices=segment_global_indices,
                residual_scores=residual_scores,
                fused_scores=fused_scores,
                frame_importance=frame_importance,
                selected=selected,
                budget=remaining_innovation,
                budget_mode=budget_mode,
            )
            if innovation_tokens.numel() > 0:
                core_tokens.append(innovation_tokens)
                core_indices.append(innovation_indices)

        if core_tokens:
            core_token_tensor = torch.cat(core_tokens, dim=0)
            core_index_tensor = torch.cat(core_indices, dim=0)
        else:
            top1 = torch.topk(fused_scores, k=1, dim=0).indices
            selected[top1] = True
            core_token_tensor = flat_features[top1]
            core_index_tensor = flat_indices[top1]

        # Semantic rescue: reclaim a small set of dropped raw tokens with
        # high question/reconstruction significance. The budget is borrowed
        # from memory tokens to keep total output size stable.
        rescue_enabled = _safe_bool(getattr(self.config, "talon_rescue_enabled", True))
        if rescue_enabled and memory_budget > 0:
            rescue_ratio = max(0.0, float(getattr(self.config, "talon_rescue_ratio", 0.08)))
            proposed_rescue = int(round(target_budget * rescue_ratio))
            proposed_rescue = max(0, proposed_rescue)
            if _safe_bool(getattr(self.config, "talon_rescue_from_memory_only", True)):
                rescue_budget = min(memory_budget, proposed_rescue)
            else:
                rescue_budget = min(proposed_rescue, int((~selected).sum().item()))
            if rescue_budget > 0:
                rescue_tokens, rescue_indices = _emit_rescue_tokens(
                    flat_features=flat_features,
                    flat_indices=flat_indices,
                    fused_scores=fused_scores,
                    residual_scores=residual_scores,
                    frame_importance=frame_importance,
                    selected_mask=selected,
                    budget=rescue_budget,
                    num_frames=num_frames,
                    num_visual_tokens=num_visual_tokens,
                    budget_mode=budget_mode,
                    config=self.config,
                )
                if rescue_tokens.numel() > 0:
                    core_token_tensor = torch.cat([core_token_tensor, rescue_tokens], dim=0)
                    core_index_tensor = torch.cat([core_index_tensor, rescue_indices], dim=0)
                    memory_budget = max(0, memory_budget - int(rescue_tokens.shape[0]))

        dropped = torch.where(~selected)[0]
        mem_tokens, mem_local_idx = _build_memory_tokens(
            flat_features=flat_features,
            dropped_indices=dropped,
            residual_scores=residual_scores,
            memory_budget=memory_budget,
            num_frames=num_frames,
            num_visual_tokens=num_visual_tokens,
            frame_importance=frame_importance,
            budget_mode=budget_mode,
            frame_balanced=_safe_bool(getattr(self.config, "talon_frame_balanced_memory", True)),
            memory_mode=str(getattr(self.config, "talon_memory_mode", "raw") or "raw"),
        )
        if mem_tokens.numel() > 0:
            all_tokens = torch.cat([core_token_tensor, mem_tokens], dim=0)
            all_indices = torch.cat([core_index_tensor, flat_indices[mem_local_idx]], dim=0)
        else:
            all_tokens = core_token_tensor
            all_indices = core_index_tensor
        return all_tokens, all_indices

    def _resolve_target_budget(
        self,
        num_frames: int,
        num_visual_tokens: int,
        num_tokens: int,
        retention_ratio: float,
        resolved_per_frame_target: int = 0,
    ) -> int:
        per_frame_target = int(resolved_per_frame_target or getattr(self.config, "talon_target_tokens_per_frame", 0) or 0)
        if per_frame_target > 0:
            budget = num_frames * max(1, min(per_frame_target, num_visual_tokens))
        else:
            budget_scale = float(getattr(self.config, "talon_budget_scale", 0.60))
            budget_scale = min(max(budget_scale, 0.05), 1.50)
            effective_ratio = retention_ratio * float(getattr(self.config, "expansion", 1.0)) * budget_scale
            effective_ratio = min(1.0, max(0.005, effective_ratio))
            budget = int(math.ceil(num_tokens * effective_ratio))

        min_total = max(1, int(getattr(self.config, "talon_min_total_tokens", 1) or 1))
        return max(1, min(num_tokens, max(min_total, int(budget))))

    def _resolve_passthrough_budget(self, core_budget: int) -> int:
        ratio = min(max(float(getattr(self.config, "talon_passthrough_ratio", 0.15)), 0.0), 0.90)
        min_tokens = max(0, int(getattr(self.config, "talon_passthrough_min", 2)))
        max_passthrough = max(0, int(core_budget) - 1)
        budget = min(max_passthrough, max(0, int(round(core_budget * ratio))))
        if budget > 0:
            budget = max(min(min_tokens, max_passthrough), budget)
        return budget

    def _select_passthrough_indices(
        self,
        fused_scores: torch.Tensor,
        raw_attention: torch.Tensor,
        frame_importance: torch.Tensor,
        budget: int,
        num_frames: int,
        num_visual_tokens: int,
        budget_mode: str,
    ) -> torch.Tensor:
        budget = min(max(0, int(budget)), int(fused_scores.numel()))
        if budget <= 0:
            return torch.empty((0,), dtype=torch.long, device=fused_scores.device)
        if not _safe_bool(getattr(self.config, "talon_frame_balanced_selection", True)):
            return torch.topk(fused_scores, k=budget, dim=0).indices
        # Anchor safety first: keep a small number of per-frame high-attention anchors.
        attention_grid = _normalize_scores(raw_attention).view(num_frames, num_visual_tokens)
        min_anchor_per_frame = max(0, int(getattr(self.config, "talon_min_anchor_per_frame", 2)))
        anchor_budget = min(budget, num_frames * min_anchor_per_frame) if min_anchor_per_frame > 0 else 0
        anchor_selected = []
        if anchor_budget > 0:
            for t in range(num_frames):
                k_t = min(min_anchor_per_frame, num_visual_tokens)
                if k_t <= 0:
                    continue
                local = torch.topk(attention_grid[t], k=k_t, dim=0).indices
                anchor_selected.append(t * num_visual_tokens + local)
            if anchor_selected:
                anchor_selected = torch.cat(anchor_selected, dim=0).unique()
                if anchor_selected.numel() > anchor_budget:
                    anchor_scores = fused_scores[anchor_selected]
                    keep_order = torch.topk(anchor_scores, k=anchor_budget, dim=0).indices
                    anchor_selected = anchor_selected[keep_order]
            else:
                anchor_selected = torch.empty((0,), dtype=torch.long, device=fused_scores.device)
        else:
            anchor_selected = torch.empty((0,), dtype=torch.long, device=fused_scores.device)

        selected_mask = torch.zeros_like(fused_scores, dtype=torch.bool)
        if anchor_selected.numel() > 0:
            selected_mask[anchor_selected] = True
        remaining_budget = max(0, budget - int(anchor_selected.numel()))
        global_ratio = float(getattr(self.config, "talon_global_topk_ratio", 0.70))
        hybrid_selected = _hybrid_global_frame_select(
            scores=fused_scores.float(),
            selected_mask=selected_mask,
            budget=remaining_budget,
            num_frames=num_frames,
            num_visual_tokens=num_visual_tokens,
            frame_importance=frame_importance,
            budget_mode=budget_mode,
            global_ratio=global_ratio,
            min_per_frame=0,
        )
        if anchor_selected.numel() == 0:
            return hybrid_selected
        if hybrid_selected.numel() == 0:
            return anchor_selected
        return torch.cat([anchor_selected, hybrid_selected], dim=0).unique().to(dtype=torch.long)

    def _select_rank_plan(
        self,
        segment_features: torch.Tensor,
        transport: _TransportState,
        basis: torch.Tensor,
        eigvals: torch.Tensor,
        selected: torch.Tensor,
        talon_budget: int,
    ) -> _RankPlan:
        num_frames, num_visual_tokens, feat_dim = segment_features.shape
        max_rank = int(basis.shape[1])
        min_rank = max(0, int(getattr(self.config, "talon_rank_min", 2)))
        strategy = str(getattr(self.config, "talon_budget_strategy", "marginal") or "marginal").strip().lower()
        if strategy == "ratio":
            raw = int(round((talon_budget / max(1, num_frames)) * float(getattr(self.config, "talon_rank_ratio", 0.40))))
            candidates = [min(max(raw, min_rank if talon_budget >= min_rank * num_frames else 0), max_rank)]
        else:
            candidates = list(range(0, max_rank + 1))
            if max_rank >= min_rank and talon_budget >= min_rank * num_frames:
                candidates = [r for r in candidates if r == 0 or r >= min_rank]

        best_rank = 0
        best_score = None
        best_residual_scores = torch.norm(segment_features.float(), p=2, dim=-1).reshape(-1) ** 2
        best_reconstruction = torch.zeros_like(segment_features)
        spectral_weight = float(getattr(self.config, "talon_rd_spectral_weight", 1.0))
        innovation_weight = float(getattr(self.config, "talon_rd_innovation_weight", 1.0))

        if _safe_bool(getattr(self.config, "talon_fast_rank_plan", True)):
            # Rate-distortion proxy: choose the background rank from spectral tail
            # and raw-token innovation energy, then reconstruct only once.
            flat_energy = torch.norm(segment_features.float(), p=2, dim=-1).reshape(-1) ** 2
            flat_energy = flat_energy.masked_fill(selected, 0.0)
            sorted_energy = torch.sort(flat_energy, descending=True).values
            prefix = F.pad(torch.cumsum(sorted_energy, dim=0), (1, 0), value=0.0)
            total_energy = prefix[-1]

            for rank in candidates:
                rank_cost = int(rank) * num_frames
                if rank_cost > talon_budget:
                    continue
                innovation_budget = min(max(0, talon_budget - rank_cost), sorted_energy.numel())
                residual_tail_proxy = total_energy - prefix[innovation_budget]
                spectral_tail = eigvals[rank:].sum() if rank < eigvals.numel() else eigvals.new_tensor(0.0)
                score = spectral_weight * spectral_tail + innovation_weight * residual_tail_proxy
                if best_score is None or score < best_score:
                    best_score = score
                    best_rank = int(rank)

            best_reconstruction = self._reconstruct_source(transport, basis[:, :best_rank])
            residual = segment_features.float() - best_reconstruction.float()
            best_residual_scores = torch.norm(residual, p=2, dim=-1).reshape(-1) ** 2
            return _RankPlan(
                rank=best_rank,
                residual_scores=best_residual_scores,
                reconstruction_source=best_reconstruction,
            )

        for rank in candidates:
            rank_cost = int(rank) * num_frames
            if rank_cost > talon_budget:
                continue
            reconstruction = self._reconstruct_source(transport, basis[:, :rank])
            residual = segment_features.float() - reconstruction.float()
            residual_scores = torch.norm(residual, p=2, dim=-1).reshape(-1) ** 2
            candidate_scores = residual_scores.masked_fill(selected, -1.0)
            innovation_budget = max(0, talon_budget - rank_cost)
            if innovation_budget > 0 and (candidate_scores > 0).any():
                keep_k = min(innovation_budget, int((candidate_scores > 0).sum().item()))
                kept = torch.topk(candidate_scores, k=keep_k, dim=0).values.sum()
            else:
                kept = candidate_scores.new_tensor(0.0)
            residual_tail = residual_scores.masked_fill(selected, 0.0).sum() - kept
            spectral_tail = eigvals[rank:].sum() if rank < eigvals.numel() else eigvals.new_tensor(0.0)
            score = spectral_weight * spectral_tail + innovation_weight * residual_tail
            if best_score is None or score < best_score:
                best_score = score
                best_rank = int(rank)
                best_residual_scores = residual_scores
                best_reconstruction = reconstruction

        return _RankPlan(rank=best_rank, residual_scores=best_residual_scores, reconstruction_source=best_reconstruction)

    def _reconstruct_source(self, transport: _TransportState, basis: torch.Tensor) -> torch.Tensor:
        aligned = transport.aligned
        num_frames, num_visual_tokens, feat_dim = aligned.shape
        if basis.numel() == 0:
            return torch.zeros_like(aligned)
        source_recon = torch.zeros_like(aligned)
        for t in range(num_frames):
            coeff = torch.matmul(basis.transpose(0, 1), aligned[t])
            bg = torch.matmul(basis, coeff)
            source_recon[t] = bg[transport.source_to_aligned[t]]
        return source_recon

    def _emit_background_tokens(
        self,
        segment_features: torch.Tensor,
        segment_global_indices: torch.Tensor,
        transport: _TransportState,
        basis: torch.Tensor,
        reconstruction_source: torch.Tensor,
        fused_grid: torch.Tensor,
        selected: torch.Tensor,
        coefficient_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_frames, num_visual_tokens, feat_dim = segment_features.shape
        rank = int(basis.shape[1])
        if rank <= 0:
            return segment_features.new_zeros((0, feat_dim)), torch.empty((0,), dtype=torch.long, device=segment_features.device)

        blend = min(max(float(getattr(self.config, "talon_reconstruction_blend", 0.25)), 0.0), 1.0)
        anchor_weight = min(max(float(getattr(self.config, "talon_anchor_score_weight", 0.35)), 0.0), 1.0)
        tokens = []
        indices = []
        for t in range(num_frames):
            frame_offset = t * num_visual_tokens
            frame_selected = selected[frame_offset : frame_offset + num_visual_tokens]
            if coefficient_output:
                coeff = torch.matmul(basis.transpose(0, 1), transport.aligned[t])
                aligned_scores = _normalize_scores(torch.norm(reconstruction_source[t].float(), p=2, dim=-1))
                aligned_scores = (1.0 - anchor_weight) * aligned_scores + anchor_weight * fused_grid[t]
                if frame_selected.any():
                    aligned_scores = aligned_scores.masked_fill(frame_selected, -1e9)
                k = min(rank, int((~frame_selected).sum().item()))
                if k <= 0:
                    continue
                anchor_idx = torch.topk(aligned_scores, k=k, dim=0).indices
                tokens.append(coeff[:k])
                indices.append(segment_global_indices[t, anchor_idx])
                selected[frame_offset + anchor_idx] = True
            else:
                energy = _normalize_scores(torch.norm(reconstruction_source[t].float(), p=2, dim=-1))
                anchor_scores = (1.0 - anchor_weight) * energy + anchor_weight * fused_grid[t]
                if frame_selected.any():
                    anchor_scores = anchor_scores.masked_fill(frame_selected, -1e9)
                k = min(rank, int((~frame_selected).sum().item()))
                if k <= 0:
                    continue
                anchor_idx = torch.topk(anchor_scores, k=k, dim=0).indices
                raw = segment_features[t, anchor_idx]
                recon = reconstruction_source[t, anchor_idx]
                tokens.append((1.0 - blend) * raw + blend * recon)
                indices.append(segment_global_indices[t, anchor_idx])
                selected[frame_offset + anchor_idx] = True

        if not tokens:
            return segment_features.new_zeros((0, feat_dim)), torch.empty((0,), dtype=torch.long, device=segment_features.device)
        return torch.cat(tokens, dim=0), torch.cat(indices, dim=0)

    def _emit_innovations(
        self,
        segment_features: torch.Tensor,
        segment_global_indices: torch.Tensor,
        residual_scores: torch.Tensor,
        fused_scores: torch.Tensor,
        frame_importance: torch.Tensor,
        selected: torch.Tensor,
        budget: int,
        budget_mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_frames, num_visual_tokens, feat_dim = segment_features.shape
        residual_core = _normalize_scores(residual_scores)
        attention_weight = min(max(float(getattr(self.config, "talon_innovation_attention_weight", 0.45)), 0.0), 1.0)
        scores = (1.0 - attention_weight) * residual_core + attention_weight * fused_scores.float()
        if _safe_bool(getattr(self.config, "talon_use_question_innovation", True)):
            q_weight = min(max(float(getattr(self.config, "talon_innovation_qweight", 0.25)), 0.0), 1.0)
            scores = (1.0 - q_weight) * scores + q_weight * fused_scores.float()
        scores = scores.masked_fill(selected, -1e9)
        k = min(int(budget), int((~selected).sum().item()))
        if k <= 0:
            return segment_features.new_zeros((0, feat_dim)), torch.empty((0,), dtype=torch.long, device=segment_features.device)

        if _safe_bool(getattr(self.config, "talon_frame_balanced_selection", True)):
            top = _hybrid_global_frame_select(
                scores=scores.float(),
                selected_mask=selected,
                budget=k,
                num_frames=num_frames,
                num_visual_tokens=num_visual_tokens,
                frame_importance=frame_importance,
                budget_mode=budget_mode,
                global_ratio=float(getattr(self.config, "talon_global_topk_ratio", 0.70)),
                min_per_frame=0,
            )
        else:
            top = torch.topk(scores, k=k, dim=0).indices

        selected[top] = True
        flat_features = segment_features.reshape(num_frames * num_visual_tokens, feat_dim)
        flat_indices = segment_global_indices.reshape(num_frames * num_visual_tokens)
        return flat_features[top], flat_indices[top]


def talon_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
    question_features: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return TalonCompressor(flashvid_config).compress(
        video_features=video_features,
        cls_attention=cls_attention,
        question_features=question_features,
    )
