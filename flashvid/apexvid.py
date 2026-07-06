from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from .configuration_flashvid import FlashVidConfig


DEFAULT_RETAIN_RATIO = 0.25
DEFAULT_MIN_K = 1
DEFAULT_BUDGET_TEMP = 0.7


def _cfg_float(config: FlashVidConfig, name: str, default: float) -> float:
    try:
        return float(getattr(config, name, default))
    except Exception:
        return float(default)


def _effective_ratio(config: FlashVidConfig) -> float:
    ratio = _cfg_float(config, "retention_ratio", DEFAULT_RETAIN_RATIO)
    ratio *= _cfg_float(config, "expansion", 1.0)
    if ratio <= 0.0:
        ratio = DEFAULT_RETAIN_RATIO
    return max(0.0, min(1.0, ratio))


def _compute_curvature(frame_reps: torch.Tensor) -> torch.Tensor:
    """Compute curvature over normalized frame representatives."""
    frame_count = int(frame_reps.shape[0])
    if frame_count <= 1:
        return torch.ones((frame_count,), device=frame_reps.device, dtype=torch.float32)

    v_in = frame_reps[1:-1] - frame_reps[:-2]
    v_out = frame_reps[2:] - frame_reps[1:-1]
    curv = 1.0 - F.cosine_similarity(v_in, v_out, dim=-1, eps=1e-6)
    ones = torch.ones((1,), device=frame_reps.device, dtype=curv.dtype)
    return torch.cat([ones, curv.float(), ones], dim=0)


def _allocate_budget_per_frame(
    curvature: torch.Tensor,
    total_budget: int,
    *,
    min_k: int,
    max_k: int,
) -> torch.Tensor:
    """Allocate per-frame budgets with deterministic rounding."""
    frame_count = int(curvature.shape[0])
    if frame_count <= 0:
        return torch.zeros((0,), device=curvature.device, dtype=torch.long)

    total_budget = int(max(0, min(int(total_budget), int(frame_count * max_k))))
    min_k = int(max(0, min(int(min_k), int(max_k))))
    min_k_eff = 0 if total_budget < frame_count * min_k else min_k

    base = torch.full((frame_count,), int(min_k_eff), device=curvature.device, dtype=torch.long)
    remaining = int(total_budget - int(base.sum().item()))
    if remaining <= 0:
        return base

    weights = curvature.float().clamp_min(0.0)
    if float(weights.sum().item()) <= 0.0:
        weights = torch.ones_like(weights)
    weights = weights / weights.sum().clamp_min(1e-6)

    raw = weights * float(remaining)
    extra = torch.floor(raw).to(torch.long)
    max_extra = max(0, max_k - min_k_eff)
    extra = torch.minimum(extra, torch.full_like(extra, int(max_extra)))
    alloc = base + extra

    remaining = int(total_budget - int(alloc.sum().item()))
    if remaining > 0:
        frac = raw - torch.floor(raw)
        frac = frac.masked_fill(alloc >= max_k, -1.0)
        for _ in range(remaining):
            idx = torch.argmax(frac)
            if float(frac[idx].item()) < 0.0:
                break
            alloc[idx] += 1
            if alloc[idx] >= max_k:
                frac[idx] = -1.0
    elif remaining < 0:
        over = -remaining
        order = torch.argsort(weights, descending=False)
        for idx in order:
            if over <= 0:
                break
            if alloc[idx] > min_k_eff:
                alloc[idx] -= 1
                over -= 1

    return alloc


def _curvature_spatial_select(
    frames: torch.Tensor,
    retain_ratio: float,
    *,
    min_k: int,
    budget_temp: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Curvature-aware temporal allocation plus coordinate-preserving spatial top-k."""
    frame_count, tokens_per_frame, dim = frames.shape
    device = frames.device
    if frame_count <= 0 or tokens_per_frame <= 0:
        empty_idx = torch.zeros((0,), device=device, dtype=torch.long)
        return frames.new_empty((0, dim)), empty_idx

    frame_reps = F.normalize(frames.float().mean(dim=1), dim=-1, eps=1e-6)
    curvature = _compute_curvature(frame_reps)
    weights = torch.softmax(curvature.float() / float(max(1e-6, budget_temp)), dim=0)

    total_tokens = int(frame_count * tokens_per_frame)
    total_budget = int(round(float(total_tokens) * float(retain_ratio)))
    total_budget = max(1, min(total_tokens, total_budget))

    min_k = int(max(0, min(int(min_k), int(tokens_per_frame))))
    if min_k > 0:
        total_budget = max(total_budget, int(frame_count * min_k))

    per_frame_budget = _allocate_budget_per_frame(
        weights,
        total_budget,
        min_k=min_k,
        max_k=tokens_per_frame,
    )

    tokens_out: list[torch.Tensor] = []
    keep_indices: list[torch.Tensor] = []
    for frame_idx in range(frame_count):
        k = int(per_frame_budget[frame_idx].item())
        if k <= 0:
            continue

        frame_tokens = frames[frame_idx]
        rep = frame_reps[frame_idx]
        sim = F.cosine_similarity(frame_tokens.float(), rep.unsqueeze(0), dim=-1, eps=1e-6)
        outlier = (1.0 - sim).float()
        norm = frame_tokens.float().norm(dim=-1)
        norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-6)
        score = outlier + norm

        topk_idx = torch.topk(score, k=k, largest=True, sorted=False).indices
        topk_idx, _ = torch.sort(topk_idx)
        tokens_out.append(frame_tokens.index_select(0, topk_idx))
        keep_indices.append(topk_idx + frame_idx * tokens_per_frame)

    if not tokens_out:
        empty_idx = torch.zeros((0,), device=device, dtype=torch.long)
        return frames.new_empty((0, dim)), empty_idx

    return torch.cat(tokens_out, dim=0), torch.cat(keep_indices, dim=0).to(torch.long)


def apexvid_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
    question_features: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Curvature-aware pruning with stable original-token ordering."""
    del cls_attention, question_features

    if video_features.ndim != 3:
        raise ValueError(f"expected video_features [T, HW, D], got shape={tuple(video_features.shape)}")

    frame_count, tokens_per_frame, _ = video_features.shape
    retain_ratio = _effective_ratio(flashvid_config)
    min_k = DEFAULT_MIN_K
    budget_temp = DEFAULT_BUDGET_TEMP

    hidden_states, selected = _curvature_spatial_select(
        video_features,
        retain_ratio,
        min_k=min_k,
        budget_temp=budget_temp,
    )

    hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=0.0, neginf=0.0).to(dtype=video_features.dtype)
    selected = selected.to(dtype=torch.long)

    raw_tokens = int(frame_count * tokens_per_frame)
    out_tokens = int(hidden_states.shape[0])
    flashvid_config.vision_token_length = out_tokens
    flashvid_config.llm_token_length = None
    flashvid_config.visual_token_length = out_tokens

    setattr(flashvid_config, "last_adapter_variant", "apexvid")
    setattr(flashvid_config, "last_adapter_output_tokens", float(out_tokens))
    setattr(flashvid_config, "last_adapter_raw_tokens", float(raw_tokens))
    setattr(flashvid_config, "last_apex_target_tokens", float(out_tokens))
    setattr(flashvid_config, "last_apex_evidence_ratio", 1.0)
    setattr(flashvid_config, "last_apex_event_ratio", 0.0)
    setattr(flashvid_config, "last_apex_memory_ratio", 0.0)
    setattr(flashvid_config, "last_apex_evidence_tokens", float(out_tokens))
    setattr(flashvid_config, "last_apex_event_tokens", 0.0)
    setattr(flashvid_config, "last_apex_memory_tokens", 0.0)
    return hidden_states, selected
