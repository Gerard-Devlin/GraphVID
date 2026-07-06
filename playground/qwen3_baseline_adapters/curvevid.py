from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F

from flashvid.configuration_flashvid import FlashVidConfig
from flashvid.fastgraphvid import (
    _cfg_float,
    _cfg_int,
    _density_score,
    _effective_ratio,
    _normalize,
    _record_fastgraph_metrics,
    _resolve_token_selection_method,
    _select_graphstm_medoids,
    _temporal_novelty,
)
from flashvid.utils import ALL_TOKEN_SELECTION_METHOD


def _frame_curvature(video_features: torch.Tensor) -> torch.Tensor:
    """Estimate temporal curvature from the frame-level visual trajectory."""
    frame_num = int(video_features.shape[0])
    device = video_features.device
    if frame_num <= 1:
        return torch.ones((frame_num,), dtype=torch.float32, device=device)

    frame_proto = video_features.float().mean(dim=1)
    frame_proto = F.normalize(frame_proto, p=2, dim=-1, eps=1e-6)

    if frame_num == 2:
        motion = (1.0 - (frame_proto[0] * frame_proto[1]).sum()).clamp(0.0, 2.0) * 0.5
        return torch.full((2,), float(motion.item()), dtype=torch.float32, device=device)

    prev_vec = F.normalize(frame_proto[1:-1] - frame_proto[:-2], p=2, dim=-1, eps=1e-6)
    next_vec = F.normalize(frame_proto[2:] - frame_proto[1:-1], p=2, dim=-1, eps=1e-6)
    middle = (1.0 - (prev_vec * next_vec).sum(dim=-1)).clamp(0.0, 2.0) * 0.5

    curvature = torch.zeros((frame_num,), dtype=torch.float32, device=device)
    curvature[1:-1] = middle
    curvature[0] = middle[0]
    curvature[-1] = middle[-1]
    return curvature


def _curvature_weights(curvature: torch.Tensor, config: FlashVidConfig) -> torch.Tensor:
    frame_num = int(curvature.numel())
    if frame_num <= 0:
        return curvature
    uniform = torch.full_like(curvature, 1.0 / max(1, frame_num))
    if frame_num == 1:
        return uniform

    mix = min(max(_cfg_float(config, "curvevid_mix", 0.65), 0.0), 1.0)
    temperature = max(_cfg_float(config, "curvevid_temperature", 0.70), 1e-3)
    curve = torch.nan_to_num(curvature.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if float(curve.max().item() - curve.min().item()) <= 1e-6:
        curve_weights = uniform
    else:
        curve = _normalize(curve)
        curve_weights = torch.softmax(curve / temperature, dim=0)
    weights = (1.0 - mix) * uniform + mix * curve_weights
    return weights / weights.sum().clamp_min(1e-6)


def _allocate_frame_budgets(
    weights: torch.Tensor,
    *,
    total_budget: int,
    min_per_frame: int,
    max_per_frame: int,
) -> torch.Tensor:
    frame_num = int(weights.numel())
    device = weights.device
    if frame_num <= 0:
        return torch.empty((0,), dtype=torch.long, device=device)

    min_per_frame = max(0, min(int(min_per_frame), int(max_per_frame)))
    max_per_frame = max(min_per_frame, int(max_per_frame))
    target = max(min_per_frame * frame_num, min(int(total_budget), max_per_frame * frame_num))

    budgets = torch.full((frame_num,), min_per_frame, dtype=torch.long, device=device)
    remaining = int(target - int(budgets.sum().item()))
    if remaining <= 0:
        return budgets

    scores = torch.nan_to_num(weights.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    if float(scores.sum().item()) <= 1e-6:
        scores = torch.ones_like(scores)

    while remaining > 0:
        capacity = max_per_frame - budgets
        active = torch.where(capacity > 0)[0]
        if active.numel() == 0:
            break
        active_scores = scores[active]
        if float(active_scores.sum().item()) <= 1e-6:
            active_scores = torch.ones_like(active_scores)
        raw = active_scores / active_scores.sum().clamp_min(1e-6) * float(remaining)
        inc = torch.floor(raw).long()
        inc = torch.minimum(inc, capacity[active])
        inc_sum = int(inc.sum().item())
        if inc_sum > 0:
            budgets[active] += inc
            remaining -= inc_sum
            continue

        fractional = raw - torch.floor(raw)
        order = torch.argsort(fractional, descending=True)
        for rel in order.tolist():
            if remaining <= 0:
                break
            pos = int(active[rel].item())
            if int(budgets[pos].item()) < max_per_frame:
                budgets[pos] += 1
                remaining -= 1
    return budgets


def _select_variable_ats(
    *,
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    frame_ats_budgets: torch.Tensor,
    config: FlashVidConfig,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    frame_num, frame_token_len, _ = video_features.shape
    device = video_features.device
    all_indices = torch.arange(frame_num * frame_token_len, dtype=torch.long, device=device).view(frame_num, frame_token_len)
    ats_mask = torch.zeros((frame_num, frame_token_len), dtype=torch.bool, device=device)
    keep_indices: list[torch.Tensor] = []

    selection_method = _resolve_token_selection_method(config)
    selector = ALL_TOKEN_SELECTION_METHOD[selection_method]
    needs_attention = "attn" in selection_method.value
    for frame_idx in range(frame_num):
        keep = max(0, min(frame_token_len, int(frame_ats_budgets[frame_idx].item())))
        if keep <= 0:
            continue
        kwargs = {"cls_attention": cls_attention[frame_idx : frame_idx + 1]} if needs_attention else {}
        _, local_idx = selector(
            features=video_features[frame_idx : frame_idx + 1],
            num_retained_tokens=keep,
            **kwargs,
        )
        local_idx = local_idx[0]
        ats_mask[frame_idx, local_idx] = True
        keep_indices.append(all_indices[frame_idx, local_idx])
    return ats_mask, keep_indices


def curvevid_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CurveVID: FastGraph-style pruning with curvature-aware frame budgets.

    The algorithm remains pruning-only: every emitted token is gathered from the
    original Qwen3 visual token grid. Curvature only reallocates the per-frame
    token budget before the ATS and residual GraphSTM branches.
    """
    frame_num, frame_token_len, feat_dim = video_features.shape
    device = video_features.device
    ratio = _effective_ratio(flashvid_config)
    ats_ratio = max(0.0, min(1.0, _cfg_float(flashvid_config, "fastgraph_ats_ratio", 0.60)))

    uniform_frame_budget = max(1, min(frame_token_len, int(frame_token_len * ratio)))
    total_budget = int(uniform_frame_budget * frame_num)
    min_per_frame = _cfg_int(flashvid_config, "curvevid_min_per_frame", 1)
    curvature = _frame_curvature(video_features)
    frame_weights = _curvature_weights(curvature, flashvid_config)
    frame_budgets = _allocate_frame_budgets(
        frame_weights,
        total_budget=total_budget,
        min_per_frame=min_per_frame,
        max_per_frame=frame_token_len,
    )
    frame_ats_budgets = torch.round(frame_budgets.float() * ats_ratio).long()
    frame_ats_budgets = torch.minimum(torch.maximum(frame_ats_budgets, torch.zeros_like(frame_ats_budgets)), frame_budgets)
    frame_graph_budgets = frame_budgets - frame_ats_budgets

    all_indices = torch.arange(frame_num * frame_token_len, dtype=torch.long, device=device).view(frame_num, frame_token_len)
    attn = cls_attention.float()
    ats_mask, keep_indices = _select_variable_ats(
        video_features=video_features,
        cls_attention=attn,
        frame_ats_budgets=frame_ats_budgets,
        config=flashvid_config,
    )

    residual_mask = ~ats_mask
    normed = F.normalize(video_features.float(), p=2, dim=-1, eps=1e-6)
    novelty = _temporal_novelty(
        normed,
        residual_mask,
        radius=max(0, _cfg_int(flashvid_config, "fastgraph_temporal_radius", 1)),
        temporal_skip=max(1, _cfg_int(flashvid_config, "fastgraph_temporal_skip", 1)),
        config=flashvid_config,
    )
    density = _density_score(video_features, residual_mask)
    attn_norm = _normalize(attn, residual_mask)

    attn_weight = _cfg_float(flashvid_config, "fastgraph_attn_weight", 0.55)
    novelty_weight = _cfg_float(flashvid_config, "fastgraph_novelty_weight", 0.30)
    density_weight = _cfg_float(flashvid_config, "fastgraph_density_weight", 0.15)
    quality = attn_weight * attn_norm + novelty_weight * novelty + density_weight * density

    graph_budget_mean = frame_graph_budgets.float().mean().clamp_min(1.0)
    frame_bias = (frame_graph_budgets.float() / graph_budget_mean).clamp(0.25, 4.0)
    quality = quality * frame_bias.unsqueeze(1)

    graphstm_target = int(frame_graph_budgets.sum().item())
    if graphstm_target > 0:
        _, graphstm_indices = _select_graphstm_medoids(
            video_features=video_features,
            normed=normed,
            candidate_mask=residual_mask,
            quality=quality,
            global_indices=all_indices,
            target_count=graphstm_target,
            config=flashvid_config,
        )
        if graphstm_indices.numel() > 0:
            keep_indices.append(graphstm_indices)

    if not keep_indices:
        hidden_states = video_features.reshape(-1, feat_dim)[:1]
        selected = torch.zeros((1,), dtype=torch.long, device=device)
    else:
        selected = torch.cat(keep_indices, dim=0)
        selected = torch.unique(selected, sorted=True)
        frame_idx = selected // frame_token_len
        token_idx = selected % frame_token_len
        hidden_states = video_features[frame_idx, token_idx]

    flashvid_config.vision_token_length = int(hidden_states.shape[0])
    flashvid_config.llm_token_length = None
    flashvid_config.visual_token_length = int(hidden_states.shape[0])
    setattr(flashvid_config, "last_fastgraph_ats_ratio", float(ats_ratio))
    setattr(flashvid_config, "last_fastgraph_frame_retain_num", float(frame_budgets.float().mean().item()))
    setattr(flashvid_config, "last_fastgraph_frame_ats_num", float(frame_ats_budgets.float().mean().item()))
    setattr(flashvid_config, "last_fastgraph_frame_graphstm_num", float(frame_graph_budgets.float().mean().item()))
    setattr(flashvid_config, "last_curvevid_curvature_mean", float(curvature.float().mean().item()))
    setattr(flashvid_config, "last_curvevid_curvature_max", float(curvature.float().max().item()))
    setattr(flashvid_config, "last_curvevid_frame_budget_min", float(frame_budgets.min().item()))
    setattr(flashvid_config, "last_curvevid_frame_budget_max", float(frame_budgets.max().item()))
    setattr(flashvid_config, "last_curvevid_temperature", float(_cfg_float(flashvid_config, "curvevid_temperature", 0.70)))
    setattr(flashvid_config, "last_curvevid_mix", float(_cfg_float(flashvid_config, "curvevid_mix", 0.65)))
    _record_fastgraph_metrics(
        flashvid_config,
        output_tokens=int(hidden_states.shape[0]),
        raw_tokens=int(frame_num * frame_token_len),
    )
    setattr(flashvid_config, "last_adapter_variant", "curvevid")
    return hidden_states, selected


__all__ = ["curvevid_compression"]
