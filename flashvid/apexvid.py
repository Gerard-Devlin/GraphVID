from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .configuration_flashvid import FlashVidConfig
from .fastgraphvid import (
    _cfg_float,
    _cfg_int,
    _density_score,
    _grid_hw,
    _normalize,
    _resolve_token_selection_method,
    _select_graphstm_medoids,
    _temporal_novelty,
)
from .utils import ALL_TOKEN_SELECTION_METHOD


@dataclass
class _Bank:
    indices: torch.Tensor
    tokens: torch.Tensor
    scores: torch.Tensor
    bank_ids: torch.Tensor


def _effective_ratio(config: FlashVidConfig) -> float:
    ratio = float(getattr(config, "retention_ratio", 0.10))
    ratio *= float(getattr(config, "expansion", 1.0))
    return max(0.0, min(1.0, ratio))


def _question_score(video_features: torch.Tensor, question_features: Optional[torch.Tensor]) -> torch.Tensor:
    if question_features is None or question_features.numel() == 0:
        return torch.zeros(video_features.shape[:2], dtype=torch.float32, device=video_features.device)
    if int(question_features.shape[-1]) != int(video_features.shape[-1]):
        return torch.zeros(video_features.shape[:2], dtype=torch.float32, device=video_features.device)
    q = F.normalize(question_features.float(), p=2, dim=-1, eps=1e-6)
    q_center = F.normalize(q.mean(dim=0), p=2, dim=-1, eps=1e-6)
    tokens = F.normalize(video_features.float(), p=2, dim=-1, eps=1e-6)
    score = (tokens * q_center.view(1, 1, -1)).sum(dim=-1)
    return _normalize(score.clamp(min=-1.0, max=1.0))


def _question_difficulty(question_features: Optional[torch.Tensor]) -> float:
    if question_features is None or question_features.numel() == 0:
        return 0.35
    q = F.normalize(question_features.float(), p=2, dim=-1, eps=1e-6)
    q_center = F.normalize(q.mean(dim=0), p=2, dim=-1, eps=1e-6)
    dispersion = (1.0 - (q * q_center).sum(dim=-1).clamp(-1.0, 1.0)).clamp(0.0, 2.0) * 0.5
    length_score = min(1.0, float(q.shape[0]) / 40.0)
    return max(0.0, min(1.0, 0.55 * float(dispersion.mean().item()) + 0.45 * length_score))


def _frame_event_score(normed: torch.Tensor) -> torch.Tensor:
    frame_num = int(normed.shape[0])
    if frame_num <= 1:
        return torch.zeros((frame_num,), dtype=torch.float32, device=normed.device)

    centers = F.normalize(normed.mean(dim=1), p=2, dim=-1, eps=1e-6)
    delta = torch.zeros((frame_num,), dtype=torch.float32, device=normed.device)
    sim = (centers[:-1] * centers[1:]).sum(dim=-1).clamp(-1.0, 1.0)
    transition = ((1.0 - sim) * 0.5).clamp(0.0, 1.0)
    delta[1:] = transition
    delta[:-1] = torch.maximum(delta[:-1], transition)

    curvature = torch.zeros_like(delta)
    if frame_num > 2:
        prev_vec = F.normalize(centers[1:-1] - centers[:-2], p=2, dim=-1, eps=1e-6)
        next_vec = F.normalize(centers[2:] - centers[1:-1], p=2, dim=-1, eps=1e-6)
        turn = ((1.0 - (prev_vec * next_vec).sum(dim=-1).clamp(-1.0, 1.0)) * 0.5).clamp(0.0, 1.0)
        curvature[1:-1] = turn
    return _normalize(0.65 * delta + 0.35 * curvature)


def _router_ratios(
    *,
    config: FlashVidConfig,
    attn: torch.Tensor,
    frame_event: torch.Tensor,
    question_features: Optional[torch.Tensor],
) -> tuple[float, float, float]:
    base = torch.tensor(
        [
            max(0.0, _cfg_float(config, "apex_evidence_ratio", 0.45)),
            max(0.0, _cfg_float(config, "apex_event_ratio", 0.30)),
            max(0.0, _cfg_float(config, "apex_memory_ratio", 0.25)),
        ],
        dtype=torch.float32,
    )
    base = base / base.sum().clamp_min(1e-6)

    strength = max(0.0, min(1.0, _cfg_float(config, "apex_router_strength", 0.50)))
    if strength <= 0.0:
        return float(base[0]), float(base[1]), float(base[2])

    flat_attn = attn.float().reshape(-1)
    topk = min(max(1, int(flat_attn.numel() * 0.08)), flat_attn.numel())
    concentration = float(flat_attn.topk(k=topk, largest=True).values.sum().div(flat_attn.sum().clamp_min(1e-6)).item())
    event_level = float(frame_event.mean().item()) if frame_event.numel() else 0.0
    qdiff = _question_difficulty(question_features)

    adjustment = torch.tensor(
        [
            0.60 * concentration + 0.30 * (1.0 - event_level) + 0.10 * qdiff,
            0.65 * event_level + 0.25 * qdiff + 0.10 * (1.0 - concentration),
            0.55 * (1.0 - concentration) + 0.35 * (1.0 - event_level) + 0.10 * (1.0 - qdiff),
        ],
        dtype=torch.float32,
    )
    adjustment = adjustment / adjustment.sum().clamp_min(1e-6)
    mixed = (1.0 - strength) * base + strength * adjustment
    mixed = mixed / mixed.sum().clamp_min(1e-6)
    return float(mixed[0]), float(mixed[1]), float(mixed[2])


def _gather_raw(video_features: torch.Tensor, flat_indices: torch.Tensor) -> torch.Tensor:
    frame_token_len = int(video_features.shape[1])
    frames = flat_indices // frame_token_len
    tokens = flat_indices % frame_token_len
    return video_features[frames, tokens]


def _topk_flat(score: torch.Tensor, count: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    count = min(max(0, int(count)), int(score.numel()))
    if count <= 0:
        return torch.empty((0,), dtype=torch.long, device=score.device)
    values = score.reshape(-1)
    if mask is not None:
        values = values.masked_fill(~mask.reshape(-1), -1e9)
        valid = int(mask.sum().item())
        count = min(count, valid)
        if count <= 0:
            return torch.empty((0,), dtype=torch.long, device=score.device)
    return torch.topk(values, k=count, largest=True).indices.long()


def _evidence_bank(
    *,
    video_features: torch.Tensor,
    attn: torch.Tensor,
    evidence_score: torch.Tensor,
    target_count: int,
    all_indices: torch.Tensor,
    config: FlashVidConfig,
) -> _Bank:
    frame_num, frame_token_len, feat_dim = video_features.shape
    device = video_features.device
    candidate_count = max(target_count * 2, frame_num)
    per_frame = max(1, min(frame_token_len, int(math.ceil(candidate_count / max(1, frame_num)))))

    indices: list[torch.Tensor] = []
    if per_frame > 0:
        method = _resolve_token_selection_method(config)
        kwargs = {"cls_attention": attn} if "attn" in method.value else {}
        _, selected = ALL_TOKEN_SELECTION_METHOD[method](
            features=video_features,
            num_retained_tokens=per_frame,
            **kwargs,
        )
        indices.append(all_indices.gather(1, selected).reshape(-1))
    indices.append(_topk_flat(evidence_score, candidate_count))

    flat = torch.unique(torch.cat(indices), sorted=True) if indices else torch.empty((0,), dtype=torch.long, device=device)
    tokens = _gather_raw(video_features, flat) if flat.numel() else torch.empty((0, feat_dim), dtype=video_features.dtype, device=device)
    scores = evidence_score.reshape(-1)[flat] if flat.numel() else torch.empty((0,), dtype=torch.float32, device=device)
    bank_ids = torch.zeros((flat.numel(),), dtype=torch.long, device=device)
    return _Bank(flat, tokens, scores, bank_ids)


def _event_bank(
    *,
    video_features: torch.Tensor,
    event_score: torch.Tensor,
    target_count: int,
    all_indices: torch.Tensor,
) -> _Bank:
    del all_indices
    feat_dim = int(video_features.shape[-1])
    device = video_features.device
    flat = _topk_flat(event_score, max(target_count * 2, int(video_features.shape[0])))
    tokens = _gather_raw(video_features, flat) if flat.numel() else torch.empty((0, feat_dim), dtype=video_features.dtype, device=device)
    scores = event_score.reshape(-1)[flat] if flat.numel() else torch.empty((0,), dtype=torch.float32, device=device)
    bank_ids = torch.ones((flat.numel(),), dtype=torch.long, device=device)
    return _Bank(flat, tokens, scores, bank_ids)


def _fuse_memory_tokens(
    *,
    video_features: torch.Tensor,
    medoid_indices: torch.Tensor,
    memory_score: torch.Tensor,
    config: FlashVidConfig,
) -> torch.Tensor:
    frame_num, frame_token_len, feat_dim = video_features.shape
    if medoid_indices.numel() == 0:
        return torch.empty((0, feat_dim), dtype=video_features.dtype, device=video_features.device)

    h, w = _grid_hw(frame_token_len, config)
    spatial_radius = max(0, _cfg_int(config, "fastgraph_temporal_radius", 1))
    temporal_radius = max(1, _cfg_int(config, "fastgraph_temporal_skip", 1))
    temperature = max(1e-4, _cfg_float(config, "apex_summary_temperature", 0.07))
    out = []

    for flat in medoid_indices.tolist():
        frame = int(flat // frame_token_len)
        token = int(flat % frame_token_len)
        row, col = divmod(token, w)
        positions: list[tuple[int, int]] = []
        for ff in range(max(0, frame - temporal_radius), min(frame_num, frame + temporal_radius + 1)):
            for rr in range(max(0, row - spatial_radius), min(h, row + spatial_radius + 1)):
                for cc in range(max(0, col - spatial_radius), min(w, col + spatial_radius + 1)):
                    pos = rr * w + cc
                    if pos < frame_token_len:
                        positions.append((ff, pos))
        if not positions:
            out.append(video_features[frame, token])
            continue
        frame_ids = torch.tensor([p[0] for p in positions], dtype=torch.long, device=video_features.device)
        token_ids = torch.tensor([p[1] for p in positions], dtype=torch.long, device=video_features.device)
        local_tokens = video_features[frame_ids, token_ids]
        local_scores = memory_score[frame_ids, token_ids].float()
        weights = torch.softmax(local_scores / temperature, dim=0).to(local_tokens.dtype)
        out.append((local_tokens * weights.unsqueeze(-1)).sum(dim=0))
    return torch.stack(out, dim=0)


def _memory_bank(
    *,
    video_features: torch.Tensor,
    normed: torch.Tensor,
    memory_score: torch.Tensor,
    target_count: int,
    all_indices: torch.Tensor,
    config: FlashVidConfig,
) -> _Bank:
    feat_dim = int(video_features.shape[-1])
    device = video_features.device
    if target_count <= 0:
        empty_idx = torch.empty((0,), dtype=torch.long, device=device)
        empty_tok = torch.empty((0, feat_dim), dtype=video_features.dtype, device=device)
        empty_score = torch.empty((0,), dtype=torch.float32, device=device)
        return _Bank(empty_idx, empty_tok, empty_score, empty_idx)

    candidate_mask = torch.ones(video_features.shape[:2], dtype=torch.bool, device=device)
    _, medoid_indices = _select_graphstm_medoids(
        video_features=video_features,
        normed=normed,
        candidate_mask=candidate_mask,
        quality=memory_score,
        global_indices=all_indices,
        target_count=max(target_count * 2, target_count),
        config=config,
    )
    medoid_indices = torch.unique(medoid_indices, sorted=True)
    tokens = _fuse_memory_tokens(
        video_features=video_features,
        medoid_indices=medoid_indices,
        memory_score=memory_score,
        config=config,
    )
    scores = memory_score.reshape(-1)[medoid_indices] if medoid_indices.numel() else torch.empty((0,), dtype=torch.float32, device=device)
    bank_ids = torch.full((medoid_indices.numel(),), 2, dtype=torch.long, device=device)
    return _Bank(medoid_indices, tokens, scores, bank_ids)


def _merge_banks(banks: list[_Bank], feat_dim: int, device: torch.device, dtype: torch.dtype) -> _Bank:
    items: dict[int, tuple[float, torch.Tensor, float, int]] = {}
    priority = {0: 0.04, 1: 0.03, 2: 0.00}
    for bank in banks:
        for idx, token, score, bank_id in zip(bank.indices.tolist(), bank.tokens, bank.scores.tolist(), bank.bank_ids.tolist()):
            key = int(idx)
            adjusted = float(score) + priority.get(int(bank_id), 0.0)
            old = items.get(key)
            if old is None or adjusted > old[0]:
                items[key] = (adjusted, token, float(score), int(bank_id))
    if not items:
        empty_idx = torch.empty((0,), dtype=torch.long, device=device)
        empty_tok = torch.empty((0, feat_dim), dtype=dtype, device=device)
        empty_score = torch.empty((0,), dtype=torch.float32, device=device)
        return _Bank(empty_idx, empty_tok, empty_score, empty_idx)

    keys = list(items.keys())
    indices = torch.tensor(keys, dtype=torch.long, device=device)
    tokens = torch.stack([items[k][1] for k in keys], dim=0).to(dtype=dtype)
    scores = torch.tensor([items[k][2] for k in keys], dtype=torch.float32, device=device)
    bank_ids = torch.tensor([items[k][3] for k in keys], dtype=torch.long, device=device)
    return _Bank(indices, tokens, scores, bank_ids)


def _arbitrate(
    *,
    candidates: _Bank,
    target_total: int,
    frame_token_len: int,
    frame_num: int,
    ratios: tuple[float, float, float],
    config: FlashVidConfig,
    fallback_score: torch.Tensor,
    video_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = video_features.device
    feat_dim = int(video_features.shape[-1])
    target_total = min(max(1, int(target_total)), frame_num * frame_token_len)
    if candidates.indices.numel() == 0:
        flat = _topk_flat(fallback_score, target_total)
        return _gather_raw(video_features, flat), flat, torch.zeros((flat.numel(),), dtype=torch.long, device=device)

    selected_mask = torch.zeros((candidates.indices.numel(),), dtype=torch.bool, device=device)
    frame_ids = candidates.indices // frame_token_len
    bank_ids = candidates.bank_ids
    frame_retain = max(1, int(math.ceil(target_total / max(1, frame_num))))
    floor_ratio = max(0.0, min(1.0, _cfg_float(config, "apex_frame_floor_ratio", 0.35)))
    frame_floor = min(frame_retain, int(math.floor(frame_retain * floor_ratio)))
    if target_total >= frame_num:
        frame_floor = max(1, frame_floor)

    chosen: list[int] = []
    if frame_floor > 0:
        for frame in range(frame_num):
            positions = torch.where(frame_ids == frame)[0]
            if positions.numel() == 0:
                continue
            k = min(frame_floor, int(positions.numel()), target_total - len(chosen))
            if k <= 0:
                break
            local = positions[torch.topk(candidates.scores[positions], k=k, largest=True).indices]
            selected_mask[local] = True
            chosen.extend(local.tolist())

    remaining_budget = target_total - int(selected_mask.sum().item())
    if remaining_budget > 0:
        remaining = torch.where(~selected_mask)[0]
        if remaining.numel() > 0:
            frame_counts = torch.bincount(frame_ids[selected_mask], minlength=frame_num).float()
            bank_counts = torch.bincount(bank_ids[selected_mask], minlength=3).float()
            bank_quota = torch.tensor(ratios, dtype=torch.float32, device=device) * float(target_total)

            frame_bonus = 0.08 / (1.0 + frame_counts[frame_ids[remaining]])
            bank_deficit = (bank_quota[bank_ids[remaining]] - bank_counts[bank_ids[remaining]]).clamp(min=-target_total, max=target_total)
            bank_bonus = 0.04 * bank_deficit / max(1.0, float(target_total))
            frame_redundancy = 0.03 * frame_counts[frame_ids[remaining]] / max(1.0, float(frame_retain))
            adjusted = candidates.scores[remaining] + frame_bonus + bank_bonus - frame_redundancy
            k = min(remaining_budget, int(remaining.numel()))
            selected_mask[remaining[torch.topk(adjusted, k=k, largest=True).indices]] = True

    selected_indices = candidates.indices[selected_mask]
    selected_tokens = candidates.tokens[selected_mask]
    selected_banks = candidates.bank_ids[selected_mask]

    if selected_indices.numel() < target_total:
        missing = target_total - int(selected_indices.numel())
        already = torch.zeros((frame_num * frame_token_len,), dtype=torch.bool, device=device)
        already[selected_indices] = True
        extra = _topk_flat(fallback_score, missing, mask=~already.view(frame_num, frame_token_len))
        if extra.numel() > 0:
            selected_indices = torch.cat([selected_indices, extra], dim=0)
            selected_tokens = torch.cat([selected_tokens, _gather_raw(video_features, extra)], dim=0)
            selected_banks = torch.cat([selected_banks, torch.zeros((extra.numel(),), dtype=torch.long, device=device)], dim=0)

    if selected_indices.numel() == 0:
        selected_indices = torch.zeros((1,), dtype=torch.long, device=device)
        selected_tokens = video_features.reshape(-1, feat_dim)[:1]
        selected_banks = torch.zeros((1,), dtype=torch.long, device=device)

    order = torch.argsort(selected_indices)
    return selected_tokens[order], selected_indices[order], selected_banks[order]


def apexvid_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
    question_features: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    frame_num, frame_token_len, feat_dim = video_features.shape
    device = video_features.device
    ratio = _effective_ratio(flashvid_config)
    frame_retain = max(1, min(frame_token_len, int(math.ceil(frame_token_len * ratio))))
    target_total = min(frame_num * frame_token_len, max(1, frame_retain * frame_num))

    all_indices = torch.arange(frame_num * frame_token_len, dtype=torch.long, device=device).view(frame_num, frame_token_len)
    attn = torch.nan_to_num(cls_attention.float(), nan=0.0, posinf=0.0, neginf=0.0)
    normed = F.normalize(video_features.float(), p=2, dim=-1, eps=1e-6)
    mask = torch.ones((frame_num, frame_token_len), dtype=torch.bool, device=device)

    attn_norm = _normalize(attn, mask)
    novelty = _temporal_novelty(
        normed,
        mask,
        radius=max(0, _cfg_int(flashvid_config, "fastgraph_temporal_radius", 1)),
        temporal_skip=max(1, _cfg_int(flashvid_config, "fastgraph_temporal_skip", 1)),
        config=flashvid_config,
    )
    density = _density_score(video_features, mask)
    q_score = _question_score(video_features, question_features)
    q_weight = max(0.0, min(1.0, _cfg_float(flashvid_config, "apex_question_weight", 0.20)))
    frame_event = _frame_event_score(normed)
    frame_event_map = frame_event.view(frame_num, 1).expand(frame_num, frame_token_len)

    evidence_ratio, event_ratio, memory_ratio = _router_ratios(
        config=flashvid_config,
        attn=attn_norm,
        frame_event=frame_event,
        question_features=question_features,
    )

    evidence_score = _normalize((0.58 - q_weight * 0.20) * attn_norm + 0.20 * density + 0.12 * novelty + q_weight * q_score)
    event_score = _normalize(0.38 * novelty + 0.30 * frame_event_map + 0.17 * attn_norm + q_weight * q_score)
    memory_score = _normalize(0.35 * density + 0.25 * (1.0 - novelty) + 0.20 * attn_norm + 0.10 * frame_event_map + q_weight * q_score)
    fallback_score = _normalize(0.42 * evidence_score + 0.34 * event_score + 0.24 * memory_score)

    evidence_target = max(1, int(round(target_total * evidence_ratio)))
    event_target = max(1, int(round(target_total * event_ratio)))
    memory_target = max(0, target_total - evidence_target - event_target)
    if memory_target <= 0 and memory_ratio > 0.0 and target_total >= 3:
        memory_target = 1
        evidence_target = max(1, evidence_target - 1)

    banks = [
        _evidence_bank(
            video_features=video_features,
            attn=attn,
            evidence_score=evidence_score,
            target_count=evidence_target,
            all_indices=all_indices,
            config=flashvid_config,
        ),
        _event_bank(
            video_features=video_features,
            event_score=event_score,
            target_count=event_target,
            all_indices=all_indices,
        ),
        _memory_bank(
            video_features=video_features,
            normed=normed,
            memory_score=memory_score,
            target_count=memory_target,
            all_indices=all_indices,
            config=flashvid_config,
        ),
    ]
    candidates = _merge_banks(banks, feat_dim, device, video_features.dtype)
    hidden_states, selected, selected_banks = _arbitrate(
        candidates=candidates,
        target_total=target_total,
        frame_token_len=frame_token_len,
        frame_num=frame_num,
        ratios=(evidence_ratio, event_ratio, memory_ratio),
        config=flashvid_config,
        fallback_score=fallback_score,
        video_features=video_features,
    )

    hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=0.0, neginf=0.0).to(dtype=video_features.dtype)
    flashvid_config.vision_token_length = int(hidden_states.shape[0])
    flashvid_config.llm_token_length = None
    flashvid_config.visual_token_length = int(hidden_states.shape[0])
    setattr(flashvid_config, "last_adapter_variant", "apexvid")
    setattr(flashvid_config, "last_adapter_output_tokens", float(hidden_states.shape[0]))
    setattr(flashvid_config, "last_adapter_raw_tokens", float(frame_num * frame_token_len))
    setattr(flashvid_config, "last_apex_target_tokens", float(target_total))
    setattr(flashvid_config, "last_apex_evidence_ratio", float(evidence_ratio))
    setattr(flashvid_config, "last_apex_event_ratio", float(event_ratio))
    setattr(flashvid_config, "last_apex_memory_ratio", float(memory_ratio))
    setattr(flashvid_config, "last_apex_evidence_tokens", float((selected_banks == 0).sum().item()))
    setattr(flashvid_config, "last_apex_event_tokens", float((selected_banks == 1).sum().item()))
    setattr(flashvid_config, "last_apex_memory_tokens", float((selected_banks == 2).sum().item()))
    return hidden_states, selected
