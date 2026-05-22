from __future__ import annotations

from typing import List, Tuple

import math
import torch
from torch.nn import functional as F

from .cats import _cats_importance, _cats_sink_tstm, _flashvid_adts_selection
from .configuration_flashvid import FlashVidConfig


def _cfg_float(config: FlashVidConfig, name: str, default: float) -> float:
    value = getattr(config, name, None)
    if value is None:
        return float(default)
    return float(value)


def _cfg_int(config: FlashVidConfig, name: str, default: int) -> int:
    value = getattr(config, name, None)
    if value is None:
        return int(default)
    return int(value)


def _reset_hedge_metrics(config: FlashVidConfig) -> None:
    defaults = {
        "last_hedge_selected_adts": 0.0,
        "last_hedge_residual_budget": 0.0,
        "last_hedge_stable_candidates": 0.0,
        "last_hedge_evidence_candidates": 0.0,
        "last_hedge_stable_selected": 0.0,
        "last_hedge_evidence_selected": 0.0,
        "last_hedge_final_tokens": 0.0,
        "last_hedge_stable_floor_ratio": 0.0,
        "last_hedge_diversity_weight": 0.0,
    }
    for key, value in defaults.items():
        setattr(config, key, value)
    setattr(config, "_hedge_segments", 0.0)
    setattr(config, "_hedge_selected_sum", 0.0)
    setattr(config, "_hedge_budget_sum", 0.0)
    setattr(config, "_hedge_stable_cand_sum", 0.0)
    setattr(config, "_hedge_evidence_cand_sum", 0.0)
    setattr(config, "_hedge_stable_sel_sum", 0.0)
    setattr(config, "_hedge_evidence_sel_sum", 0.0)
    setattr(config, "_hedge_final_sum", 0.0)


def _accumulate_hedge_metrics(
    config: FlashVidConfig,
    *,
    selected_adts: int,
    residual_budget: int,
    stable_candidates: int,
    evidence_candidates: int,
    stable_selected: int,
    evidence_selected: int,
    final_tokens: int,
) -> None:
    if not hasattr(config, "_hedge_segments"):
        _reset_hedge_metrics(config)
    setattr(config, "_hedge_segments", float(getattr(config, "_hedge_segments", 0.0)) + 1.0)
    for attr, value in (
        ("_hedge_selected_sum", selected_adts),
        ("_hedge_budget_sum", residual_budget),
        ("_hedge_stable_cand_sum", stable_candidates),
        ("_hedge_evidence_cand_sum", evidence_candidates),
        ("_hedge_stable_sel_sum", stable_selected),
        ("_hedge_evidence_sel_sum", evidence_selected),
        ("_hedge_final_sum", final_tokens),
    ):
        setattr(config, attr, float(getattr(config, attr, 0.0)) + float(value))
    segments = max(1.0, float(getattr(config, "_hedge_segments", 1.0)))
    setattr(config, "last_hedge_selected_adts", float(getattr(config, "_hedge_selected_sum", 0.0)) / segments)
    setattr(config, "last_hedge_residual_budget", float(getattr(config, "_hedge_budget_sum", 0.0)) / segments)
    setattr(config, "last_hedge_stable_candidates", float(getattr(config, "_hedge_stable_cand_sum", 0.0)) / segments)
    setattr(config, "last_hedge_evidence_candidates", float(getattr(config, "_hedge_evidence_cand_sum", 0.0)) / segments)
    setattr(config, "last_hedge_stable_selected", float(getattr(config, "_hedge_stable_sel_sum", 0.0)) / segments)
    setattr(config, "last_hedge_evidence_selected", float(getattr(config, "_hedge_evidence_sel_sum", 0.0)) / segments)
    setattr(config, "last_hedge_final_tokens", float(getattr(config, "_hedge_final_sum", 0.0)) / segments)
    setattr(config, "last_hedge_stable_floor_ratio", _cfg_float(config, "hedge_stable_floor_ratio", 0.85))
    setattr(config, "last_hedge_diversity_weight", _cfg_float(config, "hedge_diversity_weight", 0.04))


def _empty_tokens(
    feat_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty((0, feat_dim), dtype=dtype, device=device),
        torch.empty((0,), dtype=torch.long, device=device),
        torch.empty((0,), dtype=torch.float32, device=device),
    )


def _token_score_map(features: torch.Tensor, cls_attention: torch.Tensor) -> torch.Tensor:
    """Score original token positions with attention, event relevance, motion, and detail."""
    normed = F.normalize(features.float(), p=2, dim=-1, eps=1.0e-6)
    importance = _cats_importance(features, cls_attention)
    frame_proto = F.normalize(normed.mean(dim=1), p=2, dim=-1, eps=1.0e-6)
    detail = (1.0 - torch.sum(normed * frame_proto.unsqueeze(1), dim=-1)).clamp(0.0, 2.0) * 0.5

    motion = torch.zeros(features.shape[0], dtype=torch.float32, device=features.device)
    if features.shape[0] > 1:
        motion[1:] = (1.0 - torch.sum(frame_proto[1:] * frame_proto[:-1], dim=-1)).clamp(0.0, 2.0) * 0.5
    motion = motion.unsqueeze(1).expand_as(importance)
    score = 0.60 * importance + 0.20 * motion + 0.20 * detail
    return torch.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def _scores_for_globals(
    segment_features: torch.Tensor,
    segment_global_indices: torch.Tensor,
    cls_attention: torch.Tensor,
    candidate_global_indices: torch.Tensor,
    bias: float,
) -> torch.Tensor:
    if candidate_global_indices.numel() == 0:
        return torch.empty((0,), dtype=torch.float32, device=segment_features.device)
    num_frames, num_visual_tokens, _ = segment_features.shape
    base_frame = int((segment_global_indices[0, 0] // max(1, num_visual_tokens)).item())
    frame_ids = (candidate_global_indices // max(1, num_visual_tokens)) - base_frame
    token_ids = candidate_global_indices % max(1, num_visual_tokens)
    score_map = _token_score_map(segment_features, cls_attention)
    valid = (frame_ids >= 0) & (frame_ids < num_frames) & (token_ids >= 0) & (token_ids < num_visual_tokens)
    scores = torch.zeros((candidate_global_indices.shape[0],), dtype=torch.float32, device=segment_features.device)
    if valid.any():
        scores[valid] = score_map[frame_ids[valid].long(), token_ids[valid].long()]
    return scores + float(bias)


def _spatial_merge_lists(
    token_lists: List[torch.Tensor],
    global_lists: List[torch.Tensor],
    target_total: int,
    feat_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply FlashVID-style per-frame DPC merging to a candidate list."""
    from .utils import dpc_knn

    if target_total <= 0:
        tokens, globals_, _ = _empty_tokens(feat_dim, dtype, device)
        return tokens, globals_
    current_total = sum(int(tokens.shape[0]) for tokens in token_lists)
    if current_total <= 0:
        tokens, globals_, _ = _empty_tokens(feat_dim, dtype, device)
        return tokens, globals_
    if current_total <= target_total:
        return torch.cat(token_lists, dim=0), torch.cat(global_lists, dim=0)

    ratio = float(target_total) / float(max(1, current_total))
    num_frames = len(token_lists)
    max_tokens = max([int(tokens.shape[0]) for tokens in token_lists] + [1])
    batched = torch.zeros((num_frames, max_tokens, feat_dim), dtype=dtype, device=device)
    valid_mask = torch.zeros((num_frames, max_tokens), dtype=torch.bool, device=device)
    cluster_counts: list[int] = []
    k_list: list[int] = []
    for frame_idx, tokens in enumerate(token_lists):
        n = int(tokens.shape[0])
        if n > 0:
            batched[frame_idx, :n] = tokens
            valid_mask[frame_idx, :n] = True
        clusters = min(n, int(math.ceil(n * ratio)))
        cluster_counts.append(clusters)
        k_list.append(max(1, min(clusters, 7)) if clusters > 0 else 1)

    lost = int(target_total) - sum(cluster_counts)
    if lost > 0:
        # Restore any budget lost to ceil/min rounding, preferring frames with more candidates.
        order = sorted(range(num_frames), key=lambda i: int(token_lists[i].shape[0]), reverse=True)
        cursor = 0
        while lost > 0 and cursor < len(order) * max_tokens:
            frame_idx = order[cursor % len(order)]
            if cluster_counts[frame_idx] < int(token_lists[frame_idx].shape[0]):
                cluster_counts[frame_idx] += 1
                k_list[frame_idx] = max(1, min(cluster_counts[frame_idx], 7))
                lost -= 1
            cursor += 1
    elif lost < 0:
        order = sorted(range(num_frames), key=lambda i: cluster_counts[i], reverse=True)
        cursor = 0
        while lost < 0 and cursor < len(order) * max_tokens:
            frame_idx = order[cursor % len(order)]
            if cluster_counts[frame_idx] > 0:
                cluster_counts[frame_idx] -= 1
                k_list[frame_idx] = max(1, min(cluster_counts[frame_idx], 7)) if cluster_counts[frame_idx] > 0 else 1
                lost += 1
            cursor += 1

    cluster_indices_list, center_indices_list = dpc_knn(
        features=batched,
        num_clusters=cluster_counts,
        k=k_list,
        valid_token_mask=valid_mask,
    )

    merged_tokens: list[torch.Tensor] = []
    merged_globals: list[torch.Tensor] = []
    for frame_idx, (tokens, globals_) in enumerate(zip(token_lists, global_lists)):
        n = int(tokens.shape[0])
        clusters = int(cluster_counts[frame_idx])
        if n == 0 or clusters <= 0:
            continue
        cluster_indices = cluster_indices_list[frame_idx][:n]
        centers = center_indices_list[frame_idx].clamp(min=0, max=max(0, n - 1))
        if centers.numel() < clusters:
            pad = centers[-1].repeat(clusters - centers.numel()) if centers.numel() else torch.zeros((clusters,), dtype=torch.long, device=device)
            centers = torch.cat([centers, pad], dim=0)
        elif centers.numel() > clusters:
            centers = centers[:clusters]
        aggregated = torch.zeros((clusters, feat_dim), dtype=dtype, device=device)
        aggregated.scatter_add_(0, cluster_indices.unsqueeze(-1).expand(-1, feat_dim), tokens)
        counts = torch.bincount(cluster_indices, minlength=clusters).unsqueeze(-1).to(dtype)
        merged_tokens.append(aggregated / counts.clamp_min(1))
        merged_globals.append(globals_[centers])
    if not merged_tokens:
        tokens, globals_, _ = _empty_tokens(feat_dim, dtype, device)
        return tokens, globals_
    return torch.cat(merged_tokens, dim=0), torch.cat(merged_globals, dim=0)


def _deduplicate_candidates(
    features: torch.Tensor,
    globals_: torch.Tensor,
    scores: torch.Tensor,
    sources: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if globals_.numel() <= 1:
        return features, globals_, scores, sources
    best: dict[int, int] = {}
    scores_cpu = scores.detach().cpu()
    sources_cpu = sources.detach().cpu()
    for idx, global_idx in enumerate(globals_.detach().cpu().tolist()):
        prev = best.get(int(global_idx))
        if prev is None:
            best[int(global_idx)] = idx
            continue
        # Prefer higher score; stable source wins ties.
        cur_key = (float(scores_cpu[idx]), int(sources_cpu[idx] == 0))
        prev_key = (float(scores_cpu[prev]), int(sources_cpu[prev] == 0))
        if cur_key > prev_key:
            best[int(global_idx)] = idx
    keep = torch.tensor(sorted(best.values()), dtype=torch.long, device=features.device)
    return features[keep], globals_[keep], scores[keep], sources[keep]


def _mmr_select(
    features: torch.Tensor,
    globals_: torch.Tensor,
    scores: torch.Tensor,
    sources: torch.Tensor,
    target: int,
    *,
    diversity_weight: float,
    max_candidates: int,
    preselected_features: torch.Tensor | None = None,
    blocked_globals: set[int] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if target <= 0 or features.numel() == 0:
        empty_f, empty_g, _ = _empty_tokens(features.shape[-1], features.dtype, features.device)
        return empty_f, empty_g, torch.empty((0,), dtype=torch.long, device=features.device)
    blocked_globals = blocked_globals or set()
    if blocked_globals:
        keep_mask = torch.tensor(
            [int(x) not in blocked_globals for x in globals_.detach().cpu().tolist()],
            dtype=torch.bool,
            device=features.device,
        )
        features = features[keep_mask]
        globals_ = globals_[keep_mask]
        scores = scores[keep_mask]
        sources = sources[keep_mask]
    if features.numel() == 0:
        empty_f, empty_g, _ = _empty_tokens(features.shape[-1] if features.ndim == 2 else 0, features.dtype, features.device)
        return empty_f, empty_g, torch.empty((0,), dtype=torch.long, device=features.device)

    features, globals_, scores, sources = _deduplicate_candidates(features, globals_, scores, sources)
    if features.shape[0] > max_candidates:
        top = torch.topk(scores, k=max_candidates, largest=True).indices
        features = features[top]
        globals_ = globals_[top]
        scores = scores[top]
        sources = sources[top]

    target = min(int(target), int(features.shape[0]))
    normed = F.normalize(features.float(), p=2, dim=-1, eps=1.0e-6)
    max_sim = torch.zeros((features.shape[0],), dtype=torch.float32, device=features.device)
    if preselected_features is not None and preselected_features.numel() > 0:
        pre = F.normalize(preselected_features.float(), p=2, dim=-1, eps=1.0e-6)
        max_sim = torch.matmul(normed, pre.transpose(0, 1)).max(dim=1).values.clamp_min(0.0)

    selected: list[int] = []
    available = torch.ones((features.shape[0],), dtype=torch.bool, device=features.device)
    score_vec = scores.float()
    for _ in range(target):
        values = score_vec - float(diversity_weight) * max_sim
        values = values.masked_fill(~available, -float("inf"))
        idx = int(torch.argmax(values).item())
        if not bool(available[idx].item()):
            break
        selected.append(idx)
        available[idx] = False
        sim_to_chosen = torch.matmul(normed, normed[idx]).clamp_min(0.0)
        max_sim = torch.maximum(max_sim, sim_to_chosen)

    if not selected:
        empty_f, empty_g, _ = _empty_tokens(features.shape[-1], features.dtype, features.device)
        return empty_f, empty_g, torch.empty((0,), dtype=torch.long, device=features.device)
    keep = torch.tensor(selected, dtype=torch.long, device=features.device)
    return features[keep], globals_[keep], sources[keep]


def hedge_segment_compression(
    segment_features: torch.Tensor,
    segment_global_indices: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """HEDGE-VID segment compression.

    HEDGE freezes FlashVID ADTS anchors, then selects the residual budget from
    a conservative union of original FlashVID TSTM candidates and CATS
    protected-sink evidence candidates.
    """
    from .utils import spatiotemporal_compression

    num_frames, num_visual_tokens, feat_dim = segment_features.shape
    device = segment_features.device
    dtype = segment_features.dtype
    per_frame_adts = int(flashvid_config.num_attn_div_tokens or 0)
    num_other_tokens = int(flashvid_config.num_sttm_tokens or 0) * num_frames

    selected_indices, selected_mask, importance = _flashvid_adts_selection(
        features=segment_features,
        cls_attention=cls_attention,
        per_frame_tokens=per_frame_adts,
        config=flashvid_config,
    )

    selected_tokens: list[torch.Tensor] = []
    selected_globals: list[torch.Tensor] = []
    for frame_idx, indices in enumerate(selected_indices):
        if indices.numel() == 0:
            continue
        selected_tokens.append(segment_features[frame_idx, indices])
        selected_globals.append(segment_global_indices[frame_idx, indices])
    if selected_tokens:
        anchor_features = torch.cat(selected_tokens, dim=0)
        anchor_globals = torch.cat(selected_globals, dim=0)
    else:
        anchor_features = torch.empty((0, feat_dim), dtype=dtype, device=device)
        anchor_globals = torch.empty((0,), dtype=torch.long, device=device)

    if num_other_tokens <= 0:
        _accumulate_hedge_metrics(
            flashvid_config,
            selected_adts=int(selected_mask.sum().item()),
            residual_budget=0,
            stable_candidates=0,
            evidence_candidates=0,
            stable_selected=0,
            evidence_selected=0,
            final_tokens=int(anchor_features.shape[0]),
        )
        return anchor_features, anchor_globals

    residual_mask = ~selected_mask

    if num_frames > 1 and float(flashvid_config.temporal_threshold) < 1.0:
        stable_lists, stable_idx_lists = spatiotemporal_compression(
            video_features=segment_features.clone(),
            temporal_threshold=float(flashvid_config.temporal_threshold),
            token_mask=residual_mask,
            flashvid_config=flashvid_config,
        )
        stable_global_lists = [
            segment_global_indices.view(num_frames, -1)[frame_idx][indices]
            for frame_idx, indices in enumerate(stable_idx_lists)
        ]
    else:
        stable_lists = [segment_features[frame_idx, residual_mask[frame_idx]] for frame_idx in range(num_frames)]
        stable_global_lists = [segment_global_indices[frame_idx, residual_mask[frame_idx]] for frame_idx in range(num_frames)]
    stable_features, stable_globals = _spatial_merge_lists(
        stable_lists,
        stable_global_lists,
        target_total=num_other_tokens,
        feat_dim=feat_dim,
        dtype=dtype,
        device=device,
    )

    updated_features, evidence_mask, _metrics = _cats_sink_tstm(
        video_features=segment_features.clone(),
        selected_mask=selected_mask,
        importance=importance,
        temporal_threshold=float(flashvid_config.temporal_threshold),
        config=flashvid_config,
    )
    evidence_lists = [updated_features[frame_idx, evidence_mask[frame_idx]] for frame_idx in range(num_frames)]
    evidence_global_lists = [segment_global_indices[frame_idx, evidence_mask[frame_idx]] for frame_idx in range(num_frames)]
    evidence_features, evidence_globals = _spatial_merge_lists(
        evidence_lists,
        evidence_global_lists,
        target_total=num_other_tokens,
        feat_dim=feat_dim,
        dtype=dtype,
        device=device,
    )

    stable_bias = _cfg_float(flashvid_config, "hedge_stable_bias", 0.05)
    evidence_bias = _cfg_float(flashvid_config, "hedge_evidence_bias", 0.0)
    stable_scores = _scores_for_globals(segment_features, segment_global_indices, cls_attention, stable_globals, stable_bias)
    evidence_scores = _scores_for_globals(segment_features, segment_global_indices, cls_attention, evidence_globals, evidence_bias)
    stable_sources = torch.zeros((stable_globals.shape[0],), dtype=torch.long, device=device)
    evidence_sources = torch.ones((evidence_globals.shape[0],), dtype=torch.long, device=device)

    floor_ratio = min(max(_cfg_float(flashvid_config, "hedge_stable_floor_ratio", 0.85), 0.0), 1.0)
    stable_floor = min(num_other_tokens, int(round(num_other_tokens * floor_ratio)))
    diversity_weight = max(0.0, _cfg_float(flashvid_config, "hedge_diversity_weight", 0.04))
    max_candidates = max(32, _cfg_int(flashvid_config, "hedge_max_mmr_candidates", 2048))
    blocked = set(int(x) for x in anchor_globals.detach().cpu().tolist())

    stable_keep_features, stable_keep_globals, stable_keep_sources = _mmr_select(
        stable_features,
        stable_globals,
        stable_scores,
        stable_sources,
        stable_floor,
        diversity_weight=diversity_weight,
        max_candidates=max_candidates,
        preselected_features=anchor_features,
        blocked_globals=blocked,
    )
    blocked.update(int(x) for x in stable_keep_globals.detach().cpu().tolist())

    remaining = max(0, num_other_tokens - int(stable_keep_globals.shape[0]))
    if stable_features.numel() > 0 and evidence_features.numel() > 0:
        union_features = torch.cat([stable_features, evidence_features], dim=0)
        union_globals = torch.cat([stable_globals, evidence_globals], dim=0)
        union_scores = torch.cat([stable_scores, evidence_scores], dim=0)
        union_sources = torch.cat([stable_sources, evidence_sources], dim=0)
    elif evidence_features.numel() > 0:
        union_features, union_globals, union_scores, union_sources = evidence_features, evidence_globals, evidence_scores, evidence_sources
    else:
        union_features, union_globals, union_scores, union_sources = stable_features, stable_globals, stable_scores, stable_sources

    preselected = anchor_features
    if stable_keep_features.numel() > 0:
        preselected = torch.cat([anchor_features, stable_keep_features], dim=0) if anchor_features.numel() > 0 else stable_keep_features
    extra_features, extra_globals, extra_sources = _mmr_select(
        union_features,
        union_globals,
        union_scores,
        union_sources,
        remaining,
        diversity_weight=diversity_weight,
        max_candidates=max_candidates,
        preselected_features=preselected,
        blocked_globals=blocked,
    )

    residual_features = [stable_keep_features]
    residual_globals = [stable_keep_globals]
    residual_sources = [stable_keep_sources]
    if extra_features.numel() > 0:
        residual_features.append(extra_features)
        residual_globals.append(extra_globals)
        residual_sources.append(extra_sources)
    if residual_features:
        chosen_residual_features = torch.cat(residual_features, dim=0)
        chosen_residual_globals = torch.cat(residual_globals, dim=0)
        chosen_sources = torch.cat(residual_sources, dim=0)
    else:
        chosen_residual_features, chosen_residual_globals, _ = _empty_tokens(feat_dim, dtype, device)
        chosen_sources = torch.empty((0,), dtype=torch.long, device=device)

    if chosen_residual_globals.shape[0] < num_other_tokens:
        # If diversity/dedup blocked too much, fill deterministically from stable candidates.
        fill_needed = num_other_tokens - int(chosen_residual_globals.shape[0])
        refill_blocked = blocked.union(int(x) for x in chosen_residual_globals.detach().cpu().tolist())
        refill_features, refill_globals, refill_sources = _mmr_select(
            union_features,
            union_globals,
            union_scores,
            union_sources,
            fill_needed,
            diversity_weight=0.0,
            max_candidates=max_candidates,
            preselected_features=None,
            blocked_globals=refill_blocked,
        )
        if refill_features.numel() > 0:
            chosen_residual_features = torch.cat([chosen_residual_features, refill_features], dim=0)
            chosen_residual_globals = torch.cat([chosen_residual_globals, refill_globals], dim=0)
            chosen_sources = torch.cat([chosen_sources, refill_sources], dim=0)

    all_features = torch.cat([anchor_features, chosen_residual_features], dim=0)
    all_globals = torch.cat([anchor_globals, chosen_residual_globals], dim=0)
    order = all_globals.argsort()

    stable_selected = int((chosen_sources == 0).sum().item()) if chosen_sources.numel() > 0 else 0
    evidence_selected = int((chosen_sources == 1).sum().item()) if chosen_sources.numel() > 0 else 0
    _accumulate_hedge_metrics(
        flashvid_config,
        selected_adts=int(selected_mask.sum().item()),
        residual_budget=int(num_other_tokens),
        stable_candidates=int(stable_globals.shape[0]),
        evidence_candidates=int(evidence_globals.shape[0]),
        stable_selected=stable_selected,
        evidence_selected=evidence_selected,
        final_tokens=int(all_features.shape[0]),
    )
    return all_features[order], all_globals[order]
