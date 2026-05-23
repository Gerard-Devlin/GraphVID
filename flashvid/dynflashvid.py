from __future__ import annotations

from typing import List, Tuple

import math
import torch
from torch.nn import functional as F

from .configuration_flashvid import FlashVidConfig
from .graphvid import _grid_hw, _neighbor_table, _spatial_detail_score


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


def _cfg_bool(config: FlashVidConfig, name: str, default: bool) -> bool:
    value = getattr(config, name, None)
    if value is None:
        return bool(default)
    return bool(value)


def _minmax(x: torch.Tensor, dim: int = -1, eps: float = 1.0e-6) -> torch.Tensor:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    lo = x.amin(dim=dim, keepdim=True)
    hi = x.amax(dim=dim, keepdim=True)
    return ((x - lo) / (hi - lo + eps)).clamp(0.0, 1.0)


def _importance_maps(
    features: torch.Tensor,
    cls_attention: torch.Tensor,
    config: FlashVidConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_frames, num_visual_tokens, _ = features.shape
    normed = F.normalize(features.float(), p=2, dim=-1, eps=1.0e-6)

    attn = _minmax(cls_attention.float(), dim=-1)

    frame_proto = F.normalize(normed.mean(dim=1), p=2, dim=-1, eps=1.0e-6)
    event = torch.einsum("fnd,sd->fsn", normed, frame_proto).mean(dim=1)
    event = _minmax(event.clamp_min(0.0), dim=-1)

    novelty = torch.zeros((num_frames, num_visual_tokens), dtype=torch.float32, device=features.device)
    if num_frames > 1:
        prev_sim = torch.bmm(normed[1:], normed[:-1].transpose(1, 2)).max(dim=-1).values
        next_sim = torch.bmm(normed[:-1], normed[1:].transpose(1, 2)).max(dim=-1).values
        novelty[1:] = torch.maximum(novelty[1:], (1.0 - prev_sim).clamp(0.0, 2.0) * 0.5)
        novelty[:-1] = torch.maximum(novelty[:-1], (1.0 - next_sim).clamp(0.0, 2.0) * 0.5)
    novelty = _minmax(novelty, dim=-1)

    h, w = _grid_hw(num_visual_tokens, config)
    neighbor_idx, neighbor_valid = _neighbor_table(num_visual_tokens, h, w, radius=1, device=features.device)
    detail = _minmax(_spatial_detail_score(normed, neighbor_idx, neighbor_valid), dim=-1)

    attn_w = _cfg_float(config, "dyn_attn_weight", 0.50)
    event_w = _cfg_float(config, "dyn_event_weight", 0.30)
    novelty_w = _cfg_float(config, "dyn_novelty_weight", 0.15)
    detail_w = _cfg_float(config, "dyn_detail_weight", 0.05)
    denom = max(1.0e-6, attn_w + event_w + novelty_w + detail_w)
    token_importance = (
        attn_w * attn
        + event_w * event
        + novelty_w * novelty
        + detail_w * detail
    ) / denom

    motion = torch.zeros(num_frames, dtype=torch.float32, device=features.device)
    if num_frames > 1:
        motion[1:] = (1.0 - torch.sum(frame_proto[1:] * frame_proto[:-1], dim=-1)).clamp(0.0, 2.0) * 0.5
    attn_mass = attn.topk(k=min(16, num_visual_tokens), dim=-1).values.mean(dim=-1)
    event_mass = event.topk(k=min(16, num_visual_tokens), dim=-1).values.mean(dim=-1)
    detail_mass = detail.topk(k=min(16, num_visual_tokens), dim=-1).values.mean(dim=-1)
    frame_score = (
        0.35 * _minmax(motion.unsqueeze(0), dim=-1).squeeze(0)
        + 0.30 * _minmax(event_mass.unsqueeze(0), dim=-1).squeeze(0)
        + 0.20 * _minmax(attn_mass.unsqueeze(0), dim=-1).squeeze(0)
        + 0.15 * _minmax(detail_mass.unsqueeze(0), dim=-1).squeeze(0)
    )
    boundary_boost = _cfg_float(config, "dyn_boundary_boost", 0.08)
    if boundary_boost > 0 and num_frames > 0:
        frame_score[0] += boundary_boost
        frame_score[-1] += boundary_boost
    return token_importance.clamp(0.0, 1.0), frame_score.clamp_min(0.0)


def _frame_budgets(
    frame_score: torch.Tensor,
    per_frame_tokens: int,
    num_visual_tokens: int,
    config: FlashVidConfig,
) -> List[int]:
    num_frames = int(frame_score.shape[0])
    per_frame_tokens = min(max(0, int(per_frame_tokens)), num_visual_tokens)
    if per_frame_tokens <= 0:
        return [0 for _ in range(num_frames)]
    total_budget = per_frame_tokens * num_frames
    if not _cfg_bool(config, "dyn_adaptive_adts_budget", True):
        return [per_frame_tokens for _ in range(num_frames)]

    min_ratio = min(max(_cfg_float(config, "dyn_frame_budget_min_ratio", 0.50), 0.0), 1.0)
    max_ratio = max(1.0, _cfg_float(config, "dyn_frame_budget_max_ratio", 1.75))
    min_pf = min(per_frame_tokens, max(0, int(math.floor(per_frame_tokens * min_ratio))))
    max_pf = min(num_visual_tokens, max(min_pf, int(math.ceil(per_frame_tokens * max_ratio))))
    base = min_pf * num_frames
    if base >= total_budget:
        return [min_pf for _ in range(num_frames)]

    temperature = max(1.0e-3, _cfg_float(config, "dyn_budget_temperature", 0.75))
    strength = min(max(_cfg_float(config, "dyn_budget_strength", 0.45), 0.0), 1.0)
    soft = torch.softmax(frame_score.float() / temperature, dim=0)
    uniform = torch.full_like(soft, 1.0 / max(1, num_frames))
    probs = (1.0 - strength) * uniform + strength * soft
    remaining = total_budget - base
    extra_float = probs * float(remaining)
    extra = torch.floor(extra_float).long()
    budgets = (extra + min_pf).clamp(max=max_pf)

    def _fill_or_trim(values: torch.Tensor) -> torch.Tensor:
        diff = total_budget - int(values.sum().item())
        order = torch.argsort(frame_score, descending=(diff > 0)).tolist()
        guard = 0
        while diff != 0 and guard < max(1, num_frames * num_visual_tokens):
            changed = False
            for frame_idx in order:
                if diff > 0 and int(values[frame_idx].item()) < max_pf:
                    values[frame_idx] += 1
                    diff -= 1
                    changed = True
                elif diff < 0 and int(values[frame_idx].item()) > min_pf:
                    values[frame_idx] -= 1
                    diff += 1
                    changed = True
                if diff == 0:
                    break
            if not changed:
                break
            guard += 1
        return values

    budgets = _fill_or_trim(budgets)
    return [int(x) for x in budgets.tolist()]


def _select_one_frame(
    features: torch.Tensor,
    importance: torch.Tensor,
    k: int,
    beta: float,
) -> torch.Tensor:
    num_visual_tokens = int(features.shape[0])
    k = min(max(0, int(k)), num_visual_tokens)
    if k <= 0:
        return torch.empty((0,), dtype=torch.long, device=features.device)
    normed = F.normalize(features.float(), p=2, dim=-1, eps=1.0e-6)
    dist = 1.0 - torch.matmul(normed, normed.transpose(0, 1))
    keep = torch.zeros(k, dtype=torch.long, device=features.device)
    nearest = torch.topk(dist, k=min(2, num_visual_tokens), dim=0, largest=False).values[-1]
    keep[0] = torch.argmax(nearest + beta * importance)
    for i in range(1, k):
        score = dist[keep[:i]].min(dim=0).values + beta * importance
        score.scatter_(0, keep[:i], -float("inf"))
        keep[i] = torch.argmax(score)
    return keep.sort().values


def _dyn_adts_selection(
    features: torch.Tensor,
    cls_attention: torch.Tensor,
    per_frame_tokens: int,
    config: FlashVidConfig,
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, List[int], torch.Tensor]:
    importance, frame_score = _importance_maps(features, cls_attention, config)
    budgets = _frame_budgets(frame_score, per_frame_tokens, features.shape[1], config)
    beta = _cfg_float(config, "dyn_adts_beta", 0.05)

    selected: List[torch.Tensor] = []
    selected_mask = torch.zeros(features.shape[:2], dtype=torch.bool, device=features.device)
    for frame_idx, budget in enumerate(budgets):
        indices = _select_one_frame(features[frame_idx], importance[frame_idx], budget, beta)
        selected.append(indices)
        if indices.numel() > 0:
            selected_mask[frame_idx, indices] = True
    return selected, selected_mask, importance, budgets, frame_score


def _similarity_features(features: torch.Tensor, config: FlashVidConfig) -> torch.Tensor:
    x = F.normalize(features.float(), p=2, dim=-1, eps=1.0e-6)
    if not _cfg_bool(config, "dyn_similarity_debias", True):
        return x
    frame_w = _cfg_float(config, "dyn_debias_frame_weight", 0.35)
    global_w = _cfg_float(config, "dyn_debias_global_weight", 0.20)
    if frame_w != 0:
        x = x - frame_w * x.mean(dim=1, keepdim=True)
    if global_w != 0:
        x = x - global_w * x.reshape(-1, x.shape[-1]).mean(dim=0).view(1, 1, -1)
    return F.normalize(x, p=2, dim=-1, eps=1.0e-6)


def _sink_tstm(
    video_features: torch.Tensor,
    selected_mask: torch.Tensor,
    importance: torch.Tensor,
    temporal_threshold: float,
    config: FlashVidConfig,
) -> Tuple[torch.Tensor, torch.Tensor, dict[str, float | None]]:
    num_frames, num_visual_tokens, feat_dim = video_features.shape
    source_mask = ~selected_mask
    if num_frames <= 1:
        return video_features, source_mask, {
            "sink_merges": 0.0,
            "residual_merges": 0.0,
            "retained_residual_tokens": float(source_mask.sum().item()),
            "mean_merge_sim": None,
        }

    sim_features = _similarity_features(video_features, config)
    sim = torch.bmm(sim_features[1:], sim_features[:-1].transpose(1, 2))
    sim = sim.masked_fill(~source_mask[1:].unsqueeze(-1), -1.0)
    if not _cfg_bool(config, "dyn_sink_tstm", False):
        sim = sim.masked_fill(~source_mask[:-1].unsqueeze(1), -1.0)

    best_sim, best_prev = sim.max(dim=-1)
    if num_visual_tokens > 1:
        top2 = torch.topk(sim, k=2, dim=-1).values
        margin = top2[..., 0] - top2[..., 1]
    else:
        margin = torch.ones_like(best_sim)

    candidate = (best_sim > float(temporal_threshold)) & source_mask[1:]
    if _cfg_bool(config, "dyn_mutual_nn", False):
        prev_best_cur = sim.argmax(dim=1)
        cur_ids = torch.arange(num_visual_tokens, device=video_features.device).view(1, -1).expand(num_frames - 1, -1)
        mutual = prev_best_cur.gather(1, best_prev.clamp_min(0)) == cur_ids
        high_conf = best_sim > (float(temporal_threshold) + _cfg_float(config, "dyn_high_conf_bonus", 0.05))
        candidate = candidate & (mutual | high_conf)
    margin_threshold = _cfg_float(config, "dyn_margin_threshold", 0.0)
    if margin_threshold > 0:
        high_conf = best_sim > (float(temporal_threshold) + _cfg_float(config, "dyn_high_conf_bonus", 0.05))
        candidate = candidate & ((margin > margin_threshold) | high_conf)

    padded_scores = F.pad(best_sim, (0, 0, 1, 0), value=-1.0)
    padded_anchor = F.pad(best_prev, (0, 0, 1, 0), value=-1)
    merge_mask = F.pad(candidate, (0, 0, 1, 0), value=False)

    lower_bound = (int(config.num_attn_div_tokens or 0) + int(config.num_sttm_tokens or 0)) * num_frames
    selected_count = int(selected_mask.sum().item())
    retained_residual = int((~merge_mask & source_mask).sum().item())
    if selected_count + retained_residual < lower_bound:
        max_merges = max(0, num_frames * num_visual_tokens - lower_bound)
        current_merges = int(merge_mask.sum().item())
        if current_merges > max_merges:
            scores = torch.where(merge_mask, padded_scores, torch.full_like(padded_scores, -float("inf"))).view(-1)
            keep = torch.topk(scores, k=max_merges, largest=True).indices if max_merges > 0 else torch.empty((0,), dtype=torch.long, device=video_features.device)
            limited = torch.zeros_like(scores, dtype=torch.bool)
            if keep.numel() > 0:
                limited[keep] = True
            merge_mask = limited.view_as(merge_mask)

    weight_merge = _cfg_bool(config, "dyn_weighted_merge", False)
    attn_weight = max(0.0, _cfg_float(config, "dyn_confidence_attn_weight", 0.50))
    sim_weight = max(0.0, _cfg_float(config, "dyn_confidence_sim_weight", 0.50))
    token_weights = torch.ones((num_frames, num_visual_tokens), dtype=torch.float32, device=video_features.device)
    if weight_merge:
        token_weights = token_weights + attn_weight * importance.float()
    feature_sums = video_features.float() * token_weights.unsqueeze(-1)

    sink_merges = 0
    residual_merges = 0
    merge_sims: list[float] = []
    for frame_idx in range(num_frames - 1, -1, -1):
        frame_merge = merge_mask[frame_idx]
        if not frame_merge.any():
            continue
        anchor_idx = padded_anchor[frame_idx, frame_merge]
        valid = anchor_idx >= 0
        if not valid.any():
            continue
        src_idx = torch.where(frame_merge)[0][valid]
        anchor_idx = anchor_idx[valid]
        sims = padded_scores[frame_idx, src_idx].clamp_min(0.0)
        scale = 1.0 + sim_weight * sims if weight_merge else torch.ones_like(sims)

        child_sums = feature_sums[frame_idx, src_idx] * scale.unsqueeze(-1)
        child_weights = token_weights[frame_idx, src_idx] * scale
        agg_sums = torch.zeros((num_visual_tokens, feat_dim), dtype=torch.float32, device=video_features.device)
        agg_weights = torch.zeros((num_visual_tokens,), dtype=torch.float32, device=video_features.device)
        agg_sums.scatter_add_(0, anchor_idx.unsqueeze(-1).expand(-1, feat_dim), child_sums)
        agg_weights.scatter_add_(0, anchor_idx, child_weights)
        feature_sums[frame_idx - 1] += agg_sums
        token_weights[frame_idx - 1] += agg_weights
        feature_sums[frame_idx, src_idx] = 0.0
        token_weights[frame_idx, src_idx] = 0.0

        sink_merges += int(selected_mask[frame_idx - 1, anchor_idx].sum().item())
        residual_merges += int(anchor_idx.numel()) - int(selected_mask[frame_idx - 1, anchor_idx].sum().item())
        merge_sims.extend(float(x) for x in sims.detach().cpu().tolist())

    updated = feature_sums / token_weights.unsqueeze(-1).clamp_min(1.0e-6)
    residual_mask = (~merge_mask) & source_mask
    metrics = {
        "sink_merges": float(sink_merges),
        "residual_merges": float(residual_merges),
        "retained_residual_tokens": float(residual_mask.sum().item()),
        "mean_merge_sim": float(sum(merge_sims) / len(merge_sims)) if merge_sims else None,
    }
    return updated.to(dtype=video_features.dtype), residual_mask, metrics


def _spatial_merge(
    temp_tokens: List[torch.Tensor],
    temp_indices: List[torch.Tensor],
    target_tokens: int,
    feat_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[List[torch.Tensor], List[torch.Tensor], int, int]:
    from .utils import dpc_knn

    before = sum(int(x.shape[0]) for x in temp_tokens)
    if target_tokens <= 0 or before <= target_tokens:
        return temp_tokens, temp_indices, before, before
    ratio = target_tokens / max(1, before)
    num_frames = len(temp_tokens)
    max_tokens = max([int(x.shape[0]) for x in temp_tokens] + [1])
    batched = torch.zeros((num_frames, max_tokens, feat_dim), dtype=dtype, device=device)
    valid_mask = torch.zeros((num_frames, max_tokens), dtype=torch.bool, device=device)
    cluster_counts = []
    k_list = []
    for frame_idx, tokens in enumerate(temp_tokens):
        n = int(tokens.shape[0])
        if n > 0:
            batched[frame_idx, :n] = tokens
            valid_mask[frame_idx, :n] = True
        k = int(math.ceil(n * ratio))
        cluster_counts.append(k)
        k_list.append(max(1, min(k, 7)) if k > 0 else 1)

    cluster_indices_list, center_indices_list = dpc_knn(
        features=batched,
        num_clusters=cluster_counts,
        k=k_list,
        valid_token_mask=valid_mask,
    )
    out_tokens: List[torch.Tensor] = []
    out_indices: List[torch.Tensor] = []
    for frame_idx, (tokens, indices) in enumerate(zip(temp_tokens, temp_indices)):
        clusters = cluster_counts[frame_idx]
        if clusters <= 0 or tokens.numel() == 0:
            out_tokens.append(tokens)
            out_indices.append(indices)
            continue
        cluster_idx = cluster_indices_list[frame_idx][: int(tokens.shape[0])]
        centers = center_indices_list[frame_idx]
        merged = torch.zeros((clusters, feat_dim), dtype=dtype, device=device)
        merged.scatter_add_(0, cluster_idx.unsqueeze(-1).expand(-1, feat_dim), tokens)
        counts = torch.bincount(cluster_idx, minlength=clusters).unsqueeze(-1).to(dtype)
        merged = merged / counts.clamp_min(1)
        if indices.numel() > 0:
            centers = centers.clamp(min=0, max=indices.shape[0] - 1)
            if centers.numel() < clusters:
                pad = centers[-1].repeat(clusters - centers.numel()) if centers.numel() > 0 else torch.zeros((clusters,), dtype=torch.long, device=device)
                centers = torch.cat([centers, pad], dim=0)
            elif centers.numel() > clusters:
                centers = centers[:clusters]
            out_idx = indices[centers]
        else:
            out_idx = torch.empty((0,), dtype=torch.long, device=device)
        out_tokens.append(merged)
        out_indices.append(out_idx)
    after = sum(int(x.shape[0]) for x in out_tokens)
    return out_tokens, out_indices, before, after


def _reset_dyn_metrics(config: FlashVidConfig) -> None:
    defaults = {
        "last_dyn_selected_tokens": 0.0,
        "last_dyn_budget_min": 0.0,
        "last_dyn_budget_max": 0.0,
        "last_dyn_budget_std": 0.0,
        "last_dyn_sink_merges": 0.0,
        "last_dyn_residual_merges": 0.0,
        "last_dyn_retained_residual_tokens": 0.0,
        "last_dyn_spatial_tokens_before": 0.0,
        "last_dyn_spatial_tokens_after": 0.0,
        "last_dyn_mean_merge_sim": None,
        "last_dyn_similarity_debias_active": 0.0,
        "last_dyn_sink_active": 0.0,
        "last_dyn_weighted_active": 0.0,
    }
    for key, value in defaults.items():
        setattr(config, key, value)
    for key in list(defaults):
        setattr(config, "_" + key[5:] + "_sum", 0.0)
    setattr(config, "_dyn_segments", 0.0)
    setattr(config, "_dyn_merge_sim_count", 0.0)


def _accumulate_dyn_metrics(
    config: FlashVidConfig,
    *,
    selected_count: int,
    budgets: List[int],
    merge_metrics: dict[str, float | None],
    spatial_before: int,
    spatial_after: int,
) -> None:
    if not hasattr(config, "_dyn_segments"):
        _reset_dyn_metrics(config)
    segments = float(getattr(config, "_dyn_segments", 0.0)) + 1.0
    setattr(config, "_dyn_segments", segments)

    budget_tensor = torch.tensor(budgets, dtype=torch.float32)
    values = {
        "dyn_selected_tokens": float(selected_count),
        "dyn_budget_min": float(budget_tensor.min().item()) if budget_tensor.numel() else 0.0,
        "dyn_budget_max": float(budget_tensor.max().item()) if budget_tensor.numel() else 0.0,
        "dyn_budget_std": float(budget_tensor.std(unbiased=False).item()) if budget_tensor.numel() else 0.0,
        "dyn_sink_merges": float(merge_metrics.get("sink_merges") or 0.0),
        "dyn_residual_merges": float(merge_metrics.get("residual_merges") or 0.0),
        "dyn_retained_residual_tokens": float(merge_metrics.get("retained_residual_tokens") or 0.0),
        "dyn_spatial_tokens_before": float(spatial_before),
        "dyn_spatial_tokens_after": float(spatial_after),
        "dyn_similarity_debias_active": float(_cfg_bool(config, "dyn_similarity_debias", True)),
        "dyn_sink_active": float(_cfg_bool(config, "dyn_sink_tstm", False)),
        "dyn_weighted_active": float(_cfg_bool(config, "dyn_weighted_merge", False)),
    }
    for key, value in values.items():
        sum_key = f"_{key}_sum"
        setattr(config, sum_key, float(getattr(config, sum_key, 0.0)) + value)
        setattr(config, f"last_{key}", float(getattr(config, sum_key, 0.0)) / segments)
    if merge_metrics.get("mean_merge_sim") is not None:
        setattr(config, "_dyn_mean_merge_sim_sum", float(getattr(config, "_dyn_mean_merge_sim_sum", 0.0)) + float(merge_metrics["mean_merge_sim"]))
        setattr(config, "_dyn_merge_sim_count", float(getattr(config, "_dyn_merge_sim_count", 0.0)) + 1.0)
    sim_count = float(getattr(config, "_dyn_merge_sim_count", 0.0))
    setattr(
        config,
        "last_dyn_mean_merge_sim",
        float(getattr(config, "_dyn_mean_merge_sim_sum", 0.0)) / sim_count if sim_count > 0 else None,
    )


def dyn_segment_compression(
    segment_features: torch.Tensor,
    segment_global_indices: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dynamic-budget FlashVID segment compression.

    This keeps FlashVID's total ADTS/STTM budget but redistributes ADTS tokens
    across frames and uses a debiased temporal similarity for the residual merge.
    Optional protected-sink merging can be enabled for ablations.
    """
    num_frames, num_visual_tokens, feat_dim = segment_features.shape
    device = segment_features.device
    per_frame_adts = int(flashvid_config.num_attn_div_tokens or 0)

    selected_indices, selected_mask, importance, budgets, _ = _dyn_adts_selection(
        segment_features,
        cls_attention,
        per_frame_adts,
        flashvid_config,
    )
    updated_features, residual_mask, merge_metrics = _sink_tstm(
        video_features=segment_features,
        selected_mask=selected_mask,
        importance=importance,
        temporal_threshold=float(flashvid_config.temporal_threshold),
        config=flashvid_config,
    )

    selected_tokens = []
    selected_global = []
    for frame_idx, indices in enumerate(selected_indices):
        if indices.numel() == 0:
            continue
        selected_tokens.append(updated_features[frame_idx, indices])
        selected_global.append(segment_global_indices[frame_idx, indices])
    if selected_tokens:
        selected_features = torch.cat(selected_tokens, dim=0)
        selected_global_indices = torch.cat(selected_global, dim=0)
    else:
        selected_features = torch.empty((0, feat_dim), dtype=segment_features.dtype, device=device)
        selected_global_indices = torch.empty((0,), dtype=torch.long, device=device)

    temp_tokens: List[torch.Tensor] = []
    temp_indices: List[torch.Tensor] = []
    for frame_idx in range(num_frames):
        idx = torch.where(residual_mask[frame_idx])[0]
        temp_tokens.append(updated_features[frame_idx, idx])
        temp_indices.append(segment_global_indices[frame_idx, idx])

    target_residual = int(flashvid_config.num_sttm_tokens or 0) * num_frames
    temp_tokens, temp_indices, spatial_before, spatial_after = _spatial_merge(
        temp_tokens,
        temp_indices,
        target_residual,
        feat_dim,
        segment_features.dtype,
        device,
    )

    all_tokens = [selected_features] + temp_tokens
    all_indices = [selected_global_indices] + temp_indices
    final_tokens = torch.cat(all_tokens, dim=0)
    final_indices = torch.cat(all_indices, dim=0)
    _accumulate_dyn_metrics(
        flashvid_config,
        selected_count=int(selected_mask.sum().item()),
        budgets=budgets,
        merge_metrics=merge_metrics,
        spatial_before=spatial_before,
        spatial_after=spatial_after,
    )
    return final_tokens, final_indices
