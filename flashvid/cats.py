from __future__ import annotations

from typing import List, Tuple

import math
import torch
from torch.nn import functional as F

from .configuration_flashvid import FlashVidConfig


_CATS_ADTS_MODE_CODE = {
    "cats": 0.0,
    "flashvid": 1.0,
}


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


def _cats_importance(
    features: torch.Tensor,
    cls_attention: torch.Tensor,
) -> torch.Tensor:
    """Stable attention/event importance used by CATS ADTS and aggregation."""
    normed = F.normalize(features.float(), p=2, dim=-1, eps=1.0e-6)
    attn = _minmax(cls_attention.float(), dim=-1)
    frame_proto = F.normalize(normed.mean(dim=1), p=2, dim=-1, eps=1.0e-6)
    event = torch.einsum("fnd,sd->fsn", normed, frame_proto).mean(dim=1)
    event = _minmax(event.clamp_min(0.0), dim=-1)
    return torch.sqrt((attn * event).clamp_min(1.0e-6))


def _cats_frame_budgets(
    features: torch.Tensor,
    cls_attention: torch.Tensor,
    per_frame_tokens: int,
    config: FlashVidConfig,
) -> List[int]:
    num_frames, num_visual_tokens, _ = features.shape
    per_frame_tokens = min(max(0, int(per_frame_tokens)), num_visual_tokens)
    if per_frame_tokens <= 0:
        return [0 for _ in range(num_frames)]
    if not _cfg_bool(config, "cats_adaptive_adts_budget", False):
        return [per_frame_tokens for _ in range(num_frames)]

    total_budget = per_frame_tokens * num_frames
    min_per_frame = min(per_frame_tokens, max(0, _cfg_int(config, "cats_frame_budget_min", 1)))
    base = min_per_frame * num_frames
    if base >= total_budget:
        return [min_per_frame for _ in range(num_frames)]

    normed = F.normalize(features.float(), p=2, dim=-1, eps=1.0e-6)
    frame_proto = F.normalize(normed.mean(dim=1), p=2, dim=-1, eps=1.0e-6)
    motion = torch.zeros(num_frames, dtype=torch.float32, device=features.device)
    if num_frames > 1:
        motion[1:] = (1.0 - torch.sum(frame_proto[1:] * frame_proto[:-1], dim=-1)).clamp(0.0, 2.0) * 0.5

    attn = _minmax(cls_attention.float(), dim=-1)
    topk = min(16, int(attn.shape[-1]))
    top_mass = attn.topk(k=topk, dim=-1).values.sum(dim=-1)
    entropy = -(attn * (attn + 1.0e-6).log()).sum(dim=-1)
    frame_score = (
        0.50 * _minmax(motion.unsqueeze(0), dim=-1).squeeze(0)
        + 0.30 * _minmax(top_mass.unsqueeze(0), dim=-1).squeeze(0)
        + 0.20 * _minmax(entropy.unsqueeze(0), dim=-1).squeeze(0)
    )
    temperature = max(1.0e-3, _cfg_float(config, "cats_frame_budget_temperature", 0.7))
    probs = torch.softmax(frame_score / temperature, dim=0)
    remaining = total_budget - base
    extra_float = probs * float(remaining)
    extra = torch.floor(extra_float).long()
    remainder = int(remaining - int(extra.sum().item()))
    if remainder > 0:
        frac = extra_float - extra.float()
        for idx in torch.topk(frac, k=min(remainder, num_frames), largest=True).indices.tolist():
            extra[idx] += 1
    budgets = (extra + min_per_frame).clamp(max=num_visual_tokens)

    # If clamping lost tokens, fill frames with available room by score.
    lost = total_budget - int(budgets.sum().item())
    if lost > 0:
        order = torch.argsort(frame_score, descending=True).tolist()
        cursor = 0
        while lost > 0 and cursor < len(order) * max(1, num_visual_tokens):
            frame_idx = order[cursor % len(order)]
            if int(budgets[frame_idx].item()) < num_visual_tokens:
                budgets[frame_idx] += 1
                lost -= 1
            cursor += 1
    return [int(x) for x in budgets.tolist()]


def _cats_select_one_frame(
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
        sub = dist[keep[:i]]
        score = sub.min(dim=0).values + beta * importance
        score.scatter_(0, keep[:i], -float("inf"))
        keep[i] = torch.argmax(score)
    return keep.sort().values


def _cats_adts_selection(
    features: torch.Tensor,
    cls_attention: torch.Tensor,
    per_frame_tokens: int,
    config: FlashVidConfig,
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    importance = _cats_importance(features, cls_attention)
    beta = _cfg_float(config, "cats_adts_beta", 0.05)
    budgets = _cats_frame_budgets(features, cls_attention, per_frame_tokens, config)
    selected: List[torch.Tensor] = []
    selected_mask = torch.zeros(
        features.shape[:2],
        dtype=torch.bool,
        device=features.device,
    )
    for frame_idx, k in enumerate(budgets):
        indices = _cats_select_one_frame(features[frame_idx], importance[frame_idx], k, beta)
        selected.append(indices)
        if indices.numel() > 0:
            selected_mask[frame_idx, indices] = True
    return selected, selected_mask, importance


def _flashvid_adts_selection(
    features: torch.Tensor,
    cls_attention: torch.Tensor,
    per_frame_tokens: int,
    config: FlashVidConfig,
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Use FlashVID's original ADTS path, but keep CATS importance for merging weights."""
    num_frames, num_visual_tokens, _ = features.shape
    per_frame_tokens = min(max(0, int(per_frame_tokens)), num_visual_tokens)
    importance = _cats_importance(features, cls_attention)
    selected_mask = torch.zeros(
        features.shape[:2],
        dtype=torch.bool,
        device=features.device,
    )
    if per_frame_tokens <= 0:
        return [torch.empty((0,), dtype=torch.long, device=features.device) for _ in range(num_frames)], selected_mask, importance

    from .utils import ALL_TOKEN_SELECTION_METHOD

    additional_kwargs = {"cls_attention": cls_attention} if "attn" in str(config.token_selection_method) else {}
    _, selected_indices = ALL_TOKEN_SELECTION_METHOD[config.token_selection_method](
        features=features,
        num_retained_tokens=per_frame_tokens,
        **additional_kwargs,
    )
    selected_indices = selected_indices.to(device=features.device, dtype=torch.long)
    selected_mask.scatter_(1, selected_indices, True)
    selected = [selected_indices[frame_idx].sort().values for frame_idx in range(num_frames)]
    return selected, selected_mask, importance


def _cats_sink_tstm(
    video_features: torch.Tensor,
    selected_mask: torch.Tensor,
    importance: torch.Tensor,
    temporal_threshold: float,
    config: FlashVidConfig,
) -> Tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    num_frames, num_visual_tokens, feat_dim = video_features.shape
    device = video_features.device
    source_mask = ~selected_mask
    if num_frames <= 1:
        metrics = {
            "sink_merges": 0.0,
            "residual_merges": 0.0,
            "mutual_rejected": 0.0,
            "margin_rejected": 0.0,
            "threshold_rejected": 0.0,
            "retained_residual_tokens": float(source_mask.sum().item()),
            "mean_merge_sim": None,
            "mean_margin": None,
        }
        return video_features, source_mask, metrics
    normed = F.normalize(video_features.float(), p=2, dim=-1, eps=1.0e-6)
    sim = torch.bmm(normed[1:], normed[:-1].transpose(1, 2))
    sim = sim.masked_fill(~source_mask[1:].unsqueeze(-1), -1.0)

    best_sim, best_prev = sim.max(dim=-1)
    if num_visual_tokens > 1:
        top2 = torch.topk(sim, k=2, dim=-1).values
        margin = top2[..., 0] - top2[..., 1]
    else:
        margin = torch.full_like(best_sim, 1.0)

    prev_best_cur = sim.argmax(dim=1)
    cur_ids = torch.arange(num_visual_tokens, device=device).view(1, -1).expand(num_frames - 1, -1)
    mutual = prev_best_cur.gather(1, best_prev.clamp_min(0)) == cur_ids

    margin_threshold = _cfg_float(config, "cats_margin_threshold", 0.03)
    high_conf_bonus = _cfg_float(config, "cats_high_conf_bonus", 0.05)
    use_mutual = _cfg_bool(config, "cats_mutual_nn", True)
    high_conf = best_sim > (float(temporal_threshold) + high_conf_bonus)
    candidate = (best_sim > float(temporal_threshold)) & source_mask[1:]
    if use_mutual:
        candidate = candidate & (mutual | high_conf)
    candidate = candidate & ((margin > margin_threshold) | high_conf)

    padded_scores = F.pad(best_sim, (0, 0, 1, 0), value=-1.0)
    padded_margin = F.pad(margin, (0, 0, 1, 0), value=0.0)
    padded_anchor = F.pad(best_prev, (0, 0, 1, 0), value=-1)
    merge_mask = F.pad(candidate, (0, 0, 1, 0), value=False)

    lower_bound = (int(config.num_attn_div_tokens or 0) + int(config.num_sttm_tokens or 0)) * num_frames
    retained_mask = ~merge_mask
    if int(retained_mask.sum().item()) < lower_bound:
        max_merges = max(0, num_frames * num_visual_tokens - int(lower_bound))
        flat_scores = torch.where(merge_mask, padded_scores, torch.full_like(padded_scores, -float("inf"))).view(-1)
        merge_count = int(merge_mask.sum().item())
        if max_merges <= 0:
            merge_mask.zero_()
        elif merge_count > max_merges:
            keep_flat = torch.topk(flat_scores, k=max_merges, largest=True).indices
            limited = torch.zeros_like(flat_scores, dtype=torch.bool)
            limited[keep_flat] = True
            merge_mask = limited.view_as(merge_mask)
        retained_mask = ~merge_mask

    attn_weight = max(0.0, _cfg_float(config, "cats_confidence_attn_weight", 0.75))
    sim_weight = max(0.0, _cfg_float(config, "cats_confidence_sim_weight", 1.0))
    anchor_self_weight = max(1.0, _cfg_float(config, "cats_anchor_self_weight", 1.0))
    initial_weight = 1.0 + attn_weight * importance.float()
    if anchor_self_weight > 1.0:
        initial_weight = torch.where(
            selected_mask,
            initial_weight * anchor_self_weight,
            initial_weight,
        )
    feature_sums = video_features.float() * initial_weight.unsqueeze(-1)
    token_weights = initial_weight.clone()

    sink_merges = 0
    residual_merges = 0
    merge_sims: list[float] = []
    merge_margins: list[float] = []
    for frame_idx in range(num_frames - 1, -1, -1):
        frame_merge = merge_mask[frame_idx]
        if not frame_merge.any():
            continue
        anchor_token_indices = padded_anchor[frame_idx, frame_merge]
        valid = anchor_token_indices >= 0
        if not valid.any():
            continue
        source_indices = torch.where(frame_merge)[0][valid]
        anchor_token_indices = anchor_token_indices[valid]
        sims = padded_scores[frame_idx, source_indices].clamp_min(0.0)
        scale = 1.0 + sim_weight * sims

        child_sums = feature_sums[frame_idx, source_indices] * scale.unsqueeze(-1)
        child_weights = token_weights[frame_idx, source_indices] * scale
        aggregated_sums = torch.zeros((num_visual_tokens, feat_dim), dtype=torch.float32, device=device)
        aggregated_weights = torch.zeros((num_visual_tokens,), dtype=torch.float32, device=device)
        aggregated_sums.scatter_add_(0, anchor_token_indices.unsqueeze(-1).expand(-1, feat_dim), child_sums)
        aggregated_weights.scatter_add_(0, anchor_token_indices, child_weights)
        feature_sums[frame_idx - 1] += aggregated_sums
        token_weights[frame_idx - 1] += aggregated_weights
        token_weights[frame_idx, source_indices] = 0.0
        feature_sums[frame_idx, source_indices] = 0.0

        sink_merges += int(selected_mask[frame_idx - 1, anchor_token_indices].sum().item())
        residual_merges += int(anchor_token_indices.numel()) - int(selected_mask[frame_idx - 1, anchor_token_indices].sum().item())
        merge_sims.extend(float(x) for x in sims.detach().cpu().tolist())
        merge_margins.extend(float(x) for x in padded_margin[frame_idx, source_indices].detach().cpu().tolist())

    updated_features = feature_sums / token_weights.unsqueeze(-1).clamp_min(1.0e-6)
    final_residual_mask = retained_mask & source_mask
    metrics = {
        "sink_merges": float(sink_merges),
        "residual_merges": float(residual_merges),
        "mutual_rejected": float(((best_sim > float(temporal_threshold)) & source_mask[1:] & ~mutual).sum().item()),
        "margin_rejected": float(((best_sim > float(temporal_threshold)) & source_mask[1:] & (margin <= margin_threshold) & ~high_conf).sum().item()),
        "threshold_rejected": float(((best_sim <= float(temporal_threshold)) & source_mask[1:]).sum().item()),
        "retained_residual_tokens": float(final_residual_mask.sum().item()),
        "mean_merge_sim": float(sum(merge_sims) / len(merge_sims)) if merge_sims else None,
        "mean_margin": float(sum(merge_margins) / len(merge_margins)) if merge_margins else None,
    }
    return updated_features.to(dtype=video_features.dtype), final_residual_mask, metrics


def _reset_cats_metrics(config: FlashVidConfig) -> None:
    defaults = {
        "last_cats_selected_tokens": 0.0,
        "last_cats_sink_merges": 0.0,
        "last_cats_residual_merges": 0.0,
        "last_cats_mutual_rejected": 0.0,
        "last_cats_margin_rejected": 0.0,
        "last_cats_threshold_rejected": 0.0,
        "last_cats_retained_residual_tokens": 0.0,
        "last_cats_spatial_tokens_before": 0.0,
        "last_cats_spatial_tokens_after": 0.0,
        "last_cats_mean_merge_sim": None,
        "last_cats_mean_margin": None,
        "last_cats_adts_mode_code": None,
    }
    for key, value in defaults.items():
        setattr(config, key, value)
    setattr(config, "_cats_segments", 0.0)
    setattr(config, "_cats_selected_sum", 0.0)
    setattr(config, "_cats_sink_sum", 0.0)
    setattr(config, "_cats_residual_sum", 0.0)
    setattr(config, "_cats_mutual_sum", 0.0)
    setattr(config, "_cats_margin_sum", 0.0)
    setattr(config, "_cats_threshold_sum", 0.0)
    setattr(config, "_cats_retained_sum", 0.0)
    setattr(config, "_cats_spatial_before_sum", 0.0)
    setattr(config, "_cats_spatial_after_sum", 0.0)
    setattr(config, "_cats_merge_sim_sum", 0.0)
    setattr(config, "_cats_merge_sim_count", 0.0)
    setattr(config, "_cats_margin_value_sum", 0.0)
    setattr(config, "_cats_margin_value_count", 0.0)


def _accumulate_cats_metrics(config: FlashVidConfig, selected_count: int, metrics: dict[str, float], before: int, after: int) -> None:
    if not hasattr(config, "_cats_segments"):
        _reset_cats_metrics(config)
    setattr(config, "_cats_segments", float(getattr(config, "_cats_segments", 0.0)) + 1.0)
    for attr, value in (
        ("_cats_selected_sum", selected_count),
        ("_cats_sink_sum", metrics.get("sink_merges", 0.0)),
        ("_cats_residual_sum", metrics.get("residual_merges", 0.0)),
        ("_cats_mutual_sum", metrics.get("mutual_rejected", 0.0)),
        ("_cats_margin_sum", metrics.get("margin_rejected", 0.0)),
        ("_cats_threshold_sum", metrics.get("threshold_rejected", 0.0)),
        ("_cats_retained_sum", metrics.get("retained_residual_tokens", 0.0)),
        ("_cats_spatial_before_sum", before),
        ("_cats_spatial_after_sum", after),
    ):
        setattr(config, attr, float(getattr(config, attr, 0.0)) + float(value))
    if metrics.get("mean_merge_sim") is not None:
        setattr(config, "_cats_merge_sim_sum", float(getattr(config, "_cats_merge_sim_sum", 0.0)) + float(metrics["mean_merge_sim"]))
        setattr(config, "_cats_merge_sim_count", float(getattr(config, "_cats_merge_sim_count", 0.0)) + 1.0)
    if metrics.get("mean_margin") is not None:
        setattr(config, "_cats_margin_value_sum", float(getattr(config, "_cats_margin_value_sum", 0.0)) + float(metrics["mean_margin"]))
        setattr(config, "_cats_margin_value_count", float(getattr(config, "_cats_margin_value_count", 0.0)) + 1.0)

    segments = max(1.0, float(getattr(config, "_cats_segments", 1.0)))
    setattr(config, "last_cats_selected_tokens", float(getattr(config, "_cats_selected_sum", 0.0)) / segments)
    setattr(config, "last_cats_sink_merges", float(getattr(config, "_cats_sink_sum", 0.0)) / segments)
    setattr(config, "last_cats_residual_merges", float(getattr(config, "_cats_residual_sum", 0.0)) / segments)
    setattr(config, "last_cats_mutual_rejected", float(getattr(config, "_cats_mutual_sum", 0.0)) / segments)
    setattr(config, "last_cats_margin_rejected", float(getattr(config, "_cats_margin_sum", 0.0)) / segments)
    setattr(config, "last_cats_threshold_rejected", float(getattr(config, "_cats_threshold_sum", 0.0)) / segments)
    setattr(config, "last_cats_retained_residual_tokens", float(getattr(config, "_cats_retained_sum", 0.0)) / segments)
    setattr(config, "last_cats_spatial_tokens_before", float(getattr(config, "_cats_spatial_before_sum", 0.0)) / segments)
    setattr(config, "last_cats_spatial_tokens_after", float(getattr(config, "_cats_spatial_after_sum", 0.0)) / segments)
    sim_count = float(getattr(config, "_cats_merge_sim_count", 0.0))
    margin_count = float(getattr(config, "_cats_margin_value_count", 0.0))
    setattr(config, "last_cats_mean_merge_sim", float(getattr(config, "_cats_merge_sim_sum", 0.0)) / sim_count if sim_count > 0 else None)
    setattr(config, "last_cats_mean_margin", float(getattr(config, "_cats_margin_value_sum", 0.0)) / margin_count if margin_count > 0 else None)
    adts_mode = str(getattr(config, "cats_adts_mode", "cats") or "cats").strip().lower()
    setattr(config, "last_cats_adts_mode_code", _CATS_ADTS_MODE_CODE.get(adts_mode, -1.0))


def cats_segment_compression(
    segment_features: torch.Tensor,
    segment_global_indices: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CATS-FlashVID segment compression.

    CATS keeps FlashVID's ADTS/STTM budget split, but lets ADTS tokens act as
    protected sinks during temporal merging. The retained ADTS tokens are
    gathered after sink merging, so their features include absorbed evidence.
    """
    from .utils import _graphvid_apply_final_cap, dpc_knn

    num_frames, num_visual_tokens, feat_dim = segment_features.shape
    device = segment_features.device
    per_frame_adts = int(flashvid_config.num_attn_div_tokens or 0)
    adts_mode = str(getattr(flashvid_config, "cats_adts_mode", "cats") or "cats").strip().lower()
    if adts_mode == "flashvid":
        selected_indices, selected_mask, importance = _flashvid_adts_selection(
            features=segment_features,
            cls_attention=cls_attention,
            per_frame_tokens=per_frame_adts,
            config=flashvid_config,
        )
    else:
        selected_indices, selected_mask, importance = _cats_adts_selection(
            features=segment_features,
            cls_attention=cls_attention,
            per_frame_tokens=per_frame_adts,
            config=flashvid_config,
        )
    updated_features, residual_mask, metrics = _cats_sink_tstm(
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

    temp_merged_token_list: List[torch.Tensor] = []
    temp_merged_global_indices_list: List[torch.Tensor] = []
    for frame_idx in range(num_frames):
        idx = torch.where(residual_mask[frame_idx])[0]
        temp_merged_token_list.append(updated_features[frame_idx, idx])
        temp_merged_global_indices_list.append(segment_global_indices[frame_idx, idx])

    all_tokens = [selected_features]
    all_global_indices = [selected_global_indices]
    num_other_tokens = int(flashvid_config.num_sttm_tokens or 0) * num_frames
    spatial_before = sum(int(x.shape[0]) for x in temp_merged_token_list)
    if num_other_tokens > 0:
        num_current_retained_tokens = max(1, spatial_before)
        adaptive_contextual_ratio = num_other_tokens / num_current_retained_tokens
        if adaptive_contextual_ratio < 1.0:
            num_frames_in_segment = len(temp_merged_token_list)
            max_num_tokens = max([len(tokens) for tokens in temp_merged_token_list] + [1])
            batched_tokens = torch.zeros(
                (num_frames_in_segment, max_num_tokens, feat_dim),
                dtype=segment_features.dtype,
                device=device,
            )
            valid_token_mask = torch.zeros((num_frames_in_segment, max_num_tokens), dtype=torch.bool, device=device)
            num_clusters_list = []
            k_list = []
            for i, temp_merged_tokens in enumerate(temp_merged_token_list):
                num_tokens = len(temp_merged_tokens)
                if num_tokens > 0:
                    batched_tokens[i, :num_tokens] = temp_merged_tokens
                    valid_token_mask[i, :num_tokens] = True
                num_clusters = math.ceil(num_tokens * adaptive_contextual_ratio)
                num_clusters_list.append(num_clusters)
                k_list.append(max(1, min(num_clusters, 7)) if num_clusters > 0 else 1)
            cluster_indices_list, cluster_center_indices_list = dpc_knn(
                features=batched_tokens,
                num_clusters=num_clusters_list,
                k=k_list,
                valid_token_mask=valid_token_mask,
            )
            for i, (temp_merged_tokens, temp_merged_global_indices) in enumerate(
                zip(temp_merged_token_list, temp_merged_global_indices_list)
            ):
                num_clusters = num_clusters_list[i]
                if num_clusters > 0 and temp_merged_tokens.numel() > 0:
                    cluster_indices = cluster_indices_list[i][: len(temp_merged_tokens)]
                    cluster_center_indices = cluster_center_indices_list[i]
                    aggregated_tokens = torch.zeros((num_clusters, feat_dim), dtype=segment_features.dtype, device=device)
                    aggregated_tokens.scatter_add_(0, cluster_indices.unsqueeze(-1).expand(-1, feat_dim), temp_merged_tokens)
                    counts = torch.bincount(cluster_indices, minlength=num_clusters).unsqueeze(-1).to(segment_features.dtype)
                    aggregated_tokens = aggregated_tokens / counts.clamp_min(1)
                    if temp_merged_global_indices.numel() > 0:
                        cluster_center_indices = cluster_center_indices.clamp(min=0, max=temp_merged_global_indices.shape[0] - 1)
                        if cluster_center_indices.numel() < num_clusters:
                            pad = cluster_center_indices[-1].repeat(num_clusters - cluster_center_indices.numel())
                            cluster_center_indices = torch.cat([cluster_center_indices, pad], dim=0)
                        elif cluster_center_indices.numel() > num_clusters:
                            cluster_center_indices = cluster_center_indices[:num_clusters]
                        global_token_indices = temp_merged_global_indices[cluster_center_indices]
                    else:
                        global_token_indices = torch.empty((0,), dtype=torch.long, device=device)
                    all_tokens.append(aggregated_tokens)
                    all_global_indices.append(global_token_indices)
                else:
                    all_tokens.append(temp_merged_tokens)
                    all_global_indices.append(temp_merged_global_indices)
        else:
            all_tokens.extend(temp_merged_token_list)
            all_global_indices.extend(temp_merged_global_indices_list)

    segment_final_tokens = torch.cat(all_tokens, dim=0)
    segment_final_global_indices = torch.cat(all_global_indices, dim=0)
    spatial_after = int(segment_final_tokens.shape[0]) - int(selected_features.shape[0])
    _accumulate_cats_metrics(
        flashvid_config,
        selected_count=int(selected_mask.sum().item()),
        metrics=metrics,
        before=spatial_before,
        after=spatial_after,
    )
    segment_final_tokens, segment_final_global_indices = _graphvid_apply_final_cap(
        segment_final_tokens=segment_final_tokens,
        segment_final_global_indices=segment_final_global_indices,
        segment_global_indices=segment_global_indices,
        cls_attention=cls_attention,
        num_visual_tokens=num_visual_tokens,
        flashvid_config=flashvid_config,
    )
    return segment_final_tokens, segment_final_global_indices
