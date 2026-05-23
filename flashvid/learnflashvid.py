from __future__ import annotations

import math
from typing import Tuple

import torch

from .configuration_flashvid import FlashVidConfig
from .learned_selector import (
    build_scalar_token_features,
    load_selector_checkpoint,
    make_selection_mask,
    score_with_selector,
    topk_per_frame,
)
from .utils import ALL_TOKEN_SELECTION_METHOD, dpc_knn, spatiotemporal_compression


def _reset_learn_metrics(config: FlashVidConfig) -> None:
    for key in (
        "learn_selected_tokens",
        "learn_stable_tokens",
        "learn_selector_tokens",
        "learn_qaware_active",
        "learn_score_mean",
        "learn_score_std",
        "learn_teacher_keep_ratio",
    ):
        setattr(config, f"last_{key}", None)


def _get_selector(config: FlashVidConfig, device: torch.device):
    path = str(getattr(config, "learn_selector_ckpt", "") or "")
    cached_path = str(getattr(config, "_learn_selector_ckpt_path", "") or "")
    cached = getattr(config, "_learn_selector_model", None)
    if cached is not None and cached_path == path:
        return cached
    selector = load_selector_checkpoint(path, device)
    setattr(config, "_learn_selector_ckpt_path", path)
    setattr(config, "_learn_selector_model", selector)
    return selector


def _select_learned_adts(
    segment_features: torch.Tensor,
    cls_attention: torch.Tensor,
    question_features: torch.Tensor | None,
    flashvid_config: FlashVidConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_frames, num_visual_tokens, _ = segment_features.shape
    per_frame_budget = max(0, min(int(flashvid_config.num_attn_div_tokens or 0), num_visual_tokens))
    if per_frame_budget <= 0:
        empty = torch.empty((num_frames, 0), dtype=torch.long, device=segment_features.device)
        return empty, torch.zeros((num_frames, num_visual_tokens), dtype=torch.bool, device=segment_features.device)

    stable_ratio = min(max(float(getattr(flashvid_config, "learn_stable_floor_ratio", 0.50)), 0.0), 1.0)
    stable_k = min(per_frame_budget, int(math.ceil(per_frame_budget * stable_ratio)))
    selector_k = max(0, per_frame_budget - stable_k)

    if stable_k > 0:
        additional_kwargs = {"cls_attention": cls_attention} if "attn" in flashvid_config.token_selection_method else {}
        _, stable_indices = ALL_TOKEN_SELECTION_METHOD[flashvid_config.token_selection_method](
            features=segment_features,
            num_retained_tokens=stable_k,
            **additional_kwargs,
        )
    else:
        stable_indices = torch.empty((num_frames, 0), dtype=torch.long, device=segment_features.device)

    stable_mask = make_selection_mask(num_frames, num_visual_tokens, stable_indices, segment_features.device)
    density_topk = int(getattr(flashvid_config, "learn_density_topk", 8) or 8)
    scalar_features, aux = build_scalar_token_features(
        segment_features,
        cls_attention,
        question_features if bool(getattr(flashvid_config, "learn_qaware", True)) else None,
        density_topk=density_topk,
    )
    selector = _get_selector(flashvid_config, segment_features.device)
    learned_score = score_with_selector(
        selector,
        scalar_features,
        aux,
        blend=float(getattr(flashvid_config, "learn_score_blend", 0.50)),
        q_weight=float(getattr(flashvid_config, "learn_q_relevance_weight", 0.20)),
    )
    fill_indices = topk_per_frame(learned_score, selector_k, exclude=stable_mask)

    selected_rows = []
    for frame_idx in range(num_frames):
        parts = [stable_indices[frame_idx]]
        if fill_indices[frame_idx].numel() > 0:
            parts.append(fill_indices[frame_idx])
        row = torch.cat(parts, dim=0) if parts else torch.empty((0,), dtype=torch.long, device=segment_features.device)
        if row.numel() < per_frame_budget:
            row_mask = torch.zeros((num_frames, num_visual_tokens), dtype=torch.bool, device=segment_features.device)
            if row.numel() > 0:
                row_mask[frame_idx, row] = True
            fallback = topk_per_frame(
                learned_score,
                per_frame_budget - int(row.numel()),
                exclude=row_mask,
            )[frame_idx]
            row = torch.cat([row, fallback], dim=0)
        selected_rows.append(row[:per_frame_budget])
    selected_indices = torch.stack(selected_rows, dim=0)
    selected_mask = make_selection_mask(num_frames, num_visual_tokens, selected_indices, segment_features.device)

    setattr(flashvid_config, "last_learn_selected_tokens", float(int(selected_mask.sum().item())))
    setattr(flashvid_config, "last_learn_stable_tokens", float(num_frames * stable_k))
    setattr(flashvid_config, "last_learn_selector_tokens", float(num_frames * selector_k))
    setattr(flashvid_config, "last_learn_qaware_active", float(bool(getattr(flashvid_config, "learn_qaware", True)) and question_features is not None))
    setattr(flashvid_config, "last_learn_score_mean", float(learned_score.float().mean().item()))
    setattr(flashvid_config, "last_learn_score_std", float(learned_score.float().std(unbiased=False).item()))
    return selected_indices, selected_mask


def _spatial_merge_flashvid(
    temp_merged_token_list: list[torch.Tensor],
    temp_merged_global_indices_list: list[torch.Tensor],
    *,
    num_other_tokens: int,
    feat_dim: int,
    segment_features: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    num_current_retained_tokens = sum(len(tokens) for tokens in temp_merged_token_list)
    adapative_contextual_ratio = num_other_tokens / max(1, num_current_retained_tokens)
    if adapative_contextual_ratio >= 1.0 or not temp_merged_token_list:
        return temp_merged_token_list, temp_merged_global_indices_list

    num_frames_in_segment = len(temp_merged_token_list)
    max_num_tokens = max(len(tokens) for tokens in temp_merged_token_list)
    if max_num_tokens <= 0:
        return temp_merged_token_list, temp_merged_global_indices_list
    batched_tokens = torch.zeros(
        (num_frames_in_segment, max_num_tokens, feat_dim),
        dtype=segment_features.dtype,
        device=segment_features.device,
    )
    valid_token_mask = torch.zeros(
        (num_frames_in_segment, max_num_tokens),
        dtype=torch.bool,
        device=segment_features.device,
    )
    num_clusters_list = []
    k_list = []
    for i, temp_merged_tokens in enumerate(temp_merged_token_list):
        num_tokens = len(temp_merged_tokens)
        batched_tokens[i, :num_tokens] = temp_merged_tokens
        valid_token_mask[i, :num_tokens] = True
        num_clusters = math.ceil(num_tokens * adapative_contextual_ratio)
        num_clusters_list.append(num_clusters)
        k_list.append(min(num_clusters, 7))

    cluster_indices_list, cluster_center_indices_list = dpc_knn(
        features=batched_tokens,
        num_clusters=num_clusters_list,
        k=k_list,
        valid_token_mask=valid_token_mask,
    )
    out_tokens = []
    out_indices = []
    for i, (temp_tokens, temp_indices) in enumerate(zip(temp_merged_token_list, temp_merged_global_indices_list)):
        num_clusters = num_clusters_list[i]
        if num_clusters <= 0:
            out_tokens.append(temp_tokens)
            out_indices.append(temp_indices)
            continue
        cluster_indices = cluster_indices_list[i][: len(temp_tokens)]
        centers = cluster_center_indices_list[i]
        aggregated = torch.zeros((num_clusters, feat_dim), dtype=segment_features.dtype, device=segment_features.device)
        aggregated.scatter_add_(0, cluster_indices.unsqueeze(-1).expand(-1, feat_dim), temp_tokens)
        counts = torch.bincount(cluster_indices, minlength=num_clusters).unsqueeze(-1).to(segment_features.dtype)
        aggregated = aggregated / counts.clamp_min(1)
        if temp_indices.numel() > 0:
            centers = centers.clamp(min=0, max=temp_indices.shape[0] - 1)
            if centers.numel() < num_clusters:
                pad = centers[-1].repeat(num_clusters - centers.numel()) if centers.numel() else torch.zeros((num_clusters,), dtype=torch.long, device=segment_features.device)
                centers = torch.cat([centers, pad], dim=0)
            centers = centers[:num_clusters]
            global_indices = temp_indices[centers]
        else:
            global_indices = torch.zeros((num_clusters,), dtype=torch.long, device=segment_features.device)
        out_tokens.append(aggregated)
        out_indices.append(global_indices)
    return out_tokens, out_indices


def learn_segment_compression(
    segment_features: torch.Tensor,
    segment_global_indices: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
    question_features: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_frames, num_visual_tokens, feat_dim = segment_features.shape
    selected_indices, _ = _select_learned_adts(
        segment_features,
        cls_attention,
        question_features,
        flashvid_config,
    )
    selected_features = segment_features.gather(
        1,
        selected_indices.unsqueeze(-1).expand(-1, -1, feat_dim),
    )
    selected_global_indices = segment_global_indices.gather(1, index=selected_indices).view(-1)

    mask = torch.ones(num_frames, num_visual_tokens, dtype=torch.bool, device=segment_features.device)
    mask.scatter_(1, selected_indices, False)

    num_other_tokens = int(flashvid_config.num_sttm_tokens or 0) * num_frames
    if num_other_tokens > 0 and float(flashvid_config.temporal_threshold) < 1.0 and num_frames > 1:
        temp_tokens, temp_indices_local = spatiotemporal_compression(
            video_features=segment_features.clone(),
            temporal_threshold=float(flashvid_config.temporal_threshold),
            token_mask=mask,
            flashvid_config=flashvid_config,
        )
        temp_indices = [
            segment_global_indices.view(num_frames, -1)[i][local_idx]
            for i, local_idx in enumerate(temp_indices_local)
        ]
    elif num_other_tokens > 0:
        temp_tokens = [segment_features[i][mask[i]] for i in range(num_frames)]
        temp_indices = [segment_global_indices[i][mask[i]] for i in range(num_frames)]
    else:
        temp_tokens = []
        temp_indices = []

    if num_other_tokens > 0:
        temp_tokens, temp_indices = _spatial_merge_flashvid(
            temp_tokens,
            temp_indices,
            num_other_tokens=num_other_tokens,
            feat_dim=feat_dim,
            segment_features=segment_features,
        )

    all_tokens = [selected_features.reshape(-1, feat_dim)] + temp_tokens
    all_indices = [selected_global_indices] + temp_indices
    return torch.cat(all_tokens, dim=0), torch.cat(all_indices, dim=0)
