from typing import Optional, Tuple, Union, List, Iterable

import math
import torch
from torch.nn import functional as F
from .configuration_flashvid import FlashVidConfig
from .token_selection import (
    attn_based_token_selection,
    attn_div_based_token_selection,
    attn_div_v2_based_token_selection,
    div_based_token_selection,
    TokenSelectionMethod,
)

ALL_TOKEN_SELECTION_METHOD = {
    TokenSelectionMethod.ATTN: attn_based_token_selection,
    TokenSelectionMethod.ADTS_v2: attn_div_v2_based_token_selection,
    TokenSelectionMethod.ADTS: attn_div_based_token_selection,
    TokenSelectionMethod.DIV: div_based_token_selection,
}


def extract_question_features(
    input_ids: Optional[torch.Tensor],
    inputs_embeds: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    invalid_token_ids: Optional[Iterable[int]] = None,
    batch_index: int = 0,
) -> Optional[torch.Tensor]:
    """Extract text/question token embeddings for question-aware token reweighting."""
    if input_ids is None or inputs_embeds is None:
        return None
    if input_ids.ndim != 2 or inputs_embeds.ndim != 3:
        return None
    if batch_index >= input_ids.shape[0] or batch_index >= inputs_embeds.shape[0]:
        return None

    token_ids = input_ids[batch_index]
    token_mask = torch.ones_like(token_ids, dtype=torch.bool)

    if attention_mask is not None and attention_mask.ndim == 2:
        attn = attention_mask[batch_index]
        token_mask &= attn.bool() if attn.dtype == torch.bool else attn > 0

    if invalid_token_ids is not None:
        invalid_mask = torch.zeros_like(token_mask)
        for token_id in invalid_token_ids:
            if token_id is None:
                continue
            invalid_mask |= token_ids == int(token_id)
        token_mask &= ~invalid_mask

    if not token_mask.any():
        return None
    return inputs_embeds[batch_index, token_mask]


def _estimate_video_complexity(video_features: torch.Tensor) -> float:
    """Estimate video complexity in [0, 1] from temporal change + spatial dispersion."""
    num_frames = video_features.shape[0]
    normed_tokens = F.normalize(video_features.float(), p=2, dim=-1, eps=1e-6)

    frame_centers = F.normalize(normed_tokens.mean(dim=1), p=2, dim=-1, eps=1e-6)
    if num_frames > 1:
        temporal_sim = torch.sum(frame_centers[:-1] * frame_centers[1:], dim=-1)
        temporal_complexity = ((1.0 - temporal_sim).clamp(min=0.0, max=2.0) * 0.5).mean()
    else:
        temporal_complexity = torch.tensor(0.0, device=video_features.device)

    center_per_token = frame_centers.unsqueeze(1).expand_as(normed_tokens)
    spatial_sim = torch.sum(normed_tokens * center_per_token, dim=-1)
    spatial_complexity = ((1.0 - spatial_sim).clamp(min=0.0, max=2.0) * 0.5).mean()

    return float((0.6 * temporal_complexity + 0.4 * spatial_complexity).item())


def _estimate_question_difficulty(question_features: Optional[torch.Tensor]) -> float:
    """Estimate question difficulty in [0, 1] from length + semantic dispersion."""
    if question_features is None or question_features.numel() == 0:
        return 0.5

    q = F.normalize(question_features.float(), p=2, dim=-1, eps=1e-6)
    q_center = F.normalize(q.mean(dim=0), p=2, dim=-1, eps=1e-6)
    q_dispersion = 1.0 - torch.matmul(q, q_center).clamp(min=-1.0, max=1.0)
    semantic_difficulty = (q_dispersion.clamp(min=0.0, max=2.0) * 0.5).mean()
    length_difficulty = min(1.0, q.shape[0] / 32.0)

    return float(0.5 * semantic_difficulty.item() + 0.5 * length_difficulty)


def _resolve_effective_retention_ratio(
    video_features: torch.Tensor,
    question_features: Optional[torch.Tensor],
    flashvid_config: FlashVidConfig,
) -> float:
    base_ratio = float(flashvid_config.retention_ratio)
    if not bool(getattr(flashvid_config, "adaptive_token_budget", False)):
        flashvid_config.last_adaptive_retention_ratio = base_ratio
        return base_ratio

    candidate_ratios = sorted(
        [
            max(0.01, min(1.0, float(getattr(flashvid_config, "adaptive_budget_low", 0.10)))),
            max(0.01, min(1.0, float(getattr(flashvid_config, "adaptive_budget_mid", 0.15)))),
            max(0.01, min(1.0, float(getattr(flashvid_config, "adaptive_budget_high", 0.20)))),
        ]
    )

    video_complexity = _estimate_video_complexity(video_features)
    question_difficulty = _estimate_question_difficulty(question_features)
    complexity_score = 0.7 * video_complexity + 0.3 * question_difficulty
    level = min(len(candidate_ratios) - 1, int(complexity_score * len(candidate_ratios)))
    adaptive_ratio = candidate_ratios[level]

    flashvid_config.last_adaptive_retention_ratio = adaptive_ratio
    return adaptive_ratio


def flashvid_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
    question_features: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_frames, num_visual_tokens, feat_dim = video_features.shape
    compression_variant = str(getattr(flashvid_config, "compression_variant", "flashvid")).lower()
    if compression_variant == "talon":
        from .talon import talon_compression

        return talon_compression(
            video_features=video_features,
            cls_attention=cls_attention,
            flashvid_config=flashvid_config,
            question_features=question_features,
        )
    if compression_variant != "flashvid":
        raise ValueError(f"unsupported compression_variant={compression_variant!r}, expected flashvid|talon")

    retention_ratio = _resolve_effective_retention_ratio(
        video_features=video_features,
        question_features=question_features,
        flashvid_config=flashvid_config,
    )

    # 1. Partition the video frames into segments based on transition similarities.
    if flashvid_config.do_segment:
        segment_lengths = segment(
            video_features=video_features.mean(1),
            segment_threshold=flashvid_config.segment_threshold,
            min_segment_num=flashvid_config.min_segment_num,
            complementary_segment=flashvid_config.complementary_segment,
        )
    else:
        # Treat the whole video as a single segment.
        segment_lengths = torch.tensor([num_frames], dtype=torch.long, device=video_features.device)

    num_segments = segment_lengths.shape[0]
    global_indices = torch.arange(num_frames * num_visual_tokens, dtype=torch.long, device=video_features.device)

    # 2. Apply Attention and Diversity-based Token Selection(ADTS).
    token_budget = math.ceil(num_visual_tokens * retention_ratio * flashvid_config.expansion)
    num_attn_div_tokens = math.ceil(token_budget * flashvid_config.alpha)
    num_sttm_tokens = token_budget - num_attn_div_tokens
    # store in the config.
    flashvid_config.num_attn_div_tokens = num_attn_div_tokens
    flashvid_config.num_sttm_tokens = num_sttm_tokens

    all_segment_features = []
    all_segment_indices = []
    offset = 0
    for seg_idx in range(num_segments):
        seg_len = int(segment_lengths[seg_idx].item())
        segment_features = video_features[offset : offset + seg_len]
        segment_cls_attention = cls_attention[offset : offset + seg_len]
        segment_global_indices = global_indices.view(num_frames, num_visual_tokens)[offset : offset + seg_len]
        segment_features, segment_global_indices = segment_compression(
            segment_features=segment_features,
            segment_global_indices=segment_global_indices,
            cls_attention=segment_cls_attention,
            flashvid_config=flashvid_config,
        )
        all_segment_features.append(segment_features)
        all_segment_indices.append(segment_global_indices)
        offset += seg_len
    final_tokens = torch.cat(all_segment_features, dim=0)  # (num_final_tokens, feat_dim)
    final_global_indices = torch.cat(all_segment_indices, dim=0)  # (num_final_tokens,)

    sorted_indices = final_global_indices.argsort()
    sorted_tokens = final_tokens[sorted_indices]  # Sort by global indices.
    # Store the final token length in the `flashvid_config`.
    flashvid_config.visual_token_length = sorted_tokens.shape[0]
    # print(f"#Visual Tokens After Vision-Side Compression : {flashvid_config.visual_token_length}")
    return sorted_tokens, final_global_indices[sorted_indices]


def segment_compression(
    segment_features: torch.Tensor,
    segment_global_indices: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compress the segment features by applying Temporal Average Merging (TAM) and Spatial Merging.

    Args:
        segment_features (torch.Tensor): The features of the video segment, of shape (num_frames, num_visual_tokens, feat_dim).
        segment_global_indices (torch.Tensor): The global indices of the video segment, of shape (num_frames, num_visual_tokens).
        cls_attention (torch.Tensor): [CLS] attentions used for per-frame token selection, of shape (num_frames, num_visual_tokens).
        flashvid_config (FlashVidConfig): The configuration for FlashVid.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The final tokens and their global indices after compression.
    """
    num_frames, num_visual_tokens, feat_dim = segment_features.shape

    # 1. Apply Attention and Diversity-based Token Selection (ADTS).
    if flashvid_config.alpha > 0:
        additional_kwargs = {"cls_attention": cls_attention} if "attn" in flashvid_config.token_selection_method else {}
        selected_features, selected_indices = ALL_TOKEN_SELECTION_METHOD[flashvid_config.token_selection_method](
            features=segment_features,
            num_retained_tokens=flashvid_config.num_attn_div_tokens,
            **additional_kwargs,
        )
        selected_global_indices = segment_global_indices.gather(1, index=selected_indices).view(-1)
    else:
        # No token selection
        selected_features = torch.tensor([]).to(segment_features)
        selected_indices = torch.tensor([]).to(segment_global_indices)
        selected_global_indices = torch.tensor([]).to(segment_global_indices)

    mask = torch.ones(num_frames, num_visual_tokens, dtype=torch.bool, device=segment_features.device)
    mask.scatter_(1, selected_indices, False)

    num_other_tokens = flashvid_config.num_sttm_tokens * num_frames
    # 1. Apply Temporal Average Merging (TAM) to the segment features.
    if num_other_tokens > 0 and flashvid_config.temporal_threshold < 1.0:
        if num_frames > 1:
            temp_merged_token_list, temp_merged_indices_list = spatiotemporal_compression(
                video_features=segment_features,
                temporal_threshold=flashvid_config.temporal_threshold,
                token_mask=mask,
                flashvid_config=flashvid_config,
            )
            temp_merged_global_indices_list = [segment_global_indices.view(num_frames, -1)[i][temp_merged_indices] for i, temp_merged_indices in enumerate(temp_merged_indices_list)]
        else:
            # Single-frame segment, no temporal merging needed.
            temp_merged_token_list = [segment_features[0]]
            temp_merged_global_indices_list = [segment_global_indices[0]]
    else:
        # No spatial-temporal merging needed.
        temp_merged_token_list = []
        temp_merged_global_indices_list = []

    all_tokens = [selected_features.view(-1, feat_dim)]
    all_global_indices = [selected_global_indices]
    # 2. Apply Spatial Merging to the tokens after temporal merging.
    if num_other_tokens > 0: ## Only apply spatial merging when there are STTM tokens.
        # Calculate adaptive contextual ratio.
        num_current_retained_tokens = sum(len(tokens) for tokens in temp_merged_token_list)
        adapative_contextual_ratio = num_other_tokens / num_current_retained_tokens
        if adapative_contextual_ratio < 1.0:
            num_frames_in_segment = len(temp_merged_token_list)
            max_num_tokens = max(len(tokens) for tokens in temp_merged_token_list)
            batched_tokens = torch.zeros((num_frames_in_segment, max_num_tokens, feat_dim), dtype=segment_features.dtype, device=segment_features.device)
            valid_token_mask = torch.zeros((num_frames_in_segment, max_num_tokens), dtype=torch.bool, device=segment_features.device)
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
            for i, (temp_merged_tokens, temp_merged_global_indices) in enumerate(zip(temp_merged_token_list, temp_merged_global_indices_list)):
                num_clusters = num_clusters_list[i]
                if num_clusters > 0:
                    cluster_indices = cluster_indices_list[i][:len(temp_merged_tokens)]
                    cluster_center_indices = cluster_center_indices_list[i]
                    aggregated_tokens = torch.zeros((num_clusters, feat_dim), dtype=segment_features.dtype, device=segment_features.device)
                    aggregated_tokens.scatter_add_(0, cluster_indices.unsqueeze(-1).expand(-1, feat_dim), temp_merged_tokens)
                    cluster_counts = torch.bincount(cluster_indices, minlength=num_clusters).unsqueeze(-1).to(segment_features.dtype)
                    aggregated_tokens = aggregated_tokens / cluster_counts.clamp_min(1)

                    # Guard against rare invalid center indices from padded/batched clustering.
                    if temp_merged_global_indices.numel() > 0:
                        max_valid_center = temp_merged_global_indices.shape[0] - 1
                        cluster_center_indices = cluster_center_indices.clamp(min=0, max=max_valid_center)
                    else:
                        global_token_indices = torch.zeros((num_clusters,), dtype=torch.long, device=segment_features.device)
                        all_tokens.append(aggregated_tokens)
                        all_global_indices.append(global_token_indices)
                        continue

                    if cluster_center_indices.numel() < num_clusters:
                        pad_num = num_clusters - cluster_center_indices.numel()
                        pad_value = cluster_center_indices[-1] if cluster_center_indices.numel() > 0 else torch.tensor(0, device=segment_features.device)
                        pad_tensor = pad_value.repeat(pad_num)
                        cluster_center_indices = torch.cat([cluster_center_indices, pad_tensor], dim=0)
                    elif cluster_center_indices.numel() > num_clusters:
                        cluster_center_indices = cluster_center_indices[:num_clusters]

                    global_token_indices = temp_merged_global_indices[cluster_center_indices]
                else:
                    aggregated_tokens = temp_merged_tokens
                    global_token_indices = temp_merged_global_indices
                    
                all_tokens.append(aggregated_tokens)
                all_global_indices.append(global_token_indices)
        else:
            for temp_merged_tokens, temp_merged_global_indices in zip(temp_merged_token_list, temp_merged_global_indices_list):
                all_tokens.append(temp_merged_tokens)
                all_global_indices.append(temp_merged_global_indices)

    segment_final_tokens = torch.cat(all_tokens, dim=0)  # (num_final_tokens, feat_dim)
    segment_final_global_indices = torch.cat(all_global_indices, dim=0)  # (num_final_tokens,)
    return segment_final_tokens, segment_final_global_indices


def segment(
    video_features: torch.Tensor,
    segment_threshold: float,
    min_segment_num: int,
    complementary_segment: bool = True,
) -> torch.Tensor:
    """Segments the video features into distinct segments based on similarity.

    Args:
        video_features (torch.Tensor): The video features to segment.
        segment_threshold (float): The threshold for segmenting.
        min_segment_num (int): The minimum number of segments required.
        complementary_segment (int): Use complementary segmentation to ensure `min_segment_num` constraint.

    Returns:
        torch.Tensor: The lengths of the segments.
    """
    num_frames, feat_dim = video_features.shape

    # 0. Calculate transition similarities
    normed_video_features = video_features / video_features.norm(p=2, dim=-1, keepdim=True)
    transition_similarities = torch.sum(normed_video_features[:-1] * normed_video_features[1:], dim=-1)

    # 1. Find cut indices based on the segment threshold
    cut_indices = torch.where(transition_similarities < segment_threshold)[0]

    # 2. Ensure at least `min_segment_num` segments (Top-K or Uniform complementary segment)
    segment_lengths = additional_segment(
        cut_indices=cut_indices,
        num_frames=num_frames,
        min_segment_num=min_segment_num,
        transition_similarities=transition_similarities,
        segment_threshold=segment_threshold,
        complementary_segment=complementary_segment,
    )
    return segment_lengths


def additional_segment(
    cut_indices: torch.Tensor,
    num_frames: int,
    min_segment_num: int,
    transition_similarities: torch.Tensor,
    segment_threshold: float,
    complementary_segment: bool = True,
):
    num_segments = cut_indices.numel() + 1
    if num_segments < min_segment_num and complementary_segment:
        num_remaining_cut_indices = min_segment_num - num_segments
        transition_similarities[transition_similarities < segment_threshold] = 1.0
        complementary_cut_indices = torch.topk(transition_similarities, k=min(num_remaining_cut_indices, transition_similarities.shape[0]), largest=False).indices
        cut_indices = torch.cat([cut_indices, complementary_cut_indices]).sort().values

    padded_cut_indices = F.pad(cut_indices, (1, 1), value=0)
    padded_cut_indices[0] = -1
    padded_cut_indices[-1] = num_frames - 1
    segment_lengths = torch.diff(padded_cut_indices, n=1, dim=0)
    # print(f"segment lengths: {segment_lengths}")
    return segment_lengths


@torch.no_grad()
def dpc_knn(features: torch.Tensor, num_clusters: Union[int, List[int]], k: Union[int, List[int]] = 7, valid_token_mask: Optional[torch.Tensor] = None) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Union[torch.Tensor, List[torch.Tensor]]]:
    """Apply DPC-kNN clustering algorithm to the pooled image features, generating preliminary clustering result.

    Args:
        features (torch.Tensor): Pooled image features (temporal features), of shape (batch_size, seq_len, feat_dim).
        num_clusters (int or List[int]): The number of clusters. If a list, specifies the number of clusters for each batch element.
        k (int or List[int]): The number of nearest neighbors to consider for local density. Default is 7.
        valid_token_mask (Optional[torch.Tensor]): Boolean Mask indicating valid tokens, of shape (batch_size, seq_len). Default is None.

    Returns:
        Tuple[Union[torch.Tensor, List[torch.Tensor]], Union[torch.Tensor, List[torch.Tensor]]]: 
            Cluster indices and cluster center indices. If num_clusters is a list, returns lists of tensors.
    """
    invalid_token_mask = ~valid_token_mask if valid_token_mask is not None else None
    bsz, seq_len, feat_dim = features.shape

    # Calculate euclidean distance and local density
    dists = torch.cdist(features.float(), features.float()) / math.sqrt(feat_dim)

    # Mask out invalid tokens
    if valid_token_mask is not None:
        dists = torch.masked_fill(dists, invalid_token_mask.unsqueeze(1).expand(-1, seq_len, -1), dists.max() + 1)
        
    max_k = max(k) if isinstance(k, list) else k
    nearest_dist = torch.topk(dists, k=max_k, dim=-1, largest=False).values
    
    if isinstance(k, list):
        density = torch.empty((bsz, seq_len), device=features.device, dtype=features.dtype)
        for i in range(bsz):
            density[i] = torch.mean(-(nearest_dist[i, :, :k[i]]**2), dim=-1).exp()
    else:
        density = torch.mean(-(nearest_dist**2), dim=-1).exp()

    # ! [DISABLED] Add little random noise to ensure no tokens have the same density.
    # density = density + torch.rand_like(density, device=density.device, dtype=density.dtype) * 1e-6

    # Ensure the density of the empty token be 0
    if valid_token_mask is not None:
        density = torch.masked_fill(density, invalid_token_mask, 0.0)

    # Obtain the minimum distance to the point with higher density.
    mask = density[:, None, :] > density[:, :, None]
    max_dist = dists.view(bsz, -1).max(dim=-1)[0].view(-1, 1, 1)
    modified_dists = torch.where(mask, dists, max_dist)
    dist, _ = torch.min(modified_dists, dim=-1)

    # Calculate clustering score (clustering centers have the highest score)
    score = dist * density
    if isinstance(num_clusters, int):
        cluster_center_indices = torch.topk(score, k=num_clusters, dim=-1).indices
        # Obtain the distance matrix w.r.t cluster centers (batch_size, seq_len, num_clusters)
        dists = torch.gather(dists, dim=-1, index=cluster_center_indices.unsqueeze(1).expand(-1, seq_len, -1))
        cluster_indices = torch.argmin(dists, dim=-1)
        # Ensure each cluster center to merge with itself
        cluster_indices.scatter_(
            dim=-1,
            index=cluster_center_indices,
            src=torch.arange(num_clusters).to(cluster_indices).unsqueeze(0).expand(bsz, -1),
        )
        return cluster_indices, cluster_center_indices
    else:
        cluster_indices_list = []
        cluster_center_indices_list = []
        for i in range(bsz):
            k_i = num_clusters[i]
            if k_i == 0:
                cluster_center_indices_list.append(torch.tensor([], dtype=torch.long, device=features.device))
                cluster_indices_list.append(torch.zeros(seq_len, dtype=torch.long, device=features.device))
                continue
            cc_idx = torch.topk(score[i], k=k_i, dim=-1).indices
            cluster_center_indices_list.append(cc_idx)
            dists_i = torch.gather(dists[i], dim=-1, index=cc_idx.unsqueeze(0).expand(seq_len, -1))
            c_idx = torch.argmin(dists_i, dim=-1)
            c_idx.scatter_(
                dim=-1,
                index=cc_idx,
                src=torch.arange(k_i).to(c_idx),
            )
            cluster_indices_list.append(c_idx)
        return cluster_indices_list, cluster_center_indices_list


def spatiotemporal_compression(
    video_features: torch.Tensor,
    temporal_threshold: float,
    token_mask: torch.Tensor,
    flashvid_config: FlashVidConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_frames, num_visual_tokens, feat_dim = video_features.shape
    # since we pass the whole segment features, the lower bound should contain ADTS tokens.
    lower_bound = (flashvid_config.num_attn_div_tokens + flashvid_config.num_sttm_tokens) * num_frames
    normed_video_features = video_features / video_features.norm(p=2, dim=-1, keepdim=True)
    cosine_similarities = torch.bmm(normed_video_features[1:], normed_video_features[:-1].transpose(1, 2))
    # Mask out the selected tokens.
    cosine_similarities[~token_mask[1:].unsqueeze(-1).expand(-1, -1, num_visual_tokens)] = -1.0
    cosine_similarities[~token_mask[:-1].unsqueeze(1).expand(-1, num_visual_tokens, -1)] = -1.0

    max_sims, max_sim_indices = torch.max(cosine_similarities, dim=-1)

    padded_max_sims = F.pad(max_sims, (0, 0, 1, 0), value=-1)
    padded_max_sim_indices = F.pad(max_sim_indices, (0, 0, 1, 0), value=-1)

    token_counts = torch.ones(num_frames, num_visual_tokens).to(video_features)
    mask = padded_max_sims > temporal_threshold
    retaining_token_mask = ~mask

    # Ensure the number of retained tokens after TAM does not exceed the lower bound.
    if retaining_token_mask.int().sum() < lower_bound:
        soft_threshold = padded_max_sims.view(-1).topk(k=(num_frames * num_visual_tokens) - lower_bound).values[-1]
        soft_threshold = max(soft_threshold, -1.0 + 1e-6)
        mask = padded_max_sims > soft_threshold
        retaining_token_mask = ~mask

    for frame_idx in range(num_frames - 1, -1, -1):
        frame_features = video_features[frame_idx]
        frame_token_counts = token_counts[frame_idx]
        frame_max_sim_indices = padded_max_sim_indices[frame_idx]

        # Apply spatiotemporal average merging.
        tokens_to_merge = frame_features[~mask[frame_idx]]
        to_merge_token_counts = frame_token_counts[~mask[frame_idx]]
        if tokens_to_merge.numel() > 0:
            aggregated_tokens = tokens_to_merge / to_merge_token_counts.unsqueeze(-1).to(tokens_to_merge.dtype)
            video_features[frame_idx][~mask[frame_idx]] = aggregated_tokens
            token_counts[frame_idx][~mask[frame_idx]] = 1

        # other tokens are connected to the previous frame's tokens
        other_tokens = frame_features[mask[frame_idx]]
        if other_tokens.numel() > 0:
            # Distribute other tokens to the previous frame's tokens (anchor tokens)
            anchor_token_indices = frame_max_sim_indices[mask[frame_idx]]
            aggregated_tokens = torch.zeros((num_visual_tokens, feat_dim), dtype=video_features.dtype, device=video_features.device)
            aggregated_tokens.scatter_add_(0, anchor_token_indices.unsqueeze(-1).expand(-1, feat_dim), other_tokens)  # (num_visual_tokens, feat_dim)
            aggregated_token_counts = torch.bincount(anchor_token_indices, minlength=num_visual_tokens).to(video_features.dtype)  # (num_visual_tokens,)
            video_features[frame_idx - 1] += aggregated_tokens
            token_counts[frame_idx - 1] += aggregated_token_counts
            token_counts[frame_idx][mask[frame_idx]] = 0

    # Filter final tokens
    final_tokens = []
    retained_token_indices = []
    for i in range(num_frames):
        frame_mask = retaining_token_mask[i] & token_mask[i]
        frame_retained_tokens = video_features[i][frame_mask]  # (frame_retained_tokens_num, feat_dim)
        frame_retained_indices = torch.where(frame_mask)[0]  # (frame_retained_tokens_num,)
        final_tokens.append(frame_retained_tokens)
        retained_token_indices.append(frame_retained_indices)

    return final_tokens, retained_token_indices


def maybe_apply_decode_policy(
    hidden_states: torch.Tensor,
    causal_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    cache_position: Optional[torch.Tensor],
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    flashvid_config: FlashVidConfig,
    layer_idx: int,
    is_prefill: bool,
):
    """Decode-stage scaffold for Route3; default policy keeps behavior unchanged."""
    if is_prefill:
        return hidden_states, causal_mask, position_ids, cache_position, position_embeddings

    policy = str(getattr(flashvid_config, "decode_policy", "none") or "none").strip().lower()
    if policy in ("none", "off", "disabled"):
        return hidden_states, causal_mask, position_ids, cache_position, position_embeddings

    # Reserved policy name for future implementation; currently no-op with strict validation.
    if policy == "static_kv_budget":
        budget = float(getattr(flashvid_config, "decode_kv_budget_ratio", 1.0))
        update_interval = int(getattr(flashvid_config, "decode_update_interval", 4))
        start_layer = int(getattr(flashvid_config, "decode_start_layer", 0))
        if not (0.0 < budget <= 1.0):
            raise ValueError(f"decode_kv_budget_ratio must be in (0, 1], got {budget}")
        if update_interval <= 0:
            raise ValueError(f"decode_update_interval must be > 0, got {update_interval}")
        if start_layer < 0:
            raise ValueError(f"decode_start_layer must be >= 0, got {start_layer}")
        return hidden_states, causal_mask, position_ids, cache_position, position_embeddings

    raise ValueError(
        f"Unsupported decode_policy={policy!r}. "
        "Supported policies: none, static_kv_budget (scaffold/no-op)."
    )


def fastv_prune(
    hidden_states: torch.Tensor,
    causal_mask: Optional[torch.Tensor],
    attentions: Optional[torch.Tensor],
    cache_position: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    flashvid_config: FlashVidConfig,
    visual_pos_masks: Optional[torch.BoolTensor] = None,
):
    bsz, seq_length, _ = hidden_states.shape
    device = hidden_states.device
    # No-op shortcut: disable inner-LLM pruning when retention ratio is >= 1.
    if float(getattr(flashvid_config, "llm_retention_ratio", 1.0)) >= 0.9999:
        keep_indices = torch.arange(seq_length, device=device, dtype=torch.long)
        return hidden_states, causal_mask, position_ids, cache_position, position_embeddings, keep_indices

    # Obtain FlashVid arguments.
    visual_token_start_index = flashvid_config.visual_token_start_index
    visual_token_length = flashvid_config.visual_token_length
    visual_token_end_index = visual_token_start_index + visual_token_length

    retention_ratio = flashvid_config.llm_retention_ratio

    # Compatible to LLaVA-OneVision.
    if visual_pos_masks is None:
        visual_pos_masks = torch.zeros((bsz, seq_length), dtype=torch.bool, device=device)
        visual_pos_masks[:, visual_token_start_index:visual_token_end_index] = True
    non_visual_pos_masks = ~visual_pos_masks

    visual_features = hidden_states[visual_pos_masks, :]
    visual_global_indices = torch.where(visual_pos_masks[0])[0]
    non_visual_global_indices = torch.where(non_visual_pos_masks[0])[0]
    if visual_features.numel() == 0 or visual_global_indices.numel() == 0:
        return hidden_states, causal_mask, position_ids, cache_position, position_embeddings, torch.arange(seq_length, device=device)

    num_retained_tokens = math.ceil(visual_token_length * retention_ratio)
    num_retained_tokens = min(max(1, num_retained_tokens), int(visual_global_indices.numel()))
    attn = torch.mean(attentions[:, :, -1, :], dim=1)[visual_pos_masks]

    _, topk_indices = attn_based_token_selection(
        features=visual_features.unsqueeze(0),
        cls_attention=attn.unsqueeze(0),
        num_retained_tokens=num_retained_tokens,
    )
    topk_indices = topk_indices.squeeze(0)
    topk_indices = topk_indices.clamp(min=0, max=max(0, int(visual_global_indices.numel()) - 1))
    all_global_indices = [non_visual_global_indices, visual_global_indices[topk_indices]]
    keep_indices = torch.sort(torch.cat(all_global_indices).unique()).values
    keep_indices = keep_indices[keep_indices < seq_length]
    if keep_indices.numel() == 0:
        keep_indices = torch.arange(seq_length, device=device)

    # Filter
    hidden_states = hidden_states[:, keep_indices]
    # Keep RoPE positions by selecting with keep_indices, but reset cache positions to
    # contiguous range to avoid out-of-range indexing in cache update paths.
    if cache_position is None:
        cache_position = torch.arange(hidden_states.shape[1], device=device, dtype=torch.long)
    else:
        cache_position = torch.arange(hidden_states.shape[1], device=device, dtype=cache_position.dtype)
    position_ids = keep_indices.unsqueeze(0) if position_ids is None else position_ids[..., keep_indices].contiguous()
    position_embeddings = (
        position_embeddings[0][..., keep_indices, :].contiguous(),
        position_embeddings[1][..., keep_indices, :].contiguous(),
    )

    new_seq_length = hidden_states.shape[1]
    if causal_mask is not None:
        # Use index-select instead of naive truncation because keep_indices is a sparse subset.
        causal_mask = causal_mask.index_select(2, keep_indices).index_select(3, keep_indices)
    # Update flashvid config.
    flashvid_config.visual_token_length = num_retained_tokens
    return hidden_states, causal_mask, position_ids, cache_position, position_embeddings, keep_indices
