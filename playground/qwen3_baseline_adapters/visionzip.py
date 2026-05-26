from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from flashvid.configuration_flashvid import FlashVidConfig

from .common import _cfg_float, _effective_ratio, _record_adapter_metrics


def visionzip_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """VisionZip dominant-token + contextual-token compression.

    This follows the released VisionZip core: select dominant tokens by attention,
    uniformly choose contextual target tokens from the remaining tokens, assign
    dropped tokens to their most similar contextual target, and emit
    target_hidden + aggregated_hidden for contextual tokens.
    """
    num_frames, num_visual_tokens, feat_dim = video_features.shape
    device = video_features.device
    dtype = video_features.dtype
    effective_ratio = _effective_ratio(flashvid_config)
    per_frame_target = max(1, min(num_visual_tokens, int(num_visual_tokens * effective_ratio)))
    dominant_ratio = max(0.0, min(1.0, _cfg_float(flashvid_config, "visionzip_dominant_ratio", 0.85)))
    dominant_num = min(per_frame_target, max(0, int(round(per_frame_target * dominant_ratio))))
    contextual_num = max(0, per_frame_target - dominant_num)

    global_indices = torch.arange(
        num_frames * num_visual_tokens,
        dtype=torch.long,
        device=device,
    ).view(num_frames, num_visual_tokens)

    all_tokens = []
    all_indices = []
    token_positions = torch.arange(num_visual_tokens, device=device)
    for frame_idx in range(num_frames):
        hidden_states = video_features[frame_idx]
        attn = cls_attention[frame_idx].float()
        frame_selected_tokens = []
        frame_selected_indices = []

        if dominant_num > 0:
            topk_indices = torch.topk(attn, k=dominant_num, dim=0).indices.sort().values
            frame_selected_tokens.append(hidden_states[topk_indices])
            frame_selected_indices.append(global_indices[frame_idx, topk_indices])
        else:
            topk_indices = torch.empty((0,), dtype=torch.long, device=device)

        if contextual_num > 0:
            dominant_mask = torch.zeros((num_visual_tokens,), dtype=torch.bool, device=device)
            if topk_indices.numel() > 0:
                dominant_mask[topk_indices] = True
            filtered_indices = token_positions[~dominant_mask]

            if filtered_indices.numel() > 0:
                contextual_num_eff = min(contextual_num, int(filtered_indices.numel()))
                hidden_states_filtered = hidden_states[filtered_indices]
                metric_normalized = F.normalize(hidden_states_filtered.float(), p=2, dim=-1, eps=1e-6)

                step = max(1, metric_normalized.shape[0] // contextual_num_eff)
                target_positions = torch.arange(
                    0,
                    metric_normalized.shape[0],
                    step,
                    device=device,
                )[:contextual_num_eff]
                target_tokens = metric_normalized[target_positions]
                target_hidden = hidden_states_filtered[target_positions]

                merge_mask = torch.ones((metric_normalized.shape[0],), dtype=torch.bool, device=device)
                merge_mask[target_positions] = False
                tokens_to_merge = metric_normalized[merge_mask]
                hidden_to_merge = hidden_states_filtered[merge_mask]

                if tokens_to_merge.numel() > 0:
                    similarity = torch.matmul(tokens_to_merge, target_tokens.transpose(0, 1))
                    assign_one_hot = torch.zeros(
                        tokens_to_merge.shape[0],
                        contextual_num_eff,
                        dtype=dtype,
                        device=device,
                    )
                    assign_one_hot.scatter_(1, similarity.argmax(dim=1).unsqueeze(-1), 1)
                    counts = assign_one_hot.sum(dim=0).clamp(min=1).unsqueeze(-1)
                    aggregated_hidden = torch.matmul(assign_one_hot.transpose(0, 1), hidden_to_merge) / counts
                    contextual_tokens = target_hidden + aggregated_hidden
                else:
                    contextual_tokens = target_hidden

                frame_selected_tokens.append(contextual_tokens.to(dtype))
                frame_selected_indices.append(global_indices[frame_idx, filtered_indices[target_positions]])

        if frame_selected_tokens:
            tokens = torch.cat(frame_selected_tokens, dim=0)
            indices = torch.cat(frame_selected_indices, dim=0)
            order = torch.argsort(indices)
            all_tokens.append(tokens[order])
            all_indices.append(indices[order])

    if not all_tokens:
        flat = video_features.view(-1, feat_dim)
        indices = torch.arange(flat.shape[0], dtype=torch.long, device=device)[:1]
        _record_adapter_metrics(flashvid_config, variant="visionzip", output_tokens=1, raw_tokens=flat.shape[0])
        return flat[indices], indices

    final_tokens = torch.cat(all_tokens, dim=0)
    final_indices = torch.cat(all_indices, dim=0)
    order = torch.argsort(final_indices)
    final_tokens = final_tokens[order]
    final_indices = final_indices[order]
    flashvid_config.vision_token_length = int(final_tokens.shape[0])
    flashvid_config.llm_token_length = None
    flashvid_config.visual_token_length = int(final_tokens.shape[0])
    _record_adapter_metrics(
        flashvid_config,
        variant="visionzip",
        output_tokens=int(final_tokens.shape[0]),
        raw_tokens=int(num_frames * num_visual_tokens),
    )
    return final_tokens, final_indices


