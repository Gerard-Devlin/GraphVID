from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F

from flashvid.configuration_flashvid import FlashVidConfig


SUPPORTED_EXTERNAL_BASELINES = ("fastvid", "visionzip")


def _cfg_float(config: FlashVidConfig, name: str, default: float) -> float:
    value = getattr(config, name, None)
    return float(default if value is None else value)


def _cfg_int(config: FlashVidConfig, name: str, default: int) -> int:
    value = getattr(config, name, None)
    return int(default if value is None else value)


def _effective_ratio(config: FlashVidConfig) -> float:
    ratio = float(getattr(config, "retention_ratio", 0.10))
    if bool(getattr(config, "external_budget_uses_expansion", True)):
        ratio *= float(getattr(config, "expansion", 1.0))
    return max(0.0, min(1.0, ratio))


def external_baseline_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    variant = str(getattr(flashvid_config, "compression_variant", "")).strip().lower()
    if variant == "fastvid":
        return fastvid_compression(video_features, cls_attention, flashvid_config)
    if variant == "visionzip":
        return visionzip_compression(video_features, cls_attention, flashvid_config)
    if variant == "prunevid":
        raise NotImplementedError(
            "PruneVid's released code path is PLLaVA/KV-cache based and the official repo "
            "does not provide a Qwen3/OneVision implementation. Refusing to run an "
            "approximation as PruneVid."
        )
    raise ValueError(f"unsupported external baseline variant={variant!r}")


def _record_external_metrics(config: FlashVidConfig, *, variant: str, output_tokens: int, raw_tokens: int) -> None:
    setattr(config, "last_external_variant", variant)
    setattr(config, "last_external_output_tokens", float(output_tokens))
    setattr(config, "last_external_raw_tokens", float(raw_tokens))


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
        _record_external_metrics(flashvid_config, variant="visionzip", output_tokens=1, raw_tokens=flat.shape[0])
        return flat[indices], indices

    final_tokens = torch.cat(all_tokens, dim=0)
    final_indices = torch.cat(all_indices, dim=0)
    order = torch.argsort(final_indices)
    final_tokens = final_tokens[order]
    final_indices = final_indices[order]
    flashvid_config.vision_token_length = int(final_tokens.shape[0])
    flashvid_config.llm_token_length = None
    flashvid_config.visual_token_length = int(final_tokens.shape[0])
    _record_external_metrics(
        flashvid_config,
        variant="visionzip",
        output_tokens=int(final_tokens.shape[0]),
        raw_tokens=int(num_frames * num_visual_tokens),
    )
    return final_tokens, final_indices


def _fastvid_segment_sizes(frame_global_features: torch.Tensor, config: FlashVidConfig) -> list[int]:
    frame_num = int(frame_global_features.shape[0])
    if frame_num <= 1:
        return [frame_num]
    dyseg_c = max(1, _cfg_int(config, "fastvid_DySeg_c", 8))
    dyseg_tau = _cfg_float(config, "fastvid_DySeg_tau", 0.90)
    frame_global_features = F.normalize(frame_global_features.float(), p=2, dim=1, eps=1e-6)
    similarity_matrix = (frame_global_features[:-1] * frame_global_features[1:]).sum(dim=1)

    topk_count = min(max(0, dyseg_c - 1), similarity_matrix.numel())
    if topk_count > 0:
        cut_indices_topk = torch.topk(similarity_matrix, topk_count, largest=False).indices
    else:
        cut_indices_topk = torch.empty((0,), dtype=torch.long, device=frame_global_features.device)
    cut_indices_cos = torch.nonzero(similarity_matrix < dyseg_tau, as_tuple=False).squeeze(1)
    cut_indices = torch.unique(torch.cat([cut_indices_topk, cut_indices_cos])).sort().values
    if cut_indices.numel() == 0:
        return [frame_num]

    padded = F.pad(cut_indices, (1, 1), value=-1)
    padded[-1] = frame_num - 1
    return [max(1, int(v)) for v in padded.diff().tolist() if int(v) > 0]


def fastvid_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FastVID DySeg + STPrune + DTM compression adapted to Qwen3 visual tokens.

    The pruning block mirrors the released LLaVA-OneVision/Qwen2.5-VL FastVID
    logic: dynamic temporal segmentation, attention-selected salient tokens,
    density-selected context tokens, and density token merge (DTM).
    """
    frame_num, frame_token_len, hidden_states_dim = video_features.shape
    device = video_features.device
    dtype = video_features.dtype
    fastvid_retention_ratio = _effective_ratio(flashvid_config)
    fastvid_STPrune_d = max(0.0, min(1.0, _cfg_float(flashvid_config, "fastvid_STPrune_d", 0.4)))
    fastvid_DTM_p = max(1, _cfg_int(flashvid_config, "fastvid_DTM_p", 4))
    fastvid_DTM_beta = _cfg_float(flashvid_config, "fastvid_DTM_beta", 0.6)

    alltoken_indices = torch.arange(
        frame_num * frame_token_len,
        dtype=torch.long,
        device=device,
    ).view(frame_num, frame_token_len)
    video_hidden_states = video_features
    frame_attn_weights = cls_attention.float()

    segment_sizes = _fastvid_segment_sizes(video_features.mean(dim=1), flashvid_config)

    frame_retain_num = int(frame_token_len * fastvid_retention_ratio)
    frame_retain_num = max(1, min(frame_token_len, frame_retain_num))
    frame_salient_num = frame_retain_num - int(frame_retain_num * fastvid_STPrune_d)
    frame_salient_num = max(0, min(frame_token_len, frame_salient_num))
    frame_context_num = max(0, frame_retain_num - frame_salient_num)

    batchframe_indices = torch.arange(frame_num, device=device).unsqueeze(1)
    frm_context_num_list = torch.zeros(frame_num, dtype=torch.int64, device=device)

    offset = 0
    for seg_i_len in segment_sizes:
        seg_context_num = frame_context_num * seg_i_len
        temp_num = (seg_i_len + fastvid_DTM_p - 1) // fastvid_DTM_p
        cur_frm_context_num = seg_context_num // max(1, temp_num)
        end = offset + seg_i_len
        seg_indices = torch.arange(seg_i_len - 1, -1, -1, device=device)
        mask = (seg_indices % fastvid_DTM_p == 0)
        frm_context_num_list[offset:end][mask] = cur_frm_context_num
        offset = end

    final_tokens = []
    keep_indexs = []

    if frame_salient_num > 0:
        salient_indexes = torch.topk(frame_attn_weights, frame_salient_num, dim=1).indices
        batch_indices = batchframe_indices.expand(-1, frame_salient_num)
        salient_tokens = video_hidden_states[batch_indices, salient_indexes]
        salient_global_indexes = alltoken_indices[batch_indices, salient_indexes]
        final_tokens.append(salient_tokens.view(-1, hidden_states_dim))
        keep_indexs.append(salient_global_indexes.view(-1))
    else:
        salient_indexes = torch.empty((frame_num, 0), dtype=torch.long, device=device)

    if frame_context_num > 0 and frame_token_len - frame_salient_num > 0:
        all_indices = torch.arange(frame_token_len, device=device).unsqueeze(0).expand(frame_num, -1)
        all_indices_mask = torch.ones_like(all_indices, dtype=torch.bool)
        if salient_indexes.numel() > 0:
            all_indices_mask.scatter_(1, salient_indexes, False)
        filtered_indices = all_indices[all_indices_mask].view(frame_num, frame_token_len - frame_salient_num)

        batch_indices = batchframe_indices.expand(-1, frame_token_len - frame_salient_num)
        token_filtered = video_hidden_states[batch_indices, filtered_indices]
        alltoken_filtered_indices = alltoken_indices[batch_indices, filtered_indices]

        tmp_frm_hidden_states = token_filtered
        dist_matrix = torch.cdist(tmp_frm_hidden_states.float(), tmp_frm_hidden_states.float()) / (
            hidden_states_dim ** 0.5
        )
        density_k = min(4, dist_matrix.shape[-1])
        dist_nearest, _ = torch.topk(dist_matrix, k=density_k, dim=-1, largest=False)
        density = (-(dist_nearest**2).mean(dim=-1)).exp()
        density = density + torch.rand(density.shape, device=device, dtype=density.dtype) * 1e-6

        density_mask = density[:, None, :] > density[:, :, None]
        density_mask = density_mask.type(tmp_frm_hidden_states.dtype)
        dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
        dist_0, _ = (dist_matrix * density_mask + dist_max * (1 - density_mask)).min(dim=-1)
        density_score = dist_0 * density

        context_k = min(frame_context_num, token_filtered.shape[1])
        if context_k > 0:
            sampled_indexs = torch.topk(density_score, k=context_k, dim=-1).indices

            batch_indices = batchframe_indices.expand(-1, context_k)
            frm_context_tokens = token_filtered[batch_indices, sampled_indexs]
            frm_context_global_indexes = alltoken_filtered_indices[batch_indices, sampled_indexs]

            to_be_merge_tokens = F.normalize(token_filtered.float(), p=2, dim=-1, eps=1e-6).to(dtype)
            merge_target_tokens = to_be_merge_tokens[batch_indices, sampled_indexs]

            similarity = torch.bmm(to_be_merge_tokens, merge_target_tokens.transpose(1, 2))
            assign_one_hot = torch.zeros(
                frame_num,
                frame_token_len - frame_salient_num,
                context_k,
                dtype=dtype,
                device=device,
            )
            assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)

            avg_weights = (1 / (assign_one_hot.sum(dim=1).unsqueeze(-1) + 1)).clamp(min=fastvid_DTM_beta)
            counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
            aggregated_hidden = torch.bmm(assign_one_hot.transpose(1, 2), token_filtered) / counts
            frm_context_tokens = avg_weights * frm_context_tokens + (1 - avg_weights) * aggregated_hidden

            context_for_frame_mask = frm_context_num_list == context_k
            if bool(context_for_frame_mask.any()):
                context_for_frame_tokens = frm_context_tokens[context_for_frame_mask]
                context_for_frame_global_indexes = frm_context_global_indexes[context_for_frame_mask]
                final_tokens.append(context_for_frame_tokens.view(-1, hidden_states_dim))
                keep_indexs.append(context_for_frame_global_indexes.view(-1))

            idx_seg_start = 0
            for seg_i_len in segment_sizes:
                if seg_i_len > 1:
                    cur_seg_context_num_list = frm_context_num_list[idx_seg_start : idx_seg_start + seg_i_len]
                    cur_seg_context_num = int(cur_seg_context_num_list[-1].item())
                    cur_seg_target_mask = cur_seg_context_num_list > context_k
                    cur_seg_target_num = int(cur_seg_target_mask.sum().item())
                    if cur_seg_target_num > 0 and cur_seg_context_num > 0:
                        cur_seg_density_score = density_score[idx_seg_start : idx_seg_start + seg_i_len]
                        cur_seg_density_score = cur_seg_density_score[cur_seg_target_mask]
                        cur_seg_token_filtered = token_filtered[idx_seg_start : idx_seg_start + seg_i_len]
                        cur_seg_token_target = cur_seg_token_filtered[cur_seg_target_mask]
                        cur_seg_token_filtered = cur_seg_token_filtered.view(1, -1, hidden_states_dim).expand(
                            cur_seg_target_num,
                            -1,
                            -1,
                        )
                        cur_seg_alltoken_indices = alltoken_filtered_indices[idx_seg_start : idx_seg_start + seg_i_len]
                        cur_seg_alltoken_indices = cur_seg_alltoken_indices[cur_seg_target_mask]

                        cur_k = min(cur_seg_context_num, cur_seg_density_score.shape[-1])
                        sampled_indexs = torch.topk(cur_seg_density_score, k=cur_k, dim=-1).indices
                        cur_batch_indices = torch.arange(cur_seg_target_num, device=device).unsqueeze(1).expand(-1, cur_k)
                        cur_context_tokens = cur_seg_token_target[cur_batch_indices, sampled_indexs]
                        cur_context_global_indexes = cur_seg_alltoken_indices[cur_batch_indices, sampled_indexs]

                        to_be_merge_tokens = F.normalize(cur_seg_token_filtered.float(), p=2, dim=-1, eps=1e-6).to(dtype)
                        merge_target_tokens = F.normalize(cur_context_tokens.float(), p=2, dim=-1, eps=1e-6).to(dtype)
                        similarity = torch.bmm(to_be_merge_tokens, merge_target_tokens.transpose(1, 2))
                        assign_one_hot = torch.zeros(
                            cur_seg_target_num,
                            to_be_merge_tokens.shape[1],
                            cur_k,
                            dtype=dtype,
                            device=device,
                        )
                        assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)

                        avg_weights = (1 / (assign_one_hot.sum(dim=1).unsqueeze(-1) + 1)).clamp(min=fastvid_DTM_beta)
                        counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
                        aggregated_hidden = torch.bmm(assign_one_hot.transpose(1, 2), cur_seg_token_filtered) / counts
                        cur_context_tokens = avg_weights * cur_context_tokens + (1 - avg_weights) * aggregated_hidden

                        final_tokens.append(cur_context_tokens.view(-1, hidden_states_dim))
                        keep_indexs.append(cur_context_global_indexes.view(-1))

                idx_seg_start += seg_i_len

    if not final_tokens:
        final_tokens = [video_features.reshape(-1, hidden_states_dim)[:1]]
        keep_indexs = [torch.zeros((1,), dtype=torch.long, device=device)]

    hidden_states = torch.cat(final_tokens, dim=0)
    keep_indices = torch.cat(keep_indexs, dim=0)
    sorted_indexs = torch.argsort(keep_indices)
    hidden_states = hidden_states[sorted_indexs]
    keep_indices = keep_indices[sorted_indexs]
    flashvid_config.vision_token_length = int(hidden_states.shape[0])
    flashvid_config.llm_token_length = None
    flashvid_config.visual_token_length = int(hidden_states.shape[0])
    setattr(flashvid_config, "last_fastvid_segment_count", float(len(segment_sizes)))
    setattr(flashvid_config, "last_fastvid_frame_retain_num", float(frame_retain_num))
    _record_external_metrics(
        flashvid_config,
        variant="fastvid",
        output_tokens=int(hidden_states.shape[0]),
        raw_tokens=int(frame_num * frame_token_len),
    )
    return hidden_states, keep_indices
