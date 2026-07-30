from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from flashvid.configuration_flashvid import FlashVidConfig

from .common import _cfg_float, _cfg_int, _effective_ratio, _record_adapter_metrics


def _segment_sizes_from_cuts(num_frames: int, cut_indices: torch.Tensor) -> list[int]:
    cut_indices = cut_indices.to(dtype=torch.long)
    cut_indices = cut_indices[(cut_indices >= 0) & (cut_indices < num_frames - 1)]
    cut_indices = torch.unique(cut_indices, sorted=True)
    if cut_indices.numel() == 0:
        return [num_frames]

    sizes = [int(cut_indices[0].item()) + 1]
    for index in range(1, int(cut_indices.numel())):
        sizes.append(int(cut_indices[index].item() - cut_indices[index - 1].item()))
    sizes.append(int(num_frames - cut_indices[-1].item() - 1))
    return [size for size in sizes if size > 0]


def _dpc_knn_centers(tokens: torch.Tensor, cluster_num: int, k: int = 4) -> torch.Tensor:
    if cluster_num <= 0 or tokens.shape[0] == 0:
        return torch.empty((0,), dtype=torch.long, device=tokens.device)
    cluster_num = min(int(cluster_num), int(tokens.shape[0]))
    k = max(1, min(int(k), int(tokens.shape[0])))

    batched = tokens.unsqueeze(0)
    distances = torch.cdist(batched.float(), batched.float()) / (tokens.shape[-1] ** 0.5)
    nearest = torch.topk(distances, k=k, dim=-1, largest=False).values
    density = (-(nearest**2).mean(dim=-1)).exp()
    density = density + torch.rand_like(density) * 1e-6

    denser = (density[:, None, :] > density[:, :, None]).to(batched.dtype)
    distance_max = distances.flatten(1).max(dim=-1).values[:, None, None]
    distance_to_denser = (distances * denser + distance_max * (1 - denser)).min(dim=-1).values
    centers = torch.topk(distance_to_denser * density, k=cluster_num, dim=-1).indices[0]
    return centers.sort().values


def _allocate_inner_budgets(
    segment_sizes: list[int],
    frame_tokens: int,
    salient_tokens: int,
    context_tokens: int,
    dtm_period: int,
) -> tuple[list[int], list[int]]:
    num_segments = len(segment_sizes)
    if num_segments == 0:
        return [], []

    salient_base, salient_remainder = divmod(salient_tokens, num_segments)
    salient_by_segment = [
        salient_base + (1 if index < salient_remainder else 0)
        for index in range(num_segments)
    ]

    context_slots = max(1, (num_segments + dtm_period - 1) // dtm_period)
    context_base, context_remainder = divmod(context_tokens, context_slots)
    context_per_slot = [
        context_base + (1 if index < context_remainder else 0)
        for index in range(context_slots)
    ]

    context_by_segment: list[int] = []
    slot_index = 0
    for segment_index in range(num_segments):
        if segment_index % dtm_period == 0 and slot_index < len(context_per_slot):
            context_by_segment.append(min(context_per_slot[slot_index], frame_tokens // 2))
            slot_index += 1
        else:
            context_by_segment.append(0)
    context_by_segment.reverse()

    frame_salient: list[int] = []
    frame_context: list[int] = []
    for segment_index, segment_size in enumerate(segment_sizes):
        frame_salient.extend([0] * (segment_size - 1))
        frame_context.extend([0] * (segment_size - 1))
        context_count = context_by_segment[segment_index]
        frame_context.append(context_count)
        frame_salient.append(
            min(salient_by_segment[segment_index], frame_tokens - context_count)
        )

    return frame_salient, frame_context


def fastvid_qwen25_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Released Qwen2.5 FastVID: two-level DySeg, STPrune, and DTM."""

    num_frames, frame_tokens, hidden_dim = video_features.shape
    device = video_features.device
    dtype = video_features.dtype
    total_tokens = num_frames * frame_tokens
    ratio = _effective_ratio(flashvid_config)
    if ratio >= 1.0:
        setattr(flashvid_config, "_fastvid_frame_global_features", None)
        keep = torch.arange(total_tokens, dtype=torch.long, device=device)
        return video_features.reshape(total_tokens, hidden_dim), keep

    frame_global = getattr(flashvid_config, "_fastvid_frame_global_features", None)
    if (
        frame_global is None
        or frame_global.ndim != 2
        or int(frame_global.shape[0]) != num_frames
    ):
        raise RuntimeError(
            "Qwen2.5 FastVID requires post-merger frame-global features from the vision hook"
        )
    if cls_attention.shape != video_features.shape[:2]:
        raise RuntimeError(
            "Qwen2.5 FastVID attention shape mismatch: "
            f"attention={tuple(cls_attention.shape)}, features={tuple(video_features.shape[:2])}"
        )

    frame_global = F.normalize(frame_global.to(device=device).float(), dim=-1, eps=1e-6)
    setattr(flashvid_config, "_fastvid_frame_global_features", None)
    adjacent_similarity = (frame_global[:-1] * frame_global[1:]).sum(dim=-1)
    coarse_count = min(
        max(0, _cfg_int(flashvid_config, "fastvid_DySeg_c", 8) - 1),
        int(adjacent_similarity.numel()),
    )
    coarse_cuts = (
        torch.topk(adjacent_similarity, k=coarse_count, largest=False).indices
        if coarse_count > 0
        else torch.empty((0,), dtype=torch.long, device=device)
    )
    threshold_cuts = torch.nonzero(
        adjacent_similarity < _cfg_float(flashvid_config, "fastvid_DySeg_tau", 0.84),
        as_tuple=False,
    ).squeeze(-1)
    coarse_sizes = _segment_sizes_from_cuts(
        num_frames,
        torch.cat([coarse_cuts, threshold_cuts]),
    )

    frame_retain = max(1, min(frame_tokens, int(frame_tokens * ratio)))
    context_ratio = max(
        0.0,
        min(1.0, _cfg_float(flashvid_config, "fastvid_STPrune_d", 0.40)),
    )
    inner_threshold = _cfg_float(flashvid_config, "fastvid_DySeg_ignore", 0.95)
    dtm_period = max(1, _cfg_int(flashvid_config, "fastvid_DTM_p", 4))
    dtm_beta = max(
        0.0,
        min(1.0, _cfg_float(flashvid_config, "fastvid_DTM_beta", 0.60)),
    )

    all_indices = torch.arange(total_tokens, dtype=torch.long, device=device).view(
        num_frames,
        frame_tokens,
    )
    compressed_segments: list[torch.Tensor] = []
    kept_segments: list[torch.Tensor] = []
    frame_offset = 0

    for coarse_size in coarse_sizes:
        segment_features = video_features[frame_offset : frame_offset + coarse_size]
        segment_attention = cls_attention[frame_offset : frame_offset + coarse_size].float()
        segment_global = frame_global[frame_offset : frame_offset + coarse_size]
        segment_length = int(segment_features.shape[0])
        segment_budget = max(
            1,
            min(segment_length * frame_tokens, frame_retain * segment_length),
        )
        context_budget = (
            max(1, int(segment_budget * context_ratio))
            if segment_budget > 1
            else segment_budget
        )
        context_budget = min(context_budget, segment_budget)
        salient_budget = segment_budget - context_budget

        if segment_length == 1:
            frame_salient = [salient_budget]
            frame_context = [context_budget]
        else:
            inner_similarity = (segment_global[:-1] * segment_global[1:]).sum(dim=-1)
            inner_cuts = torch.nonzero(
                inner_similarity < inner_threshold,
                as_tuple=False,
            ).squeeze(-1)
            inner_sizes = _segment_sizes_from_cuts(segment_length, inner_cuts)
            frame_salient, frame_context = _allocate_inner_budgets(
                inner_sizes,
                frame_tokens,
                salient_budget,
                context_budget,
                dtm_period,
            )

        salient_indices: list[torch.Tensor] = []
        context_indices: list[torch.Tensor] = []
        for local_frame in range(segment_length):
            salient_count = min(max(0, int(frame_salient[local_frame])), frame_tokens)
            context_count = min(
                max(0, int(frame_context[local_frame])),
                frame_tokens - salient_count,
            )

            top_indices = None
            if salient_count > 0:
                top_indices = torch.topk(
                    segment_attention[local_frame],
                    k=salient_count,
                    largest=True,
                    sorted=False,
                ).indices
                salient_indices.append(top_indices + local_frame * frame_tokens)

            if context_count > 0:
                local_indices = torch.arange(frame_tokens, device=device)
                remaining = (
                    local_indices[~torch.isin(local_indices, top_indices)]
                    if top_indices is not None
                    else local_indices
                )
                if remaining.numel() > 0:
                    centers = _dpc_knn_centers(
                        segment_features[local_frame, remaining],
                        context_count,
                        k=4,
                    )
                    context_indices.append(remaining[centers] + local_frame * frame_tokens)

        flat_features = segment_features.reshape(segment_length * frame_tokens, hidden_dim)
        segment_all = torch.arange(segment_length * frame_tokens, device=device)
        salient = (
            torch.cat(salient_indices)
            if salient_indices
            else segment_all.new_empty((0,))
        )
        context = (
            torch.cat(context_indices)
            if context_indices
            else segment_all.new_empty((0,))
        )

        if salient.numel() == 0 and context.numel() == 0:
            fallback = torch.topk(
                segment_attention.flatten(),
                k=segment_budget,
                largest=True,
                sorted=False,
            ).indices.sort().values
            compressed_segments.append(flat_features[fallback])
            kept_segments.append(fallback + frame_offset * frame_tokens)
            frame_offset += coarse_size
            continue

        salient_hidden = (
            flat_features[salient]
            if salient.numel() > 0
            else flat_features.new_empty((0, hidden_dim))
        )
        normalized = F.normalize(flat_features.float(), dim=-1, eps=1e-6)
        merged_context: list[torch.Tensor] = []
        for context_group in context_indices:
            retained = torch.cat([salient, context_group])
            merge_indices = segment_all[~torch.isin(segment_all, retained)]
            targets = flat_features[context_group]
            if merge_indices.numel() == 0 or context_group.numel() == 0:
                merged_context.append(targets)
                continue

            similarity = normalized[merge_indices] @ normalized[context_group].transpose(0, 1)
            assignment = torch.zeros(
                merge_indices.shape[0],
                context_group.shape[0],
                dtype=dtype,
                device=device,
            )
            assignment.scatter_(1, similarity.argmax(dim=1, keepdim=True), 1)
            assigned_count = assignment.sum(dim=0).unsqueeze(-1)
            denominator = assigned_count.clamp(min=1)
            aggregated = assignment.transpose(0, 1) @ flat_features[merge_indices]
            aggregated = aggregated / denominator
            target_weight = (1 / (assigned_count + 1)).clamp(min=dtm_beta)
            merged_context.append(
                target_weight * targets + (1 - target_weight) * aggregated
            )

        context_hidden = (
            torch.cat(merged_context, dim=0)
            if merged_context
            else flat_features.new_empty((0, hidden_dim))
        )
        selected_indices = torch.cat([salient, context])
        selected_hidden = torch.cat([salient_hidden, context_hidden], dim=0)
        order = torch.argsort(selected_indices)
        compressed_segments.append(selected_hidden[order])
        kept_segments.append(selected_indices[order] + frame_offset * frame_tokens)
        frame_offset += coarse_size

    compressed = torch.cat(compressed_segments, dim=0)
    keep_indices = torch.cat(kept_segments, dim=0).to(dtype=torch.long)
    order = torch.argsort(keep_indices)
    compressed = compressed[order]
    keep_indices = keep_indices[order]

    flashvid_config.vision_token_length = int(compressed.shape[0])
    flashvid_config.llm_token_length = None
    flashvid_config.visual_token_length = int(compressed.shape[0])
    setattr(flashvid_config, "last_fastvid_segment_count", float(len(coarse_sizes)))
    setattr(flashvid_config, "last_fastvid_frame_retain_num", float(frame_retain))
    _record_adapter_metrics(
        flashvid_config,
        variant="fastvid",
        output_tokens=int(compressed.shape[0]),
        raw_tokens=total_tokens,
    )
    return compressed, keep_indices
