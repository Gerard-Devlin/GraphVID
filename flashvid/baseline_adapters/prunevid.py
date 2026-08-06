from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F

from flashvid.configuration_flashvid import FlashVidConfig

from .common import _cfg_float, _record_adapter_metrics


def _dpc_cluster(
    features: torch.Tensor,
    cluster_count: int,
    neighbors: int = 7,
    *,
    force_center_labels: bool = True,
):
    """DPC-kNN used by the released PruneVID visual merger."""
    count, dim = features.shape
    cluster_count = min(max(1, int(cluster_count)), count)
    if cluster_count == count:
        labels = torch.arange(count, dtype=torch.long, device=features.device)
        return labels, labels

    distances = torch.cdist(features.float(), features.float()) / math.sqrt(max(1, dim))
    k = min(max(1, int(neighbors)), count)
    nearest = torch.topk(distances, k=k, dim=-1, largest=False).values
    density = (-(nearest.square().mean(dim=-1))).exp()
    # Upstream PruneVID uses random jitter to break equal-density ties.
    density = density + torch.rand_like(density) * 1e-6
    higher = density[None, :] > density[:, None]
    max_distance = distances.max().detach()
    distance_to_higher = torch.where(higher, distances, max_distance).min(dim=-1).values
    score = density * distance_to_higher
    centers = torch.topk(score, k=cluster_count, largest=True).indices
    labels = distances[:, centers].argmin(dim=-1)
    if force_center_labels:
        labels[centers] = torch.arange(
            cluster_count,
            dtype=torch.long,
            device=features.device,
        )
    return labels, centers


def _cluster_average(
    features: torch.Tensor,
    cluster_count: int,
    *,
    force_center_labels: bool = True,
):
    labels, centers = _dpc_cluster(
        features,
        cluster_count=cluster_count,
        force_center_labels=force_center_labels,
    )
    one_hot = F.one_hot(labels, num_classes=int(centers.numel())).to(features.dtype)
    counts = one_hot.sum(dim=0).clamp_min(1).unsqueeze(-1)
    merged = one_hot.transpose(0, 1) @ features
    return merged / counts, centers


def _refine_temporal_labels(labels: torch.Tensor) -> torch.Tensor:
    """Make DPC frame clusters contiguous as in the released PruneVID path."""

    refined = labels.clone()
    count = int(labels.numel())
    for label in torch.unique(labels).tolist():
        positions = torch.where(labels == int(label))[0].tolist()
        runs: list[tuple[int, int]] = []
        if positions:
            start = previous = int(positions[0])
            for position in positions[1:]:
                position = int(position)
                if position == previous + 1:
                    previous = position
                else:
                    runs.append((start, previous))
                    start = previous = position
            runs.append((start, previous))

        longest = max((end - start + 1 for start, end in runs), default=0)
        for start, end in runs:
            if longest == 1 or end - start + 1 < longest:
                refined[start : end + 1] = -1

    index = 0
    while index < count:
        if int(refined[index].item()) != -1:
            index += 1
            continue

        start = index
        while index < count and int(refined[index].item()) == -1:
            index += 1
        end = index - 1

        left_label = int(refined[start - 1].item()) if start > 0 else None
        right_label = int(refined[end + 1].item()) if end + 1 < count else None
        left_length = 0
        if left_label is not None:
            cursor = start - 1
            while cursor >= 0 and int(refined[cursor].item()) == left_label:
                left_length += 1
                cursor -= 1
        right_length = 0
        if right_label is not None:
            cursor = end + 1
            while cursor < count and int(refined[cursor].item()) == right_label:
                right_length += 1
                cursor += 1

        if left_length >= right_length and left_label is not None:
            replacement = left_label
        elif right_label is not None:
            replacement = right_label
        else:
            replacement = 0
        refined[start : end + 1] = replacement

    return refined


def _temporal_windows(
    frame_features: torch.Tensor,
    segment_ratio: float,
    *,
    refine_labels: bool = True,
) -> list[tuple[int, int]]:
    frame_count = int(frame_features.shape[0])
    if frame_count <= 1:
        return [(0, frame_count)]
    cluster_count = min(
        frame_count,
        max(1, int(frame_count * max(0.0, min(1.0, segment_ratio)))),
    )
    labels, _ = _dpc_cluster(
        frame_features,
        cluster_count=cluster_count,
        force_center_labels=refine_labels,
    )
    if refine_labels:
        labels = _refine_temporal_labels(labels)
    windows: list[tuple[int, int]] = []
    start = 0
    for frame_idx in range(1, frame_count):
        if int(labels[frame_idx].item()) != int(labels[frame_idx - 1].item()):
            windows.append((start, frame_idx))
            start = frame_idx
    windows.append((start, frame_count))
    return windows


def prunevid_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Port the released PruneVID static/dynamic merger to LLaVA-OneVision.

    The upstream repository releases this path for PLLaVA rather than
    LLaVA-OneVision. The static/dynamic decomposition, temporal DPC windows,
    spatial DPC merge, and structured text-attention pruning are preserved.
    """
    del cls_attention
    frame_count, tokens_per_frame, _ = video_features.shape
    device = video_features.device
    raw_tokens = int(frame_count * tokens_per_frame)
    target_ratio = max(0.0, min(1.0, float(flashvid_config.retention_ratio)))
    # Never exceed the requested integer budget. Upstream's subsequent
    # per-group int() operations may retain fewer tokens than this target.
    target_tokens = max(
        0,
        min(raw_tokens, math.floor(raw_tokens * target_ratio + 1e-9)),
    )
    tau = _cfg_float(flashvid_config, "prunevid_tau", 0.8)
    cluster_ratio = max(
        1.0 / max(1, tokens_per_frame),
        min(1.0, _cfg_float(flashvid_config, "prunevid_cluster_ratio", 0.5)),
    )
    segment_ratio = _cfg_float(
        flashvid_config,
        "prunevid_temporal_segment_ratio",
        0.25,
    )
    frame_means = F.normalize(video_features.float().mean(dim=1), dim=-1, eps=1e-6)
    windows = _temporal_windows(
        frame_means,
        segment_ratio,
        refine_labels=True,
    )
    global_indices = torch.arange(raw_tokens, dtype=torch.long, device=device).view(
        frame_count,
        tokens_per_frame,
    )

    output_tokens = []
    output_indices = []
    group_sizes: list[int] = []
    for start, end in windows:
        window = video_features[start:end]
        window_size = int(end - start)
        if window_size <= 1:
            static_mask = torch.zeros(tokens_per_frame, dtype=torch.bool, device=device)
        else:
            normalized = F.normalize(window.float(), dim=-1, eps=1e-6)
            pairwise = torch.einsum("wpc,tpc->wtp", normalized, normalized)
            off_diagonal_sum = pairwise.sum(dim=(0, 1)) - float(window_size)
            mean_similarity = off_diagonal_sum / float(window_size * (window_size - 1))
            static_mask = mean_similarity > tau

        static_locations = torch.where(static_mask)[0]
        if static_locations.numel() > 0:
            static_features = window[:, static_locations].mean(dim=0)
            static_clusters = int(static_features.shape[0])
            if static_clusters > 14:
                static_clusters = max(1, int(static_clusters * cluster_ratio))
                static_features, centers = _cluster_average(
                    static_features,
                    static_clusters,
                    force_center_labels=True,
                )
                static_locations = static_locations[centers]
            output_tokens.append(static_features.to(video_features.dtype))
            output_indices.append(global_indices[start, static_locations])
            group_sizes.append(int(static_features.shape[0]))

        dynamic_locations = torch.where(~static_mask)[0]
        for local_frame, frame_idx in enumerate(range(start, end)):
            if dynamic_locations.numel() == 0:
                continue
            dynamic_features = window[local_frame, dynamic_locations]
            dynamic_clusters = int(dynamic_features.shape[0])
            chosen_locations = dynamic_locations
            if dynamic_clusters > 14:
                dynamic_clusters = max(1, int(dynamic_clusters * cluster_ratio))
                dynamic_features, centers = _cluster_average(
                    dynamic_features,
                    dynamic_clusters,
                    force_center_labels=True,
                )
                chosen_locations = dynamic_locations[centers]
            output_tokens.append(dynamic_features.to(video_features.dtype))
            output_indices.append(global_indices[frame_idx, chosen_locations])
            group_sizes.append(int(dynamic_features.shape[0]))

    if not output_tokens:
        raise RuntimeError("PruneVID produced no static or dynamic token groups")

    merged = torch.cat(output_tokens, dim=0)
    indices = torch.cat(output_indices, dim=0)
    outer_tokens = int(merged.shape[0])
    setattr(flashvid_config, "_prunevid_group_sizes", tuple(group_sizes))
    setattr(flashvid_config, "_prunevid_target_tokens", min(target_tokens, outer_tokens))
    flashvid_config.llm_retention_ratio = min(1.0, target_tokens / max(1, outer_tokens))
    flashvid_config.vision_token_length = outer_tokens
    flashvid_config.llm_token_length = None
    flashvid_config.visual_token_length = outer_tokens
    _record_adapter_metrics(
        flashvid_config,
        variant="prunevid",
        output_tokens=outer_tokens,
        raw_tokens=raw_tokens,
    )
    return merged, indices
