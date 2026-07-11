from __future__ import annotations

from collections.abc import Sequence

import torch

from .certvid import CertVidPlan, apply_certvid_plan


def compress_certvid_deepstack(
    deepstack_video_embeds: Sequence[torch.Tensor],
    plan: CertVidPlan,
) -> list[torch.Tensor]:
    """Apply the base-token CertVID plan to every Qwen3 DeepStack level."""
    compressed: list[torch.Tensor] = []
    for layer_idx, layer_features in enumerate(deepstack_video_embeds):
        if layer_features.ndim != 2:
            raise ValueError(
                f"CertVID DeepStack layer {layer_idx} must be [N, D], "
                f"got {tuple(layer_features.shape)}"
            )
        compressed.append(apply_certvid_plan(layer_features, plan))
    return compressed


def merge_certvid_visual_deepstack(
    *,
    deepstack_image_embeds: Sequence[torch.Tensor],
    compressed_video_embeds: Sequence[torch.Tensor],
    image_mask: torch.Tensor,
    video_mask: torch.Tensor,
    kept_video_indices: torch.Tensor,
) -> list[torch.Tensor]:
    """Rebuild image/video DeepStack tensors in retained input-token order."""
    if len(deepstack_image_embeds) != len(compressed_video_embeds):
        raise ValueError(
            "CertVID image/video DeepStack depth mismatch: "
            f"{len(deepstack_image_embeds)} != {len(compressed_video_embeds)}"
        )
    if image_mask.ndim != 2 or video_mask.ndim != 2 or image_mask.shape[0] != 1:
        raise ValueError(
            "CertVID mixed visual inputs currently require batch size 1 masks, "
            f"got image={tuple(image_mask.shape)}, video={tuple(video_mask.shape)}"
        )

    image_positions = torch.where(image_mask[0])[0]
    video_positions = torch.where(video_mask[0])[0]
    kept_video_indices = kept_video_indices.to(device=video_positions.device, dtype=torch.long)
    kept_video_positions = video_positions.index_select(0, kept_video_indices)
    joint_positions = torch.cat([image_positions, kept_video_positions], dim=0)
    order = torch.argsort(joint_positions, stable=True)

    merged: list[torch.Tensor] = []
    for layer_idx, (image_features, video_features) in enumerate(
        zip(deepstack_image_embeds, compressed_video_embeds)
    ):
        if int(image_features.shape[0]) != int(image_positions.numel()):
            raise ValueError(
                f"CertVID image DeepStack layer {layer_idx} has {image_features.shape[0]} "
                f"features for {image_positions.numel()} placeholders"
            )
        if int(video_features.shape[0]) != int(kept_video_positions.numel()):
            raise ValueError(
                f"CertVID video DeepStack layer {layer_idx} has {video_features.shape[0]} "
                f"features for {kept_video_positions.numel()} retained placeholders"
            )
        joint = torch.cat(
            [image_features, video_features.to(image_features.device, image_features.dtype)],
            dim=0,
        )
        merged.append(joint.index_select(0, order.to(joint.device)))
    return merged
