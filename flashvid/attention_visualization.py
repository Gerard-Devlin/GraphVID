from __future__ import annotations

from typing import Optional

import torch

from .certvid import CertVidPlan
from .configuration_flashvid import FlashVidConfig


def initialize_certvid_attention_capture(
    config: FlashVidConfig,
    plan: CertVidPlan,
    *,
    frame_count: int,
    tokens_per_frame: int,
) -> None:
    """Initialize the frame identity sidecar for an opt-in paper figure."""
    if not bool(getattr(config, "_capture_layer_frame_attention", False)):
        return
    anchors = plan.anchor_indices.detach().long()
    frame_ids = torch.div(anchors, tokens_per_frame, rounding_mode="floor")
    if frame_ids.numel() != anchors.numel():
        raise RuntimeError("CertVID visualization frame mapping is malformed")
    if frame_ids.numel() and (
        int(frame_ids.min().item()) < 0
        or int(frame_ids.max().item()) >= int(frame_count)
    ):
        raise RuntimeError("CertVID visualization frame IDs are out of range")
    setattr(config, "_visualization_current_frame_ids", frame_ids)
    setattr(config, "_visualization_frame_count", int(frame_count))
    setattr(config, "_visualization_tokens_per_frame", int(tokens_per_frame))
    setattr(config, "_visualization_layer_attention", {})


def update_visualization_after_inner_prune(
    config: FlashVidConfig,
    keep_indices: torch.Tensor,
    *,
    visual_start: int,
    visual_length: int,
) -> None:
    """Keep frame identities aligned with the visual tokens surviving FastV."""
    if not bool(getattr(config, "_capture_layer_frame_attention", False)):
        return
    frame_ids = getattr(config, "_visualization_current_frame_ids", None)
    if not torch.is_tensor(frame_ids) or frame_ids.numel() != visual_length:
        raise RuntimeError(
            "visualization frame identities do not match the pre-prune visual span"
        )
    visual_keep = keep_indices[
        (keep_indices >= visual_start)
        & (keep_indices < visual_start + visual_length)
    ] - visual_start
    frame_ids = frame_ids.to(device=visual_keep.device)
    setattr(
        config,
        "_visualization_current_frame_ids",
        frame_ids[visual_keep].contiguous(),
    )


def capture_qwen2_layer_frame_attention(
    *,
    config: Optional[FlashVidConfig],
    layer_index: int,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    num_key_value_groups: int,
) -> None:
    """Capture last-query visual mass without materializing full attention."""
    if config is None or not bool(
        getattr(config, "_capture_layer_frame_attention", False)
    ):
        return
    # Decode calls contain one query and would overwrite the prefill analysis.
    if query_states.ndim != 4 or query_states.shape[-2] <= 1:
        return

    frame_ids = getattr(config, "_visualization_current_frame_ids", None)
    frame_count = int(getattr(config, "_visualization_frame_count", 0) or 0)
    visual_start = int(getattr(config, "visual_token_start_index", -1))
    if not torch.is_tensor(frame_ids) or frame_count <= 0 or visual_start < 0:
        return
    visual_length = int(frame_ids.numel())
    key_length = int(key_states.shape[-2])
    if visual_length <= 0 or visual_start + visual_length > key_length:
        raise RuntimeError(
            "visualization visual span does not fit the current attention keys"
        )

    with torch.no_grad():
        repeated_keys = key_states
        if num_key_value_groups > 1:
            repeated_keys = key_states.repeat_interleave(
                num_key_value_groups,
                dim=1,
            )
        last_query = query_states[:, :, -1:, :]
        logits = torch.matmul(
            last_query.float(),
            repeated_keys.float().transpose(2, 3),
        ) * float(scaling)
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            # The final prefill query has no future keys. Only an additive
            # padding mask can affect it; the visualizer runs batch size one
            # without padding, while retaining compatibility with float masks.
            mask = attention_mask[..., -1:, :key_length]
            logits = logits + mask.to(device=logits.device, dtype=logits.dtype)
        attention = torch.softmax(logits, dim=-1, dtype=torch.float32)
        visual_attention = attention[
            ...,
            visual_start : visual_start + visual_length,
        ].squeeze(2)
        visual_ratio_per_head = visual_attention.sum(dim=-1).mean(dim=0)

        frame_ids = frame_ids.to(device=visual_attention.device, dtype=torch.long)
        if frame_ids.numel() and (
            int(frame_ids.min().item()) < 0
            or int(frame_ids.max().item()) >= frame_count
        ):
            raise RuntimeError("visualization frame IDs are out of range")
        head_frame_mass = torch.zeros(
            visual_attention.shape[1],
            frame_count,
            device=visual_attention.device,
            dtype=torch.float32,
        )
        source = visual_attention.mean(dim=0)
        head_frame_mass.scatter_add_(
            1,
            frame_ids.unsqueeze(0).expand(source.shape[0], -1),
            source,
        )
        frame_weights = head_frame_mass.mean(dim=0)
        frame_weights = frame_weights / frame_weights.sum().clamp_min(1e-12)

        records = getattr(config, "_visualization_layer_attention", None)
        if not isinstance(records, dict):
            records = {}
            setattr(config, "_visualization_layer_attention", records)
        records[int(layer_index) + 1] = {
            "visual_ratio_per_head": visual_ratio_per_head.detach().cpu(),
            "frame_weights": frame_weights.detach().cpu(),
            "visual_tokens": visual_length,
            "sequence_tokens": key_length,
        }


def clear_layer_attention_capture(config: FlashVidConfig) -> None:
    for name in (
        "_visualization_current_frame_ids",
        "_visualization_frame_count",
        "_visualization_tokens_per_frame",
    ):
        setattr(config, name, None)
