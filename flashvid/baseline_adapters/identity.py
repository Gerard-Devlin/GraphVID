from __future__ import annotations

from typing import Tuple

import torch

from flashvid.configuration_flashvid import FlashVidConfig

from .common import _record_adapter_metrics


def fastv_identity_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Keep every outer token so standalone FastV only prunes inside the LLM."""
    del cls_attention
    flat = video_features.reshape(-1, video_features.shape[-1])
    indices = torch.arange(flat.shape[0], dtype=torch.long, device=flat.device)
    flashvid_config.vision_token_length = int(flat.shape[0])
    flashvid_config.llm_token_length = None
    flashvid_config.visual_token_length = int(flat.shape[0])
    _record_adapter_metrics(
        flashvid_config,
        variant="fastv",
        output_tokens=int(flat.shape[0]),
        raw_tokens=int(flat.shape[0]),
    )
    return flat, indices
