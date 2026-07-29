from __future__ import annotations

from typing import Tuple

import torch

from flashvid.configuration_flashvid import FlashVidConfig

from .fastvid import fastvid_compression
from .identity import fastv_identity_compression
from .prunevid import prunevid_compression
from .visionzip import visionzip_compression


SUPPORTED_LLAVA_BASELINES = ("fastv", "fastvid", "visionzip", "prunevid")


def baseline_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    variant = str(getattr(flashvid_config, "compression_variant", "")).strip().lower()
    if variant == "fastv":
        return fastv_identity_compression(video_features, cls_attention, flashvid_config)
    if variant == "fastvid":
        return fastvid_compression(video_features, cls_attention, flashvid_config)
    if variant == "visionzip":
        return visionzip_compression(video_features, cls_attention, flashvid_config)
    if variant == "prunevid":
        return prunevid_compression(video_features, cls_attention, flashvid_config)
    raise ValueError(f"unsupported LLaVA baseline variant={variant!r}")
