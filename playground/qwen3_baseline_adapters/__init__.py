from __future__ import annotations

from typing import Tuple

import torch

from flashvid.configuration_flashvid import FlashVidConfig

from .fastvid import fastvid_compression
from .visionzip import visionzip_compression

SUPPORTED_QWEN3_BASELINE_ADAPTERS = ("fastvid", "visionzip")


def adapter_baseline_compression(
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
    raise ValueError(f"unsupported Qwen3 baseline adapter variant={variant!r}")
