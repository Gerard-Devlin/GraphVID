from __future__ import annotations

from typing import Tuple

import torch

from flashvid.configuration_flashvid import FlashVidConfig

from .curvevid import curvevid_compression
from .fastgraphvid import fastgraphvid_compression

SUPPORTED_QWEN3_BASELINE_ADAPTERS = ("fastgraphvid", "curvevid")


def adapter_baseline_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    variant = str(getattr(flashvid_config, "compression_variant", "")).strip().lower()
    if variant == "fastgraphvid":
        return fastgraphvid_compression(video_features, cls_attention, flashvid_config)
    if variant == "curvevid":
        return curvevid_compression(video_features, cls_attention, flashvid_config)
    raise ValueError(f"unsupported Qwen3 baseline adapter variant={variant!r}")
