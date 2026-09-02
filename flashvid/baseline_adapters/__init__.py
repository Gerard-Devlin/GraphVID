from __future__ import annotations

from typing import Optional, Tuple

import torch

from flashvid.configuration_flashvid import FlashVidConfig

from .cdpruner import cdpruner_compression
from .fastvid import fastvid_compression
from .fastvid_qwen25 import fastvid_qwen25_compression
from .identity import fastv_identity_compression
from .prunevid import prunevid_compression
from .visionzip import visionzip_compression


SUPPORTED_LLAVA_BASELINES = ("fastv", "fastvid", "visionzip", "prunevid", "cdpruner")


def baseline_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
    question_features: Optional[torch.Tensor] = None,
    relevance_visual_features: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    variant = str(getattr(flashvid_config, "compression_variant", "")).strip().lower()
    if variant == "fastv":
        return fastv_identity_compression(video_features, cls_attention, flashvid_config)
    if variant == "fastvid":
        if str(getattr(flashvid_config, "_baseline_backbone", "")).strip().lower() == "qwen2_5_vl":
            return fastvid_qwen25_compression(video_features, cls_attention, flashvid_config)
        return fastvid_compression(video_features, cls_attention, flashvid_config)
    if variant == "visionzip":
        return visionzip_compression(video_features, cls_attention, flashvid_config)
    if variant == "prunevid":
        return prunevid_compression(video_features, cls_attention, flashvid_config)
    if variant == "cdpruner":
        return cdpruner_compression(
            video_features,
            flashvid_config,
            relevance_visual_features=relevance_visual_features,
            relevance_text_features=question_features,
        )
    raise ValueError(f"unsupported LLaVA baseline variant={variant!r}")
