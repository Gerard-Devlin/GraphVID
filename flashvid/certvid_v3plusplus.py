from __future__ import annotations

from typing import Optional

import torch

from .certvid_v3 import certvid_v3_compression
from .certvid_v3plus import _selected_metadata
from .configuration_flashvid import FlashVidConfig
from .v3plusplus_inner import clear_v3plusplus_runtime


def certvid_v3plusplus_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
    question_features: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the V3 outer compressor unchanged and publish inner-pruning metadata."""
    clear_v3plusplus_runtime(flashvid_config)
    analysis: dict = {}
    output, indices = certvid_v3_compression(
        video_features=video_features,
        cls_attention=cls_attention,
        flashvid_config=flashvid_config,
        question_features=question_features,
        analysis_sink=analysis,
    )

    plan = getattr(flashvid_config, "_certvid_plan", None)
    if plan is None:
        raise RuntimeError("CertVID V3PlusPlus requires the CertVID V3 assignment plan")
    selected = plan.anchor_indices
    if int(output.shape[0]) != int(selected.numel()):
        raise RuntimeError("CertVID V3 output and anchor order are not aligned")

    metadata = _selected_metadata(
        selected=selected,
        plan=plan,
        analysis=analysis,
        frame_count=int(video_features.shape[0]),
        tokens_per_frame=int(video_features.shape[1]),
        config=flashvid_config,
    )
    setattr(flashvid_config, "_v3plusplus_outer_metadata", metadata)
    setattr(flashvid_config, "last_adapter_variant", "certvid_v3plusplus")
    return output, indices
