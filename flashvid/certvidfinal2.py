"""Fixed CertVID V3 configuration used by the final paper model.

Final2 intentionally delegates to the reference V3 implementation.  It fixes
hard certificates off while preserving every
other V3 selection, refinement, fusion, and diagnostic path.
"""

from __future__ import annotations

from typing import Any, MutableMapping, Optional

import torch

from .certvid_v3 import certvid_v3_compression
from .configuration_flashvid import FlashVidConfig


_MISSING = object()


def _restore_config(config: FlashVidConfig, name: str, previous: object) -> None:
    if previous is _MISSING:
        try:
            delattr(config, name)
        except AttributeError:
            pass
        return
    setattr(config, name, previous)


def certvidfinal2_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
    question_features: Optional[torch.Tensor] = None,
    *,
    analysis_sink: Optional[MutableMapping[str, Any]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run V3 with no hard certificates and no trajectory/event dynamics."""
    overrides = {
        # V3 uses this name for strict token-budget rounding.
        "compression_variant": "certvid_v3",
        "certv3_certificate_budget_ratio": 0.0,
        "certv3_use_trajectory": False,
    }
    previous = {
        name: getattr(flashvid_config, name, _MISSING)
        for name in overrides
    }
    try:
        for name, value in overrides.items():
            setattr(flashvid_config, name, value)
        return certvid_v3_compression(
            video_features=video_features,
            cls_attention=cls_attention,
            flashvid_config=flashvid_config,
            question_features=question_features,
            analysis_sink=analysis_sink,
        )
    finally:
        for name, value in previous.items():
            _restore_config(flashvid_config, name, value)
