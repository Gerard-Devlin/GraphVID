"""Canonical alias for the CertVID V3 numerical implementation.

Keeping both public names on one implementation prevents numerical drift.
All algorithm parameters, including certificate policy, remain caller-owned.
"""

from __future__ import annotations

from typing import Any, MutableMapping, Optional

import torch

from .certvid_v3 import certvid_v3_compression
from .configuration_flashvid import FlashVidConfig


_MISSING = object()


def _restore_config_value(
    config: FlashVidConfig,
    name: str,
    previous: object,
) -> None:
    if previous is _MISSING:
        try:
            delattr(config, name)
        except AttributeError:
            pass
    else:
        setattr(config, name, previous)


def certvidfinal_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
    question_features: Optional[torch.Tensor] = None,
    *,
    analysis_sink: Optional[MutableMapping[str, Any]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the exact CertVID V3 path with the caller's configuration.

    ``compression_variant`` is temporarily normalized to ``certvid_v3`` so
    strict-budget rounding and backbone-specific policy checks are identical
    to the reference V3 experiment. The caller-visible variant is restored
    even if compression raises.
    """

    variant = getattr(flashvid_config, "compression_variant", _MISSING)
    try:
        setattr(flashvid_config, "compression_variant", "certvid_v3")
        return certvid_v3_compression(
            video_features=video_features,
            cls_attention=cls_attention,
            flashvid_config=flashvid_config,
            question_features=question_features,
            analysis_sink=analysis_sink,
        )
    finally:
        _restore_config_value(
            flashvid_config,
            "compression_variant",
            variant,
        )
