from __future__ import annotations

from typing import Optional

import torch

from .certvid import _grid_hw
from .certvid_v3 import certvid_v3_compression
from .configuration_flashvid import FlashVidConfig
from .v3plus_inner import V3PlusOuterMetadata, clear_v3plus_runtime


def _selected_metadata(
    selected: torch.Tensor,
    plan,
    analysis: dict,
    frame_count: int,
    tokens_per_frame: int,
    config: FlashVidConfig,
) -> V3PlusOuterMetadata:
    device = selected.device
    selected = selected.long().reshape(-1)
    total_tokens = int(frame_count * tokens_per_frame)

    if bool(analysis.get("identity", False)):
        frame_ids = torch.div(selected, tokens_per_frame, rounding_mode="floor")
        temporal_bins = max(1, int(getattr(config, "certv3_temporal_bins", 12)))
        temporal_ids = torch.div(
            frame_ids * temporal_bins,
            max(1, frame_count),
            rounding_mode="floor",
        ).clamp_max(temporal_bins - 1)
        component_ids = selected.clone()
        demand = torch.ones(selected.numel(), dtype=torch.float32, device=device)
    else:
        required = ("frame_ids", "temporal_ids", "component_ids", "demand_weight")
        missing = [name for name in required if name not in analysis]
        if missing:
            raise RuntimeError(f"CertVID V3 analysis is missing fields: {missing}")
        frame_ids = analysis["frame_ids"].index_select(0, selected).long()
        temporal_ids = analysis["temporal_ids"].index_select(0, selected).long()
        component_ids = analysis["component_ids"].index_select(0, selected).long()
        demand = analysis["demand_weight"].index_select(0, selected).float()

    height, width = _grid_hw(tokens_per_frame, config)
    spatial_bins = max(1, int(getattr(config, "certv3_spatial_bins", 3)))
    local_ids = torch.remainder(selected, tokens_per_frame)
    rows = torch.div(local_ids, width, rounding_mode="floor").clamp_max(height - 1)
    cols = torch.remainder(local_ids, width).clamp_max(width - 1)
    row_bins = torch.div(
        rows * spatial_bins,
        max(1, height),
        rounding_mode="floor",
    ).clamp_max(spatial_bins - 1)
    col_bins = torch.div(
        cols * spatial_bins,
        max(1, width),
        rounding_mode="floor",
    ).clamp_max(spatial_bins - 1)
    spatial_ids = (row_bins * spatial_bins + col_bins).long()

    fusion_alpha = getattr(plan, "fusion_alpha", None)
    if fusion_alpha is None or int(fusion_alpha.numel()) != int(selected.numel()):
        raise RuntimeError("CertVID V3 fusion plan is not aligned with its anchors")
    certificate_mask = fusion_alpha.reshape(-1).float() <= 1e-12

    return V3PlusOuterMetadata(
        global_indices=selected.detach(),
        frame_ids=frame_ids.detach(),
        temporal_ids=temporal_ids.detach(),
        spatial_ids=spatial_ids.detach(),
        component_ids=component_ids.detach(),
        demand_weight=demand.detach(),
        certificate_mask=certificate_mask.detach(),
        raw_token_count=total_tokens,
        frame_count=frame_count,
        tokens_per_frame=tokens_per_frame,
    )


def certvid_v3plus_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
    question_features: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run CertVID V3 unchanged and publish prefill-local inner metadata."""
    clear_v3plus_runtime(flashvid_config)
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
        raise RuntimeError("CertVID V3Plus requires the CertVID V3 assignment plan")
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
    setattr(flashvid_config, "_v3plus_outer_metadata", metadata)
    setattr(flashvid_config, "last_adapter_variant", "certvid_v3plus")
    return output, indices
