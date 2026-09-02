"""CDPruner's released conditional MAP-DPP selector.

This adapter follows Theia-4869/CDPruner commit
9541616c40fcd5625de1cdb8ea6c33c129eb7864. The upstream implementation is
Copyright 2023 Haotian Liu and is licensed under Apache-2.0. The selector is
intentionally isolated from the FlashVID and CertVID feature, merging, and
inner-pruning paths.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from flashvid.configuration_flashvid import FlashVidConfig

from .common import _record_adapter_metrics


def strict_patch_budget(num_patch_tokens: int, retention_ratio: float) -> int:
    """Return CDPruner's exact video-level patch budget, ``floor(N * r)``."""

    num_patch_tokens = int(num_patch_tokens)
    retention_ratio = float(retention_ratio)
    if num_patch_tokens <= 0:
        raise ValueError("CDPruner requires at least one candidate patch token")
    if not math.isfinite(retention_ratio) or not (0.0 < retention_ratio <= 1.0):
        raise ValueError(
            "CDPruner retention_ratio must be finite and in (0, 1], "
            f"got {retention_ratio!r}"
        )
    budget = math.floor(num_patch_tokens * retention_ratio)
    if budget < 1:
        raise ValueError(
            "CDPruner's strict floor(N * retention_ratio) budget is zero: "
            f"floor({num_patch_tokens} * {retention_ratio}) = 0"
        )
    return min(num_patch_tokens, int(budget))


def conditional_dpp_kernel(
    projected_visual_tokens: torch.Tensor,
    relevance_visual_tokens: torch.Tensor,
    relevance_text_tokens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the released ``diag(r) L diag(r)`` conditional DPP kernel."""

    if projected_visual_tokens.ndim != 2:
        raise ValueError(
            "projected_visual_tokens must have shape [N, D], got "
            f"{tuple(projected_visual_tokens.shape)}"
        )
    if relevance_visual_tokens.ndim != 2:
        raise ValueError(
            "relevance_visual_tokens must have shape [N, C], got "
            f"{tuple(relevance_visual_tokens.shape)}"
        )
    if relevance_text_tokens.ndim == 1:
        relevance_text_tokens = relevance_text_tokens.unsqueeze(0)
    if relevance_text_tokens.ndim != 2:
        raise ValueError(
            "relevance_text_tokens must have shape [M, C], got "
            f"{tuple(relevance_text_tokens.shape)}"
        )
    if projected_visual_tokens.shape[0] != relevance_visual_tokens.shape[0]:
        raise ValueError(
            "CDPruner visual feature streams are not token-aligned: "
            f"{projected_visual_tokens.shape[0]} != {relevance_visual_tokens.shape[0]}"
        )
    if relevance_visual_tokens.shape[1] != relevance_text_tokens.shape[1]:
        raise ValueError(
            "CDPruner image/text relevance dimensions differ: "
            f"{relevance_visual_tokens.shape[1]} != {relevance_text_tokens.shape[1]}"
        )
    if relevance_text_tokens.shape[0] == 0:
        raise ValueError("CDPruner requires at least one text relevance token")

    projected = F.normalize(projected_visual_tokens.float(), dim=-1)
    similarity = projected @ projected.transpose(0, 1)

    image = F.normalize(relevance_visual_tokens.float(), dim=-1)
    text = F.normalize(relevance_text_tokens.float(), dim=-1)
    # Keep the sign used by the released code, including its averaging over
    # segmented text embeddings.
    relevance = -(image @ text.transpose(0, 1)).mean(dim=-1)
    minimum = relevance.min()
    span = relevance.max() - minimum
    if not torch.isfinite(span) or float(span.item()) <= torch.finfo(torch.float32).eps:
        # The released formula is undefined for constant relevance. Uniform
        # quality is the deterministic limiting case and reduces to plain DPP.
        relevance = torch.ones_like(relevance)
    else:
        relevance = (relevance - minimum + 1e-6) / span

    similarity.mul_(relevance.unsqueeze(1))
    similarity.mul_(relevance.unsqueeze(0))
    return similarity, relevance


def fast_map_dpp(
    kernel: torch.Tensor,
    budget: int,
    *,
    return_trace: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Run the released Cholesky greedy MAP-DPP inference."""

    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError(f"CDPruner kernel must be square, got {tuple(kernel.shape)}")
    num_tokens = int(kernel.shape[0])
    budget = int(budget)
    if not (1 <= budget <= num_tokens):
        raise ValueError(
            f"CDPruner budget must satisfy 1 <= K <= N, got K={budget}, N={num_tokens}"
        )

    cis = torch.zeros((budget, num_tokens), dtype=kernel.dtype, device=kernel.device)
    di2s = torch.diagonal(kernel).clone()
    original_diagonal = di2s.clone()
    available = torch.ones(num_tokens, dtype=torch.bool, device=kernel.device)
    selected = torch.empty((budget,), dtype=torch.long, device=kernel.device)
    selected_count = 0

    for step in range(budget):
        scores = di2s.masked_fill(~available, -float("inf"))
        index = torch.argmax(scores)
        pivot = scores[index]
        if not torch.isfinite(pivot) or float(pivot.item()) <= 1e-12:
            remaining = torch.where(available)[0]
            remaining_scores = original_diagonal[remaining]
            # Stable sorting gives lower source indices priority on ties.
            order = torch.argsort(remaining_scores, descending=True, stable=True)
            fill = remaining[order[: budget - selected_count]]
            selected[selected_count : selected_count + fill.numel()] = fill
            selected_count += int(fill.numel())
            break

        selected[step] = index
        if step == 0:
            correction = torch.zeros_like(kernel[index])
        else:
            correction = torch.einsum("t,tn->n", cis[:step, index], cis[:step])
        eis = (kernel[index] - correction) / torch.sqrt(pivot)
        cis[step] = eis
        di2s -= torch.square(eis)
        available[index] = False
        di2s[index] = -float("inf")
        selected_count += 1

    if selected_count != budget:
        raise RuntimeError(
            f"CDPruner MAP-DPP selected {selected_count} tokens, expected {budget}"
        )
    sorted_indices = torch.sort(selected).values
    if return_trace:
        return sorted_indices, selected
    return sorted_indices


def cdpruner_compression(
    video_features: torch.Tensor,
    flashvid_config: FlashVidConfig,
    *,
    relevance_visual_features: Optional[torch.Tensor],
    relevance_text_features: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select one strict global MAP-DPP coreset from all video patches."""

    if video_features.ndim != 3:
        raise ValueError(
            f"CDPruner video_features must have shape [F, P, D], got {tuple(video_features.shape)}"
        )
    if relevance_visual_features is None or relevance_text_features is None:
        raise RuntimeError(
            "CDPruner conditioning features are missing; the model-specific "
            "image/text feature hook did not run"
        )

    num_frames, tokens_per_frame, hidden_size = video_features.shape
    flat_features = video_features.reshape(-1, hidden_size)
    flat_relevance_visual = relevance_visual_features.reshape(
        -1, relevance_visual_features.shape[-1]
    ).to(device=flat_features.device)
    text_features = relevance_text_features.reshape(
        -1, relevance_text_features.shape[-1]
    ).to(device=flat_features.device)
    raw_tokens = int(flat_features.shape[0])
    budget = strict_patch_budget(raw_tokens, flashvid_config.retention_ratio)

    kernel, relevance = conditional_dpp_kernel(
        flat_features,
        flat_relevance_visual,
        text_features,
    )
    keep_indices = fast_map_dpp(kernel, budget)
    selected_tokens = flat_features[keep_indices]

    if int(keep_indices.numel()) != budget or int(torch.unique(keep_indices).numel()) != budget:
        raise RuntimeError(
            "CDPruner violated its strict unique-token budget: "
            f"expected {budget}, got {int(keep_indices.numel())}"
        )
    flashvid_config.vision_token_length = budget
    flashvid_config.visual_token_length = budget
    flashvid_config.llm_token_length = None
    flashvid_config.last_cdpruner_target_tokens = budget
    flashvid_config.last_cdpruner_scope = "video_global"
    flashvid_config.last_cdpruner_relevance_min = float(relevance.min().item())
    flashvid_config.last_cdpruner_relevance_max = float(relevance.max().item())
    _record_adapter_metrics(
        flashvid_config,
        variant="cdpruner",
        output_tokens=budget,
        raw_tokens=num_frames * tokens_per_frame,
    )
    return selected_tokens, keep_indices


def select_qwen3_deepstack(
    deepstack_video_embeds: Sequence[torch.Tensor],
    keep_indices: torch.Tensor,
) -> list[torch.Tensor]:
    """Apply the exact main-sequence CDPruner indices to every DeepStack layer."""

    selected = []
    for layer_idx, features in enumerate(deepstack_video_embeds):
        if features.ndim != 2:
            raise ValueError(
                f"CDPruner DeepStack layer {layer_idx} must be [N, D], "
                f"got {tuple(features.shape)}"
            )
        layer_indices = keep_indices.to(device=features.device, dtype=torch.long)
        selected.append(features.index_select(0, layer_indices))
    return selected


def merge_qwen3_visual_deepstack(
    *,
    deepstack_image_embeds: Sequence[torch.Tensor],
    selected_video_embeds: Sequence[torch.Tensor],
    image_mask: torch.Tensor,
    video_mask: torch.Tensor,
    keep_video_indices: torch.Tensor,
) -> list[torch.Tensor]:
    """Rebuild mixed image/video DeepStack features in retained prompt order."""

    if len(deepstack_image_embeds) != len(selected_video_embeds):
        raise ValueError("CDPruner image/video DeepStack depth mismatch")
    if image_mask.ndim != 2 or video_mask.ndim != 2 or image_mask.shape[0] != 1:
        raise ValueError(
            "CDPruner mixed visual inputs require batch-size-one masks, got "
            f"image={tuple(image_mask.shape)}, video={tuple(video_mask.shape)}"
        )
    image_positions = torch.where(image_mask[0])[0]
    video_positions = torch.where(video_mask[0])[0]
    keep_video_indices = keep_video_indices.to(video_positions.device, torch.long)
    kept_video_positions = video_positions.index_select(0, keep_video_indices)
    order = torch.argsort(
        torch.cat([image_positions, kept_video_positions]),
        stable=True,
    )

    merged = []
    for layer_idx, (image_features, video_features) in enumerate(
        zip(deepstack_image_embeds, selected_video_embeds)
    ):
        if int(image_features.shape[0]) != int(image_positions.numel()):
            raise ValueError(
                f"CDPruner image DeepStack layer {layer_idx} is not placeholder-aligned"
            )
        if int(video_features.shape[0]) != int(kept_video_positions.numel()):
            raise ValueError(
                f"CDPruner video DeepStack layer {layer_idx} is not selector-aligned"
            )
        joint = torch.cat(
            [image_features, video_features.to(image_features.device, image_features.dtype)],
            dim=0,
        )
        merged.append(joint.index_select(0, order.to(joint.device)))
    return merged


def load_siglip_text_tower(vision_tower, model_path: Optional[str] = None) -> None:
    """Load the SigLIP text counterpart required by CDPruner's LLaVA path."""

    if getattr(vision_tower, "_cdpruner_text_tower", None) is not None:
        return
    source = str(
        model_path
        or os.environ.get("CDPRUNER_TEXT_MODEL_PATH", "").strip()
        or getattr(vision_tower, "vision_tower_name", "")
    ).strip()
    if not source:
        raise RuntimeError(
            "CDPruner could not determine the SigLIP text checkpoint; set "
            "CDPRUNER_TEXT_MODEL_PATH"
        )

    try:
        from transformers import AutoTokenizer, SiglipTextModel

        offline = any(
            os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}
            for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
        )
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=offline)
        text_tower = SiglipTextModel.from_pretrained(source, local_files_only=offline)
    except Exception as exc:
        raise RuntimeError(
            "CDPruner requires the SigLIP text tower paired with the LLaVA "
            f"vision checkpoint ({source!r}). Set CDPRUNER_TEXT_MODEL_PATH to "
            "a complete local SigLIP checkpoint when running offline."
        ) from exc

    text_tower.requires_grad_(False)
    text_tower.eval()
    text_tower.to(device=vision_tower.device, dtype=vision_tower.dtype)
    vision_tower._cdpruner_text_tokenizer = tokenizer
    vision_tower._cdpruner_text_tower = text_tower
    vision_tower._cdpruner_text_model_path = source


@torch.no_grad()
def encode_siglip_text(vision_tower, question: str) -> torch.Tensor:
    """Encode the raw user question with the paired SigLIP text tower."""

    question = str(question or "").strip()
    if not question:
        raise ValueError("CDPruner requires a non-empty raw user question")
    tokenizer = getattr(vision_tower, "_cdpruner_text_tokenizer", None)
    text_tower = getattr(vision_tower, "_cdpruner_text_tower", None)
    if tokenizer is None or text_tower is None:
        raise RuntimeError("CDPruner SigLIP text tower has not been initialized")

    encoded = tokenizer(question, return_tensors="pt", truncation=False)
    max_positions = int(getattr(text_tower.config, "max_position_embeddings", 64))
    sequence_length = int(encoded["input_ids"].shape[1])
    segments = max(1, math.ceil(sequence_length / max_positions))
    padded_length = segments * max_positions
    model_inputs = {}
    for name, value in encoded.items():
        if value.ndim != 2:
            continue
        if value.shape[1] < padded_length:
            pad_value = (
                0
                if name == "attention_mask"
                else int(getattr(tokenizer, "pad_token_id", 0) or 0)
            )
            value = F.pad(
                value,
                (0, padded_length - value.shape[1]),
                value=pad_value,
            )
        model_inputs[name] = value.reshape(-1, max_positions).to(vision_tower.device)
    outputs = text_tower(**model_inputs)
    return outputs.pooler_output.to(dtype=vision_tower.dtype)
