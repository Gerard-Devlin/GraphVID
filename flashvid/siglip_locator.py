"""Training-free SigLIP question-to-frame localization for CertVID-G."""

from __future__ import annotations

import re
import gc
from typing import Any, Optional

import torch
import torch.nn.functional as F


def _locator_checkpoint(model: Any, config: Any) -> str:
    explicit = str(getattr(config, "certg_locator_checkpoint", "") or "").strip()
    if explicit:
        return explicit
    try:
        tower = model.get_vision_tower()
        for name in ("vision_tower_name", "vision_tower"):
            value = getattr(tower, name, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
    except Exception:
        pass
    return "google/siglip-so400m-patch14-384"


def install_siglip_locator(model: Any, config: Any) -> bool:
    """Attach a frozen SigLIP text tower without registering it in the LLM."""
    if str(getattr(config, "compression_variant", "")).lower() != "certvid_g":
        return False
    if getattr(model, "_certg_text_model", None) is not None:
        return True

    checkpoint = _locator_checkpoint(model, config)
    try:
        from transformers import AutoTokenizer, SiglipTextModel, SiglipVisionModel

        tower = model.get_vision_tower()
        device = getattr(tower, "device", None)
        dtype = getattr(tower, "dtype", None)
        if device is None or dtype is None:
            parameter = next(tower.parameters())
            device, dtype = parameter.device, parameter.dtype

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        text_model = SiglipTextModel.from_pretrained(
            checkpoint,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        text_model.eval()
        text_model.requires_grad_(False)
        text_model.to(device=device, dtype=dtype)

        restored_vision = SiglipVisionModel.from_pretrained(
            checkpoint,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        visual_tail_layer = restored_vision.vision_model.encoder.layers[-1]
        visual_pooling_head = restored_vision.vision_model.head
        visual_tail_layer.eval().requires_grad_(False)
        visual_pooling_head.eval().requires_grad_(False)
        visual_tail_layer.to(device=device, dtype=dtype)
        visual_pooling_head.to(device=device, dtype=dtype)
        del restored_vision
        gc.collect()

        # Keep the inference-only side tower outside the LLM module tree. Each
        # distributed rank owns one frozen copy and no DDP synchronization is needed.
        object.__setattr__(model, "_certg_text_model", text_model)
        object.__setattr__(model, "_certg_text_tokenizer", tokenizer)
        object.__setattr__(model, "_certg_visual_tail_layer", visual_tail_layer)
        object.__setattr__(model, "_certg_visual_pooling_head", visual_pooling_head)
        setattr(config, "_certg_locator_checkpoint", checkpoint)
        setattr(config, "_certg_locator_error", None)
        return True
    except Exception as error:
        setattr(config, "_certg_locator_checkpoint", checkpoint)
        setattr(config, "_certg_locator_error", f"{type(error).__name__}: {error}")
        return False


def _query_prompts(question: str) -> list[str]:
    question = re.sub(r"\s+", " ", question).strip()
    if not question:
        return []
    prompts = [question]
    declarative = re.sub(
        r"^(what|which|who|where|when|why|how)\s+",
        "",
        question,
        flags=re.IGNORECASE,
    ).strip(" ?.!")
    if declarative and declarative.lower() != question.lower():
        prompts.append(f"A video frame showing {declarative}.")
    return prompts[:2]


@torch.no_grad()
def compute_siglip_frame_scores(
    model: Any,
    question: str,
    frame_embeddings: Optional[torch.Tensor],
) -> tuple[Optional[torch.Tensor], str, int]:
    """Return native SigLIP-space frame/query cosine scores."""
    text_model = getattr(model, "_certg_text_model", None)
    tokenizer = getattr(model, "_certg_text_tokenizer", None)
    visual_tail_layer = getattr(model, "_certg_visual_tail_layer", None)
    visual_pooling_head = getattr(model, "_certg_visual_pooling_head", None)
    if text_model is None or tokenizer is None:
        return None, "locator_unavailable", 0
    if (
        visual_tail_layer is None
        or visual_pooling_head is None
        or frame_embeddings is None
        or frame_embeddings.ndim != 3
    ):
        return None, "frame_pool_missing", 0
    prompts = _query_prompts(question)
    if not prompts:
        return None, "query_missing", 0

    device = next(text_model.parameters()).device
    max_length = int(getattr(text_model.config, "max_position_embeddings", 64))
    encoded = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
        if key in {"input_ids", "attention_mask", "position_ids"}
    }
    text_outputs = text_model(**encoded)
    text_embeddings = F.normalize(
        text_outputs.pooler_output.float(),
        dim=-1,
        eps=1e-6,
    )
    patches = frame_embeddings.to(
        device=device,
        dtype=next(visual_tail_layer.parameters()).dtype,
    )
    tail_outputs = visual_tail_layer(
        patches,
        attention_mask=None,
        output_attentions=False,
    )
    tail_features = tail_outputs[0] if isinstance(tail_outputs, tuple) else tail_outputs
    frames = F.normalize(
        visual_pooling_head(tail_features).float(),
        dim=-1,
        eps=1e-6,
    )
    if frames.shape[-1] != text_embeddings.shape[-1]:
        return None, "embedding_dimension_mismatch", len(prompts)
    scores = (frames @ text_embeddings.T).amax(dim=1)
    if not bool(torch.isfinite(scores).all()):
        return None, "non_finite_scores", len(prompts)
    return scores, "siglip_native_cosine", len(prompts)
