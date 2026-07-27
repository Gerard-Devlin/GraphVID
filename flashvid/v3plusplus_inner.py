from __future__ import annotations

import json
import os
import time
from contextlib import nullcontext
from typing import Any, Optional

import torch
import torch.nn.functional as F

from .v3plus_inner import V3PlusOuterMetadata


def clear_v3plusplus_runtime(config: Any) -> None:
    """Release sample-local tensors while preserving completed diagnostics."""
    setattr(config, "_v3plusplus_outer_metadata", None)


def _cfg_float(config: Any, name: str, default: float) -> float:
    try:
        return float(getattr(config, name, default))
    except (TypeError, ValueError):
        return float(default)


def _cfg_int(config: Any, name: str, default: int) -> int:
    try:
        return int(getattr(config, name, default))
    except (TypeError, ValueError):
        return int(default)


def v3plusplus_strict_enabled(config: Any) -> bool:
    value = getattr(config, "v3plusplus_strict", True)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _rank_path(path: str) -> str:
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    return path.replace("{rank}", str(rank))


def _write_diagnostics(config: Any, diagnostics: dict[str, Any]) -> None:
    setattr(config, "last_v3plusplus_diagnostics", diagnostics)
    path = str(os.environ.get("V3PLUSPLUS_DIAGNOSTICS_JSONL", "")).strip()
    if not path:
        return
    path = _rank_path(path)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(diagnostics, ensure_ascii=True) + "\n")


def _stable_descending(values: torch.Tensor) -> torch.Tensor:
    try:
        return torch.argsort(values, descending=True, stable=True)
    except TypeError:
        return torch.argsort(values, descending=True)


def _gradient_nms(
    scores: torch.Tensor,
    features: torch.Tensor,
    budget: int,
    threshold: float,
) -> tuple[torch.Tensor, int]:
    """Apply TRIO-style saliency ordering followed by feature-space NMS."""
    count = int(scores.numel())
    budget = min(max(1, int(budget)), count)
    order = _stable_descending(scores.float())
    normalized = F.normalize(features.float(), dim=-1, eps=1e-6)
    similarity = normalized @ normalized.transpose(0, 1)
    neighbors = similarity >= float(threshold)

    suppressed = torch.zeros(count, dtype=torch.bool, device=scores.device)
    selected: list[int] = []
    selected_mask = torch.zeros(count, dtype=torch.bool, device=scores.device)
    for candidate in order.tolist():
        index = int(candidate)
        if bool(suppressed[index].item()):
            continue
        selected.append(index)
        selected_mask[index] = True
        suppressed |= neighbors[index]
        if len(selected) >= budget:
            break

    strict_count = len(selected)
    if len(selected) < budget:
        for candidate in order.tolist():
            index = int(candidate)
            if bool(selected_mask[index].item()):
                continue
            selected.append(index)
            selected_mask[index] = True
            if len(selected) >= budget:
                break

    result = torch.tensor(selected, dtype=torch.long, device=scores.device)
    return torch.sort(result).values, strict_count


def _proxy_positions(sequence_length: int, count: int, device: torch.device) -> torch.Tensor:
    # TRIO applies CE to the final K entries of shifted logits. Those entries
    # correspond to logits at positions S-2, S-3, ..., S-K-1.
    count = min(max(1, count), max(1, sequence_length - 1))
    start = max(0, sequence_length - 1 - count)
    return torch.arange(start, sequence_length - 1, device=device, dtype=torch.long)


def _gradient_saliency(
    *,
    hidden_states: torch.Tensor,
    decoder_layer: Any,
    output_norm: Any,
    output_head: Any,
    position_ids: Optional[torch.Tensor],
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    proxy_positions: int,
) -> tuple[torch.Tensor, float, float]:
    """Compute inference-objective gradients with an exact sparse lm-head projection."""
    started = time.perf_counter()
    try:
        inference_context = torch.inference_mode(False)
    except (AttributeError, TypeError):
        inference_context = nullcontext()

    with inference_context, torch.enable_grad():
        parameter = next(decoder_layer.parameters())
        target_dtype = parameter.dtype
        probe = hidden_states.detach().clone().to(dtype=target_dtype)
        probe.requires_grad_(True)
        probe_position_ids = (
            None if position_ids is None else position_ids.detach().clone()
        )
        probe_position_embeddings = tuple(
            value.detach().clone() for value in position_embeddings
        )

        if probe.device.type == "cuda" and target_dtype in (torch.float16, torch.bfloat16):
            autocast_context = torch.autocast(device_type="cuda", dtype=target_dtype)
        else:
            autocast_context = nullcontext()

        with autocast_context:
            layer_output = decoder_layer(
                probe,
                attention_mask=None,
                position_ids=probe_position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=None,
                position_embeddings=probe_position_embeddings,
                output_attentions=False,
            )[0]
            positions = _proxy_positions(
                sequence_length=int(layer_output.shape[1]),
                count=proxy_positions,
                device=layer_output.device,
            )
            objective_hidden = layer_output.index_select(1, positions)
            objective_hidden = output_norm(objective_hidden)
            logits = output_head(objective_hidden).float()
            pseudo_targets = logits.detach().argmax(dim=-1)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                pseudo_targets.reshape(-1),
                reduction="mean",
            )

        if not bool(loss.requires_grad):
            raise RuntimeError("proxy objective is detached")
        gradients = torch.autograd.grad(
            loss,
            probe,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if not torch.isfinite(gradients).all():
        raise RuntimeError("proxy gradient contains NaN or Inf")
    return gradients.detach(), float(loss.detach().item()), elapsed_ms


def select_v3plusplus_inner_tokens(
    *,
    hidden_states: torch.Tensor,
    visual_global_indices: torch.Tensor,
    budget: int,
    decoder_layer: Any,
    output_norm: Any,
    output_head: Any,
    position_ids: Optional[torch.Tensor],
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    config: Any,
) -> torch.Tensor:
    """Select inner visual tokens with inference-objective gradients and NMS."""
    if int(hidden_states.shape[0]) != 1:
        raise ValueError("CertVID V3PlusPlus currently supports batch size 1")
    if output_norm is None or output_head is None:
        raise RuntimeError("language-model norm or output head is unavailable")

    visual_count = int(visual_global_indices.numel())
    budget = min(max(1, int(budget)), visual_count)
    metadata: Optional[V3PlusOuterMetadata] = getattr(
        config,
        "_v3plusplus_outer_metadata",
        None,
    )
    if metadata is None or int(metadata.global_indices.numel()) != visual_count:
        raise RuntimeError("V3 outer metadata is missing or misaligned")

    gradients, proxy_loss, backward_ms = _gradient_saliency(
        hidden_states=hidden_states,
        decoder_layer=decoder_layer,
        output_norm=output_norm,
        output_head=output_head,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
        proxy_positions=max(1, _cfg_int(config, "v3plusplus_proxy_positions", 4)),
    )
    visual_gradients = gradients[0].index_select(0, visual_global_indices)
    saliency = torch.linalg.vector_norm(visual_gradients.float(), ord=2, dim=-1)
    if not torch.isfinite(saliency).all() or float(saliency.max().item()) <= 0.0:
        raise RuntimeError("gradient saliency is degenerate")

    visual_features = hidden_states[0].index_select(0, visual_global_indices)
    nms_enabled = bool(getattr(config, "v3plusplus_nms_enabled", True))
    nms_threshold = min(
        1.0,
        max(-1.0, _cfg_float(config, "v3plusplus_nms_threshold", 0.80)),
    )
    selection_started = time.perf_counter()
    if nms_enabled:
        selected, strict_count = _gradient_nms(
            scores=saliency,
            features=visual_features,
            budget=budget,
            threshold=nms_threshold,
        )
    else:
        selected = torch.sort(_stable_descending(saliency)[:budget]).values
        strict_count = budget
    selection_ms = (time.perf_counter() - selection_started) * 1000.0

    selected_frames = metadata.frame_ids.index_select(0, selected).long()
    frame_counts = torch.bincount(
        selected_frames,
        minlength=max(1, int(metadata.frame_count)),
    )
    probabilities = frame_counts.float()
    probabilities = probabilities / probabilities.sum().clamp_min(1.0)
    nonzero = probabilities > 0
    entropy = -torch.sum(probabilities[nonzero] * torch.log(probabilities[nonzero]))
    entropy = entropy / max(
        1e-6,
        float(torch.log(torch.tensor(max(2, metadata.frame_count))).item()),
    )

    diagnostics = {
        "method": "certvid_v3plusplus",
        "status": "gradient_nms",
        "selector": "trio_inference_objective_gradient_nms",
        "strict": v3plusplus_strict_enabled(config),
        "pruning_layer": _cfg_int(config, "pruning_layer", -1),
        "inner_retention_ratio": _cfg_float(
            config,
            "llm_retention_ratio",
            1.0,
        ),
        "outer_tokens": visual_count,
        "inner_tokens": int(selected.numel()),
        "proxy_positions": max(1, _cfg_int(config, "v3plusplus_proxy_positions", 4)),
        "proxy_loss": proxy_loss,
        "gradient_min": float(saliency.min().item()),
        "gradient_median": float(saliency.median().item()),
        "gradient_max": float(saliency.max().item()),
        "nms_enabled": nms_enabled,
        "nms_threshold": nms_threshold,
        "nms_strict_selected": int(strict_count),
        "nms_fill_selected": int(selected.numel()) - int(strict_count),
        "frames_with_tokens": int((frame_counts > 0).sum().item()),
        "empty_frames": int((frame_counts == 0).sum().item()),
        "temporal_entropy": float(entropy.item()),
        "per_frame_tokens": frame_counts.detach().cpu().tolist(),
        "proxy_backward_ms": backward_ms,
        "selection_host_ms": selection_ms,
        "fallback_reason": None,
    }
    _write_diagnostics(config, diagnostics)
    return selected


def record_v3plusplus_fallback(config: Any, reason: str, budget: int) -> None:
    _write_diagnostics(
        config,
        {
            "method": "certvid_v3plusplus",
            "status": "legacy_fallback",
            "inner_tokens": int(budget),
            "fallback_reason": str(reason),
        },
    )
