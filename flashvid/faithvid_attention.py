from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def _runtime_config(module):
    config = getattr(module, "flashvid_config", None)
    if config is None:
        return None
    if str(getattr(config, "compression_variant", "")).strip().lower() != "faithvid":
        return None
    return config


def clear_faithvid_runtime(config) -> None:
    for name in (
        "_faithvid_outer_group_mass",
        "_faithvid_outer_variance",
        "_faithvid_outer_log_mass",
        "_faithvid_inner_group_mass",
        "_faithvid_inner_variance",
        "_faithvid_inner_log_mass",
        "_faithvid_validated_metadata",
    ):
        setattr(config, name, None)


def _layer_metadata(config, layer_idx: int) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    use_inner = int(layer_idx) >= int(getattr(config, "pruning_layer", 0))
    prefix = "inner" if use_inner else "outer"
    return (
        getattr(config, f"_faithvid_{prefix}_group_mass", None),
        getattr(config, f"_faithvid_{prefix}_log_mass", None),
    )


def _repeat_kv(states: torch.Tensor, query_heads: int) -> torch.Tensor:
    key_heads = int(states.shape[1])
    if key_heads == query_heads:
        return states
    if query_heads % key_heads != 0:
        raise ValueError(f"query heads {query_heads} are not divisible by key/value heads {key_heads}")
    return states.repeat_interleave(query_heads // key_heads, dim=1)


def _base_attention_mask(
    *,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: Optional[torch.Tensor],
    sliding_window: Optional[int],
) -> torch.Tensor:
    batch, _, query_length, _ = query_states.shape
    key_length = int(key_states.shape[-2])
    dtype = query_states.dtype
    device = query_states.device
    min_value = torch.finfo(dtype).min

    # The patched model forwards already materialize their causal 4-D masks.
    # Reusing that tensor avoids allocating another Q x K matrix at every layer.
    if attention_mask is not None and attention_mask.ndim == 4:
        mask = attention_mask.to(device=device)
        if int(mask.shape[-2]) != query_length:
            mask = mask[..., -query_length:, :]
        mask = mask[..., :key_length]
        if mask.shape[0] == 1 and batch > 1:
            mask = mask.expand(batch, *mask.shape[1:])
        if mask.dtype == torch.bool:
            additive = torch.zeros(mask.shape, dtype=dtype, device=device)
            return additive.masked_fill(~mask, min_value)
        return mask.to(dtype=dtype)

    if cache_position is not None and int(cache_position.numel()) == query_length:
        query_positions = cache_position.reshape(-1).to(device=device, dtype=torch.long)
    else:
        start = max(0, key_length - query_length)
        query_positions = torch.arange(start, start + query_length, device=device)
    key_positions = torch.arange(key_length, device=device)
    allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    if sliding_window is not None and int(sliding_window) > 0:
        allowed &= key_positions.unsqueeze(0) > (
            query_positions.unsqueeze(1) - int(sliding_window)
        )
    additive = torch.zeros((batch, 1, query_length, key_length), dtype=dtype, device=device)
    additive.masked_fill_(~allowed.view(1, 1, query_length, key_length), min_value)

    if attention_mask is None:
        return additive
    mask = attention_mask.to(device=device)
    if mask.ndim == 2:
        mask = mask[:, :key_length]
        if mask.shape[0] == 1 and batch > 1:
            mask = mask.expand(batch, -1)
        additive.masked_fill_(~mask.bool().view(batch, 1, 1, key_length), min_value)
    else:
        raise ValueError(f"unsupported attention mask rank for FaithVID: {mask.ndim}")
    return additive


def faithvid_attention_forward(
    module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    cache_position: Optional[torch.Tensor],
    scaling: float,
    dropout: float,
    output_attentions: bool,
    sliding_window: Optional[int] = None,
) -> Optional[tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """Run mass-corrected language attention only for FaithVID.

    FlashAttention2 has no arbitrary per-key additive-bias interface, so this
    path uses PyTorch SDPA while every other compression method remains on its
    configured attention backend.
    """
    config = _runtime_config(module)
    if config is None:
        return None
    group_mass, log_mass = _layer_metadata(config, int(getattr(module, "layer_idx", 0)))
    if not isinstance(group_mass, torch.Tensor) or not isinstance(log_mass, torch.Tensor):
        setattr(config, "last_faithvid_attention_skip", "mass_metadata_missing")
        if (
            bool(getattr(config, "faith_attention_strict", True))
            and getattr(config, "visual_token_length", None) is not None
        ):
            raise RuntimeError("FaithVID visual tokens exist but attention-mass metadata is missing")
        return None
    if int(group_mass.numel()) != int(log_mass.numel()):
        raise RuntimeError(
            "FaithVID group-mass and log-mass lengths differ: "
            f"{int(group_mass.numel())} != {int(log_mass.numel())}"
        )
    if int(query_states.shape[0]) != 1:
        raise RuntimeError("FaithVID currently requires one video per language-model batch")
    validation_key = (
        int(group_mass.data_ptr()),
        int(log_mass.data_ptr()),
        int(group_mass.numel()),
    )
    if getattr(config, "_faithvid_validated_metadata", None) != validation_key:
        if not bool(torch.isfinite(group_mass).all()) or bool((group_mass <= 0).any()):
            raise RuntimeError("FaithVID group mass must be finite and strictly positive")
        if not bool(torch.isfinite(log_mass).all()):
            raise RuntimeError("FaithVID attention log-mass contains NaN or Inf")
        config._faithvid_validated_metadata = validation_key

    query_heads = int(query_states.shape[1])
    key_states = _repeat_kv(key_states, query_heads)
    value_states = _repeat_kv(value_states, query_heads)
    additive = _base_attention_mask(
        query_states=query_states,
        key_states=key_states,
        attention_mask=attention_mask,
        cache_position=cache_position,
        sliding_window=sliding_window,
    )

    key_length = int(key_states.shape[-2])
    visual_start = int(getattr(config, "visual_token_start_index", -1))
    mass_length = int(log_mass.numel())
    if visual_start < 0 or visual_start + mass_length > key_length:
        setattr(config, "last_faithvid_attention_skip", "visual_range_mismatch")
        if bool(getattr(config, "faith_attention_strict", True)):
            raise RuntimeError(
                "FaithVID attention mass does not align with the language key sequence: "
                f"start={visual_start}, mass_length={mass_length}, key_length={key_length}"
            )
        return None
    key_bias = torch.zeros(key_length, dtype=additive.dtype, device=additive.device)
    key_bias[visual_start : visual_start + mass_length] = log_mass.to(
        device=additive.device, dtype=additive.dtype
    )
    additive = additive + key_bias.view(1, 1, 1, key_length)

    # Older supported PyTorch releases do not expose SDPA's ``scale`` keyword.
    # Rescaling Q preserves the requested model-specific attention scale.
    default_scale = 1.0 / math.sqrt(float(query_states.shape[-1]))
    scaled_query_states = query_states * (float(scaling) / default_scale)
    attn_output = F.scaled_dot_product_attention(
        scaled_query_states,
        key_states,
        value_states,
        attn_mask=additive,
        dropout_p=float(dropout),
        is_causal=False,
    )
    # Match Hugging Face attention interfaces: [B, Q, H, D], not SDPA's
    # native [B, H, Q, D]. The caller flattens the final two dimensions.
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_weights = None
    if output_attentions:
        last_scores = (
            query_states[:, :, -1:, :].float()
            @ key_states.transpose(-2, -1).float()
        ) * float(scaling)
        last_scores = last_scores + additive[:, :, -1:, :].float()
        attn_weights = torch.softmax(last_scores, dim=-1).to(dtype=query_states.dtype)

    setattr(config, "last_faithvid_attention_backend", "sdpa_additive_bias")
    return attn_output, attn_weights


def update_faithvid_after_inner_prune(
    config,
    keep_indices: torch.Tensor,
    *,
    visual_start: int,
    visual_length: int,
    hidden_states: Optional[torch.Tensor] = None,
    visual_global_indices: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Mass-pool removed visual states into retained anchors at layer K."""
    if str(getattr(config, "compression_variant", "")).strip().lower() != "faithvid":
        return hidden_states
    outer_mass = getattr(config, "_faithvid_outer_group_mass", None)
    outer_variance = getattr(config, "_faithvid_outer_variance", None)
    if not isinstance(outer_mass, torch.Tensor) or int(outer_mass.numel()) != int(visual_length):
        return hidden_states
    if not isinstance(outer_variance, torch.Tensor) or int(outer_variance.numel()) != int(visual_length):
        outer_variance = torch.zeros_like(outer_mass)

    if visual_global_indices is None:
        visual_positions = torch.arange(
            visual_start,
            visual_start + visual_length,
            device=keep_indices.device,
            dtype=torch.long,
        )
    else:
        visual_positions = visual_global_indices.to(device=keep_indices.device, dtype=torch.long)
        if int(visual_positions.numel()) != int(visual_length):
            raise ValueError(
                "FaithVID visual position count differs from its mass metadata: "
                f"{int(visual_positions.numel())} != {int(visual_length)}"
            )
    local = torch.where(torch.isin(visual_positions, keep_indices))[0]
    local = local.to(device=outer_mass.device, dtype=torch.long)
    if local.numel() == 0:
        return hidden_states
    all_local = torch.arange(visual_length, device=outer_mass.device)
    destination = (all_local.unsqueeze(1) - local.unsqueeze(0)).abs().argmin(dim=1)
    inner_mass = torch.zeros(local.numel(), dtype=torch.float32, device=outer_mass.device)
    inner_mass.index_add_(0, destination, outer_mass.float())
    variance_numerator = torch.zeros_like(inner_mass)
    variance_numerator.index_add_(0, destination, outer_variance.float() * outer_mass.float())

    if hidden_states is not None:
        positions = visual_positions.to(device=hidden_states.device, dtype=torch.long)
        source = hidden_states.index_select(1, positions).float()
        destination_on_hidden = destination.to(hidden_states.device)
        source_mass = outer_mass.to(hidden_states.device).float()
        pooled = torch.zeros(
            hidden_states.shape[0],
            local.numel(),
            hidden_states.shape[-1],
            dtype=torch.float32,
            device=hidden_states.device,
        )
        pooled.index_add_(1, destination_on_hidden, source * source_mass.view(1, -1, 1))
        pooled = pooled / inner_mass.to(hidden_states.device).view(1, -1, 1).clamp_min(1e-6)

        source_norm = F.normalize(source, p=2, dim=-1, eps=1e-6)
        pooled_norm = F.normalize(pooled, p=2, dim=-1, eps=1e-6)
        dispersion = (
            1.0 - (source_norm * pooled_norm.index_select(1, destination_on_hidden)).sum(dim=-1)
        ).clamp(0.0, 2.0)
        dispersion_numerator = torch.zeros_like(inner_mass, device=hidden_states.device)
        dispersion_numerator.index_add_(
            0,
            destination_on_hidden,
            (dispersion.mean(dim=0) * source_mass),
        )
        variance_numerator = variance_numerator + dispersion_numerator.to(variance_numerator.device)

        updated = hidden_states.clone()
        retained_positions = positions.index_select(0, local.to(positions.device))
        updated[:, retained_positions] = pooled.to(dtype=hidden_states.dtype)
        hidden_states = updated
    inner_variance = variance_numerator / inner_mass.clamp_min(1e-6)

    mass_strength = max(0.0, float(getattr(config, "faith_mass_strength", 1.0)))
    variance_strength = max(0.0, float(getattr(config, "faith_variance_strength", 0.50)))
    max_log_bias = max(0.0, float(getattr(config, "faith_max_log_bias", 20.0)))
    inner_log_mass = (
        mass_strength * torch.log(inner_mass.clamp_min(1.0))
        + 0.5 * variance_strength * inner_variance
    ).clamp(0.0, max_log_bias)
    config._faithvid_inner_group_mass = inner_mass
    config._faithvid_inner_variance = inner_variance
    config._faithvid_inner_log_mass = inner_log_mass
    setattr(config, "last_faithvid_inner_mass_error", float(abs(inner_mass.sum().item() - outer_mass.sum().item())))
    return hidden_states
