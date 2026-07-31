from typing import Callable, Optional, Union, List, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs, is_flash_attn_available
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, is_torchdynamo_compiling
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
    Qwen2_5_VLAttention,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLTextModel,
    Qwen2_5_VLModel,
    Qwen2_5_VLVisionAttention,
    Qwen2_5_VLVisionBlock,
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLModelOutputWithPast,
    repeat_kv,
)
from .configuration_flashvid import FlashVidConfig
from .faithvid_attention import faithvid_attention_forward
from .utils import (
    extract_question_features,
    fastv_prune,
    flashvid_compression,
    maybe_apply_decode_policy,
)


def Qwen2_5_VLTextModel_forward(
    self: Qwen2_5_VLTextModel,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Union[tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    # torch.jit.trace() doesn't support cache objects in the output
    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache(config=self.config)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    # NOTE: we need to pass text position ids for packing. Qwen2-VL uses 3D positions
    # where each dim indicates visual spatial positions for temporal/height/width grids.
    # There are two scenarios when FA2-like packed masking might be activated.
    # 1. User specifically passed packed `position_ids` and no attention mask.
    #    In this case we expect the useer to create correct position ids for all 3 grids
    #    and prepend text-only position ids to it. The final tensor will be [4, bs, seq-len]
    # 2. User runs forward with no attention mask and no position ids. In this case, position ids
    #    are prepared by the model (`get_rope_index`) as `[4, bs, seq-len]` tensor. Text-only positions are
    #    prepended by us when creating positions so that the mask is constructed correctly. NOTE: failing to pass
    #    text-only positions will cause incorrect mask construction, do not change `prepare_input_for_generation`
    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        # If inputs are not packed (usual 3D positions), do not prepare mask from position_ids
        text_position_ids = None

    # It may already have been prepared by e.g. `generate`
    if not isinstance(causal_mask_mapping := attention_mask, dict):
        # Prepare mask arguments
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": text_position_ids,
        }
        # Create the masks
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        # The sliding window alternating layers are not always activated depending on the config
        if self.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    # Obtain FlashVid config
    if not hasattr(self, "flashvid_config"):
        raise ValueError("FlashVid configuration is not set in the model.")
    flashvid_config: FlashVidConfig = getattr(self, "flashvid_config")
    is_prefill = hidden_states.shape[1] > 1

    assert all(decoder_layer.attention_type == "full_attention" for decoder_layer in self.layers[: self.config.num_hidden_layers])
    _output_attentions = output_attentions
    causal_mask = causal_mask_mapping["full_attention"]
    for layer_idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        # Only prunes visual tokens at prefilling stage.
        if is_prefill:
            if layer_idx == flashvid_config.pruning_layer - 1:
                output_attentions = True
            elif layer_idx == flashvid_config.pruning_layer:
                output_attentions = _output_attentions
                attn = layer_outputs[1]
                (
                    hidden_states,
                    causal_mask,
                    position_ids,
                    cache_position,
                    position_embeddings,
                    keep_indices,
                ) = fastv_prune(
                    hidden_states=hidden_states,
                    causal_mask=causal_mask,
                    attentions=attn,
                    cache_position=cache_position,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    flashvid_config=flashvid_config,
                )
                # Don't forget to update text_position_ids (otherwise may occur CUDA error)
                text_position_ids = text_position_ids[..., keep_indices].contiguous()

        (
            hidden_states,
            causal_mask,
            text_position_ids,
            cache_position,
            position_embeddings,
        ) = maybe_apply_decode_policy(
            hidden_states=hidden_states,
            causal_mask=causal_mask,
            position_ids=text_position_ids,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            flashvid_config=flashvid_config,
            layer_idx=layer_idx,
            is_prefill=is_prefill,
        )

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = layer_outputs[0]

        if _output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    if not return_dict:
        return tuple(
            v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def Qwen2_5_VisionTransformerPretrainedModel_forward(
    self: Qwen2_5_VisionTransformerPretrainedModel,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
            The final hidden states of the model.
        grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
            The temporal, height and width of feature shape of each image in LLM.

    Returns:
        `torch.Tensor`: hidden_states.
    """
    hidden_states = self.patch_embed(hidden_states)
    rotary_pos_emb = self.rot_pos_emb(grid_thw)
    window_index, cu_window_seqlens = self.get_window_index(grid_thw)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens,
        device=hidden_states.device,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        # Select dtype based on the following factors:
        #  - FA2 requires that cu_seqlens_q must have dtype int32
        #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
        # See https://github.com/huggingface/transformers/pull/34852 for more information
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    flashvid_config: FlashVidConfig = getattr(self, "flashvid_config")
    variant = str(getattr(flashvid_config, "compression_variant", "")).strip().lower()
    num_blocks = len(self.blocks)
    for layer_num, blk in enumerate(self.blocks):
        if layer_num in self.fullatt_block_indexes:
            cu_seqlens_now = cu_seqlens
        else:
            cu_seqlens_now = cu_window_seqlens

        # FastV prunes from language attention and PruneVID builds its own
        # visual decomposition, so neither baseline needs the expensive final
        # vision QK metric.
        return_logits = (num_blocks - 1) == layer_num and variant not in {
            "fastv",
            "prunevid",
        }
        hidden_states, attn_weights = blk(
            hidden_states,
            cu_seqlens=cu_seqlens_now,
            position_embeddings=position_embeddings,
            return_logits=return_logits,
            score_mode=variant,
        )

    hidden_states = self.merger(hidden_states)
    reverse_indices = torch.argsort(window_index)
    hidden_states = hidden_states[reverse_indices, :]

    # FastVID's released Qwen2.5 path segments frames using post-merger frame
    # features rather than the pre-merger vision CLS/pooling representation.
    num_frames = grid_thw[0][0].item()
    if variant == "fastvid":
        if hidden_states.shape[0] % num_frames != 0:
            raise RuntimeError(
                "Qwen2.5 FastVID cannot reshape merged vision tokens into frames: "
                f"tokens={hidden_states.shape[0]}, frames={num_frames}"
            )
        setattr(
            flashvid_config,
            "_fastvid_frame_global_features",
            hidden_states.view(num_frames, -1, hidden_states.shape[-1]).float().mean(dim=1),
        )

    if variant == "visionzip":
        raw_metric = getattr(self.blocks[-1].attn, "visionzip_metric", None)
        if raw_metric is None or raw_metric.ndim != 3:
            raise RuntimeError("Qwen2.5 VisionZip did not receive the final post-RoPE key metric")
        merge_unit = int(self.spatial_merge_unit)
        if raw_metric.shape[0] % merge_unit != 0:
            raise RuntimeError(
                "Qwen2.5 VisionZip key length is not divisible by the spatial merge unit: "
                f"keys={raw_metric.shape[0]}, merge_unit={merge_unit}"
            )
        merged_metric = raw_metric.view(
            raw_metric.shape[0] // merge_unit,
            merge_unit,
            raw_metric.shape[1],
            raw_metric.shape[2],
        ).mean(dim=1)
        merged_metric = merged_metric.mean(dim=1)[reverse_indices]
        if merged_metric.shape[0] % num_frames != 0:
            raise RuntimeError(
                "Qwen2.5 VisionZip cannot reshape merged key metrics into frames: "
                f"keys={merged_metric.shape[0]}, frames={num_frames}"
            )
        setattr(
            flashvid_config,
            "_visionzip_metric",
            merged_metric.view(num_frames, -1, merged_metric.shape[-1]),
        )
        self.blocks[-1].attn.visionzip_metric = None

    merged_tokens_per_frame = int(hidden_states.shape[0] // num_frames)
    if attn_weights is None:
        # These values are shape-only placeholders. The FastV identity adapter
        # and PruneVID adapter both discard cls_attention.
        attn_weights = torch.zeros(
            (num_frames, merged_tokens_per_frame),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
    elif variant == "fastvid":
        # The released FastVID Qwen2.5 hook already scores merged 2x2 keys.
        if tuple(attn_weights.shape) != (num_frames, merged_tokens_per_frame):
            raise RuntimeError(
                "Qwen2.5 FastVID attention metric has an unexpected shape: "
                f"got {tuple(attn_weights.shape)}, expected "
                f"{(num_frames, merged_tokens_per_frame)}"
            )
        attn_weights = attn_weights.reshape(-1)[reverse_indices].view(num_frames, -1)
    else:
        # FlashVID scores within each frame. VisionZip scores all raw visual
        # keys globally. Both metrics are reduced over the spatial merger's
        # groups before restoring the original token order.
        merge_unit = int(self.spatial_merge_unit)
        raw_attention = attn_weights.reshape(-1)
        expected_raw_tokens = int(hidden_states.shape[0] * merge_unit)
        if int(raw_attention.numel()) != expected_raw_tokens:
            raise RuntimeError(
                "Qwen2.5 visual attention metric does not align with the "
                f"spatial merger: scores={int(raw_attention.numel())}, "
                f"expected={expected_raw_tokens}"
            )
        attn_weights = raw_attention.view(-1, merge_unit).mean(dim=-1)
        attn_weights = attn_weights[reverse_indices].view(num_frames, -1)

    return hidden_states, attn_weights


def Qwen2_5_VLVisionBlock_forward(
    self: Qwen2_5_VLVisionBlock,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    return_logits: bool = False,
    score_mode: str = "flashvid",
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    residual = hidden_states
    hidden_states, attn_weights = self.attn(
        self.norm1(hidden_states),
        cu_seqlens=cu_seqlens,
        rotary_pos_emb=rotary_pos_emb,
        position_embeddings=position_embeddings,
        return_logits=return_logits,
        score_mode=score_mode,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.mlp(self.norm2(hidden_states))
    hidden_states = residual + hidden_states

    return hidden_states, attn_weights


def Qwen2_5_VLAttention_forward(
    self: Qwen2_5_VLAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    dropout = 0.0 if not self.training else self.attention_dropout
    faith_output = faithvid_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        cache_position=cache_position,
        scaling=self.scaling,
        dropout=dropout,
        output_attentions=bool(output_attentions),
        sliding_window=self.sliding_window,
    )
    if faith_output is not None:
        attn_output, attn_weights = faith_output
    else:
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            position_ids=position_ids,  # pass positions for FA2
            **kwargs,
        )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if output_attentions and attn_weights is None:
        # FlashAttention2 does not return weights. Reconstruct only the rows
        # required by the active method instead of materializing full S x S.
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        flashvid_config = getattr(self, "flashvid_config", None)
        variant = str(
            getattr(flashvid_config, "compression_variant", "")
        ).strip().lower()

        if variant == "prunevid":
            # PruneVID scores visual tokens with the maximum attention over all
            # text queries after the visual span and over all heads. This is
            # intentionally different from FastV's final-query mean.
            visual_start = int(
                getattr(flashvid_config, "visual_token_start_index", 0)
            )
            visual_length = int(
                getattr(flashvid_config, "visual_token_length", 0)
            )
            visual_end = visual_start + visual_length
            key_length = int(key_states.shape[-2])
            query_length = int(query_states.shape[-2])
            text_start = min(max(visual_end, 0), query_length)

            if (
                visual_length <= 0
                or visual_start < 0
                or visual_end > key_length
                or text_start >= query_length
            ):
                raise RuntimeError(
                    "PruneVID cannot locate post-visual text queries for its "
                    "language-attention selector"
                )

            visual_scores = torch.zeros(
                (query_states.shape[0], visual_length),
                dtype=torch.float32,
                device=query_states.device,
            )
            key_positions = torch.arange(
                key_length,
                device=query_states.device,
            )
            query_chunk = 16
            for start in range(text_start, query_length, query_chunk):
                end = min(query_length, start + query_chunk)
                logits = torch.matmul(
                    query_states[:, :, start:end],
                    key_states.transpose(2, 3),
                ) / self.head_dim**0.5

                # Preserve causal semantics even when FA2 receives no explicit
                # additive mask. Any padding/additive mask is applied as well.
                query_positions = torch.arange(
                    start,
                    end,
                    device=query_states.device,
                )
                causal = key_positions.view(1, 1, 1, -1) > query_positions.view(
                    1,
                    1,
                    -1,
                    1,
                )
                logits = logits.masked_fill(
                    causal,
                    torch.finfo(logits.dtype).min,
                )
                if attention_mask is not None:
                    if attention_mask.ndim == 4:
                        if int(attention_mask.shape[-2]) == 1:
                            additive_mask = attention_mask[..., :1, :key_length]
                        else:
                            additive_mask = attention_mask[
                                ...,
                                start:end,
                                :key_length,
                            ]
                        logits = logits + additive_mask
                    elif attention_mask.ndim == 2:
                        invalid_keys = attention_mask[:, None, None, :key_length] == 0
                        logits = logits.masked_fill(
                            invalid_keys,
                            torch.finfo(logits.dtype).min,
                        )

                probabilities = nn.functional.softmax(
                    logits,
                    dim=-1,
                    dtype=torch.float32,
                )
                chunk_scores = probabilities[
                    ...,
                    visual_start:visual_end,
                ].amax(dim=(1, 2))
                visual_scores = torch.maximum(visual_scores, chunk_scores)

            # Keep the legacy attention tensor interface consumed by
            # fastv_prune while carrying PruneVID's method-specific scores.
            attn_weights = torch.zeros(
                (query_states.shape[0], 1, 1, key_length),
                dtype=query_states.dtype,
                device=query_states.device,
            )
            attn_weights[
                ...,
                visual_start:visual_end,
            ] = visual_scores[:, None, None].to(query_states.dtype)
        else:
            # Released FastV/FlashVID inner pruning uses the final query row.
            last_query = query_states[:, :, -1:, :]
            attn_weights = (
                torch.matmul(last_query, key_states.transpose(2, 3))
                / self.head_dim**0.5
            )
            attn_weights = nn.functional.softmax(
                attn_weights,
                dim=-1,
                dtype=torch.float32,
            ).to(query_states.dtype)

    return attn_output, attn_weights


@torch.no_grad()
def Qwen2_5_VLForConditionalGeneration_generate(
    self: Qwen2_5_VLForConditionalGeneration,
    **kwargs,
):
    flashvid_config: FlashVidConfig = getattr(self, "flashvid_config")
    # Obtain the visual token start index and length
    visual_token_start_index = torch.where(kwargs["input_ids"][0] == self.config.video_token_id)[0][0].item()
    visual_token_length = torch.where(kwargs["input_ids"][0] == self.config.video_token_id)[0].shape[0]
    # Update FlashVid Config.
    flashvid_config.visual_token_start_index = visual_token_start_index
    flashvid_config.visual_token_length = visual_token_length

    try:
        return self.generate_ori(**kwargs)
    finally:
        # Baseline metadata belongs to one prefill only. Clear it even when
        # generation raises so a later request cannot consume stale tensors or
        # PruneVID quotas.
        for name in (
            "_fastvid_frame_global_features",
            "_visionzip_metric",
            "_prunevid_group_sizes",
            "_prunevid_target_tokens",
        ):
            setattr(flashvid_config, name, None)
        visual = getattr(getattr(self, "model", None), "visual", None)
        blocks = getattr(visual, "blocks", None)
        if blocks:
            setattr(blocks[-1].attn, "visionzip_metric", None)


def Qwen2_5_VLModel_forward(
    self: Qwen2_5_VLModel,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Union[tuple, Qwen2_5_VLModelOutputWithPast]:
    r"""
    image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of feature shape of each image in LLM.
    video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of feature shape of each video in LLM.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
        The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
    """

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    if pixel_values is not None:
        image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_embeds, cls_attention = self.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        n_video_tokens = video_embeds.shape[0]
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    if position_ids is None:
        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids = position_ids + delta.to(position_ids.device)

    ### Applies FlashVid compression here.
    if position_ids.shape[-1] > 1 and pixel_values_videos is not None:
        num_frames, num_visual_tokens = cls_attention.shape
        flashvid_config: FlashVidConfig = getattr(self, "flashvid_config")
        setattr(flashvid_config, "_certvid_attention_source", "manual_qk")
        spatial_merge = max(1, int(getattr(self.visual, "spatial_merge_size", 2)))
        flashvid_config.H = int(video_grid_thw[0][1].item() // spatial_merge)
        flashvid_config.W = int(video_grid_thw[0][2].item() // spatial_merge)
        video_features = video_embeds.view(num_frames, num_visual_tokens, -1)
        question_features = extract_question_features(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            invalid_token_ids=[
                getattr(self.config, "video_token_id", None),
                getattr(self.config, "image_token_id", None),
                getattr(self.config, "vision_start_token_id", None),
                getattr(self.config, "vision_end_token_id", None),
            ],
        )
        compressed_video_tokens, keep_visual_global_indices = flashvid_compression(
            video_features=video_features,
            cls_attention=cls_attention,
            flashvid_config=flashvid_config,
            question_features=question_features,
        )
        visual_start_index = torch.where(input_ids[0] == self.config.video_token_id)[0][0].item()
        visual_length = n_video_tokens
        visual_end_index = visual_start_index + visual_length
        visual_token_indexes = torch.where(input_ids[0] == self.config.video_token_id)[0]
        if str(getattr(flashvid_config, "compression_variant", "")).strip().lower() == "faithvid":
            from .faithvid import apply_faithvid_position_centroids

            position_ids = apply_faithvid_position_centroids(
                flashvid_config,
                position_ids,
                visual_token_indexes,
            )
            # Qwen2.5 has no DeepStack consumer, so the GPU-heavy plan is no
            # longer needed after position centroids have been materialized.
            flashvid_config._certvid_plan = None
        # Update FlashVid config.
        flashvid_config.visual_token_start_index = visual_start_index
        flashvid_config.vision_token_length = int(compressed_video_tokens.shape[0])
        flashvid_config.llm_token_length = None
        flashvid_config.visual_token_length = compressed_video_tokens.shape[0]
        # Filter `position_ids`, `attention_mask`, `inputs_embeds`
        global_indices = torch.arange(input_ids.shape[-1]).to(input_ids)
        keep_visual_global_indices += visual_start_index
        keep_global_indices = (
            torch.cat(
                [
                    global_indices[:visual_start_index],
                    keep_visual_global_indices,
                    global_indices[visual_end_index:],
                ],
                dim=0,
            )
            .sort()
            .values
        )
        bsz, _, hidden_size = inputs_embeds.shape
        inputs_embeds.scatter_(
            dim=1,
            index=keep_visual_global_indices.unsqueeze(0).unsqueeze(-1).expand(bsz, -1, hidden_size),
            src=compressed_video_tokens.view(-1, hidden_size).unsqueeze(0),
        )
        inputs_embeds = torch.gather(
            inputs_embeds,
            dim=1,
            index=keep_global_indices.view(1, -1, 1).expand(bsz, -1, hidden_size),
        )
        position_ids = position_ids[:, :, keep_global_indices]
        attention_mask = attention_mask[:, keep_global_indices]
        cache_position = cache_position[keep_global_indices]
    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )

    output = Qwen2_5_VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )
    return output if return_dict else output.to_tuple()


def Qwen2_5_VLVisionAttention_forward(
    self: Qwen2_5_VLVisionAttention,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    return_logits: bool = False,
    score_mode: str = "flashvid",
    **kwargs,
) -> torch.Tensor:
    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    )
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
    if bool(getattr(self, "capture_visionzip", False)):
        # VisionZip's Qwen2.5 implementation merges post-RoPE keys from the
        # final vision block, after matching the spatial merger's 4-token groups.
        self.visionzip_metric = key_states.detach()

    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    if self.config._attn_implementation != "flash_attention_2":
        raise RuntimeError(
            "Qwen2.5 FlashVID vision path requires attn_implementation=flash_attention_2; "
            f"got {self.config._attn_implementation!r}."
        )
    # Flash Attention 2: Use cu_seqlens for variable length attention
    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
    attn_output, _ = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask=None,
        scaling=self.scaling,
        dropout=0.0 if not self.training else self.attention_dropout,
        cu_seq_lens_q=cu_seqlens,
        cu_seq_lens_k=cu_seqlens,
        max_length_q=max_seqlen,
        max_length_k=max_seqlen,
        is_causal=False,
        **kwargs,
    )

    attn_weights = None
    if return_logits:
        num_frames = cu_seqlens.shape[0] - 1
        q_heads = query_states.squeeze(0)
        k_heads = key_states.squeeze(0)
        raw_tokens_per_frame = int(q_heads.shape[-2] // num_frames)
        mode = str(score_mode).strip().lower()

        if mode == "fastvid":
            # Match the released Qwen2.5 FastVID metric: one pooled query per
            # frame attends to 2x2-merged visual keys.
            merge_unit = 4
            if raw_tokens_per_frame % merge_unit != 0:
                raise RuntimeError(
                    "Qwen2.5 FastVID raw frame tokens are not divisible by 4: "
                    f"{raw_tokens_per_frame}"
                )
            q_by_frame = q_heads.view(
                self.num_heads,
                num_frames,
                raw_tokens_per_frame,
                self.head_dim,
            ).permute(1, 0, 2, 3)
            k_by_frame = k_heads.view(
                self.num_heads,
                num_frames,
                raw_tokens_per_frame // merge_unit,
                merge_unit,
                self.head_dim,
            ).mean(dim=3).permute(1, 0, 2, 3)
            pooled_query = q_by_frame.mean(dim=2, keepdim=True)
            scores = torch.matmul(
                pooled_query,
                k_by_frame.transpose(-1, -2),
            ) / self.head_dim**0.5
            attn_weights = nn.functional.softmax(
                scores,
                dim=-1,
                dtype=torch.float32,
            ).to(q_heads.dtype)
            attn_weights = attn_weights.mean(dim=1).mean(dim=1)
        elif mode == "visionzip":
            # Match the released Qwen2.5 VisionZip metric: global incoming
            # attention mass over every visual query. Query chunking preserves
            # the same score while avoiding one full H x T x T allocation.
            total_tokens = int(q_heads.shape[-2])
            incoming_mass = torch.zeros(
                total_tokens,
                dtype=torch.float32,
                device=q_heads.device,
            )
            query_chunk = 32
            key_transposed = k_heads.transpose(-1, -2)
            for start in range(0, total_tokens, query_chunk):
                logits = torch.matmul(
                    q_heads[:, start : start + query_chunk],
                    key_transposed,
                ) / self.head_dim**0.5
                probabilities = nn.functional.softmax(logits, dim=-1)
                incoming_mass.add_(
                    probabilities.mean(dim=0).float().sum(dim=0)
                )
            attn_weights = incoming_mass.to(dtype=q_heads.dtype)
        else:
            # FlashVID's released Qwen2.5 score: mean incoming attention inside
            # each frame. Accumulate query chunks so high-resolution videos do
            # not materialize the full F x H x S x S attention tensor.
            q_by_frame = q_heads.view(
                self.num_heads,
                num_frames,
                raw_tokens_per_frame,
                self.head_dim,
            ).permute(1, 0, 2, 3).contiguous()
            k_by_frame = k_heads.view(
                self.num_heads,
                num_frames,
                raw_tokens_per_frame,
                self.head_dim,
            ).permute(1, 0, 2, 3).contiguous()
            incoming_mass = torch.zeros(
                (num_frames, raw_tokens_per_frame),
                dtype=torch.float32,
                device=q_heads.device,
            )
            query_chunk = 32
            key_transposed = k_by_frame.transpose(-1, -2)
            for start in range(0, raw_tokens_per_frame, query_chunk):
                logits = torch.matmul(
                    q_by_frame[:, :, start : start + query_chunk],
                    key_transposed,
                ) / self.head_dim**0.5
                probabilities = nn.functional.softmax(
                    logits,
                    dim=-1,
                    dtype=torch.float32,
                ).to(q_heads.dtype)
                incoming_mass.add_(
                    probabilities.mean(dim=1).float().sum(dim=1)
                )
            attn_weights = (
                incoming_mass / float(raw_tokens_per_frame)
            ).to(dtype=q_heads.dtype)
    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    attn_output = self.proj(attn_output)
    return attn_output, attn_weights


def Qwen2_5_VLModel_get_video_features(
    self: Qwen2_5_VLModel,
    pixel_values_videos: torch.FloatTensor,
    video_grid_thw: Optional[torch.LongTensor] = None,
):
    """
    Encodes videos into continuous embeddings that can be forwarded to the language model.

    Args:
        pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The tensors corresponding to the input videos.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
    """
    pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
    video_embeds, cls_attention = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
    split_sizes = (video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
    video_embeds = torch.split(video_embeds, split_sizes)
    return video_embeds, cls_attention
