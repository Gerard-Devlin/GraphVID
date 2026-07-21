from typing import Callable, Optional, Union, List, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, is_torchdynamo_compiling
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
    Qwen3VLVisionAttention,
    Qwen3VLVisionBlock,
    Qwen3VLVisionModel,
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
    Qwen3VLTextAttention,
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextModel,
    Qwen3VLForConditionalGeneration,
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


def _qwen3vl_bypass_enabled(module) -> bool:
    config = getattr(module, "config", None)
    if config is not None and bool(getattr(config, "flashvid_bypass_active", False)):
        return True
    attn = getattr(module, "self_attn", None)
    attn_config = getattr(attn, "config", None)
    return bool(getattr(attn_config, "flashvid_bypass_active", False))


def _set_qwen3vl_bypass_flags(model: Qwen3VLModel, active: bool) -> None:
    for cfg in (
        getattr(model, "config", None),
        getattr(getattr(model, "visual", None), "config", None),
        getattr(getattr(model, "language_model", None), "config", None),
    ):
        if cfg is not None:
            setattr(cfg, "flashvid_bypass_active", active)


def _call_qwen3vl_original_pipeline(self: Qwen3VLModel, **kwargs):
    originals = {
        (Qwen3VLVisionAttention, "forward"): Qwen3VLVisionAttention.forward,
        (Qwen3VLVisionBlock, "forward"): Qwen3VLVisionBlock.forward,
        (Qwen3VLVisionModel, "forward"): Qwen3VLVisionModel.forward,
        (Qwen3VLModel, "forward"): Qwen3VLModel.forward,
        (Qwen3VLTextAttention, "forward"): Qwen3VLTextAttention.forward,
        (Qwen3VLTextDecoderLayer, "forward"): Qwen3VLTextDecoderLayer.forward,
        (Qwen3VLTextModel, "forward"): Qwen3VLTextModel.forward,
        (Qwen3VLModel, "get_image_features"): Qwen3VLModel.get_image_features,
        (Qwen3VLModel, "get_video_features"): Qwen3VLModel.get_video_features,
    }
    try:
        Qwen3VLVisionAttention.forward = Qwen3VLVisionAttention._flashvid_original_forward
        Qwen3VLVisionBlock.forward = Qwen3VLVisionBlock._flashvid_original_forward
        Qwen3VLVisionModel.forward = Qwen3VLVisionModel._flashvid_original_forward
        Qwen3VLModel.forward = Qwen3VLModel._flashvid_original_forward
        Qwen3VLTextAttention.forward = Qwen3VLTextAttention._flashvid_original_forward
        Qwen3VLTextDecoderLayer.forward = Qwen3VLTextDecoderLayer._flashvid_original_forward
        Qwen3VLTextModel.forward = Qwen3VLTextModel._flashvid_original_forward
        Qwen3VLModel.get_image_features = Qwen3VLModel._flashvid_original_get_image_features
        Qwen3VLModel.get_video_features = Qwen3VLModel._flashvid_original_get_video_features
        return Qwen3VLModel._flashvid_original_forward(self, **kwargs)
    finally:
        for (cls, attr), fn in originals.items():
            setattr(cls, attr, fn)


def _call_qwen3vl_original_text_stack(language_model: Qwen3VLTextModel, **kwargs):
    originals = {
        (Qwen3VLTextAttention, "forward"): Qwen3VLTextAttention.forward,
        (Qwen3VLTextDecoderLayer, "forward"): Qwen3VLTextDecoderLayer.forward,
        (Qwen3VLTextModel, "forward"): Qwen3VLTextModel.forward,
    }
    try:
        Qwen3VLTextAttention.forward = Qwen3VLTextAttention._flashvid_original_forward
        Qwen3VLTextDecoderLayer.forward = Qwen3VLTextDecoderLayer._flashvid_original_forward
        Qwen3VLTextModel.forward = Qwen3VLTextModel._flashvid_original_forward
        return Qwen3VLTextModel._flashvid_original_forward(language_model, **kwargs)
    finally:
        for (cls, attr), fn in originals.items():
            setattr(cls, attr, fn)


def _talon_full_bypass_eligible(
    flashvid_config: Optional[FlashVidConfig],
    video_grid_thw: Optional[torch.LongTensor],
    spatial_merge_size: int,
    pixel_values_videos: Optional[torch.Tensor],
) -> bool:
    if flashvid_config is None or pixel_values_videos is None or video_grid_thw is None:
        return False
    variant = str(getattr(flashvid_config, "compression_variant", "flashvid")).strip().lower()
    if variant != "talon":
        return False
    if bool(getattr(flashvid_config, "adaptive_token_budget", False)):
        return False
    if float(getattr(flashvid_config, "retention_ratio", 1.0)) < 0.9999:
        return False
    if float(getattr(flashvid_config, "llm_retention_ratio", 1.0)) < 0.9999:
        return False
    decode_policy = str(getattr(flashvid_config, "decode_policy", "none") or "none").strip().lower()
    if decode_policy not in ("none", "off", "disabled"):
        return False
    frame_target = int(getattr(flashvid_config, "talon_target_tokens_per_frame", 0) or 0)
    if frame_target <= 0:
        return False
    num_visual_tokens = int((video_grid_thw[0][1].item() * video_grid_thw[0][2].item()) // max(1, spatial_merge_size**2))
    return frame_target >= num_visual_tokens


def _use_original_qwen3vl_text_stack(flashvid_config: Optional[FlashVidConfig]) -> bool:
    if flashvid_config is None:
        return False
    variant = str(getattr(flashvid_config, "compression_variant", "flashvid")).strip().lower()
    if variant != "talon":
        return False
    if float(getattr(flashvid_config, "llm_retention_ratio", 1.0)) < 0.9999:
        return False
    decode_policy = str(getattr(flashvid_config, "decode_policy", "none") or "none").strip().lower()
    if decode_policy not in ("none", "off", "disabled"):
        return False
    return True


def _prefuse_deepstack_visual_embeds(
    inputs_embeds: torch.Tensor,
    visual_pos_masks: Optional[torch.Tensor],
    deepstack_visual_embeds: Optional[list[torch.Tensor]],
    scale: float = 0.35,
) -> torch.Tensor:
    if visual_pos_masks is None or deepstack_visual_embeds is None or len(deepstack_visual_embeds) == 0:
        return inputs_embeds
    if visual_pos_masks.ndim != 2 or inputs_embeds.ndim != 3:
        return inputs_embeds
    visual_mask = visual_pos_masks[0]
    if int(visual_mask.sum().item()) <= 0:
        return inputs_embeds
    fused = torch.stack([x.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device) for x in deepstack_visual_embeds], dim=0).mean(dim=0)
    if fused.shape[0] != int(visual_mask.sum().item()) or fused.shape[-1] != inputs_embeds.shape[-1]:
        return inputs_embeds
    updated = inputs_embeds.clone()
    updated[:, visual_mask] = updated[:, visual_mask] + float(scale) * fused.unsqueeze(0)
    return updated


def _talon_should_keep_deepstack(flashvid_config: FlashVidConfig) -> bool:
    mode = str(getattr(flashvid_config, "talon_deepstack_mode", "keep") or "keep").strip().lower()
    if mode in ("keep", "on", "enabled", "true"):
        return True
    if mode in ("disable", "off", "disabled", "none", "false"):
        return False

    output_mode = str(getattr(flashvid_config, "talon_output_mode", "manifold") or "manifold").strip().lower()
    if output_mode not in ("manifold", "manifold_raw", "raw"):
        return False
    if abs(float(getattr(flashvid_config, "talon_reconstruction_blend", 0.0))) > 1e-6:
        return False
    memory_mode = str(getattr(flashvid_config, "talon_memory_mode", "raw") or "raw").strip().lower()
    return memory_mode in ("raw", "anchor", "anchors", "select")


def Qwen3VLVisionAttention_forward(
    self: Qwen3VLVisionAttention,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    return_logits: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if _qwen3vl_bypass_enabled(self):
        return type(self)._flashvid_original_forward(
            self,
            hidden_states,
            cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    if self.config._attn_implementation != "flash_attention_2":
        raise RuntimeError(
            "Qwen3 FlashVID vision path requires attn_implementation=flash_attention_2; "
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
        # Calculate attention weights manually.
        num_frames = cu_seqlens.shape[0] - 1
        q, k = query_states.squeeze(0), key_states.squeeze(0)
        # reshape to (seq_length, num_heads, head_dim)
        q, k = q.transpose(0, 1), k.transpose(0, 1)
        q = q.reshape(num_frames, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(num_frames, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / self.head_dim**0.5
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = attn_weights.mean(1).mean(1)
    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    attn_output = self.proj(attn_output)
    return attn_output, attn_weights


def Qwen3VLVisionBlock_forward(
    self: Qwen3VLVisionBlock,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> torch.Tensor:
    if _qwen3vl_bypass_enabled(self):
        return type(self)._flashvid_original_forward(
            self,
            hidden_states,
            cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
    residual = hidden_states
    hidden_states, attn_weights = self.attn(
        self.norm1(hidden_states),
        cu_seqlens=cu_seqlens,
        rotary_pos_emb=rotary_pos_emb,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.mlp(self.norm2(hidden_states))
    hidden_states = residual + hidden_states

    return hidden_states, attn_weights


def Qwen3VLVisionModel_forward(
    self: Qwen3VLVisionModel,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    if _qwen3vl_bypass_enabled(self):
        return type(self)._flashvid_original_forward(
            self,
            hidden_states,
            grid_thw,
            **kwargs,
        )
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

    pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
    hidden_states = hidden_states + pos_embeds

    rotary_pos_emb = self.rot_pos_emb(grid_thw)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)
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

    num_blocks = len(self.blocks)
    deepstack_feature_lists = []
    for layer_num, blk in enumerate(self.blocks):
        # Return attention weights of the last layer for compression.
        return_logits = (num_blocks - 1) == layer_num
        hidden_states, attn_weights = blk(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            return_logits=return_logits,
            **kwargs,
        )
        if layer_num in self.deepstack_visual_indexes:
            deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](hidden_states)
            deepstack_feature_lists.append(deepstack_feature)

    hidden_states = self.merger(hidden_states)

    # Process attn_weights
    num_frames = grid_thw[0][0].item()
    seq_len = attn_weights.shape[-1] // 4
    attn_weights = attn_weights.view(num_frames, seq_len, -1).mean(-1)

    return hidden_states, deepstack_feature_lists, attn_weights


def Qwen3VLModel_forward(
    self: Qwen3VLModel,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Union[tuple, Qwen3VLModelOutputWithPast]:
    r"""
    image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of feature shape of each image in LLM.
    video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of feature shape of each video in LLM.
    """
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    flashvid_config: Optional[FlashVidConfig] = getattr(self, "flashvid_config", None)
    spatial_merge = max(1, int(getattr(self.visual, "spatial_merge_size", 2)))
    if _talon_full_bypass_eligible(
        flashvid_config=flashvid_config,
        video_grid_thw=video_grid_thw,
        spatial_merge_size=spatial_merge,
        pixel_values_videos=pixel_values_videos,
    ):
        return _call_qwen3vl_original_pipeline(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            cache_position=cache_position,
            **kwargs,
        )

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    image_mask = None
    video_mask = None

    if pixel_values is not None:
        image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        # ! Obtain [CLS] attentions for FlashVID compression.
        video_embeds, deepstack_video_embeds, cls_attention = self.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        n_video_tokens = video_embeds.shape[0]
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    visual_pos_masks = None
    deepstack_visual_embeds = None
    if image_mask is not None and video_mask is not None:
        # aggregate visual_pos_masks and deepstack_visual_embeds
        image_mask = image_mask[..., 0]
        video_mask = video_mask[..., 0]
        visual_pos_masks = image_mask | video_mask
        deepstack_visual_embeds = []
        image_mask_joint = image_mask[visual_pos_masks]
        video_mask_joint = video_mask[visual_pos_masks]
        for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
            embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
            embed_joint[image_mask_joint, :] = img_embed
            embed_joint[video_mask_joint, :] = vid_embed
            deepstack_visual_embeds.append(embed_joint)
    elif image_mask is not None:
        image_mask = image_mask[..., 0]
        visual_pos_masks = image_mask
        deepstack_visual_embeds = deepstack_image_embeds
    elif video_mask is not None:
        video_mask = video_mask[..., 0]
        visual_pos_masks = video_mask
        deepstack_visual_embeds = deepstack_video_embeds

    if position_ids is None:
        attention_mask_tensor = (
            attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
        )
        if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
            attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
            # Only apply conversion for floating point tensors (inverted masks)
            if attention_mask_tensor.dtype.is_floating_point:
                attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()

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
                attention_mask=attention_mask_tensor,
            )
            self.rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    ### ! Applies FlashVid compression here.
    if position_ids.shape[-1] > 1 and pixel_values_videos is not None:
        num_frames, num_visual_tokens = cls_attention.shape
        flashvid_config: FlashVidConfig = getattr(self, "flashvid_config")
        setattr(flashvid_config, "_certvid_attention_source", "manual_qk")
        # Store feature map resolution.
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
            deepstack_features=(
                deepstack_video_embeds
                if str(getattr(flashvid_config, "compression_variant", "")).strip().lower() == "prismvid"
                else None
            ),
        )

        non_visual_token_indexes = torch.where(
            (input_ids[0] != self.config.vision_start_token_id)
            & (input_ids[0] != self.config.vision_end_token_id)
            & (input_ids[0] != self.config.video_token_id))[0]
        visual_token_indexes = torch.where(input_ids[0] == self.config.video_token_id)[0]
        visual_start_index = visual_token_indexes[0].item()
        visual_length = n_video_tokens
        # Update FlashVid config.
        flashvid_config.visual_token_start_index = visual_start_index
        flashvid_config.vision_token_length = int(compressed_video_tokens.shape[0])
        flashvid_config.llm_token_length = None
        flashvid_config.visual_token_length = compressed_video_tokens.shape[0] # ! NOTE
        compression_variant = str(
            getattr(flashvid_config, "compression_variant", "flashvid")
        ).strip().lower()
        if compression_variant in {"certvid", "certvid_v2", "certvid_v3", "certvid_v6", "certvid_hr", "certvid_v4", "certvid_v5", "certvid_e", "faithvid"}:
            from .certvid_qwen3 import compress_certvid_deepstack, merge_certvid_visual_deepstack

            certvid_plan = getattr(flashvid_config, "_certvid_plan", None)
            if certvid_plan is None:
                raise RuntimeError(
                    f"{compression_variant} compression did not publish its DeepStack aggregation plan"
                )
            if compression_variant == "faithvid":
                from .faithvid import apply_faithvid_position_centroids

                position_ids = apply_faithvid_position_centroids(
                    flashvid_config,
                    position_ids,
                    visual_token_indexes,
                )
            try:
                compressed_deepstack_video = compress_certvid_deepstack(
                    deepstack_video_embeds,
                    certvid_plan,
                )
                if image_mask is not None:
                    deepstack_visual_embeds = merge_certvid_visual_deepstack(
                        deepstack_image_embeds=deepstack_image_embeds,
                        compressed_video_embeds=compressed_deepstack_video,
                        image_mask=image_mask,
                        video_mask=video_mask,
                        kept_video_indices=keep_visual_global_indices,
                    )
                else:
                    deepstack_visual_embeds = compressed_deepstack_video
            finally:
                # The plan owns GPU tensors and is only valid for this prefill.
                flashvid_config._certvid_plan = None
        elif compression_variant == "prismvid":
            from .prismvid import compress_prism_deepstack, merge_prism_visual_deepstack

            compressed_deepstack_video = compress_prism_deepstack(
                deepstack_video_embeds,
                keep_visual_global_indices,
            )
            if image_mask is not None:
                deepstack_visual_embeds = merge_prism_visual_deepstack(
                    deepstack_image_embeds=deepstack_image_embeds,
                    compressed_video_embeds=compressed_deepstack_video,
                    image_mask=image_mask,
                    video_mask=video_mask,
                    kept_video_indices=keep_visual_global_indices,
                )
            else:
                deepstack_visual_embeds = compressed_deepstack_video
        elif (
            compression_variant == "talon"
            and not _talon_should_keep_deepstack(flashvid_config)
        ):
            deepstack_visual_embeds = None
        elif deepstack_visual_embeds is not None:
            # Keep DeepStack aligned only when TALON outputs stay on the raw-token manifold.
            deepstack_visual_embeds = [
                deepstack_visual_embed[
                    keep_visual_global_indices.to(deepstack_visual_embed.device)
                ]
                for deepstack_visual_embed in deepstack_visual_embeds
            ]
        keep_global_indexes = (
            torch.cat(
                [
                    visual_token_indexes[keep_visual_global_indices],
                    non_visual_token_indexes,
                ],
                dim=0,
            )
            .sort()
            .values
        )

        hidden_size = inputs_embeds.size(-1)
        assert visual_token_indexes[keep_visual_global_indices].shape[0] == compressed_video_tokens.view(-1, hidden_size).shape[0]
        inputs_embeds[:, visual_token_indexes[keep_visual_global_indices]] = compressed_video_tokens.view(-1, hidden_size).unsqueeze(0)
        inputs_embeds = inputs_embeds[:, keep_global_indexes]
        position_ids = position_ids[:, :, keep_global_indexes]
        attention_mask = attention_mask[:, keep_global_indexes]
        cache_position = cache_position[keep_global_indexes]
        visual_pos_masks = visual_pos_masks[:, keep_global_indexes]

    if _use_original_qwen3vl_text_stack(flashvid_config):
        inputs_embeds = _prefuse_deepstack_visual_embeds(
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        outputs = _call_qwen3vl_original_text_stack(
            self.language_model,
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
    else:
        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

    return Qwen3VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        rope_deltas=self.rope_deltas,
    )


def Qwen3VLTextModel_forward(
    self: Qwen3VLTextModel,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    # args for deepstack
    visual_pos_masks: Optional[torch.Tensor] = None,
    deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Union[tuple, BaseModelOutputWithPast]:
    r"""
    visual_pos_masks (`torch.Tensor` of shape `(batch_size, seqlen)`, *optional*):
        The mask of the visual positions.
    deepstack_visual_embeds (`list[torch.Tensor]`, *optional*):
        The deepstack visual embeddings. The shape is (num_layers, visual_seqlen, embed_dim).
        The feature is extracted from the different visual encoder layers, and fed to the decoder
        hidden states. It's from the paper DeepStack(https://arxiv.org/abs/2406.04334).
    """
    if _qwen3vl_bypass_enabled(self):
        return type(self)._flashvid_original_forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

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

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = position_ids[0]

    attention_mask = create_causal_mask(
        config=self.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=text_position_ids,
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # Obtain FlashVid config
    if not hasattr(self, "flashvid_config"):
        raise ValueError("FlashVid configuration is not set in the model.")
    flashvid_config: FlashVidConfig = getattr(self, "flashvid_config")
    is_prefill = hidden_states.shape[1] > 1
    is_certvid = str(
        getattr(flashvid_config, "compression_variant", "flashvid")
    ).strip().lower() in {
        "certvid",
        "certvid_v2",
        "certvid_v3",
        "certvid_v6",
        "certvid_hr",
        "certvid_v4",
        "certvid_v5",
        "certvid_e",
        "faithvid",
    }
    enable_inner_pruning = is_prefill and (
        not is_certvid
        or float(getattr(flashvid_config, "llm_retention_ratio", 1.0)) < 0.9999
    )

    # decoder layers
    for layer_idx, decoder_layer in enumerate(self.layers):
        # Only prunes visual tokens at prefilling stage.
        if enable_inner_pruning:
            if layer_idx == flashvid_config.pruning_layer - 1:
                kwargs["output_attentions"] = True
            elif layer_idx == flashvid_config.pruning_layer:
                kwargs["output_attentions"] = False
                attn = layer_outputs[1]
                (
                    hidden_states,
                    attention_mask,
                    text_position_ids,
                    cache_position,
                    position_embeddings,
                    _,
                ) = fastv_prune(
                    hidden_states=hidden_states,
                    causal_mask=attention_mask,
                    attentions=attn,
                    cache_position=cache_position,
                    position_ids=text_position_ids,
                    position_embeddings=position_embeddings,
                    flashvid_config=flashvid_config,
                    visual_pos_masks=visual_pos_masks,
                )

        (
            hidden_states,
            attention_mask,
            text_position_ids,
            cache_position,
            position_embeddings,
        ) = maybe_apply_decode_policy(
            hidden_states=hidden_states,
            causal_mask=attention_mask,
            position_ids=text_position_ids,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            flashvid_config=flashvid_config,
            layer_idx=layer_idx,
            is_prefill=is_prefill,
        )

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = layer_outputs[0]

        # add visual features to the hidden states of first several layers
        if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
            hidden_states = self._deepstack_process(
                hidden_states,
                visual_pos_masks,
                deepstack_visual_embeds[layer_idx],
            )

    hidden_states = self.norm(hidden_states)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )


def Qwen3VLModel_get_image_features(
    self: Qwen3VLModel,
    pixel_values: torch.FloatTensor,
    image_grid_thw: Optional[torch.LongTensor] = None,
):
    if _qwen3vl_bypass_enabled(self):
        return type(self)._flashvid_original_get_image_features(
            self,
            pixel_values,
            image_grid_thw=image_grid_thw,
        )
    """
    Encodes images into continuous embeddings that can be forwarded to the language model. The deepstack visual features are also returned.

    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The tensors corresponding to the input images.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
    """
    pixel_values = pixel_values.type(self.visual.dtype)
    image_embeds, deepstack_image_embeds, cls_attention = self.visual(pixel_values, grid_thw=image_grid_thw)
    split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
    image_embeds = torch.split(image_embeds, split_sizes)
    return image_embeds, deepstack_image_embeds, cls_attention


def Qwen3VLModel_get_video_features(
    self: Qwen3VLModel,
    pixel_values_videos: torch.FloatTensor,
    video_grid_thw: Optional[torch.LongTensor] = None,
):
    if _qwen3vl_bypass_enabled(self):
        return type(self)._flashvid_original_get_video_features(
            self,
            pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )
    """
    Encodes videos into continuous embeddings that can be forwarded to the language model.
    Also returns deepstack visual features and cls-attention signals for FlashVID compression.

    Args:
        pixel_values_videos (`torch.FloatTensor`):
            The tensors corresponding to the input videos.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
    """
    pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
    video_embeds, deepstack_video_embeds, cls_attention = self.visual(
        pixel_values_videos,
        grid_thw=video_grid_thw,
    )
    split_sizes = (video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
    video_embeds = torch.split(video_embeds, split_sizes)
    return video_embeds, deepstack_video_embeds, cls_attention


def Qwen3VLTextDecoderLayer_forward(
    self: Qwen3VLTextDecoderLayer,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> torch.Tensor:
    if _qwen3vl_bypass_enabled(self):
        return type(self)._flashvid_original_forward(
            self,
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    # Self Attention
    hidden_states, attn_weights = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states, attn_weights


def Qwen3VLTextAttention_forward(
    self: Qwen3VLTextAttention,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if _qwen3vl_bypass_enabled(self):
        return type(self)._flashvid_original_forward(
            self,
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
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
        output_attentions=bool(kwargs.get("output_attentions", False)),
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
            **kwargs,
        )

    if kwargs.get("output_attentions", False) and attn_weights is None:
        # * Calculate attention weights manually if not provided
        last_query = query_states[:, :, -1:, :]
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        # key_states = key_states.transpose(1, 2)
        attn_weights = torch.matmul(last_query, key_states.transpose(2, 3)) / self.head_dim**0.5
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights
