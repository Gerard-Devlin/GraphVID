from typing import Optional

from torch import nn
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2Model,
)
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaModel,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLAttention,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    Qwen2_5_VLTextModel,
    Qwen2_5_VLVisionAttention,
    Qwen2_5_VLVisionBlock,
    Qwen2_5_VisionTransformerPretrainedModel,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLVisionAttention,
    Qwen3VLVisionBlock,
    Qwen3VLVisionModel,
    Qwen3VLModel,
    Qwen3VLTextAttention,
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextModel,
)

from llava.model.llava_arch import LlavaMetaForCausalLM
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
from llava.model.multimodal_encoder.siglip_encoder import (
    SigLipAttention,
    SigLipVisionTower,
)

from .configuration_flashvid import FlashVidConfig
from .llava_arch import (
    LlavaMetaForCausalLM_encode_images,
    LlavaMetaForCausalLM_prepare_inputs_labels_for_multimodal,
)
from .modeling_qwen2 import (
    Qwen2Attention_forward,
    Qwen2DecoderLayer_forward,
    Qwen2Model_forward,
)
from .modeling_llama import (
    LlamaAttention_forward,
    LlamaDecoderLayer_forward,
    LlamaModel_forward,
)

from .modeling_qwen2_5_vl import (
    Qwen2_5_VLAttention_forward,
    Qwen2_5_VLModel_forward,
    Qwen2_5_VLTextModel_forward,
    Qwen2_5_VLModel_get_video_features,
    Qwen2_5_VLVisionAttention_forward,
    Qwen2_5_VLVisionBlock_forward,
    Qwen2_5_VisionTransformerPretrainedModel_forward,
    Qwen2_5_VLForConditionalGeneration_generate,
)

from .modeling_qwen3_vl import (
    Qwen3VLVisionAttention_forward,
    Qwen3VLVisionBlock_forward,
    Qwen3VLVisionModel_forward,
    Qwen3VLModel_forward,
    Qwen3VLTextAttention_forward,
    Qwen3VLTextDecoderLayer_forward,
    Qwen3VLTextModel_forward,
    Qwen3VLModel_get_image_features,
    Qwen3VLModel_get_video_features,
)

from .siglip_encoder import SigLipAttention_forward, SigLipVisionTower_forward


def flashvid(
    model: nn.Module,
    retention_ratio: float = 0.25,
    # 1) DySeg params (FIXED)
    do_segment: bool = True,
    segment_threshold: float = 0.9,
    min_segment_num: int = 8,
    complementary_segment: bool = True,
    # 2) ADTS and TSTM params
    token_selection_method: str = "attn_div_v2",
    alpha: float = 0.7,
    temporal_threshold: float = 0.8,
    dynamic_temporal_threshold: bool = False,
    temporal_threshold_quantile: float = 0.8,
    temporal_threshold_min: float = 0.0,
    temporal_threshold_max: float = 0.99,
    temporal_match_mode: str = "global",
    temporal_local_radius: int = 2,
    temporal_hysteresis: float = 0.0,
    min_keep_per_frame: int = 0,
    temporal_merge_mode: str = "tree",
    graph_temporal_topk: int = 3,
    graph_temporal_radius: int = 1,
    graph_temporal_skip: int = 1,
    graph_merge_protect_ratio: float = 0.15,
    graph_merge_target_ratio: float = 0.65,
    graph_merge_representative: str = "medoid",
    graph_representative_position: str = "protection",
    graph_protection_attn_weight: float = 0.70,
    graph_protection_novelty_weight: float = 0.30,
    graph_protection_detail_weight: float = 0.0,
    graph_adaptive_detail_protection: bool = False,
    graph_adaptive_detail_boost: float = 0.22,
    graph_adaptive_protect_boost: float = 0.10,
    graph_merge_importance_penalty: float = 0.0,
    graph_respect_temporal_threshold: bool = False,
    graph_final_tokens_per_frame: int = 0,
    graph_final_frame_floor_ratio: float = 0.55,
    graph_skip_spatial_merge_when_capped: bool = True,
    graft_temporal_topk: int = 3,
    graft_temporal_radius: int = 1,
    graft_temporal_skip: int = 1,
    graft_global_topk: int = 3,
    graft_input_is_residual: bool = True,
    graft_anchor_ratio: Optional[float] = None,
    graft_edge_threshold: float = 0.80,
    graft_component_radius_eps: float = 0.12,
    graft_split_radius_eps: float = 0.20,
    graft_parent_capacity: int = 1,
    graft_mutual_knn: bool = True,
    graft_one_token_per_frame: bool = True,
    graft_spatial_penalty: float = 0.10,
    graft_importance_penalty: float = 0.05,
    graft_hub_penalty: float = 0.05,
    graft_adaptive_aggregation: bool = True,
    graft_scene_threshold: float = 0.0,
    graft_min_tokens_per_frame: int = 0,
    graft_budget_correction: bool = True,
    graft_budget_diversity_weight: float = 0.35,
    graft_score_preset: str = "base",
    graft_duration_aware: bool = False,
    graft_medium_temporal_skip: Optional[int] = None,
    graft_medium_global_topk: Optional[int] = None,
    graft_medium_edge_threshold: Optional[float] = None,
    graft_medium_split_radius_eps: Optional[float] = None,
    graft_medium_spatial_penalty: Optional[float] = None,
    graft_medium_scene_threshold: Optional[float] = None,
    graft_long_temporal_skip: Optional[int] = None,
    graft_long_global_topk: Optional[int] = None,
    graft_long_edge_threshold: Optional[float] = None,
    graft_long_split_radius_eps: Optional[float] = None,
    graft_long_spatial_penalty: Optional[float] = None,
    graft_long_scene_threshold: Optional[float] = None,
    # 2.5) Experimental compression params
    compression_variant: str = "flashvid",
    question_aware_reweighting: bool = False,
    question_reweight_beta: float = 0.35,
    # Shared memory/adaptive params.
    memory_token_ratio: float = 0.10,
    memory_token_min: int = 1,
    memory_token_max: int = 16,
    adaptive_token_budget: bool = False,
    adaptive_budget_low: float = 0.10,
    adaptive_budget_mid: float = 0.15,
    adaptive_budget_high: float = 0.20,
    # 3) Inner-LLM Compression params
    expansion: float = 1.25,
    pruning_layer: int = 20,
    llm_retention_ratio: float = 0.3,
    # 4) Decode-stage policy scaffold (Route3, default no-op).
    decode_policy: str = "none",
    decode_kv_budget_ratio: float = 1.0,
    decode_update_interval: int = 4,
    decode_start_layer: int = 0,
) -> nn.Module:
    """Apply FlashVID to the model.

    Args:
        model (nn.Module): The model to apply FlashVID to.
        retention_ratio (float, optional): The retention ratio. Defaults to 0.25.
        do_segment (bool, optional): Whether to perform dynamic video segmentation. Defaults to True.
        segment_threshold (float, optional): The threshold for dynamic video segmentation. Defaults to 0.9.
        min_segment_num (int, optional): The minimum number of segments. Defaults to 8.
        complementary_segment (bool, optional): Whether to perform complementary segmentation. Defaults to True.
        token_selection_method (str, optional): The method for token selection. Defaults to "attn_div_v2".
        alpha (float, optional): The alpha for token selection. Defaults to 0.7.
        temporal_threshold (float, optional): The temporal threshold for token selection. Defaults to 0.8.
        dynamic_temporal_threshold (bool, optional): Whether to use quantile-based dynamic thresholding.
        temporal_threshold_quantile (float, optional): Quantile used when dynamic thresholding is enabled.
        temporal_threshold_min (float, optional): Lower bound for dynamic temporal threshold.
        temporal_threshold_max (float, optional): Upper bound for dynamic temporal threshold.
        temporal_match_mode (str, optional): Temporal matching mode. "global" or "local".
        temporal_local_radius (int, optional): Local matching radius when mode is "local".
        temporal_hysteresis (float, optional): Hysteresis margin for temporal merge decisions.
        min_keep_per_frame (int, optional): Minimum retained token count after TAM for each frame.
        compression_variant (str, optional): "flashvid" keeps original ADTS+TSTM;
            "graphvid" keeps ADTS/DPC but replaces tree-style temporal merging with graph merging;
            "graftvid" keeps ADTS/DPC but uses a constrained temporal forest;
        question_aware_reweighting (bool, optional): Enable question-guided token reweighting.
        question_reweight_beta (float, optional): Strength of question-aware reweighting.
        memory_token_ratio (float, optional): Budget ratio reserved for residual memory tokens.
        memory_token_min (int, optional): Minimum residual memory tokens.
        memory_token_max (int, optional): Maximum residual memory tokens.
        adaptive_token_budget (bool, optional): Enable adaptive token budget {10%, 15%, 20%}.
        adaptive_budget_low (float, optional): Low retention ratio candidate.
        adaptive_budget_mid (float, optional): Mid retention ratio candidate.
        adaptive_budget_high (float, optional): High retention ratio candidate.
        expansion (float, optional): The expansion ratio for inner-LLM compression. Defaults to 1.25.
        pruning_layer (int, optional): The layer to prune. Defaults to 20.
        llm_retention_ratio (float, optional): The retention ratio for inner-LLM compression. Defaults to 0.3.
        decode_policy (str, optional): Decode policy scaffold. "none" is a no-op.
        decode_kv_budget_ratio (float, optional): Reserved for future decode KV budgeting.
        decode_update_interval (int, optional): Reserved update interval for decode policy.
        decode_start_layer (int, optional): Reserved start layer for decode policy.

    Raises:
        NotImplementedError: If the model is not supported.

    Returns:
        nn.Module: The model with FlashVID applied.
    """

    # Replace with custom methods.
    if type(model) in (LlavaQwenForCausalLM, LlavaLlamaForCausalLM):  ## For LLaVA-OneVision or LLaVA-Video
        LlavaMetaForCausalLM.encode_images = LlavaMetaForCausalLM_encode_images
        LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = LlavaMetaForCausalLM_prepare_inputs_labels_for_multimodal
        SigLipAttention.forward = SigLipAttention_forward
        SigLipVisionTower.forward = SigLipVisionTower_forward
        # The paper configuration uses inner-LLM pruning (llm_retention_ratio=0.3).
        # Keep the stable vision-only path by default, but re-enable the Qwen2
        # language-model hooks when a caller explicitly asks for inner pruning.
        if float(llm_retention_ratio) < 0.9999:
            if type(model) is LlavaQwenForCausalLM:
                Qwen2Attention.forward = Qwen2Attention_forward
                Qwen2DecoderLayer.forward = Qwen2DecoderLayer_forward
                Qwen2Model.forward = Qwen2Model_forward
            else:
                LlamaAttention.forward = LlamaAttention_forward
                LlamaDecoderLayer.forward = LlamaDecoderLayer_forward
                LlamaModel.forward = LlamaModel_forward
        model.get_vision_tower().vision_tower.vision_model.encoder.layers[-1].self_attn.is_last_layer = True
    elif type(model) is Qwen2_5_VLForConditionalGeneration:  ## For Qwen2.5-VL
        Qwen2_5_VLAttention.forward = Qwen2_5_VLAttention_forward
        Qwen2_5_VLModel.get_video_features = Qwen2_5_VLModel_get_video_features
        Qwen2_5_VLTextModel.forward = Qwen2_5_VLTextModel_forward
        Qwen2_5_VLModel.forward = Qwen2_5_VLModel_forward
        Qwen2_5_VLVisionBlock.forward = Qwen2_5_VLVisionBlock_forward
        Qwen2_5_VLVisionAttention.forward = Qwen2_5_VLVisionAttention_forward
        Qwen2_5_VisionTransformerPretrainedModel.forward = Qwen2_5_VisionTransformerPretrainedModel_forward
        Qwen2_5_VLForConditionalGeneration.generate_ori = Qwen2_5_VLForConditionalGeneration.generate
        Qwen2_5_VLForConditionalGeneration.generate = Qwen2_5_VLForConditionalGeneration_generate
    elif type(model) is Qwen3VLForConditionalGeneration:  ## For Qwen3-VL
        if not hasattr(Qwen3VLVisionAttention, "_flashvid_original_forward"):
            Qwen3VLVisionAttention._flashvid_original_forward = Qwen3VLVisionAttention.forward
        if not hasattr(Qwen3VLVisionBlock, "_flashvid_original_forward"):
            Qwen3VLVisionBlock._flashvid_original_forward = Qwen3VLVisionBlock.forward
        if not hasattr(Qwen3VLVisionModel, "_flashvid_original_forward"):
            Qwen3VLVisionModel._flashvid_original_forward = Qwen3VLVisionModel.forward
        if not hasattr(Qwen3VLModel, "_flashvid_original_forward"):
            Qwen3VLModel._flashvid_original_forward = Qwen3VLModel.forward
        if not hasattr(Qwen3VLTextAttention, "_flashvid_original_forward"):
            Qwen3VLTextAttention._flashvid_original_forward = Qwen3VLTextAttention.forward
        if not hasattr(Qwen3VLTextDecoderLayer, "_flashvid_original_forward"):
            Qwen3VLTextDecoderLayer._flashvid_original_forward = Qwen3VLTextDecoderLayer.forward
        if not hasattr(Qwen3VLTextModel, "_flashvid_original_forward"):
            Qwen3VLTextModel._flashvid_original_forward = Qwen3VLTextModel.forward
        if not hasattr(Qwen3VLModel, "_flashvid_original_get_image_features"):
            Qwen3VLModel._flashvid_original_get_image_features = Qwen3VLModel.get_image_features
        if not hasattr(Qwen3VLModel, "_flashvid_original_get_video_features"):
            Qwen3VLModel._flashvid_original_get_video_features = Qwen3VLModel.get_video_features
        Qwen3VLVisionAttention.forward = Qwen3VLVisionAttention_forward
        Qwen3VLVisionBlock.forward = Qwen3VLVisionBlock_forward
        Qwen3VLVisionModel.forward = Qwen3VLVisionModel_forward
        Qwen3VLModel.forward = Qwen3VLModel_forward
        Qwen3VLTextAttention.forward = Qwen3VLTextAttention_forward
        Qwen3VLTextDecoderLayer.forward = Qwen3VLTextDecoderLayer_forward
        Qwen3VLTextModel.forward = Qwen3VLTextModel_forward
        Qwen3VLModel.get_image_features = Qwen3VLModel_get_image_features
        Qwen3VLModel.get_video_features = Qwen3VLModel_get_video_features
    else:
        raise NotImplementedError(f"FlashVID is not supported for {type(model)} yet.")

    variant = str(compression_variant).strip().lower()
    if variant not in ("flashvid", "graphvid", "graftvid"):
        raise ValueError(f"unsupported compression_variant={compression_variant!r}, expected flashvid|graphvid|graftvid")
    if variant == "graphvid":
        temporal_merge_mode = "graph"
    elif variant == "graftvid":
        temporal_merge_mode = "graft"

    # Create FlashVid config.
    flashvid_config = FlashVidConfig(
        retention_ratio=retention_ratio,
        do_segment=do_segment,
        segment_threshold=segment_threshold,
        min_segment_num=min_segment_num,
        complementary_segment=complementary_segment,
        alpha=alpha,
        token_selection_method=token_selection_method,
        temporal_threshold=temporal_threshold,
        dynamic_temporal_threshold=dynamic_temporal_threshold,
        temporal_threshold_quantile=temporal_threshold_quantile,
        temporal_threshold_min=temporal_threshold_min,
        temporal_threshold_max=temporal_threshold_max,
        temporal_match_mode=temporal_match_mode,
        temporal_local_radius=temporal_local_radius,
        temporal_hysteresis=temporal_hysteresis,
        min_keep_per_frame=min_keep_per_frame,
        temporal_merge_mode=temporal_merge_mode,
        graph_temporal_topk=graph_temporal_topk,
        graph_temporal_radius=graph_temporal_radius,
        graph_temporal_skip=graph_temporal_skip,
        graph_merge_protect_ratio=graph_merge_protect_ratio,
        graph_merge_target_ratio=graph_merge_target_ratio,
        graph_merge_representative=graph_merge_representative,
        graph_representative_position=graph_representative_position,
        graph_protection_attn_weight=graph_protection_attn_weight,
        graph_protection_novelty_weight=graph_protection_novelty_weight,
        graph_protection_detail_weight=graph_protection_detail_weight,
        graph_adaptive_detail_protection=graph_adaptive_detail_protection,
        graph_adaptive_detail_boost=graph_adaptive_detail_boost,
        graph_adaptive_protect_boost=graph_adaptive_protect_boost,
        graph_merge_importance_penalty=graph_merge_importance_penalty,
        graph_respect_temporal_threshold=graph_respect_temporal_threshold,
        graph_final_tokens_per_frame=graph_final_tokens_per_frame,
        graph_final_frame_floor_ratio=graph_final_frame_floor_ratio,
        graph_skip_spatial_merge_when_capped=graph_skip_spatial_merge_when_capped,
        graft_temporal_topk=graft_temporal_topk,
        graft_temporal_radius=graft_temporal_radius,
        graft_temporal_skip=graft_temporal_skip,
        graft_global_topk=graft_global_topk,
        graft_input_is_residual=graft_input_is_residual,
        graft_anchor_ratio=graft_anchor_ratio,
        graft_edge_threshold=graft_edge_threshold,
        graft_component_radius_eps=graft_component_radius_eps,
        graft_split_radius_eps=graft_split_radius_eps,
        graft_parent_capacity=graft_parent_capacity,
        graft_mutual_knn=graft_mutual_knn,
        graft_one_token_per_frame=graft_one_token_per_frame,
        graft_spatial_penalty=graft_spatial_penalty,
        graft_importance_penalty=graft_importance_penalty,
        graft_hub_penalty=graft_hub_penalty,
        graft_adaptive_aggregation=graft_adaptive_aggregation,
        graft_scene_threshold=graft_scene_threshold,
        graft_min_tokens_per_frame=graft_min_tokens_per_frame,
        graft_budget_correction=graft_budget_correction,
        graft_budget_diversity_weight=graft_budget_diversity_weight,
        graft_score_preset=graft_score_preset,
        graft_duration_aware=graft_duration_aware,
        graft_medium_temporal_skip=graft_medium_temporal_skip,
        graft_medium_global_topk=graft_medium_global_topk,
        graft_medium_edge_threshold=graft_medium_edge_threshold,
        graft_medium_split_radius_eps=graft_medium_split_radius_eps,
        graft_medium_spatial_penalty=graft_medium_spatial_penalty,
        graft_medium_scene_threshold=graft_medium_scene_threshold,
        graft_long_temporal_skip=graft_long_temporal_skip,
        graft_long_global_topk=graft_long_global_topk,
        graft_long_edge_threshold=graft_long_edge_threshold,
        graft_long_split_radius_eps=graft_long_split_radius_eps,
        graft_long_spatial_penalty=graft_long_spatial_penalty,
        graft_long_scene_threshold=graft_long_scene_threshold,
        compression_variant=variant,
        question_aware_reweighting=question_aware_reweighting,
        question_reweight_beta=question_reweight_beta,
        memory_token_ratio=memory_token_ratio,
        memory_token_min=memory_token_min,
        memory_token_max=memory_token_max,
        adaptive_token_budget=adaptive_token_budget,
        adaptive_budget_low=adaptive_budget_low,
        adaptive_budget_mid=adaptive_budget_mid,
        adaptive_budget_high=adaptive_budget_high,
        expansion=expansion,
        pruning_layer=pruning_layer,
        llm_retention_ratio=llm_retention_ratio,
        decode_policy=decode_policy,
        decode_kv_budget_ratio=decode_kv_budget_ratio,
        decode_update_interval=decode_update_interval,
        decode_start_layer=decode_start_layer,
    )

    # Store FlashVid Config in the model.
    setattr(model, "flashvid_config", flashvid_config)
    setattr(model.model, "flashvid_config", flashvid_config)
    setattr(model.config, "flashvid_bypass_active", False)
    if type(model) in (Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration):
        setattr(model.model.language_model, "flashvid_config", flashvid_config)

    return model
