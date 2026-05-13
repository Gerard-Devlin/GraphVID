from torch import nn
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2Model,
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
    # 2.5) Experimental compression params
    compression_variant: str = "flashvid",
    question_aware_reweighting: bool = False,
    question_reweight_beta: float = 0.35,
    # TALON params.
    talon_transport_radius: int = 1,
    talon_rank_ratio: float = 0.40,
    talon_rank_min: int = 2,
    talon_rank_max: int = 32,
    talon_budget_scale: float = 0.60,
    talon_target_tokens_per_frame: int = 0,
    talon_min_total_tokens: int = 1,
    talon_fast_rank_plan: bool = True,
    talon_background_max_ratio: float = 0.45,
    talon_frame_balanced_selection: bool = True,
    talon_basis_method: str = "randomized",
    talon_basis_oversample: int = 4,
    talon_innovation_attention_weight: float = 0.45,
    talon_motion_importance_weight: float = 0.35,
    talon_boundary_importance_weight: float = 0.10,
    talon_question_frame_weight: float = 0.20,
    talon_frame_balanced_memory: bool = True,
    talon_memory_mode: str = "raw",
    talon_anchor_safety_ratio: float = 0.28,
    talon_budget_strategy: str = "marginal",
    talon_budget_mode: str = "uniform",
    talon_transport_mode: str = "hard",
    talon_transport_temperature: float = 0.07,
    talon_rd_spectral_weight: float = 1.0,
    talon_rd_innovation_weight: float = 1.0,
    talon_use_question_innovation: bool = True,
    talon_innovation_qweight: float = 0.25,
    talon_output_mode: str = "manifold",
    talon_reconstruction_blend: float = 0.0,
    talon_anchor_score_weight: float = 0.35,
    talon_min_anchor_per_frame: int = 2,
    talon_passthrough_ratio: float = 0.15,
    talon_passthrough_min: int = 2,
    talon_use_segmentation: bool = True,
    talon_disable_oversegmentation: bool = True,
    talon_max_segments: int = 4,
    talon_deepstack_mode: str = "keep",
    # Shared memory/adaptive params.
    memory_token_ratio: float = 0.10,
    memory_token_min: int = 1,
    memory_token_max: int = 16,
    adaptive_token_budget: bool = False,
    adaptive_budget_low: float = 0.10,
    adaptive_budget_mid: float = 0.15,
    adaptive_budget_high: float = 0.20,
    talon_adaptive_target_low: int = 0,
    talon_adaptive_target_mid: int = 0,
    talon_adaptive_target_high: int = 0,
    talon_complexity_floor: float = 0.20,
    talon_complexity_ceil: float = 0.40,
    talon_adaptive_gamma: float = 1.0,
    talon_adaptive_target_enabled: bool = True,
    talon_force_fixed_target: bool = False,
    talon_target_mean_cap: float = 18.75,
    talon_unified_selection: bool = True,
    talon_low_budget_mode_threshold: int = 20,
    talon_low_budget_rank_cap: int = 0,
    talon_background_global_ratio: float = 0.60,
    talon_event_budget_ratio: float = 0.30,
    talon_memory_fused_weight: float = 0.50,
    talon_memory_residual_weight: float = 0.35,
    talon_memory_frame_weight: float = 0.15,
    talon_recall_memory_mode: str = "raw",
    talon_final_fused_weight: float = 0.70,
    talon_final_residual_weight: float = 0.20,
    talon_final_frame_weight: float = 0.10,
    talon_anchor_keep_bonus: float = 0.10,
    talon_recall_keep_bonus: float = 0.08,
    talon_final_anchor_min_ratio: float = 0.24,
    talon_final_recall_min_ratio: float = 0.10,
    talon_force_anchor_recall_quota: bool = True,
    talon_global_topk_ratio: float = 0.70,
    talon_rescue_enabled: bool = True,
    talon_rescue_ratio: float = 0.08,
    talon_rescue_from_memory_only: bool = True,
    talon_rescue_fused_weight: float = 0.55,
    talon_rescue_residual_weight: float = 0.35,
    talon_rescue_frame_weight: float = 0.10,
    talon_rescue_global_ratio: float = 0.85,
    talon_rerank_with_flash_prior: bool = True,
    talon_flash_prior_ratio: float = 0.20,
    talon_recall_semantic_ratio: float = 0.50,
    talon_recall_event_ratio: float = 0.25,
    talon_recall_frame_ratio: float = 0.15,
    talon_recall_global_ratio: float = 0.55,
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
            "talon" enables transport-aligned low-rank + sparse innovation compression.
        question_aware_reweighting (bool, optional): Enable question-guided token reweighting.
        question_reweight_beta (float, optional): Strength of question-aware reweighting.
        talon_transport_radius (int, optional): Local transport radius for frame-to-frame token alignment.
        talon_rank_ratio (float, optional): Per-frame low-rank share in TALON token budget.
        talon_rank_min (int, optional): Minimum TALON low-rank token count per frame when budget allows.
        talon_rank_max (int, optional): Maximum TALON low-rank token count per frame.
        talon_budget_scale (float, optional): TALON-only multiplier over the shared visual budget.
        talon_target_tokens_per_frame (int, optional): Fixed TALON target width per frame; 0 disables it.
        talon_min_total_tokens (int, optional): Lower bound on TALON output tokens per segment.
        talon_fast_rank_plan (bool, optional): Use one-pass rate-distortion rank planning.
        talon_background_max_ratio (float, optional): Max low-rank background share of per-frame TALON budget.
        talon_frame_balanced_selection (bool, optional): Keep passthrough/innovation budgets frame-balanced.
        talon_basis_method (str, optional): Low-rank basis solver, "randomized" or "covariance".
        talon_basis_oversample (int, optional): Randomized basis oversampling rank.
        talon_innovation_attention_weight (float, optional): Attention/fused-score share in innovation scoring.
        talon_motion_importance_weight (float, optional): Transition/motion share in frame budget allocation.
        talon_boundary_importance_weight (float, optional): First/last-frame prior in frame budget allocation.
        talon_question_frame_weight (float, optional): Question-frame semantic share in frame budget allocation.
        talon_frame_balanced_memory (bool, optional): Build residual memory tokens with frame coverage.
        talon_memory_mode (str, optional): "raw" keeps representative memory anchors; "merge" averages residual groups.
        talon_anchor_safety_ratio (float, optional): Extra raw attention-anchor share protected before TALON factors.
        talon_budget_strategy (str, optional): Budget split policy, one of {"ratio","marginal"}.
        talon_budget_mode (str, optional): Frame budget policy, one of {"uniform","attention"}.
        talon_transport_mode (str, optional): Local transport mode, one of {"hard","soft"}.
        talon_transport_temperature (float, optional): Soft local-transport temperature.
        talon_rd_spectral_weight (float, optional): Rate-distortion spectral-tail weight.
        talon_rd_innovation_weight (float, optional): Rate-distortion innovation-tail weight.
        talon_use_question_innovation (bool, optional): Reweight innovation selection with question cues.
        talon_innovation_qweight (float, optional): Question-aware weight for innovation scoring.
        talon_output_mode (str, optional): "manifold" keeps outputs close to pretrained visual tokens;
            "coefficient" exposes raw TALON coefficients/residuals for ablation.
        talon_reconstruction_blend (float, optional): Blend low-rank reconstruction into anchor tokens.
        talon_anchor_score_weight (float, optional): Mix question/attention score into low-rank anchor picking.
        talon_min_anchor_per_frame (int, optional): Minimum raw attention anchors kept per frame when budget allows.
        talon_passthrough_ratio (float, optional): Ratio of high-confidence raw tokens kept unchanged in TALON.
        talon_passthrough_min (int, optional): Minimum TALON passthrough token count per segment when budget allows.
        talon_use_segmentation (bool, optional): Whether TALON should segment the video before compression.
        talon_disable_oversegmentation (bool, optional): Avoid excessive short segments for TALON path.
        talon_max_segments (int, optional): Upper bound on TALON segment count when oversegmentation guard is enabled.
        talon_deepstack_mode (str, optional): TALON handling for Qwen3-VL DeepStack, one of {"disable","keep","auto"}.
        memory_token_ratio (float, optional): Budget ratio reserved for residual memory tokens.
        memory_token_min (int, optional): Minimum residual memory tokens.
        memory_token_max (int, optional): Maximum residual memory tokens.
        adaptive_token_budget (bool, optional): Enable adaptive token budget {10%, 15%, 20%}.
        adaptive_budget_low (float, optional): Low retention ratio candidate.
        adaptive_budget_mid (float, optional): Mid retention ratio candidate.
        adaptive_budget_high (float, optional): High retention ratio candidate.
        talon_adaptive_target_low (int, optional): TALON low-complexity target tokens per frame when adaptive budget is enabled.
        talon_adaptive_target_mid (int, optional): TALON mid-complexity target tokens per frame when adaptive budget is enabled.
        talon_adaptive_target_high (int, optional): TALON high-complexity target tokens per frame when adaptive budget is enabled.
        talon_complexity_floor (float, optional): Lower bound used to normalize TALON complexity score for adaptive targeting.
        talon_complexity_ceil (float, optional): Upper bound used to normalize TALON complexity score for adaptive targeting.
        talon_adaptive_gamma (float, optional): Nonlinear gain for adaptive TALON target interpolation.
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
    if type(model) is LlavaQwenForCausalLM:  ## For LLaVA-OneVision or LLaVA-Video
        LlavaMetaForCausalLM.encode_images = LlavaMetaForCausalLM_encode_images
        LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = LlavaMetaForCausalLM_prepare_inputs_labels_for_multimodal
        SigLipAttention.forward = SigLipAttention_forward
        SigLipVisionTower.forward = SigLipVisionTower_forward
        Qwen2Attention.forward = Qwen2Attention_forward
        Qwen2DecoderLayer.forward = Qwen2DecoderLayer_forward
        Qwen2Model.forward = Qwen2Model_forward
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
    if variant not in ("flashvid", "talon"):
        raise ValueError(f"unsupported compression_variant={compression_variant!r}, expected flashvid|talon")

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
        compression_variant=variant,
        question_aware_reweighting=question_aware_reweighting,
        question_reweight_beta=question_reweight_beta,
        talon_transport_radius=talon_transport_radius,
        talon_rank_ratio=talon_rank_ratio,
        talon_rank_min=talon_rank_min,
        talon_rank_max=talon_rank_max,
        talon_budget_scale=talon_budget_scale,
        talon_target_tokens_per_frame=talon_target_tokens_per_frame,
        talon_min_total_tokens=talon_min_total_tokens,
        talon_fast_rank_plan=talon_fast_rank_plan,
        talon_background_max_ratio=talon_background_max_ratio,
        talon_frame_balanced_selection=talon_frame_balanced_selection,
        talon_basis_method=talon_basis_method,
        talon_basis_oversample=talon_basis_oversample,
        talon_innovation_attention_weight=talon_innovation_attention_weight,
        talon_motion_importance_weight=talon_motion_importance_weight,
        talon_boundary_importance_weight=talon_boundary_importance_weight,
        talon_question_frame_weight=talon_question_frame_weight,
        talon_frame_balanced_memory=talon_frame_balanced_memory,
        talon_memory_mode=talon_memory_mode,
        talon_anchor_safety_ratio=talon_anchor_safety_ratio,
        talon_budget_strategy=talon_budget_strategy,
        talon_budget_mode=talon_budget_mode,
        talon_transport_mode=talon_transport_mode,
        talon_transport_temperature=talon_transport_temperature,
        talon_rd_spectral_weight=talon_rd_spectral_weight,
        talon_rd_innovation_weight=talon_rd_innovation_weight,
        talon_use_question_innovation=talon_use_question_innovation,
        talon_innovation_qweight=talon_innovation_qweight,
        talon_output_mode=talon_output_mode,
        talon_reconstruction_blend=talon_reconstruction_blend,
        talon_anchor_score_weight=talon_anchor_score_weight,
        talon_min_anchor_per_frame=talon_min_anchor_per_frame,
        talon_passthrough_ratio=talon_passthrough_ratio,
        talon_passthrough_min=talon_passthrough_min,
        talon_use_segmentation=talon_use_segmentation,
        talon_disable_oversegmentation=talon_disable_oversegmentation,
        talon_max_segments=talon_max_segments,
        talon_deepstack_mode=talon_deepstack_mode,
        memory_token_ratio=memory_token_ratio,
        memory_token_min=memory_token_min,
        memory_token_max=memory_token_max,
        adaptive_token_budget=adaptive_token_budget,
        adaptive_budget_low=adaptive_budget_low,
        adaptive_budget_mid=adaptive_budget_mid,
        adaptive_budget_high=adaptive_budget_high,
        talon_adaptive_target_low=talon_adaptive_target_low,
        talon_adaptive_target_mid=talon_adaptive_target_mid,
        talon_adaptive_target_high=talon_adaptive_target_high,
        talon_complexity_floor=talon_complexity_floor,
        talon_complexity_ceil=talon_complexity_ceil,
        talon_adaptive_gamma=talon_adaptive_gamma,
        talon_adaptive_target_enabled=talon_adaptive_target_enabled,
        talon_force_fixed_target=talon_force_fixed_target,
        talon_target_mean_cap=talon_target_mean_cap,
        talon_unified_selection=talon_unified_selection,
        talon_low_budget_mode_threshold=talon_low_budget_mode_threshold,
        talon_low_budget_rank_cap=talon_low_budget_rank_cap,
        talon_background_global_ratio=talon_background_global_ratio,
        talon_event_budget_ratio=talon_event_budget_ratio,
        talon_memory_fused_weight=talon_memory_fused_weight,
        talon_memory_residual_weight=talon_memory_residual_weight,
        talon_memory_frame_weight=talon_memory_frame_weight,
        talon_recall_memory_mode=talon_recall_memory_mode,
        talon_final_fused_weight=talon_final_fused_weight,
        talon_final_residual_weight=talon_final_residual_weight,
        talon_final_frame_weight=talon_final_frame_weight,
        talon_anchor_keep_bonus=talon_anchor_keep_bonus,
        talon_recall_keep_bonus=talon_recall_keep_bonus,
        talon_final_anchor_min_ratio=talon_final_anchor_min_ratio,
        talon_final_recall_min_ratio=talon_final_recall_min_ratio,
        talon_force_anchor_recall_quota=talon_force_anchor_recall_quota,
        talon_global_topk_ratio=talon_global_topk_ratio,
        talon_rescue_enabled=talon_rescue_enabled,
        talon_rescue_ratio=talon_rescue_ratio,
        talon_rescue_from_memory_only=talon_rescue_from_memory_only,
        talon_rescue_fused_weight=talon_rescue_fused_weight,
        talon_rescue_residual_weight=talon_rescue_residual_weight,
        talon_rescue_frame_weight=talon_rescue_frame_weight,
        talon_rescue_global_ratio=talon_rescue_global_ratio,
        talon_rerank_with_flash_prior=talon_rerank_with_flash_prior,
        talon_flash_prior_ratio=talon_flash_prior_ratio,
        talon_recall_semantic_ratio=talon_recall_semantic_ratio,
        talon_recall_event_ratio=talon_recall_event_ratio,
        talon_recall_frame_ratio=talon_recall_frame_ratio,
        talon_recall_global_ratio=talon_recall_global_ratio,
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
