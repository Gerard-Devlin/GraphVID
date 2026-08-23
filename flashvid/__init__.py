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


def _text_layer_count(model: nn.Module) -> int:
    """Read the decoder depth without depending on one Transformers config layout."""
    configs = [getattr(model, "config", None)]
    root_config = configs[0]
    if root_config is not None:
        configs.extend(
            [
                getattr(root_config, "text_config", None),
                getattr(root_config, "llm_config", None),
            ]
        )
    inner = getattr(model, "model", None)
    configs.append(getattr(inner, "config", None))
    configs.append(getattr(getattr(inner, "language_model", None), "config", None))
    configs.append(getattr(getattr(model, "language_model", None), "config", None))
    try:
        model_root = model.get_model()
        configs.append(getattr(model_root, "config", None))
        configs.append(getattr(getattr(model_root, "language_model", None), "config", None))
    except (AttributeError, TypeError):
        pass
    for config in configs:
        value = (
            config.get("num_hidden_layers")
            if isinstance(config, dict)
            else getattr(config, "num_hidden_layers", None)
        )
        if value is not None and int(value) > 0:
            return int(value)
    return 0


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
    cert_budget_uses_expansion: bool = True,
    cert_query_atoms: int = 6,
    cert_temporal_bins: int = 8,
    cert_spatial_bins: int = 3,
    cert_candidate_multiplier: float = 3.0,
    cert_query_weight: float = 0.20,
    cert_temporal_weight: float = 0.20,
    cert_detail_weight: float = 0.10,
    cert_repair_ratio: float = 0.20,
    cert_fusion_alpha: float = 0.25,
    cert_assignment_temperature: float = 0.07,
    cert_track_threshold: float = 0.82,
    cert_spatial_penalty: float = 0.08,
    cert_metric_dim: int = 256,
    certv2_budget_uses_expansion: bool = True,
    certv2_query_atoms: int = 6,
    certv2_temporal_bins: int = 8,
    certv2_spatial_bins: int = 3,
    certv2_candidate_multiplier: float = 3.0,
    certv2_query_weight: float = 0.18,
    certv2_frame_floor_ratio: float = 0.08,
    certv2_diversity_weight: float = 0.12,
    certv2_coverage_weight: float = 0.10,
    certv2_density_neighbors: int = 4,
    certv2_track_threshold: float = 0.82,
    certv2_spatial_penalty: float = 0.08,
    certv2_metric_dim: int = 256,
    certv2_repair_ratio: float = 0.05,
    certv2_repair_ratio_high: float = 0.13,
    certv2_router_strength: float = 0.65,
    certv2_protect_ratio: float = 0.30,
    certv2_swap_margin: float = 0.02,
    certv2_fusion_alpha: float = 0.25,
    certv2_repair_fusion_alpha: float = 0.08,
    certv2_assignment_temperature: float = 0.07,
    certv3_budget_uses_expansion: bool = True,
    certv3_query_atoms: int = 8,
    certv3_temporal_bins: int = 12,
    certv3_spatial_bins: int = 3,
    certv3_candidate_multiplier: float = 2.5,
    certv3_query_weight: float = 0.18,
    certv3_visual_attention_weight: float = 0.28,
    certv3_visual_novelty_weight: float = 0.20,
    certv3_visual_curvature_weight: float = 0.14,
    certv3_visual_event_weight: float = 0.12,
    certv3_visual_detail_weight: float = 0.12,
    certv3_visual_component_weight: float = 0.14,
    certv3_event_novelty_weight: float = 0.34,
    certv3_event_curvature_weight: float = 0.28,
    certv3_event_frame_weight: float = 0.18,
    certv3_event_detail_weight: float = 0.10,
    certv3_event_query_weight: float = 0.10,
    certv3_track_threshold: float = 0.82,
    certv3_spatial_penalty: float = 0.08,
    certv3_metric_dim: int = 96,
    certv3_frame_coverage_ratio: float = 1.0,
    certv3_cell_coverage_ratio: float = 0.50,
    certv3_query_threshold: float = 0.10,
    certv3_query_per_atom: int = 1,
    certv3_structural_weight: float = 0.32,
    certv3_whitening_strength: float = 0.50,
    certv3_quality_floor: float = 0.15,
    certv3_ridge: float = 0.50,
    certv3_swap_steps: int = 6,
    certv3_swap_pool: int = 24,
    certv3_swap_margin: float = 1e-4,
    certv3_fusion_alpha: float = 0.12,
    certv3_assignment_temperature: float = 0.07,
    certv3_certificate_budget_ratio: float = 1.0,
    certv3_selection_objective: str = "d_optimal",
    certv3_use_spatiotemporal_certificates: bool = True,
    certv3_use_spatiotemporal_design: bool = True,
    certv3_use_trajectory: bool = True,
    certv3_use_query: bool = True,
    certv3_use_candidate_pool: bool = True,
    # 2.5) Experimental compression params
    compression_variant: str = "flashvid",
    adapter_budget_uses_expansion: bool = False,
    fastvid_DySeg_c: int = 8,
    fastvid_DySeg_tau: float = 0.90,
    fastvid_DySeg_ignore: float = 0.95,
    fastvid_STPrune_d: float = 0.40,
    fastvid_DTM_p: int = 4,
    fastvid_DTM_beta: float = 0.60,
    visionzip_dominant_ratio: float = 65.0 / 70.0,
    prunevid_tau: float = 0.80,
    prunevid_temporal_segment_ratio: float = 0.25,
    prunevid_cluster_ratio: float = 0.50,
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
    strict_token_budget=None,
) -> nn.Module:
    """Apply the selected training-free video-token compression method.

    CertVID V1, V2, V3, the archived ``certvid_v3origin``, and the fixed
    ``certvidfinal2`` method share this entry point with the retained baselines.
    """

    variant = str(compression_variant).strip().lower()

    # Replace with custom methods.
    if type(model) is LlavaQwenForCausalLM:  ## For LLaVA-OneVision or LLaVA-Video
        LlavaMetaForCausalLM.encode_images = LlavaMetaForCausalLM_encode_images
        LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = LlavaMetaForCausalLM_prepare_inputs_labels_for_multimodal
        SigLipAttention.forward = SigLipAttention_forward
        SigLipVisionTower.forward = SigLipVisionTower_forward
        Qwen2Attention.forward = Qwen2Attention_forward
        Qwen2DecoderLayer.forward = Qwen2DecoderLayer_forward
        Qwen2Model.forward = Qwen2Model_forward
        vision_tower = model.get_vision_tower()
        vision_tower._flashvid_variant = variant
        vision_layers = vision_tower.vision_tower.vision_model.encoder.layers
        vision_layers[-1].self_attn.is_last_layer = True
        if variant == "visionzip":
            vision_layers[-1].self_attn.capture_visionzip = True
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
        if variant == "visionzip":
            model.model.visual.blocks[-1].attn.capture_visionzip = True
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
        if variant == "visionzip":
            model.model.visual.blocks[-1].attn.capture_visionzip = True
    else:
        raise NotImplementedError(f"FlashVID is not supported for {type(model)} yet.")

    supported_variants = {
        "flashvid",
        "fastv",
        "fastvid",
        "visionzip",
        "prunevid",
        "certvid",
        "certvid_v2",
        "certvid_v3",
        "certvid_v3origin",
        "certvidfinal2",
    }
    if variant not in supported_variants:
        raise ValueError(
            f"unsupported compression_variant={compression_variant!r}, "
            f"expected one of {sorted(supported_variants)}"
        )

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
        cert_budget_uses_expansion=cert_budget_uses_expansion,
        cert_query_atoms=cert_query_atoms,
        cert_temporal_bins=cert_temporal_bins,
        cert_spatial_bins=cert_spatial_bins,
        cert_candidate_multiplier=cert_candidate_multiplier,
        cert_query_weight=cert_query_weight,
        cert_temporal_weight=cert_temporal_weight,
        cert_detail_weight=cert_detail_weight,
        cert_repair_ratio=cert_repair_ratio,
        cert_fusion_alpha=cert_fusion_alpha,
        cert_assignment_temperature=cert_assignment_temperature,
        cert_track_threshold=cert_track_threshold,
        cert_spatial_penalty=cert_spatial_penalty,
        cert_metric_dim=cert_metric_dim,
        certv2_budget_uses_expansion=certv2_budget_uses_expansion,
        certv2_query_atoms=certv2_query_atoms,
        certv2_temporal_bins=certv2_temporal_bins,
        certv2_spatial_bins=certv2_spatial_bins,
        certv2_candidate_multiplier=certv2_candidate_multiplier,
        certv2_query_weight=certv2_query_weight,
        certv2_frame_floor_ratio=certv2_frame_floor_ratio,
        certv2_diversity_weight=certv2_diversity_weight,
        certv2_coverage_weight=certv2_coverage_weight,
        certv2_density_neighbors=certv2_density_neighbors,
        certv2_track_threshold=certv2_track_threshold,
        certv2_spatial_penalty=certv2_spatial_penalty,
        certv2_metric_dim=certv2_metric_dim,
        certv2_repair_ratio=certv2_repair_ratio,
        certv2_repair_ratio_high=certv2_repair_ratio_high,
        certv2_router_strength=certv2_router_strength,
        certv2_protect_ratio=certv2_protect_ratio,
        certv2_swap_margin=certv2_swap_margin,
        certv2_fusion_alpha=certv2_fusion_alpha,
        certv2_repair_fusion_alpha=certv2_repair_fusion_alpha,
        certv2_assignment_temperature=certv2_assignment_temperature,
        certv3_budget_uses_expansion=certv3_budget_uses_expansion,
        certv3_query_atoms=certv3_query_atoms,
        certv3_temporal_bins=certv3_temporal_bins,
        certv3_spatial_bins=certv3_spatial_bins,
        certv3_candidate_multiplier=certv3_candidate_multiplier,
        certv3_query_weight=certv3_query_weight,
        certv3_visual_attention_weight=certv3_visual_attention_weight,
        certv3_visual_novelty_weight=certv3_visual_novelty_weight,
        certv3_visual_curvature_weight=certv3_visual_curvature_weight,
        certv3_visual_event_weight=certv3_visual_event_weight,
        certv3_visual_detail_weight=certv3_visual_detail_weight,
        certv3_visual_component_weight=certv3_visual_component_weight,
        certv3_event_novelty_weight=certv3_event_novelty_weight,
        certv3_event_curvature_weight=certv3_event_curvature_weight,
        certv3_event_frame_weight=certv3_event_frame_weight,
        certv3_event_detail_weight=certv3_event_detail_weight,
        certv3_event_query_weight=certv3_event_query_weight,
        certv3_track_threshold=certv3_track_threshold,
        certv3_spatial_penalty=certv3_spatial_penalty,
        certv3_metric_dim=certv3_metric_dim,
        certv3_frame_coverage_ratio=certv3_frame_coverage_ratio,
        certv3_cell_coverage_ratio=certv3_cell_coverage_ratio,
        certv3_query_threshold=certv3_query_threshold,
        certv3_query_per_atom=certv3_query_per_atom,
        certv3_structural_weight=certv3_structural_weight,
        certv3_whitening_strength=certv3_whitening_strength,
        certv3_quality_floor=certv3_quality_floor,
        certv3_ridge=certv3_ridge,
        certv3_swap_steps=certv3_swap_steps,
        certv3_swap_pool=certv3_swap_pool,
        certv3_swap_margin=certv3_swap_margin,
        certv3_fusion_alpha=certv3_fusion_alpha,
        certv3_assignment_temperature=certv3_assignment_temperature,
        certv3_certificate_budget_ratio=certv3_certificate_budget_ratio,
        certv3_selection_objective=certv3_selection_objective,
        certv3_use_spatiotemporal_certificates=certv3_use_spatiotemporal_certificates,
        certv3_use_spatiotemporal_design=certv3_use_spatiotemporal_design,
        certv3_use_trajectory=certv3_use_trajectory,
        certv3_use_query=certv3_use_query,
        certv3_use_candidate_pool=certv3_use_candidate_pool,
        adapter_budget_uses_expansion=adapter_budget_uses_expansion,
        fastvid_DySeg_c=fastvid_DySeg_c,
        fastvid_DySeg_tau=fastvid_DySeg_tau,
        fastvid_DySeg_ignore=fastvid_DySeg_ignore,
        fastvid_STPrune_d=fastvid_STPrune_d,
        fastvid_DTM_p=fastvid_DTM_p,
        fastvid_DTM_beta=fastvid_DTM_beta,
        visionzip_dominant_ratio=visionzip_dominant_ratio,
        prunevid_tau=prunevid_tau,
        prunevid_temporal_segment_ratio=prunevid_temporal_segment_ratio,
        prunevid_cluster_ratio=prunevid_cluster_ratio,
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
    # Qwen-family FlashVID runs default to strict accounting. LLaVA FlashVID
    # and CertVID V3 can opt in explicitly without changing established runs.
    strict_capable_variant = variant in (
        "flashvid",
        "certvid_v3",
        "certvid_v3origin",
        "certvidfinal2",
    )
    automatic_strict_budget = bool(
        variant == "flashvid"
        and type(model)
        in (Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration)
    )
    flashvid_config.strict_token_budget = bool(
        strict_capable_variant
        and (
            automatic_strict_budget
            if strict_token_budget is None
            else bool(strict_token_budget)
        )
    )
    if flashvid_config.strict_token_budget:
        num_hidden_layers = _text_layer_count(model)
        if num_hidden_layers <= 0:
            raise ValueError(
                "cannot verify the Qwen-family FlashVID token budget "
                "without decoder depth"
            )
        if not (0 <= int(pruning_layer) <= num_hidden_layers):
            raise ValueError(
                "Qwen-family FlashVID pruning_layer must satisfy "
                f"0 <= K <= L, got K={pruning_layer}, L={num_hidden_layers}"
            )
        average_multiplier = float(expansion) * (
            int(pruning_layer)
            + (num_hidden_layers - int(pruning_layer)) * float(llm_retention_ratio)
        ) / float(num_hidden_layers)
        if average_multiplier > 1.0 + 1e-9:
            raise ValueError(
                "strict hybrid compression exceeds the nominal layer-average token budget: "
                f"multiplier={average_multiplier:.10f} > 1.0 "
                f"(E={expansion}, K={pruning_layer}, "
                f"r={llm_retention_ratio}, L={num_hidden_layers})"
            )
        flashvid_config.last_flashvid_average_budget_multiplier = float(
            average_multiplier
        )
        flashvid_config.strict_num_hidden_layers = int(num_hidden_layers)

    # Store FlashVid Config in the model.
    if type(model) is Qwen2_5_VLForConditionalGeneration:
        setattr(flashvid_config, "_baseline_backbone", "qwen2_5_vl")
    elif type(model) is Qwen3VLForConditionalGeneration:
        setattr(flashvid_config, "_baseline_backbone", "qwen3_vl")
    else:
        setattr(flashvid_config, "_baseline_backbone", "llava")
    setattr(model, "flashvid_config", flashvid_config)
    setattr(model.model, "flashvid_config", flashvid_config)
    setattr(model.config, "flashvid_bypass_active", False)
    if type(model) in (Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration):
        setattr(model.model.language_model, "flashvid_config", flashvid_config)
        setattr(model.model.visual, "flashvid_config", flashvid_config)
    for module in model.modules():
        if isinstance(module, (Qwen2Attention, Qwen2_5_VLAttention, Qwen3VLTextAttention)):
            setattr(module, "flashvid_config", flashvid_config)

    return model
