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
    temporal_merge_mode: str = "tree",
    graph_temporal_topk: int = 3,
    graph_temporal_radius: int = 1,
    graph_temporal_skip: int = 1,
    graph_merge_protect_ratio: float = 0.15,
    graph_merge_target_ratio: float = 0.65,
    graph_merge_representative: str = "medoid",
    graph_final_tokens_per_frame: int = 0,
    graph_final_frame_floor_ratio: float = 0.55,
    graph_skip_spatial_merge_when_capped: bool = True,
    fastgraph_ats_ratio: float = 0.60,
    fastgraph_budget_uses_expansion: bool = True,
    fastgraph_temporal_radius: int = 1,
    fastgraph_temporal_skip: int = 1,
    fastgraph_temporal_topk: int = 2,
    fastgraph_edge_threshold: float = 0.0,
    fastgraph_protect_ratio: float = 0.15,
    fastgraph_attn_weight: float = 0.55,
    fastgraph_novelty_weight: float = 0.30,
    fastgraph_density_weight: float = 0.15,
    apex_evidence_ratio: float = 0.45,
    apex_event_ratio: float = 0.30,
    apex_memory_ratio: float = 0.25,
    apex_router_strength: float = 0.50,
    apex_summary_temperature: float = 0.07,
    apex_frame_floor_ratio: float = 0.35,
    apex_question_weight: float = 0.20,
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
    qcert_budget_uses_expansion: bool = True,
    qcert_query_atoms: int = 8,
    qcert_temporal_bins: int = 0,
    qcert_spatial_bins: int = 0,
    qcert_candidate_multiplier: float = 2.5,
    qcert_track_threshold: float = 0.82,
    qcert_track_spatial_penalty: float = 0.08,
    qcert_frame_coverage_ratio: float = 1.0,
    qcert_cell_coverage_ratio: float = 0.35,
    qcert_query_threshold: float = 0.10,
    qcert_query_per_atom: int = 1,
    qcert_quality_query_weight: float = 0.12,
    qcert_whitening_strength: float = 0.25,
    qcert_semantic_weight: float = 0.68,
    qcert_phase_weight: float = 0.14,
    qcert_temporal_weight: float = 0.06,
    qcert_spatial_weight: float = 0.04,
    qcert_signal_weight: float = 0.04,
    qcert_design_query_weight: float = 0.04,
    qcert_phase_levels: int = 4,
    qcert_quality_floor: float = 0.15,
    qcert_ridge: float = 0.50,
    qcert_kernel_tolerance: float = 1e-4,
    qcert_max_kernel_pivots: int = 1024,
    qcert_fusion_alpha: float = 0.08,
    qcert_fusion_similarity: float = 0.82,
    qcert_fusion_temporal_radius: float = 1.0,
    qcert_fusion_spatial_radius: float = 2.0,
    qcert_assignment_temperature: float = 0.07,
    v3plus_inner_mode: str = "structured",
    v3plus_query_rows: int = 32,
    v3plus_attention_mean_weight: float = 0.75,
    v3plus_frame_floor: int = 1,
    v3plus_frame_cap_multiplier: float = 2.0,
    v3plus_pair_budget_ratio: float = 0.10,
    v3plus_attention_weight: float = 0.70,
    v3plus_outer_demand_weight: float = 0.20,
    v3plus_certificate_weight: float = 0.10,
    v3plus_diversity_weight: float = 0.15,
    v3plus_spatial_bonus: float = 0.05,
    v3plusplus_inner_mode: str = "gradient_nms",
    v3plusplus_proxy_positions: int = 4,
    v3plusplus_nms_enabled: bool = True,
    v3plusplus_nms_threshold: float = 0.80,
    v3plusplus_strict: bool = True,
    certv6_scene_temporal: bool = True,
    certv6_gate_enabled: bool = True,
    certv6_continuity_low: float = 0.55,
    certv6_continuity_high: float = 0.80,
    certv6_query_per_atom_max: int = 3,
    certv7_min_duration_seconds: float = 120.0,
    certv7_transport_spatial_bins: int = 4,
    certv7_transport_epsilon: float = 0.08,
    certv7_transport_steps: int = 8,
    certv7_transport_spatial_weight: float = 0.20,
    certv7_frame_floor_ratio: float = 1.0,
    certv7_frame_cap_ratio: float = 1.0,
    certv7_budget_temperature: float = 0.50,
    certv7_uniqueness_weight: float = 0.25,
    certv7_transport_weight: float = 0.35,
    certv7_event_weight: float = 0.20,
    certv7_query_weight: float = 0.20,
    certv7_budget_rounding: str = "per_frame_ceil",
    certv7_v3_certificate_ratio: float = 0.05,
    certv7_relay_ratio: float = 0.25,
    certv7_relay_query_share: float = 0.25,
    certv7_transition_relay_share: float = 0.45,
    certv7_query_peaks_per_atom: int = 2,
    certv7_query_min_frame_gap: int = 3,
    certv7_query_peak_threshold: float = 0.70,
    certv7_query_context_radius: int = 1,
    certv7_transition_pairs_per_boundary: int = 2,
    certv7_transition_min_similarity: float = 0.30,
    certv7_trajectory_min_span: int = 3,
    certv7_trajectory_points: int = 3,
    certv7_facility_quality_mix: float = 0.18,
    certv7_min_reallocation_ratio: float = 0.02,
    certv7_d_efficiency_floor: float = 0.80,
    certv7_assignment_topk: int = 2,
    certv7_assignment_temperature: float = 0.07,
    certv7_cross_frame_cost_quantile: float = 0.45,
    certv7_cross_frame_similarity: float = 0.82,
    certv7_cross_frame_max_seconds: float = 12.0,
    certv7_component_bonus: float = 0.08,
    certv7_design_protect_ratio: float = 0.15,
    certv7_long_fusion_alpha: float = 0.04,
    certv7_debug: bool = False,
    certv8_enabled: bool = True,
    certv8_intent_router: bool = True,
    certv8_intent_strength: float = 0.75,
    certv8_min_horizon_gap_seconds: float = 4.0,
    certv8_min_deficit: float = 0.04,
    certv8_frame_floor_ratio: float = 0.45,
    certv8_frame_cap_ratio: float = 2.00,
    certv8_max_swap_ratio: float = 0.30,
    certv8_concentration_preserve_ratio: float = 0.55,
    certv8_query_peak_count: int = 2,
    certv8_query_peak_separation: int = 2,
    certv8_query_weight: float = 0.30,
    certv8_event_weight: float = 0.25,
    certv8_balance_weight: float = 0.30,
    certv8_design_protect_ratio: float = 0.08,
    certv8_query_protect_ratio: float = 0.05,
    certv8_d_efficiency_floor: float = 0.95,
    certv8_min_objective_gain: float = 0.001,
    certv8_cross_frame_similarity: float = 0.88,
    certv8_cross_frame_max_seconds: float = 8.0,
    certv8_localized_event_boost: float = 0.0,
    certv8_attribute_query_boost: float = 0.0,
    certv8_stratified_enabled: bool = True,
    certv8_stratified_temporal_strength: float = 0.60,
    certv8_stratified_retrieval_strength: float = 0.40,
    certv8_stratified_generic_strength: float = 0.0,
    certv8_stratified_min_question_words: int = 12,
    certv8_stratified_v3_keep_ratio: float = 0.50,
    certv8_stratified_max_duration_seconds: float = 1200.0,
    certv8_stratified_d_efficiency_floor: float = 0.82,
    certv8_stratified_query_tolerance: float = 0.01,
    certv8_debug: bool = False,
    certv9_enabled: bool = True,
    certv9_merge_threshold: float = 0.80,
    certv9_uncovered_mass_threshold: float = 0.05,
    certv9_max_swap_ratio: float = 0.15,
    certv9_d_efficiency_floor: float = 0.98,
    certv9_min_objective_gain: float = 1e-4,
    certv9_state_distance_threshold: float = 0.15,
    certv9_state_min_bin_span: int = 2,
    certv9_query_max_peaks: int = 3,
    certv9_query_peak_separation: int = 2,
    certv9_event_quantile: float = 0.85,
    certv9_event_floor: float = 0.10,
    certv9_cross_segment_similarity: float = 0.92,
    certv9_cross_segment_max_seconds: float = 8.0,
    certv9_full_pool_repair_enabled: bool = True,
    certv9_merge_rejection_enabled: bool = True,
    certv9_event_mask_enabled: bool = True,
    certv9_state_pair_enabled: bool = True,
    certv9_multi_peak_enabled: bool = True,
    certv9_repair_pool: int = 128,
    certv9_remove_pool: int = 64,
    certv9_debug: bool = False,
    certv10_enabled: bool = True,
    certv10_track_similarity: float = 0.72,
    certv10_spatial_penalty: float = 0.03,
    certv10_track_min_span: int = 2,
    certv10_reliability_floor: float = 0.025,
    certv10_reliability_target: float = 0.18,
    certv10_min_swap_ratio: float = 0.08,
    certv10_max_swap_ratio: float = 0.25,
    certv10_v3_protect_ratio: float = 0.10,
    certv10_frame_floor_ratio: float = 0.55,
    certv10_frame_cap_ratio: float = 1.80,
    certv10_budget_temperature: float = 0.55,
    certv10_allocation_strength: float = 0.75,
    certv10_motion_peak_frames: int = 8,
    certv10_candidate_pool: int = 384,
    certv10_min_swap_gain: float = -0.04,
    certv10_d_soft_weight: float = 0.10,
    certv10_track_assignment_radius: int = 2,
    certv10_cross_frame_max_seconds: float = 12.0,
    certv10_assignment_topk: int = 2,
    certv10_assignment_temperature: float = 0.07,
    certv10_merge_threshold: float = 0.76,
    certv10_trajectory_fusion_scale: float = 0.25,
    certv10_debug: bool = False,
    certv11_enabled: bool = True,
    certv11_match_similarity: float = 0.72,
    certv11_match_margin: float = 0.015,
    certv11_cycle_radius: int = 1,
    certv11_max_spatial_jump: float = 0.60,
    certv11_scene_similarity: float = 0.50,
    certv11_spatial_match_weight: float = 0.04,
    certv11_time_confidence_seconds: float = 30.0,
    certv11_transition_dim: int = 32,
    certv11_state_scale: float = 0.20,
    certv11_transition_weight_min: float = 0.08,
    certv11_transition_weight_max: float = 0.24,
    certv11_reliability_floor: float = 0.015,
    certv11_reliability_target: float = 0.12,
    certv11_deficit_threshold: float = 0.025,
    certv11_deficit_scale: float = 0.20,
    certv11_min_swap_ratio: float = 0.04,
    certv11_max_swap_ratio: float = 0.12,
    certv11_add_pool: int = 160,
    certv11_remove_pool: int = 64,
    certv11_v3_protect_ratio: float = 0.10,
    certv11_node_efficiency_floor: float = 0.95,
    certv11_node_loss_weight: float = 0.35,
    certv11_edge_coverage_weight: float = 0.30,
    certv11_frame_balance_weight: float = 0.08,
    certv11_swap_margin: float = 0.0,
    certv11_cross_frame_similarity: float = 0.92,
    certv11_cross_frame_max_seconds: float = 12.0,
    certv11_assignment_topk: int = 2,
    certv11_assignment_temperature: float = 0.07,
    certv11_fusion_similarity_floor: float = 0.70,
    certv11_debug: bool = False,
    certv4_budget_mode: str = "layer_average",
    certv4_attention_policy: str = "validated",
    certv4_attention_eps: float = 1e-6,
    certv4_certificate_budget_ratio: float = 0.40,
    certv4_query_mode: str = "certificates_and_design",
    certv4_design_protect_ratio: float = 0.15,
    certv4_query_atoms: int = 8,
    certv4_temporal_bins: int = 12,
    certv4_spatial_bins: int = 3,
    certv4_candidate_multiplier: float = 2.5,
    certv4_track_threshold: float = 0.82,
    certv4_spatial_penalty: float = 0.08,
    certv4_metric_dim: int = 96,
    certv4_frame_coverage_ratio: float = 1.0,
    certv4_cell_coverage_ratio: float = 0.50,
    certv4_query_threshold: float = 0.10,
    certv4_query_per_atom: int = 1,
    certv4_structural_weight: float = 0.32,
    certv4_whitening_strength: float = 0.50,
    certv4_quality_floor: float = 0.15,
    certv4_ridge: float = 0.50,
    certv4_swap_steps: int = 6,
    certv4_swap_pool: int = 24,
    certv4_swap_margin: float = 1e-4,
    certv4_fusion_alpha: float = 0.12,
    certv4_assignment_temperature: float = 0.07,
    certv4_debug: bool = False,
    certv5_budget_mode: str = "layer_average",
    certv5_ot_enabled: bool = True,
    certv5_ot_topk: int = 4,
    certv5_ot_temperature: float = 0.07,
    certv5_ot_steps: int = 6,
    certv5_ot_capacity_tau: float = 0.10,
    certv5_ot_prior_shrink: float = 0.10,
    certv5_ot_live_fraction: float = 0.25,
    certv5_ot_cost_slack: float = 0.05,
    certv5_ot_temporal_penalty: float = 0.04,
    certv5_ot_max_displacement: float = 0.12,
    certv5_ot_min_cosine: float = 0.98,
    certv5_debug: bool = False,
    certe_budget_uses_expansion: bool = True,
    certe_ridge: float = 0.50,
    certe_bottom_k: int = 8,
    certe_swap_steps: int = 6,
    certe_remove_pool: int = 8,
    certe_add_pool: int = 16,
    certe_verify_pool: int = 4,
    certe_swap_margin: float = 1e-5,
    certe_spectral_temperature: float = 0.05,
    certe_d_efficiency_floor: float = 0.995,
    certe_rank_tolerance: float = 1e-5,
    certe_debug: bool = False,
    faith_budget_uses_expansion: bool = True,
    faith_mass_strength: float = 1.0,
    faith_variance_strength: float = 0.50,
    faith_merge_alpha: float = 1.0,
    faith_temporal_radius: int = 1,
    faith_spatial_radius: float = 0.75,
    faith_component_bonus: float = 0.08,
    faith_temporal_penalty: float = 0.04,
    faith_spatial_penalty: float = 0.04,
    faith_assignment_topk: int = 2,
    faith_assignment_temperature: float = 0.07,
    faith_max_log_bias: float = 20.0,
    faith_attention_strict: bool = True,
    faith_debug: bool = False,
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
    # TALON params.
    talon_transport_radius: int = 1,
    talon_rank_ratio: float = 0.40,
    talon_rank_min: int = 2,
    talon_rank_max: int = 32,
    talon_budget_scale: float = 0.60,
    talon_target_tokens_per_frame: int = 0,
    talon_short_target_tokens_per_frame: int = 0,
    talon_medium_target_tokens_per_frame: int = 0,
    talon_long_target_tokens_per_frame: int = 0,
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
    talon_anchor_diversity_weight: float = 0.0,
    talon_anchor_candidate_multiplier: float = 4.0,
    talon_spatial_anchor_coverage: bool = False,
    talon_spatial_anchor_ratio: float = 0.35,
    talon_spatial_anchor_rows: int = 3,
    talon_spatial_anchor_cols: int = 3,
    talon_spatial_anchor_score: str = "fused",
    talon_spatial_anchor_apply_to_short: bool = False,
    talon_frame_coverage_floor_ratio: float = 0.65,
    talon_frame_importance_pooling: str = "mean",
    talon_frame_importance_topk: int = 6,
    talon_medium_frame_coverage_floor_ratio: float = -1.0,
    talon_long_frame_coverage_floor_ratio: float = -1.0,
    talon_frame_local_budget_ratio: float = 1.0,
    talon_question_recall_ratio: float = 0.06,
    talon_question_recall_qweight: float = 0.65,
    talon_persistence_recall_ratio: float = 0.0,
    talon_persistence_recall_qweight: float = 0.50,
    talon_persistence_recall_pweight: float = 0.35,
    talon_persistence_apply_to_short: bool = False,
    talon_persistence_apply_to_medium: bool = True,
    talon_persistence_apply_to_long: bool = False,
    talon_object_evidence_ratio: float = 0.0,
    talon_object_evidence_qweight: float = 0.35,
    talon_object_evidence_sweight: float = 0.45,
    talon_object_evidence_pweight: float = 0.10,
    talon_object_evidence_apply_to_short: bool = False,
    talon_object_evidence_apply_to_medium: bool = True,
    talon_object_evidence_apply_to_long: bool = False,
    talon_question_pooling: str = "mean",
    talon_question_pooling_topk: int = 4,
    talon_question_contrast_weight: float = 0.0,
    talon_question_contrast_apply_to_short: bool = False,
    talon_monotonic_base_tokens_per_frame: int = 20,
    talon_budget_strategy: str = "marginal",
    talon_budget_mode: str = "uniform",
    talon_transport_mode: str = "hard",
    talon_transport_temperature: float = 0.07,
    talon_lite_enabled: bool = False,
    talon_echo_temperature: float = 0.07,
    talon_echo_topk_neighbors: int = 4,
    talon_echo_residual_weight: float = 0.0,
    talon_echo_score_mode: str = "mse",
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
    talon_adaptive_target_enabled: bool = False,
    talon_force_fixed_target: bool = False,
    talon_target_mean_cap: float = 0.0,
    talon_unified_selection: bool = False,
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
    talon_event_keep_bonus: float = 0.04,
    talon_legacy_base_keep_ratio: float = 0.85,
    talon_prior_candidate_ratio: float = 0.12,
    talon_prior_keep_bonus: float = 0.06,
    talon_flash_prior_channel_ratio: float = 0.12,
    talon_flash_prior_channel_method: str = "attn_div_v2",
    talon_flash_prior_channel_min_per_frame: int = 1,
    talon_flash_prior_channel_max_per_frame: int = 4,
    talon_flash_prior_channel_bonus: float = 0.06,
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
    talon_duration_aware: bool = False,
    talon_medium_anchor_safety_ratio: float = 0.72,
    talon_medium_event_budget_ratio: float = 0.30,
    talon_medium_global_topk_ratio: float = 0.70,
    talon_long_anchor_safety_ratio: float = 0.80,
    talon_long_event_budget_ratio: float = 0.14,
    talon_long_global_topk_ratio: float = 0.85,
    talon_task_aware_event: bool = False,
    talon_task_event_attention_weight: float = 0.82,
    talon_task_event_qweight: float = 0.30,
    talon_visual_task_balance: bool = False,
    talon_visual_task_anchor_ratio: float = 0.84,
    talon_visual_task_event_ratio: float = 0.12,
    talon_visual_task_recall_ratio: float = 0.02,
    talon_knowledge_visual_anchor_ratio: float = 0.78,
    talon_knowledge_visual_event_ratio: float = 0.18,
    talon_knowledge_visual_recall_ratio: float = 0.06,
    talon_adaptive_router: bool = False,
    talon_router_apply_to_short: bool = False,
    talon_router_visual_anchor_ratio: float = 0.76,
    talon_router_visual_event_ratio: float = 0.24,
    talon_router_visual_recall_ratio: float = 0.06,
    talon_router_temporal_anchor_ratio: float = 0.66,
    talon_router_temporal_event_ratio: float = 0.34,
    talon_router_temporal_recall_ratio: float = 0.08,
    talon_router_balanced_anchor_ratio: float = 0.72,
    talon_router_balanced_event_ratio: float = 0.30,
    talon_router_balanced_recall_ratio: float = 0.08,
    talon_router_visual_concentration_threshold: float = 0.28,
    talon_router_low_residual_threshold: float = 0.30,
    talon_router_temporal_entropy_threshold: float = 0.95,
    talon_router_temporal_residual_threshold: float = 0.36,
    talon_temporal_chunk_aware: bool = False,
    talon_temporal_num_chunks: int = 4,
    talon_temporal_chunk_min_ratio: float = 0.18,
    talon_temporal_chunk_score: str = "combined",
    talon_track_aware: bool = False,
    talon_track_budget_ratio: float = 0.12,
    talon_track_tokens_per_slot: int = 1,
    talon_track_score: str = "combined",
    talon_absorb_dropped_tokens: bool = False,
    talon_absorb_ratio: float = 0.35,
    talon_absorb_alpha: float = 0.25,
    talon_absorb_score: str = "combined",
    talon_summary_replacement: bool = False,
    talon_summary_raw_swap: bool = False,
    talon_summary_ratio: float = 0.08,
    talon_summary_num_chunks: int = 8,
    talon_summary_pool_topk: int = 12,
    talon_summary_alpha: float = 0.55,
    talon_summary_score: str = "combined",
    # 3) Inner-LLM Compression params
    expansion: float = 1.25,
    pruning_layer: int = 20,
    llm_retention_ratio: float = 0.3,
    # 4) Decode-stage policy scaffold (Route3, default no-op).
    decode_policy: str = "none",
    decode_kv_budget_ratio: float = 1.0,
    decode_update_interval: int = 4,
    decode_start_layer: int = 0,
    # PrismVID parameters are appended to preserve positional compatibility.
    prism_budget_uses_expansion: bool = True,
    prism_metric_dim: int = 256,
    prism_query_atoms: int = 6,
    prism_candidate_multiplier: float = 2.25,
    prism_probe_tokens: int = 512,
    prism_frame_floor_ratio: float = 0.20,
    prism_attention_weight: float = 0.30,
    prism_event_weight: float = 0.24,
    prism_query_weight: float = 0.16,
    prism_disagreement_weight: float = 0.16,
    prism_router_strength: float = 0.50,
    prism_coverage_weight: float = 0.68,
    prism_pareto_weight: float = 0.20,
    prism_batch_size: int = 8,
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
            "fastgraphvid" keeps an ATS branch plus GraphSTM residual medoids;
            "apexvid" enables adaptive evidence/event/memory compression;
            "certvid" enables constrained evidence coreset compression with shared Qwen3 DeepStack fusion;
            "certvid_v2" keeps a CertVID evidence backbone and applies gated trajectory repair;
            "certvid_v3" selects a certified regularized D-optimal evidence design;
            "certvid_qwen" uses full-feature kernel D-optimal design with
            M-RoPE-aware fusion for Qwen2.5-VL;
            "certvid_v3plus" keeps the V3 outer design and replaces only the
            LLaVA inner selector with structure-aware pruning;
            "certvid_v3plusplus" keeps the V3 outer design and uses
            inference-objective gradient saliency with feature-space NMS;
            "certvid_v6" adds continuity-gated scene structure to the V3 evidence design;
            "certvid_v7" preserves long-horizon relations with transition and trajectory evidence;
            "certvid_v8" preserves V3 anchors and repairs temporal/query evidence deficits;
            "certvid_v9" repairs missing states and rejects untrustworthy residual fusion;
            "certvid_v10" aggressively reallocates V3 evidence toward reliable motion trajectories;
            "certvid_v11" preserves the V3 evidence coreset and repairs missing
            transition endpoints using spatially validated correspondences;
            "certvid_v5" preserves V3 anchors and recovers discarded residual evidence with OT;
            "certvid_e" refines the V3 design against its weakest information direction;
            "faithvid" preserves merged-token attention mass and constrains RoPE phase dispersion;
            "prismvid" selects an exact multi-level Qwen3 DeepStack visual coreset;
            "talon" enables transport-aligned low-rank + sparse innovation compression.
        question_aware_reweighting (bool, optional): Enable question-guided token reweighting.
        question_reweight_beta (float, optional): Strength of question-aware reweighting.
        talon_transport_radius (int, optional): Local transport radius for frame-to-frame token alignment.
        talon_rank_ratio (float, optional): Per-frame low-rank share in TALON token budget.
        talon_rank_min (int, optional): Minimum TALON low-rank token count per frame when budget allows.
        talon_rank_max (int, optional): Maximum TALON low-rank token count per frame.
        talon_budget_scale (float, optional): TALON-only multiplier over the shared visual budget.
        talon_target_tokens_per_frame (int, optional): Fixed TALON target width per frame; 0 disables it.
        talon_short/medium/long_target_tokens_per_frame (int, optional): Duration-specific
            TALON targets; 0 falls back to talon_target_tokens_per_frame.
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

    variant = str(compression_variant).strip().lower()
    if variant in ("certvid_v3plus", "certvid_v3plusplus") and type(model) is not LlavaQwenForCausalLM:
        raise ValueError(f"{variant} currently supports LLaVA-OneVision only")
    if variant == "certvid_qwen" and type(model) is not Qwen2_5_VLForConditionalGeneration:
        raise ValueError("certvid_qwen currently supports Qwen2.5-VL only")

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

    if variant not in ("flashvid", "fastv", "fastvid", "visionzip", "prunevid", "talon", "graphvid", "fastgraphvid", "apexvid", "certvid", "certvid_v2", "certvid_v3", "certvid_qwen", "certvid_v3plus", "certvid_v3plusplus", "certvid_v6", "certvid_v7", "certvid_v8", "certvid_v9", "certvid_v10", "certvid_v11", "certvid_v4", "certvid_v5", "certvid_e", "faithvid", "prismvid"):
        raise ValueError(
            f"unsupported compression_variant={compression_variant!r}, "
            "expected flashvid|fastv|fastvid|visionzip|prunevid|talon|graphvid|fastgraphvid|apexvid|certvid|certvid_v2|certvid_v3|certvid_qwen|certvid_v3plus|certvid_v3plusplus|certvid_v6|certvid_v7|certvid_v8|certvid_v9|certvid_v10|certvid_v11|certvid_v4|certvid_v5|certvid_e|faithvid|prismvid"
        )
    if variant == "certvid_v3plus":
        v3plus_inner_mode = str(v3plus_inner_mode).strip().lower()
        if v3plus_inner_mode not in ("structured", "legacy"):
            raise ValueError(
                f"v3plus_inner_mode must be structured or legacy, got {v3plus_inner_mode!r}"
            )
    if variant == "certvid_v3plusplus":
        v3plusplus_inner_mode = str(v3plusplus_inner_mode).strip().lower()
        if v3plusplus_inner_mode not in ("gradient_nms", "legacy"):
            raise ValueError(
                "v3plusplus_inner_mode must be gradient_nms or legacy, "
                f"got {v3plusplus_inner_mode!r}"
            )
        if int(v3plusplus_proxy_positions) <= 0:
            raise ValueError("v3plusplus_proxy_positions must be positive")
        if not (-1.0 <= float(v3plusplus_nms_threshold) <= 1.0):
            raise ValueError("v3plusplus_nms_threshold must be in [-1, 1]")
    if variant == "graphvid":
        temporal_merge_mode = "graph"

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
        graph_final_tokens_per_frame=graph_final_tokens_per_frame,
        graph_final_frame_floor_ratio=graph_final_frame_floor_ratio,
        graph_skip_spatial_merge_when_capped=graph_skip_spatial_merge_when_capped,
        fastgraph_ats_ratio=fastgraph_ats_ratio,
        fastgraph_budget_uses_expansion=fastgraph_budget_uses_expansion,
        fastgraph_temporal_radius=fastgraph_temporal_radius,
        fastgraph_temporal_skip=fastgraph_temporal_skip,
        fastgraph_temporal_topk=fastgraph_temporal_topk,
        fastgraph_edge_threshold=fastgraph_edge_threshold,
        fastgraph_protect_ratio=fastgraph_protect_ratio,
        fastgraph_attn_weight=fastgraph_attn_weight,
        fastgraph_novelty_weight=fastgraph_novelty_weight,
        fastgraph_density_weight=fastgraph_density_weight,
        apex_evidence_ratio=apex_evidence_ratio,
        apex_event_ratio=apex_event_ratio,
        apex_memory_ratio=apex_memory_ratio,
        apex_router_strength=apex_router_strength,
        apex_summary_temperature=apex_summary_temperature,
        apex_frame_floor_ratio=apex_frame_floor_ratio,
        apex_question_weight=apex_question_weight,
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
        qcert_budget_uses_expansion=qcert_budget_uses_expansion,
        qcert_query_atoms=qcert_query_atoms,
        qcert_temporal_bins=qcert_temporal_bins,
        qcert_spatial_bins=qcert_spatial_bins,
        qcert_candidate_multiplier=qcert_candidate_multiplier,
        qcert_track_threshold=qcert_track_threshold,
        qcert_track_spatial_penalty=qcert_track_spatial_penalty,
        qcert_frame_coverage_ratio=qcert_frame_coverage_ratio,
        qcert_cell_coverage_ratio=qcert_cell_coverage_ratio,
        qcert_query_threshold=qcert_query_threshold,
        qcert_query_per_atom=qcert_query_per_atom,
        qcert_quality_query_weight=qcert_quality_query_weight,
        qcert_whitening_strength=qcert_whitening_strength,
        qcert_semantic_weight=qcert_semantic_weight,
        qcert_phase_weight=qcert_phase_weight,
        qcert_temporal_weight=qcert_temporal_weight,
        qcert_spatial_weight=qcert_spatial_weight,
        qcert_signal_weight=qcert_signal_weight,
        qcert_design_query_weight=qcert_design_query_weight,
        qcert_phase_levels=qcert_phase_levels,
        qcert_quality_floor=qcert_quality_floor,
        qcert_ridge=qcert_ridge,
        qcert_kernel_tolerance=qcert_kernel_tolerance,
        qcert_max_kernel_pivots=qcert_max_kernel_pivots,
        qcert_fusion_alpha=qcert_fusion_alpha,
        qcert_fusion_similarity=qcert_fusion_similarity,
        qcert_fusion_temporal_radius=qcert_fusion_temporal_radius,
        qcert_fusion_spatial_radius=qcert_fusion_spatial_radius,
        qcert_assignment_temperature=qcert_assignment_temperature,
        v3plus_inner_mode=v3plus_inner_mode,
        v3plus_query_rows=v3plus_query_rows,
        v3plus_attention_mean_weight=v3plus_attention_mean_weight,
        v3plus_frame_floor=v3plus_frame_floor,
        v3plus_frame_cap_multiplier=v3plus_frame_cap_multiplier,
        v3plus_pair_budget_ratio=v3plus_pair_budget_ratio,
        v3plus_attention_weight=v3plus_attention_weight,
        v3plus_outer_demand_weight=v3plus_outer_demand_weight,
        v3plus_certificate_weight=v3plus_certificate_weight,
        v3plus_diversity_weight=v3plus_diversity_weight,
        v3plus_spatial_bonus=v3plus_spatial_bonus,
        v3plusplus_inner_mode=v3plusplus_inner_mode,
        v3plusplus_proxy_positions=v3plusplus_proxy_positions,
        v3plusplus_nms_enabled=v3plusplus_nms_enabled,
        v3plusplus_nms_threshold=v3plusplus_nms_threshold,
        v3plusplus_strict=v3plusplus_strict,
        certv6_scene_temporal=certv6_scene_temporal,
        certv6_gate_enabled=certv6_gate_enabled,
        certv6_continuity_low=certv6_continuity_low,
        certv6_continuity_high=certv6_continuity_high,
        certv6_query_per_atom_max=certv6_query_per_atom_max,
        certv7_min_duration_seconds=certv7_min_duration_seconds,
        certv7_transport_spatial_bins=certv7_transport_spatial_bins,
        certv7_transport_epsilon=certv7_transport_epsilon,
        certv7_transport_steps=certv7_transport_steps,
        certv7_transport_spatial_weight=certv7_transport_spatial_weight,
        certv7_frame_floor_ratio=certv7_frame_floor_ratio,
        certv7_frame_cap_ratio=certv7_frame_cap_ratio,
        certv7_budget_temperature=certv7_budget_temperature,
        certv7_uniqueness_weight=certv7_uniqueness_weight,
        certv7_transport_weight=certv7_transport_weight,
        certv7_event_weight=certv7_event_weight,
        certv7_query_weight=certv7_query_weight,
        certv7_budget_rounding=certv7_budget_rounding,
        certv7_v3_certificate_ratio=certv7_v3_certificate_ratio,
        certv7_relay_ratio=certv7_relay_ratio,
        certv7_relay_query_share=certv7_relay_query_share,
        certv7_transition_relay_share=certv7_transition_relay_share,
        certv7_query_peaks_per_atom=certv7_query_peaks_per_atom,
        certv7_query_min_frame_gap=certv7_query_min_frame_gap,
        certv7_query_peak_threshold=certv7_query_peak_threshold,
        certv7_query_context_radius=certv7_query_context_radius,
        certv7_transition_pairs_per_boundary=certv7_transition_pairs_per_boundary,
        certv7_transition_min_similarity=certv7_transition_min_similarity,
        certv7_trajectory_min_span=certv7_trajectory_min_span,
        certv7_trajectory_points=certv7_trajectory_points,
        certv7_facility_quality_mix=certv7_facility_quality_mix,
        certv7_min_reallocation_ratio=certv7_min_reallocation_ratio,
        certv7_d_efficiency_floor=certv7_d_efficiency_floor,
        certv7_assignment_topk=certv7_assignment_topk,
        certv7_assignment_temperature=certv7_assignment_temperature,
        certv7_cross_frame_cost_quantile=certv7_cross_frame_cost_quantile,
        certv7_cross_frame_similarity=certv7_cross_frame_similarity,
        certv7_cross_frame_max_seconds=certv7_cross_frame_max_seconds,
        certv7_component_bonus=certv7_component_bonus,
        certv7_design_protect_ratio=certv7_design_protect_ratio,
        certv7_long_fusion_alpha=certv7_long_fusion_alpha,
        certv7_debug=certv7_debug,
        certv8_enabled=certv8_enabled,
        certv8_intent_router=certv8_intent_router,
        certv8_intent_strength=certv8_intent_strength,
        certv8_min_horizon_gap_seconds=certv8_min_horizon_gap_seconds,
        certv8_min_deficit=certv8_min_deficit,
        certv8_frame_floor_ratio=certv8_frame_floor_ratio,
        certv8_frame_cap_ratio=certv8_frame_cap_ratio,
        certv8_max_swap_ratio=certv8_max_swap_ratio,
        certv8_concentration_preserve_ratio=certv8_concentration_preserve_ratio,
        certv8_query_peak_count=certv8_query_peak_count,
        certv8_query_peak_separation=certv8_query_peak_separation,
        certv8_query_weight=certv8_query_weight,
        certv8_event_weight=certv8_event_weight,
        certv8_balance_weight=certv8_balance_weight,
        certv8_design_protect_ratio=certv8_design_protect_ratio,
        certv8_query_protect_ratio=certv8_query_protect_ratio,
        certv8_d_efficiency_floor=certv8_d_efficiency_floor,
        certv8_min_objective_gain=certv8_min_objective_gain,
        certv8_cross_frame_similarity=certv8_cross_frame_similarity,
        certv8_cross_frame_max_seconds=certv8_cross_frame_max_seconds,
        certv8_localized_event_boost=certv8_localized_event_boost,
        certv8_attribute_query_boost=certv8_attribute_query_boost,
        certv8_stratified_enabled=certv8_stratified_enabled,
        certv8_stratified_temporal_strength=certv8_stratified_temporal_strength,
        certv8_stratified_retrieval_strength=certv8_stratified_retrieval_strength,
        certv8_stratified_generic_strength=certv8_stratified_generic_strength,
        certv8_stratified_min_question_words=certv8_stratified_min_question_words,
        certv8_stratified_v3_keep_ratio=certv8_stratified_v3_keep_ratio,
        certv8_stratified_max_duration_seconds=certv8_stratified_max_duration_seconds,
        certv8_stratified_d_efficiency_floor=certv8_stratified_d_efficiency_floor,
        certv8_stratified_query_tolerance=certv8_stratified_query_tolerance,
        certv8_debug=certv8_debug,
        certv9_enabled=certv9_enabled,
        certv9_merge_threshold=certv9_merge_threshold,
        certv9_uncovered_mass_threshold=certv9_uncovered_mass_threshold,
        certv9_max_swap_ratio=certv9_max_swap_ratio,
        certv9_d_efficiency_floor=certv9_d_efficiency_floor,
        certv9_min_objective_gain=certv9_min_objective_gain,
        certv9_state_distance_threshold=certv9_state_distance_threshold,
        certv9_state_min_bin_span=certv9_state_min_bin_span,
        certv9_query_max_peaks=certv9_query_max_peaks,
        certv9_query_peak_separation=certv9_query_peak_separation,
        certv9_event_quantile=certv9_event_quantile,
        certv9_event_floor=certv9_event_floor,
        certv9_cross_segment_similarity=certv9_cross_segment_similarity,
        certv9_cross_segment_max_seconds=certv9_cross_segment_max_seconds,
        certv9_full_pool_repair_enabled=certv9_full_pool_repair_enabled,
        certv9_merge_rejection_enabled=certv9_merge_rejection_enabled,
        certv9_event_mask_enabled=certv9_event_mask_enabled,
        certv9_state_pair_enabled=certv9_state_pair_enabled,
        certv9_multi_peak_enabled=certv9_multi_peak_enabled,
        certv9_repair_pool=certv9_repair_pool,
        certv9_remove_pool=certv9_remove_pool,
        certv9_debug=certv9_debug,
        certv10_enabled=certv10_enabled,
        certv10_track_similarity=certv10_track_similarity,
        certv10_spatial_penalty=certv10_spatial_penalty,
        certv10_track_min_span=certv10_track_min_span,
        certv10_reliability_floor=certv10_reliability_floor,
        certv10_reliability_target=certv10_reliability_target,
        certv10_min_swap_ratio=certv10_min_swap_ratio,
        certv10_max_swap_ratio=certv10_max_swap_ratio,
        certv10_v3_protect_ratio=certv10_v3_protect_ratio,
        certv10_frame_floor_ratio=certv10_frame_floor_ratio,
        certv10_frame_cap_ratio=certv10_frame_cap_ratio,
        certv10_budget_temperature=certv10_budget_temperature,
        certv10_allocation_strength=certv10_allocation_strength,
        certv10_motion_peak_frames=certv10_motion_peak_frames,
        certv10_candidate_pool=certv10_candidate_pool,
        certv10_min_swap_gain=certv10_min_swap_gain,
        certv10_d_soft_weight=certv10_d_soft_weight,
        certv10_track_assignment_radius=certv10_track_assignment_radius,
        certv10_cross_frame_max_seconds=certv10_cross_frame_max_seconds,
        certv10_assignment_topk=certv10_assignment_topk,
        certv10_assignment_temperature=certv10_assignment_temperature,
        certv10_merge_threshold=certv10_merge_threshold,
        certv10_trajectory_fusion_scale=certv10_trajectory_fusion_scale,
        certv10_debug=certv10_debug,
        certv11_enabled=certv11_enabled,
        certv11_match_similarity=certv11_match_similarity,
        certv11_match_margin=certv11_match_margin,
        certv11_cycle_radius=certv11_cycle_radius,
        certv11_max_spatial_jump=certv11_max_spatial_jump,
        certv11_scene_similarity=certv11_scene_similarity,
        certv11_spatial_match_weight=certv11_spatial_match_weight,
        certv11_time_confidence_seconds=certv11_time_confidence_seconds,
        certv11_transition_dim=certv11_transition_dim,
        certv11_state_scale=certv11_state_scale,
        certv11_transition_weight_min=certv11_transition_weight_min,
        certv11_transition_weight_max=certv11_transition_weight_max,
        certv11_reliability_floor=certv11_reliability_floor,
        certv11_reliability_target=certv11_reliability_target,
        certv11_deficit_threshold=certv11_deficit_threshold,
        certv11_deficit_scale=certv11_deficit_scale,
        certv11_min_swap_ratio=certv11_min_swap_ratio,
        certv11_max_swap_ratio=certv11_max_swap_ratio,
        certv11_add_pool=certv11_add_pool,
        certv11_remove_pool=certv11_remove_pool,
        certv11_v3_protect_ratio=certv11_v3_protect_ratio,
        certv11_node_efficiency_floor=certv11_node_efficiency_floor,
        certv11_node_loss_weight=certv11_node_loss_weight,
        certv11_edge_coverage_weight=certv11_edge_coverage_weight,
        certv11_frame_balance_weight=certv11_frame_balance_weight,
        certv11_swap_margin=certv11_swap_margin,
        certv11_cross_frame_similarity=certv11_cross_frame_similarity,
        certv11_cross_frame_max_seconds=certv11_cross_frame_max_seconds,
        certv11_assignment_topk=certv11_assignment_topk,
        certv11_assignment_temperature=certv11_assignment_temperature,
        certv11_fusion_similarity_floor=certv11_fusion_similarity_floor,
        certv11_debug=certv11_debug,
        certv4_budget_mode=certv4_budget_mode,
        certv4_attention_policy=certv4_attention_policy,
        certv4_attention_eps=certv4_attention_eps,
        certv4_certificate_budget_ratio=certv4_certificate_budget_ratio,
        certv4_query_mode=certv4_query_mode,
        certv4_design_protect_ratio=certv4_design_protect_ratio,
        certv4_query_atoms=certv4_query_atoms,
        certv4_temporal_bins=certv4_temporal_bins,
        certv4_spatial_bins=certv4_spatial_bins,
        certv4_candidate_multiplier=certv4_candidate_multiplier,
        certv4_track_threshold=certv4_track_threshold,
        certv4_spatial_penalty=certv4_spatial_penalty,
        certv4_metric_dim=certv4_metric_dim,
        certv4_frame_coverage_ratio=certv4_frame_coverage_ratio,
        certv4_cell_coverage_ratio=certv4_cell_coverage_ratio,
        certv4_query_threshold=certv4_query_threshold,
        certv4_query_per_atom=certv4_query_per_atom,
        certv4_structural_weight=certv4_structural_weight,
        certv4_whitening_strength=certv4_whitening_strength,
        certv4_quality_floor=certv4_quality_floor,
        certv4_ridge=certv4_ridge,
        certv4_swap_steps=certv4_swap_steps,
        certv4_swap_pool=certv4_swap_pool,
        certv4_swap_margin=certv4_swap_margin,
        certv4_fusion_alpha=certv4_fusion_alpha,
        certv4_assignment_temperature=certv4_assignment_temperature,
        certv4_debug=certv4_debug,
        certv4_num_hidden_layers=_text_layer_count(model),
        certv4_inner_hook_enabled=True,
        certv5_budget_mode=certv5_budget_mode,
        certv5_ot_enabled=certv5_ot_enabled,
        certv5_ot_topk=certv5_ot_topk,
        certv5_ot_temperature=certv5_ot_temperature,
        certv5_ot_steps=certv5_ot_steps,
        certv5_ot_capacity_tau=certv5_ot_capacity_tau,
        certv5_ot_prior_shrink=certv5_ot_prior_shrink,
        certv5_ot_live_fraction=certv5_ot_live_fraction,
        certv5_ot_cost_slack=certv5_ot_cost_slack,
        certv5_ot_temporal_penalty=certv5_ot_temporal_penalty,
        certv5_ot_max_displacement=certv5_ot_max_displacement,
        certv5_ot_min_cosine=certv5_ot_min_cosine,
        certv5_debug=certv5_debug,
        certv5_num_hidden_layers=_text_layer_count(model),
        certv5_inner_hook_enabled=True,
        certe_budget_uses_expansion=certe_budget_uses_expansion,
        certe_ridge=certe_ridge,
        certe_bottom_k=certe_bottom_k,
        certe_swap_steps=certe_swap_steps,
        certe_remove_pool=certe_remove_pool,
        certe_add_pool=certe_add_pool,
        certe_verify_pool=certe_verify_pool,
        certe_swap_margin=certe_swap_margin,
        certe_spectral_temperature=certe_spectral_temperature,
        certe_d_efficiency_floor=certe_d_efficiency_floor,
        certe_rank_tolerance=certe_rank_tolerance,
        certe_debug=certe_debug,
        faith_budget_uses_expansion=faith_budget_uses_expansion,
        faith_mass_strength=faith_mass_strength,
        faith_variance_strength=faith_variance_strength,
        faith_merge_alpha=faith_merge_alpha,
        faith_temporal_radius=faith_temporal_radius,
        faith_spatial_radius=faith_spatial_radius,
        faith_component_bonus=faith_component_bonus,
        faith_temporal_penalty=faith_temporal_penalty,
        faith_spatial_penalty=faith_spatial_penalty,
        faith_assignment_topk=faith_assignment_topk,
        faith_assignment_temperature=faith_assignment_temperature,
        faith_max_log_bias=faith_max_log_bias,
        faith_attention_strict=faith_attention_strict,
        faith_debug=faith_debug,
        prism_budget_uses_expansion=prism_budget_uses_expansion,
        prism_metric_dim=prism_metric_dim,
        prism_query_atoms=prism_query_atoms,
        prism_candidate_multiplier=prism_candidate_multiplier,
        prism_probe_tokens=prism_probe_tokens,
        prism_frame_floor_ratio=prism_frame_floor_ratio,
        prism_attention_weight=prism_attention_weight,
        prism_event_weight=prism_event_weight,
        prism_query_weight=prism_query_weight,
        prism_disagreement_weight=prism_disagreement_weight,
        prism_router_strength=prism_router_strength,
        prism_coverage_weight=prism_coverage_weight,
        prism_pareto_weight=prism_pareto_weight,
        prism_batch_size=prism_batch_size,
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
        talon_transport_radius=talon_transport_radius,
        talon_rank_ratio=talon_rank_ratio,
        talon_rank_min=talon_rank_min,
        talon_rank_max=talon_rank_max,
        talon_budget_scale=talon_budget_scale,
        talon_target_tokens_per_frame=talon_target_tokens_per_frame,
        talon_short_target_tokens_per_frame=talon_short_target_tokens_per_frame,
        talon_medium_target_tokens_per_frame=talon_medium_target_tokens_per_frame,
        talon_long_target_tokens_per_frame=talon_long_target_tokens_per_frame,
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
        talon_anchor_diversity_weight=talon_anchor_diversity_weight,
        talon_anchor_candidate_multiplier=talon_anchor_candidate_multiplier,
        talon_spatial_anchor_coverage=talon_spatial_anchor_coverage,
        talon_spatial_anchor_ratio=talon_spatial_anchor_ratio,
        talon_spatial_anchor_rows=talon_spatial_anchor_rows,
        talon_spatial_anchor_cols=talon_spatial_anchor_cols,
        talon_spatial_anchor_score=talon_spatial_anchor_score,
        talon_spatial_anchor_apply_to_short=talon_spatial_anchor_apply_to_short,
        talon_frame_coverage_floor_ratio=talon_frame_coverage_floor_ratio,
        talon_frame_importance_pooling=talon_frame_importance_pooling,
        talon_frame_importance_topk=talon_frame_importance_topk,
        talon_medium_frame_coverage_floor_ratio=talon_medium_frame_coverage_floor_ratio,
        talon_long_frame_coverage_floor_ratio=talon_long_frame_coverage_floor_ratio,
        talon_frame_local_budget_ratio=talon_frame_local_budget_ratio,
        talon_question_recall_ratio=talon_question_recall_ratio,
        talon_question_recall_qweight=talon_question_recall_qweight,
        talon_persistence_recall_ratio=talon_persistence_recall_ratio,
        talon_persistence_recall_qweight=talon_persistence_recall_qweight,
        talon_persistence_recall_pweight=talon_persistence_recall_pweight,
        talon_persistence_apply_to_short=talon_persistence_apply_to_short,
        talon_persistence_apply_to_medium=talon_persistence_apply_to_medium,
        talon_persistence_apply_to_long=talon_persistence_apply_to_long,
        talon_object_evidence_ratio=talon_object_evidence_ratio,
        talon_object_evidence_qweight=talon_object_evidence_qweight,
        talon_object_evidence_sweight=talon_object_evidence_sweight,
        talon_object_evidence_pweight=talon_object_evidence_pweight,
        talon_object_evidence_apply_to_short=talon_object_evidence_apply_to_short,
        talon_object_evidence_apply_to_medium=talon_object_evidence_apply_to_medium,
        talon_object_evidence_apply_to_long=talon_object_evidence_apply_to_long,
        talon_question_pooling=talon_question_pooling,
        talon_question_pooling_topk=talon_question_pooling_topk,
        talon_question_contrast_weight=talon_question_contrast_weight,
        talon_question_contrast_apply_to_short=talon_question_contrast_apply_to_short,
        talon_monotonic_base_tokens_per_frame=talon_monotonic_base_tokens_per_frame,
        talon_budget_strategy=talon_budget_strategy,
        talon_budget_mode=talon_budget_mode,
        talon_transport_mode=talon_transport_mode,
        talon_transport_temperature=talon_transport_temperature,
        talon_lite_enabled=talon_lite_enabled,
        talon_echo_temperature=talon_echo_temperature,
        talon_echo_topk_neighbors=talon_echo_topk_neighbors,
        talon_echo_residual_weight=talon_echo_residual_weight,
        talon_echo_score_mode=talon_echo_score_mode,
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
        talon_event_keep_bonus=talon_event_keep_bonus,
        talon_legacy_base_keep_ratio=talon_legacy_base_keep_ratio,
        talon_prior_candidate_ratio=talon_prior_candidate_ratio,
        talon_prior_keep_bonus=talon_prior_keep_bonus,
        talon_flash_prior_channel_ratio=talon_flash_prior_channel_ratio,
        talon_flash_prior_channel_method=talon_flash_prior_channel_method,
        talon_flash_prior_channel_min_per_frame=talon_flash_prior_channel_min_per_frame,
        talon_flash_prior_channel_max_per_frame=talon_flash_prior_channel_max_per_frame,
        talon_flash_prior_channel_bonus=talon_flash_prior_channel_bonus,
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
        talon_duration_aware=talon_duration_aware,
        talon_medium_anchor_safety_ratio=talon_medium_anchor_safety_ratio,
        talon_medium_event_budget_ratio=talon_medium_event_budget_ratio,
        talon_medium_global_topk_ratio=talon_medium_global_topk_ratio,
        talon_long_anchor_safety_ratio=talon_long_anchor_safety_ratio,
        talon_long_event_budget_ratio=talon_long_event_budget_ratio,
        talon_long_global_topk_ratio=talon_long_global_topk_ratio,
        talon_task_aware_event=talon_task_aware_event,
        talon_task_event_attention_weight=talon_task_event_attention_weight,
        talon_task_event_qweight=talon_task_event_qweight,
        talon_visual_task_balance=talon_visual_task_balance,
        talon_visual_task_anchor_ratio=talon_visual_task_anchor_ratio,
        talon_visual_task_event_ratio=talon_visual_task_event_ratio,
        talon_visual_task_recall_ratio=talon_visual_task_recall_ratio,
        talon_knowledge_visual_anchor_ratio=talon_knowledge_visual_anchor_ratio,
        talon_knowledge_visual_event_ratio=talon_knowledge_visual_event_ratio,
        talon_knowledge_visual_recall_ratio=talon_knowledge_visual_recall_ratio,
        talon_adaptive_router=talon_adaptive_router,
        talon_router_apply_to_short=talon_router_apply_to_short,
        talon_router_visual_anchor_ratio=talon_router_visual_anchor_ratio,
        talon_router_visual_event_ratio=talon_router_visual_event_ratio,
        talon_router_visual_recall_ratio=talon_router_visual_recall_ratio,
        talon_router_temporal_anchor_ratio=talon_router_temporal_anchor_ratio,
        talon_router_temporal_event_ratio=talon_router_temporal_event_ratio,
        talon_router_temporal_recall_ratio=talon_router_temporal_recall_ratio,
        talon_router_balanced_anchor_ratio=talon_router_balanced_anchor_ratio,
        talon_router_balanced_event_ratio=talon_router_balanced_event_ratio,
        talon_router_balanced_recall_ratio=talon_router_balanced_recall_ratio,
        talon_router_visual_concentration_threshold=talon_router_visual_concentration_threshold,
        talon_router_low_residual_threshold=talon_router_low_residual_threshold,
        talon_router_temporal_entropy_threshold=talon_router_temporal_entropy_threshold,
        talon_router_temporal_residual_threshold=talon_router_temporal_residual_threshold,
        talon_temporal_chunk_aware=talon_temporal_chunk_aware,
        talon_temporal_num_chunks=talon_temporal_num_chunks,
        talon_temporal_chunk_min_ratio=talon_temporal_chunk_min_ratio,
        talon_temporal_chunk_score=talon_temporal_chunk_score,
        talon_track_aware=talon_track_aware,
        talon_track_budget_ratio=talon_track_budget_ratio,
        talon_track_tokens_per_slot=talon_track_tokens_per_slot,
        talon_track_score=talon_track_score,
        talon_absorb_dropped_tokens=talon_absorb_dropped_tokens,
        talon_absorb_ratio=talon_absorb_ratio,
        talon_absorb_alpha=talon_absorb_alpha,
        talon_absorb_score=talon_absorb_score,
        talon_summary_replacement=talon_summary_replacement,
        talon_summary_raw_swap=talon_summary_raw_swap,
        talon_summary_ratio=talon_summary_ratio,
        talon_summary_num_chunks=talon_summary_num_chunks,
        talon_summary_pool_topk=talon_summary_pool_topk,
        talon_summary_alpha=talon_summary_alpha,
        talon_summary_score=talon_summary_score,
        expansion=expansion,
        pruning_layer=pruning_layer,
        llm_retention_ratio=llm_retention_ratio,
        decode_policy=decode_policy,
        decode_kv_budget_ratio=decode_kv_budget_ratio,
        decode_update_interval=decode_update_interval,
        decode_start_layer=decode_start_layer,
    )

    if variant == "certvid_v4":
        from .certvid_v4 import _resolve_budget

        # Validate the layer-average contract before loading any benchmark sample.
        _resolve_budget(flashvid_config, total_tokens=1)

    if variant == "certvid_v5":
        from .certvid_v5 import _resolve_budget

        _resolve_budget(flashvid_config, total_tokens=1)

    # Store FlashVid Config in the model.
    if type(model) is Qwen2_5_VLForConditionalGeneration:
        setattr(flashvid_config, "_baseline_backbone", "qwen2_5_vl")
    elif type(model) is Qwen3VLForConditionalGeneration:
        setattr(flashvid_config, "_baseline_backbone", "qwen3_vl")
    else:
        setattr(flashvid_config, "_baseline_backbone", "llava")
    setattr(model, "flashvid_config", flashvid_config)
    setattr(model.model, "flashvid_config", flashvid_config)
    if variant == "certvid_v3plusplus":
        output_head = getattr(model, "lm_head", None)
        if output_head is None:
            raise RuntimeError("certvid_v3plusplus requires the LLaVA language-model head")
        # Qwen2Model.forward owns the pruning loop, while lm_head lives on the
        # causal-LM wrapper. Keep a non-registered reference for the proxy loss.
        object.__setattr__(model.model, "_v3plusplus_output_head", output_head)
    setattr(model.config, "flashvid_bypass_active", False)
    if type(model) in (Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration):
        setattr(model.model.language_model, "flashvid_config", flashvid_config)
        setattr(model.model.visual, "flashvid_config", flashvid_config)
    for module in model.modules():
        if isinstance(module, (Qwen2Attention, Qwen2_5_VLAttention, Qwen3VLTextAttention)):
            setattr(module, "flashvid_config", flashvid_config)

    return model
