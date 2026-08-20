from typing import Optional

from dataclasses import dataclass, field


@dataclass
class FlashVidConfig:
    # Average retention ratio.
    retention_ratio: float = field(default=0.25)

    # Released baseline adapters for the LLaVA family.
    adapter_budget_uses_expansion: bool = field(default=False)
    fastvid_DySeg_c: int = field(default=8)
    fastvid_DySeg_tau: float = field(default=0.90)
    fastvid_DySeg_ignore: float = field(default=0.95)
    fastvid_STPrune_d: float = field(default=0.40)
    fastvid_DTM_p: int = field(default=4)
    fastvid_DTM_beta: float = field(default=0.60)
    visionzip_dominant_ratio: float = field(default=65.0 / 70.0)
    prunevid_tau: float = field(default=0.80)
    prunevid_temporal_segment_ratio: float = field(default=0.25)
    prunevid_cluster_ratio: float = field(default=0.50)

    # 1) Token Selection Method. Defaults to ADTS.
    alpha: float = field(default=0.7) # Ratio of ADTS tokens.
    token_selection_method: str = field(default="attn_div")

    # 2) Tree-based Spatio-Temporal Token Merging.
    temporal_threshold: float = field(default=0.8)
    dynamic_temporal_threshold: bool = field(default=False)
    temporal_threshold_quantile: float = field(default=0.8)
    temporal_threshold_min: float = field(default=0.0)
    temporal_threshold_max: float = field(default=0.99)
    last_dynamic_temporal_threshold: Optional[float] = field(default=None)
    temporal_match_mode: str = field(default="global")  # global | local
    temporal_local_radius: int = field(default=2)
    temporal_hysteresis: float = field(default=0.0)
    min_keep_per_frame: int = field(default=0)

    # CertVID: constrained evidence coreset with shared DeepStack fusion.
    cert_budget_uses_expansion: bool = field(default=True)
    cert_query_atoms: int = field(default=6)
    cert_temporal_bins: int = field(default=8)
    cert_spatial_bins: int = field(default=3)
    cert_candidate_multiplier: float = field(default=3.0)
    cert_query_weight: float = field(default=0.20)
    cert_temporal_weight: float = field(default=0.20)
    cert_detail_weight: float = field(default=0.10)
    cert_repair_ratio: float = field(default=0.20)
    cert_fusion_alpha: float = field(default=0.25)
    cert_assignment_temperature: float = field(default=0.07)
    cert_track_threshold: float = field(default=0.82)
    cert_spatial_penalty: float = field(default=0.08)
    cert_metric_dim: int = field(default=256)

    # CertVID V2: evidence backbone with gated trajectory repair.
    certv2_budget_uses_expansion: bool = field(default=True)
    certv2_query_atoms: int = field(default=6)
    certv2_temporal_bins: int = field(default=8)
    certv2_spatial_bins: int = field(default=3)
    certv2_candidate_multiplier: float = field(default=3.0)
    certv2_query_weight: float = field(default=0.18)
    certv2_frame_floor_ratio: float = field(default=0.08)
    certv2_diversity_weight: float = field(default=0.12)
    certv2_coverage_weight: float = field(default=0.10)
    certv2_density_neighbors: int = field(default=4)
    certv2_track_threshold: float = field(default=0.82)
    certv2_spatial_penalty: float = field(default=0.08)
    certv2_metric_dim: int = field(default=256)
    certv2_repair_ratio: float = field(default=0.05)
    certv2_repair_ratio_high: float = field(default=0.13)
    certv2_router_strength: float = field(default=0.65)
    certv2_protect_ratio: float = field(default=0.30)
    certv2_swap_margin: float = field(default=0.02)
    certv2_fusion_alpha: float = field(default=0.25)
    certv2_repair_fusion_alpha: float = field(default=0.08)
    certv2_assignment_temperature: float = field(default=0.07)

    # CertVID V3: certified regularized D-optimal evidence design.
    certv3_budget_uses_expansion: bool = field(default=True)
    certv3_query_atoms: int = field(default=8)
    certv3_temporal_bins: int = field(default=12)
    certv3_spatial_bins: int = field(default=3)
    certv3_candidate_multiplier: float = field(default=2.5)
    certv3_query_weight: float = field(default=0.18)
    certv3_visual_attention_weight: float = field(default=0.28)
    certv3_visual_novelty_weight: float = field(default=0.20)
    certv3_visual_curvature_weight: float = field(default=0.14)
    certv3_visual_event_weight: float = field(default=0.12)
    certv3_visual_detail_weight: float = field(default=0.12)
    certv3_visual_component_weight: float = field(default=0.14)
    certv3_event_novelty_weight: float = field(default=0.34)
    certv3_event_curvature_weight: float = field(default=0.28)
    certv3_event_frame_weight: float = field(default=0.18)
    certv3_event_detail_weight: float = field(default=0.10)
    certv3_event_query_weight: float = field(default=0.10)
    certv3_track_threshold: float = field(default=0.82)
    certv3_spatial_penalty: float = field(default=0.08)
    certv3_metric_dim: int = field(default=96)
    certv3_frame_coverage_ratio: float = field(default=1.0)
    certv3_cell_coverage_ratio: float = field(default=0.50)
    certv3_query_threshold: float = field(default=0.10)
    certv3_query_per_atom: int = field(default=1)
    # Applied only to the exact CertVID V3 variant on Qwen-family backbones.
    certv3_certificate_budget_ratio: float = field(default=1.0)
    certv3_qwen_certificate_budget_ratio: float = field(default=0.35)
    certv3_structural_weight: float = field(default=0.32)
    certv3_whitening_strength: float = field(default=0.50)
    certv3_quality_floor: float = field(default=0.15)
    certv3_ridge: float = field(default=0.50)
    certv3_swap_steps: int = field(default=6)
    certv3_swap_pool: int = field(default=24)
    certv3_swap_margin: float = field(default=1e-4)
    certv3_fusion_alpha: float = field(default=0.12)
    certv3_assignment_temperature: float = field(default=0.07)
    # Paper ablations. Defaults preserve the released CertVID V3 path.
    certv3_selection_objective: str = field(default="d_optimal")
    certv3_use_spatiotemporal_certificates: bool = field(default=True)
    certv3_use_spatiotemporal_design: bool = field(default=True)
    certv3_use_trajectory: bool = field(default=True)
    certv3_use_query: bool = field(default=True)

    # Dynamic Video Segmentation (DySeg).
    do_segment: bool = field(default=True)
    segment_threshold: float = field(default=0.9)
    min_segment_num: int = field(default=8)
    complementary_segment: bool = field(default=True)

    # Vision-Side Compression params.
    num_attn_div_tokens: Optional[int] = field(default=None)
    num_sttm_tokens: Optional[int] = field(default=None)

    # Inner-LLM Compression params.
    visual_token_start_index: Optional[int] = field(default=None)
    visual_token_length: Optional[int] = field(default=None)
    vision_token_length: Optional[int] = field(default=None)
    llm_token_length: Optional[int] = field(default=None)
    expansion: float = field(default=1.25)
    pruning_layer: int = field(default=20)
    llm_retention_ratio: float = field(default=0.3)
    # Internal fairness guard enabled only for FlashVID on Qwen2.5-VL.
    strict_token_budget: bool = field(default=False)

    # Experimental compression variant.
    # "flashvid": original ADTS + TSTM path.
    compression_variant: str = field(default="flashvid")

    # Question-aware token reweighting.
    question_aware_reweighting: bool = field(default=False)
    question_reweight_beta: float = field(default=0.35)

    # Residual memory tokens.
    memory_token_ratio: float = field(default=0.10)
    memory_token_min: int = field(default=1)
    memory_token_max: int = field(default=16)

    # Adaptive token budget.
    adaptive_token_budget: bool = field(default=False)
    adaptive_budget_low: float = field(default=0.10)
    adaptive_budget_mid: float = field(default=0.15)
    adaptive_budget_high: float = field(default=0.20)
    last_adaptive_retention_ratio: Optional[float] = field(default=None)
    current_video_duration: Optional[str] = field(default=None)
    current_task_category: Optional[str] = field(default=None)
    current_category: Optional[str] = field(default=None)
    # Spatial grid metadata (set by model hooks when available).
    H: Optional[int] = field(default=None)
    W: Optional[int] = field(default=None)

    # Decode-stage policy scaffold (Route3 hook, default no-op).
    decode_policy: str = field(default="none")
    decode_kv_budget_ratio: float = field(default=1.0)
    decode_update_interval: int = field(default=4)
    decode_start_layer: int = field(default=0)
