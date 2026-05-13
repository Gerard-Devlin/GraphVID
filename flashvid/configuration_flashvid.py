from typing import Optional

from dataclasses import dataclass, field


@dataclass
class FlashVidConfig:
    # Average retention ratio.
    retention_ratio: float = field(default=0.25)

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

    # Experimental compression variant.
    # "flashvid": original ADTS + TSTM path.
    # "talon": transport-aligned low-rank + sparse innovation path.
    compression_variant: str = field(default="flashvid")

    # Question-aware token reweighting.
    question_aware_reweighting: bool = field(default=False)
    question_reweight_beta: float = field(default=0.35)

    # TALON compression.
    talon_transport_radius: int = field(default=1)
    talon_rank_ratio: float = field(default=0.40)
    talon_rank_min: int = field(default=2)
    talon_rank_max: int = field(default=32)
    talon_budget_scale: float = field(default=0.60)
    talon_target_tokens_per_frame: int = field(default=0)
    talon_min_total_tokens: int = field(default=1)
    talon_fast_rank_plan: bool = field(default=True)
    talon_background_max_ratio: float = field(default=0.45)
    talon_frame_balanced_selection: bool = field(default=True)
    talon_basis_method: str = field(default="randomized")  # covariance | randomized
    talon_basis_oversample: int = field(default=4)
    talon_innovation_attention_weight: float = field(default=0.45)
    talon_motion_importance_weight: float = field(default=0.35)
    talon_boundary_importance_weight: float = field(default=0.10)
    talon_question_frame_weight: float = field(default=0.20)
    talon_frame_balanced_memory: bool = field(default=True)
    talon_memory_mode: str = field(default="raw")  # raw | merge
    talon_anchor_safety_ratio: float = field(default=0.28)
    talon_budget_strategy: str = field(default="marginal")  # ratio | marginal
    talon_budget_mode: str = field(default="uniform")  # uniform | attention
    talon_transport_mode: str = field(default="hard")  # hard | soft
    talon_transport_temperature: float = field(default=0.07)
    talon_rd_spectral_weight: float = field(default=1.0)
    talon_rd_innovation_weight: float = field(default=1.0)
    talon_use_question_innovation: bool = field(default=True)
    talon_innovation_qweight: float = field(default=0.25)
    talon_output_mode: str = field(default="manifold")  # manifold | coefficient
    talon_reconstruction_blend: float = field(default=0.0)
    talon_anchor_score_weight: float = field(default=0.35)
    talon_min_anchor_per_frame: int = field(default=2)
    talon_passthrough_ratio: float = field(default=0.15)
    talon_passthrough_min: int = field(default=2)
    talon_use_segmentation: bool = field(default=True)
    talon_disable_oversegmentation: bool = field(default=True)
    talon_max_segments: int = field(default=4)
    talon_deepstack_mode: str = field(default="keep")  # disable | keep | auto

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
    talon_adaptive_target_low: int = field(default=0)
    talon_adaptive_target_mid: int = field(default=0)
    talon_adaptive_target_high: int = field(default=0)
    talon_complexity_floor: float = field(default=0.20)
    talon_complexity_ceil: float = field(default=0.40)
    talon_adaptive_gamma: float = field(default=1.0)
    talon_adaptive_target_enabled: bool = field(default=True)
    talon_force_fixed_target: bool = field(default=False)
    talon_target_mean_cap: float = field(default=18.75)
    talon_running_target_sum: float = field(default=0.0)
    talon_running_target_count: int = field(default=0)
    talon_unified_selection: bool = field(default=True)
    talon_low_budget_mode_threshold: int = field(default=20)
    talon_low_budget_rank_cap: int = field(default=0)
    talon_background_global_ratio: float = field(default=0.60)
    talon_event_budget_ratio: float = field(default=0.30)
    talon_memory_fused_weight: float = field(default=0.50)
    talon_memory_residual_weight: float = field(default=0.35)
    talon_memory_frame_weight: float = field(default=0.15)
    talon_recall_memory_mode: str = field(default="raw")
    talon_final_fused_weight: float = field(default=0.70)
    talon_final_residual_weight: float = field(default=0.20)
    talon_final_frame_weight: float = field(default=0.10)
    talon_anchor_keep_bonus: float = field(default=0.10)
    talon_recall_keep_bonus: float = field(default=0.08)
    talon_final_anchor_min_ratio: float = field(default=0.24)
    talon_final_recall_min_ratio: float = field(default=0.10)
    talon_force_anchor_recall_quota: bool = field(default=True)
    talon_global_topk_ratio: float = field(default=0.70)
    talon_rescue_enabled: bool = field(default=True)
    talon_rescue_ratio: float = field(default=0.08)
    talon_rescue_from_memory_only: bool = field(default=True)
    talon_rescue_fused_weight: float = field(default=0.55)
    talon_rescue_residual_weight: float = field(default=0.35)
    talon_rescue_frame_weight: float = field(default=0.10)
    talon_rescue_global_ratio: float = field(default=0.85)
    talon_rerank_with_flash_prior: bool = field(default=True)
    talon_flash_prior_ratio: float = field(default=0.20)
    talon_recall_semantic_ratio: float = field(default=0.50)
    talon_recall_event_ratio: float = field(default=0.25)
    talon_recall_frame_ratio: float = field(default=0.15)
    talon_recall_global_ratio: float = field(default=0.55)
    last_talon_target_tokens_per_frame: Optional[int] = field(default=None)
    last_talon_complexity_score: Optional[float] = field(default=None)
    last_talon_target_budget: Optional[int] = field(default=None)
    last_talon_anchor_tokens: Optional[int] = field(default=None)
    last_talon_rank_tokens: Optional[int] = field(default=None)
    last_talon_event_tokens: Optional[int] = field(default=None)
    last_talon_recall_tokens: Optional[int] = field(default=None)
    last_talon_memory_tokens: Optional[int] = field(default=None)
    last_talon_segment_count: Optional[int] = field(default=None)

    # Spatial grid metadata (set by model hooks when available).
    H: Optional[int] = field(default=None)
    W: Optional[int] = field(default=None)

    # Decode-stage policy scaffold (Route3 hook, default no-op).
    decode_policy: str = field(default="none")
    decode_kv_budget_ratio: float = field(default=1.0)
    decode_update_interval: int = field(default=4)
    decode_start_layer: int = field(default=0)
