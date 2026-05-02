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
    expansion: float = field(default=1.25)
    pruning_layer: int = field(default=20)
    llm_retention_ratio: float = field(default=0.3)

    # Experimental compression variant.
    # "flashvid": original ADTS + TSTM path.
    # "talon": transport-aligned low-rank + sparse innovation path.
    # "slot"/"graph": legacy aliases mapped to "talon" for compatibility.
    compression_variant: str = field(default="flashvid")

    # Question-aware token reweighting.
    question_aware_reweighting: bool = field(default=False)
    question_reweight_beta: float = field(default=0.35)

    # Graph-based spatiotemporal merging.
    graph_topk: int = field(default=4)
    graph_temporal_radius: int = field(default=1)

    # Slot-memory token aggregation.
    slot_base_roles: int = field(default=5)  # scene, motion, interaction, background, detail
    slot_max_per_segment: int = field(default=64)
    slot_role_allocation: str = field(default="motion,interaction,detail,scene,background")
    slot_overlap_radius: int = field(default=1)
    slot_tiebreak_eps: float = field(default=2e-2)
    slot_motion_window: int = field(default=1)
    slot_soft_cap_fraction: float = field(default=0.35)
    slot_anchor_blend: float = field(default=0.65)
    slot_passthrough_ratio: float = field(default=0.55)
    slot_passthrough_min: int = field(default=4)
    slot_fast_assignment: bool = field(default=True)

    # TALON compression.
    talon_transport_radius: int = field(default=1)
    talon_rank_ratio: float = field(default=0.40)
    talon_rank_min: int = field(default=2)
    talon_rank_max: int = field(default=32)
    talon_budget_mode: str = field(default="uniform")  # uniform | attention
    talon_use_question_innovation: bool = field(default=True)
    talon_innovation_qweight: float = field(default=0.25)
    talon_output_mode: str = field(default="manifold")  # manifold | coefficient
    talon_reconstruction_blend: float = field(default=0.25)
    talon_anchor_score_weight: float = field(default=0.35)
    talon_passthrough_ratio: float = field(default=0.15)
    talon_passthrough_min: int = field(default=2)
    talon_disable_oversegmentation: bool = field(default=True)
    talon_max_segments: int = field(default=4)

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

    # Spatial grid metadata (set by model hooks when available).
    H: Optional[int] = field(default=None)
    W: Optional[int] = field(default=None)

    # Decode-stage policy scaffold (Route3 hook, default no-op).
    decode_policy: str = field(default="none")
    decode_kv_budget_ratio: float = field(default=1.0)
    decode_update_interval: int = field(default=4)
    decode_start_layer: int = field(default=0)
