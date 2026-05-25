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
    temporal_merge_mode: str = field(default="tree")  # tree | graph
    graph_temporal_topk: int = field(default=3)
    graph_temporal_radius: int = field(default=1)
    graph_temporal_skip: int = field(default=1)
    graph_merge_protect_ratio: float = field(default=0.15)
    graph_merge_target_ratio: float = field(default=0.65)
    graph_merge_representative: str = field(default="medoid")  # medoid | mean | weighted_mean
    graph_representative_position: str = field(default="protection")  # protection | earliest | latest | medoid
    graph_protection_attn_weight: float = field(default=0.70)
    graph_protection_novelty_weight: float = field(default=0.30)
    graph_protection_detail_weight: float = field(default=0.0)
    graph_adaptive_detail_protection: bool = field(default=False)
    graph_adaptive_detail_boost: float = field(default=0.22)
    graph_adaptive_protect_boost: float = field(default=0.10)
    graph_merge_importance_penalty: float = field(default=0.0)
    graph_respect_temporal_threshold: bool = field(default=False)
    graph_final_tokens_per_frame: int = field(default=0)
    graph_final_frame_floor_ratio: float = field(default=0.55)
    graph_skip_spatial_merge_when_capped: bool = field(default=True)

    # GRAFT-VID constrained temporal forest.
    graft_temporal_topk: int = field(default=3)
    graft_temporal_radius: int = field(default=1)
    graft_temporal_skip: int = field(default=1)
    graft_global_topk: int = field(default=3)
    graft_input_is_residual: bool = field(default=True)
    graft_anchor_ratio: Optional[float] = field(default=None)
    graft_edge_threshold: float = field(default=0.80)
    graft_component_radius_eps: float = field(default=0.12)
    graft_split_radius_eps: float = field(default=0.20)
    graft_parent_capacity: int = field(default=1)
    graft_mutual_knn: bool = field(default=True)
    graft_one_token_per_frame: bool = field(default=True)
    graft_spatial_penalty: float = field(default=0.10)
    graft_importance_penalty: float = field(default=0.05)
    graft_hub_penalty: float = field(default=0.05)
    graft_adaptive_aggregation: bool = field(default=True)
    graft_scene_threshold: float = field(default=0.0)
    graft_min_tokens_per_frame: int = field(default=0)
    graft_budget_correction: bool = field(default=True)
    graft_budget_diversity_weight: float = field(default=0.35)
    graft_score_preset: str = field(default="base")  # base | event_v1 | event_v2
    graft_duration_aware: bool = field(default=False)
    graft_medium_temporal_skip: Optional[int] = field(default=None)
    graft_medium_global_topk: Optional[int] = field(default=None)
    graft_medium_edge_threshold: Optional[float] = field(default=None)
    graft_medium_split_radius_eps: Optional[float] = field(default=None)
    graft_medium_spatial_penalty: Optional[float] = field(default=None)
    graft_medium_scene_threshold: Optional[float] = field(default=None)
    graft_long_temporal_skip: Optional[int] = field(default=None)
    graft_long_global_topk: Optional[int] = field(default=None)
    graft_long_edge_threshold: Optional[float] = field(default=None)
    graft_long_split_radius_eps: Optional[float] = field(default=None)
    graft_long_spatial_penalty: Optional[float] = field(default=None)
    graft_long_scene_threshold: Optional[float] = field(default=None)

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
    # "graftvid": ADTS + constrained temporal forest path.
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
