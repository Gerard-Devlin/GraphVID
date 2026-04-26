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
    # "graph": graph-based spatiotemporal merging path.
    compression_variant: str = field(default="flashvid")

    # Question-aware token reweighting.
    question_aware_reweighting: bool = field(default=False)
    question_reweight_beta: float = field(default=0.35)

    # Graph-based spatiotemporal merging.
    graph_topk: int = field(default=4)
    graph_temporal_radius: int = field(default=1)

    # Residual memory tokens.
    memory_token_ratio: float = field(default=0.10)
    memory_token_min: int = field(default=1)
    memory_token_max: int = field(default=8)

    # Adaptive token budget.
    adaptive_token_budget: bool = field(default=False)
    adaptive_budget_low: float = field(default=0.10)
    adaptive_budget_mid: float = field(default=0.15)
    adaptive_budget_high: float = field(default=0.20)
    last_adaptive_retention_ratio: Optional[float] = field(default=None)
