import copy
import gc
import json
import math
import os
import random
import re
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from decord import VideoReader, cpu
from transformers.hf_argparser import HfArgumentParser

warnings.filterwarnings("ignore")

SEPARATOR = "=" * 72


def _canonical_method_name(value: str | None, *, default: str = "ours") -> str:
    """Return a stable lowercase phase key for logs, jsonl names, and summaries."""
    text = str(value or default).strip().lower()
    text = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in text)
    text = text.strip("_-")
    return text or default


def _ours_phase_key(args: "BenchmarkArgs") -> str:
    return _canonical_method_name(getattr(args, "compression_variant", "ours"), default="ours")


def _ours_output_path(args: "BenchmarkArgs", phase_key: str) -> str:
    attr = f"{phase_key}_output"
    phase_output = getattr(args, attr, None)
    try:
        ours_default = BenchmarkArgs.__dataclass_fields__["ours_output"].default
        phase_default = BenchmarkArgs.__dataclass_fields__[attr].default
    except Exception:
        ours_default = "logs/efficiency/ours_all_metrics.jsonl"
        phase_default = None
    if str(getattr(args, "ours_output", ours_default)) != str(ours_default) and str(phase_output or "") == str(phase_default):
        return str(args.ours_output)
    return str(phase_output or args.ours_output)


def _phase_display_name(phase_key: str) -> str:
    labels = {
        "baseline": "Baseline",
        "flashvid": "FlashVID",
        "graphvid": "GraphVID",
        "graftvid": "GraftVID",
        "cats": "CATS",
        "talon": "TALON",
        "hedgevid": "HedgeVID",
        "dynflashvid": "DynFlashVID",
        "learnflashvid": "LearnFlashVID",
        "ours": "Ours",
    }
    return labels.get(phase_key, phase_key)


def _phase_order(summary: dict[str, Any] | None = None) -> list[str]:
    preferred = ["baseline", "flashvid", "talon", "hedgevid", "dynflashvid", "learnflashvid", "ours", "graphvid", "graftvid", "cats"]
    if not summary:
        return preferred
    extras = [
        key
        for key, value in summary.items()
        if key not in preferred and key not in ("comparison", "duration_breakdown") and isinstance(value, dict)
    ]
    return preferred + sorted(extras)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if torch.is_tensor(value):
        if value.numel() == 1:
            return _json_safe(value.detach().cpu().item())
        return [_json_safe(v) for v in value.detach().cpu().tolist()]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _jsonl_line(record: dict[str, Any]) -> str:
    safe_record = _json_safe(record)
    # Use ASCII escaping for result files. Some datasets/model outputs may contain
    # odd Unicode surrogate fragments; ensure_ascii=True keeps JSONL writable and
    # parseable on every server locale.
    line = json.dumps(safe_record, ensure_ascii=True, allow_nan=False)
    # Catch malformed records at write time, not after a full benchmark phase.
    json.loads(line)
    return line + "\n"

GRAFT_METRIC_KEYS = [
    "graft_num_nodes",
    "graft_target_components",
    "graft_protected_count",
    "graft_entries_before_budget",
    "graft_entries_after_budget",
    "graft_scene_threshold",
    "graft_global_topk",
    "graft_anchor_ratio",
    "graft_input_is_residual",
    "graft_budget_diversity_weight",
    "graft_score_preset_code",
    "graft_duration_aware",
    "graft_budget_correction_active",
    "graft_protected_kept_count",
    "graft_component_count",
    "graft_avg_component_size",
    "graft_max_component_size",
    "graft_radius_mean",
    "graft_radius_max",
    "graft_edges_considered",
    "graft_edges_accepted",
    "graft_mutual_rejected",
    "graft_radius_rejected",
    "graft_capacity_rejected",
    "graft_same_frame_rejected",
]
CATS_METRIC_KEYS = [
    "cats_adts_mode_code",
    "cats_selected_tokens",
    "cats_sink_merges",
    "cats_residual_merges",
    "cats_mutual_rejected",
    "cats_margin_rejected",
    "cats_threshold_rejected",
    "cats_retained_residual_tokens",
    "cats_spatial_tokens_before",
    "cats_spatial_tokens_after",
    "cats_mean_merge_sim",
    "cats_mean_margin",
]
DYN_METRIC_KEYS = [
    "dyn_selected_tokens",
    "dyn_budget_min",
    "dyn_budget_max",
    "dyn_budget_std",
    "dyn_sink_merges",
    "dyn_residual_merges",
    "dyn_retained_residual_tokens",
    "dyn_spatial_tokens_before",
    "dyn_spatial_tokens_after",
    "dyn_mean_merge_sim",
    "dyn_similarity_debias_active",
    "dyn_sink_active",
    "dyn_weighted_active",
    "dyn_density_frame_mean",
    "dyn_event_chunk_mean",
    "dyn_motion_frame_mean",
]
HEDGE_METRIC_KEYS = [
    "hedge_selected_adts",
    "hedge_residual_budget",
    "hedge_stable_candidates",
    "hedge_evidence_candidates",
    "hedge_stable_selected",
    "hedge_evidence_selected",
    "hedge_final_tokens",
    "hedge_stable_floor_ratio",
    "hedge_diversity_weight",
]
LEARN_METRIC_KEYS = [
    "learn_selected_tokens",
    "learn_stable_tokens",
    "learn_selector_tokens",
    "learn_qaware_active",
    "learn_score_mean",
    "learn_score_std",
    "learn_teacher_keep_ratio",
]


@dataclass
class BenchmarkArgs:
    # Model
    model_path: str = field(default="Qwen/Qwen2.5-VL-7B-Instruct")
    model_backend: str = field(default="auto")  # auto | qwen2_5_vl | qwen3_vl | llava
    attn_implementation: str = field(default="flash_attention_2")
    local_files_only: bool = field(default=False)

    # Data
    dataset_jsonl: str = field(default="videomme.jsonl")
    hf_home: str | None = field(default=None)
    start_index: int = field(default=0)
    limit: int | None = field(default=100)
    shuffle: bool = field(default=True)
    duration_filter: str = field(default="")
    num_frames: int = field(default=64)
    max_pixels: int = field(default=256 * 28 * 28)
    min_pixels: int = field(default=64 * 28 * 28)

    # Runtime
    num_warmup: int = field(default=1)
    num_runs: int = field(default=3)
    max_new_tokens: int = field(default=16)

    # Which phases to run
    run_baseline: bool = field(default=True)
    run_flashvid: bool = field(default=True)
    run_ours: bool = field(default=True)
    run_graphvid: bool = field(default=False)
    run_graftvid: bool = field(default=False)
    run_cats: bool = field(default=False)
    run_dynflashvid: bool = field(default=False)
    reload_model_each_phase: bool = field(default=True)

    # FlashVID settings for phase-2
    retention_ratio: float = field(default=0.10)
    do_segment: bool = field(default=True)
    segment_threshold: float = field(default=0.9)
    min_segment_num: int = field(default=8)
    complementary_segment: bool = field(default=True)
    token_selection_method: str = field(default="attn_div_v2")
    flashvid_token_selection_method: str = field(default="attn_div_v2")
    graphvid_token_selection_method: str = field(default="")
    alpha: float = field(default=0.70)
    temporal_threshold: float = field(default=0.8)
    temporal_merge_mode: str = field(default="tree")
    graph_temporal_topk: int = field(default=3)
    graph_temporal_radius: int = field(default=1)
    graph_temporal_skip: int = field(default=1)
    graph_merge_protect_ratio: float = field(default=0.15)
    graph_merge_target_ratio: float = field(default=0.65)
    graph_merge_representative: str = field(default="medoid")
    graph_representative_position: str = field(default="protection")
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
    graft_score_preset: str = field(default="base")
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
    cats_adts_mode: str = field(default="cats")
    cats_adts_beta: float = field(default=0.05)
    cats_margin_threshold: float = field(default=0.03)
    cats_high_conf_bonus: float = field(default=0.05)
    cats_mutual_nn: bool = field(default=True)
    cats_confidence_attn_weight: float = field(default=0.75)
    cats_confidence_sim_weight: float = field(default=1.0)
    cats_anchor_self_weight: float = field(default=1.0)
    cats_adaptive_adts_budget: bool = field(default=False)
    cats_frame_budget_min: int = field(default=1)
    cats_frame_budget_temperature: float = field(default=0.7)
    dyn_adaptive_adts_budget: bool = field(default=True)
    dyn_budget_strength: float = field(default=0.45)
    dyn_budget_temperature: float = field(default=0.75)
    dyn_frame_budget_min_ratio: float = field(default=0.50)
    dyn_frame_budget_max_ratio: float = field(default=1.75)
    dyn_boundary_boost: float = field(default=0.08)
    dyn_adts_beta: float = field(default=0.05)
    dyn_attn_weight: float = field(default=0.50)
    dyn_event_weight: float = field(default=0.30)
    dyn_novelty_weight: float = field(default=0.15)
    dyn_detail_weight: float = field(default=0.05)
    dyn_density_weight: float = field(default=0.15)
    dyn_density_topk: int = field(default=8)
    dyn_event_chunk_radius: int = field(default=2)
    dyn_frame_event_weight: float = field(default=0.30)
    dyn_frame_novelty_weight: float = field(default=0.25)
    dyn_frame_attn_weight: float = field(default=0.20)
    dyn_frame_density_weight: float = field(default=0.20)
    dyn_frame_detail_weight: float = field(default=0.05)
    dyn_similarity_debias: bool = field(default=True)
    dyn_debias_frame_weight: float = field(default=0.35)
    dyn_debias_global_weight: float = field(default=0.20)
    dyn_sink_tstm: bool = field(default=False)
    dyn_mutual_nn: bool = field(default=False)
    dyn_margin_threshold: float = field(default=0.0)
    dyn_high_conf_bonus: float = field(default=0.05)
    dyn_weighted_merge: bool = field(default=False)
    dyn_confidence_attn_weight: float = field(default=0.50)
    dyn_confidence_sim_weight: float = field(default=0.50)
    learn_selector_ckpt: str = field(default="")
    learn_qaware: bool = field(default=True)
    learn_stable_floor_ratio: float = field(default=0.50)
    learn_score_blend: float = field(default=0.50)
    learn_q_relevance_weight: float = field(default=0.20)
    learn_density_topk: int = field(default=8)
    learn_collect_teacher: bool = field(default=False)
    hedge_stable_floor_ratio: float = field(default=0.85)
    hedge_diversity_weight: float = field(default=0.04)
    hedge_stable_bias: float = field(default=0.05)
    hedge_evidence_bias: float = field(default=0.0)
    hedge_max_mmr_candidates: int = field(default=2048)
    expansion: float = field(default=1.25)
    pruning_layer: int = field(default=20)
    llm_retention_ratio: float = field(default=0.3)

    # New experimental knobs (optional)
    compression_variant: str = field(default="talon")
    question_aware_reweighting: bool = field(default=False)
    question_reweight_beta: float = field(default=0.35)
    adaptive_token_budget: bool = field(default=False)
    adaptive_budget_low: float = field(default=0.10)
    adaptive_budget_mid: float = field(default=0.15)
    adaptive_budget_high: float = field(default=0.20)
    talon_core_target_tokens_per_frame: int = field(default=0)
    talon_core_neighbor_radius: int = field(default=1)
    talon_core_topk_neighbors: int = field(default=4)
    talon_core_temperature: float = field(default=0.07)
    talon_core_rank: int = field(default=4)
    talon_core_anchor_ratio: float = field(default=0.35)
    talon_core_relevance_weight: float = field(default=0.42)
    talon_core_temporal_weight: float = field(default=0.33)
    talon_core_lowrank_weight: float = field(default=0.25)
    talon_core_frame_budget_mode: str = field(default="attention")
    talon_core_min_keep_per_frame: int = field(default=1)
    talon_adaptive_target_low: int = field(default=0)
    talon_adaptive_target_mid: int = field(default=0)
    talon_adaptive_target_high: int = field(default=0)
    talon_complexity_floor: float = field(default=0.20)
    talon_complexity_ceil: float = field(default=0.40)
    talon_adaptive_gamma: float = field(default=1.0)
    talon_adaptive_target_enabled: bool = field(default=False)
    talon_force_fixed_target: bool = field(default=False)
    talon_target_mean_cap: float = field(default=0.0)
    talon_unified_selection: bool = field(default=False)
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
    talon_event_keep_bonus: float = field(default=0.04)
    talon_legacy_base_keep_ratio: float = field(default=0.85)
    talon_prior_candidate_ratio: float = field(default=0.12)
    talon_prior_keep_bonus: float = field(default=0.06)
    talon_flash_prior_channel_ratio: float = field(default=0.12)
    talon_flash_prior_channel_method: str = field(default="attn_div_v2")
    talon_flash_prior_channel_min_per_frame: int = field(default=1)
    talon_flash_prior_channel_max_per_frame: int = field(default=4)
    talon_flash_prior_channel_bonus: float = field(default=0.06)
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
    talon_duration_aware: bool = field(default=False)
    talon_medium_anchor_safety_ratio: float = field(default=0.72)
    talon_medium_event_budget_ratio: float = field(default=0.30)
    talon_medium_global_topk_ratio: float = field(default=0.70)
    talon_long_anchor_safety_ratio: float = field(default=0.80)
    talon_long_event_budget_ratio: float = field(default=0.14)
    talon_long_global_topk_ratio: float = field(default=0.85)
    talon_task_aware_event: bool = field(default=False)
    talon_task_event_attention_weight: float = field(default=0.82)
    talon_task_event_qweight: float = field(default=0.30)
    talon_visual_task_balance: bool = field(default=False)
    talon_visual_task_anchor_ratio: float = field(default=0.84)
    talon_visual_task_event_ratio: float = field(default=0.12)
    talon_visual_task_recall_ratio: float = field(default=0.02)
    talon_knowledge_visual_anchor_ratio: float = field(default=0.78)
    talon_knowledge_visual_event_ratio: float = field(default=0.18)
    talon_knowledge_visual_recall_ratio: float = field(default=0.06)
    talon_adaptive_router: bool = field(default=False)
    talon_router_apply_to_short: bool = field(default=False)
    talon_router_visual_anchor_ratio: float = field(default=0.76)
    talon_router_visual_event_ratio: float = field(default=0.24)
    talon_router_visual_recall_ratio: float = field(default=0.06)
    talon_router_temporal_anchor_ratio: float = field(default=0.66)
    talon_router_temporal_event_ratio: float = field(default=0.34)
    talon_router_temporal_recall_ratio: float = field(default=0.08)
    talon_router_balanced_anchor_ratio: float = field(default=0.72)
    talon_router_balanced_event_ratio: float = field(default=0.30)
    talon_router_balanced_recall_ratio: float = field(default=0.08)
    talon_router_visual_concentration_threshold: float = field(default=0.28)
    talon_router_low_residual_threshold: float = field(default=0.30)
    talon_router_temporal_entropy_threshold: float = field(default=0.95)
    talon_router_temporal_residual_threshold: float = field(default=0.36)
    talon_temporal_chunk_aware: bool = field(default=False)
    talon_temporal_num_chunks: int = field(default=4)
    talon_temporal_chunk_min_ratio: float = field(default=0.18)
    talon_temporal_chunk_score: str = field(default="combined")
    talon_track_aware: bool = field(default=False)
    talon_track_budget_ratio: float = field(default=0.12)
    talon_track_tokens_per_slot: int = field(default=1)
    talon_track_score: str = field(default="combined")
    talon_absorb_dropped_tokens: bool = field(default=False)
    talon_absorb_ratio: float = field(default=0.35)
    talon_absorb_alpha: float = field(default=0.25)
    talon_absorb_score: str = field(default="combined")
    talon_summary_replacement: bool = field(default=False)
    talon_summary_raw_swap: bool = field(default=False)
    talon_summary_ratio: float = field(default=0.08)
    talon_summary_num_chunks: int = field(default=8)
    talon_summary_pool_topk: int = field(default=12)
    talon_summary_alpha: float = field(default=0.55)
    talon_summary_score: str = field(default="combined")
    talon_transport_radius: int = field(default=1)
    talon_rank_ratio: float = field(default=0.40)
    talon_rank_min: int = field(default=2)
    talon_rank_max: int = field(default=32)
    talon_budget_scale: float = field(default=0.60)
    talon_target_tokens_per_frame: int = field(default=0)
    talon_short_target_tokens_per_frame: int = field(default=0)
    talon_medium_target_tokens_per_frame: int = field(default=0)
    talon_long_target_tokens_per_frame: int = field(default=0)
    talon_min_total_tokens: int = field(default=1)
    talon_fast_rank_plan: bool = field(default=True)
    talon_background_max_ratio: float = field(default=0.45)
    talon_frame_balanced_selection: bool = field(default=True)
    talon_basis_method: str = field(default="randomized")
    talon_basis_oversample: int = field(default=4)
    talon_innovation_attention_weight: float = field(default=0.45)
    talon_motion_importance_weight: float = field(default=0.35)
    talon_boundary_importance_weight: float = field(default=0.10)
    talon_question_frame_weight: float = field(default=0.20)
    talon_frame_balanced_memory: bool = field(default=True)
    talon_memory_mode: str = field(default="raw")
    talon_anchor_safety_ratio: float = field(default=0.28)
    talon_anchor_diversity_weight: float = field(default=0.0)
    talon_anchor_candidate_multiplier: float = field(default=4.0)
    talon_spatial_anchor_coverage: bool = field(default=False)
    talon_spatial_anchor_ratio: float = field(default=0.35)
    talon_spatial_anchor_rows: int = field(default=3)
    talon_spatial_anchor_cols: int = field(default=3)
    talon_spatial_anchor_score: str = field(default="fused")
    talon_spatial_anchor_apply_to_short: bool = field(default=False)
    talon_frame_coverage_floor_ratio: float = field(default=0.65)
    talon_frame_importance_pooling: str = field(default="mean")
    talon_frame_importance_topk: int = field(default=6)
    talon_medium_frame_coverage_floor_ratio: float = field(default=-1.0)
    talon_long_frame_coverage_floor_ratio: float = field(default=-1.0)
    talon_frame_local_budget_ratio: float = field(default=1.0)
    talon_question_recall_ratio: float = field(default=0.06)
    talon_question_recall_qweight: float = field(default=0.65)
    talon_persistence_recall_ratio: float = field(default=0.0)
    talon_persistence_recall_qweight: float = field(default=0.50)
    talon_persistence_recall_pweight: float = field(default=0.35)
    talon_persistence_apply_to_short: bool = field(default=False)
    talon_persistence_apply_to_medium: bool = field(default=True)
    talon_persistence_apply_to_long: bool = field(default=False)
    talon_object_evidence_ratio: float = field(default=0.0)
    talon_object_evidence_qweight: float = field(default=0.35)
    talon_object_evidence_sweight: float = field(default=0.45)
    talon_object_evidence_pweight: float = field(default=0.10)
    talon_object_evidence_apply_to_short: bool = field(default=False)
    talon_object_evidence_apply_to_medium: bool = field(default=True)
    talon_object_evidence_apply_to_long: bool = field(default=False)
    talon_question_pooling: str = field(default="mean")
    talon_question_pooling_topk: int = field(default=4)
    talon_question_contrast_weight: float = field(default=0.0)
    talon_question_contrast_apply_to_short: bool = field(default=False)
    talon_monotonic_base_tokens_per_frame: int = field(default=20)
    talon_budget_strategy: str = field(default="marginal")
    talon_budget_mode: str = field(default="attention")
    talon_transport_mode: str = field(default="hard")
    talon_transport_temperature: float = field(default=0.07)
    talon_lite_enabled: bool = field(default=False)
    talon_echo_temperature: float = field(default=0.07)
    talon_echo_topk_neighbors: int = field(default=4)
    talon_echo_residual_weight: float = field(default=0.0)
    talon_echo_score_mode: str = field(default="mse")
    talon_rd_spectral_weight: float = field(default=1.0)
    talon_rd_innovation_weight: float = field(default=1.0)
    talon_use_question_innovation: bool = field(default=True)
    talon_innovation_qweight: float = field(default=0.25)
    talon_output_mode: str = field(default="manifold")
    talon_reconstruction_blend: float = field(default=0.0)
    talon_anchor_score_weight: float = field(default=0.35)
    talon_min_anchor_per_frame: int = field(default=2)
    talon_passthrough_ratio: float = field(default=0.15)
    talon_passthrough_min: int = field(default=2)
    talon_use_segmentation: bool = field(default=True)
    talon_disable_oversegmentation: bool = field(default=True)
    talon_max_segments: int = field(default=4)
    talon_deepstack_mode: str = field(default="keep")
    memory_token_ratio: float = field(default=0.10)
    memory_token_min: int = field(default=1)
    memory_token_max: int = field(default=16)
    decode_policy: str = field(default="none")
    decode_kv_budget_ratio: float = field(default=1.0)
    decode_update_interval: int = field(default=4)
    decode_start_layer: int = field(default=0)

    # Outputs
    baseline_output: str = field(default="logs/efficiency/baseline_all_metrics.jsonl")
    flashvid_output: str = field(default="logs/efficiency/flashvid_all_metrics.jsonl")
    ours_output: str = field(default="logs/efficiency/ours_all_metrics.jsonl")
    graphvid_output: str = field(default="logs/efficiency/graphvid_all_metrics.jsonl")
    graftvid_output: str = field(default="logs/efficiency/graftvid_all_metrics.jsonl")
    cats_output: str = field(default="logs/efficiency/cats_all_metrics.jsonl")
    dynflashvid_output: str = field(default="logs/efficiency/dynflashvid_all_metrics.jsonl")
    learnflashvid_output: str = field(default="logs/efficiency/learnflashvid_all_metrics.jsonl")
    summary_output_json: str = field(default="logs/efficiency/summary_all_metrics.json")


_MCQ_CHOICES = ("A", "B", "C", "D")
_MCQ_ANSWER_PHRASES = [
    "the answer is",
    "answer is",
    "the correct answer is",
    "correct answer is",
    "the best answer is",
    "best answer is",
    "the correct option is",
    "correct option is",
    "the best option is",
    "best option is",
    "the choice is",
    "choice is",
    "the correct choice is",
    "correct choice is",
    "i choose",
    "i select",
    "i pick",
    "my answer is",
    "my choice is",
    "答案是",
    "答案为",
    "选",
]
_MCQ_FORMAT_PRIORITY = {
    "start": 10,
    "end": 9,
    "phrase": 7,
    "parentheses": 6,
    "period": 5,
    "colon": 4,
    "right_paren": 3,
    "space": 2,
    "fallback": 0,
}
_VIDEOMME_OPTION_PROMPT = (
    "Select the best answer to the following multiple-choice question based on the video and the subtitles. "
    "Respond with only the letter (A, B, C, or D) of the correct option."
)
_VIDEOMME_LEGACY_POST_PROMPT = "Answer with the option's letter from the given choices directly."


def _extract_choice_letter(text: str) -> str:
    """Match current lmms-eval VideoMME extract_mcq_answer priority rules."""
    if not text or not text.strip():
        return ""
    stripped = text.strip()
    for char in [",", ".", "!", "?", ";", ":", "'", '"']:
        stripped = stripped.strip(char)
    padded = f" {stripped} "
    candidates: list[tuple[str, int, str]] = []

    for ch in _MCQ_CHOICES:
        if f"({ch})" in padded:
            candidates.append((ch, padded.rfind(f"({ch})"), "parentheses"))
        if f"{ch}." in padded:
            candidates.append((ch, padded.rfind(f"{ch}."), "period"))
        if f"{ch}:" in padded:
            candidates.append((ch, padded.rfind(f"{ch}:"), "colon"))
        if f"{ch})" in padded:
            candidates.append((ch, padded.rfind(f"{ch})"), "right_paren"))
        if f"{ch} " in padded:
            candidates.append((ch, padded.rfind(f"{ch} "), "space"))

    padded_lower = padded.lower()
    for phrase in _MCQ_ANSWER_PHRASES:
        idx = padded_lower.find(phrase)
        if idx == -1:
            continue
        after = idx + len(phrase)
        for ch in _MCQ_CHOICES:
            ch_pos = padded.find(ch, after)
            if ch_pos != -1:
                candidates.append((ch, ch_pos, "phrase"))

    compact = padded.strip()
    for ch in _MCQ_CHOICES:
        if compact.startswith(ch) and (len(compact) == 1 or not compact[1].isalpha()):
            candidates.append((ch, 0, "start"))
        if compact.endswith(ch) and (len(compact) == 1 or not compact[-2].isalpha()):
            candidates.append((ch, len(padded) - 1, "end"))

    if not candidates:
        for ch in _MCQ_CHOICES:
            if ch in padded:
                candidates.append((ch, padded.rfind(ch), "fallback"))
    if not candidates:
        return ""
    candidates.sort(key=lambda x: (_MCQ_FORMAT_PRIORITY.get(x[2], 0), x[1]), reverse=True)
    return candidates[0][0]


def _strip_videomme_post_prompt(prompt: str) -> str:
    prompt = (prompt or "").strip()
    for suffix in (
        _VIDEOMME_LEGACY_POST_PROMPT,
        "The best answer is:",
        "Answer with the option letter only.",
        "Answer the question with A, B, C, or D.",
    ):
        if prompt.endswith(suffix):
            return prompt[: -len(suffix)].rstrip()
    return prompt


def _split_videomme_question_options(prompt_text: str) -> tuple[str, str]:
    prompt = _strip_videomme_post_prompt(prompt_text)
    lines = [line.strip() for line in prompt.splitlines() if line.strip()]
    if lines and lines[0] == _VIDEOMME_OPTION_PROMPT:
        lines = lines[1:]
    first_option = None
    for idx, line in enumerate(lines):
        if re.match(r"^[A-D]\s*[\.\)]\s+", line):
            first_option = idx
            break
    if first_option is None:
        return prompt, ""
    question = "\n".join(lines[:first_option]).strip()
    options = "\n".join(lines[first_option:]).strip()
    return question, options


def _to_lmms_videomme_prompt(prompt_text: str, backend: str = "llava") -> str:
    """Use current lmms-eval VideoMME backend-specific prompt variants."""
    backend = str(backend or "").lower()
    if backend == "qwen3_vl":
        question, options = _split_videomme_question_options(prompt_text)
        if options:
            return f"Question: {question}\nOptions:\n{options}\nAnswer with the option letter only."
        return f"Question: {_strip_videomme_post_prompt(prompt_text)}\nAnswer with the option letter only."

    prompt = _strip_videomme_post_prompt(prompt_text)
    if backend == "llava":
        if not prompt.endswith("The best answer is:"):
            prompt = f"{prompt}\nThe best answer is:"
        return prompt

    if not prompt.endswith(_VIDEOMME_LEGACY_POST_PROMPT):
        prompt = f"{prompt}\n{_VIDEOMME_LEGACY_POST_PROMPT}"
    return prompt


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "std": None}
    arr = np.array(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
    }


def _infer_backend(model_backend: str, model_path: str) -> str:
    if model_backend != "auto":
        return model_backend
    name = model_path.lower()
    if "qwen3-vl" in name:
        return "qwen3_vl"
    if "qwen2.5-vl" in name or "qwen2_5_vl" in name:
        return "qwen2_5_vl"
    return "llava"


def _resolve_attn_implementation(attn_implementation: str) -> str:
    if attn_implementation == "flash_attention_2" and not torch.cuda.is_available():
        print("[warn] CUDA is not available; fallback attn_implementation: flash_attention_2 -> eager")
        return "eager"
    return attn_implementation


def _assert_flash_attn_placement(model: Any, attn_implementation: str):
    if attn_implementation != "flash_attention_2":
        return
    device_map = getattr(model, "hf_device_map", None)
    if not isinstance(device_map, dict):
        return
    has_cpu_or_disk = any(str(dev) in ("cpu", "disk") for dev in device_map.values())
    if has_cpu_or_disk:
        raise RuntimeError(
            "flash_attention_2 requires all attention compute on CUDA, but current "
            "device_map contains CPU/DISK offload. Please use a smaller model / fewer frames, "
            "or set --attn_implementation sdpa (or eager)."
        )


def _timed_call(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        return out, float(start.elapsed_time(end))

    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) * 1000.0


def _resolve_generation_device(model: Any) -> torch.device:
    # Prefer language embedding device for generation inputs.
    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            dev = emb.weight.device
            if str(dev) != "meta":
                return dev
    except Exception:
        pass

    # Fallbacks for different wrapped model layouts.
    for path in (
        ("model", "embed_tokens", "weight"),
        ("language_model", "model", "embed_tokens", "weight"),
        ("model", "language_model", "model", "embed_tokens", "weight"),
    ):
        cur = model
        try:
            for p in path:
                cur = getattr(cur, p)
            dev = cur.device
            if str(dev) != "meta":
                return dev
        except Exception:
            continue

    if hasattr(model, "device"):
        dev = model.device
        if str(dev) != "meta":
            return dev

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move_structure_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_structure_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_structure_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_structure_to_device(v, device) for v in obj)
    return obj


def _resolve_video_path(video_id: str, hf_home_override: str | None) -> str:
    if hf_home_override:
        hf_home = Path(hf_home_override)
    else:
        hf_home = Path(os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface/")))
    base_dir = hf_home / "videomme" / "data"
    for suffix in (".mp4", ".MP4", ".mkv"):
        candidate = base_dir / f"{video_id}{suffix}"
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"missing video for videoID={video_id} under {base_dir}")


def _resolve_sample_video_path(sample: dict[str, Any], hf_home_override: str | None) -> str:
    for key in ("video_path", "video", "path", "video_file"):
        value = sample.get(key)
        if value:
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                path = Path.cwd() / path
            if path.exists():
                return str(path)
            raise FileNotFoundError(f"missing video path from sample[{key!r}]: {path}")
    if "videoID" not in sample:
        raise KeyError("sample must contain either video_path/video/path/video_file or videoID")
    return _resolve_video_path(str(sample["videoID"]), hf_home_override)


def _load_dataset(
    dataset_jsonl: str,
    limit: int | None,
    shuffle: bool,
    start_index: int = 0,
    duration_filter: str = "",
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(dataset_jsonl).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    allowed_durations = {x.strip().lower() for x in str(duration_filter or "").split(",") if x.strip()}
    if allowed_durations:
        records = [r for r in records if str(r.get("duration", "")).strip().lower() in allowed_durations]
    if shuffle:
        random.shuffle(records)
    start_index = max(0, int(start_index or 0))
    if start_index > 0:
        records = records[start_index:]
    if limit is not None:
        records = records[: max(0, int(limit))]
    return records


def _load_llava_model(args: BenchmarkArgs):
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path

    model_name = get_model_name_from_path(args.model_path)
    normalized_model_path = str(args.model_path).replace("\\", "/")
    is_llava_video_qwen2 = (
        normalized_model_path == "lmms-lab/LLaVA-Video-7B-Qwen2"
        or "LLaVA-Video-7B-Qwen2" in normalized_model_path
        or "models--lmms-lab--LLaVA-Video-7B-Qwen2" in normalized_model_path
    )
    if is_llava_video_qwen2:
        model_name = "LLaVA-Video-7B-Qwen2"
    overwrite_config = (
        {"mm_spatial_pool_mode": "average", "mm_newline_position": "frame"}
        if is_llava_video_qwen2
        else {}
    )
    attn_impl = _resolve_attn_implementation(args.attn_implementation)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        None,
        model_name,
        device_map="auto",
        attn_implementation=attn_impl,
        overwrite_config=overwrite_config,
        multimodal=True,
    )
    model.eval()
    _assert_flash_attn_placement(model, attn_impl)
    return {"model": model, "tokenizer": tokenizer, "image_processor": image_processor}


def _load_qwen_model(args: BenchmarkArgs, backend: str):
    from transformers import AutoProcessor

    if backend == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration as QwenModel
    elif backend == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration as QwenModel
    else:
        raise ValueError(f"unsupported qwen backend: {backend}")

    offline_env = str(os.getenv("HF_HUB_OFFLINE", "")).strip() == "1" or str(os.getenv("TRANSFORMERS_OFFLINE", "")).strip() == "1"
    local_path = Path(args.model_path).exists()
    local_files_only = bool(args.local_files_only or offline_env or local_path)

    attn_impl = _resolve_attn_implementation(args.attn_implementation)
    model = QwenModel.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation=attn_impl,
        local_files_only=local_files_only,
    )
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        local_files_only=local_files_only,
    )
    model.eval()
    _assert_flash_attn_placement(model, attn_impl)
    return {"model": model, "processor": processor}


def _load_backend_model(args: BenchmarkArgs):
    backend = _infer_backend(args.model_backend, args.model_path)
    if backend == "llava":
        loaded = _load_llava_model(args)
    elif backend in ("qwen2_5_vl", "qwen3_vl"):
        loaded = _load_qwen_model(args, backend)
    else:
        raise ValueError(f"unsupported backend: {backend}")
    loaded["backend"] = backend
    return loaded


def _prepare_llava_inputs(model_bundle, args: BenchmarkArgs, prompt_text: str, video_path: str):
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token

    tokenizer = model_bundle["tokenizer"]
    model = model_bundle["model"]
    target_device = _resolve_generation_device(model)
    image_processor = model_bundle["image_processor"]
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    frame_idx = np.linspace(0, total_frames - 1, args.num_frames, dtype=int)
    video_frames = vr.get_batch(frame_idx.tolist()).asnumpy()
    frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"]
    if target_device.type == "cuda":
        frames = frames.half()
    frames = frames.to(target_device)

    conv = copy.deepcopy(conv_templates["qwen_1_5"])
    conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{prompt_text}")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(target_device)
    )
    attention_mask = torch.ones_like(input_ids)
    image_sizes = [frame.size for frame in video_frames]

    raw_visual_tokens = int((input_ids[0] == IMAGE_TOKEN_INDEX).sum().item())
    return {
        "raw_visual_tokens": raw_visual_tokens,
        "prompt_len": int(input_ids.shape[1]),
        "inputs": {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images": [frames],
            "image_sizes": image_sizes,
        },
    }


def _prepare_qwen_inputs(model_bundle, args: BenchmarkArgs, prompt_text: str, video_path: str):
    processor = model_bundle["processor"]
    model = model_bundle["model"]
    backend = model_bundle.get("backend", "")
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise ImportError("qwen_vl_utils is required for Qwen video processing") from exc

    video_content: dict[str, Any] = {
        "type": "video",
        "video": video_path,
        "max_pixels": args.max_pixels,
        "min_pixels": args.min_pixels,
    }
    # lmms-eval/Qwen2.5-VL does not pass nframes into qwen_vl_utils. It decodes
    # the video first, then uniformly samples max_num_frames afterwards.
    if backend != "qwen2_5_vl":
        video_content["nframes"] = args.num_frames

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [video_content, {"type": "text", "text": prompt_text}],
        },
    ]

    video_kwargs: dict[str, Any] = {}
    video_metadata = None
    if backend == "qwen2_5_vl":
        images, videos = process_vision_info([messages])
        if videos is not None:
            if torch.is_tensor(videos):
                videos = [videos]
            elif isinstance(videos, tuple):
                videos = list(videos)
            if len(videos) > 0:
                total_frames = int(videos[0].shape[0])
                if total_frames > 0 and args.num_frames > 0:
                    indices = np.linspace(0, total_frames - 1, args.num_frames, dtype=int)
                    indices = np.unique(indices)
                    if total_frames - 1 not in indices:
                        indices = np.append(indices, total_frames - 1)
                        indices = np.unique(indices)
                    videos[0] = videos[0][indices]
    else:
        try:
            # Qwen3-VL prefers explicit video metadata for timestamp-aware prompts.
            if backend == "qwen3_vl":
                images, videos, video_kwargs = process_vision_info(
                    messages,
                    return_video_kwargs=True,
                    return_video_metadata=True,
                    image_patch_size=16,
                )
            else:
                images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        except TypeError:
            # Backward compatibility for older qwen_vl_utils versions.
            images, videos, video_kwargs = process_vision_info(
                messages,
                return_video_kwargs=True,
            )

    # qwen_vl_utils may return [(video_tensor, metadata), ...] when return_video_metadata=True.
    if videos is not None and len(videos) > 0 and isinstance(videos[0], tuple):
        videos, video_metadata = zip(*videos)
        videos = list(videos)
        video_metadata = list(video_metadata)

    template_messages = [messages] if backend == "qwen2_5_vl" else messages
    text = processor.apply_chat_template(template_messages, tokenize=False, add_generation_prompt=True)
    processor_kwargs = {
        "text": text,
        "images": images,
        "videos": videos,
        "padding": True,
        "return_tensors": "pt",
        **video_kwargs,
    }
    if video_metadata is not None:
        processor_kwargs["video_metadata"] = video_metadata

    inputs = processor(**processor_kwargs)
    target_device = _resolve_generation_device(model)
    inputs = _move_structure_to_device(inputs, target_device)

    video_token_id = getattr(model.config, "video_token_id", None)
    if video_token_id is None:
        raw_visual_tokens = 0
    else:
        raw_visual_tokens = int((inputs["input_ids"][0] == video_token_id).sum().item())

    return {
        "raw_visual_tokens": raw_visual_tokens,
        "prompt_len": int(inputs["input_ids"].shape[1]),
        "inputs": inputs,
    }


def _prepare_inputs(model_bundle, args: BenchmarkArgs, sample: dict[str, Any]):
    video_path = _resolve_sample_video_path(sample, args.hf_home)
    backend = model_bundle["backend"]
    prompt_text = _to_lmms_videomme_prompt(sample["input"], backend)
    if backend == "llava":
        return _prepare_llava_inputs(model_bundle, args, prompt_text, video_path)
    return _prepare_qwen_inputs(model_bundle, args, prompt_text, video_path)


def _clone_inputs(backend: str, prepared_inputs):
    if backend == "llava":
        src = prepared_inputs["inputs"]
        return {
            "input_ids": src["input_ids"].clone(),
            "attention_mask": src["attention_mask"].clone(),
            "images": [img.clone() for img in src["images"]],
            "image_sizes": src["image_sizes"],
        }

    src = prepared_inputs["inputs"]
    cloned = {}
    for k, v in src.items():
        if torch.is_tensor(v):
            cloned[k] = v.clone()
        else:
            cloned[k] = v
    return cloned


def _decode_prediction(model_bundle, output_ids: torch.Tensor, prompt_len: int) -> tuple[str, int]:
    backend = model_bundle["backend"]
    if backend == "llava":
        # LLaVA-Video generation is driven by inputs_embeds. On recent
        # transformers versions, generate() can return only newly generated
        # token ids rather than prompt + generation. Do not blindly slice by
        # prompt_len in that case, or every prediction becomes empty.
        generated = output_ids[:, prompt_len:] if output_ids.shape[1] > prompt_len else output_ids
        gen_tokens = int(generated.shape[1])
        if gen_tokens == 0:
            return "", 0
        tokenizer = model_bundle["tokenizer"]
        text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        answer = _extract_choice_letter(text)
        if answer:
            return answer, gen_tokens
        first_token_id = int(generated[0, 0].item())
        first_token = tokenizer.decode([first_token_id], skip_special_tokens=True)
        return _extract_choice_letter(first_token), gen_tokens

    generated = output_ids[:, prompt_len:] if output_ids.shape[1] > prompt_len else output_ids[:, :0]
    gen_tokens = int(generated.shape[1])
    if gen_tokens == 0:
        return "", 0

    processor = model_bundle["processor"]
    text = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    answer = _extract_choice_letter(text)
    if answer:
        return answer, gen_tokens
    return _extract_choice_letter(text[:8]), gen_tokens


def _safe_int_metric(value, fallback: int) -> int:
    if value is None:
        return fallback
    try:
        return int(value)
    except Exception:
        return fallback


def _get_raw_visual_token_metric(model, fallback: int, use_acceleration: bool) -> int:
    if not use_acceleration or not hasattr(model, "flashvid_config"):
        return fallback
    cfg = getattr(model, "flashvid_config")
    return _safe_int_metric(
        getattr(cfg, "raw_visual_token_length", None),
        _safe_int_metric(getattr(cfg, "raw_vision_token_length", None), fallback),
    )


def _get_visual_token_metrics(model, raw_visual_tokens: int, use_acceleration: bool) -> tuple[int, int]:
    if not use_acceleration or not hasattr(model, "flashvid_config"):
        return raw_visual_tokens, raw_visual_tokens
    cfg = getattr(model, "flashvid_config")
    vision_length = _safe_int_metric(
        getattr(cfg, "vision_token_length", None),
        _safe_int_metric(getattr(cfg, "visual_token_length", None), raw_visual_tokens),
    )
    final_length = _safe_int_metric(
        getattr(cfg, "llm_token_length", None),
        _safe_int_metric(getattr(cfg, "visual_token_length", None), vision_length),
    )
    return final_length, vision_length


def _get_talon_debug_metrics(model) -> dict[str, float | None]:
    if not hasattr(model, "flashvid_config"):
        return {
            "talon_target_tokens_per_frame": None,
            "talon_adaptive_retention_ratio": None,
            "talon_complexity_score": None,
            "talon_target_budget": None,
            "talon_anchor_tokens": None,
            "talon_rank_tokens": None,
            "talon_event_tokens": None,
            "talon_recall_tokens": None,
            "talon_persistence_tokens": None,
            "talon_object_tokens": None,
            "talon_memory_tokens": None,
            "talon_rank_cap": None,
            "talon_chosen_rank": None,
            "talon_duplicate_index_count": None,
            "talon_question_aware_active": None,
            "talon_router_mode_code": None,
            "talon_router_fused_concentration": None,
            "talon_router_residual_concentration": None,
            "talon_router_question_concentration": None,
            "talon_router_frame_entropy": None,
        }
    cfg = getattr(model, "flashvid_config")
    target = getattr(cfg, "last_talon_target_tokens_per_frame", None)
    adaptive_ratio = getattr(cfg, "last_adaptive_retention_ratio", None)
    complexity = getattr(cfg, "last_talon_complexity_score", None)
    target_budget = getattr(cfg, "last_talon_target_budget", None)
    anchor_tokens = getattr(cfg, "last_talon_anchor_tokens", None)
    rank_tokens = getattr(cfg, "last_talon_rank_tokens", None)
    event_tokens = getattr(cfg, "last_talon_event_tokens", None)
    recall_tokens = getattr(cfg, "last_talon_recall_tokens", None)
    persistence_tokens = getattr(cfg, "last_talon_persistence_tokens", None)
    object_tokens = getattr(cfg, "last_talon_object_tokens", None)
    memory_tokens = getattr(cfg, "last_talon_memory_tokens", None)
    rank_cap = getattr(cfg, "last_talon_rank_cap", None)
    chosen_rank = getattr(cfg, "last_talon_chosen_rank", None)
    duplicate_count = getattr(cfg, "last_talon_duplicate_index_count", None)
    question_active = getattr(cfg, "last_talon_question_aware_active", None)
    router_mode = getattr(cfg, "last_talon_router_mode_code", None)
    router_fused_conc = getattr(cfg, "last_talon_router_fused_concentration", None)
    router_residual_conc = getattr(cfg, "last_talon_router_residual_concentration", None)
    router_question_conc = getattr(cfg, "last_talon_router_question_concentration", None)
    router_frame_entropy = getattr(cfg, "last_talon_router_frame_entropy", None)
    return {
        "talon_target_tokens_per_frame": float(target) if target is not None else None,
        "talon_adaptive_retention_ratio": float(adaptive_ratio) if adaptive_ratio is not None else None,
        "talon_complexity_score": float(complexity) if complexity is not None else None,
        "talon_target_budget": float(target_budget) if target_budget is not None else None,
        "talon_anchor_tokens": float(anchor_tokens) if anchor_tokens is not None else None,
        "talon_rank_tokens": float(rank_tokens) if rank_tokens is not None else None,
        "talon_event_tokens": float(event_tokens) if event_tokens is not None else None,
        "talon_recall_tokens": float(recall_tokens) if recall_tokens is not None else None,
        "talon_persistence_tokens": float(persistence_tokens) if persistence_tokens is not None else None,
        "talon_object_tokens": float(object_tokens) if object_tokens is not None else None,
        "talon_memory_tokens": float(memory_tokens) if memory_tokens is not None else None,
        "talon_rank_cap": float(rank_cap) if rank_cap is not None else None,
        "talon_chosen_rank": float(chosen_rank) if chosen_rank is not None else None,
        "talon_duplicate_index_count": float(duplicate_count) if duplicate_count is not None else None,
        "talon_question_aware_active": float(bool(question_active)) if question_active is not None else None,
        "talon_router_mode_code": float(router_mode) if router_mode is not None else None,
        "talon_router_fused_concentration": float(router_fused_conc) if router_fused_conc is not None else None,
        "talon_router_residual_concentration": float(router_residual_conc) if router_residual_conc is not None else None,
        "talon_router_question_concentration": float(router_question_conc) if router_question_conc is not None else None,
        "talon_router_frame_entropy": float(router_frame_entropy) if router_frame_entropy is not None else None,
    }


def _get_talon_core_debug_metrics(model) -> dict[str, float | None]:
    empty = {
        "talon_core_target_budget": None,
        "talon_core_residual_mean": None,
        "talon_core_semantic_tokens": None,
        "talon_core_innovation_tokens": None,
        "talon_core_duplicate_index_count": None,
        "talon_core_question_aware_active": None,
        "talon_core_budget_min": None,
        "talon_core_budget_max": None,
        "talon_core_grid_h": None,
        "talon_core_grid_w": None,
    }
    if not hasattr(model, "flashvid_config"):
        return empty
    cfg = getattr(model, "flashvid_config")
    target_budget = getattr(cfg, "last_talon_core_target_budget", None)
    residual_mean = getattr(cfg, "last_talon_core_residual_mean", None)
    semantic_tokens = getattr(cfg, "last_talon_core_semantic_tokens", None)
    innovation_tokens = getattr(cfg, "last_talon_core_innovation_tokens", None)
    duplicate_count = getattr(cfg, "last_talon_core_duplicate_index_count", None)
    question_active = getattr(cfg, "last_talon_core_question_aware_active", None)
    budget_min = getattr(cfg, "last_talon_core_budget_min", None)
    budget_max = getattr(cfg, "last_talon_core_budget_max", None)
    grid_h = getattr(cfg, "last_talon_core_grid_h", None)
    grid_w = getattr(cfg, "last_talon_core_grid_w", None)
    return {
        "talon_core_target_budget": float(target_budget) if target_budget is not None else None,
        "talon_core_residual_mean": float(residual_mean) if residual_mean is not None else None,
        "talon_core_semantic_tokens": float(semantic_tokens) if semantic_tokens is not None else None,
        "talon_core_innovation_tokens": float(innovation_tokens) if innovation_tokens is not None else None,
        "talon_core_duplicate_index_count": float(duplicate_count) if duplicate_count is not None else None,
        "talon_core_question_aware_active": float(bool(question_active)) if question_active is not None else None,
        "talon_core_budget_min": float(budget_min) if budget_min is not None else None,
        "talon_core_budget_max": float(budget_max) if budget_max is not None else None,
        "talon_core_grid_h": float(grid_h) if grid_h is not None else None,
        "talon_core_grid_w": float(grid_w) if grid_w is not None else None,
    }


def _get_graft_debug_metrics(model) -> dict[str, float | None]:
    empty = {key: None for key in GRAFT_METRIC_KEYS}
    if not hasattr(model, "flashvid_config"):
        return empty
    cfg = getattr(model, "flashvid_config")
    values: dict[str, float | None] = {}
    for key in GRAFT_METRIC_KEYS:
        value = getattr(cfg, f"last_{key}", None)
        values[key] = float(value) if value is not None else None
    return values


def _get_cats_debug_metrics(model) -> dict[str, float | None]:
    empty = {key: None for key in CATS_METRIC_KEYS}
    candidates = []
    for obj in (
        model,
        getattr(model, "model", None),
        getattr(model, "language_model", None),
        getattr(model, "module", None),
        getattr(getattr(model, "module", None), "model", None),
    ):
        if obj is None:
            continue
        cfg = getattr(obj, "flashvid_config", None)
        if cfg is not None and cfg not in candidates:
            candidates.append(cfg)
    if not candidates:
        return empty
    best_values = empty
    best_score = -1
    for cfg in candidates:
        values: dict[str, float | None] = {}
        present = 0
        for key in CATS_METRIC_KEYS:
            value = getattr(cfg, f"last_{key}", None)
            values[key] = float(value) if value is not None else None
            present += int(value is not None)
        if present > best_score:
            best_score = present
            best_values = values
    return best_values


def _get_dyn_debug_metrics(model) -> dict[str, float | None]:
    empty = {key: None for key in DYN_METRIC_KEYS}
    candidates = []
    for obj in (
        model,
        getattr(model, "model", None),
        getattr(model, "language_model", None),
        getattr(model, "module", None),
        getattr(getattr(model, "module", None), "model", None),
    ):
        if obj is None:
            continue
        cfg = getattr(obj, "flashvid_config", None)
        if cfg is not None and cfg not in candidates:
            candidates.append(cfg)
    if not candidates:
        return empty
    best_values = empty
    best_score = -1
    for cfg in candidates:
        values: dict[str, float | None] = {}
        present = 0
        for key in DYN_METRIC_KEYS:
            value = getattr(cfg, f"last_{key}", None)
            values[key] = float(value) if value is not None else None
            present += int(value is not None)
        if present > best_score:
            best_score = present
            best_values = values
    return best_values


def _get_hedge_debug_metrics(model) -> dict[str, float | None]:
    empty = {key: None for key in HEDGE_METRIC_KEYS}
    candidates = []
    for obj in (
        model,
        getattr(model, "model", None),
        getattr(model, "language_model", None),
        getattr(model, "module", None),
        getattr(getattr(model, "module", None), "model", None),
    ):
        if obj is None:
            continue
        cfg = getattr(obj, "flashvid_config", None)
        if cfg is not None and cfg not in candidates:
            candidates.append(cfg)
    if not candidates:
        return empty
    best_values = empty
    best_score = -1
    for cfg in candidates:
        values: dict[str, float | None] = {}
        present = 0
        for key in HEDGE_METRIC_KEYS:
            value = getattr(cfg, f"last_{key}", None)
            values[key] = float(value) if value is not None else None
            present += int(value is not None)
        if present > best_score:
            best_score = present
            best_values = values
    return best_values


def _get_learn_debug_metrics(model) -> dict[str, float | None]:
    empty = {key: None for key in LEARN_METRIC_KEYS}
    candidates = []
    for obj in (
        model,
        getattr(model, "model", None),
        getattr(model, "language_model", None),
        getattr(model, "module", None),
        getattr(getattr(model, "module", None), "model", None),
    ):
        if obj is None:
            continue
        cfg = getattr(obj, "flashvid_config", None)
        if cfg is not None and cfg not in candidates:
            candidates.append(cfg)
    if not candidates:
        return empty
    best_values = empty
    best_score = -1
    for cfg in candidates:
        values: dict[str, float | None] = {}
        present = 0
        for key in LEARN_METRIC_KEYS:
            value = getattr(cfg, f"last_{key}", None)
            values[key] = float(value) if value is not None else None
            present += int(value is not None)
        if present > best_score:
            best_score = present
            best_values = values
    return best_values


def _run_benchmark_once(model_bundle, args: BenchmarkArgs, prepared_inputs, use_acceleration: bool):
    model = model_bundle["model"]
    backend = model_bundle["backend"]

    # Warmup
    for _ in range(args.num_warmup):
        inputs = _clone_inputs(backend, prepared_inputs)
        if backend != "llava":
            inputs = _move_structure_to_device(inputs, _resolve_generation_device(model))
        if backend == "llava":
            model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                images=inputs["images"],
                image_sizes=inputs["image_sizes"],
                do_sample=False,
                top_p=1.0,
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
                modalities=["video"],
            )
        else:
            model.generate(
                **inputs,
                do_sample=False,
                top_p=1.0,
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    latencies = []
    pred_answer = ""
    gen_tokens_per_run = []
    compressed_tokens_per_run = []
    vision_tokens_per_run = []
    talon_target_per_run = []
    talon_complexity_per_run = []
    talon_target_budget_per_run = []
    talon_anchor_per_run = []
    talon_rank_per_run = []
    talon_event_per_run = []
    talon_recall_per_run = []
    talon_persistence_per_run = []
    talon_object_per_run = []
    talon_memory_per_run = []
    talon_rank_cap_per_run = []
    talon_chosen_rank_per_run = []
    talon_duplicate_per_run = []
    talon_question_active_per_run = []
    talon_router_mode_per_run = []
    talon_router_fused_conc_per_run = []
    talon_router_residual_conc_per_run = []
    talon_router_question_conc_per_run = []
    talon_router_frame_entropy_per_run = []
    talon_core_target_budget_per_run = []
    talon_core_residual_mean_per_run = []
    talon_core_semantic_per_run = []
    talon_core_innovation_per_run = []
    talon_core_duplicate_per_run = []
    talon_core_question_active_per_run = []
    talon_core_budget_min_per_run = []
    talon_core_budget_max_per_run = []
    talon_core_grid_h_per_run = []
    talon_core_grid_w_per_run = []
    graft_metrics_per_run = {key: [] for key in GRAFT_METRIC_KEYS}
    cats_metrics_per_run = {key: [] for key in CATS_METRIC_KEYS}
    dyn_metrics_per_run = {key: [] for key in DYN_METRIC_KEYS}
    hedge_metrics_per_run = {key: [] for key in HEDGE_METRIC_KEYS}
    learn_metrics_per_run = {key: [] for key in LEARN_METRIC_KEYS}
    prompt_len = prepared_inputs["prompt_len"]
    raw_visual_tokens = int(prepared_inputs["raw_visual_tokens"])

    for run_idx in range(args.num_runs):
        inputs = _clone_inputs(backend, prepared_inputs)
        if backend != "llava":
            inputs = _move_structure_to_device(inputs, _resolve_generation_device(model))

        def _generate():
            if backend == "llava":
                return model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    images=inputs["images"],
                    image_sizes=inputs["image_sizes"],
                    do_sample=False,
                    top_p=1.0,
                    num_beams=1,
                    max_new_tokens=args.max_new_tokens,
                    modalities=["video"],
                )
            return model.generate(
                **inputs,
                do_sample=False,
                top_p=1.0,
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )

        output_ids, latency_ms = _timed_call(_generate)
        answer, gen_tokens = _decode_prediction(model_bundle, output_ids, prompt_len)
        if run_idx == 0:
            pred_answer = answer
        latencies.append(float(latency_ms))
        gen_tokens_per_run.append(float(gen_tokens))
        final_tokens, vision_tokens = _get_visual_token_metrics(model, raw_visual_tokens, use_acceleration)
        debug_metrics = _get_talon_debug_metrics(model)
        talon_core_metrics = _get_talon_core_debug_metrics(model)
        graft_metrics = _get_graft_debug_metrics(model)
        cats_metrics = _get_cats_debug_metrics(model)
        dyn_metrics = _get_dyn_debug_metrics(model)
        hedge_metrics = _get_hedge_debug_metrics(model)
        learn_metrics = _get_learn_debug_metrics(model)
        compressed_tokens_per_run.append(float(final_tokens))
        vision_tokens_per_run.append(float(vision_tokens))
        for key in GRAFT_METRIC_KEYS:
            if graft_metrics.get(key) is not None:
                graft_metrics_per_run[key].append(float(graft_metrics[key]))
        for key in CATS_METRIC_KEYS:
            if cats_metrics.get(key) is not None:
                cats_metrics_per_run[key].append(float(cats_metrics[key]))
        for key in DYN_METRIC_KEYS:
            if dyn_metrics.get(key) is not None:
                dyn_metrics_per_run[key].append(float(dyn_metrics[key]))
        for key in HEDGE_METRIC_KEYS:
            if hedge_metrics.get(key) is not None:
                hedge_metrics_per_run[key].append(float(hedge_metrics[key]))
        for key in LEARN_METRIC_KEYS:
            if learn_metrics.get(key) is not None:
                learn_metrics_per_run[key].append(float(learn_metrics[key]))
        if debug_metrics["talon_target_tokens_per_frame"] is not None:
            talon_target_per_run.append(float(debug_metrics["talon_target_tokens_per_frame"]))
        if debug_metrics["talon_complexity_score"] is not None:
            talon_complexity_per_run.append(float(debug_metrics["talon_complexity_score"]))
        if debug_metrics["talon_target_budget"] is not None:
            talon_target_budget_per_run.append(float(debug_metrics["talon_target_budget"]))
        if debug_metrics["talon_anchor_tokens"] is not None:
            talon_anchor_per_run.append(float(debug_metrics["talon_anchor_tokens"]))
        if debug_metrics["talon_rank_tokens"] is not None:
            talon_rank_per_run.append(float(debug_metrics["talon_rank_tokens"]))
        if debug_metrics["talon_event_tokens"] is not None:
            talon_event_per_run.append(float(debug_metrics["talon_event_tokens"]))
        if debug_metrics["talon_recall_tokens"] is not None:
            talon_recall_per_run.append(float(debug_metrics["talon_recall_tokens"]))
        if debug_metrics["talon_persistence_tokens"] is not None:
            talon_persistence_per_run.append(float(debug_metrics["talon_persistence_tokens"]))
        if debug_metrics["talon_object_tokens"] is not None:
            talon_object_per_run.append(float(debug_metrics["talon_object_tokens"]))
        if debug_metrics["talon_memory_tokens"] is not None:
            talon_memory_per_run.append(float(debug_metrics["talon_memory_tokens"]))
        if debug_metrics["talon_rank_cap"] is not None:
            talon_rank_cap_per_run.append(float(debug_metrics["talon_rank_cap"]))
        if debug_metrics["talon_chosen_rank"] is not None:
            talon_chosen_rank_per_run.append(float(debug_metrics["talon_chosen_rank"]))
        if debug_metrics["talon_duplicate_index_count"] is not None:
            talon_duplicate_per_run.append(float(debug_metrics["talon_duplicate_index_count"]))
        if debug_metrics["talon_question_aware_active"] is not None:
            talon_question_active_per_run.append(float(debug_metrics["talon_question_aware_active"]))
        if debug_metrics["talon_router_mode_code"] is not None:
            talon_router_mode_per_run.append(float(debug_metrics["talon_router_mode_code"]))
        if debug_metrics["talon_router_fused_concentration"] is not None:
            talon_router_fused_conc_per_run.append(float(debug_metrics["talon_router_fused_concentration"]))
        if debug_metrics["talon_router_residual_concentration"] is not None:
            talon_router_residual_conc_per_run.append(float(debug_metrics["talon_router_residual_concentration"]))
        if debug_metrics["talon_router_question_concentration"] is not None:
            talon_router_question_conc_per_run.append(float(debug_metrics["talon_router_question_concentration"]))
        if debug_metrics["talon_router_frame_entropy"] is not None:
            talon_router_frame_entropy_per_run.append(float(debug_metrics["talon_router_frame_entropy"]))
        if talon_core_metrics["talon_core_target_budget"] is not None:
            talon_core_target_budget_per_run.append(float(talon_core_metrics["talon_core_target_budget"]))
        if talon_core_metrics["talon_core_residual_mean"] is not None:
            talon_core_residual_mean_per_run.append(float(talon_core_metrics["talon_core_residual_mean"]))
        if talon_core_metrics["talon_core_semantic_tokens"] is not None:
            talon_core_semantic_per_run.append(float(talon_core_metrics["talon_core_semantic_tokens"]))
        if talon_core_metrics["talon_core_innovation_tokens"] is not None:
            talon_core_innovation_per_run.append(float(talon_core_metrics["talon_core_innovation_tokens"]))
        if talon_core_metrics["talon_core_duplicate_index_count"] is not None:
            talon_core_duplicate_per_run.append(float(talon_core_metrics["talon_core_duplicate_index_count"]))
        if talon_core_metrics["talon_core_question_aware_active"] is not None:
            talon_core_question_active_per_run.append(float(talon_core_metrics["talon_core_question_aware_active"]))
        if talon_core_metrics["talon_core_budget_min"] is not None:
            talon_core_budget_min_per_run.append(float(talon_core_metrics["talon_core_budget_min"]))
        if talon_core_metrics["talon_core_budget_max"] is not None:
            talon_core_budget_max_per_run.append(float(talon_core_metrics["talon_core_budget_max"]))
        if talon_core_metrics["talon_core_grid_h"] is not None:
            talon_core_grid_h_per_run.append(float(talon_core_metrics["talon_core_grid_h"]))
        if talon_core_metrics["talon_core_grid_w"] is not None:
            talon_core_grid_w_per_run.append(float(talon_core_metrics["talon_core_grid_w"]))

    latency_ms = float(np.mean(latencies)) if latencies else None
    generated_tokens = float(np.mean(gen_tokens_per_run)) if gen_tokens_per_run else None
    raw_visual_tokens = _get_raw_visual_token_metric(model, raw_visual_tokens, use_acceleration)
    compressed_visual_tokens = float(np.mean(compressed_tokens_per_run)) if compressed_tokens_per_run else float(raw_visual_tokens)
    vision_compressed_visual_tokens = float(np.mean(vision_tokens_per_run)) if vision_tokens_per_run else float(raw_visual_tokens)
    talon_target_tokens_per_frame = float(np.mean(talon_target_per_run)) if talon_target_per_run else None
    talon_complexity_score = float(np.mean(talon_complexity_per_run)) if talon_complexity_per_run else None
    talon_target_budget = float(np.mean(talon_target_budget_per_run)) if talon_target_budget_per_run else None
    talon_anchor_tokens = float(np.mean(talon_anchor_per_run)) if talon_anchor_per_run else None
    talon_rank_tokens = float(np.mean(talon_rank_per_run)) if talon_rank_per_run else None
    talon_event_tokens = float(np.mean(talon_event_per_run)) if talon_event_per_run else None
    talon_recall_tokens = float(np.mean(talon_recall_per_run)) if talon_recall_per_run else None
    talon_persistence_tokens = float(np.mean(talon_persistence_per_run)) if talon_persistence_per_run else None
    talon_object_tokens = float(np.mean(talon_object_per_run)) if talon_object_per_run else None
    talon_memory_tokens = float(np.mean(talon_memory_per_run)) if talon_memory_per_run else None
    talon_rank_cap = float(np.mean(talon_rank_cap_per_run)) if talon_rank_cap_per_run else None
    talon_chosen_rank = float(np.mean(talon_chosen_rank_per_run)) if talon_chosen_rank_per_run else None
    talon_duplicate_index_count = float(np.mean(talon_duplicate_per_run)) if talon_duplicate_per_run else None
    talon_question_aware_active = float(np.mean(talon_question_active_per_run)) if talon_question_active_per_run else None
    talon_router_mode_code = float(np.mean(talon_router_mode_per_run)) if talon_router_mode_per_run else None
    talon_router_fused_concentration = float(np.mean(talon_router_fused_conc_per_run)) if talon_router_fused_conc_per_run else None
    talon_router_residual_concentration = float(np.mean(talon_router_residual_conc_per_run)) if talon_router_residual_conc_per_run else None
    talon_router_question_concentration = float(np.mean(talon_router_question_conc_per_run)) if talon_router_question_conc_per_run else None
    talon_router_frame_entropy = float(np.mean(talon_router_frame_entropy_per_run)) if talon_router_frame_entropy_per_run else None
    talon_core_target_budget = float(np.mean(talon_core_target_budget_per_run)) if talon_core_target_budget_per_run else None
    talon_core_residual_mean = float(np.mean(talon_core_residual_mean_per_run)) if talon_core_residual_mean_per_run else None
    talon_core_semantic_tokens = float(np.mean(talon_core_semantic_per_run)) if talon_core_semantic_per_run else None
    talon_core_innovation_tokens = float(np.mean(talon_core_innovation_per_run)) if talon_core_innovation_per_run else None
    talon_core_duplicate_index_count = float(np.mean(talon_core_duplicate_per_run)) if talon_core_duplicate_per_run else None
    talon_core_question_aware_active = float(np.mean(talon_core_question_active_per_run)) if talon_core_question_active_per_run else None
    talon_core_budget_min = float(np.mean(talon_core_budget_min_per_run)) if talon_core_budget_min_per_run else None
    talon_core_budget_max = float(np.mean(talon_core_budget_max_per_run)) if talon_core_budget_max_per_run else None
    talon_core_grid_h = float(np.mean(talon_core_grid_h_per_run)) if talon_core_grid_h_per_run else None
    talon_core_grid_w = float(np.mean(talon_core_grid_w_per_run)) if talon_core_grid_w_per_run else None
    graft_metric_means = {
        key: float(np.mean(values)) if values else None
        for key, values in graft_metrics_per_run.items()
    }
    cats_metric_means = {
        key: float(np.mean(values)) if values else None
        for key, values in cats_metrics_per_run.items()
    }
    dyn_metric_means = {
        key: float(np.mean(values)) if values else None
        for key, values in dyn_metrics_per_run.items()
    }
    hedge_metric_means = {
        key: float(np.mean(values)) if values else None
        for key, values in hedge_metrics_per_run.items()
    }
    learn_metric_means = {
        key: float(np.mean(values)) if values else None
        for key, values in learn_metrics_per_run.items()
    }
    tps = None
    if latency_ms and latency_ms > 0 and generated_tokens is not None:
        tps = float(generated_tokens / (latency_ms / 1000.0))

    return {
        "pred_answer": pred_answer,
        "latency_ms": latency_ms,
        "generated_tokens": generated_tokens,
        "tokens_per_second": tps,
        "raw_visual_tokens": float(raw_visual_tokens),
        "compressed_visual_tokens": compressed_visual_tokens,
        "vision_compressed_visual_tokens": vision_compressed_visual_tokens,
        "talon_target_tokens_per_frame": talon_target_tokens_per_frame,
        "talon_complexity_score": talon_complexity_score,
        "talon_target_budget": talon_target_budget,
        "talon_anchor_tokens": talon_anchor_tokens,
        "talon_rank_tokens": talon_rank_tokens,
        "talon_event_tokens": talon_event_tokens,
        "talon_recall_tokens": talon_recall_tokens,
        "talon_persistence_tokens": talon_persistence_tokens,
        "talon_object_tokens": talon_object_tokens,
        "talon_memory_tokens": talon_memory_tokens,
        "talon_rank_cap": talon_rank_cap,
        "talon_chosen_rank": talon_chosen_rank,
        "talon_duplicate_index_count": talon_duplicate_index_count,
        "talon_question_aware_active": talon_question_aware_active,
        "talon_router_mode_code": talon_router_mode_code,
        "talon_router_fused_concentration": talon_router_fused_concentration,
        "talon_router_residual_concentration": talon_router_residual_concentration,
        "talon_router_question_concentration": talon_router_question_concentration,
        "talon_router_frame_entropy": talon_router_frame_entropy,
        "talon_core_target_budget": talon_core_target_budget,
        "talon_core_residual_mean": talon_core_residual_mean,
        "talon_core_semantic_tokens": talon_core_semantic_tokens,
        "talon_core_innovation_tokens": talon_core_innovation_tokens,
        "talon_core_duplicate_index_count": talon_core_duplicate_index_count,
        "talon_core_question_aware_active": talon_core_question_aware_active,
        "talon_core_budget_min": talon_core_budget_min,
        "talon_core_budget_max": talon_core_budget_max,
        "talon_core_grid_h": talon_core_grid_h,
        "talon_core_grid_w": talon_core_grid_w,
        **graft_metric_means,
        **cats_metric_means,
        **dyn_metric_means,
        **hedge_metric_means,
        **learn_metric_means,
    }


def _benchmark_single_sample(model_bundle, args: BenchmarkArgs, sample: dict[str, Any], use_acceleration: bool):
    record = {
        "question_id": sample.get("question_id"),
        "videoID": sample.get("videoID"),
        "dataset": sample.get("dataset"),
        "split": sample.get("split"),
        "subset": sample.get("subset"),
        "duration": sample.get("duration"),
        "category": sample.get("category"),
        "task_category": sample.get("task_category"),
        "sub_category": sample.get("sub_category"),
        "answer": sample.get("answer"),
        "pred_answer": "",
        "correct": None,
        "latency_ms": None,
        "generated_tokens": None,
        "tokens_per_second": None,
        "raw_visual_tokens": None,
        "compressed_visual_tokens": None,
        "vision_compressed_visual_tokens": None,
        "talon_target_tokens_per_frame": None,
        "talon_complexity_score": None,
        "talon_target_budget": None,
        "talon_anchor_tokens": None,
        "talon_rank_tokens": None,
        "talon_event_tokens": None,
        "talon_recall_tokens": None,
        "talon_persistence_tokens": None,
        "talon_object_tokens": None,
        "talon_memory_tokens": None,
        "talon_rank_cap": None,
        "talon_chosen_rank": None,
        "talon_duplicate_index_count": None,
        "talon_question_aware_active": None,
        "talon_router_mode_code": None,
        "talon_router_fused_concentration": None,
        "talon_router_residual_concentration": None,
        "talon_router_question_concentration": None,
        "talon_router_frame_entropy": None,
        "talon_core_target_budget": None,
        "talon_core_residual_mean": None,
        "talon_core_semantic_tokens": None,
        "talon_core_innovation_tokens": None,
        "talon_core_duplicate_index_count": None,
        "talon_core_question_aware_active": None,
        "visual_token_reduction_ratio": None,
        "vision_visual_token_reduction_ratio": None,
        "error": None,
        "error_traceback": None,
    }
    record.update({key: None for key in GRAFT_METRIC_KEYS})
    record.update({key: None for key in CATS_METRIC_KEYS})
    record.update({key: None for key in DYN_METRIC_KEYS})
    record.update({key: None for key in HEDGE_METRIC_KEYS})
    record.update({key: None for key in LEARN_METRIC_KEYS})

    try:
        if use_acceleration and hasattr(model_bundle.get("model"), "flashvid_config"):
            cfg = getattr(model_bundle["model"], "flashvid_config")
            setattr(cfg, "current_video_duration", sample.get("duration"))
            setattr(cfg, "current_task_category", sample.get("task_category"))
            setattr(cfg, "current_category", sample.get("category"))
        prepared_inputs = _prepare_inputs(model_bundle, args, sample)
        result = _run_benchmark_once(model_bundle, args, prepared_inputs, use_acceleration=use_acceleration)
        raw_v = result["raw_visual_tokens"]
        compressed_v = result["compressed_visual_tokens"]
        vision_compressed_v = result["vision_compressed_visual_tokens"]
        reduction_ratio = None
        vision_reduction_ratio = None
        if raw_v and raw_v > 0:
            reduction_ratio = float(max(0.0, 1.0 - (compressed_v / raw_v)))
            vision_reduction_ratio = float(max(0.0, 1.0 - (vision_compressed_v / raw_v)))

        record.update(
            {
                "pred_answer": result["pred_answer"],
                "correct": str(result["pred_answer"]).lower() == str(sample.get("answer")).lower(),
                "latency_ms": result["latency_ms"],
                "generated_tokens": result["generated_tokens"],
                "tokens_per_second": result["tokens_per_second"],
                "raw_visual_tokens": raw_v,
                "compressed_visual_tokens": compressed_v,
                "vision_compressed_visual_tokens": vision_compressed_v,
                "talon_target_tokens_per_frame": result.get("talon_target_tokens_per_frame"),
                "talon_complexity_score": result.get("talon_complexity_score"),
                "talon_target_budget": result.get("talon_target_budget"),
                "talon_anchor_tokens": result.get("talon_anchor_tokens"),
                "talon_rank_tokens": result.get("talon_rank_tokens"),
                "talon_event_tokens": result.get("talon_event_tokens"),
                "talon_recall_tokens": result.get("talon_recall_tokens"),
                "talon_persistence_tokens": result.get("talon_persistence_tokens"),
                "talon_object_tokens": result.get("talon_object_tokens"),
                "talon_memory_tokens": result.get("talon_memory_tokens"),
                "talon_rank_cap": result.get("talon_rank_cap"),
                "talon_chosen_rank": result.get("talon_chosen_rank"),
                "talon_duplicate_index_count": result.get("talon_duplicate_index_count"),
                "talon_question_aware_active": result.get("talon_question_aware_active"),
                "talon_router_mode_code": result.get("talon_router_mode_code"),
                "talon_router_fused_concentration": result.get("talon_router_fused_concentration"),
                "talon_router_residual_concentration": result.get("talon_router_residual_concentration"),
                "talon_router_question_concentration": result.get("talon_router_question_concentration"),
                "talon_router_frame_entropy": result.get("talon_router_frame_entropy"),
                "talon_core_target_budget": result.get("talon_core_target_budget"),
                "talon_core_residual_mean": result.get("talon_core_residual_mean"),
                "talon_core_semantic_tokens": result.get("talon_core_semantic_tokens"),
                "talon_core_innovation_tokens": result.get("talon_core_innovation_tokens"),
                "talon_core_duplicate_index_count": result.get("talon_core_duplicate_index_count"),
                "talon_core_question_aware_active": result.get("talon_core_question_aware_active"),
                "talon_core_budget_min": result.get("talon_core_budget_min"),
                "talon_core_budget_max": result.get("talon_core_budget_max"),
                "talon_core_grid_h": result.get("talon_core_grid_h"),
                "talon_core_grid_w": result.get("talon_core_grid_w"),
                **{key: result.get(key) for key in GRAFT_METRIC_KEYS},
                **{key: result.get(key) for key in CATS_METRIC_KEYS},
                **{key: result.get(key) for key in DYN_METRIC_KEYS},
                **{key: result.get(key) for key in HEDGE_METRIC_KEYS},
                **{key: result.get(key) for key in LEARN_METRIC_KEYS},
                "visual_token_reduction_ratio": reduction_ratio,
                "vision_visual_token_reduction_ratio": vision_reduction_ratio,
            }
        )
    except Exception as exc:  # pragma: no cover - runtime path
        err_text = str(exc).strip()
        record["error"] = f"{type(exc).__name__}: {err_text}" if err_text else f"{type(exc).__name__}"
        record["error_traceback"] = traceback.format_exc(limit=20)

    return record


def _run_phase(
    model_bundle,
    args: BenchmarkArgs,
    samples: list[dict[str, Any]],
    phase_name: str,
    use_acceleration: bool,
    output_path: str,
    phase_key: str | None = None,
):
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for idx, sample in enumerate(samples, 1):
            record = _benchmark_single_sample(model_bundle, args, sample, use_acceleration=use_acceleration)
            record["method"] = _canonical_method_name(phase_key or phase_name)

            # Normalize edge cases so logging/summary is robust:
            # - some exceptions may stringify to empty text;
            # - occasionally metrics can be missing even without an explicit error.
            err = record.get("error")
            if err == "":
                record["error"] = "unknown runtime error"
            if not record.get("error"):
                missing = [
                    key
                    for key in ("latency_ms", "compressed_visual_tokens", "raw_visual_tokens")
                    if record.get(key) is None
                ]
                if missing:
                    record["error"] = f"missing metrics: {', '.join(missing)}"

            try:
                line = _jsonl_line(record)
            except Exception as exc:
                record = {
                    "question_id": record.get("question_id"),
                    "videoID": record.get("videoID"),
                    "duration": record.get("duration"),
                    "answer": record.get("answer"),
                    "pred_answer": record.get("pred_answer", ""),
                    "correct": None,
                    "latency_ms": None,
                    "generated_tokens": None,
                    "tokens_per_second": None,
                    "raw_visual_tokens": None,
                    "compressed_visual_tokens": None,
                    "vision_compressed_visual_tokens": None,
                    "visual_token_reduction_ratio": None,
                    "vision_visual_token_reduction_ratio": None,
                    "method": _canonical_method_name(phase_key or phase_name),
                    "error": f"jsonl write failed: {type(exc).__name__}: {str(exc)[:500]}",
                    "error_traceback": traceback.format_exc(limit=8),
                }
                print(
                    f"[{phase_name}] {idx}/{len(samples)} {record.get('question_id')} "
                    f"jsonl-write-error: {record['error']}"
                )
                line = _jsonl_line(record)
            f.write(line)
            f.flush()

            if record["error"]:
                print(f"[{phase_name}] {idx}/{len(samples)} {record.get('question_id')} error: {record['error']}")
            else:
                latency_ms = float(record["latency_ms"])
                compressed_v = float(record["compressed_visual_tokens"])
                vision_v = float(record.get("vision_compressed_visual_tokens") or compressed_v)
                raw_v = float(record["raw_visual_tokens"])
                print(
                    f"[{phase_name}] {idx}/{len(samples)} {record.get('question_id')} "
                    f"acc={record['correct']} latency={latency_ms:.2f}ms "
                    f"vtoken={compressed_v:.1f}/{raw_v:.1f} vision={vision_v:.1f}/{raw_v:.1f}"
                )


def _read_jsonl(path: str):
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                snippet = line[max(0, exc.pos - 160) : exc.pos + 160].replace("\n", "\\n")
                rows.append(
                    {
                        "question_id": None,
                        "correct": None,
                        "latency_ms": None,
                        "generated_tokens": None,
                        "tokens_per_second": None,
                        "raw_visual_tokens": None,
                        "compressed_visual_tokens": None,
                        "vision_compressed_visual_tokens": None,
                        "visual_token_reduction_ratio": None,
                        "vision_visual_token_reduction_ratio": None,
                        "error": (
                            f"corrupt jsonl line in {path}:{line_no}: "
                            f"{type(exc).__name__}: {exc.msg} at col {exc.colno}; snippet={snippet}"
                        ),
                        "error_traceback": None,
                    }
                )
    return rows


def _summarize_phase(records: list[dict[str, Any]]):
    valid = [r for r in records if not r.get("error")]
    correctness = [float(r["correct"]) for r in valid if r.get("correct") is not None]
    latency = [float(r["latency_ms"]) for r in valid if r.get("latency_ms") is not None]
    gen_tokens = [float(r["generated_tokens"]) for r in valid if r.get("generated_tokens") is not None]
    tps = [float(r["tokens_per_second"]) for r in valid if r.get("tokens_per_second") is not None]
    raw_visual = [float(r["raw_visual_tokens"]) for r in valid if r.get("raw_visual_tokens") is not None]
    compressed_visual = [float(r["compressed_visual_tokens"]) for r in valid if r.get("compressed_visual_tokens") is not None]
    vision_compressed_visual = [
        float(r["vision_compressed_visual_tokens"])
        for r in valid
        if r.get("vision_compressed_visual_tokens") is not None
    ]
    talon_target = [
        float(r["talon_target_tokens_per_frame"])
        for r in valid
        if r.get("talon_target_tokens_per_frame") is not None
    ]
    talon_complexity = [
        float(r["talon_complexity_score"])
        for r in valid
        if r.get("talon_complexity_score") is not None
    ]
    talon_target_budget = [
        float(r["talon_target_budget"])
        for r in valid
        if r.get("talon_target_budget") is not None
    ]
    talon_anchor_tokens = [
        float(r["talon_anchor_tokens"])
        for r in valid
        if r.get("talon_anchor_tokens") is not None
    ]
    talon_rank_tokens = [
        float(r["talon_rank_tokens"])
        for r in valid
        if r.get("talon_rank_tokens") is not None
    ]
    talon_event_tokens = [
        float(r["talon_event_tokens"])
        for r in valid
        if r.get("talon_event_tokens") is not None
    ]
    talon_recall_tokens = [
        float(r["talon_recall_tokens"])
        for r in valid
        if r.get("talon_recall_tokens") is not None
    ]
    talon_persistence_tokens = [
        float(r["talon_persistence_tokens"])
        for r in valid
        if r.get("talon_persistence_tokens") is not None
    ]
    talon_object_tokens = [
        float(r["talon_object_tokens"])
        for r in valid
        if r.get("talon_object_tokens") is not None
    ]
    talon_memory_tokens = [
        float(r["talon_memory_tokens"])
        for r in valid
        if r.get("talon_memory_tokens") is not None
    ]
    talon_rank_cap = [
        float(r["talon_rank_cap"])
        for r in valid
        if r.get("talon_rank_cap") is not None
    ]
    talon_chosen_rank = [
        float(r["talon_chosen_rank"])
        for r in valid
        if r.get("talon_chosen_rank") is not None
    ]
    talon_duplicate_count = [
        float(r["talon_duplicate_index_count"])
        for r in valid
        if r.get("talon_duplicate_index_count") is not None
    ]
    talon_question_active = [
        float(r["talon_question_aware_active"])
        for r in valid
        if r.get("talon_question_aware_active") is not None
    ]
    talon_router_mode = [
        float(r["talon_router_mode_code"])
        for r in valid
        if r.get("talon_router_mode_code") is not None
    ]
    talon_router_fused_conc = [
        float(r["talon_router_fused_concentration"])
        for r in valid
        if r.get("talon_router_fused_concentration") is not None
    ]
    talon_router_residual_conc = [
        float(r["talon_router_residual_concentration"])
        for r in valid
        if r.get("talon_router_residual_concentration") is not None
    ]
    talon_router_question_conc = [
        float(r["talon_router_question_concentration"])
        for r in valid
        if r.get("talon_router_question_concentration") is not None
    ]
    talon_router_frame_entropy = [
        float(r["talon_router_frame_entropy"])
        for r in valid
        if r.get("talon_router_frame_entropy") is not None
    ]
    talon_core_target_budget = [
        float(r["talon_core_target_budget"])
        for r in valid
        if r.get("talon_core_target_budget") is not None
    ]
    talon_core_residual_mean = [
        float(r["talon_core_residual_mean"])
        for r in valid
        if r.get("talon_core_residual_mean") is not None
    ]
    talon_core_semantic_tokens = [
        float(r["talon_core_semantic_tokens"])
        for r in valid
        if r.get("talon_core_semantic_tokens") is not None
    ]
    talon_core_innovation_tokens = [
        float(r["talon_core_innovation_tokens"])
        for r in valid
        if r.get("talon_core_innovation_tokens") is not None
    ]
    talon_core_duplicate_count = [
        float(r["talon_core_duplicate_index_count"])
        for r in valid
        if r.get("talon_core_duplicate_index_count") is not None
    ]
    talon_core_question_active = [
        float(r["talon_core_question_aware_active"])
        for r in valid
        if r.get("talon_core_question_aware_active") is not None
    ]
    talon_core_budget_min = [
        float(r["talon_core_budget_min"])
        for r in valid
        if r.get("talon_core_budget_min") is not None
    ]
    talon_core_budget_max = [
        float(r["talon_core_budget_max"])
        for r in valid
        if r.get("talon_core_budget_max") is not None
    ]
    talon_core_grid_h = [
        float(r["talon_core_grid_h"])
        for r in valid
        if r.get("talon_core_grid_h") is not None
    ]
    talon_core_grid_w = [
        float(r["talon_core_grid_w"])
        for r in valid
        if r.get("talon_core_grid_w") is not None
    ]
    reduction = [float(r["visual_token_reduction_ratio"]) for r in valid if r.get("visual_token_reduction_ratio") is not None]
    vision_reduction = [
        float(r["vision_visual_token_reduction_ratio"])
        for r in valid
        if r.get("vision_visual_token_reduction_ratio") is not None
    ]
    graft_stats = {
        key: _stats([float(r[key]) for r in valid if r.get(key) is not None])
        for key in GRAFT_METRIC_KEYS
    }
    cats_stats = {
        key: _stats([float(r[key]) for r in valid if r.get(key) is not None])
        for key in CATS_METRIC_KEYS
    }
    dyn_stats = {
        key: _stats([float(r[key]) for r in valid if r.get(key) is not None])
        for key in DYN_METRIC_KEYS
    }
    hedge_stats = {
        key: _stats([float(r[key]) for r in valid if r.get(key) is not None])
        for key in HEDGE_METRIC_KEYS
    }
    learn_stats = {
        key: _stats([float(r[key]) for r in valid if r.get(key) is not None])
        for key in LEARN_METRIC_KEYS
    }

    return {
        "num_samples": len(records),
        "num_valid": len(valid),
        "num_errors": len(records) - len(valid),
        "accuracy": float(np.mean(correctness)) if correctness else None,
        "latency_ms": _stats(latency),
        "generated_tokens": _stats(gen_tokens),
        "tokens_per_second": _stats(tps),
        "raw_visual_tokens": _stats(raw_visual),
        "compressed_visual_tokens": _stats(compressed_visual),
        "vision_compressed_visual_tokens": _stats(vision_compressed_visual),
        "talon_target_tokens_per_frame": _stats(talon_target),
        "talon_complexity_score": _stats(talon_complexity),
        "talon_target_budget": _stats(talon_target_budget),
        "talon_anchor_tokens": _stats(talon_anchor_tokens),
        "talon_rank_tokens": _stats(talon_rank_tokens),
        "talon_event_tokens": _stats(talon_event_tokens),
        "talon_recall_tokens": _stats(talon_recall_tokens),
        "talon_persistence_tokens": _stats(talon_persistence_tokens),
        "talon_object_tokens": _stats(talon_object_tokens),
        "talon_memory_tokens": _stats(talon_memory_tokens),
        "talon_rank_cap": _stats(talon_rank_cap),
        "talon_chosen_rank": _stats(talon_chosen_rank),
        "talon_duplicate_index_count": _stats(talon_duplicate_count),
        "talon_question_aware_active": _stats(talon_question_active),
        "talon_router_mode_code": _stats(talon_router_mode),
        "talon_router_fused_concentration": _stats(talon_router_fused_conc),
        "talon_router_residual_concentration": _stats(talon_router_residual_conc),
        "talon_router_question_concentration": _stats(talon_router_question_conc),
        "talon_router_frame_entropy": _stats(talon_router_frame_entropy),
        "talon_core_target_budget": _stats(talon_core_target_budget),
        "talon_core_residual_mean": _stats(talon_core_residual_mean),
        "talon_core_semantic_tokens": _stats(talon_core_semantic_tokens),
        "talon_core_innovation_tokens": _stats(talon_core_innovation_tokens),
        "talon_core_duplicate_index_count": _stats(talon_core_duplicate_count),
        "talon_core_question_aware_active": _stats(talon_core_question_active),
        "talon_core_budget_min": _stats(talon_core_budget_min),
        "talon_core_budget_max": _stats(talon_core_budget_max),
        "talon_core_grid_h": _stats(talon_core_grid_h),
        "talon_core_grid_w": _stats(talon_core_grid_w),
        **graft_stats,
        **cats_stats,
        **dyn_stats,
        **hedge_stats,
        **learn_stats,
        "visual_token_reduction_ratio": _stats(reduction),
        "vision_visual_token_reduction_ratio": _stats(vision_reduction),
    }


def _summarize_pairwise_comparison(
    anchor_records: list[dict[str, Any]],
    target_records: list[dict[str, Any]],
    *,
    anchor_name: str,
    target_name: str,
):
    target_by_qid = {r["question_id"]: r for r in target_records if r.get("question_id")}
    matched = []
    for b in anchor_records:
        qid = b.get("question_id")
        if qid not in target_by_qid:
            continue
        t = target_by_qid[qid]
        if b.get("error") or t.get("error"):
            continue
        matched.append((b, t))

    latency_ratios = []
    token_ratios = []
    vision_token_ratios = []
    reduction_gains = []
    vision_reduction_gains = []
    both_correct = 0
    both_wrong = 0
    anchor_only_correct = 0
    target_only_correct = 0
    for b, f in matched:
        b_ok = bool(b.get("correct"))
        f_ok = bool(f.get("correct"))
        if b_ok and f_ok:
            both_correct += 1
        elif (not b_ok) and (not f_ok):
            both_wrong += 1
        elif b_ok and not f_ok:
            anchor_only_correct += 1
        elif f_ok and not b_ok:
            target_only_correct += 1

        b_lat = b.get("latency_ms")
        f_lat = f.get("latency_ms")
        if b_lat and f_lat and f_lat > 0:
            latency_ratios.append(float(b_lat) / float(f_lat))

        b_v = b.get("compressed_visual_tokens")
        f_v = f.get("compressed_visual_tokens")
        if b_v and f_v and b_v > 0:
            token_ratios.append(float(f_v) / float(b_v))
            reduction_gains.append(float(1.0 - (f_v / b_v)))

        b_vision_v = b.get("vision_compressed_visual_tokens")
        f_vision_v = f.get("vision_compressed_visual_tokens")
        if b_vision_v and f_vision_v and b_vision_v > 0:
            vision_token_ratios.append(float(f_vision_v) / float(b_vision_v))
            vision_reduction_gains.append(float(1.0 - (f_vision_v / b_vision_v)))

    return {
        "matched_samples": len(matched),
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        f"{anchor_name}_only_correct": anchor_only_correct,
        f"{target_name}_only_correct": target_only_correct,
        "latency_speedup_ratio": _stats(latency_ratios),
        f"visual_token_ratio_{target_name}_over_{anchor_name}": _stats(token_ratios),
        f"visual_token_reduction_vs_{anchor_name}": _stats(reduction_gains),
        f"vision_token_ratio_{target_name}_over_{anchor_name}": _stats(vision_token_ratios),
        f"vision_token_reduction_vs_{anchor_name}": _stats(vision_reduction_gains),
    }


def _add_duration_breakdown(
    summary: dict[str, Any],
    *,
    baseline_records: Optional[list[dict[str, Any]]] = None,
    flashvid_records: Optional[list[dict[str, Any]]] = None,
    ours_records: Optional[list[dict[str, Any]]] = None,
    ours_phase_name: str = "ours",
    graphvid_records: Optional[list[dict[str, Any]]] = None,
    graftvid_records: Optional[list[dict[str, Any]]] = None,
    cats_records: Optional[list[dict[str, Any]]] = None,
) -> None:
    ours_phase_name = _canonical_method_name(ours_phase_name)
    records_by_phase = {
        "baseline": baseline_records,
        "flashvid": flashvid_records,
        ours_phase_name: ours_records,
        "graphvid": graphvid_records,
        "graftvid": graftvid_records,
        "cats": cats_records,
    }
    durations = ["short", "medium", "long"]
    breakdown: dict[str, Any] = {}
    for duration in durations:
        bucket: dict[str, Any] = {"comparison": {}}
        phase_records: dict[str, list[dict[str, Any]]] = {}
        for phase_name, records in records_by_phase.items():
            if records is None:
                continue
            subset = [
                r for r in records
                if str(r.get("duration") or "").strip().lower() == duration
            ]
            phase_records[phase_name] = subset
            bucket[phase_name] = _summarize_phase(subset)

        if "baseline" in phase_records and "flashvid" in phase_records:
            bucket["comparison"]["baseline_vs_flashvid"] = _summarize_pairwise_comparison(
                phase_records["baseline"],
                phase_records["flashvid"],
                anchor_name="baseline",
                target_name="flashvid",
            )
        if "baseline" in phase_records and ours_phase_name in phase_records:
            bucket["comparison"][f"baseline_vs_{ours_phase_name}"] = _summarize_pairwise_comparison(
                phase_records["baseline"],
                phase_records[ours_phase_name],
                anchor_name="baseline",
                target_name=ours_phase_name,
            )
        if "flashvid" in phase_records and ours_phase_name in phase_records:
            bucket["comparison"][f"flashvid_vs_{ours_phase_name}"] = _summarize_pairwise_comparison(
                phase_records["flashvid"],
                phase_records[ours_phase_name],
                anchor_name="flashvid",
                target_name=ours_phase_name,
            )
        if "flashvid" in phase_records and "graphvid" in phase_records:
            bucket["comparison"]["flashvid_vs_graphvid"] = _summarize_pairwise_comparison(
                phase_records["flashvid"],
                phase_records["graphvid"],
                anchor_name="flashvid",
                target_name="graphvid",
            )
        if "flashvid" in phase_records and "graftvid" in phase_records:
            bucket["comparison"]["flashvid_vs_graftvid"] = _summarize_pairwise_comparison(
                phase_records["flashvid"],
                phase_records["graftvid"],
                anchor_name="flashvid",
                target_name="graftvid",
            )
        if "flashvid" in phase_records and "cats" in phase_records:
            bucket["comparison"]["flashvid_vs_cats"] = _summarize_pairwise_comparison(
                phase_records["flashvid"],
                phase_records["cats"],
                anchor_name="flashvid",
                target_name="cats",
            )
        breakdown[duration] = bucket
    summary["duration_breakdown"] = breakdown


def _resolve_llm_pruning_args(backend: str, args: BenchmarkArgs) -> tuple[int, float]:
    # For LLaVA, keep the stable vision-only path unless the caller explicitly
    # requests inner-LLM pruning with llm_retention_ratio < 1.0.
    if backend == "llava" and float(args.llm_retention_ratio) >= 0.9999:
        return 10**9, 1.0
    return args.pruning_layer, args.llm_retention_ratio


def _apply_flashvid_original(model, args: BenchmarkArgs, backend: str):
    from flashvid import flashvid
    pruning_layer, llm_retention_ratio = _resolve_llm_pruning_args(backend, args)

    return flashvid(
        model=model,
        retention_ratio=args.retention_ratio,
        do_segment=args.do_segment,
        segment_threshold=args.segment_threshold,
        min_segment_num=args.min_segment_num,
        complementary_segment=args.complementary_segment,
        token_selection_method=args.token_selection_method,
        alpha=args.alpha,
        temporal_threshold=args.temporal_threshold,
        temporal_merge_mode="tree",
        graph_temporal_topk=args.graph_temporal_topk,
        graph_temporal_radius=args.graph_temporal_radius,
        graph_temporal_skip=args.graph_temporal_skip,
        graph_merge_protect_ratio=args.graph_merge_protect_ratio,
        graph_merge_target_ratio=args.graph_merge_target_ratio,
        graph_merge_representative=args.graph_merge_representative,
        graph_representative_position=args.graph_representative_position,
        graph_protection_attn_weight=args.graph_protection_attn_weight,
        graph_protection_novelty_weight=args.graph_protection_novelty_weight,
        graph_protection_detail_weight=args.graph_protection_detail_weight,
        graph_adaptive_detail_protection=args.graph_adaptive_detail_protection,
        graph_adaptive_detail_boost=args.graph_adaptive_detail_boost,
        graph_adaptive_protect_boost=args.graph_adaptive_protect_boost,
        graph_merge_importance_penalty=args.graph_merge_importance_penalty,
        graph_respect_temporal_threshold=args.graph_respect_temporal_threshold,
        graph_final_tokens_per_frame=args.graph_final_tokens_per_frame,
        graph_final_frame_floor_ratio=args.graph_final_frame_floor_ratio,
        graph_skip_spatial_merge_when_capped=args.graph_skip_spatial_merge_when_capped,
        graft_temporal_topk=args.graft_temporal_topk,
        graft_temporal_radius=args.graft_temporal_radius,
        graft_temporal_skip=args.graft_temporal_skip,
        graft_global_topk=args.graft_global_topk,
        graft_input_is_residual=args.graft_input_is_residual,
        graft_anchor_ratio=args.graft_anchor_ratio,
        graft_edge_threshold=args.graft_edge_threshold,
        graft_component_radius_eps=args.graft_component_radius_eps,
        graft_split_radius_eps=args.graft_split_radius_eps,
        graft_parent_capacity=args.graft_parent_capacity,
        graft_mutual_knn=args.graft_mutual_knn,
        graft_one_token_per_frame=args.graft_one_token_per_frame,
        graft_spatial_penalty=args.graft_spatial_penalty,
        graft_importance_penalty=args.graft_importance_penalty,
        graft_hub_penalty=args.graft_hub_penalty,
        graft_adaptive_aggregation=args.graft_adaptive_aggregation,
        graft_scene_threshold=args.graft_scene_threshold,
        graft_min_tokens_per_frame=args.graft_min_tokens_per_frame,
        graft_budget_correction=args.graft_budget_correction,
        graft_budget_diversity_weight=args.graft_budget_diversity_weight,
        graft_score_preset=args.graft_score_preset,
        graft_duration_aware=args.graft_duration_aware,
        graft_medium_temporal_skip=args.graft_medium_temporal_skip,
        graft_medium_global_topk=args.graft_medium_global_topk,
        graft_medium_edge_threshold=args.graft_medium_edge_threshold,
        graft_medium_split_radius_eps=args.graft_medium_split_radius_eps,
        graft_medium_spatial_penalty=args.graft_medium_spatial_penalty,
        graft_medium_scene_threshold=args.graft_medium_scene_threshold,
        graft_long_temporal_skip=args.graft_long_temporal_skip,
        graft_long_global_topk=args.graft_long_global_topk,
        graft_long_edge_threshold=args.graft_long_edge_threshold,
        graft_long_split_radius_eps=args.graft_long_split_radius_eps,
        graft_long_spatial_penalty=args.graft_long_spatial_penalty,
        graft_long_scene_threshold=args.graft_long_scene_threshold,
        hedge_stable_floor_ratio=args.hedge_stable_floor_ratio,
        hedge_diversity_weight=args.hedge_diversity_weight,
        hedge_stable_bias=args.hedge_stable_bias,
        hedge_evidence_bias=args.hedge_evidence_bias,
        hedge_max_mmr_candidates=args.hedge_max_mmr_candidates,
        learn_selector_ckpt=args.learn_selector_ckpt,
        learn_qaware=args.learn_qaware,
        learn_stable_floor_ratio=args.learn_stable_floor_ratio,
        learn_score_blend=args.learn_score_blend,
        learn_q_relevance_weight=args.learn_q_relevance_weight,
        learn_density_topk=args.learn_density_topk,
        learn_collect_teacher=args.learn_collect_teacher,
        expansion=args.expansion,
        pruning_layer=pruning_layer,
        llm_retention_ratio=llm_retention_ratio,
        compression_variant="flashvid",
        question_aware_reweighting=False,
        question_reweight_beta=args.question_reweight_beta,
        adaptive_token_budget=False,
        talon_transport_radius=args.talon_transport_radius,
        talon_rank_ratio=args.talon_rank_ratio,
        talon_rank_min=args.talon_rank_min,
        talon_rank_max=args.talon_rank_max,
        talon_budget_scale=args.talon_budget_scale,
        talon_target_tokens_per_frame=args.talon_target_tokens_per_frame,
        talon_short_target_tokens_per_frame=args.talon_short_target_tokens_per_frame,
        talon_medium_target_tokens_per_frame=args.talon_medium_target_tokens_per_frame,
        talon_long_target_tokens_per_frame=args.talon_long_target_tokens_per_frame,
        talon_min_total_tokens=args.talon_min_total_tokens,
        talon_fast_rank_plan=args.talon_fast_rank_plan,
        talon_background_max_ratio=args.talon_background_max_ratio,
        talon_frame_balanced_selection=args.talon_frame_balanced_selection,
        talon_basis_method=args.talon_basis_method,
        talon_basis_oversample=args.talon_basis_oversample,
        talon_innovation_attention_weight=args.talon_innovation_attention_weight,
        talon_motion_importance_weight=args.talon_motion_importance_weight,
        talon_boundary_importance_weight=args.talon_boundary_importance_weight,
        talon_question_frame_weight=args.talon_question_frame_weight,
        talon_frame_balanced_memory=args.talon_frame_balanced_memory,
        talon_memory_mode=args.talon_memory_mode,
        talon_anchor_safety_ratio=args.talon_anchor_safety_ratio,
        talon_anchor_diversity_weight=args.talon_anchor_diversity_weight,
        talon_anchor_candidate_multiplier=args.talon_anchor_candidate_multiplier,
        talon_spatial_anchor_coverage=args.talon_spatial_anchor_coverage,
        talon_spatial_anchor_ratio=args.talon_spatial_anchor_ratio,
        talon_spatial_anchor_rows=args.talon_spatial_anchor_rows,
        talon_spatial_anchor_cols=args.talon_spatial_anchor_cols,
        talon_spatial_anchor_score=args.talon_spatial_anchor_score,
        talon_spatial_anchor_apply_to_short=args.talon_spatial_anchor_apply_to_short,
        talon_frame_coverage_floor_ratio=args.talon_frame_coverage_floor_ratio,
        talon_frame_importance_pooling=args.talon_frame_importance_pooling,
        talon_frame_importance_topk=args.talon_frame_importance_topk,
        talon_medium_frame_coverage_floor_ratio=args.talon_medium_frame_coverage_floor_ratio,
        talon_long_frame_coverage_floor_ratio=args.talon_long_frame_coverage_floor_ratio,
        talon_frame_local_budget_ratio=args.talon_frame_local_budget_ratio,
        talon_question_recall_ratio=args.talon_question_recall_ratio,
        talon_question_recall_qweight=args.talon_question_recall_qweight,
        talon_persistence_recall_ratio=args.talon_persistence_recall_ratio,
        talon_persistence_recall_qweight=args.talon_persistence_recall_qweight,
        talon_persistence_recall_pweight=args.talon_persistence_recall_pweight,
        talon_persistence_apply_to_short=args.talon_persistence_apply_to_short,
        talon_persistence_apply_to_medium=args.talon_persistence_apply_to_medium,
        talon_persistence_apply_to_long=args.talon_persistence_apply_to_long,
        talon_object_evidence_ratio=args.talon_object_evidence_ratio,
        talon_object_evidence_qweight=args.talon_object_evidence_qweight,
        talon_object_evidence_sweight=args.talon_object_evidence_sweight,
        talon_object_evidence_pweight=args.talon_object_evidence_pweight,
        talon_object_evidence_apply_to_short=args.talon_object_evidence_apply_to_short,
        talon_object_evidence_apply_to_medium=args.talon_object_evidence_apply_to_medium,
        talon_object_evidence_apply_to_long=args.talon_object_evidence_apply_to_long,
        talon_question_pooling=args.talon_question_pooling,
        talon_question_pooling_topk=args.talon_question_pooling_topk,
        talon_question_contrast_weight=args.talon_question_contrast_weight,
        talon_question_contrast_apply_to_short=args.talon_question_contrast_apply_to_short,
        talon_monotonic_base_tokens_per_frame=args.talon_monotonic_base_tokens_per_frame,
        talon_budget_strategy=args.talon_budget_strategy,
        talon_budget_mode=args.talon_budget_mode,
        talon_transport_mode=args.talon_transport_mode,
        talon_transport_temperature=args.talon_transport_temperature,
        talon_lite_enabled=args.talon_lite_enabled,
        talon_echo_temperature=args.talon_echo_temperature,
        talon_echo_topk_neighbors=args.talon_echo_topk_neighbors,
        talon_echo_residual_weight=args.talon_echo_residual_weight,
        talon_echo_score_mode=args.talon_echo_score_mode,
        talon_rd_spectral_weight=args.talon_rd_spectral_weight,
        talon_rd_innovation_weight=args.talon_rd_innovation_weight,
        talon_use_question_innovation=args.talon_use_question_innovation,
        talon_innovation_qweight=args.talon_innovation_qweight,
        talon_output_mode=args.talon_output_mode,
        talon_reconstruction_blend=args.talon_reconstruction_blend,
        talon_anchor_score_weight=args.talon_anchor_score_weight,
        talon_min_anchor_per_frame=args.talon_min_anchor_per_frame,
        talon_passthrough_ratio=args.talon_passthrough_ratio,
        talon_passthrough_min=args.talon_passthrough_min,
        talon_use_segmentation=args.talon_use_segmentation,
        talon_disable_oversegmentation=args.talon_disable_oversegmentation,
        talon_max_segments=args.talon_max_segments,
        talon_deepstack_mode=args.talon_deepstack_mode,
        memory_token_ratio=args.memory_token_ratio,
        memory_token_min=args.memory_token_min,
        memory_token_max=args.memory_token_max,
        talon_adaptive_target_low=args.talon_adaptive_target_low,
        talon_adaptive_target_mid=args.talon_adaptive_target_mid,
        talon_adaptive_target_high=args.talon_adaptive_target_high,
        talon_complexity_floor=args.talon_complexity_floor,
        talon_complexity_ceil=args.talon_complexity_ceil,
        talon_adaptive_gamma=args.talon_adaptive_gamma,
        talon_adaptive_target_enabled=args.talon_adaptive_target_enabled,
        talon_force_fixed_target=args.talon_force_fixed_target,
        talon_target_mean_cap=args.talon_target_mean_cap,
        talon_unified_selection=args.talon_unified_selection,
        talon_low_budget_mode_threshold=args.talon_low_budget_mode_threshold,
        talon_low_budget_rank_cap=args.talon_low_budget_rank_cap,
        talon_background_global_ratio=args.talon_background_global_ratio,
        talon_event_budget_ratio=args.talon_event_budget_ratio,
        talon_memory_fused_weight=args.talon_memory_fused_weight,
        talon_memory_residual_weight=args.talon_memory_residual_weight,
        talon_memory_frame_weight=args.talon_memory_frame_weight,
        talon_recall_memory_mode=args.talon_recall_memory_mode,
        talon_final_fused_weight=args.talon_final_fused_weight,
        talon_final_residual_weight=args.talon_final_residual_weight,
        talon_final_frame_weight=args.talon_final_frame_weight,
        talon_anchor_keep_bonus=args.talon_anchor_keep_bonus,
        talon_recall_keep_bonus=args.talon_recall_keep_bonus,
        talon_event_keep_bonus=args.talon_event_keep_bonus,
        talon_legacy_base_keep_ratio=args.talon_legacy_base_keep_ratio,
        talon_prior_candidate_ratio=args.talon_prior_candidate_ratio,
        talon_prior_keep_bonus=args.talon_prior_keep_bonus,
        talon_flash_prior_channel_ratio=args.talon_flash_prior_channel_ratio,
        talon_flash_prior_channel_method=args.talon_flash_prior_channel_method,
        talon_flash_prior_channel_min_per_frame=args.talon_flash_prior_channel_min_per_frame,
        talon_flash_prior_channel_max_per_frame=args.talon_flash_prior_channel_max_per_frame,
        talon_flash_prior_channel_bonus=args.talon_flash_prior_channel_bonus,
        talon_final_anchor_min_ratio=args.talon_final_anchor_min_ratio,
        talon_final_recall_min_ratio=args.talon_final_recall_min_ratio,
        talon_force_anchor_recall_quota=args.talon_force_anchor_recall_quota,
        talon_global_topk_ratio=args.talon_global_topk_ratio,
        talon_rescue_enabled=args.talon_rescue_enabled,
        talon_rescue_ratio=args.talon_rescue_ratio,
        talon_rescue_from_memory_only=args.talon_rescue_from_memory_only,
        talon_rescue_fused_weight=args.talon_rescue_fused_weight,
        talon_rescue_residual_weight=args.talon_rescue_residual_weight,
        talon_rescue_frame_weight=args.talon_rescue_frame_weight,
        talon_rescue_global_ratio=args.talon_rescue_global_ratio,
        talon_rerank_with_flash_prior=args.talon_rerank_with_flash_prior,
        talon_flash_prior_ratio=args.talon_flash_prior_ratio,
        talon_recall_semantic_ratio=args.talon_recall_semantic_ratio,
        talon_recall_event_ratio=args.talon_recall_event_ratio,
        talon_recall_frame_ratio=args.talon_recall_frame_ratio,
        talon_recall_global_ratio=args.talon_recall_global_ratio,
        talon_duration_aware=args.talon_duration_aware,
        talon_medium_anchor_safety_ratio=args.talon_medium_anchor_safety_ratio,
        talon_medium_event_budget_ratio=args.talon_medium_event_budget_ratio,
        talon_medium_global_topk_ratio=args.talon_medium_global_topk_ratio,
        talon_long_anchor_safety_ratio=args.talon_long_anchor_safety_ratio,
        talon_long_event_budget_ratio=args.talon_long_event_budget_ratio,
        talon_long_global_topk_ratio=args.talon_long_global_topk_ratio,
        talon_task_aware_event=args.talon_task_aware_event,
        talon_task_event_attention_weight=args.talon_task_event_attention_weight,
        talon_task_event_qweight=args.talon_task_event_qweight,
        talon_visual_task_balance=args.talon_visual_task_balance,
        talon_visual_task_anchor_ratio=args.talon_visual_task_anchor_ratio,
        talon_visual_task_event_ratio=args.talon_visual_task_event_ratio,
        talon_visual_task_recall_ratio=args.talon_visual_task_recall_ratio,
        talon_knowledge_visual_anchor_ratio=args.talon_knowledge_visual_anchor_ratio,
        talon_knowledge_visual_event_ratio=args.talon_knowledge_visual_event_ratio,
        talon_knowledge_visual_recall_ratio=args.talon_knowledge_visual_recall_ratio,
        talon_adaptive_router=args.talon_adaptive_router,
        talon_router_apply_to_short=args.talon_router_apply_to_short,
        talon_router_visual_anchor_ratio=args.talon_router_visual_anchor_ratio,
        talon_router_visual_event_ratio=args.talon_router_visual_event_ratio,
        talon_router_visual_recall_ratio=args.talon_router_visual_recall_ratio,
        talon_router_temporal_anchor_ratio=args.talon_router_temporal_anchor_ratio,
        talon_router_temporal_event_ratio=args.talon_router_temporal_event_ratio,
        talon_router_temporal_recall_ratio=args.talon_router_temporal_recall_ratio,
        talon_router_balanced_anchor_ratio=args.talon_router_balanced_anchor_ratio,
        talon_router_balanced_event_ratio=args.talon_router_balanced_event_ratio,
        talon_router_balanced_recall_ratio=args.talon_router_balanced_recall_ratio,
        talon_router_visual_concentration_threshold=args.talon_router_visual_concentration_threshold,
        talon_router_low_residual_threshold=args.talon_router_low_residual_threshold,
        talon_router_temporal_entropy_threshold=args.talon_router_temporal_entropy_threshold,
        talon_router_temporal_residual_threshold=args.talon_router_temporal_residual_threshold,
        talon_temporal_chunk_aware=args.talon_temporal_chunk_aware,
        talon_temporal_num_chunks=args.talon_temporal_num_chunks,
        talon_temporal_chunk_min_ratio=args.talon_temporal_chunk_min_ratio,
        talon_temporal_chunk_score=args.talon_temporal_chunk_score,
        talon_track_aware=args.talon_track_aware,
        talon_track_budget_ratio=args.talon_track_budget_ratio,
        talon_track_tokens_per_slot=args.talon_track_tokens_per_slot,
        talon_track_score=args.talon_track_score,
        talon_absorb_dropped_tokens=args.talon_absorb_dropped_tokens,
        talon_absorb_ratio=args.talon_absorb_ratio,
        talon_absorb_alpha=args.talon_absorb_alpha,
        talon_absorb_score=args.talon_absorb_score,
        talon_summary_replacement=args.talon_summary_replacement,
        talon_summary_raw_swap=args.talon_summary_raw_swap,
        talon_summary_ratio=args.talon_summary_ratio,
        talon_summary_num_chunks=args.talon_summary_num_chunks,
        talon_summary_pool_topk=args.talon_summary_pool_topk,
        talon_summary_alpha=args.talon_summary_alpha,
        talon_summary_score=args.talon_summary_score,
        decode_policy=args.decode_policy,
        decode_kv_budget_ratio=args.decode_kv_budget_ratio,
        decode_update_interval=args.decode_update_interval,
        decode_start_layer=args.decode_start_layer,
    )


def _apply_graphvid(model, args: BenchmarkArgs, backend: str):
    from flashvid import flashvid
    pruning_layer, llm_retention_ratio = _resolve_llm_pruning_args(backend, args)

    return flashvid(
        model=model,
        retention_ratio=args.retention_ratio,
        do_segment=args.do_segment,
        segment_threshold=args.segment_threshold,
        min_segment_num=args.min_segment_num,
        complementary_segment=args.complementary_segment,
        token_selection_method=args.graphvid_token_selection_method or args.token_selection_method,
        alpha=args.alpha,
        temporal_threshold=args.temporal_threshold,
        temporal_merge_mode="graph",
        graph_temporal_topk=args.graph_temporal_topk,
        graph_temporal_radius=args.graph_temporal_radius,
        graph_temporal_skip=args.graph_temporal_skip,
        graph_merge_protect_ratio=args.graph_merge_protect_ratio,
        graph_merge_target_ratio=args.graph_merge_target_ratio,
        graph_merge_representative=args.graph_merge_representative,
        graph_representative_position=args.graph_representative_position,
        graph_protection_attn_weight=args.graph_protection_attn_weight,
        graph_protection_novelty_weight=args.graph_protection_novelty_weight,
        graph_protection_detail_weight=args.graph_protection_detail_weight,
        graph_adaptive_detail_protection=args.graph_adaptive_detail_protection,
        graph_adaptive_detail_boost=args.graph_adaptive_detail_boost,
        graph_adaptive_protect_boost=args.graph_adaptive_protect_boost,
        graph_merge_importance_penalty=args.graph_merge_importance_penalty,
        graph_respect_temporal_threshold=args.graph_respect_temporal_threshold,
        graph_final_tokens_per_frame=args.graph_final_tokens_per_frame,
        graph_final_frame_floor_ratio=args.graph_final_frame_floor_ratio,
        graph_skip_spatial_merge_when_capped=args.graph_skip_spatial_merge_when_capped,
        graft_temporal_topk=args.graft_temporal_topk,
        graft_temporal_radius=args.graft_temporal_radius,
        graft_temporal_skip=args.graft_temporal_skip,
        graft_global_topk=args.graft_global_topk,
        graft_input_is_residual=args.graft_input_is_residual,
        graft_anchor_ratio=args.graft_anchor_ratio,
        graft_edge_threshold=args.graft_edge_threshold,
        graft_component_radius_eps=args.graft_component_radius_eps,
        graft_split_radius_eps=args.graft_split_radius_eps,
        graft_parent_capacity=args.graft_parent_capacity,
        graft_mutual_knn=args.graft_mutual_knn,
        graft_one_token_per_frame=args.graft_one_token_per_frame,
        graft_spatial_penalty=args.graft_spatial_penalty,
        graft_importance_penalty=args.graft_importance_penalty,
        graft_hub_penalty=args.graft_hub_penalty,
        graft_adaptive_aggregation=args.graft_adaptive_aggregation,
        graft_scene_threshold=args.graft_scene_threshold,
        graft_min_tokens_per_frame=args.graft_min_tokens_per_frame,
        graft_budget_correction=args.graft_budget_correction,
        graft_budget_diversity_weight=args.graft_budget_diversity_weight,
        graft_score_preset=args.graft_score_preset,
        graft_duration_aware=args.graft_duration_aware,
        graft_medium_temporal_skip=args.graft_medium_temporal_skip,
        graft_medium_global_topk=args.graft_medium_global_topk,
        graft_medium_edge_threshold=args.graft_medium_edge_threshold,
        graft_medium_split_radius_eps=args.graft_medium_split_radius_eps,
        graft_medium_spatial_penalty=args.graft_medium_spatial_penalty,
        graft_medium_scene_threshold=args.graft_medium_scene_threshold,
        graft_long_temporal_skip=args.graft_long_temporal_skip,
        graft_long_global_topk=args.graft_long_global_topk,
        graft_long_edge_threshold=args.graft_long_edge_threshold,
        graft_long_split_radius_eps=args.graft_long_split_radius_eps,
        graft_long_spatial_penalty=args.graft_long_spatial_penalty,
        graft_long_scene_threshold=args.graft_long_scene_threshold,
        hedge_stable_floor_ratio=args.hedge_stable_floor_ratio,
        hedge_diversity_weight=args.hedge_diversity_weight,
        hedge_stable_bias=args.hedge_stable_bias,
        hedge_evidence_bias=args.hedge_evidence_bias,
        hedge_max_mmr_candidates=args.hedge_max_mmr_candidates,
        expansion=args.expansion,
        pruning_layer=pruning_layer,
        llm_retention_ratio=llm_retention_ratio,
        compression_variant="graphvid",
        question_aware_reweighting=False,
        question_reweight_beta=args.question_reweight_beta,
        adaptive_token_budget=False,
        decode_policy=args.decode_policy,
        decode_kv_budget_ratio=args.decode_kv_budget_ratio,
        decode_update_interval=args.decode_update_interval,
        decode_start_layer=args.decode_start_layer,
    )


def _apply_graftvid(model, args: BenchmarkArgs, backend: str):
    from flashvid import flashvid
    pruning_layer, llm_retention_ratio = _resolve_llm_pruning_args(backend, args)

    return flashvid(
        model=model,
        retention_ratio=args.retention_ratio,
        do_segment=args.do_segment,
        segment_threshold=args.segment_threshold,
        min_segment_num=args.min_segment_num,
        complementary_segment=args.complementary_segment,
        token_selection_method=args.graphvid_token_selection_method or args.token_selection_method,
        alpha=args.alpha,
        temporal_threshold=args.temporal_threshold,
        temporal_merge_mode="graft",
        graph_temporal_topk=args.graph_temporal_topk,
        graph_temporal_radius=args.graph_temporal_radius,
        graph_temporal_skip=args.graph_temporal_skip,
        graph_merge_protect_ratio=args.graph_merge_protect_ratio,
        graph_merge_target_ratio=args.graph_merge_target_ratio,
        graph_merge_representative=args.graph_merge_representative,
        graph_representative_position=args.graph_representative_position,
        graph_protection_attn_weight=args.graph_protection_attn_weight,
        graph_protection_novelty_weight=args.graph_protection_novelty_weight,
        graph_protection_detail_weight=args.graph_protection_detail_weight,
        graph_adaptive_detail_protection=args.graph_adaptive_detail_protection,
        graph_adaptive_detail_boost=args.graph_adaptive_detail_boost,
        graph_adaptive_protect_boost=args.graph_adaptive_protect_boost,
        graph_merge_importance_penalty=args.graph_merge_importance_penalty,
        graph_respect_temporal_threshold=args.graph_respect_temporal_threshold,
        graph_final_tokens_per_frame=args.graph_final_tokens_per_frame,
        graph_final_frame_floor_ratio=args.graph_final_frame_floor_ratio,
        graph_skip_spatial_merge_when_capped=args.graph_skip_spatial_merge_when_capped,
        graft_temporal_topk=args.graft_temporal_topk,
        graft_temporal_radius=args.graft_temporal_radius,
        graft_temporal_skip=args.graft_temporal_skip,
        graft_global_topk=args.graft_global_topk,
        graft_input_is_residual=args.graft_input_is_residual,
        graft_anchor_ratio=args.graft_anchor_ratio,
        graft_edge_threshold=args.graft_edge_threshold,
        graft_component_radius_eps=args.graft_component_radius_eps,
        graft_split_radius_eps=args.graft_split_radius_eps,
        graft_parent_capacity=args.graft_parent_capacity,
        graft_mutual_knn=args.graft_mutual_knn,
        graft_one_token_per_frame=args.graft_one_token_per_frame,
        graft_spatial_penalty=args.graft_spatial_penalty,
        graft_importance_penalty=args.graft_importance_penalty,
        graft_hub_penalty=args.graft_hub_penalty,
        graft_adaptive_aggregation=args.graft_adaptive_aggregation,
        graft_scene_threshold=args.graft_scene_threshold,
        graft_min_tokens_per_frame=args.graft_min_tokens_per_frame,
        graft_budget_correction=args.graft_budget_correction,
        graft_budget_diversity_weight=args.graft_budget_diversity_weight,
        graft_score_preset=args.graft_score_preset,
        graft_duration_aware=args.graft_duration_aware,
        graft_medium_temporal_skip=args.graft_medium_temporal_skip,
        graft_medium_global_topk=args.graft_medium_global_topk,
        graft_medium_edge_threshold=args.graft_medium_edge_threshold,
        graft_medium_split_radius_eps=args.graft_medium_split_radius_eps,
        graft_medium_spatial_penalty=args.graft_medium_spatial_penalty,
        graft_medium_scene_threshold=args.graft_medium_scene_threshold,
        graft_long_temporal_skip=args.graft_long_temporal_skip,
        graft_long_global_topk=args.graft_long_global_topk,
        graft_long_edge_threshold=args.graft_long_edge_threshold,
        graft_long_split_radius_eps=args.graft_long_split_radius_eps,
        graft_long_spatial_penalty=args.graft_long_spatial_penalty,
        graft_long_scene_threshold=args.graft_long_scene_threshold,
        dyn_adaptive_adts_budget=args.dyn_adaptive_adts_budget,
        dyn_budget_strength=args.dyn_budget_strength,
        dyn_budget_temperature=args.dyn_budget_temperature,
        dyn_frame_budget_min_ratio=args.dyn_frame_budget_min_ratio,
        dyn_frame_budget_max_ratio=args.dyn_frame_budget_max_ratio,
        dyn_boundary_boost=args.dyn_boundary_boost,
        dyn_adts_beta=args.dyn_adts_beta,
        dyn_attn_weight=args.dyn_attn_weight,
        dyn_event_weight=args.dyn_event_weight,
        dyn_novelty_weight=args.dyn_novelty_weight,
        dyn_detail_weight=args.dyn_detail_weight,
        dyn_density_weight=args.dyn_density_weight,
        dyn_density_topk=args.dyn_density_topk,
        dyn_event_chunk_radius=args.dyn_event_chunk_radius,
        dyn_frame_event_weight=args.dyn_frame_event_weight,
        dyn_frame_novelty_weight=args.dyn_frame_novelty_weight,
        dyn_frame_attn_weight=args.dyn_frame_attn_weight,
        dyn_frame_density_weight=args.dyn_frame_density_weight,
        dyn_frame_detail_weight=args.dyn_frame_detail_weight,
        dyn_similarity_debias=args.dyn_similarity_debias,
        dyn_debias_frame_weight=args.dyn_debias_frame_weight,
        dyn_debias_global_weight=args.dyn_debias_global_weight,
        dyn_sink_tstm=args.dyn_sink_tstm,
        dyn_mutual_nn=args.dyn_mutual_nn,
        dyn_margin_threshold=args.dyn_margin_threshold,
        dyn_high_conf_bonus=args.dyn_high_conf_bonus,
        dyn_weighted_merge=args.dyn_weighted_merge,
        dyn_confidence_attn_weight=args.dyn_confidence_attn_weight,
        dyn_confidence_sim_weight=args.dyn_confidence_sim_weight,
        expansion=args.expansion,
        pruning_layer=pruning_layer,
        llm_retention_ratio=llm_retention_ratio,
        compression_variant="graftvid",
        question_aware_reweighting=False,
        question_reweight_beta=args.question_reweight_beta,
        adaptive_token_budget=False,
        decode_policy=args.decode_policy,
        decode_kv_budget_ratio=args.decode_kv_budget_ratio,
        decode_update_interval=args.decode_update_interval,
        decode_start_layer=args.decode_start_layer,
    )


def _apply_cats(model, args: BenchmarkArgs, backend: str):
    from flashvid import flashvid
    pruning_layer, llm_retention_ratio = _resolve_llm_pruning_args(backend, args)

    return flashvid(
        model=model,
        retention_ratio=args.retention_ratio,
        do_segment=args.do_segment,
        segment_threshold=args.segment_threshold,
        min_segment_num=args.min_segment_num,
        complementary_segment=args.complementary_segment,
        token_selection_method=args.graphvid_token_selection_method or args.token_selection_method,
        alpha=args.alpha,
        temporal_threshold=args.temporal_threshold,
        temporal_merge_mode="cats",
        graph_final_tokens_per_frame=args.graph_final_tokens_per_frame,
        graph_final_frame_floor_ratio=args.graph_final_frame_floor_ratio,
        graph_skip_spatial_merge_when_capped=args.graph_skip_spatial_merge_when_capped,
        cats_adts_beta=args.cats_adts_beta,
        cats_adts_mode=args.cats_adts_mode,
        cats_margin_threshold=args.cats_margin_threshold,
        cats_high_conf_bonus=args.cats_high_conf_bonus,
        cats_mutual_nn=args.cats_mutual_nn,
        cats_confidence_attn_weight=args.cats_confidence_attn_weight,
        cats_confidence_sim_weight=args.cats_confidence_sim_weight,
        cats_anchor_self_weight=args.cats_anchor_self_weight,
        cats_adaptive_adts_budget=args.cats_adaptive_adts_budget,
        cats_frame_budget_min=args.cats_frame_budget_min,
        cats_frame_budget_temperature=args.cats_frame_budget_temperature,
        expansion=args.expansion,
        pruning_layer=pruning_layer,
        llm_retention_ratio=llm_retention_ratio,
        compression_variant="cats",
        question_aware_reweighting=False,
        question_reweight_beta=args.question_reweight_beta,
        adaptive_token_budget=False,
        decode_policy=args.decode_policy,
        decode_kv_budget_ratio=args.decode_kv_budget_ratio,
        decode_update_interval=args.decode_update_interval,
        decode_start_layer=args.decode_start_layer,
    )


def _apply_ours(model, args: BenchmarkArgs, backend: str):
    from flashvid import flashvid
    pruning_layer, llm_retention_ratio = _resolve_llm_pruning_args(backend, args)

    return flashvid(
        model=model,
        retention_ratio=args.retention_ratio,
        do_segment=args.do_segment,
        segment_threshold=args.segment_threshold,
        min_segment_num=args.min_segment_num,
        complementary_segment=args.complementary_segment,
        token_selection_method=args.graphvid_token_selection_method or args.token_selection_method,
        alpha=args.alpha,
        temporal_threshold=args.temporal_threshold,
        temporal_merge_mode=args.temporal_merge_mode,
        graph_temporal_topk=args.graph_temporal_topk,
        graph_temporal_radius=args.graph_temporal_radius,
        graph_temporal_skip=args.graph_temporal_skip,
        graph_merge_protect_ratio=args.graph_merge_protect_ratio,
        graph_merge_target_ratio=args.graph_merge_target_ratio,
        graph_merge_representative=args.graph_merge_representative,
        graph_representative_position=args.graph_representative_position,
        graph_protection_attn_weight=args.graph_protection_attn_weight,
        graph_protection_novelty_weight=args.graph_protection_novelty_weight,
        graph_protection_detail_weight=args.graph_protection_detail_weight,
        graph_adaptive_detail_protection=args.graph_adaptive_detail_protection,
        graph_adaptive_detail_boost=args.graph_adaptive_detail_boost,
        graph_adaptive_protect_boost=args.graph_adaptive_protect_boost,
        graph_merge_importance_penalty=args.graph_merge_importance_penalty,
        graph_respect_temporal_threshold=args.graph_respect_temporal_threshold,
        graph_final_tokens_per_frame=args.graph_final_tokens_per_frame,
        graph_final_frame_floor_ratio=args.graph_final_frame_floor_ratio,
        graph_skip_spatial_merge_when_capped=args.graph_skip_spatial_merge_when_capped,
        graft_temporal_topk=args.graft_temporal_topk,
        graft_temporal_radius=args.graft_temporal_radius,
        graft_temporal_skip=args.graft_temporal_skip,
        graft_global_topk=args.graft_global_topk,
        graft_input_is_residual=args.graft_input_is_residual,
        graft_anchor_ratio=args.graft_anchor_ratio,
        graft_edge_threshold=args.graft_edge_threshold,
        graft_component_radius_eps=args.graft_component_radius_eps,
        graft_split_radius_eps=args.graft_split_radius_eps,
        graft_parent_capacity=args.graft_parent_capacity,
        graft_mutual_knn=args.graft_mutual_knn,
        graft_one_token_per_frame=args.graft_one_token_per_frame,
        graft_spatial_penalty=args.graft_spatial_penalty,
        graft_importance_penalty=args.graft_importance_penalty,
        graft_hub_penalty=args.graft_hub_penalty,
        graft_adaptive_aggregation=args.graft_adaptive_aggregation,
        graft_scene_threshold=args.graft_scene_threshold,
        graft_min_tokens_per_frame=args.graft_min_tokens_per_frame,
        graft_budget_correction=args.graft_budget_correction,
        graft_budget_diversity_weight=args.graft_budget_diversity_weight,
        graft_score_preset=args.graft_score_preset,
        graft_duration_aware=args.graft_duration_aware,
        graft_medium_temporal_skip=args.graft_medium_temporal_skip,
        graft_medium_global_topk=args.graft_medium_global_topk,
        graft_medium_edge_threshold=args.graft_medium_edge_threshold,
        graft_medium_split_radius_eps=args.graft_medium_split_radius_eps,
        graft_medium_spatial_penalty=args.graft_medium_spatial_penalty,
        graft_medium_scene_threshold=args.graft_medium_scene_threshold,
        graft_long_temporal_skip=args.graft_long_temporal_skip,
        graft_long_global_topk=args.graft_long_global_topk,
        graft_long_edge_threshold=args.graft_long_edge_threshold,
        graft_long_split_radius_eps=args.graft_long_split_radius_eps,
        graft_long_spatial_penalty=args.graft_long_spatial_penalty,
        graft_long_scene_threshold=args.graft_long_scene_threshold,
        dyn_adaptive_adts_budget=args.dyn_adaptive_adts_budget,
        dyn_budget_strength=args.dyn_budget_strength,
        dyn_budget_temperature=args.dyn_budget_temperature,
        dyn_frame_budget_min_ratio=args.dyn_frame_budget_min_ratio,
        dyn_frame_budget_max_ratio=args.dyn_frame_budget_max_ratio,
        dyn_boundary_boost=args.dyn_boundary_boost,
        dyn_adts_beta=args.dyn_adts_beta,
        dyn_attn_weight=args.dyn_attn_weight,
        dyn_event_weight=args.dyn_event_weight,
        dyn_novelty_weight=args.dyn_novelty_weight,
        dyn_detail_weight=args.dyn_detail_weight,
        dyn_density_weight=args.dyn_density_weight,
        dyn_density_topk=args.dyn_density_topk,
        dyn_event_chunk_radius=args.dyn_event_chunk_radius,
        dyn_frame_event_weight=args.dyn_frame_event_weight,
        dyn_frame_novelty_weight=args.dyn_frame_novelty_weight,
        dyn_frame_attn_weight=args.dyn_frame_attn_weight,
        dyn_frame_density_weight=args.dyn_frame_density_weight,
        dyn_frame_detail_weight=args.dyn_frame_detail_weight,
        dyn_similarity_debias=args.dyn_similarity_debias,
        dyn_debias_frame_weight=args.dyn_debias_frame_weight,
        dyn_debias_global_weight=args.dyn_debias_global_weight,
        dyn_sink_tstm=args.dyn_sink_tstm,
        dyn_mutual_nn=args.dyn_mutual_nn,
        dyn_margin_threshold=args.dyn_margin_threshold,
        dyn_high_conf_bonus=args.dyn_high_conf_bonus,
        dyn_weighted_merge=args.dyn_weighted_merge,
        dyn_confidence_attn_weight=args.dyn_confidence_attn_weight,
        dyn_confidence_sim_weight=args.dyn_confidence_sim_weight,
        learn_selector_ckpt=args.learn_selector_ckpt,
        learn_qaware=args.learn_qaware,
        learn_stable_floor_ratio=args.learn_stable_floor_ratio,
        learn_score_blend=args.learn_score_blend,
        learn_q_relevance_weight=args.learn_q_relevance_weight,
        learn_density_topk=args.learn_density_topk,
        learn_collect_teacher=args.learn_collect_teacher,
        expansion=args.expansion,
        pruning_layer=pruning_layer,
        llm_retention_ratio=llm_retention_ratio,
        compression_variant=args.compression_variant,
        question_aware_reweighting=args.question_aware_reweighting,
        question_reweight_beta=args.question_reweight_beta,
        adaptive_token_budget=args.adaptive_token_budget,
        adaptive_budget_low=args.adaptive_budget_low,
        adaptive_budget_mid=args.adaptive_budget_mid,
        adaptive_budget_high=args.adaptive_budget_high,
        talon_adaptive_target_low=args.talon_adaptive_target_low,
        talon_adaptive_target_mid=args.talon_adaptive_target_mid,
        talon_adaptive_target_high=args.talon_adaptive_target_high,
        talon_complexity_floor=args.talon_complexity_floor,
        talon_complexity_ceil=args.talon_complexity_ceil,
        talon_adaptive_gamma=args.talon_adaptive_gamma,
        talon_transport_radius=args.talon_transport_radius,
        talon_rank_ratio=args.talon_rank_ratio,
        talon_rank_min=args.talon_rank_min,
        talon_rank_max=args.talon_rank_max,
        talon_budget_scale=args.talon_budget_scale,
        talon_target_tokens_per_frame=args.talon_target_tokens_per_frame,
        talon_short_target_tokens_per_frame=args.talon_short_target_tokens_per_frame,
        talon_medium_target_tokens_per_frame=args.talon_medium_target_tokens_per_frame,
        talon_long_target_tokens_per_frame=args.talon_long_target_tokens_per_frame,
        talon_min_total_tokens=args.talon_min_total_tokens,
        talon_fast_rank_plan=args.talon_fast_rank_plan,
        talon_background_max_ratio=args.talon_background_max_ratio,
        talon_frame_balanced_selection=args.talon_frame_balanced_selection,
        talon_basis_method=args.talon_basis_method,
        talon_basis_oversample=args.talon_basis_oversample,
        talon_innovation_attention_weight=args.talon_innovation_attention_weight,
        talon_motion_importance_weight=args.talon_motion_importance_weight,
        talon_boundary_importance_weight=args.talon_boundary_importance_weight,
        talon_question_frame_weight=args.talon_question_frame_weight,
        talon_frame_balanced_memory=args.talon_frame_balanced_memory,
        talon_memory_mode=args.talon_memory_mode,
        talon_anchor_safety_ratio=args.talon_anchor_safety_ratio,
        talon_anchor_diversity_weight=args.talon_anchor_diversity_weight,
        talon_anchor_candidate_multiplier=args.talon_anchor_candidate_multiplier,
        talon_spatial_anchor_coverage=args.talon_spatial_anchor_coverage,
        talon_spatial_anchor_ratio=args.talon_spatial_anchor_ratio,
        talon_spatial_anchor_rows=args.talon_spatial_anchor_rows,
        talon_spatial_anchor_cols=args.talon_spatial_anchor_cols,
        talon_spatial_anchor_score=args.talon_spatial_anchor_score,
        talon_spatial_anchor_apply_to_short=args.talon_spatial_anchor_apply_to_short,
        talon_frame_coverage_floor_ratio=args.talon_frame_coverage_floor_ratio,
        talon_frame_importance_pooling=args.talon_frame_importance_pooling,
        talon_frame_importance_topk=args.talon_frame_importance_topk,
        talon_medium_frame_coverage_floor_ratio=args.talon_medium_frame_coverage_floor_ratio,
        talon_long_frame_coverage_floor_ratio=args.talon_long_frame_coverage_floor_ratio,
        talon_frame_local_budget_ratio=args.talon_frame_local_budget_ratio,
        talon_question_recall_ratio=args.talon_question_recall_ratio,
        talon_question_recall_qweight=args.talon_question_recall_qweight,
        talon_persistence_recall_ratio=args.talon_persistence_recall_ratio,
        talon_persistence_recall_qweight=args.talon_persistence_recall_qweight,
        talon_persistence_recall_pweight=args.talon_persistence_recall_pweight,
        talon_persistence_apply_to_short=args.talon_persistence_apply_to_short,
        talon_persistence_apply_to_medium=args.talon_persistence_apply_to_medium,
        talon_persistence_apply_to_long=args.talon_persistence_apply_to_long,
        talon_object_evidence_ratio=args.talon_object_evidence_ratio,
        talon_object_evidence_qweight=args.talon_object_evidence_qweight,
        talon_object_evidence_sweight=args.talon_object_evidence_sweight,
        talon_object_evidence_pweight=args.talon_object_evidence_pweight,
        talon_object_evidence_apply_to_short=args.talon_object_evidence_apply_to_short,
        talon_object_evidence_apply_to_medium=args.talon_object_evidence_apply_to_medium,
        talon_object_evidence_apply_to_long=args.talon_object_evidence_apply_to_long,
        talon_question_pooling=args.talon_question_pooling,
        talon_question_pooling_topk=args.talon_question_pooling_topk,
        talon_question_contrast_weight=args.talon_question_contrast_weight,
        talon_question_contrast_apply_to_short=args.talon_question_contrast_apply_to_short,
        talon_monotonic_base_tokens_per_frame=args.talon_monotonic_base_tokens_per_frame,
        talon_budget_strategy=args.talon_budget_strategy,
        talon_budget_mode=args.talon_budget_mode,
        talon_transport_mode=args.talon_transport_mode,
        talon_transport_temperature=args.talon_transport_temperature,
        talon_lite_enabled=args.talon_lite_enabled,
        talon_echo_temperature=args.talon_echo_temperature,
        talon_echo_topk_neighbors=args.talon_echo_topk_neighbors,
        talon_echo_residual_weight=args.talon_echo_residual_weight,
        talon_echo_score_mode=args.talon_echo_score_mode,
        talon_rd_spectral_weight=args.talon_rd_spectral_weight,
        talon_rd_innovation_weight=args.talon_rd_innovation_weight,
        talon_use_question_innovation=args.talon_use_question_innovation,
        talon_innovation_qweight=args.talon_innovation_qweight,
        talon_output_mode=args.talon_output_mode,
        talon_reconstruction_blend=args.talon_reconstruction_blend,
        talon_anchor_score_weight=args.talon_anchor_score_weight,
        talon_min_anchor_per_frame=args.talon_min_anchor_per_frame,
        talon_passthrough_ratio=args.talon_passthrough_ratio,
        talon_passthrough_min=args.talon_passthrough_min,
        talon_use_segmentation=args.talon_use_segmentation,
        talon_disable_oversegmentation=args.talon_disable_oversegmentation,
        talon_max_segments=args.talon_max_segments,
        talon_deepstack_mode=args.talon_deepstack_mode,
        memory_token_ratio=args.memory_token_ratio,
        memory_token_min=args.memory_token_min,
        memory_token_max=args.memory_token_max,
        talon_adaptive_target_enabled=args.talon_adaptive_target_enabled,
        talon_force_fixed_target=args.talon_force_fixed_target,
        talon_target_mean_cap=args.talon_target_mean_cap,
        talon_unified_selection=args.talon_unified_selection,
        talon_low_budget_mode_threshold=args.talon_low_budget_mode_threshold,
        talon_low_budget_rank_cap=args.talon_low_budget_rank_cap,
        talon_background_global_ratio=args.talon_background_global_ratio,
        talon_event_budget_ratio=args.talon_event_budget_ratio,
        talon_memory_fused_weight=args.talon_memory_fused_weight,
        talon_memory_residual_weight=args.talon_memory_residual_weight,
        talon_memory_frame_weight=args.talon_memory_frame_weight,
        talon_recall_memory_mode=args.talon_recall_memory_mode,
        talon_final_fused_weight=args.talon_final_fused_weight,
        talon_final_residual_weight=args.talon_final_residual_weight,
        talon_final_frame_weight=args.talon_final_frame_weight,
        talon_anchor_keep_bonus=args.talon_anchor_keep_bonus,
        talon_recall_keep_bonus=args.talon_recall_keep_bonus,
        talon_event_keep_bonus=args.talon_event_keep_bonus,
        talon_legacy_base_keep_ratio=args.talon_legacy_base_keep_ratio,
        talon_prior_candidate_ratio=args.talon_prior_candidate_ratio,
        talon_prior_keep_bonus=args.talon_prior_keep_bonus,
        talon_flash_prior_channel_ratio=args.talon_flash_prior_channel_ratio,
        talon_flash_prior_channel_method=args.talon_flash_prior_channel_method,
        talon_flash_prior_channel_min_per_frame=args.talon_flash_prior_channel_min_per_frame,
        talon_flash_prior_channel_max_per_frame=args.talon_flash_prior_channel_max_per_frame,
        talon_flash_prior_channel_bonus=args.talon_flash_prior_channel_bonus,
        talon_final_anchor_min_ratio=args.talon_final_anchor_min_ratio,
        talon_final_recall_min_ratio=args.talon_final_recall_min_ratio,
        talon_force_anchor_recall_quota=args.talon_force_anchor_recall_quota,
        talon_global_topk_ratio=args.talon_global_topk_ratio,
        talon_rescue_enabled=args.talon_rescue_enabled,
        talon_rescue_ratio=args.talon_rescue_ratio,
        talon_rescue_from_memory_only=args.talon_rescue_from_memory_only,
        talon_rescue_fused_weight=args.talon_rescue_fused_weight,
        talon_rescue_residual_weight=args.talon_rescue_residual_weight,
        talon_rescue_frame_weight=args.talon_rescue_frame_weight,
        talon_rescue_global_ratio=args.talon_rescue_global_ratio,
        talon_rerank_with_flash_prior=args.talon_rerank_with_flash_prior,
        talon_flash_prior_ratio=args.talon_flash_prior_ratio,
        talon_recall_semantic_ratio=args.talon_recall_semantic_ratio,
        talon_recall_event_ratio=args.talon_recall_event_ratio,
        talon_recall_frame_ratio=args.talon_recall_frame_ratio,
        talon_recall_global_ratio=args.talon_recall_global_ratio,
        talon_duration_aware=args.talon_duration_aware,
        talon_medium_anchor_safety_ratio=args.talon_medium_anchor_safety_ratio,
        talon_medium_event_budget_ratio=args.talon_medium_event_budget_ratio,
        talon_medium_global_topk_ratio=args.talon_medium_global_topk_ratio,
        talon_long_anchor_safety_ratio=args.talon_long_anchor_safety_ratio,
        talon_long_event_budget_ratio=args.talon_long_event_budget_ratio,
        talon_long_global_topk_ratio=args.talon_long_global_topk_ratio,
        talon_task_aware_event=args.talon_task_aware_event,
        talon_task_event_attention_weight=args.talon_task_event_attention_weight,
        talon_task_event_qweight=args.talon_task_event_qweight,
        talon_visual_task_balance=args.talon_visual_task_balance,
        talon_visual_task_anchor_ratio=args.talon_visual_task_anchor_ratio,
        talon_visual_task_event_ratio=args.talon_visual_task_event_ratio,
        talon_visual_task_recall_ratio=args.talon_visual_task_recall_ratio,
        talon_knowledge_visual_anchor_ratio=args.talon_knowledge_visual_anchor_ratio,
        talon_knowledge_visual_event_ratio=args.talon_knowledge_visual_event_ratio,
        talon_knowledge_visual_recall_ratio=args.talon_knowledge_visual_recall_ratio,
        talon_adaptive_router=args.talon_adaptive_router,
        talon_router_apply_to_short=args.talon_router_apply_to_short,
        talon_router_visual_anchor_ratio=args.talon_router_visual_anchor_ratio,
        talon_router_visual_event_ratio=args.talon_router_visual_event_ratio,
        talon_router_visual_recall_ratio=args.talon_router_visual_recall_ratio,
        talon_router_temporal_anchor_ratio=args.talon_router_temporal_anchor_ratio,
        talon_router_temporal_event_ratio=args.talon_router_temporal_event_ratio,
        talon_router_temporal_recall_ratio=args.talon_router_temporal_recall_ratio,
        talon_router_balanced_anchor_ratio=args.talon_router_balanced_anchor_ratio,
        talon_router_balanced_event_ratio=args.talon_router_balanced_event_ratio,
        talon_router_balanced_recall_ratio=args.talon_router_balanced_recall_ratio,
        talon_router_visual_concentration_threshold=args.talon_router_visual_concentration_threshold,
        talon_router_low_residual_threshold=args.talon_router_low_residual_threshold,
        talon_router_temporal_entropy_threshold=args.talon_router_temporal_entropy_threshold,
        talon_router_temporal_residual_threshold=args.talon_router_temporal_residual_threshold,
        talon_temporal_chunk_aware=args.talon_temporal_chunk_aware,
        talon_temporal_num_chunks=args.talon_temporal_num_chunks,
        talon_temporal_chunk_min_ratio=args.talon_temporal_chunk_min_ratio,
        talon_temporal_chunk_score=args.talon_temporal_chunk_score,
        talon_track_aware=args.talon_track_aware,
        talon_track_budget_ratio=args.talon_track_budget_ratio,
        talon_track_tokens_per_slot=args.talon_track_tokens_per_slot,
        talon_track_score=args.talon_track_score,
        talon_absorb_dropped_tokens=args.talon_absorb_dropped_tokens,
        talon_absorb_ratio=args.talon_absorb_ratio,
        talon_absorb_alpha=args.talon_absorb_alpha,
        talon_absorb_score=args.talon_absorb_score,
        talon_summary_replacement=args.talon_summary_replacement,
        talon_summary_raw_swap=args.talon_summary_raw_swap,
        talon_summary_ratio=args.talon_summary_ratio,
        talon_summary_num_chunks=args.talon_summary_num_chunks,
        talon_summary_pool_topk=args.talon_summary_pool_topk,
        talon_summary_alpha=args.talon_summary_alpha,
        talon_summary_score=args.talon_summary_score,
        decode_policy=args.decode_policy,
        decode_kv_budget_ratio=args.decode_kv_budget_ratio,
        decode_update_interval=args.decode_update_interval,
        decode_start_layer=args.decode_start_layer,
    )


def _print_header(args: BenchmarkArgs, backend: str):
    effective_attn = _resolve_attn_implementation(args.attn_implementation)
    ours_phase_name = _ours_phase_key(args)
    print(SEPARATOR)
    print("Unified Benchmark: Accuracy + Token + Latency + Speedup")
    print(SEPARATOR)
    print(f"Model path    : {args.model_path}")
    print(f"Backend       : {backend}")
    print(f"Attention impl: {effective_attn} (requested: {args.attn_implementation})")
    print(f"Dataset       : {args.dataset_jsonl}")
    print(f"HF_HOME       : {args.hf_home or os.getenv('HF_HOME', '~/.cache/huggingface')}")
    print(f"Start index   : {args.start_index}")
    print(f"Limit         : {args.limit}")
    print(f"Shuffle       : {args.shuffle}")
    print(f"Frames        : {args.num_frames}")
    print(f"Warmup/Runs   : {args.num_warmup}/{args.num_runs}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(
        "Run phases    : "
        f"baseline={args.run_baseline}, flashvid={args.run_flashvid}, "
        f"{ours_phase_name}={args.run_ours}, graphvid={args.run_graphvid}, "
        f"graftvid={args.run_graftvid}, cats={args.run_cats}"
    )
    print(f"Phase reload  : {args.reload_model_each_phase}")
    if args.run_ours:
        duration_targets = (
            f"{args.talon_short_target_tokens_per_frame}/"
            f"{args.talon_medium_target_tokens_per_frame}/"
            f"{args.talon_long_target_tokens_per_frame}"
        )
        print(
            f"{_phase_display_name(ours_phase_name)} config: "
            f"variant={args.compression_variant}, qa={args.question_aware_reweighting}, "
            f"temporal_merge={args.temporal_merge_mode}, "
            f"adaptive={args.adaptive_token_budget}, budget={args.talon_budget_strategy}, "
            f"scale={args.talon_budget_scale}, target_per_frame={args.talon_target_tokens_per_frame}, "
            f"duration_targets={duration_targets}, "
            f"event_cap={args.talon_event_budget_ratio:.2f}, "
            f"anchor_div={args.talon_anchor_diversity_weight:.2f}"
        )
        if ours_phase_name == "dynflashvid":
            print(
                "DynFlash allocator: "
                f"strength={args.dyn_budget_strength:.2f}, temp={args.dyn_budget_temperature:.2f}, "
                f"minmax={args.dyn_frame_budget_min_ratio:.2f}/{args.dyn_frame_budget_max_ratio:.2f}, "
                f"token_w=a{args.dyn_attn_weight:.2f}/e{args.dyn_event_weight:.2f}/"
                f"n{args.dyn_novelty_weight:.2f}/d{args.dyn_detail_weight:.2f}/"
                f"rho{args.dyn_density_weight:.2f}, "
                f"frame_w=event{args.dyn_frame_event_weight:.2f}/nov{args.dyn_frame_novelty_weight:.2f}/"
                f"attn{args.dyn_frame_attn_weight:.2f}/rho{args.dyn_frame_density_weight:.2f}/"
                f"detail{args.dyn_frame_detail_weight:.2f}, "
                f"density_topk={args.dyn_density_topk}, event_radius={args.dyn_event_chunk_radius}"
            )
        if ours_phase_name == "learnflashvid":
            print(
                "LearnFlash selector: "
                f"ckpt={args.learn_selector_ckpt or '<heuristic>'}, qaware={args.learn_qaware}, "
                f"stable_floor={args.learn_stable_floor_ratio:.2f}, blend={args.learn_score_blend:.2f}, "
                f"q_weight={args.learn_q_relevance_weight:.2f}, density_topk={args.learn_density_topk}, "
                f"collect_teacher={args.learn_collect_teacher}"
            )
    if args.run_graphvid:
        print(
            "GraphVID config: "
            f"merge=graph, topk={args.graph_temporal_topk}, radius={args.graph_temporal_radius}, "
            f"skip={args.graph_temporal_skip}, protect={args.graph_merge_protect_ratio:.2f}, "
            f"target_ratio={args.graph_merge_target_ratio:.2f}, final_tpf={args.graph_final_tokens_per_frame}, "
            f"skip_spatial={args.graph_skip_spatial_merge_when_capped}, "
            f"rep={args.graph_merge_representative}, "
            f"pos={args.graph_representative_position}, "
            f"detail_w={args.graph_protection_detail_weight:.2f}, "
            f"adaptive_detail={args.graph_adaptive_detail_protection}, "
            f"penalty={args.graph_merge_importance_penalty:.2f}, "
            f"respect_thr={args.graph_respect_temporal_threshold}"
        )
    if args.run_graftvid:
        print(
            "GRAFT-VID config: "
            f"merge=graft, topk={args.graft_temporal_topk}, radius={args.graft_temporal_radius}, "
            f"skip={args.graft_temporal_skip}, global_topk={args.graft_global_topk}, "
            f"residual_input={args.graft_input_is_residual}, "
            f"anchor={(args.graft_anchor_ratio if args.graft_anchor_ratio is not None else (0.15 if args.graft_input_is_residual else 0.65)):.2f}, "
            f"edge_thr={args.graft_edge_threshold:.2f}, radius_eps={args.graft_component_radius_eps:.3f}, "
            f"split_eps={args.graft_split_radius_eps:.3f}, capacity={args.graft_parent_capacity}, "
            f"mutual={args.graft_mutual_knn}, one_frame={args.graft_one_token_per_frame}, "
            f"spatial_pen={args.graft_spatial_penalty:.2f}, imp_pen={args.graft_importance_penalty:.2f}, "
            f"hub_pen={args.graft_hub_penalty:.2f}, adaptive={args.graft_adaptive_aggregation}, "
            f"scene_thr={args.graft_scene_threshold:.2f}, minpf={args.graft_min_tokens_per_frame}, "
            f"budget_fix={args.graft_budget_correction}, budget_div={args.graft_budget_diversity_weight:.2f}, "
            f"score={args.graft_score_preset}, dur_aware={args.graft_duration_aware}"
        )
    if args.run_cats:
        print(
            "CATS config  : "
            f"adts={args.cats_adts_mode}, beta={args.cats_adts_beta:.3f}, margin={args.cats_margin_threshold:.3f}, "
            f"high_bonus={args.cats_high_conf_bonus:.3f}, mutual={args.cats_mutual_nn}, "
            f"attn_w={args.cats_confidence_attn_weight:.2f}, sim_w={args.cats_confidence_sim_weight:.2f}, "
            f"anchor_w={args.cats_anchor_self_weight:.2f}, "
            f"adaptive_budget={args.cats_adaptive_adts_budget}, minpf={args.cats_frame_budget_min}, "
            f"temp={args.cats_frame_budget_temperature:.2f}"
        )
    print(SEPARATOR)


def _print_summary(summary: dict[str, Any]):
    print(SEPARATOR)
    print("Summary")
    print(SEPARATOR)
    for phase_name in _phase_order(summary):
        phase = summary.get(phase_name)
        if phase is None:
            continue
        acc = phase["accuracy"]
        acc_text = f"{acc * 100:.2f}%" if acc is not None else "N/A"
        print(f"[{phase_name}] valid={phase['num_valid']}/{phase['num_samples']} acc={acc_text}")
        lat_mean = phase["latency_ms"]["mean"]
        vt_mean = phase["compressed_visual_tokens"]["mean"]
        vision_vt_mean = phase["vision_compressed_visual_tokens"]["mean"]
        talon_target_mean = phase.get("talon_target_tokens_per_frame", {}).get("mean")
        talon_complexity_mean = phase.get("talon_complexity_score", {}).get("mean")
        talon_budget_mean = phase.get("talon_target_budget", {}).get("mean")
        talon_anchor_mean = phase.get("talon_anchor_tokens", {}).get("mean")
        talon_rank_mean = phase.get("talon_rank_tokens", {}).get("mean")
        talon_event_mean = phase.get("talon_event_tokens", {}).get("mean")
        talon_recall_mean = phase.get("talon_recall_tokens", {}).get("mean")
        talon_persistence_mean = phase.get("talon_persistence_tokens", {}).get("mean")
        talon_object_mean = phase.get("talon_object_tokens", {}).get("mean")
        talon_memory_mean = phase.get("talon_memory_tokens", {}).get("mean")
        talon_rank_cap_mean = phase.get("talon_rank_cap", {}).get("mean")
        talon_chosen_rank_mean = phase.get("talon_chosen_rank", {}).get("mean")
        talon_dup_mean = phase.get("talon_duplicate_index_count", {}).get("mean")
        talon_question_active_mean = phase.get("talon_question_aware_active", {}).get("mean")
        talon_router_mode_mean = phase.get("talon_router_mode_code", {}).get("mean")
        talon_router_fused_mean = phase.get("talon_router_fused_concentration", {}).get("mean")
        talon_router_resid_mean = phase.get("talon_router_residual_concentration", {}).get("mean")
        talon_router_q_mean = phase.get("talon_router_question_concentration", {}).get("mean")
        talon_router_entropy_mean = phase.get("talon_router_frame_entropy", {}).get("mean")
        talon_core_budget_mean = phase.get("talon_core_target_budget", {}).get("mean")
        talon_core_residual_mean = phase.get("talon_core_residual_mean", {}).get("mean")
        talon_core_semantic_mean = phase.get("talon_core_semantic_tokens", {}).get("mean")
        talon_core_innovation_mean = phase.get("talon_core_innovation_tokens", {}).get("mean")
        talon_core_dup_mean = phase.get("talon_core_duplicate_index_count", {}).get("mean")
        talon_core_question_active_mean = phase.get("talon_core_question_aware_active", {}).get("mean")
        talon_core_budget_min_mean = phase.get("talon_core_budget_min", {}).get("mean")
        talon_core_budget_max_mean = phase.get("talon_core_budget_max", {}).get("mean")
        talon_core_grid_h_mean = phase.get("talon_core_grid_h", {}).get("mean")
        talon_core_grid_w_mean = phase.get("talon_core_grid_w", {}).get("mean")
        graft_count_mean = phase.get("graft_component_count", {}).get("mean")
        graft_num_nodes_mean = phase.get("graft_num_nodes", {}).get("mean")
        graft_target_mean = phase.get("graft_target_components", {}).get("mean")
        graft_protected_mean = phase.get("graft_protected_count", {}).get("mean")
        graft_entries_before_mean = phase.get("graft_entries_before_budget", {}).get("mean")
        graft_entries_after_mean = phase.get("graft_entries_after_budget", {}).get("mean")
        graft_anchor_mean = phase.get("graft_anchor_ratio", {}).get("mean")
        graft_residual_mean = phase.get("graft_input_is_residual", {}).get("mean")
        graft_budget_div_mean = phase.get("graft_budget_diversity_weight", {}).get("mean")
        graft_score_preset_mean = phase.get("graft_score_preset_code", {}).get("mean")
        graft_budget_active_mean = phase.get("graft_budget_correction_active", {}).get("mean")
        graft_protected_kept_mean = phase.get("graft_protected_kept_count", {}).get("mean")
        graft_size_mean = phase.get("graft_avg_component_size", {}).get("mean")
        graft_max_size_mean = phase.get("graft_max_component_size", {}).get("mean")
        graft_radius_mean = phase.get("graft_radius_mean", {}).get("mean")
        graft_radius_max_mean = phase.get("graft_radius_max", {}).get("mean")
        graft_edges_mean = phase.get("graft_edges_considered", {}).get("mean")
        graft_accept_mean = phase.get("graft_edges_accepted", {}).get("mean")
        graft_mutual_rej_mean = phase.get("graft_mutual_rejected", {}).get("mean")
        graft_radius_rej_mean = phase.get("graft_radius_rejected", {}).get("mean")
        graft_capacity_rej_mean = phase.get("graft_capacity_rejected", {}).get("mean")
        graft_same_frame_rej_mean = phase.get("graft_same_frame_rejected", {}).get("mean")
        cats_selected_mean = phase.get("cats_selected_tokens", {}).get("mean")
        cats_sink_mean = phase.get("cats_sink_merges", {}).get("mean")
        cats_residual_mean = phase.get("cats_residual_merges", {}).get("mean")
        cats_mutual_rej_mean = phase.get("cats_mutual_rejected", {}).get("mean")
        cats_margin_rej_mean = phase.get("cats_margin_rejected", {}).get("mean")
        cats_threshold_rej_mean = phase.get("cats_threshold_rejected", {}).get("mean")
        cats_retained_mean = phase.get("cats_retained_residual_tokens", {}).get("mean")
        cats_before_mean = phase.get("cats_spatial_tokens_before", {}).get("mean")
        cats_after_mean = phase.get("cats_spatial_tokens_after", {}).get("mean")
        cats_sim_mean = phase.get("cats_mean_merge_sim", {}).get("mean")
        cats_margin_mean = phase.get("cats_mean_margin", {}).get("mean")
        dyn_selected_mean = phase.get("dyn_selected_tokens", {}).get("mean")
        dyn_budget_min_mean = phase.get("dyn_budget_min", {}).get("mean")
        dyn_budget_max_mean = phase.get("dyn_budget_max", {}).get("mean")
        dyn_budget_std_mean = phase.get("dyn_budget_std", {}).get("mean")
        dyn_sink_mean = phase.get("dyn_sink_merges", {}).get("mean")
        dyn_residual_mean = phase.get("dyn_residual_merges", {}).get("mean")
        dyn_retained_mean = phase.get("dyn_retained_residual_tokens", {}).get("mean")
        dyn_before_mean = phase.get("dyn_spatial_tokens_before", {}).get("mean")
        dyn_after_mean = phase.get("dyn_spatial_tokens_after", {}).get("mean")
        dyn_sim_mean = phase.get("dyn_mean_merge_sim", {}).get("mean")
        dyn_debias_mean = phase.get("dyn_similarity_debias_active", {}).get("mean")
        dyn_sink_active_mean = phase.get("dyn_sink_active", {}).get("mean")
        dyn_weighted_mean = phase.get("dyn_weighted_active", {}).get("mean")
        dyn_density_frame_mean = phase.get("dyn_density_frame_mean", {}).get("mean")
        dyn_event_chunk_mean = phase.get("dyn_event_chunk_mean", {}).get("mean")
        dyn_motion_frame_mean = phase.get("dyn_motion_frame_mean", {}).get("mean")
        hedge_budget_mean = phase.get("hedge_residual_budget", {}).get("mean")
        hedge_stable_cand_mean = phase.get("hedge_stable_candidates", {}).get("mean")
        hedge_evidence_cand_mean = phase.get("hedge_evidence_candidates", {}).get("mean")
        hedge_stable_sel_mean = phase.get("hedge_stable_selected", {}).get("mean")
        hedge_evidence_sel_mean = phase.get("hedge_evidence_selected", {}).get("mean")
        hedge_floor_mean = phase.get("hedge_stable_floor_ratio", {}).get("mean")
        hedge_div_mean = phase.get("hedge_diversity_weight", {}).get("mean")
        learn_selected_mean = phase.get("learn_selected_tokens", {}).get("mean")
        learn_stable_mean = phase.get("learn_stable_tokens", {}).get("mean")
        learn_selector_mean = phase.get("learn_selector_tokens", {}).get("mean")
        learn_qaware_mean = phase.get("learn_qaware_active", {}).get("mean")
        learn_score_mean = phase.get("learn_score_mean", {}).get("mean")
        learn_score_std_mean = phase.get("learn_score_std", {}).get("mean")
        learn_teacher_keep_mean = phase.get("learn_teacher_keep_ratio", {}).get("mean")
        red_mean = phase["visual_token_reduction_ratio"]["mean"]
        vision_red_mean = phase["vision_visual_token_reduction_ratio"]["mean"]
        if lat_mean is not None:
            print(f"  latency mean: {lat_mean:.2f} ms")
        if vt_mean is not None:
            print(f"  final visual tokens mean: {vt_mean:.2f}")
        if vision_vt_mean is not None:
            print(f"  vision-side tokens mean: {vision_vt_mean:.2f}")
        if talon_target_mean is not None:
            print(f"  talon target/frame mean: {talon_target_mean:.2f}")
        if talon_complexity_mean is not None:
            print(f"  talon complexity mean: {talon_complexity_mean:.4f}")
        if talon_budget_mean is not None:
            print(f"  talon target budget mean: {talon_budget_mean:.2f}")
        if talon_anchor_mean is not None:
            print(f"  talon anchor/event/recall mean: {talon_anchor_mean:.2f}/{(talon_event_mean or 0.0):.2f}/{(talon_recall_mean or 0.0):.2f}")
        if talon_persistence_mean is not None and talon_persistence_mean > 0:
            print(f"  talon persistence recall mean: {talon_persistence_mean:.2f}")
        if talon_object_mean is not None and talon_object_mean > 0:
            print(f"  talon object evidence mean: {talon_object_mean:.2f}")
        if talon_rank_mean is not None:
            print(f"  talon rank/memory mean: {talon_rank_mean:.2f}/{(talon_memory_mean or 0.0):.2f}")
        if talon_rank_cap_mean is not None:
            print(f"  talon rank cap/chosen mean: {talon_rank_cap_mean:.2f}/{(talon_chosen_rank_mean or 0.0):.2f}")
        if talon_dup_mean is not None:
            print(f"  talon duplicate index mean: {talon_dup_mean:.2f}")
        if talon_question_active_mean is not None:
            print(f"  talon question-aware active mean: {talon_question_active_mean:.2f}")
        if talon_router_mode_mean is not None and talon_router_mode_mean > 0:
            print(f"  talon router mode code mean: {talon_router_mode_mean:.2f} (1=visual,2=temporal,3=balanced)")
        if talon_router_fused_mean is not None:
            print(
                "  talon router fused/residual/question/entropy mean: "
                f"{talon_router_fused_mean:.3f}/{(talon_router_resid_mean or 0.0):.3f}/"
                f"{(talon_router_q_mean or 0.0):.3f}/{(talon_router_entropy_mean or 0.0):.3f}"
            )
        if talon_core_budget_mean is not None:
            print(f"  talon-core target budget mean: {talon_core_budget_mean:.2f}")
        if talon_core_residual_mean is not None:
            print(f"  talon-core residual mean: {talon_core_residual_mean:.6f}")
        if talon_core_semantic_mean is not None:
            print(
                "  talon-core semantic/innovation mean: "
                f"{talon_core_semantic_mean:.2f}/{(talon_core_innovation_mean or 0.0):.2f}"
            )
        if talon_core_dup_mean is not None:
            print(f"  talon-core duplicate index mean: {talon_core_dup_mean:.2f}")
        if talon_core_question_active_mean is not None:
            print(f"  talon-core question-aware active mean: {talon_core_question_active_mean:.2f}")
        if talon_core_budget_min_mean is not None:
            print(
                "  talon-core frame budget min/max mean: "
                f"{talon_core_budget_min_mean:.2f}/{(talon_core_budget_max_mean or 0.0):.2f}"
            )
        if talon_core_grid_h_mean is not None:
            print(f"  talon-core grid H/W mean: {talon_core_grid_h_mean:.2f}/{(talon_core_grid_w_mean or 0.0):.2f}")
        if graft_count_mean is not None:
            if graft_num_nodes_mean is not None:
                print(
                    "  graft nodes/target/protected/entries pre-post mean: "
                    f"{graft_num_nodes_mean:.2f}/{(graft_target_mean or 0.0):.2f}/"
                    f"{(graft_protected_mean or 0.0):.2f}/{(graft_entries_before_mean or 0.0):.2f}/"
                    f"{(graft_entries_after_mean or 0.0):.2f}"
                )
            if graft_anchor_mean is not None:
                print(
                    "  graft anchor/residual-input/budget-div/score/fix/kept-protect mean: "
                    f"{graft_anchor_mean:.3f}/{(graft_residual_mean or 0.0):.2f}/{(graft_budget_div_mean or 0.0):.2f}/"
                    f"{(graft_score_preset_mean or 0.0):.2f}/{(graft_budget_active_mean or 0.0):.2f}/"
                    f"{(graft_protected_kept_mean or 0.0):.2f}"
                )
            print(
                "  graft components avg/max/radius mean: "
                f"{graft_count_mean:.2f}/{(graft_size_mean or 0.0):.2f}/{(graft_max_size_mean or 0.0):.2f}/"
                f"{(graft_radius_mean or 0.0):.4f}"
            )
        if graft_radius_max_mean is not None:
            print(f"  graft radius max mean: {graft_radius_max_mean:.4f}")
        if graft_edges_mean is not None:
            print(
                "  graft edges considered/accepted/rej(m/r/c/sf) mean: "
                f"{graft_edges_mean:.2f}/{(graft_accept_mean or 0.0):.2f}/"
                f"{(graft_mutual_rej_mean or 0.0):.2f}/{(graft_radius_rej_mean or 0.0):.2f}/"
                f"{(graft_capacity_rej_mean or 0.0):.2f}/{(graft_same_frame_rej_mean or 0.0):.2f}"
            )
        if cats_selected_mean is not None:
            print(
                "  cats selected/sink/residual/retained mean: "
                f"{cats_selected_mean:.2f}/{(cats_sink_mean or 0.0):.2f}/"
                f"{(cats_residual_mean or 0.0):.2f}/{(cats_retained_mean or 0.0):.2f}"
            )
            print(
                "  cats reject thr/mutual/margin and spatial pre-post mean: "
                f"{(cats_threshold_rej_mean or 0.0):.2f}/{(cats_mutual_rej_mean or 0.0):.2f}/"
                f"{(cats_margin_rej_mean or 0.0):.2f} "
                f"{(cats_before_mean or 0.0):.2f}->{(cats_after_mean or 0.0):.2f}"
            )
        if cats_sim_mean is not None:
            print(f"  cats merge sim/margin mean: {cats_sim_mean:.4f}/{(cats_margin_mean or 0.0):.4f}")
        if dyn_selected_mean is not None:
            print(
                "  dyn selected/budget min-max-std mean: "
                f"{dyn_selected_mean:.2f}/{(dyn_budget_min_mean or 0.0):.2f}-"
                f"{(dyn_budget_max_mean or 0.0):.2f}-{(dyn_budget_std_mean or 0.0):.2f}"
            )
            if dyn_density_frame_mean is not None:
                print(
                    "  dyn density/event/motion frame score mean: "
                    f"{dyn_density_frame_mean:.3f}/{(dyn_event_chunk_mean or 0.0):.3f}/"
                    f"{(dyn_motion_frame_mean or 0.0):.3f}"
                )
            print(
                "  dyn sink/residual/retained and spatial pre-post mean: "
                f"{(dyn_sink_mean or 0.0):.2f}/{(dyn_residual_mean or 0.0):.2f}/"
                f"{(dyn_retained_mean or 0.0):.2f} "
                f"{(dyn_before_mean or 0.0):.2f}->{(dyn_after_mean or 0.0):.2f}"
            )
            print(
                "  dyn debias/sink/weighted active and merge sim mean: "
                f"{(dyn_debias_mean or 0.0):.2f}/{(dyn_sink_active_mean or 0.0):.2f}/"
                f"{(dyn_weighted_mean or 0.0):.2f}/{(dyn_sim_mean or 0.0):.4f}"
            )
        if hedge_budget_mean is not None:
            print(
                "  hedge residual budget/candidates stable+evidence/selected stable+evidence mean: "
                f"{hedge_budget_mean:.2f}/{(hedge_stable_cand_mean or 0.0):.2f}+"
                f"{(hedge_evidence_cand_mean or 0.0):.2f}/{(hedge_stable_sel_mean or 0.0):.2f}+"
                f"{(hedge_evidence_sel_mean or 0.0):.2f}"
            )
            print(
                "  hedge floor/diversity mean: "
                f"{(hedge_floor_mean or 0.0):.2f}/{(hedge_div_mean or 0.0):.2f}"
            )
        if learn_selected_mean is not None:
            print(
                "  learn selected stable/selector/qaware score mean/std: "
                f"{learn_selected_mean:.2f}/{(learn_stable_mean or 0.0):.2f}/"
                f"{(learn_selector_mean or 0.0):.2f}/{(learn_qaware_mean or 0.0):.2f} "
                f"{(learn_score_mean or 0.0):.4f}/{(learn_score_std_mean or 0.0):.4f}"
            )
        if learn_teacher_keep_mean is not None:
            print(f"  learn teacher keep ratio mean: {learn_teacher_keep_mean:.4f}")
        if red_mean is not None:
            print(f"  final token reduction mean: {red_mean * 100:.2f}%")
        if vision_red_mean is not None:
            print(f"  vision-side token reduction mean: {vision_red_mean * 100:.2f}%")

    comparison = summary.get("comparison", {})
    if comparison:
        print("[comparison]")
        preferred_comparisons = [
            "baseline_vs_flashvid",
            *[f"baseline_vs_{phase}" for phase in _phase_order(summary) if phase not in ("baseline", "flashvid")],
            *[f"flashvid_vs_{phase}" for phase in _phase_order(summary) if phase not in ("baseline", "flashvid")],
        ]
        ordered_comparisons = preferred_comparisons + sorted(k for k in comparison if k not in preferred_comparisons)
        for key in ordered_comparisons:
            comp = comparison.get(key)
            if comp is None:
                continue
            lat_sp = comp["latency_speedup_ratio"]["mean"]
            ratio_key = next((k for k in comp.keys() if k.startswith("visual_token_ratio_")), None)
            reduction_key = next((k for k in comp.keys() if k.startswith("visual_token_reduction_vs_")), None)
            vision_ratio_key = next((k for k in comp.keys() if k.startswith("vision_token_ratio_")), None)
            vision_reduction_key = next((k for k in comp.keys() if k.startswith("vision_token_reduction_vs_")), None)
            token_red = comp[reduction_key]["mean"] if reduction_key else None
            vision_token_red = comp[vision_reduction_key]["mean"] if vision_reduction_key else None
            anchor_name, target_name = key.split("_vs_", 1)
            print(f"  [{key}] matched={comp['matched_samples']}")
            print(
                "    paired correctness: "
                f"both_correct={comp.get('both_correct', 0)} both_wrong={comp.get('both_wrong', 0)} "
                f"anchor_only={comp.get(f'{anchor_name}_only_correct', 0)} "
                f"target_only={comp.get(f'{target_name}_only_correct', 0)}"
            )
            if lat_sp is not None:
                print(f"    latency speedup: {lat_sp:.3f}x")
            if ratio_key and comp[ratio_key]["mean"] is not None:
                print(f"    {ratio_key}: {comp[ratio_key]['mean']:.3f}")
            if token_red is not None:
                print(f"    token reduction: {token_red * 100:.2f}%")
            if vision_ratio_key and comp[vision_ratio_key]["mean"] is not None:
                print(f"    {vision_ratio_key}: {comp[vision_ratio_key]['mean']:.3f}")
            if vision_token_red is not None:
                print(f"    vision-side token reduction: {vision_token_red * 100:.2f}%")

    duration_breakdown = summary.get("duration_breakdown", {})
    if duration_breakdown:
        printed_header = False
        for duration in ("short", "medium", "long"):
            bucket = duration_breakdown.get(duration)
            if not bucket:
                continue
            phase_values = [
                bucket.get(phase_name)
                for phase_name in _phase_order(summary)
                if bucket.get(phase_name) is not None
            ]
            if not phase_values or all(int(phase.get("num_samples", 0) or 0) == 0 for phase in phase_values):
                continue
            if not printed_header:
                print("[by duration]")
                printed_header = True
            print(f"  [{duration}]")
            for phase_name in _phase_order(summary):
                phase = bucket.get(phase_name)
                if phase is None:
                    continue
                acc = phase.get("accuracy")
                acc_text = f"{acc * 100:.2f}%" if acc is not None else "N/A"
                vt_mean = phase.get("compressed_visual_tokens", {}).get("mean")
                vision_vt_mean = phase.get("vision_compressed_visual_tokens", {}).get("mean")
                target_mean = phase.get("talon_target_tokens_per_frame", {}).get("mean")
                channel = (
                    phase.get("talon_anchor_tokens", {}).get("mean"),
                    phase.get("talon_event_tokens", {}).get("mean"),
                    phase.get("talon_recall_tokens", {}).get("mean"),
                )
                line = f"    [{phase_name}] valid={phase['num_valid']}/{phase['num_samples']} acc={acc_text}"
                if vt_mean is not None:
                    line += f" vtoken={vt_mean:.2f}"
                if vision_vt_mean is not None:
                    line += f" vision={vision_vt_mean:.2f}"
                if target_mean is not None:
                    line += f" target/frame={target_mean:.2f}"
                if channel[0] is not None:
                    line += f" a/e/r={channel[0]:.1f}/{(channel[1] or 0.0):.1f}/{(channel[2] or 0.0):.1f}"
                print(line)
            comp_key = ""
            comp = None
            for candidate in [f"flashvid_vs_{phase}" for phase in _phase_order(summary) if phase not in ("baseline", "flashvid")]:
                comp = bucket.get("comparison", {}).get(candidate)
                if comp is not None:
                    comp_key = candidate
                    break
            if comp is not None:
                ratio_key = next((k for k in comp.keys() if k.startswith("visual_token_ratio_")), None)
                reduction_key = next((k for k in comp.keys() if k.startswith("visual_token_reduction_vs_")), None)
                ratio = comp[ratio_key]["mean"] if ratio_key else None
                reduction = comp[reduction_key]["mean"] if reduction_key else None
                comp_line = f"    [{comp_key}] matched={comp['matched_samples']}"
                if ratio is not None:
                    comp_line += f" ratio={ratio:.3f}"
                if reduction is not None:
                    comp_line += f" token_reduction={reduction * 100:.2f}%"
                print(comp_line)
    print(SEPARATOR)


def run(args: BenchmarkArgs):
    if args.run_dynflashvid:
        args.run_ours = True
        args.compression_variant = "dynflashvid"
    ours_phase_name = _ours_phase_key(args)
    ours_output_path = _ours_output_path(args, ours_phase_name)
    samples = _load_dataset(args.dataset_jsonl, args.limit, args.shuffle, args.start_index, args.duration_filter)
    if not samples:
        raise ValueError(f"No samples loaded from {args.dataset_jsonl}")
    if not (args.run_baseline or args.run_flashvid or args.run_ours or args.run_graphvid or args.run_graftvid or args.run_cats):
        raise ValueError("At least one phase must be enabled: run_baseline/run_flashvid/run_ours/run_graphvid/run_graftvid/run_cats")

    model_bundle = _load_backend_model(args)
    backend = model_bundle["backend"]
    _print_header(args, backend)
    if backend == "llava":
        if float(args.llm_retention_ratio) >= 0.9999:
            print("[info] LLaVA backend: inner-LLM pruning is disabled for stability (vision compression remains enabled).")
        else:
            print(
                "[info] LLaVA backend: inner-LLM pruning is enabled "
                f"(pruning_layer={args.pruning_layer}, llm_retention_ratio={args.llm_retention_ratio})."
            )
    print(f"Loaded {len(samples)} samples.\n")
    if args.reload_model_each_phase:
        model_bundle["model"] = None
        model_bundle["processor"] = None
        model_bundle["tokenizer"] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_phases = (
        int(args.run_baseline)
        + int(args.run_flashvid)
        + int(args.run_ours)
        + int(args.run_graphvid)
        + int(args.run_graftvid)
        + int(args.run_cats)
    )
    phase_idx = 1
    def _acquire_phase_bundle():
        if args.reload_model_each_phase:
            return _load_backend_model(args)
        return model_bundle

    def _release_phase_bundle(bundle):
        if not args.reload_model_each_phase:
            return
        if isinstance(bundle, dict):
            bundle["model"] = None
            bundle["processor"] = None
            bundle["tokenizer"] = None
        del bundle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.run_baseline:
        print(f"Phase {phase_idx}/{total_phases}: Baseline ...")
        phase_bundle = _acquire_phase_bundle()
        try:
            _run_phase(
                model_bundle=phase_bundle,
                args=args,
                samples=samples,
                phase_name="Baseline",
                use_acceleration=False,
                output_path=args.baseline_output,
            )
        finally:
            _release_phase_bundle(phase_bundle)
        phase_idx += 1

    if args.run_flashvid:
        print(f"\nPhase {phase_idx}/{total_phases}: FlashVID ...")
        print(
            "[talon-active][flashvid] "
            f"path={'unified' if args.talon_unified_selection else 'legacy'}, "
            f"rerank={args.talon_rerank_with_flash_prior}, rescue={args.talon_rescue_enabled}, "
            f"fast_rank={args.talon_fast_rank_plan}, qaware={args.question_aware_reweighting}"
        )
        phase_bundle = _acquire_phase_bundle()
        phase_backend = phase_bundle["backend"]
        phase_bundle["model"] = _apply_flashvid_original(phase_bundle["model"], args, phase_backend)
        try:
            _run_phase(
                model_bundle=phase_bundle,
                args=args,
                samples=samples,
                phase_name="FlashVID",
                use_acceleration=True,
                output_path=args.flashvid_output,
            )
        finally:
            _release_phase_bundle(phase_bundle)
        phase_idx += 1

    if args.run_graphvid:
        print(f"\nPhase {phase_idx}/{total_phases}: GraphVID ...")
        print(
            "[graphvid-active] "
            f"merge=graph, topk={args.graph_temporal_topk}, radius={args.graph_temporal_radius}, "
            f"skip={args.graph_temporal_skip}, protect={args.graph_merge_protect_ratio:.2f}, "
            f"target_ratio={args.graph_merge_target_ratio:.2f}, final_tpf={args.graph_final_tokens_per_frame}, "
            f"skip_spatial={args.graph_skip_spatial_merge_when_capped}, "
            f"rep={args.graph_merge_representative}, "
            f"pos={args.graph_representative_position}, "
            f"detail_w={args.graph_protection_detail_weight:.2f}, "
            f"adaptive_detail={args.graph_adaptive_detail_protection}, "
            f"penalty={args.graph_merge_importance_penalty:.2f}, "
            f"respect_thr={args.graph_respect_temporal_threshold}"
        )
        phase_bundle = _acquire_phase_bundle()
        phase_backend = phase_bundle["backend"]
        phase_bundle["model"] = _apply_graphvid(phase_bundle["model"], args, phase_backend)
        try:
            _run_phase(
                model_bundle=phase_bundle,
                args=args,
                samples=samples,
                phase_name="GraphVID",
                use_acceleration=True,
                output_path=args.graphvid_output,
            )
        finally:
            _release_phase_bundle(phase_bundle)
        phase_idx += 1

    if args.run_graftvid:
        print(f"\nPhase {phase_idx}/{total_phases}: GRAFT-VID ...")
        print(
            "[graftvid-active] "
            f"merge=graft, topk={args.graft_temporal_topk}, radius={args.graft_temporal_radius}, "
            f"skip={args.graft_temporal_skip}, global_topk={args.graft_global_topk}, "
            f"residual_input={args.graft_input_is_residual}, "
            f"anchor={(args.graft_anchor_ratio if args.graft_anchor_ratio is not None else (0.15 if args.graft_input_is_residual else 0.65)):.2f}, "
            f"edge_thr={args.graft_edge_threshold:.2f}, eps={args.graft_component_radius_eps:.3f}, "
            f"capacity={args.graft_parent_capacity}, mutual={args.graft_mutual_knn}, "
            f"one_frame={args.graft_one_token_per_frame}, scene_thr={args.graft_scene_threshold:.2f}, "
            f"minpf={args.graft_min_tokens_per_frame}, budget_fix={args.graft_budget_correction}, "
            f"budget_div={args.graft_budget_diversity_weight:.2f}, score={args.graft_score_preset}, "
            f"dur_aware={args.graft_duration_aware}"
        )
        phase_bundle = _acquire_phase_bundle()
        phase_backend = phase_bundle["backend"]
        phase_bundle["model"] = _apply_graftvid(phase_bundle["model"], args, phase_backend)
        try:
            _run_phase(
                model_bundle=phase_bundle,
                args=args,
                samples=samples,
                phase_name="GraftVID",
                use_acceleration=True,
                output_path=args.graftvid_output,
            )
        finally:
            _release_phase_bundle(phase_bundle)
        phase_idx += 1

    if args.run_cats:
        print(f"\nPhase {phase_idx}/{total_phases}: CATS-FlashVID ...")
        print(
            "[cats-active] "
            f"adts={args.cats_adts_mode}, beta={args.cats_adts_beta:.3f}, margin={args.cats_margin_threshold:.3f}, "
            f"bonus={args.cats_high_conf_bonus:.3f}, mutual={args.cats_mutual_nn}, "
            f"attn_w={args.cats_confidence_attn_weight:.2f}, sim_w={args.cats_confidence_sim_weight:.2f}, "
            f"anchor_w={args.cats_anchor_self_weight:.2f}, "
            f"adaptive_budget={args.cats_adaptive_adts_budget}"
        )
        phase_bundle = _acquire_phase_bundle()
        phase_backend = phase_bundle["backend"]
        phase_bundle["model"] = _apply_cats(phase_bundle["model"], args, phase_backend)
        try:
            _run_phase(
                model_bundle=phase_bundle,
                args=args,
                samples=samples,
                phase_name="CATS",
                use_acceleration=True,
                output_path=args.cats_output,
            )
        finally:
            _release_phase_bundle(phase_bundle)
        phase_idx += 1

    if args.run_ours:
        ours_display_name = _phase_display_name(ours_phase_name)
        print(f"\nPhase {phase_idx}/{total_phases}: {ours_display_name} ...")
        print(
            f"[talon-active][{ours_phase_name}] "
            f"path=clean, qaware={args.question_aware_reweighting}, "
            f"variant={args.compression_variant}, merge={args.temporal_merge_mode}, "
            f"target/frame={args.talon_target_tokens_per_frame}, "
            f"duration_targets={args.talon_short_target_tokens_per_frame}/"
            f"{args.talon_medium_target_tokens_per_frame}/"
            f"{args.talon_long_target_tokens_per_frame}, "
            f"rank_max={args.talon_rank_max}, "
            f"anchor_div={args.talon_anchor_diversity_weight:.2f}"
        )
        phase_bundle = _acquire_phase_bundle()
        phase_backend = phase_bundle["backend"]
        phase_bundle["model"] = _apply_ours(phase_bundle["model"], args, phase_backend)
        try:
            _run_phase(
                model_bundle=phase_bundle,
                args=args,
                samples=samples,
                phase_name=ours_display_name,
                use_acceleration=True,
                output_path=ours_output_path,
                phase_key=ours_phase_name,
            )
        finally:
            _release_phase_bundle(phase_bundle)

    summary: dict[str, Any] = {"comparison": {}}
    baseline_records = None
    flashvid_records = None
    ours_records = None
    graphvid_records = None
    graftvid_records = None
    cats_records = None
    if args.run_baseline:
        baseline_records = _read_jsonl(args.baseline_output)
        summary["baseline"] = _summarize_phase(baseline_records)
    if args.run_flashvid:
        flashvid_records = _read_jsonl(args.flashvid_output)
        summary["flashvid"] = _summarize_phase(flashvid_records)
    if args.run_ours:
        ours_records = _read_jsonl(ours_output_path)
        summary[ours_phase_name] = _summarize_phase(ours_records)
    if args.run_graphvid:
        graphvid_records = _read_jsonl(args.graphvid_output)
        summary["graphvid"] = _summarize_phase(graphvid_records)
    if args.run_graftvid:
        graftvid_records = _read_jsonl(args.graftvid_output)
        summary["graftvid"] = _summarize_phase(graftvid_records)
    if args.run_cats:
        cats_records = _read_jsonl(args.cats_output)
        summary["cats"] = _summarize_phase(cats_records)

    if baseline_records is not None and flashvid_records is not None:
        summary["comparison"]["baseline_vs_flashvid"] = _summarize_pairwise_comparison(
            baseline_records,
            flashvid_records,
            anchor_name="baseline",
            target_name="flashvid",
        )
    if baseline_records is not None and ours_records is not None:
        summary["comparison"][f"baseline_vs_{ours_phase_name}"] = _summarize_pairwise_comparison(
            baseline_records,
            ours_records,
            anchor_name="baseline",
            target_name=ours_phase_name,
        )
    if flashvid_records is not None and ours_records is not None:
        summary["comparison"][f"flashvid_vs_{ours_phase_name}"] = _summarize_pairwise_comparison(
            flashvid_records,
            ours_records,
            anchor_name="flashvid",
            target_name=ours_phase_name,
        )
    if flashvid_records is not None and graphvid_records is not None:
        summary["comparison"]["flashvid_vs_graphvid"] = _summarize_pairwise_comparison(
            flashvid_records,
            graphvid_records,
            anchor_name="flashvid",
            target_name="graphvid",
        )
    if flashvid_records is not None and graftvid_records is not None:
        summary["comparison"]["flashvid_vs_graftvid"] = _summarize_pairwise_comparison(
            flashvid_records,
            graftvid_records,
            anchor_name="flashvid",
            target_name="graftvid",
        )
    if flashvid_records is not None and cats_records is not None:
        summary["comparison"]["flashvid_vs_cats"] = _summarize_pairwise_comparison(
            flashvid_records,
            cats_records,
            anchor_name="flashvid",
            target_name="cats",
        )
    _add_duration_breakdown(
        summary,
        baseline_records=baseline_records,
        flashvid_records=flashvid_records,
        ours_records=ours_records,
        ours_phase_name=ours_phase_name,
        graphvid_records=graphvid_records,
        graftvid_records=graftvid_records,
        cats_records=cats_records,
    )

    summary_path = Path(args.summary_output_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _print_summary(summary)
    if args.reload_model_each_phase:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = HfArgumentParser(BenchmarkArgs)
    (args,) = parser.parse_args_into_dataclasses(return_remaining_strings=False)
    run(args)


if __name__ == "__main__":
    main()
