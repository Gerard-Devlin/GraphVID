from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _str_bool(value: bool) -> str:
    return "True" if value else "False"


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def _query_free_gpus(free_ratio: float, min_free_mb: int) -> list[int]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    gpus: list[int] = []
    print("[gpu-scan] index free/total util eligible")
    for raw in proc.stdout.strip().splitlines():
        if not raw.strip():
            continue
        parts = [p.strip() for p in raw.split(",")]
        idx = int(parts[0])
        free_mb = int(parts[1])
        total_mb = int(parts[2])
        util = int(parts[3])
        ratio = free_mb / max(1, total_mb)
        ok = ratio >= free_ratio and free_mb >= min_free_mb
        print(f"[gpu-scan] {idx} {free_mb}/{total_mb} {util}% {'yes' if ok else 'no'}")
        if ok:
            gpus.append(idx)
    return gpus


def _split_ranges(start: int, total: int, parts: int) -> list[tuple[int, int]]:
    parts = max(1, min(parts, total))
    base = total // parts
    rem = total % parts
    out = []
    cursor = start
    for i in range(parts):
        count = base + (1 if i < rem else 0)
        out.append((cursor, count))
        cursor += count
    return out


def _append_common_talon_args(cmd: list[str], args: argparse.Namespace) -> None:
    cmd.extend(
        [
            "--llm_retention_ratio",
            str(args.llm_retention_ratio),
            "--retention_ratio",
            str(args.retention_ratio),
            "--expansion",
            str(args.expansion),
            "--pruning_layer",
            str(args.pruning_layer),
            "--compression_variant",
            str(args.compression_variant),
            "--question_aware_reweighting",
            "True",
            "--question_reweight_beta",
            "0.25",
            "--apex_evidence_ratio",
            str(args.apex_evidence_ratio),
            "--apex_event_ratio",
            str(args.apex_event_ratio),
            "--apex_memory_ratio",
            str(args.apex_memory_ratio),
            "--apex_router_strength",
            str(args.apex_router_strength),
            "--apex_summary_temperature",
            str(args.apex_summary_temperature),
            "--apex_frame_floor_ratio",
            str(args.apex_frame_floor_ratio),
            "--apex_question_weight",
            str(args.apex_question_weight),
            "--cert_budget_uses_expansion",
            _str_bool(args.cert_budget_uses_expansion),
            "--cert_query_atoms",
            str(args.cert_query_atoms),
            "--cert_temporal_bins",
            str(args.cert_temporal_bins),
            "--cert_spatial_bins",
            str(args.cert_spatial_bins),
            "--cert_candidate_multiplier",
            str(args.cert_candidate_multiplier),
            "--cert_query_weight",
            str(args.cert_query_weight),
            "--cert_temporal_weight",
            str(args.cert_temporal_weight),
            "--cert_detail_weight",
            str(args.cert_detail_weight),
            "--cert_repair_ratio",
            str(args.cert_repair_ratio),
            "--cert_fusion_alpha",
            str(args.cert_fusion_alpha),
            "--cert_assignment_temperature",
            str(args.cert_assignment_temperature),
            "--cert_track_threshold",
            str(args.cert_track_threshold),
            "--cert_spatial_penalty",
            str(args.cert_spatial_penalty),
            "--cert_metric_dim",
            str(args.cert_metric_dim),
            "--certv2_budget_uses_expansion",
            _str_bool(args.certv2_budget_uses_expansion),
            "--certv2_query_atoms",
            str(args.certv2_query_atoms),
            "--certv2_temporal_bins",
            str(args.certv2_temporal_bins),
            "--certv2_spatial_bins",
            str(args.certv2_spatial_bins),
            "--certv2_candidate_multiplier",
            str(args.certv2_candidate_multiplier),
            "--certv2_query_weight",
            str(args.certv2_query_weight),
            "--certv2_frame_floor_ratio",
            str(args.certv2_frame_floor_ratio),
            "--certv2_diversity_weight",
            str(args.certv2_diversity_weight),
            "--certv2_coverage_weight",
            str(args.certv2_coverage_weight),
            "--certv2_density_neighbors",
            str(args.certv2_density_neighbors),
            "--certv2_track_threshold",
            str(args.certv2_track_threshold),
            "--certv2_spatial_penalty",
            str(args.certv2_spatial_penalty),
            "--certv2_metric_dim",
            str(args.certv2_metric_dim),
            "--certv2_repair_ratio",
            str(args.certv2_repair_ratio),
            "--certv2_repair_ratio_high",
            str(args.certv2_repair_ratio_high),
            "--certv2_router_strength",
            str(args.certv2_router_strength),
            "--certv2_protect_ratio",
            str(args.certv2_protect_ratio),
            "--certv2_swap_margin",
            str(args.certv2_swap_margin),
            "--certv2_fusion_alpha",
            str(args.certv2_fusion_alpha),
            "--certv2_repair_fusion_alpha",
            str(args.certv2_repair_fusion_alpha),
            "--certv2_assignment_temperature",
            str(args.certv2_assignment_temperature),
            "--certv3_budget_uses_expansion",
            _str_bool(args.certv3_budget_uses_expansion),
            "--certv3_query_atoms",
            str(args.certv3_query_atoms),
            "--certv3_temporal_bins",
            str(args.certv3_temporal_bins),
            "--certv3_spatial_bins",
            str(args.certv3_spatial_bins),
            "--certv3_candidate_multiplier",
            str(args.certv3_candidate_multiplier),
            "--certv3_query_weight",
            str(args.certv3_query_weight),
            "--certv3_track_threshold",
            str(args.certv3_track_threshold),
            "--certv3_spatial_penalty",
            str(args.certv3_spatial_penalty),
            "--certv3_metric_dim",
            str(args.certv3_metric_dim),
            "--certv3_frame_coverage_ratio",
            str(args.certv3_frame_coverage_ratio),
            "--certv3_cell_coverage_ratio",
            str(args.certv3_cell_coverage_ratio),
            "--certv3_query_threshold",
            str(args.certv3_query_threshold),
            "--certv3_query_per_atom",
            str(args.certv3_query_per_atom),
            "--certv3_structural_weight",
            str(args.certv3_structural_weight),
            "--certv3_whitening_strength",
            str(args.certv3_whitening_strength),
            "--certv3_quality_floor",
            str(args.certv3_quality_floor),
            "--certv3_ridge",
            str(args.certv3_ridge),
            "--certv3_swap_steps",
            str(args.certv3_swap_steps),
            "--certv3_swap_pool",
            str(args.certv3_swap_pool),
            "--certv3_swap_margin",
            str(args.certv3_swap_margin),
            "--certv3_fusion_alpha",
            str(args.certv3_fusion_alpha),
            "--certv3_assignment_temperature",
            str(args.certv3_assignment_temperature),
            "--certv6_scene_temporal" if args.certv6_scene_temporal else "--no-certv6_scene_temporal",
            "--certv6_gate_enabled" if args.certv6_gate_enabled else "--no-certv6_gate_enabled",
            "--certv6_continuity_low",
            str(args.certv6_continuity_low),
            "--certv6_continuity_high",
            str(args.certv6_continuity_high),
            "--certv6_query_per_atom_max",
            str(args.certv6_query_per_atom_max),
            "--certv7_min_duration_seconds", str(args.certv7_min_duration_seconds),
            "--certv7_transport_spatial_bins", str(args.certv7_transport_spatial_bins),
            "--certv7_transport_epsilon", str(args.certv7_transport_epsilon),
            "--certv7_transport_steps", str(args.certv7_transport_steps),
            "--certv7_transport_spatial_weight", str(args.certv7_transport_spatial_weight),
            "--certv7_frame_floor_ratio", str(args.certv7_frame_floor_ratio),
            "--certv7_frame_cap_ratio", str(args.certv7_frame_cap_ratio),
            "--certv7_budget_temperature", str(args.certv7_budget_temperature),
            "--certv7_uniqueness_weight", str(args.certv7_uniqueness_weight),
            "--certv7_transport_weight", str(args.certv7_transport_weight),
            "--certv7_event_weight", str(args.certv7_event_weight),
            "--certv7_query_weight", str(args.certv7_query_weight),
            "--certv7_budget_rounding", str(args.certv7_budget_rounding),
            "--certv7_v3_certificate_ratio", str(args.certv7_v3_certificate_ratio),
            "--certv7_relay_ratio", str(args.certv7_relay_ratio),
            "--certv7_relay_query_share", str(args.certv7_relay_query_share),
            "--certv7_transition_relay_share", str(args.certv7_transition_relay_share),
            "--certv7_query_peaks_per_atom", str(args.certv7_query_peaks_per_atom),
            "--certv7_query_min_frame_gap", str(args.certv7_query_min_frame_gap),
            "--certv7_query_peak_threshold", str(args.certv7_query_peak_threshold),
            "--certv7_query_context_radius", str(args.certv7_query_context_radius),
            "--certv7_transition_pairs_per_boundary", str(args.certv7_transition_pairs_per_boundary),
            "--certv7_transition_min_similarity", str(args.certv7_transition_min_similarity),
            "--certv7_trajectory_min_span", str(args.certv7_trajectory_min_span),
            "--certv7_trajectory_points", str(args.certv7_trajectory_points),
            "--certv7_facility_quality_mix", str(args.certv7_facility_quality_mix),
            "--certv7_min_reallocation_ratio", str(args.certv7_min_reallocation_ratio),
            "--certv7_d_efficiency_floor", str(args.certv7_d_efficiency_floor),
            "--certv7_assignment_topk", str(args.certv7_assignment_topk),
            "--certv7_assignment_temperature", str(args.certv7_assignment_temperature),
            "--certv7_cross_frame_cost_quantile", str(args.certv7_cross_frame_cost_quantile),
            "--certv7_cross_frame_similarity", str(args.certv7_cross_frame_similarity),
            "--certv7_cross_frame_max_seconds", str(args.certv7_cross_frame_max_seconds),
            "--certv7_component_bonus", str(args.certv7_component_bonus),
            "--certv7_design_protect_ratio", str(args.certv7_design_protect_ratio),
            "--certv7_long_fusion_alpha", str(args.certv7_long_fusion_alpha),
            "--certv7_debug", _str_bool(args.certv7_debug),
            "--certhr_horizon_gap_seconds",
            str(args.certhr_horizon_gap_seconds),
            "--certhr_chunk_max_seconds",
            str(args.certhr_chunk_max_seconds),
            "--certhr_chunk_max_units",
            str(args.certhr_chunk_max_units),
            "--certhr_semantic_quantile",
            str(args.certhr_semantic_quantile),
            "--certhr_semantic_floor",
            str(args.certhr_semantic_floor),
            "--certhr_coverage_floor",
            str(args.certhr_coverage_floor),
            "--certhr_deficit_threshold",
            str(args.certhr_deficit_threshold),
            "--certhr_query_peak_quantile",
            str(args.certhr_query_peak_quantile),
            "--certhr_query_peak_floor",
            str(args.certhr_query_peak_floor),
            "--certhr_max_swap_ratio",
            str(args.certhr_max_swap_ratio),
            "--certhr_d_efficiency_floor",
            str(args.certhr_d_efficiency_floor),
            "--certhr_add_pool",
            str(args.certhr_add_pool),
            "--certhr_remove_pool",
            str(args.certhr_remove_pool),
            "--certhr_debug",
            _str_bool(args.certhr_debug),
            "--certv4_budget_mode",
            str(args.certv4_budget_mode),
            "--certv4_attention_policy",
            str(args.certv4_attention_policy),
            "--certv4_attention_eps",
            str(args.certv4_attention_eps),
            "--certv4_certificate_budget_ratio",
            str(args.certv4_certificate_budget_ratio),
            "--certv4_query_mode",
            str(args.certv4_query_mode),
            "--certv4_design_protect_ratio",
            str(args.certv4_design_protect_ratio),
            "--certv4_query_atoms",
            str(args.certv4_query_atoms),
            "--certv4_temporal_bins",
            str(args.certv4_temporal_bins),
            "--certv4_spatial_bins",
            str(args.certv4_spatial_bins),
            "--certv4_candidate_multiplier",
            str(args.certv4_candidate_multiplier),
            "--certv4_track_threshold",
            str(args.certv4_track_threshold),
            "--certv4_spatial_penalty",
            str(args.certv4_spatial_penalty),
            "--certv4_metric_dim",
            str(args.certv4_metric_dim),
            "--certv4_frame_coverage_ratio",
            str(args.certv4_frame_coverage_ratio),
            "--certv4_cell_coverage_ratio",
            str(args.certv4_cell_coverage_ratio),
            "--certv4_query_threshold",
            str(args.certv4_query_threshold),
            "--certv4_query_per_atom",
            str(args.certv4_query_per_atom),
            "--certv4_structural_weight",
            str(args.certv4_structural_weight),
            "--certv4_whitening_strength",
            str(args.certv4_whitening_strength),
            "--certv4_quality_floor",
            str(args.certv4_quality_floor),
            "--certv4_ridge",
            str(args.certv4_ridge),
            "--certv4_swap_steps",
            str(args.certv4_swap_steps),
            "--certv4_swap_pool",
            str(args.certv4_swap_pool),
            "--certv4_swap_margin",
            str(args.certv4_swap_margin),
            "--certv4_fusion_alpha",
            str(args.certv4_fusion_alpha),
            "--certv4_assignment_temperature",
            str(args.certv4_assignment_temperature),
            "--certv4_debug",
            _str_bool(args.certv4_debug),
            "--certv5_budget_mode",
            str(args.certv5_budget_mode),
            "--certv5_ot_enabled",
            _str_bool(args.certv5_ot_enabled),
            "--certv5_ot_topk",
            str(args.certv5_ot_topk),
            "--certv5_ot_temperature",
            str(args.certv5_ot_temperature),
            "--certv5_ot_steps",
            str(args.certv5_ot_steps),
            "--certv5_ot_capacity_tau",
            str(args.certv5_ot_capacity_tau),
            "--certv5_ot_prior_shrink",
            str(args.certv5_ot_prior_shrink),
            "--certv5_ot_live_fraction",
            str(args.certv5_ot_live_fraction),
            "--certv5_ot_cost_slack",
            str(args.certv5_ot_cost_slack),
            "--certv5_ot_temporal_penalty",
            str(args.certv5_ot_temporal_penalty),
            "--certv5_ot_max_displacement",
            str(args.certv5_ot_max_displacement),
            "--certv5_ot_min_cosine",
            str(args.certv5_ot_min_cosine),
            "--certv5_debug",
            _str_bool(args.certv5_debug),
            "--certe_budget_uses_expansion",
            _str_bool(args.certe_budget_uses_expansion),
            "--certe_ridge",
            str(args.certe_ridge),
            "--certe_bottom_k",
            str(args.certe_bottom_k),
            "--certe_swap_steps",
            str(args.certe_swap_steps),
            "--certe_remove_pool",
            str(args.certe_remove_pool),
            "--certe_add_pool",
            str(args.certe_add_pool),
            "--certe_verify_pool",
            str(args.certe_verify_pool),
            "--certe_swap_margin",
            str(args.certe_swap_margin),
            "--certe_spectral_temperature",
            str(args.certe_spectral_temperature),
            "--certe_d_efficiency_floor",
            str(args.certe_d_efficiency_floor),
            "--certe_rank_tolerance",
            str(args.certe_rank_tolerance),
            "--certe_debug",
            _str_bool(args.certe_debug),
            "--faith_budget_uses_expansion",
            _str_bool(args.faith_budget_uses_expansion),
            "--faith_mass_strength",
            str(args.faith_mass_strength),
            "--faith_variance_strength",
            str(args.faith_variance_strength),
            "--faith_merge_alpha",
            str(args.faith_merge_alpha),
            "--faith_temporal_radius",
            str(args.faith_temporal_radius),
            "--faith_spatial_radius",
            str(args.faith_spatial_radius),
            "--faith_component_bonus",
            str(args.faith_component_bonus),
            "--faith_temporal_penalty",
            str(args.faith_temporal_penalty),
            "--faith_spatial_penalty",
            str(args.faith_spatial_penalty),
            "--faith_assignment_topk",
            str(args.faith_assignment_topk),
            "--faith_assignment_temperature",
            str(args.faith_assignment_temperature),
            "--faith_max_log_bias",
            str(args.faith_max_log_bias),
            "--faith_attention_strict",
            _str_bool(args.faith_attention_strict),
            "--faith_debug",
            _str_bool(args.faith_debug),
            "--prism_budget_uses_expansion",
            _str_bool(args.prism_budget_uses_expansion),
            "--prism_metric_dim",
            str(args.prism_metric_dim),
            "--prism_query_atoms",
            str(args.prism_query_atoms),
            "--prism_candidate_multiplier",
            str(args.prism_candidate_multiplier),
            "--prism_probe_tokens",
            str(args.prism_probe_tokens),
            "--prism_frame_floor_ratio",
            str(args.prism_frame_floor_ratio),
            "--prism_attention_weight",
            str(args.prism_attention_weight),
            "--prism_event_weight",
            str(args.prism_event_weight),
            "--prism_query_weight",
            str(args.prism_query_weight),
            "--prism_disagreement_weight",
            str(args.prism_disagreement_weight),
            "--prism_router_strength",
            str(args.prism_router_strength),
            "--prism_coverage_weight",
            str(args.prism_coverage_weight),
            "--prism_pareto_weight",
            str(args.prism_pareto_weight),
            "--prism_batch_size",
            str(args.prism_batch_size),
            "--adaptive_token_budget",
            "False",
            "--talon_adaptive_target_enabled",
            _str_bool(args.talon_adaptive_target_enabled),
            "--talon_target_mean_cap",
            str(args.talon_target_mean_cap),
            "--talon_target_tokens_per_frame",
            str(args.talon_target_tokens_per_frame),
            "--talon_short_target_tokens_per_frame",
            str(args.talon_short_target_tokens_per_frame),
            "--talon_medium_target_tokens_per_frame",
            str(args.talon_medium_target_tokens_per_frame),
            "--talon_long_target_tokens_per_frame",
            str(args.talon_long_target_tokens_per_frame),
            "--talon_adaptive_target_low",
            str(args.talon_adaptive_target_low),
            "--talon_adaptive_target_mid",
            str(args.talon_adaptive_target_mid),
            "--talon_adaptive_target_high",
            str(args.talon_adaptive_target_high),
            "--talon_complexity_floor",
            str(args.talon_complexity_floor),
            "--talon_complexity_ceil",
            str(args.talon_complexity_ceil),
            "--talon_adaptive_gamma",
            str(args.talon_adaptive_gamma),
            "--talon_question_recall_ratio",
            str(args.talon_question_recall_ratio),
            "--talon_question_recall_qweight",
            str(args.talon_question_recall_qweight),
            "--talon_persistence_recall_ratio",
            str(args.talon_persistence_recall_ratio),
            "--talon_persistence_recall_qweight",
            str(args.talon_persistence_recall_qweight),
            "--talon_persistence_recall_pweight",
            str(args.talon_persistence_recall_pweight),
            "--talon_persistence_apply_to_short",
            _str_bool(args.talon_persistence_apply_to_short),
            "--talon_persistence_apply_to_medium",
            _str_bool(args.talon_persistence_apply_to_medium),
            "--talon_persistence_apply_to_long",
            _str_bool(args.talon_persistence_apply_to_long),
            "--talon_object_evidence_ratio",
            str(args.talon_object_evidence_ratio),
            "--talon_object_evidence_qweight",
            str(args.talon_object_evidence_qweight),
            "--talon_object_evidence_sweight",
            str(args.talon_object_evidence_sweight),
            "--talon_object_evidence_pweight",
            str(args.talon_object_evidence_pweight),
            "--talon_object_evidence_apply_to_short",
            _str_bool(args.talon_object_evidence_apply_to_short),
            "--talon_object_evidence_apply_to_medium",
            _str_bool(args.talon_object_evidence_apply_to_medium),
            "--talon_object_evidence_apply_to_long",
            _str_bool(args.talon_object_evidence_apply_to_long),
            "--talon_question_pooling",
            args.talon_question_pooling,
            "--talon_question_pooling_topk",
            str(args.talon_question_pooling_topk),
            "--talon_question_contrast_weight",
            str(args.talon_question_contrast_weight),
            "--talon_question_contrast_apply_to_short",
            _str_bool(args.talon_question_contrast_apply_to_short),
            "--talon_monotonic_base_tokens_per_frame",
            str(args.talon_monotonic_base_tokens_per_frame),
            "--talon_anchor_diversity_weight",
            str(args.talon_anchor_diversity_weight),
            "--talon_spatial_anchor_coverage",
            _str_bool(args.talon_spatial_anchor_coverage),
            "--talon_spatial_anchor_ratio",
            str(args.talon_spatial_anchor_ratio),
            "--talon_spatial_anchor_rows",
            str(args.talon_spatial_anchor_rows),
            "--talon_spatial_anchor_cols",
            str(args.talon_spatial_anchor_cols),
            "--talon_spatial_anchor_score",
            args.talon_spatial_anchor_score,
            "--talon_spatial_anchor_apply_to_short",
            _str_bool(args.talon_spatial_anchor_apply_to_short),
            "--talon_frame_coverage_floor_ratio",
            str(args.talon_frame_coverage_floor_ratio),
            "--talon_frame_importance_pooling",
            args.talon_frame_importance_pooling,
            "--talon_frame_importance_topk",
            str(args.talon_frame_importance_topk),
            "--talon_medium_frame_coverage_floor_ratio",
            str(args.talon_medium_frame_coverage_floor_ratio),
            "--talon_long_frame_coverage_floor_ratio",
            str(args.talon_long_frame_coverage_floor_ratio),
            "--talon_frame_local_budget_ratio",
            str(args.talon_frame_local_budget_ratio),
            "--talon_anchor_safety_ratio",
            "0.72",
            "--talon_budget_mode",
            args.talon_budget_mode,
            "--talon_global_topk_ratio",
            "0.70",
            "--talon_event_budget_ratio",
            "0.30",
            "--talon_duration_aware",
            _str_bool(args.talon_duration_aware),
            "--talon_medium_anchor_safety_ratio",
            str(args.talon_medium_anchor_safety_ratio),
            "--talon_medium_event_budget_ratio",
            str(args.talon_medium_event_budget_ratio),
            "--talon_medium_global_topk_ratio",
            str(args.talon_medium_global_topk_ratio),
            "--talon_long_anchor_safety_ratio",
            str(args.talon_long_anchor_safety_ratio),
            "--talon_long_event_budget_ratio",
            str(args.talon_long_event_budget_ratio),
            "--talon_long_global_topk_ratio",
            str(args.talon_long_global_topk_ratio),
            "--talon_task_aware_event",
            _str_bool(args.talon_task_aware_event),
            "--talon_task_event_attention_weight",
            str(args.talon_task_event_attention_weight),
            "--talon_task_event_qweight",
            str(args.talon_task_event_qweight),
            "--talon_visual_task_balance",
            _str_bool(args.talon_visual_task_balance),
            "--talon_visual_task_anchor_ratio",
            str(args.talon_visual_task_anchor_ratio),
            "--talon_visual_task_event_ratio",
            str(args.talon_visual_task_event_ratio),
            "--talon_visual_task_recall_ratio",
            str(args.talon_visual_task_recall_ratio),
            "--talon_knowledge_visual_anchor_ratio",
            str(args.talon_knowledge_visual_anchor_ratio),
            "--talon_knowledge_visual_event_ratio",
            str(args.talon_knowledge_visual_event_ratio),
            "--talon_knowledge_visual_recall_ratio",
            str(args.talon_knowledge_visual_recall_ratio),
            "--talon_adaptive_router",
            _str_bool(args.talon_adaptive_router),
            "--talon_router_apply_to_short",
            _str_bool(args.talon_router_apply_to_short),
            "--talon_router_visual_anchor_ratio",
            str(args.talon_router_visual_anchor_ratio),
            "--talon_router_visual_event_ratio",
            str(args.talon_router_visual_event_ratio),
            "--talon_router_visual_recall_ratio",
            str(args.talon_router_visual_recall_ratio),
            "--talon_router_temporal_anchor_ratio",
            str(args.talon_router_temporal_anchor_ratio),
            "--talon_router_temporal_event_ratio",
            str(args.talon_router_temporal_event_ratio),
            "--talon_router_temporal_recall_ratio",
            str(args.talon_router_temporal_recall_ratio),
            "--talon_router_balanced_anchor_ratio",
            str(args.talon_router_balanced_anchor_ratio),
            "--talon_router_balanced_event_ratio",
            str(args.talon_router_balanced_event_ratio),
            "--talon_router_balanced_recall_ratio",
            str(args.talon_router_balanced_recall_ratio),
            "--talon_router_visual_concentration_threshold",
            str(args.talon_router_visual_concentration_threshold),
            "--talon_router_low_residual_threshold",
            str(args.talon_router_low_residual_threshold),
            "--talon_router_temporal_entropy_threshold",
            str(args.talon_router_temporal_entropy_threshold),
            "--talon_router_temporal_residual_threshold",
            str(args.talon_router_temporal_residual_threshold),
            "--talon_temporal_chunk_aware",
            _str_bool(args.talon_temporal_chunk_aware),
            "--talon_temporal_num_chunks",
            str(args.talon_temporal_num_chunks),
            "--talon_temporal_chunk_min_ratio",
            str(args.talon_temporal_chunk_min_ratio),
            "--talon_temporal_chunk_score",
            args.talon_temporal_chunk_score,
            "--talon_track_aware",
            _str_bool(args.talon_track_aware),
            "--talon_track_budget_ratio",
            str(args.talon_track_budget_ratio),
            "--talon_track_tokens_per_slot",
            str(args.talon_track_tokens_per_slot),
            "--talon_track_score",
            args.talon_track_score,
            "--talon_absorb_dropped_tokens",
            _str_bool(args.talon_absorb_dropped_tokens),
            "--talon_absorb_ratio",
            str(args.talon_absorb_ratio),
            "--talon_absorb_alpha",
            str(args.talon_absorb_alpha),
            "--talon_absorb_score",
            args.talon_absorb_score,
            "--talon_summary_replacement",
            _str_bool(args.talon_summary_replacement),
            "--talon_summary_raw_swap",
            _str_bool(args.talon_summary_raw_swap),
            "--talon_summary_ratio",
            str(args.talon_summary_ratio),
            "--talon_summary_num_chunks",
            str(args.talon_summary_num_chunks),
            "--talon_summary_pool_topk",
            str(args.talon_summary_pool_topk),
            "--talon_summary_alpha",
            str(args.talon_summary_alpha),
            "--talon_summary_score",
            args.talon_summary_score,
            "--talon_output_mode",
            args.talon_output_mode,
            "--talon_reconstruction_blend",
            str(args.talon_reconstruction_blend),
            "--talon_anchor_score_weight",
            str(args.talon_anchor_score_weight),
            "--talon_rank_ratio",
            str(args.talon_rank_ratio),
            "--talon_rank_min",
            str(args.talon_rank_min),
            "--talon_rank_max",
            str(args.talon_rank_max),
            "--talon_background_max_ratio",
            str(args.talon_background_max_ratio),
            "--talon_innovation_attention_weight",
            str(args.talon_innovation_attention_weight),
            "--talon_lite_enabled",
            _str_bool(args.talon_lite_enabled),
            "--talon_echo_residual_weight",
            str(args.talon_echo_residual_weight),
            "--talon_echo_topk_neighbors",
            str(args.talon_echo_topk_neighbors),
            "--talon_echo_temperature",
            str(args.talon_echo_temperature),
            "--talon_echo_score_mode",
            args.talon_echo_score_mode,
            "--talon_final_fused_weight",
            "0.70",
            "--talon_final_residual_weight",
            "0.20",
            "--talon_final_frame_weight",
            "0.10",
            "--talon_use_question_innovation",
            "True",
            "--talon_innovation_qweight",
            "0.20",
            "--talon_deepstack_mode",
            "keep",
        ]
    )


def _append_graphvid_args(cmd: list[str], args: argparse.Namespace) -> None:
    cmd.extend(
        [
            "--llm_retention_ratio",
            str(args.llm_retention_ratio),
            "--retention_ratio",
            str(args.retention_ratio),
            "--expansion",
            str(args.expansion),
            "--temporal_merge_mode",
            "graph",
            "--graph_temporal_topk",
            str(args.graph_temporal_topk),
            "--graph_temporal_radius",
            str(args.graph_temporal_radius),
            "--graph_temporal_skip",
            str(args.graph_temporal_skip),
            "--graph_merge_protect_ratio",
            str(args.graph_merge_protect_ratio),
            "--graph_merge_target_ratio",
            str(args.graph_merge_target_ratio),
            "--graph_merge_representative",
            args.graph_merge_representative,
            "--graph_final_tokens_per_frame",
            str(args.graph_final_tokens_per_frame),
            "--graph_final_frame_floor_ratio",
            str(args.graph_final_frame_floor_ratio),
            "--graph_skip_spatial_merge_when_capped",
            _str_bool(args.graph_skip_spatial_merge_when_capped),
            "--graphvid_token_selection_method",
            args.graphvid_token_selection_method,
        ]
    )


def _launch_shards(args: argparse.Namespace, gpu_ids: list[int], work_dir: Path) -> list[dict[str, object]]:
    ranges = _split_ranges(args.start_index, args.total_limit, len(gpu_ids))
    shard_dir = work_dir / "logs" / "efficiency" / "parallel" / args.tag
    shard_dir.mkdir(parents=True, exist_ok=True)
    jobs = []

    for shard_idx, ((start, limit), gpu_id) in enumerate(zip(ranges, gpu_ids)):
        flashvid_out = shard_dir / f"flashvid_shard{shard_idx:02d}.jsonl"
        ours_out = shard_dir / f"ours_shard{shard_idx:02d}.jsonl"
        graphvid_out = shard_dir / f"graphvid_shard{shard_idx:02d}.jsonl"
        summary_out = shard_dir / f"summary_shard{shard_idx:02d}.json"
        log_out = shard_dir / f"run_shard{shard_idx:02d}.log"

        cmd = [
            sys.executable,
            "-u",
            "playground/bench_all_metrics.py",
            "--model_backend",
            args.model_backend,
            "--model_path",
            args.model_path,
            "--local_files_only",
            _str_bool(args.local_files_only),
            "--dataset_jsonl",
            args.dataset_jsonl,
            "--duration_filter",
            args.duration_filter,
            "--start_index",
            str(start),
            "--limit",
            str(limit),
            "--shuffle",
            "False",
            "--num_frames",
            str(args.num_frames),
            "--min_pixels",
            str(args.min_pixels),
            "--max_pixels",
            str(args.max_pixels),
            "--num_warmup",
            str(args.num_warmup),
            "--num_runs",
            str(args.num_runs),
            "--max_new_tokens",
            str(args.max_new_tokens),
            "--videomme_eval_style",
            args.videomme_eval_style,
            "--attn_implementation",
            args.attn_implementation,
            "--token_selection_method",
            args.token_selection_method,
            "--flashvid_token_selection_method",
            args.flashvid_token_selection_method,
            "--run_baseline",
            "False",
            "--run_flashvid",
            _str_bool(args.run_flashvid),
            "--run_ours",
            _str_bool(args.run_ours and not args.run_graphvid),
            "--run_graphvid",
            _str_bool(args.run_graphvid),
            "--flashvid_output",
            str(flashvid_out),
            "--ours_output",
            str(ours_out),
            "--graphvid_output",
            str(graphvid_out),
            "--summary_output_json",
            str(summary_out),
        ]
        if args.run_graphvid:
            _append_graphvid_args(cmd, args)
        else:
            _append_common_talon_args(cmd, args)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env.setdefault("HF_HOME", args.hf_home)
        env.setdefault("HF_HUB_CACHE", str(Path(env["HF_HOME"]) / "hub"))
        env.setdefault("HF_DATASETS_CACHE", str(Path(env["HF_HOME"]) / "datasets"))

        log_handle = log_out.open("w", encoding="utf-8")
        print(f"[launch] shard={shard_idx} gpu={gpu_id} start={start} limit={limit} log={log_out}")
        proc = subprocess.Popen(
            cmd,
            cwd=work_dir,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        jobs.append(
            {
                "proc": proc,
                "log_handle": log_handle,
                "gpu": gpu_id,
                "start": start,
                "limit": limit,
                "flashvid_out": flashvid_out,
                "ours_out": ours_out,
                "graphvid_out": graphvid_out,
                "summary_out": summary_out,
                "log_out": log_out,
            }
        )
    return jobs


def _wait_jobs(jobs: list[dict[str, object]]) -> None:
    failed = []
    while True:
        running = 0
        for job in jobs:
            proc = job["proc"]
            assert isinstance(proc, subprocess.Popen)
            if proc.poll() is None:
                running += 1
        if running == 0:
            break
        print(f"[wait] running={running}/{len(jobs)}")
        time.sleep(30)

    for i, job in enumerate(jobs):
        proc = job["proc"]
        log_handle = job["log_handle"]
        assert isinstance(proc, subprocess.Popen)
        log_handle.close()
        if proc.returncode != 0:
            failed.append((i, proc.returncode, job["log_out"]))
    if failed:
        for idx, code, log in failed:
            print(f"[failed] shard={idx} code={code} log={log}")
        raise SystemExit(1)


def _combine_jsonl(paths: list[Path], out_path: Path) -> list[dict]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with out_path.open("w", encoding="utf-8") as w:
        for path in paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as r:
                for line in r:
                    if not line.strip():
                        continue
                    rows.append(json.loads(line))
                    w.write(line if line.endswith("\n") else line + "\n")
    return rows


def _write_summary(args: argparse.Namespace, jobs: list[dict[str, object]], shard_dir: Path) -> None:
    sys.path.insert(0, str(REPO_ROOT))
    from playground.bench_all_metrics import (
        _add_duration_breakdown,
        _print_summary,
        _summarize_pairwise_comparison,
        _summarize_phase,
    )

    combined_flashvid = shard_dir / f"{args.tag}_flashvid.jsonl"
    combined_ours = shard_dir / f"{args.tag}_ours.jsonl"
    combined_graphvid = shard_dir / f"{args.tag}_graphvid.jsonl"
    combined_summary = shard_dir / f"{args.tag}_summary.json"

    flashvid_records = []
    if args.run_flashvid:
        flashvid_records = _combine_jsonl([Path(j["flashvid_out"]) for j in jobs], combined_flashvid)
    ours_records = []
    graphvid_records = []
    if args.run_graphvid:
        graphvid_records = _combine_jsonl([Path(j["graphvid_out"]) for j in jobs], combined_graphvid)
    elif args.run_ours:
        ours_records = _combine_jsonl([Path(j["ours_out"]) for j in jobs], combined_ours)

    summary: dict[str, object] = {"comparison": {}}
    if args.run_flashvid:
        summary["flashvid"] = _summarize_phase(flashvid_records)
    if args.run_graphvid:
        summary["graphvid"] = _summarize_phase(graphvid_records)
    elif args.run_ours:
        summary["ours"] = _summarize_phase(ours_records)
    if args.run_flashvid and args.run_graphvid:
        summary["comparison"]["flashvid_vs_graphvid"] = _summarize_pairwise_comparison(
            flashvid_records,
            graphvid_records,
            anchor_name="flashvid",
            target_name="graphvid",
        )
    elif args.run_flashvid and args.run_ours:
        summary["comparison"]["flashvid_vs_ours"] = _summarize_pairwise_comparison(
            flashvid_records,
            ours_records,
            anchor_name="flashvid",
            target_name="ours",
        )
    _add_duration_breakdown(
        summary,
        flashvid_records=flashvid_records if args.run_flashvid else None,
        ours_records=None if args.run_graphvid else ours_records,
        graphvid_records=graphvid_records if args.run_graphvid else None,
    )
    with combined_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    if args.run_graphvid:
        print(f"[combined] graphvid={combined_graphvid}")
    elif args.run_ours:
        print(f"[combined] ours={combined_ours}")
    if args.run_flashvid:
        print(f"[combined] flashvid={combined_flashvid}")
    print(f"[combined] summary={combined_summary}")
    _print_summary(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel TALON recall08 benchmark launcher.")
    parser.add_argument("--model_path", default="/gluster/envs/users/wuzhijian/hf_home/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
    parser.add_argument("--model_backend", default="qwen3_vl")
    parser.add_argument("--dataset_jsonl", default="assets/videomme.jsonl")
    parser.add_argument("--duration_filter", default="", help="Comma-separated durations: short,medium,long.")
    parser.add_argument("--hf_home", default=os.environ.get("HF_HOME", "/gluster/envs/users/wuzhijian/hf_home"))
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--total_limit", type=int, default=200)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--min_pixels", type=int, default=64 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--num_warmup", type=int, default=1)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--videomme_eval_style", default="jsonl")
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--token_selection_method", default="attn_div_v2")
    parser.add_argument("--flashvid_token_selection_method", default="attn_div_v2")
    parser.add_argument("--graphvid_token_selection_method", default="attn_div_v2")
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run_flashvid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run_ours", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run_graphvid", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--retention_ratio", type=float, default=0.10)
    parser.add_argument("--expansion", type=float, default=1.25)
    parser.add_argument("--llm_retention_ratio", type=float, default=1.0)
    parser.add_argument("--compression_variant", default="talon", choices=["talon", "apexvid", "certvid", "certvid_v2", "certvid_v3", "certvid_v6", "certvid_v7", "certvid_v8", "certvid_hr", "certvid_lh", "certvid_v4", "certvid_v5", "certvid_e", "faithvid", "prismvid"])
    parser.add_argument("--pruning_layer", type=int, default=20)
    parser.add_argument("--apex_evidence_ratio", type=float, default=0.45)
    parser.add_argument("--apex_event_ratio", type=float, default=0.30)
    parser.add_argument("--apex_memory_ratio", type=float, default=0.25)
    parser.add_argument("--apex_router_strength", type=float, default=0.50)
    parser.add_argument("--apex_summary_temperature", type=float, default=0.07)
    parser.add_argument("--apex_frame_floor_ratio", type=float, default=0.35)
    parser.add_argument("--apex_question_weight", type=float, default=0.20)
    parser.add_argument("--cert_budget_uses_expansion", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cert_query_atoms", type=int, default=6)
    parser.add_argument("--cert_temporal_bins", type=int, default=8)
    parser.add_argument("--cert_spatial_bins", type=int, default=3)
    parser.add_argument("--cert_candidate_multiplier", type=float, default=3.0)
    parser.add_argument("--cert_query_weight", type=float, default=0.20)
    parser.add_argument("--cert_temporal_weight", type=float, default=0.20)
    parser.add_argument("--cert_detail_weight", type=float, default=0.10)
    parser.add_argument("--cert_repair_ratio", type=float, default=0.20)
    parser.add_argument("--cert_fusion_alpha", type=float, default=0.25)
    parser.add_argument("--cert_assignment_temperature", type=float, default=0.07)
    parser.add_argument("--cert_track_threshold", type=float, default=0.82)
    parser.add_argument("--cert_spatial_penalty", type=float, default=0.08)
    parser.add_argument("--cert_metric_dim", type=int, default=256)
    parser.add_argument("--certv2_budget_uses_expansion", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--certv2_query_atoms", type=int, default=6)
    parser.add_argument("--certv2_temporal_bins", type=int, default=8)
    parser.add_argument("--certv2_spatial_bins", type=int, default=3)
    parser.add_argument("--certv2_candidate_multiplier", type=float, default=3.0)
    parser.add_argument("--certv2_query_weight", type=float, default=0.18)
    parser.add_argument("--certv2_frame_floor_ratio", type=float, default=0.08)
    parser.add_argument("--certv2_diversity_weight", type=float, default=0.12)
    parser.add_argument("--certv2_coverage_weight", type=float, default=0.10)
    parser.add_argument("--certv2_density_neighbors", type=int, default=4)
    parser.add_argument("--certv2_track_threshold", type=float, default=0.82)
    parser.add_argument("--certv2_spatial_penalty", type=float, default=0.08)
    parser.add_argument("--certv2_metric_dim", type=int, default=256)
    parser.add_argument("--certv2_repair_ratio", type=float, default=0.05)
    parser.add_argument("--certv2_repair_ratio_high", type=float, default=0.13)
    parser.add_argument("--certv2_router_strength", type=float, default=0.65)
    parser.add_argument("--certv2_protect_ratio", type=float, default=0.30)
    parser.add_argument("--certv2_swap_margin", type=float, default=0.02)
    parser.add_argument("--certv2_fusion_alpha", type=float, default=0.25)
    parser.add_argument("--certv2_repair_fusion_alpha", type=float, default=0.08)
    parser.add_argument("--certv2_assignment_temperature", type=float, default=0.07)
    parser.add_argument("--certv3_budget_uses_expansion", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--certv3_query_atoms", type=int, default=8)
    parser.add_argument("--certv3_temporal_bins", type=int, default=12)
    parser.add_argument("--certv3_spatial_bins", type=int, default=3)
    parser.add_argument("--certv3_candidate_multiplier", type=float, default=2.5)
    parser.add_argument("--certv3_query_weight", type=float, default=0.18)
    parser.add_argument("--certv3_track_threshold", type=float, default=0.82)
    parser.add_argument("--certv3_spatial_penalty", type=float, default=0.08)
    parser.add_argument("--certv3_metric_dim", type=int, default=96)
    parser.add_argument("--certv3_frame_coverage_ratio", type=float, default=1.0)
    parser.add_argument("--certv3_cell_coverage_ratio", type=float, default=0.50)
    parser.add_argument("--certv3_query_threshold", type=float, default=0.10)
    parser.add_argument("--certv3_query_per_atom", type=int, default=1)
    parser.add_argument("--certv3_structural_weight", type=float, default=0.32)
    parser.add_argument("--certv3_whitening_strength", type=float, default=0.50)
    parser.add_argument("--certv3_quality_floor", type=float, default=0.15)
    parser.add_argument("--certv3_ridge", type=float, default=0.50)
    parser.add_argument("--certv3_swap_steps", type=int, default=6)
    parser.add_argument("--certv3_swap_pool", type=int, default=24)
    parser.add_argument("--certv3_swap_margin", type=float, default=1e-4)
    parser.add_argument("--certv3_fusion_alpha", type=float, default=0.12)
    parser.add_argument("--certv3_assignment_temperature", type=float, default=0.07)
    parser.add_argument("--certv6_scene_temporal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--certv6_gate_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--certv6_continuity_low", type=float, default=0.55)
    parser.add_argument("--certv6_continuity_high", type=float, default=0.80)
    parser.add_argument("--certv6_query_per_atom_max", type=int, default=3)
    parser.add_argument("--certv7_min_duration_seconds", type=float, default=120.0)
    parser.add_argument("--certv7_transport_spatial_bins", type=int, default=4)
    parser.add_argument("--certv7_transport_epsilon", type=float, default=0.08)
    parser.add_argument("--certv7_transport_steps", type=int, default=8)
    parser.add_argument("--certv7_transport_spatial_weight", type=float, default=0.20)
    parser.add_argument("--certv7_frame_floor_ratio", type=float, default=1.0)
    parser.add_argument("--certv7_frame_cap_ratio", type=float, default=1.0)
    parser.add_argument("--certv7_budget_temperature", type=float, default=0.50)
    parser.add_argument("--certv7_uniqueness_weight", type=float, default=0.25)
    parser.add_argument("--certv7_transport_weight", type=float, default=0.35)
    parser.add_argument("--certv7_event_weight", type=float, default=0.20)
    parser.add_argument("--certv7_query_weight", type=float, default=0.20)
    parser.add_argument(
        "--certv7_budget_rounding",
        choices=["per_frame_ceil", "global_round"],
        default="per_frame_ceil",
    )
    parser.add_argument("--certv7_v3_certificate_ratio", type=float, default=0.05)
    parser.add_argument("--certv7_relay_ratio", type=float, default=0.25)
    parser.add_argument("--certv7_relay_query_share", type=float, default=0.25)
    parser.add_argument("--certv7_transition_relay_share", type=float, default=0.45)
    parser.add_argument("--certv7_query_peaks_per_atom", type=int, default=2)
    parser.add_argument("--certv7_query_min_frame_gap", type=int, default=3)
    parser.add_argument("--certv7_query_peak_threshold", type=float, default=0.70)
    parser.add_argument("--certv7_query_context_radius", type=int, default=1)
    parser.add_argument("--certv7_transition_pairs_per_boundary", type=int, default=2)
    parser.add_argument("--certv7_transition_min_similarity", type=float, default=0.30)
    parser.add_argument("--certv7_trajectory_min_span", type=int, default=3)
    parser.add_argument("--certv7_trajectory_points", type=int, default=3)
    parser.add_argument("--certv7_facility_quality_mix", type=float, default=0.18)
    parser.add_argument("--certv7_min_reallocation_ratio", type=float, default=0.02)
    parser.add_argument("--certv7_d_efficiency_floor", type=float, default=0.80)
    parser.add_argument("--certv7_assignment_topk", type=int, default=2)
    parser.add_argument("--certv7_assignment_temperature", type=float, default=0.07)
    parser.add_argument("--certv7_cross_frame_cost_quantile", type=float, default=0.45)
    parser.add_argument("--certv7_cross_frame_similarity", type=float, default=0.82)
    parser.add_argument("--certv7_cross_frame_max_seconds", type=float, default=12.0)
    parser.add_argument("--certv7_component_bonus", type=float, default=0.08)
    parser.add_argument("--certv7_design_protect_ratio", type=float, default=0.15)
    parser.add_argument("--certv7_long_fusion_alpha", type=float, default=0.04)
    parser.add_argument("--certv7_debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--certhr_horizon_gap_seconds", type=float, default=4.0)
    parser.add_argument("--certhr_chunk_max_seconds", type=float, default=60.0)
    parser.add_argument("--certhr_chunk_max_units", type=int, default=4)
    parser.add_argument("--certhr_semantic_quantile", type=float, default=0.85)
    parser.add_argument("--certhr_semantic_floor", type=float, default=0.10)
    parser.add_argument("--certhr_coverage_floor", type=float, default=0.70)
    parser.add_argument("--certhr_deficit_threshold", type=float, default=0.05)
    parser.add_argument("--certhr_query_peak_quantile", type=float, default=0.90)
    parser.add_argument("--certhr_query_peak_floor", type=float, default=0.75)
    parser.add_argument("--certhr_max_swap_ratio", type=float, default=0.05)
    parser.add_argument("--certhr_d_efficiency_floor", type=float, default=0.995)
    parser.add_argument("--certhr_add_pool", type=int, default=32)
    parser.add_argument("--certhr_remove_pool", type=int, default=24)
    parser.add_argument("--certhr_debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--certv4_budget_mode", default="layer_average", choices=["layer_average", "outer_only"])
    parser.add_argument("--certv4_attention_policy", default="validated", choices=["validated", "strict", "off"])
    parser.add_argument("--certv4_attention_eps", type=float, default=1e-6)
    parser.add_argument("--certv4_certificate_budget_ratio", type=float, default=0.40)
    parser.add_argument("--certv4_query_mode", default="certificates_and_design", choices=["certificates_only", "design_only", "certificates_and_design", "off"])
    parser.add_argument("--certv4_design_protect_ratio", type=float, default=0.15)
    parser.add_argument("--certv4_query_atoms", type=int, default=8)
    parser.add_argument("--certv4_temporal_bins", type=int, default=12)
    parser.add_argument("--certv4_spatial_bins", type=int, default=3)
    parser.add_argument("--certv4_candidate_multiplier", type=float, default=2.5)
    parser.add_argument("--certv4_track_threshold", type=float, default=0.82)
    parser.add_argument("--certv4_spatial_penalty", type=float, default=0.08)
    parser.add_argument("--certv4_metric_dim", type=int, default=96)
    parser.add_argument("--certv4_frame_coverage_ratio", type=float, default=1.0)
    parser.add_argument("--certv4_cell_coverage_ratio", type=float, default=0.50)
    parser.add_argument("--certv4_query_threshold", type=float, default=0.10)
    parser.add_argument("--certv4_query_per_atom", type=int, default=1)
    parser.add_argument("--certv4_structural_weight", type=float, default=0.32)
    parser.add_argument("--certv4_whitening_strength", type=float, default=0.50)
    parser.add_argument("--certv4_quality_floor", type=float, default=0.15)
    parser.add_argument("--certv4_ridge", type=float, default=0.50)
    parser.add_argument("--certv4_swap_steps", type=int, default=6)
    parser.add_argument("--certv4_swap_pool", type=int, default=24)
    parser.add_argument("--certv4_swap_margin", type=float, default=1e-4)
    parser.add_argument("--certv4_fusion_alpha", type=float, default=0.12)
    parser.add_argument("--certv4_assignment_temperature", type=float, default=0.07)
    parser.add_argument("--certv4_debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--certv5_budget_mode", default="layer_average", choices=["layer_average", "outer_only"])
    parser.add_argument("--certv5_ot_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--certv5_ot_topk", type=int, default=4)
    parser.add_argument("--certv5_ot_temperature", type=float, default=0.07)
    parser.add_argument("--certv5_ot_steps", type=int, default=6)
    parser.add_argument("--certv5_ot_capacity_tau", type=float, default=0.10)
    parser.add_argument("--certv5_ot_prior_shrink", type=float, default=0.10)
    parser.add_argument("--certv5_ot_live_fraction", type=float, default=0.25)
    parser.add_argument("--certv5_ot_cost_slack", type=float, default=0.05)
    parser.add_argument("--certv5_ot_temporal_penalty", type=float, default=0.04)
    parser.add_argument("--certv5_ot_max_displacement", type=float, default=0.12)
    parser.add_argument("--certv5_ot_min_cosine", type=float, default=0.98)
    parser.add_argument("--certv5_debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--certe_budget_uses_expansion", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--certe_ridge", type=float, default=0.50)
    parser.add_argument("--certe_bottom_k", type=int, default=8)
    parser.add_argument("--certe_swap_steps", type=int, default=6)
    parser.add_argument("--certe_remove_pool", type=int, default=8)
    parser.add_argument("--certe_add_pool", type=int, default=16)
    parser.add_argument("--certe_verify_pool", type=int, default=4)
    parser.add_argument("--certe_swap_margin", type=float, default=1e-5)
    parser.add_argument("--certe_spectral_temperature", type=float, default=0.05)
    parser.add_argument("--certe_d_efficiency_floor", type=float, default=0.995)
    parser.add_argument("--certe_rank_tolerance", type=float, default=1e-5)
    parser.add_argument("--certe_debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--faith_budget_uses_expansion", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--faith_mass_strength", type=float, default=1.0)
    parser.add_argument("--faith_variance_strength", type=float, default=0.50)
    parser.add_argument("--faith_merge_alpha", type=float, default=1.0)
    parser.add_argument("--faith_temporal_radius", type=int, default=1)
    parser.add_argument("--faith_spatial_radius", type=float, default=0.75)
    parser.add_argument("--faith_component_bonus", type=float, default=0.08)
    parser.add_argument("--faith_temporal_penalty", type=float, default=0.04)
    parser.add_argument("--faith_spatial_penalty", type=float, default=0.04)
    parser.add_argument("--faith_assignment_topk", type=int, default=2)
    parser.add_argument("--faith_assignment_temperature", type=float, default=0.07)
    parser.add_argument("--faith_max_log_bias", type=float, default=20.0)
    parser.add_argument("--faith_attention_strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--faith_debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prism_budget_uses_expansion", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prism_metric_dim", type=int, default=256)
    parser.add_argument("--prism_query_atoms", type=int, default=6)
    parser.add_argument("--prism_candidate_multiplier", type=float, default=2.25)
    parser.add_argument("--prism_probe_tokens", type=int, default=512)
    parser.add_argument("--prism_frame_floor_ratio", type=float, default=0.20)
    parser.add_argument("--prism_attention_weight", type=float, default=0.30)
    parser.add_argument("--prism_event_weight", type=float, default=0.24)
    parser.add_argument("--prism_query_weight", type=float, default=0.16)
    parser.add_argument("--prism_disagreement_weight", type=float, default=0.16)
    parser.add_argument("--prism_router_strength", type=float, default=0.50)
    parser.add_argument("--prism_coverage_weight", type=float, default=0.68)
    parser.add_argument("--prism_pareto_weight", type=float, default=0.20)
    parser.add_argument("--prism_batch_size", type=int, default=8)
    parser.add_argument("--free_ratio", type=float, default=0.75)
    parser.add_argument("--min_free_mb", type=int, default=18000)
    parser.add_argument("--max_gpus", type=int, default=0, help="0 means use all eligible GPUs.")
    parser.add_argument("--gpu_ids", default="", help="Comma-separated GPU ids. Overrides auto selection.")
    parser.add_argument("--tag", default="talon_recall08_t20_parallel")
    parser.add_argument("--talon_target_tokens_per_frame", type=int, default=20)
    parser.add_argument("--graph_temporal_topk", type=int, default=3)
    parser.add_argument("--graph_temporal_radius", type=int, default=1)
    parser.add_argument("--graph_temporal_skip", type=int, default=1)
    parser.add_argument("--graph_merge_protect_ratio", type=float, default=0.15)
    parser.add_argument("--graph_merge_target_ratio", type=float, default=0.65)
    parser.add_argument("--graph_merge_representative", default="medoid", choices=["medoid", "mean"])
    parser.add_argument("--graph_final_tokens_per_frame", type=int, default=0)
    parser.add_argument("--graph_final_frame_floor_ratio", type=float, default=0.55)
    parser.add_argument("--graph_skip_spatial_merge_when_capped", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--talon_short_target_tokens_per_frame", type=int, default=0)
    parser.add_argument("--talon_medium_target_tokens_per_frame", type=int, default=0)
    parser.add_argument("--talon_long_target_tokens_per_frame", type=int, default=0)
    parser.add_argument("--talon_adaptive_target_enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--talon_adaptive_target_low", type=int, default=0)
    parser.add_argument("--talon_adaptive_target_mid", type=int, default=0)
    parser.add_argument("--talon_adaptive_target_high", type=int, default=0)
    parser.add_argument("--talon_complexity_floor", type=float, default=0.20)
    parser.add_argument("--talon_complexity_ceil", type=float, default=0.40)
    parser.add_argument("--talon_adaptive_gamma", type=float, default=1.0)
    parser.add_argument("--talon_target_mean_cap", type=float, default=0.0)
    parser.add_argument("--talon_question_recall_ratio", type=float, default=0.08)
    parser.add_argument("--talon_question_recall_qweight", type=float, default=0.65)
    parser.add_argument("--talon_persistence_recall_ratio", type=float, default=0.0)
    parser.add_argument("--talon_persistence_recall_qweight", type=float, default=0.50)
    parser.add_argument("--talon_persistence_recall_pweight", type=float, default=0.35)
    parser.add_argument("--talon_persistence_apply_to_short", type=_parse_bool, default=False)
    parser.add_argument("--talon_persistence_apply_to_medium", type=_parse_bool, default=True)
    parser.add_argument("--talon_persistence_apply_to_long", type=_parse_bool, default=False)
    parser.add_argument("--talon_object_evidence_ratio", type=float, default=0.0)
    parser.add_argument("--talon_object_evidence_qweight", type=float, default=0.35)
    parser.add_argument("--talon_object_evidence_sweight", type=float, default=0.45)
    parser.add_argument("--talon_object_evidence_pweight", type=float, default=0.10)
    parser.add_argument("--talon_object_evidence_apply_to_short", type=_parse_bool, default=False)
    parser.add_argument("--talon_object_evidence_apply_to_medium", type=_parse_bool, default=True)
    parser.add_argument("--talon_object_evidence_apply_to_long", type=_parse_bool, default=False)
    parser.add_argument("--talon_question_pooling", default="mean")
    parser.add_argument("--talon_question_pooling_topk", type=int, default=4)
    parser.add_argument("--talon_question_contrast_weight", type=float, default=0.0)
    parser.add_argument("--talon_question_contrast_apply_to_short", type=_parse_bool, default=False)
    parser.add_argument("--talon_monotonic_base_tokens_per_frame", type=int, default=20)
    parser.add_argument("--talon_frame_local_budget_ratio", type=float, default=1.0)
    parser.add_argument("--talon_anchor_diversity_weight", type=float, default=0.0)
    parser.add_argument("--talon_spatial_anchor_coverage", type=_parse_bool, default=False)
    parser.add_argument("--talon_spatial_anchor_ratio", type=float, default=0.35)
    parser.add_argument("--talon_spatial_anchor_rows", type=int, default=3)
    parser.add_argument("--talon_spatial_anchor_cols", type=int, default=3)
    parser.add_argument("--talon_spatial_anchor_score", default="fused", choices=["combined", "fused", "question", "event"])
    parser.add_argument("--talon_spatial_anchor_apply_to_short", type=_parse_bool, default=False)
    parser.add_argument("--talon_frame_coverage_floor_ratio", type=float, default=0.65)
    parser.add_argument("--talon_frame_importance_pooling", default="mean", choices=["mean", "topk", "max", "evidence"])
    parser.add_argument("--talon_frame_importance_topk", type=int, default=6)
    parser.add_argument("--talon_medium_frame_coverage_floor_ratio", type=float, default=-1.0)
    parser.add_argument("--talon_long_frame_coverage_floor_ratio", type=float, default=-1.0)
    parser.add_argument("--talon_budget_mode", default="attention", choices=["attention", "uniform"])
    parser.add_argument("--talon_lite_enabled", type=_parse_bool, default=False)
    parser.add_argument("--talon_echo_residual_weight", type=float, default=0.0)
    parser.add_argument("--talon_echo_topk_neighbors", type=int, default=4)
    parser.add_argument("--talon_echo_temperature", type=float, default=0.07)
    parser.add_argument("--talon_echo_score_mode", default="mse", choices=["mse", "cosine"])
    parser.add_argument("--talon_output_mode", default="manifold", choices=["manifold", "full", "lowrank", "coefficient"])
    parser.add_argument("--talon_reconstruction_blend", type=float, default=0.0)
    parser.add_argument("--talon_anchor_score_weight", type=float, default=0.35)
    parser.add_argument("--talon_rank_ratio", type=float, default=0.40)
    parser.add_argument("--talon_rank_min", type=int, default=1)
    parser.add_argument("--talon_rank_max", type=int, default=8)
    parser.add_argument("--talon_background_max_ratio", type=float, default=0.35)
    parser.add_argument("--talon_innovation_attention_weight", type=float, default=0.65)
    parser.add_argument("--talon_duration_aware", type=_parse_bool, default=False)
    parser.add_argument("--talon_medium_anchor_safety_ratio", type=float, default=0.72)
    parser.add_argument("--talon_medium_event_budget_ratio", type=float, default=0.30)
    parser.add_argument("--talon_medium_global_topk_ratio", type=float, default=0.70)
    parser.add_argument("--talon_long_anchor_safety_ratio", type=float, default=0.80)
    parser.add_argument("--talon_long_event_budget_ratio", type=float, default=0.14)
    parser.add_argument("--talon_long_global_topk_ratio", type=float, default=0.85)
    parser.add_argument("--talon_task_aware_event", type=_parse_bool, default=False)
    parser.add_argument("--talon_task_event_attention_weight", type=float, default=0.82)
    parser.add_argument("--talon_task_event_qweight", type=float, default=0.30)
    parser.add_argument("--talon_visual_task_balance", type=_parse_bool, default=False)
    parser.add_argument("--talon_visual_task_anchor_ratio", type=float, default=0.84)
    parser.add_argument("--talon_visual_task_event_ratio", type=float, default=0.12)
    parser.add_argument("--talon_visual_task_recall_ratio", type=float, default=0.02)
    parser.add_argument("--talon_knowledge_visual_anchor_ratio", type=float, default=0.78)
    parser.add_argument("--talon_knowledge_visual_event_ratio", type=float, default=0.18)
    parser.add_argument("--talon_knowledge_visual_recall_ratio", type=float, default=0.06)
    parser.add_argument("--talon_adaptive_router", type=_parse_bool, default=False)
    parser.add_argument("--talon_router_apply_to_short", type=_parse_bool, default=False)
    parser.add_argument("--talon_router_visual_anchor_ratio", type=float, default=0.76)
    parser.add_argument("--talon_router_visual_event_ratio", type=float, default=0.24)
    parser.add_argument("--talon_router_visual_recall_ratio", type=float, default=0.06)
    parser.add_argument("--talon_router_temporal_anchor_ratio", type=float, default=0.66)
    parser.add_argument("--talon_router_temporal_event_ratio", type=float, default=0.34)
    parser.add_argument("--talon_router_temporal_recall_ratio", type=float, default=0.08)
    parser.add_argument("--talon_router_balanced_anchor_ratio", type=float, default=0.72)
    parser.add_argument("--talon_router_balanced_event_ratio", type=float, default=0.30)
    parser.add_argument("--talon_router_balanced_recall_ratio", type=float, default=0.08)
    parser.add_argument("--talon_router_visual_concentration_threshold", type=float, default=0.28)
    parser.add_argument("--talon_router_low_residual_threshold", type=float, default=0.30)
    parser.add_argument("--talon_router_temporal_entropy_threshold", type=float, default=0.95)
    parser.add_argument("--talon_router_temporal_residual_threshold", type=float, default=0.36)
    parser.add_argument("--talon_temporal_chunk_aware", type=_parse_bool, default=False)
    parser.add_argument("--talon_temporal_num_chunks", type=int, default=4)
    parser.add_argument("--talon_temporal_chunk_min_ratio", type=float, default=0.18)
    parser.add_argument("--talon_temporal_chunk_score", default="combined", choices=["combined", "fused", "question", "event"])
    parser.add_argument("--talon_track_aware", type=_parse_bool, default=False)
    parser.add_argument("--talon_track_budget_ratio", type=float, default=0.12)
    parser.add_argument("--talon_track_tokens_per_slot", type=int, default=1)
    parser.add_argument("--talon_track_score", default="combined", choices=["combined", "fused", "question", "event"])
    parser.add_argument("--talon_absorb_dropped_tokens", type=_parse_bool, default=False)
    parser.add_argument("--talon_absorb_ratio", type=float, default=0.35)
    parser.add_argument("--talon_absorb_alpha", type=float, default=0.25)
    parser.add_argument("--talon_absorb_score", default="combined", choices=["combined", "fused", "question", "event"])
    parser.add_argument("--talon_summary_replacement", type=_parse_bool, default=False)
    parser.add_argument("--talon_summary_raw_swap", type=_parse_bool, default=False)
    parser.add_argument("--talon_summary_ratio", type=float, default=0.08)
    parser.add_argument("--talon_summary_num_chunks", type=int, default=8)
    parser.add_argument("--talon_summary_pool_topk", type=int, default=12)
    parser.add_argument("--talon_summary_alpha", type=float, default=0.55)
    parser.add_argument("--talon_summary_score", default="combined", choices=["combined", "fused", "question", "event"])
    args = parser.parse_args()

    if args.gpu_ids.strip():
        gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
    else:
        gpu_ids = _query_free_gpus(args.free_ratio, args.min_free_mb)
    if args.max_gpus > 0:
        gpu_ids = gpu_ids[: args.max_gpus]
    if not gpu_ids:
        raise SystemExit("No eligible GPU found. Lower --free_ratio/--min_free_mb or pass --gpu_ids.")
    if args.total_limit <= 0:
        raise SystemExit("--total_limit must be positive.")

    jobs = _launch_shards(args, gpu_ids, REPO_ROOT)
    _wait_jobs(jobs)
    shard_dir = REPO_ROOT / "logs" / "efficiency" / "parallel" / args.tag
    _write_summary(args, jobs, shard_dir)


if __name__ == "__main__":
    main()
