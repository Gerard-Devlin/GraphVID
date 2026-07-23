import gc
import inspect
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)

process_vision_info, _has_qwen_vl = optional_import("qwen_vl_utils", "process_vision_info")
if not _has_qwen_vl:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


SUPPORTED_QWEN3_BASELINE_ADAPTERS = ("fastvid", "visionzip", "fastgraphvid")
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def _safe_nframes_for_video(total_frames: Optional[int], requested: int) -> int:
    if total_frames is None:
        return int(requested)
    total_frames = int(total_frames)
    requested = int(requested)
    if total_frames <= 0:
        return requested
    capped = min(requested, total_frames)
    if capped > 2 and capped % 2 == 1:
        capped -= 1
    if total_frames >= 2:
        capped = max(2, min(capped, total_frames))
    return capped


def _parse_qwen_nframes_limit(error: ValueError) -> Optional[int]:
    match = re.search(r"nframes should in interval \[\d+,\s*(\d+)\], but got \d+", str(error))
    if not match:
        return None
    max_allowed = int(match.group(1))
    if max_allowed < 2:
        return None
    if max_allowed > 2 and max_allowed % 2 == 1:
        max_allowed -= 1
    return max(2, max_allowed)


def _set_video_frame_limit(messages: List[List[dict]], nframes: int) -> None:
    for message in messages:
        for turn in message:
            content = turn.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "video":
                    part.pop("fps", None)
                    part.pop("max_frames", None)
                    part["nframes"] = int(nframes)


def _release_video_file_cache(paths: List[str]) -> None:
    if os.environ.get("LMMS_EVAL_FADVISE_DONTNEED", "1").lower() in {"0", "false", "no"}:
        return
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return
    for path in dict.fromkeys(paths):
        if not isinstance(path, str) or not os.path.isfile(path):
            continue
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError:
            pass


def _metadata_value(metadata, name):
    if isinstance(metadata, dict):
        return metadata.get(name)
    return getattr(metadata, name, None)


def _qwen_frame_timing(video_metadata):
    if video_metadata is None or len(video_metadata) != 1:
        return None
    metadata = video_metadata[0]
    frame_indices = _metadata_value(metadata, "frames_indices")
    fps = _metadata_value(metadata, "fps")
    if frame_indices is None or fps is None:
        return None
    try:
        indices = torch.as_tensor(frame_indices, dtype=torch.float64).reshape(-1)
        fps_value = float(torch.as_tensor(fps).reshape(-1)[0].item())
    except (TypeError, ValueError, RuntimeError, IndexError):
        return None
    if indices.numel() == 0 or fps_value <= 0.0 or not torch.isfinite(indices).all():
        return None
    return indices.div(fps_value).tolist(), "qwen3_video_metadata"


def _flashvid_runtime_config(model):
    candidates = [model, getattr(model, "model", None)]
    nested = getattr(getattr(model, "model", None), "language_model", None)
    candidates.append(nested)
    for candidate in candidates:
        config = getattr(candidate, "flashvid_config", None) if candidate is not None else None
        if config is not None:
            return config
    return None


def _publish_certhr_timing(model, timing):
    config = _flashvid_runtime_config(model)
    if config is None:
        return None
    config._certvid_frame_times_sec = None
    config._certvid_frame_times_source = "missing"
    if str(getattr(config, "compression_variant", "")).strip().lower() in {"certvid_hr", "certvid_lh", "certvid_v7", "certvid_v8"} and timing is not None:
        config._certvid_frame_times_sec, config._certvid_frame_times_source = timing
    return config


def _clear_certhr_timing(config) -> None:
    if config is not None:
        config._certvid_frame_times_sec = None
        config._certvid_frame_times_source = "missing"


def _doc_value(doc, *names):
    if not isinstance(doc, dict):
        return None
    for name in names:
        value = doc.get(name)
        if value is not None and str(value).strip():
            return value
    return None


def _publish_certvid_sample(config, doc, doc_id, context, task):
    if config is None:
        return
    sample_id = _doc_value(doc, "id", "question_id", "uid", "video_id")
    question = _doc_value(doc, "question", "query", "instruction", "input")
    category = _doc_value(
        doc,
        "question_category",
        "question_type",
        "category",
        "duration",
    )
    config._debug_sample_id = str(sample_id if sample_id is not None else doc_id)
    config._certvid_query_text = str(question if question is not None else context)
    config._certvid_eval_category = None if category is None else str(category)
    config._certvid_task_name = str(task)


def _clear_certvid_sample(config) -> None:
    if config is None:
        return
    config._debug_sample_id = "unknown"
    config._certvid_query_text = ""
    config._certvid_eval_category = None
    config._certvid_task_name = None


def _install_qwen3_baseline_adapter_patch() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import flashvid.modeling_qwen3_vl as modeling_qwen3_vl
    from playground.qwen3_baseline_adapters import adapter_baseline_compression

    original = getattr(
        modeling_qwen3_vl,
        "_lmms_eval_original_flashvid_compression",
        modeling_qwen3_vl.flashvid_compression,
    )
    modeling_qwen3_vl._lmms_eval_original_flashvid_compression = original

    def patched_flashvid_compression(
        *,
        video_features,
        cls_attention,
        flashvid_config,
        question_features=None,
        deepstack_features=None,
    ):
        variant = str(getattr(flashvid_config, "compression_variant", "")).strip().lower()
        if variant in SUPPORTED_QWEN3_BASELINE_ADAPTERS:
            return adapter_baseline_compression(video_features, cls_attention, flashvid_config)
        return original(
            video_features=video_features,
            cls_attention=cls_attention,
            flashvid_config=flashvid_config,
            question_features=question_features,
            deepstack_features=deepstack_features,
        )

    modeling_qwen3_vl.flashvid_compression = patched_flashvid_compression


@register_model("qwen3_vl")
class Qwen3_VL(lmms):
    """
    Qwen3_VL Model
    "https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct"
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-4B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        total_pixels: Optional[int] = None,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,  # Only applicable if use_custom_video_loader is True
        max_image_size: Optional[int] = None,  # Only applicable if use_custom_video_loader is True
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        # ! FlashVid parameters.
        enable_flashvid: bool = False,
        retention_ratio: float = 0.25,
        # DySeg parameters (Fixed)
        do_segment: bool = True,
        segment_threshold: float = 0.9,
        min_segment_num: int = 8,
        complementary_segment: bool = True,
        # ADTS and TSTM parameters
        token_selection_method: str = "attn_div_v2",
        flashvid_token_selection_method: Optional[str] = None,
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
        # Slot-memory experimental parameters
        compression_variant: str = "flashvid",
        question_aware_reweighting: bool = False,
        question_reweight_beta: float = 0.35,
        graph_topk: int = 4,
        graph_temporal_topk: Optional[int] = None,
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
        # CertVID V8 parameters
        certv8_enabled: bool = True,
        certv8_intent_router: bool = True,
        certv8_intent_strength: float = 0.75,
        certv8_min_horizon_gap_seconds: float = 4.0,
        certv8_min_deficit: float = 0.04,
        certv8_frame_floor_ratio: float = 0.45,
        certv8_frame_cap_ratio: float = 2.00,
        certv8_max_swap_ratio: float = 0.30,
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
        certv8_debug: bool = False,
        certhr_horizon_gap_seconds: float = 4.0,
        certhr_chunk_max_seconds: float = 60.0,
        certhr_chunk_max_units: int = 4,
        certhr_semantic_quantile: float = 0.85,
        certhr_semantic_floor: float = 0.10,
        certhr_coverage_floor: float = 0.70,
        certhr_deficit_threshold: float = 0.05,
        certhr_query_peak_quantile: float = 0.90,
        certhr_query_peak_floor: float = 0.75,
        certhr_max_swap_ratio: float = 0.05,
        certhr_d_efficiency_floor: float = 0.995,
        certhr_add_pool: int = 32,
        certhr_remove_pool: int = 24,
        certhr_debug: bool = False,
        certlh_min_duration_seconds: float = 120.0,
        certlh_horizon_gap_seconds: float = 4.0,
        certlh_gate_threshold: float = 0.55,
        certlh_min_groups: int = 4,
        certlh_max_groups: int = 8,
        certlh_min_group_units: int = 2,
        certlh_max_group_units: int = 8,
        certlh_event_quantile: float = 0.80,
        certlh_event_floor: float = 0.08,
        certlh_group_floor_ratio: float = 0.50,
        certlh_budget_temperature: float = 0.25,
        certlh_query_weight: float = 0.35,
        certlh_relay_ratio: float = 0.10,
        certlh_query_peaks_per_atom: int = 2,
        certlh_query_peak_quantile: float = 0.90,
        certlh_query_peak_floor: float = 0.75,
        certlh_query_min_group_distance: int = 2,
        certlh_cross_group_similarity: float = 0.90,
        certlh_cross_group_max_seconds: float = 8.0,
        certlh_debug: bool = False,
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
        adapter_budget_uses_expansion: bool = True,
        external_budget_uses_expansion: bool = True,
        fastvid_DySeg_c: int = 8,
        fastvid_DySeg_tau: float = 0.90,
        fastvid_DySeg_ignore: float = 0.95,
        fastvid_STPrune_d: float = 0.40,
        fastvid_DTM_p: int = 4,
        fastvid_DTM_beta: float = 0.60,
        visionzip_dominant_ratio: float = 0.85,
        slot_base_roles: int = 5,
        slot_max_per_segment: int = 64,
        slot_role_allocation: str = "motion,interaction,detail,scene,background",
        slot_overlap_radius: int = 1,
        slot_tiebreak_eps: float = 2e-2,
        slot_motion_window: int = 1,
        slot_soft_cap_fraction: float = 0.35,
        slot_anchor_blend: float = 0.65,
        slot_passthrough_ratio: float = 0.55,
        slot_passthrough_min: int = 4,
        slot_fast_assignment: bool = True,
        talon_transport_radius: int = 1,
        talon_rank_ratio: float = 0.40,
        talon_rank_min: int = 2,
        talon_rank_max: int = 32,
        talon_budget_mode: str = "uniform",
        talon_use_question_innovation: bool = True,
        talon_innovation_qweight: float = 0.25,
        talon_output_mode: str = "manifold",
        talon_reconstruction_blend: float = 0.25,
        talon_anchor_score_weight: float = 0.35,
        memory_token_ratio: float = 0.10,
        memory_token_min: int = 1,
        memory_token_max: int = 16,
        adaptive_token_budget: bool = False,
        adaptive_budget_low: float = 0.10,
        adaptive_budget_mid: float = 0.15,
        adaptive_budget_high: float = 0.20,
        # Inner-LLM Pruning parameters
        expansion: float = 1.25,
        pruning_layer: int = 20,
        llm_retention_ratio: float = 0.3,
        # Decode-stage policy scaffold (default no-op)
        decode_policy: str = "none",
        decode_kv_budget_ratio: float = 1.0,
        decode_update_interval: int = 4,
        decode_start_layer: int = 0,
        # Appended to keep existing positional model_args compatible.
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
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Validate attention implementation
        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        # if self.fps and not self.use_custom_video_loader:
        #     raise ValueError("FPS is only applicable if use_custom_video_loader is True")
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # Prepare model loading arguments
        model_kwargs = {
            "dtype": "bfloat16",
            "device_map": self.device_map,
        }

        # Add attention implementation if specified
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        # check whether its an MoE model
        match = re.search(r"A\d+B", pretrained)
        model_fn = Qwen3VLMoeForConditionalGeneration if match else Qwen3VLForConditionalGeneration
        self._model = model_fn.from_pretrained(pretrained, **model_kwargs)
        # ! Enable FlashVID
        if enable_flashvid:
            from flashvid import flashvid

            variant = str(compression_variant).strip().lower()
            effective_token_selection_method = flashvid_token_selection_method or token_selection_method
            flashvid_init_variant = variant
            if variant in SUPPORTED_QWEN3_BASELINE_ADAPTERS:
                _install_qwen3_baseline_adapter_patch()
                # Install the standard Qwen3 FlashVID hook first, then switch
                # runtime compression through flashvid_config. This preserves
                # each adapter algorithm while using the official lmms-eval path.
                flashvid_init_variant = "flashvid"

            flashvid_kwargs = dict(
                model=self._model,
                retention_ratio=retention_ratio,
                expansion=expansion,
                do_segment=do_segment,
                segment_threshold=segment_threshold,
                min_segment_num=min_segment_num,
                complementary_segment=complementary_segment,
                token_selection_method=effective_token_selection_method,
                alpha=alpha,
                temporal_threshold=temporal_threshold,
                dynamic_temporal_threshold=dynamic_temporal_threshold,
                temporal_threshold_quantile=temporal_threshold_quantile,
                temporal_threshold_min=temporal_threshold_min,
                temporal_threshold_max=temporal_threshold_max,
                temporal_match_mode=temporal_match_mode,
                temporal_local_radius=temporal_local_radius,
                temporal_hysteresis=temporal_hysteresis,
                min_keep_per_frame=min_keep_per_frame,
                compression_variant=flashvid_init_variant,
                question_aware_reweighting=question_aware_reweighting,
                question_reweight_beta=question_reweight_beta,
                graph_temporal_topk=graph_temporal_topk if graph_temporal_topk is not None else graph_topk,
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
                certv8_debug=certv8_debug,
                certhr_horizon_gap_seconds=certhr_horizon_gap_seconds,
                certhr_chunk_max_seconds=certhr_chunk_max_seconds,
                certhr_chunk_max_units=certhr_chunk_max_units,
                certhr_semantic_quantile=certhr_semantic_quantile,
                certhr_semantic_floor=certhr_semantic_floor,
                certhr_coverage_floor=certhr_coverage_floor,
                certhr_deficit_threshold=certhr_deficit_threshold,
                certhr_query_peak_quantile=certhr_query_peak_quantile,
                certhr_query_peak_floor=certhr_query_peak_floor,
                certhr_max_swap_ratio=certhr_max_swap_ratio,
                certhr_d_efficiency_floor=certhr_d_efficiency_floor,
                certhr_add_pool=certhr_add_pool,
                certhr_remove_pool=certhr_remove_pool,
                certhr_debug=certhr_debug,
                certlh_min_duration_seconds=certlh_min_duration_seconds,
                certlh_horizon_gap_seconds=certlh_horizon_gap_seconds,
                certlh_gate_threshold=certlh_gate_threshold,
                certlh_min_groups=certlh_min_groups,
                certlh_max_groups=certlh_max_groups,
                certlh_min_group_units=certlh_min_group_units,
                certlh_max_group_units=certlh_max_group_units,
                certlh_event_quantile=certlh_event_quantile,
                certlh_event_floor=certlh_event_floor,
                certlh_group_floor_ratio=certlh_group_floor_ratio,
                certlh_budget_temperature=certlh_budget_temperature,
                certlh_query_weight=certlh_query_weight,
                certlh_relay_ratio=certlh_relay_ratio,
                certlh_query_peaks_per_atom=certlh_query_peaks_per_atom,
                certlh_query_peak_quantile=certlh_query_peak_quantile,
                certlh_query_peak_floor=certlh_query_peak_floor,
                certlh_query_min_group_distance=certlh_query_min_group_distance,
                certlh_cross_group_similarity=certlh_cross_group_similarity,
                certlh_cross_group_max_seconds=certlh_cross_group_max_seconds,
                certlh_debug=certlh_debug,
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
                slot_base_roles=slot_base_roles,
                slot_max_per_segment=slot_max_per_segment,
                slot_role_allocation=slot_role_allocation,
                slot_overlap_radius=slot_overlap_radius,
                slot_tiebreak_eps=slot_tiebreak_eps,
                slot_motion_window=slot_motion_window,
                slot_soft_cap_fraction=slot_soft_cap_fraction,
                slot_anchor_blend=slot_anchor_blend,
                slot_passthrough_ratio=slot_passthrough_ratio,
                slot_passthrough_min=slot_passthrough_min,
                slot_fast_assignment=slot_fast_assignment,
                talon_transport_radius=talon_transport_radius,
                talon_rank_ratio=talon_rank_ratio,
                talon_rank_min=talon_rank_min,
                talon_rank_max=talon_rank_max,
                talon_budget_mode=talon_budget_mode,
                talon_use_question_innovation=talon_use_question_innovation,
                talon_innovation_qweight=talon_innovation_qweight,
                talon_output_mode=talon_output_mode,
                talon_reconstruction_blend=talon_reconstruction_blend,
                talon_anchor_score_weight=talon_anchor_score_weight,
                memory_token_ratio=memory_token_ratio,
                memory_token_min=memory_token_min,
                memory_token_max=memory_token_max,
                adaptive_token_budget=adaptive_token_budget,
                adaptive_budget_low=adaptive_budget_low,
                adaptive_budget_mid=adaptive_budget_mid,
                adaptive_budget_high=adaptive_budget_high,
                pruning_layer=pruning_layer,
                llm_retention_ratio=llm_retention_ratio,
                decode_policy=decode_policy,
                decode_kv_budget_ratio=decode_kv_budget_ratio,
                decode_update_interval=decode_update_interval,
                decode_start_layer=decode_start_layer,
            )
            supported_flashvid_args = set(inspect.signature(flashvid).parameters)
            dropped_flashvid_args = sorted(set(flashvid_kwargs) - supported_flashvid_args)
            if dropped_flashvid_args:
                eval_logger.debug(f"Dropping unsupported FlashVID wrapper args: {dropped_flashvid_args}")
            self._model = flashvid(**{k: v for k, v in flashvid_kwargs.items() if k in supported_flashvid_args})
            if variant in SUPPORTED_QWEN3_BASELINE_ADAPTERS:
                cfg = getattr(self._model, "flashvid_config")
                setattr(cfg, "compression_variant", variant)
                setattr(cfg, "adapter_budget_uses_expansion", bool(adapter_budget_uses_expansion))
                setattr(cfg, "external_budget_uses_expansion", bool(external_budget_uses_expansion))
                setattr(cfg, "fastvid_DySeg_c", int(fastvid_DySeg_c))
                setattr(cfg, "fastvid_DySeg_tau", float(fastvid_DySeg_tau))
                setattr(cfg, "fastvid_DySeg_ignore", float(fastvid_DySeg_ignore))
                setattr(cfg, "fastvid_STPrune_d", float(fastvid_STPrune_d))
                setattr(cfg, "fastvid_DTM_p", int(fastvid_DTM_p))
                setattr(cfg, "fastvid_DTM_beta", float(fastvid_DTM_beta))
                setattr(cfg, "fastgraph_ats_ratio", float(fastgraph_ats_ratio))
                setattr(cfg, "fastgraph_temporal_radius", int(fastgraph_temporal_radius))
                setattr(cfg, "fastgraph_temporal_skip", int(fastgraph_temporal_skip))
                setattr(cfg, "fastgraph_temporal_topk", int(fastgraph_temporal_topk))
                setattr(cfg, "fastgraph_edge_threshold", float(fastgraph_edge_threshold))
                setattr(cfg, "fastgraph_protect_ratio", float(fastgraph_protect_ratio))
                setattr(cfg, "fastgraph_attn_weight", float(fastgraph_attn_weight))
                setattr(cfg, "fastgraph_novelty_weight", float(fastgraph_novelty_weight))
                setattr(cfg, "fastgraph_density_weight", float(fastgraph_density_weight))
                setattr(cfg, "visionzip_dominant_ratio", float(visionzip_dominant_ratio))
            # print(f"[INFO] Enable FlashVID with retention_ratio={retention_ratio}, expansion={expansion}, do_segment={do_segment}, segment_threshold={segment_threshold}, min_segment_num={min_segment_num}, complementary_segment={complementary_segment}, token_selection_method={token_selection_method}, alpha={alpha}, temporal_threshold={temporal_threshold}, pruning_layer={pruning_layer}, llm_retention_ratio={llm_retention_ratio}")

        self._model.eval()
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.total_pixels = total_pixels
        self.max_num_frames = max_num_frames

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals

        self._config = self.model.config
        self._max_length = kwargs.get("max_length", 2048)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _build_video_kwargs(self) -> dict:
        video_kwargs = {
            "max_pixels": self.max_pixels,
            "min_pixels": self.min_pixels,
        }
        if self.total_pixels is not None:
            video_kwargs["total_pixels"] = self.total_pixels
        if self.fps is not None:
            video_kwargs["fps"] = self.fps
            video_kwargs["max_frames"] = int(self.max_num_frames)
        else:
            video_kwargs["nframes"] = int(self.max_num_frames)
        return video_kwargs

    def _build_current_gen_kwargs(self, gen_kwargs: dict) -> dict:
        current_gen_kwargs = {
            "max_new_tokens": 128,
            "temperature": 0.0,
            "top_p": None,
            "num_beams": 1,
            **gen_kwargs,
        }
        if current_gen_kwargs["temperature"] > 0:
            current_gen_kwargs["do_sample"] = True
        else:
            current_gen_kwargs["do_sample"] = False
            current_gen_kwargs["temperature"] = None
            current_gen_kwargs["top_p"] = None
        return current_gen_kwargs

    def _preprocess_chunk(self, chunk):
        contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
        task = task[0]
        split = split[0]
        visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
        gen_kwargs = all_gen_kwargs[0]

        until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
        if isinstance(until, str):
            until = [until]
        elif not isinstance(until, list):
            raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")
        until = [item for item in until if item != "\n\n"]

        if isinstance(contexts, tuple):
            contexts = list(contexts)
        contexts = [context.replace("<image>", "") for context in contexts]

        video_paths_to_release = []
        batched_messages = []
        base_video_kwargs = self._build_video_kwargs()
        for idx, context in enumerate(contexts):
            message = [{"role": "system", "content": self.system_prompt}]
            if self.reasoning_prompt:
                context = context.strip() + self.reasoning_prompt
                contexts[idx] = context

            processed_visuals = []
            if visual_list[idx] is not None:
                for visual in visual_list[idx]:
                    if isinstance(visual, str) and visual.lower().endswith(VIDEO_SUFFIXES):
                        video_paths_to_release.append(visual)
                        processed_visuals.append({"type": "video", "video": visual, **base_video_kwargs})
                    elif isinstance(visual, Image.Image):
                        processed_visuals.append(
                            {
                                "type": "image",
                                "image": visual,
                                "max_pixels": self.max_pixels,
                                "min_pixels": self.min_pixels,
                            }
                        )

            if self.interleave_visuals is False:
                message.append(
                    {
                        "role": "user",
                        "content": processed_visuals + [{"type": "text", "text": context}],
                    }
                )
            else:
                image_placeholders = re.findall(r"<image \d+>", context)
                content_parts = []
                text_parts = re.split(r"<image \d+>", context)
                if text_parts[0]:
                    content_parts.append({"type": "text", "text": text_parts[0]})
                for image_pos, placeholder in enumerate(image_placeholders):
                    image_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                    image_idx = min(image_idx, len(processed_visuals) - 1) if processed_visuals else 0
                    if processed_visuals and image_idx < len(processed_visuals):
                        content_parts.append(processed_visuals[image_idx])
                    if image_pos + 1 < len(text_parts) and text_parts[image_pos + 1]:
                        content_parts.append({"type": "text", "text": text_parts[image_pos + 1]})
                message.append({"role": "user", "content": content_parts})
            batched_messages.append(message)

        texts = self.processor.apply_chat_template(batched_messages, tokenize=False, add_generation_prompt=True)
        while True:
            try:
                image_inputs, video_inputs, processed_video_kwargs = process_vision_info(
                    batched_messages,
                    return_video_kwargs=True,
                    image_patch_size=16,
                    return_video_metadata=True,
                )
                break
            except ValueError as exc:
                fallback_nframes = _parse_qwen_nframes_limit(exc)
                if fallback_nframes is None:
                    raise
                _set_video_frame_limit(batched_messages, fallback_nframes)

        video_metadata = None
        if video_inputs is not None and len(video_inputs) > 0 and isinstance(video_inputs[0], tuple):
            video_inputs, video_metadata = zip(*video_inputs)
            video_inputs = list(video_inputs)
            video_metadata = list(video_metadata)

        processor_kwargs = {
            "text": texts,
            "images": image_inputs,
            "videos": video_inputs,
            "padding": True,
            "return_tensors": "pt",
            "do_resize": False,
            **processed_video_kwargs,
        }
        if self.batch_size > 1:
            processor_kwargs["padding_side"] = "left"
        if video_metadata is not None:
            processor_kwargs["video_metadata"] = video_metadata

        inputs = self.processor(**processor_kwargs)
        if self.device_map == "auto":
            inputs = inputs.to("cuda")
        else:
            inputs = inputs.to(self.device)

        frame_timing = _qwen_frame_timing(video_metadata)
        return inputs, contexts, gen_kwargs, until, video_paths_to_release, frame_timing

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # Keep VideoMME execution order aligned with bench_all_metrics. lmms-eval's
        # default length-based reordering is fine for pure text, but video
        # compression can be sensitive to per-process CUDA/RNG ordering.
        preserve_order = os.environ.get("LMMS_EVAL_PRESERVE_REQUEST_ORDER", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        if preserve_order:
            request_args = [reg.args for reg in requests]
            chunks = [request_args[i : i + self.batch_size] for i in range(0, len(request_args), self.batch_size)]
            re_ords = None
        else:
            # We group requests by their generation kwargs, so that we don't try
            # to execute e.g. greedy sampling and temp=0.8 sampling in one batch.
            re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
            chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        if self.rank == 0 and os.environ.get("LMMS_EVAL_DEBUG_ORDER", "0").lower() in {"1", "true", "yes"}:
            eval_logger.info(
                f"Qwen3_VL generate_until file={__file__} preserve_order={preserve_order} "
                f"requests={len(requests)} chunks={len(chunks)} batch_size={self.batch_size}"
            )
        for chunk in chunks:
            inputs, contexts, gen_kwargs, until, video_paths_to_release, frame_timing = self._preprocess_chunk(chunk)
            _, _, _, doc_ids, tasks, splits = zip(*chunk)
            task = tasks[0]
            split = splits[0]
            sample_doc = self.task_dict[task][split][doc_ids[0]]

            current_gen_kwargs = self._build_current_gen_kwargs(gen_kwargs)
            pad_token_id = self.tokenizer.pad_token_id

            runtime_config = _publish_certhr_timing(self.model, frame_timing)
            _publish_certvid_sample(
                runtime_config,
                sample_doc,
                doc_ids[0],
                contexts[0],
                task,
            )
            try:
                cont = self.model.generate(
                    **inputs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    do_sample=current_gen_kwargs["do_sample"],
                    temperature=current_gen_kwargs["temperature"],
                    top_p=current_gen_kwargs["top_p"],
                    num_beams=current_gen_kwargs["num_beams"],
                    max_new_tokens=current_gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                )
            finally:
                _clear_certvid_sample(runtime_config)
                _clear_certhr_timing(runtime_config)

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            del inputs, cont, generated_ids_trimmed
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            for ans, context in zip(answers, contexts):
                clean_ans = parse_reasoning_model_answer(ans)
                res.append(clean_ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), clean_ans)
                pbar.update(1)

                # eval_logger.debug(f"Question: {context}")
                # eval_logger.debug(f"Model Raw Response: {ans}")
                # eval_logger.debug(f"Model Clean Response: {clean_ans}")
            if video_paths_to_release:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                _release_video_file_cache(video_paths_to_release)
                gc.collect()
            # reorder this group of results back to original unsorted form
        if re_ords is not None:
            res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
