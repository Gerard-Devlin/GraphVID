import copy
import json
import logging
import os
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from packaging import version
from tqdm import tqdm
from transformers import AutoConfig

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
eval_logger = logging.getLogger("lmms-eval")

# Enable TF32 for CUDA
torch.backends.cuda.matmul.allow_tf32 = True

# Import LLaVA modules
try:
    from llava.constants import (
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        IGNORE_INDEX,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import (
        KeywordsStoppingCriteria,
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
    from llava.model.builder import load_pretrained_model
except ImportError as e:
    eval_logger.debug(f"LLaVA is not installed. Please install LLaVA to use this model.\nError: {e}")


# Determine best attention implementation
if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"


def _flashvid_runtime_config(model):
    candidates = [model, getattr(model, "model", None)]
    nested = getattr(getattr(model, "model", None), "language_model", None)
    candidates.append(nested)
    for candidate in candidates:
        config = getattr(candidate, "flashvid_config", None) if candidate is not None else None
        if config is not None:
            return config
    return None


def _publish_frame_timing(model, timing):
    config = _flashvid_runtime_config(model)
    if config is None:
        return None
    config._certvid_frame_times_sec = None
    config._certvid_frame_times_source = "missing"
    if str(getattr(config, "compression_variant", "")).strip().lower() in {"flashvid", "certvid_v7", "certvid_v8", "certvid_v9", "certvid_v10", "certvid_v11"} and timing is not None:
        config._certvid_frame_times_sec, config._certvid_frame_times_source = timing
    return config


def _clear_frame_timing(config) -> None:
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
    if str(getattr(config, "compression_variant", "")).strip().lower() == "certvid_v3plus":
        from flashvid.v3plus_inner import clear_v3plus_runtime

        clear_v3plus_runtime(config)
    if str(getattr(config, "compression_variant", "")).strip().lower() == "certvid_v3plusplus":
        from flashvid.v3plusplus_inner import clear_v3plusplus_runtime

        clear_v3plusplus_runtime(config)
    config._debug_sample_id = "unknown"
    config._certvid_query_text = ""
    config._certvid_eval_category = None
    config._certvid_task_name = None


@register_model("llava_onevision")
class Llava_OneVision(lmms):
    """
    Llava Model
    """

    def __init__(
        self,
        pretrained: str = "lmms-lab/llava-onevision-qwen2-7b-ov",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name: Optional[str] = None,
        attn_implementation: Optional[str] = best_fit_attn_implementation,
        device_map: Optional[str] = "cuda:0",
        conv_template: Optional[str] = "qwen_1_5",
        use_cache: Optional[bool] = True,
        truncate_context: Optional[bool] = False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        customized_config: Optional[str] = None,  # ends in json
        max_frames_num: Optional[int] = 32,
        mm_spatial_pool_stride: Optional[int] = 2,
        mm_spatial_pool_mode: Optional[str] = "bilinear",
        token_strategy: Optional[str] = "single",  # could be "single" or "multiple", "multiple" denotes adding multiple <image> tokens for each frame
        video_decode_backend: str = "decord",
        # ! FlashVid parameters.
        enable_flashvid: bool = False,
        retention_ratio: float = 0.25,
        # DySeg parameters (FIXED)
        do_segment: bool = True,
        segment_threshold: float = 0.9,
        min_segment_num: int = 8,
        complementary_segment: bool = True,
        # ADTS and TSTM parameters
        token_selection_method: str = "attn_div_v2",
        alpha: float = 0.7,
        temporal_threshold: float = 0.8,
        compression_variant: str = "flashvid",
        adapter_budget_uses_expansion: bool = False,
        fastvid_DySeg_c: int = 8,
        fastvid_DySeg_tau: float = 0.90,
        fastvid_STPrune_d: float = 0.40,
        fastvid_DTM_p: int = 4,
        fastvid_DTM_beta: float = 0.60,
        visionzip_dominant_ratio: float = 65.0 / 70.0,
        prunevid_tau: float = 0.80,
        prunevid_temporal_segment_ratio: float = 0.25,
        prunevid_cluster_ratio: float = 0.50,
        # CertVID parameters
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
        # CertVID V2 parameters
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
        # CertVID V3 parameters
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
        # CertVID V3Plus inner-selector parameters
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
        # CertVID V6 parameters
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
        # CertVID V9 parameters
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
        # CertVID V10 parameters
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
        # CertVID V11 parameters
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
        # CertVID V4 parameters
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
        # CertVID V5 parameters
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
        # CertVID-E parameters
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
        # Inner-LLM Pruning parameters (FIXED)
        expansion: float = 1.25,
        pruning_layer: int = 20,
        llm_retention_ratio: float = 0.3,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        llava_model_args = {
            "multimodal": True,
        }
        if customized_config is not None:
            llava_model_args["customized_config"] = customized_config
        if attn_implementation is not None:
            llava_model_args["attn_implementation"] = attn_implementation
        if "use_flash_attention_2" in kwargs:
            llava_model_args["use_flash_attention_2"] = kwargs["use_flash_attention_2"]
        model_name = model_name if model_name is not None else get_model_name_from_path(pretrained)

        self.pretrained = pretrained
        self.token_strategy = token_strategy
        self.max_frames_num = max_frames_num
        self.mm_spatial_pool_stride = mm_spatial_pool_stride
        self.mm_spatial_pool_mode = mm_spatial_pool_mode
        self.video_decode_backend = video_decode_backend

        overwrite_config = {}
        overwrite_config["mm_spatial_pool_stride"] = self.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_mode"] = self.mm_spatial_pool_mode
        cfg_pretrained = AutoConfig.from_pretrained(self.pretrained)

        llava_model_args["overwrite_config"] = overwrite_config
        try:
            # Try to load the model with the multimodal argument
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
                pretrained,
                None,
                model_name,
                device_map=self.device_map,
                **llava_model_args,
            )
        except TypeError:
            # for older versions of LLaVA that don't have multimodal argument
            llava_model_args.pop("multimodal", None)
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
                pretrained,
                None,
                model_name,
                device_map=self.device_map,
                **llava_model_args,
            )

        # ! Enable FlashVID
        if enable_flashvid:
            from flashvid import flashvid

            self._model = flashvid(
                model=self._model,
                retention_ratio=retention_ratio,
                do_segment=do_segment,
                segment_threshold=segment_threshold,
                min_segment_num=min_segment_num,
                complementary_segment=complementary_segment,
                alpha=alpha,
                token_selection_method=token_selection_method,
                temporal_threshold=temporal_threshold,
                compression_variant=compression_variant,
                adapter_budget_uses_expansion=adapter_budget_uses_expansion,
                fastvid_DySeg_c=fastvid_DySeg_c,
                fastvid_DySeg_tau=fastvid_DySeg_tau,
                fastvid_STPrune_d=fastvid_STPrune_d,
                fastvid_DTM_p=fastvid_DTM_p,
                fastvid_DTM_beta=fastvid_DTM_beta,
                visionzip_dominant_ratio=visionzip_dominant_ratio,
                prunevid_tau=prunevid_tau,
                prunevid_temporal_segment_ratio=prunevid_temporal_segment_ratio,
                prunevid_cluster_ratio=prunevid_cluster_ratio,
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
                expansion=expansion,
                pruning_layer=pruning_layer,
                llm_retention_ratio=llm_retention_ratio,
            )
            # print(f"[INFO] Enable FlashVID with retention_ratio={retention_ratio}, expansion={expansion}, do_segment={do_segment}, segment_threshold={segment_threshold}, min_segment_num={min_segment_num}, complementary_segment={complementary_segment}, token_selection_method={token_selection_method}, alpha={alpha}, temporal_threshold={temporal_threshold}, pruning_layer={pruning_layer}, llm_retention_ratio={llm_retention_ratio}")

        self._config = self._model.config
        self.model.eval()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._world_size = 1

        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
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
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

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

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        origin_image_aspect_ratio = getattr(self._config, "image_aspect_ratio", None)

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            visual = doc_to_visual(self.task_dict[task][split][doc_id])

            if origin_image_aspect_ratio is not None and self._config.image_aspect_ratio != origin_image_aspect_ratio:
                self._config.image_aspect_ratio = origin_image_aspect_ratio
                eval_logger.info(f"Resetting image aspect ratio to {origin_image_aspect_ratio}")

            if visual is None or visual == []:
                visual = None
                task_type = "text"
                image_tensor = None
            else:
                if len(visual) > 1 or "image_aspect_ratio" not in self._config.__dict__:
                    self._config.image_aspect_ratio = "pad"
                    eval_logger.info(f"In Multi-Image setting, image aspect ratio: {self._config.image_aspect_ratio}")

                if "task_type" in self.metadata and self.metadata["task_type"] == "video" and "sample_frames" in self.metadata:
                    assert type(visual) == list, "sample_frames must be specified for video task"
                    sample_indices = np.linspace(0, len(visual) - 1, self.metadata["sample_frames"], dtype=int)
                    visual = [visual[i] for i in sample_indices]
                    assert len(visual) == self.metadata["sample_frames"]

                    image_tensor = process_images(visual, self._image_processor, self._config)
                    if type(image_tensor) is list:
                        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                    else:
                        image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

                    task_type = "video"

                # elif type(visual[0]) == PIL.Image.Image:
                elif isinstance(visual[0], PIL.Image.Image):
                    image_tensor = process_images(visual, self._image_processor, self._config)
                    if type(image_tensor) is list:
                        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                    else:
                        image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

                    task_type = "image"

                elif type(visual[0]) == str:
                    image_tensor = []
                    try:
                        if self.video_decode_backend == "decord":
                            frames = self.load_video(visual, self.max_frames_num)
                        elif self.video_decode_backend == "pyav":
                            frames = read_video_pyav(visual[0], num_frm=self.max_frames_num)
                        frames = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().to(self._device)
                        image_tensor.append(frames)
                    except Exception as e:
                        eval_logger.error(f"Error {e} in loading video")
                        image_tensor = None

                    task_type = "video"

            if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in contexts:
                placeholder_count = len(visual) if isinstance(visual, list) else 1
                if task_type == "video":
                    placeholder_count = len(frames) if self.token_strategy == "multiple" else 1
                image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
                image_tokens = " ".join(image_tokens)
                prompts_input = image_tokens + "\n" + contexts
            else:
                prompts_input = contexts

            if "llama_3" in self.conv_template:
                conv = copy.deepcopy(conv_templates[self.conv_template])
            else:
                conv = conv_templates[self.conv_template].copy()

            conv.append_message(conv.roles[0], prompts_input)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])

            conv.messages[-1][1] = continuation
            full_prompt = conv.get_prompt()
            full_input_ids = tokenizer_image_token(full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

            labels = full_input_ids.clone()
            labels[0, : input_ids.shape[1]] = -100

            kwargs = {}
            if task_type == "image":
                kwargs["image_sizes"] = [[v.size[0], v.size[1]] for v in visual] if isinstance(visual, list) else [[visual.size[0], visual.size[1]]]
            elif task_type == "video":
                kwargs["modalities"] = ["video"]
                self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
                self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode

            with torch.inference_mode():
                outputs = self.model(
                    input_ids=full_input_ids,
                    labels=labels,
                    images=image_tensor,
                    use_cache=True,
                    **kwargs,
                )

            loss = outputs["loss"]
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = full_input_ids[:, input_ids.shape[1] :]
            greedy_tokens = greedy_tokens[:, input_ids.shape[1] : full_input_ids.shape[1]]
            max_equal = (greedy_tokens == cont_toks).all()

            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)

        pbar.close()
        return res

    def flatten(self, input):
        if not input or any(i is None for i in input):
            return []
        new_list = []
        for i in input:
            if i:
                for j in i:
                    new_list.append(j)
        return new_list

    def load_video(self, video_path, max_frames_num):
        self._pending_frame_timing = None
        if type(video_path) == str:
            vr = VideoReader(video_path, ctx=cpu(0))
        else:
            vr = VideoReader(video_path[0], ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        try:
            fps = float(vr.get_avg_fps())
            if np.isfinite(fps) and fps > 0.0:
                self._pending_frame_timing = (
                    (uniform_sampled_frames.astype(np.float64) / fps).tolist(),
                    "llava_decord_indices_fps",
                )
        except (AttributeError, TypeError, ValueError, RuntimeError):
            self._pending_frame_timing = None
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames  # (frames, height, width, channels)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        metadata = requests[0].metadata
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        origin_image_aspect_ratio = getattr(self._config, "image_aspect_ratio", None)

        for chunk in chunks:
            (
                batched_contexts,
                all_gen_kwargs,
                batched_doc_to_visual,
                batched_doc_id,
                batched_task,
                batched_split,
            ) = zip(*chunk)
            task = batched_task[0]
            split = batched_split[0]
            sample_doc = self.task_dict[task][split][batched_doc_id[0]]
            sample_identifier = sample_doc.get("id", batched_doc_id[0])
            if os.environ.get("FLASHVID_DEBUG_SAMPLE_ID", "0") == "1":
                print(
                    "[flashvid-sample] "
                    f"id={sample_identifier} doc_id={batched_doc_id[0]} "
                    f"task={task}"
                )
            batched_visuals = [batched_doc_to_visual[0](self.task_dict[task][split][ids]) for ids in batched_doc_id]  # [B, N]
            assert len(batched_visuals) == 1

            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            question_input = []
            # import ipdb; ipdb.set_trace()
            for visual, context in zip(batched_visuals, batched_contexts):
                self._pending_frame_timing = None
                if origin_image_aspect_ratio is not None and self._config.image_aspect_ratio != origin_image_aspect_ratio:
                    self._config.image_aspect_ratio = origin_image_aspect_ratio
                    eval_logger.info(f"Resetting image aspect ratio to {origin_image_aspect_ratio}")

                if visual is None or visual == []:  # for text-only tasks.
                    visual = None
                    task_type = "text"
                    placeholder_count = 0
                    image_tensor = None
                else:
                    if len(visual) > 1 or "image_aspect_ratio" not in self._config.__dict__:  # for multi image case, we treat per image aspect ratio as "pad" by default.
                        self._config.image_aspect_ratio = getattr(gen_kwargs, "image_aspect_ratio", "pad")
                        eval_logger.info(f"In Multi-Image setting, image aspect ratio: {self._config.image_aspect_ratio}")

                    if "task_type" in metadata and metadata["task_type"] == "video" and "sample_frames" in metadata:  # overwrite logic for video task with multiple static image frames
                        assert type(visual) == list, "sample_frames must be specified for video task"
                        sample_indices = np.linspace(0, len(visual) - 1, metadata["sample_frames"], dtype=int)
                        visual = [visual[i] for i in sample_indices]
                        assert len(visual) == metadata["sample_frames"]

                        image_tensor = process_images(visual, self._image_processor, self._config)
                        if type(image_tensor) is list:
                            image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                        else:
                            image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

                        task_type = "video"
                        placeholder_count = 1

                    elif type(visual[0]) == PIL.Image.Image:  # For image, multi-image tasks
                        image_tensor = process_images(visual, self._image_processor, self._config)
                        if type(image_tensor) is list:
                            image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                        else:
                            image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

                        task_type = "image"
                        placeholder_count = len(visual) if isinstance(visual, list) else 1

                    elif type(visual[0]) == str:  # For video task
                        image_tensor = []
                        try:
                            if self.video_decode_backend == "decord":
                                frames = self.load_video(visual, self.max_frames_num)
                            elif self.video_decode_backend == "pyav":
                                frames = read_video_pyav(visual[0], num_frm=self.max_frames_num)
                            frames = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().to(self._device)
                            image_tensor.append(frames)
                        except Exception as e:
                            eval_logger.error(f"Error {e} in loading video")
                            image_tensor = None

                        task_type = "video"
                        placeholder_count = len(frames) if self.token_strategy == "multiple" else 1

                if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                    """
                    Three senarios:
                    1. No image, and there for, no image token should be added.
                    2. image token is already specified in the context, so we don't need to add it.
                    3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                    4. For video tasks, we could add a <image> token or multiple <image> tokens for each frame in the context. This depends on the training strategy and should balance in test to decide which is better
                    """
                    # if task_type == "image": # indeed in multi-image case, not the video in frames.
                    #     image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count if isinstance(visual, list) else [DEFAULT_IMAGE_TOKEN]
                    # elif task_type == "video":
                    # image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count if self.token_strategy == "multiple" else [DEFAULT_IMAGE_TOKEN]
                    image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
                    image_tokens = " ".join(image_tokens)
                    question = image_tokens + "\n" + context
                else:
                    question = context

                # This is much safer for llama3, as we now have some object type in it
                if "llama_3" in self.conv_template:
                    conv = copy.deepcopy(conv_templates[self.conv_template])
                else:
                    conv = conv_templates[self.conv_template].copy()

                if utils.is_json(question):  # conversational question input
                    question = json.loads(question)
                    for idx, item in enumerate(question):
                        role = conv.roles[idx % 2]
                        message = item["value"]
                        conv.append_message(role, message)

                    assert len(conv.messages) % 2 == 1
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    question_input.append(prompt_question)
                else:  # only simple string for question
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    question_input.append(prompt_question)

            input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in question_input]
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
            attention_masks = input_ids.ne(pad_token_ids).to(self.device)

            if task_type == "image":
                gen_kwargs["image_sizes"] = [batched_visuals[0][idx].size for idx in range(len(batched_visuals[0]))]
            elif task_type == "video":
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
                gen_kwargs["modalities"] = ["video"]
                gen_kwargs["stopping_criteria"] = [stopping_criteria]
                self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
                self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode

            # These steps are not in LLaVA's original code, but are necessary for generation to work
            # TODO: attention to this major generation step...
            # preconfigure gen_kwargs with defaults
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024

            if "image_aspect_ratio" in gen_kwargs.keys():
                gen_kwargs.pop("image_aspect_ratio")
            # When do_sample=False, remove sampling-related parameters to avoid warnings
            # These might be in gen_kwargs or in the model's generation_config
            if not gen_kwargs.get("do_sample", False):
                gen_kwargs.pop("temperature", None)
                gen_kwargs.pop("top_p", None)
                gen_kwargs.pop("top_k", None)
            runtime_config = _publish_frame_timing(
                self.model,
                getattr(self, "_pending_frame_timing", None),
            )
            _publish_certvid_sample(
                runtime_config,
                sample_doc,
                batched_doc_id[0],
                batched_contexts[0],
                task,
            )
            try:
                with torch.inference_mode():
                    cont = self.model.generate(
                        input_ids,
                        attention_mask=attention_masks,
                        pad_token_id=pad_token_ids,
                        images=image_tensor,
                        use_cache=self.use_cache,
                        **gen_kwargs,
                    )
                    # cont = self.model.generate(qwen_input_ids, pad_token_id=pad_token_ids, images=image_tensor, use_cache=self.use_cache, **gen_kwargs)

                text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            except Exception as e:
                raise e
            finally:
                _clear_certvid_sample(runtime_config)
                _clear_frame_timing(runtime_config)
                self._pending_frame_timing = None

            text_outputs = [response.strip() for response in text_outputs]
            res.extend(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        metadata = requests[0].metadata
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        origin_image_aspect_ratio = getattr(self._config, "image_aspect_ratio", None)

        for chunk in chunks:
            (
                batched_contexts,
                all_gen_kwargs,
                batched_doc_to_visual,
                batched_doc_to_text,
                batched_doc_id,
                batched_task,
                batched_split,
            ) = zip(*chunk)
            task = batched_task[0]
            split = batched_split[0]
            sample_doc = self.task_dict[task][split][batched_doc_id[0]]
            batched_visuals = [batched_doc_to_visual[0](self.task_dict[task][split][ids]) for ids in batched_doc_id]  # [B, N]
            assert len(batched_visuals) == 1

            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            # multi round inference: terminate when receiving signal from the doc_to_text
            round_idx = 0
            batched_round_res = []
            batched_previous_round_info = None
            while True:
                question_input = []

                if round_idx != 0:  # get current round visual and context from doc_to_text function
                    (
                        batched_visuals,
                        batched_contexts,
                        batched_terminal_singal,
                        batched_round_res,
                        batched_previous_round_info,
                    ) = list(
                        zip(
                            *[
                                batched_doc_to_text[0](
                                    self.task_dict[task][split][ids],
                                    previous_output=[round_res[ids_idx] for round_res in batched_round_res],
                                    round_idx=round_idx,
                                    previous_round_info=batched_previous_round_info[ids_idx] if batched_previous_round_info is not None else None,
                                )
                                for ids_idx, ids in enumerate(batched_doc_id)
                            ]
                        )
                    )
                    # import ipdb; ipdb.set_trace()
                    batched_round_res = list(zip(*batched_round_res))  # [(r1_1, r1_2), (r2_1, r2_2), ...]
                    if batched_terminal_singal[0]:  # terminal signal from doc_to_text function
                        break

                for visual, context in zip(batched_visuals, batched_contexts):
                    self._pending_frame_timing = None
                    if origin_image_aspect_ratio is not None and self._config.image_aspect_ratio != origin_image_aspect_ratio:
                        self._config.image_aspect_ratio = origin_image_aspect_ratio
                        eval_logger.info(f"Resetting image aspect ratio to {origin_image_aspect_ratio}")

                    if visual is None or visual == []:  # for text-only tasks.
                        visual = None
                        task_type = "text"
                        placeholder_count = 0
                        image_tensor = None
                    else:
                        if len(visual) > 1 or "image_aspect_ratio" not in self._config.__dict__:  # for multi image case, we treat per image aspect ratio as "pad" by default.
                            self._config.image_aspect_ratio = getattr(gen_kwargs, "image_aspect_ratio", "pad")
                            eval_logger.info(f"In Multi-Image setting, image aspect ratio: {self._config.image_aspect_ratio}")

                        if "task_type" in metadata and metadata["task_type"] == "video" and "sample_frames" in metadata:  # overwrite logic for video task with multiple static image frames
                            assert type(visual) == list, "sample_frames must be specified for video task"
                            sample_indices = np.linspace(0, len(visual) - 1, metadata["sample_frames"], dtype=int)
                            visual = [visual[i] for i in sample_indices]
                            assert len(visual) == metadata["sample_frames"]

                            image_tensor = process_images(visual, self._image_processor, self._config)
                            if type(image_tensor) is list:
                                image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                            else:
                                image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

                            task_type = "video"
                            placeholder_count = 1

                        elif type(visual[0]) == PIL.Image.Image:  # For image, multi-image tasks
                            image_tensor = process_images(visual, self._image_processor, self._config)
                            if type(image_tensor) is list:
                                image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                            else:
                                image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

                            task_type = "image"
                            placeholder_count = len(visual) if isinstance(visual, list) else 1

                        elif type(visual[0]) == str:  # For video task
                            image_tensor = []
                            try:
                                if self.video_decode_backend == "decord":
                                    frames = self.load_video(visual, self.max_frames_num)
                                elif self.video_decode_backend == "pyav":
                                    frames = read_video_pyav(visual[0], num_frm=self.max_frames_num)
                                frames = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().to(self._device)
                                image_tensor.append(frames)
                            except Exception as e:
                                eval_logger.error(f"Error {e} in loading video")
                                image_tensor = None

                            task_type = "video"
                            placeholder_count = len(frames) if self.token_strategy == "multiple" else 1

                    if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                        """
                        Three senarios:
                        1. No image, and there for, no image token should be added.
                        2. image token is already specified in the context, so we don't need to add it.
                        3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                        4. For video tasks, we could add a <image> token or multiple <image> tokens for each frame in the context. This depends on the training strategy and should balance in test to decide which is better
                        """
                        # if task_type == "image": # indeed in multi-image case, not the video in frames.
                        #     image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count if isinstance(visual, list) else [DEFAULT_IMAGE_TOKEN]
                        # elif task_type == "video":
                        # image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count if self.token_strategy == "multiple" else [DEFAULT_IMAGE_TOKEN]
                        image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
                        image_tokens = " ".join(image_tokens)
                        question = image_tokens + "\n" + context
                    else:
                        question = context

                    # This is much safer for llama3, as we now have some object type in it
                    if "llama_3" in self.conv_template:
                        conv = copy.deepcopy(conv_templates[self.conv_template])
                    else:
                        conv = conv_templates[self.conv_template].copy()

                    if utils.is_json(question):  # conversational question input
                        question = json.loads(question)
                        for idx, item in enumerate(question):
                            role = conv.roles[idx % 2]
                            message = item["value"]
                            conv.append_message(role, message)

                        assert len(conv.messages) % 2 == 1
                        conv.append_message(conv.roles[1], None)
                        prompt_question = conv.get_prompt()
                        question_input.append(prompt_question)
                    else:  # only simple string for question
                        conv.append_message(conv.roles[0], question)
                        conv.append_message(conv.roles[1], None)
                        prompt_question = conv.get_prompt()
                        question_input.append(prompt_question)

                # preconfigure gen_kwargs with defaults
                if "max_new_tokens" not in gen_kwargs:
                    gen_kwargs["max_new_tokens"] = 1024
                if "do_sample" not in gen_kwargs:
                    gen_kwargs["do_sample"] = False
                # Only set temperature and top_p if do_sample is True
                if gen_kwargs.get("do_sample", False):
                    if "temperature" not in gen_kwargs:
                        gen_kwargs["temperature"] = 1.0  # Default temperature for sampling
                    if "top_p" not in gen_kwargs:
                        gen_kwargs["top_p"] = 1.0  # Default top_p for sampling
                if "num_beams" not in gen_kwargs:
                    gen_kwargs["num_beams"] = 1

                input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in question_input]
                pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
                attention_masks = input_ids.ne(pad_token_ids).to(self.device)

                if task_type == "image":
                    gen_kwargs["image_sizes"] = [batched_visuals[0][idx].size for idx in range(len(batched_visuals[0]))]
                elif task_type == "video":
                    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    keywords = [stop_str]
                    stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
                    gen_kwargs["modalities"] = ["video"]
                    gen_kwargs["stopping_criteria"] = [stopping_criteria]
                    self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
                    self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode

                # These steps are not in LLaVA's original code, but are necessary for generation to work
                # TODO: attention to this major generation step...
                if "image_aspect_ratio" in gen_kwargs.keys():
                    gen_kwargs.pop("image_aspect_ratio")
                # Remove temperature and top_p when do_sample=False to avoid warnings
                if not gen_kwargs.get("do_sample", False):
                    gen_kwargs.pop("temperature", None)
                    gen_kwargs.pop("top_p", None)
                runtime_config = _publish_frame_timing(
                    self.model,
                    getattr(self, "_pending_frame_timing", None),
                )
                _publish_certvid_sample(
                    runtime_config,
                    sample_doc,
                    batched_doc_id[0],
                    batched_contexts[0],
                    task,
                )
                try:
                    with torch.inference_mode():
                        cont = self.model.generate(
                            input_ids,
                            attention_mask=attention_masks,
                            pad_token_id=pad_token_ids,
                            images=image_tensor,
                            use_cache=self.use_cache,
                            **gen_kwargs,
                        )
                        # cont = self.model.generate(qwen_input_ids, pad_token_id=pad_token_ids, images=image_tensor, use_cache=self.use_cache, **gen_kwargs)

                    text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
                except Exception as e:
                    raise e
                finally:
                    _clear_certvid_sample(runtime_config)
                    _clear_frame_timing(runtime_config)
                    self._pending_frame_timing = None

                text_outputs = [response.strip() for response in text_outputs]
                batched_round_res.append(text_outputs)

                round_idx += 1

            res.extend(list(zip(*batched_round_res)))
            self.cache_hook.add_partial("generate_until_multi_round", (context, gen_kwargs), batched_round_res)
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res
