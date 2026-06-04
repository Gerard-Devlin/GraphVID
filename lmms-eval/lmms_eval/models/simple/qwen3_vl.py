import gc
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
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
    ):
        variant = str(getattr(flashvid_config, "compression_variant", "")).strip().lower()
        if variant in SUPPORTED_QWEN3_BASELINE_ADAPTERS:
            return adapter_baseline_compression(video_features, cls_attention, flashvid_config)
        return original(
            video_features=video_features,
            cls_attention=cls_attention,
            flashvid_config=flashvid_config,
            question_features=question_features,
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

            self._model = flashvid(
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
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]
            video_paths_to_release = []

            # Set default until or update values from gen_kwargs if present
            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])

            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")

            # Avoid using '\n\n' as a stopper for Qwen2.5VL to prevent truncation, which can lead to incorrect results
            until = [item for item in until if item != "\n\n"]

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            batched_messages = []
            for i, context in enumerate(contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")

                message = [{"role": "system", "content": self.system_prompt}]
                if self.reasoning_prompt:
                    context = context.strip() + self.reasoning_prompt
                    contexts[i] = context

                processed_visuals = []
                if visual_list[i] is not None:
                    for visual in visual_list[i]:
                        if isinstance(visual, str) and visual.lower().endswith(VIDEO_SUFFIXES):  # Video file
                            video_paths_to_release.append(visual)
                            total_frames = None
                            vr = None
                            try:
                                vr = decord.VideoReader(visual)
                                total_frames = len(vr)
                            finally:
                                if vr is not None:
                                    del vr
                            video_item = {
                                "type": "video",
                                "video": visual,
                                "max_pixels": self.max_pixels,
                                "min_pixels": self.min_pixels,
                            }
                            if self.fps is not None:
                                video_item["fps"] = self.fps
                                video_item["max_frames"] = _safe_nframes_for_video(total_frames, self.max_num_frames)
                            else:
                                video_item["nframes"] = _safe_nframes_for_video(total_frames, self.max_num_frames)
                            processed_visuals.append(video_item)
                        elif isinstance(visual, Image.Image):  # Handle both single and multiple images
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
                else:  # currently support find <image x> in the context
                    image_placeholders = re.findall(r"<image \d+>", context)
                    content_parts = []
                    text_parts = re.split(r"<image \d+>", context)
                    if text_parts[0]:
                        content_parts.append({"type": "text", "text": text_parts[0]})

                    for i, placeholder in enumerate(image_placeholders):
                        img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                        image_idx = min(img_idx, len(processed_visuals) - 1) if processed_visuals else 0
                        if processed_visuals and image_idx < len(processed_visuals):
                            content_parts.append(processed_visuals[image_idx])
                        if i + 1 < len(text_parts) and text_parts[i + 1]:
                            content_parts.append({"type": "text", "text": text_parts[i + 1]})

                    message.append(
                        {
                            "role": "user",
                            "content": content_parts,
                        }
                    )

                batched_messages.append(message)
            texts = self.processor.apply_chat_template(batched_messages, tokenize=False, add_generation_prompt=True)
            # TODO: refactor code to allow return_video_kwargs and return_video_metadata
            image_inputs, video_inputs = process_vision_info(
                batched_messages,
                return_video_kwargs=False,
                image_patch_size=16,
                return_video_metadata=False,
            )
            if video_inputs is not None:
                total_frames = video_inputs[0].shape[0]
                if total_frames > self.max_num_frames:
                    indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                    # Ensure unique indices if linspace produces duplicates for few frames
                    indices = np.unique(indices)
                    # Append the last frame index if not already included
                    if total_frames - 1 not in indices:
                        indices = np.append(indices, total_frames - 1)
                        indices = np.unique(indices)  # Ensure uniqueness again
                    video_inputs[0] = video_inputs[0][indices]
            if self.batch_size > 1:
                inputs = self.processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    do_resize=False,
                    padding=True,
                    padding_side="left",
                    return_tensors="pt",
                )
            else:
                inputs = self.processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    do_resize=False,
                    return_tensors="pt",
                )
            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            # Set default generation kwargs
            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,  # Set to 0 for greedy default
                "top_p": None,
                "num_beams": 1,
            }
            # Update with provided kwargs
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None

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

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
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
                _release_video_file_cache(video_paths_to_release)
                gc.collect()
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
