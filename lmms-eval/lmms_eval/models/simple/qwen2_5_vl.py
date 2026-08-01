import base64
import re
from io import BytesIO
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
    Qwen2_5_VLForConditionalGeneration,
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


@register_model("qwen2_5_vl")
class Qwen2_5_VL(lmms):
    """
    Qwen2.5_VL Model
    "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct"
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
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
        # Baseline adapter parameters
        compression_variant: str = "flashvid",
        adapter_budget_uses_expansion: bool = False,
        fastvid_DySeg_c: int = 8,
        fastvid_DySeg_tau: float = 0.84,
        fastvid_DySeg_ignore: float = 0.95,
        fastvid_STPrune_d: float = 0.40,
        fastvid_DTM_p: int = 4,
        fastvid_DTM_beta: float = 0.60,
        visionzip_dominant_ratio: float = 65.0 / 70.0,
        prunevid_tau: float = 0.80,
        prunevid_temporal_segment_ratio: float = 0.25,
        prunevid_cluster_ratio: float = 0.50,
        question_aware_reweighting: bool = False,
        question_reweight_beta: float = 0.35,
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
        # CertVID V3 selector parameters used by FaithVID.
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
        # FaithVID functional-faithfulness parameters.
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
            "torch_dtype": "bfloat16",
            "device_map": self.device_map,
        }

        # Add attention implementation if specified
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained, **model_kwargs)
        
        
        # ! Enable FlashVID
        if enable_flashvid:
            from flashvid import flashvid

            self._model = flashvid(
                model=self._model,
                retention_ratio=retention_ratio,
                expansion=expansion,
                do_segment=do_segment,
                segment_threshold=segment_threshold,
                min_segment_num=min_segment_num,
                complementary_segment=complementary_segment,
                token_selection_method=token_selection_method,
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
                compression_variant=compression_variant,
                adapter_budget_uses_expansion=adapter_budget_uses_expansion,
                fastvid_DySeg_c=fastvid_DySeg_c,
                fastvid_DySeg_tau=fastvid_DySeg_tau,
                fastvid_DySeg_ignore=fastvid_DySeg_ignore,
                fastvid_STPrune_d=fastvid_STPrune_d,
                fastvid_DTM_p=fastvid_DTM_p,
                fastvid_DTM_beta=fastvid_DTM_beta,
                visionzip_dominant_ratio=visionzip_dominant_ratio,
                prunevid_tau=prunevid_tau,
                prunevid_temporal_segment_ratio=prunevid_temporal_segment_ratio,
                prunevid_cluster_ratio=prunevid_cluster_ratio,
                question_aware_reweighting=question_aware_reweighting,
                question_reweight_beta=question_reweight_beta,
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
                pruning_layer=pruning_layer,
                llm_retention_ratio=llm_retention_ratio,
                decode_policy=decode_policy,
                decode_kv_budget_ratio=decode_kv_budget_ratio,
                decode_update_interval=decode_update_interval,
                decode_start_layer=decode_start_layer,
            )
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
                        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                            vr = decord.VideoReader(visual)
                            first_frame = vr[0].asnumpy()
                            height, width = first_frame.shape[:2]
                            # max_pixels = height * width
                            processed_visuals.append(
                                {
                                    "type": "video",
                                    "video": visual,
                                    "max_pixels": self.max_pixels,
                                    "min_pixels": self.min_pixels,
                                }
                            )
                        elif isinstance(visual, Image.Image):  # Handle both single and multiple images
                            base64_image = visual.convert("RGB")
                            buffer = BytesIO()
                            base64_image.save(buffer, format="JPEG")
                            base64_bytes = base64.b64encode(buffer.getvalue())
                            base64_string = base64_bytes.decode("utf-8")
                            processed_visuals.append(
                                {
                                    "type": "image",
                                    "image": f"data:image/jpeg;base64,{base64_string}",
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
            image_inputs, video_inputs = process_vision_info(batched_messages)
            if video_inputs is not None:
                total_frames = video_inputs[0].shape[0]
                indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                # Ensure unique indices if linspace produces duplicates for few frames
                indices = np.unique(indices)
                # Append the last frame index if not already included
                if total_frames - 1 not in indices:
                    indices = np.append(indices, total_frames - 1)
                    indices = np.unique(indices)  # Ensure uniqueness again
                video_inputs[0] = video_inputs[0][indices]
            padding_side = "left" if self.batch_size > 1 else "right"
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                padding_side=padding_side,
                return_tensors="pt",
            )
            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            # Set default generation kwargs
            default_gen_kwargs = {
                "max_new_tokens": 32768,
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

            flashvid_config = getattr(self.model, "flashvid_config", None)
            if flashvid_config is not None:
                if len(doc_id) == 1:
                    flashvid_config._debug_sample_id = str(doc_id[0])
                    flashvid_config._certvid_task_name = str(task)
                    flashvid_config._certvid_query_text = str(contexts[0])
                else:
                    flashvid_config._debug_sample_id = "batched"
                    flashvid_config._certvid_task_name = str(task)
                    flashvid_config._certvid_query_text = ""
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
                if flashvid_config is not None:
                    flashvid_config._debug_sample_id = "unknown"
                    flashvid_config._certvid_task_name = None
                    flashvid_config._certvid_query_text = ""

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
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        metadata = requests[0].metadata
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

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
            batched_visuals = [batched_doc_to_visual[0](self.task_dict[task][split][ids]) for ids in batched_doc_id]
            assert len(batched_visuals) == 1

            gen_kwargs = all_gen_kwargs[0]
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            round_idx = 0
            batched_round_res = []
            batched_previous_round_info = None
            while True:
                contexts = []
                visuals_list = []

                if round_idx != 0:
                    (
                        visuals_list,
                        contexts,
                        batched_terminal_signal,
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
                    batched_round_res = list(zip(*batched_round_res))
                    if batched_terminal_signal[0]:
                        break
                else:
                    visuals_list = batched_visuals
                    contexts = list(batched_contexts)

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

                    processed_visuals = []
                    if visuals_list[i] is not None:
                        for visual in visuals_list[i]:
                            if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):
                                vr = decord.VideoReader(visual)
                                first_frame = vr[0].asnumpy()
                                height, width = first_frame.shape[:2]
                                processed_visuals.append(
                                    {
                                        "type": "video",
                                        "video": visual,
                                        "max_pixels": self.max_pixels,
                                        "min_pixels": self.min_pixels,
                                    }
                                )
                            elif isinstance(visual, Image.Image):
                                base64_image = visual.convert("RGB")
                                buffer = BytesIO()
                                base64_image.save(buffer, format="JPEG")
                                base64_bytes = base64.b64encode(buffer.getvalue())
                                base64_string = base64_bytes.decode("utf-8")
                                processed_visuals.append(
                                    {
                                        "type": "image",
                                        "image": f"data:image/jpeg;base64,{base64_string}",
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

                        for j, placeholder in enumerate(image_placeholders):
                            img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                            image_idx = min(img_idx, len(processed_visuals) - 1) if processed_visuals else 0
                            if processed_visuals and image_idx < len(processed_visuals):
                                content_parts.append(processed_visuals[image_idx])
                            if j + 1 < len(text_parts) and text_parts[j + 1]:
                                content_parts.append({"type": "text", "text": text_parts[j + 1]})

                        message.append(
                            {
                                "role": "user",
                                "content": content_parts,
                            }
                        )

                    batched_messages.append(message)

                texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batched_messages]
                image_inputs, video_inputs = process_vision_info(batched_messages)
                if video_inputs is not None:
                    total_frames = video_inputs[0].shape[0]
                    indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                    indices = np.unique(indices)
                    if total_frames - 1 not in indices:
                        indices = np.append(indices, total_frames - 1)
                        indices = np.unique(indices)
                    video_inputs[0] = video_inputs[0][indices]
                inputs = self.processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )

                if self.device_map == "auto":
                    inputs = inputs.to("cuda")
                else:
                    inputs = inputs.to(self.device)

                default_gen_kwargs = {
                    "max_new_tokens": 32768,
                    "temperature": 0.0,
                    "top_p": None,
                    "num_beams": 1,
                }
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

                clean_answers = []
                for ans in answers:
                    clean_ans = parse_reasoning_model_answer(ans)
                    clean_answers.append(clean_ans)

                batched_round_res.append(clean_answers)
                round_idx += 1

            res.extend(list(zip(*batched_round_res)))
            self.cache_hook.add_partial(
                "generate_until_multi_round",
                (batched_contexts[0], gen_kwargs),
                batched_round_res,
            )
            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res
