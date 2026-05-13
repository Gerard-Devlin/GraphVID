import copy
import gc
import json
import os
import random
import re
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from decord import VideoReader, cpu
from transformers.hf_argparser import HfArgumentParser

warnings.filterwarnings("ignore")

SEPARATOR = "=" * 72


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
    reload_model_each_phase: bool = field(default=True)

    # FlashVID settings for phase-2
    retention_ratio: float = field(default=0.10)
    do_segment: bool = field(default=True)
    segment_threshold: float = field(default=0.9)
    min_segment_num: int = field(default=8)
    complementary_segment: bool = field(default=True)
    token_selection_method: str = field(default="attn_div_v2")
    alpha: float = field(default=0.70)
    temporal_threshold: float = field(default=0.8)
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
    talon_medium_anchor_safety_ratio: float = field(default=0.78)
    talon_medium_event_budget_ratio: float = field(default=0.18)
    talon_medium_global_topk_ratio: float = field(default=0.80)
    talon_long_anchor_safety_ratio: float = field(default=0.80)
    talon_long_event_budget_ratio: float = field(default=0.14)
    talon_long_global_topk_ratio: float = field(default=0.85)
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
    talon_frame_coverage_floor_ratio: float = field(default=0.65)
    talon_frame_local_budget_ratio: float = field(default=1.0)
    talon_question_recall_ratio: float = field(default=0.06)
    talon_question_recall_qweight: float = field(default=0.65)
    talon_question_pooling: str = field(default="mean")
    talon_question_pooling_topk: int = field(default=4)
    talon_monotonic_base_tokens_per_frame: int = field(default=20)
    talon_budget_strategy: str = field(default="marginal")
    talon_budget_mode: str = field(default="attention")
    talon_transport_mode: str = field(default="hard")
    talon_transport_temperature: float = field(default=0.07)
    talon_echo_temperature: float = field(default=0.07)
    talon_echo_topk_neighbors: int = field(default=4)
    talon_echo_residual_weight: float = field(default=0.0)
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
    summary_output_json: str = field(default="logs/efficiency/summary_all_metrics.json")


def _extract_choice_letter(text: str) -> str:
    if not text:
        return ""
    t = (text or "").strip().upper()
    if not t:
        return ""

    # 1) Strict single-token answer forms: "A", "(B)", "C.", "[D]".
    m = re.match(r"^\s*[\(\[]?\s*([ABCD])\s*[\)\].,:;!?\u3002\uff0c\uff1a\uff1b]?[\s]*$", t)
    if m:
        return m.group(1)

    # 2) Common prefixed forms: "Answer: B", "Option C", "Choice is D".
    prefixed_patterns = [
        r"\b(?:ANSWER|OPTION|CHOICE)\b\s*[:=\-]?\s*[\(\[]?\s*([ABCD])\b",
        r"\b(?:THE\s+ANSWER\s+IS|I\s+CHOOSE|I\s+PICK)\b\s*[:=\-]?\s*[\(\[]?\s*([ABCD])\b",
    ]
    for pat in prefixed_patterns:
        m = re.search(pat, t)
        if m:
            return m.group(1)

    # 3) Fallback: first standalone option token (avoid letters inside words).
    m = re.search(r"\b([ABCD])\b", t)
    if m:
        return m.group(1)
    return ""


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


def _load_dataset(dataset_jsonl: str, limit: int | None, shuffle: bool, start_index: int = 0) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(dataset_jsonl).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
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
    overwrite_config = (
        {"mm_spatial_pool_mode": "average", "mm_newline_position": "frame"}
        if args.model_path == "lmms-lab/LLaVA-Video-7B-Qwen2"
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

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "video": video_path,
                    "max_pixels": args.max_pixels,
                    "min_pixels": args.min_pixels,
                    "nframes": args.num_frames,
                },
                {"type": "text", "text": prompt_text},
            ],
        },
    ]

    video_kwargs: dict[str, Any] = {}
    video_metadata = None
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
        images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    # qwen_vl_utils may return [(video_tensor, metadata), ...] when return_video_metadata=True.
    if videos is not None and len(videos) > 0 and isinstance(videos[0], tuple):
        videos, video_metadata = zip(*videos)
        videos = list(videos)
        video_metadata = list(video_metadata)

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
    video_path = _resolve_video_path(sample["videoID"], args.hf_home)
    prompt_text = sample["input"]
    backend = model_bundle["backend"]
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
    generated = output_ids[:, prompt_len:] if output_ids.shape[1] > prompt_len else output_ids[:, :0]
    gen_tokens = int(generated.shape[1])
    if gen_tokens == 0:
        return "", 0

    backend = model_bundle["backend"]
    if backend == "llava":
        tokenizer = model_bundle["tokenizer"]
        text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        answer = _extract_choice_letter(text)
        if answer:
            return answer, gen_tokens
        first_token_id = int(generated[0, 0].item())
        first_token = tokenizer.decode([first_token_id], skip_special_tokens=True)
        return _extract_choice_letter(first_token), gen_tokens

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
            "talon_memory_tokens": None,
            "talon_rank_cap": None,
            "talon_chosen_rank": None,
            "talon_duplicate_index_count": None,
            "talon_question_aware_active": None,
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
    memory_tokens = getattr(cfg, "last_talon_memory_tokens", None)
    rank_cap = getattr(cfg, "last_talon_rank_cap", None)
    chosen_rank = getattr(cfg, "last_talon_chosen_rank", None)
    duplicate_count = getattr(cfg, "last_talon_duplicate_index_count", None)
    question_active = getattr(cfg, "last_talon_question_aware_active", None)
    return {
        "talon_target_tokens_per_frame": float(target) if target is not None else None,
        "talon_adaptive_retention_ratio": float(adaptive_ratio) if adaptive_ratio is not None else None,
        "talon_complexity_score": float(complexity) if complexity is not None else None,
        "talon_target_budget": float(target_budget) if target_budget is not None else None,
        "talon_anchor_tokens": float(anchor_tokens) if anchor_tokens is not None else None,
        "talon_rank_tokens": float(rank_tokens) if rank_tokens is not None else None,
        "talon_event_tokens": float(event_tokens) if event_tokens is not None else None,
        "talon_recall_tokens": float(recall_tokens) if recall_tokens is not None else None,
        "talon_memory_tokens": float(memory_tokens) if memory_tokens is not None else None,
        "talon_rank_cap": float(rank_cap) if rank_cap is not None else None,
        "talon_chosen_rank": float(chosen_rank) if chosen_rank is not None else None,
        "talon_duplicate_index_count": float(duplicate_count) if duplicate_count is not None else None,
        "talon_question_aware_active": float(bool(question_active)) if question_active is not None else None,
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
                max_new_tokens=args.max_new_tokens,
                modalities=["video"],
            )
        else:
            model.generate(
                **inputs,
                do_sample=False,
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
    talon_memory_per_run = []
    talon_rank_cap_per_run = []
    talon_chosen_rank_per_run = []
    talon_duplicate_per_run = []
    talon_question_active_per_run = []
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
                    max_new_tokens=args.max_new_tokens,
                    modalities=["video"],
                )
            return model.generate(
                **inputs,
                do_sample=False,
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
        compressed_tokens_per_run.append(float(final_tokens))
        vision_tokens_per_run.append(float(vision_tokens))
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
    compressed_visual_tokens = float(np.mean(compressed_tokens_per_run)) if compressed_tokens_per_run else float(raw_visual_tokens)
    vision_compressed_visual_tokens = float(np.mean(vision_tokens_per_run)) if vision_tokens_per_run else float(raw_visual_tokens)
    talon_target_tokens_per_frame = float(np.mean(talon_target_per_run)) if talon_target_per_run else None
    talon_complexity_score = float(np.mean(talon_complexity_per_run)) if talon_complexity_per_run else None
    talon_target_budget = float(np.mean(talon_target_budget_per_run)) if talon_target_budget_per_run else None
    talon_anchor_tokens = float(np.mean(talon_anchor_per_run)) if talon_anchor_per_run else None
    talon_rank_tokens = float(np.mean(talon_rank_per_run)) if talon_rank_per_run else None
    talon_event_tokens = float(np.mean(talon_event_per_run)) if talon_event_per_run else None
    talon_recall_tokens = float(np.mean(talon_recall_per_run)) if talon_recall_per_run else None
    talon_memory_tokens = float(np.mean(talon_memory_per_run)) if talon_memory_per_run else None
    talon_rank_cap = float(np.mean(talon_rank_cap_per_run)) if talon_rank_cap_per_run else None
    talon_chosen_rank = float(np.mean(talon_chosen_rank_per_run)) if talon_chosen_rank_per_run else None
    talon_duplicate_index_count = float(np.mean(talon_duplicate_per_run)) if talon_duplicate_per_run else None
    talon_question_aware_active = float(np.mean(talon_question_active_per_run)) if talon_question_active_per_run else None
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
        "talon_memory_tokens": talon_memory_tokens,
        "talon_rank_cap": talon_rank_cap,
        "talon_chosen_rank": talon_chosen_rank,
        "talon_duplicate_index_count": talon_duplicate_index_count,
        "talon_question_aware_active": talon_question_aware_active,
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
    }


def _benchmark_single_sample(model_bundle, args: BenchmarkArgs, sample: dict[str, Any], use_acceleration: bool):
    record = {
        "question_id": sample.get("question_id"),
        "videoID": sample.get("videoID"),
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
        "talon_memory_tokens": None,
        "talon_rank_cap": None,
        "talon_chosen_rank": None,
        "talon_duplicate_index_count": None,
        "talon_question_aware_active": None,
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
                "correct": result["pred_answer"] == sample.get("answer"),
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
                "talon_memory_tokens": result.get("talon_memory_tokens"),
                "talon_rank_cap": result.get("talon_rank_cap"),
                "talon_chosen_rank": result.get("talon_chosen_rank"),
                "talon_duplicate_index_count": result.get("talon_duplicate_index_count"),
                "talon_question_aware_active": result.get("talon_question_aware_active"),
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
):
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for idx, sample in enumerate(samples, 1):
            record = _benchmark_single_sample(model_bundle, args, sample, use_acceleration=use_acceleration)

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

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
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
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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
        "talon_memory_tokens": _stats(talon_memory_tokens),
        "talon_rank_cap": _stats(talon_rank_cap),
        "talon_chosen_rank": _stats(talon_chosen_rank),
        "talon_duplicate_index_count": _stats(talon_duplicate_count),
        "talon_question_aware_active": _stats(talon_question_active),
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
    for b, f in matched:
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
        "latency_speedup_ratio": _stats(latency_ratios),
        f"visual_token_ratio_{target_name}_over_{anchor_name}": _stats(token_ratios),
        f"visual_token_reduction_vs_{anchor_name}": _stats(reduction_gains),
        f"vision_token_ratio_{target_name}_over_{anchor_name}": _stats(vision_token_ratios),
        f"vision_token_reduction_vs_{anchor_name}": _stats(vision_reduction_gains),
    }


def _resolve_llm_pruning_args(backend: str, args: BenchmarkArgs) -> tuple[int, float]:
    # LLaVA backend currently has instability in inner-LLM token pruning path.
    # Keep visual-side compression enabled, but disable LLM pruning for stable benchmarking.
    if backend == "llava":
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
        talon_frame_coverage_floor_ratio=args.talon_frame_coverage_floor_ratio,
        talon_frame_local_budget_ratio=args.talon_frame_local_budget_ratio,
        talon_question_recall_ratio=args.talon_question_recall_ratio,
        talon_question_recall_qweight=args.talon_question_recall_qweight,
        talon_question_pooling=args.talon_question_pooling,
        talon_question_pooling_topk=args.talon_question_pooling_topk,
        talon_monotonic_base_tokens_per_frame=args.talon_monotonic_base_tokens_per_frame,
        talon_budget_strategy=args.talon_budget_strategy,
        talon_budget_mode=args.talon_budget_mode,
        talon_transport_mode=args.talon_transport_mode,
        talon_transport_temperature=args.talon_transport_temperature,
        talon_echo_temperature=args.talon_echo_temperature,
        talon_echo_topk_neighbors=args.talon_echo_topk_neighbors,
        talon_echo_residual_weight=args.talon_echo_residual_weight,
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
        token_selection_method=args.token_selection_method,
        alpha=args.alpha,
        temporal_threshold=args.temporal_threshold,
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
        talon_frame_coverage_floor_ratio=args.talon_frame_coverage_floor_ratio,
        talon_frame_local_budget_ratio=args.talon_frame_local_budget_ratio,
        talon_question_recall_ratio=args.talon_question_recall_ratio,
        talon_question_recall_qweight=args.talon_question_recall_qweight,
        talon_question_pooling=args.talon_question_pooling,
        talon_question_pooling_topk=args.talon_question_pooling_topk,
        talon_monotonic_base_tokens_per_frame=args.talon_monotonic_base_tokens_per_frame,
        talon_budget_strategy=args.talon_budget_strategy,
        talon_budget_mode=args.talon_budget_mode,
        talon_transport_mode=args.talon_transport_mode,
        talon_transport_temperature=args.talon_transport_temperature,
        talon_echo_temperature=args.talon_echo_temperature,
        talon_echo_topk_neighbors=args.talon_echo_topk_neighbors,
        talon_echo_residual_weight=args.talon_echo_residual_weight,
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
        decode_policy=args.decode_policy,
        decode_kv_budget_ratio=args.decode_kv_budget_ratio,
        decode_update_interval=args.decode_update_interval,
        decode_start_layer=args.decode_start_layer,
    )


def _print_header(args: BenchmarkArgs, backend: str):
    effective_attn = _resolve_attn_implementation(args.attn_implementation)
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
        f"baseline={args.run_baseline}, flashvid={args.run_flashvid}, ours={args.run_ours}"
    )
    print(f"Phase reload  : {args.reload_model_each_phase}")
    if args.run_ours:
        print(
            "Ours config   : "
            f"variant={args.compression_variant}, qa={args.question_aware_reweighting}, "
            f"adaptive={args.adaptive_token_budget}, budget={args.talon_budget_strategy}, "
            f"scale={args.talon_budget_scale}, target_per_frame={args.talon_target_tokens_per_frame}, "
            f"event_cap={args.talon_event_budget_ratio:.2f}, "
            f"anchor_div={args.talon_anchor_diversity_weight:.2f}"
        )
    print(SEPARATOR)


def _print_summary(summary: dict[str, Any]):
    print(SEPARATOR)
    print("Summary")
    print(SEPARATOR)
    for phase_name in ("baseline", "flashvid", "ours"):
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
        talon_memory_mean = phase.get("talon_memory_tokens", {}).get("mean")
        talon_rank_cap_mean = phase.get("talon_rank_cap", {}).get("mean")
        talon_chosen_rank_mean = phase.get("talon_chosen_rank", {}).get("mean")
        talon_dup_mean = phase.get("talon_duplicate_index_count", {}).get("mean")
        talon_question_active_mean = phase.get("talon_question_aware_active", {}).get("mean")
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
        if talon_rank_mean is not None:
            print(f"  talon rank/memory mean: {talon_rank_mean:.2f}/{(talon_memory_mean or 0.0):.2f}")
        if talon_rank_cap_mean is not None:
            print(f"  talon rank cap/chosen mean: {talon_rank_cap_mean:.2f}/{(talon_chosen_rank_mean or 0.0):.2f}")
        if talon_dup_mean is not None:
            print(f"  talon duplicate index mean: {talon_dup_mean:.2f}")
        if talon_question_active_mean is not None:
            print(f"  talon question-aware active mean: {talon_question_active_mean:.2f}")
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
        if red_mean is not None:
            print(f"  final token reduction mean: {red_mean * 100:.2f}%")
        if vision_red_mean is not None:
            print(f"  vision-side token reduction mean: {vision_red_mean * 100:.2f}%")

    comparison = summary.get("comparison", {})
    if comparison:
        print("[comparison]")
        for key in ("baseline_vs_flashvid", "baseline_vs_ours", "flashvid_vs_ours"):
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
            print(f"  [{key}] matched={comp['matched_samples']}")
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
    print(SEPARATOR)


def run(args: BenchmarkArgs):
    samples = _load_dataset(args.dataset_jsonl, args.limit, args.shuffle, args.start_index)
    if not samples:
        raise ValueError(f"No samples loaded from {args.dataset_jsonl}")
    if not (args.run_baseline or args.run_flashvid or args.run_ours):
        raise ValueError("At least one phase must be enabled: run_baseline/run_flashvid/run_ours")

    model_bundle = _load_backend_model(args)
    backend = model_bundle["backend"]
    _print_header(args, backend)
    if backend == "llava":
        print("[info] LLaVA backend: inner-LLM pruning is disabled for stability (vision compression remains enabled).")
    print(f"Loaded {len(samples)} samples.\n")
    if args.reload_model_each_phase:
        model_bundle["model"] = None
        model_bundle["processor"] = None
        model_bundle["tokenizer"] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_phases = int(args.run_baseline) + int(args.run_flashvid) + int(args.run_ours)
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

    if args.run_ours:
        print(f"\nPhase {phase_idx}/{total_phases}: Ours ...")
        print(
            "[talon-active][ours] "
            f"path=clean, qaware={args.question_aware_reweighting}, "
            f"target/frame={args.talon_target_tokens_per_frame}, "
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
                phase_name="Ours",
                use_acceleration=True,
                output_path=args.ours_output,
            )
        finally:
            _release_phase_bundle(phase_bundle)

    summary: dict[str, Any] = {"comparison": {}}
    baseline_records = None
    flashvid_records = None
    ours_records = None
    if args.run_baseline:
        baseline_records = _read_jsonl(args.baseline_output)
        summary["baseline"] = _summarize_phase(baseline_records)
    if args.run_flashvid:
        flashvid_records = _read_jsonl(args.flashvid_output)
        summary["flashvid"] = _summarize_phase(flashvid_records)
    if args.run_ours:
        ours_records = _read_jsonl(args.ours_output)
        summary["ours"] = _summarize_phase(ours_records)

    if baseline_records is not None and flashvid_records is not None:
        summary["comparison"]["baseline_vs_flashvid"] = _summarize_pairwise_comparison(
            baseline_records,
            flashvid_records,
            anchor_name="baseline",
            target_name="flashvid",
        )
    if baseline_records is not None and ours_records is not None:
        summary["comparison"]["baseline_vs_ours"] = _summarize_pairwise_comparison(
            baseline_records,
            ours_records,
            anchor_name="baseline",
            target_name="ours",
        )
    if flashvid_records is not None and ours_records is not None:
        summary["comparison"]["flashvid_vs_ours"] = _summarize_pairwise_comparison(
            flashvid_records,
            ours_records,
            anchor_name="flashvid",
            target_name="ours",
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
