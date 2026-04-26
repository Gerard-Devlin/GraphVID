import copy
import json
import os
import random
import re
import time
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

    # Data
    dataset_jsonl: str = field(default="videomme.jsonl")
    hf_home: str | None = field(default=None)
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
    run_flashvid: bool = field(default=True)
    run_ours: bool = field(default=True)

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
    compression_variant: str = field(default="graph")
    question_aware_reweighting: bool = field(default=True)
    adaptive_token_budget: bool = field(default=True)
    adaptive_budget_low: float = field(default=0.10)
    adaptive_budget_mid: float = field(default=0.15)
    adaptive_budget_high: float = field(default=0.20)
    graph_topk: int = field(default=4)
    graph_temporal_radius: int = field(default=1)
    memory_token_ratio: float = field(default=0.10)

    # Outputs
    baseline_output: str = field(default="logs/efficiency/baseline_all_metrics.jsonl")
    flashvid_output: str = field(default="logs/efficiency/flashvid_all_metrics.jsonl")
    ours_output: str = field(default="logs/efficiency/ours_all_metrics.jsonl")
    summary_output_json: str = field(default="logs/efficiency/summary_all_metrics.json")


def _extract_choice_letter(text: str) -> str:
    match = re.search(r"[ABCD]", (text or "").upper())
    return match.group(0) if match else ""


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


def _load_dataset(dataset_jsonl: str, limit: int | None, shuffle: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(dataset_jsonl).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    if shuffle:
        random.shuffle(records)
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
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        None,
        model_name,
        device_map="auto",
        attn_implementation=args.attn_implementation,
        overwrite_config=overwrite_config,
        multimodal=True,
    )
    model.eval()
    return {"model": model, "tokenizer": tokenizer, "image_processor": image_processor}


def _load_qwen_model(args: BenchmarkArgs, backend: str):
    from transformers import AutoProcessor

    if backend == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration as QwenModel
    elif backend == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration as QwenModel
    else:
        raise ValueError(f"unsupported qwen backend: {backend}")

    model = QwenModel.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation=args.attn_implementation,
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    model.eval()
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
    image_processor = model_bundle["image_processor"]
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    frame_idx = np.linspace(0, total_frames - 1, args.num_frames, dtype=int)
    video_frames = vr.get_batch(frame_idx.tolist()).asnumpy()
    frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()

    conv = copy.deepcopy(conv_templates["qwen_1_5"])
    conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{prompt_text}")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to("cuda")
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

    images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")

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


def _get_compressed_visual_tokens(model, raw_visual_tokens: int, use_acceleration: bool) -> int:
    if not use_acceleration:
        return raw_visual_tokens
    if not hasattr(model, "flashvid_config"):
        return raw_visual_tokens
    cfg = getattr(model, "flashvid_config")
    length = getattr(cfg, "visual_token_length", None)
    if length is None:
        return raw_visual_tokens
    try:
        return int(length)
    except Exception:
        return raw_visual_tokens


def _run_benchmark_once(model_bundle, args: BenchmarkArgs, prepared_inputs, use_acceleration: bool):
    model = model_bundle["model"]
    backend = model_bundle["backend"]

    # Warmup
    for _ in range(args.num_warmup):
        inputs = _clone_inputs(backend, prepared_inputs)
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
    prompt_len = prepared_inputs["prompt_len"]
    raw_visual_tokens = int(prepared_inputs["raw_visual_tokens"])

    for run_idx in range(args.num_runs):
        inputs = _clone_inputs(backend, prepared_inputs)

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
        compressed_tokens_per_run.append(
            float(_get_compressed_visual_tokens(model, raw_visual_tokens, use_acceleration))
        )

    latency_ms = float(np.mean(latencies)) if latencies else None
    generated_tokens = float(np.mean(gen_tokens_per_run)) if gen_tokens_per_run else None
    compressed_visual_tokens = float(np.mean(compressed_tokens_per_run)) if compressed_tokens_per_run else float(raw_visual_tokens)
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
    }


def _benchmark_single_sample(model_bundle, args: BenchmarkArgs, sample: dict[str, Any], use_acceleration: bool):
    record = {
        "question_id": sample.get("question_id"),
        "videoID": sample.get("videoID"),
        "answer": sample.get("answer"),
        "pred_answer": "",
        "correct": None,
        "latency_ms": None,
        "generated_tokens": None,
        "tokens_per_second": None,
        "raw_visual_tokens": None,
        "compressed_visual_tokens": None,
        "visual_token_reduction_ratio": None,
        "error": None,
    }

    try:
        prepared_inputs = _prepare_inputs(model_bundle, args, sample)
        result = _run_benchmark_once(model_bundle, args, prepared_inputs, use_acceleration=use_acceleration)
        raw_v = result["raw_visual_tokens"]
        compressed_v = result["compressed_visual_tokens"]
        reduction_ratio = None
        if raw_v and raw_v > 0:
            reduction_ratio = float(max(0.0, 1.0 - (compressed_v / raw_v)))

        record.update(
            {
                "pred_answer": result["pred_answer"],
                "correct": result["pred_answer"] == sample.get("answer"),
                "latency_ms": result["latency_ms"],
                "generated_tokens": result["generated_tokens"],
                "tokens_per_second": result["tokens_per_second"],
                "raw_visual_tokens": raw_v,
                "compressed_visual_tokens": compressed_v,
                "visual_token_reduction_ratio": reduction_ratio,
            }
        )
    except Exception as exc:  # pragma: no cover - runtime path
        record["error"] = str(exc)

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
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            if record["error"]:
                print(f"[{phase_name}] {idx}/{len(samples)} {record.get('question_id')} error: {record['error']}")
            else:
                print(
                    f"[{phase_name}] {idx}/{len(samples)} {record.get('question_id')} "
                    f"acc={record['correct']} latency={record['latency_ms']:.2f}ms "
                    f"vtoken={record['compressed_visual_tokens']:.1f}/{record['raw_visual_tokens']:.1f}"
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
    reduction = [float(r["visual_token_reduction_ratio"]) for r in valid if r.get("visual_token_reduction_ratio") is not None]

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
        "visual_token_reduction_ratio": _stats(reduction),
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
    reduction_gains = []
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

    return {
        "matched_samples": len(matched),
        "latency_speedup_ratio": _stats(latency_ratios),
        f"visual_token_ratio_{target_name}_over_{anchor_name}": _stats(token_ratios),
        f"visual_token_reduction_vs_{anchor_name}": _stats(reduction_gains),
    }


def _apply_flashvid_original(model, args: BenchmarkArgs):
    from flashvid import flashvid

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
        pruning_layer=args.pruning_layer,
        llm_retention_ratio=args.llm_retention_ratio,
        compression_variant="flashvid",
        question_aware_reweighting=False,
        adaptive_token_budget=False,
    )


def _apply_ours(model, args: BenchmarkArgs):
    from flashvid import flashvid

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
        pruning_layer=args.pruning_layer,
        llm_retention_ratio=args.llm_retention_ratio,
        compression_variant=args.compression_variant,
        question_aware_reweighting=args.question_aware_reweighting,
        adaptive_token_budget=args.adaptive_token_budget,
        adaptive_budget_low=args.adaptive_budget_low,
        adaptive_budget_mid=args.adaptive_budget_mid,
        adaptive_budget_high=args.adaptive_budget_high,
        graph_topk=args.graph_topk,
        graph_temporal_radius=args.graph_temporal_radius,
        memory_token_ratio=args.memory_token_ratio,
    )


def _print_header(args: BenchmarkArgs, backend: str):
    print(SEPARATOR)
    print("Unified Benchmark: Accuracy + Token + Latency + Speedup")
    print(SEPARATOR)
    print(f"Model path    : {args.model_path}")
    print(f"Backend       : {backend}")
    print(f"Dataset       : {args.dataset_jsonl}")
    print(f"HF_HOME       : {args.hf_home or os.getenv('HF_HOME', '~/.cache/huggingface')}")
    print(f"Limit         : {args.limit}")
    print(f"Shuffle       : {args.shuffle}")
    print(f"Frames        : {args.num_frames}")
    print(f"Warmup/Runs   : {args.num_warmup}/{args.num_runs}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Run phases    : baseline, flashvid={args.run_flashvid}, ours={args.run_ours}")
    if args.run_ours:
        print(
            "Ours config   : "
            f"variant={args.compression_variant}, qa={args.question_aware_reweighting}, "
            f"adaptive={args.adaptive_token_budget}"
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
        red_mean = phase["visual_token_reduction_ratio"]["mean"]
        if lat_mean is not None:
            print(f"  latency mean: {lat_mean:.2f} ms")
        if vt_mean is not None:
            print(f"  visual tokens mean: {vt_mean:.2f}")
        if red_mean is not None:
            print(f"  token reduction mean: {red_mean * 100:.2f}%")

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
            token_red = comp[reduction_key]["mean"] if reduction_key else None
            print(f"  [{key}] matched={comp['matched_samples']}")
            if lat_sp is not None:
                print(f"    latency speedup: {lat_sp:.3f}x")
            if ratio_key and comp[ratio_key]["mean"] is not None:
                print(f"    {ratio_key}: {comp[ratio_key]['mean']:.3f}")
            if token_red is not None:
                print(f"    token reduction: {token_red * 100:.2f}%")
    print(SEPARATOR)


def run(args: BenchmarkArgs):
    samples = _load_dataset(args.dataset_jsonl, args.limit, args.shuffle)
    if not samples:
        raise ValueError(f"No samples loaded from {args.dataset_jsonl}")

    model_bundle = _load_backend_model(args)
    backend = model_bundle["backend"]
    _print_header(args, backend)
    print(f"Loaded {len(samples)} samples.\n")

    total_phases = 1 + int(args.run_flashvid) + int(args.run_ours)
    phase_idx = 1

    print(f"Phase {phase_idx}/{total_phases}: Baseline ...")
    _run_phase(
        model_bundle=model_bundle,
        args=args,
        samples=samples,
        phase_name="Baseline",
        use_acceleration=False,
        output_path=args.baseline_output,
    )
    phase_idx += 1

    if args.run_flashvid:
        print(f"\nPhase {phase_idx}/{total_phases}: FlashVID ...")
        model_bundle["model"] = _apply_flashvid_original(model_bundle["model"], args)
        _run_phase(
            model_bundle=model_bundle,
            args=args,
            samples=samples,
            phase_name="FlashVID",
            use_acceleration=True,
            output_path=args.flashvid_output,
        )
        phase_idx += 1

    if args.run_ours:
        print(f"\nPhase {phase_idx}/{total_phases}: Ours ...")
        model_bundle["model"] = _apply_ours(model_bundle["model"], args)
        _run_phase(
            model_bundle=model_bundle,
            args=args,
            samples=samples,
            phase_name="Ours",
            use_acceleration=True,
            output_path=args.ours_output,
        )

    baseline_records = _read_jsonl(args.baseline_output)
    summary: dict[str, Any] = {
        "baseline": _summarize_phase(baseline_records),
        "comparison": {},
    }
    flashvid_records = None
    ours_records = None
    if args.run_flashvid:
        flashvid_records = _read_jsonl(args.flashvid_output)
        summary["flashvid"] = _summarize_phase(flashvid_records)
        summary["comparison"]["baseline_vs_flashvid"] = _summarize_pairwise_comparison(
            baseline_records,
            flashvid_records,
            anchor_name="baseline",
            target_name="flashvid",
        )
    if args.run_ours:
        ours_records = _read_jsonl(args.ours_output)
        summary["ours"] = _summarize_phase(ours_records)
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


def main():
    parser = HfArgumentParser(BenchmarkArgs)
    (args,) = parser.parse_args_into_dataclasses(return_remaining_strings=False)
    run(args)


if __name__ == "__main__":
    main()
