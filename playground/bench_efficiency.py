"""Reproducible LLaVA-OneVision efficiency benchmark at 1% retention.

The benchmark has three deliberately separate stages:

* ``manifest`` selects one question from each of 100 distinct VideoMME videos.
* ``run`` benchmarks exactly one method in one fresh Python process.
* ``summarize`` validates all raw records and writes JSON/CSV/Markdown/LaTeX.

Video decoding, preprocessing, and model loading happen outside every timed
region. CUDA events measure vision encoding, outer compression, and the first
LLM prefill independently. TTFT follows the FlashVID efficiency convention and
is the sum of those three measured stages.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import statistics
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from decord import VideoReader, cpu

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.model.builder import load_pretrained_model

warnings.filterwarnings("ignore")

METHOD_ORDER = ("vanilla", "fastv", "visionzip", "fastvid", "flashvid", "ours")
METHOD_LABELS = {
    "vanilla": "Vanilla",
    "fastv": "FastV",
    "visionzip": "VisionZip",
    "fastvid": "FastVID",
    "flashvid": "FlashVID",
    "ours": "Ours",
}
DURATION_ORDER = ("short", "medium", "long")
DEFAULT_DURATION_COUNTS = {"short": 34, "medium": 33, "long": 33}
DEFAULT_SCORE_FILE = Path("scripts/efficiency/llava_ov_r1_scores.json")
SAMPLING_PROTOCOL = "uniform_segment_centers_tail_guard_v1"
CERTVID_PHASE_PREFIX = "certvid_phase_"


@dataclass(frozen=True)
class MethodSpec:
    variant: str | None
    expansion: float
    pruning_layer: int
    inner_retention: float
    budget_contract: str
    token_selection_method: str = "attn_div_v2"


METHOD_SPECS = {
    "vanilla": MethodSpec(None, 1.0, 28, 1.0, "uncompressed"),
    # FastV's published contract applies R at its layer-2 pruning point. Its
    # first two dense layers are reported in TFLOPs and in the layer-average
    # audit rather than being silently hidden.
    "fastv": MethodSpec("fastv", 1.0, 2, 0.01, "post_prune"),
    "visionzip": MethodSpec("visionzip", 1.0, 28, 1.0, "outer"),
    "fastvid": MethodSpec("fastvid", 1.0, 28, 1.0, "outer"),
    "flashvid": MethodSpec("flashvid", 1.25, 20, 0.3, "layer_average"),
    "ours": MethodSpec(
        "certvidfinal",
        1.30,
        20,
        0.1923076923,
        "layer_average",
        token_selection_method="attn_div_stable",
    ),
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _stable_rank(seed: int, *parts: object) -> str:
    text = ":".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _question_id(sample: dict[str, Any]) -> str:
    value = sample.get("question_id") or sample.get("id") or sample.get("q_uid")
    if value is None:
        raise ValueError("VideoMME record is missing question_id/id/q_uid")
    return str(value)


def _duration(sample: dict[str, Any]) -> str:
    value = str(sample.get("duration") or sample.get("duration_category") or "")
    value = value.strip().lower()
    if value not in DURATION_ORDER:
        raise ValueError(f"invalid or missing VideoMME duration category: {value!r}")
    return value


def _duration_targets(sample_count: int) -> dict[str, int]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    exact = {
        key: sample_count * DEFAULT_DURATION_COUNTS[key] / 100.0
        for key in DURATION_ORDER
    }
    targets = {key: int(math.floor(exact[key])) for key in DURATION_ORDER}
    remainder = sample_count - sum(targets.values())
    ranked = sorted(
        DURATION_ORDER,
        key=lambda key: (-(exact[key] - targets[key]), DURATION_ORDER.index(key)),
    )
    for key in ranked[:remainder]:
        targets[key] += 1
    return targets


def build_manifest(args: argparse.Namespace) -> None:
    source = Path(args.dataset_jsonl)
    samples = _read_jsonl(source)
    if not samples:
        raise ValueError(f"no records found in {source}")

    # First choose one question deterministically per video, then stratify the
    # unique videos. This prevents videos with many questions from dominating.
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        video_id = str(sample.get("videoID") or "").strip()
        if not video_id:
            raise ValueError("VideoMME record is missing videoID")
        _duration(sample)
        _question_id(sample)
        by_video[video_id].append(sample)

    one_question_per_video: dict[str, dict[str, Any]] = {}
    for video_id, candidates in by_video.items():
        ordered = sorted(
            candidates,
            key=lambda row: _stable_rank(args.seed, video_id, _question_id(row)),
        )
        one_question_per_video[video_id] = ordered[0]

    targets = _duration_targets(args.sample_count)
    selected: list[dict[str, Any]] = []
    actual_counts: dict[str, int] = {}
    for duration in DURATION_ORDER:
        candidates = [
            sample
            for sample in one_question_per_video.values()
            if _duration(sample) == duration
        ]
        candidates.sort(
            key=lambda row: _stable_rank(
                args.seed, duration, row["videoID"], _question_id(row)
            )
        )
        required = targets[duration]
        if len(candidates) < required:
            raise ValueError(
                f"duration={duration} has {len(candidates)} unique videos; "
                f"need {required}"
            )
        chosen = candidates[:required]
        actual_counts[duration] = len(chosen)
        selected.extend(chosen)

    selected.sort(key=lambda row: _stable_rank(args.seed, row["videoID"]))
    manifest_records = []
    for index, sample in enumerate(selected):
        record = dict(sample)
        record["efficiency_sample_index"] = index
        record["efficiency_question_id"] = _question_id(sample)
        record["efficiency_duration"] = _duration(sample)
        manifest_records.append(record)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in manifest_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    metadata = {
        "source": str(source.resolve()),
        "manifest": str(output.resolve()),
        "seed": args.seed,
        "sample_count": len(manifest_records),
        "unique_video_count": len({row["videoID"] for row in manifest_records}),
        "duration_counts": actual_counts,
        "question_ids_sha256": hashlib.sha256(
            "\n".join(
                row["efficiency_question_id"] for row in manifest_records
            ).encode()
        ).hexdigest(),
    }
    _write_json(output.with_suffix(output.suffix + ".meta.json"), metadata)
    print(json.dumps(metadata, indent=2))


def load_video(video_path: str, num_frames: int) -> np.ndarray:
    """Decode deterministic segment-center frames while avoiding fragile EOFs."""
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")

    errors: list[str] = []
    # Segment-center sampling avoids forcing Decord to seek to a frequently
    # damaged or inaccurately indexed final frame. Wider guards are only used
    # when the first deterministic schedule still encounters a corrupt tail.
    for tail_guard in (1.0 / 64.0, 1.0 / 32.0, 1.0 / 16.0):
        reader = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frames = len(reader)
        if total_frames <= 0:
            raise ValueError(f"video has no decodable frames: {video_path}")
        usable_frames = max(1, int(math.floor(total_frames * (1.0 - tail_guard))))
        edges = np.linspace(0.0, float(usable_frames), num_frames + 1)
        indices = np.floor((edges[:-1] + edges[1:]) * 0.5).astype(np.int64)
        indices = np.clip(indices, 0, total_frames - 1)
        try:
            return reader.get_batch(indices.tolist()).asnumpy()
        except Exception as exc:  # Decord exposes decoder failures per backend.
            errors.append(f"tail_guard={tail_guard:.5f}: {exc}")

    details = " | ".join(errors)
    raise RuntimeError(f"failed to decode stable frames from {video_path}: {details}")


def frame_content_sha256(frames: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(frames)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def resolve_video_path(video_id: str, video_root: Path) -> Path:
    for suffix in (".mp4", ".MP4", ".mkv", ".webm"):
        candidate = video_root / f"{video_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing videoID={video_id} under {video_root}")


def get_model(args: argparse.Namespace):
    overwrite_config = {"mm_spatial_pool_mode": args.mm_spatial_pool_mode}
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        None,
        args.model_name,
        device_map=args.device,
        attn_implementation="flash_attention_2",
        overwrite_config=overwrite_config,
        multimodal=True,
    )
    model.eval()
    return tokenizer, model, image_processor


def prepare_inputs(tokenizer, image_processor, frames: np.ndarray, prompt_text: str):
    pixel_values = image_processor.preprocess(frames, return_tensors="pt")[
        "pixel_values"
    ]
    pixel_values = pixel_values.to(device="cuda", dtype=torch.float16)

    conversation = copy.deepcopy(conv_templates["qwen_1_5"])
    conversation.append_message(
        conversation.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{prompt_text}"
    )
    conversation.append_message(conversation.roles[1], None)
    prompt = conversation.get_prompt()
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to("cuda")
    attention_mask = torch.ones_like(input_ids)
    image_sizes = [(int(frame.shape[1]), int(frame.shape[0])) for frame in frames]
    return input_ids, attention_mask, [pixel_values], image_sizes


def extract_choice_letter(text: str) -> str:
    patterns = (
        r"^\s*[\(\[]?\s*([A-Da-d])\s*[\)\].,:;!?]?\s*$",
        r"\b(?:ANSWER|OPTION|CHOICE)\b\s*(?:IS)?\s*[:=\-]?\s*[\(\[]?\s*([A-Da-d])\b",
        r"[\(\[]\s*([A-Da-d])\s*[\)\]]",
        r"\b([A-Da-d])\s*\.",
        r"\b([A-Da-d])\b",
    )
    for pattern in patterns:
        match = re.search(pattern, str(text or ""), flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return ""


def decode_generated_output(tokenizer, output_ids: torch.Tensor, prompt_length: int) -> str:
    generated = output_ids[:, prompt_length:] if output_ids.shape[1] > prompt_length else output_ids
    if generated.shape[1] == 0:
        return ""
    text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
    return extract_choice_letter(text)


def apply_method(model, method: str, retention_ratio: float):
    if method == "vanilla":
        return model
    from flashvid import flashvid

    spec = METHOD_SPECS[method]
    variant = spec.variant
    if method == "ours":
        variant = os.getenv("EFFICIENCY_OURS_VARIANT", variant)
    kwargs: dict[str, Any] = {
        "retention_ratio": retention_ratio,
        "compression_variant": variant,
        "expansion": spec.expansion,
        "pruning_layer": spec.pruning_layer,
        # FastV applies the requested budget directly at its pruning layer.
        # Hybrid methods instead use a fixed inner ratio paired with expansion
        # to satisfy their layer-average retention contract.
        "llm_retention_ratio": (
            retention_ratio if method == "fastv" else spec.inner_retention
        ),
        "token_selection_method": spec.token_selection_method,
        "do_segment": True,
        "segment_threshold": 0.9,
        "min_segment_num": 8,
        "complementary_segment": True,
        "alpha": 0.70,
        "temporal_threshold": 0.8,
        "adapter_budget_uses_expansion": False,
    }
    if method == "fastvid":
        kwargs.update(
            fastvid_DySeg_c=8,
            fastvid_DySeg_tau=0.90,
            fastvid_STPrune_d=0.40,
            fastvid_DTM_p=4,
            fastvid_DTM_beta=0.60,
        )
    elif method == "visionzip":
        kwargs["visionzip_dominant_ratio"] = 65.0 / 70.0
    elif method == "flashvid":
        # Explicit strict mode removes the original per-frame ceil surplus.
        kwargs["strict_token_budget"] = True
    elif method == "ours":
        kwargs.update(
            certv3_budget_uses_expansion=True,
            certv3_certificate_budget_ratio=0.0,
            strict_token_budget=True,
        )
    return flashvid(model, **kwargs)


class StageTimer:
    """CUDA-event timer for one generate call."""

    def __init__(self, model, accelerated: bool):
        self.model = model
        self.accelerated = accelerated
        self.events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        self.finished: set[str] = set()
        self.hooks: list[Any] = []
        self.original_compression: tuple[Any, Any] | None = None
        self.raw_visual_tokens: int | None = None
        self.outer_visual_tokens: int | None = None
        self.llm_outer_visual_tokens: int | None = None
        self.certvid_phase_events: dict[
            str, tuple[torch.cuda.Event, torch.cuda.Event]
        ] | None = None

    def _events_for(self, name: str) -> tuple[torch.cuda.Event, torch.cuda.Event]:
        pair = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        self.events[name] = pair
        return pair

    def install(self) -> None:
        vision_tower = self.model.get_model().get_vision_tower()
        projector = self.model.get_model().mm_projector
        llm_backbone = self.model.model

        def start_once(name: str):
            def hook(_module, _inputs):
                if name not in self.events:
                    start, _ = self._events_for(name)
                    start.record()
            return hook

        def end_once(name: str):
            def hook(_module, _inputs, _output):
                if name in self.events and name not in self.finished:
                    _, end = self.events[name]
                    end.record()
                    self.finished.add(name)
            return hook

        def llm_start(_module, _inputs):
            if "llm_forward" not in self.events:
                start, _ = self._events_for("llm_forward")
                start.record()
                config = getattr(self.model, "flashvid_config", None)
                if config is not None:
                    value = getattr(config, "visual_token_length", None)
                    if value is not None:
                        self.llm_outer_visual_tokens = int(value)

        self.hooks.extend(
            [
                vision_tower.register_forward_pre_hook(start_once("vision_encoding")),
                projector.register_forward_hook(end_once("vision_encoding")),
                llm_backbone.register_forward_pre_hook(llm_start),
                llm_backbone.register_forward_hook(end_once("llm_forward")),
            ]
        )

        if not self.accelerated:
            return
        import flashvid.llava_arch as flashvid_arch

        original = flashvid_arch.flashvid_compression
        self.original_compression = (flashvid_arch, original)

        def timed_compression(*call_args, **call_kwargs):
            video_features = call_kwargs.get("video_features")
            if video_features is None and call_args:
                video_features = call_args[0]
            if torch.is_tensor(video_features):
                self.raw_visual_tokens = int(np.prod(video_features.shape[:-1]))

            first_call = "compression" not in self.events
            if first_call:
                start, end = self._events_for("compression")
                start.record()
            result = original(*call_args, **call_kwargs)
            if first_call:
                end.record()
                output = result[0] if isinstance(result, tuple) else result
                if torch.is_tensor(output):
                    self.outer_visual_tokens = int(output.shape[0])
                config = getattr(self.model, "flashvid_config", None)
                phase_events = getattr(config, "_certv3_profile_events", None)
                if phase_events:
                    self.certvid_phase_events = dict(phase_events)
            return result

        flashvid_arch.flashvid_compression = timed_compression

    def read(self) -> dict[str, float]:
        torch.cuda.synchronize()
        missing = {"vision_encoding", "llm_forward"} - set(self.events)
        if missing:
            raise RuntimeError(f"missing CUDA timing events: {sorted(missing)}")
        values = {
            name: float(start.elapsed_time(end))
            for name, (start, end) in self.events.items()
        }
        values.setdefault("compression", 0.0)
        if self.certvid_phase_events:
            for name, (start, end) in self.certvid_phase_events.items():
                values[f"{CERTVID_PHASE_PREFIX}{name}"] = float(
                    start.elapsed_time(end)
                )
        return values

    def close(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        if self.original_compression is not None:
            module, original = self.original_compression
            module.flashvid_compression = original
            self.original_compression = None


def llm_tflops(sequence_tokens: int, layers: int) -> float:
    # FlashVID's published LLaVA-OneVision formula (visual-token prefill only).
    hidden = 3584
    heads = 28
    kv_groups = 4
    ff_hidden = 18944
    seq = int(sequence_tokens)
    per_layer = (
        2 * seq * hidden**2
        + 2 * seq * hidden**2 * (kv_groups / heads)
        + 2 * seq**2 * hidden
        + 3 * seq * hidden * ff_hidden
    )
    return float(layers * per_layer / 1e12)


def token_audit(
    method: str,
    retention_ratio: float,
    raw_tokens: int,
    outer_tokens: int,
    inner_tokens: int,
) -> dict[str, Any]:
    spec = METHOD_SPECS[method]
    layers = 28
    dense_layers = min(max(spec.pruning_layer, 0), layers)
    layer_average = (
        dense_layers * outer_tokens + (layers - dense_layers) * inner_tokens
    ) / layers
    target_float = raw_tokens * retention_ratio
    target_floor = int(math.floor(target_float + 1e-9))

    if spec.budget_contract == "uncompressed":
        audited_tokens = raw_tokens
        budget_ok = outer_tokens == raw_tokens and inner_tokens == raw_tokens
    elif spec.budget_contract == "outer":
        audited_tokens = outer_tokens
        # VisionZip's global allocator is exact. FastVID intentionally keeps
        # its released grouped flooring/DTM behavior and may under-fill.
        budget_ok = (
            outer_tokens == target_floor
            if method == "visionzip"
            else outer_tokens <= target_floor
        )
    elif spec.budget_contract == "post_prune":
        audited_tokens = inner_tokens
        # FastV is a direct global Top-K at the pruning layer, so unlike
        # grouping-based adapters it must realize the requested floor exactly.
        budget_ok = inner_tokens == target_floor
    else:
        audited_tokens = layer_average
        budget_ok = layer_average <= target_float + 1e-6

    tflops = llm_tflops(outer_tokens, dense_layers) + llm_tflops(
        inner_tokens, layers - dense_layers
    )
    return {
        "budget_contract": spec.budget_contract,
        "nominal_retention_ratio": 1.0 if method == "vanilla" else retention_ratio,
        "raw_visual_tokens": raw_tokens,
        "outer_visual_tokens": outer_tokens,
        "inner_visual_tokens": inner_tokens,
        "layer_average_visual_tokens": layer_average,
        "layer_average_retention_ratio": layer_average / max(1, raw_tokens),
        "audited_visual_tokens": audited_tokens,
        "audited_retention_ratio": audited_tokens / max(1, raw_tokens),
        "strict_floor_budget": raw_tokens if method == "vanilla" else target_floor,
        "budget_ok": bool(budget_ok),
        "tflops": tflops,
    }


@torch.inference_mode()
def timed_generate(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_tensors: list[torch.Tensor],
    image_sizes: list[tuple[int, int]],
    method: str,
    retention_ratio: float,
    max_new_tokens: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    timer = StageTimer(model, accelerated=method != "vanilla")
    timer.install()
    try:
        output_ids = model.generate(
            input_ids.clone(),
            attention_mask=attention_mask.clone(),
            images=[tensor.clone() for tensor in image_tensors],
            image_sizes=image_sizes,
            modalities=["video"],
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
        times = timer.read()
        if method == "vanilla":
            # LLaVA-OneVision appends one global newline to the visual span.
            raw_tokens = outer_tokens = inner_tokens = 32 * 196 + 1
        else:
            config = model.flashvid_config
            raw_patch_tokens = int(
                timer.raw_visual_tokens
                or getattr(config, "last_adapter_raw_tokens", 0)
            )
            outer_patch_tokens = int(
                timer.outer_visual_tokens
                or getattr(config, "last_adapter_output_tokens", 0)
            )
            outer_tokens = int(timer.llm_outer_visual_tokens or outer_patch_tokens)
            raw_tokens = raw_patch_tokens + max(0, outer_tokens - outer_patch_tokens)
            inner_tokens = int(getattr(config, "llm_token_length", 0) or outer_tokens)
        if min(raw_tokens, outer_tokens, inner_tokens) < 0 or raw_tokens <= 0:
            raise RuntimeError(
                f"invalid token audit raw={raw_tokens} outer={outer_tokens} inner={inner_tokens}"
            )
        audit = token_audit(
            method, retention_ratio, raw_tokens, outer_tokens, inner_tokens
        )
        if not audit["budget_ok"]:
            raise RuntimeError(
                f"strict {retention_ratio:.4%} token budget failed: {audit}"
            )
        vision = times["vision_encoding"]
        compression = times["compression"]
        llm = times["llm_forward"]
        phase_metrics = {
            f"{name}_ms": value
            for name, value in times.items()
            if name.startswith(CERTVID_PHASE_PREFIX)
        }
        return (
            {
                "vision_encoding_ms": vision,
                "compression_ms": compression,
                "llm_forward_ms": llm,
                "prefilling_total_ms": compression + llm,
                "ttft_ms": vision + compression + llm,
                **phase_metrics,
                **audit,
            },
            output_ids,
        )
    finally:
        timer.close()


@torch.inference_mode()
def warmup_generate(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_tensors: list[torch.Tensor],
    image_sizes: list[tuple[int, int]],
    max_new_tokens: int,
) -> None:
    model.generate(
        input_ids.clone(),
        attention_mask=attention_mask.clone(),
        images=[tensor.clone() for tensor in image_tensors],
        image_sizes=image_sizes,
        modalities=["video"],
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )
    torch.cuda.synchronize()


def run_method(args: argparse.Namespace) -> None:
    method = args.method.lower()
    if method not in METHOD_ORDER:
        raise ValueError(f"unsupported method {method!r}")
    manifest = _read_jsonl(Path(args.manifest))
    if not manifest:
        raise ValueError(f"empty manifest: {args.manifest}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "efficiency protocol requires exactly one visible GPU; "
            f"found {torch.cuda.device_count()}"
        )
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if args.require_gpu_name.lower() not in gpu_name.lower():
        raise RuntimeError(
            f"expected GPU containing {args.require_gpu_name!r}, found {gpu_name!r}"
        )

    retention_ratio = 1.0 if method == "vanilla" else float(args.retention_ratio)
    if not 0.0 < retention_ratio <= 1.0:
        raise ValueError(f"invalid retention ratio: {retention_ratio}")

    tokenizer, model, image_processor = get_model(args)
    model = apply_method(model, method, retention_ratio)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    video_root = Path(args.video_root)
    phase_records: list[dict[str, float]] = []
    phase_compression_totals: list[float] = []

    print(
        f"[setup] method={method} samples={len(manifest)} "
        f"retention={retention_ratio:.4%} "
        f"warmup={args.num_warmup} repeats={args.num_repeats} "
        f"gpu={gpu_name} memory={gpu_memory_gb:.1f}GiB output={output}"
    )
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for sample_index, sample in enumerate(manifest):
            qid = str(sample.get("efficiency_question_id") or _question_id(sample))
            video_id = str(sample["videoID"])
            video_path = resolve_video_path(video_id, video_root)

            # Decode and preprocess once per sample, before any CUDA timer.
            frames = load_video(str(video_path), args.num_frames)
            frame_sha256 = frame_content_sha256(frames)
            prepared = prepare_inputs(
                tokenizer,
                image_processor,
                frames,
                str(sample["input"]),
            )
            input_ids, attention_mask, image_tensors, image_sizes = prepared
            for _ in range(args.num_warmup):
                warmup_generate(
                    model,
                    input_ids,
                    attention_mask,
                    image_tensors,
                    image_sizes,
                    args.max_new_tokens,
                )

            pred_answer = ""
            for repeat_index in range(args.num_repeats):
                metrics, output_ids = timed_generate(
                    model,
                    tokenizer,
                    input_ids,
                    attention_mask,
                    image_tensors,
                    image_sizes,
                    method,
                    retention_ratio,
                    args.max_new_tokens,
                )
                if repeat_index == 0:
                    pred_answer = decode_generated_output(
                        tokenizer, output_ids, int(input_ids.shape[1])
                    )
                record = {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "sample_index": int(sample.get("efficiency_sample_index", sample_index)),
                    "repeat_index": repeat_index,
                    "question_id": qid,
                    "videoID": video_id,
                    "duration": str(sample.get("efficiency_duration") or _duration(sample)),
                    "answer": str(sample.get("answer") or "").strip().upper(),
                    "pred_answer": pred_answer,
                    "correct": pred_answer == str(sample.get("answer") or "").strip().upper(),
                    "num_frames": args.num_frames,
                    "sampling_protocol": SAMPLING_PROTOCOL,
                    "frame_sha256": frame_sha256,
                    "gpu_name": gpu_name,
                    "gpu_memory_gb": gpu_memory_gb,
                    "error": None,
                    **metrics,
                }
                phase_record = {
                    key: float(value)
                    for key, value in metrics.items()
                    if key.startswith(CERTVID_PHASE_PREFIX)
                }
                if phase_record:
                    phase_records.append(phase_record)
                    phase_compression_totals.append(float(metrics["compression_ms"]))
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"[{method}] {sample_index + 1}/{len(manifest)} "
                    f"repeat={repeat_index + 1}/{args.num_repeats} qid={qid} "
                    f"prefill={record['prefilling_total_ms']:.2f}ms "
                    f"ttft={record['ttft_ms']:.2f}ms "
                    f"tokens={record['outer_visual_tokens']}->{record['inner_visual_tokens']}"
                )

            del frames, input_ids, attention_mask, image_tensors
            torch.cuda.empty_cache()

    if phase_records:
        phase_keys = tuple(sorted(phase_records[0]))
        if any(tuple(sorted(record)) != phase_keys for record in phase_records):
            raise RuntimeError("CertVID phase profiler returned inconsistent fields")
        summary = {
            "method": method,
            "num_records": len(phase_records),
            "phases": {
                key.removeprefix(CERTVID_PHASE_PREFIX).removesuffix("_ms"): _stats(
                    record[key] for record in phase_records
                )
                for key in phase_keys
            },
        }
        phase_sum = statistics.fmean(
            sum(record[key] for key in phase_keys) for record in phase_records
        )
        compression_stats = _stats(phase_compression_totals)
        summary["mean_profiled_phase_sum_ms"] = phase_sum
        summary["compression_total_ms"] = compression_stats
        summary["mean_unprofiled_gap_ms"] = float(compression_stats["mean"]) - phase_sum
        summary["phase_share_of_compression"] = {
            name: float(stats["mean"]) / max(1e-9, float(compression_stats["mean"]))
            for name, stats in summary["phases"].items()
        }
        profile_path = output.with_suffix(".profile.json")
        _write_json(profile_path, summary)
        print("[profile] CertVID phase means (ms):")
        for name, stats in summary["phases"].items():
            share = 100.0 * summary["phase_share_of_compression"][name]
            print(f"  {name}: {stats['mean']:.3f} ({share:.1f}%)")
        print(f"  profiled_phase_sum: {phase_sum:.3f}")
        print(f"  compression_total: {compression_stats['mean']:.3f}")
        print(f"  unprofiled_gap: {summary['mean_unprofiled_gap_ms']:.3f}")
        print(f"[profile] wrote {profile_path}")


def _stats(values: Iterable[float]) -> dict[str, float | int]:
    items = [float(value) for value in values]
    if not items:
        raise ValueError("cannot summarize an empty value list")
    return {
        "count": len(items),
        "mean": statistics.fmean(items),
        "std": statistics.pstdev(items),
        "median": statistics.median(items),
    }


def _sample_means(records: list[dict[str, Any]], key: str) -> list[float]:
    values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        values[str(record["question_id"])].append(float(record[key]))
    return [statistics.fmean(group) for group in values.values()]


def _format_ms(mean: float, speedup: float) -> str:
    return f"{mean:.1f} ({speedup:.1f}x)"


def summarize(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = json.loads(Path(args.score_file).read_text(encoding="utf-8"))

    manifest = _read_jsonl(Path(args.manifest))
    manifest_ids = [str(row.get("efficiency_question_id") or _question_id(row)) for row in manifest]
    expected_ids = set(manifest_ids)
    combined: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    observed_gpu_names: set[str] = set()
    reference_frame_hashes: dict[str, str] | None = None
    methods = tuple(args.methods)
    if not methods or methods[0] != "vanilla" or len(set(methods)) != len(methods):
        raise ValueError("summary methods must be unique and start with vanilla")

    for method in methods:
        path = input_dir / f"{method}.jsonl"
        records = _read_jsonl(path)
        if len(records) != len(manifest) * args.num_repeats:
            raise ValueError(
                f"{method}: expected {len(manifest) * args.num_repeats} records, "
                f"found {len(records)}"
            )
        if any(record.get("error") for record in records):
            raise ValueError(f"{method}: raw records contain errors")
        if any(
            record.get("sampling_protocol") != SAMPLING_PROTOCOL
            for record in records
        ):
            raise ValueError(f"{method}: stale or mismatched sampling protocol")
        counts: dict[str, int] = defaultdict(int)
        for record in records:
            counts[str(record["question_id"])] += 1
        if set(counts) != expected_ids or any(value != args.num_repeats for value in counts.values()):
            raise ValueError(f"{method}: question IDs/repeat counts differ from manifest")
        method_frame_hashes: dict[str, set[str]] = defaultdict(set)
        for record in records:
            frame_hash = str(record.get("frame_sha256") or "")
            if not frame_hash:
                raise ValueError(f"{method}: missing sampled-frame content hash")
            method_frame_hashes[str(record["question_id"])].add(frame_hash)
        if any(len(hashes) != 1 for hashes in method_frame_hashes.values()):
            raise ValueError(f"{method}: sampled frames changed between repeats")
        flattened_hashes = {
            question_id: next(iter(hashes))
            for question_id, hashes in method_frame_hashes.items()
        }
        if reference_frame_hashes is None:
            reference_frame_hashes = flattened_hashes
        elif flattened_hashes != reference_frame_hashes:
            raise ValueError(f"{method}: sampled frames differ from Vanilla")
        if any(not bool(record.get("budget_ok")) for record in records):
            raise ValueError(f"{method}: one or more token audits failed")
        observed_gpu_names.update(str(record.get("gpu_name") or "") for record in records)
        for record in records:
            expected_total = float(record["compression_ms"]) + float(record["llm_forward_ms"])
            if abs(expected_total - float(record["prefilling_total_ms"])) > 1e-4:
                raise ValueError(f"{method}: invalid prefilling total")
        combined.extend(records)

        metric_keys = (
            "vision_encoding_ms",
            "compression_ms",
            "llm_forward_ms",
            "prefilling_total_ms",
            "ttft_ms",
            "tflops",
            "raw_visual_tokens",
            "outer_visual_tokens",
            "inner_visual_tokens",
            "layer_average_visual_tokens",
            "layer_average_retention_ratio",
            "audited_retention_ratio",
        )
        summaries[method] = {
            "method_label": METHOD_LABELS[method],
            "num_samples": len(manifest),
            "num_repeats": args.num_repeats,
            "num_records": len(records),
            "duration_counts": {
                duration: len(
                    {
                        record["question_id"]
                        for record in records
                        if record["duration"] == duration
                    }
                )
                for duration in DURATION_ORDER
            },
            "budget_contract": records[0]["budget_contract"],
            "retention_ratio": float(records[0]["nominal_retention_ratio"]),
            "metrics": {
                key: _stats(_sample_means(records, key)) for key in metric_keys
            },
            "token_audit": {
                "all_passed": True,
                "max_audited_retention_ratio": max(
                    float(record["audited_retention_ratio"]) for record in records
                ),
                "max_layer_average_retention_ratio": max(
                    float(record["layer_average_retention_ratio"]) for record in records
                ),
            },
        }

    vanilla_prefill = summaries["vanilla"]["metrics"]["prefilling_total_ms"]["mean"]
    vanilla_ttft = summaries["vanilla"]["metrics"]["ttft_ms"]["mean"]
    table_rows: list[dict[str, Any]] = []
    for method in methods:
        metrics = summaries[method]["metrics"]
        prefill = metrics["prefilling_total_ms"]["mean"]
        ttft = metrics["ttft_ms"]["mean"]
        row = {
            "method": METHOD_LABELS[method],
            "retention_ratio": (
                "100%"
                if method == "vanilla"
                else f"{100.0 * summaries[method]['retention_ratio']:g}%"
            ),
            "tflops": metrics["tflops"]["mean"],
            "vision_encoding_ms": metrics["vision_encoding_ms"]["mean"],
            "compression_ms": metrics["compression_ms"]["mean"],
            "llm_forward_ms": metrics["llm_forward_ms"]["mean"],
            "prefilling_total_ms": prefill,
            "prefilling_speedup": vanilla_prefill / prefill,
            "ttft_ms": ttft,
            "ttft_speedup": vanilla_ttft / ttft,
            "avg_score": float(scores[method]["avg_score"]),
            "rel_acc": float(scores[method]["rel_acc"]),
        }
        table_rows.append(row)
        summaries[method]["speedup"] = {
            "prefilling": row["prefilling_speedup"],
            "ttft": row["ttft_speedup"],
        }

    samples_path = output_dir / "efficiency_samples.jsonl"
    with samples_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in combined:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_payload = {
        "protocol": {
            "model": "LLaVA-OneVision-7B",
            "gpu": "NVIDIA GeForce RTX 5090 32GB",
            "retention_ratios": {
                method: summaries[method]["retention_ratio"] for method in methods
            },
            "num_frames": 32,
            "num_samples": len(manifest),
            "num_warmup": args.num_warmup,
            "num_repeats": args.num_repeats,
            "fastv_budget_note": (
                "FastV applies 1% after layer 2; its first two dense layers are "
                "included in TFLOPs and layer-average diagnostics."
            ),
            "timing_excludes": ["model loading", "video decoding", "input preprocessing"],
            "ttft_definition": "vision encoding + compression + first LLM prefill forward",
            "scores_source": str(Path(args.score_file).resolve()),
            "observed_gpu_names": sorted(name for name in observed_gpu_names if name),
        },
        "methods": summaries,
        "table": table_rows,
    }
    _write_json(output_dir / "efficiency_summary.json", summary_payload)

    csv_fields = list(table_rows[0].keys())
    with (output_dir / "efficiency_table.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(table_rows)

    headers = (
        "Method",
        "R",
        "TFLOPs",
        "Vision Encoding",
        "Compression",
        "LLM Forward",
        "Prefilling Total",
        "TTFT",
        "Avg. Score",
        "Rel. Acc.",
    )
    md_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for index, row in enumerate(table_rows):
        is_vanilla = index == 0
        md_lines.append(
            "| "
            + " | ".join(
                [
                    row["method"],
                    row["retention_ratio"],
                    f"{row['tflops']:.1f}",
                    f"{row['vision_encoding_ms']:.1f}",
                    "--" if is_vanilla else f"{row['compression_ms']:.1f}",
                    f"{row['llm_forward_ms']:.1f}",
                    _format_ms(row["prefilling_total_ms"], row["prefilling_speedup"]),
                    _format_ms(row["ttft_ms"], row["ttft_speedup"]),
                    f"{row['avg_score']:.1f}",
                    f"{row['rel_acc']:.1f}",
                ]
            )
            + " |"
        )
    (output_dir / "efficiency_table.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )

    tex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{Efficiency on LLaVA-OneVision. Times are averaged over {len(manifest)} VideoMME videos on an NVIDIA GeForce RTX 5090 32GB GPU.}}",
        r"\label{tab:efficiency}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccccccc}",
        r"\toprule",
        r"Method & $R$ & TFLOPs & Vision Enc. & Compression & LLM Forward & Prefill Total & TTFT & Avg. Score & Rel. Acc. (\%) \\",
        r"\midrule",
    ]
    for index, row in enumerate(table_rows):
        is_vanilla = index == 0
        tex_lines.append(
            " & ".join(
                [
                    row["method"],
                    row["retention_ratio"].replace("%", r"\%"),
                    f"{row['tflops']:.1f}",
                    f"{row['vision_encoding_ms']:.1f}",
                    "--" if is_vanilla else f"{row['compression_ms']:.1f}",
                    f"{row['llm_forward_ms']:.1f}",
                    _format_ms(row["prefilling_total_ms"], row["prefilling_speedup"]).replace("x", r"$\times$"),
                    _format_ms(row["ttft_ms"], row["ttft_speedup"]).replace("x", r"$\times$"),
                    f"{row['avg_score']:.1f}",
                    f"{row['rel_acc']:.1f}",
                ]
            )
            + r" \\"
        )
    tex_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
        ]
    )
    (output_dir / "efficiency_table.tex").write_text(
        "\n".join(tex_lines) + "\n", encoding="utf-8"
    )

    print("\n".join(md_lines))
    print(f"\nWrote efficiency artifacts to {output_dir.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest", help="build fixed VideoMME manifest")
    manifest.add_argument("--dataset-jsonl", required=True)
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--sample-count", type=int, default=100)
    manifest.add_argument("--seed", type=int, default=20260813)
    manifest.set_defaults(func=build_manifest)

    run = subparsers.add_parser("run", help="benchmark one method")
    run.add_argument("--method", choices=METHOD_ORDER, required=True)
    run.add_argument("--model-path", required=True)
    run.add_argument("--model-name", default="llava_qwen")
    run.add_argument("--manifest", required=True)
    run.add_argument("--video-root", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--retention-ratio", type=float, default=0.01)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--mm-spatial-pool-mode", default="bilinear")
    run.add_argument("--num-frames", type=int, default=32)
    run.add_argument("--num-warmup", type=int, default=1)
    run.add_argument("--num-repeats", type=int, default=3)
    run.add_argument("--max-new-tokens", type=int, default=16)
    run.add_argument("--require-gpu-name", default="RTX 5090")
    run.set_defaults(func=run_method)

    report = subparsers.add_parser("summarize", help="validate and render tables")
    report.add_argument("--input-dir", required=True)
    report.add_argument("--output-dir", required=True)
    report.add_argument("--manifest", required=True)
    report.add_argument("--score-file", default=str(DEFAULT_SCORE_FILE))
    report.add_argument("--num-warmup", type=int, default=1)
    report.add_argument("--num-repeats", type=int, default=3)
    report.add_argument(
        "--methods", nargs="+", choices=METHOD_ORDER, default=list(METHOD_ORDER)
    )
    report.set_defaults(func=summarize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
