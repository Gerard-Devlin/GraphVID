#!/usr/bin/env python3
"""Compare vanilla visual attention with CertVID V3 on two videos.

The script samples videos directly from a dataset directory, runs the real
LLaVA-OneVision path with and without CertVID V3, and exports separate token
maps plus question/answer metadata. It intentionally does not depend on
lmms-eval result files or precomputed diagnostics.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image, ImageDraw

from flashvid import flashvid
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.model.builder import load_pretrained_model


VIDEO_SUFFIXES = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".MP4"}
OPTION_PATTERN = re.compile(r"^\s*([A-E])\s*[.)]\s*(.+?)\s*$")


@dataclass(frozen=True)
class QuestionRecord:
    question_id: str | None
    prompt: str
    question: str
    options: tuple[tuple[str, str], ...]
    answer: str | None
    answer_text: str | None


@dataclass
class Example:
    video_path: Path
    question_record: QuestionRecord
    frame: Image.Image
    attention_overlay: Image.Image
    certvid_overlay: Image.Image
    sampled_frame_index: int
    source_frame_index: int
    timestamp_seconds: float | None
    grid_height: int
    grid_width: int
    attention_selected_in_frame: list[int]
    selected_in_frame: list[int]
    per_frame_counts: list[int]
    anchor_indices: list[int]
    raw_token_count: int
    output_token_count: int
    without_certvid_prediction: str
    without_certvid_answer: str | None
    without_certvid_correct: bool | None
    certvid_prediction: str
    certvid_answer: str | None
    certvid_correct: bool | None


def parse_args() -> argparse.Namespace:
    hf_home = Path(os.environ.get("HF_HOME", "/home/xuyouwen/hf_home_local"))
    parser = argparse.ArgumentParser(
        description="Randomly sample dataset videos and visualize real CertVID V3 anchors."
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get(
            "PRETRAINED", "/home/xuyouwen/models/llava-onevision-qwen2-7b-ov"
        ),
        help="Local LLaVA-OneVision model or resolved Hugging Face snapshot.",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(hf_home / "videomme" / "data"),
        help="Directory recursively containing dataset videos.",
    )
    parser.add_argument(
        "--metadata-jsonl",
        default="assets/videomme.jsonl",
        help="Optional JSONL with video IDs and questions; pass an empty string to disable.",
    )
    parser.add_argument(
        "--video-id",
        default="",
        help=(
            "Optional comma-separated video stems/q_uids to visualize instead "
            "of random sampling. The count must match --num-examples."
        ),
    )
    parser.add_argument("--output-dir", default="logs/visualizations/certvid_examples")
    parser.add_argument("--num-examples", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--retention-ratio", type=float, default=0.10)
    parser.add_argument("--expansion", type=float, default=1.30)
    parser.add_argument("--pruning-layer", type=int, default=20)
    parser.add_argument("--llm-retention-ratio", type=float, default=0.1923076923)
    parser.add_argument("--pool-mode", choices=("bilinear", "average", "max"), default="bilinear")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--selection-mode",
        choices=("improvement", "any"),
        default="improvement",
        help=(
            "improvement keeps only cases where full-token inference is wrong "
            "and CertVID V3 is correct"
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=20,
        help="Maximum videos attempted while skipping corrupt or incompatible files.",
    )
    return parser.parse_args()


def discover_videos(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    videos = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {suffix.lower() for suffix in VIDEO_SUFFIXES}
    )
    if not videos:
        raise FileNotFoundError(f"no videos found under: {root}")
    return videos


def _first_present(record: dict[str, Any], keys: Iterable[str]) -> str | None:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def display_question(question: str) -> str:
    lines = [line.strip() for line in question.splitlines() if line.strip()]
    candidates = [
        line
        for line in lines
        if not line.startswith(("A.", "B.", "C.", "D.", "E."))
        and "answer with" not in line.lower()
        and "select the best answer" not in line.lower()
    ]
    interrogatives = [line for line in candidates if "?" in line]
    text = interrogatives[-1] if interrogatives else (candidates[-1] if candidates else question)
    return " ".join(text.split())


def _parse_options(record: dict[str, Any], prompt: str) -> tuple[tuple[str, str], ...]:
    raw_options = record.get("options")
    parsed: list[tuple[str, str]] = []
    if isinstance(raw_options, dict):
        for label, text in raw_options.items():
            parsed.append((str(label).strip().upper(), str(text).strip()))
    elif isinstance(raw_options, (list, tuple)):
        for index, value in enumerate(raw_options):
            text = str(value).strip()
            match = OPTION_PATTERN.match(text)
            if match:
                parsed.append((match.group(1), match.group(2)))
            else:
                parsed.append((chr(ord("A") + index), text))
    else:
        for line in prompt.splitlines():
            match = OPTION_PATTERN.match(line)
            if match:
                parsed.append((match.group(1), match.group(2)))
    return tuple(parsed)


def _normalize_answer(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip().upper()
    if text.isdigit() and 0 <= int(text) < 5:
        return chr(ord("A") + int(text))
    match = re.search(r"[A-E]", text)
    return match.group(0) if match else None


def _question_record(
    record: dict[str, Any],
    fallback_question_id: str | None = None,
) -> QuestionRecord | None:
    prompt = _first_present(record, ("input", "prompt", "query", "question"))
    if prompt is None:
        return None
    question = _first_present(record, ("question",)) or display_question(prompt)
    options = _parse_options(record, prompt)
    answer_raw = _first_present(record, ("answer", "target", "label"))
    answer = _normalize_answer(answer_raw)
    answer_lookup = dict(options)
    return QuestionRecord(
        question_id=(
            _first_present(record, ("question_id", "questionID", "qid", "id"))
            or fallback_question_id
        ),
        prompt=prompt,
        question=question,
        options=options,
        answer=answer,
        answer_text=answer_lookup.get(answer) if answer else None,
    )


def load_questions(path: Path | None) -> dict[str, list[QuestionRecord]]:
    questions: dict[str, list[QuestionRecord]] = {}
    if path is None or not path.is_file():
        return questions
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            video_id = _first_present(
                record,
                ("videoID", "video_id", "video_idx", "video", "video_name"),
            )
            if video_id is None and isinstance(record.get("submission"), dict):
                video_id = next(iter(record["submission"]), None)
            question_record = _question_record(record, video_id)
            if video_id and question_record is not None:
                questions.setdefault(Path(video_id).stem, []).append(question_record)
    return questions


def sample_video(path: Path, num_frames: int) -> tuple[np.ndarray, np.ndarray, float | None]:
    reader = VideoReader(str(path), ctx=cpu(0))
    total_frames = len(reader)
    if total_frames <= 0:
        raise ValueError("video has no decodable frames")
    source_indices = np.linspace(0, total_frames - 1, num_frames, dtype=np.int64)
    frames = reader.get_batch(source_indices.tolist()).asnumpy()
    try:
        fps = float(reader.get_avg_fps())
        if not np.isfinite(fps) or fps <= 0.0:
            fps = None
    except (AttributeError, TypeError, ValueError, RuntimeError):
        fps = None
    return frames, source_indices, fps


def tensor_frames_to_pil(pixel_values: torch.Tensor, image_processor) -> list[Image.Image]:
    values = pixel_values.detach().float().cpu()
    mean = torch.tensor(
        getattr(image_processor, "image_mean", [0.5, 0.5, 0.5]), dtype=torch.float32
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        getattr(image_processor, "image_std", [0.5, 0.5, 0.5]), dtype=torch.float32
    ).view(1, 3, 1, 1)
    values = (values * std + mean).clamp(0.0, 1.0)
    arrays = (values.permute(0, 2, 3, 1).numpy() * 255.0).round().astype(np.uint8)
    return [Image.fromarray(array, mode="RGB") for array in arrays]


def factor_grid(tokens_per_frame: int) -> tuple[int, int]:
    height = max(1, int(math.sqrt(tokens_per_frame)))
    while height > 1 and tokens_per_frame % height:
        height -= 1
    return height, tokens_per_frame // height


def overlay_selection(
    frame: Image.Image,
    selected_local: list[int],
    grid_height: int,
    grid_width: int,
) -> Image.Image:
    base = frame.convert("RGBA")
    width, height = base.size
    selected = set(selected_local)

    mask = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask)
    for row in range(grid_height):
        y0 = round(row * height / grid_height)
        y1 = round((row + 1) * height / grid_height)
        for col in range(grid_width):
            x0 = round(col * width / grid_width)
            x1 = round((col + 1) * width / grid_width)
            token = row * grid_width + col
            if token in selected:
                draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=(0, 175, 176, 34))
                line_width = max(1, round(min(width, height) / 240))
                draw.rectangle(
                    (x0, y0, x1 - 1, y1 - 1),
                    outline=(0, 185, 186, 255),
                    width=line_width,
                )
            else:
                draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=(28, 32, 37, 150))
    return Image.alpha_composite(base, mask).convert("RGB")


def prepare_prompt(tokenizer, question: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    conv = copy.deepcopy(conv_templates["qwen_1_5"])
    conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{question}")
    conv.append_message(conv.roles[1], None)
    input_ids = tokenizer_image_token(
        conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    return input_ids, torch.ones_like(input_ids)


def extract_answer_label(text: str, valid_labels: Iterable[str]) -> str | None:
    labels = "".join(
        label for label in (str(item).strip().upper()[:1] for item in valid_labels) if label
    )
    labels = "".join(dict.fromkeys(labels)) or "ABCD"
    choice_class = re.escape(labels)
    patterns = (
        rf"^\s*[\(\[]?\s*([{choice_class}])\s*[\)\].,:;!?]?\s*$",
        rf"\b(?:ANSWER|OPTION|CHOICE)\b\s*(?:IS)?\s*[:=\-]?\s*[\(\[]?\s*([{choice_class}])\b",
        rf"\b(?:THE\s+ANSWER\s+IS|I\s+CHOOSE|I\s+PICK)\b\s*[:=\-]?\s*[\(\[]?\s*([{choice_class}])\b",
        rf"[\(\[]\s*([{choice_class}])\s*[\)\]]",
        rf"\b([{choice_class}])\s*\.",
        rf"\b([{choice_class}])\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def load_certvid_model(args: argparse.Namespace):
    model_path = str(Path(args.model_path).expanduser().resolve())
    if not Path(model_path, "config.json").is_file():
        raise FileNotFoundError(f"model config not found: {model_path}/config.json")
    # A resolved Hugging Face snapshot ends in a commit hash, so deriving the
    # architecture from the directory name incorrectly routes OneVision to
    # LlavaLlamaForCausalLM. This visualizer intentionally supports the
    # LLaVA-OneVision Qwen2 checkpoint only and therefore uses its canonical
    # builder name explicitly.
    model_name = "llava-onevision-qwen2-7b-ov"
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        None,
        model_name,
        device_map="auto",
        attn_implementation="flash_attention_2",
        overwrite_config={
            "mm_spatial_pool_stride": 2,
            "mm_spatial_pool_mode": args.pool_mode,
        },
        multimodal=True,
    )
    if model.__class__.__name__ != "LlavaQwenForCausalLM":
        raise RuntimeError(
            "expected LlavaQwenForCausalLM for LLaVA-OneVision, "
            f"but loaded {model.__class__.__name__}"
        )
    model.eval()
    model.config.mm_spatial_pool_stride = 2
    model.config.mm_spatial_pool_mode = args.pool_mode
    model = flashvid(
        model,
        retention_ratio=args.retention_ratio,
        expansion=args.expansion,
        pruning_layer=args.pruning_layer,
        llm_retention_ratio=args.llm_retention_ratio,
        compression_variant="certvid_v3",
        certv3_budget_uses_expansion=True,
        do_segment=True,
        segment_threshold=0.9,
        min_segment_num=8,
        complementary_segment=True,
        token_selection_method="attn_div_stable",
        alpha=0.70,
        temporal_threshold=0.8,
    )
    setattr(model.flashvid_config, "_capture_visualization_attention", True)
    device = next(model.parameters()).device
    return tokenizer, model, image_processor, device


def generate_once(
    *,
    model,
    tokenizer,
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    retention_ratio: float,
    expansion: float,
    llm_retention_ratio: float,
    max_new_tokens: int,
) -> tuple[str, torch.Tensor, Any]:
    config = model.flashvid_config
    config.retention_ratio = retention_ratio
    config.expansion = expansion
    config.llm_retention_ratio = llm_retention_ratio
    setattr(config, "_certvid_plan", None)
    setattr(config, "_visualization_cls_attention", None)
    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    with torch.inference_mode():
        generated = model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
            images=[pixel_values],
            modalities=["video"],
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    prediction = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
    cls_attention = getattr(config, "_visualization_cls_attention", None)
    if not torch.is_tensor(cls_attention) or cls_attention.ndim != 2:
        raise RuntimeError("visual hook did not publish pre-compression attention")
    plan = getattr(config, "_certvid_plan", None)
    return prediction, cls_attention.detach().float().cpu().clone(), plan


def run_one_example(
    path: Path,
    question_record: QuestionRecord,
    tokenizer,
    model,
    image_processor,
    device: torch.device,
    args: argparse.Namespace,
) -> Example:
    frames, source_indices, fps = sample_video(path, args.num_frames)
    pixel_values_cpu = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
    display_frames = tensor_frames_to_pil(pixel_values_cpu, image_processor)
    pixel_values = pixel_values_cpu.to(device=device, dtype=torch.float16)
    input_ids, attention_mask = prepare_prompt(tokenizer, question_record.prompt, device)
    valid_labels = [label for label, _ in question_record.options]
    target = question_record.answer

    print(f"[inference] {path.name}: full-token start", flush=True)
    without_prediction, without_attention, _ = generate_once(
        model=model,
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        retention_ratio=1.0,
        expansion=1.0,
        llm_retention_ratio=1.0,
        max_new_tokens=args.max_new_tokens,
    )
    without_answer = extract_answer_label(without_prediction, valid_labels)
    print(
        f"[inference] {path.name}: full-token done "
        f"prediction={without_answer!r} target={target!r}",
        flush=True,
    )
    print(f"[inference] {path.name}: CertVID V3 start", flush=True)
    certvid_prediction, cls_attention, plan = generate_once(
        model=model,
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        retention_ratio=args.retention_ratio,
        expansion=args.expansion,
        llm_retention_ratio=args.llm_retention_ratio,
        max_new_tokens=args.max_new_tokens,
    )
    certvid_answer = extract_answer_label(certvid_prediction, valid_labels)
    print(
        f"[inference] {path.name}: CertVID V3 done "
        f"prediction={certvid_answer!r} target={target!r}",
        flush=True,
    )
    del pixel_values, input_ids, attention_mask

    if plan is None:
        raise RuntimeError("CertVID did not publish an anchor plan")
    anchors = plan.anchor_indices.detach().long().cpu().tolist()
    raw_token_count = int(plan.raw_token_count)
    frame_count = len(display_frames)
    if raw_token_count % frame_count:
        raise RuntimeError(
            f"raw token count {raw_token_count} is not divisible by {frame_count} frames"
        )
    tokens_per_frame = raw_token_count // frame_count
    if tuple(cls_attention.shape) != (frame_count, tokens_per_frame):
        raise RuntimeError(
            "pre-compression attention shape mismatch: "
            f"expected {(frame_count, tokens_per_frame)}, got {tuple(cls_attention.shape)}"
        )
    if tuple(without_attention.shape) != (frame_count, tokens_per_frame):
        raise RuntimeError(
            "uncompressed attention shape mismatch: "
            f"expected {(frame_count, tokens_per_frame)}, got {tuple(without_attention.shape)}"
        )
    grid_height, grid_width = factor_grid(tokens_per_frame)

    per_frame: list[list[int]] = [[] for _ in range(frame_count)]
    for anchor in anchors:
        frame_index, local_index = divmod(int(anchor), tokens_per_frame)
        if 0 <= frame_index < frame_count:
            per_frame[frame_index].append(local_index)
    counts = [len(indices) for indices in per_frame]

    attention_budget = min(len(anchors), int(without_attention.numel()))
    attention_global = (
        torch.argsort(
            without_attention.reshape(-1).float(),
            descending=True,
            stable=True,
        )[:attention_budget]
        .sort()
        .values.tolist()
    )
    attention_per_frame: list[list[int]] = [[] for _ in range(frame_count)]
    for global_index in attention_global:
        frame_index, local_index = divmod(int(global_index), tokens_per_frame)
        attention_per_frame[frame_index].append(local_index)

    # Show the frame where CertVID contributes the most evidence that pure
    # global attention Top-K would omit. Ties favor richer, central frames.
    representative = max(
        range(frame_count),
        key=lambda index: (
            len(set(per_frame[index]) - set(attention_per_frame[index])),
            len(per_frame[index]),
            -abs(index - (frame_count - 1) / 2.0),
            -index,
        ),
    )
    selected_local = sorted(per_frame[representative])
    attention_selected = sorted(attention_per_frame[representative])
    frame = display_frames[representative]
    attention_overlay = overlay_selection(
        frame,
        attention_selected,
        grid_height,
        grid_width,
    )
    certvid_overlay = overlay_selection(
        frame,
        selected_local,
        grid_height,
        grid_width,
    )
    timestamp = (
        float(source_indices[representative]) / fps if fps is not None else None
    )
    return Example(
        video_path=path,
        question_record=question_record,
        frame=frame,
        attention_overlay=attention_overlay,
        certvid_overlay=certvid_overlay,
        sampled_frame_index=representative,
        source_frame_index=int(source_indices[representative]),
        timestamp_seconds=timestamp,
        grid_height=grid_height,
        grid_width=grid_width,
        attention_selected_in_frame=[int(index) for index in attention_selected],
        selected_in_frame=selected_local,
        per_frame_counts=counts,
        anchor_indices=[int(index) for index in anchors],
        raw_token_count=raw_token_count,
        output_token_count=len(anchors),
        without_certvid_prediction=without_prediction,
        without_certvid_answer=without_answer,
        without_certvid_correct=(without_answer == target) if target else None,
        certvid_prediction=certvid_prediction,
        certvid_answer=certvid_answer,
        certvid_correct=(certvid_answer == target) if target else None,
    )


def save_image(image: Image.Image, output_path: Path) -> None:
    """Save the visualization directly without padding, borders, or labels."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path, format="PNG")


def example_metadata(
    example: Example,
    example_number: int,
    attention_name: str,
    certvid_name: str,
) -> dict[str, Any]:
    question = example.question_record
    attention_tokens = set(example.attention_selected_in_frame)
    certvid_tokens = set(example.selected_in_frame)
    return {
        "example_id": example_number,
        "video_id": example.video_path.stem,
        "video_file": example.video_path.name,
        "question_id": question.question_id,
        "question": question.question,
        "options": [
            {"label": label, "text": text} for label, text in question.options
        ],
        "answer": {
            "label": question.answer,
            "text": question.answer_text,
        },
        "model_outputs": {
            "without_certvid_v3": {
                "raw_prediction": example.without_certvid_prediction,
                "predicted_answer": example.without_certvid_answer,
                "correct": example.without_certvid_correct,
            },
            "with_certvid_v3": {
                "raw_prediction": example.certvid_prediction,
                "predicted_answer": example.certvid_answer,
                "correct": example.certvid_correct,
            },
        },
        "without_certvid_v3_correct": example.without_certvid_correct,
        "with_certvid_v3_correct": example.certvid_correct,
        "images": {
            "attention_without_certvid_v3": attention_name,
            "with_certvid_v3": certvid_name,
        },
        "visualized_frame": {
            "sampled_index": example.sampled_frame_index,
            "source_index": example.source_frame_index,
            "timestamp_seconds": example.timestamp_seconds,
        },
        "certvid": {
            "raw_tokens": example.raw_token_count,
            "selected_tokens": example.output_token_count,
            "attention_topk_in_visualized_frame": len(
                example.attention_selected_in_frame
            ),
            "selected_tokens_in_visualized_frame": len(example.selected_in_frame),
            "overlap_with_attention_in_visualized_frame": len(
                attention_tokens & certvid_tokens
            ),
            "new_tokens_vs_attention_in_visualized_frame": len(
                certvid_tokens - attention_tokens
            ),
            "grid_height": example.grid_height,
            "grid_width": example.grid_width,
        },
    }


def main() -> None:
    args = parse_args()
    if args.num_examples <= 0:
        raise ValueError("num-examples must be positive")
    requested_video_ids = [
        Path(value.strip()).stem
        for value in args.video_id.split(",")
        if value.strip()
    ]
    if requested_video_ids and len(requested_video_ids) != args.num_examples:
        raise ValueError(
            "the number of comma-separated --video-id values must match "
            "--num-examples"
        )
    if len(requested_video_ids) != len(set(requested_video_ids)):
        raise ValueError("--video-id values must be unique")
    if args.num_frames <= 0:
        raise ValueError("num-frames must be positive")
    if not (0.0 < args.retention_ratio <= 1.0):
        raise ValueError("retention-ratio must be in (0, 1]")

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        png_count = 2 * args.num_examples
        raise RuntimeError(
            "output directory must be empty so it contains only "
            f"{png_count} PNG files and examples.json: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    metadata_path = (
        Path(args.metadata_jsonl).expanduser().resolve() if args.metadata_jsonl.strip() else None
    )
    questions = load_questions(metadata_path)
    if not questions:
        raise RuntimeError(
            "metadata JSONL is required to export questions, options, and answers"
        )
    candidates = discover_videos(dataset_root)
    rng = random.Random(args.seed)
    if requested_video_ids:
        paths_by_id = {path.stem: path for path in candidates}
        missing_ids = [
            video_id for video_id in requested_video_ids if video_id not in paths_by_id
        ]
        if missing_ids:
            raise FileNotFoundError(
                f"video IDs {missing_ids!r} were not found under {dataset_root}"
            )
        candidates = [paths_by_id[video_id] for video_id in requested_video_ids]
    else:
        rng.shuffle(candidates)

    print(f"[setup] dataset={dataset_root}")
    print(f"[setup] discovered_videos={len(candidates)} seed={args.seed}")
    print(f"[setup] metadata_videos={len(questions)}")
    print(f"[setup] output={output_dir}")
    tokenizer, model, image_processor, device = load_certvid_model(args)

    examples: list[Example] = []
    attempt_limit = min(len(candidates), max(args.max_attempts, args.num_examples))
    for attempt, path in enumerate(candidates[:attempt_limit], start=1):
        print(
            f"[try] {attempt}/{attempt_limit} video={path.name} "
            f"accepted={len(examples)}/{args.num_examples}",
            flush=True,
        )
        video_questions = questions.get(path.stem)
        if not video_questions:
            print(f"[skip] {path.name}: no matching question metadata", flush=True)
            continue
        question_record = rng.choice(video_questions)
        try:
            example = run_one_example(
                path,
                question_record,
                tokenizer,
                model,
                image_processor,
                device,
                args,
            )
        except Exception as exc:
            print(f"[skip] {path.name}: {exc}", flush=True)
            continue

        if args.selection_mode == "improvement" and not (
            example.without_certvid_correct is False
            and example.certvid_correct is True
        ):
            print(
                f"[reject] {path.name}: full={example.without_certvid_answer}/"
                f"{example.without_certvid_correct} certvid={example.certvid_answer}/"
                f"{example.certvid_correct}",
                flush=True,
            )
            continue

        examples.append(example)
        print(
            f"[ok] {len(examples)}/{args.num_examples} {path.name} "
            f"full={example.without_certvid_answer} certvid={example.certvid_answer} "
            f"target={example.question_record.answer} "
            f"anchors={example.output_token_count}/{example.raw_token_count}",
            flush=True,
        )
        if len(examples) >= args.num_examples:
            break

    metadata_examples: list[dict[str, Any]] = []
    for number, example in enumerate(examples, start=1):
        video_id = example.video_path.stem
        attention_name = (
            f"example_{number:02d}_{video_id}_without_certvid_v3.png"
        )
        certvid_name = f"example_{number:02d}_{video_id}_with_certvid_v3.png"
        save_image(example.attention_overlay, output_dir / attention_name)
        save_image(example.certvid_overlay, output_dir / certvid_name)
        metadata_examples.append(
            example_metadata(example, number, attention_name, certvid_name)
        )

    metadata = {
        "visualization": {
            "without_certvid_v3": (
                "Global Top-K pre-compression visual-attention tokens"
            ),
            "with_certvid_v3": "CertVID V3 anchors on the same displayed frame",
            "matched_global_token_budget": True,
            "frame_policy": (
                "Frame with the most CertVID anchors absent from global attention Top-K"
            ),
            "selection_mode": args.selection_mode,
            "requested_video_ids": requested_video_ids or None,
            "requested_examples": args.num_examples,
            "produced_examples": len(examples),
        },
        "examples": metadata_examples,
    }
    (output_dir / "examples.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[done] generated {2 * len(examples)} PNG files and examples.json "
        f"in {output_dir}"
    )
    if len(examples) < args.num_examples:
        raise RuntimeError(
            f"only produced {len(examples)}/{args.num_examples} examples; "
            "accepted examples were saved before this error"
        )


if __name__ == "__main__":
    main()
