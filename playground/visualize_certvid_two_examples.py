#!/usr/bin/env python3
"""Create paper-style qualitative visualizations of CertVID token selection.

The script samples videos directly from a dataset directory, runs the real
LLaVA-OneVision + CertVID V3 path once per video, and overlays the selected
outer anchors on a representative frame. It intentionally does not depend on
lmms-eval result files or precomputed diagnostics.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image, ImageDraw, ImageFont

from flashvid import flashvid
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.model.builder import load_pretrained_model


VIDEO_SUFFIXES = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".MP4"}
GENERIC_QUESTION = "What are the most important visual events in this video?"


@dataclass
class Example:
    video_path: Path
    question: str
    display_question: str
    frame: Image.Image
    overlay: Image.Image
    sampled_frame_index: int
    source_frame_index: int
    timestamp_seconds: float | None
    grid_height: int
    grid_width: int
    selected_in_frame: list[int]
    per_frame_counts: list[int]
    anchor_indices: list[int]
    raw_token_count: int
    output_token_count: int


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
    parser.add_argument("--output-dir", default="logs/visualizations/certvid_examples")
    parser.add_argument("--num-examples", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--retention-ratio", type=float, default=0.10)
    parser.add_argument("--expansion", type=float, default=1.30)
    parser.add_argument("--pruning-layer", type=int, default=20)
    parser.add_argument("--llm-retention-ratio", type=float, default=0.1923076923)
    parser.add_argument("--pool-mode", choices=("bilinear", "average", "max"), default="bilinear")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=50,
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


def load_questions(path: Path | None) -> dict[str, list[str]]:
    questions: dict[str, list[str]] = {}
    if path is None or not path.is_file():
        return questions
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            video_id = _first_present(
                record,
                ("videoID", "video_id", "video_idx", "video", "video_name", "id"),
            )
            question = _first_present(record, ("input", "question", "prompt", "query"))
            if video_id and question:
                questions.setdefault(Path(video_id).stem, []).append(question)
    return questions


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


def load_font(size: int, *, serif: bool = False, bold: bool = False) -> ImageFont.ImageFont:
    names = []
    if serif:
        names.extend(
            [
                "DejaVuSerif-Bold.ttf" if bold else "DejaVuSerif.ttf",
                "LiberationSerif-Bold.ttf" if bold else "LiberationSerif-Regular.ttf",
            ]
        )
    else:
        names.extend(
            [
                "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
                "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf",
            ]
        )
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


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
    device = next(model.parameters()).device
    return tokenizer, model, image_processor, device


def run_one_example(
    path: Path,
    question: str,
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
    input_ids, attention_mask = prepare_prompt(tokenizer, question, device)

    config = model.flashvid_config
    setattr(config, "_certvid_plan", None)
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
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
    del generated, pixel_values, input_ids, attention_mask

    plan = getattr(config, "_certvid_plan", None)
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
    grid_height, grid_width = factor_grid(tokens_per_frame)

    per_frame: list[list[int]] = [[] for _ in range(frame_count)]
    for anchor in anchors:
        frame_index, local_index = divmod(int(anchor), tokens_per_frame)
        if 0 <= frame_index < frame_count:
            per_frame[frame_index].append(local_index)
    counts = [len(indices) for indices in per_frame]
    max_count = max(counts)
    representative_candidates = [index for index, count in enumerate(counts) if count == max_count]
    representative = min(
        representative_candidates,
        key=lambda index: (abs(index - (frame_count - 1) / 2.0), index),
    )
    selected_local = sorted(per_frame[representative])
    frame = display_frames[representative]
    overlay = overlay_selection(frame, selected_local, grid_height, grid_width)
    timestamp = (
        float(source_indices[representative]) / fps if fps is not None else None
    )
    return Example(
        video_path=path,
        question=question,
        display_question=display_question(question),
        frame=frame,
        overlay=overlay,
        sampled_frame_index=representative,
        source_frame_index=int(source_indices[representative]),
        timestamp_seconds=timestamp,
        grid_height=grid_height,
        grid_width=grid_width,
        selected_in_frame=selected_local,
        per_frame_counts=counts,
        anchor_indices=[int(index) for index in anchors],
        raw_token_count=raw_token_count,
        output_token_count=len(anchors),
    )


def fit_image(image: Image.Image, size: int) -> Image.Image:
    fitted = image.copy()
    resampling = getattr(Image, "Resampling", Image)
    fitted.thumbnail((size, size), resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    x = (size - fitted.width) // 2
    y = (size - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def draw_centered(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    draw.text((xy[0] - (box[2] - box[0]) // 2, xy[1]), text, font=font, fill=fill)


def render_figure(examples: list[Example], output_path: Path, args: argparse.Namespace) -> None:
    tile = 430
    image_gap = 22
    card_gap = 48
    margin = 54
    columns = min(2, len(examples))
    rows = math.ceil(len(examples) / columns)
    pair_width = tile * 2 + image_gap
    question_height = 105
    caption_height = 92
    card_height = question_height + tile + caption_height
    header_height = 155
    width = margin * 2 + columns * pair_width + (columns - 1) * card_gap
    height = header_height + margin + rows * card_height + (rows - 1) * card_gap + margin

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(36, serif=True, bold=True)
    subtitle_font = load_font(23, serif=True)
    question_font = load_font(23, serif=True)
    caption_font = load_font(22, serif=True)
    detail_font = load_font(18, serif=False)

    draw_centered(
        draw,
        (width // 2, 28),
        "Qualitative visualization of CertVID evidence selection",
        title_font,
        (15, 22, 28),
    )
    outer_ratio = args.retention_ratio * args.expansion * 100.0
    subtitle = (
        f"Random dataset samples | {args.num_frames} frames | nominal R={args.retention_ratio * 100:.2f}% "
        f"| outer anchor ratio={outer_ratio:.2f}%"
    )
    draw_centered(draw, (width // 2, 83), subtitle, subtitle_font, (62, 72, 82))
    draw.line((margin, 125, width - margin, 125), fill=(182, 188, 193), width=2)

    for index, example in enumerate(examples):
        row, col = divmod(index, columns)
        x = margin + col * (pair_width + card_gap)
        y = header_height + margin + row * (card_height + card_gap)

        wrapped = textwrap.wrap(f"Question: {example.display_question}", width=72)[:3]
        question_text = "\n".join(wrapped)
        qbox = draw.multiline_textbbox((0, 0), question_text, font=question_font, spacing=5)
        qwidth = qbox[2] - qbox[0]
        draw.multiline_text(
            (x + pair_width // 2 - qwidth // 2, y),
            question_text,
            font=question_font,
            fill=(18, 26, 33),
            spacing=5,
            align="center",
        )

        image_y = y + question_height
        original = fit_image(example.frame, tile)
        selected = fit_image(example.overlay, tile)
        canvas.paste(original, (x, image_y))
        canvas.paste(selected, (x + tile + image_gap, image_y))
        draw.rectangle((x, image_y, x + tile - 1, image_y + tile - 1), outline=(120, 128, 136), width=2)
        draw.rectangle(
            (x + tile + image_gap, image_y, x + pair_width - 1, image_y + tile - 1),
            outline=(0, 154, 155),
            width=3,
        )

        caption_y = image_y + tile + 13
        draw_centered(draw, (x + tile // 2, caption_y), "Original sampled frame", caption_font, (25, 31, 36))
        draw_centered(
            draw,
            (x + tile + image_gap + tile // 2, caption_y),
            "CertVID selected evidence",
            caption_font,
            (0, 132, 133),
        )
        time_text = (
            f"{example.timestamp_seconds:.1f}s" if example.timestamp_seconds is not None else "time unavailable"
        )
        details = (
            f"{example.video_path.stem} | sampled frame {example.sampled_frame_index + 1}/{args.num_frames} "
            f"({time_text}) | {len(example.selected_in_frame)} selected patches"
        )
        draw_centered(draw, (x + pair_width // 2, caption_y + 38), details, detail_font, (82, 89, 96))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def save_manifest(examples: list[Example], output_path: Path, args: argparse.Namespace) -> None:
    records = []
    for example in examples:
        records.append(
            {
                "video_path": str(example.video_path),
                "question": example.question,
                "display_question": example.display_question,
                "sampled_frame_index": example.sampled_frame_index,
                "source_frame_index": example.source_frame_index,
                "timestamp_seconds": example.timestamp_seconds,
                "grid": [example.grid_height, example.grid_width],
                "selected_patch_indices_in_frame": example.selected_in_frame,
                "anchors_per_frame": example.per_frame_counts,
                "anchor_indices": example.anchor_indices,
                "raw_token_count": example.raw_token_count,
                "output_token_count": example.output_token_count,
            }
        )
    payload = {
        "seed": args.seed,
        "num_frames": args.num_frames,
        "retention_ratio": args.retention_ratio,
        "expansion": args.expansion,
        "pruning_layer": args.pruning_layer,
        "llm_retention_ratio": args.llm_retention_ratio,
        "examples": records,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.num_examples <= 0 or args.num_frames <= 0:
        raise ValueError("num-examples and num-frames must be positive")
    if not (0.0 < args.retention_ratio <= 1.0):
        raise ValueError("retention-ratio must be in (0, 1]")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    metadata_path = (
        Path(args.metadata_jsonl).expanduser().resolve() if args.metadata_jsonl.strip() else None
    )
    questions = load_questions(metadata_path)
    candidates = discover_videos(dataset_root)
    rng = random.Random(args.seed)
    rng.shuffle(candidates)

    print(f"[setup] dataset={dataset_root}")
    print(f"[setup] discovered_videos={len(candidates)} seed={args.seed}")
    print(f"[setup] metadata_videos={len(questions)}")
    print(f"[setup] output={output_dir}")
    tokenizer, model, image_processor, device = load_certvid_model(args)

    examples: list[Example] = []
    failures: list[dict[str, str]] = []
    for path in candidates[: max(args.max_attempts, args.num_examples)]:
        video_questions = questions.get(path.stem, [GENERIC_QUESTION])
        question = rng.choice(video_questions)
        try:
            example = run_one_example(
                path,
                question,
                tokenizer,
                model,
                image_processor,
                device,
                args,
            )
        except Exception as exc:
            failures.append({"video_path": str(path), "error": str(exc)})
            print(f"[skip] {path.name}: {exc}")
            continue

        examples.append(example)
        per_example_path = output_dir / f"example_{len(examples):02d}_{path.stem}.png"
        render_figure([example], per_example_path, args)
        print(
            f"[ok] {len(examples)}/{args.num_examples} {path.name} "
            f"anchors={example.output_token_count}/{example.raw_token_count} "
            f"figure={per_example_path}"
        )
        if len(examples) >= args.num_examples:
            break

    failures_path = output_dir / "failures.json"
    failures_path.write_text(
        json.dumps(failures, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if len(examples) < args.num_examples:
        raise RuntimeError(
            f"only produced {len(examples)}/{args.num_examples} examples; "
            f"see {failures_path}"
        )

    figure_path = output_dir / "certvid_qualitative_examples.png"
    render_figure(examples, figure_path, args)
    save_manifest(examples, output_dir / "manifest.json", args)
    print(f"[done] combined_figure={figure_path}")
    print(f"[done] manifest={output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
