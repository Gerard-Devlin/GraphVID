#!/usr/bin/env python3
"""Export three paper-ready views of CertVID's real attention dynamics."""

from __future__ import annotations

import argparse
import os
import random
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from visualize_certvid_two_examples import (
    QuestionRecord,
    discover_videos,
    generate_once,
    load_certvid_model,
    load_questions,
    prepare_prompt,
    sample_video,
)


OUTPUT_NAMES = (
    "01_filmstrip.png",
    "02_visual_attention_heatmap.png",
    "03_frame_weight_curves.png",
)


def parse_args() -> argparse.Namespace:
    hf_home = Path(os.environ.get("HF_HOME", "/home/xuyouwen/hf_home_local"))
    parser = argparse.ArgumentParser(
        description=(
            "Run one real CertVID V3 sample and export a filmstrip, a "
            "layer-by-head visual-attention heatmap, and four frame curves."
        )
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get(
            "PRETRAINED",
            "/home/xuyouwen/models/llava-onevision-qwen2-7b-ov",
        ),
    )
    parser.add_argument(
        "--dataset-root",
        default=str(hf_home / "videomme" / "data"),
    )
    parser.add_argument("--metadata-jsonl", default="assets/videomme.jsonl")
    parser.add_argument("--answers-json", default="")
    parser.add_argument(
        "--video-id",
        default="",
        help="Video filename or stem. When omitted, a metadata-backed video is sampled.",
    )
    parser.add_argument(
        "--question-id",
        default="",
        help="Optional question ID when a video has multiple questions.",
    )
    parser.add_argument(
        "--output-dir",
        default="logs/visualizations/certvid_attention_dynamics",
    )
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--filmstrip-frames", type=int, default=8)
    parser.add_argument("--curve-layers", default="2,11,20,22")
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--retention-ratio", type=float, default=0.25)
    parser.add_argument("--expansion", type=float, default=1.30)
    parser.add_argument("--pruning-layer", type=int, default=20)
    parser.add_argument("--llm-retention-ratio", type=float, default=0.1923076923)
    parser.add_argument(
        "--pool-mode",
        choices=("bilinear", "average", "max"),
        default="bilinear",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def parse_layers(value: str) -> list[int]:
    layers = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(layers) != 4 or len(set(layers)) != 4 or min(layers) <= 0:
        raise ValueError("--curve-layers must contain four distinct positive layers")
    return layers


def choose_example(
    videos: list[Path],
    questions: dict[str, list[QuestionRecord]],
    *,
    video_id: str,
    question_id: str,
    seed: int,
) -> tuple[Path, QuestionRecord]:
    by_stem = {path.stem: path for path in videos}
    if video_id:
        stem = Path(video_id).stem
        if stem not in by_stem:
            raise FileNotFoundError(f"requested video was not found: {video_id}")
        candidates = questions.get(stem, [])
        if not candidates:
            raise KeyError(f"metadata has no question for video: {stem}")
        video_path = by_stem[stem]
    else:
        available = sorted(set(by_stem).intersection(questions))
        if not available:
            raise RuntimeError("no discovered video has matching question metadata")
        stem = random.Random(seed).choice(available)
        video_path = by_stem[stem]
        candidates = questions[stem]

    if question_id:
        matches = [
            record
            for record in candidates
            if str(record.question_id or "") == str(question_id)
        ]
        if not matches:
            raise KeyError(
                f"question {question_id!r} was not found for video {video_path.stem}"
            )
        return video_path, matches[0]
    return video_path, candidates[0]


def select_filmstrip_indices(
    frame_count: int,
    panel_count: int,
    peak_frame: int,
) -> list[int]:
    panel_count = min(frame_count, max(2, panel_count))
    indices = np.linspace(0, frame_count - 1, panel_count).round().astype(int).tolist()
    indices = sorted(set(indices))
    if peak_frame not in indices:
        replace_at = min(
            range(len(indices)),
            key=lambda idx: (abs(indices[idx] - peak_frame), idx in (0, len(indices) - 1)),
        )
        indices[replace_at] = peak_frame
        indices = sorted(set(indices))
    return indices


def build_filmstrip(
    frames: list[Image.Image],
    frame_indices: list[int],
    peak_frame: int,
) -> Image.Image:
    tile_width, tile_height = 360, 210
    gap, rail = 10, 36
    left_right = 12
    width = left_right * 2 + len(frame_indices) * tile_width + (len(frame_indices) - 1) * gap
    height = tile_height + rail * 2
    strip = Image.new("RGB", (width, height), "black")

    for position, frame_index in enumerate(frame_indices):
        x0 = left_right + position * (tile_width + gap)
        tile = ImageOps.fit(
            frames[frame_index].convert("RGB"),
            (tile_width, tile_height),
            method=Image.Resampling.LANCZOS,
        )
        strip.paste(tile, (x0, rail))
        if frame_index == peak_frame:
            draw = ImageDraw.Draw(strip)
            draw.rectangle(
                (x0, rail, x0 + tile_width - 1, rail + tile_height - 1),
                outline=(218, 45, 38),
                width=7,
            )

    draw = ImageDraw.Draw(strip)
    hole_width, hole_height, hole_gap = 20, 14, 12
    x = 8
    while x + hole_width < width:
        draw.rectangle((x, 8, x + hole_width, 8 + hole_height), fill="white")
        draw.rectangle(
            (x, height - 8 - hole_height, x + hole_width, height - 8),
            fill="white",
        )
        x += hole_width + hole_gap
    return strip


def save_filmstrip(
    *,
    frames: list[Image.Image],
    question: str,
    frame_weights: np.ndarray,
    output_path: Path,
    panel_count: int,
    dpi: int,
) -> int:
    if len(frame_weights) != len(frames):
        raise RuntimeError(
            "deep-layer frame weights do not match the sampled video frames"
        )
    peak_frame = int(np.argmax(frame_weights))
    frame_indices = select_filmstrip_indices(len(frames), panel_count, peak_frame)
    strip = build_filmstrip(frames, frame_indices, peak_frame)

    figure, axis = plt.subplots(figsize=(18, 3.5), facecolor="white")
    axis.imshow(strip)
    axis.axis("off")
    title = textwrap.fill(question.strip(), width=105)
    figure.suptitle(
        f"Question: {title}",
        fontsize=19,
        fontfamily="DejaVu Serif",
        fontweight="semibold",
        y=0.99,
    )
    figure.subplots_adjust(left=0.015, right=0.985, bottom=0.03, top=0.79)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return peak_frame


def stack_attention_records(
    records: dict[int, dict[str, object]],
) -> tuple[list[int], np.ndarray]:
    layer_numbers = sorted(int(layer) for layer in records)
    if not layer_numbers:
        raise RuntimeError("the language model did not publish layer attention")
    rows = []
    head_count = None
    for layer in layer_numbers:
        values = records[layer]["visual_ratio_per_head"]
        if not torch.is_tensor(values) or values.ndim != 1:
            raise RuntimeError(f"layer {layer} has malformed per-head attention")
        row = values.detach().float().cpu().numpy()
        if not np.isfinite(row).all():
            raise RuntimeError(f"layer {layer} attention contains NaN/Inf")
        head_count = len(row) if head_count is None else head_count
        if len(row) != head_count:
            raise RuntimeError("attention head count changes across layers")
        rows.append(row)
    return layer_numbers, np.stack(rows, axis=0)


def save_attention_heatmap(
    *,
    layer_numbers: list[int],
    heatmap: np.ndarray,
    pruning_layer: int,
    output_path: Path,
    dpi: int,
) -> None:
    figure, axis = plt.subplots(figsize=(9.2, 8.2), facecolor="white")
    positive = heatmap[heatmap > 0]
    vmax = float(np.quantile(positive, 0.985)) if positive.size else 1.0
    image = axis.imshow(
        heatmap,
        origin="lower",
        aspect="auto",
        cmap="Blues",
        vmin=0.0,
        vmax=max(vmax, 1e-8),
        interpolation="nearest",
    )
    axis.set_title("Visual Attention Ratio Across Layers and Heads", fontsize=15, pad=12)
    axis.set_xlabel("Attention Head", fontsize=12)
    axis.set_ylabel("Transformer Layer", fontsize=12)
    x_step = max(1, heatmap.shape[1] // 8)
    x_positions = np.arange(0, heatmap.shape[1], x_step)
    axis.set_xticks(x_positions, [str(value + 1) for value in x_positions])
    y_step = max(1, len(layer_numbers) // 14)
    y_positions = np.arange(0, len(layer_numbers), y_step)
    axis.set_yticks(y_positions, [str(layer_numbers[value]) for value in y_positions])

    boundary = sum(layer <= pruning_layer for layer in layer_numbers) - 0.5
    if 0.0 <= boundary <= len(layer_numbers) - 1:
        axis.axhline(boundary, color="#D7261E", linestyle="--", linewidth=1.8)
        axis.text(
            heatmap.shape[1] - 0.2,
            boundary + 0.25,
            "inner pruning",
            color="#B51E18",
            fontsize=9,
            ha="right",
            va="bottom",
        )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Attention mass on visual tokens", fontsize=10)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def save_frame_curves(
    *,
    records: dict[int, dict[str, object]],
    curve_layers: list[int],
    output_path: Path,
    dpi: int,
) -> None:
    missing = [layer for layer in curve_layers if layer not in records]
    if missing:
        raise RuntimeError(
            f"requested curve layers are unavailable: {missing}; "
            f"available={sorted(records)}"
        )

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.0), facecolor="white")
    for axis, layer in zip(axes.flat, curve_layers):
        values = records[layer]["frame_weights"]
        if not torch.is_tensor(values) or values.ndim != 1:
            raise RuntimeError(f"layer {layer} has malformed frame weights")
        weights = values.detach().float().cpu().numpy()
        if not np.isfinite(weights).all():
            raise RuntimeError(f"layer {layer} frame weights contain NaN/Inf")
        frame_indices = np.arange(1, len(weights) + 1)
        peak = int(np.argmax(weights))
        axis.plot(
            frame_indices,
            weights,
            color="#176B9C",
            linewidth=1.8,
            marker="o",
            markersize=3.8,
        )
        axis.scatter(
            [peak + 1],
            [weights[peak]],
            marker="*",
            s=260,
            color="#C51610",
            edgecolors="white",
            linewidths=0.7,
            zorder=5,
        )
        axis.set_title(f"Frame Weight Curve at Layer {layer}", fontsize=12)
        axis.set_xlabel("Sampled Frame Index")
        axis.set_ylabel("Normalized Frame Weight")
        axis.set_xlim(1, len(weights))
        axis.set_ylim(bottom=0.0)
        axis.grid(True, alpha=0.28, linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle(
        "Question-conditioned Temporal Evidence Evolution",
        fontsize=17,
        y=1.01,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.num_frames <= 1:
        raise ValueError("--num-frames must be greater than one")
    curve_layers = parse_layers(args.curve_layers)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = Path(args.metadata_jsonl).expanduser().resolve()
    answers_path = (
        Path(args.answers_json).expanduser().resolve() if args.answers_json else None
    )
    questions = load_questions(metadata_path, answers_path)
    videos = discover_videos(Path(args.dataset_root).expanduser().resolve())
    video_path, question_record = choose_example(
        videos,
        questions,
        video_id=args.video_id,
        question_id=args.question_id,
        seed=args.seed,
    )
    print(
        f"[sample] video={video_path.name} question_id={question_record.question_id}",
        flush=True,
    )
    print(f"[question] {question_record.question}", flush=True)

    frames_array, _, _ = sample_video(video_path, args.num_frames)
    display_frames = [Image.fromarray(frame, mode="RGB") for frame in frames_array]
    tokenizer, model, image_processor, device = load_certvid_model(args)
    config = model.flashvid_config
    setattr(config, "_capture_layer_frame_attention", True)

    pixel_values_cpu = image_processor.preprocess(
        frames_array,
        return_tensors="pt",
    )["pixel_values"]
    pixel_values = pixel_values_cpu.to(device=device, dtype=torch.float16)
    input_ids, attention_mask = prepare_prompt(
        tokenizer,
        question_record.prompt,
        device,
    )

    try:
        generate_once(
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
        captured = getattr(config, "_visualization_layer_attention", None)
        if not isinstance(captured, dict) or not captured:
            raise RuntimeError("no per-layer attention was captured")
        records = {
            int(layer): {
                key: value.detach().float().cpu().clone()
                if torch.is_tensor(value)
                else value
                for key, value in record.items()
            }
            for layer, record in captured.items()
        }
    finally:
        setattr(config, "_capture_layer_frame_attention", False)
        setattr(config, "_visualization_current_frame_ids", None)

    layer_numbers, heatmap = stack_attention_records(records)
    missing = [layer for layer in curve_layers if layer not in records]
    if missing:
        raise RuntimeError(
            f"model did not capture requested layers {missing}; available={layer_numbers}"
        )
    deepest_layer = max(curve_layers)
    deepest_weights = (
        records[deepest_layer]["frame_weights"].detach().float().cpu().numpy()
    )

    peak_frame = save_filmstrip(
        frames=display_frames,
        question=question_record.question,
        frame_weights=deepest_weights,
        output_path=output_dir / OUTPUT_NAMES[0],
        panel_count=args.filmstrip_frames,
        dpi=args.dpi,
    )
    save_attention_heatmap(
        layer_numbers=layer_numbers,
        heatmap=heatmap,
        pruning_layer=args.pruning_layer,
        output_path=output_dir / OUTPUT_NAMES[1],
        dpi=args.dpi,
    )
    save_frame_curves(
        records=records,
        curve_layers=curve_layers,
        output_path=output_dir / OUTPUT_NAMES[2],
        dpi=args.dpi,
    )

    print(
        f"[done] peak_frame={peak_frame + 1} source_layer={deepest_layer}",
        flush=True,
    )
    for name in OUTPUT_NAMES:
        print(output_dir / name, flush=True)


if __name__ == "__main__":
    main()
