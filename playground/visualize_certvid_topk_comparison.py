#!/usr/bin/env python3
"""Visualize equal-budget Quality Top-K and CertVID selections over video.

The paper figure follows a compact qualitative layout: an eight-frame filmstrip
provides temporal context, followed by two rows of four identical frames. The
first row shows global quality Top-K and the second shows the actual CertVID V3
anchors. Both methods always use the same global token budget.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from visualize_certvid_two_examples import (
    discover_videos,
    extract_answer_label,
    factor_grid,
    generate_once,
    load_certvid_model,
    load_questions,
    prepare_prompt,
    sample_video,
    tensor_frames_to_pil,
)
from visualize_certvid_volume import _information_summary


BASELINE_COLOR = "#D98C84"
BASELINE_OUTLINE_COLOR = "#9F3232"
OURS_COLOR = "#2A9D6F"
UNSELECTED_COLOR = "#16324A"
GRID_COLOR = "#F4F0EA"
INK_COLOR = "#20242A"


@dataclass
class ComparisonCase:
    video_id: str
    video_path: Path
    question_id: str | None
    question: str
    options: list[dict[str, str]]
    target: str | None
    prediction: str | None
    raw_prediction: str
    frames: list[Image.Image]
    quality: torch.Tensor
    ours_indices: torch.Tensor
    topk_indices: torch.Tensor
    comparison_frames: list[int]
    frame_count: int
    tokens_per_frame: int
    grid_height: int
    grid_width: int
    d_efficiency: float
    overlap_ratio: float
    selection_score: float


def parse_args() -> argparse.Namespace:
    hf_home = Path(os.environ.get("HF_HOME", "/home/xuyouwen/hf_home_local"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=os.environ.get(
            "PRETRAINED", "/home/xuyouwen/models/llava-onevision-qwen2-7b-ov"
        ),
    )
    parser.add_argument("--dataset-root", default=str(hf_home / "videomme" / "data"))
    parser.add_argument("--metadata-jsonl", default="assets/videomme.jsonl")
    parser.add_argument(
        "--candidate-ids-file",
        default=None,
        help=(
            "Optional ordered video-ID allowlist. Blank lines and text after '#' "
            "are ignored. When provided, only listed videos are scanned."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-count", type=int, default=20)
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--filmstrip-frames", type=int, default=8)
    parser.add_argument(
        "--comparison-frames",
        default="1,8,17,26",
        help="One-based sampled-frame indices, or 'auto' for four spread frames.",
    )
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--retention-ratio", type=float, default=0.01)
    parser.add_argument("--expansion", type=float, default=1.30)
    parser.add_argument("--pruning-layer", type=int, default=20)
    parser.add_argument("--llm-retention-ratio", type=float, default=0.1923076923)
    parser.add_argument(
        "--pool-mode", choices=("bilinear", "average", "max"), default="bilinear"
    )
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _per_frame(
    indices: torch.Tensor,
    frame_count: int,
    tokens_per_frame: int,
) -> list[list[int]]:
    result: list[list[int]] = [[] for _ in range(frame_count)]
    for global_index in indices.tolist():
        frame_index, local_index = divmod(int(global_index), tokens_per_frame)
        if 0 <= frame_index < frame_count:
            result[frame_index].append(local_index)
    return [sorted(values) for values in result]


def _parse_comparison_frames(specification: str, frame_count: int) -> list[int] | None:
    if specification.strip().lower() == "auto":
        return None
    values = [value.strip() for value in specification.split(",") if value.strip()]
    if len(values) != 4:
        raise ValueError("--comparison-frames must contain exactly four indices")
    result = [int(value) - 1 for value in values]
    if any(value < 0 or value >= frame_count for value in result):
        raise ValueError(
            f"comparison frames must be in [1, {frame_count}], got {specification!r}"
        )
    if len(set(result)) != 4:
        raise ValueError("comparison frames must be unique")
    return result


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return values
    span = float(values.max() - values.min())
    if span <= 1e-12:
        return np.zeros_like(values)
    return (values - values.min()) / span


def _frame_detail(frame: Image.Image) -> float:
    gray = np.asarray(frame.convert("L"), dtype=np.float32) / 255.0
    contrast = float(gray.std())
    edge_x = float(np.abs(np.diff(gray, axis=1)).mean()) if gray.shape[1] > 1 else 0.0
    edge_y = float(np.abs(np.diff(gray, axis=0)).mean()) if gray.shape[0] > 1 else 0.0
    return contrast + 2.5 * (edge_x + edge_y)


def _auto_comparison_frames(
    frames: list[Image.Image],
    ours_per_frame: list[list[int]],
    topk_per_frame: list[list[int]],
) -> list[int]:
    frame_count = len(frames)
    detail = _normalize(np.asarray([_frame_detail(frame) for frame in frames]))
    disagreement = _normalize(
        np.asarray(
            [
                len(set(ours_per_frame[index]) ^ set(topk_per_frame[index]))
                for index in range(frame_count)
            ],
            dtype=np.float64,
        )
    )
    support = _normalize(
        np.asarray(
            [
                len(ours_per_frame[index]) + len(topk_per_frame[index])
                for index in range(frame_count)
            ],
            dtype=np.float64,
        )
    )
    score = 0.42 * detail + 0.43 * disagreement + 0.15 * support

    # Select one frame from each temporal quartile. This preserves the visual
    # narrative and prevents all four panels from collapsing onto one event.
    boundaries = np.linspace(0, frame_count, 5, dtype=int)
    selected: list[int] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        candidates = list(range(start, max(start + 1, end)))
        selected.append(max(candidates, key=lambda index: (float(score[index]), -index)))
    return selected


def _normalized_entropy(
    indices: torch.Tensor, frame_count: int, tokens_per_frame: int
) -> float:
    frame_ids = torch.div(indices.long(), tokens_per_frame, rounding_mode="floor")
    counts = torch.bincount(frame_ids, minlength=frame_count).double()
    probabilities = counts / counts.sum().clamp_min(1.0)
    positive = probabilities > 0
    entropy = -(probabilities[positive] * probabilities[positive].log()).sum()
    return float((entropy / math.log(max(2, frame_count))).item())


def _case_score(
    frames: list[Image.Image],
    comparison_frames: list[int],
    ours_per_frame: list[list[int]],
    topk_per_frame: list[list[int]],
    d_efficiency: float,
    overlap_ratio: float,
    entropy_gain: float,
) -> float:
    disagreement = np.asarray(
        [
            len(set(ours_per_frame[index]) ^ set(topk_per_frame[index]))
            for index in comparison_frames
        ],
        dtype=np.float64,
    )
    detail = np.asarray(
        [_frame_detail(frames[index]) for index in comparison_frames],
        dtype=np.float64,
    )
    visible_disagreement = float(np.mean(np.log1p(disagreement) * (0.5 + detail)))
    return (
        math.log(max(1e-8, d_efficiency))
        + 0.30 * visible_disagreement
        + 0.20 * entropy_gain
        + 0.20 * (1.0 - overlap_ratio)
    )


def _retention_map(
    frame: Image.Image,
    selected: list[int],
    quality: np.ndarray,
    grid_height: int,
    grid_width: int,
    accent: str,
    outline: str | None = None,
) -> Image.Image:
    base = frame.convert("RGB")
    width, height = base.size
    selected_set = set(int(value) for value in selected)
    quality = _normalize(np.asarray(quality, dtype=np.float64).reshape(-1))
    if quality.size != grid_height * grid_width:
        raise ValueError("quality map and visual-token grid do not match")

    result = base.convert("RGBA")
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    unselected_rgb = tuple(int(UNSELECTED_COLOR[index : index + 2], 16) for index in (1, 3, 5))
    accent_rgb = tuple(int(accent[index : index + 2], 16) for index in (1, 3, 5))
    outline = outline or accent
    outline_rgb = tuple(int(outline[index : index + 2], 16) for index in (1, 3, 5))
    border_width = max(1, int(round(min(width, height) / 260)))
    selected_boxes: list[tuple[int, int, int, int]] = []

    for row in range(grid_height):
        y0 = int(round(row * height / grid_height))
        y1 = int(round((row + 1) * height / grid_height))
        for column in range(grid_width):
            x0 = int(round(column * width / grid_width))
            x1 = int(round((column + 1) * width / grid_width))
            token = row * grid_width + column
            if token in selected_set:
                # Keep the underlying evidence visible while making the two
                # method colors legible on saturated sports footage.
                alpha = int(round(58 + 46 * quality[token]))
                draw.rectangle((x0, y0, x1, y1), fill=(*accent_rgb, alpha))
                selected_boxes.append((x0, y0, x1, y1))
            else:
                alpha = int(round(112 + 38 * (1.0 - quality[token])))
                draw.rectangle((x0, y0, x1, y1), fill=(*unselected_rgb, alpha))

    # A very subtle patch grid preserves the token interpretation without
    # overpowering the underlying frame.
    grid_rgb = tuple(int(GRID_COLOR[index : index + 2], 16) for index in (1, 3, 5))
    for row in range(1, grid_height):
        y = int(round(row * height / grid_height))
        draw.line((0, y, width, y), fill=(*grid_rgb, 42), width=1)
    for column in range(1, grid_width):
        x = int(round(column * width / grid_width))
        draw.line((x, 0, x, height), fill=(*grid_rgb, 42), width=1)

    # Draw selection boundaries last. Insetting them keeps all four edges inside
    # the image and prevents neighboring fills or grid lines from covering them.
    inset = max(1, (border_width + 1) // 2)
    for x0, y0, x1, y1 in selected_boxes:
        left = min(x1 - 1, x0 + inset)
        top = min(y1 - 1, y0 + inset)
        right = max(left, x1 - 1 - inset)
        bottom = max(top, y1 - 1 - inset)
        draw.rectangle(
            (left, top, right, bottom),
            outline=(*outline_rgb, 255),
            width=border_width,
        )
    return Image.alpha_composite(result, overlay).convert("RGB")


def _filmstrip_indices(frame_count: int, required: list[int], count: int) -> list[int]:
    count = max(len(required), min(count, frame_count))
    candidates = set(required)
    candidates.update(
        int(round(value))
        for value in np.linspace(0, frame_count - 1, count, dtype=np.float64)
    )
    if len(candidates) > count:
        optional = [value for value in sorted(candidates) if value not in required]
        while len(candidates) > count and optional:
            candidates.remove(optional.pop(len(optional) // 2))
    if len(candidates) < count:
        for value in range(frame_count):
            candidates.add(value)
            if len(candidates) == count:
                break
    return sorted(candidates)


def _filmstrip(
    frames: list[Image.Image],
    frame_indices: list[int],
    highlighted: set[int],
) -> Image.Image:
    cell_width = 320
    cell_height = 180
    border = 20
    sprocket_height = 15
    separator = 7
    canvas_width = border * 2 + len(frame_indices) * cell_width + (len(frame_indices) - 1) * separator
    canvas_height = border * 2 + sprocket_height * 2 + cell_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), "black")
    draw = ImageDraw.Draw(canvas)
    y0 = border + sprocket_height

    x = border
    for frame_index in frame_indices:
        frame = frames[frame_index].convert("RGB")
        frame.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (cell_width, cell_height), "black")
        tile.paste(frame, ((cell_width - frame.width) // 2, (cell_height - frame.height) // 2))
        canvas.paste(tile, (x, y0))
        if frame_index in highlighted:
            line_width = 5
            draw.rectangle(
                (x + 1, y0 + 1, x + cell_width - 2, y0 + cell_height - 2),
                outline=OURS_COLOR,
                width=line_width,
            )
        x += cell_width + separator

    hole_width = 18
    hole_gap = 13
    for x0 in range(border, canvas_width - border - hole_width + 1, hole_width + hole_gap):
        draw.rounded_rectangle(
            (x0, 4, x0 + hole_width, 4 + sprocket_height - 4),
            radius=2,
            fill="white",
        )
        bottom = canvas_height - sprocket_height
        draw.rounded_rectangle(
            (x0, bottom, x0 + hole_width, canvas_height - 4),
            radius=2,
            fill="white",
        )
    return canvas


def _plot(
    case: ComparisonCase,
    output_dir: Path,
    filmstrip_frames: int,
    dpi: int,
    stem_name: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    ours_per_frame = _per_frame(case.ours_indices, case.frame_count, case.tokens_per_frame)
    topk_per_frame = _per_frame(case.topk_indices, case.frame_count, case.tokens_per_frame)
    quality_2d = case.quality.reshape(case.frame_count, case.tokens_per_frame).numpy()
    film_indices = _filmstrip_indices(
        case.frame_count, case.comparison_frames, filmstrip_frames
    )
    strip = _filmstrip(case.frames, film_indices, set(case.comparison_frames))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "font.size": 10.0,
            "text.color": INK_COLOR,
        }
    )
    fig = plt.figure(figsize=(13.2, 7.10), facecolor="white")
    grid = GridSpec(
        3,
        4,
        figure=fig,
        height_ratios=(0.62, 1.0, 1.0),
        hspace=0.10,
        wspace=0.055,
        left=0.105,
        right=0.992,
        bottom=0.025,
        top=0.985,
    )
    film_axis = fig.add_subplot(grid[0, :])
    film_axis.imshow(strip)
    film_axis.set_axis_off()

    row_axes: dict[int, list[Any]] = {1: [], 2: []}
    for column, frame_index in enumerate(case.comparison_frames):
        topk_map = _retention_map(
            case.frames[frame_index],
            topk_per_frame[frame_index],
            quality_2d[frame_index],
            case.grid_height,
            case.grid_width,
            BASELINE_COLOR,
            BASELINE_OUTLINE_COLOR,
        )
        ours_map = _retention_map(
            case.frames[frame_index],
            ours_per_frame[frame_index],
            quality_2d[frame_index],
            case.grid_height,
            case.grid_width,
            OURS_COLOR,
        )
        for row, image in ((1, topk_map), (2, ours_map)):
            axis = fig.add_subplot(grid[row, column])
            row_axes[row].append(axis)
            axis.imshow(image)
            axis.set_axis_off()
            if row == 1:
                axis.set_title(f"Frame {frame_index + 1}", fontsize=11.2, pad=5.0)

    # Derive label positions from the rendered panels instead of fragile constants.
    fig.canvas.draw()
    row_centers = {
        row: sum(
            0.5 * (axis.get_position().y0 + axis.get_position().y1)
            for axis in axes
        )
        / len(axes)
        for row, axes in row_axes.items()
    }
    first_panel_left = min(axis.get_position().x0 for axis in row_axes[1])
    # Keep row labels close to the first panel without overlapping the image.
    label_x = 0.72 * first_panel_left
    fig.text(
        label_x,
        row_centers[1],
        "Quality\nTop-K",
        ha="center",
        va="center",
        fontsize=12.0,
        fontweight="semibold",
        color=INK_COLOR,
    )
    fig.text(
        label_x,
        row_centers[2],
        "CertVID",
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color=INK_COLOR,
    )

    stem = output_dir / stem_name
    for extension in ("png", "pdf"):
        fig.savefig(
            stem.with_suffix(f".{extension}"),
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.03,
            facecolor="white",
        )
    plt.close(fig)


def _audit_record(case: ComparisonCase, selected: bool) -> dict[str, Any]:
    ours_per_frame = _per_frame(case.ours_indices, case.frame_count, case.tokens_per_frame)
    topk_per_frame = _per_frame(case.topk_indices, case.frame_count, case.tokens_per_frame)
    return {
        "selected_for_figure": selected,
        "video_id": case.video_id,
        "video_path": str(case.video_path),
        "question_id": case.question_id,
        "question": case.question,
        "options": case.options,
        "target": case.target,
        "prediction": case.prediction,
        "raw_prediction": case.raw_prediction,
        "selection_policy_uses_answer": False,
        "raw_visual_tokens": case.frame_count * case.tokens_per_frame,
        "equal_global_budget": int(case.ours_indices.numel()),
        "comparison_frames_one_based": [value + 1 for value in case.comparison_frames],
        "filmstrip_highlight_semantics": (
            "Rose borders indicate the four frames visualized in both comparison rows."
        ),
        "retention_map_semantics": {
            "selected": "Original image remains visible with a method-colored patch boundary.",
            "unselected": "Patch is dimmed with a shared navy overlay.",
            "attention_heatmap": False,
        },
        "per_frame": [
            {
                "frame_one_based": frame_index + 1,
                "quality_topk_count": len(topk_per_frame[frame_index]),
                "certvid_count": len(ours_per_frame[frame_index]),
                "quality_topk_local_indices": topk_per_frame[frame_index],
                "certvid_local_indices": ours_per_frame[frame_index],
            }
            for frame_index in case.comparison_frames
        ],
        "global_overlap_ratio": case.overlap_ratio,
        "d_efficiency_certvid_vs_quality_topk": case.d_efficiency,
        "selection_score": case.selection_score,
    }


def main() -> None:
    args = parse_args()
    if args.candidate_count <= 0:
        raise ValueError("--candidate-count must be positive")
    if args.num_examples <= 0:
        raise ValueError("--num-examples must be positive")
    if args.filmstrip_frames < 4:
        raise ValueError("--filmstrip-frames must be at least four")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(args.metadata_jsonl).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    questions = load_questions(metadata_path)
    videos = [path for path in discover_videos(dataset_root) if path.stem in questions]
    candidate_ids_path: Path | None = None
    if args.candidate_ids_file:
        candidate_ids_path = Path(args.candidate_ids_file).expanduser().resolve()
        if not candidate_ids_path.is_file():
            raise FileNotFoundError(
                f"candidate video-ID file does not exist: {candidate_ids_path}"
            )
        candidate_ids = []
        for line in candidate_ids_path.read_text(encoding="utf-8").splitlines():
            video_id = line.split("#", 1)[0].strip()
            if video_id and video_id not in candidate_ids:
                candidate_ids.append(video_id)
        if not candidate_ids:
            raise RuntimeError(f"candidate video-ID file is empty: {candidate_ids_path}")
        candidate_order = {
            video_id: position for position, video_id in enumerate(candidate_ids)
        }
        videos = [path for path in videos if path.stem in candidate_order]
        videos.sort(key=lambda path: candidate_order[path.stem])
        missing_ids = [
            video_id
            for video_id in candidate_ids
            if all(path.stem != video_id for path in videos)
        ]
        if missing_ids:
            print(
                f"[setup] candidate IDs without local videos: {','.join(missing_ids)}",
                flush=True,
            )
    else:
        rng = random.Random(args.seed)
        rng.shuffle(videos)
    videos = videos[: args.candidate_count]
    if not videos:
        raise RuntimeError("no VideoMME videos matched the metadata")

    print(
        f"[setup] candidates={len(videos)} seed={args.seed} output={output_dir}",
        flush=True,
    )
    tokenizer, model, image_processor, device = load_certvid_model(args)
    config = model.flashvid_config
    setattr(config, "certv3_certificate_budget_ratio", 0.0)
    setattr(config, "_capture_visualization_design", True)

    top_cases: list[ComparisonCase] = []
    audits: list[dict[str, Any]] = []
    for number, video_path in enumerate(videos, start=1):
        question_record = questions[video_path.stem][0]
        print(f"[scan] {number}/{len(videos)} {video_path.name}", flush=True)
        try:
            frames_np, _, _ = sample_video(video_path, args.num_frames)
            pixel_values_cpu = image_processor.preprocess(frames_np, return_tensors="pt")[
                "pixel_values"
            ]
            display_frames = tensor_frames_to_pil(pixel_values_cpu, image_processor)
            pixel_values = pixel_values_cpu.to(device=device, dtype=torch.float16)
            input_ids, attention_mask = prepare_prompt(
                tokenizer, question_record.prompt, device
            )
            prediction_raw, _, plan = generate_once(
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
            analysis = getattr(config, "_visualization_certvid_analysis", None)
            if plan is None or not isinstance(analysis, dict):
                raise RuntimeError("CertVID did not publish a visualization plan")
            if "design" not in analysis or "quality" not in analysis:
                raise RuntimeError("visualization capture is missing design or quality")

            design = analysis["design"].float().cpu()
            quality = analysis["quality"].float().reshape(-1).cpu()
            ours_indices = plan.anchor_indices.detach().long().cpu().sort().values
            budget = int(ours_indices.numel())
            topk_indices = (
                torch.argsort(quality, descending=True, stable=True)[:budget]
                .sort()
                .values
            )
            frame_count = len(display_frames)
            if design.ndim != 2 or len(design) != len(quality):
                raise RuntimeError("captured design and quality shapes do not match")
            if len(design) % frame_count:
                raise RuntimeError("visual-token count is not divisible by frame count")
            tokens_per_frame = len(design) // frame_count
            grid_height, grid_width = factor_grid(tokens_per_frame)
            ours_per_frame = _per_frame(ours_indices, frame_count, tokens_per_frame)
            topk_per_frame = _per_frame(topk_indices, frame_count, tokens_per_frame)
            comparison_frames = _parse_comparison_frames(
                args.comparison_frames, frame_count
            )
            if comparison_frames is None:
                comparison_frames = _auto_comparison_frames(
                    display_frames, ours_per_frame, topk_per_frame
                )

            ridge = float(analysis.get("ridge", 0.5))
            ours_summary = _information_summary(design, ours_indices, ridge)
            topk_summary = _information_summary(design, topk_indices, ridge)
            normalized_logdet_gain = (
                ours_summary.logdet_gain - topk_summary.logdet_gain
            ) / max(1, int(design.shape[1]))
            d_efficiency = math.exp(max(-30.0, min(30.0, normalized_logdet_gain)))
            overlap_ratio = float(
                torch.isin(ours_indices, topk_indices).float().mean().item()
            )
            entropy_gain = _normalized_entropy(
                ours_indices, frame_count, tokens_per_frame
            ) - _normalized_entropy(topk_indices, frame_count, tokens_per_frame)
            score = _case_score(
                display_frames,
                comparison_frames,
                ours_per_frame,
                topk_per_frame,
                d_efficiency,
                overlap_ratio,
                entropy_gain,
            )
            valid_labels = [label for label, _ in question_record.options]
            case = ComparisonCase(
                video_id=video_path.stem,
                video_path=video_path,
                question_id=question_record.question_id,
                question=question_record.question,
                options=[
                    {"label": label, "text": text}
                    for label, text in question_record.options
                ],
                target=question_record.answer,
                prediction=extract_answer_label(prediction_raw, valid_labels),
                raw_prediction=prediction_raw,
                frames=display_frames,
                quality=quality,
                ours_indices=ours_indices,
                topk_indices=topk_indices,
                comparison_frames=comparison_frames,
                frame_count=frame_count,
                tokens_per_frame=tokens_per_frame,
                grid_height=grid_height,
                grid_width=grid_width,
                d_efficiency=d_efficiency,
                overlap_ratio=overlap_ratio,
                selection_score=score,
            )
            audits.append(_audit_record(case, False))
            top_cases.append(case)
            top_cases.sort(key=lambda item: item.selection_score, reverse=True)
            del top_cases[args.num_examples :]
            print(
                f"[score] {video_path.stem} score={score:.4f} "
                f"D-eff={d_efficiency:.3f}x overlap={100.0 * overlap_ratio:.1f}% "
                f"frames={','.join(str(value + 1) for value in comparison_frames)}",
                flush=True,
            )
            del pixel_values, pixel_values_cpu, input_ids, attention_mask
        except Exception as error:
            print(f"[skip] {video_path.name}: {error}", flush=True)
            traceback.print_exc()
        finally:
            setattr(config, "_certvid_plan", None)
            setattr(config, "_visualization_certvid_analysis", None)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not top_cases:
        raise RuntimeError("no candidate produced a valid visualization capture")
    selected_ranks = {
        case.video_id: rank for rank, case in enumerate(top_cases, start=1)
    }
    for rank, case in enumerate(top_cases, start=1):
        stem_name = (
            f"certvid_equal_budget_retention_maps_{rank:02d}_{case.video_id}"
        )
        _plot(
            case,
            output_dir,
            args.filmstrip_frames,
            args.dpi,
            stem_name,
        )
        print(
            f"[figure] rank={rank} video={case.video_id} "
            f"file={output_dir / (stem_name + '.pdf')}",
            flush=True,
        )
    audits.sort(key=lambda item: float(item["selection_score"]), reverse=True)
    for record in audits:
        record["selected_for_figure"] = record["video_id"] in selected_ranks
        record["figure_rank"] = selected_ranks.get(record["video_id"])
    audit = {
        "figure": "Equal-budget Quality Top-K versus CertVID retention maps",
        "selection_policy": (
            "Candidates are ranked without answer correctness using D-efficiency, "
            "keep-set disagreement, temporal entropy, and frame visual detail."
        ),
        "seed": args.seed,
        "candidate_ids_file": (
            str(candidate_ids_path) if candidate_ids_path is not None else None
        ),
        "configuration": {
            "dataset": "VideoMME",
            "model": "LLaVA-OneVision-7B",
            "frames": args.num_frames,
            "retention_ratio": args.retention_ratio,
            "expansion": args.expansion,
            "pruning_layer": args.pruning_layer,
            "llm_retention_ratio": args.llm_retention_ratio,
            "certificate_budget_ratio": 0.0,
            "comparison_frames_argument": args.comparison_frames,
            "num_examples": args.num_examples,
        },
        "ranked_candidates": audits,
    }
    (output_dir / "certvid_equal_budget_retention_maps.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "[complete] selected="
        + ",".join(case.video_id for case in top_cases),
        flush=True,
    )


if __name__ == "__main__":
    main()
