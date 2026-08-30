#!/usr/bin/env python3
"""Render two real DOVE patch-level anchor examples from a video.

The script runs the repository's LLaVA-OneVision CertVID V3/DOVE path, reads
the actual anchor indices published by the compression plan, and exports two
paper-ready patch maps. It never synthesizes or hand-picks token locations.

Run from the repository root so the local ``flashvid`` and ``playground``
modules are importable.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance


ROOT = Path(__file__).resolve().parents[1]
PLAYGROUND = ROOT / "playground"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PLAYGROUND) not in sys.path:
    sys.path.insert(0, str(PLAYGROUND))

from visualize_certvid_two_examples import (  # noqa: E402
    factor_grid,
    generate_once,
    load_certvid_model,
    prepare_prompt,
    sample_video,
    tensor_frames_to_pil,
)


SELECTED_FILL = (22, 111, 65, 102)
SELECTED_EDGE = (10, 91, 49, 255)
DISCARDED_FILL = (235, 237, 234, 124)
GRID_COLOR = (255, 255, 255, 150)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-path", required=True, help="Real input video.")
    parser.add_argument("--question", required=True, help="Question used by DOVE.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model-path",
        default=os.environ.get(
            "PRETRAINED", "/home/xuyouwen/models/llava-onevision-qwen2-7b-ov"
        ),
        help="Resolved local LLaVA-OneVision checkpoint.",
    )
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--num-examples", type=int, default=2)
    parser.add_argument(
        "--frame-indices",
        default="auto",
        help="One-based sampled-frame indices, for example '8,21', or 'auto'.",
    )
    parser.add_argument("--retention-ratio", type=float, default=0.01)
    parser.add_argument("--expansion", type=float, default=1.30)
    parser.add_argument("--pruning-layer", type=int, default=20)
    parser.add_argument("--llm-retention-ratio", type=float, default=0.1923076923)
    parser.add_argument(
        "--pool-mode", choices=("bilinear", "average", "max"), default="bilinear"
    )
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--certificate-budget-ratio",
        type=float,
        default=0.0,
        help="Keep at 0.0 for the paper configuration.",
    )
    return parser.parse_args()


def _detail_score(frame: Image.Image) -> float:
    gray = np.asarray(frame.convert("L"), dtype=np.float32) / 255.0
    contrast = float(gray.std())
    dx = float(np.abs(np.diff(gray, axis=1)).mean()) if gray.shape[1] > 1 else 0.0
    dy = float(np.abs(np.diff(gray, axis=0)).mean()) if gray.shape[0] > 1 else 0.0
    return contrast + 2.5 * (dx + dy)


def _luminance(frame: Image.Image) -> float:
    return float(np.asarray(frame.convert("L"), dtype=np.float32).mean() / 255.0)


def _parse_manual_frames(specification: str, frame_count: int) -> list[int] | None:
    if specification.strip().lower() == "auto":
        return None
    result = [int(value.strip()) - 1 for value in specification.split(",") if value.strip()]
    if not result:
        raise ValueError("--frame-indices did not contain any frame index")
    if any(index < 0 or index >= frame_count for index in result):
        raise ValueError(
            f"--frame-indices must lie in [1, {frame_count}], got {specification!r}"
        )
    if len(set(result)) != len(result):
        raise ValueError("--frame-indices must be unique")
    return result


def _choose_frames(
    frames: list[Image.Image],
    per_frame_indices: list[list[int]],
    count: int,
) -> list[int]:
    """Choose clear, anchor-rich, temporally separated frames."""
    frame_count = len(frames)
    count = min(max(1, count), frame_count)
    details = np.asarray([_detail_score(frame) for frame in frames], dtype=np.float64)
    if float(np.ptp(details)) > 1e-12:
        details = (details - details.min()) / np.ptp(details)
    counts = np.asarray([len(values) for values in per_frame_indices], dtype=np.float64)
    if float(counts.max()) > 0.0:
        counts /= counts.max()
    luminance = np.asarray([_luminance(frame) for frame in frames], dtype=np.float64)
    usable = (luminance >= 0.10).astype(np.float64)
    score = 0.68 * counts + 0.22 * details + 0.10 * usable

    ranked = sorted(range(frame_count), key=lambda index: (score[index], -index), reverse=True)
    selected: list[int] = []
    minimum_separation = max(2, frame_count // (count + 2))
    for frame_index in ranked:
        if all(abs(frame_index - other) >= minimum_separation for other in selected):
            selected.append(frame_index)
        if len(selected) == count:
            break
    if len(selected) < count:
        selected.extend(index for index in ranked if index not in selected)
    return sorted(selected[:count])


def _overlay_patch_map(
    frame: Image.Image,
    selected_local: list[int],
    grid_height: int,
    grid_width: int,
) -> Image.Image:
    """Dim discarded patches and outline actual DOVE anchors in green."""
    base = ImageEnhance.Contrast(frame.convert("RGB")).enhance(1.03).convert("RGBA")
    width, height = base.size
    selected = set(int(value) for value in selected_local)
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    border_width = max(2, int(round(min(width, height) / 180)))
    selected_boxes: list[tuple[int, int, int, int]] = []

    for row in range(grid_height):
        y0 = int(round(row * height / grid_height))
        y1 = int(round((row + 1) * height / grid_height))
        for column in range(grid_width):
            x0 = int(round(column * width / grid_width))
            x1 = int(round((column + 1) * width / grid_width))
            token = row * grid_width + column
            if token in selected:
                draw.rectangle((x0, y0, x1, y1), fill=SELECTED_FILL)
                selected_boxes.append((x0, y0, x1, y1))
            else:
                draw.rectangle((x0, y0, x1, y1), fill=DISCARDED_FILL)

    # Draw the patch lattice first, then all four edges of every selected patch.
    for row in range(1, grid_height):
        y = int(round(row * height / grid_height))
        draw.line((0, y, width, y), fill=GRID_COLOR, width=1)
    for column in range(1, grid_width):
        x = int(round(column * width / grid_width))
        draw.line((x, 0, x, height), fill=GRID_COLOR, width=1)

    inset = max(1, (border_width + 1) // 2)
    for x0, y0, x1, y1 in selected_boxes:
        draw.rectangle(
            (
                x0 + inset,
                y0 + inset,
                max(x0 + inset, x1 - 1 - inset),
                max(y0 + inset, y1 - 1 - inset),
            ),
            outline=SELECTED_EDGE,
            width=border_width,
        )
    return Image.alpha_composite(base, overlay).convert("RGB")


def _save_panel(image: Image.Image, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, dpi=(dpi, dpi), optimize=True)
    image.save(path.with_suffix(".pdf"), "PDF", resolution=float(dpi))


def _combined_preview(images: list[Image.Image], output_path: Path, dpi: int) -> None:
    target_height = max(image.height for image in images)
    resized: list[Image.Image] = []
    for image in images:
        width = int(round(image.width * target_height / image.height))
        resized.append(image.resize((width, target_height), Image.Resampling.LANCZOS))
    gap = max(8, target_height // 35)
    canvas = Image.new(
        "RGB",
        (sum(image.width for image in resized) + gap * (len(resized) - 1), target_height),
        "white",
    )
    x = 0
    for image in resized:
        canvas.paste(image, (x, 0))
        x += image.width + gap
    _save_panel(canvas, output_path, dpi)


def main() -> None:
    args = parse_args()
    video_path = Path(args.video_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"video does not exist: {video_path}")
    if args.num_examples <= 0:
        raise ValueError("--num-examples must be positive")

    tokenizer, model, image_processor, device = load_certvid_model(args)
    config = model.flashvid_config
    setattr(config, "certv3_certificate_budget_ratio", args.certificate_budget_ratio)
    setattr(config, "certv3_use_trajectory", True)
    setattr(config, "certv3_use_query", True)
    setattr(config, "certv3_use_candidate_pool", True)
    setattr(config, "_capture_visualization_design", True)

    print(f"[video] {video_path}", flush=True)
    frames_np, source_indices, fps = sample_video(video_path, args.num_frames)
    pixel_values_cpu = image_processor.preprocess(frames_np, return_tensors="pt")[
        "pixel_values"
    ]
    display_frames = tensor_frames_to_pil(pixel_values_cpu, image_processor)
    pixel_values = pixel_values_cpu.to(device=device, dtype=torch.float16)
    input_ids, attention_mask = prepare_prompt(tokenizer, args.question, device)

    prediction, _, plan = generate_once(
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
    if plan is None:
        raise RuntimeError("DOVE did not publish an anchor plan")

    anchors = plan.anchor_indices.detach().long().cpu()
    raw_token_count = int(plan.raw_token_count)
    frame_count = len(display_frames)
    if raw_token_count % frame_count:
        raise RuntimeError(
            f"raw token count {raw_token_count} is not divisible by {frame_count} frames"
        )
    tokens_per_frame = raw_token_count // frame_count
    grid_height, grid_width = factor_grid(tokens_per_frame)
    per_frame_indices: list[list[int]] = [[] for _ in range(frame_count)]
    for anchor in anchors.tolist():
        frame_index, local_index = divmod(int(anchor), tokens_per_frame)
        if 0 <= frame_index < frame_count:
            per_frame_indices[frame_index].append(local_index)
    per_frame_indices = [sorted(values) for values in per_frame_indices]

    chosen_frames = _parse_manual_frames(args.frame_indices, frame_count)
    if chosen_frames is None:
        chosen_frames = _choose_frames(
            display_frames, per_frame_indices, args.num_examples
        )
    else:
        chosen_frames = chosen_frames[: args.num_examples]

    rendered: list[Image.Image] = []
    records: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for example_number, frame_index in enumerate(chosen_frames, start=1):
        image = _overlay_patch_map(
            display_frames[frame_index],
            per_frame_indices[frame_index],
            grid_height,
            grid_width,
        )
        rendered.append(image)
        output_path = output_dir / f"patch_example_{example_number:02d}.png"
        _save_panel(image, output_path, args.dpi)
        source_frame = int(source_indices[frame_index])
        timestamp = source_frame / fps if fps else None
        records.append(
            {
                "example": example_number,
                "sampled_frame_one_based": frame_index + 1,
                "source_frame_zero_based": source_frame,
                "timestamp_seconds": timestamp,
                "selected_local_patch_indices": per_frame_indices[frame_index],
                "selected_patch_count": len(per_frame_indices[frame_index]),
                "grid_height": grid_height,
                "grid_width": grid_width,
                "png": str(output_path),
            }
        )
        print(
            f"[saved] {output_path} frame={frame_index + 1} "
            f"selected={len(per_frame_indices[frame_index])}",
            flush=True,
        )

    _combined_preview(rendered, output_dir / "patch_examples_combined.png", args.dpi)
    audit = {
        "video_path": str(video_path),
        "question": args.question,
        "prediction": prediction,
        "num_sampled_frames": frame_count,
        "raw_token_count": raw_token_count,
        "anchor_count": int(anchors.numel()),
        "retention_ratio": args.retention_ratio,
        "expansion": args.expansion,
        "certificate_budget_ratio": args.certificate_budget_ratio,
        "actual_dove_anchor_indices": anchors.tolist(),
        "examples": records,
    }
    (output_dir / "patch_examples.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    del pixel_values, pixel_values_cpu, input_ids, attention_mask, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[complete] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
