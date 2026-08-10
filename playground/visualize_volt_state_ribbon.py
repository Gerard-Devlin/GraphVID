#!/usr/bin/env python3
"""Visualize CertVID V3 question relevance on its most relevant video frame."""

from __future__ import annotations

import argparse
import json
import os
import random
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import torch

from visualize_certvid_two_examples import (
    discover_videos,
    extract_answer_label,
    generate_once,
    load_certvid_model,
    load_questions,
    prepare_prompt,
    sample_video,
    tensor_frames_to_pil,
)


def parse_args() -> argparse.Namespace:
    hf_home = Path(os.environ.get("HF_HOME", "/home/xuyouwen/hf_home_local"))
    parser = argparse.ArgumentParser(
        description="Visualize CertVID V3 query relevance without hard certificates."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-root", default=str(hf_home / "videomme" / "data"))
    parser.add_argument("--metadata-jsonl", required=True)
    parser.add_argument("--video-id", default="")
    parser.add_argument("--num-examples", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional deterministic sampling seed; omit it for a fresh random video.",
    )
    parser.add_argument("--retention-ratio", type=float, default=0.01)
    parser.add_argument("--expansion", type=float, default=1.30)
    parser.add_argument("--pruning-layer", type=int, default=20)
    parser.add_argument("--llm-retention-ratio", type=float, default=0.1923076923)
    parser.add_argument(
        "--pool-mode", choices=("bilinear", "average", "max"), default="bilinear"
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _choose_videos(
    root: Path,
    questions: dict[str, Any],
    video_id: str,
    count: int,
    seed: int | None,
) -> list[Path]:
    videos = discover_videos(root)
    by_id = {path.stem: path for path in videos}
    requested = [Path(value.strip()).stem for value in video_id.split(",") if value.strip()]
    if requested:
        missing = [value for value in requested if value not in by_id]
        if missing:
            raise FileNotFoundError(f"videos not found: {missing}")
        return [by_id[value] for value in requested]
    eligible = [path for path in videos if path.stem in questions]
    random.Random(seed).shuffle(eligible)
    return eligible[:count]


def _run_certvid(
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
) -> tuple[str, np.ndarray, dict[str, Any]]:
    config = model.flashvid_config
    config.certv3_certificate_budget_ratio = 0.0
    config.certv3_selection_objective = "d_optimal"
    config.certv3_use_trajectory = True
    setattr(config, "_capture_visualization_design", True)
    setattr(config, "_capture_visualization_trajectory", True)
    prediction, _, plan = generate_once(
        model=model,
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        retention_ratio=retention_ratio,
        expansion=expansion,
        llm_retention_ratio=llm_retention_ratio,
        max_new_tokens=max_new_tokens,
    )
    analysis = getattr(config, "_visualization_certvid_analysis", None)
    if plan is None or not isinstance(analysis, dict):
        raise RuntimeError("CertVID plan or visualization sidecar is missing")
    copied = {
        key: value.detach().cpu().clone() if torch.is_tensor(value) else value
        for key, value in analysis.items()
    }
    indices = plan.anchor_indices.detach().long().cpu().numpy().copy()
    setattr(config, "_certvid_plan", None)
    setattr(config, "_visualization_certvid_analysis", None)
    return prediction, indices, copied


def _tensor(analysis: dict[str, Any], name: str, dtype) -> np.ndarray:
    value = analysis.get(name)
    if not torch.is_tensor(value):
        raise RuntimeError(f"visualization sidecar is missing {name!r}")
    return value.numpy().astype(dtype, copy=False)


def _focus_frame(query_score: np.ndarray) -> tuple[int, np.ndarray]:
    tokens_per_frame = int(query_score.shape[1])
    topk = max(1, int(round(0.08 * tokens_per_frame)))
    partitioned = np.partition(query_score, tokens_per_frame - topk, axis=1)
    frame_relevance = partitioned[:, -topk:].mean(axis=1)
    return int(np.argmax(frame_relevance)), frame_relevance


def _plot_relevance(
    *,
    output_path: Path,
    frame: Any,
    relevance: np.ndarray,
    relevance_low: float,
    relevance_high: float,
    grid_height: int,
    grid_width: int,
    question: str,
    frame_index: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = np.asarray(frame)
    relevance_grid = relevance.reshape(grid_height, grid_width)
    normalized = np.clip(
        (relevance_grid - relevance_low)
        / max(1e-12, relevance_high - relevance_low),
        0.0,
        1.0,
    )
    # Keep ordinary patches cool while preserving the ordering of rare peaks.
    normalized = normalized**1.6

    fig = plt.figure(figsize=(6.0, 6.7), facecolor="white")
    ax = fig.add_axes((0.04, 0.19, 0.92, 0.70))
    extent = (0, image.shape[1], image.shape[0], 0)
    ax.imshow(image)
    ax.imshow(
        normalized,
        extent=extent,
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
        cmap="coolwarm",
        alpha=0.50,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.text(
        0.5,
        0.935,
        f"CertVID V3 relevance | Frame {frame_index + 1} | "
        "Blue: low, Red: high",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#17212B",
    )
    fig.text(
        0.5,
        0.085,
        textwrap.fill(question, width=60),
        ha="center",
        va="center",
        fontsize=15,
        fontfamily="serif",
        color="#111111",
    )
    fig.savefig(output_path, dpi=240, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    questions = load_questions(Path(args.metadata_jsonl).expanduser().resolve())
    videos = _choose_videos(
        Path(args.dataset_root).expanduser().resolve(),
        questions,
        args.video_id,
        args.num_examples,
        args.seed,
    )
    if len(videos) < args.num_examples:
        raise RuntimeError(f"only found {len(videos)}/{args.num_examples} videos")

    tokenizer, model, image_processor, device = load_certvid_model(args)
    records: list[dict[str, Any]] = []
    for number, video_path in enumerate(videos, start=1):
        record = questions[video_path.stem][0]
        print(f"[run] {number}/{len(videos)} video={video_path.name}", flush=True)
        sampled, source_indices, fps = sample_video(video_path, args.num_frames)
        pixels_cpu = image_processor.preprocess(sampled, return_tensors="pt")["pixel_values"]
        display_frames = tensor_frames_to_pil(pixels_cpu, image_processor)
        pixels = pixels_cpu.to(device=device, dtype=torch.float16)
        input_ids, attention_mask = prepare_prompt(tokenizer, record.prompt, device)

        prediction_raw, selected, analysis = _run_certvid(
            model=model,
            tokenizer=tokenizer,
            pixel_values=pixels,
            input_ids=input_ids,
            attention_mask=attention_mask,
            retention_ratio=args.retention_ratio,
            expansion=args.expansion,
            llm_retention_ratio=args.llm_retention_ratio,
            max_new_tokens=args.max_new_tokens,
        )

        frame_count = int(analysis["frame_count"])
        tokens_per_frame = int(analysis["tokens_per_frame"])
        grid_height = int(analysis["grid_height"])
        grid_width = int(analysis["grid_width"])
        if len(display_frames) != frame_count:
            raise RuntimeError("display-frame count does not match CertVID geometry")
        if grid_height * grid_width != tokens_per_frame:
            raise RuntimeError("patch grid does not match tokens_per_frame")
        query_score = _tensor(analysis, "query_score", np.float32)
        expected_tokens = frame_count * tokens_per_frame
        if query_score.size != expected_tokens:
            raise RuntimeError("captured query scores do not match the visual grid")
        query_matrix = query_score.reshape(frame_count, tokens_per_frame)
        focus_frame, frame_relevance = _focus_frame(query_matrix)
        relevance_low = float(np.quantile(query_matrix, 0.85))
        relevance_high = float(np.quantile(query_matrix, 0.995))
        if relevance_high - relevance_low <= 1e-12:
            relevance_low = float(query_matrix.min())
            relevance_high = float(query_matrix.max()) + 1e-12
        query_confidence = float(analysis.get("query_confidence", 0.0))
        labels = [label for label, _ in record.options]
        prediction = extract_answer_label(prediction_raw, labels)
        image_name = f"certvid_relevance_{number:02d}_{video_path.stem}.png"
        _plot_relevance(
            output_path=output_dir / image_name,
            frame=display_frames[focus_frame],
            relevance=query_matrix[focus_frame],
            relevance_low=relevance_low,
            relevance_high=relevance_high,
            grid_height=grid_height,
            grid_width=grid_width,
            question=record.question,
            frame_index=focus_frame,
        )
        records.append(
            {
                "video_id": video_path.stem,
                "question_id": record.question_id,
                "question": record.question,
                "options": [{"label": a, "text": b} for a, b in record.options],
                "target": record.answer,
                "prediction": prediction,
                "correct": prediction == record.answer,
                "raw_tokens": frame_count * tokens_per_frame,
                "selected_tokens": int(selected.size),
                "certificate_budget_ratio": 0.0,
                "query_confidence": query_confidence,
                "focus_frame_relevance": float(frame_relevance[focus_frame]),
                "visualization_relevance_low": relevance_low,
                "visualization_relevance_high": relevance_high,
                "focus_sampled_frame": focus_frame,
                "focus_source_frame": int(source_indices[focus_frame]),
                "video_fps": fps,
                "figure": image_name,
            }
        )
        (output_dir / "certvid_relevance_metadata.json").write_text(
            json.dumps({"examples": records}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        del pixels, pixels_cpu, input_ids, attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[done] figure={image_name}", flush=True)

    print(f"[complete] outputs={output_dir}", flush=True)


if __name__ == "__main__":
    main()
