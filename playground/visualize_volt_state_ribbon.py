#!/usr/bin/env python3
"""Create a paper-ready spatial token-demand comparison for VOLT-Vid.

The script compares the complete CertVID V3 selector (hard certificates off)
against the exact same selector with trajectory signals disabled. Both variants
use the same model, video, question, token budget, and D-optimal objective.
"""

from __future__ import annotations

import argparse
import json
import os
import random
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
        description="Compare spatial token demand with and without trajectory structure."
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


def _run_variant(
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
    use_trajectory: bool,
) -> tuple[str, np.ndarray, dict[str, Any]]:
    config = model.flashvid_config
    config.certv3_certificate_budget_ratio = 0.0
    config.certv3_selection_objective = "d_optimal"
    config.certv3_use_trajectory = bool(use_trajectory)
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


def _focus_frame(
    full_demand: np.ndarray,
    no_trajectory_demand: np.ndarray,
    state_change: np.ndarray,
    frame_event: np.ndarray,
) -> int:
    frame_count = int(full_demand.shape[0])
    distribution_shift = np.abs(full_demand - no_trajectory_demand).sum(axis=1)
    state_score = state_change.max(axis=1)

    def normalize(values: np.ndarray) -> np.ndarray:
        low = float(values.min())
        high = float(values.max())
        return (values - low) / max(1e-12, high - low)

    score = (
        0.60 * normalize(distribution_shift)
        + 0.25 * normalize(state_score)
        + 0.15 * normalize(frame_event)
    )
    return int(np.argmax(score[:frame_count]))


def _plot_demand_panel(
    ax,
    frame: Any,
    demand_grid: np.ndarray,
    low: float,
    high: float,
):
    image = np.asarray(frame)
    normalized = np.clip((demand_grid - low) / max(1e-12, high - low), 0.0, 1.0)
    extent = (0, image.shape[1], image.shape[0], 0)
    ax.imshow(image)
    heatmap = ax.imshow(
        normalized,
        extent=extent,
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
        cmap="turbo",
        alpha=0.54,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return heatmap


def _plot_spatial_comparison(
    *,
    output_path: Path,
    frame: Any,
    frame_index: int,
    full_demand: np.ndarray,
    no_trajectory_demand: np.ndarray,
    frame_event: np.ndarray,
    grid_height: int,
    grid_width: int,
    question: str,
    video_id: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    full_grid = full_demand.reshape(grid_height, grid_width)
    no_grid = no_trajectory_demand.reshape(grid_height, grid_width)
    compared = np.concatenate([full_grid.reshape(-1), no_grid.reshape(-1)])
    low = float(np.quantile(compared, 0.02))
    high = float(np.quantile(compared, 0.98))
    if high - low <= 1e-12:
        low = float(compared.min())
        high = float(compared.max()) + 1e-12

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.7), constrained_layout=True)
    panels = (
        (no_grid, "(a) Without trajectory structure"),
        (full_grid, "(b) Ours: trajectory-aware"),
    )
    for ax, (demand, label) in zip(axes, panels):
        _plot_demand_panel(ax, frame, demand, low, high)
        ax.set_title(label, fontsize=12, fontweight="bold", color="#17212B", pad=9)

    fig.suptitle(
        f"Blue: low token demand     Red: high token demand\n"
        f"Frame {frame_index + 1} | Event {frame_event[frame_index]:.2f} | {video_id}",
        fontsize=11,
        color="#17212B",
    )
    fig.text(
        0.5,
        0.015,
        question,
        ha="center",
        va="bottom",
        fontsize=15,
        fontfamily="serif",
        color="#111111",
    )
    fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
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

        no_prediction_raw, no_selected, no_analysis = _run_variant(
            model=model,
            tokenizer=tokenizer,
            pixel_values=pixels,
            input_ids=input_ids,
            attention_mask=attention_mask,
            retention_ratio=args.retention_ratio,
            expansion=args.expansion,
            llm_retention_ratio=args.llm_retention_ratio,
            max_new_tokens=args.max_new_tokens,
            use_trajectory=False,
        )
        full_prediction_raw, full_selected, analysis = _run_variant(
            model=model,
            tokenizer=tokenizer,
            pixel_values=pixels,
            input_ids=input_ids,
            attention_mask=attention_mask,
            retention_ratio=args.retention_ratio,
            expansion=args.expansion,
            llm_retention_ratio=args.llm_retention_ratio,
            max_new_tokens=args.max_new_tokens,
            use_trajectory=True,
        )

        frame_count = int(analysis["frame_count"])
        tokens_per_frame = int(analysis["tokens_per_frame"])
        grid_height = int(analysis["grid_height"])
        grid_width = int(analysis["grid_width"])
        if len(display_frames) != frame_count:
            raise RuntimeError("display-frame count does not match CertVID geometry")
        if grid_height * grid_width != tokens_per_frame:
            raise RuntimeError("patch grid does not match tokens_per_frame")
        if full_selected.size != no_selected.size:
            raise RuntimeError("full and no-trajectory token budgets differ")
        no_frame_count = int(no_analysis["frame_count"])
        no_tokens_per_frame = int(no_analysis["tokens_per_frame"])
        if (no_frame_count, no_tokens_per_frame) != (frame_count, tokens_per_frame):
            raise RuntimeError("full and no-trajectory visual geometries differ")
        novelty = _tensor(analysis, "novelty", np.float32)
        curvature = _tensor(analysis, "curvature", np.float32)
        frame_event = _tensor(analysis, "frame_event", np.float32)
        full_demand = _tensor(analysis, "demand_weight", np.float32)
        no_demand = _tensor(no_analysis, "demand_weight", np.float32)
        expected_tokens = frame_count * tokens_per_frame
        if full_demand.size != expected_tokens or no_demand.size != expected_tokens:
            raise RuntimeError("captured demand weights do not match the visual grid")
        full_demand_matrix = full_demand.reshape(frame_count, tokens_per_frame)
        no_demand_matrix = no_demand.reshape(frame_count, tokens_per_frame)
        state_change = np.maximum(novelty, curvature).reshape(
            frame_count, tokens_per_frame
        )
        focus_frame = _focus_frame(
            full_demand_matrix,
            no_demand_matrix,
            state_change,
            frame_event,
        )
        labels = [label for label, _ in record.options]
        full_prediction = extract_answer_label(full_prediction_raw, labels)
        no_prediction = extract_answer_label(no_prediction_raw, labels)
        image_name = f"trajectory_relevance_{number:02d}_{video_path.stem}.png"
        _plot_spatial_comparison(
            output_path=output_dir / image_name,
            frame=display_frames[focus_frame],
            frame_index=focus_frame,
            full_demand=full_demand_matrix[focus_frame],
            no_trajectory_demand=no_demand_matrix[focus_frame],
            frame_event=frame_event,
            grid_height=grid_height,
            grid_width=grid_width,
            question=record.question,
            video_id=video_path.stem,
        )
        records.append(
            {
                "video_id": video_path.stem,
                "question_id": record.question_id,
                "question": record.question,
                "options": [{"label": a, "text": b} for a, b in record.options],
                "target": record.answer,
                "full_prediction": full_prediction,
                "without_trajectory_prediction": no_prediction,
                "full_correct": full_prediction == record.answer,
                "without_trajectory_correct": no_prediction == record.answer,
                "raw_tokens": frame_count * tokens_per_frame,
                "selected_tokens": int(full_selected.size),
                "certificate_budget_ratio": 0.0,
                "focus_sampled_frame": focus_frame,
                "focus_source_frame": int(source_indices[focus_frame]),
                "video_fps": fps,
                "figure": image_name,
            }
        )
        (output_dir / "trajectory_relevance_metadata.json").write_text(
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
