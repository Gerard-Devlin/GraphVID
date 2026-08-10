#!/usr/bin/env python3
"""Visualize VOLT-Vid's real trajectory geometry and dynamic signals.

The script runs the ordinary CertVID V3 implementation with hard certificates
disabled, captures its opt-in analysis sidecar, and renders two paper-ready
figures: a trajectory overlay across a contiguous sampled-frame window and
aligned novelty/curvature/support/event diagnostics. No signal is synthesized.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ComponentSummary:
    component_id: int
    members: np.ndarray
    frames: np.ndarray
    score: float
    support: float
    max_novelty: float
    max_curvature: float
    selected_count: int


def parse_args() -> argparse.Namespace:
    hf_home = Path(os.environ.get("HF_HOME", "/home/xuyouwen/hf_home_local"))
    parser = argparse.ArgumentParser(
        description="Visualize VOLT-Vid trajectory links and dynamic signals."
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get(
            "PRETRAINED", "/home/xuyouwen/models/llava-onevision-qwen2-7b-ov"
        ),
    )
    parser.add_argument(
        "--dataset-root", default=str(hf_home / "videomme" / "data")
    )
    parser.add_argument("--metadata-jsonl", required=True)
    parser.add_argument(
        "--video-id",
        default="",
        help="Optional comma-separated video stems. Random matching videos are used otherwise.",
    )
    parser.add_argument("--num-examples", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--filmstrip-frames", type=int, default=8)
    parser.add_argument("--top-components", type=int, default=5)
    parser.add_argument("--min-component-frames", type=int, default=2)
    parser.add_argument("--heatmap-alpha", type=float, default=0.58)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--retention-ratio", type=float, default=0.10)
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


def _requested_ids(raw: str) -> list[str]:
    values = [Path(value.strip()).stem for value in raw.split(",") if value.strip()]
    if len(values) != len(set(values)):
        raise ValueError("--video-id values must be unique")
    return values


def _choose_videos(
    dataset_root: Path,
    questions: dict[str, Any],
    requested: list[str],
    count: int,
    seed: int,
) -> list[Path]:
    videos = discover_videos(dataset_root)
    by_id = {path.stem: path for path in videos}
    if requested:
        missing = [video_id for video_id in requested if video_id not in by_id]
        if missing:
            raise FileNotFoundError(f"videos not found: {missing}")
        return [by_id[video_id] for video_id in requested]
    eligible = [path for path in videos if path.stem in questions]
    random.Random(seed).shuffle(eligible)
    return eligible[:count]


def _component_summaries(
    component_ids: np.ndarray,
    component_support: np.ndarray,
    novelty: np.ndarray,
    curvature: np.ndarray,
    selected_mask: np.ndarray,
    tokens_per_frame: int,
    frame_count: int,
) -> list[ComponentSummary]:
    summaries: list[ComponentSummary] = []
    for component_id in np.unique(component_ids):
        members = np.flatnonzero(component_ids == component_id)
        frames = np.unique(members // tokens_per_frame)
        support = float(component_support[members].mean())
        max_novelty = float(novelty[members].max())
        max_curvature = float(curvature[members].max())
        selected_count = int(selected_mask[members].sum())
        span = float(len(frames)) / max(1, frame_count)
        selection_coverage = float(selected_count) / max(1, len(frames))
        score = (
            0.34 * span
            + 0.25 * support
            + 0.18 * max_novelty
            + 0.15 * max_curvature
            + 0.08 * min(1.0, selection_coverage)
        )
        summaries.append(
            ComponentSummary(
                component_id=int(component_id),
                members=members,
                frames=frames,
                score=score,
                support=support,
                max_novelty=max_novelty,
                max_curvature=max_curvature,
                selected_count=selected_count,
            )
        )
    return sorted(summaries, key=lambda item: (-item.score, item.component_id))


def _best_window(
    frame_event: np.ndarray,
    novelty: np.ndarray,
    curvature: np.ndarray,
    selected_mask: np.ndarray,
    frame_count: int,
    tokens_per_frame: int,
    window_size: int,
) -> list[int]:
    window_size = min(frame_count, max(2, window_size))
    novelty_2d = novelty.reshape(frame_count, tokens_per_frame)
    curvature_2d = curvature.reshape(frame_count, tokens_per_frame)
    selected_2d = selected_mask.reshape(frame_count, tokens_per_frame)
    dynamic = (
        0.45 * frame_event
        + 0.25 * novelty_2d.max(axis=1)
        + 0.20 * curvature_2d.max(axis=1)
        + 0.10
        * selected_2d.sum(axis=1)
        / max(1.0, float(selected_2d.sum(axis=1).max()))
    )
    scores = np.convolve(dynamic, np.ones(window_size), mode="valid")
    start = int(np.argmax(scores)) if scores.size else 0
    return list(range(start, start + window_size))


def _top_window_components(
    summaries: list[ComponentSummary],
    window: list[int],
    top_count: int,
    min_frames: int,
) -> list[ComponentSummary]:
    window_set = set(window)
    ranked = sorted(
        summaries,
        key=lambda item: (
            -sum(int(frame) in window_set for frame in item.frames),
            -item.score,
            item.component_id,
        ),
    )
    eligible = [
        item
        for item in ranked
        if sum(int(frame) in window_set for frame in item.frames) >= min_frames
    ]
    if not eligible:
        eligible = [
            item for item in ranked if any(int(frame) in window_set for frame in item.frames)
        ]
    return eligible[: max(1, top_count)]


def _short_text(text: str, limit: int) -> str:
    value = " ".join(text.split())
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _plot_trajectory_graph(
    output_path: Path,
    frames: list[Any],
    window: list[int],
    components: list[ComponentSummary],
    frame_event: np.ndarray,
    selected_mask: np.ndarray,
    component_ids: np.ndarray,
    match_previous: np.ndarray,
    match_mutual: np.ndarray,
    match_similarity: np.ndarray,
    grid_height: int,
    grid_width: int,
    tokens_per_frame: int,
    question: str,
    video_id: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    teal = "#00AEB3"
    red = "#E64B35"
    ink = "#17212B"
    colors = plt.get_cmap("Dark2")(np.linspace(0.0, 1.0, max(3, len(components))))
    gap = 0.075
    panel_width = 1.0
    total_width = len(window) * panel_width + (len(window) - 1) * gap
    fig, ax = plt.subplots(figsize=(max(12.0, len(window) * 2.05), 4.7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    frame_x: dict[int, float] = {}
    for position, frame_idx in enumerate(window):
        x0 = position * (panel_width + gap)
        frame_x[frame_idx] = x0
        ax.imshow(frames[frame_idx], extent=(x0, x0 + panel_width, 0.0, 1.0))
        event_value = float(frame_event[frame_idx])
        ax.add_patch(
            plt.Rectangle(
                (x0, 0.0),
                panel_width,
                1.0,
                fill=False,
                edgecolor=red,
                linewidth=0.7 + 3.0 * event_value,
                alpha=0.25 + 0.65 * event_value,
            )
        )
        ax.text(
            x0 + panel_width / 2,
            -0.055,
            f"Frame {frame_idx + 1}\nEvent {event_value:.2f}",
            ha="center",
            va="top",
            fontsize=8,
            color=ink,
        )

    displayed_components = {
        component.component_id: rank for rank, component in enumerate(components)
    }
    visible_nodes: dict[int, set[int]] = {
        component.component_id: set() for component in components
    }
    for frame_idx in window[1:]:
        match_row = frame_idx - 1
        if match_row >= match_previous.shape[0]:
            continue
        for local in range(tokens_per_frame):
            if not bool(match_mutual[match_row, local]):
                continue
            current_global = frame_idx * tokens_per_frame + local
            component_id = int(component_ids[current_global])
            rank = displayed_components.get(component_id)
            if rank is None:
                continue
            previous_local = int(match_previous[match_row, local])
            previous_global = (frame_idx - 1) * tokens_per_frame + previous_local
            # Equal component ids mean this exact mutual edge survived CertVID's
            # similarity and event-boundary gates when components were built.
            if int(component_ids[previous_global]) != component_id:
                continue
            previous_row, previous_col = divmod(previous_local, grid_width)
            current_row, current_col = divmod(local, grid_width)
            x0 = frame_x[frame_idx - 1] + (previous_col + 0.5) / grid_width
            y0 = 1.0 - (previous_row + 0.5) / grid_height
            x1 = frame_x[frame_idx] + (current_col + 0.5) / grid_width
            y1 = 1.0 - (current_row + 0.5) / grid_height
            similarity = float(match_similarity[match_row, local])
            support = components[rank].support
            ax.plot(
                [x0, x1],
                [y0, y1],
                color=colors[rank],
                linewidth=1.4 + 2.0 * support,
                alpha=0.42 + 0.48 * max(0.0, min(1.0, similarity)),
                zorder=5,
            )
            visible_nodes[component_id].update((previous_global, current_global))

    for rank, component in enumerate(components):
        color = colors[rank]
        points: list[tuple[int, int, float, float]] = []
        for global_index in sorted(visible_nodes[component.component_id]):
            frame_idx, local = divmod(int(global_index), tokens_per_frame)
            if frame_idx not in frame_x:
                continue
            row, col = divmod(local, grid_width)
            x = frame_x[frame_idx] + (col + 0.5) / grid_width
            y = 1.0 - (row + 0.5) / grid_height
            points.append((frame_idx, int(global_index), x, y))
        points.sort()
        for _, global_index, x, y in points:
            if selected_mask[global_index]:
                ax.scatter(
                    [x], [y], s=54, marker="s", color=teal, edgecolors="white",
                    linewidths=0.9, zorder=9,
                )
            else:
                ax.scatter(
                    [x], [y], s=19, facecolors="white", edgecolors=[color],
                    linewidths=1.15, alpha=0.95, zorder=8,
                )

    legend = [
        Line2D([0], [0], color=colors[0], linewidth=3, label="Mutual token match"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=teal,
               markeredgecolor="white", markersize=8, label="Selected anchor"),
        Line2D([0], [0], color=red, linewidth=2.5, alpha=0.7,
               label="Frame event strength"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 1.10),
              ncol=3, frameon=False, fontsize=9)
    ax.set_xlim(-0.02, total_width + 0.02)
    ax.set_ylim(-0.16, 1.02)
    ax.axis("off")
    fig.suptitle(
        f"Trajectory-aware visual token geometry | {video_id}\n{_short_text(question, 150)}",
        fontsize=12,
        fontweight="bold",
        color=ink,
        y=0.995,
    )
    fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _focus_frame(
    frame_event: np.ndarray,
    novelty: np.ndarray,
    curvature: np.ndarray,
    component_support: np.ndarray,
    frame_count: int,
    tokens_per_frame: int,
) -> int:
    def top_mean(values: np.ndarray) -> np.ndarray:
        matrix = values.reshape(frame_count, tokens_per_frame)
        count = max(1, int(np.ceil(tokens_per_frame * 0.10)))
        partitioned = np.partition(matrix, tokens_per_frame - count, axis=1)
        return partitioned[:, -count:].mean(axis=1)

    score = (
        0.36 * frame_event
        + 0.28 * top_mean(novelty)
        + 0.22 * top_mean(curvature)
        + 0.14 * top_mean(component_support)
    )
    return int(np.argmax(score))


def _signal_grid(
    values: np.ndarray,
    frame_idx: int,
    tokens_per_frame: int,
    grid_height: int,
    grid_width: int,
) -> np.ndarray:
    start = frame_idx * tokens_per_frame
    return np.clip(
        values[start : start + tokens_per_frame].reshape(grid_height, grid_width),
        0.0,
        1.0,
    )


def _plot_signal_maps(
    output_path: Path,
    frame: Any,
    novelty: np.ndarray,
    curvature: np.ndarray,
    component_support: np.ndarray,
    demand_weight: np.ndarray,
    frame_event: np.ndarray,
    selected_mask: np.ndarray,
    focus_frame: int,
    tokens_per_frame: int,
    grid_height: int,
    grid_width: int,
    heatmap_alpha: float,
    demand_label: str,
    question: str,
    video_id: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    novelty_grid = _signal_grid(
        novelty, focus_frame, tokens_per_frame, grid_height, grid_width
    )
    curvature_grid = _signal_grid(
        curvature, focus_frame, tokens_per_frame, grid_height, grid_width
    )
    support_grid = _signal_grid(
        component_support, focus_frame, tokens_per_frame, grid_height, grid_width
    )
    event_value = float(frame_event[focus_frame])
    demand_grid = _signal_grid(
        demand_weight, focus_frame, tokens_per_frame, grid_height, grid_width
    )
    maps = [
        ("(a) Novelty", novelty_grid),
        ("(b) Curvature", curvature_grid),
        ("(c) Trajectory support", support_grid),
        (f"(d) {demand_label}", demand_grid),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(15.6, 4.1), constrained_layout=True)
    ink = "#17212B"
    teal = "#00AEB3"
    frame_array = np.asarray(frame)
    extent = (0, frame_array.shape[1], frame_array.shape[0], 0)
    alpha = min(0.90, max(0.0, float(heatmap_alpha)))
    for panel, (ax, (title, matrix)) in enumerate(zip(axes, maps)):
        ax.imshow(frame_array)
        image = ax.imshow(
            matrix,
            extent=extent,
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
            cmap="turbo",
            alpha=alpha,
        )
        ax.set_title(title, fontweight="bold", color=ink)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if panel == 3:
            frame_selected = selected_mask.reshape(-1, tokens_per_frame)[focus_frame]
            for local in np.flatnonzero(frame_selected):
                row, col = divmod(int(local), grid_width)
                width = frame_array.shape[1] / grid_width
                height = frame_array.shape[0] / grid_height
                ax.add_patch(
                    plt.Rectangle(
                        (col * width, row * height),
                        width,
                        height,
                        fill=False,
                        edgecolor=teal,
                        linewidth=1.0,
                    )
                )
    colorbar = fig.colorbar(image, ax=axes, fraction=0.018, pad=0.012)
    colorbar.set_label("Normalized signal", color=ink)
    fig.suptitle(
        f"Spatial trajectory evidence | {video_id} | sampled frame {focus_frame + 1} "
        f"| event strength {event_value:.2f}\n{_short_text(question, 150)}",
        fontsize=12,
        fontweight="bold",
        color=ink,
    )
    fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _tensor_numpy(analysis: dict[str, Any], key: str, dtype=None) -> np.ndarray:
    value = analysis.get(key)
    if not torch.is_tensor(value):
        raise RuntimeError(f"visualization capture is missing tensor {key!r}")
    array = value.detach().cpu().numpy()
    return array.astype(dtype, copy=False) if dtype is not None else array


def main() -> None:
    args = parse_args()
    if args.num_examples <= 0:
        raise ValueError("--num-examples must be positive")
    requested = _requested_ids(args.video_id)
    if requested and len(requested) != args.num_examples:
        raise ValueError("--video-id count must equal --num-examples")

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(args.metadata_jsonl).expanduser().resolve()
    questions = load_questions(metadata_path)
    videos = _choose_videos(
        Path(args.dataset_root).expanduser().resolve(),
        questions,
        requested,
        args.num_examples,
        args.seed,
    )
    if len(videos) < args.num_examples:
        raise RuntimeError(f"only found {len(videos)}/{args.num_examples} matching videos")

    tokenizer, model, image_processor, device = load_certvid_model(args)
    config = model.flashvid_config
    config.certv3_certificate_budget_ratio = 0.0
    setattr(config, "_capture_visualization_design", True)
    setattr(config, "_capture_visualization_trajectory", True)
    records: list[dict[str, Any]] = []

    for number, video_path in enumerate(videos, start=1):
        question_record = questions[video_path.stem][0]
        print(f"[run] {number}/{len(videos)} video={video_path.name}", flush=True)
        sampled, source_indices, fps = sample_video(video_path, args.num_frames)
        pixels_cpu = image_processor.preprocess(sampled, return_tensors="pt")["pixel_values"]
        display_frames = tensor_frames_to_pil(pixels_cpu, image_processor)
        pixels = pixels_cpu.to(device=device, dtype=torch.float16)
        input_ids, attention_mask = prepare_prompt(tokenizer, question_record.prompt, device)
        prediction, _, plan = generate_once(
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
        analysis = getattr(config, "_visualization_certvid_analysis", None)
        if plan is None or not isinstance(analysis, dict):
            raise RuntimeError("CertVID plan or trajectory analysis capture is missing")

        frame_count = int(analysis["frame_count"])
        tokens_per_frame = int(analysis["tokens_per_frame"])
        grid_height = int(analysis["grid_height"])
        grid_width = int(analysis["grid_width"])
        if len(display_frames) != frame_count or grid_height * grid_width != tokens_per_frame:
            raise RuntimeError("captured trajectory geometry does not match sampled frames")

        component_ids = _tensor_numpy(analysis, "component_ids", np.int64)
        component_support = _tensor_numpy(analysis, "component_support", np.float32)
        demand_tensor = analysis.get("demand_weight")
        demand_weight = (
            demand_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            if torch.is_tensor(demand_tensor)
            else None
        )
        novelty = _tensor_numpy(analysis, "novelty", np.float32)
        curvature = _tensor_numpy(analysis, "curvature", np.float32)
        frame_event = _tensor_numpy(analysis, "frame_event", np.float32)
        match_previous = _tensor_numpy(analysis, "match_previous", np.int64)
        match_mutual = _tensor_numpy(analysis, "match_mutual", bool)
        match_similarity = _tensor_numpy(analysis, "match_similarity", np.float32)
        selected = plan.anchor_indices.detach().long().cpu().numpy()
        selected_mask = np.zeros(frame_count * tokens_per_frame, dtype=bool)
        selected_mask[selected] = True
        if demand_weight is None:
            # Compatibility with visualization sidecars created before demand
            # was added to the publication whitelist in flashvid/utils.py.
            demand_weight = selected_mask.astype(np.float32)
            demand_label = "Selected anchor mask"
            demand_source = "anchor_mask_fallback"
        else:
            demand_label = "Final token demand"
            demand_source = "certvid_v3_demand_weight"
        summaries = _component_summaries(
            component_ids,
            component_support,
            novelty,
            curvature,
            selected_mask,
            tokens_per_frame,
            frame_count,
        )
        window = _best_window(
            frame_event,
            novelty,
            curvature,
            selected_mask,
            frame_count,
            tokens_per_frame,
            args.filmstrip_frames,
        )
        top_components = _top_window_components(
            summaries,
            window,
            args.top_components,
            max(1, args.min_component_frames),
        )
        focus_frame = _focus_frame(
            frame_event,
            novelty,
            curvature,
            component_support,
            frame_count,
            tokens_per_frame,
        )

        prefix = f"{number:02d}_{video_path.stem}"
        graph_name = f"trajectory_tracks_{prefix}.png"
        signals_name = f"trajectory_heatmaps_{prefix}.png"
        _plot_trajectory_graph(
            output_dir / graph_name,
            display_frames,
            window,
            top_components,
            frame_event,
            selected_mask,
            component_ids,
            match_previous,
            match_mutual,
            match_similarity,
            grid_height,
            grid_width,
            tokens_per_frame,
            question_record.question,
            video_path.stem,
        )
        _plot_signal_maps(
            output_dir / signals_name,
            display_frames[focus_frame],
            novelty,
            curvature,
            component_support,
            demand_weight,
            frame_event,
            selected_mask,
            focus_frame,
            tokens_per_frame,
            grid_height,
            grid_width,
            args.heatmap_alpha,
            demand_label,
            question_record.question,
            video_path.stem,
        )

        valid_labels = [label for label, _ in question_record.options]
        records.append(
            {
                "video_id": video_path.stem,
                "video_path": str(video_path),
                "question_id": question_record.question_id,
                "question": question_record.question,
                "options": [
                    {"label": label, "text": text}
                    for label, text in question_record.options
                ],
                "target": question_record.answer,
                "prediction": extract_answer_label(prediction, valid_labels),
                "raw_prediction": prediction,
                "retention_ratio": args.retention_ratio,
                "expansion": args.expansion,
                "certificate_budget_ratio": 0.0,
                "raw_tokens": int(plan.raw_token_count),
                "selected_tokens": int(len(selected)),
                "sampled_source_frame_indices": [int(value) for value in source_indices],
                "video_fps": fps,
                "displayed_sampled_frames": [int(value) for value in window],
                "displayed_source_frames": [int(source_indices[value]) for value in window],
                "heatmap_sampled_frame": int(focus_frame),
                "heatmap_source_frame": int(source_indices[focus_frame]),
                "heatmap_event_strength": float(frame_event[focus_frame]),
                "heatmap_demand_source": demand_source,
                "figures": {
                    "trajectory_tracks": graph_name,
                    "trajectory_heatmaps": signals_name,
                },
                "components": [
                    {
                        "label": f"T{rank + 1:02d}",
                        "component_id": component.component_id,
                        "size": int(len(component.members)),
                        "sampled_frames": [int(value) for value in component.frames],
                        "selected_count": component.selected_count,
                        "support": component.support,
                        "max_novelty": component.max_novelty,
                        "max_curvature": component.max_curvature,
                        "display_score": component.score,
                    }
                    for rank, component in enumerate(top_components)
                ],
            }
        )
        (output_dir / "trajectory_metadata.json").write_text(
            json.dumps({"examples": records}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        setattr(config, "_certvid_plan", None)
        setattr(config, "_visualization_certvid_analysis", None)
        del pixels, pixels_cpu, input_ids, attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[done] graph={graph_name} signals={signals_name}", flush=True)

    print(f"[complete] outputs={output_dir}", flush=True)


if __name__ == "__main__":
    main()
