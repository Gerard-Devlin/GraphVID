#!/usr/bin/env python3
"""Create a paper-ready trajectory-state ribbon for VOLT-Vid.

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
        description="Visualize trajectory-state preservation under an equal token budget."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-root", default=str(hf_home / "videomme" / "data"))
    parser.add_argument("--metadata-jsonl", required=True)
    parser.add_argument("--video-id", default="")
    parser.add_argument("--num-examples", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--key-frames", type=int, default=4)
    parser.add_argument("--top-components", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260810)
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
    seed: int,
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


def _state_ribbon(
    component_ids: np.ndarray,
    novelty: np.ndarray,
    curvature: np.ndarray,
    support: np.ndarray,
    frame_count: int,
    tokens_per_frame: int,
    top_components: int,
) -> tuple[np.ndarray, list[int], np.ndarray]:
    state = np.maximum(novelty, curvature)
    rows: list[tuple[float, int, np.ndarray]] = []
    for component_id in np.unique(component_ids):
        members = np.flatnonzero(component_ids == component_id)
        frames = np.unique(members // tokens_per_frame)
        if frames.size < 2:
            continue
        ribbon = np.full(frame_count, np.nan, dtype=np.float32)
        for frame in frames:
            frame_members = members[members // tokens_per_frame == frame]
            ribbon[int(frame)] = float(state[frame_members].max())
        span = float(frames[-1] - frames[0] + 1) / max(1, frame_count)
        persistence = float(support[members].mean())
        peak = float(np.nanmax(ribbon))
        score = 0.44 * persistence + 0.34 * span + 0.22 * peak
        rows.append((score, int(component_id), ribbon))
    rows.sort(key=lambda item: (-item[0], item[1]))
    rows = rows[: max(1, top_components)]
    if not rows:
        raise RuntimeError("no multi-frame trajectory components were found")
    matrix = np.stack([item[2] for item in rows], axis=0)
    component_order = [item[1] for item in rows]
    frame_energy = np.max(np.where(np.isnan(matrix), 0.0, matrix), axis=0)
    return matrix, component_order, frame_energy


def _selection_counts(
    selected: np.ndarray,
    component_ids: np.ndarray,
    component_order: list[int],
    frame_count: int,
    tokens_per_frame: int,
) -> np.ndarray:
    counts = np.zeros((len(component_order), frame_count), dtype=np.int32)
    row_by_component = {component_id: row for row, component_id in enumerate(component_order)}
    for index in selected:
        component_id = int(component_ids[int(index)])
        row = row_by_component.get(component_id)
        if row is not None:
            counts[row, int(index) // tokens_per_frame] += 1
    return counts


def _key_frame_ids(frame_score: np.ndarray, count: int) -> list[int]:
    count = min(max(1, count), int(frame_score.size))
    ranked = np.argsort(-frame_score, kind="stable")
    chosen: list[int] = []
    minimum_gap = max(1, int(frame_score.size // max(2, count * 2)))
    for frame in ranked:
        value = int(frame)
        if all(abs(value - previous) >= minimum_gap for previous in chosen):
            chosen.append(value)
            if len(chosen) == count:
                break
    if len(chosen) < count:
        for frame in ranked:
            value = int(frame)
            if value not in chosen:
                chosen.append(value)
                if len(chosen) == count:
                    break
    return sorted(chosen)


def _draw_anchor_boxes(ax, frame: Any, selected_local: np.ndarray, height: int, width: int) -> None:
    import matplotlib.pyplot as plt

    image = np.asarray(frame)
    ax.imshow(image)
    patch_width = image.shape[1] / width
    patch_height = image.shape[0] / height
    for local in selected_local:
        row, col = divmod(int(local), width)
        ax.add_patch(
            plt.Rectangle(
                (col * patch_width, row * patch_height),
                patch_width,
                patch_height,
                fill=False,
                edgecolor="#00AEB3",
                linewidth=1.15,
            )
        )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _plot_case(
    *,
    output_path: Path,
    frames: list[Any],
    key_frames: list[int],
    full_selected: np.ndarray,
    no_trajectory_selected: np.ndarray,
    ribbon: np.ndarray,
    component_order: list[int],
    full_counts: np.ndarray,
    no_trajectory_counts: np.ndarray,
    frame_event: np.ndarray,
    frame_count: int,
    tokens_per_frame: int,
    grid_height: int,
    grid_width: int,
    question: str,
    target: str,
    full_prediction: str,
    no_trajectory_prediction: str,
    video_id: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15.8, 8.2), constrained_layout=True)
    grid = fig.add_gridspec(3, len(key_frames), height_ratios=(1.25, 1.0, 1.0))
    full_mask = np.zeros(frame_count * tokens_per_frame, dtype=bool)
    full_mask[full_selected] = True
    for column, frame_idx in enumerate(key_frames):
        ax = fig.add_subplot(grid[0, column])
        start = frame_idx * tokens_per_frame
        local = np.flatnonzero(full_mask[start : start + tokens_per_frame])
        _draw_anchor_boxes(ax, frames[frame_idx], local, grid_height, grid_width)
        ax.set_title(
            f"Frame {frame_idx + 1} | Event {frame_event[frame_idx]:.2f}",
            fontsize=10,
            color="#17212B",
        )

    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#ECEFF1")
    axes = [fig.add_subplot(grid[1, :]), fig.add_subplot(grid[2, :])]
    labels = [
        "(a) Without trajectory structure",
        "(b) Full trajectory-aware design",
    ]
    count_matrices = [no_trajectory_counts, full_counts]
    last_image = None
    for ax, label, counts in zip(axes, labels, count_matrices):
        last_image = ax.imshow(
            ribbon,
            aspect="auto",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
            cmap=cmap,
        )
        ys, xs = np.where(counts > 0)
        if xs.size:
            sizes = 34.0 + 18.0 * np.minimum(counts[ys, xs], 4)
            ax.scatter(
                xs,
                ys,
                s=sizes,
                facecolors="#00AEB3",
                edgecolors="white",
                linewidths=0.85,
                zorder=3,
            )
        ax.set_title(label, loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel("Trajectory component")
        ax.set_yticks(np.arange(len(component_order)))
        ax.set_yticklabels([f"T{row + 1:02d}" for row in range(len(component_order))])
        ax.set_xlim(-0.5, frame_count - 0.5)
        ax.set_xticks(np.arange(0, frame_count, 2))
        ax.set_xticklabels(np.arange(1, frame_count + 1, 2))
        ax.set_xlabel("Sampled frame")
        for frame_idx in key_frames:
            ax.axvline(frame_idx, color="white", alpha=0.28, linewidth=0.8)
    if last_image is not None:
        colorbar = fig.colorbar(last_image, ax=axes, fraction=0.015, pad=0.012)
        colorbar.set_label("State-change energy  max(novelty, curvature)")

    full_ok = full_prediction == target
    ablated_ok = no_trajectory_prediction == target
    fig.suptitle(
        f"Trajectory-state preservation at {len(full_selected)}/{frame_count * tokens_per_frame} "
        f"visual tokens | {video_id}\n{question}",
        fontsize=14,
        fontweight="bold",
        color="#17212B",
    )
    fig.text(
        0.5,
        0.008,
        f"Target: {target}    |    w/o trajectory: {no_trajectory_prediction} "
        f"({'correct' if ablated_ok else 'wrong'})    |    Full: {full_prediction} "
        f"({'correct' if full_ok else 'wrong'})",
        ha="center",
        fontsize=11,
        color="#17212B",
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

        no_prediction_raw, no_selected, _ = _run_variant(
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
        component_ids = _tensor(analysis, "component_ids", np.int64)
        novelty = _tensor(analysis, "novelty", np.float32)
        curvature = _tensor(analysis, "curvature", np.float32)
        support = _tensor(analysis, "component_support", np.float32)
        frame_event = _tensor(analysis, "frame_event", np.float32)
        ribbon, component_order, frame_energy = _state_ribbon(
            component_ids,
            novelty,
            curvature,
            support,
            frame_count,
            tokens_per_frame,
            args.top_components,
        )
        full_counts = _selection_counts(
            full_selected, component_ids, component_order, frame_count, tokens_per_frame
        )
        no_counts = _selection_counts(
            no_selected, component_ids, component_order, frame_count, tokens_per_frame
        )
        key_frames = _key_frame_ids(
            0.68 * frame_energy + 0.32 * frame_event,
            args.key_frames,
        )
        labels = [label for label, _ in record.options]
        full_prediction = extract_answer_label(full_prediction_raw, labels)
        no_prediction = extract_answer_label(no_prediction_raw, labels)
        image_name = f"state_ribbon_{number:02d}_{video_path.stem}.png"
        _plot_case(
            output_path=output_dir / image_name,
            frames=display_frames,
            key_frames=key_frames,
            full_selected=full_selected,
            no_trajectory_selected=no_selected,
            ribbon=ribbon,
            component_order=component_order,
            full_counts=full_counts,
            no_trajectory_counts=no_counts,
            frame_event=frame_event,
            frame_count=frame_count,
            tokens_per_frame=tokens_per_frame,
            grid_height=grid_height,
            grid_width=grid_width,
            question=record.question,
            target=record.answer,
            full_prediction=full_prediction,
            no_trajectory_prediction=no_prediction,
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
                "key_sampled_frames": key_frames,
                "key_source_frames": [int(source_indices[index]) for index in key_frames],
                "video_fps": fps,
                "figure": image_name,
            }
        )
        (output_dir / "state_ribbon_metadata.json").write_text(
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
