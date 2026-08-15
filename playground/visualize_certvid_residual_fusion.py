#!/usr/bin/env python3
"""Visualize CertVID V3's real residual-to-anchor fusion assignments."""

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

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import ConnectionPatch, Rectangle
from PIL import Image, ImageOps

from visualize_certvid_two_examples import (
    QuestionRecord,
    discover_videos,
    factor_grid,
    generate_once,
    load_certvid_model,
    load_questions,
    prepare_prompt,
    sample_video,
    tensor_frames_to_pil,
)


@dataclass
class FusionCase:
    video_id: str
    video_path: Path
    question: QuestionRecord
    frames: list[Image.Image]
    source_indices: np.ndarray
    prediction: str
    anchor_indices: np.ndarray
    assignment_indices: np.ndarray
    assignment_weights: np.ndarray
    source_mass: np.ndarray
    fusion_alpha: np.ndarray
    frame_count: int
    tokens_per_frame: int
    grid_height: int
    grid_width: int
    score: float


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
    parser.add_argument("--answers-json", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--video-id", default="")
    parser.add_argument("--question-id", default="")
    parser.add_argument("--candidate-count", type=int, default=12)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--panel-frames", type=int, default=5)
    parser.add_argument("--max-edges", type=int, default=32)
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


def _candidate_videos(
    videos: list[Path], questions: dict[str, list[QuestionRecord]], args: argparse.Namespace
) -> list[Path]:
    by_stem = {path.stem: path for path in videos}
    if args.video_id:
        stem = Path(args.video_id).stem
        if stem not in by_stem:
            raise FileNotFoundError(f"requested video was not found: {args.video_id}")
        if stem not in questions:
            raise KeyError(f"metadata has no question for video: {stem}")
        return [by_stem[stem]]
    matched = [by_stem[stem] for stem in sorted(set(by_stem).intersection(questions))]
    random.Random(args.seed).shuffle(matched)
    return matched[: max(1, args.candidate_count)]


def _question_for(records: list[QuestionRecord], question_id: str) -> QuestionRecord:
    if question_id:
        for record in records:
            if str(record.question_id or "") == str(question_id):
                return record
        raise KeyError(f"question id was not found: {question_id}")
    return records[0]


def _effective_edges(case: FusionCase) -> tuple[np.ndarray, ...]:
    source = np.repeat(
        np.arange(case.frame_count * case.tokens_per_frame, dtype=np.int64),
        case.assignment_indices.shape[1],
    )
    target_positions = case.assignment_indices.reshape(-1).astype(np.int64)
    target = case.anchor_indices[target_positions]
    weights = case.assignment_weights.reshape(-1).astype(np.float64)
    source_mass = np.repeat(case.source_mass.astype(np.float64), case.assignment_indices.shape[1])
    alpha = case.fusion_alpha[target_positions].astype(np.float64)
    effective = weights * source_mass * alpha
    selected_mask = np.isin(source, case.anchor_indices)
    valid = (effective > 1e-10) & (~selected_mask) & (source != target)
    return (
        source[valid],
        target[valid],
        effective[valid],
        target_positions[valid],
        weights[valid],
    )


def _best_window(case: FusionCase, panel_frames: int) -> tuple[int, int, float]:
    panel_frames = min(case.frame_count, max(3, panel_frames))
    source, target, mass, _, _ = _effective_edges(case)
    source_frame = source // case.tokens_per_frame
    target_frame = target // case.tokens_per_frame
    best = (0, panel_frames, -1.0)
    for start in range(case.frame_count - panel_frames + 1):
        stop = start + panel_frames
        inside = (
            (source_frame >= start)
            & (source_frame < stop)
            & (target_frame >= start)
            & (target_frame < stop)
        )
        score = float(mass[inside].sum())
        if score > best[2]:
            best = (start, stop, score)
    return best


def _case_score(case: FusionCase) -> float:
    source, target, mass, _, _ = _effective_edges(case)
    if not len(mass):
        return -math.inf
    source_frame = source // case.tokens_per_frame
    target_frame = target // case.tokens_per_frame
    cross_frame = float(mass[source_frame != target_frame].sum())
    active_sources = len(np.unique(source)) / max(1, case.frame_count * case.tokens_per_frame)
    concentration = float(np.quantile(mass, 0.90) / max(1e-12, mass.mean()))
    return math.log1p(float(mass.sum())) + 0.35 * math.log1p(cross_frame) + active_sources + 0.04 * concentration


def _token_center(local_index: int, grid_height: int, grid_width: int) -> tuple[float, float]:
    row, col = divmod(int(local_index), grid_width)
    return ((col + 0.5) / grid_width, 1.0 - (row + 0.5) / grid_height)


def _draw_grid_axis(
    axis: plt.Axes,
    frame: Image.Image,
    frame_index: int,
    case: FusionCase,
    anchor_color: str,
) -> None:
    axis.imshow(ImageOps.fit(frame.convert("RGB"), (560, 340), Image.Resampling.LANCZOS), extent=(0, 1, 0, 1))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(f"Frame {frame_index + 1}", fontsize=10, pad=5, color="#25313C")
    anchors = case.anchor_indices // case.tokens_per_frame
    local = case.anchor_indices % case.tokens_per_frame
    for anchor_position in np.flatnonzero(anchors == frame_index):
        token = int(local[anchor_position])
        row, col = divmod(token, case.grid_width)
        x0 = col / case.grid_width
        y0 = 1.0 - (row + 1) / case.grid_height
        alpha = float(case.fusion_alpha[anchor_position])
        axis.add_patch(
            Rectangle(
                (x0, y0),
                1.0 / case.grid_width,
                1.0 / case.grid_height,
                fill=True,
                facecolor=anchor_color,
                edgecolor="white",
                linewidth=0.8,
                alpha=0.20 + 0.62 * min(1.0, alpha / 0.12),
                zorder=5,
            )
        )
        axis.add_patch(
            Rectangle(
                (x0, y0),
                1.0 / case.grid_width,
                1.0 / case.grid_height,
                fill=False,
                edgecolor=anchor_color,
                linewidth=1.5,
                zorder=6,
            )
        )
    for spine in axis.spines.values():
        spine.set_color("#CAD2D8")
        spine.set_linewidth(0.8)


def _save_figure(case: FusionCase, output_dir: Path, args: argparse.Namespace) -> None:
    start, stop, window_mass = _best_window(case, args.panel_frames)
    frame_indices = list(range(start, stop))
    source, target, effective, target_positions, raw_weights = _effective_edges(case)
    source_frame = source // case.tokens_per_frame
    target_frame = target // case.tokens_per_frame
    inside = (
        (source_frame >= start)
        & (source_frame < stop)
        & (target_frame >= start)
        & (target_frame < stop)
    )
    edge_ids = np.flatnonzero(inside)
    edge_ids = edge_ids[np.argsort(effective[edge_ids])[::-1]][: max(1, args.max_edges)]

    anchor_color = "#BD4444"
    residual_color = "#4E74B3"
    flow_color = "#2A9D8F"
    figure, axes = plt.subplots(1, len(frame_indices), figsize=(3.25 * len(frame_indices), 3.25), facecolor="white")
    axes = np.atleast_1d(axes).tolist()
    by_frame = {frame: axis for frame, axis in zip(frame_indices, axes)}
    for axis, frame_index in zip(axes, frame_indices):
        _draw_grid_axis(axis, case.frames[frame_index], frame_index, case, anchor_color)

    vmax = max(float(effective[edge_ids].max()) if len(edge_ids) else 0.0, 1e-12)
    for rank, edge_id in enumerate(edge_ids):
        source_index = int(source[edge_id])
        target_index = int(target[edge_id])
        source_f, source_local = divmod(source_index, case.tokens_per_frame)
        target_f, target_local = divmod(target_index, case.tokens_per_frame)
        source_xy = _token_center(source_local, case.grid_height, case.grid_width)
        target_xy = _token_center(target_local, case.grid_height, case.grid_width)
        strength = float(effective[edge_id] / vmax)
        source_axis = by_frame[source_f]
        target_axis = by_frame[target_f]
        source_axis.scatter(
            [source_xy[0]], [source_xy[1]],
            s=8 + 22 * strength,
            color=residual_color,
            edgecolors="white",
            linewidths=0.35,
            alpha=0.60 + 0.35 * strength,
            zorder=8,
        )
        connection = ConnectionPatch(
            xyA=source_xy,
            coordsA=source_axis.transData,
            xyB=target_xy,
            coordsB=target_axis.transData,
            arrowstyle="-|>",
            mutation_scale=7.0 + 3.0 * strength,
            linewidth=0.45 + 1.25 * strength,
            color=flow_color,
            alpha=0.20 + 0.48 * strength,
            connectionstyle=f"arc3,rad={0.06 * ((rank % 3) - 1)}",
            zorder=7,
        )
        figure.add_artist(connection)

    handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=anchor_color, markeredgecolor=anchor_color, markersize=8, label="Selected anchor"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=residual_color, markeredgecolor="white", markersize=7, label="Residual source"),
        plt.Line2D([0, 1], [0, 0], color=flow_color, linewidth=1.5, label="Effective soft assignment"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    figure.subplots_adjust(left=0.012, right=0.988, top=0.92, bottom=0.16, wspace=0.055)

    stem = output_dir / "certvid_residual_to_anchor_fusion"
    figure.savefig(stem.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(figure)

    edge_records = []
    for edge_id in edge_ids:
        source_index = int(source[edge_id])
        target_index = int(target[edge_id])
        edge_records.append(
            {
                "source_global_index": source_index,
                "source_frame": int(source_index // case.tokens_per_frame) + 1,
                "source_local_index": int(source_index % case.tokens_per_frame),
                "anchor_global_index": target_index,
                "anchor_frame": int(target_index // case.tokens_per_frame) + 1,
                "anchor_local_index": int(target_index % case.tokens_per_frame),
                "assignment_weight": float(raw_weights[edge_id]),
                "fusion_alpha": float(case.fusion_alpha[int(target_positions[edge_id])]),
                "effective_mass": float(effective[edge_id]),
            }
        )
    metadata = {
        "video_id": case.video_id,
        "video_path": str(case.video_path),
        "question_id": case.question.question_id,
        "question": case.question.question,
        "options": [{"label": label, "text": text} for label, text in case.question.options],
        "target": case.question.answer,
        "prediction": case.prediction,
        "sampled_source_indices": case.source_indices.astype(int).tolist(),
        "visualized_frame_window": [start + 1, stop],
        "window_effective_fusion_mass": window_mass,
        "shown_edges": edge_records,
        "total_anchor_count": int(len(case.anchor_indices)),
        "geometry_only_selection_score": case.score,
        "configuration": {
            "frames": args.num_frames,
            "retention_ratio": args.retention_ratio,
            "expansion": args.expansion,
            "pruning_layer": args.pruning_layer,
            "llm_retention_ratio": args.llm_retention_ratio,
            "certificate_budget_ratio": 0.0,
        },
    }
    (output_dir / "certvid_residual_to_anchor_fusion.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.num_frames <= 1 or args.candidate_count <= 0 or args.max_edges <= 0:
        raise ValueError("frame, candidate, and edge counts must be positive")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    answers = Path(args.answers_json).expanduser().resolve() if args.answers_json else None
    questions = load_questions(Path(args.metadata_jsonl).expanduser().resolve(), answers)
    videos = discover_videos(Path(args.dataset_root).expanduser().resolve())
    candidates = _candidate_videos(videos, questions, args)
    if not candidates:
        raise RuntimeError("no VideoMME video matched the metadata")

    tokenizer, model, image_processor, device = load_certvid_model(args)
    config = model.flashvid_config
    setattr(config, "certv3_certificate_budget_ratio", 0.0)
    setattr(config, "_capture_visualization_design", True)

    best: FusionCase | None = None
    audit: list[dict[str, Any]] = []
    for number, video_path in enumerate(candidates, start=1):
        print(f"[scan] {number}/{len(candidates)} {video_path.name}", flush=True)
        try:
            question = _question_for(questions[video_path.stem], args.question_id)
            frames_np, source_indices, _ = sample_video(video_path, args.num_frames)
            pixel_values_cpu = image_processor.preprocess(frames_np, return_tensors="pt")["pixel_values"]
            # Patch coordinates refer to the vision processor's square crop,
            # so render that exact tensor rather than the uncropped video frame.
            frames = tensor_frames_to_pil(pixel_values_cpu, image_processor)
            pixel_values = pixel_values_cpu.to(device=device, dtype=torch.float16)
            input_ids, attention_mask = prepare_prompt(tokenizer, question.prompt, device)
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
            analysis = getattr(config, "_visualization_certvid_analysis", None)
            if plan is None or not isinstance(analysis, dict):
                raise RuntimeError("CertVID did not publish plan and analysis tensors")
            frame_count = int(args.num_frames)
            tokens_per_frame = int(plan.raw_token_count) // frame_count
            grid_height, grid_width = factor_grid(tokens_per_frame)
            case = FusionCase(
                video_id=video_path.stem,
                video_path=video_path,
                question=question,
                frames=frames,
                source_indices=source_indices,
                prediction=prediction,
                anchor_indices=plan.anchor_indices.detach().long().cpu().numpy(),
                assignment_indices=plan.assignment_indices.detach().long().cpu().numpy(),
                assignment_weights=plan.assignment_weights.detach().float().cpu().numpy(),
                source_mass=plan.source_mass.detach().float().cpu().numpy(),
                fusion_alpha=plan.fusion_alpha.detach().float().cpu().numpy(),
                frame_count=frame_count,
                tokens_per_frame=tokens_per_frame,
                grid_height=grid_height,
                grid_width=grid_width,
                score=0.0,
            )
            case.score = _case_score(case)
            if not math.isfinite(case.score):
                raise RuntimeError("sample has no effective residual fusion edges")
            audit.append({"video_id": video_path.stem, "score": case.score, "valid": True})
            if best is None or case.score > best.score:
                best = case
            print(f"[score] {video_path.stem} fusion={case.score:.4f}", flush=True)
            del pixel_values, pixel_values_cpu, input_ids, attention_mask
        except Exception as error:
            audit.append({"video_id": video_path.stem, "valid": False, "error": str(error)})
            print(f"[skip] {video_path.name}: {error}", flush=True)
            traceback.print_exc()
        finally:
            setattr(config, "_certvid_plan", None)
            setattr(config, "_visualization_certvid_analysis", None)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if best is None:
        raise RuntimeError("no candidate produced a valid residual-fusion map")
    _save_figure(best, output_dir, args)
    (output_dir / "residual_fusion_scan.json").write_text(
        json.dumps({"seed": args.seed, "candidates": audit}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[done] selected={best.video_id} output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
