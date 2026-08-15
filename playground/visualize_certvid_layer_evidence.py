#!/usr/bin/env python3
"""Visualize question-conditioned evidence preservation across CertVID layers.

All quantities in the figure come from the real LLaVA-OneVision prefill.  The
script uses CertVID's opt-in attention sidecar to aggregate the last textual
query's attention over the surviving visual tokens of every sampled frame.
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

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageOps

from visualize_certvid_two_examples import (
    QuestionRecord,
    discover_videos,
    generate_once,
    load_certvid_model,
    load_questions,
    prepare_prompt,
    sample_video,
)


@dataclass
class LayerEvidenceCase:
    video_id: str
    video_path: Path
    question: QuestionRecord
    frames: list[Image.Image]
    source_indices: np.ndarray
    records: dict[int, dict[str, Any]]
    anchor_counts: np.ndarray
    prediction: str
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
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=12,
        help="Geometry-only scan size when --video-id is omitted.",
    )
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--filmstrip-frames", type=int, default=8)
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
    videos: list[Path],
    questions: dict[str, list[QuestionRecord]],
    args: argparse.Namespace,
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


def _question_for(
    records: list[QuestionRecord], question_id: str
) -> QuestionRecord:
    if question_id:
        for record in records:
            if str(record.question_id or "") == str(question_id):
                return record
        raise KeyError(f"question id was not found: {question_id}")
    return records[0]


def _frame_matrix(records: dict[int, dict[str, Any]]) -> tuple[list[int], np.ndarray]:
    layers = sorted(int(layer) for layer in records)
    rows: list[np.ndarray] = []
    frame_count = None
    for layer in layers:
        values = records[layer].get("frame_weights")
        if not torch.is_tensor(values) or values.ndim != 1:
            raise RuntimeError(f"layer {layer} has malformed frame weights")
        row = values.detach().float().cpu().numpy()
        if not np.isfinite(row).all():
            raise RuntimeError(f"layer {layer} frame weights contain NaN/Inf")
        frame_count = len(row) if frame_count is None else frame_count
        if len(row) != frame_count:
            raise RuntimeError("captured frame count changes across layers")
        rows.append(row)
    if not rows:
        raise RuntimeError("no layer-wise frame attention was captured")
    return layers, np.stack(rows, axis=0)


def _js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left = np.clip(left.astype(np.float64), 1e-12, None)
    right = np.clip(right.astype(np.float64), 1e-12, None)
    left /= left.sum()
    right /= right.sum()
    middle = 0.5 * (left + right)
    return float(
        0.5 * np.sum(left * np.log(left / middle))
        + 0.5 * np.sum(right * np.log(right / middle))
    )


def _case_score(matrix: np.ndarray, anchor_counts: np.ndarray) -> float:
    early = matrix[0]
    late = matrix[-1]
    anchors = anchor_counts.astype(np.float64)
    anchors /= max(1.0, anchors.sum())
    late_norm = late / max(1e-12, late.sum())
    alignment = float(np.dot(np.sqrt(anchors), np.sqrt(late_norm)))
    concentration = float(late.max() / max(1e-12, late.mean()))
    active = float(np.count_nonzero(anchor_counts)) / max(1, len(anchor_counts))
    return _js_divergence(early, late) + 0.35 * alignment + 0.08 * concentration + 0.15 * active


def _copy_records(captured: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {
        int(layer): {
            key: value.detach().float().cpu().clone() if torch.is_tensor(value) else value
            for key, value in record.items()
        }
        for layer, record in captured.items()
    }


def _filmstrip_indices(frame_count: int, panel_count: int, peaks: list[int]) -> list[int]:
    panel_count = min(frame_count, max(4, panel_count))
    base = np.linspace(0, frame_count - 1, panel_count).round().astype(int).tolist()
    indices = sorted(set(base))
    for peak in peaks:
        if peak in indices:
            continue
        replace = min(
            range(len(indices)),
            key=lambda position: (
                abs(indices[position] - peak),
                indices[position] in (0, frame_count - 1),
            ),
        )
        indices[replace] = peak
        indices = sorted(set(indices))
    return indices


def _draw_filmstrip(
    axes: list[plt.Axes],
    frames: list[Image.Image],
    indices: list[int],
    evidence: np.ndarray,
    anchor_counts: np.ndarray,
) -> None:
    vmax = max(float(evidence.max()), 1e-12)
    for axis, frame_index in zip(axes, indices):
        frame = ImageOps.fit(
            frames[frame_index].convert("RGB"),
            (480, 285),
            method=Image.Resampling.LANCZOS,
        )
        axis.imshow(frame)
        axis.set_xticks([])
        axis.set_yticks([])
        strength = float(evidence[frame_index] / vmax)
        color = plt.cm.RdPu(0.28 + 0.70 * strength)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(1.3 + 2.5 * strength)
        axis.text(
            0.02,
            0.04,
            f"F{frame_index + 1:02d}  A{int(anchor_counts[frame_index])}",
            transform=axis.transAxes,
            fontsize=8.5,
            color="white",
            ha="left",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "#17202A", "alpha": 0.72, "edgecolor": "none"},
        )


def _save_figure(case: LayerEvidenceCase, output_dir: Path, args: argparse.Namespace) -> None:
    layers, matrix = _frame_matrix(case.records)
    late = matrix[-1]
    peaks = np.argsort(late)[-2:][::-1].astype(int).tolist()
    film_indices = _filmstrip_indices(len(case.frames), args.filmstrip_frames, peaks)

    cmap = LinearSegmentedColormap.from_list(
        "certvid_evidence",
        ["#102A43", "#2F6F8F", "#93C5B5", "#F4D6A0", "#BD4444"],
        N=256,
    )
    figure = plt.figure(figsize=(15.8, 7.0), facecolor="white")
    outer = figure.add_gridspec(2, 1, height_ratios=(1.0, 2.25), hspace=0.12)
    film = outer[0].subgridspec(1, len(film_indices), wspace=0.035)
    film_axes = [figure.add_subplot(film[0, index]) for index in range(len(film_indices))]
    _draw_filmstrip(film_axes, case.frames, film_indices, late, case.anchor_counts)

    axis = figure.add_subplot(outer[1])
    positive = matrix[matrix > 0]
    vmax = float(np.quantile(positive, 0.99)) if positive.size else 1.0
    image = axis.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0.0,
        vmax=max(vmax, 1e-8),
    )
    axis.set_xlabel("Sampled frame", fontsize=11)
    axis.set_ylabel("Transformer layer", fontsize=11)
    x_ticks = np.arange(0, matrix.shape[1], 2)
    axis.set_xticks(x_ticks, [str(value + 1) for value in x_ticks], fontsize=8)
    y_step = max(1, len(layers) // 10)
    y_positions = np.arange(0, len(layers), y_step)
    axis.set_yticks(y_positions, [str(layers[value]) for value in y_positions], fontsize=8)

    boundary = sum(layer <= args.pruning_layer for layer in layers) - 0.5
    if 0 <= boundary <= len(layers):
        axis.axhline(boundary, color="#F7F4EA", linewidth=1.3, linestyle="--", alpha=0.95)

    selected_frames = np.flatnonzero(case.anchor_counts > 0)
    if selected_frames.size:
        selected_y = np.full(selected_frames.shape, len(layers) - 0.55)
        sizes = 18.0 + 8.0 * np.sqrt(case.anchor_counts[selected_frames])
        axis.scatter(
            selected_frames,
            selected_y,
            s=sizes,
            marker="v",
            color="#F7F4EA",
            edgecolors="#8D1B3D",
            linewidths=0.7,
            clip_on=False,
            zorder=6,
        )

    colorbar = figure.colorbar(image, ax=axis, fraction=0.018, pad=0.018)
    colorbar.set_label("Normalized question-to-frame evidence", fontsize=9)
    axis.spines[["top", "right"]].set_visible(False)
    figure.subplots_adjust(left=0.06, right=0.955, bottom=0.09, top=0.985)

    stem = output_dir / "certvid_layer_evidence_preservation"
    figure.savefig(stem.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(figure)

    metadata = {
        "video_id": case.video_id,
        "video_path": str(case.video_path),
        "question_id": case.question.question_id,
        "question": case.question.question,
        "options": [{"label": label, "text": text} for label, text in case.question.options],
        "target": case.question.answer,
        "prediction": case.prediction,
        "sampled_source_indices": case.source_indices.astype(int).tolist(),
        "captured_layers": layers,
        "anchor_counts_per_frame": case.anchor_counts.astype(int).tolist(),
        "final_layer_frame_evidence": late.astype(float).tolist(),
        "filmstrip_frames": [int(value + 1) for value in film_indices],
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
    (output_dir / "certvid_layer_evidence_preservation.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.num_frames <= 1 or args.candidate_count <= 0:
        raise ValueError("frame and candidate counts must be positive")
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
    setattr(config, "_capture_layer_frame_attention", True)

    best: LayerEvidenceCase | None = None
    audit: list[dict[str, Any]] = []
    for number, video_path in enumerate(candidates, start=1):
        print(f"[scan] {number}/{len(candidates)} {video_path.name}", flush=True)
        try:
            question = _question_for(questions[video_path.stem], args.question_id)
            frames_np, source_indices, _ = sample_video(video_path, args.num_frames)
            frames = [Image.fromarray(frame, mode="RGB") for frame in frames_np]
            pixel_values_cpu = image_processor.preprocess(frames_np, return_tensors="pt")["pixel_values"]
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
            captured = getattr(config, "_visualization_layer_attention", None)
            if not isinstance(captured, dict) or not captured or plan is None:
                raise RuntimeError("CertVID did not publish layer attention and plan")
            records = _copy_records(captured)
            _, matrix = _frame_matrix(records)
            tokens_per_frame = int(plan.raw_token_count) // args.num_frames
            anchor_frames = torch.div(
                plan.anchor_indices.detach().long().cpu(),
                tokens_per_frame,
                rounding_mode="floor",
            )
            anchor_counts = torch.bincount(anchor_frames, minlength=args.num_frames).numpy()
            score = _case_score(matrix, anchor_counts)
            case = LayerEvidenceCase(
                video_id=video_path.stem,
                video_path=video_path,
                question=question,
                frames=frames,
                source_indices=source_indices,
                records=records,
                anchor_counts=anchor_counts,
                prediction=prediction,
                score=score,
            )
            audit.append({"video_id": video_path.stem, "score": score, "valid": True})
            if best is None or score > best.score:
                best = case
            print(f"[score] {video_path.stem} evidence={score:.4f}", flush=True)
            del pixel_values, pixel_values_cpu, input_ids, attention_mask
        except Exception as error:
            audit.append({"video_id": video_path.stem, "valid": False, "error": str(error)})
            print(f"[skip] {video_path.name}: {error}", flush=True)
            traceback.print_exc()
        finally:
            setattr(config, "_certvid_plan", None)
            setattr(config, "_visualization_layer_attention", {})
            setattr(config, "_visualization_current_frame_ids", None)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if best is None:
        raise RuntimeError("no candidate produced valid layer evidence")
    _save_figure(best, output_dir, args)
    (output_dir / "layer_evidence_scan.json").write_text(
        json.dumps({"seed": args.seed, "candidates": audit}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[done] selected={best.video_id} output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
