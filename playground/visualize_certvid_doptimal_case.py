#!/usr/bin/env python3
"""Select and visualize a representative CertVID D-optimal VideoMME case.

The script scans a deterministic random subset of VideoMME, runs the real
CertVID V3 compression path, and ranks examples without using answer labels.
The selected example maximizes the measured advantage over equal-budget global
quality Top-K in the exact design space used by CertVID.
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
from PIL import Image

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
from visualize_certvid_volume import InformationSummary, _information_summary, _pca_coordinates


@dataclass
class Candidate:
    video_id: str
    video_path: Path
    question_id: str | None
    question: str
    options: list[dict[str, str]]
    target: str | None
    prediction: str | None
    raw_prediction: str
    frames: list[Image.Image]
    design: torch.Tensor
    quality: torch.Tensor
    ours_indices: torch.Tensor
    topk_indices: torch.Tensor
    ours_summary: InformationSummary
    topk_summary: InformationSummary
    ridge: float
    frame_count: int
    tokens_per_frame: int
    grid_height: int
    grid_width: int
    overlap_ratio: float
    temporal_entropy_ours: float
    temporal_entropy_topk: float
    d_efficiency: float
    rank_ratio: float
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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-count", type=int, default=20)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--retention-ratio", type=float, default=0.01)
    parser.add_argument("--expansion", type=float, default=1.30)
    parser.add_argument("--pruning-layer", type=int, default=20)
    parser.add_argument("--llm-retention-ratio", type=float, default=0.1923076923)
    parser.add_argument(
        "--pool-mode", choices=("bilinear", "average", "max"), default="bilinear"
    )
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def _normalized_entropy(indices: torch.Tensor, frame_count: int, tokens_per_frame: int) -> float:
    frame_ids = torch.div(indices.long(), tokens_per_frame, rounding_mode="floor")
    counts = torch.bincount(frame_ids, minlength=frame_count).double()
    probabilities = counts / counts.sum().clamp_min(1.0)
    positive = probabilities > 0
    entropy = -(probabilities[positive] * probabilities[positive].log()).sum()
    return float((entropy / math.log(max(2, frame_count))).item())


def _selection_score(
    ours: InformationSummary,
    topk: InformationSummary,
    dimension: int,
    overlap_ratio: float,
    entropy_ours: float,
    entropy_topk: float,
) -> tuple[float, float, float]:
    normalized_logdet_gain = (ours.logdet_gain - topk.logdet_gain) / max(1, dimension)
    d_efficiency = math.exp(max(-30.0, min(30.0, normalized_logdet_gain)))
    rank_ratio = ours.effective_rank / max(1e-8, topk.effective_rank)
    score = (
        normalized_logdet_gain
        + 0.35 * math.log(max(1e-8, rank_ratio))
        + 0.25 * (entropy_ours - entropy_topk)
        + 0.15 * (1.0 - overlap_ratio)
    )
    return score, d_efficiency, rank_ratio


def _per_frame(indices: torch.Tensor, frame_count: int, tokens_per_frame: int) -> list[list[int]]:
    result: list[list[int]] = [[] for _ in range(frame_count)]
    for global_index in indices.tolist():
        frame_index, local_index = divmod(int(global_index), tokens_per_frame)
        if 0 <= frame_index < frame_count:
            result[frame_index].append(local_index)
    return [sorted(values) for values in result]


def _representative_frame(candidate: Candidate) -> int:
    ours = _per_frame(candidate.ours_indices, candidate.frame_count, candidate.tokens_per_frame)
    topk = _per_frame(candidate.topk_indices, candidate.frame_count, candidate.tokens_per_frame)

    # A large keep-set disagreement is only useful when the underlying frame is
    # visually informative. Rank both factors so dark transitions and title
    # cards cannot win merely because their sparse selections differ.
    visual_detail: list[float] = []
    disagreement: list[float] = []
    support: list[float] = []
    for frame_index, frame in enumerate(candidate.frames):
        image = np.asarray(frame.convert("L"), dtype=np.float32) / 255.0
        contrast = float(image.std())
        edge_x = float(np.abs(np.diff(image, axis=1)).mean()) if image.shape[1] > 1 else 0.0
        edge_y = float(np.abs(np.diff(image, axis=0)).mean()) if image.shape[0] > 1 else 0.0
        visual_detail.append(contrast + 2.5 * (edge_x + edge_y))
        ours_set = set(ours[frame_index])
        topk_set = set(topk[frame_index])
        disagreement.append(float(len(ours_set ^ topk_set)))
        support.append(float(min(len(ours_set), len(topk_set))))

    def normalize(values: list[float]) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        span = float(array.max() - array.min())
        return (array - array.min()) / span if span > 1e-12 else np.zeros_like(array)

    scores = (
        0.55 * normalize(visual_detail)
        + 0.35 * normalize(disagreement)
        + 0.10 * normalize(support)
    )
    valid = [
        index
        for index in range(candidate.frame_count)
        if ours[index] and topk[index] and disagreement[index] > 0
    ]
    pool = valid or list(range(candidate.frame_count))
    return max(pool, key=lambda index: (float(scores[index]), -abs(index - 15.5)))


def _selection_heatmap(
    frame: Image.Image,
    selected: list[int],
    quality: np.ndarray,
    grid_height: int,
    grid_width: int,
) -> Image.Image:
    import matplotlib

    token_count = grid_height * grid_width
    values = np.asarray(quality, dtype=np.float32).reshape(-1)
    if len(values) != token_count:
        raise ValueError(
            f"heatmap quality has {len(values)} values, expected {token_count}"
        )

    # Selection is the dominant signal. Quality only modulates color within
    # selected and unselected groups, preserving an honest keep-set comparison.
    normalized = np.clip(values, 0.0, 1.0)
    heat = 0.04 + 0.18 * normalized
    selected_array = np.asarray(selected, dtype=np.int64)
    if selected_array.size:
        heat[selected_array] = 0.80 + 0.20 * normalized[selected_array]
    heat = heat.reshape(grid_height, grid_width)

    rgba = matplotlib.colormaps["turbo"](heat, bytes=True)
    heatmap = Image.fromarray(rgba[:, :, :3], mode="RGB").resize(
        frame.size, resample=Image.Resampling.NEAREST
    )
    return Image.blend(frame.convert("RGB"), heatmap, alpha=0.52)


def _plot(candidate: Candidate, output_dir: Path, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Muted, print-safe colors following the visual hierarchy used in paper figures:
    # the baseline recedes while our method carries the single strong accent.
    ours_color = "#0F4D92"
    baseline_color = "#D88F8A"
    token_gray = "#AEB7BF"
    grid_color = "#D8DEE4"
    ink = "#26323D"
    frame_index = _representative_frame(candidate)
    ours_per_frame = _per_frame(
        candidate.ours_indices, candidate.frame_count, candidate.tokens_per_frame
    )
    topk_per_frame = _per_frame(
        candidate.topk_indices, candidate.frame_count, candidate.tokens_per_frame
    )
    frame_quality = candidate.quality.reshape(
        candidate.frame_count, candidate.tokens_per_frame
    )[frame_index].numpy()
    topk_overlay = _selection_heatmap(
        candidate.frames[frame_index],
        topk_per_frame[frame_index],
        frame_quality,
        candidate.grid_height,
        candidate.grid_width,
    )
    ours_overlay = _selection_heatmap(
        candidate.frames[frame_index],
        ours_per_frame[frame_index],
        frame_quality,
        candidate.grid_height,
        candidate.grid_width,
    )

    coordinates, explained = _pca_coordinates(candidate.design)
    ours_np = candidate.ours_indices.numpy()
    topk_np = candidate.topk_indices.numpy()
    spectrum_ours = np.sort(candidate.ours_summary.axis_information)[::-1]
    spectrum_topk = np.sort(candidate.topk_summary.axis_information)[::-1]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9.0,
            "axes.labelsize": 9.0,
            "axes.titlesize": 10.2,
            "axes.titleweight": "semibold",
            "axes.edgecolor": ink,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.8,
        }
    )

    def save_figure(fig, stem: str, *, pad: float = 0.025) -> None:
        for extension in ("png", "pdf"):
            fig.savefig(
                output_dir / f"{stem}.{extension}",
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=pad,
                facecolor="white",
            )
        plt.close(fig)

    def save_token_map(image: Image.Image, stem: str) -> None:
        width, height = image.size
        fig, axis = plt.subplots(figsize=(4.2, 4.2 * height / max(1, width)))
        axis.imshow(image)
        axis.set_axis_off()
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        save_figure(fig, stem, pad=0.0)

    # The first two outputs are intentionally image-only. Method names, frame
    # indices, and token counts are recorded in the JSON audit for the caption.
    save_token_map(topk_overlay, "01_quality_topk")
    save_token_map(ours_overlay, "02_doptimal_selection")

    x = np.arange(candidate.frame_count)
    fig, ax_timeline = plt.subplots(figsize=(5.2, 3.35))
    ax_timeline.plot(
        x,
        [len(values) for values in topk_per_frame],
        color=baseline_color,
        marker="o",
        ms=2.7,
        linewidth=1.45,
        label="Quality Top-K",
    )
    ax_timeline.plot(
        x,
        [len(values) for values in ours_per_frame],
        color=ours_color,
        marker="o",
        ms=2.7,
        linewidth=1.65,
        label="Ours",
    )
    ax_timeline.axvline(frame_index, color="#7B858E", linestyle="--", linewidth=0.9)
    ax_timeline.set_xlabel("Sampled frame")
    ax_timeline.set_ylabel("Selected tokens")
    ax_timeline.legend(frameon=False, loc="upper right", handlelength=2.0)
    ax_timeline.grid(color=grid_color, alpha=0.62, linewidth=0.65)
    ax_timeline.spines["top"].set_visible(False)
    ax_timeline.spines["right"].set_visible(False)
    ax_timeline.tick_params(width=0.75, length=3.0, color=ink)
    fig.tight_layout(pad=0.45)
    save_figure(fig, "03_temporal_allocation")

    fig, ax_pca = plt.subplots(figsize=(4.7, 4.05))
    ax_pca.scatter(
        coordinates[:, 0], coordinates[:, 1], s=3.2, color=token_gray, alpha=0.13,
        linewidths=0, label="All visual tokens"
    )
    ax_pca.scatter(
        coordinates[topk_np, 0], coordinates[topk_np, 1], s=18, marker="x",
        color=baseline_color, alpha=0.82, linewidths=0.8, label="Quality Top-K"
    )
    ax_pca.scatter(
        coordinates[ours_np, 0], coordinates[ours_np, 1], s=15, color=ours_color,
        alpha=0.78, linewidths=0, label="Ours"
    )
    ax_pca.set_xlabel(f"PC 1 ({100.0 * explained[0]:.1f}% variance)")
    ax_pca.set_ylabel(f"PC 2 ({100.0 * explained[1]:.1f}% variance)")
    ax_pca.legend(frameon=False, loc="best", handletextpad=0.4)
    ax_pca.spines["top"].set_visible(False)
    ax_pca.spines["right"].set_visible(False)
    ax_pca.tick_params(width=0.75, length=3.0, color=ink)
    fig.tight_layout(pad=0.45)
    save_figure(fig, "04_design_space_coverage")

    metric_names = ["D-efficiency", "Effective rank", "Temporal entropy"]
    ours_metrics = [
        candidate.d_efficiency,
        candidate.rank_ratio,
        candidate.temporal_entropy_ours / max(1e-8, candidate.temporal_entropy_topk),
    ]
    positions = np.arange(len(metric_names))
    width = 0.36
    fig, ax_metrics = plt.subplots(figsize=(5.0, 3.55))
    ax_metrics.bar(positions - width / 2, np.ones(3), width, color=baseline_color, alpha=0.72,
                   label="Quality Top-K")
    bars = ax_metrics.bar(positions + width / 2, ours_metrics, width, color=ours_color, alpha=0.90,
                          label="Ours")
    for bar, value in zip(bars, ours_metrics):
        ax_metrics.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}x",
                        ha="center", va="bottom", fontsize=8.2, fontweight="semibold",
                        color=ours_color)
    ax_metrics.set_xticks(positions, metric_names, rotation=10)
    ax_metrics.set_ylabel("Relative to Quality Top-K")
    ax_metrics.axhline(1.0, color="#7B858E", linewidth=0.8, linestyle="--")
    ax_metrics.legend(frameon=False, loc="upper left")
    ax_metrics.grid(axis="y", color=grid_color, alpha=0.62, linewidth=0.65)
    ax_metrics.spines["top"].set_visible(False)
    ax_metrics.spines["right"].set_visible(False)
    ax_metrics.tick_params(width=0.75, length=3.0, color=ink)
    fig.tight_layout(pad=0.45)
    save_figure(fig, "05_structural_quality")

    axes_index = np.arange(1, len(spectrum_ours) + 1)
    fig, ax_spectrum = plt.subplots(figsize=(5.0, 3.55))
    ax_spectrum.plot(axes_index, spectrum_topk, color=baseline_color, linewidth=1.45,
                     label="Quality Top-K")
    ax_spectrum.plot(axes_index, spectrum_ours, color=ours_color, linewidth=1.75, label="Ours")
    ax_spectrum.fill_between(
        axes_index, spectrum_topk, spectrum_ours, color=ours_color, alpha=0.075
    )
    ax_spectrum.set_xlabel("Design eigen-directions")
    ax_spectrum.set_ylabel(r"$\log(1 + \lambda_j / \lambda)$")
    ax_spectrum.legend(frameon=False, loc="upper right")
    ax_spectrum.grid(color=grid_color, alpha=0.62, linewidth=0.65)
    ax_spectrum.spines["top"].set_visible(False)
    ax_spectrum.spines["right"].set_visible(False)
    ax_spectrum.tick_params(width=0.75, length=3.0, color=ink)
    fig.tight_layout(pad=0.45)
    save_figure(fig, "06_information_spectrum")


def _audit_record(candidate: Candidate, selected: bool) -> dict[str, Any]:
    frame_index = _representative_frame(candidate)
    ours_per_frame = _per_frame(
        candidate.ours_indices, candidate.frame_count, candidate.tokens_per_frame
    )
    topk_per_frame = _per_frame(
        candidate.topk_indices, candidate.frame_count, candidate.tokens_per_frame
    )
    return {
        "selected_for_figure": selected,
        "video_id": candidate.video_id,
        "question_id": candidate.question_id,
        "question": candidate.question,
        "options": candidate.options,
        "target": candidate.target,
        "prediction": candidate.prediction,
        "raw_prediction": candidate.raw_prediction,
        "raw_tokens": int(len(candidate.design)),
        "selected_tokens": int(len(candidate.ours_indices)),
        "representative_frame": {
            "zero_based_index": frame_index,
            "one_based_index": frame_index + 1,
            "quality_topk_tokens": len(topk_per_frame[frame_index]),
            "doptimal_tokens": len(ours_per_frame[frame_index]),
            "selection_symmetric_difference": len(
                set(topk_per_frame[frame_index]) ^ set(ours_per_frame[frame_index])
            ),
        },
        "selection_heatmap": {
            "semantics": (
                "Blue denotes unselected patches and red denotes selected patches; "
                "the shared V3 quality score only modulates intensity within each group."
            ),
            "shared_frame_and_scale": True,
            "attention_map": False,
        },
        "design_dimension": int(candidate.design.shape[1]),
        "ridge": candidate.ridge,
        "selection_overlap_ratio": candidate.overlap_ratio,
        "d_efficiency_ours_vs_quality_topk": candidate.d_efficiency,
        "effective_rank_ratio_ours_vs_quality_topk": candidate.rank_ratio,
        "temporal_entropy": {
            "ours": candidate.temporal_entropy_ours,
            "quality_topk": candidate.temporal_entropy_topk,
        },
        "logdet_gain": {
            "ours": candidate.ours_summary.logdet_gain,
            "quality_topk": candidate.topk_summary.logdet_gain,
        },
        "effective_rank": {
            "ours": candidate.ours_summary.effective_rank,
            "quality_topk": candidate.topk_summary.effective_rank,
        },
        "selection_score": candidate.selection_score,
    }


def main() -> None:
    args = parse_args()
    if args.candidate_count <= 0:
        raise ValueError("--candidate-count must be positive")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(args.metadata_jsonl).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    questions = load_questions(metadata_path)
    videos = [path for path in discover_videos(dataset_root) if path.stem in questions]
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

    audit_records: list[dict[str, Any]] = []
    best: Candidate | None = None
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
            input_ids, attention_mask = prepare_prompt(tokenizer, question_record.prompt, device)
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
            topk_indices = torch.argsort(quality, descending=True, stable=True)[:budget].sort().values
            if design.ndim != 2 or len(design) != len(quality):
                raise RuntimeError("captured design and quality shapes do not match")
            frame_count = len(display_frames)
            if len(design) % frame_count:
                raise RuntimeError("raw visual-token count is not divisible by frame count")
            tokens_per_frame = len(design) // frame_count
            grid_height, grid_width = factor_grid(tokens_per_frame)
            ridge = float(analysis.get("ridge", 0.5))
            ours_summary = _information_summary(design, ours_indices, ridge)
            topk_summary = _information_summary(design, topk_indices, ridge)
            overlap_ratio = float(torch.isin(ours_indices, topk_indices).float().mean().item())
            entropy_ours = _normalized_entropy(ours_indices, frame_count, tokens_per_frame)
            entropy_topk = _normalized_entropy(topk_indices, frame_count, tokens_per_frame)
            score, d_efficiency, rank_ratio = _selection_score(
                ours_summary,
                topk_summary,
                int(design.shape[1]),
                overlap_ratio,
                entropy_ours,
                entropy_topk,
            )
            valid_labels = [label for label, _ in question_record.options]
            candidate = Candidate(
                video_id=video_path.stem,
                video_path=video_path,
                question_id=question_record.question_id,
                question=question_record.question,
                options=[
                    {"label": label, "text": text} for label, text in question_record.options
                ],
                target=question_record.answer,
                prediction=extract_answer_label(prediction_raw, valid_labels),
                raw_prediction=prediction_raw,
                frames=display_frames,
                design=design,
                quality=quality,
                ours_indices=ours_indices,
                topk_indices=topk_indices,
                ours_summary=ours_summary,
                topk_summary=topk_summary,
                ridge=ridge,
                frame_count=frame_count,
                tokens_per_frame=tokens_per_frame,
                grid_height=grid_height,
                grid_width=grid_width,
                overlap_ratio=overlap_ratio,
                temporal_entropy_ours=entropy_ours,
                temporal_entropy_topk=entropy_topk,
                d_efficiency=d_efficiency,
                rank_ratio=rank_ratio,
                selection_score=score,
            )
            audit_records.append(_audit_record(candidate, False))
            if best is None or candidate.selection_score > best.selection_score:
                if best is not None:
                    del best
                best = candidate
            print(
                f"[score] {video_path.stem} score={score:.4f} "
                f"D-eff={d_efficiency:.3f}x rank={rank_ratio:.3f}x "
                f"overlap={100.0 * overlap_ratio:.1f}%",
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

    if best is None:
        raise RuntimeError("no candidate produced a valid CertVID visualization capture")
    _plot(best, output_dir, args.dpi)
    audit_records.sort(key=lambda item: float(item["selection_score"]), reverse=True)
    for record in audit_records:
        record["selected_for_figure"] = record["video_id"] == best.video_id
    audit = {
        "selection_policy": (
            "Geometry-only ranking: normalized log-det gain, effective-rank gain, "
            "temporal-entropy gain, and keep-set disagreement. Answers are not used."
        ),
        "seed": args.seed,
        "candidate_count_requested": args.candidate_count,
        "candidate_count_valid": len(audit_records),
        "configuration": {
            "dataset": "VideoMME",
            "model": "LLaVA-OneVision-7B",
            "frames": args.num_frames,
            "retention_ratio": args.retention_ratio,
            "expansion": args.expansion,
            "pruning_layer": args.pruning_layer,
            "llm_retention_ratio": args.llm_retention_ratio,
            "certificate_budget_ratio": 0.0,
        },
        "ranked_candidates": audit_records,
    }
    (output_dir / "doptimal_case_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[complete] selected={best.video_id}", flush=True)
    print(f"[complete] figures={output_dir / '01_quality_topk.png'} ... 06", flush=True)
    print(f"[complete] audit={output_dir / 'doptimal_case_audit.json'}", flush=True)


if __name__ == "__main__":
    main()
