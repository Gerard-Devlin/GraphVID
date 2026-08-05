#!/usr/bin/env python3
"""Visualize CertVID V3's regularized D-optimal token volume.

The figure compares three equal-budget selections in the exact design space
used by CertVID V3: CertVID anchors, global visual-attention Top-K, and random
tokens. It reports the regularized log-determinant gain rather than pretending
that a two-dimensional projection is the original high-dimensional volume.
"""

from __future__ import annotations

import argparse
import json
import math
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
)


@dataclass(frozen=True)
class InformationSummary:
    logdet_gain: float
    geometric_gain: float
    effective_rank: float
    axis_information: np.ndarray


def parse_args() -> argparse.Namespace:
    hf_home = Path(os.environ.get("HF_HOME", "/home/xuyouwen/hf_home_local"))
    parser = argparse.ArgumentParser(
        description="Visualize CertVID V3's equal-budget D-optimal token volume."
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get(
            "PRETRAINED", "/home/xuyouwen/models/llava-onevision-qwen2-7b-ov"
        ),
    )
    parser.add_argument(
        "--dataset-root",
        default=str(hf_home / "egoschema" / "videos"),
    )
    parser.add_argument("--metadata-jsonl", required=True)
    parser.add_argument(
        "--video-id",
        default="",
        help="Optional comma-separated video stems/q_uids.",
    )
    parser.add_argument("--num-examples", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--random-trials", type=int, default=20)
    parser.add_argument("--retention-ratio", type=float, default=0.25)
    parser.add_argument("--expansion", type=float, default=1.30)
    parser.add_argument("--pruning-layer", type=int, default=20)
    parser.add_argument("--llm-retention-ratio", type=float, default=0.1923076923)
    parser.add_argument(
        "--pool-mode", choices=("bilinear", "average", "max"), default="bilinear"
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _requested_ids(raw: str) -> list[str]:
    values = [Path(value.strip()).stem for value in raw.split(",") if value.strip()]
    if len(values) != len(set(values)):
        raise ValueError("--video-id values must be unique")
    return values


def _information_summary(
    design: torch.Tensor,
    indices: torch.Tensor,
    ridge: float,
) -> InformationSummary:
    rows = design[indices].double()
    signal = rows.transpose(0, 1) @ rows
    eigenvalues = torch.linalg.eigvalsh(signal).clamp_min(0.0)
    axis_information = torch.log1p(eigenvalues / ridge)
    logdet_gain = float(axis_information.sum().item())
    dimension = max(1, int(design.shape[1]))
    geometric_gain = float(math.exp(min(80.0, logdet_gain / dimension)))

    total = eigenvalues.sum()
    if float(total.item()) <= 0.0:
        effective_rank = 0.0
    else:
        probabilities = (eigenvalues / total).clamp_min(1e-30)
        entropy = -(probabilities * probabilities.log()).sum()
        effective_rank = float(torch.exp(entropy).item())
    return InformationSummary(
        logdet_gain=logdet_gain,
        geometric_gain=geometric_gain,
        effective_rank=effective_rank,
        axis_information=axis_information.cpu().numpy(),
    )


def _pca_coordinates(design: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    values = design.double()
    centered = values - values.mean(dim=0, keepdim=True)
    covariance = centered.transpose(0, 1) @ centered
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    components = eigenvectors[:, [-1, -2]]
    coordinates = centered @ components
    explained = eigenvalues[[-1, -2]].clamp_min(0.0)
    explained = explained / eigenvalues.clamp_min(0.0).sum().clamp_min(1e-30)
    return coordinates.cpu().numpy(), explained.cpu().numpy()


def _random_summaries(
    design: torch.Tensor,
    budget: int,
    ridge: float,
    trials: int,
    seed: int,
) -> list[InformationSummary]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    summaries: list[InformationSummary] = []
    for _ in range(trials):
        indices = torch.randperm(design.shape[0], generator=generator)[:budget]
        summaries.append(_information_summary(design, indices, ridge))
    return summaries


def _plot_volume(
    *,
    output_path: Path,
    video_id: str,
    question: str,
    design: torch.Tensor,
    certvid_indices: torch.Tensor,
    attention_indices: torch.Tensor,
    certvid: InformationSummary,
    attention: InformationSummary,
    random_summaries: list[InformationSummary],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coordinates, explained = _pca_coordinates(design)
    certvid_np = certvid_indices.cpu().numpy()
    attention_np = attention_indices.cpu().numpy()
    random_logdet = np.asarray(
        [summary.logdet_gain for summary in random_summaries], dtype=np.float64
    )
    random_axis = np.stack(
        [summary.axis_information for summary in random_summaries], axis=0
    )

    teal = "#009E9A"
    orange = "#E58A2B"
    gray = "#9AA0A6"
    ink = "#18212B"
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6), constrained_layout=True)
    fig.patch.set_facecolor("white")

    ax = axes[0]
    ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=5,
        color=gray,
        alpha=0.18,
        linewidths=0,
        label="All visual tokens",
    )
    ax.scatter(
        coordinates[attention_np, 0],
        coordinates[attention_np, 1],
        s=13,
        marker="x",
        color=orange,
        alpha=0.55,
        linewidths=0.7,
        label="Attention Top-K",
    )
    ax.scatter(
        coordinates[certvid_np, 0],
        coordinates[certvid_np, 1],
        s=12,
        color=teal,
        alpha=0.62,
        linewidths=0,
        label="CertVID",
    )
    ax.set_title("(a) Design-space coverage", fontweight="bold", color=ink)
    ax.set_xlabel(f"PC 1 ({100.0 * explained[0]:.1f}% variance)")
    ax.set_ylabel(f"PC 2 ({100.0 * explained[1]:.1f}% variance)")
    ax.legend(frameon=False, fontsize=8, loc="best")

    ax = axes[1]
    values = [random_logdet.mean(), attention.logdet_gain, certvid.logdet_gain]
    errors = [random_logdet.std(ddof=1) if len(random_logdet) > 1 else 0.0, 0.0, 0.0]
    bars = ax.bar(
        ["Random", "Attention\nTop-K", "CertVID"],
        values,
        yerr=errors,
        capsize=4,
        color=[gray, orange, teal],
        edgecolor="white",
        linewidth=0.8,
    )
    span = max(values) - min(values)
    label_offset = max(0.5, 0.025 * max(span, max(values)))
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + label_offset,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=ink,
        )
    ax.set_title("(b) Equal-budget log-volume", fontweight="bold", color=ink)
    ax.set_ylabel(r"$\log\det(I + Y_S^\top Y_S / \lambda)$")
    ax.grid(axis="y", alpha=0.18, linewidth=0.7)
    ax.set_axisbelow(True)

    ax = axes[2]
    x = np.arange(1, design.shape[1] + 1)
    random_mean = np.sort(random_axis, axis=1).mean(axis=0)
    random_std = np.sort(random_axis, axis=1).std(axis=0)
    attention_axis = np.sort(attention.axis_information)
    certvid_axis = np.sort(certvid.axis_information)
    ax.fill_between(
        x,
        np.maximum(0.0, random_mean - random_std),
        random_mean + random_std,
        color=gray,
        alpha=0.20,
        linewidth=0,
    )
    ax.plot(x, random_mean, color=gray, linewidth=1.2, label="Random")
    ax.plot(x, attention_axis, color=orange, linewidth=1.5, label="Attention Top-K")
    ax.plot(x, certvid_axis, color=teal, linewidth=1.8, label="CertVID")
    ax.set_title("(c) Information spectrum", fontweight="bold", color=ink)
    ax.set_xlabel("Design axes, weakest to strongest")
    ax.set_ylabel(r"$\log(1 + \lambda_j / \lambda)$")
    ax.grid(alpha=0.18, linewidth=0.7)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    short_question = " ".join(question.split())
    if len(short_question) > 112:
        short_question = short_question[:109] + "..."
    fig.suptitle(
        f"CertVID D-optimal token volume | {video_id} | "
        f"K={len(certvid_indices)}/{len(design)}\n{short_question}",
        fontsize=12,
        fontweight="bold",
        color=ink,
    )
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _summary_json(summary: InformationSummary) -> dict[str, float]:
    return {
        "logdet_gain": summary.logdet_gain,
        "geometric_information_gain": summary.geometric_gain,
        "effective_rank": summary.effective_rank,
    }


def main() -> None:
    args = parse_args()
    if args.num_examples <= 0:
        raise ValueError("num-examples must be positive")
    if args.random_trials <= 0:
        raise ValueError("random-trials must be positive")
    requested = _requested_ids(args.video_id)
    if requested and len(requested) != args.num_examples:
        raise ValueError("--video-id count must match --num-examples")

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    metadata_path = Path(args.metadata_jsonl).expanduser().resolve()
    questions = load_questions(metadata_path)
    if not questions:
        raise RuntimeError(f"no question metadata loaded from {metadata_path}")
    all_videos = discover_videos(dataset_root)
    paths_by_id = {path.stem: path for path in all_videos}
    if requested:
        missing = [video_id for video_id in requested if video_id not in paths_by_id]
        if missing:
            raise FileNotFoundError(f"videos not found: {missing}")
        videos = [paths_by_id[video_id] for video_id in requested]
    else:
        rng = random.Random(args.seed)
        videos = [path for path in all_videos if path.stem in questions]
        rng.shuffle(videos)
        videos = videos[: args.num_examples]
    if len(videos) < args.num_examples:
        raise RuntimeError(f"only found {len(videos)}/{args.num_examples} videos")

    print(f"[setup] videos={len(videos)} output={output_dir}", flush=True)
    tokenizer, model, image_processor, device = load_certvid_model(args)
    setattr(model.flashvid_config, "_capture_visualization_design", True)

    records: list[dict[str, Any]] = []
    for number, path in enumerate(videos, start=1):
        question_record = questions[path.stem][0]
        print(f"[run] {number}/{len(videos)} {path.name}", flush=True)
        frames, _, _ = sample_video(path, args.num_frames)
        pixel_values_cpu = image_processor.preprocess(frames, return_tensors="pt")[
            "pixel_values"
        ]
        pixel_values = pixel_values_cpu.to(device=device, dtype=torch.float16)
        input_ids, attention_mask = prepare_prompt(
            tokenizer, question_record.prompt, device
        )
        prediction, cls_attention, plan = generate_once(
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
        analysis = getattr(
            model.flashvid_config, "_visualization_certvid_analysis", None
        )
        if plan is None or not isinstance(analysis, dict) or "design" not in analysis:
            raise RuntimeError(f"CertVID design capture failed for {path.name}")

        design = analysis["design"].float().cpu()
        ridge = float(analysis.get("ridge", 0.5))
        selected = plan.anchor_indices.detach().long().cpu()
        if design.ndim != 2 or design.shape[0] != plan.raw_token_count:
            raise RuntimeError("captured design shape does not match the CertVID plan")
        budget = int(selected.numel())
        attention_scores = torch.nan_to_num(
            cls_attention.reshape(-1).float(), nan=-float("inf")
        )
        attention_indices = torch.topk(
            attention_scores, k=budget, largest=True, sorted=False
        ).indices.cpu()

        certvid_summary = _information_summary(design, selected, ridge)
        attention_summary = _information_summary(design, attention_indices, ridge)
        random_summaries = _random_summaries(
            design,
            budget,
            ridge,
            args.random_trials,
            args.seed + number * 1009,
        )
        random_logdet = np.asarray(
            [summary.logdet_gain for summary in random_summaries]
        )
        random_geometric = np.asarray(
            [summary.geometric_gain for summary in random_summaries]
        )
        random_rank = np.asarray(
            [summary.effective_rank for summary in random_summaries]
        )

        image_name = f"volume_{number:02d}_{path.stem}.png"
        _plot_volume(
            output_path=output_dir / image_name,
            video_id=path.stem,
            question=question_record.question,
            design=design,
            certvid_indices=selected,
            attention_indices=attention_indices,
            certvid=certvid_summary,
            attention=attention_summary,
            random_summaries=random_summaries,
        )
        dimension = max(1, int(design.shape[1]))
        d_efficiency_vs_attention = math.exp(
            max(
                -80.0,
                min(
                    80.0,
                    (certvid_summary.logdet_gain - attention_summary.logdet_gain)
                    / dimension,
                ),
            )
        )
        valid_labels = [label for label, _ in question_record.options]
        records.append(
            {
                "video_id": path.stem,
                "question_id": question_record.question_id,
                "question": question_record.question,
                "options": [
                    {"label": label, "text": text}
                    for label, text in question_record.options
                ],
                "target": question_record.answer,
                "prediction": extract_answer_label(prediction, valid_labels),
                "raw_prediction": prediction,
                "image": image_name,
                "raw_tokens": int(design.shape[0]),
                "selected_tokens": budget,
                "design_dimension": int(design.shape[1]),
                "ridge": ridge,
                "selection_overlap": int(
                    torch.isin(selected, attention_indices).sum().item()
                ),
                "certvid": _summary_json(certvid_summary),
                "attention_topk": _summary_json(attention_summary),
                "random": {
                    "trials": args.random_trials,
                    "logdet_gain_mean": float(random_logdet.mean()),
                    "logdet_gain_std": float(random_logdet.std()),
                    "geometric_information_gain_mean": float(
                        random_geometric.mean()
                    ),
                    "effective_rank_mean": float(random_rank.mean()),
                },
                "certvid_d_efficiency_vs_attention": d_efficiency_vs_attention,
            }
        )
        (output_dir / "volume_metrics.json").write_text(
            json.dumps({"examples": records}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        setattr(model.flashvid_config, "_certvid_plan", None)
        del pixel_values, pixel_values_cpu, input_ids, attention_mask, design
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"[done] {path.stem} logdet: CertVID={certvid_summary.logdet_gain:.2f} "
            f"Attention={attention_summary.logdet_gain:.2f} "
            f"Random={random_logdet.mean():.2f}",
            flush=True,
        )

    print(f"[complete] figures and metrics written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
