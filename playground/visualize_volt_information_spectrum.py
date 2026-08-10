#!/usr/bin/env python3
"""Compare VOLT-Vid's true high-dimensional information spectrum.

The script runs quality Top-K and D-optimal selection on the same video and
budget, then evaluates both selections and random candidate-pool subsets in the
exact design space emitted by CertVID V3. Hard certificates are disabled.
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
class SpectrumSummary:
    information: np.ndarray
    logdet_gain: float
    effective_rank: float
    median_information: float
    weak_axis_information: float


def parse_args() -> argparse.Namespace:
    hf_home = Path(os.environ.get("HF_HOME", "/home/xuyouwen/hf_home_local"))
    parser = argparse.ArgumentParser(
        description="Visualize equal-budget D-optimal information spectra."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-root", default=str(hf_home / "videomme" / "data"))
    parser.add_argument("--metadata-jsonl", required=True)
    parser.add_argument("--video-id", default="")
    parser.add_argument("--num-examples", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--random-trials", type=int, default=40)
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


def _run_objective(
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
    objective: str,
) -> tuple[str, torch.Tensor, dict[str, Any]]:
    config = model.flashvid_config
    config.certv3_certificate_budget_ratio = 0.0
    config.certv3_use_trajectory = True
    config.certv3_selection_objective = objective
    setattr(config, "_capture_visualization_design", True)
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
    if plan is None or not isinstance(analysis, dict) or "design" not in analysis:
        raise RuntimeError("CertVID design capture failed")
    copied = {
        key: value.detach().cpu().clone() if torch.is_tensor(value) else value
        for key, value in analysis.items()
    }
    selected = plan.anchor_indices.detach().long().cpu().clone()
    setattr(config, "_certvid_plan", None)
    setattr(config, "_visualization_certvid_analysis", None)
    return prediction, selected, copied


def _spectrum(design: torch.Tensor, indices: torch.Tensor, ridge: float) -> SpectrumSummary:
    rows = design.index_select(0, indices).double()
    eigenvalues = torch.linalg.eigvalsh(rows.transpose(0, 1) @ rows).clamp_min(0.0)
    information = torch.log1p(eigenvalues / max(1e-12, ridge))
    information, _ = torch.sort(information, descending=True)
    total = eigenvalues.sum()
    if float(total.item()) <= 0.0:
        effective_rank = 0.0
    else:
        probability = (eigenvalues / total).clamp_min(1e-30)
        effective_rank = float(torch.exp(-(probability * probability.log()).sum()).item())
    dimension = int(information.numel())
    weak_count = max(1, int(math.ceil(0.20 * dimension)))
    return SpectrumSummary(
        information=information.cpu().numpy(),
        logdet_gain=float(information.sum().item()),
        effective_rank=effective_rank,
        median_information=float(information.median().item()),
        weak_axis_information=float(information[-weak_count:].mean().item()),
    )


def _random_spectra(
    design: torch.Tensor,
    candidates: torch.Tensor,
    budget: int,
    ridge: float,
    trials: int,
    seed: int,
) -> list[SpectrumSummary]:
    if candidates.numel() < budget:
        raise RuntimeError("candidate pool is smaller than the selection budget")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    outputs: list[SpectrumSummary] = []
    for _ in range(trials):
        permutation = torch.randperm(candidates.numel(), generator=generator)[:budget]
        outputs.append(_spectrum(design, candidates[permutation], ridge))
    return outputs


def _plot_spectrum(
    *,
    output_path: Path,
    video_id: str,
    question: str,
    budget: int,
    total_tokens: int,
    doptimal: SpectrumSummary,
    quality: SpectrumSummary,
    random_runs: list[SpectrumSummary],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    random_information = np.stack([run.information for run in random_runs], axis=0)
    random_mean = random_information.mean(axis=0)
    random_std = random_information.std(axis=0)
    random_logdet = np.asarray([run.logdet_gain for run in random_runs])
    random_rank = np.asarray([run.effective_rank for run in random_runs])
    random_weak = np.asarray([run.weak_axis_information for run in random_runs])

    teal = "#009E9A"
    orange = "#E58A2B"
    gray = "#9299A1"
    ink = "#17212B"
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.8), constrained_layout=True)

    ax = axes[0]
    x = np.arange(1, doptimal.information.size + 1)
    ax.fill_between(
        x,
        np.maximum(0.0, random_mean - random_std),
        random_mean + random_std,
        color=gray,
        alpha=0.20,
        linewidth=0,
    )
    ax.plot(x, random_mean, color=gray, linewidth=1.4, label="Random candidate subsets")
    ax.plot(x, quality.information, color=orange, linewidth=2.0, label="Quality Top-K")
    ax.plot(x, doptimal.information, color=teal, linewidth=2.5, label="D-optimal (ours)")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel("Information axis, strongest to weakest")
    ax.set_ylabel(r"$\log(1 + \lambda_j / \lambda)$")
    ax.set_title("(a) High-dimensional information spectrum", fontweight="bold")
    ax.grid(alpha=0.18, linewidth=0.7)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    metric_names = ["Log-det\ngain", "Effective\nrank", "Weak-axis\ninformation"]
    random_values = [random_logdet.mean(), random_rank.mean(), random_weak.mean()]
    quality_values = [quality.logdet_gain, quality.effective_rank, quality.weak_axis_information]
    ours_values = [doptimal.logdet_gain, doptimal.effective_rank, doptimal.weak_axis_information]
    values = np.asarray([random_values, quality_values, ours_values], dtype=np.float64)
    normalized = values / np.maximum(values.max(axis=0, keepdims=True), 1e-12)
    positions = np.arange(len(metric_names))
    width = 0.24
    for offset, row, color, label in (
        (-width, 0, gray, "Random"),
        (0.0, 1, orange, "Quality Top-K"),
        (width, 2, teal, "D-optimal"),
    ):
        bars = ax.bar(positions + offset, normalized[row], width, color=color, label=label)
        for column, bar in enumerate(bars):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.025,
                f"{values[row, column]:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=ink,
                rotation=0,
            )
    ax.set_xticks(positions)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0.0, 1.17)
    ax.set_ylabel("Normalized to the best method")
    ax.set_title("(b) Equal-budget geometric coverage", fontweight="bold")
    ax.grid(axis="y", alpha=0.18, linewidth=0.7)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    d_efficiency = math.exp(
        (doptimal.logdet_gain - quality.logdet_gain)
        / max(1, doptimal.information.size)
    )
    short_question = " ".join(question.split())
    if len(short_question) > 130:
        short_question = short_question[:127] + "..."
    fig.suptitle(
        f"D-optimal evidence geometry | {video_id} | K={budget}/{total_tokens} "
        f"| D-efficiency vs Top-K={d_efficiency:.3f}\n{short_question}",
        fontsize=13,
        fontweight="bold",
        color=ink,
    )
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _summary(value: SpectrumSummary) -> dict[str, float]:
    return {
        "logdet_gain": value.logdet_gain,
        "effective_rank": value.effective_rank,
        "median_information": value.median_information,
        "weak_axis_information": value.weak_axis_information,
    }


def main() -> None:
    args = parse_args()
    if args.random_trials <= 0:
        raise ValueError("--random-trials must be positive")
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
        pixels = pixels_cpu.to(device=device, dtype=torch.float16)
        input_ids, attention_mask = prepare_prompt(tokenizer, record.prompt, device)

        quality_raw, quality_indices, _ = _run_objective(
            model=model,
            tokenizer=tokenizer,
            pixel_values=pixels,
            input_ids=input_ids,
            attention_mask=attention_mask,
            retention_ratio=args.retention_ratio,
            expansion=args.expansion,
            llm_retention_ratio=args.llm_retention_ratio,
            max_new_tokens=args.max_new_tokens,
            objective="quality_topk",
        )
        doptimal_raw, doptimal_indices, analysis = _run_objective(
            model=model,
            tokenizer=tokenizer,
            pixel_values=pixels,
            input_ids=input_ids,
            attention_mask=attention_mask,
            retention_ratio=args.retention_ratio,
            expansion=args.expansion,
            llm_retention_ratio=args.llm_retention_ratio,
            max_new_tokens=args.max_new_tokens,
            objective="d_optimal",
        )
        design_value = analysis.get("design")
        candidate_value = analysis.get("candidate_indices")
        if not torch.is_tensor(design_value):
            raise RuntimeError("visualization sidecar is missing the design matrix")
        design = design_value.float().cpu()
        candidates = (
            candidate_value.long().cpu()
            if torch.is_tensor(candidate_value)
            else torch.arange(design.shape[0], dtype=torch.long)
        )
        if quality_indices.numel() != doptimal_indices.numel():
            raise RuntimeError("quality Top-K and D-optimal budgets differ")
        ridge = float(analysis.get("ridge", 0.5))
        budget = int(doptimal_indices.numel())
        doptimal = _spectrum(design, doptimal_indices, ridge)
        quality = _spectrum(design, quality_indices, ridge)
        random_runs = _random_spectra(
            design,
            candidates,
            budget,
            ridge,
            args.random_trials,
            args.seed + 1009 * number,
        )
        image_name = f"information_spectrum_{number:02d}_{video_path.stem}.png"
        _plot_spectrum(
            output_path=output_dir / image_name,
            video_id=video_path.stem,
            question=record.question,
            budget=budget,
            total_tokens=int(design.shape[0]),
            doptimal=doptimal,
            quality=quality,
            random_runs=random_runs,
        )
        labels = [label for label, _ in record.options]
        doptimal_prediction = extract_answer_label(doptimal_raw, labels)
        quality_prediction = extract_answer_label(quality_raw, labels)
        random_logdet = np.asarray([run.logdet_gain for run in random_runs])
        records.append(
            {
                "video_id": video_path.stem,
                "question_id": record.question_id,
                "question": record.question,
                "target": record.answer,
                "doptimal_prediction": doptimal_prediction,
                "quality_topk_prediction": quality_prediction,
                "doptimal_correct": doptimal_prediction == record.answer,
                "quality_topk_correct": quality_prediction == record.answer,
                "raw_tokens": int(design.shape[0]),
                "selected_tokens": budget,
                "candidate_tokens": int(candidates.numel()),
                "certificate_budget_ratio": 0.0,
                "ridge": ridge,
                "selection_overlap": int(torch.isin(doptimal_indices, quality_indices).sum()),
                "doptimal": _summary(doptimal),
                "quality_topk": _summary(quality),
                "random_logdet_mean": float(random_logdet.mean()),
                "random_logdet_std": float(random_logdet.std()),
                "d_efficiency_vs_quality_topk": math.exp(
                    (doptimal.logdet_gain - quality.logdet_gain)
                    / max(1, design.shape[1])
                ),
                "sampled_source_frame_indices": [int(value) for value in source_indices],
                "video_fps": fps,
                "figure": image_name,
            }
        )
        (output_dir / "information_spectrum_metrics.json").write_text(
            json.dumps({"examples": records}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        del pixels, pixels_cpu, input_ids, attention_mask, design
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"[done] D-opt={doptimal.logdet_gain:.2f} "
            f"Top-K={quality.logdet_gain:.2f} Random={random_logdet.mean():.2f}",
            flush=True,
        )

    print(f"[complete] outputs={output_dir}", flush=True)


if __name__ == "__main__":
    main()
