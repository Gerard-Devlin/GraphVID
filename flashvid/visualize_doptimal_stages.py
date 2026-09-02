#!/usr/bin/env python3
"""Render three real DOVE D-optimal selection stages as separate figures."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


# LLaVA registers unused remote-backed templates during import. Keep this
# visualization fully local, just like the evaluation scripts.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_EVALUATE_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
PLAYGROUND = ROOT / "playground"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PLAYGROUND) not in sys.path:
    sys.path.insert(0, str(PLAYGROUND))

from flashvid.certvid_v3 import _d_optimal_unconstrained_columns  # noqa: E402
from visualize_certvid_two_examples import (  # noqa: E402
    generate_once,
    load_certvid_model,
    prepare_prompt,
    sample_video,
)


ALL_TOKEN = "#315f78"
CANDIDATE_TOKEN = "#244f68"
SELECTED_TOKEN = "#c94b43"
EXCHANGED_TOKEN = "#179b68"
ENVELOPE_FILL = "#e88b82"
ENVELOPE_EDGE = "#a83432"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model-path",
        default=os.environ.get(
            "PRETRAINED", "/home/xuyouwen/models/llava-onevision-qwen2-7b-ov"
        ),
    )
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--retention-ratio", type=float, default=0.01)
    parser.add_argument("--expansion", type=float, default=1.30)
    parser.add_argument("--pruning-layer", type=int, default=20)
    parser.add_argument("--llm-retention-ratio", type=float, default=0.1923076923)
    parser.add_argument(
        "--pool-mode", choices=("bilinear", "average", "max"), default="bilinear"
    )
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument(
        "--seed-count",
        type=int,
        default=12,
        help="Number of earliest real greedy selections shown as the seed snapshot.",
    )
    parser.add_argument(
        "--max-swap-arrows",
        type=int,
        default=6,
        help="Maximum number of real exchanged-in tokens emphasized in stage three.",
    )
    parser.add_argument(
        "--envelope-bandwidth",
        type=float,
        default=0.0,
        help="KDE bandwidth in PCA coordinates; zero selects it from token spacing.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _pca2(candidate_design: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    values = candidate_design.double()
    centered = values - values.mean(dim=0, keepdim=True)
    covariance = centered.transpose(0, 1) @ centered
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)[:2]
    coordinates = centered @ eigenvectors[:, order]
    explained = eigenvalues[order].clamp_min(0.0)
    explained = explained / eigenvalues.clamp_min(0.0).sum().clamp_min(1e-30)
    coordinates_np = coordinates.float().cpu().numpy()
    scale = float(np.quantile(np.linalg.norm(coordinates_np, axis=1), 0.985))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    coordinates_np = coordinates_np / scale
    return coordinates_np, explained.float().cpu().numpy()


def _replay_greedy_order(
    design: torch.Tensor,
    candidates: torch.Tensor,
    budget: int,
    ridge: float,
    device: torch.device,
) -> torch.Tensor:
    design_device = design.to(device=device, dtype=torch.float32)
    candidates_device = candidates.to(device=device, dtype=torch.long)
    rows = design_device[candidates_device]
    columns = _d_optimal_unconstrained_columns(
        rows,
        int(budget),
        max(1e-4, float(ridge)),
    )
    return candidates_device[columns].detach().long().cpu()


def _fedorov_trace(
    *,
    selected: torch.Tensor,
    candidates: torch.Tensor,
    design: torch.Tensor,
    ridge: float,
    steps: int,
    pool_size: int,
    margin: float,
    device: torch.device,
) -> tuple[torch.Tensor, list[dict[str, float | int]], float]:
    """Replay the repository's no-certificate eager exchange and record swaps."""
    candidates_device = candidates.to(device=device, dtype=torch.long)
    rows = design.to(device=device, dtype=torch.float32)[candidates_device]
    selected_device = selected.to(device=device, dtype=torch.long)
    selected_columns = torch.searchsorted(candidates_device, selected_device)
    dimension = int(rows.shape[1])
    identity = torch.eye(dimension, dtype=torch.float32, device=device)
    ridge = max(1e-4, float(ridge))
    trace: list[dict[str, float | int]] = []

    for step in range(max(0, int(steps))):
        selected_rows = rows[selected_columns]
        information = ridge * identity + selected_rows.transpose(0, 1) @ selected_rows
        inverse = torch.linalg.inv(information)
        selected_leverage = torch.sum(
            (selected_rows @ inverse) * selected_rows,
            dim=1,
        ).clamp(0.0, 1.0 - 1e-5)
        removal_loss = -torch.log1p(-selected_leverage)

        removable_positions = torch.arange(
            len(selected_columns), dtype=torch.long, device=device
        )
        token_order = torch.argsort(
            candidates_device[selected_columns[removable_positions]], stable=True
        )
        removable_positions = removable_positions[token_order]
        loss_order = torch.argsort(removal_loss[removable_positions], stable=True)
        removable_positions = removable_positions[loss_order]
        removable_positions = removable_positions[: max(1, int(pool_size))]

        outside_mask = torch.ones(len(candidates_device), dtype=torch.bool, device=device)
        outside_mask[selected_columns] = False
        outside_columns = torch.where(outside_mask)[0]
        if not len(outside_columns):
            break
        outside_rows = rows[outside_columns]
        outside_leverage = torch.sum(
            (outside_rows @ inverse) * outside_rows,
            dim=1,
        )
        outside_order = torch.argsort(
            outside_leverage, descending=True, stable=True
        )[: max(1, int(pool_size))]
        outside_columns = outside_columns[outside_order]
        outside_rows = rows[outside_columns]

        local_deltas: list[torch.Tensor] = []
        local_adds: list[torch.Tensor] = []
        for position in removable_positions.unbind():
            removed = selected_rows[position]
            direction = inverse @ removed
            remove_denominator = (
                1.0 - torch.dot(removed, direction)
            ).clamp_min(1e-5)
            inverse_without = (
                inverse
                + torch.outer(direction, direction) / remove_denominator
            )
            add_leverage = torch.sum(
                (outside_rows @ inverse_without) * outside_rows,
                dim=1,
            ).clamp_min(0.0)
            delta = torch.log(remove_denominator) + torch.log1p(add_leverage)
            local = torch.argmax(delta)
            local_deltas.append(delta[local])
            local_adds.append(outside_columns[local])

        local_deltas_tensor = torch.stack(local_deltas)
        best_local = torch.argmax(local_deltas_tensor)
        best_delta = float(local_deltas_tensor[best_local].item())
        if best_delta <= float(margin):
            break
        remove_position = removable_positions[best_local]
        add_column = torch.stack(local_adds)[best_local]
        removed_token = int(candidates_device[selected_columns[remove_position]].item())
        added_token = int(candidates_device[add_column].item())
        selected_columns[remove_position] = add_column
        trace.append(
            {
                "step": step + 1,
                "removed_token": removed_token,
                "added_token": added_token,
                "delta_logdet": best_delta,
            }
        )

    final_rows = rows[selected_columns]
    final_information = ridge * identity + final_rows.transpose(0, 1) @ final_rows
    sign, logabsdet = torch.linalg.slogdet(final_information)
    logdet = float(logabsdet.item()) if float(sign.item()) > 0 else float("-inf")
    return candidates_device[selected_columns].detach().long().cpu(), trace, logdet


def _style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 11.0,
        }
    )


def _base_axis(
    axis: Any,
    coordinates: np.ndarray,
    candidate_tokens: np.ndarray,
) -> None:
    axis.set_facecolor("none")
    axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=1.35,
        c=ALL_TOKEN,
        alpha=0.16,
        linewidths=0,
        zorder=1,
    )
    if len(candidate_tokens):
        axis.scatter(
            coordinates[candidate_tokens, 0],
            coordinates[candidate_tokens, 1],
            s=2.25,
            c=CANDIDATE_TOKEN,
            alpha=0.32,
            linewidths=0,
            zorder=2,
        )
    axis.set_xlim(-1.23, 1.23)
    axis.set_ylim(-1.03, 1.03)
    axis.set_aspect("auto")
    axis.set_axis_off()


def _draw_selected(
    axis: Any,
    coordinates: np.ndarray,
    selected_tokens: np.ndarray,
    *,
    color: str = SELECTED_TOKEN,
    size: float = 8.5,
) -> None:
    if not len(selected_tokens):
        return
    axis.scatter(
        coordinates[selected_tokens, 0],
        coordinates[selected_tokens, 1],
        s=size,
        c=color,
        alpha=0.90,
        linewidths=0,
        zorder=4,
    )


def _draw_evidence_envelope(
    axis: Any,
    coordinates: np.ndarray,
    selected_tokens: np.ndarray,
    requested_bandwidth: float,
) -> float:
    """Draw a data-derived nonparametric envelope around selected tokens."""
    points = coordinates[selected_tokens]
    if len(points) < 2:
        return 0.0

    pairwise = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    pairwise[np.diag_indices_from(pairwise)] = np.inf
    nearest = np.min(pairwise, axis=1)
    finite_nearest = nearest[np.isfinite(nearest) & (nearest > 1e-8)]
    if requested_bandwidth > 0.0:
        bandwidth = float(requested_bandwidth)
    elif len(finite_nearest):
        bandwidth = float(np.median(finite_nearest) * 1.35)
    else:
        bandwidth = 0.10
    bandwidth = float(np.clip(bandwidth, 0.040, 0.14))

    x = np.linspace(-1.23, 1.23, 320)
    y = np.linspace(-1.03, 1.03, 210)
    xx, yy = np.meshgrid(x, y)
    dx = xx[..., None] - points[:, 0]
    dy = yy[..., None] - points[:, 1]
    density = np.exp(
        -0.5 * (dx * dx + dy * dy) / (bandwidth * bandwidth)
    ).sum(axis=2)

    point_dx = points[:, None, 0] - points[None, :, 0]
    point_dy = points[:, None, 1] - points[None, :, 1]
    point_density = np.exp(
        -0.5
        * (point_dx * point_dx + point_dy * point_dy)
        / (bandwidth * bandwidth)
    ).sum(axis=1)
    level = max(1e-5, float(point_density.min()) * 0.34)
    maximum = float(density.max())
    if level >= maximum:
        level = maximum * 0.5

    axis.contourf(
        xx,
        yy,
        density,
        levels=[level, maximum + 1e-6],
        colors=[ENVELOPE_FILL],
        alpha=0.09,
        antialiased=True,
        zorder=0,
    )
    axis.contour(
        xx,
        yy,
        density,
        levels=[level],
        colors=[ENVELOPE_EDGE],
        linewidths=0.65,
        alpha=0.65,
        antialiased=True,
        zorder=3,
    )
    return bandwidth


def _save_stage(fig: Any, output_dir: Path, stem: str, dpi: int) -> None:
    for extension in ("png", "pdf", "svg"):
        fig.savefig(
            output_dir / f"{stem}.{extension}",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.0,
            transparent=True,
        )


def _plot_stages(
    *,
    output_dir: Path,
    coordinates: np.ndarray,
    candidate_tokens: np.ndarray,
    greedy_tokens: np.ndarray,
    final_tokens: np.ndarray,
    seed_count: int,
    trace_tokens: list[tuple[int, int, float]],
    envelope_bandwidth: float,
    dpi: int,
) -> dict[str, float]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style()
    seed_tokens = greedy_tokens[:seed_count]
    bandwidths: dict[str, float] = {}

    fig, axis = plt.subplots(figsize=(5.2, 2.75), facecolor="none")
    _base_axis(axis, coordinates, candidate_tokens)
    bandwidths["seed_set"] = _draw_evidence_envelope(
        axis,
        coordinates,
        seed_tokens,
        envelope_bandwidth,
    )
    _draw_selected(axis, coordinates, seed_tokens, size=9.0)
    _save_stage(fig, output_dir, "01_seed_set", dpi)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5.2, 2.75), facecolor="none")
    _base_axis(axis, coordinates, candidate_tokens)
    bandwidths["greedy"] = _draw_evidence_envelope(
        axis,
        coordinates,
        greedy_tokens,
        envelope_bandwidth,
    )
    _draw_selected(axis, coordinates, greedy_tokens)
    _save_stage(fig, output_dir, "02_greedy_max_delta", dpi)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5.2, 2.75), facecolor="none")
    _base_axis(axis, coordinates, candidate_tokens)
    bandwidths["fedorov"] = _draw_evidence_envelope(
        axis,
        coordinates,
        final_tokens,
        envelope_bandwidth,
    )
    _draw_selected(axis, coordinates, final_tokens)
    exchanged_in = np.asarray(
        [added for _, added, _ in trace_tokens],
        dtype=np.int64,
    )
    _draw_selected(
        axis,
        coordinates,
        exchanged_in,
        color=EXCHANGED_TOKEN,
        size=12.5,
    )
    _save_stage(fig, output_dir, "03_fedorov_exchange", dpi)
    plt.close(fig)
    return bandwidths


def main() -> None:
    args = parse_args()
    video_path = Path(args.video_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"video does not exist: {video_path}")
    if args.seed_count <= 0:
        raise ValueError("--seed-count must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model, image_processor, device = load_certvid_model(args)
    config = model.flashvid_config
    setattr(config, "certv3_certificate_budget_ratio", 0.0)
    setattr(config, "certv3_use_trajectory", True)
    setattr(config, "certv3_use_query", True)
    setattr(config, "certv3_use_candidate_pool", True)
    setattr(config, "_capture_visualization_design", True)

    frames, source_indices, fps = sample_video(video_path, args.num_frames)
    pixel_values_cpu = image_processor.preprocess(frames, return_tensors="pt")[
        "pixel_values"
    ]
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
    analysis = getattr(config, "_visualization_certvid_analysis", None)
    if plan is None or not isinstance(analysis, dict):
        raise RuntimeError("DOVE did not publish its visualization state")
    if "design" not in analysis or "candidate_indices" not in analysis:
        raise RuntimeError("captured state is missing design or candidate indices")

    design = analysis["design"].float().cpu()
    candidates = analysis["candidate_indices"].long().cpu().sort().values
    final_selected = plan.anchor_indices.detach().long().cpu().sort().values
    budget = int(final_selected.numel())
    ridge = float(analysis.get("ridge", 0.5))
    greedy_order = _replay_greedy_order(
        design, candidates, budget, ridge, device
    )
    refined, trace, replay_logdet = _fedorov_trace(
        selected=greedy_order,
        candidates=candidates,
        design=design,
        ridge=ridge,
        steps=int(getattr(config, "certv3_swap_steps", 6)),
        pool_size=int(getattr(config, "certv3_swap_pool", 24)),
        margin=float(getattr(config, "certv3_swap_margin", 1e-4)),
        device=device,
    )
    if not torch.equal(refined.sort().values, final_selected):
        raise RuntimeError(
            "visualization replay differs from the actual final DOVE selection"
        )

    # Fit one shared display projection to all raw visual tokens. Candidate and
    # selected sets are then overlaid using their original flattened token ids.
    coordinates, explained = _pca2(design)
    candidate_tokens = candidates.numpy()
    greedy_tokens = greedy_order.numpy()
    final_tokens = final_selected.numpy()
    trace_tokens = [
        (
            int(record["removed_token"]),
            int(record["added_token"]),
            float(record["delta_logdet"]),
        )
        for record in trace[: max(0, args.max_swap_arrows)]
    ]
    seed_count = min(args.seed_count, len(greedy_tokens))
    envelope_bandwidths = _plot_stages(
        output_dir=output_dir,
        coordinates=coordinates,
        candidate_tokens=candidate_tokens,
        greedy_tokens=greedy_tokens,
        final_tokens=final_tokens,
        seed_count=seed_count,
        trace_tokens=trace_tokens,
        envelope_bandwidth=args.envelope_bandwidth,
        dpi=args.dpi,
    )

    audit: dict[str, Any] = {
        "video_path": str(video_path),
        "question": args.question,
        "prediction": prediction,
        "num_frames": args.num_frames,
        "sampled_source_frame_indices": [int(value) for value in source_indices],
        "fps": float(fps),
        "retention_ratio": args.retention_ratio,
        "raw_token_count": int(design.shape[0]),
        "candidate_count": int(len(candidates)),
        "selected_count": budget,
        "seed_snapshot_count": seed_count,
        "seed_snapshot_definition": (
            "The first K selections in the actual no-certificate greedy D-optimal order; "
            "this is a visualization checkpoint, not a certificate set."
        ),
        "ridge": ridge,
        "pca_explained_variance": [float(value) for value in explained],
        "candidate_indices": [int(value) for value in candidates.tolist()],
        "greedy_order_before_fedorov": [int(value) for value in greedy_order.tolist()],
        "final_selected_indices": [int(value) for value in final_selected.tolist()],
        "fedorov_swap_count": len(trace),
        "fedorov_trace": trace,
        "replayed_final_logdet": replay_logdet,
        "projected_envelope": {
            "method": "Gaussian KDE level set over selected tokens in the shared PCA plane",
            "bandwidths": envelope_bandwidths,
            "requested_bandwidth": args.envelope_bandwidth,
        },
        "visualization_note": (
            "All three transparent figures share one PCA projection fitted to every captured "
            "whitened visual-token design vector. Candidate and selected tokens are overlaid "
            "using their original flattened token ids. D-optimal selection itself runs in "
            "the complete design space."
        ),
    }
    (output_dir / "doptimal_stages.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    del model, pixel_values, pixel_values_cpu, input_ids, attention_mask, design
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[complete] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
