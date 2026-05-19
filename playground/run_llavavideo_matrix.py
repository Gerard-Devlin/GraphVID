from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "lmms-lab/LLaVA-Video-7B-Qwen2"


def _parse_rates(text: str) -> list[tuple[str, float]]:
    rates: list[tuple[str, float]] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        ratio = value / 100.0 if value > 1.0 else value
        label = f"{ratio * 100:g}".replace(".", "p")
        rates.append((label, ratio))
    if not rates:
        raise ValueError("no retention rates provided")
    return rates


def _count_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _mean(summary: dict[str, Any] | None, phase: str, key: str) -> float | None:
    if not summary:
        return None
    section = summary.get(phase)
    if not isinstance(section, dict):
        return None
    stat = section.get(key)
    if isinstance(stat, dict) and stat.get("mean") is not None:
        return float(stat["mean"])
    return None


def _acc(summary: dict[str, Any] | None, phase: str) -> float | None:
    if not summary:
        return None
    section = summary.get(phase)
    if not isinstance(section, dict) or section.get("accuracy") is None:
        return None
    return float(section["accuracy"]) * 100.0


def _duration_acc(summary: dict[str, Any] | None, phase: str, duration: str) -> float | None:
    if not summary:
        return None
    bucket = summary.get("duration_breakdown", {}).get(duration, {})
    section = bucket.get(phase) if isinstance(bucket, dict) else None
    if not isinstance(section, dict) or section.get("accuracy") is None:
        return None
    return float(section["accuracy"]) * 100.0


def _stat_mean(value: Any) -> float | None:
    if isinstance(value, dict) and value.get("mean") is not None:
        return float(value["mean"])
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _comparison(summary: dict[str, Any] | None, key: str) -> float | None:
    if not summary:
        return None
    comp = summary.get("comparison", {}).get("flashvid_vs_graphvid", {})
    if not isinstance(comp, dict):
        return None
    key_aliases = {
        "latency_speedup": ["latency_speedup_ratio", "latency_speedup"],
        "visual_token_reduction": [
            "visual_token_reduction_vs_flashvid",
            "visual_token_reduction",
        ],
    }
    for candidate in key_aliases.get(key, [key]):
        value = _stat_mean(comp.get(candidate))
        if value is not None:
            return value
    return None


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _build_command(args: argparse.Namespace, rate_label: str, ratio: float, total_limit: int) -> tuple[list[str], Path]:
    run_tag = f"{args.tag}_r{rate_label}_videomme{total_limit}"
    summary_path = REPO_ROOT / "logs" / "efficiency" / "parallel" / run_tag / f"{run_tag}_summary.json"
    cmd = [
        sys.executable,
        "-u",
        "playground/run_talon_parallel.py",
        "--model_backend",
        "llava",
        "--model_path",
        args.model_path,
        "--dataset_jsonl",
        args.dataset_jsonl,
        "--hf_home",
        args.hf_home,
        "--total_limit",
        str(total_limit),
        "--num_frames",
        str(args.num_frames),
        "--num_warmup",
        str(args.num_warmup),
        "--num_runs",
        str(args.num_runs),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--attn_implementation",
        args.attn_implementation,
        "--free_ratio",
        str(args.free_ratio),
        "--min_free_mb",
        str(args.min_free_mb),
        "--max_gpus",
        str(args.max_gpus),
        "--tag",
        run_tag,
        "--no-run_ours",
        "--run_flashvid",
        "--run_graphvid",
        "--retention_ratio",
        str(ratio),
        "--expansion",
        str(args.expansion),
        "--llm_retention_ratio",
        str(args.llm_retention_ratio),
        "--token_selection_method",
        args.token_selection_method,
        "--flashvid_token_selection_method",
        args.flashvid_token_selection_method,
        "--graphvid_token_selection_method",
        args.graphvid_token_selection_method,
        "--graph_temporal_topk",
        str(args.graph_temporal_topk),
        "--graph_temporal_radius",
        str(args.graph_temporal_radius),
        "--graph_temporal_skip",
        str(args.graph_temporal_skip),
        "--graph_merge_protect_ratio",
        str(args.graph_merge_protect_ratio),
        "--graph_merge_target_ratio",
        str(args.graph_merge_target_ratio),
        "--graph_merge_representative",
        args.graph_merge_representative,
        "--graph_final_tokens_per_frame",
        str(args.graph_final_tokens_per_frame),
        "--graph_final_frame_floor_ratio",
        str(args.graph_final_frame_floor_ratio),
    ]
    if args.graph_skip_spatial_merge_when_capped:
        cmd.append("--graph_skip_spatial_merge_when_capped")
    else:
        cmd.append("--no-graph_skip_spatial_merge_when_capped")
    if args.gpu_ids:
        cmd.extend(["--gpu_ids", args.gpu_ids])
    cmd.extend(args.extra_args)
    return cmd, summary_path


def _load_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _row(rate_label: str, summary: dict[str, Any] | None) -> dict[str, Any]:
    flash_tokens = _mean(summary, "flashvid", "compressed_visual_tokens")
    graph_tokens = _mean(summary, "graphvid", "compressed_visual_tokens")
    flash_acc = _acc(summary, "flashvid")
    graph_acc = _acc(summary, "graphvid")
    return {
        "retention_ratio": f"{rate_label.replace('p', '.')}%",
        "flashvid_acc": flash_acc,
        "graphvid_acc": graph_acc,
        "acc_delta": None if flash_acc is None or graph_acc is None else graph_acc - flash_acc,
        "flashvid_short": _duration_acc(summary, "flashvid", "short"),
        "graphvid_short": _duration_acc(summary, "graphvid", "short"),
        "flashvid_medium": _duration_acc(summary, "flashvid", "medium"),
        "graphvid_medium": _duration_acc(summary, "graphvid", "medium"),
        "flashvid_long": _duration_acc(summary, "flashvid", "long"),
        "graphvid_long": _duration_acc(summary, "graphvid", "long"),
        "flashvid_tokens": flash_tokens,
        "graphvid_tokens": graph_tokens,
        "token_reduction": _comparison(summary, "visual_token_reduction"),
        "flashvid_latency_ms": _mean(summary, "flashvid", "latency_ms"),
        "graphvid_latency_ms": _mean(summary, "graphvid", "latency_ms"),
        "latency_speedup": _comparison(summary, "latency_speedup"),
    }


def _write_tables(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "llavavideo_matrix.csv"
    md_path = out_dir / "llavavideo_matrix.md"
    json_path = out_dir / "llavavideo_matrix.json"
    fields = [
        "retention_ratio",
        "flashvid_acc",
        "graphvid_acc",
        "acc_delta",
        "flashvid_short",
        "graphvid_short",
        "flashvid_medium",
        "graphvid_medium",
        "flashvid_long",
        "graphvid_long",
        "flashvid_tokens",
        "graphvid_tokens",
        "token_reduction",
        "flashvid_latency_ms",
        "graphvid_latency_ms",
        "latency_speedup",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    lines = [
        "| R | FlashVID Acc | GraphVID Acc | Delta | F Short | G Short | F Medium | G Medium | F Long | G Long | F Tokens | G Tokens | Token Red. | F Lat. | G Lat. | Speedup |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {r} | {fa} | {ga} | {d} | {fs} | {gs} | {fm} | {gm} | {fl} | {gl} | {ft} | {gt} | {tr} | {flat} | {glat} | {sp} |".format(
                r=row["retention_ratio"],
                fa=_fmt(row.get("flashvid_acc")),
                ga=_fmt(row.get("graphvid_acc")),
                d=_fmt(row.get("acc_delta")),
                fs=_fmt(row.get("flashvid_short")),
                gs=_fmt(row.get("graphvid_short")),
                fm=_fmt(row.get("flashvid_medium")),
                gm=_fmt(row.get("graphvid_medium")),
                fl=_fmt(row.get("flashvid_long")),
                gl=_fmt(row.get("graphvid_long")),
                ft=_fmt(row.get("flashvid_tokens")),
                gt=_fmt(row.get("graphvid_tokens")),
                tr=_fmt(row.get("token_reduction")),
                flat=_fmt(row.get("flashvid_latency_ms")),
                glat=_fmt(row.get("graphvid_latency_ms")),
                sp=_fmt(row.get("latency_speedup"), 3),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[matrix] csv={csv_path}")
    print(f"[matrix] md={md_path}")
    print(f"[matrix] json={json_path}")


def main() -> None:
    hf_home = os.environ.get("HF_HOME", "/gluster/envs/users/wuzhijian/hf_home")
    parser = argparse.ArgumentParser(description="Run LLaVA-Video FlashVID vs GraphVID matrix.")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_ID)
    parser.add_argument("--dataset_jsonl", default="assets/videomme.jsonl")
    parser.add_argument("--hf_home", default=hf_home)
    parser.add_argument("--rates", default="10,20")
    parser.add_argument("--tag", default="llavavideo_graphvid_vs_flashvid")
    parser.add_argument("--output_dir", default="logs/efficiency/matrix/llavavideo")
    parser.add_argument("--total_limit", type=int, default=2700, help="0 means all rows in dataset_jsonl.")
    parser.add_argument("--num_frames", type=int, default=64)
    parser.add_argument("--num_warmup", type=int, default=0)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--free_ratio", type=float, default=0.75)
    parser.add_argument("--min_free_mb", type=int, default=20000)
    parser.add_argument("--max_gpus", type=int, default=0)
    parser.add_argument("--gpu_ids", default="")
    parser.add_argument("--expansion", type=float, default=1.25)
    parser.add_argument("--llm_retention_ratio", type=float, default=1.0)
    parser.add_argument("--token_selection_method", default="attn_div_v2")
    parser.add_argument("--flashvid_token_selection_method", default="attn_div_v2")
    parser.add_argument("--graphvid_token_selection_method", default="attn_div_stable")
    parser.add_argument("--graph_temporal_topk", type=int, default=2)
    parser.add_argument("--graph_temporal_radius", type=int, default=1)
    parser.add_argument("--graph_temporal_skip", type=int, default=1)
    parser.add_argument("--graph_merge_protect_ratio", type=float, default=0.15)
    parser.add_argument("--graph_merge_target_ratio", type=float, default=1.00)
    parser.add_argument("--graph_merge_representative", default="medoid", choices=["medoid", "mean"])
    parser.add_argument("--graph_final_tokens_per_frame", type=int, default=0)
    parser.add_argument("--graph_final_frame_floor_ratio", type=float, default=0.55)
    parser.add_argument("--graph_skip_spatial_merge_when_capped", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args, extra_args = parser.parse_known_args()
    args.extra_args = extra_args

    dataset_path = Path(args.dataset_jsonl)
    if not dataset_path.is_absolute():
        dataset_path = REPO_ROOT / dataset_path
    total_limit = int(args.total_limit) if int(args.total_limit) > 0 else _count_jsonl(dataset_path)

    rows: list[dict[str, Any]] = []
    for rate_label, ratio in _parse_rates(args.rates):
        cmd, summary_path = _build_command(args, rate_label, ratio, total_limit)
        print("[run]", " ".join(shlex.quote(x) for x in cmd))
        if not args.dry_run and not (args.resume and summary_path.exists()):
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        rows.append(_row(rate_label, _load_summary(summary_path)))
        _write_tables(REPO_ROOT / args.output_dir / args.tag, rows)


if __name__ == "__main__":
    main()
