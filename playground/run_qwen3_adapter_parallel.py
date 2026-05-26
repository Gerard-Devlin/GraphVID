from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_METHODS = ("fastvid", "visionzip")


def _artifact_method_name(method: str) -> str:
    return f"{method}_qwen3_adapter"


def _query_free_gpus(free_ratio: float, min_free_mb: int) -> list[int]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out: list[int] = []
    print("[gpu-scan] index free/total util eligible")
    for raw in proc.stdout.strip().splitlines():
        if not raw.strip():
            continue
        idx_text, free_text, total_text, util_text = [part.strip() for part in raw.split(",")]
        idx = int(idx_text)
        free_mb = int(free_text)
        total_mb = int(total_text)
        util = int(util_text)
        ok = free_mb >= min_free_mb and (free_mb / max(1, total_mb)) >= free_ratio
        print(f"[gpu-scan] {idx} {free_mb}/{total_mb} {util}% {'yes' if ok else 'no'}")
        if ok:
            out.append(idx)
    return out


def _split_ranges(start: int, total: int, parts: int) -> list[tuple[int, int]]:
    parts = max(1, min(parts, total))
    base = total // parts
    rem = total % parts
    cursor = start
    out: list[tuple[int, int]] = []
    for idx in range(parts):
        count = base + (1 if idx < rem else 0)
        out.append((cursor, count))
        cursor += count
    return out


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import playground.bench_all_metrics as bench

    return bench._summarize_phase(rows)


def _print_summary(summary: dict[str, Any], method: str) -> None:
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    phase = summary.get(method, {})
    acc = phase.get("accuracy")
    acc_text = f"{acc * 100:.2f}%" if acc is not None else "N/A"
    valid = phase.get("num_valid", 0)
    total = phase.get("num_samples", 0)
    print(f"[{method}] valid={valid}/{total} acc={acc_text}")
    for key, label in (
        ("latency_ms", "latency mean"),
        ("compressed_visual_tokens", "final visual tokens mean"),
        ("vision_compressed_visual_tokens", "vision-side tokens mean"),
    ):
        value = phase.get(key, {}).get("mean")
        if value is not None:
            unit = " ms" if key == "latency_ms" else ""
            print(f"  {label}: {value:.2f}{unit}")
    by_duration = summary.get("duration_breakdown", {})
    if by_duration:
        print("[by duration]")
        for duration in ("short", "medium", "long"):
            bucket = by_duration.get(duration, {}).get(method)
            if not bucket:
                continue
            acc = bucket.get("accuracy")
            acc_text = f"{acc * 100:.2f}%" if acc is not None else "N/A"
            vt = bucket.get("compressed_visual_tokens", {}).get("mean")
            vision = bucket.get("vision_compressed_visual_tokens", {}).get("mean")
            line = f"  [{duration}] [{method}] valid={bucket.get('num_valid', 0)}/{bucket.get('num_samples', 0)} acc={acc_text}"
            if vt is not None:
                line += f" vtoken={vt:.2f}"
            if vision is not None:
                line += f" vision={vision:.2f}"
            print(line)
    print("=" * 72)


def _combine_outputs(args: argparse.Namespace, shard_dir: Path, shard_count: int) -> None:
    artifact_method = _artifact_method_name(args.method)
    all_rows: list[dict[str, Any]] = []
    for idx in range(shard_count):
        all_rows.extend(_read_jsonl(shard_dir / f"{artifact_method}_shard{idx:02d}.jsonl"))
    all_rows.sort(key=lambda row: (str(row.get("sample_id", "")), int(row.get("sample_index", 0) or 0)))

    combined_jsonl = shard_dir / f"{args.tag}_{artifact_method}.jsonl"
    summary_path = shard_dir / f"{args.tag}_summary.json"
    _write_jsonl(combined_jsonl, all_rows)

    summary: dict[str, Any] = {"comparison": {}, args.method: _summarize(all_rows)}
    duration_breakdown: dict[str, Any] = {}
    for duration in ("short", "medium", "long"):
        rows = [row for row in all_rows if str(row.get("duration") or "").strip().lower() == duration]
        if rows:
            duration_breakdown[duration] = {args.method: _summarize(rows), "comparison": {}}
    summary["duration_breakdown"] = duration_breakdown
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[combined] {args.method}={combined_jsonl}")
    print(f"[combined] summary={summary_path}")
    _print_summary(summary, args.method)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel launcher for sidecar Qwen3 baseline adapters.")
    parser.add_argument("--method", choices=SUPPORTED_METHODS, required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_backend", default="qwen3_vl")
    parser.add_argument("--dataset_jsonl", required=True)
    parser.add_argument("--hf_home", default="")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--total_limit", type=int, default=900)
    parser.add_argument("--duration_filter", default="")
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--min_pixels", type=int, default=64 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--num_warmup", type=int, default=1)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--free_ratio", type=float, default=0.75)
    parser.add_argument("--min_free_mb", type=int, default=18000)
    parser.add_argument("--max_gpus", type=int, default=0)
    parser.add_argument("--gpu_ids", default="")
    parser.add_argument("--retention_ratio", type=float, default=0.10)
    parser.add_argument("--expansion", type=float, default=1.25)
    parser.add_argument("--llm_retention_ratio", type=float, default=1.0)
    parser.add_argument("--token_selection_method", default="attn_div_v2")
    parser.add_argument("--adapter_budget_uses_expansion", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--external_budget_uses_expansion",
        dest="adapter_budget_uses_expansion",
        action=argparse.BooleanOptionalAction,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--fastvid_DySeg_c", type=int, default=8)
    parser.add_argument("--fastvid_DySeg_tau", type=float, default=0.90)
    parser.add_argument("--fastvid_DySeg_ignore", type=float, default=0.95)
    parser.add_argument("--fastvid_STPrune_d", type=float, default=0.40)
    parser.add_argument("--fastvid_DTM_p", type=int, default=4)
    parser.add_argument("--fastvid_DTM_beta", type=float, default=0.60)
    parser.add_argument("--visionzip_dominant_ratio", type=float, default=0.85)
    args = parser.parse_args()

    if args.gpu_ids:
        gpu_ids = [int(part.strip()) for part in args.gpu_ids.split(",") if part.strip()]
    else:
        gpu_ids = _query_free_gpus(args.free_ratio, args.min_free_mb)
    if args.max_gpus > 0:
        gpu_ids = gpu_ids[: args.max_gpus]
    if not gpu_ids:
        raise SystemExit("No eligible GPUs found")

    ranges = _split_ranges(args.start_index, args.total_limit, len(gpu_ids))
    shard_dir = REPO_ROOT / "logs" / "efficiency" / "parallel" / args.tag
    shard_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for shard_idx, ((start, limit), gpu_id) in enumerate(zip(ranges, gpu_ids)):
        artifact_method = _artifact_method_name(args.method)
        output_jsonl = shard_dir / f"{artifact_method}_shard{shard_idx:02d}.jsonl"
        summary_json = shard_dir / f"summary_shard{shard_idx:02d}.json"
        log_path = shard_dir / f"run_shard{shard_idx:02d}.log"
        cmd = [
            sys.executable,
            "-u",
            "playground/bench_qwen3_baseline_adapter.py",
            "--method",
            args.method,
            "--model_path",
            args.model_path,
            "--model_backend",
            args.model_backend,
            "--dataset_jsonl",
            args.dataset_jsonl,
            "--hf_home",
            args.hf_home,
            "--tag",
            f"{args.tag}_shard{shard_idx:02d}",
            "--output_jsonl",
            str(output_jsonl),
            "--summary_output_json",
            str(summary_json),
            "--start_index",
            str(start),
            "--limit",
            str(limit),
            "--duration_filter",
            args.duration_filter,
            "--num_frames",
            str(args.num_frames),
            "--min_pixels",
            str(args.min_pixels),
            "--max_pixels",
            str(args.max_pixels),
            "--num_warmup",
            str(args.num_warmup),
            "--num_runs",
            str(args.num_runs),
            "--max_new_tokens",
            str(args.max_new_tokens),
            "--attn_implementation",
            args.attn_implementation,
            "--retention_ratio",
            str(args.retention_ratio),
            "--expansion",
            str(args.expansion),
            "--llm_retention_ratio",
            str(args.llm_retention_ratio),
            "--token_selection_method",
            args.token_selection_method,
            "--adapter_budget_uses_expansion" if args.adapter_budget_uses_expansion else "--no-adapter_budget_uses_expansion",
            "--fastvid_DySeg_c",
            str(args.fastvid_DySeg_c),
            "--fastvid_DySeg_tau",
            str(args.fastvid_DySeg_tau),
            "--fastvid_DySeg_ignore",
            str(args.fastvid_DySeg_ignore),
            "--fastvid_STPrune_d",
            str(args.fastvid_STPrune_d),
            "--fastvid_DTM_p",
            str(args.fastvid_DTM_p),
            "--fastvid_DTM_beta",
            str(args.fastvid_DTM_beta),
            "--visionzip_dominant_ratio",
            str(args.visionzip_dominant_ratio),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[launch] shard={shard_idx} gpu={gpu_id} start={start} limit={limit} log={log_path}")
        log_f = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        jobs.append({"proc": proc, "log_f": log_f, "shard": shard_idx, "log": log_path})

    while True:
        running = 0
        failed: list[dict[str, Any]] = []
        for job in jobs:
            ret = job["proc"].poll()
            if ret is None:
                running += 1
            elif ret != 0 and not job.get("reported"):
                job["reported"] = True
                failed.append(job)
        for job in failed:
            print(f"[failed] shard={job['shard']} code={job['proc'].returncode} log={job['log']}")
        if running == 0:
            break
        print(f"[wait] running={running}/{len(jobs)}")
        time.sleep(30)

    exit_codes = []
    for job in jobs:
        exit_codes.append(job["proc"].wait())
        job["log_f"].close()
    if any(code != 0 for code in exit_codes):
        raise SystemExit(f"Some shards failed: {exit_codes}")

    _combine_outputs(args, shard_dir, len(jobs))


if __name__ == "__main__":
    main()
