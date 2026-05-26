from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GPU_CAP = 4
EXTERNAL_METHODS = ("fastvid", "visionzip", "prunevid")


def _str_bool(value: bool) -> str:
    return "True" if value else "False"


def _query_free_gpus(free_ratio: float, min_free_mb: int) -> list[int]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    gpus: list[int] = []
    print("[gpu-scan] index free/total util eligible")
    for raw in proc.stdout.strip().splitlines():
        if not raw.strip():
            continue
        parts = [p.strip() for p in raw.split(",")]
        idx = int(parts[0])
        free_mb = int(parts[1])
        total_mb = int(parts[2])
        util = int(parts[3])
        ratio = free_mb / max(1, total_mb)
        ok = ratio >= free_ratio and free_mb >= min_free_mb
        print(f"[gpu-scan] {idx} {free_mb}/{total_mb} {util}% {'yes' if ok else 'no'}")
        if ok:
            gpus.append(idx)
    return gpus


def _apply_gpu_cap(gpu_ids: list[int], gpu_cap: int) -> list[int]:
    if gpu_cap <= 0 or len(gpu_ids) <= gpu_cap:
        return gpu_ids
    capped = gpu_ids[:gpu_cap]
    print(f"[gpu-cap] limiting GPUs from {gpu_ids} to {capped} (cap={gpu_cap})")
    return capped


def _split_ranges(start: int, total: int, parts: int) -> list[tuple[int, int]]:
    parts = max(1, min(parts, total))
    base = total // parts
    rem = total % parts
    out = []
    cursor = start
    for i in range(parts):
        count = base + (1 if i < rem else 0)
        out.append((cursor, count))
        cursor += count
    return out


def _append_graphvid_args(cmd: list[str], args: argparse.Namespace, merge_mode: str = "graph") -> None:
    cmd.extend(
        [
            "--llm_retention_ratio",
            str(args.llm_retention_ratio),
            "--retention_ratio",
            str(args.retention_ratio),
            "--expansion",
            str(args.expansion),
            "--temporal_merge_mode",
            merge_mode,
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
            "--graph_representative_position",
            args.graph_representative_position,
            "--graph_protection_attn_weight",
            str(args.graph_protection_attn_weight),
            "--graph_protection_novelty_weight",
            str(args.graph_protection_novelty_weight),
            "--graph_protection_detail_weight",
            str(args.graph_protection_detail_weight),
            "--graph_adaptive_detail_boost",
            str(args.graph_adaptive_detail_boost),
            "--graph_adaptive_protect_boost",
            str(args.graph_adaptive_protect_boost),
            "--graph_merge_importance_penalty",
            str(args.graph_merge_importance_penalty),
            "--graph_final_tokens_per_frame",
            str(args.graph_final_tokens_per_frame),
            "--graph_final_frame_floor_ratio",
            str(args.graph_final_frame_floor_ratio),
        ]
    )
    # bench_all_metrics.py uses HfArgumentParser; pass bool values explicitly
    # instead of argparse-style --no-* flags.
    cmd.extend(["--graph_respect_temporal_threshold", str(bool(args.graph_respect_temporal_threshold))])
    cmd.extend(["--graph_adaptive_detail_protection", str(bool(args.graph_adaptive_detail_protection))])
    cmd.extend(["--graph_skip_spatial_merge_when_capped", str(bool(args.graph_skip_spatial_merge_when_capped))])


def _append_graftvid_args(cmd: list[str], args: argparse.Namespace) -> None:
    _append_graphvid_args(cmd, args, merge_mode="graft")
    cmd.extend(
        [
            "--compression_variant",
            "graftvid",
            "--graft_temporal_topk",
            str(args.graft_temporal_topk),
            "--graft_temporal_radius",
            str(args.graft_temporal_radius),
            "--graft_temporal_skip",
            str(args.graft_temporal_skip),
            "--graft_global_topk",
            str(args.graft_global_topk),
            "--graft_edge_threshold",
            str(args.graft_edge_threshold),
            "--graft_component_radius_eps",
            str(args.graft_component_radius_eps),
            "--graft_split_radius_eps",
            str(args.graft_split_radius_eps),
            "--graft_parent_capacity",
            str(args.graft_parent_capacity),
            "--graft_spatial_penalty",
            str(args.graft_spatial_penalty),
            "--graft_importance_penalty",
            str(args.graft_importance_penalty),
            "--graft_hub_penalty",
            str(args.graft_hub_penalty),
            "--graft_scene_threshold",
            str(args.graft_scene_threshold),
            "--graft_min_tokens_per_frame",
            str(args.graft_min_tokens_per_frame),
            "--graft_budget_diversity_weight",
            str(args.graft_budget_diversity_weight),
            "--graft_score_preset",
            str(args.graft_score_preset),
        ]
    )
    if args.graft_anchor_ratio is not None:
        cmd.extend(["--graft_anchor_ratio", str(args.graft_anchor_ratio)])
    cmd.append("--graft_mutual_knn" if args.graft_mutual_knn else "--no-graft_mutual_knn")
    cmd.append("--graft_one_token_per_frame" if args.graft_one_token_per_frame else "--no-graft_one_token_per_frame")
    cmd.append("--graft_adaptive_aggregation" if args.graft_adaptive_aggregation else "--no-graft_adaptive_aggregation")
    cmd.append("--graft_budget_correction" if args.graft_budget_correction else "--no-graft_budget_correction")
    cmd.append("--graft_input_is_residual" if args.graft_input_is_residual else "--no-graft_input_is_residual")


def _selected_external_method(args: argparse.Namespace) -> str | None:
    selected = [name for name in EXTERNAL_METHODS if bool(getattr(args, f"run_{name}", False))]
    if len(selected) > 1:
        raise SystemExit(f"Enable at most one external baseline per run_parallel launch: {', '.join(selected)}")
    method = selected[0] if selected else None
    if method == "prunevid":
        raise SystemExit(
            "PruneVid is not launched here because the released repository only provides "
            "a PLLaVA/KV-cache implementation, not a runnable Qwen3 adapter. "
            "Refusing to run an approximation as PruneVid."
        )
    return method


def _append_external_args(cmd: list[str], args: argparse.Namespace, method: str) -> None:
    cmd.extend(
        [
            "--compression_variant",
            method,
            "--llm_retention_ratio",
            str(args.llm_retention_ratio),
            "--retention_ratio",
            str(args.retention_ratio),
            "--expansion",
            str(args.expansion),
            "--external_budget_uses_expansion",
            str(bool(args.external_budget_uses_expansion)),
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
    )


def _launch_shards(args: argparse.Namespace, gpu_ids: list[int], work_dir: Path) -> list[dict[str, object]]:
    external_method = _selected_external_method(args)
    ranges = _split_ranges(args.start_index, args.total_limit, len(gpu_ids))
    shard_dir = work_dir / "logs" / "efficiency" / "parallel" / args.tag
    shard_dir.mkdir(parents=True, exist_ok=True)
    jobs = []

    for shard_idx, ((start, limit), gpu_id) in enumerate(zip(ranges, gpu_ids)):
        flashvid_out = shard_dir / f"flashvid_shard{shard_idx:02d}.jsonl"
        graphvid_out = shard_dir / f"graphvid_shard{shard_idx:02d}.jsonl"
        graftvid_out = shard_dir / f"graftvid_shard{shard_idx:02d}.jsonl"
        external_out = shard_dir / f"{external_method}_shard{shard_idx:02d}.jsonl" if external_method else None
        summary_out = shard_dir / f"summary_shard{shard_idx:02d}.json"
        log_out = shard_dir / f"run_shard{shard_idx:02d}.log"

        cmd = [
            sys.executable,
            "-u",
            "playground/bench_all_metrics.py",
            "--model_backend",
            args.model_backend,
            "--model_path",
            args.model_path,
            "--local_files_only",
            _str_bool(args.local_files_only),
            "--dataset_jsonl",
            args.dataset_jsonl,
            "--duration_filter",
            args.duration_filter,
            "--start_index",
            str(start),
            "--limit",
            str(limit),
            "--shuffle",
            "False",
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
            "--token_selection_method",
            args.token_selection_method,
            "--flashvid_token_selection_method",
            args.flashvid_token_selection_method,
            "--graphvid_token_selection_method",
            args.graphvid_token_selection_method,
            "--run_baseline",
            "False",
            "--run_flashvid",
            _str_bool(args.run_flashvid),
            "--run_ours",
            _str_bool(external_method is not None),
            "--run_graphvid",
            _str_bool(args.run_graphvid),
            "--run_graftvid",
            _str_bool(args.run_graftvid),
            "--flashvid_output",
            str(flashvid_out),
            "--graphvid_output",
            str(graphvid_out),
            "--graftvid_output",
            str(graftvid_out),
            "--ours_output",
            str(external_out or (shard_dir / f"ours_shard{shard_idx:02d}.jsonl")),
            "--summary_output_json",
            str(summary_out),
        ]
        if args.run_graphvid:
            _append_graphvid_args(cmd, args)
        if args.run_graftvid:
            _append_graftvid_args(cmd, args)
        if external_method is not None:
            _append_external_args(cmd, args, external_method)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env.setdefault("HF_HOME", args.hf_home)
        env.setdefault("HF_HUB_CACHE", str(Path(env["HF_HOME"]) / "hub"))
        env.setdefault("HF_DATASETS_CACHE", str(Path(env["HF_HOME"]) / "datasets"))

        log_handle = log_out.open("w", encoding="utf-8")
        print(f"[launch] shard={shard_idx} gpu={gpu_id} start={start} limit={limit} log={log_out}")
        proc = subprocess.Popen(
            cmd,
            cwd=work_dir,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        jobs.append(
            {
                "proc": proc,
                "log_handle": log_handle,
                "gpu": gpu_id,
                "start": start,
                "limit": limit,
                "flashvid_out": flashvid_out,
                "graphvid_out": graphvid_out,
                "graftvid_out": graftvid_out,
                "external_method": external_method,
                "external_out": external_out,
                "summary_out": summary_out,
                "log_out": log_out,
            }
        )
    return jobs


def _wait_jobs(jobs: list[dict[str, object]]) -> None:
    failed = []
    while True:
        running = 0
        for job in jobs:
            proc = job["proc"]
            assert isinstance(proc, subprocess.Popen)
            if proc.poll() is None:
                running += 1
        if running == 0:
            break
        print(f"[wait] running={running}/{len(jobs)}")
        time.sleep(30)

    for i, job in enumerate(jobs):
        proc = job["proc"]
        log_handle = job["log_handle"]
        assert isinstance(proc, subprocess.Popen)
        log_handle.close()
        if proc.returncode != 0:
            failed.append((i, proc.returncode, job["log_out"]))
    if failed:
        for idx, code, log in failed:
            print(f"[failed] shard={idx} code={code} log={log}")
        raise SystemExit(1)


def _combine_jsonl(paths: list[Path], out_path: Path) -> list[dict]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with out_path.open("w", encoding="utf-8") as w:
        for path in paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as r:
                for line in r:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    rows.append(row)
                    w.write(line if line.endswith("\n") else line + "\n")
    return rows


def _write_summary(args: argparse.Namespace, jobs: list[dict[str, object]], shard_dir: Path) -> None:
    external_method = _selected_external_method(args)
    sys.path.insert(0, str(REPO_ROOT))
    from playground.bench_all_metrics import (
        _add_duration_breakdown,
        _print_summary,
        _summarize_pairwise_comparison,
        _summarize_phase,
    )

    combined_flashvid = shard_dir / f"{args.tag}_flashvid.jsonl"
    combined_graphvid = shard_dir / f"{args.tag}_graphvid.jsonl"
    combined_graftvid = shard_dir / f"{args.tag}_graftvid.jsonl"
    combined_external = shard_dir / f"{args.tag}_{external_method}.jsonl" if external_method else None
    combined_summary = shard_dir / f"{args.tag}_summary.json"

    flashvid_records = []
    graphvid_records = []
    graftvid_records = []
    external_records = []
    if args.run_flashvid:
        flashvid_records = _combine_jsonl([Path(j["flashvid_out"]) for j in jobs], combined_flashvid)
    if args.run_graphvid:
        graphvid_records = _combine_jsonl([Path(j["graphvid_out"]) for j in jobs], combined_graphvid)
    if args.run_graftvid:
        graftvid_records = _combine_jsonl([Path(j["graftvid_out"]) for j in jobs], combined_graftvid)
    if external_method and combined_external is not None:
        external_records = _combine_jsonl([Path(j["external_out"]) for j in jobs if j.get("external_out")], combined_external)

    summary: dict[str, object] = {"comparison": {}}
    if args.run_flashvid:
        summary["flashvid"] = _summarize_phase(flashvid_records)
    if args.run_graphvid:
        summary["graphvid"] = _summarize_phase(graphvid_records)
    if args.run_graftvid:
        summary["graftvid"] = _summarize_phase(graftvid_records)
    if external_method:
        summary[external_method] = _summarize_phase(external_records)
    if args.run_flashvid and args.run_graphvid:
        summary["comparison"]["flashvid_vs_graphvid"] = _summarize_pairwise_comparison(
            flashvid_records,
            graphvid_records,
            anchor_name="flashvid",
            target_name="graphvid",
        )
    if args.run_flashvid and args.run_graftvid:
        summary["comparison"]["flashvid_vs_graftvid"] = _summarize_pairwise_comparison(
            flashvid_records,
            graftvid_records,
            anchor_name="flashvid",
            target_name="graftvid",
        )
    if args.run_flashvid and external_method:
        summary["comparison"][f"flashvid_vs_{external_method}"] = _summarize_pairwise_comparison(
            flashvid_records,
            external_records,
            anchor_name="flashvid",
            target_name=external_method,
        )
    _add_duration_breakdown(
        summary,
        flashvid_records=flashvid_records if args.run_flashvid else None,
        ours_records=external_records if external_method else None,
        ours_phase_name=external_method or "ours",
        graphvid_records=graphvid_records if args.run_graphvid else None,
        graftvid_records=graftvid_records if args.run_graftvid else None,
    )
    with combined_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    if args.run_graphvid:
        print(f"[combined] graphvid={combined_graphvid}")
    if args.run_graftvid:
        print(f"[combined] graftvid={combined_graftvid}")
    if args.run_flashvid:
        print(f"[combined] flashvid={combined_flashvid}")
    if external_method and combined_external is not None:
        print(f"[combined] {external_method}={combined_external}")
    print(f"[combined] summary={combined_summary}")
    _print_summary(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel FlashVID/GraphVID/GRAFTVID benchmark launcher.")
    parser.add_argument("--model_path", default="/gluster/envs/users/wuzhijian/hf_home/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
    parser.add_argument("--model_backend", default="qwen3_vl")
    parser.add_argument("--dataset_jsonl", default="assets/videomme.jsonl")
    parser.add_argument("--duration_filter", default="", help="Comma-separated durations: short,medium,long.")
    parser.add_argument("--hf_home", default=os.environ.get("HF_HOME", "/gluster/envs/users/wuzhijian/hf_home"))
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--total_limit", type=int, default=200)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--min_pixels", type=int, default=64 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--num_warmup", type=int, default=1)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--token_selection_method", default="attn_div_v2")
    parser.add_argument("--flashvid_token_selection_method", default="attn_div_v2")
    parser.add_argument("--graphvid_token_selection_method", default="attn_div_stable")
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run_flashvid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run_ours", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run_graphvid", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run_graftvid", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run_fastvid", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run_visionzip", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run_prunevid", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--retention_ratio", type=float, default=0.10)
    parser.add_argument("--expansion", type=float, default=1.25)
    parser.add_argument("--llm_retention_ratio", type=float, default=1.0)
    parser.add_argument("--free_ratio", type=float, default=0.75)
    parser.add_argument("--min_free_mb", type=int, default=18000)
    parser.add_argument("--max_gpus", type=int, default=0, help="0 means use eligible GPUs up to --gpu_cap.")
    parser.add_argument("--gpu_cap", type=int, default=DEFAULT_GPU_CAP, help="Hard cap for GPUs per launch. 0 disables the cap.")
    parser.add_argument("--gpu_ids", default="", help="Comma-separated GPU ids. Overrides auto selection.")
    parser.add_argument("--tag", default="parallel_benchmark")
    parser.add_argument("--graph_temporal_topk", type=int, default=3)
    parser.add_argument("--graph_temporal_radius", type=int, default=1)
    parser.add_argument("--graph_temporal_skip", type=int, default=1)
    parser.add_argument("--graph_merge_protect_ratio", type=float, default=0.15)
    parser.add_argument("--graph_merge_target_ratio", type=float, default=0.65)
    parser.add_argument("--graph_merge_representative", default="medoid", choices=["medoid", "mean", "weighted_mean"])
    parser.add_argument(
        "--graph_representative_position",
        default="protection",
        choices=["protection", "earliest", "latest", "medoid", "position_medoid", "temporal_medoid"],
    )
    parser.add_argument("--graph_protection_attn_weight", type=float, default=0.70)
    parser.add_argument("--graph_protection_novelty_weight", type=float, default=0.30)
    parser.add_argument("--graph_protection_detail_weight", type=float, default=0.0)
    parser.add_argument("--graph_adaptive_detail_protection", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--graph_adaptive_detail_boost", type=float, default=0.22)
    parser.add_argument("--graph_adaptive_protect_boost", type=float, default=0.10)
    parser.add_argument("--graph_merge_importance_penalty", type=float, default=0.0)
    parser.add_argument("--graph_respect_temporal_threshold", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--graph_final_tokens_per_frame", type=int, default=0)
    parser.add_argument("--graph_final_frame_floor_ratio", type=float, default=0.55)
    parser.add_argument("--graph_skip_spatial_merge_when_capped", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graft_temporal_topk", type=int, default=3)
    parser.add_argument("--graft_temporal_radius", type=int, default=1)
    parser.add_argument("--graft_temporal_skip", type=int, default=1)
    parser.add_argument("--graft_global_topk", type=int, default=3)
    parser.add_argument("--graft_input_is_residual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graft_anchor_ratio", type=float, default=None)
    parser.add_argument("--graft_edge_threshold", type=float, default=0.80)
    parser.add_argument("--graft_component_radius_eps", type=float, default=0.12)
    parser.add_argument("--graft_split_radius_eps", type=float, default=0.20)
    parser.add_argument("--graft_parent_capacity", type=int, default=1)
    parser.add_argument("--graft_mutual_knn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graft_one_token_per_frame", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graft_spatial_penalty", type=float, default=0.10)
    parser.add_argument("--graft_importance_penalty", type=float, default=0.05)
    parser.add_argument("--graft_hub_penalty", type=float, default=0.05)
    parser.add_argument("--graft_adaptive_aggregation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graft_scene_threshold", type=float, default=0.0)
    parser.add_argument("--graft_min_tokens_per_frame", type=int, default=0)
    parser.add_argument("--graft_budget_correction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graft_budget_diversity_weight", type=float, default=0.35)
    parser.add_argument("--graft_score_preset", default="base", choices=["base", "legacy", "event", "event_v1", "event_v2"])
    parser.add_argument("--external_budget_uses_expansion", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fastvid_DySeg_c", type=int, default=8)
    parser.add_argument("--fastvid_DySeg_tau", type=float, default=0.90)
    parser.add_argument("--fastvid_DySeg_ignore", type=float, default=0.95)
    parser.add_argument("--fastvid_STPrune_d", type=float, default=0.40)
    parser.add_argument("--fastvid_DTM_p", type=int, default=4)
    parser.add_argument("--fastvid_DTM_beta", type=float, default=0.60)
    parser.add_argument("--visionzip_dominant_ratio", type=float, default=0.85)
    args = parser.parse_args()

    work_dir = REPO_ROOT
    if args.gpu_ids.strip():
        gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
    else:
        gpu_ids = _query_free_gpus(args.free_ratio, args.min_free_mb)
        gpu_ids = _apply_gpu_cap(gpu_ids, args.gpu_cap)
        if args.max_gpus > 0:
            gpu_ids = gpu_ids[: args.max_gpus]
    if not gpu_ids:
        raise SystemExit("No eligible GPU found.")
    if args.total_limit <= 0:
        raise SystemExit("--total_limit must be positive for parallel sharding.")
    external_method = _selected_external_method(args)
    if not (args.run_flashvid or args.run_graphvid or args.run_graftvid or external_method):
        raise SystemExit(
            "Enable at least one of --run_flashvid/--run_graphvid/--run_graftvid/"
            "--run_fastvid/--run_visionzip/--run_prunevid."
        )

    jobs = _launch_shards(args, gpu_ids, work_dir)
    _wait_jobs(jobs)
    shard_dir = work_dir / "logs" / "efficiency" / "parallel" / args.tag
    _write_summary(args, jobs, shard_dir)


if __name__ == "__main__":
    main()
