from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _str_bool(value: bool) -> str:
    return "True" if value else "False"


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


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


def _append_common_talon_args(cmd: list[str], args: argparse.Namespace) -> None:
    cmd.extend(
        [
            "--llm_retention_ratio",
            str(args.llm_retention_ratio),
            "--retention_ratio",
            str(args.retention_ratio),
            "--expansion",
            str(args.expansion),
            "--compression_variant",
            "talon",
            "--question_aware_reweighting",
            "True",
            "--question_reweight_beta",
            "0.25",
            "--adaptive_token_budget",
            "False",
            "--talon_adaptive_target_enabled",
            _str_bool(args.talon_adaptive_target_enabled),
            "--talon_target_mean_cap",
            str(args.talon_target_mean_cap),
            "--talon_target_tokens_per_frame",
            str(args.talon_target_tokens_per_frame),
            "--talon_short_target_tokens_per_frame",
            str(args.talon_short_target_tokens_per_frame),
            "--talon_medium_target_tokens_per_frame",
            str(args.talon_medium_target_tokens_per_frame),
            "--talon_long_target_tokens_per_frame",
            str(args.talon_long_target_tokens_per_frame),
            "--talon_adaptive_target_low",
            str(args.talon_adaptive_target_low),
            "--talon_adaptive_target_mid",
            str(args.talon_adaptive_target_mid),
            "--talon_adaptive_target_high",
            str(args.talon_adaptive_target_high),
            "--talon_complexity_floor",
            str(args.talon_complexity_floor),
            "--talon_complexity_ceil",
            str(args.talon_complexity_ceil),
            "--talon_adaptive_gamma",
            str(args.talon_adaptive_gamma),
            "--talon_question_recall_ratio",
            str(args.talon_question_recall_ratio),
            "--talon_question_recall_qweight",
            str(args.talon_question_recall_qweight),
            "--talon_persistence_recall_ratio",
            str(args.talon_persistence_recall_ratio),
            "--talon_persistence_recall_qweight",
            str(args.talon_persistence_recall_qweight),
            "--talon_persistence_recall_pweight",
            str(args.talon_persistence_recall_pweight),
            "--talon_persistence_apply_to_short",
            _str_bool(args.talon_persistence_apply_to_short),
            "--talon_persistence_apply_to_medium",
            _str_bool(args.talon_persistence_apply_to_medium),
            "--talon_persistence_apply_to_long",
            _str_bool(args.talon_persistence_apply_to_long),
            "--talon_object_evidence_ratio",
            str(args.talon_object_evidence_ratio),
            "--talon_object_evidence_qweight",
            str(args.talon_object_evidence_qweight),
            "--talon_object_evidence_sweight",
            str(args.talon_object_evidence_sweight),
            "--talon_object_evidence_pweight",
            str(args.talon_object_evidence_pweight),
            "--talon_object_evidence_apply_to_short",
            _str_bool(args.talon_object_evidence_apply_to_short),
            "--talon_object_evidence_apply_to_medium",
            _str_bool(args.talon_object_evidence_apply_to_medium),
            "--talon_object_evidence_apply_to_long",
            _str_bool(args.talon_object_evidence_apply_to_long),
            "--talon_question_pooling",
            args.talon_question_pooling,
            "--talon_question_pooling_topk",
            str(args.talon_question_pooling_topk),
            "--talon_question_contrast_weight",
            str(args.talon_question_contrast_weight),
            "--talon_question_contrast_apply_to_short",
            _str_bool(args.talon_question_contrast_apply_to_short),
            "--talon_monotonic_base_tokens_per_frame",
            str(args.talon_monotonic_base_tokens_per_frame),
            "--talon_anchor_diversity_weight",
            str(args.talon_anchor_diversity_weight),
            "--talon_spatial_anchor_coverage",
            _str_bool(args.talon_spatial_anchor_coverage),
            "--talon_spatial_anchor_ratio",
            str(args.talon_spatial_anchor_ratio),
            "--talon_spatial_anchor_rows",
            str(args.talon_spatial_anchor_rows),
            "--talon_spatial_anchor_cols",
            str(args.talon_spatial_anchor_cols),
            "--talon_spatial_anchor_score",
            args.talon_spatial_anchor_score,
            "--talon_spatial_anchor_apply_to_short",
            _str_bool(args.talon_spatial_anchor_apply_to_short),
            "--talon_frame_coverage_floor_ratio",
            str(args.talon_frame_coverage_floor_ratio),
            "--talon_frame_importance_pooling",
            args.talon_frame_importance_pooling,
            "--talon_frame_importance_topk",
            str(args.talon_frame_importance_topk),
            "--talon_medium_frame_coverage_floor_ratio",
            str(args.talon_medium_frame_coverage_floor_ratio),
            "--talon_long_frame_coverage_floor_ratio",
            str(args.talon_long_frame_coverage_floor_ratio),
            "--talon_frame_local_budget_ratio",
            str(args.talon_frame_local_budget_ratio),
            "--talon_anchor_safety_ratio",
            "0.72",
            "--talon_budget_mode",
            args.talon_budget_mode,
            "--talon_global_topk_ratio",
            "0.70",
            "--talon_event_budget_ratio",
            "0.30",
            "--talon_duration_aware",
            _str_bool(args.talon_duration_aware),
            "--talon_medium_anchor_safety_ratio",
            str(args.talon_medium_anchor_safety_ratio),
            "--talon_medium_event_budget_ratio",
            str(args.talon_medium_event_budget_ratio),
            "--talon_medium_global_topk_ratio",
            str(args.talon_medium_global_topk_ratio),
            "--talon_long_anchor_safety_ratio",
            str(args.talon_long_anchor_safety_ratio),
            "--talon_long_event_budget_ratio",
            str(args.talon_long_event_budget_ratio),
            "--talon_long_global_topk_ratio",
            str(args.talon_long_global_topk_ratio),
            "--talon_task_aware_event",
            _str_bool(args.talon_task_aware_event),
            "--talon_task_event_attention_weight",
            str(args.talon_task_event_attention_weight),
            "--talon_task_event_qweight",
            str(args.talon_task_event_qweight),
            "--talon_visual_task_balance",
            _str_bool(args.talon_visual_task_balance),
            "--talon_visual_task_anchor_ratio",
            str(args.talon_visual_task_anchor_ratio),
            "--talon_visual_task_event_ratio",
            str(args.talon_visual_task_event_ratio),
            "--talon_visual_task_recall_ratio",
            str(args.talon_visual_task_recall_ratio),
            "--talon_knowledge_visual_anchor_ratio",
            str(args.talon_knowledge_visual_anchor_ratio),
            "--talon_knowledge_visual_event_ratio",
            str(args.talon_knowledge_visual_event_ratio),
            "--talon_knowledge_visual_recall_ratio",
            str(args.talon_knowledge_visual_recall_ratio),
            "--talon_adaptive_router",
            _str_bool(args.talon_adaptive_router),
            "--talon_router_apply_to_short",
            _str_bool(args.talon_router_apply_to_short),
            "--talon_router_visual_anchor_ratio",
            str(args.talon_router_visual_anchor_ratio),
            "--talon_router_visual_event_ratio",
            str(args.talon_router_visual_event_ratio),
            "--talon_router_visual_recall_ratio",
            str(args.talon_router_visual_recall_ratio),
            "--talon_router_temporal_anchor_ratio",
            str(args.talon_router_temporal_anchor_ratio),
            "--talon_router_temporal_event_ratio",
            str(args.talon_router_temporal_event_ratio),
            "--talon_router_temporal_recall_ratio",
            str(args.talon_router_temporal_recall_ratio),
            "--talon_router_balanced_anchor_ratio",
            str(args.talon_router_balanced_anchor_ratio),
            "--talon_router_balanced_event_ratio",
            str(args.talon_router_balanced_event_ratio),
            "--talon_router_balanced_recall_ratio",
            str(args.talon_router_balanced_recall_ratio),
            "--talon_router_visual_concentration_threshold",
            str(args.talon_router_visual_concentration_threshold),
            "--talon_router_low_residual_threshold",
            str(args.talon_router_low_residual_threshold),
            "--talon_router_temporal_entropy_threshold",
            str(args.talon_router_temporal_entropy_threshold),
            "--talon_router_temporal_residual_threshold",
            str(args.talon_router_temporal_residual_threshold),
            "--talon_temporal_chunk_aware",
            _str_bool(args.talon_temporal_chunk_aware),
            "--talon_temporal_num_chunks",
            str(args.talon_temporal_num_chunks),
            "--talon_temporal_chunk_min_ratio",
            str(args.talon_temporal_chunk_min_ratio),
            "--talon_temporal_chunk_score",
            args.talon_temporal_chunk_score,
            "--talon_track_aware",
            _str_bool(args.talon_track_aware),
            "--talon_track_budget_ratio",
            str(args.talon_track_budget_ratio),
            "--talon_track_tokens_per_slot",
            str(args.talon_track_tokens_per_slot),
            "--talon_track_score",
            args.talon_track_score,
            "--talon_absorb_dropped_tokens",
            _str_bool(args.talon_absorb_dropped_tokens),
            "--talon_absorb_ratio",
            str(args.talon_absorb_ratio),
            "--talon_absorb_alpha",
            str(args.talon_absorb_alpha),
            "--talon_absorb_score",
            args.talon_absorb_score,
            "--talon_summary_replacement",
            _str_bool(args.talon_summary_replacement),
            "--talon_summary_raw_swap",
            _str_bool(args.talon_summary_raw_swap),
            "--talon_summary_ratio",
            str(args.talon_summary_ratio),
            "--talon_summary_num_chunks",
            str(args.talon_summary_num_chunks),
            "--talon_summary_pool_topk",
            str(args.talon_summary_pool_topk),
            "--talon_summary_alpha",
            str(args.talon_summary_alpha),
            "--talon_summary_score",
            args.talon_summary_score,
            "--talon_output_mode",
            args.talon_output_mode,
            "--talon_reconstruction_blend",
            str(args.talon_reconstruction_blend),
            "--talon_anchor_score_weight",
            str(args.talon_anchor_score_weight),
            "--talon_rank_ratio",
            str(args.talon_rank_ratio),
            "--talon_rank_min",
            str(args.talon_rank_min),
            "--talon_rank_max",
            str(args.talon_rank_max),
            "--talon_background_max_ratio",
            str(args.talon_background_max_ratio),
            "--talon_innovation_attention_weight",
            str(args.talon_innovation_attention_weight),
            "--talon_lite_enabled",
            _str_bool(args.talon_lite_enabled),
            "--talon_echo_residual_weight",
            str(args.talon_echo_residual_weight),
            "--talon_echo_topk_neighbors",
            str(args.talon_echo_topk_neighbors),
            "--talon_echo_temperature",
            str(args.talon_echo_temperature),
            "--talon_echo_score_mode",
            args.talon_echo_score_mode,
            "--talon_final_fused_weight",
            "0.70",
            "--talon_final_residual_weight",
            "0.20",
            "--talon_final_frame_weight",
            "0.10",
            "--talon_use_question_innovation",
            "True",
            "--talon_innovation_qweight",
            "0.20",
            "--talon_deepstack_mode",
            "keep",
        ]
    )


def _append_graphvid_args(cmd: list[str], args: argparse.Namespace) -> None:
    cmd.extend(
        [
            "--llm_retention_ratio",
            str(args.llm_retention_ratio),
            "--retention_ratio",
            str(args.retention_ratio),
            "--expansion",
            str(args.expansion),
            "--temporal_merge_mode",
            "graph",
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
            "--graph_skip_spatial_merge_when_capped",
            _str_bool(args.graph_skip_spatial_merge_when_capped),
            "--graphvid_token_selection_method",
            args.graphvid_token_selection_method,
        ]
    )


def _launch_shards(args: argparse.Namespace, gpu_ids: list[int], work_dir: Path) -> list[dict[str, object]]:
    ranges = _split_ranges(args.start_index, args.total_limit, len(gpu_ids))
    shard_dir = work_dir / "logs" / "efficiency" / "parallel" / args.tag
    shard_dir.mkdir(parents=True, exist_ok=True)
    jobs = []

    for shard_idx, ((start, limit), gpu_id) in enumerate(zip(ranges, gpu_ids)):
        flashvid_out = shard_dir / f"flashvid_shard{shard_idx:02d}.jsonl"
        ours_out = shard_dir / f"ours_shard{shard_idx:02d}.jsonl"
        graphvid_out = shard_dir / f"graphvid_shard{shard_idx:02d}.jsonl"
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
            "--run_baseline",
            "False",
            "--run_flashvid",
            _str_bool(args.run_flashvid),
            "--run_ours",
            _str_bool(args.run_ours and not args.run_graphvid),
            "--run_graphvid",
            _str_bool(args.run_graphvid),
            "--flashvid_output",
            str(flashvid_out),
            "--ours_output",
            str(ours_out),
            "--graphvid_output",
            str(graphvid_out),
            "--summary_output_json",
            str(summary_out),
        ]
        if args.run_graphvid:
            _append_graphvid_args(cmd, args)
        else:
            _append_common_talon_args(cmd, args)

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
                "ours_out": ours_out,
                "graphvid_out": graphvid_out,
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
                    rows.append(json.loads(line))
                    w.write(line if line.endswith("\n") else line + "\n")
    return rows


def _write_summary(args: argparse.Namespace, jobs: list[dict[str, object]], shard_dir: Path) -> None:
    sys.path.insert(0, str(REPO_ROOT))
    from playground.bench_all_metrics import (
        _add_duration_breakdown,
        _print_summary,
        _summarize_pairwise_comparison,
        _summarize_phase,
    )

    combined_flashvid = shard_dir / f"{args.tag}_flashvid.jsonl"
    combined_ours = shard_dir / f"{args.tag}_ours.jsonl"
    combined_graphvid = shard_dir / f"{args.tag}_graphvid.jsonl"
    combined_summary = shard_dir / f"{args.tag}_summary.json"

    flashvid_records = []
    if args.run_flashvid:
        flashvid_records = _combine_jsonl([Path(j["flashvid_out"]) for j in jobs], combined_flashvid)
    ours_records = []
    graphvid_records = []
    if args.run_graphvid:
        graphvid_records = _combine_jsonl([Path(j["graphvid_out"]) for j in jobs], combined_graphvid)
    elif args.run_ours:
        ours_records = _combine_jsonl([Path(j["ours_out"]) for j in jobs], combined_ours)

    summary: dict[str, object] = {"comparison": {}}
    if args.run_flashvid:
        summary["flashvid"] = _summarize_phase(flashvid_records)
    if args.run_graphvid:
        summary["graphvid"] = _summarize_phase(graphvid_records)
    elif args.run_ours:
        summary["ours"] = _summarize_phase(ours_records)
    if args.run_flashvid and args.run_graphvid:
        summary["comparison"]["flashvid_vs_graphvid"] = _summarize_pairwise_comparison(
            flashvid_records,
            graphvid_records,
            anchor_name="flashvid",
            target_name="graphvid",
        )
    elif args.run_flashvid and args.run_ours:
        summary["comparison"]["flashvid_vs_ours"] = _summarize_pairwise_comparison(
            flashvid_records,
            ours_records,
            anchor_name="flashvid",
            target_name="ours",
        )
    _add_duration_breakdown(
        summary,
        flashvid_records=flashvid_records if args.run_flashvid else None,
        ours_records=None if args.run_graphvid else ours_records,
        graphvid_records=graphvid_records if args.run_graphvid else None,
    )
    with combined_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    if args.run_graphvid:
        print(f"[combined] graphvid={combined_graphvid}")
    elif args.run_ours:
        print(f"[combined] ours={combined_ours}")
    if args.run_flashvid:
        print(f"[combined] flashvid={combined_flashvid}")
    print(f"[combined] summary={combined_summary}")
    _print_summary(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel TALON recall08 benchmark launcher.")
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
    parser.add_argument("--graphvid_token_selection_method", default="attn_div_v2")
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run_flashvid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run_ours", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run_graphvid", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--retention_ratio", type=float, default=0.10)
    parser.add_argument("--expansion", type=float, default=1.25)
    parser.add_argument("--llm_retention_ratio", type=float, default=1.0)
    parser.add_argument("--free_ratio", type=float, default=0.75)
    parser.add_argument("--min_free_mb", type=int, default=18000)
    parser.add_argument("--max_gpus", type=int, default=0, help="0 means use all eligible GPUs.")
    parser.add_argument("--gpu_ids", default="", help="Comma-separated GPU ids. Overrides auto selection.")
    parser.add_argument("--tag", default="talon_recall08_t20_parallel")
    parser.add_argument("--talon_target_tokens_per_frame", type=int, default=20)
    parser.add_argument("--graph_temporal_topk", type=int, default=3)
    parser.add_argument("--graph_temporal_radius", type=int, default=1)
    parser.add_argument("--graph_temporal_skip", type=int, default=1)
    parser.add_argument("--graph_merge_protect_ratio", type=float, default=0.15)
    parser.add_argument("--graph_merge_target_ratio", type=float, default=0.65)
    parser.add_argument("--graph_merge_representative", default="medoid", choices=["medoid", "mean"])
    parser.add_argument("--graph_final_tokens_per_frame", type=int, default=0)
    parser.add_argument("--graph_final_frame_floor_ratio", type=float, default=0.55)
    parser.add_argument("--graph_skip_spatial_merge_when_capped", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--talon_short_target_tokens_per_frame", type=int, default=0)
    parser.add_argument("--talon_medium_target_tokens_per_frame", type=int, default=0)
    parser.add_argument("--talon_long_target_tokens_per_frame", type=int, default=0)
    parser.add_argument("--talon_adaptive_target_enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--talon_adaptive_target_low", type=int, default=0)
    parser.add_argument("--talon_adaptive_target_mid", type=int, default=0)
    parser.add_argument("--talon_adaptive_target_high", type=int, default=0)
    parser.add_argument("--talon_complexity_floor", type=float, default=0.20)
    parser.add_argument("--talon_complexity_ceil", type=float, default=0.40)
    parser.add_argument("--talon_adaptive_gamma", type=float, default=1.0)
    parser.add_argument("--talon_target_mean_cap", type=float, default=0.0)
    parser.add_argument("--talon_question_recall_ratio", type=float, default=0.08)
    parser.add_argument("--talon_question_recall_qweight", type=float, default=0.65)
    parser.add_argument("--talon_persistence_recall_ratio", type=float, default=0.0)
    parser.add_argument("--talon_persistence_recall_qweight", type=float, default=0.50)
    parser.add_argument("--talon_persistence_recall_pweight", type=float, default=0.35)
    parser.add_argument("--talon_persistence_apply_to_short", type=_parse_bool, default=False)
    parser.add_argument("--talon_persistence_apply_to_medium", type=_parse_bool, default=True)
    parser.add_argument("--talon_persistence_apply_to_long", type=_parse_bool, default=False)
    parser.add_argument("--talon_object_evidence_ratio", type=float, default=0.0)
    parser.add_argument("--talon_object_evidence_qweight", type=float, default=0.35)
    parser.add_argument("--talon_object_evidence_sweight", type=float, default=0.45)
    parser.add_argument("--talon_object_evidence_pweight", type=float, default=0.10)
    parser.add_argument("--talon_object_evidence_apply_to_short", type=_parse_bool, default=False)
    parser.add_argument("--talon_object_evidence_apply_to_medium", type=_parse_bool, default=True)
    parser.add_argument("--talon_object_evidence_apply_to_long", type=_parse_bool, default=False)
    parser.add_argument("--talon_question_pooling", default="mean")
    parser.add_argument("--talon_question_pooling_topk", type=int, default=4)
    parser.add_argument("--talon_question_contrast_weight", type=float, default=0.0)
    parser.add_argument("--talon_question_contrast_apply_to_short", type=_parse_bool, default=False)
    parser.add_argument("--talon_monotonic_base_tokens_per_frame", type=int, default=20)
    parser.add_argument("--talon_frame_local_budget_ratio", type=float, default=1.0)
    parser.add_argument("--talon_anchor_diversity_weight", type=float, default=0.0)
    parser.add_argument("--talon_spatial_anchor_coverage", type=_parse_bool, default=False)
    parser.add_argument("--talon_spatial_anchor_ratio", type=float, default=0.35)
    parser.add_argument("--talon_spatial_anchor_rows", type=int, default=3)
    parser.add_argument("--talon_spatial_anchor_cols", type=int, default=3)
    parser.add_argument("--talon_spatial_anchor_score", default="fused", choices=["combined", "fused", "question", "event"])
    parser.add_argument("--talon_spatial_anchor_apply_to_short", type=_parse_bool, default=False)
    parser.add_argument("--talon_frame_coverage_floor_ratio", type=float, default=0.65)
    parser.add_argument("--talon_frame_importance_pooling", default="mean", choices=["mean", "topk", "max", "evidence"])
    parser.add_argument("--talon_frame_importance_topk", type=int, default=6)
    parser.add_argument("--talon_medium_frame_coverage_floor_ratio", type=float, default=-1.0)
    parser.add_argument("--talon_long_frame_coverage_floor_ratio", type=float, default=-1.0)
    parser.add_argument("--talon_budget_mode", default="attention", choices=["attention", "uniform"])
    parser.add_argument("--talon_lite_enabled", type=_parse_bool, default=False)
    parser.add_argument("--talon_echo_residual_weight", type=float, default=0.0)
    parser.add_argument("--talon_echo_topk_neighbors", type=int, default=4)
    parser.add_argument("--talon_echo_temperature", type=float, default=0.07)
    parser.add_argument("--talon_echo_score_mode", default="mse", choices=["mse", "cosine"])
    parser.add_argument("--talon_output_mode", default="manifold", choices=["manifold", "full", "lowrank", "coefficient"])
    parser.add_argument("--talon_reconstruction_blend", type=float, default=0.0)
    parser.add_argument("--talon_anchor_score_weight", type=float, default=0.35)
    parser.add_argument("--talon_rank_ratio", type=float, default=0.40)
    parser.add_argument("--talon_rank_min", type=int, default=1)
    parser.add_argument("--talon_rank_max", type=int, default=8)
    parser.add_argument("--talon_background_max_ratio", type=float, default=0.35)
    parser.add_argument("--talon_innovation_attention_weight", type=float, default=0.65)
    parser.add_argument("--talon_duration_aware", type=_parse_bool, default=False)
    parser.add_argument("--talon_medium_anchor_safety_ratio", type=float, default=0.72)
    parser.add_argument("--talon_medium_event_budget_ratio", type=float, default=0.30)
    parser.add_argument("--talon_medium_global_topk_ratio", type=float, default=0.70)
    parser.add_argument("--talon_long_anchor_safety_ratio", type=float, default=0.80)
    parser.add_argument("--talon_long_event_budget_ratio", type=float, default=0.14)
    parser.add_argument("--talon_long_global_topk_ratio", type=float, default=0.85)
    parser.add_argument("--talon_task_aware_event", type=_parse_bool, default=False)
    parser.add_argument("--talon_task_event_attention_weight", type=float, default=0.82)
    parser.add_argument("--talon_task_event_qweight", type=float, default=0.30)
    parser.add_argument("--talon_visual_task_balance", type=_parse_bool, default=False)
    parser.add_argument("--talon_visual_task_anchor_ratio", type=float, default=0.84)
    parser.add_argument("--talon_visual_task_event_ratio", type=float, default=0.12)
    parser.add_argument("--talon_visual_task_recall_ratio", type=float, default=0.02)
    parser.add_argument("--talon_knowledge_visual_anchor_ratio", type=float, default=0.78)
    parser.add_argument("--talon_knowledge_visual_event_ratio", type=float, default=0.18)
    parser.add_argument("--talon_knowledge_visual_recall_ratio", type=float, default=0.06)
    parser.add_argument("--talon_adaptive_router", type=_parse_bool, default=False)
    parser.add_argument("--talon_router_apply_to_short", type=_parse_bool, default=False)
    parser.add_argument("--talon_router_visual_anchor_ratio", type=float, default=0.76)
    parser.add_argument("--talon_router_visual_event_ratio", type=float, default=0.24)
    parser.add_argument("--talon_router_visual_recall_ratio", type=float, default=0.06)
    parser.add_argument("--talon_router_temporal_anchor_ratio", type=float, default=0.66)
    parser.add_argument("--talon_router_temporal_event_ratio", type=float, default=0.34)
    parser.add_argument("--talon_router_temporal_recall_ratio", type=float, default=0.08)
    parser.add_argument("--talon_router_balanced_anchor_ratio", type=float, default=0.72)
    parser.add_argument("--talon_router_balanced_event_ratio", type=float, default=0.30)
    parser.add_argument("--talon_router_balanced_recall_ratio", type=float, default=0.08)
    parser.add_argument("--talon_router_visual_concentration_threshold", type=float, default=0.28)
    parser.add_argument("--talon_router_low_residual_threshold", type=float, default=0.30)
    parser.add_argument("--talon_router_temporal_entropy_threshold", type=float, default=0.95)
    parser.add_argument("--talon_router_temporal_residual_threshold", type=float, default=0.36)
    parser.add_argument("--talon_temporal_chunk_aware", type=_parse_bool, default=False)
    parser.add_argument("--talon_temporal_num_chunks", type=int, default=4)
    parser.add_argument("--talon_temporal_chunk_min_ratio", type=float, default=0.18)
    parser.add_argument("--talon_temporal_chunk_score", default="combined", choices=["combined", "fused", "question", "event"])
    parser.add_argument("--talon_track_aware", type=_parse_bool, default=False)
    parser.add_argument("--talon_track_budget_ratio", type=float, default=0.12)
    parser.add_argument("--talon_track_tokens_per_slot", type=int, default=1)
    parser.add_argument("--talon_track_score", default="combined", choices=["combined", "fused", "question", "event"])
    parser.add_argument("--talon_absorb_dropped_tokens", type=_parse_bool, default=False)
    parser.add_argument("--talon_absorb_ratio", type=float, default=0.35)
    parser.add_argument("--talon_absorb_alpha", type=float, default=0.25)
    parser.add_argument("--talon_absorb_score", default="combined", choices=["combined", "fused", "question", "event"])
    parser.add_argument("--talon_summary_replacement", type=_parse_bool, default=False)
    parser.add_argument("--talon_summary_raw_swap", type=_parse_bool, default=False)
    parser.add_argument("--talon_summary_ratio", type=float, default=0.08)
    parser.add_argument("--talon_summary_num_chunks", type=int, default=8)
    parser.add_argument("--talon_summary_pool_topk", type=int, default=12)
    parser.add_argument("--talon_summary_alpha", type=float, default=0.55)
    parser.add_argument("--talon_summary_score", default="combined", choices=["combined", "fused", "question", "event"])
    args = parser.parse_args()

    if args.gpu_ids.strip():
        gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
    else:
        gpu_ids = _query_free_gpus(args.free_ratio, args.min_free_mb)
    if args.max_gpus > 0:
        gpu_ids = gpu_ids[: args.max_gpus]
    if not gpu_ids:
        raise SystemExit("No eligible GPU found. Lower --free_ratio/--min_free_mb or pass --gpu_ids.")
    if args.total_limit <= 0:
        raise SystemExit("--total_limit must be positive.")

    jobs = _launch_shards(args, gpu_ids, REPO_ROOT)
    _wait_jobs(jobs)
    shard_dir = REPO_ROOT / "logs" / "efficiency" / "parallel" / args.tag
    _write_summary(args, jobs, shard_dir)


if __name__ == "__main__":
    main()
