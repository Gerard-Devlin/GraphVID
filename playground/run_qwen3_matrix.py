from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = (
    "/gluster/envs/users/wuzhijian/hf_home/hub/"
    "models--Qwen--Qwen3-VL-8B-Instruct/snapshots/"
    "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)

DEFAULT_DATASETS = OrderedDict(
    [
        ("videomme", "assets/videomme.jsonl"),
        ("egoschema_subset", "assets/egoschema_subset.jsonl"),
        ("egoschema_total", "assets/egoschema.jsonl"),
        ("longvideobench", "assets/longvideobench.jsonl"),
        ("mvbench", "assets/mvbench.jsonl"),
    ]
)


def _parse_rates(text: str) -> list[tuple[str, float]]:
    rates: list[tuple[str, float]] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        ratio = value / 100.0 if value > 1.0 else value
        label = f"{ratio * 100:g}"
        rates.append((label, ratio))
    if not rates:
        raise ValueError("no retention rates provided")
    return rates


def _parse_dataset_map(text: str) -> OrderedDict[str, str]:
    if not text.strip():
        return DEFAULT_DATASETS.copy()
    out: OrderedDict[str, str] = OrderedDict()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"dataset spec must be name=path, got: {item}")
        name, path = item.split("=", 1)
        out[name.strip()] = path.strip()
    return out


def _parse_method_list(text: str) -> list[str]:
    methods = [x.strip().lower() for x in str(text).split(",") if x.strip()]
    allowed = {"graphvid", "graftvid", "flashvid", "talon", "cats"}
    unknown = sorted(set(methods) - allowed)
    if unknown:
        raise ValueError(f"unknown methods: {unknown}; allowed={sorted(allowed)}")
    return methods or ["graphvid"]


def _count_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _safe_pct(value: Any) -> float | None:
    if value is None:
        return None
    return float(value) * 100.0


def _stat_mean(phase: dict[str, Any] | None, key: str) -> float | None:
    if not phase:
        return None
    stat = phase.get(key)
    if isinstance(stat, dict):
        value = stat.get("mean")
        return None if value is None else float(value)
    return None


def _phase_name(method: str) -> str:
    if method == "talon":
        return "ours"
    return method


def _format_num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _format_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def _graph_cap_for_rate(args: argparse.Namespace, ratio: float) -> int:
    mode = str(args.graph_final_cap_mode).lower()
    if mode == "none":
        return int(args.graph_final_tokens_per_frame)
    if mode == "custom":
        if not args.graph_final_tokens_per_frame_by_rate:
            return int(args.graph_final_tokens_per_frame)
        mapping: dict[str, int] = {}
        for item in args.graph_final_tokens_per_frame_by_rate.split(","):
            if not item.strip():
                continue
            k, v = item.split(":", 1)
            mapping[f"{float(k):g}"] = int(v)
            if float(k) > 1:
                mapping[f"{float(k) / 100.0 * 100:g}"] = int(v)
        label = f"{ratio * 100:g}"
        return int(mapping.get(label, args.graph_final_tokens_per_frame))
    multiplier = 1.0
    if mode == "expanded":
        multiplier = float(args.expansion)
    elif mode != "strict":
        raise ValueError(f"unknown graph_final_cap_mode={args.graph_final_cap_mode}")
    budget = float(args.raw_visual_tokens) * ratio * multiplier
    return int(math.ceil(budget / max(1, int(args.visual_time_units))))


def _build_command(
    args: argparse.Namespace,
    method: str,
    dataset_name: str,
    dataset_path: Path,
    rate_label: str,
    ratio: float,
    total_limit: int,
) -> tuple[list[str], Path]:
    method_tag = f"{args.tag}_{method}_r{rate_label.replace('.', 'p')}_{dataset_name}"
    summary_path = REPO_ROOT / "logs" / "efficiency" / "parallel" / method_tag / f"{method_tag}_summary.json"
    cmd = [
        sys.executable,
        "-u",
        "playground/run_talon_parallel.py",
        "--model_backend",
        args.model_backend,
        "--model_path",
        args.model_path,
        "--dataset_jsonl",
        str(dataset_path),
        "--hf_home",
        args.hf_home,
        "--total_limit",
        str(total_limit),
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
        "--free_ratio",
        str(args.free_ratio),
        "--min_free_mb",
        str(args.min_free_mb),
        "--max_gpus",
        str(args.max_gpus),
        "--gpu_cap",
        str(args.gpu_cap),
        "--retention_ratio",
        str(ratio),
        "--expansion",
        str(args.expansion),
        "--llm_retention_ratio",
        str(args.llm_retention_ratio),
        "--tag",
        method_tag,
    ]
    if args.gpu_ids:
        cmd.extend(["--gpu_ids", args.gpu_ids])
    if method in ("graphvid", "graftvid", "cats"):
        graph_cap = _graph_cap_for_rate(args, ratio)
        cmd.extend(
            [
                "--run_graphvid" if method == "graphvid" else "--no-run_graphvid",
                "--run_graftvid" if method == "graftvid" else "--no-run_graftvid",
                "--run_cats" if method == "cats" else "--no-run_cats",
                "--no-run_flashvid",
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
                str(graph_cap),
                "--graph_final_frame_floor_ratio",
                str(args.graph_final_frame_floor_ratio),
                "--token_selection_method",
                args.token_selection_method,
                "--graphvid_token_selection_method",
                args.graphvid_token_selection_method,
            ]
        )
        if method == "graftvid":
            cmd.extend(
                [
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
        elif method == "cats":
            cmd.extend(
                [
                    "--cats_adts_beta",
                    str(args.cats_adts_beta),
                    "--cats_margin_threshold",
                    str(args.cats_margin_threshold),
                    "--cats_high_conf_bonus",
                    str(args.cats_high_conf_bonus),
                    "--cats_confidence_attn_weight",
                    str(args.cats_confidence_attn_weight),
                    "--cats_confidence_sim_weight",
                    str(args.cats_confidence_sim_weight),
                    "--cats_anchor_self_weight",
                    str(args.cats_anchor_self_weight),
                    "--cats_frame_budget_min",
                    str(args.cats_frame_budget_min),
                    "--cats_frame_budget_temperature",
                    str(args.cats_frame_budget_temperature),
                ]
            )
            cmd.append("--cats_mutual_nn" if args.cats_mutual_nn else "--no-cats_mutual_nn")
            cmd.append("--cats_adaptive_adts_budget" if args.cats_adaptive_adts_budget else "--no-cats_adaptive_adts_budget")
        if args.graph_skip_spatial_merge_when_capped:
            cmd.append("--graph_skip_spatial_merge_when_capped")
        else:
            cmd.append("--no-graph_skip_spatial_merge_when_capped")
        if args.graph_respect_temporal_threshold:
            cmd.append("--graph_respect_temporal_threshold")
        else:
            cmd.append("--no-graph_respect_temporal_threshold")
        if args.graph_adaptive_detail_protection:
            cmd.append("--graph_adaptive_detail_protection")
        else:
            cmd.append("--no-graph_adaptive_detail_protection")
    elif method == "flashvid":
        cmd.extend(["--run_flashvid", "--no-run_ours"])
    else:
        cmd.extend(
            [
                "--no-run_flashvid",
                "--talon_target_tokens_per_frame",
                str(args.talon_target_tokens_per_frame),
                "--talon_question_recall_ratio",
                str(args.talon_question_recall_ratio),
                "--talon_question_recall_qweight",
                str(args.talon_question_recall_qweight),
                "--talon_question_pooling",
                args.talon_question_pooling,
                "--talon_question_pooling_topk",
                str(args.talon_question_pooling_topk),
            ]
        )
    cmd.extend(args.extra_args)
    return cmd, summary_path


def _load_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_score(summary: dict[str, Any] | None, method: str, dataset_name: str) -> dict[str, float | None]:
    phase_key = _phase_name(method)
    phase = summary.get(phase_key) if summary else None
    result: dict[str, float | None] = {
        "acc": _safe_pct(phase.get("accuracy")) if phase else None,
        "tokens": _stat_mean(phase, "compressed_visual_tokens"),
    }
    if dataset_name == "videomme" and summary:
        by_duration = summary.get("duration_breakdown", {})
        for duration in ("short", "medium", "long"):
            bucket = by_duration.get(duration, {})
            sub_phase = bucket.get(phase_key)
            result[duration] = _safe_pct(sub_phase.get("accuracy")) if sub_phase else None
    return result


def _row_average(row: dict[str, float | None]) -> float | None:
    keys = [
        "videomme_overall",
        "egoschema_subset",
        "egoschema_total",
        "longvideobench",
        "mvbench",
    ]
    vals = [row.get(k) for k in keys if row.get(k) is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _write_tables(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "qwen3_matrix_table.csv"
    md_path = out_dir / "qwen3_matrix_table.md"
    json_path = out_dir / "qwen3_matrix_summary.json"
    fieldnames = [
        "method",
        "retention_ratio",
        "videomme_short",
        "videomme_medium",
        "videomme_long",
        "videomme_overall",
        "egoschema_subset",
        "egoschema_total",
        "longvideobench",
        "mvbench",
        "avg_score",
        "rel_acc",
        "mean_tokens",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    lines = [
        "| Method | Retention Ratio R | VideoMME Short | VideoMME Medium | VideoMME Long | VideoMME Overall | EgoSchema Subset | EgoSchema Total | LongVideoBench | MVBench | Avg. Score | Rel. Acc (%) | Tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {r} | {vs} | {vm} | {vl} | {vo} | {es} | {et} | {lvb} | {mvb} | {avg} | {rel} | {tok} |".format(
                method=row.get("method", "-"),
                r=row.get("retention_ratio", "-"),
                vs=_format_pct(row.get("videomme_short")),
                vm=_format_pct(row.get("videomme_medium")),
                vl=_format_pct(row.get("videomme_long")),
                vo=_format_pct(row.get("videomme_overall")),
                es=_format_pct(row.get("egoschema_subset")),
                et=_format_pct(row.get("egoschema_total")),
                lvb=_format_pct(row.get("longvideobench")),
                mvb=_format_pct(row.get("mvbench")),
                avg=_format_pct(row.get("avg_score")),
                rel=_format_pct(row.get("rel_acc")),
                tok=_format_num(row.get("mean_tokens")),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[table] csv={csv_path}")
    print(f"[table] md={md_path}")
    print(f"[table] json={json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Qwen3-VL-8B compression sweeps and emit paper-style benchmark tables."
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model_backend", default="qwen3_vl")
    parser.add_argument("--hf_home", default=os.environ.get("HF_HOME", "/gluster/envs/users/wuzhijian/hf_home"))
    parser.add_argument("--datasets", default="", help="Comma list: name=path. Defaults to standard asset names.")
    parser.add_argument("--methods", default="graphvid", help="Comma list: graphvid,graftvid,cats,flashvid,talon.")
    parser.add_argument("--rates", default="10,15,20,25", help="Retention ratios in percent or decimals.")
    parser.add_argument("--tag", default="qwen3_matrix")
    parser.add_argument("--output_dir", default="logs/efficiency/matrix")
    parser.add_argument("--limit", type=int, default=0, help="0 means all samples in each JSONL.")
    parser.add_argument("--strict_datasets", action="store_true", help="Fail if a default dataset JSONL is missing.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip runs whose combined summary already exists.")
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--min_pixels", type=int, default=64 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--num_warmup", type=int, default=1)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--free_ratio", type=float, default=0.75)
    parser.add_argument("--min_free_mb", type=int, default=18000)
    parser.add_argument("--max_gpus", type=int, default=4)
    parser.add_argument("--gpu_cap", type=int, default=4)
    parser.add_argument("--gpu_ids", default="")
    parser.add_argument("--retention_expansion", dest="expansion", type=float, default=1.25)
    parser.add_argument("--llm_retention_ratio", type=float, default=1.0)
    parser.add_argument("--raw_visual_tokens", type=int, default=2880)
    parser.add_argument("--visual_time_units", type=int, default=16)
    parser.add_argument(
        "--graph_final_cap_mode",
        default="expanded",
        choices=["expanded", "strict", "none", "custom"],
        help="expanded mimics FlashVID R*expansion; strict uses raw*R; none leaves graph final cap at the provided value.",
    )
    parser.add_argument("--graph_final_tokens_per_frame", type=int, default=0)
    parser.add_argument("--graph_final_tokens_per_frame_by_rate", default="")
    parser.add_argument("--graph_temporal_topk", type=int, default=2)
    parser.add_argument("--graph_temporal_radius", type=int, default=1)
    parser.add_argument("--graph_temporal_skip", type=int, default=1)
    parser.add_argument("--graph_merge_protect_ratio", type=float, default=0.15)
    parser.add_argument("--graph_merge_target_ratio", type=float, default=1.00)
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
    parser.add_argument("--graph_final_frame_floor_ratio", type=float, default=0.55)
    parser.add_argument("--graph_skip_spatial_merge_when_capped", action=argparse.BooleanOptionalAction, default=False)
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
    parser.add_argument("--cats_adts_beta", type=float, default=0.05)
    parser.add_argument("--cats_margin_threshold", type=float, default=0.03)
    parser.add_argument("--cats_high_conf_bonus", type=float, default=0.05)
    parser.add_argument("--cats_mutual_nn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cats_confidence_attn_weight", type=float, default=0.75)
    parser.add_argument("--cats_confidence_sim_weight", type=float, default=1.0)
    parser.add_argument("--cats_anchor_self_weight", type=float, default=1.0)
    parser.add_argument("--cats_adaptive_adts_budget", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cats_frame_budget_min", type=int, default=1)
    parser.add_argument("--cats_frame_budget_temperature", type=float, default=0.7)
    parser.add_argument("--token_selection_method", default="attn_div_stable")
    parser.add_argument("--graphvid_token_selection_method", default="attn_div_stable")
    parser.add_argument("--talon_target_tokens_per_frame", type=int, default=22)
    parser.add_argument("--talon_question_recall_ratio", type=float, default=0.08)
    parser.add_argument("--talon_question_recall_qweight", type=float, default=0.65)
    parser.add_argument("--talon_question_pooling", default="topk")
    parser.add_argument("--talon_question_pooling_topk", type=int, default=4)
    args, extra_args = parser.parse_known_args()
    args.extra_args = extra_args

    datasets = _parse_dataset_map(args.datasets)
    rates = _parse_rates(args.rates)
    methods = _parse_method_list(args.methods)
    out_dir = REPO_ROOT / args.output_dir / args.tag
    rows: list[dict[str, Any]] = []
    summaries: dict[tuple[str, str, str], dict[str, Any] | None] = {}

    for method in methods:
        for rate_label, ratio in rates:
            row: dict[str, Any] = {
                "method": method,
                "retention_ratio": f"{rate_label}%",
            }
            token_values: list[float] = []
            for dataset_name, dataset_spec in datasets.items():
                dataset_path = (REPO_ROOT / dataset_spec).resolve() if not Path(dataset_spec).is_absolute() else Path(dataset_spec)
                if not dataset_path.exists():
                    msg = f"[skip] missing dataset {dataset_name}: {dataset_path}"
                    if args.strict_datasets:
                        raise FileNotFoundError(msg)
                    print(msg)
                    continue
                total_limit = int(args.limit) if int(args.limit) > 0 else _count_jsonl(dataset_path)
                cmd, summary_path = _build_command(
                    args=args,
                    method=method,
                    dataset_name=dataset_name,
                    dataset_path=dataset_path,
                    rate_label=rate_label,
                    ratio=ratio,
                    total_limit=total_limit,
                )
                print("[run]", " ".join(shlex.quote(x) for x in cmd))
                if not args.dry_run and not (args.resume and summary_path.exists()):
                    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
                summary = _load_summary(summary_path)
                summaries[(method, rate_label, dataset_name)] = summary
                score = _extract_score(summary, method, dataset_name)
                if score.get("tokens") is not None:
                    token_values.append(float(score["tokens"]))
                if dataset_name == "videomme":
                    row["videomme_short"] = score.get("short")
                    row["videomme_medium"] = score.get("medium")
                    row["videomme_long"] = score.get("long")
                    row["videomme_overall"] = score.get("acc")
                elif dataset_name == "egoschema_subset":
                    row["egoschema_subset"] = score.get("acc")
                elif dataset_name == "egoschema_total":
                    row["egoschema_total"] = score.get("acc")
                elif dataset_name == "longvideobench":
                    row["longvideobench"] = score.get("acc")
                elif dataset_name == "mvbench":
                    row["mvbench"] = score.get("acc")
            row["avg_score"] = _row_average(row)
            row["rel_acc"] = None
            row["mean_tokens"] = float(sum(token_values) / len(token_values)) if token_values else None
            rows.append(row)

    _write_tables(out_dir, rows)


if __name__ == "__main__":
    main()
