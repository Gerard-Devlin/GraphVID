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
    allowed = {"graphvid", "flashvid", "talon", "fastvid", "visionzip", "fastgraphvid", "curvevid"}
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


def _run_method_name(method: str) -> str:
    if method in ("fastvid", "visionzip", "fastgraphvid", "curvevid"):
        return f"{method}_qwen3_adapter"
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
    run_method = _run_method_name(method)
    method_tag = f"{args.tag}_{run_method}_r{rate_label.replace('.', 'p')}_{dataset_name}"
    summary_path = REPO_ROOT / "logs" / "efficiency" / "parallel" / method_tag / f"{method_tag}_summary.json"
    if method in ("fastvid", "visionzip", "fastgraphvid", "curvevid"):
        cmd = [
            sys.executable,
            "-u",
            "playground/run_qwen3_adapter_parallel.py",
            "--method",
            method,
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
            "--retention_ratio",
            str(ratio),
            "--expansion",
            str(args.expansion),
            "--llm_retention_ratio",
            str(args.llm_retention_ratio),
            "--tag",
            method_tag,
            "--duration_filter",
            args.duration_filter,
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
            "--fastgraph_ats_ratio",
            str(args.fastgraph_ats_ratio),
            "--fastgraph_temporal_radius",
            str(args.fastgraph_temporal_radius),
            "--fastgraph_temporal_skip",
            str(args.fastgraph_temporal_skip),
            "--fastgraph_temporal_topk",
            str(args.fastgraph_temporal_topk),
            "--fastgraph_edge_threshold",
            str(args.fastgraph_edge_threshold),
            "--fastgraph_protect_ratio",
            str(args.fastgraph_protect_ratio),
            "--fastgraph_attn_weight",
            str(args.fastgraph_attn_weight),
            "--fastgraph_novelty_weight",
            str(args.fastgraph_novelty_weight),
            "--fastgraph_density_weight",
            str(args.fastgraph_density_weight),
            "--curvevid_temperature",
            str(args.curvevid_temperature),
            "--curvevid_mix",
            str(args.curvevid_mix),
            "--curvevid_min_per_frame",
            str(args.curvevid_min_per_frame),
            "--visionzip_dominant_ratio",
            str(args.visionzip_dominant_ratio),
        ]
        if args.gpu_ids:
            cmd.extend(["--gpu_ids", args.gpu_ids])
        return cmd, summary_path

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
        "--duration_filter",
        args.duration_filter,
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
    if method == "graphvid":
        graph_cap = _graph_cap_for_rate(args, ratio)
        cmd.extend(
            [
                "--run_graphvid",
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
        if args.graph_skip_spatial_merge_when_capped:
            cmd.append("--graph_skip_spatial_merge_when_capped")
        else:
            cmd.append("--no-graph_skip_spatial_merge_when_capped")
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
    parser.add_argument("--methods", default="graphvid", help="Comma list: graphvid,flashvid,talon,fastvid,visionzip,fastgraphvid,curvevid.")
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
    parser.add_argument("--duration_filter", default="")
    parser.add_argument("--num_warmup", type=int, default=1)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--free_ratio", type=float, default=0.75)
    parser.add_argument("--min_free_mb", type=int, default=18000)
    parser.add_argument("--max_gpus", type=int, default=0)
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
    parser.add_argument("--graph_merge_representative", default="medoid", choices=["medoid", "mean"])
    parser.add_argument("--graph_final_frame_floor_ratio", type=float, default=0.55)
    parser.add_argument("--graph_skip_spatial_merge_when_capped", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--token_selection_method", default="attn_div_stable")
    parser.add_argument("--graphvid_token_selection_method", default="attn_div_stable")
    parser.add_argument("--talon_target_tokens_per_frame", type=int, default=22)
    parser.add_argument("--talon_question_recall_ratio", type=float, default=0.08)
    parser.add_argument("--talon_question_recall_qweight", type=float, default=0.65)
    parser.add_argument("--talon_question_pooling", default="topk")
    parser.add_argument("--talon_question_pooling_topk", type=int, default=4)
    parser.add_argument(
        "--adapter_budget_uses_expansion",
        "--external_budget_uses_expansion",
        dest="adapter_budget_uses_expansion",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--fastvid_DySeg_c", type=int, default=8)
    parser.add_argument("--fastvid_DySeg_tau", type=float, default=0.90)
    parser.add_argument("--fastvid_DySeg_ignore", type=float, default=0.95)
    parser.add_argument("--fastvid_STPrune_d", type=float, default=0.40)
    parser.add_argument("--fastvid_DTM_p", type=int, default=4)
    parser.add_argument("--fastvid_DTM_beta", type=float, default=0.60)
    parser.add_argument("--fastgraph_ats_ratio", type=float, default=0.60)
    parser.add_argument("--fastgraph_temporal_radius", type=int, default=1)
    parser.add_argument("--fastgraph_temporal_skip", type=int, default=1)
    parser.add_argument("--fastgraph_temporal_topk", type=int, default=2)
    parser.add_argument("--fastgraph_edge_threshold", type=float, default=0.0)
    parser.add_argument("--fastgraph_protect_ratio", type=float, default=0.15)
    parser.add_argument("--fastgraph_attn_weight", type=float, default=0.55)
    parser.add_argument("--fastgraph_novelty_weight", type=float, default=0.30)
    parser.add_argument("--fastgraph_density_weight", type=float, default=0.15)
    parser.add_argument("--curvevid_temperature", type=float, default=0.70)
    parser.add_argument("--curvevid_mix", type=float, default=0.65)
    parser.add_argument("--curvevid_min_per_frame", type=int, default=1)
    parser.add_argument("--visionzip_dominant_ratio", type=float, default=0.85)
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
