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


def _parse_methods(text: str) -> set[str]:
    aliases = {
        "flash": "flashvid",
        "flashvid": "flashvid",
        "graph": "graphvid",
        "graphvid": "graphvid",
        "graft": "graftvid",
        "graftvid": "graftvid",
        "cats": "cats",
        "catsvid": "cats",
        "dyn": "dynflashvid",
        "dynflash": "dynflashvid",
        "dynflashvid": "dynflashvid",
        "learn": "learnflashvid",
        "learnflash": "learnflashvid",
        "learnflashvid": "learnflashvid",
        "hedge": "hedgevid",
        "hedgevid": "hedgevid",
    }
    methods: set[str] = set()
    for part in str(text).split(","):
        item = part.strip().lower()
        if not item:
            continue
        if item not in aliases:
            allowed = ", ".join(sorted(set(aliases)))
            raise ValueError(f"unknown method {item!r}; allowed: {allowed}")
        methods.add(aliases[item])
    if not methods:
        raise ValueError("no methods provided")
    return methods


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


def _comparison(summary: dict[str, Any] | None, key: str, target: str = "graphvid") -> float | None:
    if not summary:
        return None
    comp = summary.get("comparison", {}).get(f"flashvid_vs_{target}", {})
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


def _first_not_none(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def _comparison_any(summary: dict[str, Any] | None, key: str, *targets: str) -> float | None:
    for target in targets:
        value = _comparison(summary, key, target=target)
        if value is not None:
            return value
    return None


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _build_command(
    args: argparse.Namespace,
    rate_label: str,
    ratio: float,
    total_limit: int,
    methods: set[str],
) -> tuple[list[str], Path]:
    ours_like = sorted(methods & {"hedgevid", "dynflashvid", "learnflashvid"})
    if len(ours_like) > 1:
        raise ValueError(f"Only one run_ours-style method can be launched at once; got {ours_like}")
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
        "--gpu_cap",
        str(args.gpu_cap),
        "--tag",
        run_tag,
        "--run_ours" if ("hedgevid" in methods or "dynflashvid" in methods or "learnflashvid" in methods) else "--no-run_ours",
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
    cmd.append("--run_flashvid" if "flashvid" in methods else "--no-run_flashvid")
    cmd.append("--run_graphvid" if "graphvid" in methods else "--no-run_graphvid")
    cmd.append("--run_graftvid" if "graftvid" in methods else "--no-run_graftvid")
    cmd.append("--run_cats" if "cats" in methods else "--no-run_cats")
    if "hedgevid" in methods:
        cmd.extend(["--compression_variant", "hedgevid"])
    if "dynflashvid" in methods:
        cmd.extend(["--compression_variant", "dynflashvid"])
    if "learnflashvid" in methods:
        cmd.extend(["--compression_variant", "learnflashvid"])
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
    if args.graph_respect_temporal_threshold:
        cmd.append("--graph_respect_temporal_threshold")
    else:
        cmd.append("--no-graph_respect_temporal_threshold")
    if args.graph_adaptive_detail_protection:
        cmd.append("--graph_adaptive_detail_protection")
    else:
        cmd.append("--no-graph_adaptive_detail_protection")
    if args.graph_skip_spatial_merge_when_capped:
        cmd.append("--graph_skip_spatial_merge_when_capped")
    else:
        cmd.append("--no-graph_skip_spatial_merge_when_capped")
    cmd.append("--graft_mutual_knn" if args.graft_mutual_knn else "--no-graft_mutual_knn")
    cmd.append("--graft_one_token_per_frame" if args.graft_one_token_per_frame else "--no-graft_one_token_per_frame")
    cmd.append("--graft_adaptive_aggregation" if args.graft_adaptive_aggregation else "--no-graft_adaptive_aggregation")
    cmd.append("--graft_budget_correction" if args.graft_budget_correction else "--no-graft_budget_correction")
    cmd.append("--graft_input_is_residual" if args.graft_input_is_residual else "--no-graft_input_is_residual")
    cmd.extend(
        [
            "--cats_adts_beta",
            str(args.cats_adts_beta),
            "--cats_adts_mode",
            str(args.cats_adts_mode),
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
    cmd.extend(
        [
            "--hedge_stable_floor_ratio",
            str(args.hedge_stable_floor_ratio),
            "--hedge_diversity_weight",
            str(args.hedge_diversity_weight),
            "--hedge_stable_bias",
            str(args.hedge_stable_bias),
            "--hedge_evidence_bias",
            str(args.hedge_evidence_bias),
            "--hedge_max_mmr_candidates",
            str(args.hedge_max_mmr_candidates),
        ]
    )
    cmd.extend(
        [
            "--learn_selector_ckpt",
            str(args.learn_selector_ckpt),
            "--learn_stable_floor_ratio",
            str(args.learn_stable_floor_ratio),
            "--learn_score_blend",
            str(args.learn_score_blend),
            "--learn_q_relevance_weight",
            str(args.learn_q_relevance_weight),
            "--learn_density_topk",
            str(args.learn_density_topk),
        ]
    )
    cmd.append("--learn_qaware" if args.learn_qaware else "--no-learn_qaware")
    cmd.append("--learn_collect_teacher" if args.learn_collect_teacher else "--no-learn_collect_teacher")
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
    graft_tokens = _mean(summary, "graftvid", "compressed_visual_tokens")
    cats_tokens = _mean(summary, "cats", "compressed_visual_tokens")
    hedge_tokens = _first_not_none(_mean(summary, "hedgevid", "compressed_visual_tokens"), _mean(summary, "ours", "compressed_visual_tokens"))
    dyn_tokens = _mean(summary, "dynflashvid", "compressed_visual_tokens")
    learn_tokens = _mean(summary, "learnflashvid", "compressed_visual_tokens")
    flash_acc = _acc(summary, "flashvid")
    graph_acc = _acc(summary, "graphvid")
    graft_acc = _acc(summary, "graftvid")
    cats_acc = _acc(summary, "cats")
    hedge_acc = _first_not_none(_acc(summary, "hedgevid"), _acc(summary, "ours"))
    dyn_acc = _acc(summary, "dynflashvid")
    learn_acc = _acc(summary, "learnflashvid")
    return {
        "retention_ratio": f"{rate_label.replace('p', '.')}%",
        "flashvid_acc": flash_acc,
        "graphvid_acc": graph_acc,
        "graftvid_acc": graft_acc,
        "cats_acc": cats_acc,
        "hedgevid_acc": hedge_acc,
        "dynflashvid_acc": dyn_acc,
        "learnflashvid_acc": learn_acc,
        "acc_delta": None if flash_acc is None or graph_acc is None else graph_acc - flash_acc,
        "graft_acc_delta": None if flash_acc is None or graft_acc is None else graft_acc - flash_acc,
        "cats_acc_delta": None if flash_acc is None or cats_acc is None else cats_acc - flash_acc,
        "hedge_acc_delta": None if flash_acc is None or hedge_acc is None else hedge_acc - flash_acc,
        "dyn_acc_delta": None if flash_acc is None or dyn_acc is None else dyn_acc - flash_acc,
        "learn_acc_delta": None if flash_acc is None or learn_acc is None else learn_acc - flash_acc,
        "flashvid_short": _duration_acc(summary, "flashvid", "short"),
        "graphvid_short": _duration_acc(summary, "graphvid", "short"),
        "graftvid_short": _duration_acc(summary, "graftvid", "short"),
        "cats_short": _duration_acc(summary, "cats", "short"),
        "hedgevid_short": _first_not_none(_duration_acc(summary, "hedgevid", "short"), _duration_acc(summary, "ours", "short")),
        "dynflashvid_short": _duration_acc(summary, "dynflashvid", "short"),
        "learnflashvid_short": _duration_acc(summary, "learnflashvid", "short"),
        "flashvid_medium": _duration_acc(summary, "flashvid", "medium"),
        "graphvid_medium": _duration_acc(summary, "graphvid", "medium"),
        "graftvid_medium": _duration_acc(summary, "graftvid", "medium"),
        "cats_medium": _duration_acc(summary, "cats", "medium"),
        "hedgevid_medium": _first_not_none(_duration_acc(summary, "hedgevid", "medium"), _duration_acc(summary, "ours", "medium")),
        "dynflashvid_medium": _duration_acc(summary, "dynflashvid", "medium"),
        "learnflashvid_medium": _duration_acc(summary, "learnflashvid", "medium"),
        "flashvid_long": _duration_acc(summary, "flashvid", "long"),
        "graphvid_long": _duration_acc(summary, "graphvid", "long"),
        "graftvid_long": _duration_acc(summary, "graftvid", "long"),
        "cats_long": _duration_acc(summary, "cats", "long"),
        "hedgevid_long": _first_not_none(_duration_acc(summary, "hedgevid", "long"), _duration_acc(summary, "ours", "long")),
        "dynflashvid_long": _duration_acc(summary, "dynflashvid", "long"),
        "learnflashvid_long": _duration_acc(summary, "learnflashvid", "long"),
        "flashvid_tokens": flash_tokens,
        "graphvid_tokens": graph_tokens,
        "graftvid_tokens": graft_tokens,
        "cats_tokens": cats_tokens,
        "hedgevid_tokens": hedge_tokens,
        "dynflashvid_tokens": dyn_tokens,
        "learnflashvid_tokens": learn_tokens,
        "token_reduction": _comparison(summary, "visual_token_reduction"),
        "graft_token_reduction": _comparison(summary, "visual_token_reduction", target="graftvid"),
        "cats_token_reduction": _comparison(summary, "visual_token_reduction", target="cats"),
        "hedge_token_reduction": _comparison_any(summary, "visual_token_reduction", "hedgevid", "ours"),
        "dyn_token_reduction": _comparison(summary, "visual_token_reduction", target="dynflashvid"),
        "learn_token_reduction": _comparison(summary, "visual_token_reduction", target="learnflashvid"),
        "flashvid_latency_ms": _mean(summary, "flashvid", "latency_ms"),
        "graphvid_latency_ms": _mean(summary, "graphvid", "latency_ms"),
        "graftvid_latency_ms": _mean(summary, "graftvid", "latency_ms"),
        "cats_latency_ms": _mean(summary, "cats", "latency_ms"),
        "hedgevid_latency_ms": _first_not_none(_mean(summary, "hedgevid", "latency_ms"), _mean(summary, "ours", "latency_ms")),
        "dynflashvid_latency_ms": _mean(summary, "dynflashvid", "latency_ms"),
        "learnflashvid_latency_ms": _mean(summary, "learnflashvid", "latency_ms"),
        "latency_speedup": _comparison(summary, "latency_speedup"),
        "graft_latency_speedup": _comparison(summary, "latency_speedup", target="graftvid"),
        "cats_latency_speedup": _comparison(summary, "latency_speedup", target="cats"),
        "hedge_latency_speedup": _comparison_any(summary, "latency_speedup", "hedgevid", "ours"),
        "dyn_latency_speedup": _comparison(summary, "latency_speedup", target="dynflashvid"),
        "learn_latency_speedup": _comparison(summary, "latency_speedup", target="learnflashvid"),
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
        "graftvid_acc",
        "cats_acc",
        "hedgevid_acc",
        "dynflashvid_acc",
        "learnflashvid_acc",
        "acc_delta",
        "graft_acc_delta",
        "cats_acc_delta",
        "hedge_acc_delta",
        "dyn_acc_delta",
        "learn_acc_delta",
        "flashvid_short",
        "graphvid_short",
        "graftvid_short",
        "cats_short",
        "hedgevid_short",
        "dynflashvid_short",
        "learnflashvid_short",
        "flashvid_medium",
        "graphvid_medium",
        "graftvid_medium",
        "cats_medium",
        "hedgevid_medium",
        "dynflashvid_medium",
        "learnflashvid_medium",
        "flashvid_long",
        "graphvid_long",
        "graftvid_long",
        "cats_long",
        "hedgevid_long",
        "dynflashvid_long",
        "learnflashvid_long",
        "flashvid_tokens",
        "graphvid_tokens",
        "graftvid_tokens",
        "cats_tokens",
        "hedgevid_tokens",
        "dynflashvid_tokens",
        "learnflashvid_tokens",
        "token_reduction",
        "graft_token_reduction",
        "cats_token_reduction",
        "hedge_token_reduction",
        "dyn_token_reduction",
        "learn_token_reduction",
        "flashvid_latency_ms",
        "graphvid_latency_ms",
        "graftvid_latency_ms",
        "cats_latency_ms",
        "hedgevid_latency_ms",
        "dynflashvid_latency_ms",
        "learnflashvid_latency_ms",
        "latency_speedup",
        "graft_latency_speedup",
        "cats_latency_speedup",
        "hedge_latency_speedup",
        "dyn_latency_speedup",
        "learn_latency_speedup",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    lines = [
        "| R | FlashVID Acc | GraphVID Acc | GRAFT Acc | CATS Acc | HEDGE Acc | DYN Acc | G Delta | GRAFT Delta | CATS Delta | HEDGE Delta | DYN Delta | F Short | G Short | GRAFT Short | CATS Short | HEDGE Short | DYN Short | F Medium | G Medium | GRAFT Medium | CATS Medium | HEDGE Medium | DYN Medium | F Long | G Long | GRAFT Long | CATS Long | HEDGE Long | DYN Long | F Tokens | G Tokens | GRAFT Tokens | CATS Tokens | HEDGE Tokens | DYN Tokens | G Token Red. | GRAFT Token Red. | CATS Token Red. | HEDGE Token Red. | DYN Token Red. | F Lat. | G Lat. | GRAFT Lat. | CATS Lat. | HEDGE Lat. | DYN Lat. | G Speedup | GRAFT Speedup | CATS Speedup | HEDGE Speedup | DYN Speedup |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {r} | {fa} | {ga} | {gfa} | {ca} | {ha} | {da} | {d} | {gd} | {cd} | {hd} | {dd} | {fs} | {gs} | {gfs} | {cs} | {hs} | {ds} | {fm} | {gm} | {gfm} | {cm} | {hm} | {dm} | {fl} | {gl} | {gfl} | {cl} | {hl} | {dl} | {ft} | {gt} | {gft} | {ct} | {ht} | {dt} | {tr} | {gtr} | {ctr} | {htr} | {dtr} | {flat} | {glat} | {gflat} | {clat} | {hlat} | {dlat} | {sp} | {gsp} | {csp} | {hsp} | {dsp} |".format(
                r=row["retention_ratio"],
                fa=_fmt(row.get("flashvid_acc")),
                ga=_fmt(row.get("graphvid_acc")),
                gfa=_fmt(row.get("graftvid_acc")),
                ca=_fmt(row.get("cats_acc")),
                ha=_fmt(row.get("hedgevid_acc")),
                da=_fmt(row.get("dynflashvid_acc")),
                d=_fmt(row.get("acc_delta")),
                gd=_fmt(row.get("graft_acc_delta")),
                cd=_fmt(row.get("cats_acc_delta")),
                hd=_fmt(row.get("hedge_acc_delta")),
                dd=_fmt(row.get("dyn_acc_delta")),
                fs=_fmt(row.get("flashvid_short")),
                gs=_fmt(row.get("graphvid_short")),
                gfs=_fmt(row.get("graftvid_short")),
                cs=_fmt(row.get("cats_short")),
                hs=_fmt(row.get("hedgevid_short")),
                ds=_fmt(row.get("dynflashvid_short")),
                fm=_fmt(row.get("flashvid_medium")),
                gm=_fmt(row.get("graphvid_medium")),
                gfm=_fmt(row.get("graftvid_medium")),
                cm=_fmt(row.get("cats_medium")),
                hm=_fmt(row.get("hedgevid_medium")),
                dm=_fmt(row.get("dynflashvid_medium")),
                fl=_fmt(row.get("flashvid_long")),
                gl=_fmt(row.get("graphvid_long")),
                gfl=_fmt(row.get("graftvid_long")),
                cl=_fmt(row.get("cats_long")),
                hl=_fmt(row.get("hedgevid_long")),
                dl=_fmt(row.get("dynflashvid_long")),
                ft=_fmt(row.get("flashvid_tokens")),
                gt=_fmt(row.get("graphvid_tokens")),
                gft=_fmt(row.get("graftvid_tokens")),
                ct=_fmt(row.get("cats_tokens")),
                ht=_fmt(row.get("hedgevid_tokens")),
                dt=_fmt(row.get("dynflashvid_tokens")),
                tr=_fmt(row.get("token_reduction")),
                gtr=_fmt(row.get("graft_token_reduction")),
                ctr=_fmt(row.get("cats_token_reduction")),
                htr=_fmt(row.get("hedge_token_reduction")),
                dtr=_fmt(row.get("dyn_token_reduction")),
                flat=_fmt(row.get("flashvid_latency_ms")),
                glat=_fmt(row.get("graphvid_latency_ms")),
                gflat=_fmt(row.get("graftvid_latency_ms")),
                clat=_fmt(row.get("cats_latency_ms")),
                hlat=_fmt(row.get("hedgevid_latency_ms")),
                dlat=_fmt(row.get("dynflashvid_latency_ms")),
                sp=_fmt(row.get("latency_speedup"), 3),
                gsp=_fmt(row.get("graft_latency_speedup"), 3),
                csp=_fmt(row.get("cats_latency_speedup"), 3),
                hsp=_fmt(row.get("hedge_latency_speedup"), 3),
                dsp=_fmt(row.get("dyn_latency_speedup"), 3),
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
    parser.add_argument("--methods", default="flashvid,graphvid", help="Comma list: flashvid,graphvid,graftvid,cats,dynflashvid,learnflashvid,hedgevid.")
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
    parser.add_argument("--max_gpus", type=int, default=4)
    parser.add_argument("--gpu_cap", type=int, default=4)
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
    parser.add_argument("--cats_adts_mode", default="cats", choices=["cats", "flashvid"])
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
    parser.add_argument("--hedge_stable_floor_ratio", type=float, default=0.85)
    parser.add_argument("--hedge_diversity_weight", type=float, default=0.04)
    parser.add_argument("--hedge_stable_bias", type=float, default=0.05)
    parser.add_argument("--hedge_evidence_bias", type=float, default=0.0)
    parser.add_argument("--hedge_max_mmr_candidates", type=int, default=2048)
    parser.add_argument("--learn_selector_ckpt", default="")
    parser.add_argument("--learn_qaware", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learn_stable_floor_ratio", type=float, default=0.50)
    parser.add_argument("--learn_score_blend", type=float, default=0.50)
    parser.add_argument("--learn_q_relevance_weight", type=float, default=0.20)
    parser.add_argument("--learn_density_topk", type=int, default=8)
    parser.add_argument("--learn_collect_teacher", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args, extra_args = parser.parse_known_args()
    args.extra_args = extra_args

    dataset_path = Path(args.dataset_jsonl)
    if not dataset_path.is_absolute():
        dataset_path = REPO_ROOT / dataset_path
    total_limit = int(args.total_limit) if int(args.total_limit) > 0 else _count_jsonl(dataset_path)

    rows: list[dict[str, Any]] = []
    methods = _parse_methods(args.methods)
    for rate_label, ratio in _parse_rates(args.rates):
        cmd, summary_path = _build_command(args, rate_label, ratio, total_limit, methods)
        print("[run]", " ".join(shlex.quote(x) for x in cmd))
        if not args.dry_run and not (args.resume and summary_path.exists()):
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        rows.append(_row(rate_label, _load_summary(summary_path)))
        _write_tables(REPO_ROOT / args.output_dir / args.tag, rows)


if __name__ == "__main__":
    main()
