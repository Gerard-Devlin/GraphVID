#!/usr/bin/env python3
"""Summarize fair CertVID V3 outer/inner schedule searches."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


RATE_PATTERN = re.compile(r"_r([0-9]+(?:\.[0-9]+)?)_")


def _metric(task_result: dict[str, Any], prefix: str) -> float | None:
    for key, value in task_result.items():
        if key.startswith(prefix) and isinstance(value, (int, float)):
            return float(value)
    return None


def _rate_from_path(path: Path) -> str:
    match = RATE_PATTERN.search(str(path))
    return match.group(1) if match else "unknown"


def _load_manifest(root: Path) -> dict[str, dict[str, str]]:
    path = root / "search_matrix.tsv"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            row["name"]: row
            for row in csv.DictReader(handle, delimiter="\t")
            if row.get("name")
        }


def _run_name(root: Path, path: Path) -> str:
    relative = path.relative_to(root)
    for part in relative.parts:
        if part.startswith("search_"):
            return part.removeprefix("search_")
    return path.parent.name


def _result_files(root: Path) -> Iterable[Path]:
    return sorted(root.rglob("*_results.json"))


def _scores(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    results = payload.get("results", {})
    scores: dict[str, float] = {}

    videomme = results.get("videomme")
    if isinstance(videomme, dict):
        score = _metric(videomme, "videomme_perception_score")
        if score is not None:
            scores["videomme"] = score if score > 1.0 else 100.0 * score

    lvb = results.get("longvideobench_val_v")
    if isinstance(lvb, dict):
        score = _metric(lvb, "lvb_acc")
        if score is not None:
            scores["longvideobench"] = score if score > 1.0 else 100.0 * score

    mvbench_values = []
    for task_name, task_result in results.items():
        if not task_name.startswith("mvbench_") or not isinstance(task_result, dict):
            continue
        score = _metric(task_result, "mvbench_accuracy")
        if score is not None:
            mvbench_values.append(score if score > 1.0 else 100.0 * score)
    if mvbench_values:
        scores["mvbench"] = sum(mvbench_values) / len(mvbench_values)

    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--tasks",
        default="videomme,mvbench,longvideobench_val_v",
        help="Comma-separated lmms-eval tasks required for a complete row.",
    )
    args = parser.parse_args()

    requested_tasks = {
        task.strip() for task in args.tasks.split(",") if task.strip()
    }
    required_metrics: set[str] = set()
    if "videomme" in requested_tasks:
        required_metrics.add("videomme")
    if "mvbench" in requested_tasks:
        required_metrics.add("mvbench")
    if "longvideobench_val_v" in requested_tasks:
        required_metrics.add("longvideobench")

    manifest = _load_manifest(args.root)
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    result_paths: dict[tuple[str, str], list[str]] = defaultdict(list)

    for path in _result_files(args.root):
        run = _run_name(args.root, path)
        rate = _rate_from_path(path)
        grouped[(run, rate)].update(_scores(path))
        result_paths[(run, rate)].append(str(path))

    rows = []
    for (run, rate), scores in grouped.items():
        config = manifest.get(run, {})
        available = [
            scores[name]
            for name in ("videomme", "mvbench", "longvideobench")
            if name in scores
        ]
        rows.append(
            {
                "run": run,
                "rate": rate,
                "outer_layers": config.get("outer_layers", ""),
                "inner_layers": config.get("inner_layers", ""),
                "expansion": config.get("expansion", ""),
                "inner_retention": config.get("inner_retention", ""),
                "average_multiplier": config.get("average_multiplier", ""),
                "videomme": scores.get("videomme", ""),
                "mvbench": scores.get("mvbench", ""),
                "longvideobench": scores.get("longvideobench", ""),
                "mean_score": sum(available) / len(available) if available else "",
                "complete": bool(required_metrics)
                and required_metrics.issubset(scores),
                "result_paths": ";".join(result_paths[(run, rate)]),
            }
        )

    rows.sort(
        key=lambda row: (
            bool(row["complete"]),
            float(row["mean_score"]) if row["mean_score"] != "" else -1.0,
        ),
        reverse=True,
    )
    output = args.output or args.root / "v3_schedule_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
        "rate",
        "outer_layers",
        "inner_layers",
        "expansion",
        "inner_retention",
        "average_multiplier",
        "videomme",
        "mvbench",
        "longvideobench",
        "mean_score",
        "complete",
        "result_paths",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        "| schedule | R | E | inner r | VideoMME | MVBench | LVB | mean |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    for row in rows:
        def show(name: str) -> str:
            value = row[name]
            return "-" if value == "" else f"{float(value):.2f}"

        schedule = f"{row['outer_layers']}/{row['inner_layers']}"
        print(
            f"| {schedule} | {row['rate']} | {row['expansion']} | "
            f"{row['inner_retention']} | {show('videomme')} | "
            f"{show('mvbench')} | {show('longvideobench')} | "
            f"{show('mean_score')} |"
        )
    print(f"\nCSV: {output}")


if __name__ == "__main__":
    main()
