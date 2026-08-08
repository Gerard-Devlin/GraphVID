#!/usr/bin/env python3
"""Build paper-ready CSV/Markdown tables from CertVID V3 ablation runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


ABLATIONS = [
    ("full", "Full CertVID"),
    ("no_doptimal", "w/o D-optimal (Quality Top-K)"),
    ("no_certificates", "w/o Spatiotemporal Certificates"),
    ("no_trajectory", "w/o Trajectory Structure"),
    ("no_query", "w/o Query Evidence"),
    ("no_fusion", "w/o Residual Fusion"),
]

EXPANSION_ABLATIONS = [
    ("e130", "Expansion 1.30 (20+8, r=0.1923)"),
    ("e125", "Expansion 1.25 (20+8, r=0.3000)"),
    ("e120", "Expansion 1.20 (20+8, r=0.4167)"),
    ("e115", "Expansion 1.15 (20+8, r=0.5435)"),
]

TASK_METRICS = {
    "videomme": ("videomme", "videomme_perception_score"),
    "egoschema_subset": ("egoschema_subset", "score"),
    # EgoSchema total is a hidden-label test set. This remains empty until the
    # generated CSV is scored by Kaggle, but keeping the column makes the
    # paper table ready for the returned score.
    "egoschema": ("egoschema", "score"),
    "longvideobench_val_v": ("longvideobench_val_v", "lvb_acc"),
    "mvbench": ("mvbench", "mvbench_accuracy"),
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _result_files(directory: Path) -> list[Path]:
    patterns = ("**/*results*.json", "**/results.json")
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(path for path in directory.glob(pattern) if path.is_file())
    return sorted(paths, key=lambda path: path.stat().st_mtime, reverse=True)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result * 100.0 if -1.0 <= result <= 1.0 else result


def _metric_from_mapping(mapping: dict[str, Any], prefix: str) -> float | None:
    for key, value in mapping.items():
        name = str(key).lower()
        if name.startswith(prefix.lower()) and "stderr" not in name:
            parsed = _number(value)
            if parsed is not None:
                return parsed
    return None


def _task_metric(path: Path, task: str) -> float | None:
    data = _read_json(path)
    task_name, metric = TASK_METRICS[task]
    for root_name in ("results", "groups"):
        root = data.get(root_name)
        if not isinstance(root, dict):
            continue
        exact = root.get(task_name)
        if isinstance(exact, dict):
            value = _metric_from_mapping(exact, metric)
            if value is not None:
                return value

        if task == "mvbench":
            values = []
            for name, payload in root.items():
                if str(name).startswith("mvbench_") and isinstance(payload, dict):
                    value = _metric_from_mapping(payload, metric)
                    if value is not None:
                        values.append(value)
            if values:
                return sum(values) / len(values)
    return None


def _load_metric(root: Path, ablation: str, rate: str, task: str) -> float | None:
    run_dir = root / ablation / f"certvid_v3_r{rate}_{task}"
    for path in _result_files(run_dir):
        value = _task_metric(path, task)
        if value is not None:
            return value
    return None


def _videomme_duration_scores(log_path: Path) -> dict[str, float | None]:
    values: dict[str, float | None] = {"short": None, "medium": None, "long": None}
    if not log_path.is_file():
        return values
    text = log_path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(
        r"Evaluation on video Type:\s*(short|medium|long)\s*:\s*([0-9]+(?:\.[0-9]+)?)%",
        re.IGNORECASE,
    )
    for duration, score in pattern.findall(text):
        values[duration.lower()] = float(score)
    return values


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.1f}"


def _average(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def build_table(
    root: Path,
    rate: str,
    configurations: list[tuple[str, str]] = ABLATIONS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for slug, label in configurations:
        duration = _videomme_duration_scores(root / slug / "videomme.log")
        row: dict[str, Any] = {
            "configuration": label,
            "videomme_short": duration["short"],
            "videomme_medium": duration["medium"],
            "videomme_long": duration["long"],
            "videomme_overall": _load_metric(root, slug, rate, "videomme"),
            "egoschema_subset": _load_metric(root, slug, rate, "egoschema_subset"),
            "egoschema_total": _load_metric(root, slug, rate, "egoschema"),
            "longvideobench": _load_metric(root, slug, rate, "longvideobench_val_v"),
            "mvbench": _load_metric(root, slug, rate, "mvbench"),
        }
        row["average"] = _average(
            [
                row["videomme_overall"],
                row["egoschema_subset"],
                row["egoschema_total"],
                row["longvideobench"],
                row["mvbench"],
            ]
        )
        rows.append(row)
    return rows


def write_outputs(root: Path, rows: list[dict[str, Any]], stem: str = "certvid_v3_ablation_table") -> None:
    fields = [
        "configuration",
        "videomme_short",
        "videomme_medium",
        "videomme_long",
        "videomme_overall",
        "egoschema_subset",
        "egoschema_total",
        "longvideobench",
        "mvbench",
        "average",
    ]
    csv_path = root / f"{stem}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row[field]) if field != "configuration" else row[field] for field in fields})

    lines = [
        "| Configuration | VideoMME Short | Medium | Long | Overall | EgoSchema Subset | EgoSchema Total | LongVideoBench | MVBench | Avg. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {configuration} | {short} | {medium} | {long} | {overall} | {ego_subset} | {ego_total} | {lvb} | {mvbench} | {average} |".format(
                configuration=row["configuration"],
                short=_fmt(row["videomme_short"]),
                medium=_fmt(row["videomme_medium"]),
                long=_fmt(row["videomme_long"]),
                overall=_fmt(row["videomme_overall"]),
                ego_subset=_fmt(row["egoschema_subset"]),
                ego_total=_fmt(row["egoschema_total"]),
                lvb=_fmt(row["longvideobench"]),
                mvbench=_fmt(row["mvbench"]),
                average=_fmt(row["average"]),
            )
        )
    markdown_path = root / f"{stem}.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / f"{stem}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))
    print(f"\nCSV: {csv_path}")
    print(f"Markdown: {markdown_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rate", default="0.10")
    parser.add_argument("--mode", choices=("components", "expansion"), default="components")
    args = parser.parse_args()
    if args.mode == "expansion":
        configurations = EXPANSION_ABLATIONS
        stem = "certvid_v3_expansion_ablation_table"
    else:
        configurations = ABLATIONS
        stem = "certvid_v3_ablation_table"
    write_outputs(
        args.root,
        build_table(args.root, args.rate, configurations),
        stem=stem,
    )


if __name__ == "__main__":
    main()
