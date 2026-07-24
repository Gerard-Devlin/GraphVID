#!/usr/bin/env python3
"""Summarize paired CertVID V8 LongVideoBench hyperparameter runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _find_samples(result_path: Path) -> Path | None:
    candidates = list(result_path.parent.glob("*samples_longvideobench_val_v.jsonl"))
    return candidates[0] if candidates else None


def _sample_metrics(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    records = _read_jsonl(path)
    rows = [record["lvb_acc"] for record in records if "lvb_acc" in record]
    if not rows:
        return {}
    correct = [row["answer"] == row["parsed_pred"] for row in rows]
    categories = Counter()
    durations = Counter()
    for row, is_correct in zip(rows, correct):
        categories[str(row.get("question_category"))] += int(is_correct)
        durations[str(row.get("duration_group"))] += int(is_correct)
    category_totals = Counter(str(row.get("question_category")) for row in rows)
    duration_totals = Counter(str(row.get("duration_group")) for row in rows)
    return {
        "sample_count": len(rows),
        "sample_accuracy": sum(correct) / len(correct),
        "worst_category_accuracy": min(
            categories[key] / category_totals[key] for key in category_totals
        ),
        "worst_duration_accuracy": min(
            durations[key] / duration_totals[key] for key in duration_totals
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = []
    for result_path in sorted(args.root.rglob("*_results.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        task_result = result.get("results", {}).get("longvideobench_val_v")
        if not task_result:
            continue
        score = task_result.get("lvb_acc,none")
        run_name = next(
            (
                part
                for part in result_path.parts
                if part.startswith("search_") or part.startswith("fine_")
            ),
            result_path.parent.name,
        )
        row = {
            "run": run_name,
            "accuracy": float(score),
            **_sample_metrics(_find_samples(result_path)),
            "result_path": str(result_path),
        }
        # Keep the score on the accuracy scale and penalize brittle group regressions.
        worst_category = row.get("worst_category_accuracy", row["accuracy"])
        worst_duration = row.get("worst_duration_accuracy", row["accuracy"])
        row["selection_score"] = (
            row["accuracy"]
            - 0.10 * max(0.0, row["accuracy"] - worst_category)
            - 0.10 * max(0.0, row["accuracy"] - worst_duration)
        )
        rows.append(row)

    rows.sort(key=lambda row: (row["selection_score"], row["accuracy"]), reverse=True)
    output = args.output or args.root / "lvb_v8_search_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
        "accuracy",
        "sample_count",
        "sample_accuracy",
        "worst_category_accuracy",
        "worst_duration_accuracy",
        "selection_score",
        "result_path",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        "| run | acc | worst category | worst duration | selection score |\n"
        "|---|---:|---:|---:|---:|"
    )
    for row in rows:
        print(
            f"| {row['run']} | {100 * row['accuracy']:.2f} | "
            f"{100 * row.get('worst_category_accuracy', 0):.2f} | "
            f"{100 * row.get('worst_duration_accuracy', 0):.2f} | "
            f"{100 * row['selection_score']:.2f} |"
        )
    print(f"\nCSV: {output}")


if __name__ == "__main__":
    main()
