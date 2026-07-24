#!/usr/bin/env python3
"""Build deterministic, label-blind LongVideoBench search/holdout subsets."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return records


def _read_diagnostics(directory: Path) -> dict[str, dict[str, Any]]:
    diagnostics = {}
    for path in sorted(directory.glob("certvid_v8_diagnostics.rank*.jsonl")):
        for record in _read_jsonl(path):
            sample_id = str(record.get("sample_id", ""))
            if sample_id:
                diagnostics[sample_id] = record
    return diagnostics


def _read_samples(path: Path) -> dict[str, dict[str, Any]]:
    samples = {}
    for record in _read_jsonl(path):
        metric = record.get("lvb_acc", {})
        sample_id = str(metric.get("id", ""))
        if not sample_id:
            raise ValueError(f"Missing lvb_acc.id in {path}")
        samples[sample_id] = {
            "id": sample_id,
            "doc_id": int(record["doc_id"]),
            "duration_group": str(metric.get("duration_group", "unknown")),
            "question_category": str(metric.get("question_category", "unknown")),
            "answer": metric.get("answer"),
            "prediction": metric.get("parsed_pred"),
        }
    return samples


def _largest_remainder(values: list[str], size: int) -> dict[str, int]:
    counts = collections.Counter(values)
    total = sum(counts.values())
    exact = {key: size * count / total for key, count in counts.items()}
    targets = {key: min(counts[key], int(math.floor(value))) for key, value in exact.items()}
    remaining = size - sum(targets.values())
    order = sorted(
        counts,
        key=lambda key: (exact[key] - targets[key], counts[key], key),
        reverse=True,
    )
    for key in order:
        if remaining <= 0:
            break
        if targets[key] < counts[key]:
            targets[key] += 1
            remaining -= 1
    return targets


def _stable_noise(seed: str, sample_id: str) -> float:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _select_balanced(
    pool: list[dict[str, Any]],
    size: int,
    seed: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if size > len(pool):
        raise ValueError(f"Requested {size} samples from a pool of {len(pool)}")

    dimensions = (
        "duration_group",
        "question_category",
        "route",
        "fallback",
        "reference_disagreement",
        "duration_category",
    )
    targets = {
        dimension: _largest_remainder([str(row[dimension]) for row in pool], size)
        for dimension in dimensions
    }
    current = {dimension: collections.Counter() for dimension in dimensions}
    remaining = list(pool)
    selected = []

    while len(selected) < size:
        best_index = -1
        best_score = None
        for index, row in enumerate(remaining):
            score = 0.0
            for dimension in dimensions:
                value = str(row[dimension])
                target = targets[dimension][value]
                count = current[dimension][value]
                if count < target:
                    score += 1.0 + (target - count) / max(target, 1)
                else:
                    score -= (count - target + 1) / max(size, 1)
            score += _stable_noise(seed, row["id"]) * 1e-6
            if best_score is None or score > best_score:
                best_score = score
                best_index = index

        row = remaining.pop(best_index)
        selected.append(row)
        for dimension in dimensions:
            current[dimension][str(row[dimension])] += 1

    return selected, remaining


def _write_split(output_dir: Path, name: str, records: list[dict[str, Any]]) -> None:
    ids_path = output_dir / f"{name}.ids"
    manifest_path = output_dir / f"{name}.jsonl"
    ids_path.write_text(
        "".join(f"{record['id']}\n" for record in records),
        encoding="utf-8",
    )
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "duration_group",
        "question_category",
        "route",
        "fallback",
        "reference_disagreement",
    )
    return {
        "size": len(records),
        **{
            field: dict(sorted(collections.Counter(str(row[field]) for row in records).items()))
            for field in fields
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-jsonl", type=Path, required=True)
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--reference-samples-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-size", type=int, default=192)
    parser.add_argument("--holdout-size", type=int, default=192)
    parser.add_argument("--seed", default="20260724")
    args = parser.parse_args()

    samples = _read_samples(args.samples_jsonl)
    diagnostics = _read_diagnostics(args.diagnostics_dir)
    reference = (
        _read_samples(args.reference_samples_jsonl)
        if args.reference_samples_jsonl
        else {}
    )

    rows = []
    for sample_id, sample in samples.items():
        diagnostic = diagnostics.get(sample_id, {})
        reference_prediction = reference.get(sample_id, {}).get("prediction")
        if reference_prediction is None:
            disagreement = "unknown"
        elif reference_prediction == sample["prediction"]:
            disagreement = "agree"
        else:
            disagreement = "disagree"
        row = {
            **sample,
            "route": str(diagnostic.get("query_intent", "unknown")),
            "fallback": str(diagnostic.get("fallback_reason") or "active"),
            "reference_disagreement": disagreement,
            "duration_category": (
                f"{sample['duration_group']}::{sample['question_category']}"
            ),
            "baseline_correct": sample["answer"] == sample["prediction"],
        }
        rows.append(row)

    rows.sort(key=lambda row: row["doc_id"])
    search, remaining = _select_balanced(rows, args.search_size, f"{args.seed}:search")
    holdout, _ = _select_balanced(
        remaining,
        args.holdout_size,
        f"{args.seed}:holdout",
    )
    search.sort(key=lambda row: row["doc_id"])
    holdout.sort(key=lambda row: row["doc_id"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_split(args.output_dir, f"lvb_v8_search_{len(search)}", search)
    _write_split(args.output_dir, f"lvb_v8_holdout_{len(holdout)}", holdout)

    summary = {
        "selection_is_label_blind": True,
        "selection_dimensions": [
            "duration_group",
            "question_category",
            "route",
            "fallback",
            "reference_disagreement",
            "duration_group x question_category",
        ],
        "seed": args.seed,
        "source_size": len(rows),
        "search": _summary(search),
        "holdout": _summary(holdout),
        "overlap": len({row["id"] for row in search} & {row["id"] for row in holdout}),
    }
    (args.output_dir / "lvb_v8_subset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
