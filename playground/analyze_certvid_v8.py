from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Optional


def _expand(patterns: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            paths.update(Path(match) for match in matches)
        elif Path(pattern).is_file():
            paths.add(Path(pattern))
    return sorted(paths)


def _read_json_records(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        text = path.read_text(encoding="utf-8-sig")
        stripped = text.lstrip()
        if not stripped:
            continue
        if stripped.startswith("["):
            payload = json.loads(text)
            for record in payload:
                if isinstance(record, dict):
                    yield record
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if isinstance(record, dict):
                yield record


def _nested(record: Any, *paths: str) -> Any:
    for path in paths:
        current = record
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                current = None
                break
            current = current[part]
        if current is not None:
            return current
    return None


def _sample_id(record: dict[str, Any]) -> Optional[str]:
    value = _nested(
        record,
        "sample_id",
        "doc_id",
        "id",
        "question_id",
        "doc.id",
        "doc.question_id",
        "arguments.doc_id",
    )
    return None if value is None else str(value)


def _choice_letter(value: Any) -> Optional[str]:
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, dict):
        value = _nested(value, "answer", "prediction", "text")
    if value is None:
        return None
    text = str(value).strip().upper()
    for character in text:
        if character in "ABCDEFG":
            return character
    return None


def _correctness(record: dict[str, Any]) -> Optional[bool]:
    direct = _nested(
        record,
        "correct",
        "is_correct",
        "exact_match",
        "acc",
        "metrics.exact_match",
        "metrics.acc",
    )
    if isinstance(direct, bool):
        return direct
    if isinstance(direct, (int, float)):
        return bool(direct)

    prediction = _choice_letter(
        _nested(
            record,
            "parsed_pred",
            "prediction",
            "filtered_resps",
            "resps",
            "response",
        )
    )
    answer = _nested(
        record,
        "answer",
        "target",
        "doc.answer",
        "doc.correct_answer",
    )
    correct_choice = _nested(record, "correct_choice", "doc.correct_choice")
    if answer is None and isinstance(correct_choice, int):
        answer = chr(ord("A") + correct_choice)
    answer_letter = _choice_letter(answer)
    if prediction is None or answer_letter is None:
        return None
    return prediction == answer_letter


def _fold(sample_id: str, folds: int) -> int:
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % folds


def _number(record: dict[str, Any], path: str, default: float = 0.0) -> float:
    value = _nested(record, path)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _summarize(records: list[dict[str, Any]], result_map: dict[str, bool]) -> dict[str, Any]:
    sample_ids = [str(record["sample_id"]) for record in records]
    labels = [result_map[sample_id] for sample_id in sample_ids if sample_id in result_map]
    return {
        "samples": len(records),
        "labeled": len(labels),
        "accuracy": mean(labels) if labels else None,
        "fallback_rate": mean(
            record.get("fallback_reason") is not None for record in records
        )
        if records
        else 0.0,
        "swap_mean": mean(_number(record, "swap_count") for record in records)
        if records
        else 0.0,
        "modified_ratio_mean": mean(
            _number(record, "modified_ratio") for record in records
        )
        if records
        else 0.0,
        "v3_overlap_mean": mean(
            _number(record, "v3_overlap_ratio", 1.0) for record in records
        )
        if records
        else 1.0,
        "d_efficiency_mean": mean(
            _number(record, "d_efficiency", 1.0) for record in records
        )
        if records
        else 1.0,
        "frame_cv_mean": mean(
            _number(record, "final_frame_distribution.cv") for record in records
        )
        if records
        else 0.0,
        "frame_entropy_mean": mean(
            _number(record, "final_frame_distribution.normalized_entropy")
            for record in records
        )
        if records
        else 0.0,
        "query_coverage_mean": mean(
            _number(record, "final_query_coverage", 1.0) for record in records
        )
        if records
        else 1.0,
        "objective_gain_mean": mean(
            _number(record, "objective_gain") for record in records
        )
        if records
        else 0.0,
    }


def _format(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _print_table(title: str, rows: list[dict[str, Any]]) -> None:
    columns = (
        "group",
        "samples",
        "labeled",
        "accuracy",
        "fallback_rate",
        "swap_mean",
        "modified_ratio_mean",
        "v3_overlap_mean",
        "d_efficiency_mean",
        "frame_cv_mean",
        "frame_entropy_mean",
        "query_coverage_mean",
        "objective_gain_mean",
    )
    print(f"\n{title}")
    print("\t".join(columns))
    for row in rows:
        print("\t".join(_format(row.get(column)) for column in columns))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize CertVID V8 diagnostics. Labels are joined only for "
            "offline reporting and are never consumed by the compressor."
        )
    )
    parser.add_argument("--diagnostics", nargs="+", required=True)
    parser.add_argument("--samples", nargs="*", default=[])
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--holdout-fold", type=int, default=0)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    if args.num_folds < 2:
        raise ValueError("--num-folds must be at least 2")
    if not 0 <= args.holdout_fold < args.num_folds:
        raise ValueError("--holdout-fold must be in [0, num-folds)")

    diagnostic_paths = _expand(args.diagnostics)
    if not diagnostic_paths:
        raise FileNotFoundError("no diagnostic JSONL files matched")
    diagnostics: dict[str, dict[str, Any]] = {}
    for record in _read_json_records(diagnostic_paths):
        sample_id = _sample_id(record)
        if sample_id is None or sample_id == "unknown":
            continue
        normalized = dict(record)
        normalized["sample_id"] = sample_id
        diagnostics[sample_id] = normalized

    result_map: dict[str, bool] = {}
    for record in _read_json_records(_expand(args.samples)):
        sample_id = _sample_id(record)
        correct = _correctness(record)
        if sample_id is not None and correct is not None:
            result_map[sample_id] = correct

    partitions: dict[str, list[dict[str, Any]]] = {"tune": [], "holdout": []}
    for sample_id, record in diagnostics.items():
        name = (
            "holdout"
            if _fold(sample_id, args.num_folds) == args.holdout_fold
            else "tune"
        )
        partitions[name].append(record)

    rows: list[dict[str, Any]] = []
    for partition, records in partitions.items():
        overall = _summarize(records, result_map)
        overall["group"] = f"{partition}/all"
        rows.append(overall)

        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[("intent", str(record.get("query_intent", "unknown")))].append(
                record
            )
            grouped[("category", str(record.get("eval_category", "unknown")))].append(
                record
            )
            grouped[
                ("fallback", str(record.get("fallback_reason") or "repaired"))
            ].append(record)
        for (kind, value), group_records in sorted(grouped.items()):
            summary = _summarize(group_records, result_map)
            summary["group"] = f"{partition}/{kind}:{value}"
            rows.append(summary)

    _print_table("CertVID V8 leakage-safe tune/holdout summary", rows)
    print(
        "\nAnswer labels are used only in this offline report. "
        "Tune parameters on tune rows, then inspect holdout once."
    )

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        columns = list(rows[0].keys()) if rows else ["group"]
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
