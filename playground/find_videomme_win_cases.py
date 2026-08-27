#!/usr/bin/env python3
"""Find paired VideoMME questions where CertVID wins over every baseline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METHODS = ("fastv", "visionzip", "fastvid", "flashvid", "certvidfinal2")
BASELINES = METHODS[:-1]
OURS = METHODS[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rate", default="0.01")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def find_sample_file(root: Path, method: str, rate: str) -> Path:
    run_name = f"{method}_r{rate}_videomme"
    matches = [
        path
        for path in root.rglob("*_samples_videomme.jsonl")
        if run_name in path.parts
    ]
    if len(matches) != 1:
        listing = "\n".join(str(path) for path in matches) or "<none>"
        raise RuntimeError(f"expected one sample file for {run_name}, found {len(matches)}:\n{listing}")
    return matches[0]


def metric_payload(sample: dict[str, Any]) -> dict[str, Any]:
    payload = sample.get("videomme_perception_score")
    if isinstance(payload, list) and len(payload) == 1:
        payload = payload[0]
    if not isinstance(payload, dict):
        raise ValueError("sample has no VideoMME per-question metric payload")
    return payload


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sample = json.loads(line)
            payload = metric_payload(sample)
            question_id = str(payload["question_id"])
            if question_id in predictions:
                raise ValueError(f"duplicate question_id {question_id} in {path}")
            predictions[question_id] = {
                "question_id": question_id,
                "videoID": str(payload["videoID"]),
                "answer": str(payload["answer"]).strip().upper(),
                "pred_answer": str(payload["pred_answer"]).strip().upper(),
                "score": float(payload["score"]),
                "input": sample.get("input", ""),
                "duration": payload.get("duration"),
                "category": payload.get("category"),
                "sub_category": payload.get("sub_category"),
                "task_category": payload.get("task_category"),
            }
    return predictions


def main() -> None:
    args = parse_args()
    sample_files = {
        method: find_sample_file(args.root, method, args.rate)
        for method in METHODS
    }
    predictions = {
        method: load_predictions(path)
        for method, path in sample_files.items()
    }

    key_sets = {method: set(values) for method, values in predictions.items()}
    common = set.intersection(*key_sets.values())
    if any(keys != common for keys in key_sets.values()):
        counts = {method: len(keys) for method, keys in key_sets.items()}
        raise RuntimeError(f"methods did not evaluate identical questions: {counts}, common={len(common)}")

    cases = []
    for question_id in sorted(common):
        ours = predictions[OURS][question_id]
        if ours["score"] != 1.0:
            continue
        if not all(predictions[method][question_id]["score"] == 0.0 for method in BASELINES):
            continue
        record = {
            key: ours[key]
            for key in (
                "question_id",
                "videoID",
                "answer",
                "input",
                "duration",
                "category",
                "sub_category",
                "task_category",
            )
        }
        record["predictions"] = {
            method: predictions[method][question_id]["pred_answer"]
            for method in METHODS
        }
        cases.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "strict_win_cases.jsonl"
    csv_path = args.output_dir / "strict_win_cases.csv"
    video_path = args.output_dir / "strict_win_video_ids.txt"
    summary_path = args.output_dir / "summary.json"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")

    fieldnames = [
        "question_id",
        "videoID",
        "answer",
        *[f"pred_{method}" for method in METHODS],
        "duration",
        "category",
        "sub_category",
        "task_category",
        "input",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            row = {key: case.get(key) for key in fieldnames}
            for method in METHODS:
                row[f"pred_{method}"] = case["predictions"][method]
            writer.writerow(row)

    win_videos = sorted({case["videoID"] for case in cases})
    video_path.write_text("\n".join(win_videos) + ("\n" if win_videos else ""), encoding="utf-8")
    summary = {
        "paired_questions": len(common),
        "strict_win_questions": len(cases),
        "strict_win_videos": len(win_videos),
        "sample_files": {method: str(path) for method, path in sample_files.items()},
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"cases_jsonl={jsonl_path}")
    print(f"cases_csv={csv_path}")


if __name__ == "__main__":
    main()
