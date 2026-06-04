#!/usr/bin/env python3
"""Build LongVideoBench validation JSONL records for the GraphVID runner.

This follows the LMMs-Eval `longvideobench_val_v` prompt style: video only,
question/options in text, and direct multiple-choice letter output.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _load_annotations(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("validation", "val", "data", "questions", "annotations"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        if all(isinstance(value, dict) for value in data.values()):
            return list(data.values())
    raise ValueError(f"unsupported LongVideoBench annotation format: {path}")


def _get_options(item: dict[str, Any]) -> list[str]:
    if isinstance(item.get("options"), list):
        return [str(option) for option in item["options"] if str(option).strip() and str(option).strip() != "N/A"]

    options: list[str] = []
    for idx in range(10):
        value = item.get(f"option{idx}")
        if value is None:
            value = item.get(f"option_{idx}")
        if value is None or str(value).strip() == "N/A":
            continue
        options.append(str(value).strip())
    if not options:
        raise ValueError(f"missing options for LongVideoBench item id={item.get('id')}")
    return options


def _answer_letter(item: dict[str, Any], options: list[str]) -> str:
    if "correct_choice" in item and item["correct_choice"] is not None:
        idx = int(item["correct_choice"])
        if 0 <= idx < len(LETTERS):
            return LETTERS[idx]
    answer = item.get("answer")
    if answer is None:
        raise ValueError(f"missing answer/correct_choice for LongVideoBench item id={item.get('id')}")
    answer_text = str(answer).strip()
    if answer_text in options:
        return LETTERS[options.index(answer_text)]
    if answer_text.isdigit():
        idx = int(answer_text)
        if 0 <= idx < len(LETTERS):
            return LETTERS[idx]
    return answer_text.upper()[:1]


def _format_prompt(item: dict[str, Any], options: list[str]) -> str:
    lines = [str(item.get("question", "")).strip()]
    lines.extend(f"{LETTERS[idx]}. {option}" for idx, option in enumerate(options))
    lines.append("Answer with the option's letter from the given choices directly.")
    return "\n".join(lines)


def _resolve_video_path(video_root: Path, rel_path: str) -> Path | None:
    rel = Path(rel_path)
    candidates = [
        video_root / rel,
        video_root / rel.name,
    ]
    if rel.suffix == "":
        candidates.extend(
            [
                video_root / f"{rel_path}.mp4",
                video_root / f"{rel.name}.mp4",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    matches = list(video_root.rglob(rel.name))
    if matches:
        return matches[0].resolve()
    return None


def build_records(annotation_path: Path, video_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for idx, item in enumerate(_load_annotations(annotation_path)):
        options = _get_options(item)
        rel_video = str(item.get("video_path") or item.get("video") or item.get("path") or "").strip()
        if not rel_video:
            raise ValueError(f"missing video_path for LongVideoBench item id={item.get('id', idx)}")
        visual_path = _resolve_video_path(video_root, rel_video)
        question_id = str(item.get("id") or item.get("question_id") or f"longvideobench-{idx:04d}")
        record = {
            "question_id": question_id,
            "videoID": rel_video,
            "video_path": str(visual_path) if visual_path else "",
            "dataset": "longvideobench",
            "subset": "longvideobench_val_v",
            "duration": str(item.get("duration_group") or item.get("duration") or "long"),
            "category": str(item.get("question_category") or item.get("category") or ""),
            "answer": _answer_letter(item, options),
            "options": options,
            "input": _format_prompt(item, options),
        }
        if visual_path:
            records.append(record)
        else:
            missing.append(record)
    return records, missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LongVideoBench val_v JSONL for GraphVID.")
    parser.add_argument("--annotation", default="")
    parser.add_argument("--video_root", default="")
    parser.add_argument("--output", default="assets/longvideobench.jsonl")
    parser.add_argument("--missing_output", default="assets/longvideobench_missing.txt")
    args = parser.parse_args()

    hf_home = Path(os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface")))
    annotation = Path(args.annotation).expanduser() if args.annotation else hf_home / "longvideobench" / "lvb_val.json"
    video_root = Path(args.video_root).expanduser() if args.video_root else hf_home / "longvideobench" / "videos"

    records, missing = build_records(annotation, video_root)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    missing_output = Path(args.missing_output)
    missing_output.parent.mkdir(parents=True, exist_ok=True)
    missing_output.write_text(
        "\n".join(f"{item.get('question_id')}\t{item.get('videoID', '')}\t{item.get('video_path', '')}" for item in missing),
        encoding="utf-8",
    )

    print(f"annotation={annotation}")
    print(f"video_root={video_root}")
    print(f"records={len(records)}")
    print(f"missing={len(missing)}")
    print(f"output={output}")
    print(f"missing_output={missing_output}")


if __name__ == "__main__":
    main()
