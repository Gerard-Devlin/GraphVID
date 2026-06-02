#!/usr/bin/env python3
"""Build EgoSchema JSONL records in the same prompt style used by LMMs-Eval."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


CHOICE_LETTERS = "ABCDE"
LMMS_EVAL_SUFFIX = "\nAnswer with the option's letter from the given choices directly."


def _answer_to_letter(answer: Any) -> str:
    if isinstance(answer, int):
        if 0 <= answer < len(CHOICE_LETTERS):
            return CHOICE_LETTERS[answer]
        return str(answer)
    value = str(answer).strip()
    if value.isdigit():
        idx = int(value)
        if 0 <= idx < len(CHOICE_LETTERS):
            return CHOICE_LETTERS[idx]
    value = value.upper()
    if value in CHOICE_LETTERS:
        return value
    return value


def _format_prompt(doc: dict[str, Any]) -> str:
    question = str(doc["question"]).strip()
    options = doc.get("option") or doc.get("options") or []
    option_lines = [str(option).strip() for option in options if str(option).strip()]
    return "\n".join([question, *option_lines]) + LMMS_EVAL_SUFFIX


def _resolve_video_path(video_root: Path, video_idx: str) -> str:
    candidates = []
    for base in (video_root, video_root / "videos", video_root / "data"):
        for suffix in (".mp4", ".MP4", ".mkv", ".webm"):
            candidates.append(base / f"{video_idx}{suffix}")
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    # Keep a deterministic expected path so missing files fail clearly at run time.
    return str(video_root / "videos" / f"{video_idx}.mp4")


def _load_egoschema(dataset_id: str, config: str, split: str, cache_dir: str | None, local_files_only: bool):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Please install datasets first: pip install datasets") from exc

    download_mode = "reuse_dataset_if_exists" if local_files_only else None
    return load_dataset(
        dataset_id,
        config,
        split=split,
        cache_dir=cache_dir,
        download_mode=download_mode,
    )


def build_records(dataset, video_root: Path, subset_name: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, doc in enumerate(dataset):
        video_idx = str(doc.get("video_idx") or doc.get("videoID") or doc.get("video_id") or doc.get("id") or idx)
        options = [str(option).strip() for option in (doc.get("option") or doc.get("options") or [])]
        records.append(
            {
                "question_id": video_idx,
                "videoID": video_idx,
                "video_path": _resolve_video_path(video_root, video_idx),
                "dataset": "egoschema",
                "subset": subset_name.lower(),
                "split": str(doc.get("split") or "test"),
                "answer": _answer_to_letter(doc.get("answer")),
                "options": options,
                "input": _format_prompt(doc),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build EgoSchema JSONL with LMMs-Eval-compatible prompts.")
    parser.add_argument("--dataset_id", default="lmms-lab/egoschema")
    parser.add_argument("--config", default="Subset", help="Hugging Face dataset config, e.g. Subset or MC.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="assets/egoschema_subset.jsonl")
    parser.add_argument("--video_root", default="")
    parser.add_argument("--cache_dir", default="")
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    hf_home = Path(os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface")))
    video_root = Path(args.video_root).expanduser() if args.video_root else hf_home / "egoschema" / "data"
    dataset = _load_egoschema(
        args.dataset_id,
        args.config,
        args.split,
        args.cache_dir or None,
        args.local_files_only,
    )
    records = build_records(dataset, video_root, args.config)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} records to {output}")


if __name__ == "__main__":
    main()
