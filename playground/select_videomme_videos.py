#!/usr/bin/env python3
"""Select a deterministic subset of unique VideoMME videos."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--ids-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    video_ids = sorted({str(record["videoID"]) for record in records})
    if not 1 <= args.count <= len(video_ids):
        raise ValueError(f"count must be in [1, {len(video_ids)}], got {args.count}")

    selected = set(random.Random(args.seed).sample(video_ids, args.count))
    selected_records = [record for record in records if str(record["videoID"]) in selected]
    selected_ids = sorted(selected)

    args.ids_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.ids_output.write_text("\n".join(selected_ids) + "\n", encoding="utf-8")
    with args.manifest_output.open("w", encoding="utf-8") as handle:
        for record in selected_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    question_count = len(selected_records)
    print(f"selected_videos={len(selected_ids)}")
    print(f"selected_questions={question_count}")
    print(f"seed={args.seed}")
    print(f"ids={args.ids_output}")
    print(f"manifest={args.manifest_output}")


if __name__ == "__main__":
    main()
