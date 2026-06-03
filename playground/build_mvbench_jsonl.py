#!/usr/bin/env python3
"""Build MVBench JSONL records from the Hugging Face / ModelScope layout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

DATA_LIST = {
    "object_interaction": "star/Charades_segment",
    "action_sequence": "star/Charades_segment",
    "action_prediction": "star/Charades_segment",
    "action_localization": "sta/sta_video_segment",
    "moving_count": "clevrer/video_validation",
    "fine_grained_pose": "nturgbd_convert",
    "character_order": "perception/videos",
    "object_shuffle": "perception/videos",
    "egocentric_navigation": "vlnqa",
    "moving_direction": "clevrer/video_validation",
    "episodic_reasoning": "tvqa/video_fps3_hq_segment",
    "fine_grained_action": "Moments_in_Time_Raw/videos",
    "scene_transition": "scene_qa/video",
    "state_change": "perception/videos",
    "moving_attribute": "clevrer/video_validation",
    "action_antonym": "ssv2_video_mp4",
    "unexpected_action": "FunQA_test/test",
    "counterfactual_inference": "clevrer/video_validation",
    "object_existence": "clevrer/video_validation",
    "action_count": "perception/videos",
}

PATH_ALIASES = {
    "star/Charades_segment": ["star/Charades_v1_480"],
    "sta/sta_video_segment": ["sta/sta_video"],
    "tvqa/video_fps3_hq_segment": ["tvqa/frames_fps3_hq"],
    "ssv2_video_mp4": ["ssv2_video"],
}


def _load_task_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("data") or data.get("questions") or list(data.values())
    if not isinstance(data, list):
        raise ValueError(f"unsupported MVBench json format: {path}")
    return data


def _answer_letter(item: dict[str, Any]) -> str:
    candidates = [str(x) for x in item.get("candidates", [])]
    answer = item.get("answer")
    if answer in candidates:
        return LETTERS[candidates.index(answer)]
    text = str(answer).strip()
    if text.isdigit():
        idx = int(text)
        if 0 <= idx < len(LETTERS):
            return LETTERS[idx]
    return text.upper()[:1]


def _format_prompt(item: dict[str, Any]) -> str:
    lines = ["Question:" + str(item.get("question", "")).strip(), "Option:"]
    for idx, option in enumerate(item.get("candidates", [])):
        lines.append(f"({LETTERS[idx]}) {option}")
    lines.append("Answer with the option's letter from the given choices directly.")
    return "\n".join(lines)


def _candidate_rel_dirs(rel_dir: str) -> list[str]:
    out = [rel_dir]
    out.extend(PATH_ALIASES.get(rel_dir, []))
    out.extend(f"data0613/{path}" for path in out[:])
    deduped: list[str] = []
    for path in out:
        if path not in deduped:
            deduped.append(path)
    return deduped


def _resolve_visual_path(video_roots: list[Path], rel_dir: str, video_name: str) -> Path | None:
    # First try the official relative path and known layout aliases.
    for root in video_roots:
        for rel in _candidate_rel_dirs(rel_dir):
            candidate = root / rel / video_name
            if candidate.exists():
                return candidate

    # Some local mirrors use the right filename under a different task directory.
    video_basename = Path(video_name).name
    for root in video_roots:
        matches = list(root.rglob(video_basename))
        if matches:
            return matches[0]
    return None


def build_records(json_dir: Path, video_roots: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for task, rel_dir in DATA_LIST.items():
        json_path = json_dir / f"{task}.json"
        if not json_path.exists():
            missing.append({"question_id": f"{task}-missing-json", "video_path": str(json_path)})
            continue
        for idx, item in enumerate(_load_task_json(json_path)):
            video_name = str(item.get("video", "")).strip()
            visual_path = _resolve_visual_path(video_roots, rel_dir, video_name)
            record = {
                "question_id": f"{task}-{idx:04d}",
                "videoID": video_name,
                "video_path": str(visual_path) if visual_path else "",
                "dataset": "mvbench",
                "subset": "mvbench",
                "duration": "medium",
                "category": task,
                "task_category": task,
                "answer": _answer_letter(item),
                "options": [str(x) for x in item.get("candidates", [])],
                "input": _format_prompt(item),
            }
            if visual_path:
                records.append(record)
            else:
                missing.append(record)
    return records, missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MVBench JSONL for the GraphVID benchmark runner.")
    parser.add_argument("--json_dir", default="/root/autodl-tmp/mvbench_raw/json")
    parser.add_argument("--video_root", default="")
    parser.add_argument("--extra_video_root", action="append", default=[])
    parser.add_argument("--output", default="assets/mvbench.jsonl")
    parser.add_argument("--missing_output", default="assets/mvbench_missing.txt")
    args = parser.parse_args()

    hf_home = Path(os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface")))
    roots = []
    if args.video_root:
        roots.append(Path(args.video_root).expanduser())
    else:
        roots.extend(
            [
                hf_home / "mvbench" / "data",
                Path("/root/autodl-tmp/mvbench_video"),
            ]
        )
    roots.extend(Path(path).expanduser() for path in args.extra_video_root)
    roots = [root for root in roots if root.exists()]

    records, missing = build_records(Path(args.json_dir).expanduser(), roots)

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

    print(f"video_roots={','.join(str(root) for root in roots)}")
    print(f"records={len(records)}")
    print(f"missing={len(missing)}")
    print(f"output={output}")
    print(f"missing_output={missing_output}")


if __name__ == "__main__":
    main()
