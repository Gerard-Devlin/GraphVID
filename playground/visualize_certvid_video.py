#!/usr/bin/env python3
"""Render CertVID token maps as videos over the actual model input frames.

The exported MP4 follows the 32 sampled frames consumed by LLaVA-OneVision.
It deliberately does not interpolate attention onto frames the model never saw.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from visualize_certvid_two_examples import (
    Example,
    discover_videos,
    example_metadata,
    load_certvid_model,
    load_questions,
    overlay_selection,
    run_one_example,
)


def parse_args() -> argparse.Namespace:
    hf_home = Path(os.environ.get("HF_HOME", "/home/xuyouwen/hf_home_local"))
    parser = argparse.ArgumentParser(
        description="Export per-frame Vanilla-attention or CertVID token maps as MP4."
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get(
            "PRETRAINED", "/home/xuyouwen/models/llava-onevision-qwen2-7b-ov"
        ),
    )
    parser.add_argument(
        "--dataset-root",
        default=str(hf_home / "videomme" / "data"),
    )
    parser.add_argument("--metadata-jsonl", default="assets/videomme.jsonl")
    parser.add_argument(
        "--video-id",
        default="",
        help="Optional comma-separated video stems/q_uids."
    )
    parser.add_argument("--output-dir", default="logs/visualizations/certvid_video")
    parser.add_argument("--num-examples", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--retention-ratio", type=float, default=0.25)
    parser.add_argument("--expansion", type=float, default=1.30)
    parser.add_argument("--pruning-layer", type=int, default=20)
    parser.add_argument("--llm-retention-ratio", type=float, default=0.1923076923)
    parser.add_argument(
        "--pool-mode", choices=("bilinear", "average", "max"), default="bilinear"
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--selection-mode",
        choices=("improvement", "any"),
        default="any",
    )
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument(
        "--overlay-mode",
        choices=("certvid", "attention", "both"),
        default="certvid",
        help=(
            "certvid writes the selected-anchor map, attention writes matched-budget "
            "global attention Top-K, and both writes two separate MP4 files."
        ),
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        default=4.0,
        help="Playback rate for the sampled-frame timeline."
    )
    parser.add_argument(
        "--frame-repeat",
        type=int,
        default=1,
        help="Repeat each sampled frame this many times in the encoded video."
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="FourCC codec used by OpenCV; mp4v is broadly available."
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> list[str]:
    if args.num_examples <= 0:
        raise ValueError("num-examples must be positive")
    if args.num_frames <= 0:
        raise ValueError("num-frames must be positive")
    if args.output_fps <= 0.0:
        raise ValueError("output-fps must be positive")
    if args.frame_repeat <= 0:
        raise ValueError("frame-repeat must be positive")
    if len(args.codec) != 4:
        raise ValueError("codec must contain exactly four characters")
    if not (0.0 < args.retention_ratio <= 1.0):
        raise ValueError("retention-ratio must be in (0, 1]")
    requested = [
        Path(value.strip()).stem for value in args.video_id.split(",") if value.strip()
    ]
    if requested and len(requested) != args.num_examples:
        raise ValueError(
            "the number of comma-separated --video-id values must match --num-examples"
        )
    if len(requested) != len(set(requested)):
        raise ValueError("--video-id values must be unique")
    return requested


def _write_overlay_video(
    example: Example,
    selected_per_frame: list[list[int]],
    output_path: Path,
    fps: float,
    frame_repeat: int,
    codec: str,
) -> None:
    if len(example.timeline_frames) != len(selected_per_frame):
        raise RuntimeError("timeline frame/token-map count mismatch")
    if not example.timeline_frames:
        raise RuntimeError("cannot encode an empty frame timeline")

    first = example.timeline_frames[0]
    width = int(first.width) - int(first.width) % 2
    height = int(first.height) - int(first.height) % 2
    if width <= 0 or height <= 0:
        raise RuntimeError(f"invalid output frame size: {first.size}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"OpenCV could not open MP4 writer for {output_path} with codec {codec!r}"
        )

    try:
        for frame, selected in zip(example.timeline_frames, selected_per_frame):
            overlay = overlay_selection(
                frame,
                selected,
                example.grid_height,
                example.grid_width,
            )
            rgb = np.asarray(overlay.convert("RGB"), dtype=np.uint8)
            if rgb.shape[1] != width or rgb.shape[0] != height:
                rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for _ in range(frame_repeat):
                writer.write(bgr)
    finally:
        writer.release()

    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"video encoder produced no data: {output_path}")


def _timeline_metadata(example: Example) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sampled_index, (source_index, attention, certvid) in enumerate(
        zip(
            example.source_frame_indices,
            example.attention_per_frame,
            example.certvid_per_frame,
        )
    ):
        timestamp = (
            float(source_index) / example.video_fps
            if example.video_fps is not None
            else None
        )
        records.append(
            {
                "sampled_index": sampled_index,
                "source_frame_index": source_index,
                "timestamp_seconds": timestamp,
                "attention_topk_tokens": len(attention),
                "certvid_tokens": len(certvid),
                "overlap_tokens": len(set(attention) & set(certvid)),
            }
        )
    return records


def main() -> None:
    args = parse_args()
    requested_video_ids = _validate_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    metadata_path = Path(args.metadata_jsonl).expanduser().resolve()
    questions = load_questions(metadata_path)
    if not questions:
        raise RuntimeError("metadata JSONL is required")
    candidates = discover_videos(dataset_root)
    rng = random.Random(args.seed)
    if requested_video_ids:
        paths_by_id = {path.stem: path for path in candidates}
        missing = [video_id for video_id in requested_video_ids if video_id not in paths_by_id]
        if missing:
            raise FileNotFoundError(f"video IDs not found under {dataset_root}: {missing}")
        candidates = [paths_by_id[video_id] for video_id in requested_video_ids]
    else:
        rng.shuffle(candidates)

    print(f"[setup] dataset={dataset_root}", flush=True)
    print(f"[setup] output={output_dir}", flush=True)
    print(f"[setup] overlay_mode={args.overlay_mode}", flush=True)
    tokenizer, model, image_processor, device = load_certvid_model(args)

    examples: list[Example] = []
    attempt_limit = min(len(candidates), max(args.max_attempts, args.num_examples))
    for attempt, path in enumerate(candidates[:attempt_limit], start=1):
        print(
            f"[try] {attempt}/{attempt_limit} video={path.name} "
            f"accepted={len(examples)}/{args.num_examples}",
            flush=True,
        )
        video_questions = questions.get(path.stem)
        if not video_questions:
            print(f"[skip] {path.name}: no matching metadata", flush=True)
            continue
        try:
            example = run_one_example(
                path,
                rng.choice(video_questions),
                tokenizer,
                model,
                image_processor,
                device,
                args,
            )
        except Exception as exc:
            print(f"[skip] {path.name}: {exc}", flush=True)
            continue
        if args.selection_mode == "improvement" and not (
            example.without_certvid_correct is False and example.certvid_correct is True
        ):
            print(
                f"[reject] {path.name}: full={example.without_certvid_answer}/"
                f"{example.without_certvid_correct} certvid={example.certvid_answer}/"
                f"{example.certvid_correct}",
                flush=True,
            )
            continue
        examples.append(example)
        if len(examples) >= args.num_examples:
            break

    output_records: list[dict[str, Any]] = []
    for number, example in enumerate(examples, start=1):
        video_files: dict[str, str] = {}
        modes = (
            ("attention", "certvid") if args.overlay_mode == "both" else (args.overlay_mode,)
        )
        for mode in modes:
            selected = (
                example.certvid_per_frame
                if mode == "certvid"
                else example.attention_per_frame
            )
            output_name = f"example_{number:02d}_{example.video_path.stem}_{mode}.mp4"
            print(f"[encode] {output_name}", flush=True)
            _write_overlay_video(
                example,
                selected,
                output_dir / output_name,
                args.output_fps,
                args.frame_repeat,
                args.codec,
            )
            video_files[mode] = output_name

        record = example_metadata(example, number, "", "")
        record.pop("images", None)
        record["videos"] = video_files
        record["sampled_timeline"] = _timeline_metadata(example)
        output_records.append(record)

    metadata = {
        "visualization": {
            "overlay_mode": args.overlay_mode,
            "timeline": "actual model-input sampled frames in temporal order",
            "interpolated_attention": False,
            "audio_preserved": False,
            "sampled_frames": args.num_frames,
            "output_fps": args.output_fps,
            "frame_repeat": args.frame_repeat,
            "retention_ratio": args.retention_ratio,
            "selection_mode": args.selection_mode,
            "requested_video_ids": requested_video_ids or None,
            "requested_examples": args.num_examples,
            "produced_examples": len(examples),
        },
        "examples": output_records,
    }
    (output_dir / "examples.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[done] generated {sum(len(item['videos']) for item in output_records)} "
        f"MP4 file(s) and examples.json in {output_dir}",
        flush=True,
    )
    if len(examples) < args.num_examples:
        raise RuntimeError(
            f"only produced {len(examples)}/{args.num_examples} examples; "
            "completed outputs were retained"
        )


if __name__ == "__main__":
    main()
