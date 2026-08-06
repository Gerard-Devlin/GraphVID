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
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from decord import VideoReader, cpu

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
    parser.add_argument("--answers-json", default="")
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
    parser.add_argument("--device-map", default="cuda:0")
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
        help="Playback rate used only by --timeline-mode sampled."
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
    parser.add_argument(
        "--timeline-mode",
        choices=("full", "sampled"),
        default="full",
        help="full preserves every source-video frame; sampled exports only model inputs.",
    )
    parser.add_argument(
        "--overlay-hold-seconds",
        type=float,
        default=0.6,
        help=(
            "In full mode, show each sampled token map for this duration around "
            "its source timestamp. Other source frames remain unmodified."
        ),
    )
    parser.add_argument(
        "--max-output-width",
        type=int,
        default=960,
        help="Resize wide full-video outputs to this width; use 0 for native resolution.",
    )
    parser.add_argument(
        "--decode-batch-size",
        type=int,
        default=16,
        help="Number of source frames decoded per batch in full mode.",
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
    if args.overlay_hold_seconds < 0.0:
        raise ValueError("overlay-hold-seconds must be non-negative")
    if args.max_output_width < 0:
        raise ValueError("max-output-width must be non-negative")
    if args.decode_batch_size <= 0:
        raise ValueError("decode-batch-size must be positive")
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


def _write_sampled_overlay_video(
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


def _output_size(width: int, height: int, max_width: int) -> tuple[int, int]:
    if max_width > 0 and width > max_width:
        scale = max_width / float(width)
        width = max_width
        height = max(2, round(height * scale))
    width -= width % 2
    height -= height % 2
    if width <= 0 or height <= 0:
        raise RuntimeError(f"invalid video frame size: {(width, height)}")
    return width, height


def _draw_selection_boxes_bgr(
    frame: np.ndarray,
    selected_local: list[int],
    grid_height: int,
    grid_width: int,
) -> np.ndarray:
    selected = sorted(
        {
            int(token)
            for token in selected_local
            if 0 <= int(token) < grid_height * grid_width
        }
    )
    if not selected:
        return frame

    height, width = frame.shape[:2]
    fill = frame.copy()
    boxes: list[tuple[int, int, int, int]] = []
    for token in selected:
        row, col = divmod(token, grid_width)
        x0 = round(col * width / grid_width)
        x1 = round((col + 1) * width / grid_width) - 1
        y0 = round(row * height / grid_height)
        y1 = round((row + 1) * height / grid_height) - 1
        boxes.append((x0, y0, x1, y1))
        cv2.rectangle(fill, (x0, y0), (x1, y1), (186, 175, 0), thickness=-1)

    result = cv2.addWeighted(fill, 0.14, frame, 0.86, 0.0)
    line_width = max(1, round(min(width, height) / 240))
    for x0, y0, x1, y1 in boxes:
        cv2.rectangle(
            result,
            (x0, y0),
            (x1, y1),
            (186, 185, 0),
            thickness=line_width,
        )
    return result


def _write_full_overlay_video(
    example: Example,
    selected_per_frame: list[list[int]],
    output_path: Path,
    hold_seconds: float,
    max_width: int,
    decode_batch_size: int,
    codec: str,
) -> dict[str, Any]:
    if len(example.source_frame_indices) != len(selected_per_frame):
        raise RuntimeError("source index/token-map count mismatch")
    reader = VideoReader(str(example.video_path), ctx=cpu(0))
    total_frames = len(reader)
    if total_frames <= 0:
        raise RuntimeError("source video contains no decodable frames")
    try:
        source_fps = float(reader.get_avg_fps())
    except (AttributeError, TypeError, ValueError, RuntimeError):
        source_fps = 30.0
    if not np.isfinite(source_fps) or source_fps <= 0.0:
        source_fps = 30.0

    first = reader[0].asnumpy()
    output_width, output_height = _output_size(
        int(first.shape[1]),
        int(first.shape[0]),
        max_width,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        source_fps,
        (output_width, output_height),
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"OpenCV could not open MP4 writer for {output_path} with codec {codec!r}"
        )

    sample_indices = np.asarray(example.source_frame_indices, dtype=np.int64)
    hold_radius = max(0, round(source_fps * hold_seconds / 2.0))
    nearest_sample = 0
    boxed_frames = 0
    next_progress = 10
    try:
        for batch_start in range(0, total_frames, decode_batch_size):
            batch_end = min(total_frames, batch_start + decode_batch_size)
            source_batch = reader.get_batch(list(range(batch_start, batch_end))).asnumpy()
            for offset, rgb in enumerate(source_batch):
                source_index = batch_start + offset
                while (
                    nearest_sample + 1 < len(sample_indices)
                    and abs(int(sample_indices[nearest_sample + 1]) - source_index)
                    <= abs(int(sample_indices[nearest_sample]) - source_index)
                ):
                    nearest_sample += 1
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if bgr.shape[1] != output_width or bgr.shape[0] != output_height:
                    bgr = cv2.resize(
                        bgr,
                        (output_width, output_height),
                        interpolation=cv2.INTER_AREA,
                    )
                if (
                    abs(int(sample_indices[nearest_sample]) - source_index) <= hold_radius
                    and selected_per_frame[nearest_sample]
                ):
                    bgr = _draw_selection_boxes_bgr(
                        bgr,
                        selected_per_frame[nearest_sample],
                        example.grid_height,
                        example.grid_width,
                    )
                    boxed_frames += 1
                writer.write(bgr)

            progress = int(100 * batch_end / total_frames)
            if progress >= next_progress:
                print(
                    f"[encode] {output_path.name}: {progress}% "
                    f"({batch_end}/{total_frames})",
                    flush=True,
                )
                next_progress += 10
    finally:
        writer.release()

    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"video encoder produced no data: {output_path}")
    return {
        "source_frames": total_frames,
        "boxed_frames": boxed_frames,
        "unboxed_frames": total_frames - boxed_frames,
        "source_fps": source_fps,
        "output_width": output_width,
        "output_height": output_height,
        "overlay_hold_seconds": hold_seconds,
    }


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
    answers_path = (
        Path(args.answers_json).expanduser().resolve() if args.answers_json.strip() else None
    )
    questions = load_questions(metadata_path, answers_path)
    if not questions:
        raise RuntimeError(
            f"no question records were loaded from metadata file: {metadata_path}"
        )
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
            traceback.print_exc()
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
        render_stats: dict[str, dict[str, Any]] = {}
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
            if args.timeline_mode == "full":
                render_stats[mode] = _write_full_overlay_video(
                    example,
                    selected,
                    output_dir / output_name,
                    args.overlay_hold_seconds,
                    args.max_output_width,
                    args.decode_batch_size,
                    args.codec,
                )
            else:
                _write_sampled_overlay_video(
                    example,
                    selected,
                    output_dir / output_name,
                    args.output_fps,
                    args.frame_repeat,
                    args.codec,
                )
                render_stats[mode] = {
                    "source_frames": len(example.timeline_frames),
                    "boxed_frames": sum(bool(tokens) for tokens in selected),
                    "unboxed_frames": sum(not tokens for tokens in selected),
                    "source_fps": None,
                    "output_width": example.timeline_frames[0].width,
                    "output_height": example.timeline_frames[0].height,
                    "overlay_hold_seconds": None,
                }
            video_files[mode] = output_name

        record = example_metadata(example, number, "", "")
        record.pop("images", None)
        record["videos"] = video_files
        record["video_render"] = render_stats
        record["sampled_timeline"] = _timeline_metadata(example)
        output_records.append(record)

    metadata = {
        "visualization": {
            "overlay_mode": args.overlay_mode,
            "timeline_mode": args.timeline_mode,
            "timeline": (
                "all original source frames; sampled token maps are briefly held "
                "around their true timestamps"
                if args.timeline_mode == "full"
                else "actual model-input sampled frames in temporal order"
            ),
            "interpolated_attention": False,
            "overlay_hold_seconds": (
                args.overlay_hold_seconds if args.timeline_mode == "full" else None
            ),
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
