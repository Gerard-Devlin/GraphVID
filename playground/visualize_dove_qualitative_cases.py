#!/usr/bin/env python3
"""Render VideoMME qualitative wins as publication-ready filmstrip panels.

The input is the strict-win CSV produced by ``find_videomme_win_cases.py``.
Each example contains a uniformly sampled video filmstrip, the multiple-choice
question, and the predictions from four baselines and DOVE. The script exports
one six-example figure and one standalone figure per example as both PDF and
PNG. No model inference is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageOps


VIDEO_SUFFIXES = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
OPTION_RE = re.compile(r"^\s*([A-E])\s*[.)]\s*(.*?)\s*$")
DEFAULT_QUESTION_IDS = ("052-3", "116-3", "141-2", "208-3", "264-2", "460-3")

INK = "#1E242B"
CORRECT = "#3F7D55"
WRONG = "#B74848"
DOVE = "#BD4444"


@dataclass(frozen=True)
class QualitativeCase:
    question_id: str
    video_id: str
    answer: str
    question: str
    options: dict[str, str]
    predictions: dict[str, str]
    duration: str
    category: str
    task_category: str


def parse_args() -> argparse.Namespace:
    hf_home = Path(os.environ.get("HF_HOME", "/home/xuyouwen/hf_home_local"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-csv", required=True, help="strict_win_cases.csv")
    parser.add_argument(
        "--video-root",
        default=str(hf_home / "videomme" / "data"),
        help="Directory recursively containing VideoMME videos.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--question-ids",
        default=",".join(DEFAULT_QUESTION_IDS),
        help="Ordered comma-separated question IDs. Empty means the first N rows.",
    )
    parser.add_argument("--num-examples", type=int, default=6)
    parser.add_argument("--filmstrip-frames", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _clean_prompt_line(line: str) -> str:
    return " ".join(line.strip().split())


def parse_prompt(prompt: str) -> tuple[str, dict[str, str]]:
    """Extract the question and labeled options from an lmms-eval prompt."""
    prompt = prompt or ""
    lines = [_clean_prompt_line(line) for line in prompt.splitlines() if line.strip()]
    content: list[str] = []
    options: dict[str, str] = {}
    current_label: str | None = None

    for line in lines:
        lower = line.lower()
        if lower.startswith("select the best answer"):
            continue
        if lower.startswith("respond with only"):
            continue
        if lower.startswith("answer with the option"):
            continue

        match = OPTION_RE.match(line)
        if match:
            current_label = match.group(1).upper()
            options[current_label] = match.group(2).strip()
        elif current_label is not None:
            # VideoMME sometimes wraps a long option onto the next CSV line.
            options[current_label] = f"{options[current_label]} {line}".strip()
        else:
            content.append(line)

    question = " ".join(content).strip()
    if not question:
        question = "Video question"
    return question, options


def _answer_letter(value: str | None) -> str:
    match = re.search(r"[A-E]", str(value or "").upper())
    return match.group(0) if match else "?"


def _case_from_row(row: dict[str, str | None]) -> QualitativeCase | None:
    question_id = str(row.get("question_id") or "").strip()
    video_id = str(row.get("videoID") or "").strip()
    prompt = str(row.get("input") or "").strip()
    if not question_id or not video_id or not prompt:
        return None
    question, options = parse_prompt(prompt)
    return QualitativeCase(
        question_id=question_id,
        video_id=video_id,
        answer=_answer_letter(row.get("answer")),
        question=question,
        options=options,
        predictions={
            "FastV": _answer_letter(row.get("pred_fastv")),
            "VisionZip": _answer_letter(row.get("pred_visionzip")),
            "FastVID": _answer_letter(row.get("pred_fastvid")),
            "FlashVID": _answer_letter(row.get("pred_flashvid")),
            "DOVE": _answer_letter(row.get("pred_certvidfinal2")),
        },
        duration=str(row.get("duration") or "").strip(),
        category=str(row.get("category") or "").strip(),
        task_category=str(row.get("task_category") or "").strip(),
    )


def _read_loose_multiline_csv(path: Path) -> list[dict[str, str]]:
    """Recover pasted CSVs whose multiline prompt lost its CSV quoting."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if not lines:
        return []
    fieldnames = next(csv.reader([lines[0]]))
    fixed_count = len(fieldnames) - 1
    records: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    prompt_lines: list[str] = []

    def finish() -> None:
        if current is None:
            return
        prompt = "\n".join(prompt_lines).strip()
        if prompt.startswith('"'):
            prompt = prompt[1:]
        if prompt.endswith('"'):
            prompt = prompt[:-1]
        current["input"] = prompt.replace('""', '"')
        records.append(current)

    for line in lines[1:]:
        if re.match(r"^\d{3}-\d+,", line):
            finish()
            values = next(csv.reader([line]))
            if len(values) < fixed_count:
                current = None
                prompt_lines = []
                continue
            current = dict(zip(fieldnames[:fixed_count], values[:fixed_count]))
            prompt_lines = [",".join(values[fixed_count:])]
        elif current is not None:
            prompt_lines.append(line)
    finish()
    return records


def read_cases(path: Path) -> list[QualitativeCase]:
    if not path.is_file():
        raise FileNotFoundError(f"cases CSV does not exist: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    cases = [case for row in rows if (case := _case_from_row(row)) is not None]

    # Normal files written by csv.DictWriter take this path. The fallback also
    # accepts CSV text copied through tools that strip multiline field quotes.
    if not cases or not any(case.options for case in cases):
        cases = [
            case
            for row in _read_loose_multiline_csv(path)
            if (case := _case_from_row(row)) is not None
        ]
    if not cases:
        raise ValueError(f"no rows found in {path}")
    return cases


def choose_cases(
    cases: list[QualitativeCase], question_ids: str, num_examples: int
) -> list[QualitativeCase]:
    requested = [value.strip() for value in question_ids.split(",") if value.strip()]
    if not requested:
        return cases[:num_examples]

    by_id = {case.question_id: case for case in cases}
    missing = [question_id for question_id in requested if question_id not in by_id]
    if missing:
        available = ", ".join(case.question_id for case in cases)
        raise KeyError(f"question IDs not found: {missing}. Available: {available}")
    return [by_id[question_id] for question_id in requested[:num_examples]]


def discover_video_map(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"video root does not exist: {root}")
    result: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            result.setdefault(path.stem, path)
    if not result:
        raise FileNotFoundError(f"no video files found under: {root}")
    return result


def sample_frames(video_path: Path, count: int) -> list[Image.Image]:
    try:
        from decord import VideoReader, cpu
    except ImportError as error:
        raise RuntimeError(
            "decord is required to read videos; install it in the experiment environment"
        ) from error
    reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    if len(reader) == 0:
        raise ValueError(f"video contains no frames: {video_path}")
    indices = np.linspace(0, len(reader) - 1, count).round().astype(np.int64)
    batch = reader.get_batch(indices.tolist()).asnumpy()
    return [Image.fromarray(frame).convert("RGB") for frame in batch]


def _fit_frame(frame: Image.Image, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "black")
    fitted = ImageOps.contain(frame, size, Image.Resampling.LANCZOS)
    x = (size[0] - fitted.width) // 2
    y = (size[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def make_filmstrip(
    frames: Iterable[Image.Image], width: int = 2400, height: int = 310
) -> Image.Image:
    frames = list(frames)
    if not frames:
        raise ValueError("filmstrip requires at least one frame")

    outer = 8
    band = 31
    gap = 5
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width - 1, height - 1), fill="black", outline=INK, width=3)

    content_y = band + outer
    content_h = height - 2 * (band + outer)
    content_x = outer
    content_w = width - 2 * outer
    cell_w = (content_w - gap * (len(frames) - 1)) // len(frames)

    for index, frame in enumerate(frames):
        x = content_x + index * (cell_w + gap)
        fitted = _fit_frame(frame, (cell_w, content_h))
        image.paste(fitted, (x, content_y))
        draw.rectangle(
            (x, content_y, x + cell_w - 1, content_y + content_h - 1),
            outline="#DCE3E8",
            width=2,
        )

    hole_w = 22
    hole_h = 14
    hole_gap = 14
    holes = max(1, (width - 2 * outer + hole_gap) // (hole_w + hole_gap))
    used = holes * hole_w + (holes - 1) * hole_gap
    start_x = (width - used) // 2
    for index in range(holes):
        x = start_x + index * (hole_w + hole_gap)
        for y in (8, height - 8 - hole_h):
            draw.rounded_rectangle(
                (x, y, x + hole_w, y + hole_h), radius=3, fill="white"
            )
    return image


def _wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def _draw_question(ax: plt.Axes, case: QualitativeCase) -> None:
    ax.text(
        0.012,
        0.93,
        "Question:",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12.0,
        fontweight="bold",
        fontstyle="italic",
        color=INK,
    )
    ax.text(
        0.090,
        0.93,
        _wrap(case.question, 105),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.8,
        fontstyle="italic",
        color=INK,
        linespacing=1.12,
    )

    option_items = list(case.options.items())
    y_positions = (0.59, 0.41, 0.23, 0.05)
    for index, (label, text) in enumerate(option_items[:4]):
        y = y_positions[index]
        color = CORRECT if label == case.answer else INK
        weight = "bold" if label == case.answer else "normal"
        ax.text(
            0.020,
            y,
            _wrap(f'"{label}. {text}"', 67),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10.5,
            fontweight=weight,
            color=color,
            linespacing=1.06,
        )

    ax.plot(
        [0.545, 0.545],
        [0.015, 0.66],
        transform=ax.transAxes,
        color=INK,
        linewidth=0.9,
        clip_on=False,
    )


def _draw_answer_cards(ax: plt.Axes, case: QualitativeCase) -> None:
    placements = {
        "FastV": (0.625, 0.54),
        "VisionZip": (0.785, 0.54),
        "FastVID": (0.935, 0.54),
        "FlashVID": (0.700, 0.20),
        "DOVE": (0.865, 0.20),
    }
    for method, (x, y) in placements.items():
        prediction = case.predictions[method]
        correct = prediction == case.answer
        color = CORRECT if correct else WRONG
        ax.text(
            x,
            y,
            method,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=11.1,
            fontweight="bold",
            color=DOVE if method == "DOVE" else INK,
        )
        ax.text(
            x,
            y - 0.035,
            f'"{prediction}."',
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12.0,
            fontweight="bold",
            color=color,
        )


def draw_info_panel(ax: plt.Axes, case: QualitativeCase) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _draw_question(ax, case)
    _draw_answer_cards(ax, case)


def draw_filmstrip_axis(ax: plt.Axes, filmstrip: Image.Image) -> None:
    ax.imshow(filmstrip)
    ax.set_aspect("auto")
    ax.axis("off")


def render_figure(
    cases: list[QualitativeCase],
    filmstrips: dict[str, Image.Image],
    output_stem: Path,
    dpi: int,
) -> None:
    count = len(cases)
    figure = plt.figure(figsize=(13.9, 3.02 * count), facecolor="white")
    grid = figure.add_gridspec(
        2 * count,
        1,
        height_ratios=[0.72, 1.08] * count,
        hspace=0.035,
        left=0.025,
        right=0.975,
        top=0.992,
        bottom=0.012,
    )

    for index, case in enumerate(cases):
        filmstrip_ax = figure.add_subplot(grid[2 * index])
        panel_ax = figure.add_subplot(grid[2 * index + 1])
        draw_filmstrip_axis(filmstrip_ax, filmstrips[case.question_id])
        draw_info_panel(panel_ax, case)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_stem.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.035,
        facecolor="white",
    )
    figure.savefig(
        output_stem.with_suffix(".png"),
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.035,
        facecolor="white",
    )
    plt.close(figure)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def main() -> None:
    args = parse_args()
    if args.num_examples <= 0:
        raise ValueError("--num-examples must be positive")
    if args.filmstrip_frames <= 1:
        raise ValueError("--filmstrip-frames must be greater than one")

    configure_matplotlib()
    all_cases = read_cases(Path(args.cases_csv).expanduser().resolve())
    cases = choose_cases(all_cases, args.question_ids, args.num_examples)
    video_map = discover_video_map(Path(args.video_root).expanduser().resolve())
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    filmstrips: dict[str, Image.Image] = {}
    metadata: list[dict[str, object]] = []
    for case in cases:
        video_path = video_map.get(case.video_id)
        if video_path is None:
            raise FileNotFoundError(
                f"video {case.video_id!r} for question {case.question_id!r} "
                f"was not found under {args.video_root}"
            )
        print(f"[{case.question_id}] sampling {video_path}", flush=True)
        frames = sample_frames(video_path, args.filmstrip_frames)
        filmstrips[case.question_id] = make_filmstrip(frames)
        metadata.append({**asdict(case), "video_path": str(video_path)})

    render_figure(
        cases,
        filmstrips,
        output_dir / "dove_qualitative_comparison",
        args.dpi,
    )
    for case in cases:
        render_figure(
            [case],
            filmstrips,
            output_dir / f"dove_case_{case.question_id.replace('/', '_')}",
            args.dpi,
        )

    (output_dir / "selected_cases.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote combined PDF/PNG and {len(cases)} standalone pairs to {output_dir}")


if __name__ == "__main__":
    main()
