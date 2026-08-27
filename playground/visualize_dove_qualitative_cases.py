#!/usr/bin/env python3
"""Export VideoMME filmstrips and a LaTeX qualitative-comparison fragment.

The input is the strict-win CSV produced by ``find_videomme_win_cases.py``.
Python only renders uniformly sampled filmstrips. Questions, options, method
predictions, and correctness marks are emitted as LaTeX so their typography is
controlled by the paper template. No model inference is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageOps


VIDEO_SUFFIXES = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
OPTION_RE = re.compile(r"^\s*([A-E])\s*[.)]\s*(.*?)\s*$")
DEFAULT_QUESTION_IDS = ("052-3", "116-3", "141-2", "208-3", "264-2", "460-3")

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
    parser.add_argument("--filmstrip-frames", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--tex-output",
        default="",
        help="LaTeX fragment path. Defaults to OUTPUT_DIR/dove_qualitative_cases.tex.",
    )
    parser.add_argument(
        "--tex-image-prefix",
        default="figures/appendix/dove_cases",
        help="Image path prefix written into the LaTeX fragment.",
    )
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


def make_filmstrip(frames: Iterable[Image.Image]) -> Image.Image:
    """Use the same filmstrip geometry as the other paper visualizations."""
    frames = list(frames)
    if not frames:
        raise ValueError("filmstrip requires at least one frame")

    tile_width, tile_height = 360, 210
    gap, rail = 10, 36
    side_margin = 12
    width = side_margin * 2 + len(frames) * tile_width + (len(frames) - 1) * gap
    height = tile_height + rail * 2
    strip = Image.new("RGB", (width, height), "black")

    for position, frame in enumerate(frames):
        x0 = side_margin + position * (tile_width + gap)
        tile = ImageOps.fit(
            frame.convert("RGB"),
            (tile_width, tile_height),
            method=Image.Resampling.LANCZOS,
        )
        strip.paste(tile, (x0, rail))

    draw = ImageDraw.Draw(strip)
    hole_width, hole_height, hole_gap = 20, 14, 12
    x = 8
    while x + hole_width < width:
        draw.rectangle((x, 8, x + hole_width, 8 + hole_height), fill="white")
        draw.rectangle(
            (x, height - 8 - hole_height, x + hole_width, height - 8),
            fill="white",
        )
        x += hole_width + hole_gap
    return strip


def _tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _tex_prediction(case: QualitativeCase, method: str) -> str:
    prediction = _tex_escape(case.predictions[method])
    macro = "DOVECorrectPrediction" if case.predictions[method] == case.answer else "DOVEWrongPrediction"
    return rf"\{macro}{{{prediction}}}"


def _tex_case(case: QualitativeCase, image_prefix: str) -> str:
    image_name = f"dove_filmstrip_{case.question_id.replace('/', '_')}.png"
    image_path = f"{image_prefix.rstrip('/')}/{image_name}" if image_prefix else image_name
    options: list[str] = []
    for index, (label, text) in enumerate(case.options.items()):
        punctuation = "," if index < len(case.options) - 1 else ""
        option = f"``{_tex_escape(label)}. {_tex_escape(text)}''{punctuation}"
        if label == case.answer:
            option = rf"\DOVECorrectOption{{{option}}}"
        options.append(option)
    option_lines = " \\\\[0.18em]\n".join(options)

    return rf"""\noindent
\begin{{minipage}}{{\linewidth}}
    \centering
    \includegraphics[width=\linewidth]{{{image_path}}}
    \par\vspace{{0.10em}}
    \raggedright
    {{\large\bfseries\itshape Question:}}
    {{\large\itshape {_tex_escape(case.question)}}}
    \par\vspace{{0.45em}}

    \noindent
    \begin{{minipage}}[t]{{0.505\linewidth}}
    \vspace{{0pt}}\raggedright
    {option_lines}
    \end{{minipage}}%
    \hfill
    \begin{{minipage}}[t]{{0.012\linewidth}}
    \vspace{{0pt}}\centering\rule{{0.45pt}}{{7.2em}}
    \end{{minipage}}%
    \hfill
    \begin{{minipage}}[t]{{0.445\linewidth}}
    \vspace{{0pt}}\centering
    \begin{{tabular*}}{{\linewidth}}[t]{{@{{\extracolsep{{\fill}}}}ccc@{{}}}}
    FastV & VisionZip & FastVID \\
    {_tex_prediction(case, 'FastV')} & {_tex_prediction(case, 'VisionZip')} & {_tex_prediction(case, 'FastVID')} \\[0.30em]
    FlashVID & \textbf{{DOVE}} & \\
    {_tex_prediction(case, 'FlashVID')} & {_tex_prediction(case, 'DOVE')} &
    \end{{tabular*}}
    \end{{minipage}}
\end{{minipage}}"""


def write_tex_fragment(
    cases: list[QualitativeCase], output_path: Path, image_prefix: str
) -> None:
    header = r"""% Auto-generated by playground/visualize_dove_qualitative_cases.py.
% Required packages: graphicx, xcolor, amssymb.
\definecolor{doveWrong}{HTML}{C90000}
\definecolor{doveCorrect}{HTML}{00A53C}
\newcommand{\DOVEWrongPrediction}[1]{{\color{doveWrong}\emph{``#1..''}\,\raisebox{-0.15ex}{\scalebox{1.55}{$\times$}}}}
\newcommand{\DOVECorrectPrediction}[1]{{\color{doveCorrect}\emph{``#1..''}\,\raisebox{-0.10ex}{\scalebox{1.45}{$\checkmark$}}}}
\newcommand{\DOVECorrectOption}[1]{{\color{doveCorrect}\bfseries #1}}
"""
    blocks: list[str] = [header.rstrip()]
    for page_index, start in enumerate(range(0, len(cases), 3), start=1):
        page_cases = cases[start : start + 3]
        body = "\n\n\\vspace{0.75em}\n\n".join(
            _tex_case(case, image_prefix) for case in page_cases
        )
        blocks.append(
            rf"""\begin{{figure*}}[p]
    \centering
{body}
    \caption{{\textbf{{Qualitative comparison on VideoMME at 1\% retention.}}
    DOVE answers each question correctly, while all four competing video-token
    compression methods produce incorrect predictions.}}
    \label{{fig:dove_qualitative_{page_index}}}
\end{{figure*}}"""
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.num_examples <= 0:
        raise ValueError("--num-examples must be positive")
    if args.filmstrip_frames <= 1:
        raise ValueError("--filmstrip-frames must be greater than one")

    all_cases = read_cases(Path(args.cases_csv).expanduser().resolve())
    cases = choose_cases(all_cases, args.question_ids, args.num_examples)
    video_map = discover_video_map(Path(args.video_root).expanduser().resolve())
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_output = (
        Path(args.tex_output).expanduser().resolve()
        if args.tex_output
        else output_dir / "dove_qualitative_cases.tex"
    )

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
        strip = make_filmstrip(frames)
        safe_id = case.question_id.replace("/", "_")
        png_path = output_dir / f"dove_filmstrip_{safe_id}.png"
        strip.save(
            png_path,
            format="PNG",
            optimize=True,
            dpi=(float(args.dpi), float(args.dpi)),
        )
        metadata.append(
            {
                **asdict(case),
                "video_path": str(video_path),
                "filmstrip_png": str(png_path),
            }
        )

    write_tex_fragment(cases, tex_output, args.tex_image_prefix)

    (output_dir / "selected_cases.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(cases)} filmstrip PNG files to {output_dir}")
    print(f"LaTeX fragment: {tex_output}")


if __name__ == "__main__":
    main()
