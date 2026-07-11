from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

FIELDNAMES = [
    "method",
    "retention_ratio",
    "videomme_short",
    "videomme_medium",
    "videomme_long",
    "videomme_overall",
    "egoschema_subset",
    "egoschema_total",
    "longvideobench",
    "mvbench",
    "avg_score",
    "rel_acc",
    "mean_tokens",
]

DATASET_TO_COLUMN = {
    "videomme": "videomme_overall",
    "egoschema_subset": "egoschema_subset",
    "egoschema_total": "egoschema_total",
    "longvideobench": "longvideobench",
    "mvbench": "mvbench",
}


def _as_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _format_pct(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.2f}"


def _format_num(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _stat_mean(phase: dict[str, Any] | None, key: str) -> float | None:
    if not phase:
        return None
    value = phase.get(key)
    if isinstance(value, dict):
        mean = value.get("mean")
        return None if mean is None else float(mean)
    return None


def _safe_pct(value: Any) -> float | None:
    return None if value is None else float(value) * 100.0


def _row_average(row: dict[str, Any]) -> float | None:
    keys = [
        "videomme_overall",
        "egoschema_subset",
        "egoschema_total",
        "longvideobench",
        "mvbench",
    ]
    return _mean([row.get(k) for k in keys if row.get(k) is not None])


def _normalize_rate(rate: str) -> str:
    text = str(rate).strip()
    if not text:
        return text
    if text.endswith("%"):
        return text
    value = float(text)
    if value <= 1.0:
        value *= 100.0
    return f"{value:g}%"


def _parse_kv_spec(spec: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"expected key=value in --parallel_summary item, got {part!r}")
        key, value = part.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _load_matrix_rows(patterns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pattern in patterns:
        matches = sorted(glob.glob(str(_as_path(pattern))))
        if not matches and _as_path(pattern).exists():
            matches = [str(_as_path(pattern))]
        for path_text in matches:
            path = Path(path_text)
            data = _load_json(path)
            if not isinstance(data, list):
                raise ValueError(f"matrix json must be a row list: {path}")
            for row in data:
                rows.append({k: row.get(k) for k in FIELDNAMES})
    return rows


def _row_from_parallel_summary(spec: str) -> dict[str, Any]:
    meta = _parse_kv_spec(spec)
    method = meta.get("method")
    rate = meta.get("rate")
    dataset = meta.get("dataset", "videomme")
    path_text = meta.get("path")
    if not method or not rate or not path_text:
        raise ValueError(
            "--parallel_summary requires method=...,rate=...,path=... "
            "and optional dataset=..."
        )

    summary = _load_json(_as_path(path_text))
    phase = summary.get(method)
    if phase is None and method == "talon":
        phase = summary.get("ours")
    if phase is None:
        raise KeyError(f"method {method!r} not found in {path_text}")

    row: dict[str, Any] = {k: None for k in FIELDNAMES}
    row["method"] = method
    row["retention_ratio"] = _normalize_rate(rate)
    row["mean_tokens"] = _stat_mean(phase, "compressed_visual_tokens")

    dataset_col = DATASET_TO_COLUMN.get(dataset)
    if dataset_col:
        row[dataset_col] = _safe_pct(phase.get("accuracy"))
    if dataset == "videomme":
        by_duration = summary.get("duration_breakdown", {})
        for duration, key in (
            ("short", "videomme_short"),
            ("medium", "videomme_medium"),
            ("long", "videomme_long"),
        ):
            sub_phase = by_duration.get(duration, {}).get(method)
            if sub_phase is None and method == "talon":
                sub_phase = by_duration.get(duration, {}).get("ours")
            if sub_phase is not None:
                row[key] = _safe_pct(sub_phase.get("accuracy"))
    row["avg_score"] = _row_average(row)
    return row


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Later inputs win so users can override stale rows intentionally.
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        method = str(row.get("method") or "").strip().lower()
        rate = _normalize_rate(str(row.get("retention_ratio") or ""))
        if not method or not rate:
            continue
        normalized = {k: row.get(k) for k in FIELDNAMES}
        normalized["method"] = method
        normalized["retention_ratio"] = rate
        if normalized.get("avg_score") is None:
            normalized["avg_score"] = _row_average(normalized)
        merged[(method, rate)] = normalized
    return sorted(merged.values(), key=_sort_key)


def _sort_key(row: dict[str, Any]) -> tuple[float, int, str]:
    order = {
        "flashvid": 0,
        "graphvid": 1,
        "fastvid": 2,
        "fastgraphvid": 3,
        "certvid": 4,
        "visionzip": 4,
        "prunevid": 5,
        "talon": 6,
        "ours": 6,
    }
    rate_text = str(row.get("retention_ratio") or "0").replace("%", "")
    try:
        rate = float(rate_text)
    except ValueError:
        rate = 0.0
    method = str(row.get("method") or "").lower()
    return (rate, order.get(method, 99), method)


def _write_tables(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "qwen3_comparison_table.csv"
    md_path = out_dir / "qwen3_comparison_table.md"
    json_path = out_dir / "qwen3_comparison_summary.json"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in FIELDNAMES})

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    lines = [
        "| Method | Retention Ratio R | VideoMME Short | VideoMME Medium | VideoMME Long | VideoMME Overall | EgoSchema Subset | EgoSchema Total | LongVideoBench | MVBench | Avg. Score | Rel. Acc (%) | Tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {rate} | {short} | {medium} | {long} | {overall} | {ego_s} | {ego_t} | {lvb} | {mvb} | {avg} | {rel} | {tokens} |".format(
                method=row.get("method", "-"),
                rate=row.get("retention_ratio", "-"),
                short=_format_pct(row.get("videomme_short")),
                medium=_format_pct(row.get("videomme_medium")),
                long=_format_pct(row.get("videomme_long")),
                overall=_format_pct(row.get("videomme_overall")),
                ego_s=_format_pct(row.get("egoschema_subset")),
                ego_t=_format_pct(row.get("egoschema_total")),
                lvb=_format_pct(row.get("longvideobench")),
                mvb=_format_pct(row.get("mvbench")),
                avg=_format_pct(row.get("avg_score")),
                rel=_format_pct(row.get("rel_acc")),
                tokens=_format_num(row.get("mean_tokens")),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[table] csv={csv_path}")
    print(f"[table] md={md_path}")
    print(f"[table] json={json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge GraphVID/FlashVID matrix rows with Qwen3 baseline-adapter rows "
            "without touching any compression implementation."
        )
    )
    parser.add_argument(
        "--matrix_json",
        action="append",
        default=[],
        help="Existing qwen3_matrix_summary.json path or glob. May be repeated.",
    )
    parser.add_argument(
        "--parallel_summary",
        action="append",
        default=[],
        help=(
            "Add one row from a parallel summary: "
            "method=fastvid,rate=10,dataset=videomme,path=logs/..._summary.json"
        ),
    )
    parser.add_argument(
        "--out_dir",
        default="logs/efficiency/matrix/qwen3_comparison_merged",
        help="Output directory for merged csv/md/json tables.",
    )
    args = parser.parse_args()

    rows = _load_matrix_rows(args.matrix_json)
    rows.extend(_row_from_parallel_summary(item) for item in args.parallel_summary)
    rows = _dedupe_rows(rows)
    _write_tables(_as_path(args.out_dir), rows)


if __name__ == "__main__":
    main()
