import argparse
import json
import re
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pct(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value * 100.0 if value <= 1.0 else value


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _phase_name(method: str) -> str:
    return "ours" if method in {"talon"} else method


def _bench_scores(summary_path: Path, method: str) -> dict[str, float | None]:
    summary = _read_json(summary_path)
    phase_key = _phase_name(method)
    phase = summary.get(phase_key, {})
    scores: dict[str, float | None] = {"overall": _pct(phase.get("accuracy"))}
    by_duration = summary.get("duration_breakdown", {})
    for duration in ("short", "medium", "long"):
        bucket = by_duration.get(duration, {})
        scores[duration] = _pct(bucket.get(phase_key, {}).get("accuracy"))
    return scores


def _find_lmms_results(output_dir: Path) -> list[Path]:
    patterns = ("**/*results*.json", "**/results.json")
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(output_dir.glob(pattern))
    return sorted(set(p for p in paths if p.is_file()))


def _extract_lmms_from_results_json(path: Path) -> dict[str, float | None] | None:
    data = _read_json(path)
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, dict):
        return None
    task = results.get("videomme")
    if not isinstance(task, dict):
        return None
    value = None
    for key, candidate in task.items():
        if str(key).startswith("videomme_perception_score"):
            value = candidate
            break
    if value is None:
        return None
    return {"overall": _pct(value), "short": None, "medium": None, "long": None}


def _sample_metric_payload(sample: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("videomme_perception_score", "metrics", "metric_vals"):
        value = sample.get(key)
        if isinstance(value, dict):
            if "duration" in value and "score" in value:
                return value
            nested = value.get("videomme_perception_score")
            if isinstance(nested, dict):
                return nested
    for value in sample.values():
        if isinstance(value, dict) and "duration" in value and "score" in value:
            return value
    return None


def _find_lmms_samples(output_dir: Path) -> list[Path]:
    paths = list(output_dir.glob("**/*samples*.json")) + list(output_dir.glob("**/*eval_samples*.json"))
    return sorted(set(p for p in paths if p.is_file()))


def _extract_lmms_from_samples(output_dir: Path) -> dict[str, float | None] | None:
    samples: list[dict[str, Any]] = []
    for path in _find_lmms_samples(output_dir):
        try:
            data = _read_json(path)
        except Exception:
            continue
        if isinstance(data, list):
            samples.extend(x for x in data if isinstance(x, dict))
        elif isinstance(data, dict):
            for value in data.values():
                if isinstance(value, list):
                    samples.extend(x for x in value if isinstance(x, dict))
    rows = []
    for sample in samples:
        payload = _sample_metric_payload(sample)
        if payload:
            rows.append(payload)
    if not rows:
        return None

    scores: dict[str, float | None] = {}
    total = [float(r.get("score", 0.0)) for r in rows]
    scores["overall"] = (sum(total) / len(total) * 100.0) if total else None
    for duration in ("short", "medium", "long"):
        vals = [float(r.get("score", 0.0)) for r in rows if str(r.get("duration", "")).lower() == duration]
        scores[duration] = (sum(vals) / len(vals) * 100.0) if vals else None
    return scores


def _extract_lmms_from_log(log_path: Path) -> dict[str, float | None] | None:
    if not log_path.exists():
        return None
    scores: dict[str, float | None] = {"overall": None, "short": None, "medium": None, "long": None}
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    for duration in ("short", "medium", "long"):
        match = re.findall(rf"Evaluation on video Type:\s*{duration}:\s*([0-9.]+)%", text, flags=re.IGNORECASE)
        if match:
            scores[duration] = float(match[-1])
    overall = re.findall(r"Overall Performance:\s*([0-9.]+)%", text, flags=re.IGNORECASE)
    if overall:
        scores["overall"] = float(overall[-1])
    if any(value is not None for value in scores.values()):
        return scores
    return None


def _lmms_scores(output_dir: Path, log_path: Path | None = None) -> dict[str, float | None]:
    from_samples = _extract_lmms_from_samples(output_dir)
    if from_samples:
        return from_samples
    for path in _find_lmms_results(output_dir):
        parsed = _extract_lmms_from_results_json(path)
        if parsed:
            return parsed
    if log_path:
        parsed = _extract_lmms_from_log(log_path)
        if parsed:
            return parsed
    return {"overall": None, "short": None, "medium": None, "long": None}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare VideoMME bench_all_metrics and lmms-eval results.")
    parser.add_argument("--bench_summary", required=True)
    parser.add_argument("--lmms_output", required=True)
    parser.add_argument("--lmms_log", default="")
    parser.add_argument("--method", required=True)
    parser.add_argument("--rate", required=True)
    parser.add_argument("--out_md", default="")
    args = parser.parse_args()

    bench = _bench_scores(Path(args.bench_summary), args.method)
    lmms = _lmms_scores(Path(args.lmms_output), Path(args.lmms_log) if args.lmms_log else None)

    lines = [
        "| Method | R | Split | bench_all_metrics | lmms-eval | Delta |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for split in ("overall", "short", "medium", "long"):
        b = bench.get(split)
        l = lmms.get(split)
        delta = None if b is None or l is None else l - b
        lines.append(f"| {args.method} | {args.rate} | {split} | {_fmt(b)} | {_fmt(l)} | {_fmt(delta)} |")
    table = "\n".join(lines) + "\n"
    print(table)
    if args.out_md:
        Path(args.out_md).write_text(table, encoding="utf-8")


if __name__ == "__main__":
    main()
