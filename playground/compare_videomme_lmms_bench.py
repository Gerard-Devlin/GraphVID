import argparse
import json
import re
import sys
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
    return "ours" if method in {"talon", "apexvid", "certvid", "certvid_v2"} else method


def _bench_record_phase_name(method: str) -> str:
    return "ours" if method in {"fastvid", "visionzip", "fastgraphvid", "curvevid", "talon", "apexvid", "certvid", "certvid_v2"} else method


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
    paths = (
        list(output_dir.glob("**/*samples*.json"))
        + list(output_dir.glob("**/*samples*.jsonl"))
        + list(output_dir.glob("**/*eval_samples*.json"))
        + list(output_dir.glob("**/*eval_samples*.jsonl"))
    )
    return sorted(set(p for p in paths if p.is_file()))


def _find_bench_jsonl(summary_path: Path, method: str) -> Path | None:
    phase = _bench_record_phase_name(method)
    candidates = [
        summary_path.with_name(summary_path.name.replace("_summary.json", f"_{phase}.jsonl")),
        summary_path.with_name(summary_path.name.replace("_summary.json", f"_{method}.jsonl")),
    ]
    if method in {"fastvid", "visionzip", "fastgraphvid", "curvevid", "apexvid", "certvid", "certvid_v2"}:
        candidates.append(summary_path.with_name(summary_path.name.replace("_summary.json", "_ours.jsonl")))
    for path in candidates:
        if path.exists():
            return path
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
    return rows


def _extract_doc_from_sample(sample: dict[str, Any]) -> dict[str, Any]:
    for key in ("doc", "doc_hash", "arguments", "metadata"):
        value = sample.get(key)
        if isinstance(value, dict):
            if "question_id" in value or "videoID" in value:
                return value
    return sample


def _sample_question_id(sample: dict[str, Any]) -> str:
    payload = _sample_metric_payload(sample)
    if payload and payload.get("question_id") is not None:
        return str(payload.get("question_id"))
    doc = _extract_doc_from_sample(sample)
    for container in (sample, doc):
        for key in ("question_id", "questionId", "id"):
            value = container.get(key) if isinstance(container, dict) else None
            if value is not None:
                return str(value)
    return ""


def _sample_prediction(sample: dict[str, Any]) -> str:
    payload = _sample_metric_payload(sample)
    if payload and payload.get("pred_answer") is not None:
        return str(payload.get("pred_answer"))
    keys = ("pred_answer", "filtered_resps", "resps", "response", "prediction", "pred", "exact_match")
    for key in keys:
        value = sample.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            if not value:
                continue
            value = value[0]
            if isinstance(value, list) and value:
                value = value[0]
        if isinstance(value, dict):
            nested = value.get("pred_answer") or value.get("prediction") or value.get("response")
            if nested is not None:
                value = nested
            else:
                continue
        return str(value)
    return ""


def _unwrap_response_value(value: Any) -> str:
    if value is None:
        return ""
    while isinstance(value, list):
        if not value:
            return ""
        value = value[0]
    if isinstance(value, dict):
        for key in ("response", "prediction", "pred", "text", "answer"):
            if value.get(key) is not None:
                return _unwrap_response_value(value.get(key))
        return ""
    return str(value)


def _sample_raw_response(sample: dict[str, Any]) -> str:
    for key in ("resps", "filtered_resps", "response", "prediction", "pred"):
        value = sample.get(key)
        raw = _unwrap_response_value(value)
        if raw:
            return raw
    return _sample_prediction(sample)


def _sample_answer(sample: dict[str, Any]) -> str:
    doc = _extract_doc_from_sample(sample)
    for container in (sample, doc):
        if not isinstance(container, dict):
            continue
        for key in ("answer", "target", "gt_answer"):
            value = container.get(key)
            if value is not None:
                return str(value)
    payload = _sample_metric_payload(sample)
    if payload and payload.get("answer") is not None:
        return str(payload.get("answer"))
    return ""


def _sample_correct(sample: dict[str, Any]) -> bool | None:
    payload = _sample_metric_payload(sample)
    if payload and payload.get("score") is not None:
        return bool(float(payload.get("score")))
    for key in ("correct", "acc", "exact_match"):
        value = sample.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(float(value))
    return None


def _lmms_sample_records(output_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in _find_lmms_samples(output_dir):
        try:
            data = _read_jsonl(path) if path.suffix == ".jsonl" else _read_json(path)
        except Exception:
            continue
        samples: list[dict[str, Any]] = []
        if isinstance(data, list):
            samples.extend(x for x in data if isinstance(x, dict))
        elif isinstance(data, dict):
            for value in data.values():
                if isinstance(value, list):
                    samples.extend(x for x in value if isinstance(x, dict))
        for sample in samples:
            qid = _sample_question_id(sample)
            if not qid:
                continue
            records.append(
                {
                    "question_id": qid,
                    "pred": _sample_prediction(sample),
                    "raw": _sample_raw_response(sample),
                    "answer": _sample_answer(sample),
                    "correct": _sample_correct(sample),
                }
            )
    return records


def _bench_sample_records(summary_path: Path, method: str) -> list[dict[str, Any]]:
    jsonl_path = _find_bench_jsonl(summary_path, method)
    if jsonl_path is None:
        return []
    records = []
    for row in _read_jsonl(jsonl_path):
        qid = str(row.get("question_id") or "")
        if not qid:
            continue
        records.append(
            {
                "question_id": qid,
                "pred": str(row.get("pred_answer") or ""),
                "raw": str(row.get("raw_response") or row.get("pred_answer") or ""),
                "answer": str(row.get("answer") or ""),
                "correct": row.get("correct"),
            }
        )
    return records


def _diff_samples(bench_summary: Path, lmms_output: Path, method: str, max_rows: int) -> tuple[list[str], int | None]:
    bench_rows = _bench_sample_records(bench_summary, method)
    lmms_rows = _lmms_sample_records(lmms_output)
    if not bench_rows or not lmms_rows:
        return [], None

    bench_by_qid = {str(row["question_id"]): row for row in bench_rows}
    lmms_by_qid = {str(row["question_id"]): row for row in lmms_rows}
    common_qids = [qid for qid in bench_by_qid if qid in lmms_by_qid]
    mismatches = []
    for qid in common_qids:
        b = bench_by_qid[qid]
        l = lmms_by_qid[qid]
        if b.get("pred") != l.get("pred") or b.get("correct") != l.get("correct"):
            mismatches.append((qid, b, l))

    lines = [
        "",
        "| question_id | bench_pred | lmms_pred | answer | bench_correct | lmms_correct | bench_raw | lmms_raw |",
        "|---|---|---|---|---:|---:|---|---|",
    ]
    for qid, b, l in mismatches[:max_rows]:
        answer = b.get("answer") or l.get("answer") or ""
        bench_raw = str(b.get("raw", "")).replace("\n", "\\n")[:120]
        lmms_raw = str(l.get("raw", "")).replace("\n", "\\n")[:120]
        lines.append(
            f"| {qid} | {b.get('pred', '')} | {l.get('pred', '')} | {answer} | {b.get('correct')} | {l.get('correct')} | {bench_raw} | {lmms_raw} |"
        )
    if len(mismatches) > max_rows:
        lines.append(f"| ... | ... | ... | ... | ... | ... | ... | ... |")
    lines.insert(0, f"\nMatched samples: {len(common_qids)}; mismatches: {len(mismatches)}")
    return lines, len(mismatches)


def _extract_lmms_from_samples(output_dir: Path) -> dict[str, float | None] | None:
    samples: list[dict[str, Any]] = []
    for path in _find_lmms_samples(output_dir):
        try:
            data = _read_jsonl(path) if path.suffix == ".jsonl" else _read_json(path)
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
    scores: dict[str, float | None] = {"overall": None, "short": None, "medium": None, "long": None}
    for path in _find_lmms_results(output_dir):
        parsed = _extract_lmms_from_results_json(path)
        if parsed:
            scores.update({key: value for key, value in parsed.items() if value is not None})
            break
    if log_path:
        parsed = _extract_lmms_from_log(log_path)
        if parsed:
            scores.update({key: value for key, value in parsed.items() if value is not None})
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare VideoMME bench_all_metrics and lmms-eval results.")
    parser.add_argument("--bench_summary", required=True)
    parser.add_argument("--lmms_output", required=True)
    parser.add_argument("--lmms_log", default="")
    parser.add_argument("--method", required=True)
    parser.add_argument("--rate", required=True)
    parser.add_argument("--out_md", default="")
    parser.add_argument("--max_sample_diffs", type=int, default=20)
    parser.add_argument("--fail_on_mismatch", action="store_true")
    parser.add_argument("--tolerance", type=float, default=0.0)
    args = parser.parse_args()

    bench = _bench_scores(Path(args.bench_summary), args.method)
    lmms = _lmms_scores(Path(args.lmms_output), Path(args.lmms_log) if args.lmms_log else None)
    should_fail = False

    lines = [
        "| Method | R | Split | bench_all_metrics | lmms-eval | Delta |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for split in ("overall", "short", "medium", "long"):
        b = bench.get(split)
        l = lmms.get(split)
        delta = None if b is None or l is None else l - b
        if args.fail_on_mismatch and delta is not None and abs(delta) > args.tolerance:
            should_fail = True
        lines.append(f"| {args.method} | {args.rate} | {split} | {_fmt(b)} | {_fmt(l)} | {_fmt(delta)} |")
    table = "\n".join(lines) + "\n"
    sample_diff_lines, sample_mismatches = _diff_samples(
        Path(args.bench_summary),
        Path(args.lmms_output),
        args.method,
        args.max_sample_diffs,
    )
    if args.fail_on_mismatch and sample_mismatches is not None and sample_mismatches > 0:
        should_fail = True
    if sample_diff_lines:
        table += "\n" + "\n".join(sample_diff_lines) + "\n"
    print(table)
    if args.out_md:
        Path(args.out_md).write_text(table, encoding="utf-8")
    if args.fail_on_mismatch and should_fail:
        sys.exit(2)


if __name__ == "__main__":
    main()
