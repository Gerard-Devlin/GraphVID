from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from playground.bench_all_metrics import (
    BenchmarkArgs,
    _apply_flashvid_original,
    _benchmark_single_sample,
    _load_backend_model,
    _load_dataset,
)


def _flashvid_configs(model) -> list[Any]:
    out = []
    for obj in (
        model,
        getattr(model, "model", None),
        getattr(model, "language_model", None),
        getattr(model, "module", None),
        getattr(getattr(model, "module", None), "model", None),
    ):
        if obj is None:
            continue
        cfg = getattr(obj, "flashvid_config", None)
        if cfg is not None and cfg not in out:
            out.append(cfg)
    return out


def _best_teacher_config(model):
    best = None
    best_score = -1
    for cfg in _flashvid_configs(model):
        score = int(getattr(cfg, "last_learn_teacher_features", None) is not None)
        score += int(getattr(cfg, "last_learn_teacher_labels", None) is not None)
        score += int(getattr(cfg, "last_learn_teacher_keep_ratio", None) is not None)
        if score > best_score:
            best_score = score
            best = cfg
    return best


def _clear_teacher_state(model) -> None:
    for cfg in _flashvid_configs(model):
        for name in (
            "last_learn_teacher_features",
            "last_learn_teacher_raw_indices",
            "last_learn_teacher_labels",
            "last_learn_teacher_keep_indices",
            "last_learn_teacher_keep_ratio",
            "last_learn_teacher_visual_keep_positions",
            "last_learn_teacher_shape",
            "last_learn_vision_global_indices",
            "visual_seq_global_indices",
        ):
            if hasattr(cfg, name):
                setattr(cfg, name, None)


def collect(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    bench_args = BenchmarkArgs(
        model_path=args.model_path,
        model_backend=args.model_backend,
        dataset_jsonl=args.dataset_jsonl,
        hf_home=args.hf_home,
        start_index=args.start_index,
        limit=args.limit,
        shuffle=False,
        duration_filter=args.duration_filter,
        num_frames=args.num_frames,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        num_warmup=0,
        num_runs=1,
        max_new_tokens=args.max_new_tokens,
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
        retention_ratio=args.retention_ratio,
        expansion=args.expansion,
        llm_retention_ratio=args.llm_retention_ratio,
        token_selection_method=args.token_selection_method,
        flashvid_token_selection_method=args.flashvid_token_selection_method,
        alpha=args.alpha,
        temporal_threshold=args.temporal_threshold,
        learn_collect_teacher=True,
        learn_density_topk=args.learn_density_topk,
    )

    print("[load] model")
    model_bundle = _load_backend_model(bench_args)
    model = model_bundle["model"]
    _apply_flashvid_original(model, bench_args, model_bundle["backend"])
    model.eval()

    samples = _load_dataset(
        args.dataset_jsonl,
        args.limit,
        shuffle=False,
        start_index=args.start_index,
        duration_filter=args.duration_filter,
    )
    print(f"[data] samples={len(samples)} out={out_dir}")

    written = 0
    skipped = 0
    with manifest_path.open("w", encoding="utf-8") as mf:
        for idx, sample in enumerate(samples, 1):
            _clear_teacher_state(model)
            if hasattr(model, "flashvid_config"):
                cfg0 = getattr(model, "flashvid_config")
                setattr(cfg0, "current_video_duration", sample.get("duration"))
                setattr(cfg0, "current_task_category", sample.get("task_category"))
                setattr(cfg0, "current_category", sample.get("category"))
            record = _benchmark_single_sample(model_bundle, bench_args, sample, use_acceleration=True)
            cfg = _best_teacher_config(model)
            features = getattr(cfg, "last_learn_teacher_features", None) if cfg is not None else None
            labels = getattr(cfg, "last_learn_teacher_labels", None) if cfg is not None else None
            if features is None or labels is None:
                skipped += 1
                print(f"[skip] {idx}/{len(samples)} qid={sample.get('question_id')} labels_missing error={record.get('error')}")
                continue
            qid = str(sample.get("question_id") or f"sample_{args.start_index + idx - 1}")
            safe_qid = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in qid)
            path = out_dir / f"{safe_qid}.pt"
            payload = {
                "features": features.cpu().float(),
                "labels": labels.cpu().float(),
                "feature_names": getattr(cfg, "last_learn_teacher_feature_names", None),
                "shape": getattr(cfg, "last_learn_teacher_shape", None),
                "keep_ratio": getattr(cfg, "last_learn_teacher_keep_ratio", None),
                "question_id": sample.get("question_id"),
                "videoID": sample.get("videoID"),
                "duration": sample.get("duration"),
                "answer": sample.get("answer"),
                "pred_answer": record.get("pred_answer"),
                "correct": record.get("correct"),
                "error": record.get("error"),
            }
            torch.save(payload, str(path))
            mf.write(
                json.dumps(
                    {
                        "path": str(path),
                        "question_id": sample.get("question_id"),
                        "videoID": sample.get("videoID"),
                        "duration": sample.get("duration"),
                        "num_tokens": int(labels.numel()),
                        "positive_ratio": float(labels.float().mean().item()) if labels.numel() else 0.0,
                        "correct": record.get("correct"),
                        "error": record.get("error"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            mf.flush()
            written += 1
            print(
                f"[write] {idx}/{len(samples)} qid={sample.get('question_id')} "
                f"pos={float(labels.float().mean().item()):.4f} correct={record.get('correct')} file={path.name}"
            )
            if torch.cuda.is_available() and idx % 20 == 0:
                torch.cuda.empty_cache()
    print(json.dumps({"written": written, "skipped": skipped, "manifest": str(manifest_path)}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect LearnFlashVID teacher token labels from FlashVID inner-LLM pruning.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_backend", default="llava")
    parser.add_argument("--dataset_jsonl", default="assets/videomme.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--hf_home", default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--duration_filter", default="")
    parser.add_argument("--num_frames", type=int, default=64)
    parser.add_argument("--min_pixels", type=int, default=64 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retention_ratio", type=float, default=0.10)
    parser.add_argument("--expansion", type=float, default=1.25)
    parser.add_argument("--llm_retention_ratio", type=float, default=0.30)
    parser.add_argument("--alpha", type=float, default=0.70)
    parser.add_argument("--temporal_threshold", type=float, default=0.80)
    parser.add_argument("--token_selection_method", default="attn_div_v2")
    parser.add_argument("--flashvid_token_selection_method", default="attn_div_v2")
    parser.add_argument("--learn_density_topk", type=int, default=8)
    collect(parser.parse_args())


if __name__ == "__main__":
    main()
