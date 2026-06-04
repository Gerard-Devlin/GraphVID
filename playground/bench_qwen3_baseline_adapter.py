from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SUPPORTED_METHODS = ("fastvid", "visionzip", "fastgraphvid", "curvevid")


def _install_adapter_compression_patch(method: str) -> None:
    import flashvid.modeling_qwen3_vl as modeling_qwen3_vl
    from playground.qwen3_baseline_adapters import adapter_baseline_compression

    original = modeling_qwen3_vl.flashvid_compression

    def patched_flashvid_compression(
        *,
        video_features,
        cls_attention,
        flashvid_config,
        question_features=None,
    ):
        variant = str(getattr(flashvid_config, "compression_variant", "")).strip().lower()
        if variant in SUPPORTED_METHODS:
            return adapter_baseline_compression(video_features, cls_attention, flashvid_config)
        return original(
            video_features=video_features,
            cls_attention=cls_attention,
            flashvid_config=flashvid_config,
            question_features=question_features,
        )

    modeling_qwen3_vl.flashvid_compression = patched_flashvid_compression


def _patch_bench_apply_ours(method: str, adapter_args: argparse.Namespace) -> None:
    import playground.bench_all_metrics as bench

    original_apply_ours = bench._apply_ours

    def apply_adapter(model, args, backend):
        # 949's flashvid() only accepts flashvid/talon/graphvid at setup time.
        # Initialize the standard hook, then switch only this phase's runtime
        # compression variant to the Qwen3 adapter implementation.
        original_variant = args.compression_variant
        args.compression_variant = "flashvid"
        try:
            model = original_apply_ours(model, args, backend)
        finally:
            args.compression_variant = original_variant

        cfg = getattr(model, "flashvid_config")
        setattr(cfg, "compression_variant", method)
        setattr(cfg, "adapter_budget_uses_expansion", bool(adapter_args.adapter_budget_uses_expansion))
        setattr(cfg, "external_budget_uses_expansion", bool(adapter_args.adapter_budget_uses_expansion))
        setattr(cfg, "fastvid_DySeg_c", int(adapter_args.fastvid_DySeg_c))
        setattr(cfg, "fastvid_DySeg_tau", float(adapter_args.fastvid_DySeg_tau))
        setattr(cfg, "fastvid_DySeg_ignore", float(adapter_args.fastvid_DySeg_ignore))
        setattr(cfg, "fastvid_STPrune_d", float(adapter_args.fastvid_STPrune_d))
        setattr(cfg, "fastvid_DTM_p", int(adapter_args.fastvid_DTM_p))
        setattr(cfg, "fastvid_DTM_beta", float(adapter_args.fastvid_DTM_beta))
        setattr(cfg, "fastgraph_ats_ratio", float(adapter_args.fastgraph_ats_ratio))
        setattr(cfg, "fastgraph_temporal_radius", int(adapter_args.fastgraph_temporal_radius))
        setattr(cfg, "fastgraph_temporal_skip", int(adapter_args.fastgraph_temporal_skip))
        setattr(cfg, "fastgraph_temporal_topk", int(adapter_args.fastgraph_temporal_topk))
        setattr(cfg, "fastgraph_edge_threshold", float(adapter_args.fastgraph_edge_threshold))
        setattr(cfg, "fastgraph_protect_ratio", float(adapter_args.fastgraph_protect_ratio))
        setattr(cfg, "fastgraph_attn_weight", float(adapter_args.fastgraph_attn_weight))
        setattr(cfg, "fastgraph_novelty_weight", float(adapter_args.fastgraph_novelty_weight))
        setattr(cfg, "fastgraph_density_weight", float(adapter_args.fastgraph_density_weight))
        setattr(cfg, "curvevid_temperature", float(adapter_args.curvevid_temperature))
        setattr(cfg, "curvevid_mix", float(adapter_args.curvevid_mix))
        setattr(cfg, "curvevid_min_per_frame", int(adapter_args.curvevid_min_per_frame))
        setattr(cfg, "visionzip_dominant_ratio", float(adapter_args.visionzip_dominant_ratio))
        return model

    bench._apply_ours = apply_adapter


def _rename_summary_phase(summary_path: Path, method: str) -> None:
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if "ours" in summary:
        summary[method] = summary.pop("ours")
    comparison = summary.get("comparison")
    if isinstance(comparison, dict):
        for key in list(comparison.keys()):
            if "_ours" in key or "ours_" in key:
                comparison.pop(key, None)
    breakdown = summary.get("duration_breakdown")
    if isinstance(breakdown, dict):
        for bucket in breakdown.values():
            if isinstance(bucket, dict) and "ours" in bucket:
                bucket[method] = bucket.pop("ours")
            comp = bucket.get("comparison") if isinstance(bucket, dict) else None
            if isinstance(comp, dict):
                for key in list(comp.keys()):
                    if "_ours" in key or "ours_" in key:
                        comp.pop(key, None)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _build_bench_args(cli: argparse.Namespace):
    import playground.bench_all_metrics as bench

    args = bench.BenchmarkArgs()
    args.model_path = cli.model_path
    args.model_backend = cli.model_backend
    args.dataset_jsonl = cli.dataset_jsonl
    args.hf_home = cli.hf_home
    args.start_index = cli.start_index
    args.limit = cli.limit
    args.shuffle = False
    args.duration_filter = cli.duration_filter
    args.num_frames = cli.num_frames
    args.min_pixels = cli.min_pixels
    args.max_pixels = cli.max_pixels
    args.num_warmup = cli.num_warmup
    args.num_runs = cli.num_runs
    args.max_new_tokens = cli.max_new_tokens
    args.attn_implementation = cli.attn_implementation
    args.retention_ratio = cli.retention_ratio
    args.expansion = cli.expansion
    args.llm_retention_ratio = cli.llm_retention_ratio
    args.token_selection_method = cli.token_selection_method
    args.flashvid_token_selection_method = cli.token_selection_method
    args.graphvid_token_selection_method = cli.token_selection_method
    args.local_files_only = cli.local_files_only
    args.reload_model_each_phase = True

    args.run_baseline = False
    args.run_flashvid = False
    args.run_graphvid = False
    args.run_ours = True
    args.compression_variant = cli.method

    out_dir = REPO_ROOT / "logs" / "efficiency" / "qwen3_baseline_adapters" / cli.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    args.ours_output = cli.output_jsonl or str(out_dir / f"{cli.method}.jsonl")
    args.summary_output_json = cli.summary_output_json or str(out_dir / f"{cli.method}_summary.json")
    return args, out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run sidecar Qwen3 baseline adapters without modifying GraphVID/FlashVID core."
    )
    parser.add_argument("--method", choices=SUPPORTED_METHODS, required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_backend", default="qwen3_vl")
    parser.add_argument("--dataset_jsonl", default="assets/videomme.jsonl")
    parser.add_argument("--hf_home", default="")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output_jsonl", default="")
    parser.add_argument("--summary_output_json", default="")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=900)
    parser.add_argument("--duration_filter", default="short")
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--min_pixels", type=int, default=64 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--num_warmup", type=int, default=1)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--retention_ratio", type=float, default=0.10)
    parser.add_argument("--expansion", type=float, default=1.25)
    parser.add_argument("--llm_retention_ratio", type=float, default=1.0)
    parser.add_argument("--token_selection_method", default="attn_div_v2")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--adapter_budget_uses_expansion", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--external_budget_uses_expansion",
        dest="adapter_budget_uses_expansion",
        action=argparse.BooleanOptionalAction,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--fastvid_DySeg_c", type=int, default=8)
    parser.add_argument("--fastvid_DySeg_tau", type=float, default=0.90)
    parser.add_argument("--fastvid_DySeg_ignore", type=float, default=0.95)
    parser.add_argument("--fastvid_STPrune_d", type=float, default=0.40)
    parser.add_argument("--fastvid_DTM_p", type=int, default=4)
    parser.add_argument("--fastvid_DTM_beta", type=float, default=0.60)
    parser.add_argument("--fastgraph_ats_ratio", type=float, default=0.60)
    parser.add_argument("--fastgraph_temporal_radius", type=int, default=1)
    parser.add_argument("--fastgraph_temporal_skip", type=int, default=1)
    parser.add_argument("--fastgraph_temporal_topk", type=int, default=2)
    parser.add_argument("--fastgraph_edge_threshold", type=float, default=0.0)
    parser.add_argument("--fastgraph_protect_ratio", type=float, default=0.15)
    parser.add_argument("--fastgraph_attn_weight", type=float, default=0.55)
    parser.add_argument("--fastgraph_novelty_weight", type=float, default=0.30)
    parser.add_argument("--fastgraph_density_weight", type=float, default=0.15)
    parser.add_argument("--curvevid_temperature", type=float, default=0.70)
    parser.add_argument("--curvevid_mix", type=float, default=0.65)
    parser.add_argument("--curvevid_min_per_frame", type=int, default=1)
    parser.add_argument("--visionzip_dominant_ratio", type=float, default=0.85)
    cli = parser.parse_args()

    import playground.bench_all_metrics as bench

    _install_adapter_compression_patch(cli.method)
    _patch_bench_apply_ours(cli.method, cli)
    bench_args, out_dir = _build_bench_args(cli)
    print(f"[qwen3-adapter] method={cli.method} out_dir={out_dir}")
    bench.run(bench_args)
    _rename_summary_phase(Path(bench_args.summary_output_json), cli.method)
    print(f"[qwen3-adapter] jsonl={bench_args.ours_output}")
    print(f"[qwen3-adapter] summary={bench_args.summary_output_json}")


if __name__ == "__main__":
    main()
