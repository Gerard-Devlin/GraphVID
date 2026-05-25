from __future__ import annotations

import argparse

from collect_learnflashvid_teacher import DEFAULT_GPU_CAP, collect


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect baseline/full-visual-token teacher labels for learned token value. "
            "This keeps before-LLM vision retention at 100% and uses inner-LLM attention "
            "as the teacher signal, so the selector learns from the uncompressed model path "
            "rather than from FlashVID's token subset."
        )
    )
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
    parser.add_argument("--llm_retention_ratio", type=float, default=0.30)
    parser.add_argument("--token_selection_method", default="attn_div_v2")
    parser.add_argument("--flashvid_token_selection_method", default="attn_div_v2")
    parser.add_argument("--learn_density_topk", type=int, default=8)
    parser.add_argument("--free_ratio", type=float, default=0.90)
    parser.add_argument("--min_free_mb", type=int, default=22000)
    parser.add_argument("--max_gpus", type=int, default=1, help="0 means use eligible GPUs up to --gpu_cap.")
    parser.add_argument("--gpu_cap", type=int, default=DEFAULT_GPU_CAP, help="Hard cap for GPUs per launch. 0 disables the cap.")
    parser.add_argument("--gpu_ids", default="", help="Comma-separated GPU ids. Overrides auto selection.")
    args = parser.parse_args()

    # Full visual-token teacher: ADTS selects every token and TSTM is disabled.
    args.retention_ratio = 1.0
    args.expansion = 1.0
    args.alpha = 1.0
    args.temporal_threshold = 1.0
    collect(args)


if __name__ == "__main__":
    main()
