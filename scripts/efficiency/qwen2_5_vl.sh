#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# Optional:
# export HF_HOME=/gluster/envs/users/wuzhijian/hf_home

python playground/bench_all_metrics.py \
    --model_backend qwen2_5_vl \
    --model_path "Qwen/Qwen2.5-VL-7B-Instruct" \
    --dataset_jsonl "videomme.jsonl" \
    --limit 100 \
    --shuffle True \
    --num_frames 64 \
    --num_warmup 1 \
    --num_runs 3 \
    --max_new_tokens 16 \
    --retention_ratio 0.10 \
    --compression_variant "flashvid" \
    --question_aware_reweighting False \
    --adaptive_token_budget False \
    --baseline_output "logs/efficiency/baseline_qwen2_5_all_metrics.jsonl" \
    --flashvid_output "logs/efficiency/flashvid_qwen2_5_all_metrics.jsonl" \
    --summary_output_json "logs/efficiency/summary_qwen2_5_all_metrics.json"
