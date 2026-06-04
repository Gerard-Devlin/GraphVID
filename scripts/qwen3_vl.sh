#!/usr/bin/env bash
set -euo pipefail

# Official lmms-eval runner for Qwen3-8B method comparisons.
# This script only maps method names to model_args; task parsers/scorers remain
# the official lmms-eval ones.

cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-/gluster/envs/users/wuzhijian/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-20480}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="$PWD:$PWD/lmms-eval:${PYTHONPATH:-}"

ACCELERATE="${ACCELERATE:-accelerate}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-18888}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"

PRETRAINED="${PRETRAINED:-$HF_HOME/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b}"
METHODS="${METHODS:-flashvid,graphvid,fastvid,fastgraphvid,visionzip}"
RATES="${RATES:-0.10,0.15,0.20,0.25}"
TASKS="${TASKS:-videomme,egoschema,mvbench,longvideobench_val_v}"
OUTPUT_PATH="${OUTPUT_PATH:-./logs/lmms_eval_qwen3_8b}"
LOG_SAMPLES_SUFFIX="${LOG_SAMPLES_SUFFIX:-qwen3_vl}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GEN_KWARGS="${GEN_KWARGS:-max_new_tokens=16,temperature=0}"
LIMIT="${LIMIT:-}"

MAX_NUM_FRAMES="${MAX_NUM_FRAMES:-32}"
MIN_PIXELS="${MIN_PIXELS:-50176}"
MAX_PIXELS="${MAX_PIXELS:-200704}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"

DO_SEGMENT="${DO_SEGMENT:-True}"
SEGMENT_THRESHOLD="${SEGMENT_THRESHOLD:-0.9}"
MIN_SEGMENT_NUM="${MIN_SEGMENT_NUM:-8}"
COMPLEMENTARY_SEGMENT="${COMPLEMENTARY_SEGMENT:-True}"
ALPHA="${ALPHA:-0.70}"
TEMPORAL_THRESHOLD="${TEMPORAL_THRESHOLD:-0.8}"
EXPANSION="${EXPANSION:-1.25}"
PRUNING_LAYER="${PRUNING_LAYER:-20}"
LLM_RETENTION_RATIO="${LLM_RETENTION_RATIO:-1.0}"

FLASHVID_TOKEN_SELECTION_METHOD="${FLASHVID_TOKEN_SELECTION_METHOD:-attn_div_v2}"
GRAPHVID_TOKEN_SELECTION_METHOD="${GRAPHVID_TOKEN_SELECTION_METHOD:-attn_div_stable}"
ADAPTER_TOKEN_SELECTION_METHOD="${ADAPTER_TOKEN_SELECTION_METHOD:-attn_div_stable}"

GRAPH_TEMPORAL_TOPK="${GRAPH_TEMPORAL_TOPK:-2}"
GRAPH_TEMPORAL_RADIUS="${GRAPH_TEMPORAL_RADIUS:-1}"
GRAPH_TEMPORAL_SKIP="${GRAPH_TEMPORAL_SKIP:-1}"
GRAPH_PROTECT_RATIO="${GRAPH_PROTECT_RATIO:-0.15}"
GRAPH_TARGET_RATIO="${GRAPH_TARGET_RATIO:-1.0}"
GRAPH_REPRESENTATIVE="${GRAPH_REPRESENTATIVE:-medoid}"
GRAPH_FINAL_TPF="${GRAPH_FINAL_TPF:-0}"
GRAPH_FRAME_FLOOR_RATIO="${GRAPH_FRAME_FLOOR_RATIO:-0.55}"
GRAPH_SKIP_SPATIAL_MERGE_WHEN_CAPPED="${GRAPH_SKIP_SPATIAL_MERGE_WHEN_CAPPED:-False}"

ADAPTER_BUDGET_USES_EXPANSION="${ADAPTER_BUDGET_USES_EXPANSION:-True}"
FASTVID_DYSEG_C="${FASTVID_DYSEG_C:-8}"
FASTVID_DYSEG_TAU="${FASTVID_DYSEG_TAU:-0.90}"
FASTVID_DYSEG_IGNORE="${FASTVID_DYSEG_IGNORE:-0.95}"
FASTVID_STPRUNE_D="${FASTVID_STPRUNE_D:-0.40}"
FASTVID_DTM_P="${FASTVID_DTM_P:-4}"
FASTVID_DTM_BETA="${FASTVID_DTM_BETA:-0.60}"

FASTGRAPH_ATS_RATIO="${FASTGRAPH_ATS_RATIO:-0.60}"
FASTGRAPH_TEMPORAL_RADIUS="${FASTGRAPH_TEMPORAL_RADIUS:-1}"
FASTGRAPH_TEMPORAL_SKIP="${FASTGRAPH_TEMPORAL_SKIP:-1}"
FASTGRAPH_TEMPORAL_TOPK="${FASTGRAPH_TEMPORAL_TOPK:-2}"
FASTGRAPH_EDGE_THRESHOLD="${FASTGRAPH_EDGE_THRESHOLD:-0.0}"
FASTGRAPH_PROTECT_RATIO="${FASTGRAPH_PROTECT_RATIO:-0.15}"
FASTGRAPH_ATTN_WEIGHT="${FASTGRAPH_ATTN_WEIGHT:-0.55}"
FASTGRAPH_NOVELTY_WEIGHT="${FASTGRAPH_NOVELTY_WEIGHT:-0.30}"
FASTGRAPH_DENSITY_WEIGHT="${FASTGRAPH_DENSITY_WEIGHT:-0.15}"
VISIONZIP_DOMINANT_RATIO="${VISIONZIP_DOMINANT_RATIO:-0.85}"

split_csv() {
  local text="$1"
  text="${text//,/ }"
  # shellcheck disable=SC2086
  printf '%s\n' $text
}

base_model_args() {
  printf 'pretrained=%s,max_num_frames=%s,min_pixels=%s,max_pixels=%s,attn_implementation=%s' \
    "$PRETRAINED" "$MAX_NUM_FRAMES" "$MIN_PIXELS" "$MAX_PIXELS" "$ATTN_IMPLEMENTATION"
}

common_flash_args() {
  local retention_ratio="$1"
  printf 'enable_flashvid=True,retention_ratio=%s,expansion=%s,do_segment=%s,segment_threshold=%s,min_segment_num=%s,complementary_segment=%s,alpha=%s,temporal_threshold=%s,pruning_layer=%s,llm_retention_ratio=%s' \
    "$retention_ratio" "$EXPANSION" "$DO_SEGMENT" "$SEGMENT_THRESHOLD" "$MIN_SEGMENT_NUM" "$COMPLEMENTARY_SEGMENT" "$ALPHA" "$TEMPORAL_THRESHOLD" "$PRUNING_LAYER" "$LLM_RETENTION_RATIO"
}

method_flash_args() {
  local method="$1"
  case "$method" in
    flashvid)
      printf 'compression_variant=flashvid,token_selection_method=%s' "$FLASHVID_TOKEN_SELECTION_METHOD"
      ;;
    graphvid)
      printf 'compression_variant=graphvid,token_selection_method=%s,graph_temporal_topk=%s,graph_temporal_radius=%s,graph_temporal_skip=%s,graph_merge_protect_ratio=%s,graph_merge_target_ratio=%s,graph_merge_representative=%s,graph_final_tokens_per_frame=%s,graph_final_frame_floor_ratio=%s,graph_skip_spatial_merge_when_capped=%s' \
        "$GRAPHVID_TOKEN_SELECTION_METHOD" "$GRAPH_TEMPORAL_TOPK" "$GRAPH_TEMPORAL_RADIUS" "$GRAPH_TEMPORAL_SKIP" "$GRAPH_PROTECT_RATIO" "$GRAPH_TARGET_RATIO" "$GRAPH_REPRESENTATIVE" "$GRAPH_FINAL_TPF" "$GRAPH_FRAME_FLOOR_RATIO" "$GRAPH_SKIP_SPATIAL_MERGE_WHEN_CAPPED"
      ;;
    fastvid)
      printf 'compression_variant=fastvid,token_selection_method=%s,adapter_budget_uses_expansion=%s,external_budget_uses_expansion=%s,fastvid_DySeg_c=%s,fastvid_DySeg_tau=%s,fastvid_DySeg_ignore=%s,fastvid_STPrune_d=%s,fastvid_DTM_p=%s,fastvid_DTM_beta=%s' \
        "$ADAPTER_TOKEN_SELECTION_METHOD" "$ADAPTER_BUDGET_USES_EXPANSION" "$ADAPTER_BUDGET_USES_EXPANSION" "$FASTVID_DYSEG_C" "$FASTVID_DYSEG_TAU" "$FASTVID_DYSEG_IGNORE" "$FASTVID_STPRUNE_D" "$FASTVID_DTM_P" "$FASTVID_DTM_BETA"
      ;;
    fastgraphvid)
      printf 'compression_variant=fastgraphvid,token_selection_method=%s,adapter_budget_uses_expansion=%s,external_budget_uses_expansion=%s,fastvid_DySeg_c=%s,fastvid_DySeg_tau=%s,fastvid_DySeg_ignore=%s,fastvid_STPrune_d=%s,fastvid_DTM_p=%s,fastvid_DTM_beta=%s,fastgraph_ats_ratio=%s,fastgraph_temporal_radius=%s,fastgraph_temporal_skip=%s,fastgraph_temporal_topk=%s,fastgraph_edge_threshold=%s,fastgraph_protect_ratio=%s,fastgraph_attn_weight=%s,fastgraph_novelty_weight=%s,fastgraph_density_weight=%s' \
        "$ADAPTER_TOKEN_SELECTION_METHOD" "$ADAPTER_BUDGET_USES_EXPANSION" "$ADAPTER_BUDGET_USES_EXPANSION" "$FASTVID_DYSEG_C" "$FASTVID_DYSEG_TAU" "$FASTVID_DYSEG_IGNORE" "$FASTVID_STPRUNE_D" "$FASTVID_DTM_P" "$FASTVID_DTM_BETA" "$FASTGRAPH_ATS_RATIO" "$FASTGRAPH_TEMPORAL_RADIUS" "$FASTGRAPH_TEMPORAL_SKIP" "$FASTGRAPH_TEMPORAL_TOPK" "$FASTGRAPH_EDGE_THRESHOLD" "$FASTGRAPH_PROTECT_RATIO" "$FASTGRAPH_ATTN_WEIGHT" "$FASTGRAPH_NOVELTY_WEIGHT" "$FASTGRAPH_DENSITY_WEIGHT"
      ;;
    visionzip)
      printf 'compression_variant=visionzip,token_selection_method=%s,adapter_budget_uses_expansion=%s,external_budget_uses_expansion=%s,visionzip_dominant_ratio=%s' \
        "$ADAPTER_TOKEN_SELECTION_METHOD" "$ADAPTER_BUDGET_USES_EXPANSION" "$ADAPTER_BUDGET_USES_EXPANSION" "$VISIONZIP_DOMINANT_RATIO"
      ;;
    *)
      echo "Unknown method: $method" >&2
      return 1
      ;;
  esac
}

for method in $(split_csv "$METHODS"); do
  for retention_ratio in $(split_csv "$RATES"); do
    model_args="$(base_model_args),$(common_flash_args "$retention_ratio"),$(method_flash_args "$method")"
    for task in $(split_csv "$TASKS"); do
      cmd=(
        "$ACCELERATE" launch
        --main_process_port "$MAIN_PROCESS_PORT"
        --num_processes "$NUM_PROCESSES"
        -m lmms_eval
        --model qwen3_vl
        --model_args "$model_args"
        --tasks "$task"
        --batch_size "$BATCH_SIZE"
        --gen_kwargs "$GEN_KWARGS"
        --log_samples
        --log_samples_suffix "${LOG_SAMPLES_SUFFIX}_${method}_r${retention_ratio}"
        --output_path "$OUTPUT_PATH"
      )
      if [[ -n "$LIMIT" ]]; then
        cmd+=(--limit "$LIMIT")
      fi
      echo "[lmms-eval] method=$method rate=$retention_ratio task=$task"
      echo "[lmms-eval] ${cmd[*]}"
      if [[ "${DRY_RUN:-0}" != "1" ]]; then
        "${cmd[@]}"
      fi
    done
  done
done
