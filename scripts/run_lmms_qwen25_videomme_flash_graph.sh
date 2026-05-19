#!/usr/bin/env bash
set -euo pipefail

cd "${GRAPHVID_ROOT:-$HOME/GraphVID}"

export HF_HOME="${HF_HOME:-/gluster/envs/users/wuzhijian/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-20480}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTHONPATH="$PWD:$PWD/lmms-eval:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-/gluster/envs/users/wuzhijian/hf_home/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5}"
RATE="${RATE:-0.10}"
LIMIT="${LIMIT:-100}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
PORT_FLASH="${PORT_FLASH:-18888}"
PORT_GRAPH="${PORT_GRAPH:-18889}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/lmms_eval}"
TAG="${TAG:-qwen25_r10_smoke100}"
TASK_NAME="${TASK_NAME:-videomme_local}"
RUN_FLASHVID="${RUN_FLASHVID:-1}"
RUN_GRAPHVID="${RUN_GRAPHVID:-1}"
GRAPH_FINAL_TPF="${GRAPH_FINAL_TPF:-0}"

mkdir -p "$OUTPUT_ROOT"

BASE_MODEL_ARGS="pretrained=$MODEL_PATH,max_num_frames=32,attn_implementation=flash_attention_2"
COMMON_FLASH_ARGS="enable_flashvid=True,expansion=1.25,do_segment=True,min_segment_num=4,complementary_segment=True,token_selection_method=attn_div,alpha=0.70,temporal_threshold=0.8,pruning_layer=20,llm_retention_ratio=0.3,retention_ratio=$RATE"

LIMIT_ARGS=()
if [[ "$LIMIT" != "0" && "$LIMIT" != "none" && "$LIMIT" != "None" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

if [[ "$RUN_FLASHVID" == "1" ]]; then
  echo "[lmms] Qwen2.5-VL VideoMME FlashVID RATE=$RATE LIMIT=$LIMIT"
  accelerate launch \
    --main_process_port "$PORT_FLASH" \
    --num_processes "$NUM_PROCESSES" \
    -m lmms_eval \
    --model qwen2_5_vl \
    --model_args "$BASE_MODEL_ARGS,$COMMON_FLASH_ARGS,compression_variant=flashvid" \
    --tasks "$TASK_NAME" \
    --batch_size 1 \
    "${LIMIT_ARGS[@]}" \
    --log_samples \
    --log_samples_suffix "${TAG}_flashvid" \
    --output_path "$OUTPUT_ROOT/${TAG}_flashvid"
fi

if [[ "$RUN_GRAPHVID" == "1" ]]; then
  echo "[lmms] Qwen2.5-VL VideoMME GraphVID RATE=$RATE LIMIT=$LIMIT"
  accelerate launch \
    --main_process_port "$PORT_GRAPH" \
    --num_processes "$NUM_PROCESSES" \
    -m lmms_eval \
    --model qwen2_5_vl \
    --model_args "$BASE_MODEL_ARGS,$COMMON_FLASH_ARGS,compression_variant=graphvid,graph_temporal_topk=2,graph_temporal_radius=1,graph_temporal_skip=1,graph_merge_protect_ratio=0.15,graph_merge_target_ratio=1.00,graph_merge_representative=medoid,graph_final_tokens_per_frame=$GRAPH_FINAL_TPF" \
    --tasks "$TASK_NAME" \
    --batch_size 1 \
    "${LIMIT_ARGS[@]}" \
    --log_samples \
    --log_samples_suffix "${TAG}_graphvid" \
    --output_path "$OUTPUT_ROOT/${TAG}_graphvid"
fi
