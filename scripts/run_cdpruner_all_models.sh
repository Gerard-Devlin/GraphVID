#!/usr/bin/env bash
set -u -o pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"

export PATH="${CONDA_PREFIX:+$CONDA_PREFIX/bin:}$PATH"
export PYTHONPATH="$ROOT/lmms-eval:$ROOT:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/home/xuyouwen/hf_home_local}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/home/xuyouwen/hf_hub_local}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_EVALUATE_OFFLINE="${HF_EVALUATE_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export LMMS_EVAL_SERIALIZE_DATASET_LOAD="${LMMS_EVAL_SERIALIZE_DATASET_LOAD:-1}"
export TMPDIR="${TMPDIR:-/home/xuyouwen/tmp}"
mkdir -p "$TMPDIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export NUM_PROCESSES="${NUM_PROCESSES:-6}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export ACCELERATE="${ACCELERATE:-accelerate}"

RATES="${RATES:-0.005,0.01}"
TASKS="${TASKS:-videomme,egoschema_subset,egoschema,longvideobench_val_v,mvbench}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
BASE_PORT="${BASE_PORT:-19160}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/logs/lmms_eval/cdpruner_all_models_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTPUT_ROOT"
MASTER_LOG="$OUTPUT_ROOT/all_models.log"
if [[ "${CDPRUNER_MASTER_LOG_ACTIVE:-0}" != "1" ]]; then
  export CDPRUNER_MASTER_LOG_ACTIVE=1
  exec > >(tee "$MASTER_LOG") 2>&1
fi

resolve_model() {
  local repo="$1"
  local fallback="$2"
  local revision
  if [[ -f "$repo/refs/main" ]]; then
    revision="$(tr -d '\r\n' < "$repo/refs/main")"
    readlink -f "$repo/snapshots/$revision"
    return
  fi
  if [[ "${HF_HUB_OFFLINE,,}" == "1" || "${HF_HUB_OFFLINE,,}" == "true" ]]; then
    echo "ERROR: offline model cache not found: $repo" >&2
    return 1
  fi
  printf '%s\n' "$fallback"
}

OV_REPO="${OV_REPO:-$HF_HUB_CACHE/models--lmms-lab--llava-onevision-qwen2-7b-ov}"
VIDEO_REPO="${VIDEO_REPO:-$HF_HUB_CACHE/models--lmms-lab--LLaVA-Video-7B-Qwen2}"
QWEN3_REPO="${QWEN3_REPO:-$HF_HUB_CACHE/models--Qwen--Qwen3-VL-8B-Instruct}"

OV_MODEL="${OV_MODEL:-$(resolve_model "$OV_REPO" lmms-lab/llava-onevision-qwen2-7b-ov)}"
VIDEO_MODEL="${VIDEO_MODEL:-$(resolve_model "$VIDEO_REPO" lmms-lab/LLaVA-Video-7B-Qwen2)}"
QWEN3_MODEL="${QWEN3_MODEL:-$(resolve_model "$QWEN3_REPO" Qwen/Qwen3-VL-8B-Instruct)}"

if [[ -z "${CDPRUNER_TEXT_MODEL_PATH:-}" ]]; then
  SIGLIP_REPO="$HF_HUB_CACHE/models--google--siglip-so400m-patch14-384"
  CDPRUNER_TEXT_MODEL_PATH="$(resolve_model "$SIGLIP_REPO" google/siglip-so400m-patch14-384)"
fi
export CDPRUNER_TEXT_MODEL_PATH

FAILURES=0
run_stage() {
  local name="$1"
  local log_name="$2"
  shift 2
  local log_path="$OUTPUT_ROOT/$log_name.log"
  echo "============================================================"
  echo "Starting: $name"
  echo "Rates: $RATES"
  echo "Tasks: $TASKS"
  echo "GPUs: $CUDA_VISIBLE_DEVICES"
  echo "Log: $log_path"
  echo "============================================================"
  if "$@" 2>&1 | tee "$log_path"; then
    echo "COMPLETED: $name"
  else
    echo "FAILED: $name"
    FAILURES=$((FAILURES + 1))
  fi
  sleep "${STAGE_PAUSE_SECONDS:-15}"
}

run_stage "LLaVA-OneVision" llava_onevision \
  env \
    PRETRAINED="$OV_MODEL" \
    METHODS=cdpruner \
    RATES="$RATES" \
    TASKS="$TASKS" \
    MAX_FRAMES_NUM=32 \
    ATTN_IMPLEMENTATION="$ATTN_IMPLEMENTATION" \
    MAIN_PROCESS_PORT="$BASE_PORT" \
    OUTPUT_PATH="$OUTPUT_ROOT/llava_onevision" \
    bash scripts/llava_ov.sh

run_stage "LLaVA-Video" llava_video \
  env \
    PRETRAINED="$VIDEO_MODEL" \
    METHODS=cdpruner \
    RATES="$RATES" \
    TASKS="$TASKS" \
    MAX_FRAMES_NUM=64 \
    ATTN_IMPLEMENTATION="$ATTN_IMPLEMENTATION" \
    MAIN_PROCESS_PORT="$((BASE_PORT + 1))" \
    OUTPUT_PATH="$OUTPUT_ROOT/llava_video" \
    bash scripts/llava_vid.sh

run_stage "Qwen3-VL" qwen3_vl \
  env \
    PRETRAINED="$QWEN3_MODEL" \
    METHODS=cdpruner \
    RATES="$RATES" \
    TASKS="$TASKS" \
    MAX_NUM_FRAMES=32 \
    ATTN_IMPLEMENTATION="$ATTN_IMPLEMENTATION" \
    MAIN_PROCESS_PORT="$((BASE_PORT + 2))" \
    OUTPUT_PATH="$OUTPUT_ROOT/qwen3_vl" \
    bash scripts/qwen3_vl.sh

echo "============================================================"
echo "CDPruner stages finished"
echo "failures=$FAILURES"
echo "results=$OUTPUT_ROOT"
echo "============================================================"
test "$FAILURES" -eq 0
