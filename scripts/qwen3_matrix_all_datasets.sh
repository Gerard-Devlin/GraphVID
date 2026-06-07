#!/usr/bin/env bash
set -euo pipefail

# LMMS-Eval-style launcher for our Qwen3-VL compression matrix.
#
# It mirrors the structure of common lmms-eval scripts:
#   1. define TASK_NAMES / DATASET_JSONLS
#   2. define METHODS / RETENTION_RATIOS
#   3. define model + video input args
#   4. loop over tasks and run one benchmark command per task
#
# We call playground/run_qwen3_matrix.py instead of `python -m lmms_eval`
# because our runner records visual-token/latency metrics and dispatches
# FlashVID/GraphVID/FastVID/VisionZip/CurveVID through the same hooks.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-graphvid311}"
if [[ -n "${CONDA_ENV}" ]] && command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

PYTHON="${PYTHON:-python}"

export HF_HOME="${HF_HOME:-/gluster/envs/users/wuzhijian/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------
TASK_NAMES=(
  "videomme"
  "egoschema_subset"
  "egoschema_total"
  "longvideobench"
  "mvbench"
)

DATASET_JSONLS=(
  "assets/videomme.jsonl"
  "assets/egoschema_subset.jsonl"
  "assets/egoschema.jsonl"
  "assets/longvideobench.jsonl"
  "assets/mvbench.jsonl"
)

# ---------------------------------------------------------------------------
# Model and benchmark configuration
# ---------------------------------------------------------------------------
PRETRAINED_MODEL="${PRETRAINED_MODEL:-qwen3_vl}"
PRETRAINED="${PRETRAINED:-${HF_HUB_CACHE}/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b}"

METHODS=(
  "flashvid"
  "graphvid"
  "fastvid"
  "visionzip"
  "curvevid"
)

RETENTION_RATIOS=(
  "10"
  "15"
  "20"
  "25"
)

MAX_NUM_FRAMES="${MAX_NUM_FRAMES:-32}"
MIN_PIXELS="${MIN_PIXELS:-50176}"       # 64 * 28 * 28
MAX_PIXELS="${MAX_PIXELS:-200704}"      # 256 * 28 * 28
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"

# Parallel launcher controls. run_qwen3_matrix.py will shard each run.
CUDA_DEVICES="${CUDA_DEVICES:-}"
MAX_GPUS="${MAX_GPUS:-4}"
FREE_RATIO="${FREE_RATIO:-0.75}"
MIN_FREE_MB="${MIN_FREE_MB:-18000}"
NUM_WARMUP="${NUM_WARMUP:-1}"
NUM_RUNS="${NUM_RUNS:-1}"
LIMIT="${LIMIT:-0}"                     # 0 = all samples in that JSONL
DURATION_FILTER="${DURATION_FILTER:-}"  # e.g. short
RESUME="${RESUME:-1}"
STRICT_DATASETS="${STRICT_DATASETS:-0}"

# FlashVID/GraphVID budget convention used by our paper tables.
RETENTION_EXPANSION="${RETENTION_EXPANSION:-1.25}"
LLM_RETENTION_RATIO="${LLM_RETENTION_RATIO:-1.0}"
TOKEN_SELECTION_METHOD="${TOKEN_SELECTION_METHOD:-attn_div_v2}"
GRAPHVID_TOKEN_SELECTION_METHOD="${GRAPHVID_TOKEN_SELECTION_METHOD:-attn_div_stable}"

BASE_LOG_DIR="${BASE_LOG_DIR:-logs/efficiency/matrix}"
RUN_TAG="${RUN_TAG:-qwen3_8b_all_datasets_lmms_eval}"
mkdir -p "${BASE_LOG_DIR}"

join_by_comma() {
  local IFS=","
  echo "$*"
}

METHODS_CSV="${METHODS_CSV:-$(join_by_comma "${METHODS[@]}")}"
RATES_CSV="${RATES_CSV:-$(join_by_comma "${RETENTION_RATIOS[@]}")}"

echo "========================================"
echo "Qwen3 compression matrix"
echo " Model      : ${PRETRAINED}"
echo " Methods    : ${METHODS_CSV}"
echo " Ratios     : ${RATES_CSV}"
echo " Frames     : ${MAX_NUM_FRAMES}"
echo " Pixels     : min=${MIN_PIXELS}, max=${MAX_PIXELS}"
echo " Max GPUs   : ${MAX_GPUS}, GPU_IDS=${CUDA_DEVICES:-auto}"
echo " Logs       : ${BASE_LOG_DIR}/${RUN_TAG}"
echo "========================================"

for task_idx in "${!TASK_NAMES[@]}"; do
  task_name="${TASK_NAMES[$task_idx]}"
  dataset_jsonl="${DATASET_JSONLS[$task_idx]}"

  if [[ ! -f "${dataset_jsonl}" ]]; then
    echo "[skip] task=${task_name} missing ${dataset_jsonl}"
    if [[ "${STRICT_DATASETS}" == "1" || "${STRICT_DATASETS}" == "true" || "${STRICT_DATASETS}" == "True" ]]; then
      exit 1
    fi
    continue
  fi

  ts="$(date +"%m-%d-%H-%M-%S")"
  task_tag="${RUN_TAG}_${task_name}"
  log_file="${BASE_LOG_DIR}/${ts}_${task_tag}.log"

  cmd=(
    "${PYTHON}" -u playground/run_qwen3_matrix.py
    --model_backend "${PRETRAINED_MODEL}"
    --model_path "${PRETRAINED}"
    --datasets "${task_name}=${dataset_jsonl}"
    --methods "${METHODS_CSV}"
    --rates "${RATES_CSV}"
    --tag "${task_tag}"
    --output_dir "${BASE_LOG_DIR}"
    --limit "${LIMIT}"
    --num_frames "${MAX_NUM_FRAMES}"
    --min_pixels "${MIN_PIXELS}"
    --max_pixels "${MAX_PIXELS}"
    --num_warmup "${NUM_WARMUP}"
    --num_runs "${NUM_RUNS}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --attn_implementation "${ATTN_IMPLEMENTATION}"
    --free_ratio "${FREE_RATIO}"
    --min_free_mb "${MIN_FREE_MB}"
    --max_gpus "${MAX_GPUS}"
    --retention_expansion "${RETENTION_EXPANSION}"
    --llm_retention_ratio "${LLM_RETENTION_RATIO}"
    --token_selection_method "${TOKEN_SELECTION_METHOD}"
    --graphvid_token_selection_method "${GRAPHVID_TOKEN_SELECTION_METHOD}"
  )

  if [[ -n "${DURATION_FILTER}" ]]; then
    cmd+=(--duration_filter "${DURATION_FILTER}")
  fi
  if [[ -n "${CUDA_DEVICES}" ]]; then
    cmd+=(--gpu_ids "${CUDA_DEVICES}")
  fi
  if [[ "${RESUME}" == "1" || "${RESUME}" == "true" || "${RESUME}" == "True" ]]; then
    cmd+=(--resume)
  fi

  echo "========================================"
  echo "Running evaluation:"
  echo " Task       : ${task_name}"
  echo " Dataset    : ${dataset_jsonl}"
  echo " Output tag : ${task_tag}"
  echo " Log        : ${log_file}"
  echo "========================================"
  printf '[command]'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  "${cmd[@]}" 2>&1 | tee "${log_file}"

  echo "Completed evaluation for task: ${task_name}"
  echo "----------------------------------------"
done
