#!/usr/bin/env bash
set -euo pipefail

# Qwen2.5-VL runner for official baseline adapters and CertVID.
# Task prompts, parsers, and metrics are provided by lmms-eval unchanged.

cd "$(dirname "$0")/.."

if [[ -z "${HF_HOME:-}" ]]; then
  if [[ -d /home/xuyouwen/hf_home_local ]]; then
    export HF_HOME=/home/xuyouwen/hf_home_local
  elif [[ -d /root/autodl-tmp/hf_home ]]; then
    export HF_HOME=/root/autodl-tmp/hf_home
  else
    export HF_HOME=/gluster/envs/users/xuyouwen/hf_home
  fi
fi
if [[ -z "${HF_HUB_CACHE:-}" ]]; then
  if [[ -d /home/xuyouwen/hf_hub_local ]]; then
    export HF_HUB_CACHE=/home/xuyouwen/hf_hub_local
  else
    export HF_HUB_CACHE="$HF_HOME/hub"
  fi
fi
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_EVALUATE_OFFLINE="${HF_EVALUATE_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export LMMS_EVAL_SERIALIZE_DATASET_LOAD="${LMMS_EVAL_SERIALIZE_DATASET_LOAD:-1}"
export LMMS_EVAL_FADVISE_DONTNEED="${LMMS_EVAL_FADVISE_DONTNEED:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-20480}"
if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="$PWD:$PWD/lmms-eval:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"
ACCELERATE="${ACCELERATE:-accelerate}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-18888}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"

if [[ -z "${PRETRAINED:-}" ]]; then
  model_repo="$HF_HUB_CACHE/models--Qwen--Qwen2.5-VL-7B-Instruct"
  if [[ -f "$model_repo/refs/main" ]]; then
    revision="$(tr -d '\r\n' < "$model_repo/refs/main")"
    PRETRAINED="$(readlink -f "$model_repo/snapshots/$revision")"
  else
    PRETRAINED="Qwen/Qwen2.5-VL-7B-Instruct"
  fi
fi

METHODS="${METHODS:-fastv,prunevid,visionzip,fastvid,flashvid,certvid_v3}"
RATES="${RATES:-0.10,0.15,0.20,0.25}"
TASKS="${TASKS:-videomme,egoschema_subset,mvbench,longvideobench_val_v}"
OUTPUT_PATH="${OUTPUT_PATH:-./logs/lmms_eval_qwen2_5_vl}"
LOG_SAMPLES_SUFFIX="${LOG_SAMPLES_SUFFIX:-qwen2_5_vl}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GEN_KWARGS="${GEN_KWARGS:-max_new_tokens=16,temperature=0}"
LIMIT="${LIMIT:-}"
LOG_SAMPLES="${LOG_SAMPLES:-0}"

MAX_NUM_FRAMES="${MAX_NUM_FRAMES:-32}"
MIN_PIXELS="${MIN_PIXELS:-200704}"
MAX_PIXELS="${MAX_PIXELS:-1605632}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"

DO_SEGMENT="${DO_SEGMENT:-True}"
SEGMENT_THRESHOLD="${SEGMENT_THRESHOLD:-0.9}"
MIN_SEGMENT_NUM="${MIN_SEGMENT_NUM:-8}"
COMPLEMENTARY_SEGMENT="${COMPLEMENTARY_SEGMENT:-True}"
ALPHA="${ALPHA:-0.70}"
TEMPORAL_THRESHOLD="${TEMPORAL_THRESHOLD:-0.8}"

# FlashVID's released Qwen2.5 token selector and fair layer-average schedule.
FLASHVID_EXPANSION="${FLASHVID_EXPANSION:-1.25}"
FLASHVID_PRUNING_LAYER="${FLASHVID_PRUNING_LAYER:-20}"
FLASHVID_LLM_RETENTION_RATIO="${FLASHVID_LLM_RETENTION_RATIO:-0.3}"
FLASHVID_TOKEN_SELECTION_METHOD="${FLASHVID_TOKEN_SELECTION_METHOD:-attn_div}"

# CertVID V3 defaults to the strongest tested 28-layer schedule.
CERTV3_EXPANSION="${CERTV3_EXPANSION:-1.30}"
CERTV3_PRUNING_LAYER="${CERTV3_PRUNING_LAYER:-20}"
CERTV3_LLM_RETENTION_RATIO="${CERTV3_LLM_RETENTION_RATIO:-0.1923076923}"
CERTV3_TOKEN_SELECTION_METHOD="${CERTV3_TOKEN_SELECTION_METHOD:-attn_div_stable}"
CERTV3_BUDGET_USES_EXPANSION="${CERTV3_BUDGET_USES_EXPANSION:-True}"
CERTV3_QUERY_ATOMS="${CERTV3_QUERY_ATOMS:-8}"
CERTV3_TEMPORAL_BINS="${CERTV3_TEMPORAL_BINS:-12}"
CERTV3_SPATIAL_BINS="${CERTV3_SPATIAL_BINS:-3}"
CERTV3_CANDIDATE_MULTIPLIER="${CERTV3_CANDIDATE_MULTIPLIER:-2.5}"
CERTV3_QUERY_WEIGHT="${CERTV3_QUERY_WEIGHT:-0.18}"
CERTV3_TRACK_THRESHOLD="${CERTV3_TRACK_THRESHOLD:-0.82}"
CERTV3_SPATIAL_PENALTY="${CERTV3_SPATIAL_PENALTY:-0.08}"
CERTV3_METRIC_DIM="${CERTV3_METRIC_DIM:-96}"
CERTV3_FRAME_COVERAGE_RATIO="${CERTV3_FRAME_COVERAGE_RATIO:-1.0}"
CERTV3_CELL_COVERAGE_RATIO="${CERTV3_CELL_COVERAGE_RATIO:-0.50}"
CERTV3_QUERY_THRESHOLD="${CERTV3_QUERY_THRESHOLD:-0.10}"
CERTV3_QUERY_PER_ATOM="${CERTV3_QUERY_PER_ATOM:-1}"
CERTV3_STRUCTURAL_WEIGHT="${CERTV3_STRUCTURAL_WEIGHT:-0.32}"
CERTV3_WHITENING_STRENGTH="${CERTV3_WHITENING_STRENGTH:-0.50}"
CERTV3_QUALITY_FLOOR="${CERTV3_QUALITY_FLOOR:-0.15}"
CERTV3_RIDGE="${CERTV3_RIDGE:-0.50}"
CERTV3_SWAP_STEPS="${CERTV3_SWAP_STEPS:-6}"
CERTV3_SWAP_POOL="${CERTV3_SWAP_POOL:-24}"
CERTV3_SWAP_MARGIN="${CERTV3_SWAP_MARGIN:-0.0001}"
CERTV3_FUSION_ALPHA="${CERTV3_FUSION_ALPHA:-0.12}"
CERTV3_ASSIGNMENT_TEMPERATURE="${CERTV3_ASSIGNMENT_TEMPERATURE:-0.07}"

FASTV_PRUNING_LAYER="${FASTV_PRUNING_LAYER:-2}"
OUTER_ONLY_PRUNING_LAYER="${OUTER_ONLY_PRUNING_LAYER:-999}"
FASTVID_DYSEG_C="${FASTVID_DYSEG_C:-8}"
FASTVID_DYSEG_TAU="${FASTVID_DYSEG_TAU:-0.84}"
FASTVID_DYSEG_IGNORE="${FASTVID_DYSEG_IGNORE:-0.95}"
FASTVID_STPRUNE_D="${FASTVID_STPRUNE_D:-0.40}"
FASTVID_DTM_P="${FASTVID_DTM_P:-4}"
FASTVID_DTM_BETA="${FASTVID_DTM_BETA:-0.60}"
VISIONZIP_DOMINANT_RATIO="${VISIONZIP_DOMINANT_RATIO:-0.9285714286}"
PRUNEVID_SELECTED_LAYER="${PRUNEVID_SELECTED_LAYER:-10}"
PRUNEVID_TAU="${PRUNEVID_TAU:-0.80}"
PRUNEVID_TEMPORAL_SEGMENT_RATIO="${PRUNEVID_TEMPORAL_SEGMENT_RATIO:-0.25}"
PRUNEVID_CLUSTER_RATIO="${PRUNEVID_CLUSTER_RATIO:-0.50}"

FADVISE_DURING_RUN="${FADVISE_DURING_RUN:-1}"
FADVISE_INTERVAL="${FADVISE_INTERVAL:-20}"
FADVISE_ROOTS="${FADVISE_ROOTS:-$HF_HOME/videomme:$HF_HOME/egoschema:$HF_HOME/mvbench_video:$HF_HOME/longvideobench}"

split_csv() {
  local text="$1"
  text="${text//,/ }"
  # shellcheck disable=SC2086
  printf '%s\n' $text
}

fadvise_cached_video_files() {
  FADVISE_ROOTS="$FADVISE_ROOTS" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

roots = [Path(path) for path in os.environ.get("FADVISE_ROOTS", "").split(":") if path]
suffixes = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
count = 0
for root in roots:
    if not root.exists():
        continue
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        try:
            descriptor = os.open(str(path), os.O_RDONLY)
            try:
                os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(descriptor)
            count += 1
        except Exception:
            pass
print(f"[fadvise-loop] files={count}", flush=True)
PY
}

janitor_pid=""
start_fadvise_loop() {
  local label="$1"
  janitor_pid=""
  case "$FADVISE_DURING_RUN" in
    1|true|True|yes|Yes) ;;
    *) return 0 ;;
  esac
  (
    while true; do
      sleep "$FADVISE_INTERVAL" || exit 0
      printf '[fadvise-loop] label=%s time=%s\n' "$label" "$(date '+%F %T')" >&2
      fadvise_cached_video_files >&2 || true
    done
  ) &
  janitor_pid="$!"
}

stop_fadvise_loop() {
  local pid="${1:-}"
  if [[ -n "$pid" ]]; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}
trap 'stop_fadvise_loop "${janitor_pid:-}"' EXIT INT TERM

base_model_args() {
  local args="pretrained=$PRETRAINED,max_num_frames=$MAX_NUM_FRAMES,attn_implementation=$ATTN_IMPLEMENTATION"
  if [[ -n "$MIN_PIXELS" ]]; then
    args+=",min_pixels=$MIN_PIXELS"
  fi
  if [[ -n "$MAX_PIXELS" ]]; then
    args+=",max_pixels=$MAX_PIXELS"
  fi
  printf '%s' "$args"
}

common_flash_args() {
  local retention_ratio="$1"
  local method="$2"
  local expansion="1.0"
  local pruning_layer="$OUTER_ONLY_PRUNING_LAYER"
  local llm_retention_ratio="1.0"

  case "$method" in
    fastv)
      pruning_layer="$FASTV_PRUNING_LAYER"
      llm_retention_ratio="$retention_ratio"
      ;;
    prunevid)
      pruning_layer="$PRUNEVID_SELECTED_LAYER"
      ;;
    flashvid)
      expansion="$FLASHVID_EXPANSION"
      pruning_layer="$FLASHVID_PRUNING_LAYER"
      llm_retention_ratio="$FLASHVID_LLM_RETENTION_RATIO"
      ;;
    certvid_v3)
      expansion="$CERTV3_EXPANSION"
      pruning_layer="$CERTV3_PRUNING_LAYER"
      llm_retention_ratio="$CERTV3_LLM_RETENTION_RATIO"
      ;;
    fastvid|visionzip)
      ;;
    *)
      echo "Unsupported Qwen2.5 method: $method" >&2
      return 1
      ;;
  esac

  printf 'enable_flashvid=True,retention_ratio=%s,expansion=%s,do_segment=%s,segment_threshold=%s,min_segment_num=%s,complementary_segment=%s,alpha=%s,temporal_threshold=%s,pruning_layer=%s,llm_retention_ratio=%s' \
    "$retention_ratio" "$expansion" "$DO_SEGMENT" "$SEGMENT_THRESHOLD" \
    "$MIN_SEGMENT_NUM" "$COMPLEMENTARY_SEGMENT" "$ALPHA" \
    "$TEMPORAL_THRESHOLD" "$pruning_layer" "$llm_retention_ratio"
}

method_flash_args() {
  local method="$1"
  case "$method" in
    fastv)
      printf 'compression_variant=fastv,adapter_budget_uses_expansion=False'
      ;;
    fastvid)
      printf 'compression_variant=fastvid,adapter_budget_uses_expansion=False,fastvid_DySeg_c=%s,fastvid_DySeg_tau=%s,fastvid_DySeg_ignore=%s,fastvid_STPrune_d=%s,fastvid_DTM_p=%s,fastvid_DTM_beta=%s' \
        "$FASTVID_DYSEG_C" "$FASTVID_DYSEG_TAU" "$FASTVID_DYSEG_IGNORE" \
        "$FASTVID_STPRUNE_D" "$FASTVID_DTM_P" "$FASTVID_DTM_BETA"
      ;;
    visionzip)
      printf 'compression_variant=visionzip,adapter_budget_uses_expansion=False,visionzip_dominant_ratio=%s' \
        "$VISIONZIP_DOMINANT_RATIO"
      ;;
    prunevid)
      printf 'compression_variant=prunevid,adapter_budget_uses_expansion=False,prunevid_tau=%s,prunevid_temporal_segment_ratio=%s,prunevid_cluster_ratio=%s' \
        "$PRUNEVID_TAU" "$PRUNEVID_TEMPORAL_SEGMENT_RATIO" "$PRUNEVID_CLUSTER_RATIO"
      ;;
    flashvid)
      printf 'compression_variant=flashvid,token_selection_method=%s' \
        "$FLASHVID_TOKEN_SELECTION_METHOD"
      ;;
    certvid_v3)
      printf 'compression_variant=certvid_v3,token_selection_method=%s,certv3_budget_uses_expansion=%s,certv3_query_atoms=%s,certv3_temporal_bins=%s,certv3_spatial_bins=%s,certv3_candidate_multiplier=%s,certv3_query_weight=%s,certv3_track_threshold=%s,certv3_spatial_penalty=%s,certv3_metric_dim=%s,certv3_frame_coverage_ratio=%s,certv3_cell_coverage_ratio=%s,certv3_query_threshold=%s,certv3_query_per_atom=%s,certv3_structural_weight=%s,certv3_whitening_strength=%s,certv3_quality_floor=%s,certv3_ridge=%s,certv3_swap_steps=%s,certv3_swap_pool=%s,certv3_swap_margin=%s,certv3_fusion_alpha=%s,certv3_assignment_temperature=%s' \
        "$CERTV3_TOKEN_SELECTION_METHOD" "$CERTV3_BUDGET_USES_EXPANSION" \
        "$CERTV3_QUERY_ATOMS" "$CERTV3_TEMPORAL_BINS" "$CERTV3_SPATIAL_BINS" \
        "$CERTV3_CANDIDATE_MULTIPLIER" "$CERTV3_QUERY_WEIGHT" \
        "$CERTV3_TRACK_THRESHOLD" "$CERTV3_SPATIAL_PENALTY" "$CERTV3_METRIC_DIM" \
        "$CERTV3_FRAME_COVERAGE_RATIO" "$CERTV3_CELL_COVERAGE_RATIO" \
        "$CERTV3_QUERY_THRESHOLD" "$CERTV3_QUERY_PER_ATOM" \
        "$CERTV3_STRUCTURAL_WEIGHT" "$CERTV3_WHITENING_STRENGTH" \
        "$CERTV3_QUALITY_FLOOR" "$CERTV3_RIDGE" "$CERTV3_SWAP_STEPS" \
        "$CERTV3_SWAP_POOL" "$CERTV3_SWAP_MARGIN" "$CERTV3_FUSION_ALPHA" \
        "$CERTV3_ASSIGNMENT_TEMPERATURE"
      ;;
    *)
      echo "Unsupported Qwen2.5 method: $method" >&2
      return 1
      ;;
  esac
}

mkdir -p "$OUTPUT_PATH"

for method in $(split_csv "$METHODS"); do
  for retention_ratio in $(split_csv "$RATES"); do
    model_args="$(base_model_args),$(common_flash_args "$retention_ratio" "$method"),$(method_flash_args "$method")"
    for task in $(split_csv "$TASKS"); do
      cmd=(
        "$ACCELERATE" launch
        --main_process_port "$MAIN_PROCESS_PORT"
        --num_processes "$NUM_PROCESSES"
        --num_machines 1
        --mixed_precision no
        --dynamo_backend no
        -m lmms_eval
        --model qwen2_5_vl
        --model_args "$model_args"
        --tasks "$task"
        --batch_size "$BATCH_SIZE"
        --gen_kwargs "$GEN_KWARGS"
        --output_path "$OUTPUT_PATH"
      )
      if [[ "$LOG_SAMPLES" == "1" || "$LOG_SAMPLES" == "true" || "$LOG_SAMPLES" == "True" ]]; then
        cmd+=(--log_samples --log_samples_suffix "${LOG_SAMPLES_SUFFIX}_${method}_r${retention_ratio}")
      fi
      if [[ -n "$LIMIT" ]]; then
        cmd+=(--limit "$LIMIT")
      fi

      echo "[lmms-eval] method=$method rate=$retention_ratio task=$task"
      echo "[lmms-eval] ${cmd[*]}"
      if [[ "${DRY_RUN:-0}" == "1" ]]; then
        continue
      fi

      janitor_pid=""
      start_fadvise_loop "${method}_r${retention_ratio}_${task}"
      set +e
      "${cmd[@]}"
      code=$?
      set -e
      stop_fadvise_loop "$janitor_pid"
      janitor_pid=""
      if [[ "$code" -ne 0 ]]; then
        exit "$code"
      fi
    done
  done
done
