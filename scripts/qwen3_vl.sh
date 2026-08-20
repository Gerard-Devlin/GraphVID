#!/usr/bin/env bash
set -euo pipefail

# Official lmms-eval runner for Qwen3-8B method comparisons.
# This script only maps method names to model_args; task parsers/scorers remain
# the official lmms-eval ones.

cd "$(dirname "$0")/.."

if [[ -z "${HF_HOME:-}" ]]; then
  if [[ -d /home/xuyouwen/hf_home_local ]]; then
    export HF_HOME=/home/xuyouwen/hf_home_local
  else
    export HF_HOME="${HOME}/.cache/huggingface"
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-20480}"
export LMMS_EVAL_FADVISE_DONTNEED="${LMMS_EVAL_FADVISE_DONTNEED:-1}"
if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=1
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="$PWD:$PWD/lmms-eval:${PYTHONPATH:-}"

ACCELERATE="${ACCELERATE:-accelerate}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-18888}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"

if [[ -z "${PRETRAINED:-}" ]]; then
  model_repo="$HF_HUB_CACHE/models--Qwen--Qwen3-VL-8B-Instruct"
  if [[ -f "$model_repo/refs/main" ]]; then
    revision="$(tr -d '\r\n' < "$model_repo/refs/main")"
    PRETRAINED="$(readlink -f "$model_repo/snapshots/$revision")"
  else
    PRETRAINED="Qwen/Qwen3-VL-8B-Instruct"
  fi
fi
METHODS="${METHODS:-flashvid}"
RATES="${RATES:-0.10,0.15,0.20,0.25}"
TASKS="${TASKS:-videomme,egoschema,mvbench,longvideobench_val_v}"
OUTPUT_PATH="${OUTPUT_PATH:-./logs/lmms_eval_qwen3_8b}"
LOG_SAMPLES_SUFFIX="${LOG_SAMPLES_SUFFIX:-qwen3_vl}"
BATCH_SIZE="${BATCH_SIZE:-1}"
# Leave generation settings to each lmms-eval task by default, matching the
# LLaVA-OneVision and Qwen2.5-VL runners. Set GEN_KWARGS to override them.
GEN_KWARGS="${GEN_KWARGS:-}"
VERBOSITY="${VERBOSITY:-INFO}"
LIMIT="${LIMIT:-}"
LOG_SAMPLES="${LOG_SAMPLES:-0}"
FADVISE_DURING_RUN="${FADVISE_DURING_RUN:-1}"
FADVISE_INTERVAL="${FADVISE_INTERVAL:-20}"
FADVISE_ROOTS="${FADVISE_ROOTS:-$HF_HOME/videomme:$HF_HOME/egoschema:$HF_HOME/mvbench_video:$HF_HOME/longvideobench}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MAX_NUM_FRAMES="${MAX_NUM_FRAMES:-32}"
MIN_PIXELS="${MIN_PIXELS:-200704}"
# qwen_vl_utils caps Qwen3 video inputs at 786432 pixels. Using the effective
# limit directly avoids repeated warnings without changing processed inputs.
MAX_PIXELS="${MAX_PIXELS:-786432}"
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
FLASHVID_EXPANSION="${FLASHVID_EXPANSION:-1.25}"
FLASHVID_PRUNING_LAYER="${FLASHVID_PRUNING_LAYER:-28}"
FLASHVID_LLM_RETENTION_RATIO="${FLASHVID_LLM_RETENTION_RATIO:-0.1}"
FASTV_PRUNING_LAYER="${FASTV_PRUNING_LAYER:-2}"
FASTVID_DYSEG_C="${FASTVID_DYSEG_C:-8}"
FASTVID_DYSEG_TAU="${FASTVID_DYSEG_TAU:-0.84}"
FASTVID_STPRUNE_D="${FASTVID_STPRUNE_D:-0.40}"
FASTVID_DTM_P="${FASTVID_DTM_P:-4}"
FASTVID_DTM_BETA="${FASTVID_DTM_BETA:-0.60}"
VISIONZIP_DOMINANT_RATIO="${VISIONZIP_DOMINANT_RATIO:-0.9285714286}"
PRUNEVID_SELECTED_LAYER="${PRUNEVID_SELECTED_LAYER:-10}"
PRUNEVID_TAU="${PRUNEVID_TAU:-0.80}"
PRUNEVID_TEMPORAL_SEGMENT_RATIO="${PRUNEVID_TEMPORAL_SEGMENT_RATIO:-0.25}"
PRUNEVID_CLUSTER_RATIO="${PRUNEVID_CLUSTER_RATIO:-0.50}"

FLASHVID_TOKEN_SELECTION_METHOD="${FLASHVID_TOKEN_SELECTION_METHOD:-attn_div_v2}"
DEFAULT_CERT_TOKEN_SELECTION_METHOD="${DEFAULT_CERT_TOKEN_SELECTION_METHOD:-attn_div_stable}"
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

ADAPTER_BUDGET_USES_EXPANSION="${ADAPTER_BUDGET_USES_EXPANSION:-False}"




CERT_TOKEN_SELECTION_METHOD="${CERT_TOKEN_SELECTION_METHOD:-$DEFAULT_CERT_TOKEN_SELECTION_METHOD}"
CERT_BUDGET_USES_EXPANSION="${CERT_BUDGET_USES_EXPANSION:-True}"
CERT_QUERY_ATOMS="${CERT_QUERY_ATOMS:-6}"
CERT_TEMPORAL_BINS="${CERT_TEMPORAL_BINS:-8}"
CERT_SPATIAL_BINS="${CERT_SPATIAL_BINS:-3}"
CERT_CANDIDATE_MULTIPLIER="${CERT_CANDIDATE_MULTIPLIER:-3.0}"
CERT_QUERY_WEIGHT="${CERT_QUERY_WEIGHT:-0.20}"
CERT_TEMPORAL_WEIGHT="${CERT_TEMPORAL_WEIGHT:-0.20}"
CERT_DETAIL_WEIGHT="${CERT_DETAIL_WEIGHT:-0.10}"
CERT_REPAIR_RATIO="${CERT_REPAIR_RATIO:-0.20}"
CERT_FUSION_ALPHA="${CERT_FUSION_ALPHA:-0.25}"
CERT_ASSIGNMENT_TEMPERATURE="${CERT_ASSIGNMENT_TEMPERATURE:-0.07}"
CERT_TRACK_THRESHOLD="${CERT_TRACK_THRESHOLD:-0.82}"
CERT_SPATIAL_PENALTY="${CERT_SPATIAL_PENALTY:-0.08}"
CERT_METRIC_DIM="${CERT_METRIC_DIM:-256}"

CERTV2_TOKEN_SELECTION_METHOD="${CERTV2_TOKEN_SELECTION_METHOD:-$DEFAULT_CERT_TOKEN_SELECTION_METHOD}"
CERTV2_BUDGET_USES_EXPANSION="${CERTV2_BUDGET_USES_EXPANSION:-True}"
CERTV2_QUERY_ATOMS="${CERTV2_QUERY_ATOMS:-6}"
CERTV2_TEMPORAL_BINS="${CERTV2_TEMPORAL_BINS:-8}"
CERTV2_SPATIAL_BINS="${CERTV2_SPATIAL_BINS:-3}"
CERTV2_CANDIDATE_MULTIPLIER="${CERTV2_CANDIDATE_MULTIPLIER:-3.0}"
CERTV2_QUERY_WEIGHT="${CERTV2_QUERY_WEIGHT:-0.18}"
CERTV2_FRAME_FLOOR_RATIO="${CERTV2_FRAME_FLOOR_RATIO:-0.08}"
CERTV2_DIVERSITY_WEIGHT="${CERTV2_DIVERSITY_WEIGHT:-0.12}"
CERTV2_COVERAGE_WEIGHT="${CERTV2_COVERAGE_WEIGHT:-0.10}"
CERTV2_DENSITY_NEIGHBORS="${CERTV2_DENSITY_NEIGHBORS:-4}"
CERTV2_TRACK_THRESHOLD="${CERTV2_TRACK_THRESHOLD:-0.82}"
CERTV2_SPATIAL_PENALTY="${CERTV2_SPATIAL_PENALTY:-0.08}"
CERTV2_METRIC_DIM="${CERTV2_METRIC_DIM:-256}"
CERTV2_REPAIR_RATIO="${CERTV2_REPAIR_RATIO:-0.05}"
CERTV2_REPAIR_RATIO_HIGH="${CERTV2_REPAIR_RATIO_HIGH:-0.13}"
CERTV2_ROUTER_STRENGTH="${CERTV2_ROUTER_STRENGTH:-0.65}"
CERTV2_PROTECT_RATIO="${CERTV2_PROTECT_RATIO:-0.30}"
CERTV2_SWAP_MARGIN="${CERTV2_SWAP_MARGIN:-0.02}"
CERTV2_FUSION_ALPHA="${CERTV2_FUSION_ALPHA:-0.25}"
CERTV2_REPAIR_FUSION_ALPHA="${CERTV2_REPAIR_FUSION_ALPHA:-0.08}"
CERTV2_ASSIGNMENT_TEMPERATURE="${CERTV2_ASSIGNMENT_TEMPERATURE:-0.07}"

CERTV3_BUDGET_USES_EXPANSION="${CERTV3_BUDGET_USES_EXPANSION:-True}"
CERTV3_EXPANSION="${CERTV3_EXPANSION:-$EXPANSION}"
CERTV3_PRUNING_LAYER="${CERTV3_PRUNING_LAYER:-$PRUNING_LAYER}"
CERTV3_LLM_RETENTION_RATIO="${CERTV3_LLM_RETENTION_RATIO:-$LLM_RETENTION_RATIO}"
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
CERTV3_CERTIFICATE_BUDGET_RATIO="${CERTV3_CERTIFICATE_BUDGET_RATIO:-1.0}"



FLASHVID_DIAGNOSTICS_DETAIL="${FLASHVID_DIAGNOSTICS_DETAIL:-summary}"





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

roots = [Path(p) for p in os.environ.get("FADVISE_ROOTS", "").split(":") if p]
suffixes = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".zip"}
count = 0
for root in roots:
    try:
        exists = root.exists()
    except OSError:
        continue
    if not exists:
        continue
    try:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            try:
                fd = os.open(str(path), os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
                count += 1
            except Exception:
                pass
    except OSError:
        continue
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
  printf 'pretrained=%s,max_num_frames=%s,min_pixels=%s,max_pixels=%s,attn_implementation=%s' \
    "$PRETRAINED" "$MAX_NUM_FRAMES" "$MIN_PIXELS" "$MAX_PIXELS" "$ATTN_IMPLEMENTATION"
}

common_flash_args() {
  local retention_ratio="$1"
  local method="$2"
  local expansion="$EXPANSION"
  local pruning_layer="$PRUNING_LAYER"
  local llm_retention_ratio="$LLM_RETENTION_RATIO"
  if [[ "$method" == "fastv" ]]; then
    expansion="1.0"
    pruning_layer="$FASTV_PRUNING_LAYER"
    llm_retention_ratio="$retention_ratio"
  elif [[ "$method" == "fastvid" || "$method" == "visionzip" ]]; then
    expansion="1.0"
    llm_retention_ratio="1.0"
  elif [[ "$method" == "prunevid" ]]; then
    expansion="1.0"
    pruning_layer="$PRUNEVID_SELECTED_LAYER"
    llm_retention_ratio="1.0"
  elif [[ "$method" == "flashvid" ]]; then
    expansion="$FLASHVID_EXPANSION"
    pruning_layer="$FLASHVID_PRUNING_LAYER"
    llm_retention_ratio="$FLASHVID_LLM_RETENTION_RATIO"
  elif [[ "$method" == "certvid_v3" || "$method" == "certvidfinal" ]]; then
    expansion="$CERTV3_EXPANSION"
    pruning_layer="$CERTV3_PRUNING_LAYER"
    llm_retention_ratio="$CERTV3_LLM_RETENTION_RATIO"
  fi
  printf 'enable_flashvid=True,retention_ratio=%s,expansion=%s,do_segment=%s,segment_threshold=%s,min_segment_num=%s,complementary_segment=%s,alpha=%s,temporal_threshold=%s,pruning_layer=%s,llm_retention_ratio=%s' \
    "$retention_ratio" "$expansion" "$DO_SEGMENT" "$SEGMENT_THRESHOLD" "$MIN_SEGMENT_NUM" "$COMPLEMENTARY_SEGMENT" "$ALPHA" "$TEMPORAL_THRESHOLD" "$pruning_layer" "$llm_retention_ratio"
}

method_flash_args() {
  local method="$1"
  case "$method" in
    fastv)
      printf 'compression_variant=fastv,adapter_budget_uses_expansion=False'
      ;;
    fastvid)
      printf 'compression_variant=fastvid,adapter_budget_uses_expansion=%s,fastvid_DySeg_c=%s,fastvid_DySeg_tau=%s,fastvid_STPrune_d=%s,fastvid_DTM_p=%s,fastvid_DTM_beta=%s' \
        "$ADAPTER_BUDGET_USES_EXPANSION" "$FASTVID_DYSEG_C" "$FASTVID_DYSEG_TAU" "$FASTVID_STPRUNE_D" "$FASTVID_DTM_P" "$FASTVID_DTM_BETA"
      ;;
    visionzip)
      printf 'compression_variant=visionzip,adapter_budget_uses_expansion=%s,visionzip_dominant_ratio=%s' \
        "$ADAPTER_BUDGET_USES_EXPANSION" "$VISIONZIP_DOMINANT_RATIO"
      ;;
    prunevid)
      printf 'compression_variant=prunevid,adapter_budget_uses_expansion=False,prunevid_tau=%s,prunevid_temporal_segment_ratio=%s,prunevid_cluster_ratio=%s' \
        "$PRUNEVID_TAU" "$PRUNEVID_TEMPORAL_SEGMENT_RATIO" "$PRUNEVID_CLUSTER_RATIO"
      ;;
    flashvid)
      printf 'compression_variant=flashvid,token_selection_method=%s' "$FLASHVID_TOKEN_SELECTION_METHOD"
      ;;
    certvid)
      printf 'compression_variant=certvid,token_selection_method=%s,cert_budget_uses_expansion=%s,cert_query_atoms=%s,cert_temporal_bins=%s,cert_spatial_bins=%s,cert_candidate_multiplier=%s,cert_query_weight=%s,cert_temporal_weight=%s,cert_detail_weight=%s,cert_repair_ratio=%s,cert_fusion_alpha=%s,cert_assignment_temperature=%s,cert_track_threshold=%s,cert_spatial_penalty=%s,cert_metric_dim=%s' \
        "$CERT_TOKEN_SELECTION_METHOD" "$CERT_BUDGET_USES_EXPANSION" "$CERT_QUERY_ATOMS" "$CERT_TEMPORAL_BINS" "$CERT_SPATIAL_BINS" "$CERT_CANDIDATE_MULTIPLIER" "$CERT_QUERY_WEIGHT" "$CERT_TEMPORAL_WEIGHT" "$CERT_DETAIL_WEIGHT" "$CERT_REPAIR_RATIO" "$CERT_FUSION_ALPHA" "$CERT_ASSIGNMENT_TEMPERATURE" "$CERT_TRACK_THRESHOLD" "$CERT_SPATIAL_PENALTY" "$CERT_METRIC_DIM"
      ;;
    certvid_v2)
      printf 'compression_variant=certvid_v2,token_selection_method=%s,certv2_budget_uses_expansion=%s,certv2_query_atoms=%s,certv2_temporal_bins=%s,certv2_spatial_bins=%s,certv2_candidate_multiplier=%s,certv2_query_weight=%s,certv2_frame_floor_ratio=%s,certv2_diversity_weight=%s,certv2_coverage_weight=%s,certv2_density_neighbors=%s,certv2_track_threshold=%s,certv2_spatial_penalty=%s,certv2_metric_dim=%s,certv2_repair_ratio=%s,certv2_repair_ratio_high=%s,certv2_router_strength=%s,certv2_protect_ratio=%s,certv2_swap_margin=%s,certv2_fusion_alpha=%s,certv2_repair_fusion_alpha=%s,certv2_assignment_temperature=%s' \
        "$CERTV2_TOKEN_SELECTION_METHOD" "$CERTV2_BUDGET_USES_EXPANSION" "$CERTV2_QUERY_ATOMS" "$CERTV2_TEMPORAL_BINS" "$CERTV2_SPATIAL_BINS" "$CERTV2_CANDIDATE_MULTIPLIER" "$CERTV2_QUERY_WEIGHT" "$CERTV2_FRAME_FLOOR_RATIO" "$CERTV2_DIVERSITY_WEIGHT" "$CERTV2_COVERAGE_WEIGHT" "$CERTV2_DENSITY_NEIGHBORS" "$CERTV2_TRACK_THRESHOLD" "$CERTV2_SPATIAL_PENALTY" "$CERTV2_METRIC_DIM" "$CERTV2_REPAIR_RATIO" "$CERTV2_REPAIR_RATIO_HIGH" "$CERTV2_ROUTER_STRENGTH" "$CERTV2_PROTECT_RATIO" "$CERTV2_SWAP_MARGIN" "$CERTV2_FUSION_ALPHA" "$CERTV2_REPAIR_FUSION_ALPHA" "$CERTV2_ASSIGNMENT_TEMPERATURE"
      ;;
    certvid_v3|certvidfinal)
      printf 'compression_variant=%s,certv3_budget_uses_expansion=%s,certv3_query_atoms=%s,certv3_temporal_bins=%s,certv3_spatial_bins=%s,certv3_candidate_multiplier=%s,certv3_query_weight=%s,certv3_track_threshold=%s,certv3_spatial_penalty=%s,certv3_metric_dim=%s,certv3_frame_coverage_ratio=%s,certv3_cell_coverage_ratio=%s,certv3_query_threshold=%s,certv3_query_per_atom=%s,certv3_structural_weight=%s,certv3_whitening_strength=%s,certv3_quality_floor=%s,certv3_ridge=%s,certv3_swap_steps=%s,certv3_swap_pool=%s,certv3_swap_margin=%s,certv3_fusion_alpha=%s,certv3_assignment_temperature=%s,certv3_certificate_budget_ratio=%s' \
        "$method" "$CERTV3_BUDGET_USES_EXPANSION" "$CERTV3_QUERY_ATOMS" "$CERTV3_TEMPORAL_BINS" "$CERTV3_SPATIAL_BINS" "$CERTV3_CANDIDATE_MULTIPLIER" "$CERTV3_QUERY_WEIGHT" "$CERTV3_TRACK_THRESHOLD" "$CERTV3_SPATIAL_PENALTY" "$CERTV3_METRIC_DIM" "$CERTV3_FRAME_COVERAGE_RATIO" "$CERTV3_CELL_COVERAGE_RATIO" "$CERTV3_QUERY_THRESHOLD" "$CERTV3_QUERY_PER_ATOM" "$CERTV3_STRUCTURAL_WEIGHT" "$CERTV3_WHITENING_STRENGTH" "$CERTV3_QUALITY_FLOOR" "$CERTV3_RIDGE" "$CERTV3_SWAP_STEPS" "$CERTV3_SWAP_POOL" "$CERTV3_SWAP_MARGIN" "$CERTV3_FUSION_ALPHA" "$CERTV3_ASSIGNMENT_TEMPERATURE" "$CERTV3_CERTIFICATE_BUDGET_RATIO"
      ;;
    *)
      echo "Unknown method: $method" >&2
      return 1
      ;;
  esac
}

for retention_ratio in $(split_csv "$RATES"); do
  for method in $(split_csv "$METHODS"); do
    model_args="$(base_model_args),$(common_flash_args "$retention_ratio" "$method"),$(method_flash_args "$method")"
    for task in $(split_csv "$TASKS"); do
      run_output="$OUTPUT_PATH/${method}_r${retention_ratio}_${task}"
      cmd=(
        "$ACCELERATE" launch
        --main_process_port "$MAIN_PROCESS_PORT"
        --num_processes "$NUM_PROCESSES"
        --num_machines 1
        --mixed_precision no
        --dynamo_backend no
        -m lmms_eval
        --model qwen3_vl
        --model_args "$model_args"
        --tasks "$task"
        --batch_size "$BATCH_SIZE"
        --verbosity "$VERBOSITY"
        --output_path "$run_output"
      )
      if [[ -n "$GEN_KWARGS" ]]; then
        cmd+=(--gen_kwargs "$GEN_KWARGS")
      fi
      if [[ "$LOG_SAMPLES" == "1" || "$LOG_SAMPLES" == "true" || "$LOG_SAMPLES" == "True" ]]; then
        cmd+=(
          --log_samples
          --log_samples_suffix "${LOG_SAMPLES_SUFFIX}_${method}_r${retention_ratio}"
        )
      fi
      if [[ -n "$LIMIT" ]]; then
        cmd+=(--limit "$LIMIT")
      fi
      echo "[lmms-eval] method=$method rate=$retention_ratio task=$task"
      echo "[lmms-eval] ${cmd[*]}"
      if [[ "${DRY_RUN:-0}" != "1" ]]; then
        janitor_pid=""
        start_fadvise_loop "${method}_r${retention_ratio}_${task}"
        set +e
        if [[ "$method" == "flashvid" ]]; then
          diagnostics_path="${FLASHVID_DIAGNOSTICS_JSONL:-$run_output/flashvid_diagnostics.jsonl}"
          echo "[flashvid] diagnostics=$diagnostics_path detail=$FLASHVID_DIAGNOSTICS_DETAIL"
          env \
            FLASHVID_DIAGNOSTICS_JSONL="$diagnostics_path" \
            FLASHVID_DIAGNOSTICS_DETAIL="$FLASHVID_DIAGNOSTICS_DETAIL" \
            "${cmd[@]}"
        else
          "${cmd[@]}"
        fi
        code=$?
        stop_fadvise_loop "$janitor_pid"
        set -e
        if [[ "$code" -ne 0 ]]; then
          exit "$code"
        fi
      fi
    done
  done
done
