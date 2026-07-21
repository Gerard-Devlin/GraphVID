#!/usr/bin/env bash
set -euo pipefail

# Official lmms-eval runner for Qwen3-8B method comparisons.
# This script only maps method names to model_args; task parsers/scorers remain
# the official lmms-eval ones.

cd "$(dirname "$0")/.."

if [[ -z "${HF_HOME:-}" ]]; then
  if [[ -d /root/autodl-tmp/hf_home ]]; then
    export HF_HOME=/root/autodl-tmp/hf_home
  else
    export HF_HOME=/gluster/envs/users/wuzhijian/hf_home
  fi
fi
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
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

PRETRAINED="${PRETRAINED:-$HF_HOME/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b}"
METHODS="${METHODS:-flashvid,graphvid,fastvid,fastgraphvid,visionzip}"
RATES="${RATES:-0.10,0.15,0.20,0.25}"
TASKS="${TASKS:-videomme,egoschema,mvbench,longvideobench_val_v}"
OUTPUT_PATH="${OUTPUT_PATH:-./logs/lmms_eval_qwen3_8b}"
LOG_SAMPLES_SUFFIX="${LOG_SAMPLES_SUFFIX:-qwen3_vl}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GEN_KWARGS="${GEN_KWARGS:-max_new_tokens=16,temperature=0}"
LIMIT="${LIMIT:-}"
LOG_SAMPLES="${LOG_SAMPLES:-1}"
FADVISE_DURING_RUN="${FADVISE_DURING_RUN:-1}"
FADVISE_INTERVAL="${FADVISE_INTERVAL:-20}"
FADVISE_ROOTS="${FADVISE_ROOTS:-$HF_HOME/videomme/data:$HF_HOME/videomme:/root/autodl-tmp/videomme_raw}"
PYTHON_BIN="${PYTHON_BIN:-python}"

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
APEX_TOKEN_SELECTION_METHOD="${APEX_TOKEN_SELECTION_METHOD:-$FLASHVID_TOKEN_SELECTION_METHOD}"

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

APEX_EVIDENCE_RATIO="${APEX_EVIDENCE_RATIO:-0.45}"
APEX_EVENT_RATIO="${APEX_EVENT_RATIO:-0.30}"
APEX_MEMORY_RATIO="${APEX_MEMORY_RATIO:-0.25}"
APEX_ROUTER_STRENGTH="${APEX_ROUTER_STRENGTH:-0.50}"
APEX_SUMMARY_TEMPERATURE="${APEX_SUMMARY_TEMPERATURE:-0.07}"
APEX_FRAME_FLOOR_RATIO="${APEX_FRAME_FLOOR_RATIO:-0.35}"
APEX_QUESTION_WEIGHT="${APEX_QUESTION_WEIGHT:-0.20}"

PRISM_TOKEN_SELECTION_METHOD="${PRISM_TOKEN_SELECTION_METHOD:-$GRAPHVID_TOKEN_SELECTION_METHOD}"
PRISM_BUDGET_USES_EXPANSION="${PRISM_BUDGET_USES_EXPANSION:-True}"
PRISM_METRIC_DIM="${PRISM_METRIC_DIM:-256}"
PRISM_QUERY_ATOMS="${PRISM_QUERY_ATOMS:-6}"
PRISM_CANDIDATE_MULTIPLIER="${PRISM_CANDIDATE_MULTIPLIER:-2.25}"
PRISM_PROBE_TOKENS="${PRISM_PROBE_TOKENS:-512}"
PRISM_FRAME_FLOOR_RATIO="${PRISM_FRAME_FLOOR_RATIO:-0.20}"
PRISM_ATTENTION_WEIGHT="${PRISM_ATTENTION_WEIGHT:-0.30}"
PRISM_EVENT_WEIGHT="${PRISM_EVENT_WEIGHT:-0.24}"
PRISM_QUERY_WEIGHT="${PRISM_QUERY_WEIGHT:-0.16}"
PRISM_DISAGREEMENT_WEIGHT="${PRISM_DISAGREEMENT_WEIGHT:-0.16}"
PRISM_ROUTER_STRENGTH="${PRISM_ROUTER_STRENGTH:-0.50}"
PRISM_COVERAGE_WEIGHT="${PRISM_COVERAGE_WEIGHT:-0.68}"
PRISM_PARETO_WEIGHT="${PRISM_PARETO_WEIGHT:-0.20}"
PRISM_BATCH_SIZE="${PRISM_BATCH_SIZE:-8}"

CERT_TOKEN_SELECTION_METHOD="${CERT_TOKEN_SELECTION_METHOD:-$GRAPHVID_TOKEN_SELECTION_METHOD}"
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

CERTV2_TOKEN_SELECTION_METHOD="${CERTV2_TOKEN_SELECTION_METHOD:-$GRAPHVID_TOKEN_SELECTION_METHOD}"
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

CERTV3_TOKEN_SELECTION_METHOD="${CERTV3_TOKEN_SELECTION_METHOD:-$GRAPHVID_TOKEN_SELECTION_METHOD}"
CERTV6_TOKEN_SELECTION_METHOD="${CERTV6_TOKEN_SELECTION_METHOD:-$CERTV3_TOKEN_SELECTION_METHOD}"
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

CERTV6_SCENE_TEMPORAL="${CERTV6_SCENE_TEMPORAL:-True}"
CERTV6_GATE_ENABLED="${CERTV6_GATE_ENABLED:-True}"
CERTV6_CONTINUITY_LOW="${CERTV6_CONTINUITY_LOW:-0.55}"
CERTV6_CONTINUITY_HIGH="${CERTV6_CONTINUITY_HIGH:-0.80}"
CERTV6_QUERY_PER_ATOM_MAX="${CERTV6_QUERY_PER_ATOM_MAX:-3}"

CERTHR_TOKEN_SELECTION_METHOD="${CERTHR_TOKEN_SELECTION_METHOD:-$CERTV3_TOKEN_SELECTION_METHOD}"
CERTHR_HORIZON_GAP_SECONDS="${CERTHR_HORIZON_GAP_SECONDS:-4.0}"
CERTHR_CHUNK_MAX_SECONDS="${CERTHR_CHUNK_MAX_SECONDS:-60.0}"
CERTHR_CHUNK_MAX_UNITS="${CERTHR_CHUNK_MAX_UNITS:-4}"
CERTHR_SEMANTIC_QUANTILE="${CERTHR_SEMANTIC_QUANTILE:-0.85}"
CERTHR_SEMANTIC_FLOOR="${CERTHR_SEMANTIC_FLOOR:-0.10}"
CERTHR_COVERAGE_FLOOR="${CERTHR_COVERAGE_FLOOR:-0.70}"
CERTHR_DEFICIT_THRESHOLD="${CERTHR_DEFICIT_THRESHOLD:-0.05}"
CERTHR_QUERY_PEAK_QUANTILE="${CERTHR_QUERY_PEAK_QUANTILE:-0.90}"
CERTHR_QUERY_PEAK_FLOOR="${CERTHR_QUERY_PEAK_FLOOR:-0.75}"
CERTHR_MAX_SWAP_RATIO="${CERTHR_MAX_SWAP_RATIO:-0.05}"
CERTHR_D_EFFICIENCY_FLOOR="${CERTHR_D_EFFICIENCY_FLOOR:-0.995}"
CERTHR_ADD_POOL="${CERTHR_ADD_POOL:-32}"
CERTHR_REMOVE_POOL="${CERTHR_REMOVE_POOL:-24}"
CERTHR_DEBUG="${CERTHR_DEBUG:-False}"

CERTV4_TOKEN_SELECTION_METHOD="${CERTV4_TOKEN_SELECTION_METHOD:-$GRAPHVID_TOKEN_SELECTION_METHOD}"
CERTV4_EXPANSION="${CERTV4_EXPANSION:-$EXPANSION}"
CERTV4_PRUNING_LAYER="${CERTV4_PRUNING_LAYER:-28}"
CERTV4_LLM_RETENTION_RATIO="${CERTV4_LLM_RETENTION_RATIO:-0.10}"
CERTV4_BUDGET_MODE="${CERTV4_BUDGET_MODE:-layer_average}"
CERTV4_ATTENTION_POLICY="${CERTV4_ATTENTION_POLICY:-validated}"
CERTV4_ATTENTION_EPS="${CERTV4_ATTENTION_EPS:-0.000001}"
CERTV4_CERTIFICATE_BUDGET_RATIO="${CERTV4_CERTIFICATE_BUDGET_RATIO:-0.40}"
CERTV4_QUERY_MODE="${CERTV4_QUERY_MODE:-certificates_and_design}"
CERTV4_DESIGN_PROTECT_RATIO="${CERTV4_DESIGN_PROTECT_RATIO:-0.15}"
CERTV4_QUERY_ATOMS="${CERTV4_QUERY_ATOMS:-8}"
CERTV4_TEMPORAL_BINS="${CERTV4_TEMPORAL_BINS:-12}"
CERTV4_SPATIAL_BINS="${CERTV4_SPATIAL_BINS:-3}"
CERTV4_CANDIDATE_MULTIPLIER="${CERTV4_CANDIDATE_MULTIPLIER:-2.5}"
CERTV4_TRACK_THRESHOLD="${CERTV4_TRACK_THRESHOLD:-0.82}"
CERTV4_SPATIAL_PENALTY="${CERTV4_SPATIAL_PENALTY:-0.08}"
CERTV4_METRIC_DIM="${CERTV4_METRIC_DIM:-96}"
CERTV4_FRAME_COVERAGE_RATIO="${CERTV4_FRAME_COVERAGE_RATIO:-1.0}"
CERTV4_CELL_COVERAGE_RATIO="${CERTV4_CELL_COVERAGE_RATIO:-0.50}"
CERTV4_QUERY_THRESHOLD="${CERTV4_QUERY_THRESHOLD:-0.10}"
CERTV4_QUERY_PER_ATOM="${CERTV4_QUERY_PER_ATOM:-1}"
CERTV4_STRUCTURAL_WEIGHT="${CERTV4_STRUCTURAL_WEIGHT:-0.32}"
CERTV4_WHITENING_STRENGTH="${CERTV4_WHITENING_STRENGTH:-0.50}"
CERTV4_QUALITY_FLOOR="${CERTV4_QUALITY_FLOOR:-0.15}"
CERTV4_RIDGE="${CERTV4_RIDGE:-0.50}"
CERTV4_SWAP_STEPS="${CERTV4_SWAP_STEPS:-6}"
CERTV4_SWAP_POOL="${CERTV4_SWAP_POOL:-24}"
CERTV4_SWAP_MARGIN="${CERTV4_SWAP_MARGIN:-0.0001}"
CERTV4_FUSION_ALPHA="${CERTV4_FUSION_ALPHA:-0.12}"
CERTV4_ASSIGNMENT_TEMPERATURE="${CERTV4_ASSIGNMENT_TEMPERATURE:-0.07}"
CERTV4_DEBUG="${CERTV4_DEBUG:-False}"

CERTV5_TOKEN_SELECTION_METHOD="${CERTV5_TOKEN_SELECTION_METHOD:-$GRAPHVID_TOKEN_SELECTION_METHOD}"
CERTV5_EXPANSION="${CERTV5_EXPANSION:-$EXPANSION}"
CERTV5_PRUNING_LAYER="${CERTV5_PRUNING_LAYER:-28}"
CERTV5_LLM_RETENTION_RATIO="${CERTV5_LLM_RETENTION_RATIO:-0.10}"
CERTV5_BUDGET_MODE="${CERTV5_BUDGET_MODE:-layer_average}"
CERTV5_OT_ENABLED="${CERTV5_OT_ENABLED:-True}"
CERTV5_OT_TOPK="${CERTV5_OT_TOPK:-4}"
CERTV5_OT_TEMPERATURE="${CERTV5_OT_TEMPERATURE:-0.07}"
CERTV5_OT_STEPS="${CERTV5_OT_STEPS:-6}"
CERTV5_OT_CAPACITY_TAU="${CERTV5_OT_CAPACITY_TAU:-0.10}"
CERTV5_OT_PRIOR_SHRINK="${CERTV5_OT_PRIOR_SHRINK:-0.10}"
CERTV5_OT_LIVE_FRACTION="${CERTV5_OT_LIVE_FRACTION:-0.25}"
CERTV5_OT_COST_SLACK="${CERTV5_OT_COST_SLACK:-0.05}"
CERTV5_OT_TEMPORAL_PENALTY="${CERTV5_OT_TEMPORAL_PENALTY:-0.04}"
CERTV5_OT_MAX_DISPLACEMENT="${CERTV5_OT_MAX_DISPLACEMENT:-0.12}"
CERTV5_OT_MIN_COSINE="${CERTV5_OT_MIN_COSINE:-0.98}"
CERTV5_DEBUG="${CERTV5_DEBUG:-False}"

CERTE_TOKEN_SELECTION_METHOD="${CERTE_TOKEN_SELECTION_METHOD:-$GRAPHVID_TOKEN_SELECTION_METHOD}"
CERTE_EXPANSION="${CERTE_EXPANSION:-$EXPANSION}"
CERTE_PRUNING_LAYER="${CERTE_PRUNING_LAYER:-28}"
CERTE_LLM_RETENTION_RATIO="${CERTE_LLM_RETENTION_RATIO:-0.10}"
CERTE_BUDGET_USES_EXPANSION="${CERTE_BUDGET_USES_EXPANSION:-True}"
CERTE_RIDGE="${CERTE_RIDGE:-0.50}"
CERTE_BOTTOM_K="${CERTE_BOTTOM_K:-8}"
CERTE_SWAP_STEPS="${CERTE_SWAP_STEPS:-6}"
CERTE_REMOVE_POOL="${CERTE_REMOVE_POOL:-8}"
CERTE_ADD_POOL="${CERTE_ADD_POOL:-16}"
CERTE_VERIFY_POOL="${CERTE_VERIFY_POOL:-4}"
CERTE_SWAP_MARGIN="${CERTE_SWAP_MARGIN:-0.00001}"
CERTE_SPECTRAL_TEMPERATURE="${CERTE_SPECTRAL_TEMPERATURE:-0.05}"
CERTE_D_EFFICIENCY_FLOOR="${CERTE_D_EFFICIENCY_FLOOR:-0.995}"
CERTE_RANK_TOLERANCE="${CERTE_RANK_TOLERANCE:-0.00001}"
CERTE_DEBUG="${CERTE_DEBUG:-False}"

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
    if not root.exists():
        continue
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
  if [[ "$method" == "certvid_v4" ]]; then
    expansion="$CERTV4_EXPANSION"
    pruning_layer="$CERTV4_PRUNING_LAYER"
    llm_retention_ratio="$CERTV4_LLM_RETENTION_RATIO"
  elif [[ "$method" == "certvid_v5" ]]; then
    expansion="$CERTV5_EXPANSION"
    pruning_layer="$CERTV5_PRUNING_LAYER"
    llm_retention_ratio="$CERTV5_LLM_RETENTION_RATIO"
  elif [[ "$method" == "certvid_e" ]]; then
    expansion="$CERTE_EXPANSION"
    pruning_layer="$CERTE_PRUNING_LAYER"
    llm_retention_ratio="$CERTE_LLM_RETENTION_RATIO"
  fi
  printf 'enable_flashvid=True,retention_ratio=%s,expansion=%s,do_segment=%s,segment_threshold=%s,min_segment_num=%s,complementary_segment=%s,alpha=%s,temporal_threshold=%s,pruning_layer=%s,llm_retention_ratio=%s' \
    "$retention_ratio" "$expansion" "$DO_SEGMENT" "$SEGMENT_THRESHOLD" "$MIN_SEGMENT_NUM" "$COMPLEMENTARY_SEGMENT" "$ALPHA" "$TEMPORAL_THRESHOLD" "$pruning_layer" "$llm_retention_ratio"
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
    apexvid)
      printf 'compression_variant=apexvid,token_selection_method=%s,apex_evidence_ratio=%s,apex_event_ratio=%s,apex_memory_ratio=%s,apex_router_strength=%s,apex_summary_temperature=%s,apex_frame_floor_ratio=%s,apex_question_weight=%s' \
        "$APEX_TOKEN_SELECTION_METHOD" "$APEX_EVIDENCE_RATIO" "$APEX_EVENT_RATIO" "$APEX_MEMORY_RATIO" "$APEX_ROUTER_STRENGTH" "$APEX_SUMMARY_TEMPERATURE" "$APEX_FRAME_FLOOR_RATIO" "$APEX_QUESTION_WEIGHT"
      ;;
    prismvid)
      printf 'compression_variant=prismvid,token_selection_method=%s,prism_budget_uses_expansion=%s,prism_metric_dim=%s,prism_query_atoms=%s,prism_candidate_multiplier=%s,prism_probe_tokens=%s,prism_frame_floor_ratio=%s,prism_attention_weight=%s,prism_event_weight=%s,prism_query_weight=%s,prism_disagreement_weight=%s,prism_router_strength=%s,prism_coverage_weight=%s,prism_pareto_weight=%s,prism_batch_size=%s' \
        "$PRISM_TOKEN_SELECTION_METHOD" "$PRISM_BUDGET_USES_EXPANSION" "$PRISM_METRIC_DIM" "$PRISM_QUERY_ATOMS" "$PRISM_CANDIDATE_MULTIPLIER" "$PRISM_PROBE_TOKENS" "$PRISM_FRAME_FLOOR_RATIO" "$PRISM_ATTENTION_WEIGHT" "$PRISM_EVENT_WEIGHT" "$PRISM_QUERY_WEIGHT" "$PRISM_DISAGREEMENT_WEIGHT" "$PRISM_ROUTER_STRENGTH" "$PRISM_COVERAGE_WEIGHT" "$PRISM_PARETO_WEIGHT" "$PRISM_BATCH_SIZE"
      ;;
    certvid)
      printf 'compression_variant=certvid,token_selection_method=%s,cert_budget_uses_expansion=%s,cert_query_atoms=%s,cert_temporal_bins=%s,cert_spatial_bins=%s,cert_candidate_multiplier=%s,cert_query_weight=%s,cert_temporal_weight=%s,cert_detail_weight=%s,cert_repair_ratio=%s,cert_fusion_alpha=%s,cert_assignment_temperature=%s,cert_track_threshold=%s,cert_spatial_penalty=%s,cert_metric_dim=%s' \
        "$CERT_TOKEN_SELECTION_METHOD" "$CERT_BUDGET_USES_EXPANSION" "$CERT_QUERY_ATOMS" "$CERT_TEMPORAL_BINS" "$CERT_SPATIAL_BINS" "$CERT_CANDIDATE_MULTIPLIER" "$CERT_QUERY_WEIGHT" "$CERT_TEMPORAL_WEIGHT" "$CERT_DETAIL_WEIGHT" "$CERT_REPAIR_RATIO" "$CERT_FUSION_ALPHA" "$CERT_ASSIGNMENT_TEMPERATURE" "$CERT_TRACK_THRESHOLD" "$CERT_SPATIAL_PENALTY" "$CERT_METRIC_DIM"
      ;;
    certvid_v2)
      printf 'compression_variant=certvid_v2,token_selection_method=%s,certv2_budget_uses_expansion=%s,certv2_query_atoms=%s,certv2_temporal_bins=%s,certv2_spatial_bins=%s,certv2_candidate_multiplier=%s,certv2_query_weight=%s,certv2_frame_floor_ratio=%s,certv2_diversity_weight=%s,certv2_coverage_weight=%s,certv2_density_neighbors=%s,certv2_track_threshold=%s,certv2_spatial_penalty=%s,certv2_metric_dim=%s,certv2_repair_ratio=%s,certv2_repair_ratio_high=%s,certv2_router_strength=%s,certv2_protect_ratio=%s,certv2_swap_margin=%s,certv2_fusion_alpha=%s,certv2_repair_fusion_alpha=%s,certv2_assignment_temperature=%s' \
        "$CERTV2_TOKEN_SELECTION_METHOD" "$CERTV2_BUDGET_USES_EXPANSION" "$CERTV2_QUERY_ATOMS" "$CERTV2_TEMPORAL_BINS" "$CERTV2_SPATIAL_BINS" "$CERTV2_CANDIDATE_MULTIPLIER" "$CERTV2_QUERY_WEIGHT" "$CERTV2_FRAME_FLOOR_RATIO" "$CERTV2_DIVERSITY_WEIGHT" "$CERTV2_COVERAGE_WEIGHT" "$CERTV2_DENSITY_NEIGHBORS" "$CERTV2_TRACK_THRESHOLD" "$CERTV2_SPATIAL_PENALTY" "$CERTV2_METRIC_DIM" "$CERTV2_REPAIR_RATIO" "$CERTV2_REPAIR_RATIO_HIGH" "$CERTV2_ROUTER_STRENGTH" "$CERTV2_PROTECT_RATIO" "$CERTV2_SWAP_MARGIN" "$CERTV2_FUSION_ALPHA" "$CERTV2_REPAIR_FUSION_ALPHA" "$CERTV2_ASSIGNMENT_TEMPERATURE"
      ;;
    certvid_v3)
      printf 'compression_variant=certvid_v3,token_selection_method=%s,certv3_budget_uses_expansion=%s,certv3_query_atoms=%s,certv3_temporal_bins=%s,certv3_spatial_bins=%s,certv3_candidate_multiplier=%s,certv3_query_weight=%s,certv3_track_threshold=%s,certv3_spatial_penalty=%s,certv3_metric_dim=%s,certv3_frame_coverage_ratio=%s,certv3_cell_coverage_ratio=%s,certv3_query_threshold=%s,certv3_query_per_atom=%s,certv3_structural_weight=%s,certv3_whitening_strength=%s,certv3_quality_floor=%s,certv3_ridge=%s,certv3_swap_steps=%s,certv3_swap_pool=%s,certv3_swap_margin=%s,certv3_fusion_alpha=%s,certv3_assignment_temperature=%s' \
        "$CERTV3_TOKEN_SELECTION_METHOD" "$CERTV3_BUDGET_USES_EXPANSION" "$CERTV3_QUERY_ATOMS" "$CERTV3_TEMPORAL_BINS" "$CERTV3_SPATIAL_BINS" "$CERTV3_CANDIDATE_MULTIPLIER" "$CERTV3_QUERY_WEIGHT" "$CERTV3_TRACK_THRESHOLD" "$CERTV3_SPATIAL_PENALTY" "$CERTV3_METRIC_DIM" "$CERTV3_FRAME_COVERAGE_RATIO" "$CERTV3_CELL_COVERAGE_RATIO" "$CERTV3_QUERY_THRESHOLD" "$CERTV3_QUERY_PER_ATOM" "$CERTV3_STRUCTURAL_WEIGHT" "$CERTV3_WHITENING_STRENGTH" "$CERTV3_QUALITY_FLOOR" "$CERTV3_RIDGE" "$CERTV3_SWAP_STEPS" "$CERTV3_SWAP_POOL" "$CERTV3_SWAP_MARGIN" "$CERTV3_FUSION_ALPHA" "$CERTV3_ASSIGNMENT_TEMPERATURE"
      ;;
    certvid_v6)
      printf 'compression_variant=certvid_v6,token_selection_method=%s,certv3_budget_uses_expansion=%s,certv3_query_atoms=%s,certv3_temporal_bins=%s,certv3_spatial_bins=%s,certv3_candidate_multiplier=%s,certv3_query_weight=%s,certv3_track_threshold=%s,certv3_spatial_penalty=%s,certv3_metric_dim=%s,certv3_frame_coverage_ratio=%s,certv3_cell_coverage_ratio=%s,certv3_query_threshold=%s,certv3_query_per_atom=%s,certv3_structural_weight=%s,certv3_whitening_strength=%s,certv3_quality_floor=%s,certv3_ridge=%s,certv3_swap_steps=%s,certv3_swap_pool=%s,certv3_swap_margin=%s,certv3_fusion_alpha=%s,certv3_assignment_temperature=%s,certv6_scene_temporal=%s,certv6_gate_enabled=%s,certv6_continuity_low=%s,certv6_continuity_high=%s,certv6_query_per_atom_max=%s' \
        "$CERTV6_TOKEN_SELECTION_METHOD" "$CERTV3_BUDGET_USES_EXPANSION" "$CERTV3_QUERY_ATOMS" "$CERTV3_TEMPORAL_BINS" "$CERTV3_SPATIAL_BINS" "$CERTV3_CANDIDATE_MULTIPLIER" "$CERTV3_QUERY_WEIGHT" "$CERTV3_TRACK_THRESHOLD" "$CERTV3_SPATIAL_PENALTY" "$CERTV3_METRIC_DIM" "$CERTV3_FRAME_COVERAGE_RATIO" "$CERTV3_CELL_COVERAGE_RATIO" "$CERTV3_QUERY_THRESHOLD" "$CERTV3_QUERY_PER_ATOM" "$CERTV3_STRUCTURAL_WEIGHT" "$CERTV3_WHITENING_STRENGTH" "$CERTV3_QUALITY_FLOOR" "$CERTV3_RIDGE" "$CERTV3_SWAP_STEPS" "$CERTV3_SWAP_POOL" "$CERTV3_SWAP_MARGIN" "$CERTV3_FUSION_ALPHA" "$CERTV3_ASSIGNMENT_TEMPERATURE" "$CERTV6_SCENE_TEMPORAL" "$CERTV6_GATE_ENABLED" "$CERTV6_CONTINUITY_LOW" "$CERTV6_CONTINUITY_HIGH" "$CERTV6_QUERY_PER_ATOM_MAX"
      ;;
    certvid_hr)
      printf 'compression_variant=certvid_hr,token_selection_method=%s,certv3_budget_uses_expansion=%s,certv3_query_atoms=%s,certv3_temporal_bins=%s,certv3_spatial_bins=%s,certv3_candidate_multiplier=%s,certv3_query_weight=%s,certv3_track_threshold=%s,certv3_spatial_penalty=%s,certv3_metric_dim=%s,certv3_frame_coverage_ratio=%s,certv3_cell_coverage_ratio=%s,certv3_query_threshold=%s,certv3_query_per_atom=%s,certv3_structural_weight=%s,certv3_whitening_strength=%s,certv3_quality_floor=%s,certv3_ridge=%s,certv3_swap_steps=%s,certv3_swap_pool=%s,certv3_swap_margin=%s,certv3_fusion_alpha=%s,certv3_assignment_temperature=%s,certhr_horizon_gap_seconds=%s,certhr_chunk_max_seconds=%s,certhr_chunk_max_units=%s,certhr_semantic_quantile=%s,certhr_semantic_floor=%s,certhr_coverage_floor=%s,certhr_deficit_threshold=%s,certhr_query_peak_quantile=%s,certhr_query_peak_floor=%s,certhr_max_swap_ratio=%s,certhr_d_efficiency_floor=%s,certhr_add_pool=%s,certhr_remove_pool=%s,certhr_debug=%s' \
        "$CERTHR_TOKEN_SELECTION_METHOD" "$CERTV3_BUDGET_USES_EXPANSION" "$CERTV3_QUERY_ATOMS" "$CERTV3_TEMPORAL_BINS" "$CERTV3_SPATIAL_BINS" "$CERTV3_CANDIDATE_MULTIPLIER" "$CERTV3_QUERY_WEIGHT" "$CERTV3_TRACK_THRESHOLD" "$CERTV3_SPATIAL_PENALTY" "$CERTV3_METRIC_DIM" "$CERTV3_FRAME_COVERAGE_RATIO" "$CERTV3_CELL_COVERAGE_RATIO" "$CERTV3_QUERY_THRESHOLD" "$CERTV3_QUERY_PER_ATOM" "$CERTV3_STRUCTURAL_WEIGHT" "$CERTV3_WHITENING_STRENGTH" "$CERTV3_QUALITY_FLOOR" "$CERTV3_RIDGE" "$CERTV3_SWAP_STEPS" "$CERTV3_SWAP_POOL" "$CERTV3_SWAP_MARGIN" "$CERTV3_FUSION_ALPHA" "$CERTV3_ASSIGNMENT_TEMPERATURE" "$CERTHR_HORIZON_GAP_SECONDS" "$CERTHR_CHUNK_MAX_SECONDS" "$CERTHR_CHUNK_MAX_UNITS" "$CERTHR_SEMANTIC_QUANTILE" "$CERTHR_SEMANTIC_FLOOR" "$CERTHR_COVERAGE_FLOOR" "$CERTHR_DEFICIT_THRESHOLD" "$CERTHR_QUERY_PEAK_QUANTILE" "$CERTHR_QUERY_PEAK_FLOOR" "$CERTHR_MAX_SWAP_RATIO" "$CERTHR_D_EFFICIENCY_FLOOR" "$CERTHR_ADD_POOL" "$CERTHR_REMOVE_POOL" "$CERTHR_DEBUG"
      ;;
    certvid_v4)
      printf 'compression_variant=certvid_v4,token_selection_method=%s,certv4_budget_mode=%s,certv4_attention_policy=%s,certv4_attention_eps=%s,certv4_certificate_budget_ratio=%s,certv4_query_mode=%s,certv4_design_protect_ratio=%s,certv4_query_atoms=%s,certv4_temporal_bins=%s,certv4_spatial_bins=%s,certv4_candidate_multiplier=%s,certv4_track_threshold=%s,certv4_spatial_penalty=%s,certv4_metric_dim=%s,certv4_frame_coverage_ratio=%s,certv4_cell_coverage_ratio=%s,certv4_query_threshold=%s,certv4_query_per_atom=%s,certv4_structural_weight=%s,certv4_whitening_strength=%s,certv4_quality_floor=%s,certv4_ridge=%s,certv4_swap_steps=%s,certv4_swap_pool=%s,certv4_swap_margin=%s,certv4_fusion_alpha=%s,certv4_assignment_temperature=%s,certv4_debug=%s' \
        "$CERTV4_TOKEN_SELECTION_METHOD" "$CERTV4_BUDGET_MODE" "$CERTV4_ATTENTION_POLICY" "$CERTV4_ATTENTION_EPS" "$CERTV4_CERTIFICATE_BUDGET_RATIO" "$CERTV4_QUERY_MODE" "$CERTV4_DESIGN_PROTECT_RATIO" "$CERTV4_QUERY_ATOMS" "$CERTV4_TEMPORAL_BINS" "$CERTV4_SPATIAL_BINS" "$CERTV4_CANDIDATE_MULTIPLIER" "$CERTV4_TRACK_THRESHOLD" "$CERTV4_SPATIAL_PENALTY" "$CERTV4_METRIC_DIM" "$CERTV4_FRAME_COVERAGE_RATIO" "$CERTV4_CELL_COVERAGE_RATIO" "$CERTV4_QUERY_THRESHOLD" "$CERTV4_QUERY_PER_ATOM" "$CERTV4_STRUCTURAL_WEIGHT" "$CERTV4_WHITENING_STRENGTH" "$CERTV4_QUALITY_FLOOR" "$CERTV4_RIDGE" "$CERTV4_SWAP_STEPS" "$CERTV4_SWAP_POOL" "$CERTV4_SWAP_MARGIN" "$CERTV4_FUSION_ALPHA" "$CERTV4_ASSIGNMENT_TEMPERATURE" "$CERTV4_DEBUG"
      ;;
    certvid_v5)
      printf 'compression_variant=certvid_v5,token_selection_method=%s,certv3_budget_uses_expansion=%s,certv3_query_atoms=%s,certv3_temporal_bins=%s,certv3_spatial_bins=%s,certv3_candidate_multiplier=%s,certv3_query_weight=%s,certv3_track_threshold=%s,certv3_spatial_penalty=%s,certv3_metric_dim=%s,certv3_frame_coverage_ratio=%s,certv3_cell_coverage_ratio=%s,certv3_query_threshold=%s,certv3_query_per_atom=%s,certv3_structural_weight=%s,certv3_whitening_strength=%s,certv3_quality_floor=%s,certv3_ridge=%s,certv3_swap_steps=%s,certv3_swap_pool=%s,certv3_swap_margin=%s,certv3_fusion_alpha=%s,certv3_assignment_temperature=%s,certv5_budget_mode=%s,certv5_ot_enabled=%s,certv5_ot_topk=%s,certv5_ot_temperature=%s,certv5_ot_steps=%s,certv5_ot_capacity_tau=%s,certv5_ot_prior_shrink=%s,certv5_ot_live_fraction=%s,certv5_ot_cost_slack=%s,certv5_ot_temporal_penalty=%s,certv5_ot_max_displacement=%s,certv5_ot_min_cosine=%s,certv5_debug=%s' \
        "$CERTV5_TOKEN_SELECTION_METHOD" "$CERTV3_BUDGET_USES_EXPANSION" "$CERTV3_QUERY_ATOMS" "$CERTV3_TEMPORAL_BINS" "$CERTV3_SPATIAL_BINS" "$CERTV3_CANDIDATE_MULTIPLIER" "$CERTV3_QUERY_WEIGHT" "$CERTV3_TRACK_THRESHOLD" "$CERTV3_SPATIAL_PENALTY" "$CERTV3_METRIC_DIM" "$CERTV3_FRAME_COVERAGE_RATIO" "$CERTV3_CELL_COVERAGE_RATIO" "$CERTV3_QUERY_THRESHOLD" "$CERTV3_QUERY_PER_ATOM" "$CERTV3_STRUCTURAL_WEIGHT" "$CERTV3_WHITENING_STRENGTH" "$CERTV3_QUALITY_FLOOR" "$CERTV3_RIDGE" "$CERTV3_SWAP_STEPS" "$CERTV3_SWAP_POOL" "$CERTV3_SWAP_MARGIN" "$CERTV3_FUSION_ALPHA" "$CERTV3_ASSIGNMENT_TEMPERATURE" "$CERTV5_BUDGET_MODE" "$CERTV5_OT_ENABLED" "$CERTV5_OT_TOPK" "$CERTV5_OT_TEMPERATURE" "$CERTV5_OT_STEPS" "$CERTV5_OT_CAPACITY_TAU" "$CERTV5_OT_PRIOR_SHRINK" "$CERTV5_OT_LIVE_FRACTION" "$CERTV5_OT_COST_SLACK" "$CERTV5_OT_TEMPORAL_PENALTY" "$CERTV5_OT_MAX_DISPLACEMENT" "$CERTV5_OT_MIN_COSINE" "$CERTV5_DEBUG"
      ;;
    certvid_e)
      printf 'compression_variant=certvid_e,token_selection_method=%s,certv3_query_atoms=%s,certv3_temporal_bins=%s,certv3_spatial_bins=%s,certv3_candidate_multiplier=%s,certv3_query_weight=%s,certv3_track_threshold=%s,certv3_spatial_penalty=%s,certv3_metric_dim=%s,certv3_frame_coverage_ratio=%s,certv3_cell_coverage_ratio=%s,certv3_query_threshold=%s,certv3_query_per_atom=%s,certv3_structural_weight=%s,certv3_whitening_strength=%s,certv3_quality_floor=%s,certv3_ridge=%s,certv3_swap_steps=%s,certv3_swap_pool=%s,certv3_swap_margin=%s,certv3_fusion_alpha=%s,certv3_assignment_temperature=%s,certe_budget_uses_expansion=%s,certe_ridge=%s,certe_bottom_k=%s,certe_swap_steps=%s,certe_remove_pool=%s,certe_add_pool=%s,certe_verify_pool=%s,certe_swap_margin=%s,certe_spectral_temperature=%s,certe_d_efficiency_floor=%s,certe_rank_tolerance=%s,certe_debug=%s' \
        "$CERTE_TOKEN_SELECTION_METHOD" "$CERTV3_QUERY_ATOMS" "$CERTV3_TEMPORAL_BINS" "$CERTV3_SPATIAL_BINS" "$CERTV3_CANDIDATE_MULTIPLIER" "$CERTV3_QUERY_WEIGHT" "$CERTV3_TRACK_THRESHOLD" "$CERTV3_SPATIAL_PENALTY" "$CERTV3_METRIC_DIM" "$CERTV3_FRAME_COVERAGE_RATIO" "$CERTV3_CELL_COVERAGE_RATIO" "$CERTV3_QUERY_THRESHOLD" "$CERTV3_QUERY_PER_ATOM" "$CERTV3_STRUCTURAL_WEIGHT" "$CERTV3_WHITENING_STRENGTH" "$CERTV3_QUALITY_FLOOR" "$CERTV3_RIDGE" "$CERTV3_SWAP_STEPS" "$CERTV3_SWAP_POOL" "$CERTV3_SWAP_MARGIN" "$CERTV3_FUSION_ALPHA" "$CERTV3_ASSIGNMENT_TEMPERATURE" "$CERTE_BUDGET_USES_EXPANSION" "$CERTE_RIDGE" "$CERTE_BOTTOM_K" "$CERTE_SWAP_STEPS" "$CERTE_REMOVE_POOL" "$CERTE_ADD_POOL" "$CERTE_VERIFY_POOL" "$CERTE_SWAP_MARGIN" "$CERTE_SPECTRAL_TEMPERATURE" "$CERTE_D_EFFICIENCY_FLOOR" "$CERTE_RANK_TOLERANCE" "$CERTE_DEBUG"
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
    model_args="$(base_model_args),$(common_flash_args "$retention_ratio" "$method"),$(method_flash_args "$method")"
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
        --output_path "$OUTPUT_PATH"
      )
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
        "${cmd[@]}"
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
