#!/usr/bin/env bash
set -euo pipefail

# LLaVA-OneVision lmms-eval runner. Defaults reproduce the FlashVID table
# settings while allowing one method, task, or retention rate to be selected.
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$PWD:$PWD/lmms-eval:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-20480}"
export LMMS_EVAL_SERIALIZE_DATASET_LOAD="${LMMS_EVAL_SERIALIZE_DATASET_LOAD:-1}"

ACCELERATE="${ACCELERATE:-accelerate}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-18888}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"

PRETRAINED="${PRETRAINED:-lmms-lab/llava-onevision-qwen2-7b-ov}"
MODEL_NAME="${MODEL_NAME:-llava_qwen}"
METHODS="${METHODS:-flashvid}"
RATES="${RATES:-0.10,0.15,0.20,0.25}"
TASKS="${TASKS:-videomme,mvbench,longvideobench_val_v,egoschema}"
OUTPUT_PATH="${OUTPUT_PATH:-./logs/llava_onevision}"
LIMIT="${LIMIT:-}"
LOG_SAMPLES="${LOG_SAMPLES:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"

DO_SEGMENT="${DO_SEGMENT:-True}"
SEGMENT_THRESHOLD="${SEGMENT_THRESHOLD:-0.9}"
MIN_SEGMENT_NUM="${MIN_SEGMENT_NUM:-8}"
COMPLEMENTARY_SEGMENT="${COMPLEMENTARY_SEGMENT:-True}"
ALPHA="${ALPHA:-0.70}"
TEMPORAL_THRESHOLD="${TEMPORAL_THRESHOLD:-0.8}"
EXPANSION="${EXPANSION:-1.25}"
PRUNING_LAYER="${PRUNING_LAYER:-20}"
LLM_RETENTION_RATIO="${LLM_RETENTION_RATIO:-0.3}"
STRICT_TOKEN_BUDGET="${STRICT_TOKEN_BUDGET:-False}"
FASTV_PRUNING_LAYER="${FASTV_PRUNING_LAYER:-2}"
ADAPTER_BUDGET_USES_EXPANSION="${ADAPTER_BUDGET_USES_EXPANSION:-False}"
FASTVID_DYSEG_C="${FASTVID_DYSEG_C:-8}"
FASTVID_DYSEG_TAU="${FASTVID_DYSEG_TAU:-0.90}"
FASTVID_STPRUNE_D="${FASTVID_STPRUNE_D:-0.40}"
FASTVID_DTM_P="${FASTVID_DTM_P:-4}"
FASTVID_DTM_BETA="${FASTVID_DTM_BETA:-0.60}"
VISIONZIP_DOMINANT_RATIO="${VISIONZIP_DOMINANT_RATIO:-0.9285714286}"
PRUNEVID_SELECTED_LAYER="${PRUNEVID_SELECTED_LAYER:-10}"
PRUNEVID_TAU="${PRUNEVID_TAU:-0.80}"
PRUNEVID_TEMPORAL_SEGMENT_RATIO="${PRUNEVID_TEMPORAL_SEGMENT_RATIO:-0.25}"
PRUNEVID_CLUSTER_RATIO="${PRUNEVID_CLUSTER_RATIO:-0.50}"

MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-32}"
CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_1_5}"
MM_SPATIAL_POOL_MODE="${MM_SPATIAL_POOL_MODE:-bilinear}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"

FLASHVID_TOKEN_SELECTION_METHOD="${FLASHVID_TOKEN_SELECTION_METHOD:-attn_div_v2}"
CERT_TOKEN_SELECTION_METHOD="${CERT_TOKEN_SELECTION_METHOD:-attn_div_stable}"
CERTV2_TOKEN_SELECTION_METHOD="${CERTV2_TOKEN_SELECTION_METHOD:-attn_div_stable}"

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
CERTV3_QUERY_ATOMS="${CERTV3_QUERY_ATOMS:-8}"
CERTV3_TEMPORAL_BINS="${CERTV3_TEMPORAL_BINS:-12}"
CERTV3_SPATIAL_BINS="${CERTV3_SPATIAL_BINS:-3}"
CERTV3_CANDIDATE_MULTIPLIER="${CERTV3_CANDIDATE_MULTIPLIER:-2.5}"
CERTV3_QUERY_WEIGHT="${CERTV3_QUERY_WEIGHT:-0.18}"
CERTV3_VISUAL_ATTENTION_WEIGHT="${CERTV3_VISUAL_ATTENTION_WEIGHT:-0.28}"
CERTV3_VISUAL_NOVELTY_WEIGHT="${CERTV3_VISUAL_NOVELTY_WEIGHT:-0.20}"
CERTV3_VISUAL_CURVATURE_WEIGHT="${CERTV3_VISUAL_CURVATURE_WEIGHT:-0.14}"
CERTV3_VISUAL_EVENT_WEIGHT="${CERTV3_VISUAL_EVENT_WEIGHT:-0.12}"
CERTV3_VISUAL_DETAIL_WEIGHT="${CERTV3_VISUAL_DETAIL_WEIGHT:-0.12}"
CERTV3_VISUAL_COMPONENT_WEIGHT="${CERTV3_VISUAL_COMPONENT_WEIGHT:-0.14}"
CERTV3_EVENT_NOVELTY_WEIGHT="${CERTV3_EVENT_NOVELTY_WEIGHT:-0.34}"
CERTV3_EVENT_CURVATURE_WEIGHT="${CERTV3_EVENT_CURVATURE_WEIGHT:-0.28}"
CERTV3_EVENT_FRAME_WEIGHT="${CERTV3_EVENT_FRAME_WEIGHT:-0.18}"
CERTV3_EVENT_DETAIL_WEIGHT="${CERTV3_EVENT_DETAIL_WEIGHT:-0.10}"
CERTV3_EVENT_QUERY_WEIGHT="${CERTV3_EVENT_QUERY_WEIGHT:-0.10}"
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
CERTV3_SELECTION_OBJECTIVE="${CERTV3_SELECTION_OBJECTIVE:-d_optimal}"
CERTV3_USE_SPATIOTEMPORAL_CERTIFICATES="${CERTV3_USE_SPATIOTEMPORAL_CERTIFICATES:-True}"
CERTV3_USE_SPATIOTEMPORAL_DESIGN="${CERTV3_USE_SPATIOTEMPORAL_DESIGN:-True}"
CERTV3_USE_TRAJECTORY="${CERTV3_USE_TRAJECTORY:-True}"
CERTV3_USE_QUERY="${CERTV3_USE_QUERY:-True}"
CERTV3_USE_CANDIDATE_POOL="${CERTV3_USE_CANDIDATE_POOL:-True}"




FLASHVID_DIAGNOSTICS_DETAIL="${FLASHVID_DIAGNOSTICS_DETAIL:-summary}"





split_csv() {
  local text="$1"
  text="${text//,/ }"
  # shellcheck disable=SC2086
  printf '%s\n' $text
}

resolve_accelerate_launcher() {
  if command -v "$ACCELERATE" >/dev/null 2>&1; then
    ACCELERATE_LAUNCHER=("$ACCELERATE" launch)
    return
  fi
  if "$PYTHON_BIN" -c 'import accelerate' >/dev/null 2>&1; then
    ACCELERATE_LAUNCHER=("$PYTHON_BIN" -m accelerate.commands.launch)
    return
  fi
  echo "Accelerate is not installed in the active Python environment." >&2
  echo "Install it with: $PYTHON_BIN -m pip install 'accelerate>=0.29.1'" >&2
  exit 127
}

base_model_args() {
  printf 'pretrained=%s,model_name=%s,conv_template=%s,mm_spatial_pool_mode=%s,max_frames_num=%s,attn_implementation=%s' \
    "$PRETRAINED" "$MODEL_NAME" "$CONV_TEMPLATE" "$MM_SPATIAL_POOL_MODE" "$MAX_FRAMES_NUM" "$ATTN_IMPLEMENTATION"
}

common_compression_args() {
  local method="$1"
  local rate="$2"
  local expansion="$EXPANSION"
  local pruning_layer="$PRUNING_LAYER"
  local inner_ratio="$LLM_RETENTION_RATIO"
  case "$method" in
    fastv)
      expansion="1.0"
      pruning_layer="$FASTV_PRUNING_LAYER"
      inner_ratio="$rate"
      ;;
    fastvid|visionzip)
      expansion="1.0"
      inner_ratio="1.0"
      ;;
    prunevid)
      expansion="1.0"
      pruning_layer="$PRUNEVID_SELECTED_LAYER"
      inner_ratio="1.0"
      ;;
  esac
  printf 'enable_flashvid=True,retention_ratio=%s,expansion=%s,do_segment=%s,segment_threshold=%s,min_segment_num=%s,complementary_segment=%s,alpha=%s,temporal_threshold=%s,pruning_layer=%s,llm_retention_ratio=%s' \
    "$rate" "$expansion" "$DO_SEGMENT" "$SEGMENT_THRESHOLD" "$MIN_SEGMENT_NUM" "$COMPLEMENTARY_SEGMENT" "$ALPHA" "$TEMPORAL_THRESHOLD" "$pruning_layer" "$inner_ratio"
}

method_args() {
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
      printf 'compression_variant=flashvid,token_selection_method=%s,strict_token_budget=%s' \
        "$FLASHVID_TOKEN_SELECTION_METHOD" "$STRICT_TOKEN_BUDGET"
      ;;
    certvid)
      printf 'compression_variant=certvid,token_selection_method=%s,cert_budget_uses_expansion=%s,cert_query_atoms=%s,cert_temporal_bins=%s,cert_spatial_bins=%s,cert_candidate_multiplier=%s,cert_query_weight=%s,cert_temporal_weight=%s,cert_detail_weight=%s,cert_repair_ratio=%s,cert_fusion_alpha=%s,cert_assignment_temperature=%s,cert_track_threshold=%s,cert_spatial_penalty=%s,cert_metric_dim=%s' \
        "$CERT_TOKEN_SELECTION_METHOD" "$CERT_BUDGET_USES_EXPANSION" "$CERT_QUERY_ATOMS" "$CERT_TEMPORAL_BINS" "$CERT_SPATIAL_BINS" "$CERT_CANDIDATE_MULTIPLIER" "$CERT_QUERY_WEIGHT" "$CERT_TEMPORAL_WEIGHT" "$CERT_DETAIL_WEIGHT" "$CERT_REPAIR_RATIO" "$CERT_FUSION_ALPHA" "$CERT_ASSIGNMENT_TEMPERATURE" "$CERT_TRACK_THRESHOLD" "$CERT_SPATIAL_PENALTY" "$CERT_METRIC_DIM"
      ;;
    certvid_v2)
      printf 'compression_variant=certvid_v2,token_selection_method=%s,certv2_budget_uses_expansion=%s,certv2_query_atoms=%s,certv2_temporal_bins=%s,certv2_spatial_bins=%s,certv2_candidate_multiplier=%s,certv2_query_weight=%s,certv2_frame_floor_ratio=%s,certv2_diversity_weight=%s,certv2_coverage_weight=%s,certv2_density_neighbors=%s,certv2_track_threshold=%s,certv2_spatial_penalty=%s,certv2_metric_dim=%s,certv2_repair_ratio=%s,certv2_repair_ratio_high=%s,certv2_router_strength=%s,certv2_protect_ratio=%s,certv2_swap_margin=%s,certv2_fusion_alpha=%s,certv2_repair_fusion_alpha=%s,certv2_assignment_temperature=%s' \
        "$CERTV2_TOKEN_SELECTION_METHOD" "$CERTV2_BUDGET_USES_EXPANSION" "$CERTV2_QUERY_ATOMS" "$CERTV2_TEMPORAL_BINS" "$CERTV2_SPATIAL_BINS" "$CERTV2_CANDIDATE_MULTIPLIER" "$CERTV2_QUERY_WEIGHT" "$CERTV2_FRAME_FLOOR_RATIO" "$CERTV2_DIVERSITY_WEIGHT" "$CERTV2_COVERAGE_WEIGHT" "$CERTV2_DENSITY_NEIGHBORS" "$CERTV2_TRACK_THRESHOLD" "$CERTV2_SPATIAL_PENALTY" "$CERTV2_METRIC_DIM" "$CERTV2_REPAIR_RATIO" "$CERTV2_REPAIR_RATIO_HIGH" "$CERTV2_ROUTER_STRENGTH" "$CERTV2_PROTECT_RATIO" "$CERTV2_SWAP_MARGIN" "$CERTV2_FUSION_ALPHA" "$CERTV2_REPAIR_FUSION_ALPHA" "$CERTV2_ASSIGNMENT_TEMPERATURE"
      ;;
    certvid_v3|certvid_v3origin|certvidfinal2)
      local certv3_model_args
      printf -v certv3_model_args 'compression_variant=%s,certv3_budget_uses_expansion=%s,certv3_query_atoms=%s,certv3_temporal_bins=%s,certv3_spatial_bins=%s,certv3_candidate_multiplier=%s,certv3_query_weight=%s,certv3_visual_attention_weight=%s,certv3_visual_novelty_weight=%s,certv3_visual_curvature_weight=%s,certv3_visual_event_weight=%s,certv3_visual_detail_weight=%s,certv3_visual_component_weight=%s,certv3_event_novelty_weight=%s,certv3_event_curvature_weight=%s,certv3_event_frame_weight=%s,certv3_event_detail_weight=%s,certv3_event_query_weight=%s,certv3_track_threshold=%s,certv3_spatial_penalty=%s,certv3_metric_dim=%s,certv3_frame_coverage_ratio=%s,certv3_cell_coverage_ratio=%s,certv3_query_threshold=%s,certv3_query_per_atom=%s,certv3_structural_weight=%s,certv3_whitening_strength=%s,certv3_quality_floor=%s,certv3_ridge=%s,certv3_swap_steps=%s,certv3_swap_pool=%s,certv3_swap_margin=%s,certv3_fusion_alpha=%s,certv3_assignment_temperature=%s,certv3_certificate_budget_ratio=%s,certv3_selection_objective=%s,certv3_use_spatiotemporal_certificates=%s,certv3_use_spatiotemporal_design=%s,certv3_use_trajectory=%s,certv3_use_query=%s,certv3_use_candidate_pool=%s' \
        "$method" "$CERTV3_BUDGET_USES_EXPANSION" "$CERTV3_QUERY_ATOMS" "$CERTV3_TEMPORAL_BINS" "$CERTV3_SPATIAL_BINS" "$CERTV3_CANDIDATE_MULTIPLIER" "$CERTV3_QUERY_WEIGHT" "$CERTV3_VISUAL_ATTENTION_WEIGHT" "$CERTV3_VISUAL_NOVELTY_WEIGHT" "$CERTV3_VISUAL_CURVATURE_WEIGHT" "$CERTV3_VISUAL_EVENT_WEIGHT" "$CERTV3_VISUAL_DETAIL_WEIGHT" "$CERTV3_VISUAL_COMPONENT_WEIGHT" "$CERTV3_EVENT_NOVELTY_WEIGHT" "$CERTV3_EVENT_CURVATURE_WEIGHT" "$CERTV3_EVENT_FRAME_WEIGHT" "$CERTV3_EVENT_DETAIL_WEIGHT" "$CERTV3_EVENT_QUERY_WEIGHT" "$CERTV3_TRACK_THRESHOLD" "$CERTV3_SPATIAL_PENALTY" "$CERTV3_METRIC_DIM" "$CERTV3_FRAME_COVERAGE_RATIO" "$CERTV3_CELL_COVERAGE_RATIO" "$CERTV3_QUERY_THRESHOLD" "$CERTV3_QUERY_PER_ATOM" "$CERTV3_STRUCTURAL_WEIGHT" "$CERTV3_WHITENING_STRENGTH" "$CERTV3_QUALITY_FLOOR" "$CERTV3_RIDGE" "$CERTV3_SWAP_STEPS" "$CERTV3_SWAP_POOL" "$CERTV3_SWAP_MARGIN" "$CERTV3_FUSION_ALPHA" "$CERTV3_ASSIGNMENT_TEMPERATURE" "$CERTV3_CERTIFICATE_BUDGET_RATIO" "$CERTV3_SELECTION_OBJECTIVE" "$CERTV3_USE_SPATIOTEMPORAL_CERTIFICATES" "$CERTV3_USE_SPATIOTEMPORAL_DESIGN" "$CERTV3_USE_TRAJECTORY" "$CERTV3_USE_QUERY" "$CERTV3_USE_CANDIDATE_POOL"
      if [[ "$method" == "certvid_v3origin" || "$method" == "certvidfinal2" ]]; then
        certv3_model_args="${certv3_model_args/,certv3_certificate_budget_ratio=$CERTV3_CERTIFICATE_BUDGET_RATIO/}"
      fi
      if [[ "$method" == "certvidfinal2" ]]; then
        certv3_model_args="${certv3_model_args/,certv3_use_trajectory=$CERTV3_USE_TRAJECTORY/}"
      fi
      printf '%s' "$certv3_model_args"
      ;;
    *)
      echo "Unsupported LLaVA-OneVision method: $method" >&2
      return 1
      ;;
  esac
}

mkdir -p "$OUTPUT_PATH"
resolve_accelerate_launcher

for method in $(split_csv "$METHODS"); do
  for rate in $(split_csv "$RATES"); do
    model_args="$(base_model_args),$(common_compression_args "$method" "$rate"),$(method_args "$method")"
    for task in $(split_csv "$TASKS"); do
      run_output="$OUTPUT_PATH/${method}_r${rate}_${task}"
      cmd=(
        "${ACCELERATE_LAUNCHER[@]}"
        --main_process_port "$MAIN_PROCESS_PORT"
        --num_processes "$NUM_PROCESSES"
        -m lmms_eval
        --model llava_onevision
        --model_args "$model_args"
        --tasks "$task"
        --batch_size "$BATCH_SIZE"
        --output_path "$run_output"
      )
      if [[ -n "$LIMIT" ]]; then
        cmd+=(--limit "$LIMIT")
      fi
      if [[ "$LOG_SAMPLES" == "1" ]]; then
        cmd+=(--log_samples --log_samples_suffix "llava_onevision_${method}_r${rate}")
      fi

      printf '[lmms-eval] method=%s rate=%s task=%s\n' "$method" "$rate" "$task"
      printf '[lmms-eval]'; printf ' %q' "${cmd[@]}"; printf '\n'
      if [[ "$method" == "flashvid" ]]; then
        diagnostics_path="${FLASHVID_DIAGNOSTICS_JSONL:-$run_output/flashvid_diagnostics.jsonl}"
        printf '[flashvid] diagnostics=%s detail=%s\n' \
          "$diagnostics_path" "$FLASHVID_DIAGNOSTICS_DETAIL"
        env \
          FLASHVID_DIAGNOSTICS_JSONL="$diagnostics_path" \
          FLASHVID_DIAGNOSTICS_DETAIL="$FLASHVID_DIAGNOSTICS_DETAIL" \
          "${cmd[@]}"
      else
        "${cmd[@]}"
      fi
      status=$?
      if [[ "$status" -ne 0 ]]; then
        printf '[lmms-eval] FAILED method=%s rate=%s task=%s exit=%s\n' \
          "$method" "$rate" "$task" "$status" >&2
        exit "$status"
      fi
    done
  done
done
