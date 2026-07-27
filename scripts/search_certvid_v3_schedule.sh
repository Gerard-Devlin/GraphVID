#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

STAGE="${STAGE:-coarse}"
TOTAL_LAYERS="${TOTAL_LAYERS:-28}"
MODEL="${PRETRAINED:-/home/xuyouwen/models/llava-onevision-qwen2-7b-ov}"
PYTHON_BIN="${PYTHON_BIN:-/home/xuyouwen/.conda/envs/graphvid311/bin/python}"
ACCELERATE="${ACCELERATE:-/home/xuyouwen/.conda/envs/graphvid311/bin/accelerate}"
OUTPUT_ROOT="${OUTPUT_PATH:-$REPO_ROOT/logs/v3_schedule_${STAGE}_$(date +%Y%m%d_%H%M%S)}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
PROCESSES="${NUM_PROCESSES:-6}"
PORT="${MAIN_PROCESS_PORT:-18960}"
TASKS="${TASKS:-videomme,mvbench,longvideobench_val_v}"
RATES="${RATES:-0.10}"
LIMIT="${LIMIT-100}"
SAMPLE_IDS_FILE="${SAMPLE_IDS_FILE:-}"
SAMPLE_ID_FIELD="${SAMPLE_ID_FIELD:-id}"
SLEEP_SECONDS="${SLEEP_SECONDS:-10}"
FAIL_FAST="${FAIL_FAST:-1}"
DRY_RUN="${DRY_RUN:-0}"

case "$STAGE" in
  quick)
    DEFAULT_LAYERS="12,16,20"
    DEFAULT_EXPANSIONS="1.20,1.25,1.30"
    ;;
  coarse)
    DEFAULT_LAYERS="8,12,16,20,24"
    DEFAULT_EXPANSIONS="1.15,1.20,1.25,1.30"
    ;;
  full)
    DEFAULT_LAYERS="8,12,16,20,24"
    DEFAULT_EXPANSIONS="1.10,1.15,1.20,1.225,1.25,1.275,1.30,1.325,1.35"
    ;;
  custom)
    DEFAULT_LAYERS="8,12,16,20"
    DEFAULT_EXPANSIONS="1.20,1.25,1.30"
    ;;
  *)
    echo "Unknown STAGE=$STAGE (expected quick, coarse, full, or custom)" >&2
    exit 1
    ;;
esac

PRUNING_LAYERS="${PRUNING_LAYERS:-$DEFAULT_LAYERS}"
EXPANSIONS="${EXPANSIONS:-$DEFAULT_EXPANSIONS}"

if [[ "$PYTHON_BIN" == */* && ! -x "$PYTHON_BIN" ]] \
  || [[ "$PYTHON_BIN" != */* && -z "$(command -v "$PYTHON_BIN" 2>/dev/null)" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ "$DRY_RUN" != "1" && "$ACCELERATE" == */* && ! -x "$ACCELERATE" ]] \
  || [[ "$DRY_RUN" != "1" && "$ACCELERATE" != */* && -z "$(command -v "$ACCELERATE" 2>/dev/null)" ]]; then
  echo "Accelerate executable not found: $ACCELERATE" >&2
  exit 1
fi
if [[ -n "$SAMPLE_IDS_FILE" ]]; then
  if [[ ! -f "$SAMPLE_IDS_FILE" ]]; then
    echo "Sample-id file not found: $SAMPLE_IDS_FILE" >&2
    exit 1
  fi
  SAMPLE_IDS_FILE="$(cd "$(dirname "$SAMPLE_IDS_FILE")" && pwd)/$(basename "$SAMPLE_IDS_FILE")"
  if [[ -n "$LIMIT" ]]; then
    echo "Fixed sample IDs supplied; disabling LIMIT=$LIMIT."
    LIMIT=""
  fi
fi

export HF_HOME="${HF_HOME:-/gluster/envs/users/xuyouwen/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_EVALUATE_OFFLINE="${HF_EVALUATE_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export LMMS_EVAL_SERIALIZE_DATASET_LOAD="${LMMS_EVAL_SERIALIZE_DATASET_LOAD:-1}"

mkdir -p "$OUTPUT_ROOT"
MATRIX_PATH="$OUTPUT_ROOT/search_matrix.tsv"
SKIPPED_PATH="$OUTPUT_ROOT/skipped_invalid.tsv"
SUMMARY_PATH="$OUTPUT_ROOT/v3_schedule_summary.csv"
if [[ -n "$SAMPLE_IDS_FILE" ]]; then
  cp "$SAMPLE_IDS_FILE" "$OUTPUT_ROOT/selected_sample_ids.txt"
fi

printf 'name\touter_layers\tinner_layers\texpansion\tinner_retention\taverage_multiplier\n' \
  > "$MATRIX_PATH"
printf 'outer_layers\tinner_layers\texpansion\treason\n' > "$SKIPPED_PATH"

IFS=',' read -r -a LAYER_VALUES <<< "$PRUNING_LAYERS"
IFS=',' read -r -a EXPANSION_VALUES <<< "$EXPANSIONS"
IFS=',' read -r -a TASK_VALUES <<< "$TASKS"
IFS=',' read -r -a RATE_VALUES <<< "$RATES"
EXPECTED_RESULTS=$(( ${#TASK_VALUES[@]} * ${#RATE_VALUES[@]} ))
MAX_RATE="$("$PYTHON_BIN" - "$RATES" <<'PY'
import sys

rates = [float(value.strip()) for value in sys.argv[1].split(",") if value.strip()]
if not rates:
    raise SystemExit("RATES must contain at least one value")
print(max(rates))
PY
)"

declare -a CONFIG_ROWS=()
for raw_layer in "${LAYER_VALUES[@]}"; do
  layer="${raw_layer//[[:space:]]/}"
  [[ -n "$layer" ]] || continue
  for raw_expansion in "${EXPANSION_VALUES[@]}"; do
    expansion="${raw_expansion//[[:space:]]/}"
    [[ -n "$expansion" ]] || continue

    budget_line="$("$PYTHON_BIN" - "$TOTAL_LAYERS" "$layer" "$expansion" "$MAX_RATE" <<'PY'
import sys

layers = int(sys.argv[1])
outer_layers = int(sys.argv[2])
expansion = float(sys.argv[3])
max_rate = float(sys.argv[4])

if not 0 < outer_layers < layers:
    print("INVALID pruning_layer_out_of_range")
    raise SystemExit(0)
if expansion < 1.0:
    print("INVALID expansion_below_one")
    raise SystemExit(0)
if max_rate * expansion > 1.0:
    print(f"INVALID outer_retention_{max_rate * expansion:.10f}_exceeds_one")
    raise SystemExit(0)

inner_layers = layers - outer_layers
inner_retention = (layers / expansion - outer_layers) / inner_layers
if not 0.0 < inner_retention <= 1.0:
    print(f"INVALID solved_inner_retention_{inner_retention:.10f}")
    raise SystemExit(0)

multiplier = expansion * (
    outer_layers + inner_layers * inner_retention
) / layers
if abs(multiplier - 1.0) > 1e-10:
    print(f"INVALID multiplier_{multiplier:.12f}")
    raise SystemExit(0)

print(f"VALID {inner_retention:.10f} {multiplier:.12f}")
PY
)"

    status="${budget_line%% *}"
    if [[ "$status" != "VALID" ]]; then
      reason="${budget_line#* }"
      printf '%s\t%s\t%s\t%s\n' \
        "$layer" "$((TOTAL_LAYERS - layer))" "$expansion" "$reason" \
        >> "$SKIPPED_PATH"
      continue
    fi

    read -r _ inner_retention multiplier <<< "$budget_line"
    expansion_tag="${expansion/./p}"
    name="k${layer}_$((TOTAL_LAYERS - layer))_e${expansion_tag}"
    row="$name"$'\t'"$layer"$'\t'"$((TOTAL_LAYERS - layer))"$'\t'"$expansion"$'\t'"$inner_retention"$'\t'"$multiplier"
    CONFIG_ROWS+=("$row")
    printf '%s\n' "$row" >> "$MATRIX_PATH"
  done
done

if [[ "${#CONFIG_ROWS[@]}" -eq 0 ]]; then
  echo "No valid schedule configurations were generated." >&2
  exit 1
fi

echo "Search root: $OUTPUT_ROOT"
echo "Configurations: ${#CONFIG_ROWS[@]}"
echo "Tasks: $TASKS"
echo "Rates: $RATES"
echo "Limit per task: ${LIMIT:-full}"
echo "Fixed sample IDs: ${SAMPLE_IDS_FILE:-none}"
echo "GPUs: $GPUS"
echo
column -t -s $'\t' "$MATRIX_PATH" 2>/dev/null || cat "$MATRIX_PATH"

if [[ "$DRY_RUN" == "1" ]]; then
  echo
  echo "DRY_RUN=1, no experiments were launched."
  exit 0
fi

FAILURES=0
for row in "${CONFIG_ROWS[@]}"; do
  IFS=$'\t' read -r \
    name pruning_layer inner_layers expansion inner_retention multiplier \
    <<< "$row"

  run_dir="$OUTPUT_ROOT/search_$name"
  mkdir -p "$run_dir"
  completed_count="$(
    find "$run_dir" -type f -name '*_results.json' -print 2>/dev/null \
      | wc -l \
      | tr -d '[:space:]'
  )"
  if [[ "$completed_count" -ge "$EXPECTED_RESULTS" ]]; then
    echo "[$(date)] skipping completed configuration: $name"
    PORT=$((PORT + 1))
    continue
  fi

  echo "================================================================"
  echo "[$(date)] configuration=$name"
  echo "outer/inner layers=$pruning_layer/$inner_layers"
  echo "expansion=$expansion inner_retention=$inner_retention"
  echo "continuous layer-average multiplier=$multiplier"
  echo "output=$run_dir"
  echo "================================================================"

  cmd=(
    env
    CUDA_VISIBLE_DEVICES="$GPUS"
    NUM_PROCESSES="$PROCESSES"
    MAIN_PROCESS_PORT="$PORT"
    PYTHON_BIN="$PYTHON_BIN"
    ACCELERATE="$ACCELERATE"
    PRETRAINED="$MODEL"
    METHODS=certvid_v3
    TASKS="$TASKS"
    RATES="$RATES"
    MAX_FRAMES_NUM=32
    EXPANSION="$expansion"
    PRUNING_LAYER="$pruning_layer"
    LLM_RETENTION_RATIO="$inner_retention"
    CERTV3_BUDGET_USES_EXPANSION=True
    CERTV3_TOKEN_SELECTION_METHOD=attn_div_stable
    BATCH_SIZE=1
    LOG_SAMPLES=0
    OUTPUT_PATH="$run_dir"
  )
  if [[ -n "$LIMIT" ]]; then
    cmd+=(LIMIT="$LIMIT")
  fi
  if [[ -n "$SAMPLE_IDS_FILE" ]]; then
    cmd+=(
      LMMS_EVAL_SAMPLE_IDS_FILE="$SAMPLE_IDS_FILE"
      LMMS_EVAL_SAMPLE_ID_FIELD="$SAMPLE_ID_FIELD"
    )
  fi
  cmd+=(bash scripts/llava_ov.sh)

  run_succeeded=0
  if "${cmd[@]}" 2>&1 | tee "$run_dir/search.log"; then
    result_count="$(
      find "$run_dir" -type f -name '*_results.json' -print 2>/dev/null \
        | wc -l \
        | tr -d '[:space:]'
    )"
    if [[ "$result_count" -ge "$EXPECTED_RESULTS" ]]; then
      run_succeeded=1
      echo "[$(date)] completed: $name"
    else
      echo "[$(date)] FAILED: $name produced $result_count/$EXPECTED_RESULTS result files" >&2
    fi
  else
    echo "[$(date)] FAILED: $name" >&2
  fi

  if [[ "$run_succeeded" != "1" ]]; then
    FAILURES=$((FAILURES + 1))
    if [[ "$FAIL_FAST" == "1" ]]; then
      exit 1
    fi
  fi

  "$PYTHON_BIN" playground/summarize_v3_schedule_search.py \
    --root "$OUTPUT_ROOT" \
    --output "$SUMMARY_PATH" || true

  PORT=$((PORT + 1))
  sleep "$SLEEP_SECONDS"
done

"$PYTHON_BIN" playground/summarize_v3_schedule_search.py \
  --root "$OUTPUT_ROOT" \
  --output "$SUMMARY_PATH"

echo "Search complete: $OUTPUT_ROOT"
echo "Summary: $SUMMARY_PATH"
echo "Failures: $FAILURES"
test "$FAILURES" -eq 0
