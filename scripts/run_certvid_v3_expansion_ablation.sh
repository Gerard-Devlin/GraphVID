#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

RATE="${RATE:-0.01}"
TASKS="${TASKS:-videomme,egoschema_subset,egoschema,longvideobench_val_v,mvbench}"
OUTPUT_PATH="${OUTPUT_PATH:-$PWD/logs/lmms_eval/certvid_v3_expansion_ablation}"
RESUME="${RESUME:-1}"
FAIL_FAST="${FAIL_FAST:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXPANSIONS="${EXPANSIONS:-e125,e120,e115,e100}"

mkdir -p "$OUTPUT_PATH"

split_csv() {
  local text="$1"
  text="${text//,/ }"
  # shellcheck disable=SC2086
  printf '%s\n' $text
}

has_result() {
  local directory="$1"
  [[ -d "$directory" ]] && find "$directory" -type f \( -name '*results*.json' -o -name 'results.json' \) -print -quit | grep -q .
}

FAILURES=0

for slug in $(split_csv "$EXPANSIONS"); do
  # E * (20 + 8r) / 28 = 1 for an equal average-layer token budget.
  case "$slug" in
    e140) expansion=1.40; inner_retention=0.0 ;;
    e135) expansion=1.35; inner_retention=0.0925925926 ;;
    e130) expansion=1.30; inner_retention=0.1923076923 ;;
    e125) expansion=1.25; inner_retention=0.3 ;;
    e120) expansion=1.20; inner_retention=0.4166666667 ;;
    e115) expansion=1.15; inner_retention=0.5434782609 ;;
    e110) expansion=1.10; inner_retention=0.6818181818 ;;
    e100) expansion=1.00; inner_retention=1.0 ;;
    *) echo "Unknown expansion ablation: $slug" >&2; exit 2 ;;
  esac
  config_dir="$OUTPUT_PATH/$slug"
  mkdir -p "$config_dir"

  for task in $(split_csv "$TASKS"); do
    run_dir="$config_dir/certvid_v3_r${RATE}_${task}"
    task_log="$config_dir/${task}.log"
    if [[ "$RESUME" == "1" ]] && has_result "$run_dir"; then
      echo "[skip] expansion=$expansion task=$task result already exists"
      continue
    fi

    echo "================================================================"
    echo "[run] rate=$RATE expansion=$expansion outer_layers=20 inner_layers=8 inner_retention=$inner_retention task=$task"
    echo "================================================================"

    task_status=0
    if env \
      METHODS=certvid_v3 \
      RATES="$RATE" \
      TASKS="$task" \
      EXPANSION="$expansion" \
      PRUNING_LAYER=20 \
      LLM_RETENTION_RATIO="$inner_retention" \
      CERTV3_BUDGET_USES_EXPANSION=True \
      CERTV3_CERTIFICATE_BUDGET_RATIO=0.0 \
      CERTV3_SELECTION_OBJECTIVE=d_optimal \
      CERTV3_USE_SPATIOTEMPORAL_DESIGN=True \
      CERTV3_USE_TRAJECTORY=True \
      CERTV3_USE_QUERY=True \
      CERTV3_FUSION_ALPHA=0.12 \
      OUTPUT_PATH="$config_dir" \
      CERTV3_DIAGNOSTICS_JSONL="$run_dir/certvid_v3_diagnostics_rank{rank}.jsonl" \
      bash scripts/llava_ov.sh 2>&1 | tee "$task_log"
    then
      if has_result "$run_dir"; then
        echo "[completed] expansion=$expansion task=$task"
      else
        echo "[failed] expansion=$expansion task=$task: launcher returned zero but no result JSON was produced" >&2
        task_status=1
      fi
    else
      echo "[failed] expansion=$expansion task=$task" >&2
      task_status=1
    fi
    if [[ "$task_status" -ne 0 ]]; then
      FAILURES=$((FAILURES + 1))
      if [[ "$FAIL_FAST" == "1" ]]; then
        exit 1
      fi
    fi
  done

  "$PYTHON_BIN" playground/summarize_certvid_v3_ablation.py \
    --root "$OUTPUT_PATH" \
    --rate "$RATE" \
    --mode expansion || true
done

echo "================================================================"
echo "[summary] failures=$FAILURES"
"$PYTHON_BIN" playground/summarize_certvid_v3_ablation.py \
  --root "$OUTPUT_PATH" \
  --rate "$RATE" \
  --mode expansion
echo "[summary] results=$OUTPUT_PATH"
echo "================================================================"

test "$FAILURES" -eq 0
