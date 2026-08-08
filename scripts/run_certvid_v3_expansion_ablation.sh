#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

RATE="${RATE:-0.01}"
TASKS="${TASKS:-videomme,egoschema_subset,egoschema,longvideobench_val_v,mvbench}"
OUTPUT_PATH="${OUTPUT_PATH:-$PWD/logs/lmms_eval/certvid_v3_expansion_ablation}"
RESUME="${RESUME:-1}"
FAIL_FAST="${FAIL_FAST:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

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

# E * (20 + 8r) / 28 = 1, so every row has the same average-layer budget.
CONFIGS=(
  "e130:1.30:0.1923076923"
  "e125:1.25:0.3"
  "e120:1.20:0.4166666667"
  "e115:1.15:0.5434782609"
)

FAILURES=0

for config in "${CONFIGS[@]}"; do
  IFS=: read -r slug expansion inner_retention <<< "$config"
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

    if env \
      METHODS=certvid_v3 \
      RATES="$RATE" \
      TASKS="$task" \
      EXPANSION="$expansion" \
      PRUNING_LAYER=20 \
      LLM_RETENTION_RATIO="$inner_retention" \
      CERTV3_BUDGET_USES_EXPANSION=True \
      OUTPUT_PATH="$config_dir" \
      CERTV3_DIAGNOSTICS_JSONL="$run_dir/certvid_v3_diagnostics_rank{rank}.jsonl" \
      bash scripts/llava_ov.sh 2>&1 | tee "$task_log"
    then
      echo "[completed] expansion=$expansion task=$task"
    else
      echo "[failed] expansion=$expansion task=$task" >&2
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
