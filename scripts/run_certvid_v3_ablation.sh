#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

RATE="${RATE:-0.10}"
TASKS="${TASKS:-videomme,egoschema_subset,longvideobench_val_v,mvbench}"
OUTPUT_PATH="${OUTPUT_PATH:-$PWD/logs/lmms_eval/certvid_v3_ablation}"
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

ABLATIONS=(
  full
  no_doptimal
  no_certificates
  no_trajectory
  no_query
  no_fusion
)

FAILURES=0

for ablation in "${ABLATIONS[@]}"; do
  selection_objective=d_optimal
  use_spatiotemporal_certificates=True
  use_trajectory=True
  use_query=True
  fusion_alpha=0.12

  case "$ablation" in
    full) ;;
    no_doptimal) selection_objective=quality_topk ;;
    no_certificates) use_spatiotemporal_certificates=False ;;
    no_trajectory) use_trajectory=False ;;
    no_query) use_query=False ;;
    no_fusion) fusion_alpha=0.0 ;;
    *) echo "Unknown ablation: $ablation" >&2; exit 2 ;;
  esac

  ablation_dir="$OUTPUT_PATH/$ablation"
  mkdir -p "$ablation_dir"

  for task in $(split_csv "$TASKS"); do
    run_dir="$ablation_dir/certvid_v3_r${RATE}_${task}"
    task_log="$ablation_dir/${task}.log"
    if [[ "$RESUME" == "1" ]] && has_result "$run_dir"; then
      echo "[skip] ablation=$ablation task=$task result already exists"
      continue
    fi

    echo "================================================================"
    echo "[run] ablation=$ablation rate=$RATE task=$task"
    echo "[run] objective=$selection_objective spatiotemporal_certificates=$use_spatiotemporal_certificates trajectory=$use_trajectory query=$use_query fusion_alpha=$fusion_alpha"
    echo "================================================================"

    if env \
      METHODS=certvid_v3 \
      RATES="$RATE" \
      TASKS="$task" \
      OUTPUT_PATH="$ablation_dir" \
      CERTV3_SELECTION_OBJECTIVE="$selection_objective" \
      CERTV3_USE_SPATIOTEMPORAL_CERTIFICATES="$use_spatiotemporal_certificates" \
      CERTV3_USE_TRAJECTORY="$use_trajectory" \
      CERTV3_USE_QUERY="$use_query" \
      CERTV3_FUSION_ALPHA="$fusion_alpha" \
      CERTV3_DIAGNOSTICS_JSONL="$run_dir/certvid_v3_diagnostics_rank{rank}.jsonl" \
      bash scripts/llava_ov.sh 2>&1 | tee "$task_log"
    then
      echo "[completed] ablation=$ablation task=$task"
    else
      echo "[failed] ablation=$ablation task=$task" >&2
      FAILURES=$((FAILURES + 1))
      if [[ "$FAIL_FAST" == "1" ]]; then
        exit 1
      fi
    fi
  done

  "$PYTHON_BIN" playground/summarize_certvid_v3_ablation.py \
    --root "$OUTPUT_PATH" \
    --rate "$RATE" || true
done

echo "================================================================"
echo "[summary] failures=$FAILURES"
"$PYTHON_BIN" playground/summarize_certvid_v3_ablation.py \
  --root "$OUTPUT_PATH" \
  --rate "$RATE"
echo "[summary] results=$OUTPUT_PATH"
echo "================================================================"

test "$FAILURES" -eq 0
