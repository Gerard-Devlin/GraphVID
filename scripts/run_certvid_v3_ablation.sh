#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

RATE="${RATE:-0.01}"
TASKS="${TASKS:-videomme,egoschema_subset,egoschema,longvideobench_val_v,mvbench}"
EXPANSION="${EXPANSION:-1.30}"
PRUNING_LAYER="${PRUNING_LAYER:-20}"
LLM_RETENTION_RATIO="${LLM_RETENTION_RATIO:-0.1923076923}"
OUTPUT_PATH="${OUTPUT_PATH:-$PWD/logs/lmms_eval/certvid_v3_ablation}"
RESUME="${RESUME:-1}"
FAIL_FAST="${FAIL_FAST:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ABLATIONS="${ABLATIONS:-no_doptimal,no_quality_aware_weighting,no_spatiotemporal,no_all_trajectory_dynamics,no_query,no_whitening,no_candidate_pool,no_exchange_refinement,no_fusion}"

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

for ablation in $(split_csv "$ABLATIONS"); do
  selection_objective=d_optimal
  use_spatiotemporal_design=True
  use_trajectory=True
  use_query=True
  fusion_alpha=0.12
  quality_floor=0.15
  whitening_strength=0.50
  use_candidate_pool=True
  swap_steps=6

  case "$ablation" in
    full) ;;
    no_doptimal) selection_objective=score_only ;;
    no_quality_aware_weighting) quality_floor=1.0 ;;
    no_spatiotemporal) use_spatiotemporal_design=False ;;
    no_all_trajectory_dynamics|no_trajectory) use_trajectory=False ;;
    no_query) use_query=False ;;
    no_whitening) whitening_strength=0.0 ;;
    no_candidate_pool) use_candidate_pool=False ;;
    no_exchange_refinement) swap_steps=0 ;;
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
    echo "[run] objective=$selection_objective quality_floor=$quality_floor spatiotemporal_design=$use_spatiotemporal_design all_trajectory_dynamics=$use_trajectory query=$use_query whitening=$whitening_strength candidate_pool=$use_candidate_pool exchange_steps=$swap_steps fusion_alpha=$fusion_alpha certificate_ratio=0"
    echo "================================================================"

    task_status=0
    if env \
      METHODS=certvid_v3 \
      RATES="$RATE" \
      TASKS="$task" \
      EXPANSION="$EXPANSION" \
      PRUNING_LAYER="$PRUNING_LAYER" \
      LLM_RETENTION_RATIO="$LLM_RETENTION_RATIO" \
      OUTPUT_PATH="$ablation_dir" \
      CERTV3_BUDGET_USES_EXPANSION=True \
      CERTV3_SELECTION_OBJECTIVE="$selection_objective" \
      CERTV3_QUALITY_FLOOR="$quality_floor" \
      CERTV3_CERTIFICATE_BUDGET_RATIO=0.0 \
      CERTV3_USE_SPATIOTEMPORAL_DESIGN="$use_spatiotemporal_design" \
      CERTV3_USE_TRAJECTORY="$use_trajectory" \
      CERTV3_USE_QUERY="$use_query" \
      CERTV3_WHITENING_STRENGTH="$whitening_strength" \
      CERTV3_USE_CANDIDATE_POOL="$use_candidate_pool" \
      CERTV3_SWAP_STEPS="$swap_steps" \
      CERTV3_FUSION_ALPHA="$fusion_alpha" \
      CERTV3_DIAGNOSTICS_JSONL="$run_dir/certvid_v3_diagnostics_rank{rank}.jsonl" \
      bash scripts/llava_ov.sh 2>&1 | tee "$task_log"
    then
      if has_result "$run_dir"; then
        echo "[completed] ablation=$ablation task=$task"
      else
        echo "[failed] ablation=$ablation task=$task: launcher returned zero but no result JSON was produced" >&2
        task_status=1
      fi
    else
      echo "[failed] ablation=$ablation task=$task" >&2
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
