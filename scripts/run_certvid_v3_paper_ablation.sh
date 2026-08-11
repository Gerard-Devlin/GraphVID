#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

RATE="${RATE:-0.01}"
OUTPUT_PATH="${OUTPUT_PATH:-$PWD/logs/lmms_eval/certvid_v3_paper_ablation}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESUME="${RESUME:-1}"
FAIL_FAST="${FAIL_FAST:-0}"
ABLATIONS="${ABLATIONS:-no_doptimal,no_spatiotemporal,no_trajectory,no_query,no_fusion}"
EXPANSIONS="${EXPANSIONS:-e125,e120,e115,e100}"

TABLE1="$OUTPUT_PATH/table1_components"
TABLE2="$OUTPUT_PATH/table2_expansion_schedule"
mkdir -p "$TABLE1" "$TABLE2"

echo "================================================================"
echo "[table 1] component ablations at R=$RATE"
echo "[table 1] certificates disabled for every row"
echo "================================================================"

if env \
  RATE="$RATE" \
  OUTPUT_PATH="$TABLE1" \
  PYTHON_BIN="$PYTHON_BIN" \
  RESUME="$RESUME" \
  FAIL_FAST="$FAIL_FAST" \
  ABLATIONS="$ABLATIONS" \
  bash scripts/run_certvid_v3_ablation.sh
then
  TABLE1_STATUS=0
else
  TABLE1_STATUS=$?
fi

echo "================================================================"
echo "[table 2] expansion/schedule ablations at R=$RATE"
echo "[table 2] fixed 28-layer average token budget"
echo "================================================================"

if env \
  RATE="$RATE" \
  OUTPUT_PATH="$TABLE2" \
  PYTHON_BIN="$PYTHON_BIN" \
  RESUME="$RESUME" \
  FAIL_FAST="$FAIL_FAST" \
  EXPANSIONS="$EXPANSIONS" \
  bash scripts/run_certvid_v3_expansion_ablation.sh
then
  TABLE2_STATUS=0
else
  TABLE2_STATUS=$?
fi

echo "================================================================"
echo "[final tables]"
"$PYTHON_BIN" playground/summarize_certvid_v3_ablation.py \
  --root "$TABLE1" --rate "$RATE" --mode components
"$PYTHON_BIN" playground/summarize_certvid_v3_ablation.py \
  --root "$TABLE2" --rate "$RATE" --mode expansion
echo "table1=$TABLE1/table1_component_ablation.tsv"
echo "table2=$TABLE2/table2_expansion_schedule.tsv"
echo "================================================================"

test "$TABLE1_STATUS" -eq 0 && test "$TABLE2_STATUS" -eq 0
