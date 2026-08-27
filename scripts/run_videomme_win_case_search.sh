#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE_JSONL="${SOURCE_JSONL:-$PWD/assets/videomme.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-$PWD/logs/lmms_eval/videomme_win_case_search}"
VIDEO_COUNT="${VIDEO_COUNT:-200}"
SEED="${SEED:-20260827}"

SUBSET_DIR="$OUTPUT_PATH/subset"
RUNS_DIR="$OUTPUT_PATH/runs"
ANALYSIS_DIR="$OUTPUT_PATH/analysis"
IDS_FILE="$SUBSET_DIR/video_ids.txt"
MANIFEST_FILE="$SUBSET_DIR/videomme_subset.jsonl"
mkdir -p "$SUBSET_DIR" "$RUNS_DIR" "$ANALYSIS_DIR"

"$PYTHON_BIN" playground/select_videomme_videos.py \
  --input "$SOURCE_JSONL" \
  --count "$VIDEO_COUNT" \
  --seed "$SEED" \
  --ids-output "$IDS_FILE" \
  --manifest-output "$MANIFEST_FILE"

echo "Running five methods on the same $VIDEO_COUNT VideoMME videos"
env \
  LMMS_EVAL_SAMPLE_IDS_FILE="$IDS_FILE" \
  LMMS_EVAL_SAMPLE_ID_FIELD=videoID \
  METHODS=fastv,visionzip,fastvid,flashvid,certvidfinal2 \
  RATES=0.01 \
  TASKS=videomme \
  MAX_FRAMES_NUM=32 \
  EXPANSION=1.30 \
  PRUNING_LAYER=20 \
  LLM_RETENTION_RATIO=0.1923076923 \
  CERTV3_BUDGET_USES_EXPANSION=True \
  CERTV3_USE_EXACT_CUDA_GRAPHS="${CERTV3_USE_EXACT_CUDA_GRAPHS:-1}" \
  CERTV3_AUDIT_EXACT_OPTIMIZATIONS=0 \
  CERTV3_PROFILE_PHASES=0 \
  LOG_SAMPLES=1 \
  LIMIT= \
  OUTPUT_PATH="$RUNS_DIR" \
  bash scripts/llava_ov.sh

"$PYTHON_BIN" playground/find_videomme_win_cases.py \
  --root "$RUNS_DIR" \
  --rate 0.01 \
  --output-dir "$ANALYSIS_DIR"

echo "Finished: $OUTPUT_PATH"
echo "Strict cases: $ANALYSIS_DIR/strict_win_cases.csv"
