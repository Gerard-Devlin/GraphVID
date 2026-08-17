#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/GraphVID}"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/xuyouwen/.conda/envs/graphvid311/bin/python}"
MODEL_PATH="${MODEL_PATH:-}"
MODEL_REPO="${MODEL_REPO:-${HF_HUB_CACHE:-/home/xuyouwen/hf_hub_local}/models--lmms-lab--llava-onevision-qwen2-7b-ov}"

if [[ -z "$MODEL_PATH" ]]; then
  if [[ -f "$MODEL_REPO/refs/main" ]]; then
    REVISION=$(tr -d '\r\n' < "$MODEL_REPO/refs/main")
    MODEL_PATH=$(readlink -f "$MODEL_REPO/snapshots/$REVISION")
  else
    MODEL_PATH=$(find "$MODEL_REPO/snapshots" -mindepth 1 -maxdepth 1 -type d | head -n 1)
  fi
fi

test -f "$MODEL_PATH/config.json" || {
  echo "LLaVA-OneVision model not found: $MODEL_PATH" >&2
  exit 1
}

DATASET_JSONL="${DATASET_JSONL:-$ROOT/assets/videomme.jsonl}"
if [[ ! -f "$DATASET_JSONL" && -f "$ROOT/videomme.jsonl" ]]; then
  DATASET_JSONL="$ROOT/videomme.jsonl"
fi
test -f "$DATASET_JSONL" || {
  echo "VideoMME JSONL not found: $DATASET_JSONL" >&2
  exit 1
}

VIDEO_ROOT="${VIDEO_ROOT:-${HF_HOME:-/home/xuyouwen/hf_home_local}/videomme/data}"
test -d "$VIDEO_ROOT" || {
  echo "VideoMME video root not found: $VIDEO_ROOT" >&2
  exit 1
}

GPU="${GPU:-0}"
SAMPLE_COUNT="${SAMPLE_COUNT:-100}"
NUM_WARMUP="${NUM_WARMUP:-1}"
NUM_REPEATS="${NUM_REPEATS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
SEED="${SEED:-20260813}"
METHODS="${METHODS:-vanilla fastv visionzip fastvid flashvid ours}"
SUMMARIZE="${SUMMARIZE:-1}"
SCORE_FILE="${SCORE_FILE:-$ROOT/scripts/efficiency/llava_ov_r1_scores.json}"
OUT="${OUT:-$ROOT/logs/efficiency/llavaov_r1_$(date +%Y%m%d_%H%M%S)}"
RAW_DIR="$OUT/raw"
MANIFEST="$OUT/efficiency_manifest.jsonl"
mkdir -p "$RAW_DIR" "${TMPDIR:-/home/xuyouwen/tmp}"

export PYTHONPATH="$ROOT/lmms-eval:$ROOT:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/home/xuyouwen/hf_home_local}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/home/xuyouwen/hf_hub_local}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/xuyouwen/hf_home_local/datasets}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="${TMPDIR:-/home/xuyouwen/tmp}"

echo "============================================================"
echo "LLaVA-OneVision efficiency benchmark"
echo "GPU=$GPU samples=$SAMPLE_COUNT warmup=$NUM_WARMUP repeats=$NUM_REPEATS"
echo "frames=32 retention=1%"
echo "methods=$METHODS"
echo "model=$MODEL_PATH"
echo "videos=$VIDEO_ROOT"
echo "output=$OUT"
echo "============================================================"

"$PYTHON_BIN" playground/bench_efficiency.py manifest \
  --dataset-jsonl "$DATASET_JSONL" \
  --output "$MANIFEST" \
  --sample-count "$SAMPLE_COUNT" \
  --seed "$SEED"

for METHOD in $METHODS; do
  echo "============================================================"
  echo "[$(date)] starting method=$METHOD in a fresh process"
  echo "============================================================"
  CUDA_VISIBLE_DEVICES="$GPU" \
    "$PYTHON_BIN" playground/bench_efficiency.py run \
      --method "$METHOD" \
      --model-path "$MODEL_PATH" \
      --model-name llava_qwen \
      --manifest "$MANIFEST" \
      --video-root "$VIDEO_ROOT" \
      --output "$RAW_DIR/$METHOD.jsonl" \
      --device cuda:0 \
      --num-frames 32 \
      --num-warmup "$NUM_WARMUP" \
      --num-repeats "$NUM_REPEATS" \
      --max-new-tokens "$MAX_NEW_TOKENS"
done

if [[ "$SUMMARIZE" == "1" ]]; then
  "$PYTHON_BIN" playground/bench_efficiency.py summarize \
    --input-dir "$RAW_DIR" \
    --output-dir "$OUT" \
    --manifest "$MANIFEST" \
    --score-file "$SCORE_FILE" \
    --num-warmup "$NUM_WARMUP" \
    --num-repeats "$NUM_REPEATS" \
    --methods $METHODS
else
  echo "Skipping cross-method summary (SUMMARIZE=$SUMMARIZE)."
  echo "Raw measurements: $RAW_DIR"
fi

echo "============================================================"
echo "Efficiency benchmark complete: $OUT"
if [[ "$SUMMARIZE" == "1" ]]; then
  echo "Table: $OUT/efficiency_table.md"
fi
echo "============================================================"
