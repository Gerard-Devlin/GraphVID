#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/xuyouwen/GraphVID}
cd "$ROOT"

export PATH=/home/xuyouwen/.conda/envs/graphvid311/bin:$PATH
export PYTHONPATH="$ROOT/lmms-eval:$ROOT/playground:$ROOT:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-/home/xuyouwen/hf_home_local}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/home/xuyouwen/hf_hub_local}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/home/xuyouwen/hf_home_local/datasets}
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON_BIN=${PYTHON_BIN:-/home/xuyouwen/.conda/envs/graphvid311/bin/python}
MODEL_REPO=${MODEL_REPO:-$HF_HUB_CACHE/models--lmms-lab--llava-onevision-qwen2-7b-ov}
DATASET_ROOT=${DATASET_ROOT:-$HF_HOME/videomme/data}
METADATA_JSONL=${METADATA_JSONL:-$ROOT/assets/videomme.jsonl}
CANDIDATE_IDS_FILE=${CANDIDATE_IDS_FILE:-$ROOT/assets/videomme_sports_visualization.ids}
GPU=${GPU:-4}
CANDIDATE_COUNT=${CANDIDATE_COUNT:-12}
SEED=${SEED:-20260815}
RETENTION_RATIO=${RETENTION_RATIO:-0.01}
COMPARISON_FRAMES=${COMPARISON_FRAMES:-1,8,17,26}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/logs/visualizations/certvid_topk_$(date +%Y%m%d_%H%M%S)}

if [[ -f "$MODEL_REPO/refs/main" ]]; then
    REVISION=$(tr -d '\r\n' < "$MODEL_REPO/refs/main")
    MODEL=$(readlink -f "$MODEL_REPO/snapshots/$REVISION")
else
    MODEL=$(find "$MODEL_REPO/snapshots" -mindepth 1 -maxdepth 1 -type d | head -n 1)
fi

test -f "$MODEL/config.json" || {
    echo "Model not found: $MODEL" >&2
    exit 1
}
test -d "$DATASET_ROOT" || {
    echo "VideoMME root not found: $DATASET_ROOT" >&2
    exit 1
}
test -f "$METADATA_JSONL" || {
    echo "VideoMME metadata not found: $METADATA_JSONL" >&2
    exit 1
}
test -f "$CANDIDATE_IDS_FILE" || {
    echo "Candidate video-ID file not found: $CANDIDATE_IDS_FILE" >&2
    exit 1
}

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "Equal-budget Quality Top-K versus CertVID visualization"
echo "GPU=$GPU candidates=$CANDIDATE_COUNT R=$RETENTION_RATIO"
echo "comparison_frames=$COMPARISON_FRAMES"
echo "candidate_ids=$CANDIDATE_IDS_FILE"
echo "model=$MODEL"
echo "output=$OUTPUT_DIR"
echo "============================================================"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
    playground/visualize_certvid_topk_comparison.py \
    --model-path "$MODEL" \
    --dataset-root "$DATASET_ROOT" \
    --metadata-jsonl "$METADATA_JSONL" \
    --candidate-ids-file "$CANDIDATE_IDS_FILE" \
    --candidate-count "$CANDIDATE_COUNT" \
    --num-frames 32 \
    --filmstrip-frames 8 \
    --comparison-frames "$COMPARISON_FRAMES" \
    --seed "$SEED" \
    --retention-ratio "$RETENTION_RATIO" \
    --expansion 1.30 \
    --pruning-layer 20 \
    --llm-retention-ratio 0.1923076923 \
    --device-map cuda:0 \
    --dpi 300 \
    --output-dir "$OUTPUT_DIR"

echo "============================================================"
echo "Completed: $OUTPUT_DIR"
find "$OUTPUT_DIR" -maxdepth 1 -type f -print | sort
echo "============================================================"
