#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

STAGE="${STAGE:-coarse}"
SAMPLE_IDS_FILE="${SAMPLE_IDS_FILE:-$REPO_ROOT/assets/lvb_search/lvb_v8_search_192.ids}"
MODEL="${PRETRAINED:-/home/xuyouwen/models/llava-onevision-qwen2-7b-ov}"
OUTPUT_ROOT="${OUTPUT_PATH:-$REPO_ROOT/logs/lvb_v8_search_${STAGE}_$(date +%Y%m%d_%H%M%S)}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
PROCESSES="${NUM_PROCESSES:-6}"
PORT="${MAIN_PROCESS_PORT:-18950}"
OFFLINE="${OFFLINE:-1}"

if [[ ! -f "$SAMPLE_IDS_FILE" ]]; then
  echo "Sample-id file not found: $SAMPLE_IDS_FILE" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
cp "$SAMPLE_IDS_FILE" "$OUTPUT_ROOT/selected_sample_ids.txt"

case "$STAGE" in
  baseline)
    CONFIGS=$(cat <<'EOF'
baseline	0.75	0.45	2.00	0.30	2	0.30	0.25	0.30	0.95	0.04	0.001	0.88	8.0
EOF
)
    ;;
  coarse)
    CONFIGS=$(cat <<'EOF'
baseline	0.75	0.45	2.00	0.30	2	0.30	0.25	0.30	0.95	0.04	0.001	0.88	8.0
query_low	0.75	0.45	2.00	0.30	2	0.15	0.25	0.30	0.95	0.04	0.001	0.88	8.0
query_high	0.75	0.45	2.00	0.30	3	0.45	0.25	0.30	0.95	0.04	0.001	0.88	8.0
event_high	0.75	0.45	2.00	0.30	2	0.25	0.45	0.20	0.95	0.04	0.001	0.88	8.0
balance_low	0.75	0.45	2.00	0.30	2	0.30	0.25	0.10	0.95	0.04	0.001	0.88	8.0
swap_low	0.75	0.45	2.00	0.15	2	0.30	0.25	0.30	0.95	0.04	0.001	0.88	8.0
swap_high	0.75	0.45	2.00	0.45	2	0.30	0.25	0.30	0.95	0.04	0.001	0.88	8.0
d_strict	0.75	0.45	2.00	0.30	2	0.30	0.25	0.30	0.98	0.04	0.001	0.88	8.0
d_relaxed	0.75	0.45	2.00	0.30	2	0.30	0.25	0.30	0.92	0.04	0.001	0.88	8.0
EOF
)
    ;;
  fine)
    CONFIGS=$(cat <<'EOF'
fine_conservative	0.60	0.40	1.80	0.20	2	0.18	0.30	0.15	0.98	0.05	0.0020	0.90	6.0
fine_event	0.75	0.40	2.20	0.32	2	0.20	0.40	0.15	0.95	0.035	0.0010	0.88	8.0
fine_focused	0.85	0.30	2.50	0.35	3	0.38	0.25	0.10	0.94	0.03	0.0005	0.86	10.0
fine_coverage	0.70	0.60	1.60	0.25	2	0.20	0.30	0.45	0.97	0.05	0.0010	0.92	6.0
fine_local	0.70	0.45	2.00	0.25	2	0.22	0.30	0.20	0.97	0.04	0.0010	0.94	4.0
EOF
)
    ;;
  *)
    echo "Unknown STAGE=$STAGE (expected baseline, coarse, or fine)" >&2
    exit 1
    ;;
esac

printf '%s\n' "$CONFIGS" > "$OUTPUT_ROOT/search_matrix.tsv"

while IFS=$'\t' read -r \
  name intent floor cap swap peaks query event balance d_floor deficit min_gain cross_sim cross_seconds
do
  [[ -n "$name" ]] || continue
  run_dir="$OUTPUT_ROOT/search_$name"
  echo "================================================================"
  echo "[$(date)] $name"
  echo "samples=$SAMPLE_IDS_FILE output=$run_dir"
  echo "================================================================"

  env \
    CUDA_VISIBLE_DEVICES="$GPUS" \
    NUM_PROCESSES="$PROCESSES" \
    MAIN_PROCESS_PORT="$PORT" \
    HF_HUB_OFFLINE="$OFFLINE" \
    HF_DATASETS_OFFLINE="$OFFLINE" \
    TRANSFORMERS_OFFLINE="$OFFLINE" \
    PRETRAINED="$MODEL" \
    METHODS=certvid_v8 \
    TASKS=longvideobench_val_v \
    RATES=0.10 \
    EXPANSION=1.30 \
    PRUNING_LAYER=20 \
    LLM_RETENTION_RATIO=0.1923076923 \
    CERTV3_BUDGET_USES_EXPANSION=True \
    CERTV8_INTENT_STRENGTH="$intent" \
    CERTV8_FRAME_FLOOR_RATIO="$floor" \
    CERTV8_FRAME_CAP_RATIO="$cap" \
    CERTV8_MAX_SWAP_RATIO="$swap" \
    CERTV8_QUERY_PEAK_COUNT="$peaks" \
    CERTV8_QUERY_WEIGHT="$query" \
    CERTV8_EVENT_WEIGHT="$event" \
    CERTV8_BALANCE_WEIGHT="$balance" \
    CERTV8_D_EFFICIENCY_FLOOR="$d_floor" \
    CERTV8_MIN_DEFICIT="$deficit" \
    CERTV8_MIN_OBJECTIVE_GAIN="$min_gain" \
    CERTV8_CROSS_FRAME_SIMILARITY="$cross_sim" \
    CERTV8_CROSS_FRAME_MAX_SECONDS="$cross_seconds" \
    CERTV8_DEBUG=False \
    CERTV8_DIAGNOSTICS_DETAIL=summary \
    LMMS_EVAL_SAMPLE_IDS_FILE="$SAMPLE_IDS_FILE" \
    LMMS_EVAL_SAMPLE_ID_FIELD=id \
    LOG_SAMPLES=1 \
    OUTPUT_PATH="$run_dir" \
    bash scripts/llava_ov.sh

  if ! find "$run_dir" -type f -name '*_results.json' -print -quit | grep -q .; then
    echo "No lmms-eval result JSON was produced for $name; stopping search." >&2
    exit 1
  fi

  PORT=$((PORT + 1))
  sleep 10
done <<< "$CONFIGS"

python playground/summarize_lvb_v8_search.py --root "$OUTPUT_ROOT"
echo "Search complete: $OUTPUT_ROOT"
