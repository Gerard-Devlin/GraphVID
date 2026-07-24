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
DIAGNOSTICS_DETAIL="${CERTV8_DIAGNOSTICS_DETAIL:-summary}"
SOURCE_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/datasets}"
LOCAL_DATASETS_CACHE="${LOCAL_DATASETS_CACHE:-/tmp/${USER:-graphvid}/graphvid_hf_datasets}"
LVB_CACHE_NAME="longvideobench___long_video_bench"

if [[ ! -f "$SAMPLE_IDS_FILE" ]]; then
  echo "Sample-id file not found: $SAMPLE_IDS_FILE" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
cp "$SAMPLE_IDS_FILE" "$OUTPUT_ROOT/selected_sample_ids.txt"

# Gluster file locks can fail when all ranks initialize the cached dataset.
# Keep videos on shared storage, but copy the small Arrow dataset cache locally.
if [[ ! -d "$SOURCE_DATASETS_CACHE/$LVB_CACHE_NAME" ]]; then
  echo "LongVideoBench dataset cache not found: $SOURCE_DATASETS_CACHE/$LVB_CACHE_NAME" >&2
  exit 1
fi
mkdir -p "$LOCAL_DATASETS_CACHE/$LVB_CACHE_NAME"
cp -a "$SOURCE_DATASETS_CACHE/$LVB_CACHE_NAME/." \
  "$LOCAL_DATASETS_CACHE/$LVB_CACHE_NAME/"
echo "Using node-local datasets cache: $LOCAL_DATASETS_CACHE"

case "$STAGE" in
  baseline)
    CONFIGS=$(cat <<'EOF'
baseline	0.75	0.45	2.00	0.30	2	0.30	0.25	0.30	0.95	0.04	0.001	0.88	8.0	0.0	0.0
EOF
)
    ;;
  coarse)
    CONFIGS=$(cat <<'EOF'
baseline	0.75	0.45	2.00	0.30	2	0.30	0.25	0.30	0.95	0.04	0.001	0.88	8.0	0.0	0.0
query_low	0.75	0.45	2.00	0.30	2	0.15	0.25	0.30	0.95	0.04	0.001	0.88	8.0	0.0	0.0
query_high	0.75	0.45	2.00	0.30	3	0.45	0.25	0.30	0.95	0.04	0.001	0.88	8.0	0.0	0.0
event_high	0.75	0.45	2.00	0.30	2	0.25	0.45	0.20	0.95	0.04	0.001	0.88	8.0	0.0	0.0
balance_low	0.75	0.45	2.00	0.30	2	0.30	0.25	0.10	0.95	0.04	0.001	0.88	8.0	0.0	0.0
swap_low	0.75	0.45	2.00	0.15	2	0.30	0.25	0.30	0.95	0.04	0.001	0.88	8.0	0.0	0.0
swap_high	0.75	0.45	2.00	0.45	2	0.30	0.25	0.30	0.95	0.04	0.001	0.88	8.0	0.0	0.0
d_strict	0.75	0.45	2.00	0.30	2	0.30	0.25	0.30	0.98	0.04	0.001	0.88	8.0	0.0	0.0
d_relaxed	0.75	0.45	2.00	0.30	2	0.30	0.25	0.30	0.92	0.04	0.001	0.88	8.0	0.0	0.0
EOF
)
    ;;
  fine)
    CONFIGS=$(cat <<'EOF'
fine_conservative	0.60	0.40	1.80	0.20	2	0.18	0.30	0.15	0.98	0.05	0.0020	0.90	6.0	0.0	0.0
fine_event	0.75	0.40	2.20	0.32	2	0.20	0.40	0.15	0.95	0.035	0.0010	0.88	8.0	0.0	0.0
fine_focused	0.85	0.30	2.50	0.35	3	0.38	0.25	0.10	0.94	0.03	0.0005	0.86	10.0	0.0	0.0
fine_coverage	0.70	0.60	1.60	0.25	2	0.20	0.30	0.45	0.97	0.05	0.0010	0.92	6.0	0.0	0.0
fine_local	0.70	0.45	2.00	0.25	2	0.22	0.30	0.20	0.97	0.04	0.0010	0.94	4.0	0.0	0.0
EOF
)
    ;;
  targeted)
    CONFIGS=$(cat <<'EOF'
target_baseline	0.75	0.45	2.00	0.30	2	0.30	0.25	0.30	0.95	0.04	0.001	0.88	8.0	0.000	0.000
cue_e025	0.75	0.45	2.00	0.30	2	0.30	0.25	0.30	0.95	0.04	0.001	0.88	8.0	0.025	0.000
cue_e050	0.75	0.45	2.00	0.30	2	0.30	0.25	0.30	0.95	0.04	0.001	0.88	8.0	0.050	0.000
cue_e075	0.75	0.45	2.00	0.30	2	0.30	0.25	0.30	0.95	0.04	0.001	0.88	8.0	0.075	0.000
attr_q025	0.75	0.45	2.00	0.30	2	0.30	0.25	0.30	0.95	0.04	0.001	0.88	8.0	0.000	0.025
attr_q050	0.75	0.45	2.00	0.30	2	0.30	0.25	0.30	0.95	0.04	0.001	0.88	8.0	0.000	0.050
combo_e025_q025	0.75	0.45	2.00	0.30	2	0.30	0.25	0.30	0.95	0.04	0.001	0.88	8.0	0.025	0.025
combo_e050_q025	0.75	0.45	2.00	0.30	2	0.30	0.25	0.30	0.95	0.04	0.001	0.88	8.0	0.050	0.025
combo_e050_q050	0.75	0.45	2.00	0.30	2	0.30	0.25	0.30	0.95	0.04	0.001	0.88	8.0	0.050	0.050
EOF
)
    ;;
  budget)
    CONFIGS=$(cat <<'EOF'
budget_e115	0.75	0.45	2.00	0.30	2	0.30	0.25	0.30	0.95	0.04	0.001	0.88	8.0	0.0	0.0
budget_e120	0.75	0.45	2.00	0.30	2	0.30	0.25	0.30	0.95	0.04	0.001	0.88	8.0	0.0	0.0
budget_e1225	0.75	0.45	2.00	0.30	2	0.30	0.25	0.30	0.95	0.04	0.001	0.88	8.0	0.0	0.0
budget_e125	0.75	0.45	2.00	0.30	2	0.30	0.25	0.30	0.95	0.04	0.001	0.88	8.0	0.0	0.0
budget_e1275	0.75	0.45	2.00	0.30	2	0.30	0.25	0.30	0.95	0.04	0.001	0.88	8.0	0.0	0.0
budget_e130	0.75	0.45	2.00	0.30	2	0.30	0.25	0.30	0.95	0.04	0.001	0.88	8.0	0.0	0.0
budget_e1325	0.75	0.45	2.00	0.30	2	0.30	0.25	0.30	0.95	0.04	0.001	0.88	8.0	0.0	0.0
budget_e135	0.75	0.45	2.00	0.30	2	0.30	0.25	0.30	0.95	0.04	0.001	0.88	8.0	0.0	0.0
budget_e1375	0.75	0.45	2.00	0.30	2	0.30	0.25	0.30	0.95	0.04	0.001	0.88	8.0	0.0	0.0
EOF
)
    ;;
  *)
    echo "Unknown STAGE=$STAGE (expected baseline, coarse, fine, targeted, or budget)" >&2
    exit 1
    ;;
esac

printf '%s\n' "$CONFIGS" > "$OUTPUT_ROOT/search_matrix.tsv"

while IFS=$'\t' read -r \
  name intent floor cap swap peaks query event balance d_floor deficit min_gain cross_sim cross_seconds localized_event attribute_query
do
  [[ -n "$name" ]] || continue
  expansion="${EXPANSION:-1.30}"
  llm_retention_ratio="${LLM_RETENTION_RATIO:-0.1923076923}"
  if [[ "$STAGE" == "budget" ]]; then
    case "$name" in
      budget_e115)  expansion=1.15;  llm_retention_ratio=0.5434782609 ;;
      budget_e120)  expansion=1.20;  llm_retention_ratio=0.4166666667 ;;
      budget_e1225) expansion=1.225; llm_retention_ratio=0.3571428571 ;;
      budget_e125)  expansion=1.25;  llm_retention_ratio=0.3000000000 ;;
      budget_e1275) expansion=1.275; llm_retention_ratio=0.2450980392 ;;
      budget_e130)  expansion=1.30;  llm_retention_ratio=0.1923076923 ;;
      budget_e1325) expansion=1.325; llm_retention_ratio=0.1415094340 ;;
      budget_e135)  expansion=1.35;  llm_retention_ratio=0.0925925926 ;;
      budget_e1375) expansion=1.375; llm_retention_ratio=0.0454545455 ;;
      *) echo "Missing budget pair for $name" >&2; exit 1 ;;
    esac
  fi

  python - "$expansion" "$llm_retention_ratio" <<'PY'
import sys

expansion = float(sys.argv[1])
retention = float(sys.argv[2])
multiplier = expansion * (20.0 + 8.0 * retention) / 28.0
if abs(multiplier - 1.0) > 1e-4:
    raise SystemExit(
        f"unfair layer-average budget: E={expansion}, r={retention}, "
        f"multiplier={multiplier:.8f}"
    )
PY

  run_dir="$OUTPUT_ROOT/search_$name"
  if find "$run_dir" -type f -name '*_results.json' -print -quit 2>/dev/null \
    | grep -q .; then
    echo "[$(date)] skipping completed configuration: $name"
    PORT=$((PORT + 1))
    continue
  fi
  echo "================================================================"
  echo "[$(date)] $name"
  echo "samples=$SAMPLE_IDS_FILE output=$run_dir"
  echo "budget_split: expansion=$expansion pruning_layer=20 llm_retention_ratio=$llm_retention_ratio"
  echo "================================================================"

  env \
    CUDA_VISIBLE_DEVICES="$GPUS" \
    NUM_PROCESSES="$PROCESSES" \
    MAIN_PROCESS_PORT="$PORT" \
    HF_HUB_OFFLINE="$OFFLINE" \
    HF_DATASETS_OFFLINE="$OFFLINE" \
    TRANSFORMERS_OFFLINE="$OFFLINE" \
    HF_DATASETS_CACHE="$LOCAL_DATASETS_CACHE" \
    PRETRAINED="$MODEL" \
    METHODS=certvid_v8 \
    TASKS=longvideobench_val_v \
    RATES=0.10 \
    EXPANSION="$expansion" \
    PRUNING_LAYER=20 \
    LLM_RETENTION_RATIO="$llm_retention_ratio" \
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
    CERTV8_LOCALIZED_EVENT_BOOST="$localized_event" \
    CERTV8_ATTRIBUTE_QUERY_BOOST="$attribute_query" \
    CERTV8_DEBUG=False \
    CERTV8_DIAGNOSTICS_DETAIL="$DIAGNOSTICS_DETAIL" \
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
