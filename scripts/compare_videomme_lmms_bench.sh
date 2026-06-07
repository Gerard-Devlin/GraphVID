#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON="${PYTHON:-python}"
PRETRAINED="${PRETRAINED:-/root/models}"
DATASET_JSONL="${DATASET_JSONL:-assets/videomme.jsonl}"
METHODS="${METHODS:-flashvid,graphvid}"
RATES="${RATES:-0.10}"
LIMIT="${LIMIT:-10}"
TAG="${TAG:-compare_lmms_bench_videomme_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-logs/compare_lmms_bench/${TAG}}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MAX_GPUS="${MAX_GPUS:-1}"
MIN_FREE_MB="${MIN_FREE_MB:-0}"
FREE_RATIO="${FREE_RATIO:-0.0}"
LOG_SAMPLES="${LOG_SAMPLES:-0}"

MAX_NUM_FRAMES="${MAX_NUM_FRAMES:-32}"
MIN_PIXELS="${MIN_PIXELS:-50176}"
MAX_PIXELS="${MAX_PIXELS:-200704}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"

EXPANSION="${EXPANSION:-1.25}"
LLM_RETENTION_RATIO="${LLM_RETENTION_RATIO:-1.0}"
RAW_VISUAL_TOKENS="${RAW_VISUAL_TOKENS:-2880}"
VISUAL_TIME_UNITS="${VISUAL_TIME_UNITS:-16}"
GRAPH_FINAL_CAP_MODE="${GRAPH_FINAL_CAP_MODE:-expanded}"
GRAPH_FINAL_TPF="${GRAPH_FINAL_TPF:-0}"
GRAPH_FINAL_TPF_BY_RATE="${GRAPH_FINAL_TPF_BY_RATE:-}"
TOKEN_SELECTION_METHOD="${TOKEN_SELECTION_METHOD:-attn_div_v2}"
FLASHVID_TOKEN_SELECTION_METHOD="${FLASHVID_TOKEN_SELECTION_METHOD:-attn_div_v2}"
GRAPHVID_TOKEN_SELECTION_METHOD="${GRAPHVID_TOKEN_SELECTION_METHOD:-attn_div_stable}"

mkdir -p "$OUT_ROOT"

BENCH_DATASET_JSONL="$DATASET_JSONL"

split_csv() {
  local text="$1"
  text="${text//,/ }"
  # shellcheck disable=SC2086
  printf '%s\n' $text
}

rate_label() {
  "$PYTHON" - "$1" <<'PY'
import sys
r = float(sys.argv[1])
if r > 1:
    r = r / 100.0
print(f"{r * 100:g}".replace(".", "p"))
PY
}

bench_method_name() {
  case "$1" in
    fastvid|visionzip|fastgraphvid|curvevid) printf '%s_qwen3_adapter' "$1" ;;
    *) printf '%s' "$1" ;;
  esac
}

bench_summary_path() {
  local method="$1"
  local rate="$2"
  local label
  local run_method
  label="$(rate_label "$rate")"
  run_method="$(bench_method_name "$method")"
  local method_tag="${TAG}_bench_${run_method}_r${label}_videomme"
  printf 'logs/efficiency/parallel/%s/%s_summary.json' "$method_tag" "$method_tag"
}

graph_final_tpf_for_rate() {
  "$PYTHON" - "$1" "$GRAPH_FINAL_CAP_MODE" "$GRAPH_FINAL_TPF" "$GRAPH_FINAL_TPF_BY_RATE" "$RAW_VISUAL_TOKENS" "$VISUAL_TIME_UNITS" "$EXPANSION" <<'PY'
import math
import sys

ratio = float(sys.argv[1])
if ratio > 1:
    ratio /= 100.0
mode = sys.argv[2].lower()
base = int(sys.argv[3])
by_rate = sys.argv[4]
raw = int(sys.argv[5])
units = max(1, int(sys.argv[6]))
expansion = float(sys.argv[7])

if mode == "none":
    print(base)
elif mode == "custom":
    mapping = {}
    for item in by_rate.split(","):
        if not item.strip():
            continue
        k, v = item.split(":", 1)
        key = float(k)
        if key > 1:
            key /= 100.0
        mapping[f"{key * 100:g}"] = int(v)
    print(mapping.get(f"{ratio * 100:g}", base))
elif mode == "strict":
    print(math.ceil(raw * ratio / units))
elif mode == "expanded":
    print(math.ceil(raw * ratio * expansion / units))
else:
    raise SystemExit(f"unknown GRAPH_FINAL_CAP_MODE={mode}")
PY
}

make_lmms_order_jsonl() {
  if [[ -z "$LIMIT" || "$LIMIT" == "0" ]]; then
    return 0
  fi
  local out_jsonl="$OUT_ROOT/videomme_lmms_order_limit${LIMIT}.jsonl"
  echo "[data] writing lmms-eval ordered bench subset: $out_jsonl"
  "$PYTHON" - "$DATASET_JSONL" "$out_jsonl" "$LIMIT" <<'PY'
import json
import os
import sys
from pathlib import Path

asset_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
limit = int(float(sys.argv[3]))

by_qid = {}
with asset_path.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        qid = str(row.get("question_id") or "")
        if qid:
            by_qid[qid] = row

try:
    import datasets

    ds = datasets.load_dataset("lmms-lab/Video-MME", split="test", token=True)
    qids = [str(ds[i].get("question_id") or "") for i in range(min(limit, len(ds)))]
except Exception as exc:
    raise SystemExit(f"failed to load cached lmms-eval VideoMME order: {type(exc).__name__}: {exc}")

rows = []
missing = []
for qid in qids:
    row = by_qid.get(qid)
    if row is None:
        missing.append(qid)
    else:
        rows.append(row)
if missing:
    raise SystemExit(f"assets JSONL is missing {len(missing)} lmms-eval qids, first={missing[:10]}")

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"[data] wrote {len(rows)} rows")
PY
  BENCH_DATASET_JSONL="$out_jsonl"
}

run_bench() {
  local method="$1"
  local rate="$2"
  local label
  label="$(rate_label "$rate")"
  local bench_tag="${TAG}_bench"
  local log="$OUT_ROOT/bench_${method}_r${label}.log"

  echo "[bench] method=$method rate=$rate limit=$LIMIT log=$log"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON" -u playground/run_qwen3_matrix.py \
    --model_backend qwen3_vl \
    --model_path "$PRETRAINED" \
    --datasets "videomme=$BENCH_DATASET_JSONL" \
    --methods "$method" \
    --rates "$rate" \
    --limit "$LIMIT" \
    --tag "$bench_tag" \
    --output_dir logs/efficiency/matrix \
    --num_frames "$MAX_NUM_FRAMES" \
    --min_pixels "$MIN_PIXELS" \
    --max_pixels "$MAX_PIXELS" \
    --num_warmup 0 \
    --num_runs 1 \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --attn_implementation "$ATTN_IMPLEMENTATION" \
    --free_ratio "$FREE_RATIO" \
    --min_free_mb "$MIN_FREE_MB" \
    --max_gpus "$MAX_GPUS" \
    --retention_expansion "$EXPANSION" \
    --llm_retention_ratio "$LLM_RETENTION_RATIO" \
    --raw_visual_tokens "$RAW_VISUAL_TOKENS" \
    --visual_time_units "$VISUAL_TIME_UNITS" \
    --graph_final_cap_mode "$GRAPH_FINAL_CAP_MODE" \
    --graph_final_tokens_per_frame "$GRAPH_FINAL_TPF" \
    --graph_final_tokens_per_frame_by_rate "$GRAPH_FINAL_TPF_BY_RATE" \
    --token_selection_method "$TOKEN_SELECTION_METHOD" \
    --flashvid_token_selection_method "$FLASHVID_TOKEN_SELECTION_METHOD" \
    --graphvid_token_selection_method "$GRAPHVID_TOKEN_SELECTION_METHOD" \
    > "$log" 2>&1
}

run_lmms() {
  local method="$1"
  local rate="$2"
  local label
  label="$(rate_label "$rate")"
  local out="$OUT_ROOT/lmms_${method}_r${label}"
  local log="$OUT_ROOT/lmms_${method}_r${label}.log"
  local graph_final_tpf
  graph_final_tpf="$(graph_final_tpf_for_rate "$rate")"

  echo "[lmms] method=$method rate=$rate limit=$LIMIT log=$log"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  NUM_PROCESSES="$NUM_PROCESSES" \
  PRETRAINED="$PRETRAINED" \
  METHODS="$method" \
  TASKS=videomme \
  RATES="$rate" \
  LIMIT="$LIMIT" \
  LOG_SAMPLES="$LOG_SAMPLES" \
  FADVISE_DURING_RUN=0 \
  OUTPUT_PATH="$out" \
  MAX_NUM_FRAMES="$MAX_NUM_FRAMES" \
  MIN_PIXELS="$MIN_PIXELS" \
  MAX_PIXELS="$MAX_PIXELS" \
  GEN_KWARGS="max_new_tokens=${MAX_NEW_TOKENS},temperature=0" \
  EXPANSION="$EXPANSION" \
  LLM_RETENTION_RATIO="$LLM_RETENTION_RATIO" \
  GRAPH_FINAL_TPF="$graph_final_tpf" \
  FLASHVID_TOKEN_SELECTION_METHOD="$FLASHVID_TOKEN_SELECTION_METHOD" \
  GRAPHVID_TOKEN_SELECTION_METHOD="$GRAPHVID_TOKEN_SELECTION_METHOD" \
  ADAPTER_TOKEN_SELECTION_METHOD="$TOKEN_SELECTION_METHOD" \
  bash scripts/qwen3_vl.sh > "$log" 2>&1
}

compare_one() {
  local method="$1"
  local rate="$2"
  local label
  label="$(rate_label "$rate")"
  local bench_summary
  local lmms_out
  local lmms_log
  local out_md

  bench_summary="$(bench_summary_path "$method" "$rate")"
  lmms_out="$OUT_ROOT/lmms_${method}_r${label}"
  lmms_log="$OUT_ROOT/lmms_${method}_r${label}.log"
  out_md="$OUT_ROOT/compare_${method}_r${label}.md"

  "$PYTHON" playground/compare_videomme_lmms_bench.py \
    --method "$method" \
    --rate "$rate" \
    --bench_summary "$bench_summary" \
    --lmms_output "$lmms_out" \
    --lmms_log "$lmms_log" \
    --out_md "$out_md"
}

echo "========================================"
echo "VideoMME lmms-eval vs bench_all_metrics"
echo " Methods : $METHODS"
echo " Rates   : $RATES"
echo " Limit   : $LIMIT"
echo " Model   : $PRETRAINED"
echo " Output  : $OUT_ROOT"
echo "========================================"

make_lmms_order_jsonl

for method in $(split_csv "$METHODS"); do
  for rate in $(split_csv "$RATES"); do
    run_bench "$method" "$rate"
    run_lmms "$method" "$rate"
    compare_one "$method" "$rate"
  done
done

echo "[done] compare tables under $OUT_ROOT"
