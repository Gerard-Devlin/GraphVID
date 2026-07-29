#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TAG="${TAG:-videomme_parity_four_methods_r10_limit50_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-logs/compare_lmms_bench/${TAG}}"
LOG="${LOG:-logs/compare_lmms_bench/${TAG}.log}"
PARITY_TOLERANCE="${PARITY_TOLERANCE:-0.0}"

mkdir -p "$(dirname "$LOG")"

echo "========================================"
echo "VideoMME parity check"
echo " TAG     : $TAG"
echo " OUT_ROOT: $OUT_ROOT"
echo " LOG     : $LOG"
echo "========================================"

set +e
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
PRETRAINED="${PRETRAINED:-/root/models}" \
HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_home}" \
METHODS="${METHODS:-flashvid,graphvid,fastgraphvid}" \
RATES="${RATES:-0.10}" \
LIMIT="${LIMIT:-50}" \
LOG_SAMPLES="${LOG_SAMPLES:-1}" \
NUM_PROCESSES="${NUM_PROCESSES:-1}" \
MAX_GPUS="${MAX_GPUS:-1}" \
FREE_RATIO="${FREE_RATIO:-0.0}" \
MIN_FREE_MB="${MIN_FREE_MB:-0}" \
FAIL_ON_MISMATCH=0 \
OUT_ROOT="$OUT_ROOT" \
PARITY_TOLERANCE="$PARITY_TOLERANCE" \
bash scripts/compare_videomme_lmms_bench.sh > "$LOG" 2>&1
run_code=$?
set -e

echo "========================================"
echo "Compare log tail"
echo "========================================"
tail -n 80 "$LOG" || true

echo "========================================"
echo "Compare tables"
echo "========================================"
find "$OUT_ROOT" -maxdepth 1 -name 'compare_*.md' -print -exec cat {} \; || true

if [[ "$run_code" -ne 0 ]]; then
  echo "[parity] compare run failed before final parity check, code=$run_code"
  exit "$run_code"
fi

python - "$OUT_ROOT" "$PARITY_TOLERANCE" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
tol = float(sys.argv[2])
bad = []
tables = sorted(root.glob("compare_*.md"))
if not tables:
    print(f"[parity] no compare_*.md files under {root}")
    sys.exit(2)

for path in tables:
    text = path.read_text(encoding="utf-8", errors="ignore")
    mm = re.search(r"mismatches:\s*(\d+)", text)
    if mm and int(mm.group(1)) != 0:
        bad.append(f"{path.name}: sample mismatches={mm.group(1)}")
    for line in text.splitlines():
        if not line.startswith("| ") or "Delta" in line or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 6:
            continue
        delta = cells[5]
        if delta == "-":
            continue
        try:
            value = float(delta)
        except ValueError:
            continue
        if abs(value) > tol:
            bad.append(f"{path.name}: {cells[2]} delta={value:.4f}")

if bad:
    print("[parity] NOT aligned")
    for item in bad:
        print("  - " + item)
    sys.exit(2)

print("[parity] aligned: all score deltas within tolerance and no sample mismatches")
PY
