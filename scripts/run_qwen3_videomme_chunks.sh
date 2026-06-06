#!/usr/bin/env bash
set -euo pipefail

# Chunked lmms-eval runner for AutoDL-style memory limits.
# It keeps the normal scripts/qwen3_vl.sh method mapping, but feeds VideoMME
# through short JSONL chunks so one long lmms-eval process cannot pin all video
# file cache until the cgroup OOM killer steps in.

cd "$(dirname "$0")/.."

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-base}"
fi

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-20480}"
export LMMS_EVAL_FADVISE_DONTNEED="${LMMS_EVAL_FADVISE_DONTNEED:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUM_PROCESSES="${NUM_PROCESSES:-1}"
export ACCELERATE="${ACCELERATE:-/root/miniconda3/bin/accelerate}"
export PRETRAINED="${PRETRAINED:-/root/models}"
export LOG_SAMPLES="${LOG_SAMPLES:-0}"

METHODS="${METHODS:-flashvid,fastgraphvid}"
RATES="${RATES:-0.10}"
SRC_JSONL="${SRC_JSONL:-assets/videomme.jsonl}"
CHUNK_SIZE="${CHUNK_SIZE:-300}"
FADVISE_INTERVAL="${FADVISE_INTERVAL:-20}"
FADVISE_DURING_RUN="${FADVISE_DURING_RUN:-1}"
TAG="${TAG:-qwen3_videomme_chunks_r10_${CHUNK_SIZE}_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-logs/lmms_eval_videomme/${TAG}}"
CHUNK_DIR="${RUN_DIR}/chunks"
CURRENT_JSONL="${RUN_DIR}/videomme_current.jsonl"
YAML_PATH="lmms-eval/lmms_eval/tasks/videomme/videomme.yaml"
YAML_BACKUP="${RUN_DIR}/videomme.yaml.backup"

mkdir -p "$CHUNK_DIR"
cp "$YAML_PATH" "$YAML_BACKUP"

restore_yaml() {
  if [[ -f "$YAML_BACKUP" ]]; then
    cp "$YAML_BACKUP" "$YAML_PATH"
  fi
}
trap restore_yaml EXIT

cat > "$YAML_PATH" <<YAML
dataset_path: json
dataset_kwargs:
  data_files: ${CURRENT_JSONL}
  cache_dir: ${HF_HOME}/videomme
task: videomme
test_split: train
output_type: generate_until
cluster_key: videoID
doc_to_visual: !function utils.videomme_doc_to_visual
doc_to_text: !function utils.videomme_doc_to_text
doc_to_target: "answer"
generation_kwargs:
  max_new_tokens: 16
  temperature: 0
  top_p: 1.0
  num_beams: 1
  do_sample: false
process_results: !function utils.videomme_process_results
metric_list:
  - metric: videomme_perception_score
    aggregation: !function utils.videomme_aggregate_results
    higher_is_better: true
lmms_eval_specific_kwargs:
  default:
    pre_prompt: ""
    post_prompt: "\nAnswer with the option's letter from the given choices directly."
  qwen3_vl:
    format: "qwen3_vl"
    pre_prompt: "Question: "
    post_prompt: "Answer with the option letter only."
metadata:
  - version: 0.0
YAML

printf '[chunk-run] methods=%s rates=%s\n' "$METHODS" "$RATES"
printf '[chunk-run] source=%s chunk_size=%s\n' "$SRC_JSONL" "$CHUNK_SIZE"
printf '[chunk-run] fadvise_during_run=%s interval=%ss\n' "$FADVISE_DURING_RUN" "$FADVISE_INTERVAL"
printf '[chunk-run] run_dir=%s\n' "$RUN_DIR"

python - "$SRC_JSONL" "$CHUNK_DIR" "$CHUNK_SIZE" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
out = Path(sys.argv[2])
size = int(sys.argv[3])
rows = [line for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
manifest = []
for idx, start in enumerate(range(0, len(rows), size)):
    chunk = rows[start : start + size]
    path = out / f"chunk_{idx:03d}_{start:04d}_{start + len(chunk) - 1:04d}.jsonl"
    path.write_text("\n".join(chunk) + "\n", encoding="utf-8")
    durations = {}
    for line in chunk:
        try:
            duration = str(json.loads(line).get("duration", "unknown"))
        except Exception:
            duration = "unknown"
        durations[duration] = durations.get(duration, 0) + 1
    manifest.append(
        {
            "index": idx,
            "start": start,
            "end": start + len(chunk) - 1,
            "count": len(chunk),
            "path": str(path),
            "durations": durations,
        }
    )
(out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"wrote {len(manifest)} chunks / {len(rows)} rows")
PY

fadvise_videos() {
  python - <<'PY'
import os
from pathlib import Path

roots = [
    Path("/root/autodl-tmp/hf_home/videomme/data"),
    Path("/root/autodl-tmp/videomme_raw"),
]
count = 0
for root in roots:
    if not root.exists():
        continue
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".mp4", ".mkv", ".avi", ".mov", ".webm", ".zip"}:
            continue
        try:
            fd = os.open(str(path), os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
            count += 1
        except Exception:
            pass
print(f"[fadvise] files={count}")
PY
}

record_mem() {
  local label="$1"
  {
    echo "=== ${label} ==="
    date
    cat /sys/fs/cgroup/memory.current 2>/dev/null || true
    grep -E '^(anon|file|active_file|inactive_file) ' /sys/fs/cgroup/memory.stat 2>/dev/null || true
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true
  } | tee -a "${RUN_DIR}/memory.log"
}

record_mem_quiet() {
  local label="$1"
  {
    echo "=== ${label} ==="
    date
    cat /sys/fs/cgroup/memory.current 2>/dev/null || true
    grep -E '^(anon|file|active_file|inactive_file) ' /sys/fs/cgroup/memory.stat 2>/dev/null || true
  } >> "${RUN_DIR}/memory.log"
}

start_fadvise_loop() {
  local label="$1"
  janitor_pid=""
  case "$FADVISE_DURING_RUN" in
    1|true|True|yes|Yes) ;;
    *) return 0 ;;
  esac
  (
    while true; do
      sleep "$FADVISE_INTERVAL" || exit 0
      printf '[fadvise-loop] label=%s\n' "$label" >> "${RUN_DIR}/memory.log"
      fadvise_videos >> "${RUN_DIR}/memory.log" 2>&1 || true
      record_mem_quiet "during_${label}"
    done
  ) &
  janitor_pid="$!"
}

stop_fadvise_loop() {
  local pid="${1:-}"
  if [[ -n "$pid" ]]; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}

split_csv() {
  local text="$1"
  text="${text//,/ }"
  # shellcheck disable=SC2086
  printf '%s\n' $text
}

status_csv="${RUN_DIR}/chunk_status.csv"
echo "method,rate,chunk,start,end,count,exit_code,log,output_dir" > "$status_csv"
record_mem "before_all"
fadvise_videos | tee -a "${RUN_DIR}/memory.log"

for method in $(split_csv "$METHODS"); do
  for rate in $(split_csv "$RATES"); do
    for chunk in "$CHUNK_DIR"/chunk_*.jsonl; do
      name=$(basename "$chunk" .jsonl)
      idx=$(echo "$name" | cut -d_ -f2)
      start=$(echo "$name" | cut -d_ -f3)
      end=$(echo "$name" | cut -d_ -f4)
      count=$(wc -l < "$chunk" | tr -d " ")
      cp "$chunk" "$CURRENT_JSONL"

      out_dir="${RUN_DIR}/out_${method}_r${rate}_${name}"
      log="${RUN_DIR}/${method}_r${rate}_${name}.log"
      mkdir -p "$out_dir"

      printf '[chunk-run] method=%s rate=%s chunk=%s start=%s end=%s count=%s\n' \
        "$method" "$rate" "$idx" "$start" "$end" "$count" | tee -a "${RUN_DIR}/launcher.log"
      record_mem "before_${method}_r${rate}_${name}"

      janitor_pid=""
      start_fadvise_loop "${method}_r${rate}_${name}"
      set +e
      METHODS="$method" \
      RATES="$rate" \
      TASKS=videomme \
      OUTPUT_PATH="$out_dir" \
      bash scripts/qwen3_vl.sh > "$log" 2>&1
      code=$?
      stop_fadvise_loop "$janitor_pid"
      set -e

      echo "${method},${rate},${idx},${start},${end},${count},${code},${log},${out_dir}" >> "$status_csv"
      tail -n 30 "$log" >> "${RUN_DIR}/launcher.log" || true
      record_mem "after_${method}_r${rate}_${name}"
      fadvise_videos | tee -a "${RUN_DIR}/memory.log"
      record_mem "after_fadvise_${method}_r${rate}_${name}"

      if [[ "$code" -ne 0 ]]; then
        echo "[chunk-run] failed method=${method} rate=${rate} chunk=${idx} code=${code}" | tee -a "${RUN_DIR}/launcher.log"
        exit "$code"
      fi
    done
  done
done

python - "$RUN_DIR" <<'PY'
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

run = Path(sys.argv[1])
rows = []
for log in sorted(run.glob("*_chunk_*.log")):
    text = log.read_text(errors="ignore")
    match = re.findall(r"Overall Performance:\s+([0-9.]+)%", text)
    score = float(match[-1]) if match else None
    parts = log.name.split("_chunk_", 1)[0]
    method_rate = parts.rsplit("_r", 1)
    method = method_rate[0]
    rate = method_rate[1] if len(method_rate) == 2 else ""
    rows.append({"method": method, "rate": rate, "log": log.name, "overall_percent": score})

summary_path = run / "summary.csv"
with summary_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["method", "rate", "log", "overall_percent"])
    writer.writeheader()
    writer.writerows(rows)

groups = defaultdict(list)
for row in rows:
    if row["overall_percent"] is not None:
        groups[(row["method"], row["rate"])].append(row["overall_percent"])

print(f"[chunk-run] summary={summary_path}")
for (method, rate), values in sorted(groups.items()):
    print(f"[chunk-run] {method} r{rate} chunks={len(values)} simple_mean={sum(values) / len(values):.4f}")
PY

record_mem "after_all"
printf '[chunk-run] done run_dir=%s\n' "$RUN_DIR"
