#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/FlashVID_official
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

export PYTHONPATH=/root/autodl-tmp/FlashVID_official:/root/autodl-tmp/FlashVID_official/lmms-eval:${PYTHONPATH:-}
export HF_HOME=/root/autodl-tmp/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DECORD_EOF_RETRY_MAX=20480
export CUDA_VISIBLE_DEVICES=0

SRC_JSONL=${SRC_JSONL:-/root/autodl-tmp/GraphVID/assets/videomme.jsonl}
CHUNK_SIZE=${CHUNK_SIZE:-300}
TAG=${TAG:-official_flashvid_qwen3_videomme_r10_chunked${CHUNK_SIZE}_$(date +%Y%m%d_%H%M%S)}
RUN_DIR=/root/autodl-tmp/FlashVID_official/logs/official_flashvid/${TAG}
CHUNK_DIR=${RUN_DIR}/chunks
CURRENT_JSONL=${RUN_DIR}/videomme_current.jsonl
YAML_PATH=/root/autodl-tmp/FlashVID_official/lmms-eval/lmms_eval/tasks/videomme/videomme.yaml
UTILS_PATH=/root/autodl-tmp/FlashVID_official/lmms-eval/lmms_eval/tasks/videomme/utils.py

mkdir -p "$CHUNK_DIR"

printf '[chunk-run] tag=%s\n' "$TAG"
printf '[chunk-run] run_dir=%s\n' "$RUN_DIR"
printf '[chunk-run] source=%s chunk_size=%s\n' "$SRC_JSONL" "$CHUNK_SIZE"

# Use the local VideoMME JSONL adapter only. Official FlashVID algorithm and
# official qwen3_vl wrapper stay untouched.
cp /root/autodl-tmp/GraphVID/lmms-eval/lmms_eval/tasks/videomme/utils.py "$UTILS_PATH"
cat > "$YAML_PATH" <<YAML
dataset_path: json
dataset_kwargs:
  data_files: ${CURRENT_JSONL}
  cache_dir: /root/autodl-tmp/hf_home/videomme
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

/root/miniconda3/bin/python - "$SRC_JSONL" "$CHUNK_DIR" "$CHUNK_SIZE" <<'PY'
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
    durs = {}
    for line in chunk:
        try:
            dur = str(json.loads(line).get("duration", "unknown"))
        except Exception:
            dur = "unknown"
        durs[dur] = durs.get(dur, 0) + 1
    manifest.append(
        {
            "index": idx,
            "start": start,
            "end": start + len(chunk) - 1,
            "count": len(chunk),
            "path": str(path),
            "durations": durs,
        }
    )
(out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"wrote {len(manifest)} chunks / {len(rows)} rows")
PY

fadvise_videos() {
  /root/miniconda3/bin/python - <<'PY'
import os
from pathlib import Path

n = 0
for root in [Path("/root/autodl-tmp/hf_home/videomme/data"), Path("/root/autodl-tmp/videomme_raw")]:
    if not root.exists():
        continue
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".mp4", ".mkv", ".avi", ".mov", ".webm", ".zip"}:
            try:
                fd = os.open(str(p), os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
                n += 1
            except Exception:
                pass
print(f"[fadvise] files={n}")
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

MODEL_ARGS="pretrained=/root/models,max_num_frames=32,min_pixels=50176,max_pixels=200704,attn_implementation=flash_attention_2,enable_flashvid=True,retention_ratio=0.10,expansion=1.25,do_segment=True,segment_threshold=0.9,min_segment_num=8,complementary_segment=True,alpha=0.70,temporal_threshold=0.8,pruning_layer=20,llm_retention_ratio=1.0,token_selection_method=attn_div_v2"

record_mem "before_all"
fadvise_videos

status_csv=${RUN_DIR}/chunk_status.csv
echo "chunk,start,end,count,exit_code,log,output_dir" > "$status_csv"

for chunk in "$CHUNK_DIR"/chunk_*.jsonl; do
  name=$(basename "$chunk" .jsonl)
  idx=$(echo "$name" | cut -d_ -f2)
  start=$(echo "$name" | cut -d_ -f3)
  end=$(echo "$name" | cut -d_ -f4)
  count=$(wc -l < "$chunk" | tr -d " ")
  cp "$chunk" "$CURRENT_JSONL"
  out_dir=${RUN_DIR}/out_${name}
  log=${RUN_DIR}/${name}.log
  mkdir -p "$out_dir"

  printf '[chunk-run] chunk=%s start=%s end=%s count=%s\n' "$idx" "$start" "$end" "$count" | tee -a "${RUN_DIR}/launcher.log"
  record_mem "before_${name}"

  set +e
  /root/miniconda3/bin/accelerate launch \
    --main_process_port 18890 \
    --num_processes 1 \
    -m lmms_eval \
    --model qwen3_vl \
    --model_args "$MODEL_ARGS" \
    --tasks videomme \
    --batch_size 1 \
    --gen_kwargs max_new_tokens=16,temperature=0 \
    --output_path "$out_dir" \
    > "$log" 2>&1
  code=$?
  set -e

  echo "${idx},${start},${end},${count},${code},${log},${out_dir}" >> "$status_csv"
  tail -n 30 "$log" >> "${RUN_DIR}/launcher.log" || true
  record_mem "after_${name}"
  fadvise_videos | tee -a "${RUN_DIR}/memory.log"
  record_mem "after_fadvise_${name}"
  if [ "$code" -ne 0 ]; then
    echo "[chunk-run] failed chunk=${idx} code=${code}; stopping" | tee -a "${RUN_DIR}/launcher.log"
    exit "$code"
  fi
done

/root/miniconda3/bin/python - "$RUN_DIR" <<'PY'
import csv
import re
import sys
from pathlib import Path

run = Path(sys.argv[1])
rows = []
for log in sorted(run.glob("chunk_*.log")):
    text = log.read_text(errors="ignore")
    m = re.findall(r"Overall Performance:\s+([0-9.]+)%", text)
    score = float(m[-1]) if m else None
    rows.append((log.name, score))
summary = run / "summary.csv"
with summary.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["log", "overall_percent"])
    w.writerows(rows)
vals = [v for _, v in rows if v is not None]
print(f"[chunk-run] completed_chunks={len(vals)}/{len(rows)}")
if vals:
    print(f"[chunk-run] simple_mean_overall={sum(vals) / len(vals):.4f}")
print(f"[chunk-run] summary={summary}")
PY

record_mem "after_all"
printf '[chunk-run] done run_dir=%s\n' "$RUN_DIR"
