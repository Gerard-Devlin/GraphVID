# CertVID V8 LongVideoBench Search Split

These files define deterministic, disjoint subsets of the official
LongVideoBench validation split:

- `lvb_v8_search_192.ids`: coarse hyperparameter search.
- `lvb_v8_holdout_384.ids`: validation after choosing candidates.
- Matching JSONL files contain metadata for auditing the split.
- `lvb_v8_subset_summary.json` records the distributions and seed.

Selection is label-blind. It balances:

- duration group;
- question category;
- CertVID V8 query route;
- active versus V3-fallback path;
- prediction agreement versus the supplied reference run;
- duration-by-category cells.

The answer and baseline-correctness fields are retained in the manifest for
post-hoc analysis, but they are not used by the selection algorithm.

Run the coarse search:

```bash
TAG=certvid_v8_lvb_coarse_$(date +%Y%m%d_%H%M%S)
LOG=logs/${TAG}.log
mkdir -p logs

nohup env \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  NUM_PROCESSES=6 \
  STAGE=coarse \
  SAMPLE_IDS_FILE="$PWD/assets/lvb_search/lvb_v8_search_192.ids" \
  PRETRAINED=/home/xuyouwen/models/llava-onevision-qwen2-7b-ov \
  OUTPUT_PATH="$PWD/logs/lvb_v8_search/$TAG" \
  bash scripts/search_certvid_v8_lvb.sh > "$LOG" 2>&1 &
```

After choosing the best few settings, rerun the relevant matrix on
`lvb_v8_holdout_384.ids`. Do not select a final configuration using the
holdout repeatedly; use the full 1337-example validation run once after the
holdout check.
