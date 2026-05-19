from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from playground import run_qwen3_matrix


DEFAULT_QWEN25_MODEL_PATH = (
    "/gluster/envs/users/wuzhijian/hf_home/hub/"
    "models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots"
)


def _resolve_default_model_path() -> str:
    root = Path(DEFAULT_QWEN25_MODEL_PATH)
    if root.is_dir():
        snapshots = [p for p in root.iterdir() if p.is_dir()]
        if snapshots:
            snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return str(snapshots[0])
    return DEFAULT_QWEN25_MODEL_PATH


def main() -> None:
    if not any(arg == "--model_backend" for arg in sys.argv[1:]):
        sys.argv.extend(["--model_backend", "qwen2_5_vl"])
    if not any(arg == "--model_path" for arg in sys.argv[1:]):
        sys.argv.extend(["--model_path", _resolve_default_model_path()])
    if not any(arg == "--llm_retention_ratio" for arg in sys.argv[1:]):
        sys.argv.extend(["--llm_retention_ratio", "0.3"])
    if not any(arg == "--min_pixels" for arg in sys.argv[1:]):
        sys.argv.extend(["--min_pixels", str(256 * 28 * 28)])
    if not any(arg == "--max_pixels" for arg in sys.argv[1:]):
        sys.argv.extend(["--max_pixels", str(1605632)])
    if not any(arg == "--token_selection_method" for arg in sys.argv[1:]):
        sys.argv.extend(["--token_selection_method", "attn_div"])
    if not any(arg == "--flashvid_token_selection_method" for arg in sys.argv[1:]):
        sys.argv.extend(["--flashvid_token_selection_method", "attn_div"])
    if not any(arg == "--graphvid_token_selection_method" for arg in sys.argv[1:]):
        sys.argv.extend(["--graphvid_token_selection_method", "attn_div"])
    if not any(arg == "--tag" for arg in sys.argv[1:]):
        sys.argv.extend(["--tag", "qwen25_7b_matrix"])
    run_qwen3_matrix.main()


if __name__ == "__main__":
    main()
