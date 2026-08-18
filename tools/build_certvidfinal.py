#!/usr/bin/env python3
"""Build the self-contained CertVID final implementation from exact sources."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FLASHVID = ROOT / "flashvid"

V1_SYMBOLS = (
    "CertVidPlan",
    "_cfg_float",
    "_cfg_int",
    "_grid_hw",
    "_minmax",
    "_rank_normalize",
    "_metric_features",
    "_spatial_layout",
    "_temporal_signals",
    "_build_components",
    "_question_atoms",
    "_question_relevance",
    "_local_detail",
    "_build_plan",
    "apply_certvid_plan",
)
V2_SYMBOLS = (
    "_exact_cuda_graph_enabled",
    "_trajectory_signals_eager",
    "_trajectory_signals",
    "_component_support",
)


def _definitions(path: Path, names: tuple[str, ...]) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    definitions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing = [name for name in names if name not in definitions]
    if missing:
        raise RuntimeError(f"missing definitions in {path}: {missing}")
    blocks = []
    for name in names:
        node = definitions[name]
        start_line = min(
            [node.lineno]
            + [decorator.lineno for decorator in getattr(node, "decorator_list", [])]
        )
        blocks.append("\n".join(source.splitlines()[start_line - 1 : node.end_lineno]))
    return "\n\n\n".join(blocks)


def _v3_body(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    first_definition = next(
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.ClassDef, ast.FunctionDef))
        and getattr(node, "lineno", 0) > 1
        and not isinstance(node, (ast.Import, ast.ImportFrom))
    )
    body = "\n".join(source.splitlines()[first_definition.lineno - 1 :])
    body = body.replace("def certvid_v3_compression(", "def certvidfinal_compression(", 1)
    body = body.replace(
        '== "certvid_v3"\n        and getattr(flashvid_config, "strict_token_budget", False)',
        'in {"certvid_v3", "certvidfinal"}\n        and getattr(flashvid_config, "strict_token_budget", False)',
        1,
    )
    body = body.replace(
        'and compression_variant == "certvid_v3"',
        'and compression_variant in {"certvid_v3", "certvidfinal"}',
        1,
    )
    body = body.replace(
        'setattr(flashvid_config, "last_adapter_variant", "certvid_v3")',
        'setattr(flashvid_config, "last_adapter_variant", "certvidfinal")',
        1,
    )
    return body


def main() -> None:
    header = '''"""Self-contained CertVID implementation equivalent to CertVID V3.

This file inlines the exact helper implementations used by CertVID V3 so the
complete algorithm can be read from one module. The legacy V1/V2/V3 modules
remain unchanged.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, MutableMapping, Optional

import torch
import torch.nn.functional as F

from .configuration_flashvid import FlashVidConfig
'''
    v1 = _definitions(FLASHVID / "certvid.py", V1_SYMBOLS)
    v2 = _definitions(FLASHVID / "certvid_v2.py", V2_SYMBOLS)
    v3 = _v3_body(FLASHVID / "certvid_v3.py")
    generated = (
        header
        + "\n\n# Exact helpers formerly imported from certvid.py.\n\n"
        + v1
        + "\n\n\n_build_components_reference = _build_components\n"
        + "\n\n\n# Exact trajectory helpers formerly imported from certvid_v2.py.\n"
        + "_TRAJECTORY_GRAPH_CACHE: dict[tuple[object, ...], dict[str, object]] = {}\n\n"
        + v2
        + "\n\n\n# CertVID V3 implementation.\n\n"
        + v3
        + "\n"
    )
    output = FLASHVID / "certvidfinal.py"
    output.write_text(generated, encoding="utf-8", newline="\n")
    print(f"wrote {output} ({len(generated.splitlines())} lines)")


if __name__ == "__main__":
    main()
