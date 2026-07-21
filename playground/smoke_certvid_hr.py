"""CPU smoke tests for CertVID-HR invariants."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from flashvid.certvid_hr import (
    _V3Analysis,
    _analysis_from_sink,
    _normalize_frame_times,
    _query_requirements,
    _repair_selection,
    _v3_analysis,
    certvid_hr_compression,
)
from flashvid.certvid_qwen3 import compress_certvid_deepstack
from flashvid.certvid_v3 import certvid_v3_compression
from flashvid.configuration_flashvid import FlashVidConfig
from flashvid.utils import flashvid_compression


def _same_plan(left, right) -> bool:
    tensor_fields = (
        "anchor_indices",
        "assignment_indices",
        "assignment_weights",
        "source_mass",
        "fusion_alpha",
    )
    return left.raw_token_count == right.raw_token_count and all(
        torch.equal(getattr(left, field), getattr(right, field)) for field in tensor_fields
    )


def _fallback_tests() -> None:
    torch.manual_seed(17)
    video = torch.randn(8, 16, 64)
    attention = torch.randn(8, 16)
    question = torch.randn(10, 64)

    v3_config = FlashVidConfig(retention_ratio=0.25, expansion=1.0)
    v3_output, v3_indices = certvid_v3_compression(video, attention, v3_config, question)
    v3_plan = v3_config._certvid_plan

    missing_config = FlashVidConfig(retention_ratio=0.25, expansion=1.0)
    missing_output, missing_indices = certvid_hr_compression(
        video,
        attention,
        missing_config,
        question,
    )
    assert missing_config.last_certhr_fallback_reason == "missing_timestamps"
    assert torch.equal(v3_output, missing_output)
    assert torch.equal(v3_indices, missing_indices)
    assert _same_plan(v3_plan, missing_config._certvid_plan)

    short_config = FlashVidConfig(retention_ratio=0.25, expansion=1.0)
    short_config._certvid_frame_times_sec = list(range(8))
    short_config._certvid_frame_times_source = "smoke"
    short_output, short_indices = certvid_hr_compression(video, attention, short_config, question)
    assert short_config.last_certhr_fallback_reason == "short_horizon"
    assert torch.equal(v3_output, short_output)
    assert torch.equal(v3_indices, short_indices)
    assert _same_plan(v3_plan, short_config._certvid_plan)


def _timestamp_tests() -> None:
    raw = torch.arange(32, dtype=torch.float32)
    mapped, error = _normalize_frame_times(raw, 16, device=torch.device("cpu"))
    assert error is None
    assert mapped is not None
    assert torch.equal(mapped, raw.reshape(16, 2).mean(dim=1))

    mapped, error = _normalize_frame_times([0.0, 2.0, 1.0], 3, device=torch.device("cpu"))
    assert mapped is None and error == "nonmonotonic_timestamps"
    mapped, error = _normalize_frame_times([0.0, float("nan")], 2, device=torch.device("cpu"))
    assert mapped is None and error == "nonfinite_timestamps"


def _analysis_capture_test() -> None:
    torch.manual_seed(23)
    video = torch.randn(6, 16, 64)
    attention = torch.randn(6, 16)
    question = torch.randn(9, 64)

    baseline_config = FlashVidConfig(retention_ratio=0.25, expansion=1.0, H=4, W=4)
    baseline_output, baseline_indices = certvid_v3_compression(
        video,
        attention,
        baseline_config,
        question,
    )

    captured: dict[str, object] = {}
    capture_config = FlashVidConfig(retention_ratio=0.25, expansion=1.0, H=4, W=4)
    capture_output, capture_indices = certvid_v3_compression(
        video,
        attention,
        capture_config,
        question,
        analysis_sink=captured,
    )
    assert torch.equal(capture_output, baseline_output)
    assert torch.equal(capture_indices, baseline_indices)
    assert _same_plan(capture_config._certvid_plan, baseline_config._certvid_plan)

    captured_analysis = _analysis_from_sink(captured)
    rebuilt_analysis = _v3_analysis(video, attention, question, capture_config)
    tensor_fields = (
        "metric_flat",
        "design",
        "demand_weight",
        "attention",
        "query_score",
        "query_relevance",
        "component_ids",
        "frame_ids",
        "temporal_ids",
    )
    for field in tensor_fields:
        torch.testing.assert_close(
            getattr(captured_analysis, field),
            getattr(rebuilt_analysis, field),
            rtol=0.0,
            atol=0.0,
        )
    assert captured_analysis.query_confidence == rebuilt_analysis.query_confidence
    assert captured_analysis.ridge == rebuilt_analysis.ridge


def _long_horizon_fallback_test() -> None:
    video = torch.ones(8, 16, 64)
    attention = torch.linspace(0.0, 1.0, steps=128).reshape(8, 16)
    v3_config = FlashVidConfig(retention_ratio=0.25, expansion=1.0, H=4, W=4)
    v3_output, v3_indices = certvid_v3_compression(video, attention, v3_config, None)

    hr_config = FlashVidConfig(
        retention_ratio=0.25,
        expansion=1.0,
        compression_variant="certvid_hr",
        H=4,
        W=4,
    )
    hr_config._certvid_frame_times_sec = [float(frame * 10) for frame in range(8)]
    hr_config._certvid_frame_times_source = "smoke_long"
    hr_output, hr_indices = flashvid_compression(video, attention, hr_config, None)

    assert hr_config.last_certhr_fallback_reason == "coverage_sufficient"
    assert torch.equal(hr_output, v3_output)
    assert torch.equal(hr_indices, v3_indices)
    assert _same_plan(v3_config._certvid_plan, hr_config._certvid_plan)

    deepstack = [torch.randn(128, 24), torch.randn(128, 40)]
    compressed = compress_certvid_deepstack(deepstack, hr_config._certvid_plan)
    assert [tuple(layer.shape) for layer in compressed] == [(32, 24), (32, 40)]
    assert all(torch.isfinite(layer).all() for layer in compressed)


def _query_deficit_test() -> None:
    relevance = torch.tensor([[1.0, 0.9, 0.1, 0.0, 1.0, 0.1]])
    chunks = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    selected = torch.tensor([0, 2, 5], dtype=torch.long)
    requirements, missing = _query_requirements(
        relevance,
        1.0,
        chunks,
        selected,
        confidence_threshold=0.1,
        peak_quantile=0.9,
        peak_floor=0.75,
    )
    assert {(atom, chunk) for atom, chunk, _ in requirements} == {(0, 0), (0, 2)}
    assert [(atom, chunk) for atom, chunk, _ in missing] == [(0, 2)]


def _repair_test() -> None:
    # Three chunks: chunk 2's selected anchor is an outlier, while token 9 is
    # representative.  Token 1 is redundant in a well-covered source chunk.
    metric = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.99, 0.01, 0.0, 0.0],
            [0.98, 0.02, 0.0, 0.0],
            [0.97, 0.03, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.99, 0.01, 0.0],
            [0.0, 0.98, 0.02, 0.0],
            [0.0, 0.97, 0.03, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    metric = F.normalize(metric, dim=-1)
    design = torch.randn(12, 6, generator=torch.Generator().manual_seed(4))
    design[9] = design[1]
    selected = torch.tensor([0, 1, 4, 8], dtype=torch.long)
    chunk_ids = torch.arange(3).repeat_interleave(4)
    analysis = _V3Analysis(
        metric_flat=metric,
        design=design,
        demand_weight=torch.full((12,), 1.0 / 12),
        attention=torch.zeros(12),
        query_score=torch.zeros(12),
        query_relevance=torch.empty(0, 12),
        query_confidence=0.0,
        component_ids=torch.arange(12),
        frame_ids=torch.arange(3).repeat_interleave(4),
        temporal_ids=torch.arange(3).repeat_interleave(4),
        ridge=0.5,
    )
    config = SimpleNamespace(
        certhr_max_swap_ratio=0.05,
        certhr_coverage_floor=0.70,
        certhr_deficit_threshold=0.05,
        certhr_add_pool=32,
        certhr_remove_pool=24,
        certhr_d_efficiency_floor=0.995,
        certv3_query_threshold=0.10,
        certhr_query_peak_quantile=0.90,
        certhr_query_peak_floor=0.75,
    )
    repaired, diagnostics = _repair_selection(analysis, selected, {0}, chunk_ids, config)
    assert diagnostics["swap_count"] == 1
    assert repaired.numel() == selected.numel()
    assert repaired.unique().numel() == repaired.numel()
    assert torch.equal(repaired, torch.sort(repaired).values)
    assert 9 in repaired.tolist()
    assert 0 in repaired.tolist()
    assert 4 in repaired.tolist()
    assert 1 not in repaired.tolist()
    assert diagnostics["d_efficiency"] >= 0.995
    assert diagnostics["swap_count"] <= int(torch.ceil(torch.tensor(0.05 * selected.numel())).item())


def main() -> None:
    _fallback_tests()
    _timestamp_tests()
    _analysis_capture_test()
    _long_horizon_fallback_test()
    _query_deficit_test()
    _repair_test()
    print("CertVID-HR smoke checks passed")


if __name__ == "__main__":
    main()
