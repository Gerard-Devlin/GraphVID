from __future__ import annotations

import copy

import torch

from flashvid.certvid_qwen3 import compress_certvid_deepstack
from flashvid.certvid_v3 import certvid_v3_compression
from flashvid.certvid_v8 import certvid_v8_compression
from flashvid.configuration_flashvid import FlashVidConfig
from flashvid.utils import flashvid_compression


def _config(variant: str) -> FlashVidConfig:
    config = FlashVidConfig()
    config.compression_variant = variant
    config.retention_ratio = 0.25
    config.expansion = 1.0
    config.certv3_budget_uses_expansion = True
    config.certv3_metric_dim = 32
    config.certv3_temporal_bins = 4
    config.certv3_spatial_bins = 2
    config.certv3_frame_coverage_ratio = 1.0
    config.certv3_cell_coverage_ratio = 0.0
    config.certv3_candidate_multiplier = 2.5
    config.certv3_swap_steps = 2
    config.certv3_swap_pool = 12
    return config


def _assert_plan_equal(left, right) -> None:
    assert left.raw_token_count == right.raw_token_count
    for name in (
        "anchor_indices",
        "assignment_indices",
        "assignment_weights",
        "source_mass",
        "fusion_alpha",
    ):
        assert torch.equal(getattr(left, name), getattr(right, name)), name


def _v3_reference(features, attention, question):
    config = _config("certvid_v3")
    output, indices = certvid_v3_compression(features, attention, config, question)
    return output, indices, copy.deepcopy(config._certvid_plan)


def main() -> None:
    defaults = _config("certvid_v8")
    assert defaults.certv8_long_max_swap_ratio == 0.06
    assert defaults.certv8_long_d_efficiency_floor == 0.98

    torch.manual_seed(8)
    frames, tokens, dimension = 8, 16, 32
    base = torch.randn(tokens, dimension)
    static = base.unsqueeze(0).repeat(frames, 1, 1)
    attention = torch.randn(frames, tokens)
    question = torch.randn(4, dimension)

    v3_output, v3_indices, v3_plan = _v3_reference(static, attention, question)
    static_config = _config("certvid_v8")
    static_output, static_indices = certvid_v8_compression(
        static,
        attention,
        static_config,
        question,
    )
    assert static_config.last_certv8_fallback_reason == "weak_relation_signal"
    assert torch.equal(static_output, v3_output)
    assert torch.equal(static_indices, v3_indices)
    _assert_plan_equal(static_config._certvid_plan, v3_plan)

    dynamic_frames = []
    for frame in range(frames):
        current = base + 0.01 * torch.randn(tokens, dimension)
        if frame >= 3:
            current[:5] = current[:5] + 1.8 * torch.randn(5, dimension)
        if frame >= 6:
            current[5:10] = current[5:10] - 1.5 * torch.randn(5, dimension)
        dynamic_frames.append(current)
    dynamic = torch.stack(dynamic_frames)

    disabled_v3_output, disabled_v3_indices, disabled_v3_plan = _v3_reference(
        dynamic,
        attention,
        question,
    )
    disabled = _config("certvid_v8")
    disabled.certv8_enabled = False
    disabled_output, disabled_indices = certvid_v8_compression(
        dynamic,
        attention,
        disabled,
        question,
    )
    assert disabled.last_certv8_fallback_reason == "disabled"
    assert torch.equal(disabled_output, disabled_v3_output)
    assert torch.equal(disabled_indices, disabled_v3_indices)
    _assert_plan_equal(disabled._certvid_plan, disabled_v3_plan)

    active = _config("certvid_v8")
    active.certv8_gate_threshold = 0.0
    active.certv8_min_relation_deficit = 0.0
    active.certv8_min_relation_gain = 0.0
    active.certv8_d_efficiency_floor = 0.0
    active.certv8_short_max_swap_ratio = 0.20
    active._certvid_frame_times_sec = torch.linspace(0.0, 420.0, frames)
    active._certvid_frame_times_source = "smoke"
    output, indices = flashvid_compression(
        dynamic,
        attention,
        active,
        question,
    )
    budget = round(frames * tokens * active.retention_ratio * active.expansion)
    assert active.last_certv8_fallback_reason is None, active.last_certv8_diagnostics
    assert output.shape == (budget, dimension)
    assert indices.numel() == budget
    assert indices.unique().numel() == budget
    assert torch.equal(indices, torch.sort(indices).values)
    assert torch.isfinite(output).all()
    assert active.last_certv8_swap_count > 0
    assert active.last_certv8_modified_ratio <= active.certv8_long_max_swap_ratio + 1e-8
    assert active.last_certv8_v3_overlap_ratio >= 1.0 - active.certv8_long_max_swap_ratio - 1e-8
    assert (
        active.last_certv8_final_relation_coverage
        >= active.last_certv8_base_relation_coverage
    )
    assert active.last_certv8_unsafe_assignment_count == 0

    plan = active._certvid_plan
    source_frames = torch.arange(frames).repeat_interleave(tokens)
    anchor_frames = source_frames[indices]
    assigned_frames = anchor_frames[plan.assignment_indices]
    assert int((source_frames.unsqueeze(1) - assigned_frames).abs().max().item()) <= 1

    repeat = _config("certvid_v8")
    for name in (
        "certv8_gate_threshold",
        "certv8_min_relation_deficit",
        "certv8_min_relation_gain",
        "certv8_d_efficiency_floor",
        "certv8_short_max_swap_ratio",
        "_certvid_frame_times_sec",
        "_certvid_frame_times_source",
    ):
        setattr(repeat, name, copy.deepcopy(getattr(active, name)))
    repeat_output, repeat_indices = certvid_v8_compression(
        dynamic,
        attention,
        repeat,
        question,
    )
    assert torch.equal(repeat_output, output)
    assert torch.equal(repeat_indices, indices)
    _assert_plan_equal(repeat._certvid_plan, plan)

    deepstack = [torch.randn(frames * tokens, dimension) for _ in range(3)]
    compressed = compress_certvid_deepstack(deepstack, plan)
    assert len(compressed) == len(deepstack)
    assert all(layer.shape == (budget, dimension) for layer in compressed)
    assert all(torch.isfinite(layer).all() for layer in compressed)
    print("CertVID V8 relation-witness smoke passed")


if __name__ == "__main__":
    main()
