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
    assert defaults.certv8_long_max_swap_ratio == 0.20
    assert defaults.certv8_long_d_efficiency_floor == 0.95
    assert defaults.certv8_local_mix == 0.55

    torch.manual_seed(8)
    frames, tokens, dimension = 8, 16, 32
    base = torch.randn(tokens, dimension)
    attention = torch.randn(frames, tokens)
    question = torch.randn(4, dimension)

    dynamic_frames = []
    for frame in range(frames):
        current = base + 0.02 * torch.randn(tokens, dimension)
        current = current.roll(shifts=frame % 4, dims=0)
        if frame >= 2:
            current[:5] += (0.5 + 0.3 * frame) * torch.randn(5, dimension)
        if frame >= 5:
            current[7:12] -= 1.6 * torch.randn(5, dimension)
        dynamic_frames.append(current)
    dynamic = torch.stack(dynamic_frames)

    v3_output, v3_indices, v3_plan = _v3_reference(
        dynamic,
        attention,
        question,
    )

    short = _config("certvid_v8")
    short._certvid_frame_times_sec = torch.linspace(0.0, 7.0, frames)
    short._certvid_frame_times_source = "smoke"
    short_output, short_indices = certvid_v8_compression(
        dynamic,
        attention,
        short,
        question,
    )
    assert short.last_certv8_fallback_reason == "short_horizon"
    assert torch.equal(short_output, v3_output)
    assert torch.equal(short_indices, v3_indices)
    _assert_plan_equal(short._certvid_plan, v3_plan)

    disabled = _config("certvid_v8")
    disabled.certv8_enabled = False
    disabled_output, disabled_indices = certvid_v8_compression(
        dynamic,
        attention,
        disabled,
        question,
    )
    assert disabled.last_certv8_fallback_reason == "disabled"
    assert torch.equal(disabled_output, v3_output)
    assert torch.equal(disabled_indices, v3_indices)
    _assert_plan_equal(disabled._certvid_plan, v3_plan)

    active = _config("certvid_v8")
    active.certv8_local_mix = 1.0
    active.certv8_min_disagreement_ratio = 0.0
    active.certv8_min_joint_gain = 0.0
    active.certv8_v3_coverage_weight = 0.10
    active.certv8_long_d_efficiency_floor = 0.0
    active.certv8_long_max_swap_ratio = 0.50
    active.certv8_design_protect_ratio = 0.0
    active.certv8_swap_margin = -1.0
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
    assert (
        active.last_certv8_v3_overlap_ratio
        >= 1.0 - active.certv8_long_max_swap_ratio - 1e-8
    )
    assert (
        active.last_certv8_final_joint_coverage
        >= active.last_certv8_base_joint_coverage
    )
    assert active.last_certv8_unsafe_assignment_count == 0

    plan = active._certvid_plan
    source_frames = torch.arange(frames).repeat_interleave(tokens)
    anchor_frames = source_frames[indices]
    assigned_frames = anchor_frames[plan.assignment_indices]
    assert int((source_frames.unsqueeze(1) - assigned_frames).abs().max().item()) <= 1

    repeat = _config("certvid_v8")
    for name in (
        "certv8_local_mix",
        "certv8_min_disagreement_ratio",
        "certv8_min_joint_gain",
        "certv8_v3_coverage_weight",
        "certv8_long_d_efficiency_floor",
        "certv8_long_max_swap_ratio",
        "certv8_design_protect_ratio",
        "certv8_swap_margin",
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
    print("CertVID V8 complementary-coreset smoke passed")


if __name__ == "__main__":
    main()
