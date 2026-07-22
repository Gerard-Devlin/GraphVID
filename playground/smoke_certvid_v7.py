from __future__ import annotations

import copy
import math

import torch

from flashvid.certvid_qwen3 import compress_certvid_deepstack
from flashvid.certvid_v3 import certvid_v3_compression
from flashvid.certvid_v7 import _long_horizon_budget, certvid_v7_compression
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
    config.certv3_frame_coverage_ratio = 0.50
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


def _assert_flashvid_shape_parity() -> None:
    torch.manual_seed(17)
    frames, tokens, dim = 32, 196, 32
    token_basis = torch.randn(tokens, dim)
    video = []
    for frame in range(frames):
        current = token_basis + 0.015 * frame * torch.randn(1, dim)
        current = current + 0.02 * torch.randn(tokens, dim)
        if frame in (7, 15, 23):
            current[:32] = current[:32] + 1.2 * torch.randn(32, dim)
        video.append(current)
    video = torch.stack(video)
    attention = torch.randn(frames, tokens)
    question = torch.randn(6, dim)

    flash_config = FlashVidConfig()
    flash_config.compression_variant = "flashvid"
    flash_config.retention_ratio = 0.10
    flash_config.expansion = 1.30
    flash_output, _ = flashvid_compression(video, attention, flash_config, question)

    v7_config = _config("certvid_v7")
    v7_config.retention_ratio = 0.10
    v7_config.expansion = 1.30
    v7_config.certv7_min_duration_seconds = 1.0
    v7_config.certv7_min_reallocation_ratio = 0.0
    v7_config.certv7_d_efficiency_floor = 0.0
    v7_config._certvid_frame_times_sec = torch.linspace(0.0, 1800.0, frames)
    v7_config._certvid_frame_times_source = "shape_parity"
    v7_output, v7_indices = flashvid_compression(video, attention, v7_config, question)

    per_frame_budget = math.ceil(tokens * v7_config.retention_ratio * v7_config.expansion)
    target_budget = frames * per_frame_budget
    frame_counts = torch.bincount(v7_indices // tokens, minlength=frames)
    assert target_budget == 832
    assert v7_output.shape[0] == target_budget
    assert torch.all(frame_counts == per_frame_budget), frame_counts.tolist()
    assert v7_config.last_certv7_target_tokens == target_budget
    assert v7_config.last_certv7_native_v3_tokens == 815
    # FlashVID's nested cluster ceil can add a handful of content-dependent tokens.
    assert abs(flash_output.shape[0] - v7_output.shape[0]) <= 8
    assert v7_config.last_certv7_unsafe_assignment_count == 0
    print(
        "shape parity: "
        f"flashvid={flash_output.shape[0]} v7={v7_output.shape[0]} "
        f"v7_per_frame={int(frame_counts.min())}-{int(frame_counts.max())}"
    )


def main() -> None:
    torch.manual_seed(5)
    frames, tokens, dim = 8, 16, 48
    token_basis = torch.randn(tokens, dim)
    features = []
    for frame in range(frames):
        drift = 0.025 * frame * torch.randn(1, dim)
        current = token_basis + drift + 0.015 * torch.randn(tokens, dim)
        if frame >= 4:
            current[:6] = current[:6] + 1.5 * torch.randn(6, dim)
        features.append(current)
    features = torch.stack(features)
    attention = torch.randn(frames, tokens)
    question = torch.randn(6, dim)

    parity_config = _config("certvid_v7")
    parity_config.retention_ratio = 0.10
    parity_config.expansion = 1.30
    target, native, mode = _long_horizon_budget(32, 196, parity_config)
    assert (target, native, mode) == (832, 815, "per_frame_ceil")

    v3_config = _config("certvid_v3")
    v3_output, v3_indices = certvid_v3_compression(
        features,
        attention,
        v3_config,
        question,
    )
    v3_plan = copy.deepcopy(v3_config._certvid_plan)

    short_config = _config("certvid_v7")
    short_config._certvid_frame_times_sec = torch.linspace(0.0, 70.0, frames)
    short_config._certvid_frame_times_source = "smoke_short"
    short_output, short_indices = certvid_v7_compression(
        features,
        attention,
        short_config,
        question,
    )
    assert torch.equal(short_indices, v3_indices)
    assert torch.equal(short_output, v3_output)
    _assert_plan_equal(short_config._certvid_plan, v3_plan)
    assert short_config.last_certv7_fallback_reason == "short_horizon"

    missing_config = _config("certvid_v7")
    missing_output, missing_indices = certvid_v7_compression(
        features,
        attention,
        missing_config,
        question,
    )
    assert torch.equal(missing_indices, v3_indices)
    assert torch.equal(missing_output, v3_output)
    _assert_plan_equal(missing_config._certvid_plan, v3_plan)
    assert missing_config.last_certv7_fallback_reason == "missing_timestamps"

    long_config = _config("certvid_v7")
    long_config._certvid_frame_times_sec = torch.tensor(
        [0.0, 35.0, 80.0, 145.0, 230.0, 340.0, 470.0, 620.0]
    )
    long_config._certvid_frame_times_source = "smoke_long"
    long_config.certv7_min_duration_seconds = 1.0
    long_config.certv7_min_reallocation_ratio = 0.0
    long_config.certv7_d_efficiency_floor = 0.0
    long_config.certv7_transport_steps = 6
    long_output, long_indices = certvid_v7_compression(
        features,
        attention,
        long_config,
        question,
    )
    expected_budget = round(frames * tokens * long_config.retention_ratio)
    assert long_config.last_certv7_fallback_reason is None, long_config.last_certv7_diagnostics
    assert long_output.shape == (expected_budget, dim)
    assert long_indices.numel() == expected_budget
    assert long_indices.unique().numel() == expected_budget
    assert torch.equal(long_indices, torch.sort(long_indices).values)
    assert torch.isfinite(long_output).all()
    assert sum(long_config.last_certv7_diagnostics["frame_budgets"]) == expected_budget
    assert long_config.last_certv7_frame_budget_sum == expected_budget
    assert long_config.last_certv7_unsafe_assignment_count == 0
    assert long_config.last_certv7_frame_budget_min >= 1
    assert long_config.last_certv7_pair_cost_max >= long_config.last_certv7_pair_cost_mean
    assert long_config.last_certv7_selection_change_ratio > 0.10
    assert long_config.last_certv7_v3_anchor_overlap_ratio < 0.90
    certificate_cap = round(expected_budget * long_config.certv7_v3_certificate_ratio)
    assert long_config.last_certv7_v3_certificate_count <= certificate_cap
    assert long_config.last_certv7_transition_relay_count > 0
    assert long_config.last_certv7_trajectory_relay_count > 0
    assert long_config.last_certv7_final_coverage > 0.0

    plan = long_config._certvid_plan
    source_frames = torch.arange(frames).repeat_interleave(tokens)
    anchor_frames = source_frames[long_indices]
    assigned_frames = anchor_frames[plan.assignment_indices]
    assert int((source_frames.unsqueeze(1) - assigned_frames).abs().max().item()) <= 1
    mandatory_count = (
        long_config.last_certv7_v3_certificate_count
        + long_config.last_certv7_query_relay_count
        + long_config.last_certv7_transition_relay_count
        + long_config.last_certv7_trajectory_relay_count
    )
    assert int((plan.fusion_alpha <= 1e-12).sum().item()) >= mandatory_count
    assert float(plan.fusion_alpha.max().item()) <= long_config.certv7_long_fusion_alpha + 1e-12

    repeat_config = _config("certvid_v7")
    for name in (
        "_certvid_frame_times_sec",
        "_certvid_frame_times_source",
        "certv7_min_duration_seconds",
        "certv7_min_reallocation_ratio",
        "certv7_d_efficiency_floor",
        "certv7_transport_steps",
    ):
        setattr(repeat_config, name, copy.deepcopy(getattr(long_config, name)))
    repeat_output, repeat_indices = certvid_v7_compression(
        features,
        attention,
        repeat_config,
        question,
    )
    assert torch.equal(repeat_indices, long_indices)
    assert torch.equal(repeat_output, long_output)
    _assert_plan_equal(repeat_config._certvid_plan, long_config._certvid_plan)

    deepstack = [torch.randn(frames * tokens, dim) for _ in range(3)]
    compressed = compress_certvid_deepstack(deepstack, long_config._certvid_plan)
    assert len(compressed) == len(deepstack)
    assert all(layer.shape == (expected_budget, dim) for layer in compressed)
    assert all(torch.isfinite(layer).all() for layer in compressed)
    _assert_flashvid_shape_parity()
    print("CertVID V7 relation-evidence smoke passed")


if __name__ == "__main__":
    main()
