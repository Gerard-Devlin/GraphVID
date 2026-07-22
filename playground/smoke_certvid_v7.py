from __future__ import annotations

import copy

import torch

from flashvid.certvid_qwen3 import compress_certvid_deepstack
from flashvid.certvid_v3 import certvid_v3_compression
from flashvid.certvid_v7 import (
    _Edges,
    _causal_distortion,
    _fit_transition,
    _query_edges,
    certvid_v7_compression,
)
from flashvid.configuration_flashvid import FlashVidConfig


def _config(variant: str) -> FlashVidConfig:
    config = FlashVidConfig()
    config.compression_variant = variant
    config.retention_ratio = 0.25
    config.expansion = 1.0
    config.certv3_budget_uses_expansion = True
    config.certv3_metric_dim = 32
    config.certv3_temporal_bins = 8
    config.certv3_spatial_bins = 2
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


def main() -> None:
    torch.manual_seed(5)
    frames, tokens, dim = 8, 16, 48
    token_basis = torch.randn(tokens, dim)
    temporal_drift = torch.randn(frames, 1, dim) * 0.04
    features = token_basis.unsqueeze(0) + temporal_drift + 0.02 * torch.randn(frames, tokens, dim)
    attention = torch.randn(frames, tokens)
    question = torch.randn(6, dim)

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
    long_config.certv7_min_path_residual = 0.0
    long_config.certv7_max_swap_ratio = 0.25
    long_config.certv7_add_pool = 32
    long_config.certv7_remove_pool = 32
    long_output, long_indices = certvid_v7_compression(
        features,
        attention,
        long_config,
        question,
    )
    expected_budget = round(frames * tokens * long_config.retention_ratio)
    assert long_output.shape == (expected_budget, dim)
    assert long_indices.numel() == expected_budget
    assert long_indices.unique().numel() == expected_budget
    assert torch.equal(long_indices, torch.sort(long_indices).values)
    assert torch.isfinite(long_output).all()
    assert long_config.last_certv7_swap_count > 0
    assert long_config.last_certv7_modified_ratio <= 0.25 + 1e-8
    assert long_config.last_certv7_final_path_loss < long_config.last_certv7_base_path_loss
    assert long_config.last_certv7_d_efficiency >= long_config.certv7_d_efficiency_floor
    assert long_config.last_certv7_local_edge_count > 0
    assert long_config.last_certv7_skip_edge_count > 0

    query_relevance = torch.zeros(1, frames * tokens)
    query_relevance[0, 2] = 1.0
    query_relevance[0, 5 * tokens + 7] = 1.0
    query_edges = _query_edges(
        query_relevance,
        query_confidence=1.0,
        frame_count=frames,
        tokens_per_frame=tokens,
        frame_times=long_config._certvid_frame_times_sec,
        max_peaks=3,
        min_frame_gap=2,
    )
    assert query_edges.count == 1
    assert int(query_edges.source[0].item()) < int(query_edges.target[0].item())

    # A learned forward operator must not reduce to sign-invariant edge loss.
    causal_metric = torch.tensor(
        [[1.0, 0.0], [0.8, 0.3], [0.1, 1.0], [-0.8, 0.4]],
        dtype=torch.float32,
    )
    causal_edges = _Edges(
        source=torch.tensor([0, 1, 2]),
        target=torch.tensor([1, 2, 3]),
        weight=torch.ones(3),
    )
    causal_family = _fit_transition(causal_metric, causal_edges, ridge=0.2)
    causal_reconstruction = causal_metric.clone()
    causal_reconstruction[1] = 0.65 * causal_metric[0] + 0.35 * causal_metric[1]
    forward_loss = _causal_distortion(
        causal_metric,
        causal_reconstruction,
        causal_family,
    ).mean()
    reverse_metric = causal_metric.flip(0)
    reverse_reconstruction = causal_reconstruction.flip(0)
    reverse_family = _fit_transition(reverse_metric, causal_edges, ridge=0.2)
    reverse_loss = _causal_distortion(
        reverse_metric,
        reverse_reconstruction,
        reverse_family,
    ).mean()
    assert not torch.isclose(forward_loss, reverse_loss, atol=1e-5)

    # Collapsing two states of one persistent entity must create transition loss.
    state_metric = torch.tensor([[1.0, 0.0], [0.9, 0.2], [0.8, 0.7]])
    state_edges = _Edges(torch.tensor([0, 1]), torch.tensor([1, 2]), torch.ones(2))
    state_family = _fit_transition(state_metric, state_edges, ridge=0.1)
    collapsed_states = state_metric.clone()
    collapsed_states[2] = collapsed_states[0]
    exact_state_loss = _causal_distortion(state_metric, state_metric, state_family).mean()
    collapsed_state_loss = _causal_distortion(
        state_metric,
        collapsed_states,
        state_family,
    ).mean()
    assert exact_state_loss + 1e-4 < collapsed_state_loss

    # Equal node error can have very different causal transition fidelity.
    scalar_metric = torch.tensor([[0.1], [0.2], [0.4], [0.8]])
    scalar_family = _fit_transition(scalar_metric, causal_edges, ridge=1e-6)
    faithful_error = torch.tensor([[0.01], [0.02], [0.04], [0.08]])
    scrambled_error = torch.tensor([[0.08], [-0.04], [0.02], [-0.01]])
    assert torch.isclose(faithful_error.square().mean(), scrambled_error.square().mean())
    faithful_loss = _causal_distortion(
        scalar_metric,
        scalar_metric + faithful_error,
        scalar_family,
    ).mean()
    scrambled_loss = _causal_distortion(
        scalar_metric,
        scalar_metric + scrambled_error,
        scalar_family,
    ).mean()
    assert faithful_loss + 1e-4 < scrambled_loss

    repeated_metric = torch.ones(4, 2)
    repeated_family = _fit_transition(repeated_metric, causal_edges, ridge=0.1)
    assert float(
        _causal_distortion(repeated_metric, repeated_metric, repeated_family).max().item()
    ) == 0.0

    repeat_config = _config("certvid_v7")
    for name in (
        "_certvid_frame_times_sec",
        "_certvid_frame_times_source",
        "certv7_min_duration_seconds",
        "certv7_min_path_residual",
        "certv7_d_efficiency_floor",
        "certv7_path_margin",
        "certv7_max_swap_ratio",
        "certv7_add_pool",
        "certv7_remove_pool",
        "certv7_cross_time_similarity",
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

    deepstack = [torch.randn(frames * tokens, dim) for _ in range(3)]
    compressed = compress_certvid_deepstack(deepstack, long_config._certvid_plan)
    assert len(compressed) == len(deepstack)
    assert all(layer.shape == (expected_budget, dim) for layer in compressed)
    assert all(torch.isfinite(layer).all() for layer in compressed)
    print("CertVID V7 smoke passed")


if __name__ == "__main__":
    main()
