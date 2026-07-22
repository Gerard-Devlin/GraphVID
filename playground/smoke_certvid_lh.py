from __future__ import annotations

from types import SimpleNamespace

import torch

from flashvid.certvid_lh import certvid_lh_compression
from flashvid.certvid_v3 import certvid_v3_compression


def _config(*, long_video: bool) -> SimpleNamespace:
    frame_times = torch.linspace(0.0, 900.0 if long_video else 30.0, steps=8).tolist()
    return SimpleNamespace(
        retention_ratio=0.25,
        expansion=1.0,
        certv3_budget_uses_expansion=False,
        compression_variant="certvid_lh",
        H=4,
        W=4,
        certv3_metric_dim=32,
        certv3_query_atoms=4,
        certv3_temporal_bins=4,
        certv3_spatial_bins=2,
        certv3_candidate_multiplier=2.0,
        certv3_query_weight=0.18,
        certv3_track_threshold=0.82,
        certv3_spatial_penalty=0.08,
        certv3_frame_coverage_ratio=1.0,
        certv3_cell_coverage_ratio=0.5,
        certv3_query_threshold=0.1,
        certv3_query_per_atom=1,
        certv3_structural_weight=0.32,
        certv3_whitening_strength=0.5,
        certv3_quality_floor=0.15,
        certv3_ridge=0.5,
        certv3_swap_steps=2,
        certv3_swap_pool=8,
        certv3_swap_margin=1e-4,
        certv3_fusion_alpha=0.12,
        certv3_assignment_temperature=0.07,
        certlh_min_duration_seconds=120.0,
        certlh_horizon_gap_seconds=4.0,
        certlh_gate_threshold=0.55,
        certlh_min_groups=4,
        certlh_max_groups=4,
        certlh_min_group_units=2,
        certlh_max_group_units=2,
        certlh_event_quantile=0.80,
        certlh_event_floor=0.08,
        certlh_group_floor_ratio=0.50,
        certlh_budget_temperature=0.25,
        certlh_query_weight=0.35,
        certlh_relay_ratio=0.10,
        certlh_query_peaks_per_atom=2,
        certlh_query_peak_quantile=0.90,
        certlh_query_peak_floor=0.75,
        certlh_query_min_group_distance=2,
        certlh_cross_group_similarity=0.90,
        certlh_cross_group_max_seconds=8.0,
        certlh_debug=False,
        _certvid_frame_times_sec=frame_times,
        _certvid_frame_times_source="smoke",
    )


def _assert_plan_equal(left, right) -> None:
    assert torch.equal(left.anchor_indices, right.anchor_indices)
    assert torch.equal(left.assignment_indices, right.assignment_indices)
    assert torch.equal(left.assignment_weights, right.assignment_weights)
    assert torch.equal(left.source_mass, right.source_mass)
    assert torch.equal(left.fusion_alpha, right.fusion_alpha)


def main() -> None:
    torch.manual_seed(7)
    video = torch.randn(8, 16, 48, dtype=torch.float32)
    attention = torch.randn(8, 16, dtype=torch.float32)
    question = torch.randn(12, 48, dtype=torch.float32)

    short_lh = _config(long_video=False)
    short_v3 = _config(long_video=False)
    lh_output, lh_indices = certvid_lh_compression(video, attention, short_lh, question)
    v3_output, v3_indices = certvid_v3_compression(video, attention, short_v3, question)
    assert torch.equal(lh_output, v3_output)
    assert torch.equal(lh_indices, v3_indices)
    _assert_plan_equal(short_lh._certvid_plan, short_v3._certvid_plan)
    assert short_lh.last_certlh_diagnostics["mode"] == "v3"

    missing_lh = _config(long_video=True)
    missing_v3 = _config(long_video=True)
    missing_lh._certvid_frame_times_sec = None
    missing_output, missing_indices = certvid_lh_compression(video, attention, missing_lh, question)
    expected_output, expected_indices = certvid_v3_compression(video, attention, missing_v3, question)
    assert torch.equal(missing_output, expected_output)
    assert torch.equal(missing_indices, expected_indices)
    _assert_plan_equal(missing_lh._certvid_plan, missing_v3._certvid_plan)
    assert missing_lh.last_certlh_diagnostics["fallback_reason"] == "missing_timestamps"

    long_config = _config(long_video=True)
    output, indices = certvid_lh_compression(video, attention, long_config, question)
    expected = round(video.shape[0] * video.shape[1] * long_config.retention_ratio)
    assert output.shape == (expected, video.shape[-1])
    assert indices.numel() == expected
    assert indices.unique().numel() == expected
    assert torch.equal(indices, torch.sort(indices).values)
    assert torch.isfinite(output).all()
    assert long_config.last_certlh_diagnostics["mode"] == "long_horizon"
    assert sum(long_config.last_certlh_diagnostics["group_budgets"]) + int(
        long_config.last_certlh_diagnostics["relay_tokens"]
    ) == expected
    assert sum(
        int(long_config.last_certlh_diagnostics[name])
        for name in (
            "relay_query_tokens",
            "relay_boundary_tokens",
            "relay_transition_tokens",
            "relay_context_tokens",
            "relay_fill_tokens",
        )
    ) == int(long_config.last_certlh_diagnostics["relay_tokens"])
    frame_groups = torch.empty(video.shape[0], dtype=torch.long)
    for group, (start, end) in enumerate(long_config.last_certlh_diagnostics["group_boundaries"]):
        frame_groups[start:end] = group
    token_groups = frame_groups.repeat_interleave(video.shape[1])
    plan = long_config._certvid_plan
    assigned_groups = token_groups[plan.anchor_indices][plan.assignment_indices]
    source_groups = token_groups.unsqueeze(1)
    active_edges = plan.assignment_weights > 1e-6
    assert torch.all((assigned_groups - source_groups).abs()[active_edges] <= 1)

    repeat = _config(long_video=True)
    repeat_output, repeat_indices = certvid_lh_compression(video, attention, repeat, question)
    assert torch.equal(indices, repeat_indices)
    assert torch.equal(output, repeat_output)
    print("CertVID-LH smoke passed")


if __name__ == "__main__":
    main()
