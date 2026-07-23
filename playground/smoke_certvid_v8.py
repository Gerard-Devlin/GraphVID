from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path

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


def _active_config(frames: int) -> FlashVidConfig:
    config = _config("certvid_v8")
    config.certv8_intent_router = False
    config.certv8_intent_strength = 0.0
    config.certv8_frame_floor_ratio = 0.95
    config.certv8_frame_cap_ratio = 1.0
    config.certv8_max_swap_ratio = 0.50
    config.certv8_concentration_preserve_ratio = 0.0
    config.certv8_query_weight = 0.0
    config.certv8_event_weight = 0.0
    config.certv8_balance_weight = 0.60
    config.certv8_design_protect_ratio = 0.0
    config.certv8_query_protect_ratio = 0.0
    config.certv8_d_efficiency_floor = 0.0
    config.certv8_min_deficit = 0.0
    config.certv8_min_objective_gain = 0.0
    config._certvid_frame_times_sec = torch.linspace(0.0, 420.0, frames)
    config._certvid_frame_times_source = "smoke"
    config._debug_sample_id = "smoke-sequence"
    config._certvid_query_text = "What happened first and then after the event?"
    config._certvid_eval_category = "sequence"
    config._certvid_task_name = "longvideobench_val_v"
    return config


def main() -> None:
    defaults = _config("certvid_v8")
    assert defaults.certv8_intent_router is True
    assert defaults.certv8_max_swap_ratio == 0.09
    assert defaults.certv8_frame_floor_ratio == 0.30
    assert defaults.certv8_frame_cap_ratio == 2.30
    assert defaults.certv8_concentration_preserve_ratio == 0.70
    assert defaults.certv8_d_efficiency_floor == 0.98
    assert defaults.certv8_query_peak_count == 2

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

    for reason, config in (
        ("missing_real_timestamps", _config("certvid_v8")),
        ("disabled", _config("certvid_v8")),
        ("short_horizon", _config("certvid_v8")),
    ):
        if reason == "disabled":
            config.certv8_enabled = False
        elif reason == "short_horizon":
            config._certvid_frame_times_sec = torch.linspace(0.0, 7.0, frames)
            config._certvid_frame_times_source = "smoke"
        output, indices = certvid_v8_compression(
            dynamic,
            attention,
            config,
            question,
        )
        assert config.last_certv8_fallback_reason == reason
        assert torch.equal(output, v3_output)
        assert torch.equal(indices, v3_indices)
        _assert_plan_equal(config._certvid_plan, v3_plan)

    active = _active_config(frames)
    with tempfile.TemporaryDirectory() as temp_dir:
        diagnostics_path = Path(temp_dir) / "certv8_{rank}.jsonl"
        previous_path = os.environ.get("CERTV8_DIAGNOSTICS_JSONL")
        previous_detail = os.environ.get("CERTV8_DIAGNOSTICS_DETAIL")
        os.environ["CERTV8_DIAGNOSTICS_JSONL"] = str(diagnostics_path)
        os.environ["CERTV8_DIAGNOSTICS_DETAIL"] = "tokens"
        try:
            output, indices = flashvid_compression(
                dynamic,
                attention,
                active,
                question,
            )
        finally:
            if previous_path is None:
                os.environ.pop("CERTV8_DIAGNOSTICS_JSONL", None)
            else:
                os.environ["CERTV8_DIAGNOSTICS_JSONL"] = previous_path
            if previous_detail is None:
                os.environ.pop("CERTV8_DIAGNOSTICS_DETAIL", None)
            else:
                os.environ["CERTV8_DIAGNOSTICS_DETAIL"] = previous_detail

        written = Path(str(diagnostics_path).replace("{rank}", "0"))
        record = json.loads(written.read_text(encoding="utf-8").splitlines()[-1])
        assert record["sample_id"] == "smoke-sequence"
        assert record["task"] == "longvideobench_val_v"
        assert record["raw_token_count"] == frames * tokens
        assert record["budget"] == indices.numel()
        assert len(record["v3_frame_counts"]) == frames
        assert len(record["final_frame_counts"]) == frames
        assert "v3_selected_indices" in record
        assert "final_selected_indices" in record

    budget = round(frames * tokens * active.retention_ratio * active.expansion)
    assert active.last_certv8_fallback_reason is None, active.last_certv8_diagnostics
    assert output.shape == (budget, dimension)
    assert indices.numel() == budget
    assert indices.unique().numel() == budget
    assert torch.equal(indices, torch.sort(indices).values)
    assert torch.isfinite(output).all()
    assert active.last_certv8_swap_count > 0
    assert active.last_certv8_modified_ratio <= active.certv8_max_swap_ratio + 1e-8
    assert (
        active.last_certv8_v3_overlap_ratio
        >= 1.0 - active.certv8_max_swap_ratio - 1e-8
    )
    assert active.last_certv8_d_efficiency >= active.certv8_d_efficiency_floor
    base_cv = active.last_certv8_diagnostics["v3_frame_distribution"]["cv"]
    final_cv = active.last_certv8_diagnostics["final_frame_distribution"]["cv"]
    assert (
        final_cv + 1e-8
        >= active.certv8_concentration_preserve_ratio * base_cv
    )
    assert active.last_certv8_diagnostics["final_deficit"] <= (
        active.last_certv8_diagnostics["base_deficit"] + 1e-8
    )
    assert active.last_certv8_diagnostics["final_query_coverage"] >= (
        active.last_certv8_diagnostics["base_query_coverage"] - 1e-8
    )
    assert active.last_certv8_diagnostics["unsafe_assignment_count"] == 0

    plan = active._certvid_plan
    source_frames = torch.arange(frames).repeat_interleave(tokens)
    anchor_frames = source_frames[indices]
    assigned_frames = anchor_frames[plan.assignment_indices]
    assert int((source_frames.unsqueeze(1) - assigned_frames).abs().max().item()) <= 1

    repeat = _active_config(frames)
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
    print("CertVID V8 V3-repair smoke passed")


if __name__ == "__main__":
    main()
