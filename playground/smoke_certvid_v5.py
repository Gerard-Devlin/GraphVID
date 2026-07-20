from __future__ import annotations

import math
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Unit tests only need the compression modules. Avoid importing model-specific
# transformers hooks so the smoke can also run in a lightweight CPU env.
package = types.ModuleType("flashvid")
package.__path__ = [str(REPO_ROOT / "flashvid")]
sys.modules.setdefault("flashvid", package)

import torch

from flashvid.certvid import apply_certvid_plan
from flashvid.certvid_qwen3 import compress_certvid_deepstack
from flashvid.certvid_v5 import (
    _CERTIFICATE_SHARES,
    _CertificateRequest,
    _admit_certificates,
    _capped_entropic_tail_weights,
    _resolve_budget,
    _scene_pyramid,
    _tail_risk_spectral_selection,
    _tie_safe_rank_normalize,
    _validated_attention,
    certvid_v5_compression,
)
from flashvid.configuration_flashvid import FlashVidConfig
from flashvid.utils import flashvid_compression


def _config(**overrides) -> FlashVidConfig:
    values = dict(
        retention_ratio=0.10,
        expansion=1.25,
        pruning_layer=20,
        llm_retention_ratio=0.30,
        compression_variant="certvid_v5",
        certv5_num_hidden_layers=28,
        certv5_inner_hook_enabled=True,
        certv5_scene_threshold=0.30,
        certv5_motion_threshold=0.0,
        certv5_motion_confidence_threshold=0.0,
    )
    values.update(overrides)
    config = FlashVidConfig(**values)
    config._certvid_attention_source = "manual_qk"
    return config


def test_attention_and_budget() -> None:
    ranks, used, reason = _tie_safe_rank_normalize(torch.tensor([0.0, 0.0, 1.0, 1.0]))
    assert used and reason == "validated"
    assert ranks[0] == ranks[1] and ranks[2] == ranks[3]

    degenerate, used, reason = _tie_safe_rank_normalize(torch.ones(8))
    assert not used and reason == "degenerate"
    assert torch.count_nonzero(degenerate) == 0

    config = _config()
    normalized, diagnostics = _validated_attention(torch.randn(2, 4), 2, 4, config)
    assert normalized.shape == (8,)
    assert diagnostics["used"] and diagnostics["source"] == "manual_qk"

    fallback_config = _config()
    fallback_config._certvid_attention_source = "feature_norm"
    fallback, diagnostics = _validated_attention(torch.randn(2, 4), 2, 4, fallback_config)
    assert torch.count_nonzero(fallback) == 0
    assert not diagnostics["used"]

    for invalid in (torch.randn(8), torch.full((2, 4), float("nan"))):
        try:
            _validated_attention(invalid, 2, 4, config)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid attention should be rejected")

    budget, diagnostics = _resolve_budget(config, 2880)
    assert budget == 360
    assert math.isclose(float(diagnostics["average_layer_multiplier"]), 1.0, abs_tol=1e-6)

    qwen_config = _config(
        pruning_layer=28,
        llm_retention_ratio=0.10,
        certv5_num_hidden_layers=36,
    )
    qwen_budget, diagnostics = _resolve_budget(qwen_config, 2880)
    assert qwen_budget == 360
    assert math.isclose(float(diagnostics["average_layer_multiplier"]), 1.0, abs_tol=1e-6)

    e130_config = _config(expansion=1.30, llm_retention_ratio=0.1923076923)
    e130_budget, diagnostics = _resolve_budget(e130_config, 2880)
    assert e130_budget == 374
    assert math.isclose(float(diagnostics["average_layer_multiplier"]), 1.0, abs_tol=1e-6)

    try:
        _resolve_budget(_config(expansion=1.30, llm_retention_ratio=0.30), 2880)
    except ValueError:
        pass
    else:
        raise AssertionError("misaligned layer-average budget should be rejected")


def test_scene_pyramid_and_atomic_certificates() -> None:
    frame_event = torch.tensor([0.0, 0.1, 0.92, 0.2, 0.88, 0.1, 0.84, 0.0])
    scene_ids, fine_ids, coarse_ids, diagnostics = _scene_pyramid(
        frame_event,
        frame_count=8,
        max_scenes=4,
        boundary_threshold=0.50,
        min_scene_frames=2,
        fine_bins=8,
        coarse_bins=4,
    )
    assert diagnostics["boundaries"] == [2, 4, 6]
    assert int(scene_ids.max()) + 1 == 4
    assert int(fine_ids.max()) + 1 == 8
    assert int(coarse_ids.max()) + 1 == 4

    requests = [
        _CertificateRequest("motion", "motion:0", (1, 2), 2.0),
        _CertificateRequest("track", "track:0", (4, 9), 1.8),
        _CertificateRequest("scene", "scene:0", (3,), 1.0),
        _CertificateRequest("frame", "frame:0", (0,), 0.9),
    ]
    locked, diagnostics, _ = _admit_certificates(
        requests,
        budget=10,
        budget_ratio=0.50,
        shares=_CERTIFICATE_SHARES,
    )
    locked_set = set(locked)
    for pair in ({1, 2}, {4, 9}):
        assert pair.issubset(locked_set) or pair.isdisjoint(locked_set)
    assert len(locked) <= 5
    assert diagnostics["admitted_unique"] == len(locked)


def test_tail_risk_recovers_weak_directions() -> None:
    tied_weights = _capped_entropic_tail_weights(
        torch.ones(4),
        tail_fraction=0.25,
        temperature=0.10,
        ridge=0.05,
    )
    assert torch.allclose(tied_weights, torch.full((4,), 0.25))
    ordered_weights = _capped_entropic_tail_weights(
        torch.tensor([0.1, 0.2, 1.0, 2.0]),
        tail_fraction=0.50,
        temperature=0.10,
        ridge=0.05,
    )
    assert math.isclose(float(ordered_weights.sum()), 1.0, abs_tol=1e-6)
    assert float(ordered_weights.max()) <= 0.5 + 1e-6
    assert ordered_weights[0] > ordered_weights[-1]

    design = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.8, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.8],
        ]
    )
    candidates = torch.arange(6, dtype=torch.long)
    groups = torch.arange(6, dtype=torch.long)
    selected, diagnostics, tail_vectors, tail_weights = _tail_risk_spectral_selection(
        design=design,
        candidates=candidates,
        demand_weight=torch.full((6,), 1.0 / 6.0),
        quality=torch.ones(6),
        locked=[0],
        budget=3,
        fine_temporal_ids=groups,
        scene_ids=groups,
        spatial_ids=groups,
        motion_sectors=groups,
        query_relevance=torch.empty((0, 6)),
        atom_weights=torch.empty(0),
        query_enabled=False,
        tail_fraction=0.34,
        tail_temperature=0.10,
        ridge=0.05,
        refresh_interval=1,
        dual_strength=0.0,
        mean_weight=0.0,
    )
    selected_set = set(selected.tolist())
    assert 0 in selected_set
    assert selected_set.intersection({2, 3})
    assert selected_set.intersection({4, 5})
    assert diagnostics["tail_cvar"] > 0.05
    assert diagnostics["minimum_eigenvalue"] > 0.05
    assert tail_vectors.shape == (3, 3)
    assert tail_weights.shape == (3,)
    assert math.isclose(float(tail_weights.sum()), 1.0, abs_tol=1e-6)


def test_query_off_isolation() -> None:
    torch.manual_seed(31)
    video = torch.randn(4, 9, 32)
    attention = torch.randn(4, 9)
    question = torch.randn(6, 32)
    without_question, indices_without = certvid_v5_compression(
        video,
        attention,
        _config(certv5_query_mode="off"),
        None,
    )
    with_question, indices_with = certvid_v5_compression(
        video,
        attention,
        _config(certv5_query_mode="off"),
        question,
    )
    assert torch.equal(indices_without, indices_with)
    assert torch.allclose(without_question, with_question)


def test_integration() -> None:
    torch.manual_seed(23)
    frame_count, tokens_per_frame, feature_dim = 8, 16, 48
    video = torch.randn(frame_count, tokens_per_frame, feature_dim)
    # Add a controlled moving feature so directed correspondences are exercised.
    for frame_idx in range(frame_count):
        video[frame_idx, (frame_idx * 2) % tokens_per_frame, :8] += 6.0
    attention = torch.randn(frame_count, tokens_per_frame)
    question = torch.randn(7, feature_dim)
    config = _config()

    output, indices = certvid_v5_compression(video, attention, config, question)
    target = round(frame_count * tokens_per_frame * 0.10 * 1.25)
    assert output.shape == (target, feature_dim)
    assert indices.numel() == target
    assert torch.equal(indices, torch.sort(indices).values)
    assert torch.unique(indices).numel() == target
    assert torch.isfinite(output).all()

    diagnostics = config.last_certv5_diagnostics
    assert diagnostics["motion_pair_count"] > 0
    assert diagnostics["certificates"]["ratio"] <= 0.28
    assert diagnostics["scene_pyramid"]["fine_count"] == frame_count
    assert 0.0 <= diagnostics["router"]["motion_activity"] <= 1.0
    assert diagnostics["spectral_design"]["objective"] > 0.0
    assert diagnostics["spectral_design"]["tail_cvar"] > 0.0
    assert diagnostics["spectral_design"]["iterations"] >= 0
    assert diagnostics["spectral_weights"]["appearance"] > 0.0
    assert diagnostics["spectral_weights"]["instance"] > 0.0
    assert "facility_location" not in diagnostics
    assert "divide_and_conquer" not in diagnostics
    assert "d_optimal" not in diagnostics
    assert "logdet" not in diagnostics

    plan = config._certvid_plan
    expected = apply_certvid_plan(video.reshape(-1, feature_dim), plan)
    assert torch.allclose(output, expected)
    assert torch.all((plan.fusion_alpha >= 0.0) & (plan.fusion_alpha <= 0.10 + 1e-6))

    second_config = _config()
    second_output, second_indices = certvid_v5_compression(
        video,
        attention,
        second_config,
        question,
    )
    assert torch.equal(second_indices, indices)
    assert torch.allclose(second_output, output)

    deepstack = [torch.randn(frame_count * tokens_per_frame, feature_dim) for _ in range(3)]
    compressed = compress_certvid_deepstack(deepstack, plan)
    assert all(features.shape[0] == target for features in compressed)

    routed, routed_indices = flashvid_compression(video, attention, config, question)
    assert routed.shape == output.shape
    assert torch.equal(routed_indices, indices)
    assert config.last_adapter_variant == "certvid_v5"


def main() -> None:
    test_attention_and_budget()
    test_scene_pyramid_and_atomic_certificates()
    test_tail_risk_recovers_weak_directions()
    test_query_off_isolation()
    test_integration()
    print("CertVID V5 smoke tests passed.")


if __name__ == "__main__":
    main()
