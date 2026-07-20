from __future__ import annotations

import math
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Compression tests do not need model-specific Transformers hooks.
package = types.ModuleType("flashvid")
package.__path__ = [str(REPO_ROOT / "flashvid")]
sys.modules.setdefault("flashvid", package)

import torch

from flashvid.certvid import CertVidPlan, apply_certvid_plan
from flashvid.certvid_qwen3 import compress_certvid_deepstack
from flashvid.certvid_v3 import certvid_v3_compression
from flashvid.certvid_v5 import (
    _recover_residual_plan,
    _resolve_budget,
    certvid_v5_compression,
)
from flashvid.configuration_flashvid import FlashVidConfig
from flashvid.utils import flashvid_compression


def _config(**overrides: object) -> FlashVidConfig:
    values: dict[str, object] = {
        "retention_ratio": 0.10,
        "expansion": 1.25,
        "pruning_layer": 20,
        "llm_retention_ratio": 0.30,
        "compression_variant": "certvid_v5",
        "certv3_budget_uses_expansion": True,
        "certv5_budget_mode": "layer_average",
        "certv5_num_hidden_layers": 28,
        "certv5_inner_hook_enabled": True,
        "H": 4,
        "W": 4,
    }
    values.update(overrides)
    return FlashVidConfig(**values)


def _clone_plan(plan: CertVidPlan) -> CertVidPlan:
    return CertVidPlan(
        anchor_indices=plan.anchor_indices.clone(),
        assignment_indices=plan.assignment_indices.clone(),
        assignment_weights=plan.assignment_weights.clone(),
        source_mass=plan.source_mass.clone(),
        fusion_alpha=plan.fusion_alpha.clone(),
        raw_token_count=plan.raw_token_count,
    )


def _assert_plan_equal(left: CertVidPlan, right: CertVidPlan) -> None:
    assert left.raw_token_count == right.raw_token_count
    assert torch.equal(left.anchor_indices, right.anchor_indices)
    assert torch.equal(left.assignment_indices, right.assignment_indices)
    assert torch.equal(left.assignment_weights, right.assignment_weights)
    assert torch.equal(left.source_mass, right.source_mass)
    assert torch.equal(left.fusion_alpha, right.fusion_alpha)


def test_budget_contract() -> None:
    budget, diagnostics = _resolve_budget(_config(), 2880)
    assert budget == 360
    assert math.isclose(float(diagnostics["average_layer_multiplier"]), 1.0, abs_tol=1e-6)

    qwen = _config(
        pruning_layer=28,
        llm_retention_ratio=0.10,
        certv5_num_hidden_layers=36,
    )
    budget, diagnostics = _resolve_budget(qwen, 2880)
    assert budget == 360
    assert math.isclose(float(diagnostics["average_layer_multiplier"]), 1.0, abs_tol=1e-6)

    e1275 = _config(expansion=1.275, llm_retention_ratio=0.2450980392)
    _, diagnostics = _resolve_budget(e1275, 2880)
    assert math.isclose(float(diagnostics["average_layer_multiplier"]), 1.0, abs_tol=1e-6)

    e130 = _config(expansion=1.30, llm_retention_ratio=0.1923076923)
    budget, diagnostics = _resolve_budget(e130, 2880)
    assert budget == 374
    assert math.isclose(float(diagnostics["average_layer_multiplier"]), 1.0, abs_tol=1e-6)

    outer = _config(
        expansion=1.0,
        llm_retention_ratio=1.0,
        certv5_budget_mode="outer_only",
        certv5_inner_hook_enabled=False,
    )
    budget, diagnostics = _resolve_budget(outer, 2880)
    assert budget == 288 and diagnostics["mode"] == "outer_only"

    invalid = (
        _config(expansion=1.30, llm_retention_ratio=0.30),
        _config(certv5_inner_hook_enabled=False),
        _config(certv5_budget_mode="outer_only"),
    )
    for config in invalid:
        try:
            _resolve_budget(config, 2880)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid V5 budget contract should be rejected")


def test_ot_disabled_is_exact_v3() -> None:
    torch.manual_seed(31)
    video = torch.randn(8, 16, 48)
    attention = torch.rand(8, 16)
    question = torch.randn(9, 48)

    v3_config = _config(compression_variant="certvid_v3")
    v3_output, v3_indices = certvid_v3_compression(video, attention, v3_config, question)
    v3_plan = _clone_plan(v3_config._certvid_plan)

    v5_config = _config(certv5_ot_enabled=False)
    v5_output, v5_indices = certvid_v5_compression(video, attention, v5_config, question)

    assert torch.equal(v5_indices, v3_indices)
    assert torch.equal(v5_output, v3_output)
    _assert_plan_equal(v5_config._certvid_plan, v3_plan)
    assert v5_config.last_certv5_diagnostics["v3_anchor_match"]
    assert v5_config.last_certv5_diagnostics["transport"]["fallback_reason"] == "ot_disabled_exact_v3"
    assert v5_config.last_adapter_variant == "certvid_v5"


def test_residual_recovery_transport() -> None:
    video = torch.zeros(2, 4, 32)
    video[0, 0, :2] = torch.tensor([1.0, 0.0])
    video[0, 1, :2] = torch.tensor([1.0, 0.02])
    video[0, 2, :2] = torch.tensor([1.0, 0.01])
    video[0, 3, :2] = torch.tensor([1.0, 0.03])
    video[1, 0, :2] = torch.tensor([0.0, 1.0])
    video[1, 1, :2] = torch.tensor([0.02, 1.0])
    video[1, 2, :2] = torch.tensor([0.01, 1.0])
    video[1, 3, :2] = torch.tensor([0.03, 1.0])

    baseline = CertVidPlan(
        anchor_indices=torch.tensor([0, 1, 4, 5]),
        assignment_indices=torch.tensor([[0], [1], [0], [1], [2], [3], [2], [3]]),
        assignment_weights=torch.ones(8, 1),
        source_mass=torch.ones(8),
        fusion_alpha=torch.tensor([0.0, 0.10, 0.0, 0.10]),
        raw_token_count=8,
    )
    plan, diagnostics = _recover_residual_plan(
        video_features=video,
        baseline=baseline,
        config=_config(certv5_ot_live_fraction=0.25),
    )

    assert not diagnostics["fallback"]
    assert diagnostics["dead_mass_before"] > 0.0
    assert diagnostics["dead_mass_after"] < diagnostics["dead_mass_before"]
    assert diagnostics["rerouted_mass"] > 0.0
    assert diagnostics["row_mass_error"] <= 1e-5
    assert diagnostics["live_transport_fraction"] <= 0.25
    assert diagnostics["max_cost_excess"] <= 0.05 + 1e-5
    assert diagnostics["capacity_kl_after"] <= diagnostics["capacity_kl_before"] + 1e-5
    assert diagnostics["max_relative_displacement"] <= 0.12 + 1e-5
    assert diagnostics["min_output_anchor_cosine"] >= 0.98 - 1e-5
    assert torch.equal(plan.anchor_indices, baseline.anchor_indices)
    assert torch.allclose(plan.assignment_weights.sum(dim=1), torch.ones(8), atol=1e-6)
    assert torch.all(plan.fusion_alpha <= baseline.fusion_alpha + 1e-7)
    assert torch.equal(plan.fusion_alpha[baseline.fusion_alpha == 0.0], torch.zeros(2))

    for anchor_position, source_index in enumerate(baseline.anchor_indices.tolist()):
        assert int(plan.assignment_indices[source_index, 0]) == anchor_position
        assert float(plan.assignment_weights[source_index, 0]) == 1.0


def test_integration_and_deepstack() -> None:
    torch.manual_seed(47)
    frame_count, tokens_per_frame, feature_dim = 8, 16, 48
    video = torch.randn(frame_count, tokens_per_frame, feature_dim)
    attention = torch.rand(frame_count, tokens_per_frame)
    question = torch.randn(8, feature_dim)

    v3_config = _config(compression_variant="certvid_v3")
    _, v3_indices = certvid_v3_compression(video, attention, v3_config, question)
    v3_plan = _clone_plan(v3_config._certvid_plan)

    config = _config()
    output, indices = certvid_v5_compression(video, attention, config, question)
    plan = config._certvid_plan
    expected_tokens = round(frame_count * tokens_per_frame * 0.10 * 1.25)

    assert output.shape == (expected_tokens, feature_dim)
    assert torch.equal(indices, v3_indices)
    assert torch.equal(indices, torch.sort(indices).values)
    assert torch.unique(indices).numel() == expected_tokens
    assert torch.isfinite(output).all()
    assert torch.allclose(output, apply_certvid_plan(video.reshape(-1, feature_dim), plan))
    assert torch.all(plan.fusion_alpha <= v3_plan.fusion_alpha + 1e-7)
    assert torch.equal(
        plan.fusion_alpha[v3_plan.fusion_alpha == 0.0],
        torch.zeros_like(plan.fusion_alpha[v3_plan.fusion_alpha == 0.0]),
    )
    assert torch.allclose(plan.assignment_weights.sum(dim=1), torch.ones(frame_count * tokens_per_frame))
    assert config.last_certv5_diagnostics["v3_anchor_match"]

    second_config = _config()
    second_output, second_indices = certvid_v5_compression(
        video,
        attention,
        second_config,
        question,
    )
    assert torch.equal(second_indices, indices)
    assert torch.equal(second_output, output)

    deepstack = [
        torch.randn(frame_count * tokens_per_frame, 40),
        torch.randn(frame_count * tokens_per_frame, 64),
    ]
    compressed = compress_certvid_deepstack(deepstack, plan)
    assert [tuple(layer.shape) for layer in compressed] == [
        (expected_tokens, 40),
        (expected_tokens, 64),
    ]

    routed_config = _config()
    routed, routed_indices = flashvid_compression(
        video,
        attention,
        routed_config,
        question,
    )
    assert torch.equal(routed_indices, indices)
    assert torch.equal(routed, output)
    assert routed_config.last_adapter_variant == "certvid_v5"


def test_all_public_rates_preserve_v3_anchors() -> None:
    torch.manual_seed(67)
    video = torch.randn(16, 72, 32)
    attention = torch.rand(16, 72)
    question = torch.randn(8, 32)
    for rate in (0.10, 0.15, 0.20, 0.25):
        v3_config = _config(
            retention_ratio=rate,
            compression_variant="certvid_v3",
            H=8,
            W=9,
        )
        _, v3_indices = certvid_v3_compression(video, attention, v3_config, question)
        v3_plan = _clone_plan(v3_config._certvid_plan)

        v5_config = _config(retention_ratio=rate, H=8, W=9)
        output, indices = certvid_v5_compression(video, attention, v5_config, question)
        plan = v5_config._certvid_plan
        expected = int(v3_indices.numel())
        assert output.shape == (expected, video.shape[-1])
        assert torch.equal(indices, v3_indices)
        assert torch.all(plan.fusion_alpha <= v3_plan.fusion_alpha + 1e-7)
        assert torch.equal(
            plan.fusion_alpha[v3_plan.fusion_alpha == 0.0],
            torch.zeros_like(plan.fusion_alpha[v3_plan.fusion_alpha == 0.0]),
        )
        assert torch.allclose(plan.assignment_weights.sum(dim=1), torch.ones(1152), atol=1e-6)
        assert torch.isfinite(output).all()
        transport = v5_config.last_certv5_diagnostics["transport"]
        assert not transport["fallback"]
        assert transport["dead_mass_after"] < transport["dead_mass_before"]
        assert transport["max_cost_excess"] <= 0.05 + 1e-5
        assert transport["capacity_kl_after"] <= transport["capacity_kl_before"] + 1e-5


def main() -> None:
    test_budget_contract()
    test_ot_disabled_is_exact_v3()
    test_residual_recovery_transport()
    test_integration_and_deepstack()
    test_all_public_rates_preserve_v3_anchors()
    print("CertVID V5 V3-anchor residual-recovery smoke passed")


if __name__ == "__main__":
    main()
