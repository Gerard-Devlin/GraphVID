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

from flashvid.certvid import apply_certvid_plan
from flashvid.certvid_e import _e_optimal_refine, certvid_e_compression
from flashvid.certvid_qwen3 import compress_certvid_deepstack
from flashvid.certvid_v3 import certvid_v3_compression
from flashvid.configuration_flashvid import FlashVidConfig
from flashvid.utils import flashvid_compression


def _config(**overrides: object) -> FlashVidConfig:
    values: dict[str, object] = {
        "retention_ratio": 0.10,
        "expansion": 1.25,
        "compression_variant": "certvid_e",
        "certv3_budget_uses_expansion": True,
        "certe_budget_uses_expansion": True,
        "H": 4,
        "W": 4,
    }
    values.update(overrides)
    return FlashVidConfig(**values)


def test_e_refinement_improves_weakest_direction() -> None:
    design = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.8],
        ],
        dtype=torch.float32,
    )
    candidates = torch.arange(design.shape[0])
    selected = torch.tensor([0, 1, 2])
    (
        refined,
        swaps,
        active_rank,
        lambda_before,
        lambda_after,
        tail_mean,
        logdet,
        d_efficiency,
    ) = _e_optimal_refine(
        selected=selected,
        candidates=candidates,
        design=design,
        mandatory=[0],
        ridge=0.10,
        steps=4,
        bottom_k=2,
        remove_pool=3,
        add_pool=2,
        verify_pool=8,
        margin=1e-6,
        spectral_temperature=0.05,
        d_efficiency_floor=0.0,
        rank_tolerance=1e-5,
    )

    assert swaps >= 1
    assert active_rank == 3
    assert lambda_after > lambda_before
    assert 0 in refined.tolist()
    assert int(torch.unique(refined).numel()) == refined.numel()
    assert all(math.isfinite(value) for value in (tail_mean, logdet, d_efficiency))


def test_disabled_refinement_is_exact_v3() -> None:
    torch.manual_seed(37)
    video = torch.randn(8, 16, 48)
    attention = torch.rand(8, 16)
    question = torch.randn(9, 48)

    v3_config = _config(compression_variant="certvid_v3")
    v3_output, v3_indices = certvid_v3_compression(video, attention, v3_config, question)

    e_config = _config(certe_swap_steps=0)
    e_output, e_indices = certvid_e_compression(video, attention, e_config, question)

    assert torch.equal(e_indices, v3_indices)
    assert torch.equal(e_output, v3_output)
    assert torch.equal(e_config._certvid_plan.anchor_indices, v3_config._certvid_plan.anchor_indices)
    assert torch.equal(e_config._certvid_plan.assignment_indices, v3_config._certvid_plan.assignment_indices)
    assert torch.equal(e_config._certvid_plan.assignment_weights, v3_config._certvid_plan.assignment_weights)
    assert torch.equal(e_config._certvid_plan.source_mass, v3_config._certvid_plan.source_mass)
    assert torch.equal(e_config._certvid_plan.fusion_alpha, v3_config._certvid_plan.fusion_alpha)
    assert e_config.last_certe_e_swap_count == 0.0


def test_integration_and_deepstack() -> None:
    torch.manual_seed(53)
    frames, tokens_per_frame, feature_dim = 8, 16, 48
    video = torch.randn(frames, tokens_per_frame, feature_dim)
    attention = torch.rand(frames, tokens_per_frame)
    question = torch.randn(8, feature_dim)

    config = _config()
    output, indices = certvid_e_compression(video, attention, config, question)
    expected = round(frames * tokens_per_frame * 0.10 * 1.25)

    assert output.shape == (expected, feature_dim)
    assert indices.shape == (expected,)
    assert torch.equal(indices, torch.sort(indices).values)
    assert int(torch.unique(indices).numel()) == expected
    assert int(indices.min()) >= 0
    assert int(indices.max()) < frames * tokens_per_frame
    assert torch.isfinite(output).all()
    assert torch.equal(output, apply_certvid_plan(video.reshape(-1, feature_dim), config._certvid_plan))
    assert config.last_adapter_variant == "certvid_e"

    diagnostics = config.last_certe_diagnostics
    assert diagnostics["lambda_min"] + 1e-6 >= diagnostics["lambda_min_before"]
    assert diagnostics["d_efficiency"] + 1e-6 >= config.certe_d_efficiency_floor
    assert diagnostics["e_swap_count"] <= config.certe_swap_steps
    assert all(math.isfinite(float(value)) for value in diagnostics.values())

    second_config = _config()
    second_output, second_indices = certvid_e_compression(
        video,
        attention,
        second_config,
        question,
    )
    assert torch.equal(second_indices, indices)
    assert torch.equal(second_output, output)

    deepstack = [
        torch.randn(frames * tokens_per_frame, 40),
        torch.randn(frames * tokens_per_frame, 64),
    ]
    compressed = compress_certvid_deepstack(deepstack, config._certvid_plan)
    assert [tuple(layer.shape) for layer in compressed] == [(expected, 40), (expected, 64)]
    assert all(torch.isfinite(layer).all() for layer in compressed)

    routed_config = _config()
    routed, routed_indices = flashvid_compression(video, attention, routed_config, question)
    assert torch.equal(routed_indices, indices)
    assert torch.equal(routed, output)


def test_public_rates_and_identity() -> None:
    torch.manual_seed(71)
    video = torch.randn(8, 16, 32)
    attention = torch.rand(8, 16)
    question = torch.randn(6, 32)
    for rate in (0.10, 0.15, 0.20, 0.25):
        config = _config(retention_ratio=rate)
        output, indices = certvid_e_compression(video, attention, config, question)
        expected = round(video.shape[0] * video.shape[1] * rate * 1.25)
        assert output.shape == (expected, video.shape[-1])
        assert indices.numel() == expected
        assert torch.isfinite(output).all()

    identity_config = _config(retention_ratio=1.0, expansion=1.0)
    identity_output, identity_indices = certvid_e_compression(
        video,
        attention,
        identity_config,
        question,
    )
    assert torch.equal(identity_output, video.reshape(-1, video.shape[-1]))
    assert torch.equal(identity_indices, torch.arange(video.shape[0] * video.shape[1]))


def main() -> None:
    test_e_refinement_improves_weakest_direction()
    test_disabled_refinement_is_exact_v3()
    test_integration_and_deepstack()
    test_public_rates_and_identity()
    print("CertVID-E V3-compatible E-optimal smoke passed")


if __name__ == "__main__":
    main()
