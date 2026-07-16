from __future__ import annotations

import math

import torch

from flashvid.certvid import apply_certvid_plan
from flashvid.certvid_qwen3 import compress_certvid_deepstack
from flashvid.certvid_v3 import _d_optimal_greedy, certvid_v3_compression
from flashvid.configuration_flashvid import FlashVidConfig
from flashvid.utils import flashvid_compression


def _run_rate(rate: float) -> None:
    torch.manual_seed(31)
    frames, tokens_per_frame, feature_dim = 8, 36, 96
    video = torch.randn(frames, tokens_per_frame, feature_dim)
    attention = torch.rand(frames, tokens_per_frame)
    question = torch.randn(16, feature_dim)
    config = FlashVidConfig(
        retention_ratio=rate,
        expansion=1.25,
        compression_variant="certvid_v3",
        certv3_budget_uses_expansion=True,
        H=6,
        W=6,
    )
    output, indices = certvid_v3_compression(video, attention, config, question)
    plan = config._certvid_plan
    output_again, indices_again = certvid_v3_compression(video, attention, config, question)
    expected = round(frames * tokens_per_frame * rate * 1.25)

    assert output.shape == (expected, feature_dim)
    assert indices.shape == (expected,)
    assert int(torch.unique(indices).numel()) == expected
    assert bool(torch.all(indices[1:] > indices[:-1]))
    assert int(indices.min()) >= 0
    assert int(indices.max()) < frames * tokens_per_frame
    assert torch.isfinite(output).all()
    assert torch.equal(indices, indices_again)
    assert torch.allclose(output, output_again, rtol=0.0, atol=1e-6)
    assert torch.equal(output, apply_certvid_plan(video.reshape(-1, feature_dim), plan))

    selected_frames = torch.div(indices, tokens_per_frame, rounding_mode="floor")
    assert int(torch.unique(selected_frames).numel()) == frames
    certificate_count = int(config.last_certv3_certificate_count)
    assert int(torch.count_nonzero(plan.fusion_alpha == 0.0)) >= certificate_count
    assert math.isfinite(float(config.last_certv3_logdet))
    assert int(config.last_certv3_swap_count) <= config.certv3_swap_steps

    deepstack = [
        torch.randn(frames * tokens_per_frame, 40),
        torch.randn(frames * tokens_per_frame, 64),
    ]
    compressed = compress_certvid_deepstack(deepstack, plan)
    assert [tuple(layer.shape) for layer in compressed] == [(expected, 40), (expected, 64)]
    assert all(torch.isfinite(layer).all() for layer in compressed)
    assert config.last_adapter_variant == "certvid_v3"


def _check_complementary_directions() -> None:
    design = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.70, 0.70, 0.0],
        ],
        dtype=torch.float32,
    )
    design = torch.nn.functional.normalize(design, p=2, dim=-1)
    candidates = torch.arange(design.shape[0])
    selected = _d_optimal_greedy(
        design=design,
        candidates=candidates,
        mandatory=[],
        budget=3,
        ridge=0.25,
    )
    assert set(selected.tolist()) == {0, 2, 3}


def main() -> None:
    _check_complementary_directions()
    for rate in (0.10, 0.15, 0.20, 0.25):
        _run_rate(rate)

    torch.manual_seed(41)
    video = torch.randn(4, 16, 48)
    attention = torch.rand(4, 16)
    config = FlashVidConfig(
        retention_ratio=0.10,
        expansion=1.25,
        compression_variant="certvid_v3",
        H=4,
        W=4,
    )
    routed, routed_indices = flashvid_compression(video, attention, config, None)
    assert torch.equal(routed, apply_certvid_plan(video.reshape(-1, 48), config._certvid_plan))
    assert torch.equal(routed_indices, config._certvid_plan.anchor_indices)
    assert routed.shape[0] == round(4 * 16 * 0.10 * 1.25)

    constant = torch.zeros(4, 16, 48)
    constant_attention = torch.zeros(4, 16)
    constant_output, constant_indices = certvid_v3_compression(
        constant,
        constant_attention,
        config,
        None,
    )
    assert torch.isfinite(constant_output).all()
    assert int(torch.unique(constant_indices).numel()) == constant_output.shape[0]

    identity_config = FlashVidConfig(
        retention_ratio=1.0,
        expansion=1.0,
        compression_variant="certvid_v3",
        H=4,
        W=4,
    )
    identity_output, identity_indices = certvid_v3_compression(
        video,
        attention,
        identity_config,
        None,
    )
    assert torch.equal(identity_output, video.reshape(-1, 48))
    assert torch.equal(identity_indices, torch.arange(video.numel() // 48))
    print("CertVID V3 smoke passed")


if __name__ == "__main__":
    main()
