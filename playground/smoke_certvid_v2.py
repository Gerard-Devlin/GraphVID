from __future__ import annotations

import torch

from flashvid.certvid import apply_certvid_plan, certvid_compression
from flashvid.certvid_qwen3 import compress_certvid_deepstack
from flashvid.certvid_v2 import certvid_v2_compression
from flashvid.configuration_flashvid import FlashVidConfig
from flashvid.utils import flashvid_compression


def _run_rate(rate: float) -> None:
    torch.manual_seed(7)
    frames, tokens_per_frame, feature_dim = 8, 36, 96
    video = torch.randn(frames, tokens_per_frame, feature_dim)
    attention = torch.rand(frames, tokens_per_frame)
    question = torch.randn(14, feature_dim)
    config = FlashVidConfig(
        retention_ratio=rate,
        expansion=1.25,
        compression_variant="certvid_v2",
        certv2_budget_uses_expansion=True,
        H=6,
        W=6,
    )
    output, indices = certvid_v2_compression(video, attention, config, question)
    plan = config._certvid_plan
    output_again, indices_again = certvid_v2_compression(video, attention, config, question)
    expected = round(frames * tokens_per_frame * rate * 1.25)

    assert output.shape == (expected, feature_dim)
    assert indices.shape == (expected,)
    assert int(torch.unique(indices).numel()) == expected
    assert bool(torch.all(indices[1:] > indices[:-1]))
    assert int(indices.min()) >= 0
    assert int(indices.max()) < frames * tokens_per_frame
    assert torch.isfinite(output).all()
    assert torch.equal(indices, indices_again)
    assert torch.equal(output, output_again)
    assert torch.equal(output, apply_certvid_plan(video.reshape(-1, feature_dim), plan))
    assert plan.anchor_indices.shape == indices.shape
    assert plan.assignment_indices.shape[0] == frames * tokens_per_frame
    assert plan.assignment_weights.shape == plan.assignment_indices.shape
    assert plan.fusion_alpha.shape == indices.shape
    assert torch.isfinite(plan.assignment_weights).all()
    assert torch.isfinite(plan.fusion_alpha).all()
    assert bool(torch.all(plan.fusion_alpha >= 0.0))
    assert bool(torch.all(plan.fusion_alpha <= 0.25 + 1e-6))
    deepstack = [torch.randn(frames * tokens_per_frame, 40), torch.randn(frames * tokens_per_frame, 64)]
    compressed_deepstack = compress_certvid_deepstack(deepstack, plan)
    assert [tuple(layer.shape) for layer in compressed_deepstack] == [
        (expected, 40),
        (expected, 64),
    ]
    assert all(torch.isfinite(layer).all() for layer in compressed_deepstack)
    assert config.last_adapter_variant == "certvid_v2"
    assert int(config.last_certv2_target_tokens) == expected
    assert int(config.last_certv2_repair_tokens) <= round(expected * 0.18)


def main() -> None:
    for rate in (0.10, 0.15, 0.20, 0.25):
        _run_rate(rate)

    torch.manual_seed(9)
    fallback_video = torch.randn(6, 25, 64)
    fallback_attention = torch.rand(6, 25)
    fallback_question = torch.randn(10, 64)
    v1_config = FlashVidConfig(
        retention_ratio=0.10,
        expansion=1.25,
        compression_variant="certvid",
        H=5,
        W=5,
    )
    v2_fallback_config = FlashVidConfig(
        retention_ratio=0.10,
        expansion=1.25,
        compression_variant="certvid_v2",
        certv2_repair_ratio=0.0,
        certv2_repair_ratio_high=0.0,
        H=5,
        W=5,
    )
    v1_output, v1_indices = certvid_compression(
        fallback_video,
        fallback_attention,
        v1_config,
        fallback_question,
    )
    fallback_output, fallback_indices = certvid_v2_compression(
        fallback_video,
        fallback_attention,
        v2_fallback_config,
        fallback_question,
    )
    assert torch.equal(v1_indices, fallback_indices)
    assert torch.equal(v1_output, fallback_output)

    torch.manual_seed(11)
    video = torch.randn(4, 16, 48)
    attention = torch.rand(4, 16)
    config = FlashVidConfig(
        retention_ratio=0.10,
        expansion=1.25,
        compression_variant="certvid_v2",
        certv2_budget_uses_expansion=True,
        H=4,
        W=4,
    )
    routed, routed_indices = flashvid_compression(video, attention, config, None)
    assert torch.equal(routed, apply_certvid_plan(video.reshape(-1, 48), config._certvid_plan))
    assert torch.equal(routed_indices, config._certvid_plan.anchor_indices)
    assert routed.shape[0] == round(4 * 16 * 0.10 * 1.25)

    constant = torch.zeros(4, 16, 48)
    constant_attention = torch.zeros(4, 16)
    constant_output, constant_indices = certvid_v2_compression(
        constant,
        constant_attention,
        config,
        None,
    )
    assert torch.isfinite(constant_output).all()
    assert int(torch.unique(constant_indices).numel()) == constant_output.shape[0]

    torch.manual_seed(19)
    base_frame = torch.randn(1, 16, 48)
    static_video = base_frame.repeat(8, 1, 1) + 0.01 * torch.randn(8, 16, 48)
    dynamic_video = torch.randn(8, 16, 48)
    router_attention = torch.rand(8, 16)
    static_config = FlashVidConfig(
        retention_ratio=0.10,
        expansion=1.25,
        compression_variant="certvid_v2",
        H=4,
        W=4,
    )
    dynamic_config = FlashVidConfig(
        retention_ratio=0.10,
        expansion=1.25,
        compression_variant="certvid_v2",
        H=4,
        W=4,
    )
    certvid_v2_compression(static_video, router_attention, static_config, None)
    certvid_v2_compression(dynamic_video, router_attention, dynamic_config, None)
    assert static_config.last_certv2_trajectory_complexity < dynamic_config.last_certv2_trajectory_complexity
    assert static_config.last_certv2_repair_fraction <= dynamic_config.last_certv2_repair_fraction
    print("CertVID V2 smoke passed")


if __name__ == "__main__":
    main()
