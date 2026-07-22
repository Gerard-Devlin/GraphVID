from __future__ import annotations

import torch

from flashvid.certvid import apply_certvid_plan
from flashvid.certvid_qwen3 import compress_certvid_deepstack
from flashvid.certvid_sb import certvid_sb_compression
from flashvid.certvid_v3 import certvid_v3_compression
from flashvid.configuration_flashvid import FlashVidConfig
from flashvid.utils import flashvid_compression


def _config(rate: float, **overrides) -> FlashVidConfig:
    values = {
        "retention_ratio": rate,
        "expansion": 1.25,
        "compression_variant": "certvid_sb",
        "certv3_budget_uses_expansion": True,
        "certsb_temporal_bins": 8,
        "H": 4,
        "W": 4,
    }
    values.update(overrides)
    return FlashVidConfig(**values)


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(73)
    frames, tokens_per_frame, feature_dim = 8, 16, 64
    base = torch.randn(tokens_per_frame, feature_dim)
    video = torch.stack(
        [base + 0.08 * frame * torch.randn_like(base) for frame in range(frames)]
    )
    # Add persistent local motion without turning every change into a scene cut.
    video[2:6, 5:9] += torch.linspace(0.0, 1.0, 4).view(4, 1, 1)
    attention = torch.rand(frames, tokens_per_frame)
    question = torch.randn(12, feature_dim)
    return video, attention, question


def _run_rate(
    rate: float,
    video: torch.Tensor,
    attention: torch.Tensor,
    question: torch.Tensor,
) -> None:
    config = _config(rate)
    output, indices = certvid_sb_compression(video, attention, config, question)
    plan = config._certvid_plan
    expected = round(video.shape[0] * video.shape[1] * rate * 1.25)

    assert output.shape == (expected, video.shape[-1])
    assert indices.shape == (expected,)
    assert int(torch.unique(indices).numel()) == expected
    assert bool(torch.all(indices[1:] > indices[:-1]))
    assert int(indices.min()) >= 0
    assert int(indices.max()) < video.shape[0] * video.shape[1]
    assert torch.isfinite(output).all()
    assert torch.equal(output, apply_certvid_plan(video.reshape(-1, video.shape[-1]), plan))

    diagnostics = config.last_certsb_diagnostics
    assert diagnostics["target_tokens"] == expected
    assert (
        diagnostics["semantic_tokens"]
        + diagnostics["temporal_tokens"]
        + diagnostics["coverage_tokens"]
        == expected
    )
    assert 0.0 <= diagnostics["v3_anchor_overlap"] <= 1.0
    assert 0.0 < diagnostics["temporal_bin_coverage"] <= 1.0
    assert int(torch.count_nonzero(plan.fusion_alpha == 0.0)) >= int(
        diagnostics["protected_structured_tokens"]
    )
    assert config.last_adapter_variant == "certvid_sb"

    second_config = _config(rate)
    second_output, second_indices = certvid_sb_compression(
        video, attention, second_config, question
    )
    assert torch.equal(indices, second_indices)
    assert torch.allclose(output, second_output, rtol=0.0, atol=1e-6)

    deepstack = [
        torch.randn(video.shape[0] * video.shape[1], 32),
        torch.randn(video.shape[0] * video.shape[1], 48),
    ]
    compressed = compress_certvid_deepstack(deepstack, plan)
    assert [tuple(layer.shape) for layer in compressed] == [(expected, 32), (expected, 48)]


def _check_v3_unchanged(
    video: torch.Tensor,
    attention: torch.Tensor,
    question: torch.Tensor,
) -> None:
    before = FlashVidConfig(
        retention_ratio=0.10,
        expansion=1.25,
        compression_variant="certvid_v3",
        certv3_budget_uses_expansion=True,
        H=4,
        W=4,
    )
    before_output, before_indices = certvid_v3_compression(
        video, attention, before, question
    )
    certvid_sb_compression(video, attention, _config(0.10), question)
    after = FlashVidConfig(
        retention_ratio=0.10,
        expansion=1.25,
        compression_variant="certvid_v3",
        certv3_budget_uses_expansion=True,
        H=4,
        W=4,
    )
    after_output, after_indices = certvid_v3_compression(video, attention, after, question)
    assert torch.equal(before_indices, after_indices)
    assert torch.equal(before_output, after_output)


def main() -> None:
    video, attention, question = _inputs()
    for rate in (0.10, 0.15, 0.20, 0.25):
        _run_rate(rate, video, attention, question)

    explicit = _config(
        0.20,
        certsb_semantic_ratio=0.50,
        certsb_temporal_ratio=0.30,
        certsb_coverage_ratio=0.20,
    )
    certvid_sb_compression(video, attention, explicit, question)
    explicit_diagnostics = explicit.last_certsb_diagnostics
    assert explicit_diagnostics["semantic_share"] == 0.50
    assert explicit_diagnostics["temporal_share"] == 0.30
    assert explicit_diagnostics["coverage_share"] == 0.20

    routed_config = _config(0.10)
    routed, routed_indices = flashvid_compression(
        video, attention, routed_config, question
    )
    assert torch.equal(routed_indices, routed_config._certvid_plan.anchor_indices)
    assert torch.equal(
        routed,
        apply_certvid_plan(video.reshape(-1, video.shape[-1]), routed_config._certvid_plan),
    )

    identity = _config(1.0, expansion=1.0)
    identity_output, identity_indices = certvid_sb_compression(
        video, attention, identity, question
    )
    assert torch.equal(identity_output, video.reshape(-1, video.shape[-1]))
    assert torch.equal(
        identity_indices,
        torch.arange(video.shape[0] * video.shape[1]),
    )
    _check_v3_unchanged(video, attention, question)
    print("CertVID-SB smoke passed")


if __name__ == "__main__":
    main()
