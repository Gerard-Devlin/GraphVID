from __future__ import annotations

import torch

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
    assert torch.equal(output, video.reshape(-1, feature_dim)[indices])
    assert int(torch.unique(torch.div(indices, tokens_per_frame, rounding_mode="floor")).numel()) == frames
    assert config.last_adapter_variant == "certvid_v2"
    assert int(config.last_certv2_target_tokens) == expected


def main() -> None:
    for rate in (0.10, 0.15, 0.20, 0.25):
        _run_rate(rate)

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
    assert torch.equal(routed, video.reshape(-1, 48)[routed_indices])
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
    print("CertVID V2 smoke passed")


if __name__ == "__main__":
    main()
