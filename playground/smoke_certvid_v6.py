from __future__ import annotations

import math

import torch

from flashvid.certvid import apply_certvid_plan
from flashvid.certvid_qwen3 import compress_certvid_deepstack
from flashvid.certvid_v6 import _continuity_gate, _scene_ids, certvid_v6_compression
from flashvid.configuration_flashvid import FlashVidConfig
from flashvid.utils import flashvid_compression


def _config(rate: float, **overrides) -> FlashVidConfig:
    values = {
        "retention_ratio": rate,
        "expansion": 1.25,
        "compression_variant": "certvid_v6",
        "certv3_budget_uses_expansion": True,
        "certv6_scene_temporal": True,
        "certv6_gate_enabled": True,
        "H": 6,
        "W": 6,
    }
    values.update(overrides)
    return FlashVidConfig(**values)


def _run_rate(rate: float) -> None:
    torch.manual_seed(61)
    frames, tokens_per_frame, feature_dim = 8, 36, 96
    video = torch.randn(frames, tokens_per_frame, feature_dim)
    attention = torch.rand(frames, tokens_per_frame)
    question = torch.randn(16, feature_dim)
    config = _config(rate)

    output, indices = certvid_v6_compression(video, attention, config, question)
    plan = config._certvid_plan
    output_again, indices_again = certvid_v6_compression(video, attention, config, question)
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

    assert 0.0 <= float(config.last_certv6_gate) <= 1.0
    assert 1 <= int(config.last_certv6_scene_count) <= frames
    assert math.isfinite(float(config.last_certv6_continuity))
    assert math.isfinite(float(config.last_certv6_logdet))

    deepstack = [
        torch.randn(frames * tokens_per_frame, 40),
        torch.randn(frames * tokens_per_frame, 64),
    ]
    compressed = compress_certvid_deepstack(deepstack, plan)
    assert [tuple(layer.shape) for layer in compressed] == [(expected, 40), (expected, 64)]
    assert all(torch.isfinite(layer).all() for layer in compressed)
    assert config.last_adapter_variant == "certvid_v6"


def _check_scene_and_gate_controls() -> None:
    torch.manual_seed(67)
    base = torch.randn(1, 16, 48)
    smooth = base.repeat(8, 1, 1) + 1e-3 * torch.randn(8, 16, 48)
    discontinuous = torch.randn(8, 16, 48)
    config = _config(0.10, H=4, W=4, min_segment_num=4)

    smooth_ids, smooth_count, smooth_continuity = _scene_ids(smooth, config)
    rough_ids, rough_count, rough_continuity = _scene_ids(discontinuous, config)
    assert smooth_ids.shape == rough_ids.shape == (8,)
    assert int(torch.unique(smooth_ids).numel()) == smooth_count
    assert int(torch.unique(rough_ids).numel()) == rough_count
    assert smooth_continuity > rough_continuity
    assert _continuity_gate(smooth_continuity, config) >= _continuity_gate(rough_continuity, config)

    config.certv6_gate_enabled = False
    assert _continuity_gate(-1.0, config) == 1.0

    config.certv6_scene_temporal = False
    uniform_ids, uniform_count, _ = _scene_ids(discontinuous, config)
    assert uniform_count == min(8, config.certv3_temporal_bins)
    assert int(uniform_ids.min()) == 0
    assert int(uniform_ids.max()) == uniform_count - 1


def main() -> None:
    _check_scene_and_gate_controls()
    for rate in (0.10, 0.15, 0.20, 0.25):
        _run_rate(rate)

    torch.manual_seed(71)
    video = torch.randn(4, 16, 48)
    attention = torch.rand(4, 16)
    routed_config = _config(0.10, H=4, W=4, min_segment_num=4)
    routed, routed_indices = flashvid_compression(video, attention, routed_config, None)
    assert torch.equal(routed, apply_certvid_plan(video.reshape(-1, 48), routed_config._certvid_plan))
    assert torch.equal(routed_indices, routed_config._certvid_plan.anchor_indices)

    identity_config = _config(
        1.0,
        expansion=1.0,
        certv3_budget_uses_expansion=False,
        H=4,
        W=4,
    )
    identity_output, identity_indices = certvid_v6_compression(
        video,
        attention,
        identity_config,
        None,
    )
    assert torch.equal(identity_output, video.reshape(-1, 48))
    assert torch.equal(identity_indices, torch.arange(video.numel() // 48))
    print("CertVID V6 smoke passed")


if __name__ == "__main__":
    main()
