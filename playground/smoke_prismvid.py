from __future__ import annotations

import math

import torch

from flashvid.configuration_flashvid import FlashVidConfig
from flashvid.prismvid import (
    compress_prism_deepstack,
    merge_prism_visual_deepstack,
    prismvid_compression,
)


def _config(**overrides) -> FlashVidConfig:
    values = dict(
        retention_ratio=0.20,
        expansion=1.25,
        compression_variant="prismvid",
        prism_metric_dim=32,
        prism_probe_tokens=64,
        prism_candidate_multiplier=2.0,
        prism_batch_size=4,
    )
    values.update(overrides)
    return FlashVidConfig(**values)


def main() -> None:
    torch.manual_seed(17)
    frame_count, tokens_per_frame, feature_dim = 8, 16, 64
    video = torch.randn(frame_count, tokens_per_frame, feature_dim)
    attention = torch.softmax(torch.randn(frame_count, tokens_per_frame), dim=-1)
    question = torch.randn(12, feature_dim)
    deepstack = [torch.randn(frame_count * tokens_per_frame, feature_dim) for _ in range(3)]

    config = _config()
    output, indices = prismvid_compression(video, attention, config, question, deepstack)
    expected_budget = frame_count * math.ceil(tokens_per_frame * 0.20 * 1.25)
    assert output.shape == (expected_budget, feature_dim)
    assert indices.shape == (expected_budget,)
    assert torch.equal(output, video.reshape(-1, feature_dim)[indices])
    assert torch.equal(indices, torch.unique(indices, sorted=True))
    assert bool(torch.all(indices[1:] > indices[:-1]))
    assert int(indices.min()) >= 0 and int(indices.max()) < frame_count * tokens_per_frame
    assert bool(torch.isfinite(output).all())
    assert config.last_prism_budget == expected_budget
    assert config.last_prism_per_frame_budget == math.ceil(tokens_per_frame * 0.20 * 1.25)
    assert config.last_prism_levels == 4

    compressed_deepstack = compress_prism_deepstack(deepstack, indices)
    for source, compressed in zip(deepstack, compressed_deepstack):
        assert torch.equal(compressed, source[indices])

    image_mask = torch.tensor([[False, True, False, False, True, False, False, False]])
    video_mask = torch.tensor([[False, False, True, True, False, False, True, True]])
    kept_video = torch.tensor([1, 3])
    image_levels = [torch.tensor([[10.0], [40.0]])]
    video_levels = [torch.tensor([[20.0], [30.0], [60.0], [70.0]])]
    selected_video = compress_prism_deepstack(video_levels, kept_video)
    merged = merge_prism_visual_deepstack(
        deepstack_image_embeds=image_levels,
        compressed_video_embeds=selected_video,
        image_mask=image_mask,
        video_mask=video_mask,
        kept_video_indices=kept_video,
    )
    assert torch.equal(merged[0].flatten(), torch.tensor([10.0, 30.0, 40.0, 70.0]))

    repeat_output, repeat_indices = prismvid_compression(
        video,
        attention,
        _config(),
        question,
        deepstack,
    )
    assert torch.equal(indices, repeat_indices)
    assert torch.equal(output, repeat_output)

    base_only_output, base_only_indices = prismvid_compression(
        video,
        attention,
        _config(),
        question_features=None,
        deepstack_features=None,
    )
    assert base_only_output.shape[0] == expected_budget
    assert base_only_indices.shape[0] == expected_budget

    full_config = _config(retention_ratio=1.0, expansion=1.0)
    full_output, full_indices = prismvid_compression(video, attention, full_config, question, deepstack)
    assert torch.equal(full_output, video.reshape(-1, feature_dim))
    assert torch.equal(full_indices, torch.arange(frame_count * tokens_per_frame))

    static_config = _config(prism_frame_floor_ratio=0.50)
    static_output, static_indices = prismvid_compression(
        torch.zeros_like(video),
        torch.full_like(attention, 1.0 / tokens_per_frame),
        static_config,
    )
    static_frames = torch.div(static_indices, tokens_per_frame, rounding_mode="floor")
    assert torch.unique(static_frames).numel() == frame_count
    assert bool(torch.isfinite(static_output).all())

    try:
        prismvid_compression(video, attention[:, :-1], _config())
    except ValueError:
        pass
    else:
        raise AssertionError("shape mismatch must raise ValueError")

    print(
        "PrismVID smoke passed: "
        f"raw={frame_count * tokens_per_frame} output={expected_budget} "
        f"levels={config.last_prism_levels} query_confidence={config.last_prism_query_confidence:.3f}"
    )


if __name__ == "__main__":
    main()
