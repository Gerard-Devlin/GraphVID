from __future__ import annotations

import torch

from flashvid.certvid import apply_certvid_plan, certvid_compression
from flashvid.certvid_qwen3 import compress_certvid_deepstack, merge_certvid_visual_deepstack
from flashvid.configuration_flashvid import FlashVidConfig


def main() -> None:
    torch.manual_seed(7)
    frame_count, height, width, hidden_size = 8, 4, 4, 64
    tokens_per_frame = height * width
    raw_tokens = frame_count * tokens_per_frame
    config = FlashVidConfig(
        retention_ratio=0.10,
        expansion=1.25,
        compression_variant="certvid",
        llm_retention_ratio=1.0,
    )
    config.H = height
    config.W = width
    video = torch.randn(frame_count, tokens_per_frame, hidden_size)
    attention = torch.rand(frame_count, tokens_per_frame)
    question = torch.randn(12, hidden_size)

    output, indices = certvid_compression(video, attention, config, question)
    expected_budget = round(raw_tokens * 0.10 * 1.25)
    plan = config._certvid_plan
    assert output.shape == (expected_budget, hidden_size)
    assert indices.shape == (expected_budget,)
    assert torch.equal(indices, torch.sort(indices).values)
    assert torch.unique(indices).numel() == indices.numel()
    assert int(indices.min()) >= 0 and int(indices.max()) < raw_tokens
    assert torch.isfinite(output).all()

    deepstack = [torch.randn(raw_tokens, 48), torch.randn(raw_tokens, 32)]
    compressed_deepstack = compress_certvid_deepstack(deepstack, plan)
    assert [tuple(x.shape) for x in compressed_deepstack] == [
        (expected_budget, 48),
        (expected_budget, 32),
    ]
    assert torch.allclose(compressed_deepstack[0], apply_certvid_plan(deepstack[0], plan))

    image_count = 3
    sequence_length = raw_tokens + image_count + 5
    image_mask = torch.zeros((1, sequence_length), dtype=torch.bool)
    video_mask = torch.zeros_like(image_mask)
    image_mask[0, torch.tensor([1, 10, sequence_length - 1])] = True
    video_positions = torch.tensor([idx for idx in range(sequence_length) if not image_mask[0, idx]][:raw_tokens])
    video_mask[0, video_positions] = True
    image_deepstack = [torch.randn(image_count, 48), torch.randn(image_count, 32)]
    merged = merge_certvid_visual_deepstack(
        deepstack_image_embeds=image_deepstack,
        compressed_video_embeds=compressed_deepstack,
        image_mask=image_mask,
        video_mask=video_mask,
        kept_video_indices=indices,
    )
    assert [tuple(x.shape) for x in merged] == [
        (image_count + expected_budget, 48),
        (image_count + expected_budget, 32),
    ]
    assert all(torch.isfinite(layer).all() for layer in merged)
    print(
        "CertVID smoke passed: "
        f"raw={raw_tokens} budget={expected_budget} deepstack_layers={len(merged)}"
    )


if __name__ == "__main__":
    main()
