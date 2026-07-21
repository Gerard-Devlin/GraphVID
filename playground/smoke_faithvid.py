from __future__ import annotations

import math
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Compression tests do not require the model-specific Transformers hooks.
package = types.ModuleType("flashvid")
package.__path__ = [str(REPO_ROOT / "flashvid")]
sys.modules.setdefault("flashvid", package)

import torch
import torch.nn.functional as F

from flashvid.certvid_qwen3 import compress_certvid_deepstack
from flashvid.configuration_flashvid import FlashVidConfig
from flashvid.faithvid import (
    FaithVidPlan,
    append_faithvid_neutral_tokens,
    apply_faithvid_position_centroids,
    faithvid_compression,
    pack_faithvid_frame_newlines,
)
from flashvid.faithvid_attention import (
    faithvid_attention_forward,
    update_faithvid_after_inner_prune,
)
from flashvid.utils import flashvid_compression


def _config(**overrides: object) -> FlashVidConfig:
    values: dict[str, object] = {
        "retention_ratio": 0.10,
        "expansion": 1.25,
        "compression_variant": "faithvid",
        "certv3_budget_uses_expansion": True,
        "faith_budget_uses_expansion": True,
        "faith_variance_strength": 0.0,
        "H": 4,
        "W": 4,
    }
    values.update(overrides)
    return FlashVidConfig(**values)


def test_fixed_budget_mass_and_determinism() -> None:
    torch.manual_seed(101)
    frames, tokens_per_frame, feature_dim = 8, 16, 48
    video = torch.randn(frames, tokens_per_frame, feature_dim)
    attention = torch.rand(frames, tokens_per_frame)
    question = torch.randn(8, feature_dim)

    config = _config()
    output, indices = faithvid_compression(video, attention, config, question)
    plan = config._certvid_plan
    expected = round(frames * tokens_per_frame * 0.10 * 1.25)

    assert isinstance(plan, FaithVidPlan)
    assert output.shape == (expected, feature_dim)
    assert indices.shape == (expected,)
    assert torch.equal(indices, torch.sort(indices).values)
    assert int(torch.unique(indices).numel()) == expected
    assert torch.isfinite(output).all()
    assert torch.isfinite(plan.attention_log_mass).all()
    assert torch.allclose(plan.assignment_weights.sum(dim=1), torch.ones(frames * tokens_per_frame))
    assert abs(float(plan.group_mass.sum().item()) - frames * tokens_per_frame) < 1e-4
    assert torch.equal(plan.source_mass, torch.ones_like(plan.source_mass))
    assert torch.equal(plan.assignment_indices[indices, 0], torch.arange(expected))
    assert torch.equal(plan.assignment_weights[indices, 0], torch.ones(expected))

    second_config = _config()
    second_output, second_indices = faithvid_compression(video, attention, second_config, question)
    assert torch.equal(second_indices, indices)
    assert torch.equal(second_output, output)

    deepstack = [
        torch.randn(frames * tokens_per_frame, 40),
        torch.randn(frames * tokens_per_frame, 64),
    ]
    compressed = compress_certvid_deepstack(deepstack, plan)
    assert [tuple(layer.shape) for layer in compressed] == [(expected, 40), (expected, 64)]
    assert all(torch.isfinite(layer).all() for layer in compressed)

    routed_config = _config()
    routed, routed_indices = flashvid_compression(video, attention, routed_config, question)
    assert torch.equal(routed_indices, indices)
    assert torch.equal(routed, output)


def test_mass_corrected_attention_exactness() -> None:
    # Three identical tokens are merged into A while B remains a competing key.
    query = torch.tensor([[[[0.7, -0.2]]]], dtype=torch.float32)
    key_a = torch.tensor([0.4, 0.8], dtype=torch.float32)
    key_b = torch.tensor([-0.5, 0.3], dtype=torch.float32)
    value_a = torch.tensor([1.5, -0.5], dtype=torch.float32)
    value_b = torch.tensor([-0.25, 2.0], dtype=torch.float32)
    full_keys = torch.stack([key_a, key_a, key_a, key_b]).view(1, 1, 4, 2)
    full_values = torch.stack([value_a, value_a, value_a, value_b]).view(1, 1, 4, 2)
    compressed_keys = torch.stack([key_a, key_b]).view(1, 1, 2, 2)
    compressed_values = torch.stack([value_a, value_b]).view(1, 1, 2, 2)
    scaling = 1.0 / math.sqrt(2.0)

    full_scores = (query @ full_keys.transpose(-2, -1)) * scaling
    full_output = torch.softmax(full_scores, dim=-1) @ full_values

    config = _config(expansion=1.0)
    config.visual_token_start_index = 0
    config.pruning_layer = 20
    config._faithvid_outer_group_mass = torch.tensor([3.0, 1.0])
    config._faithvid_outer_log_mass = torch.tensor([math.log(3.0), 0.0])
    config._faithvid_inner_group_mass = config._faithvid_outer_group_mass
    config._faithvid_inner_log_mass = config._faithvid_outer_log_mass

    module = types.SimpleNamespace(layer_idx=0, flashvid_config=config)
    corrected = faithvid_attention_forward(
        module,
        query,
        compressed_keys,
        compressed_values,
        torch.zeros(1, 1, 1, 2),
        cache_position=torch.tensor([1]),
        scaling=scaling,
        dropout=0.0,
        output_attentions=True,
    )
    assert corrected is not None
    corrected_output, corrected_weights = corrected
    assert torch.allclose(corrected_output.transpose(1, 2), full_output, rtol=1e-5, atol=1e-6)
    assert corrected_weights is not None and torch.isfinite(corrected_weights).all()

    uncorrected = F.scaled_dot_product_attention(
        query,
        compressed_keys,
        compressed_values,
        dropout_p=0.0,
        is_causal=False,
    )
    assert float((uncorrected - full_output).abs().max().item()) > 1e-3

    missing = _config(expansion=1.0)
    missing.visual_token_start_index = 0
    missing.visual_token_length = 2
    missing_module = types.SimpleNamespace(layer_idx=0, flashvid_config=missing)
    try:
        faithvid_attention_forward(
            missing_module,
            query,
            compressed_keys,
            compressed_values,
            None,
            cache_position=torch.tensor([1]),
            scaling=scaling,
            dropout=0.0,
            output_attentions=False,
        )
    except RuntimeError as error:
        assert "metadata is missing" in str(error)
    else:
        raise AssertionError("strict FaithVID attention accepted missing mass metadata")

    # HF attention expects BQHD and Qwen uses grouped-query attention.
    gqa_config = _config(expansion=1.0)
    gqa_config.visual_token_start_index = 0
    gqa_config.pruning_layer = 20
    gqa_config._faithvid_outer_group_mass = torch.ones(3)
    gqa_config._faithvid_outer_log_mass = torch.zeros(3)
    gqa_config._faithvid_inner_group_mass = torch.ones(3)
    gqa_config._faithvid_inner_log_mass = torch.zeros(3)
    gqa_module = types.SimpleNamespace(layer_idx=0, flashvid_config=gqa_config)
    gqa_output = faithvid_attention_forward(
        gqa_module,
        torch.randn(1, 4, 3, 8),
        torch.randn(1, 2, 3, 8),
        torch.randn(1, 2, 3, 8),
        None,
        cache_position=torch.arange(3),
        scaling=1.0 / math.sqrt(8.0),
        dropout=0.0,
        output_attentions=False,
    )
    assert gqa_output is not None and gqa_output[0].shape == (1, 3, 4, 8)

    other_config = _config(compression_variant="certvid_v3")
    other_module = types.SimpleNamespace(layer_idx=0, flashvid_config=other_config)
    assert faithvid_attention_forward(
        other_module,
        query,
        compressed_keys,
        compressed_values,
        None,
        cache_position=torch.tensor([1]),
        scaling=scaling,
        dropout=0.0,
        output_attentions=False,
    ) is None


def test_position_centroid_and_inner_mass() -> None:
    torch.manual_seed(131)
    frames, tokens_per_frame = 4, 16
    video = torch.randn(frames, tokens_per_frame, 32)
    attention = torch.rand(frames, tokens_per_frame)
    config = _config(retention_ratio=0.25, expansion=1.0)
    _, indices = faithvid_compression(video, attention, config, None)
    plan = config._certvid_plan

    prefix = 3
    sequence_length = prefix + frames * tokens_per_frame + 2
    position_ids = torch.stack(
        [
            torch.arange(sequence_length),
            torch.arange(sequence_length) * 2,
            torch.arange(sequence_length) * 3,
        ]
    ).unsqueeze(1)
    visual_positions = torch.arange(prefix, prefix + frames * tokens_per_frame)
    updated = apply_faithvid_position_centroids(config, position_ids, visual_positions)
    assert updated.shape == position_ids.shape
    assert torch.equal(updated[:, :, :prefix], position_ids[:, :, :prefix])
    assert config.last_faithvid_position_mode == "quantized_mrope_centroid"
    anchor_positions = visual_positions[indices]
    assert torch.isfinite(updated[:, :, anchor_positions].float()).all()

    outer_mass = plan.group_mass.clone()
    config.visual_token_start_index = prefix
    config.visual_token_length = int(indices.numel())
    retained_local = torch.arange(0, indices.numel(), 2)
    keep_indices = torch.cat(
        [
            torch.arange(prefix),
            retained_local + prefix,
            torch.tensor([prefix + indices.numel()]),
        ]
    )
    hidden = torch.randn(1, prefix + indices.numel() + 1, 12)
    merged_hidden = update_faithvid_after_inner_prune(
        config,
        keep_indices,
        visual_start=prefix,
        visual_length=int(indices.numel()),
        hidden_states=hidden,
        visual_global_indices=torch.arange(prefix, prefix + indices.numel()),
    )
    assert merged_hidden is not None
    assert torch.equal(merged_hidden[:, :prefix], hidden[:, :prefix])
    all_local = torch.arange(indices.numel())
    destination = (all_local.unsqueeze(1) - retained_local.unsqueeze(0)).abs().argmin(dim=1)
    expected = torch.zeros(1, retained_local.numel(), hidden.shape[-1])
    expected.index_add_(
        1,
        destination,
        hidden[:, prefix : prefix + indices.numel()] * outer_mass.view(1, -1, 1),
    )
    expected = expected / config._faithvid_inner_group_mass.view(1, -1, 1)
    assert torch.allclose(merged_hidden[:, prefix + retained_local], expected, atol=1e-6)
    assert config._faithvid_inner_group_mass.numel() == retained_local.numel()
    assert torch.allclose(config._faithvid_inner_group_mass.sum(), outer_mass.sum(), atol=1e-5)
    assert config.last_faithvid_inner_mass_error < 1e-5


def test_public_rates_and_identity() -> None:
    torch.manual_seed(151)
    video = torch.randn(4, 16, 32)
    attention = torch.rand(4, 16)
    for rate in (0.01, 0.02, 0.10, 0.15, 0.20, 0.25):
        config = _config(retention_ratio=rate)
        output, indices = faithvid_compression(video, attention, config, None)
        expected = round(video.shape[0] * video.shape[1] * rate * 1.25)
        assert output.shape == (expected, video.shape[-1])
        assert indices.numel() == expected

    identity = _config(retention_ratio=1.0, expansion=1.0)
    output, indices = faithvid_compression(video, attention, identity, None)
    assert torch.equal(output, video.reshape(-1, video.shape[-1]))
    assert torch.equal(indices, torch.arange(video.shape[0] * video.shape[1]))
    assert torch.equal(identity._certvid_plan.group_mass, torch.ones(video.shape[0] * video.shape[1]))


def test_llava_irregular_frame_packing() -> None:
    config = _config()
    config._faithvid_outer_group_mass = torch.tensor([2.0, 3.0, 1.0])
    config._faithvid_outer_variance = torch.tensor([0.1, 0.2, 0.3])
    config._faithvid_outer_log_mass = torch.log(config._faithvid_outer_group_mass)
    config._faithvid_inner_group_mass = config._faithvid_outer_group_mass
    config._faithvid_inner_variance = config._faithvid_outer_variance
    config._faithvid_inner_log_mass = config._faithvid_outer_log_mass
    tokens = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    keep = torch.tensor([0, 5, 11])
    newline = torch.full((4,), -1.0)

    packed = pack_faithvid_frame_newlines(config, tokens, keep, 3, 4, newline)
    assert packed.shape == (6, 4)
    assert torch.equal(packed[::2], tokens)
    assert torch.equal(packed[1::2], newline.expand(3, -1))
    assert torch.equal(
        config._faithvid_outer_group_mass,
        torch.tensor([2.0, 1.0, 3.0, 1.0, 1.0, 1.0]),
    )
    assert config.visual_token_length == 6

    append_faithvid_neutral_tokens(config, 1)
    assert torch.equal(
        config._faithvid_outer_group_mass,
        torch.tensor([2.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0]),
    )
    assert config._faithvid_outer_log_mass[-1].item() == 0.0


def main() -> None:
    test_fixed_budget_mass_and_determinism()
    test_mass_corrected_attention_exactness()
    test_position_centroid_and_inner_mass()
    test_public_rates_and_identity()
    test_llava_irregular_frame_packing()
    print("FaithVID functional-faithfulness smoke passed")


if __name__ == "__main__":
    main()
