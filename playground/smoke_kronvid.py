from __future__ import annotations

import torch

from flashvid.configuration_flashvid import FlashVidConfig
from flashvid.kronvid import (
    _allocate_budget,
    _effective_dimension,
    _harmonic_prolongation,
    _resolve_budget,
    compress_kronvid_deepstack,
    kronvid_compression,
)


def _outer_config(**overrides) -> FlashVidConfig:
    values = {
        "compression_variant": "kronvid",
        "retention_ratio": 0.25,
        "expansion": 1.0,
        "llm_retention_ratio": 1.0,
        "kron_budget_mode": "outer_only",
        "kron_temporal_segments": 3,
        "kron_metric_dim": 16,
    }
    values.update(overrides)
    return FlashVidConfig(**values)


def _hybrid_config(*, layers: int, pruning_layer: int, inner_ratio: float) -> FlashVidConfig:
    return FlashVidConfig(
        compression_variant="kronvid",
        retention_ratio=0.10,
        expansion=1.25,
        pruning_layer=pruning_layer,
        llm_retention_ratio=inner_ratio,
        kron_budget_mode="layer_average",
        kron_num_hidden_layers=layers,
        kron_inner_hook_enabled=True,
    )


def _test_budget_contracts() -> None:
    llava = _hybrid_config(layers=28, pruning_layer=20, inner_ratio=0.30)
    qwen = _hybrid_config(layers=36, pruning_layer=28, inner_ratio=0.10)
    assert _resolve_budget(llava, 2880)[0] == 360
    assert _resolve_budget(qwen, 2880)[0] == 360

    invalid = _hybrid_config(layers=28, pruning_layer=20, inner_ratio=0.20)
    try:
        _resolve_budget(invalid, 2880)
    except ValueError as error:
        assert "not aligned" in str(error)
    else:
        raise AssertionError("invalid layer-average budget was accepted")

    invalid_outer = _outer_config(expansion=1.25)
    try:
        _resolve_budget(invalid_outer, 48)
    except ValueError as error:
        assert "outer_only requires" in str(error)
    else:
        raise AssertionError("outer-only expansion mismatch was accepted")


def _test_budget_allocation_and_dimension() -> None:
    allocation = _allocate_budget(11, [8, 8, 8], [1.0, 3.0, 2.0], 0.35)
    assert sum(allocation) == 11
    assert all(0 <= value <= 8 for value in allocation)
    assert allocation[1] >= allocation[0]

    duplicate = torch.ones(12, 8)
    independent = torch.eye(8).repeat(2, 1)[:12]
    assert _effective_dimension(independent, 0.10) > _effective_dimension(duplicate, 0.10)


def _test_harmonic_coordinates() -> None:
    weights = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    laplacian = torch.diag(weights.sum(dim=1)) - weights
    metric = torch.nn.functional.normalize(torch.randn(4, 6), dim=1)
    anchors = torch.tensor([0, 3])
    prolongation, fallback = _harmonic_prolongation(laplacian, metric, anchors, 0.01)
    assert not fallback
    torch.testing.assert_close(prolongation.sum(dim=1), torch.ones(4), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(prolongation[anchors], torch.eye(2), atol=0.0, rtol=0.0)
    assert bool((prolongation >= 0).all())


def _test_end_to_end() -> None:
    torch.manual_seed(17)
    features = torch.randn(4, 12, 24)
    attention_a = torch.randn(4, 12)
    attention_b = torch.randn(4, 12)
    question_a = torch.randn(5, 24)
    question_b = torch.randn(7, 24)

    config_a = _outer_config()
    output_a, indices_a = kronvid_compression(
        features,
        attention_a,
        config_a,
        question_a,
    )
    config_b = _outer_config()
    output_b, indices_b = kronvid_compression(
        features,
        attention_b,
        config_b,
        question_b,
    )

    assert output_a.shape == (12, 24)
    assert indices_a.shape == (12,)
    assert int(torch.unique(indices_a).numel()) == 12
    assert bool((indices_a[1:] > indices_a[:-1]).all())
    assert int(indices_a.min()) >= 0 and int(indices_a.max()) < 48
    assert bool(torch.isfinite(output_a).all())
    torch.testing.assert_close(indices_a, indices_b, atol=0, rtol=0)
    torch.testing.assert_close(output_a, output_b, atol=0, rtol=0)
    for segment in config_a._kronvid_plan.segments:
        reduced = segment.reduced_laplacian
        assert bool(torch.isfinite(reduced).all())
        torch.testing.assert_close(reduced, reduced.transpose(0, 1), atol=1e-5, rtol=1e-5)
        assert float(torch.linalg.eigvalsh(reduced).min().item()) >= -1e-4

    deepstack = [torch.randn(48, 10), torch.randn(48, 10)]
    compressed_deepstack = compress_kronvid_deepstack(deepstack, config_a._kronvid_plan)
    assert [tuple(value.shape) for value in compressed_deepstack] == [(12, 10), (12, 10)]
    assert all(bool(torch.isfinite(value).all()) for value in compressed_deepstack)

    prune_config = _outer_config(kron_merge_mode="prune")
    prune_output, prune_indices = kronvid_compression(
        features,
        attention_a,
        prune_config,
        question_a,
    )
    torch.testing.assert_close(prune_output, features.reshape(48, 24)[prune_indices])

    assert config_a.last_kron_target_tokens == 12.0
    assert config_a.last_kron_segment_count == 3.0
    assert config_a.last_kron_graph_edges > 0.0


def main() -> None:
    _test_budget_contracts()
    _test_budget_allocation_and_dimension()
    _test_harmonic_coordinates()
    _test_end_to_end()
    print("KronVID smoke tests passed")


if __name__ == "__main__":
    main()
