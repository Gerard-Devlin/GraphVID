from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from flashvid.certvid import apply_certvid_plan
from flashvid.certvid_qwen3 import compress_certvid_deepstack
from flashvid.certvid_v4 import (
    _CertificateRequest,
    _admit_certificates,
    _candidate_pool,
    _factor_information,
    _resolve_budget,
    _swap_refine,
    _tie_safe_rank_normalize,
    _validated_attention,
    certvid_v4_compression,
)
from flashvid.configuration_flashvid import FlashVidConfig
from flashvid.utils import flashvid_compression


def _llava_config(**overrides) -> FlashVidConfig:
    values = dict(
        retention_ratio=0.10,
        expansion=1.25,
        pruning_layer=20,
        llm_retention_ratio=0.30,
        compression_variant="certvid_v4",
        certv4_num_hidden_layers=28,
        certv4_inner_hook_enabled=True,
        certv4_swap_steps=2,
    )
    values.update(overrides)
    config = FlashVidConfig(**values)
    config._certvid_attention_source = "manual_qk"
    return config


def _expect_error(fn, text: str) -> None:
    try:
        fn()
    except (ValueError, RuntimeError) as exc:
        assert text in str(exc), (text, str(exc))
    else:
        raise AssertionError(f"expected error containing {text!r}")


def test_attention() -> None:
    tied, used, reason = _tie_safe_rank_normalize(torch.tensor([0.0, 0.0, 1.0, 1.0]))
    assert used and reason == "validated"
    assert tied[0] == tied[1] and tied[2] == tied[3]
    assert tied[0] < tied[2]

    per_frame, used, reason = _tie_safe_rank_normalize(
        torch.tensor([[0.0, 0.0, 1.0, 1.0], [10.0, 10.0, 20.0, 20.0]])
    )
    assert used and reason == "validated"
    assert torch.equal(per_frame[0], per_frame[1])

    constant, used, reason = _tie_safe_rank_normalize(torch.ones(8))
    assert not used and reason == "degenerate"
    assert torch.equal(constant, torch.zeros_like(constant))

    config = _llava_config()
    attention, diagnostics = _validated_attention(torch.ones(2, 4), 2, 4, config)
    assert not diagnostics["used"] and diagnostics["reason"] == "degenerate"
    assert torch.count_nonzero(attention) == 0

    config._certvid_attention_source = "feature_norm"
    attention, diagnostics = _validated_attention(torch.arange(8).view(2, 4), 2, 4, config)
    assert not diagnostics["used"] and diagnostics["reason"] == "unvalidated_source"
    assert torch.count_nonzero(attention) == 0

    config._certvid_attention_source = "missing"
    attention, diagnostics = _validated_attention(torch.arange(8).view(2, 4), 2, 4, config)
    assert not diagnostics["used"] and diagnostics["source"] == "missing"
    assert torch.count_nonzero(attention) == 0

    _expect_error(
        lambda: _validated_attention(torch.ones(8), 2, 4, config),
        "must have shape",
    )
    _expect_error(
        lambda: _validated_attention(torch.tensor([[0.0, float("nan")], [1.0, 2.0]]), 2, 2, config),
        "NaN or Inf",
    )
    config.certv4_attention_policy = "strict"
    _expect_error(
        lambda: _validated_attention(torch.ones(2, 4), 2, 4, config),
        "requires attention provenance",
    )


def test_budget() -> None:
    budget, diagnostics = _resolve_budget(_llava_config(), 2880)
    assert budget == 360
    assert abs(float(diagnostics["average_layer_multiplier"]) - 1.0) < 1e-6
    assert diagnostics["post_inner_tokens"] == 108
    assert math.isclose(float(diagnostics["average_layer_tokens"]), 288.0)

    qwen = _llava_config(
        expansion=1.25,
        pruning_layer=28,
        llm_retention_ratio=0.10,
        certv4_num_hidden_layers=36,
    )
    assert _resolve_budget(qwen, 2880)[0] == 360

    tuned = _llava_config(expansion=1.275, llm_retention_ratio=0.245098)
    assert _resolve_budget(tuned, 2880)[0] == 367

    invalid = _llava_config(expansion=1.30, llm_retention_ratio=0.30)
    _expect_error(lambda: _resolve_budget(invalid, 2880), "not aligned")
    missing_hook = _llava_config(certv4_inner_hook_enabled=False)
    _expect_error(lambda: _resolve_budget(missing_hook, 2880), "inner-pruning hook")
    over_budget = _llava_config(retention_ratio=0.90)
    _expect_error(lambda: _resolve_budget(over_budget, 2880), "must not exceed 1")

    outer = _llava_config(
        expansion=1.0,
        llm_retention_ratio=1.0,
        certv4_budget_mode="outer_only",
    )
    assert _resolve_budget(outer, 2880)[0] == 288
    outer.expansion = 1.25
    _expect_error(lambda: _resolve_budget(outer, 2880), "outer_only requires")


def test_certificates_and_candidates() -> None:
    requests = []
    for category in ("query", "frame", "temporal", "spatial"):
        for index in range(20):
            token = index if category == "frame" else index + 20 * (1 + ("query", "temporal", "spatial").index(category))
            requests.append(_CertificateRequest(category, f"{category}:{index}", token, 1.0 - index / 100.0))
    requests.append(_CertificateRequest("query", "query:duplicate", 0, 2.0))
    locked, diagnostics = _admit_certificates(requests, budget=20, budget_ratio=0.40)
    assert len(locked) <= 8
    assert float(diagnostics["ratio"]) <= 0.40
    assert all(diagnostics["categories"][name]["contributed_unique"] > 0 for name in ("query", "frame", "temporal", "spatial"))

    quality = torch.linspace(0.0, 1.0, 100)
    component_ids = torch.arange(100) // 5
    temporal_ids = torch.arange(100) // 20
    spatial_ids = torch.arange(100) % 9
    relevance = torch.stack([quality, quality.flip(0)])
    atom_weights = torch.tensor([0.5, 0.5])
    candidates, candidate_diagnostics = _candidate_pool(
        budget=20,
        quality=quality,
        component_ids=component_ids,
        temporal_ids=temporal_ids,
        spatial_ids=spatial_ids,
        query_relevance=relevance,
        atom_weights=atom_weights,
        query_mode="certificates_and_design",
        locked=locked,
        multiplier=2.5,
    )
    assert candidates.numel() == 50
    assert torch.equal(candidates, torch.sort(candidates).values)
    assert set(locked).issubset(set(candidates.tolist()))
    assert all(candidate_diagnostics["sources"][name]["admitted"] > 0 for name in ("query", "trajectory", "spatial", "global"))

    _, certificates_only = _candidate_pool(
        budget=20,
        quality=quality,
        component_ids=component_ids,
        temporal_ids=temporal_ids,
        spatial_ids=spatial_ids,
        query_relevance=relevance,
        atom_weights=atom_weights,
        query_mode="certificates_only",
        locked=locked,
        multiplier=2.5,
    )
    assert certificates_only["sources"]["query"]["offered"] == 0


def test_cholesky_and_swap() -> None:
    torch.manual_seed(11)
    rows = torch.randn(12, 7)
    ridge = 0.5
    _, logdet, _ = _factor_information(rows, ridge)
    information = ridge * torch.eye(7) + rows.transpose(0, 1) @ rows
    sign, reference = torch.linalg.slogdet(information)
    assert sign > 0 and math.isclose(logdet, float(reference), rel_tol=1e-5, abs_tol=1e-5)

    design = torch.randn(30, 7)
    candidates = torch.arange(30)
    selected = torch.arange(10)
    refined, swaps, zero_step_logdet, _ = _swap_refine(
        selected=selected,
        candidates=candidates,
        design=design,
        locked=[0, 1],
        ridge=ridge,
        steps=0,
        pool_size=8,
        margin=1e-4,
    )
    assert swaps == 0 and torch.equal(refined, selected)
    _, expected, _ = _factor_information(design[selected], ridge)
    assert math.isclose(zero_step_logdet, expected, rel_tol=1e-6, abs_tol=1e-6)

    refined, _, refined_logdet, _ = _swap_refine(
        selected=selected,
        candidates=candidates,
        design=design,
        locked=[0, 1],
        ridge=ridge,
        steps=4,
        pool_size=12,
        margin=1e-4,
    )
    assert {0, 1}.issubset(set(refined.tolist()))
    assert refined_logdet + 1e-6 >= expected


def test_integration() -> None:
    torch.manual_seed(17)
    frame_count, tokens_per_frame, feature_dim = 6, 16, 48
    video = torch.randn(frame_count, tokens_per_frame, feature_dim)
    attention = torch.randn(frame_count, tokens_per_frame)
    question = torch.randn(7, feature_dim)
    config = _llava_config()

    output, indices = certvid_v4_compression(video, attention, config, question)
    first_output = output.clone()
    first_indices = indices.clone()
    first_logdet = config.last_certv4_logdet
    output_again, indices_again = certvid_v4_compression(video, attention, config, question)
    assert torch.equal(first_indices, indices_again)
    assert torch.allclose(first_output, output_again)
    assert math.isclose(first_logdet, config.last_certv4_logdet, rel_tol=1e-6, abs_tol=1e-6)

    target = round(frame_count * tokens_per_frame * 0.10 * 1.25)
    assert output.shape == (target, feature_dim)
    assert indices.numel() == target
    assert torch.equal(indices, torch.sort(indices).values)
    assert torch.unique(indices).numel() == target
    assert torch.isfinite(output).all()
    diagnostics = config.last_certv4_diagnostics
    assert diagnostics["certificates"]["ratio"] <= 0.40
    assert diagnostics["attention"]["source"] == "manual_qk"

    plan = config._certvid_plan
    assert torch.allclose(output, apply_certvid_plan(video.reshape(-1, feature_dim), plan))
    deepstack = [torch.randn(frame_count * tokens_per_frame, feature_dim) for _ in range(3)]
    compressed_deepstack = compress_certvid_deepstack(deepstack, plan)
    assert all(tensor.shape[0] == target for tensor in compressed_deepstack)

    routed, routed_indices = flashvid_compression(video, attention, config, question)
    assert routed.shape == output.shape
    assert torch.equal(routed_indices, first_indices)
    assert config.last_adapter_variant == "certvid_v4"

    for query_mode in ("certificates_only", "design_only", "certificates_and_design", "off"):
        mode_config = _llava_config(certv4_query_mode=query_mode, certv4_swap_steps=0)
        mode_output, mode_indices = certvid_v4_compression(video, attention, mode_config, question)
        assert mode_output.shape == output.shape
        assert mode_indices.numel() == target
        assert torch.isfinite(mode_output).all()
        if query_mode == "off":
            assert mode_config.last_certv4_diagnostics["certificates"]["categories"]["query"]["requested"] == 0
            assert mode_config.last_certv4_diagnostics["candidates"]["sources"]["query"]["offered"] == 0


def main() -> None:
    test_attention()
    test_budget()
    test_certificates_and_candidates()
    test_cholesky_and_swap()
    test_integration()
    print("CertVID V4 smoke passed")


if __name__ == "__main__":
    main()
