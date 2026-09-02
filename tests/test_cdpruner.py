from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Isolate the selector tests from flashvid/__init__.py, whose model imports are
# intentionally tied to the server's Transformers build.
flashvid_package = types.ModuleType("flashvid")
flashvid_package.__path__ = [str(ROOT / "flashvid")]
sys.modules.setdefault("flashvid", flashvid_package)
baseline_package = types.ModuleType("flashvid.baseline_adapters")
baseline_package.__path__ = [str(ROOT / "flashvid" / "baseline_adapters")]
sys.modules.setdefault("flashvid.baseline_adapters", baseline_package)

configuration = _load_module(
    "flashvid.configuration_flashvid",
    ROOT / "flashvid" / "configuration_flashvid.py",
)
_load_module(
    "flashvid.baseline_adapters.common",
    ROOT / "flashvid" / "baseline_adapters" / "common.py",
)
cdpruner = _load_module(
    "flashvid.baseline_adapters.cdpruner",
    ROOT / "flashvid" / "baseline_adapters" / "cdpruner.py",
)

FlashVidConfig = configuration.FlashVidConfig
cdpruner_compression = cdpruner.cdpruner_compression
conditional_dpp_kernel = cdpruner.conditional_dpp_kernel
encode_siglip_text = cdpruner.encode_siglip_text
fast_map_dpp = cdpruner.fast_map_dpp
merge_qwen3_visual_deepstack = cdpruner.merge_qwen3_visual_deepstack
select_qwen3_deepstack = cdpruner.select_qwen3_deepstack
strict_patch_budget = cdpruner.strict_patch_budget


def _released_fast_map_dpp(kernel: torch.Tensor, budget: int) -> torch.Tensor:
    batched = kernel.unsqueeze(0)
    batch_size, num_tokens, _ = batched.shape
    cis = torch.zeros((budget, batch_size, num_tokens), dtype=kernel.dtype)
    di2s = torch.diagonal(batched, dim1=1, dim2=2).clone()
    selected = torch.empty((budget, batch_size), dtype=torch.long)
    for step in range(budget):
        index = torch.argmax(di2s, dim=-1)
        selected[step] = index
        eis = (
            batched[torch.arange(batch_size), index]
            - torch.einsum(
                "tb,tbn->bn",
                cis[:step, torch.arange(batch_size), index],
                cis[:step],
            )
        ) / torch.sqrt(di2s[torch.arange(batch_size), index]).unsqueeze(-1)
        cis[step] = eis
        di2s -= torch.square(eis)
        di2s[torch.arange(batch_size), index] = -float("inf")
    return selected[:, 0]


def test_kernel_and_every_greedy_index_match_released_code() -> None:
    generator = torch.Generator().manual_seed(19)
    projected = torch.randn(11, 16, generator=generator)
    image = torch.randn(11, 12, generator=generator)
    text = torch.randn(3, 12, generator=generator)

    kernel, relevance = conditional_dpp_kernel(projected, image, text)
    projected_norm = F.normalize(projected.float(), dim=-1)
    image_norm = F.normalize(image.float(), dim=-1)
    text_norm = F.normalize(text.float(), dim=-1)
    expected_relevance = -(image_norm @ text_norm.T).mean(dim=-1)
    expected_relevance = (
        expected_relevance - expected_relevance.min() + 1e-6
    ) / (expected_relevance.max() - expected_relevance.min())
    expected_kernel = (
        expected_relevance[:, None]
        * (projected_norm @ projected_norm.T)
        * expected_relevance[None, :]
    )
    torch.testing.assert_close(relevance, expected_relevance, rtol=0, atol=0)
    torch.testing.assert_close(kernel, expected_kernel, rtol=0, atol=0)

    _, trace = fast_map_dpp(kernel, 5, return_trace=True)
    torch.testing.assert_close(trace, _released_fast_map_dpp(kernel, 5), rtol=0, atol=0)


@pytest.mark.parametrize(
    ("num_tokens", "ratio", "expected"),
    [
        (6272, 0.005, 31),
        (6272, 0.01, 62),
        (10816, 0.005, 54),
        (10816, 0.01, 108),
    ],
)
def test_strict_table_budgets(num_tokens: int, ratio: float, expected: int) -> None:
    assert strict_patch_budget(num_tokens, ratio) == expected


def test_zero_budget_is_an_error_instead_of_being_rounded_up() -> None:
    with pytest.raises(ValueError, match="budget is zero"):
        strict_patch_budget(10, 0.01)


def test_constant_relevance_and_degenerate_kernel_stay_finite_and_unique() -> None:
    projected = torch.ones(8, 4)
    image = torch.ones(8, 3)
    text = torch.ones(1, 3)
    kernel, relevance = conditional_dpp_kernel(projected, image, text)
    assert torch.isfinite(kernel).all()
    assert torch.equal(relevance, torch.ones_like(relevance))
    selected = fast_map_dpp(kernel, 4)
    assert selected.shape == (4,)
    assert torch.unique(selected).numel() == 4


def test_empty_text_features_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least one text"):
        conditional_dpp_kernel(
            torch.randn(4, 3),
            torch.randn(4, 2),
            torch.empty(0, 2),
        )


def test_siglip_text_padding_uses_the_checkpoint_pad_token() -> None:
    class FakeTokenizer:
        pad_token_id = 7

        def __call__(self, question, **kwargs):
            assert question == "raw question"
            return {
                "input_ids": torch.tensor([[3, 4]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }

    class FakeTextTower:
        config = types.SimpleNamespace(max_position_embeddings=4)

        def __call__(self, **model_inputs):
            self.model_inputs = model_inputs
            return types.SimpleNamespace(pooler_output=torch.ones(1, 5))

    text_tower = FakeTextTower()
    vision_tower = types.SimpleNamespace(
        _cdpruner_text_tokenizer=FakeTokenizer(),
        _cdpruner_text_tower=text_tower,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    result = encode_siglip_text(vision_tower, "raw question")
    assert torch.equal(text_tower.model_inputs["input_ids"], torch.tensor([[3, 4, 7, 7]]))
    assert torch.equal(text_tower.model_inputs["attention_mask"], torch.tensor([[1, 1, 0, 0]]))
    assert result.shape == (1, 5)


def test_compression_returns_exact_dynamic_budget() -> None:
    generator = torch.Generator().manual_seed(7)
    features = torch.randn(4, 25, 12, generator=generator)
    text = torch.randn(2, 12, generator=generator)
    config = FlashVidConfig(retention_ratio=0.13, compression_variant="cdpruner")
    tokens, indices = cdpruner_compression(
        features,
        config,
        relevance_visual_features=features,
        relevance_text_features=text,
    )
    assert tokens.shape == (13, 12)
    assert indices.shape == (13,)
    assert torch.equal(indices, torch.sort(indices).values)
    assert config.last_cdpruner_target_tokens == 13
    assert config.last_adapter_raw_tokens == 100.0


def test_qwen3_deepstack_reuses_main_indices_and_prompt_order() -> None:
    keep = torch.tensor([1, 3, 5])
    video_layers = [torch.arange(12).view(6, 2), torch.arange(18).view(6, 3)]
    selected = select_qwen3_deepstack(video_layers, keep)
    for source, result in zip(video_layers, selected):
        assert torch.equal(result, source.index_select(0, keep))

    image_layers = [torch.tensor([[100, 101]]), torch.tensor([[100, 101, 102]])]
    image_mask = torch.tensor([[False, True, False, False, False, False, False, False]])
    video_mask = torch.tensor([[True, False, True, True, True, True, True, False]])
    merged = merge_qwen3_visual_deepstack(
        deepstack_image_embeds=image_layers,
        selected_video_embeds=selected,
        image_mask=image_mask,
        video_mask=video_mask,
        keep_video_indices=keep,
    )
    assert merged[0].shape[0] == 4
    assert torch.equal(merged[0][0], image_layers[0][0])
