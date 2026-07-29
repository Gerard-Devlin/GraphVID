from __future__ import annotations

from flashvid.configuration_flashvid import FlashVidConfig


def _cfg_float(config: FlashVidConfig, name: str, default: float) -> float:
    value = getattr(config, name, None)
    return float(default if value is None else value)


def _cfg_int(config: FlashVidConfig, name: str, default: int) -> int:
    value = getattr(config, name, None)
    return int(default if value is None else value)


def _effective_ratio(config: FlashVidConfig) -> float:
    ratio = float(getattr(config, "retention_ratio", 0.10))
    uses_expansion = bool(
        getattr(
            config,
            "adapter_budget_uses_expansion",
            getattr(config, "external_budget_uses_expansion", True),
        )
    )
    if uses_expansion:
        ratio *= float(getattr(config, "expansion", 1.0))
    return max(0.0, min(1.0, ratio))


def _record_adapter_metrics(config: FlashVidConfig, *, variant: str, output_tokens: int, raw_tokens: int) -> None:
    setattr(config, "last_adapter_variant", variant)
    setattr(config, "last_adapter_output_tokens", float(output_tokens))
    setattr(config, "last_adapter_raw_tokens", float(raw_tokens))
