"""CertVID-G: confidence-gated SigLIP localization over the V3 design."""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Optional

import torch
import torch.nn.functional as F

from .certvid import _cfg_float, _cfg_int
from .certvid_v3 import certvid_v3_compression


_TEXT_ONLY_PATTERNS = (
    r"\bsubtitle",
    r"\bcaption",
    r"\baccording to",
    r"\bwhat (?:did|does|do|is|was|were) .+ (?:say|said|tell|told|mention)",
    r"\b(?:word|words|text|letter|number|name) (?:is|are|was|were) (?:shown|written|displayed)",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _write_diagnostics(config: Any, diagnostics: dict[str, Any]) -> None:
    config.last_certg_diagnostics = diagnostics
    config.last_certg_active = bool(diagnostics.get("active", False))
    config.last_certg_confidence = float(diagnostics.get("confidence", 0.0))
    template = os.environ.get("CERTG_DIAGNOSTICS_JSONL", "").strip()
    if template:
        rank = os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))
        path = template.replace("{rank}", rank).replace("{pid}", str(os.getpid()))
        if "{rank}" not in template and "{pid}" not in template:
            root, extension = os.path.splitext(path)
            path = f"{root}.rank{rank}{extension or '.jsonl'}"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        record = dict(diagnostics)
        record["sample_id"] = str(getattr(config, "_debug_sample_id", "unknown"))
        record["task"] = getattr(config, "_certvid_task_name", None)
        record["question"] = str(getattr(config, "_certvid_query_text", "") or "")
        category = getattr(config, "_certvid_eval_category", None)
        if category is not None:
            record["eval_category"] = str(category)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")
    if bool(getattr(config, "certg_debug", False)):
        print(
            "[certvid-g] "
            f"sample={getattr(config, '_debug_sample_id', 'unknown')} "
            f"active={diagnostics.get('active', False)} "
            f"fallback={diagnostics.get('fallback_reason')} "
            f"confidence={diagnostics.get('confidence', 0.0):.4f} "
            f"peaks={diagnostics.get('peak_frames', [])}"
        )


def _duration_seconds(config: Any, frame_count: int) -> tuple[float, bool]:
    raw = getattr(config, "_certvid_frame_times_sec", None)
    if raw is None:
        return 0.0, False
    times = torch.as_tensor(raw, dtype=torch.float32).flatten()
    valid = (
        times.numel() == frame_count
        and bool(torch.isfinite(times).all())
        and (times.numel() <= 1 or bool(torch.all(times[1:] > times[:-1])))
    )
    if not valid:
        return 0.0, False
    return float((times[-1] - times[0]).item()), True


def _smooth(scores: torch.Tensor) -> torch.Tensor:
    if scores.numel() < 3:
        return scores.float()
    values = scores.float().view(1, 1, -1)
    values = F.pad(values, (1, 1), mode="replicate")
    kernel = scores.new_tensor([0.25, 0.50, 0.25], dtype=torch.float32).view(1, 1, 3)
    return F.conv1d(values, kernel).flatten()


def _locator_confidence(scores: torch.Tensor, radius: int) -> tuple[float, torch.Tensor]:
    smooth = _smooth(scores)
    median = smooth.median()
    mad = (smooth - median).abs().median() * 1.4826
    dynamic_range = smooth.max() - smooth.min()
    if float(mad.item()) < 1e-6 or float(dynamic_range.item()) < 1e-6:
        return 0.0, torch.zeros_like(smooth)
    z = (smooth - median) / mad.clamp_min(1e-6)
    probabilities = torch.softmax(z / 1.25, dim=0)
    peak = int(torch.argmax(z).item())
    left, right = max(0, peak - radius), min(z.numel(), peak + radius + 1)
    window_mass = probabilities[left:right].sum()
    uniform_mass = float(right - left) / float(z.numel())
    mass_confidence = (
        (window_mass - uniform_mass) / max(1e-6, 0.60 - uniform_mass)
    ).clamp(0.0, 1.0)
    peak_confidence = ((z.max() - 2.0) / 2.5).clamp(0.0, 1.0)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    entropy_confidence = (
        1.0 - entropy / math.log(max(2, int(z.numel())))
    ).clamp(0.0, 1.0)
    confidence = (
        0.45 * peak_confidence
        + 0.35 * mass_confidence
        + 0.20 * entropy_confidence
    )
    return float(confidence.item()), z


def _peak_frames(z: torch.Tensor, count: int, separation: int) -> list[int]:
    peaks: list[int] = []
    for index in torch.argsort(z, descending=True, stable=True).tolist():
        if float(z[index].item()) <= 0.0:
            break
        if all(abs(index - previous) >= separation for previous in peaks):
            peaks.append(int(index))
        if len(peaks) >= count:
            break
    return sorted(peaks)


def _frame_multiplier(
    z: torch.Tensor,
    peaks: list[int],
    confidence: float,
    radius: int,
    max_tilt: float,
) -> torch.Tensor:
    positions = torch.arange(z.numel(), device=z.device, dtype=torch.float32)
    kernel = torch.zeros_like(positions)
    scale = max(1.0, float(radius))
    z_positive = z.clamp_min(0.0)
    for peak in peaks:
        amplitude = float(torch.sigmoid(z_positive[peak] - 1.0).item())
        distance = (positions - float(peak)) / scale
        kernel = torch.maximum(kernel, amplitude * torch.exp(-0.5 * distance.square()))
    strength = max(0.0, max_tilt) * max(0.0, min(1.0, confidence))
    multiplier = 1.0 + strength * kernel
    return multiplier / multiplier.mean().clamp_min(1e-6)


def _frame_counts(indices: torch.Tensor, tokens_per_frame: int, frame_count: int) -> list[int]:
    frames = torch.div(indices, tokens_per_frame, rounding_mode="floor")
    counts = torch.bincount(frames, minlength=frame_count)
    return [int(value) for value in counts.detach().cpu().tolist()]


def certvid_g_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: Any,
    question_features: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tilt V3 only when the native SigLIP locator is confidently localized."""
    if video_features.ndim != 3:
        raise ValueError(f"expected video_features [T, HW, D], got {tuple(video_features.shape)}")
    frame_count, tokens_per_frame, _ = video_features.shape
    question = str(getattr(flashvid_config, "_certvid_query_text", "") or "").strip()
    duration, has_real_times = _duration_seconds(flashvid_config, frame_count)
    scores = getattr(flashvid_config, "_certg_frame_scores", None)
    score_source = str(getattr(flashvid_config, "_certg_score_source", "missing"))
    diagnostics: dict[str, Any] = {
        "active": False,
        "fallback_reason": None,
        "score_source": score_source,
        "locator_checkpoint": getattr(
            flashvid_config,
            "_certg_locator_checkpoint",
            None,
        ),
        "locator_error": getattr(flashvid_config, "_certg_locator_error", None),
        "locator_runtime_error": getattr(
            flashvid_config,
            "_certg_locator_runtime_error",
            None,
        ),
        "prompt_count": int(getattr(flashvid_config, "_certg_prompt_count", 0)),
        "duration_seconds": duration,
        "has_real_timestamps": has_real_times,
        "frame_count": frame_count,
        "raw_token_count": int(frame_count * tokens_per_frame),
        "question_word_count": len(re.findall(r"\b\w+\b", question)),
    }
    setattr(flashvid_config, "_certvid_design_mass_multiplier", None)

    fallback_reason: Optional[str] = None
    if not bool(getattr(flashvid_config, "certg_enabled", True)):
        fallback_reason = "disabled"
    elif not has_real_times:
        fallback_reason = "timestamps_missing"
    elif duration < _cfg_float(flashvid_config, "certg_min_duration_seconds", 120.0):
        fallback_reason = "short_horizon"
    elif diagnostics["question_word_count"] < _cfg_int(
        flashvid_config,
        "certg_min_question_words",
        6,
    ):
        fallback_reason = "short_query"
    elif bool(getattr(flashvid_config, "certg_subtitle_fallback", True)) and any(
        re.search(pattern, question.lower()) for pattern in _TEXT_ONLY_PATTERNS
    ):
        fallback_reason = "text_only_query"
    elif scores is None:
        fallback_reason = score_source or "locator_missing"
    else:
        scores = torch.as_tensor(
            scores,
            dtype=torch.float32,
            device=video_features.device,
        ).flatten()
        if scores.numel() != frame_count:
            fallback_reason = "score_shape_mismatch"
        elif not bool(torch.isfinite(scores).all()):
            fallback_reason = "score_non_finite"

    confidence = 0.0
    z = torch.zeros(frame_count, dtype=torch.float32, device=video_features.device)
    radius = max(1, _cfg_int(flashvid_config, "certg_window_radius", 2))
    if fallback_reason is None:
        confidence, z = _locator_confidence(scores, radius)
        diagnostics["score_min"] = float(scores.min().item())
        diagnostics["score_mean"] = float(scores.mean().item())
        diagnostics["score_max"] = float(scores.max().item())
        diagnostics["score_std"] = float(scores.std(unbiased=False).item())
        diagnostics["confidence"] = confidence
        if confidence < _cfg_float(
            flashvid_config,
            "certg_confidence_threshold",
            0.55,
        ):
            fallback_reason = "low_confidence"

    if fallback_reason is not None:
        diagnostics["fallback_reason"] = fallback_reason
        output, indices = certvid_v3_compression(
            video_features,
            cls_attention,
            flashvid_config,
            question_features,
        )
        diagnostics["selected_frame_counts"] = _frame_counts(
            indices,
            tokens_per_frame,
            frame_count,
        )
        diagnostics["budget"] = int(indices.numel())
        diagnostics.setdefault("confidence", confidence)
        _write_diagnostics(flashvid_config, diagnostics)
        return output, indices

    peaks = _peak_frames(
        z,
        max(1, _cfg_int(flashvid_config, "certg_peak_count", 2)),
        max(1, _cfg_int(flashvid_config, "certg_peak_separation", 3)),
    )
    if not peaks:
        diagnostics["fallback_reason"] = "no_positive_peak"
        output, indices = certvid_v3_compression(
            video_features,
            cls_attention,
            flashvid_config,
            question_features,
        )
        diagnostics["selected_frame_counts"] = _frame_counts(
            indices,
            tokens_per_frame,
            frame_count,
        )
        diagnostics["budget"] = int(indices.numel())
        _write_diagnostics(flashvid_config, diagnostics)
        return output, indices

    frame_multiplier = _frame_multiplier(
        z,
        peaks,
        confidence,
        radius,
        _cfg_float(flashvid_config, "certg_max_tilt", 2.0),
    )
    token_multiplier = frame_multiplier.repeat_interleave(tokens_per_frame)
    setattr(flashvid_config, "_certvid_design_mass_multiplier", token_multiplier)
    try:
        output, indices = certvid_v3_compression(
            video_features,
            cls_attention,
            flashvid_config,
            None
            if bool(
                getattr(
                    flashvid_config,
                    "certg_disable_v3_query_when_active",
                    True,
                )
            )
            else question_features,
        )
    finally:
        setattr(flashvid_config, "_certvid_design_mass_multiplier", None)

    diagnostics.update(
        {
            "active": True,
            "fallback_reason": None,
            "confidence": confidence,
            "peak_frames": peaks,
            "frame_scores": [
                round(float(value), 6)
                for value in scores.detach().cpu().tolist()
            ],
            "frame_z_scores": [
                round(float(value), 6)
                for value in z.detach().cpu().tolist()
            ],
            "frame_multipliers": [
                round(float(value), 6)
                for value in frame_multiplier.detach().cpu().tolist()
            ],
            "selected_frame_counts": _frame_counts(
                indices,
                tokens_per_frame,
                frame_count,
            ),
            "budget": int(indices.numel()),
            "v3_query_disabled": bool(
                getattr(
                    flashvid_config,
                    "certg_disable_v3_query_when_active",
                    True,
                )
            ),
        }
    )
    _write_diagnostics(flashvid_config, diagnostics)
    return output, indices
