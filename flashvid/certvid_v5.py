from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .certvid import (
    CertVidPlan,
    _build_components,
    _cfg_float,
    _cfg_int,
    _grid_hw,
    _local_detail,
    _metric_features,
    _minmax,
    _question_atoms,
    _question_relevance,
    _spatial_layout,
    apply_certvid_plan,
)
from .certvid_v2 import _component_support, _trajectory_signals
from .configuration_flashvid import FlashVidConfig


_CERTIFICATE_CATEGORIES = ("query", "frame", "scene", "motion", "track")
_CERTIFICATE_SHARES = {
    "query": 0.16,
    "frame": 0.22,
    "scene": 0.24,
    "motion": 0.20,
    "track": 0.18,
}
_CANDIDATE_SOURCES = ("motion", "query", "track", "scene", "spatial", "global")
_CANDIDATE_SHARES = {
    "motion": 0.20,
    "query": 0.15,
    "track": 0.17,
    "scene": 0.17,
    "spatial": 0.14,
    "global": 0.17,
}
_QUERY_MODES = {
    "certificates_only",
    "kernel_only",
    "certificates_and_kernel",
    # Aliases retained so an old launcher fails neither parsing nor resumption.
    "design_only",
    "certificates_and_design",
    "off",
}


@dataclass(frozen=True)
class _CertificateRequest:
    category: str
    request_id: str
    tokens: tuple[int, ...]
    score: float


def _identity_plan(total_tokens: int, device: torch.device) -> CertVidPlan:
    indices = torch.arange(total_tokens, dtype=torch.long, device=device)
    return CertVidPlan(
        anchor_indices=indices,
        assignment_indices=indices.unsqueeze(1),
        assignment_weights=torch.ones((total_tokens, 1), dtype=torch.float32, device=device),
        source_mass=torch.ones(total_tokens, dtype=torch.float32, device=device),
        fusion_alpha=torch.zeros(total_tokens, dtype=torch.float32, device=device),
        raw_token_count=total_tokens,
    )


def _tie_safe_rank_normalize(values: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, bool, str]:
    """Return mid-ranks so tied or degenerate inputs cannot encode token position."""
    tensor = values.float()
    flat = tensor.reshape(-1)
    if flat.numel() == 0:
        return tensor, False, "empty"
    if not bool(torch.isfinite(flat).all()):
        raise ValueError("cls_attention contains NaN or Inf")
    if tensor.ndim == 0 or tensor.shape[-1] <= 1:
        return torch.zeros_like(tensor), False, "single_value"

    eps = max(0.0, float(eps))
    rows = tensor.reshape(-1, tensor.shape[-1])
    normalized = torch.zeros_like(rows)
    valid_rows = 0
    for row_idx, row in enumerate(rows):
        spread = float((row.max() - row.min()).item())
        standard_deviation = float(row.std(unbiased=False).item())
        if spread < eps or standard_deviation < eps:
            continue
        _, inverse, counts = torch.unique(row, sorted=True, return_inverse=True, return_counts=True)
        counts_float = counts.float()
        starts = torch.cumsum(counts_float, dim=0) - counts_float
        mid_ranks = starts + 0.5 * (counts_float - 1.0)
        normalized[row_idx] = mid_ranks[inverse] / float(row.numel() - 1)
        valid_rows += 1

    if valid_rows == 0:
        return normalized.reshape_as(tensor), False, "degenerate"
    reason = "validated" if valid_rows == rows.shape[0] else "validated_partial"
    return normalized.reshape_as(tensor), True, reason


def _validated_attention(
    cls_attention: torch.Tensor,
    frame_count: int,
    tokens_per_frame: int,
    config: FlashVidConfig,
) -> tuple[torch.Tensor, dict[str, object]]:
    expected = (int(frame_count), int(tokens_per_frame))
    if tuple(cls_attention.shape) != expected:
        raise ValueError(f"cls_attention must have shape {expected}, got {tuple(cls_attention.shape)}")
    raw = cls_attention.float()
    if not bool(torch.isfinite(raw).all()):
        raise ValueError("cls_attention contains NaN or Inf")

    policy = str(getattr(config, "certv5_attention_policy", "validated")).strip().lower()
    if policy not in {"validated", "strict", "off"}:
        raise ValueError(f"unsupported certv5_attention_policy={policy!r}")
    source = str(getattr(config, "_certvid_attention_source", "missing")).strip().lower()
    diagnostic: dict[str, object] = {
        "policy": policy,
        "source": source,
        "used": False,
        "reason": "policy_off" if policy == "off" else "unvalidated_source",
    }
    if policy == "strict" and source != "manual_qk":
        raise ValueError(
            "certv5_attention_policy='strict' requires attention provenance 'manual_qk', "
            f"got {source!r}"
        )
    if policy == "off" or source != "manual_qk":
        return torch.zeros(frame_count * tokens_per_frame, dtype=torch.float32, device=raw.device), diagnostic

    normalized, used, reason = _tie_safe_rank_normalize(
        raw,
        _cfg_float(config, "certv5_attention_eps", 1e-6),
    )
    diagnostic["used"] = bool(used)
    diagnostic["reason"] = reason
    diagnostic["raw_std"] = float(raw.std(unbiased=False).item())
    diagnostic["raw_range"] = float((raw.max() - raw.min()).item())
    return normalized.reshape(-1), diagnostic


def _resolve_budget(
    config: FlashVidConfig,
    total_tokens: int,
) -> tuple[int, dict[str, object]]:
    mode = str(getattr(config, "certv5_budget_mode", "layer_average")).strip().lower()
    if mode not in {"layer_average", "outer_only"}:
        raise ValueError(f"unsupported certv5_budget_mode={mode!r}")

    nominal = _cfg_float(config, "retention_ratio", 0.10)
    expansion = _cfg_float(config, "expansion", 1.0)
    pruning_layer = _cfg_int(config, "pruning_layer", 0)
    inner_retention = _cfg_float(config, "llm_retention_ratio", 1.0)
    layers = _cfg_int(config, "certv5_num_hidden_layers", 0)
    tolerance = 1e-4
    if not (0.0 < nominal <= 1.0):
        raise ValueError(f"retention_ratio must be in (0, 1], got {nominal}")

    if mode == "outer_only":
        if abs(expansion - 1.0) > tolerance or abs(inner_retention - 1.0) > tolerance:
            raise ValueError(
                "certv5 outer_only requires expansion=1 and llm_retention_ratio=1; "
                f"got expansion={expansion}, llm_retention_ratio={inner_retention}"
            )
        outer_retention = nominal
        post_inner_retention = nominal
        layer_multiplier = 1.0
        average_retention = nominal
    else:
        if layers <= 1:
            raise ValueError("certv5 layer_average requires certv5_num_hidden_layers from the model config")
        if not (0 < pruning_layer < layers):
            raise ValueError(
                f"pruning_layer must satisfy 0 < K < L, got K={pruning_layer}, L={layers}"
            )
        if not (0.0 < inner_retention < 1.0):
            raise ValueError(
                "certv5 layer_average requires 0 < llm_retention_ratio < 1, "
                f"got {inner_retention}"
            )
        if not bool(getattr(config, "certv5_inner_hook_enabled", False)):
            raise ValueError("certv5 layer_average requires an installed inner-pruning hook")
        if nominal * expansion > 1.0 + tolerance:
            raise ValueError(
                f"outer retention R*E must not exceed 1, got {nominal * expansion:.8f}"
            )
        layer_multiplier = expansion * (
            pruning_layer + (layers - pruning_layer) * inner_retention
        ) / float(layers)
        if abs(layer_multiplier - 1.0) > tolerance:
            raise ValueError(
                "certv5 layer_average budget is not aligned: "
                f"E*(K+(L-K)*r)/L={layer_multiplier:.8f}, expected 1 within {tolerance}"
            )
        outer_retention = nominal * expansion
        post_inner_retention = outer_retention * inner_retention
        average_retention = nominal * layer_multiplier

    budget = max(1, int(round(total_tokens * outer_retention)))
    if budget > total_tokens:
        raise ValueError(f"certv5 outer budget {budget} exceeds raw token count {total_tokens}")
    post_inner_tokens = (
        max(1, int(round(budget * inner_retention)))
        if mode == "layer_average"
        else budget
    )
    average_layer_tokens = (
        (pruning_layer * budget + (layers - pruning_layer) * post_inner_tokens) / float(layers)
        if mode == "layer_average"
        else float(budget)
    )
    diagnostics: dict[str, object] = {
        "mode": mode,
        "nominal_retention": nominal,
        "outer_retention": outer_retention,
        "post_inner_retention": post_inner_retention,
        "average_retention": average_retention,
        "average_layer_multiplier": layer_multiplier,
        "expansion": expansion,
        "pruning_layer": pruning_layer,
        "inner_retention": inner_retention,
        "num_hidden_layers": layers,
        "raw_tokens": total_tokens,
        "target_tokens": budget,
        "post_inner_tokens": post_inner_tokens,
        "average_layer_tokens": average_layer_tokens,
    }
    return budget, diagnostics


def _whiten_features(features: torch.Tensor, strength: float) -> torch.Tensor:
    strength = min(1.0, max(0.0, float(strength)))
    centered = features.float() - features.float().mean(dim=0, keepdim=True)
    if strength <= 1e-6 or centered.shape[0] <= 1:
        return F.normalize(centered, p=2, dim=-1, eps=1e-6)
    covariance = centered.transpose(0, 1) @ centered / float(max(1, centered.shape[0] - 1))
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    mean_eigenvalue = eigenvalues.mean().clamp_min(1e-6)
    eigenvalues = eigenvalues.clamp_min(mean_eigenvalue * 1e-4)
    scales = torch.pow(eigenvalues / mean_eigenvalue, -0.5 * strength)
    whitened = (centered @ eigenvectors) * scales.unsqueeze(0)
    return F.normalize(
        torch.nan_to_num(whitened, nan=0.0, posinf=0.0, neginf=0.0),
        p=2,
        dim=-1,
        eps=1e-6,
    )


def _query_uses_certificates(mode: str) -> bool:
    return mode in {"certificates_only", "certificates_and_kernel", "certificates_and_design"}


def _query_uses_kernel(mode: str) -> bool:
    return mode in {"kernel_only", "certificates_and_kernel", "design_only", "certificates_and_design"}


def _balanced_target_mass(
    *,
    quality: torch.Tensor,
    event_score: torch.Tensor,
    component_support: torch.Tensor,
    query_score: torch.Tensor,
    frame_ids: torch.Tensor,
    frame_event: torch.Tensor,
    query_weight: float,
    quality_floor: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Build the quality-weighted empirical measure that the coreset represents."""
    quality_floor = min(1.0, max(1e-4, float(quality_floor)))
    query_weight = min(0.20, max(0.0, float(query_weight)))
    visual_weight = max(0.0, 1.0 - query_weight)
    local_mass = quality_floor + visual_weight * (
        0.48 * quality + 0.30 * event_score + 0.22 * component_support
    )
    if query_weight > 0.0:
        local_mass = local_mass + query_weight * query_score
    local_mass = local_mass.clamp_min(1e-6)

    frame_count = int(frame_event.numel())
    frame_quality = torch.zeros(frame_count, dtype=torch.float32, device=quality.device)
    frame_query = torch.zeros_like(frame_quality)
    within_frame = torch.zeros_like(local_mass)
    for frame_idx in range(frame_count):
        members = torch.where(frame_ids == frame_idx)[0]
        if members.numel() == 0:
            continue
        values = local_mass[members]
        within_frame[members] = values / values.sum().clamp_min(1e-6)
        frame_quality[frame_idx] = quality[members].mean()
        frame_query[frame_idx] = query_score[members].max()

    # Quiet frames retain a majority uniform prior; event and query evidence only
    # redistribute the remaining mass. This prevents long-video blind spots.
    frame_mass = 0.60 + 0.24 * frame_event.float() + 0.16 * frame_quality
    if query_weight > 0.0:
        frame_mass = frame_mass + query_weight * frame_query
    frame_mass = frame_mass / frame_mass.sum().clamp_min(1e-6)
    target_mass = within_frame * frame_mass[frame_ids]
    target_mass = target_mass / target_mass.sum().clamp_min(1e-6)
    entropy = float((-(target_mass * torch.log(target_mass.clamp_min(1e-12))).sum()).item())
    return target_mass, {
        "entropy": entropy,
        "effective_support": float(math.exp(min(50.0, entropy))),
        "min_frame_mass": float(frame_mass.min().item()),
        "max_frame_mass": float(frame_mass.max().item()),
    }


def _evenly_spaced_ids(count: int, keep: int, device: torch.device) -> torch.Tensor:
    if keep >= count:
        return torch.arange(count, dtype=torch.long, device=device)
    if keep <= 1:
        return torch.tensor([count // 2], dtype=torch.long, device=device)
    return torch.round(torch.linspace(0, count - 1, steps=keep, device=device)).long().unique()


def _top_fraction_mean(values: torch.Tensor, fraction: float) -> float:
    flat = values.reshape(-1).float()
    if flat.numel() == 0:
        return 0.0
    count = min(int(flat.numel()), max(1, int(math.ceil(flat.numel() * float(fraction)))))
    return float(torch.topk(flat, k=count).values.mean().item())


def _scene_pyramid(
    frame_event: torch.Tensor,
    frame_count: int,
    max_scenes: int,
    boundary_threshold: float,
    min_scene_frames: int,
    fine_bins: int,
    coarse_bins: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    """Create event-aware scenes plus fixed fine/coarse temporal coordinates."""
    device = frame_event.device
    max_scenes = min(frame_count, max(1, int(max_scenes)))
    min_gap = max(1, int(min_scene_frames))
    threshold = min(1.0, max(0.0, float(boundary_threshold)))
    candidates = [
        (float(frame_event[idx].item()), idx)
        for idx in range(1, frame_count)
        if float(frame_event[idx].item()) >= threshold
    ]
    candidates.sort(key=lambda item: (-item[0], item[1]))
    boundaries: list[int] = []
    for _, frame_idx in candidates:
        if len(boundaries) >= max_scenes - 1:
            break
        if frame_idx < min_gap or frame_count - frame_idx < min_gap:
            continue
        if any(abs(frame_idx - existing) < min_gap for existing in boundaries):
            continue
        boundaries.append(frame_idx)
    boundaries.sort()

    frame_range = torch.arange(frame_count, dtype=torch.long, device=device)
    if boundaries:
        boundary_tensor = torch.tensor(boundaries, dtype=torch.long, device=device)
        frame_scene_ids = torch.bucketize(frame_range, boundary_tensor, right=True)
    else:
        frame_scene_ids = torch.zeros(frame_count, dtype=torch.long, device=device)
    scene_count = len(boundaries) + 1

    fine_count = min(frame_count, max(1, int(fine_bins)))
    coarse_count = min(frame_count, max(1, int(coarse_bins)))
    frame_fine_ids = torch.div(
        frame_range * fine_count,
        max(1, frame_count),
        rounding_mode="floor",
    ).clamp_max(fine_count - 1)
    frame_coarse_ids = torch.div(
        frame_range * coarse_count,
        max(1, frame_count),
        rounding_mode="floor",
    ).clamp_max(coarse_count - 1)
    diagnostics: dict[str, object] = {
        "boundaries": boundaries,
        "scene_count": scene_count,
        "fine_count": fine_count,
        "coarse_count": coarse_count,
        "boundary_threshold": threshold,
    }
    return frame_scene_ids, frame_fine_ids, frame_coarse_ids, diagnostics


def _directed_motion_signals(
    metric_frames: torch.Tensor,
    coords: torch.Tensor,
    matches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    novelty: torch.Tensor,
    curvature: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover signed patch trajectories instead of retaining only motion magnitude."""
    frame_count, tokens_per_frame, _ = metric_frames.shape
    device = metric_frames.device
    velocity = torch.zeros((frame_count, tokens_per_frame, 2), dtype=torch.float32, device=device)
    acceleration = torch.zeros_like(velocity)
    confidence = torch.zeros((frame_count, tokens_per_frame), dtype=torch.float32, device=device)
    previous_global = torch.full(
        (frame_count, tokens_per_frame),
        -1,
        dtype=torch.long,
        device=device,
    )

    for frame_idx, (best_previous, mutual, similarity) in enumerate(matches, start=1):
        velocity[frame_idx] = coords - coords[best_previous]
        confidence[frame_idx] = ((similarity.float() + 1.0) * 0.5).clamp(0.0, 1.0)
        confidence[frame_idx] *= 0.35 + 0.65 * mutual.float()
        previous_global[frame_idx] = (frame_idx - 1) * tokens_per_frame + best_previous
        if frame_idx >= 2:
            acceleration[frame_idx] = velocity[frame_idx] - velocity[frame_idx - 1][best_previous]

    speed = velocity.norm(dim=-1).clamp(0.0, math.sqrt(2.0)) / math.sqrt(2.0)
    acceleration_norm = acceleration.norm(dim=-1).clamp(0.0, 2.0 * math.sqrt(2.0)) / (
        2.0 * math.sqrt(2.0)
    )
    motion_score = _minmax(
        0.38 * speed
        + 0.24 * acceleration_norm
        + 0.26 * novelty.float()
        + 0.12 * curvature.float(),
        dim=-1,
    )
    direction_axes = torch.cat(
        [velocity, acceleration, speed.unsqueeze(-1), acceleration_norm.unsqueeze(-1)],
        dim=-1,
    )
    return (
        motion_score.reshape(-1),
        confidence.reshape(-1),
        direction_axes.reshape(frame_count * tokens_per_frame, -1),
        previous_global.reshape(-1),
    )


def _adaptive_source_shares(
    motion_score: torch.Tensor,
    frame_event: torch.Tensor,
    scene_count: int,
    router_strength: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    motion_activity = min(
        1.0,
        max(
            0.0,
            0.55 * _top_fraction_mean(motion_score, 0.20)
            + 0.25 * float(motion_score.mean().item())
            + 0.20 * float((motion_score > 0.55).float().mean().item()),
        ),
    )
    event_activity = min(
        1.0,
        max(
            0.0,
            0.60 * _top_fraction_mean(frame_event, 0.25)
            + 0.40 * min(1.0, max(0.0, (scene_count - 1) / 7.0)),
        ),
    )
    strength = min(1.0, max(0.0, float(router_strength)))

    certificate = dict(_CERTIFICATE_SHARES)
    certificate["motion"] += strength * 0.14 * motion_activity
    certificate["track"] += strength * (0.06 * motion_activity + 0.04 * event_activity)
    certificate["scene"] += strength * 0.10 * event_activity
    certificate["frame"] = max(
        0.08,
        certificate["frame"] - strength * (0.08 * motion_activity + 0.04 * event_activity),
    )

    candidate = dict(_CANDIDATE_SHARES)
    candidate["motion"] += strength * 0.16 * motion_activity
    candidate["track"] += strength * 0.06 * motion_activity
    candidate["scene"] += strength * 0.08 * event_activity
    candidate["global"] = max(
        0.06,
        candidate["global"] - strength * (0.10 * motion_activity + 0.04 * event_activity),
    )

    def normalize(values: dict[str, float]) -> dict[str, float]:
        total = sum(max(0.0, value) for value in values.values())
        return {name: max(0.0, value) / max(total, 1e-6) for name, value in values.items()}

    diagnostics = {
        "motion_activity": motion_activity,
        "event_activity": event_activity,
        "router_strength": strength,
    }
    return normalize(certificate), normalize(candidate), diagnostics


def _largest_remainder(total: int, shares: dict[str, float], names: list[str]) -> dict[str, int]:
    if total <= 0 or not names:
        return {name: 0 for name in names}
    weight_sum = sum(max(0.0, shares[name]) for name in names)
    if weight_sum <= 0.0:
        weight_sum = float(len(names))
        normalized = {name: 1.0 / len(names) for name in names}
    else:
        normalized = {name: max(0.0, shares[name]) / weight_sum for name in names}
    raw = {name: total * normalized[name] for name in names}
    output = {name: int(math.floor(raw[name])) for name in names}
    remaining = total - sum(output.values())
    order = sorted(names, key=lambda name: (-(raw[name] - output[name]), names.index(name)))
    for name in order[:remaining]:
        output[name] += 1
    return output


def _build_certificate_requests(
    *,
    quality: torch.Tensor,
    event_score: torch.Tensor,
    motion_score: torch.Tensor,
    motion_confidence: torch.Tensor,
    previous_global: torch.Tensor,
    frame_ids: torch.Tensor,
    fine_temporal_ids: torch.Tensor,
    coarse_temporal_ids: torch.Tensor,
    scene_ids: torch.Tensor,
    spatial_ids: torch.Tensor,
    component_ids: torch.Tensor,
    query_relevance: torch.Tensor,
    atom_weights: torch.Tensor,
    query_confidence: float,
    query_mode: str,
    frame_count: int,
    scene_count: int,
    spatial_count: int,
    frame_coverage_ratio: float,
    query_threshold: float,
    query_per_atom: int,
    motion_threshold: float,
    motion_confidence_threshold: float,
) -> list[_CertificateRequest]:
    requests: list[_CertificateRequest] = []
    frame_keep = max(1, int(math.ceil(frame_count * min(1.0, max(0.0, frame_coverage_ratio)))))
    for frame_id in _evenly_spaced_ids(frame_count, frame_keep, quality.device).tolist():
        members = torch.where(frame_ids == int(frame_id))[0]
        if members.numel() > 0:
            token = int(members[torch.argmax(quality[members])].item())
            requests.append(
                _CertificateRequest("frame", f"frame:{frame_id}", (token,), float(quality[token]))
            )

    # Scene entry, exit, and event nodes preserve the long-range state chain.
    for scene_id in range(scene_count):
        members = torch.where(scene_ids == scene_id)[0]
        if members.numel() == 0:
            continue
        scene_frames = frame_ids[members]
        first_frame = int(scene_frames.min().item())
        last_frame = int(scene_frames.max().item())
        for label, target_frame in (("entry", first_frame), ("exit", last_frame)):
            local_members = members[scene_frames == target_frame]
            token = int(local_members[torch.argmax(quality[local_members])].item())
            requests.append(
                _CertificateRequest(
                    "scene",
                    f"scene:{scene_id}:{label}",
                    (token,),
                    float(quality[token]),
                )
            )
        event_token = int(members[torch.argmax(event_score[members])].item())
        requests.append(
            _CertificateRequest(
                "scene",
                f"scene:{scene_id}:event",
                (event_token,),
                float(event_score[event_token]),
            )
        )

    # A direction claim requires both ends of a correspondence. Keep pairs atomic.
    motion_gate = (
        (previous_global >= 0)
        & (motion_score >= float(motion_threshold))
        & (motion_confidence >= float(motion_confidence_threshold))
    )
    joint_motion_cells = (
        fine_temporal_ids * spatial_count + spatial_ids
    )
    for cell_id in torch.unique(joint_motion_cells[motion_gate]).detach().cpu().tolist():
        members = torch.where(motion_gate & (joint_motion_cells == int(cell_id)))[0]
        if members.numel() == 0:
            continue
        score = (
            0.55 * motion_score[members]
            + 0.25 * motion_confidence[members]
            + 0.20 * quality[members]
        )
        current = int(members[torch.argmax(score)].item())
        previous = int(previous_global[current].item())
        tokens = tuple(sorted({previous, current}))
        requests.append(
            _CertificateRequest(
                "motion",
                f"motion:{cell_id}:{previous}:{current}",
                tokens,
                float(score.max().item()),
            )
        )

    # Persistent components contribute ordered endpoints, supporting count and identity.
    for component_id in torch.unique(component_ids).detach().cpu().tolist():
        members = torch.where(component_ids == int(component_id))[0]
        if members.numel() < 2:
            continue
        component_frames = frame_ids[members]
        first_frame = int(component_frames.min().item())
        last_frame = int(component_frames.max().item())
        if first_frame == last_frame:
            continue
        first_members = members[component_frames == first_frame]
        last_members = members[component_frames == last_frame]
        first_score = 0.65 * quality[first_members] + 0.35 * motion_score[first_members]
        last_score = 0.65 * quality[last_members] + 0.35 * motion_score[last_members]
        first = int(first_members[torch.argmax(first_score)].item())
        last = int(last_members[torch.argmax(last_score)].item())
        span = (last_frame - first_frame) / max(1, frame_count - 1)
        score = 0.45 * span + 0.30 * float(motion_score[members].max()) + 0.25 * float(quality[members].mean())
        requests.append(
            _CertificateRequest(
                "track",
                f"track:{component_id}:{first}:{last}",
                tuple(sorted({first, last})),
                score,
            )
        )

    if (
        _query_uses_certificates(query_mode)
        and query_relevance.numel() > 0
        and query_confidence >= float(query_threshold)
    ):
        per_atom = max(1, int(query_per_atom))
        for atom_idx, atom_scores in enumerate(query_relevance):
            alternatives: list[tuple[float, int, int]] = []
            for scene_id in range(scene_count):
                members = torch.where(scene_ids == scene_id)[0]
                if members.numel() == 0:
                    continue
                token = int(members[torch.argmax(atom_scores[members])].item())
                score = float(atom_scores[token] * atom_weights[atom_idx].clamp_min(1e-6))
                alternatives.append((score, token, scene_id))
            alternatives.sort(key=lambda item: (-item[0], item[1]))
            for rank, (score, token, scene_id) in enumerate(alternatives[:per_atom]):
                requests.append(
                    _CertificateRequest(
                        "query",
                        f"query:{atom_idx}:{rank}:{scene_id}",
                        (token,),
                        score,
                    )
                )
    return requests


def _admit_certificates(
    requests: list[_CertificateRequest],
    budget: int,
    budget_ratio: float,
    shares: dict[str, float],
) -> tuple[list[int], dict[str, object], dict[str, set[int]]]:
    cap = min(budget, max(0, int(math.floor(budget * min(1.0, max(0.0, budget_ratio))))))
    grouped = {name: [] for name in _CERTIFICATE_CATEGORIES}
    for request in requests:
        grouped[request.category].append(request)
    for category in _CERTIFICATE_CATEGORIES:
        grouped[category].sort(key=lambda request: (-request.score, request.tokens, request.request_id))

    active = [category for category in _CERTIFICATE_CATEGORIES if grouped[category]]
    quotas = _largest_remainder(cap, shares, active)
    selected: set[int] = set()
    admitted_by_source = {category: 0 for category in _CERTIFICATE_CATEGORIES}
    pointers = {category: 0 for category in _CERTIFICATE_CATEGORIES}

    def offer(category: str, source_remaining: int, total_remaining: int) -> int:
        entries = grouped[category]
        while pointers[category] < len(entries):
            request = entries[pointers[category]]
            pointers[category] += 1
            new_tokens = [token for token in request.tokens if token not in selected]
            if not new_tokens:
                continue
            # Multi-token requests are atomic: never admit only one end of a tracklet.
            # A two-token tracklet may exceed its category quota by one, but it
            # must never be split or exceed the global certificate cap.
            if source_remaining <= 0 or len(new_tokens) > total_remaining:
                continue
            selected.update(new_tokens)
            admitted_by_source[category] += len(new_tokens)
            return len(new_tokens)
        return 0

    for category in active:
        while admitted_by_source[category] < quotas[category]:
            added = offer(
                category,
                quotas[category] - admitted_by_source[category],
                cap - len(selected),
            )
            if added <= 0:
                break

    cycle: list[str] = []
    for category in _CERTIFICATE_CATEGORIES:
        cycle.extend([category] * max(1, int(round(shares.get(category, 0.0) * 20))))
    while len(selected) < cap:
        progress = False
        for category in cycle:
            if len(selected) >= cap:
                break
            added = offer(category, cap - len(selected), cap - len(selected))
            progress = added > 0 or progress
        if not progress:
            break

    selected_sorted = sorted(selected)
    selected_set = set(selected_sorted)
    protected_by_category = {category: set() for category in _CERTIFICATE_CATEGORIES}
    category_stats: dict[str, dict[str, int]] = {}
    for category in _CERTIFICATE_CATEGORIES:
        entries = grouped[category]
        requested_tokens = {token for entry in entries for token in entry.tokens}
        admitted_entries = [entry for entry in entries if set(entry.tokens).issubset(selected_set)]
        admitted_tokens = {token for entry in admitted_entries for token in entry.tokens}
        protected_by_category[category].update(admitted_tokens)
        category_stats[category] = {
            "requested": len(entries),
            "deduplicated": len(requested_tokens),
            "admitted": len(admitted_entries),
            "admitted_unique": len(admitted_tokens),
            "truncated_or_unmet": len(entries) - len(admitted_entries),
            "quota": quotas.get(category, 0),
            "contributed_unique": admitted_by_source[category],
        }
    diagnostics: dict[str, object] = {
        "cap": cap,
        "admitted_unique": len(selected_sorted),
        "ratio": len(selected_sorted) / float(max(1, budget)),
        "requested": len(requests),
        "deduplicated": len({token for request in requests for token in request.tokens}),
        "categories": category_stats,
        "shares": shares,
    }
    return selected_sorted, diagnostics, protected_by_category


def _unique_ranked(tokens: list[tuple[float, int]]) -> list[int]:
    tokens.sort(key=lambda item: (-item[0], item[1]))
    output: list[int] = []
    seen: set[int] = set()
    for _, token in tokens:
        if token not in seen:
            output.append(token)
            seen.add(token)
    return output


def _candidate_pool(
    *,
    budget: int,
    quality: torch.Tensor,
    event_score: torch.Tensor,
    motion_score: torch.Tensor,
    component_ids: torch.Tensor,
    fine_temporal_ids: torch.Tensor,
    scene_ids: torch.Tensor,
    spatial_ids: torch.Tensor,
    query_relevance: torch.Tensor,
    atom_weights: torch.Tensor,
    query_mode: str,
    requests: list[_CertificateRequest],
    locked: list[int],
    multiplier: float,
    shares: dict[str, float],
) -> tuple[torch.Tensor, dict[str, object]]:
    total_tokens = int(quality.numel())
    limit = min(total_tokens, max(budget, int(math.ceil(budget * max(1.0, multiplier)))))
    offers: dict[str, list[int]] = {name: [] for name in _CANDIDATE_SOURCES}

    if _query_uses_kernel(query_mode) and query_relevance.numel() > 0:
        ranked: list[tuple[float, int]] = []
        for atom_idx, scores in enumerate(query_relevance):
            for scene_id in torch.unique(scene_ids).tolist():
                members = torch.where(scene_ids == int(scene_id))[0]
                if members.numel() > 0:
                    token = int(members[torch.argmax(scores[members])].item())
                    ranked.append((float(scores[token] * atom_weights[atom_idx].clamp_min(1e-6)), token))
        offers["query"] = _unique_ranked(ranked)

    motion_ranked: list[tuple[float, int]] = []
    track_ranked: list[tuple[float, int]] = []
    scene_ranked: list[tuple[float, int]] = []
    for request in requests:
        target = {
            "motion": motion_ranked,
            "track": track_ranked,
            "scene": scene_ranked,
        }.get(request.category)
        if target is not None:
            target.extend((request.score, int(token)) for token in request.tokens)
    motion_ranked.extend(
        (float(motion_score[token]), int(token))
        for token in torch.argsort(motion_score, descending=True, stable=True).detach().cpu().tolist()
    )
    offers["motion"] = _unique_ranked(motion_ranked)

    quality_cpu = quality.detach().float().cpu().tolist()
    representatives: dict[int, int] = {}
    for token, component in enumerate(component_ids.detach().cpu().tolist()):
        previous = representatives.get(int(component))
        if previous is None or quality_cpu[token] > quality_cpu[previous]:
            representatives[int(component)] = token
    track_ranked.extend((quality_cpu[token], token) for token in representatives.values())
    offers["track"] = _unique_ranked(track_ranked)

    for scene_id in torch.unique(scene_ids).tolist():
        members = torch.where(scene_ids == int(scene_id))[0]
        if members.numel() == 0:
            continue
        scene_quality = 0.55 * quality[members] + 0.45 * event_score[members]
        token = int(members[torch.argmax(scene_quality)].item())
        scene_ranked.append((float(scene_quality.max()), token))
    offers["scene"] = _unique_ranked(scene_ranked)

    joint_cells = fine_temporal_ids * (int(spatial_ids.max().item()) + 1) + spatial_ids
    cells: list[tuple[float, int]] = []
    for cell_id in torch.unique(joint_cells).tolist():
        members = torch.where(joint_cells == int(cell_id))[0]
        token = int(members[torch.argmax(quality[members])].item())
        cells.append((float(quality[token]), token))
    offers["spatial"] = _unique_ranked(cells)
    offers["global"] = torch.argsort(quality, descending=True, stable=True).detach().cpu().tolist()

    selected: set[int] = set(int(token) for token in locked)
    contributed = {name: 0 for name in _CANDIDATE_SOURCES}
    pointers = {name: 0 for name in _CANDIDATE_SOURCES}
    remaining = max(0, limit - len(selected))
    active = [name for name in _CANDIDATE_SOURCES if offers[name]]
    quotas = _largest_remainder(remaining, shares, active)

    def offer(source: str) -> bool:
        source_tokens = offers[source]
        while pointers[source] < len(source_tokens):
            token = int(source_tokens[pointers[source]])
            pointers[source] += 1
            if token in selected:
                continue
            if len(selected) >= limit:
                return False
            selected.add(token)
            contributed[source] += 1
            return True
        return False

    for source in active:
        while contributed[source] < quotas[source]:
            if not offer(source):
                break

    cycle: list[str] = []
    for source in _CANDIDATE_SOURCES:
        cycle.extend([source] * max(1, int(round(shares.get(source, 0.0) * 20))))
    while len(selected) < limit:
        progress = False
        for source in cycle:
            if len(selected) >= limit:
                break
            progress = offer(source) or progress
        if not progress:
            break
    if len(selected) < limit:
        for token in offers["global"]:
            selected.add(token)
            if len(selected) >= limit:
                break
    if len(selected) < budget:
        raise RuntimeError(f"candidate pool has {len(selected)} candidates for budget {budget}")

    diagnostics: dict[str, object] = {
        "limit": limit,
        "locked": len(locked),
        "shares": shares,
        "sources": {
            source: {
                "offered": len(offers[source]),
                "unique": len(set(offers[source])),
                "quota": quotas.get(source, 0),
                "admitted": contributed[source],
                "covered": len(set(offers[source]) & selected),
            }
            for source in _CANDIDATE_SOURCES
        },
    }
    return torch.tensor(sorted(selected), dtype=torch.long, device=quality.device), diagnostics


def _kernel_weights(
    config: FlashVidConfig,
    router: dict[str, float],
    query_confidence: float,
    query_enabled: bool,
) -> dict[str, float]:
    motion = float(router["motion_activity"])
    event = float(router["event_activity"])
    strength = float(router["router_strength"])
    weights = {
        "appearance": max(0.0, _cfg_float(config, "certv5_appearance_kernel_weight", 0.46)),
        "temporal": max(0.0, _cfg_float(config, "certv5_temporal_kernel_weight", 0.16))
        * (1.0 + strength * 0.45 * event),
        "motion": max(0.0, _cfg_float(config, "certv5_motion_kernel_weight", 0.16))
        * (1.0 + strength * 0.75 * motion),
        "event": max(0.0, _cfg_float(config, "certv5_event_kernel_weight", 0.10))
        * (1.0 + strength * 0.40 * event),
        "spatial": max(0.0, _cfg_float(config, "certv5_spatial_kernel_weight", 0.07)),
        "query": max(0.0, _cfg_float(config, "certv5_query_kernel_weight", 0.05))
        * min(1.0, max(0.0, float(query_confidence)))
        if query_enabled
        else 0.0,
    }
    total = sum(weights.values())
    if total <= 1e-8:
        weights["appearance"] = 1.0
        total = 1.0
    return {name: value / total for name, value in weights.items()}


def _prepare_kernel_state(
    *,
    metric_features: torch.Tensor,
    frame_ids: torch.Tensor,
    frame_count: int,
    spatial_coords: torch.Tensor,
    scene_ids: torch.Tensor,
    novelty: torch.Tensor,
    curvature: torch.Tensor,
    event: torch.Tensor,
    detail: torch.Tensor,
    component_support: torch.Tensor,
    motion_score: torch.Tensor,
    motion_confidence: torch.Tensor,
    direction_axes: torch.Tensor,
    query_relevance: torch.Tensor,
    atom_weights: torch.Tensor,
    whitening_strength: float,
) -> dict[str, torch.Tensor]:
    query_signature = metric_features.new_empty((metric_features.shape[0], 0))
    if query_relevance.numel() > 0:
        query_signature = query_relevance.transpose(0, 1) * torch.sqrt(
            atom_weights.clamp_min(1e-6)
        ).unsqueeze(0)
        query_signature = F.normalize(query_signature, p=2, dim=-1, eps=1e-6)
    event_signature = torch.stack(
        [novelty, curvature, event, detail, component_support, motion_score],
        dim=1,
    ).float()
    return {
        "appearance": _whiten_features(metric_features, whitening_strength),
        "time": frame_ids.float() / float(max(1, frame_count - 1)),
        "spatial": spatial_coords.float(),
        "scene": scene_ids.long(),
        "event": event_signature,
        "motion": F.normalize(direction_axes.float(), p=2, dim=-1, eps=1e-6),
        "motion_gate": (motion_score * motion_confidence).clamp(0.0, 1.0),
        "motion_score": motion_score.float(),
        "query": query_signature,
    }


def _multi_kernel_block(
    left: torch.Tensor,
    right: torch.Tensor,
    state: dict[str, torch.Tensor],
    weights: dict[str, float],
    config: FlashVidConfig,
) -> torch.Tensor:
    device = left.device
    output = torch.zeros((left.numel(), right.numel()), dtype=torch.float32, device=device)

    if weights["appearance"] > 0.0:
        temperature = max(1e-3, _cfg_float(config, "certv5_appearance_temperature", 0.18))
        cosine = (state["appearance"][left] @ state["appearance"][right].transpose(0, 1)).clamp(-1.0, 1.0)
        output += weights["appearance"] * torch.exp((cosine - 1.0) / temperature)

    if weights["temporal"] > 0.0:
        temperature = max(1e-3, _cfg_float(config, "certv5_temporal_temperature", 0.20))
        distance = torch.abs(state["time"][left].unsqueeze(1) - state["time"][right].unsqueeze(0))
        same_scene = state["scene"][left].unsqueeze(1) == state["scene"][right].unsqueeze(0)
        temporal = torch.exp(-distance / temperature) * (0.35 + 0.65 * same_scene.float())
        output += weights["temporal"] * temporal

    if weights["motion"] > 0.0:
        temperature = max(1e-3, _cfg_float(config, "certv5_motion_temperature", 0.20))
        cosine = (state["motion"][left] @ state["motion"][right].transpose(0, 1)).clamp(-1.0, 1.0)
        directional = torch.exp((cosine - 1.0) / temperature)
        scalar_distance = torch.abs(
            state["motion_score"][left].unsqueeze(1)
            - state["motion_score"][right].unsqueeze(0)
        )
        scalar = torch.exp(-scalar_distance / temperature)
        left_gate = state["motion_gate"][left].clamp(0.0, 1.0)
        right_gate = state["motion_gate"][right].clamp(0.0, 1.0)
        active_gate = torch.sqrt(left_gate.unsqueeze(1) * right_gate.unsqueeze(0))
        static_gate = torch.sqrt((1.0 - left_gate).unsqueeze(1) * (1.0 - right_gate).unsqueeze(0))
        output += weights["motion"] * (active_gate * directional + static_gate * scalar)

    if weights["event"] > 0.0:
        temperature = max(1e-3, _cfg_float(config, "certv5_event_temperature", 0.35))
        distance = torch.cdist(state["event"][left], state["event"][right], p=2)
        output += weights["event"] * torch.exp(-distance.square() / (2.0 * temperature**2))

    if weights["spatial"] > 0.0:
        temperature = max(1e-3, _cfg_float(config, "certv5_spatial_temperature", 0.35))
        distance = torch.cdist(state["spatial"][left], state["spatial"][right], p=1)
        output += weights["spatial"] * torch.exp(-distance / temperature)

    if weights["query"] > 0.0 and state["query"].shape[1] > 0:
        temperature = max(1e-3, _cfg_float(config, "certv5_query_temperature", 0.20))
        cosine = (state["query"][left] @ state["query"][right].transpose(0, 1)).clamp(-1.0, 1.0)
        output += weights["query"] * torch.exp((cosine - 1.0) / temperature)
    return torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)


def _kernel_matrices(
    *,
    candidates: torch.Tensor,
    target_mass: torch.Tensor,
    state: dict[str, torch.Tensor],
    weights: dict[str, float],
    config: FlashVidConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_tokens = int(target_mass.numel())
    chunk_size = max(64, _cfg_int(config, "certv5_kernel_chunk_size", 512))
    kernel = torch.empty(
        (int(candidates.numel()), total_tokens),
        dtype=torch.float32,
        device=candidates.device,
    )
    for start in range(0, total_tokens, chunk_size):
        end = min(total_tokens, start + chunk_size)
        right = torch.arange(start, end, dtype=torch.long, device=candidates.device)
        kernel[:, start:end] = _multi_kernel_block(candidates, right, state, weights, config)
    target_mean = kernel @ target_mass.float()
    candidate_kernel = kernel.index_select(1, candidates)
    candidate_kernel = 0.5 * (candidate_kernel + candidate_kernel.transpose(0, 1))
    return kernel, candidate_kernel, target_mean


def _composable_divide_and_conquer(
    *,
    kernel: torch.Tensor,
    candidate_kernel: torch.Tensor,
    target_mean: torch.Tensor,
    candidates: torch.Tensor,
    target_mass: torch.Tensor,
    quality: torch.Tensor,
    coarse_temporal_ids: torch.Tensor,
    scene_ids: torch.Tensor,
    locked: list[int],
    budget: int,
    merge_multiplier: float,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Compose local scene-time coresets before the global submodular pass."""
    candidate_count = int(candidates.numel())
    limit = min(
        candidate_count,
        max(budget, int(math.ceil(budget * max(1.0, float(merge_multiplier))))),
    )
    if limit >= candidate_count:
        return torch.arange(candidate_count, dtype=torch.long, device=candidates.device), {
            "input_candidates": candidate_count,
            "survivors": candidate_count,
            "shards": 1,
            "covered_shards": 1,
            "bypassed": True,
        }

    coarse_count = int(coarse_temporal_ids.max().item()) + 1
    shard_ids = scene_ids * coarse_count + coarse_temporal_ids
    candidate_shards = shard_ids[candidates]
    shard_values = sorted(int(value) for value in torch.unique(candidate_shards).tolist())
    token_to_position = {
        int(token): position for position, token in enumerate(candidates.detach().cpu().tolist())
    }
    locked_positions = {
        token_to_position[int(token)] for token in locked if int(token) in token_to_position
    }
    survivor_positions: set[int] = set(locked_positions)

    shard_candidates: dict[int, torch.Tensor] = {}
    shard_locked: dict[int, list[int]] = {}
    shard_capacity: dict[int, int] = {}
    shard_mass: dict[int, float] = {}
    extras: dict[int, int] = {shard: 0 for shard in shard_values}
    for shard in shard_values:
        positions = torch.where(candidate_shards == shard)[0]
        shard_candidates[shard] = positions
        local_locked = [position for position in positions.tolist() if position in locked_positions]
        shard_locked[shard] = local_locked
        shard_capacity[shard] = max(0, int(positions.numel()) - len(local_locked))
        members = torch.where(shard_ids == shard)[0]
        shard_mass[shard] = float(target_mass[members].sum().item())

    # Reserve one representative per uncovered shard when the merge budget
    # permits, then use discrete water filling for the remaining capacity.
    # Locked certificates are charged first.
    remaining = max(0, limit - len(survivor_positions))
    uncovered = sorted(
        (
            shard
            for shard in shard_values
            if not shard_locked[shard] and shard_capacity[shard] > 0
        ),
        key=lambda value: (-shard_mass[value], value),
    )
    for shard in uncovered:
        if remaining <= 0:
            break
        extras[shard] += 1
        remaining -= 1
    while remaining > 0:
        active_shards = [shard for shard in shard_values if extras[shard] < shard_capacity[shard]]
        if not active_shards:
            break
        shard = max(
            active_shards,
            key=lambda value: (
                shard_mass[value] / float(extras[value] + len(shard_locked[value]) + 1),
                -value,
            ),
        )
        extras[shard] += 1
        remaining -= 1

    shard_diagnostics: dict[str, object] = {}
    for shard in shard_values:
        positions = shard_candidates[shard]
        chosen = list(shard_locked[shard])
        quota = len(chosen) + extras[shard]
        if quota > len(chosen):
            members = torch.where(shard_ids == shard)[0]
            local_mass = target_mass[members]
            local_mass = local_mass / local_mass.sum().clamp_min(1e-6)
            local_mean = kernel[positions][:, members] @ local_mass
            local_active = torch.ones(positions.numel(), dtype=torch.bool, device=candidates.device)
            redundancy = torch.zeros_like(local_mean)
            if chosen:
                chosen_tensor = torch.tensor(chosen, dtype=torch.long, device=candidates.device)
                local_columns = torch.searchsorted(positions, chosen_tensor)
                local_active[local_columns] = False
                redundancy = candidate_kernel[positions][:, chosen_tensor].max(dim=1).values
            while len(chosen) < quota:
                score = (
                    local_mean
                    - 0.45 * redundancy
                    + 0.04 * quality[candidates[positions]]
                ).masked_fill(~local_active, float("-inf"))
                local_position = int(torch.argmax(score))
                if not math.isfinite(float(score[local_position])):
                    break
                global_position = int(positions[local_position])
                chosen.append(global_position)
                local_active[local_position] = False
                redundancy = torch.maximum(
                    redundancy,
                    candidate_kernel[positions, global_position],
                )
        survivor_positions.update(chosen)
        shard_diagnostics[str(shard)] = {
            "mass": shard_mass[shard],
            "candidates": int(positions.numel()),
            "locked": len(shard_locked[shard]),
            "survivors": len(chosen),
        }

    if len(survivor_positions) < limit:
        global_order = torch.argsort(target_mean, descending=True, stable=True).tolist()
        for position in global_order:
            survivor_positions.add(int(position))
            if len(survivor_positions) >= limit:
                break
    if len(survivor_positions) < budget:
        raise RuntimeError(
            f"divide-and-conquer stage kept {len(survivor_positions)} candidates for budget {budget}"
        )
    survivors = torch.tensor(
        sorted(survivor_positions),
        dtype=torch.long,
        device=candidates.device,
    )
    return survivors, {
        "input_candidates": candidate_count,
        "survivors": int(survivors.numel()),
        "shards": len(shard_values),
        "covered_shards": sum(
            1 for shard in shard_values if len(shard_locked[shard]) + extras[shard] > 0
        ),
        "bypassed": False,
        "details": shard_diagnostics,
    }


def _motion_sectors(
    direction_axes: torch.Tensor,
    motion_score: torch.Tensor,
    motion_confidence: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    velocity = direction_axes[:, :2].float()
    angle = torch.atan2(velocity[:, 0], velocity[:, 1])
    sector = torch.floor((angle + math.pi) * (8.0 / (2.0 * math.pi))).long().remainder(8)
    active = (motion_score >= float(threshold)) & (motion_confidence >= 0.20)
    return torch.where(active, sector, torch.full_like(sector, 8))


def _group_targets(
    group_ids: torch.Tensor,
    target_mass: torch.Tensor,
    budget: int,
    fraction: float,
) -> torch.Tensor:
    group_count = int(group_ids.max().item()) + 1
    mass = torch.zeros(group_count, dtype=torch.float32, device=group_ids.device)
    mass.index_add_(0, group_ids, target_mass.float())
    present = mass > 0.0
    targets = float(budget) * max(0.0, float(fraction)) * mass / mass.sum().clamp_min(1e-6)
    targets[present] = targets[present].clamp_min(1.0)
    return targets


def _constrained_facility_location(
    *,
    kernel: torch.Tensor,
    candidate_kernel: torch.Tensor,
    target_mean: torch.Tensor,
    candidates: torch.Tensor,
    target_mass: torch.Tensor,
    quality: torch.Tensor,
    locked: list[int],
    budget: int,
    fine_temporal_ids: torch.Tensor,
    scene_ids: torch.Tensor,
    spatial_ids: torch.Tensor,
    motion_sectors: torch.Tensor,
    query_relevance: torch.Tensor,
    atom_weights: torch.Tensor,
    query_enabled: bool,
    dual_strength: float,
    mmd_weight: float,
    sample_size: int,
) -> tuple[torch.Tensor, dict[str, object], torch.Tensor]:
    candidate_tokens = [int(token) for token in candidates.detach().cpu().tolist()]
    token_to_position = {token: position for position, token in enumerate(candidate_tokens)}
    selected_positions: list[int] = []
    active = torch.ones(candidates.numel(), dtype=torch.bool, device=candidates.device)
    coverage = torch.zeros(target_mass.numel(), dtype=torch.float32, device=candidates.device)
    repulsion = torch.zeros(candidates.numel(), dtype=torch.float32, device=candidates.device)

    fine_targets = _group_targets(fine_temporal_ids, target_mass, budget, 0.75)
    scene_targets = _group_targets(scene_ids, target_mass, budget, 0.65)
    spatial_targets = _group_targets(spatial_ids, target_mass, budget, 0.30)
    motion_targets = _group_targets(motion_sectors, target_mass, budget, 0.30)
    fine_counts = torch.zeros_like(fine_targets)
    scene_counts = torch.zeros_like(scene_targets)
    spatial_counts = torch.zeros_like(spatial_targets)
    motion_counts = torch.zeros_like(motion_targets)
    query_coverage = (
        torch.zeros(query_relevance.shape[0], dtype=torch.float32, device=candidates.device)
        if query_enabled and query_relevance.numel() > 0
        else torch.empty(0, dtype=torch.float32, device=candidates.device)
    )
    query_targets = (
        0.85 * query_relevance.max(dim=1).values
        if query_coverage.numel() > 0
        else query_coverage
    )

    def add(position: int) -> None:
        if not bool(active[position]):
            return
        token = int(candidates[position])
        selected_positions.append(position)
        active[position] = False
        coverage.copy_(torch.maximum(coverage, kernel[position]))
        repulsion.add_(candidate_kernel[:, position])
        fine_counts[fine_temporal_ids[token]] += 1.0
        scene_counts[scene_ids[token]] += 1.0
        spatial_counts[spatial_ids[token]] += 1.0
        motion_counts[motion_sectors[token]] += 1.0
        if query_coverage.numel() > 0:
            query_coverage.copy_(torch.maximum(query_coverage, query_relevance[:, token]))

    for token in locked:
        if len(selected_positions) >= budget:
            break
        position = token_to_position.get(int(token))
        if position is not None:
            add(position)

    dual_strength = max(0.0, float(dual_strength))
    mmd_weight = max(0.0, float(mmd_weight))
    sample_size = max(8, int(sample_size))
    tie_break = (
        1.0
        - torch.arange(candidates.numel(), device=candidates.device).float()
        / float(max(1, candidates.numel()))
    ) * 1e-7
    step = 0
    while len(selected_positions) < budget:
        available = torch.where(active)[0]
        if available.numel() == 0:
            break
        take = min(sample_size, int(available.numel()))
        priority = (
            target_mean
            - repulsion / float(max(1, len(selected_positions)))
            + tie_break
        )
        priority = priority.masked_fill(~active, float("-inf"))
        top_take = min((take + 1) // 2, int(available.numel()))
        top_positions = torch.topk(priority, k=top_take, largest=True, sorted=True).indices

        rotate_take = take - top_take
        if rotate_take > 0:
            start = (step * max(1, rotate_take) * 17) % int(available.numel())
            order = torch.cat([available[start:], available[:start]])
            rotating = order[~torch.isin(order, top_positions)][:rotate_take]
            shortlist = torch.cat([top_positions, rotating])
        else:
            shortlist = top_positions

        marginal = (
            (kernel[shortlist] - coverage.unsqueeze(0)).clamp_min(0.0)
            * target_mass.unsqueeze(0)
        ).sum(dim=1)
        witness = target_mean[shortlist] - repulsion[shortlist] / float(
            max(1, len(selected_positions))
        )
        witness = _minmax(witness, dim=0)

        fine_deficit = ((fine_targets - fine_counts).clamp_min(0.0) / fine_targets.clamp_min(1.0))
        scene_deficit = ((scene_targets - scene_counts).clamp_min(0.0) / scene_targets.clamp_min(1.0))
        spatial_deficit = ((spatial_targets - spatial_counts).clamp_min(0.0) / spatial_targets.clamp_min(1.0))
        motion_deficit = ((motion_targets - motion_counts).clamp_min(0.0) / motion_targets.clamp_min(1.0))
        tokens = candidates[shortlist]
        dual_bonus = (
            0.34 * fine_deficit[fine_temporal_ids[tokens]]
            + 0.26 * scene_deficit[scene_ids[tokens]]
            + 0.16 * spatial_deficit[spatial_ids[tokens]]
            + 0.24 * motion_deficit[motion_sectors[tokens]]
        )
        if query_coverage.numel() > 0:
            query_deficit = (
                (query_targets - query_coverage).clamp_min(0.0)
                / query_targets.clamp_min(1e-6)
            )
            query_bonus = (
                query_relevance[:, tokens]
                * (query_deficit * atom_weights.clamp_min(1e-6)).unsqueeze(1)
            ).sum(dim=0)
            dual_bonus = dual_bonus + 0.30 * query_bonus

        score = (
            marginal
            + mmd_weight * witness
            + dual_strength * dual_bonus
            + 0.01 * quality[tokens]
            + tie_break[shortlist]
        )
        best = int(shortlist[int(torch.argmax(score))])
        add(best)
        step += 1

    if len(selected_positions) != budget:
        raise RuntimeError(
            f"facility-location selector produced {len(selected_positions)} tokens for budget {budget}"
        )
    positions = torch.tensor(selected_positions, dtype=torch.long, device=candidates.device)
    selected = candidates[positions]
    if not set(int(token) for token in locked).issubset(set(int(token) for token in selected.tolist())):
        raise RuntimeError("facility-location selector removed a locked certificate")
    selected_kernel = candidate_kernel[positions][:, positions]
    mmd_proxy = float(
        (
            selected_kernel.mean()
            - 2.0 * target_mean[positions].mean()
        ).item()
    )
    diagnostics: dict[str, object] = {
        "objective": float(torch.dot(target_mass, coverage).item()),
        "mmd_proxy": mmd_proxy,
        "iterations": step,
        "sample_size": sample_size,
        "locked": len(locked),
        "coverage": {
            "fine_unmet": int((fine_counts + 1e-6 < fine_targets).sum().item()),
            "scene_unmet": int((scene_counts + 1e-6 < scene_targets).sum().item()),
            "spatial_unmet": int((spatial_counts + 1e-6 < spatial_targets).sum().item()),
            "motion_unmet": int((motion_counts + 1e-6 < motion_targets).sum().item()),
            "query_unmet": int((query_coverage + 1e-6 < query_targets).sum().item())
            if query_coverage.numel() > 0
            else 0,
        },
    }
    return selected, diagnostics, target_mean


def _kernel_protected_tokens(
    selected: torch.Tensor,
    candidates: torch.Tensor,
    candidate_kernel: torch.Tensor,
    target_mean: torch.Tensor,
    ratio: float,
) -> set[int]:
    count = min(int(selected.numel()), max(0, int(math.ceil(selected.numel() * max(0.0, ratio)))))
    if count <= 0:
        return set()
    token_to_position = {
        int(token): position for position, token in enumerate(candidates.detach().cpu().tolist())
    }
    positions = torch.tensor(
        [token_to_position[int(token)] for token in selected.detach().cpu().tolist()],
        dtype=torch.long,
        device=selected.device,
    )
    similarity = candidate_kernel[positions][:, positions]
    redundancy = (
        (similarity.sum(dim=1) - torch.diagonal(similarity))
        / float(max(1, selected.numel() - 1))
    )
    witness = target_mean[positions] - 0.50 * redundancy
    protected_positions = torch.argsort(witness, descending=True, stable=True)[:count]
    return {int(selected[position]) for position in protected_positions}


def _build_plan(
    *,
    selected: torch.Tensor,
    metric_features: torch.Tensor,
    demand_weight: torch.Tensor,
    scene_ids: torch.Tensor,
    component_ids: torch.Tensor,
    motion_score: torch.Tensor,
    fusion_alpha: float,
    temperature: float,
    protected_tokens: set[int],
    motion_fusion_threshold: float,
    transport_steps: int,
    transport_balance: float,
) -> tuple[CertVidPlan, dict[str, float]]:
    total_tokens = int(metric_features.shape[0])
    budget = int(selected.numel())
    similarity = metric_features @ metric_features[selected].transpose(0, 1)
    same_scene = scene_ids.unsqueeze(1) == scene_ids[selected].unsqueeze(0)
    similarity = similarity.masked_fill(~same_scene, -2.0)
    same_component = component_ids.unsqueeze(1) == component_ids[selected].unsqueeze(0)
    similarity = similarity + 0.08 * same_component.float()

    # A small transport dual discourages all residual mass from collapsing onto
    # one visually generic anchor. It only changes fusion assignments, not the
    # selected raw-token coreset.
    neighbor_count = min(2, budget)
    temperature = max(1e-4, float(temperature))
    balance = max(0.0, float(transport_balance))
    dual = torch.zeros(budget, dtype=torch.float32, device=selected.device)
    normalized_demand = demand_weight.float() / demand_weight.float().sum().clamp_min(1e-6)
    target_load = torch.full_like(dual, 1.0 / float(max(1, budget)))
    for _ in range(max(0, int(transport_steps))):
        values, assignment = torch.topk(
            similarity - dual.unsqueeze(0),
            k=neighbor_count,
            dim=1,
            largest=True,
        )
        weights = torch.softmax(values.float() / temperature, dim=1)
        load = torch.zeros_like(dual)
        for neighbor in range(neighbor_count):
            load.index_add_(
                0,
                assignment[:, neighbor],
                normalized_demand * weights[:, neighbor],
            )
        dual.add_(
            balance
            * torch.log((load + 1e-6) / (target_load + 1e-6))
        )
        dual.sub_(dual.mean())

    values, assignment = torch.topk(
        similarity - dual.unsqueeze(0),
        k=neighbor_count,
        dim=1,
        largest=True,
    )
    weights = torch.softmax(values.float() / temperature, dim=1)

    positions = torch.arange(budget, dtype=torch.long, device=selected.device)
    assignment[selected, 0] = positions
    weights[selected] = 0.0
    weights[selected, 0] = 1.0

    load = torch.zeros(budget, dtype=torch.float32, device=selected.device)
    for neighbor in range(neighbor_count):
        load.index_add_(
            0,
            assignment[:, neighbor],
            normalized_demand * weights[:, neighbor],
        )
    source_mass = (0.5 + 0.5 * demand_weight * float(total_tokens)).clamp(0.25, 2.0)
    base_alpha = min(max(float(fusion_alpha), 0.0), 0.75)
    alpha = base_alpha * (1.0 - motion_score[selected].clamp(0.0, 1.0))
    alpha = alpha.to(dtype=torch.float32)
    alpha[motion_score[selected] >= float(motion_fusion_threshold)] = 0.0
    if protected_tokens:
        protected = torch.tensor(sorted(protected_tokens), dtype=torch.long, device=selected.device)
        alpha[torch.isin(selected, protected)] = 0.0
    plan = CertVidPlan(
        anchor_indices=selected,
        assignment_indices=assignment,
        assignment_weights=weights,
        source_mass=source_mass,
        fusion_alpha=alpha,
        raw_token_count=total_tokens,
    )
    diagnostics = {
        "steps": float(max(0, int(transport_steps))),
        "load_cv": float((load.std(unbiased=False) / load.mean().clamp_min(1e-6)).item()),
        "min_load": float(load.min().item()),
        "max_load": float(load.max().item()),
    }
    return plan, diagnostics


def _publish_diagnostics(config: FlashVidConfig, diagnostics: dict[str, object]) -> None:
    setattr(config, "last_certv5_diagnostics", diagnostics)
    budget = diagnostics["budget"]
    certificates = diagnostics["certificates"]
    facility = diagnostics["facility_location"]
    attention = diagnostics["attention"]
    scene_pyramid = diagnostics["scene_pyramid"]
    router = diagnostics["router"]
    divide = diagnostics["divide_and_conquer"]
    transport = diagnostics["transport"]
    setattr(config, "last_certv5_target_tokens", float(budget["target_tokens"]))
    setattr(config, "last_certv5_nominal_retention", float(budget["nominal_retention"]))
    setattr(config, "last_certv5_outer_retention", float(budget["outer_retention"]))
    setattr(config, "last_certv5_post_inner_retention", float(budget["post_inner_retention"]))
    setattr(config, "last_certv5_average_layer_multiplier", float(budget["average_layer_multiplier"]))
    setattr(config, "last_certv5_post_inner_tokens", float(budget["post_inner_tokens"]))
    setattr(config, "last_certv5_average_layer_tokens", float(budget["average_layer_tokens"]))
    setattr(config, "last_certv5_certificate_count", float(certificates["admitted_unique"]))
    setattr(config, "last_certv5_candidate_tokens", float(diagnostics["candidate_count"]))
    setattr(config, "last_certv5_component_count", float(diagnostics["component_count"]))
    setattr(config, "last_certv5_query_confidence", float(diagnostics["query_confidence"]))
    setattr(config, "last_certv5_facility_objective", float(facility["objective"]))
    setattr(config, "last_certv5_mmd_proxy", float(facility["mmd_proxy"]))
    setattr(config, "last_certv5_facility_iterations", float(facility["iterations"]))
    setattr(config, "last_certv5_divide_survivors", float(divide["survivors"]))
    setattr(config, "last_certv5_transport_load_cv", float(transport["load_cv"]))
    setattr(
        config,
        "last_certv5_kernel_protected_count",
        float(diagnostics["kernel_protected_count"]),
    )
    setattr(config, "last_certv5_attention_used", float(bool(attention["used"])))
    setattr(config, "last_certv5_scene_count", float(scene_pyramid["scene_count"]))
    setattr(config, "last_certv5_motion_activity", float(router["motion_activity"]))
    setattr(config, "last_certv5_event_activity", float(router["event_activity"]))
    setattr(config, "last_certv5_motion_pair_count", float(diagnostics["motion_pair_count"]))
    if bool(getattr(config, "certv5_debug", False)):
        print(f"[certvid-v5] {diagnostics}")


def certvid_v5_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
    question_features: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a composable, quality-weighted facility-location visual coreset."""
    if video_features.ndim != 3:
        raise ValueError(f"expected video_features [T, HW, D], got {tuple(video_features.shape)}")
    frame_count, tokens_per_frame, _ = video_features.shape
    total_tokens = int(frame_count * tokens_per_frame)
    budget, budget_diagnostics = _resolve_budget(flashvid_config, total_tokens)
    attention, attention_diagnostics = _validated_attention(
        cls_attention,
        frame_count,
        tokens_per_frame,
        flashvid_config,
    )
    flat_features = video_features.reshape(total_tokens, -1)
    query_mode = str(
        getattr(flashvid_config, "certv5_query_mode", "certificates_and_kernel")
    ).strip().lower()
    if query_mode not in _QUERY_MODES:
        raise ValueError(f"unsupported certv5_query_mode={query_mode!r}")

    if budget >= total_tokens:
        plan = _identity_plan(total_tokens, video_features.device)
        output = flat_features
        diagnostics: dict[str, object] = {
            "budget": budget_diagnostics,
            "attention": attention_diagnostics,
            "query_mode": query_mode,
            "query_confidence": 0.0,
            "scene_pyramid": {"boundaries": [], "scene_count": 1, "fine_count": 1, "coarse_count": 1},
            "router": {"motion_activity": 0.0, "event_activity": 0.0, "router_strength": 0.0},
            "certificates": {
                "cap": int(math.floor(total_tokens * min(1.0, max(0.0, _cfg_float(flashvid_config, "certv5_certificate_budget_ratio", 0.28))))),
                "admitted_unique": 0,
                "ratio": 0.0,
                "requested": 0,
                "deduplicated": 0,
                "categories": {},
            },
            "candidates": {"limit": total_tokens, "locked": 0, "sources": {}},
            "candidate_count": total_tokens,
            "component_count": 0,
            "motion_pair_count": 0,
            "target_measure": {
                "entropy": float(math.log(max(1, total_tokens))),
                "effective_support": float(total_tokens),
                "min_frame_mass": 1.0 / float(max(1, frame_count)),
                "max_frame_mass": 1.0 / float(max(1, frame_count)),
            },
            "kernel_weights": {
                "appearance": 1.0,
                "temporal": 0.0,
                "motion": 0.0,
                "event": 0.0,
                "spatial": 0.0,
                "query": 0.0,
            },
            "divide_and_conquer": {
                "input_candidates": total_tokens,
                "survivors": total_tokens,
                "shards": 1,
                "covered_shards": 1,
                "bypassed": True,
            },
            "facility_location": {
                "objective": 1.0,
                "mmd_proxy": -1.0,
                "iterations": 0,
                "sample_size": 0,
                "locked": 0,
                "coverage": {},
            },
            "kernel_protected_count": 0,
            "fusion_protected_count": 0,
            "transport": {
                "steps": 0.0,
                "load_cv": 0.0,
                "min_load": 1.0 / float(max(1, total_tokens)),
                "max_load": 1.0 / float(max(1, total_tokens)),
            },
        }
    else:
        metric_dim = max(32, _cfg_int(flashvid_config, "certv5_metric_dim", 96))
        metric_flat = _metric_features(video_features, metric_dim)
        metric_frames = metric_flat.view(frame_count, tokens_per_frame, -1)
        height, width = _grid_hw(tokens_per_frame, flashvid_config)
        spatial_bins = max(1, _cfg_int(flashvid_config, "certv5_spatial_bins", 3))
        coords, frame_spatial_ids = _spatial_layout(
            tokens_per_frame,
            height,
            width,
            spatial_bins,
            video_features.device,
        )
        frame_event, _, novelty_2d, curvature_2d, matches = _trajectory_signals(
            metric_frames,
            coords,
            _cfg_float(flashvid_config, "certv5_spatial_penalty", 0.08),
        )
        component_ids_cpu, component_sizes_cpu = _build_components(
            frame_count,
            tokens_per_frame,
            frame_event,
            matches,
            _cfg_float(flashvid_config, "certv5_track_threshold", 0.82),
        )
        component_ids = component_ids_cpu.to(video_features.device)
        component_sizes = component_sizes_cpu.to(video_features.device)
        frame_ids = torch.arange(frame_count, device=video_features.device).repeat_interleave(tokens_per_frame)
        component_value = _component_support(
            metric_flat,
            component_ids,
            component_sizes,
            frame_ids,
            frame_count,
        )
        frame_scene_ids, frame_fine_ids, frame_coarse_ids, scene_diagnostics = _scene_pyramid(
            frame_event=frame_event,
            frame_count=frame_count,
            max_scenes=_cfg_int(flashvid_config, "certv5_max_scenes", 8),
            boundary_threshold=_cfg_float(flashvid_config, "certv5_scene_threshold", 0.58),
            min_scene_frames=_cfg_int(flashvid_config, "certv5_min_scene_frames", 2),
            fine_bins=_cfg_int(flashvid_config, "certv5_temporal_bins", 12),
            coarse_bins=_cfg_int(flashvid_config, "certv5_coarse_bins", 4),
        )
        scene_ids = frame_scene_ids.repeat_interleave(tokens_per_frame)
        fine_temporal_ids = frame_fine_ids.repeat_interleave(tokens_per_frame)
        coarse_temporal_ids = frame_coarse_ids.repeat_interleave(tokens_per_frame)
        scene_count = int(scene_diagnostics["scene_count"])
        fine_temporal_count = int(scene_diagnostics["fine_count"])
        coarse_temporal_count = int(scene_diagnostics["coarse_count"])
        spatial_ids = frame_spatial_ids.repeat(frame_count)
        spatial_count = spatial_bins * spatial_bins

        novelty = novelty_2d.reshape(-1)
        curvature = curvature_2d.reshape(-1)
        detail = _local_detail(video_features, height, width).reshape(-1)
        event = frame_event.repeat_interleave(tokens_per_frame)
        motion_score, motion_confidence, direction_axes, previous_global = _directed_motion_signals(
            metric_frames=metric_frames,
            coords=coords,
            matches=matches,
            novelty=novelty_2d,
            curvature=curvature_2d,
        )
        certificate_shares, candidate_shares, router_diagnostics = _adaptive_source_shares(
            motion_score=motion_score,
            frame_event=frame_event,
            scene_count=scene_count,
            router_strength=_cfg_float(flashvid_config, "certv5_router_strength", 0.65),
        )
        atoms = _question_atoms(
            question_features,
            max(0, _cfg_int(flashvid_config, "certv5_query_atoms", 8)),
            metric_dim,
        ).to(video_features.device)
        query_relevance, atom_weights, query_confidence = _question_relevance(atoms, metric_flat)
        query_score = (
            (query_relevance * atom_weights.unsqueeze(1)).sum(dim=0)
            if query_relevance.numel() > 0
            else torch.zeros_like(attention)
        )

        motion_weight = 0.12 + 0.10 * float(router_diagnostics["motion_activity"])
        quality_weight_sum = 0.18 + 0.16 + 0.10 + 0.10 + 0.10 + 0.14 + motion_weight
        visual_quality = _minmax(
            (
                0.18 * attention
                + 0.16 * novelty
                + 0.10 * curvature
                + 0.10 * event
                + 0.10 * detail
                + 0.14 * component_value
                + motion_weight * motion_score
            )
            / quality_weight_sum,
            dim=0,
        )
        query_scalar_enabled = query_mode != "off" and query_relevance.numel() > 0
        query_weight = (
            min(
                0.20,
                max(0.0, _cfg_float(flashvid_config, "certv5_query_weight", 0.12))
                * float(query_confidence),
            )
            if query_scalar_enabled
            else 0.0
        )
        # Query evidence is represented by certificates, target mass, and the
        # query kernel. Keeping it out of scalar quality avoids counting the
        # same signal repeatedly.
        quality = visual_quality
        event_score = _minmax(
            0.24 * novelty
            + 0.18 * curvature
            + 0.18 * event
            + 0.10 * detail
            + 0.18 * motion_score
            + 0.12 * component_value,
            dim=0,
        )
        demand_weight, target_diagnostics = _balanced_target_mass(
            quality=quality,
            event_score=event_score,
            component_support=component_value,
            query_score=query_score,
            frame_ids=frame_ids,
            frame_event=frame_event,
            query_weight=query_weight,
            quality_floor=_cfg_float(flashvid_config, "certv5_quality_floor", 0.18),
        )

        requests = _build_certificate_requests(
            quality=quality,
            event_score=event_score,
            motion_score=motion_score,
            motion_confidence=motion_confidence,
            previous_global=previous_global,
            frame_ids=frame_ids,
            fine_temporal_ids=fine_temporal_ids,
            coarse_temporal_ids=coarse_temporal_ids,
            scene_ids=scene_ids,
            spatial_ids=spatial_ids,
            component_ids=component_ids,
            query_relevance=query_relevance,
            atom_weights=atom_weights,
            query_confidence=query_confidence,
            query_mode=query_mode,
            frame_count=frame_count,
            scene_count=scene_count,
            spatial_count=spatial_count,
            frame_coverage_ratio=_cfg_float(flashvid_config, "certv5_frame_coverage_ratio", 0.75),
            query_threshold=_cfg_float(flashvid_config, "certv5_query_threshold", 0.10),
            query_per_atom=_cfg_int(flashvid_config, "certv5_query_per_atom", 1),
            motion_threshold=_cfg_float(flashvid_config, "certv5_motion_threshold", 0.42),
            motion_confidence_threshold=_cfg_float(
                flashvid_config,
                "certv5_motion_confidence_threshold",
                0.35,
            ),
        )
        locked, certificate_diagnostics, protected_by_category = _admit_certificates(
            requests,
            budget,
            _cfg_float(flashvid_config, "certv5_certificate_budget_ratio", 0.28),
            certificate_shares,
        )
        candidates, candidate_diagnostics = _candidate_pool(
            budget=budget,
            quality=quality,
            event_score=event_score,
            motion_score=motion_score,
            component_ids=component_ids,
            fine_temporal_ids=fine_temporal_ids,
            scene_ids=scene_ids,
            spatial_ids=spatial_ids,
            query_relevance=query_relevance,
            atom_weights=atom_weights,
            query_mode=query_mode,
            requests=requests,
            locked=locked,
            multiplier=_cfg_float(flashvid_config, "certv5_candidate_multiplier", 2.5),
            shares=candidate_shares,
        )

        kernel_query_enabled = _query_uses_kernel(query_mode) and query_relevance.numel() > 0
        kernel_weights = _kernel_weights(
            flashvid_config,
            router_diagnostics,
            query_confidence,
            kernel_query_enabled,
        )
        global_coords = coords.repeat(frame_count, 1)
        kernel_state = _prepare_kernel_state(
            metric_features=metric_flat,
            frame_ids=frame_ids,
            frame_count=frame_count,
            spatial_coords=global_coords,
            scene_ids=scene_ids,
            novelty=novelty,
            curvature=curvature,
            event=event,
            detail=detail,
            component_support=component_value,
            motion_score=motion_score,
            motion_confidence=motion_confidence,
            direction_axes=direction_axes,
            query_relevance=query_relevance,
            atom_weights=atom_weights,
            whitening_strength=_cfg_float(flashvid_config, "certv5_whitening_strength", 0.25),
        )
        kernel, candidate_kernel, target_mean = _kernel_matrices(
            candidates=candidates,
            target_mass=demand_weight,
            state=kernel_state,
            weights=kernel_weights,
            config=flashvid_config,
        )

        survivor_positions, divide_diagnostics = _composable_divide_and_conquer(
            kernel=kernel,
            candidate_kernel=candidate_kernel,
            target_mean=target_mean,
            candidates=candidates,
            target_mass=demand_weight,
            quality=quality,
            coarse_temporal_ids=coarse_temporal_ids,
            scene_ids=scene_ids,
            locked=locked,
            budget=budget,
            merge_multiplier=_cfg_float(flashvid_config, "certv5_merge_multiplier", 2.0),
        )
        candidates = candidates[survivor_positions]
        kernel = kernel[survivor_positions]
        candidate_kernel = candidate_kernel[survivor_positions][:, survivor_positions]
        target_mean = target_mean[survivor_positions]

        motion_sectors = _motion_sectors(
            direction_axes,
            motion_score,
            motion_confidence,
            _cfg_float(flashvid_config, "certv5_motion_sector_threshold", 0.24),
        )
        selected, facility_diagnostics, target_mean = _constrained_facility_location(
            kernel=kernel,
            candidate_kernel=candidate_kernel,
            target_mean=target_mean,
            candidates=candidates,
            target_mass=demand_weight,
            quality=quality,
            locked=locked,
            budget=budget,
            fine_temporal_ids=fine_temporal_ids,
            scene_ids=scene_ids,
            spatial_ids=spatial_ids,
            motion_sectors=motion_sectors,
            query_relevance=query_relevance,
            atom_weights=atom_weights,
            query_enabled=kernel_query_enabled,
            dual_strength=_cfg_float(flashvid_config, "certv5_dual_strength", 0.20),
            mmd_weight=_cfg_float(flashvid_config, "certv5_mmd_weight", 0.10),
            sample_size=_cfg_int(flashvid_config, "certv5_greedy_sample_size", 48),
        )
        selected = torch.sort(selected).values
        kernel_protected = _kernel_protected_tokens(
            selected=selected,
            candidates=candidates,
            candidate_kernel=candidate_kernel,
            target_mean=target_mean,
            ratio=_cfg_float(flashvid_config, "certv5_kernel_protect_ratio", 0.12),
        )
        protected_tokens = set(locked) | kernel_protected
        plan, transport_diagnostics = _build_plan(
            selected=selected,
            metric_features=metric_flat,
            demand_weight=demand_weight,
            scene_ids=scene_ids,
            component_ids=component_ids,
            motion_score=motion_score,
            fusion_alpha=_cfg_float(flashvid_config, "certv5_fusion_alpha", 0.04),
            temperature=_cfg_float(flashvid_config, "certv5_assignment_temperature", 0.07),
            protected_tokens=protected_tokens,
            motion_fusion_threshold=_cfg_float(
                flashvid_config,
                "certv5_motion_fusion_threshold",
                0.45,
            ),
            transport_steps=_cfg_int(flashvid_config, "certv5_transport_steps", 4),
            transport_balance=_cfg_float(flashvid_config, "certv5_transport_balance", 0.20),
        )
        output = apply_certvid_plan(flat_features, plan)
        diagnostics = {
            "budget": budget_diagnostics,
            "attention": attention_diagnostics,
            "query_mode": query_mode,
            "query_confidence": float(query_confidence),
            "scene_pyramid": scene_diagnostics,
            "router": router_diagnostics,
            "certificates": certificate_diagnostics,
            "candidates": candidate_diagnostics,
            "candidate_count": int(candidates.numel()),
            "component_count": int(component_sizes.numel()),
            "motion_pair_count": sum(
                1 for request in requests if request.category == "motion"
            ),
            "target_measure": target_diagnostics,
            "kernel_weights": kernel_weights,
            "divide_and_conquer": divide_diagnostics,
            "facility_location": facility_diagnostics,
            "kernel_protected_count": len(kernel_protected),
            "fusion_protected_count": len(protected_tokens),
            "transport": transport_diagnostics,
        }

    setattr(flashvid_config, "_certvid_plan", plan)
    flashvid_config.vision_token_length = int(output.shape[0])
    flashvid_config.visual_token_length = int(output.shape[0])
    flashvid_config.llm_token_length = None
    setattr(flashvid_config, "last_adapter_variant", "certvid_v5")
    setattr(flashvid_config, "last_adapter_raw_tokens", float(total_tokens))
    setattr(flashvid_config, "last_adapter_output_tokens", float(output.shape[0]))
    _publish_diagnostics(flashvid_config, diagnostics)
    return output, plan.anchor_indices

