from __future__ import annotations

from typing import Tuple

import torch
from torch.nn import functional as F

from .configuration_flashvid import FlashVidConfig


def _cfg_float(config: FlashVidConfig, name: str, default: float) -> float:
    value = getattr(config, name, None)
    if value is None:
        return float(default)
    return float(value)


def _cfg_int(config: FlashVidConfig, name: str, default: int) -> int:
    value = getattr(config, name, None)
    if value is None:
        return int(default)
    return int(value)


def _cfg_bool(config: FlashVidConfig, name: str, default: bool) -> bool:
    value = getattr(config, name, None)
    if value is None:
        return bool(default)
    return bool(value)


def _minmax_flat(x: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    lo = x.min()
    hi = x.max()
    return ((x - lo) / (hi - lo + eps)).clamp(0.0, 1.0)


def _safe_topk(values: torch.Tensor, k: int, *, largest: bool = True) -> torch.Tensor:
    k = min(max(0, int(k)), int(values.numel()))
    if k <= 0:
        return torch.empty((0,), dtype=torch.long, device=values.device)
    return torch.topk(values, k=k, largest=largest).indices


def _reset_pivot_metrics(config: FlashVidConfig) -> None:
    defaults = {
        "last_pivot_target_tokens": 0.0,
        "last_pivot_selected_tokens": 0.0,
        "last_pivot_candidate_count": 0.0,
        "last_pivot_use_fuse": 0.0,
        "last_pivot_budget_scale": 0.0,
        "last_pivot_avg_cluster_size": 0.0,
        "last_pivot_max_cluster_size": 0.0,
        "last_pivot_coverage_mean": None,
        "last_pivot_selected_utility_mean": None,
        "last_pivot_bridge_mean": None,
        "last_pivot_surprise_mean": None,
        "last_pivot_background_mean": None,
    }
    for key, value in defaults.items():
        setattr(config, key, value)
    setattr(config, "_pivot_segments", 0.0)
    for key in defaults:
        setattr(config, f"_{key[5:]}_sum", 0.0)
    setattr(config, "_pivot_coverage_count", 0.0)
    setattr(config, "_pivot_utility_count", 0.0)
    setattr(config, "_pivot_bridge_count", 0.0)
    setattr(config, "_pivot_surprise_count", 0.0)
    setattr(config, "_pivot_background_count", 0.0)


def _accumulate_pivot_metrics(
    config: FlashVidConfig,
    *,
    target_tokens: int,
    selected_tokens: int,
    candidate_count: int,
    use_fuse: bool,
    budget_scale: float,
    avg_cluster_size: float,
    max_cluster_size: int,
    coverage_mean: float | None,
    selected_utility_mean: float | None,
    bridge_mean: float | None,
    surprise_mean: float | None,
    background_mean: float | None,
) -> None:
    if not hasattr(config, "_pivot_segments"):
        _reset_pivot_metrics(config)
    setattr(config, "_pivot_segments", float(getattr(config, "_pivot_segments", 0.0)) + 1.0)
    simple_values = {
        "pivot_target_tokens": target_tokens,
        "pivot_selected_tokens": selected_tokens,
        "pivot_candidate_count": candidate_count,
        "pivot_use_fuse": float(bool(use_fuse)),
        "pivot_budget_scale": budget_scale,
        "pivot_avg_cluster_size": avg_cluster_size,
        "pivot_max_cluster_size": max_cluster_size,
    }
    for key, value in simple_values.items():
        attr = f"_{key}_sum"
        setattr(config, attr, float(getattr(config, attr, 0.0)) + float(value))

    optional_values = {
        "pivot_coverage": coverage_mean,
        "pivot_utility": selected_utility_mean,
        "pivot_bridge": bridge_mean,
        "pivot_surprise": surprise_mean,
        "pivot_background": background_mean,
    }
    for key, value in optional_values.items():
        if value is None:
            continue
        setattr(config, f"_{key}_sum", float(getattr(config, f"_{key}_sum", 0.0)) + float(value))
        setattr(config, f"_{key}_count", float(getattr(config, f"_{key}_count", 0.0)) + 1.0)

    segments = max(1.0, float(getattr(config, "_pivot_segments", 1.0)))
    for key in simple_values:
        setattr(config, f"last_{key}", float(getattr(config, f"_{key}_sum", 0.0)) / segments)
    for key, last_key in (
        ("pivot_coverage", "last_pivot_coverage_mean"),
        ("pivot_utility", "last_pivot_selected_utility_mean"),
        ("pivot_bridge", "last_pivot_bridge_mean"),
        ("pivot_surprise", "last_pivot_surprise_mean"),
        ("pivot_background", "last_pivot_background_mean"),
    ):
        count = float(getattr(config, f"_{key}_count", 0.0))
        setattr(config, last_key, float(getattr(config, f"_{key}_sum", 0.0)) / count if count > 0 else None)


def _compute_pivot_utility(
    features: torch.Tensor,
    cls_attention: torch.Tensor,
    config: FlashVidConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_frames, num_visual_tokens, _ = features.shape
    normed = F.normalize(features.float(), p=2, dim=-1, eps=1.0e-6)

    if cls_attention is None or cls_attention.numel() == 0:
        attention = torch.ones((num_frames, num_visual_tokens), dtype=torch.float32, device=features.device)
    else:
        attention = cls_attention.float().reshape(num_frames, num_visual_tokens)
    attention = _minmax_flat(attention)

    b_prev = torch.zeros((num_frames, num_visual_tokens), dtype=torch.float32, device=features.device)
    b_next = torch.zeros_like(b_prev)
    if num_frames > 1:
        sim = torch.bmm(normed[1:], normed[:-1].transpose(1, 2)).clamp_min(0.0)
        b_prev[1:] = sim.max(dim=-1).values
        b_next[:-1] = sim.max(dim=1).values
    bridge = _minmax_flat(torch.sqrt((b_prev + 1.0e-6) * (b_next + 1.0e-6)))

    if num_visual_tokens > 1:
        same = torch.bmm(normed, normed.transpose(1, 2)).clamp_min(0.0)
        eye = torch.eye(num_visual_tokens, dtype=torch.bool, device=features.device).unsqueeze(0)
        same = same.masked_fill(eye, 0.0)
        k = min(max(1, _cfg_int(config, "pivot_surprise_topk", 8)), num_visual_tokens - 1)
        local_mean = same.topk(k=k, dim=-1).values.mean(dim=-1)
        surprise = (1.0 - local_mean).clamp(0.0, 1.0)
    else:
        surprise = torch.zeros_like(attention)
    gated_surprise = _minmax_flat(surprise * torch.sqrt((attention + bridge + 1.0e-6).clamp_min(1.0e-6)))

    persistence = 0.5 * (b_prev + b_next)
    background = _minmax_flat(persistence * (1.0 - gated_surprise) * (1.0 - attention))

    alpha = _cfg_float(config, "pivot_alpha", 0.35)
    beta = _cfg_float(config, "pivot_beta", 0.25)
    gamma = _cfg_float(config, "pivot_gamma", 0.30)
    delta = _cfg_float(config, "pivot_delta", 0.10)
    utility = _minmax_flat(alpha * attention + beta * bridge + gamma * gated_surprise - delta * background)
    evidence_weight = (utility + 1.0e-6).clamp_min(1.0e-6)
    return utility, evidence_weight, bridge, gated_surprise, background, normed


def _select_pivots(
    normed_flat: torch.Tensor,
    utility: torch.Tensor,
    evidence_weight: torch.Tensor,
    frame_ids: torch.Tensor,
    target_tokens: int,
    num_frames: int,
    config: FlashVidConfig,
) -> tuple[torch.Tensor, int]:
    total_tokens = int(normed_flat.shape[0])
    target_tokens = min(max(1, int(target_tokens)), total_tokens)
    if target_tokens >= total_tokens:
        all_idx = torch.arange(total_tokens, dtype=torch.long, device=normed_flat.device)
        return all_idx, total_tokens

    selected: list[int] = []
    selected_mask = torch.zeros((total_tokens,), dtype=torch.bool, device=normed_flat.device)
    min_per_frame = max(0, _cfg_int(config, "pivot_min_keep_per_frame", 0))
    if min_per_frame > 0:
        for frame_idx in range(num_frames):
            frame_positions = torch.where(frame_ids == frame_idx)[0]
            if frame_positions.numel() == 0:
                continue
            take = min(min_per_frame, target_tokens - len(selected), int(frame_positions.numel()))
            if take <= 0:
                break
            local = _safe_topk(utility[frame_positions], take, largest=True)
            chosen = frame_positions[local]
            for idx in chosen.tolist():
                if not bool(selected_mask[idx]):
                    selected.append(int(idx))
                    selected_mask[idx] = True

    candidate_factor = max(1.0, _cfg_float(config, "pivot_candidate_factor", 4.0))
    max_candidates = max(target_tokens, _cfg_int(config, "pivot_max_candidates", 2048))
    candidate_count = min(total_tokens, max(target_tokens, min(max_candidates, int(candidate_factor * target_tokens))))
    candidates = _safe_topk(utility, candidate_count, largest=True)
    if candidates.numel() == 0:
        candidates = torch.arange(total_tokens, dtype=torch.long, device=normed_flat.device)
    candidate_mask = selected_mask[candidates].clone()
    candidate_normed = normed_flat[candidates]
    coverage = torch.zeros((int(candidates.numel()),), dtype=torch.float32, device=normed_flat.device)
    lam = min(max(_cfg_float(config, "pivot_lambda", 0.40), 0.0), 1.0)
    utility_c = utility[candidates]
    evidence_c = evidence_weight[candidates]

    while len(selected) < target_tokens:
        gain = lam * utility_c + (1.0 - lam) * evidence_c * (1.0 - coverage).clamp_min(0.0)
        gain = gain.masked_fill(candidate_mask, -float("inf"))
        best_pos = int(torch.argmax(gain).item())
        if not torch.isfinite(gain[best_pos]):
            remaining = torch.where(~selected_mask)[0]
            if remaining.numel() == 0:
                break
            best_idx = int(remaining[torch.argmax(utility[remaining])].item())
        else:
            best_idx = int(candidates[best_pos].item())
            candidate_mask[best_pos] = True
        if bool(selected_mask[best_idx]):
            break
        selected.append(best_idx)
        selected_mask[best_idx] = True
        best_vec = normed_flat[best_idx]
        coverage = torch.maximum(coverage, torch.matmul(candidate_normed, best_vec).clamp_min(0.0))

    if len(selected) < target_tokens:
        remaining = torch.where(~selected_mask)[0]
        fill = _safe_topk(utility[remaining], target_tokens - len(selected), largest=True)
        selected.extend(int(x) for x in remaining[fill].tolist())
    selected_tensor = torch.tensor(selected[:target_tokens], dtype=torch.long, device=normed_flat.device)
    return selected_tensor, int(candidates.numel())


def _fuse_to_pivots(
    features_flat: torch.Tensor,
    normed_flat: torch.Tensor,
    selected: torch.Tensor,
    utility: torch.Tensor,
    evidence_weight: torch.Tensor,
    config: FlashVidConfig,
) -> tuple[torch.Tensor, torch.Tensor, float, int, float]:
    pivots = features_flat[selected].float()
    pivot_normed = normed_flat[selected]
    sim = torch.matmul(normed_flat, pivot_normed.transpose(0, 1)).clamp_min(0.0)
    assignment_sim, assignment = sim.max(dim=1)
    cluster_counts = torch.bincount(assignment, minlength=int(selected.numel())).float()

    if not _cfg_bool(config, "pivot_use_fuse", True):
        return features_flat[selected], assignment_sim, float(cluster_counts.mean().item()), int(cluster_counts.max().item()), float(assignment_sim.mean().item())

    tau = max(0.0, _cfg_float(config, "pivot_tau", 1.0))
    weights = evidence_weight * assignment_sim.clamp_min(0.0).pow(tau)
    weights[selected] = 0.0
    mu0 = max(1.0e-6, _cfg_float(config, "pivot_mu0", 1.0))
    mu = mu0 * (1.0 + utility[selected]).float()
    numerator = mu.unsqueeze(-1) * pivots
    denominator = mu.clone()
    feat_dim = int(features_flat.shape[-1])
    numerator.scatter_add_(0, assignment.unsqueeze(-1).expand(-1, feat_dim), weights.unsqueeze(-1) * features_flat.float())
    denominator.scatter_add_(0, assignment, weights.float())
    fused = numerator / denominator.unsqueeze(-1).clamp_min(1.0e-6)
    return fused.to(dtype=features_flat.dtype), assignment_sim, float(cluster_counts.mean().item()), int(cluster_counts.max().item()), float(assignment_sim.mean().item())


def pivotfuse_segment_compression(
    segment_features: torch.Tensor,
    segment_global_indices: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evidence-pivot token selection plus residual-preserving fusion.

    This is a before-LLM full segment compressor. It keeps the same outer
    FlashVID retention budget, selects evidence pivots with an approximate
    coverage objective, and fuses non-pivot residual information into pivots.
    """
    num_frames, num_visual_tokens, feat_dim = segment_features.shape
    total_tokens = int(num_frames * num_visual_tokens)
    if total_tokens <= 0:
        return (
            torch.empty((0, feat_dim), dtype=segment_features.dtype, device=segment_features.device),
            torch.empty((0,), dtype=torch.long, device=segment_features.device),
        )

    per_frame_budget = int((flashvid_config.num_attn_div_tokens or 0) + (flashvid_config.num_sttm_tokens or 0))
    if per_frame_budget <= 0:
        per_frame_budget = max(1, int(round(num_visual_tokens * float(flashvid_config.retention_ratio) * float(flashvid_config.expansion))))
    budget_scale = max(0.01, _cfg_float(flashvid_config, "pivot_budget_scale", 1.0))
    target_tokens = min(total_tokens, max(1, int(round(per_frame_budget * num_frames * budget_scale))))

    utility_map, evidence_map, bridge_map, surprise_map, background_map, normed = _compute_pivot_utility(
        segment_features,
        cls_attention,
        flashvid_config,
    )
    features_flat = segment_features.reshape(total_tokens, feat_dim)
    globals_flat = segment_global_indices.reshape(total_tokens)
    normed_flat = normed.reshape(total_tokens, feat_dim)
    utility = utility_map.reshape(total_tokens)
    evidence_weight = evidence_map.reshape(total_tokens)
    bridge = bridge_map.reshape(total_tokens)
    surprise = surprise_map.reshape(total_tokens)
    background = background_map.reshape(total_tokens)
    frame_ids = torch.arange(num_frames, device=segment_features.device, dtype=torch.long).repeat_interleave(num_visual_tokens)

    selected, candidate_count = _select_pivots(
        normed_flat=normed_flat,
        utility=utility,
        evidence_weight=evidence_weight,
        frame_ids=frame_ids,
        target_tokens=target_tokens,
        num_frames=num_frames,
        config=flashvid_config,
    )
    fused, assignment_sim, avg_cluster, max_cluster, coverage_mean = _fuse_to_pivots(
        features_flat=features_flat,
        normed_flat=normed_flat,
        selected=selected,
        utility=utility,
        evidence_weight=evidence_weight,
        config=flashvid_config,
    )
    selected_globals = globals_flat[selected]
    order = torch.argsort(selected_globals)
    fused = fused[order]
    selected_globals = selected_globals[order]

    _accumulate_pivot_metrics(
        flashvid_config,
        target_tokens=target_tokens,
        selected_tokens=int(selected.numel()),
        candidate_count=candidate_count,
        use_fuse=_cfg_bool(flashvid_config, "pivot_use_fuse", True),
        budget_scale=budget_scale,
        avg_cluster_size=avg_cluster,
        max_cluster_size=max_cluster,
        coverage_mean=coverage_mean,
        selected_utility_mean=float(utility[selected].mean().item()) if selected.numel() > 0 else None,
        bridge_mean=float(bridge[selected].mean().item()) if selected.numel() > 0 else None,
        surprise_mean=float(surprise[selected].mean().item()) if selected.numel() > 0 else None,
        background_mean=float(background[selected].mean().item()) if selected.numel() > 0 else None,
    )
    return fused.to(dtype=segment_features.dtype), selected_globals
