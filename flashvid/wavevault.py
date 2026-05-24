from __future__ import annotations

from typing import Tuple

import torch

from .configuration_flashvid import FlashVidConfig
from .pivotfuse import _cfg_float, _cfg_int, _compute_pivot_utility, _safe_topk


def _reset_wave_metrics(config: FlashVidConfig) -> None:
    defaults = {
        "last_wave_target_tokens": 0.0,
        "last_wave_anchor_tokens": 0.0,
        "last_wave_vault_tokens": 0.0,
        "last_wave_candidate_count": 0.0,
        "last_wave_anchor_ratio": 0.0,
        "last_wave_sim_threshold": 0.0,
        "last_wave_budget_scale": 0.0,
        "last_wave_coverage_mean": None,
        "last_wave_residual_mean": None,
        "last_wave_anchor_utility_mean": None,
        "last_wave_vault_residual_mean": None,
        "last_wave_anchor_drift": 0.0,
    }
    for key, value in defaults.items():
        setattr(config, key, value)
    setattr(config, "_wave_segments", 0.0)
    for key in defaults:
        setattr(config, f"_{key[5:]}_sum", 0.0)
    for key in (
        "wave_coverage",
        "wave_residual",
        "wave_anchor_utility",
        "wave_vault_residual",
    ):
        setattr(config, f"_{key}_count", 0.0)


def _accumulate_wave_metrics(
    config: FlashVidConfig,
    *,
    target_tokens: int,
    anchor_tokens: int,
    vault_tokens: int,
    candidate_count: int,
    anchor_ratio: float,
    sim_threshold: float,
    budget_scale: float,
    coverage_mean: float | None,
    residual_mean: float | None,
    anchor_utility_mean: float | None,
    vault_residual_mean: float | None,
) -> None:
    if not hasattr(config, "_wave_segments"):
        _reset_wave_metrics(config)
    setattr(config, "_wave_segments", float(getattr(config, "_wave_segments", 0.0)) + 1.0)
    simple = {
        "wave_target_tokens": target_tokens,
        "wave_anchor_tokens": anchor_tokens,
        "wave_vault_tokens": vault_tokens,
        "wave_candidate_count": candidate_count,
        "wave_anchor_ratio": anchor_ratio,
        "wave_sim_threshold": sim_threshold,
        "wave_budget_scale": budget_scale,
        "wave_anchor_drift": 0.0,
    }
    for key, value in simple.items():
        attr = f"_{key}_sum"
        setattr(config, attr, float(getattr(config, attr, 0.0)) + float(value))

    optional = {
        "wave_coverage": coverage_mean,
        "wave_residual": residual_mean,
        "wave_anchor_utility": anchor_utility_mean,
        "wave_vault_residual": vault_residual_mean,
    }
    for key, value in optional.items():
        if value is None:
            continue
        setattr(config, f"_{key}_sum", float(getattr(config, f"_{key}_sum", 0.0)) + float(value))
        setattr(config, f"_{key}_count", float(getattr(config, f"_{key}_count", 0.0)) + 1.0)

    segments = max(1.0, float(getattr(config, "_wave_segments", 1.0)))
    for key in simple:
        setattr(config, f"last_{key}", float(getattr(config, f"_{key}_sum", 0.0)) / segments)
    for key, last_key in (
        ("wave_coverage", "last_wave_coverage_mean"),
        ("wave_residual", "last_wave_residual_mean"),
        ("wave_anchor_utility", "last_wave_anchor_utility_mean"),
        ("wave_vault_residual", "last_wave_vault_residual_mean"),
    ):
        count = float(getattr(config, f"_{key}_count", 0.0))
        setattr(config, last_key, float(getattr(config, f"_{key}_sum", 0.0)) / count if count > 0 else None)


def _choose_from_candidates(
    *,
    normed_flat: torch.Tensor,
    candidates: torch.Tensor,
    base_score: torch.Tensor,
    count: int,
    sim_threshold: float,
    already_selected: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate safe facility-location greedy over the candidate pool."""
    count = min(max(0, int(count)), int(candidates.numel()))
    if count <= 0 or candidates.numel() == 0:
        empty = torch.empty((0,), dtype=torch.long, device=normed_flat.device)
        return empty, torch.zeros((int(candidates.numel()),), dtype=torch.float32, device=normed_flat.device)

    candidate_normed = normed_flat[candidates]
    coverage = torch.zeros((int(candidates.numel()),), dtype=torch.float32, device=normed_flat.device)
    blocked = torch.zeros_like(coverage, dtype=torch.bool)
    if already_selected is not None and already_selected.numel() > 0:
        selected_set = set(int(x) for x in already_selected.detach().cpu().tolist())
        blocked_cpu = [int(x) in selected_set for x in candidates.detach().cpu().tolist()]
        blocked = torch.tensor(blocked_cpu, dtype=torch.bool, device=normed_flat.device)

    selected: list[int] = []
    for _ in range(count):
        score = base_score[candidates] * (1.0 - coverage).clamp_min(0.0)
        score = score.masked_fill(blocked, -float("inf"))
        best_pos = int(torch.argmax(score).item())
        if not torch.isfinite(score[best_pos]):
            break
        best_idx = int(candidates[best_pos].item())
        selected.append(best_idx)
        blocked[best_pos] = True
        sim = torch.matmul(candidate_normed, normed_flat[best_idx]).clamp_min(0.0)
        safe_sim = sim * (sim >= sim_threshold).float()
        coverage = torch.maximum(coverage, safe_sim)

    if len(selected) < count:
        remaining_mask = ~blocked
        remaining = candidates[remaining_mask]
        if remaining.numel() > 0:
            fill = _safe_topk(base_score[remaining], count - len(selected), largest=True)
            selected.extend(int(x) for x in remaining[fill].tolist())
    selected_tensor = torch.tensor(selected[:count], dtype=torch.long, device=normed_flat.device)
    return selected_tensor, coverage


def _safe_coverage(
    normed_flat: torch.Tensor,
    selected: torch.Tensor,
    sim_threshold: float,
    chunk_size: int = 512,
) -> torch.Tensor:
    if selected.numel() == 0:
        return torch.zeros((int(normed_flat.shape[0]),), dtype=torch.float32, device=normed_flat.device)
    coverage = torch.zeros((int(normed_flat.shape[0]),), dtype=torch.float32, device=normed_flat.device)
    selected_normed = normed_flat[selected]
    for start in range(0, int(selected_normed.shape[0]), max(1, int(chunk_size))):
        block = selected_normed[start : start + max(1, int(chunk_size))]
        sim = torch.matmul(normed_flat, block.transpose(0, 1)).clamp_min(0.0)
        safe_sim = sim * (sim >= sim_threshold).float()
        coverage = torch.maximum(coverage, safe_sim.max(dim=1).values)
    return coverage


def wavevault_segment_compression(
    segment_features: torch.Tensor,
    segment_global_indices: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """WAVE-VAULT: drift-free anchor selection plus residual vault tokens.

    Unlike PIVOT-FUSE, every emitted token is an unchanged original visual token:
    anchors are invariant and vault tokens are medoids from uncovered regions.
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
    budget_scale = max(0.01, _cfg_float(flashvid_config, "wave_budget_scale", 1.0))
    target_tokens = min(total_tokens, max(1, int(round(per_frame_budget * num_frames * budget_scale))))
    anchor_ratio = min(max(_cfg_float(flashvid_config, "wave_anchor_ratio", 0.80), 0.0), 1.0)
    anchor_tokens = min(target_tokens, max(0, int(round(target_tokens * anchor_ratio))))
    vault_tokens = max(0, target_tokens - anchor_tokens)
    if anchor_tokens <= 0 and target_tokens > 0:
        anchor_tokens = 1
        vault_tokens = max(0, target_tokens - 1)
    sim_threshold = min(max(_cfg_float(flashvid_config, "wave_sim_threshold", 0.60), 0.0), 1.0)

    utility_map, evidence_map, _bridge, _surprise, _background, normed = _compute_pivot_utility(
        segment_features,
        cls_attention,
        flashvid_config,
    )
    features_flat = segment_features.reshape(total_tokens, feat_dim)
    globals_flat = segment_global_indices.reshape(total_tokens)
    normed_flat = normed.reshape(total_tokens, feat_dim)
    utility = utility_map.reshape(total_tokens)
    evidence = evidence_map.reshape(total_tokens)
    q = evidence / evidence.sum().clamp_min(1.0e-6)
    base_score = (q * utility.clamp_min(1.0e-6)).clamp_min(1.0e-9)

    candidate_factor = max(1.0, _cfg_float(flashvid_config, "wave_candidate_factor", 4.0))
    max_candidates = max(target_tokens, _cfg_int(flashvid_config, "wave_max_candidates", 2048))
    candidate_count = min(total_tokens, max(target_tokens, min(max_candidates, int(candidate_factor * target_tokens))))
    candidates = _safe_topk(base_score, candidate_count, largest=True)
    if candidates.numel() == 0:
        candidates = torch.arange(total_tokens, dtype=torch.long, device=segment_features.device)

    anchors, _ = _choose_from_candidates(
        normed_flat=normed_flat,
        candidates=candidates,
        base_score=base_score,
        count=anchor_tokens,
        sim_threshold=sim_threshold,
    )
    coverage_after_anchor = _safe_coverage(normed_flat, anchors, sim_threshold)
    residual_score = q * (1.0 - coverage_after_anchor).clamp_min(0.0)
    if anchors.numel() > 0:
        residual_score[anchors] = -float("inf")
    vault_candidate_count = min(total_tokens, max(vault_tokens, min(max_candidates, int(candidate_factor * max(1, vault_tokens)))))
    vault_candidates = _safe_topk(residual_score, vault_candidate_count, largest=True)
    vaults, _ = _choose_from_candidates(
        normed_flat=normed_flat,
        candidates=vault_candidates,
        base_score=residual_score.clamp_min(0.0),
        count=vault_tokens,
        sim_threshold=sim_threshold,
        already_selected=anchors,
    )
    selected = torch.cat([anchors, vaults], dim=0) if vaults.numel() > 0 else anchors
    if selected.numel() < target_tokens:
        selected_mask = torch.zeros((total_tokens,), dtype=torch.bool, device=segment_features.device)
        if selected.numel() > 0:
            selected_mask[selected] = True
        remaining = torch.where(~selected_mask)[0]
        fill = _safe_topk(base_score[remaining], target_tokens - int(selected.numel()), largest=True)
        selected = torch.cat([selected, remaining[fill]], dim=0)
    selected = selected[:target_tokens]
    selected_globals = globals_flat[selected]
    order = torch.argsort(selected_globals)
    selected = selected[order]
    selected_globals = selected_globals[order]
    selected_tokens = features_flat[selected]

    final_coverage = _safe_coverage(normed_flat, selected, sim_threshold)
    _accumulate_wave_metrics(
        flashvid_config,
        target_tokens=target_tokens,
        anchor_tokens=int(anchors.numel()),
        vault_tokens=int(vaults.numel()),
        candidate_count=int(candidates.numel()),
        anchor_ratio=anchor_ratio,
        sim_threshold=sim_threshold,
        budget_scale=budget_scale,
        coverage_mean=float(final_coverage.mean().item()) if final_coverage.numel() > 0 else None,
        residual_mean=float((1.0 - final_coverage).mean().item()) if final_coverage.numel() > 0 else None,
        anchor_utility_mean=float(utility[anchors].mean().item()) if anchors.numel() > 0 else None,
        vault_residual_mean=float(residual_score[vaults].mean().item()) if vaults.numel() > 0 else None,
    )
    return selected_tokens.to(dtype=segment_features.dtype), selected_globals
