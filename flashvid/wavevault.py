from __future__ import annotations

import heapq
from typing import Tuple

import torch

from .configuration_flashvid import FlashVidConfig
from .pivotfuse import _cfg_float, _cfg_int, _compute_pivot_utility, _safe_topk
from .learned_selector import build_scalar_token_features, load_selector_checkpoint, score_with_selector


def _reset_wave_metrics(config: FlashVidConfig) -> None:
    defaults = {
        "last_wave_target_tokens": 0.0,
        "last_wave_anchor_tokens": 0.0,
        "last_wave_vault_tokens": 0.0,
        "last_wave_candidate_count": 0.0,
        "last_wave_anchor_ratio": 0.0,
        "last_wave_sim_threshold": 0.0,
        "last_wave_budget_scale": 0.0,
        "last_wave_intrinsic_weight": 0.0,
        "last_wave_vault_intrinsic_weight": 0.0,
        "last_wave_q_floor": 0.0,
        "last_wave_coverage_mean": None,
        "last_wave_residual_mean": None,
        "last_wave_anchor_utility_mean": None,
        "last_wave_vault_residual_mean": None,
        "last_wave_anchor_drift": 0.0,
        "last_wave_learn_active": 0.0,
        "last_wave_learn_blend": 0.0,
        "last_wave_learn_score_mean": None,
        "last_wave_learn_score_std": None,
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
        "wave_learn_score",
        "wave_learn_score_std",
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
    intrinsic_weight: float,
    vault_intrinsic_weight: float,
    q_floor: float,
    learn_active: bool,
    learn_blend: float,
    coverage_mean: float | None,
    residual_mean: float | None,
    anchor_utility_mean: float | None,
    vault_residual_mean: float | None,
    learn_score_mean: float | None,
    learn_score_std: float | None,
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
        "wave_intrinsic_weight": intrinsic_weight,
        "wave_vault_intrinsic_weight": vault_intrinsic_weight,
        "wave_q_floor": q_floor,
        "wave_anchor_drift": 0.0,
        "wave_learn_active": float(bool(learn_active)),
        "wave_learn_blend": learn_blend,
    }
    for key, value in simple.items():
        attr = f"_{key}_sum"
        setattr(config, attr, float(getattr(config, attr, 0.0)) + float(value))

    optional = {
        "wave_coverage": coverage_mean,
        "wave_residual": residual_mean,
        "wave_anchor_utility": anchor_utility_mean,
        "wave_vault_residual": vault_residual_mean,
        "wave_learn_score": learn_score_mean,
        "wave_learn_score_std": learn_score_std,
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
        ("wave_learn_score", "last_wave_learn_score_mean"),
        ("wave_learn_score_std", "last_wave_learn_score_std"),
    ):
        count = float(getattr(config, f"_{key}_count", 0.0))
        setattr(config, last_key, float(getattr(config, f"_{key}_sum", 0.0)) / count if count > 0 else None)


def _minmax_per_frame(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    lo = values.amin(dim=1, keepdim=True)
    hi = values.amax(dim=1, keepdim=True)
    return ((values - lo) / (hi - lo + 1.0e-6)).clamp(0.0, 1.0)


def _get_wave_selector(config: FlashVidConfig, device: torch.device):
    path = str(getattr(config, "learn_selector_ckpt", "") or "")
    if not path:
        return None
    cached_path = str(getattr(config, "_wave_selector_ckpt_path", "") or "")
    cached = getattr(config, "_wave_selector_model", None)
    if cached is not None and cached_path == path:
        return cached
    selector = load_selector_checkpoint(path, device)
    setattr(config, "_wave_selector_ckpt_path", path)
    setattr(config, "_wave_selector_model", selector)
    return selector


def _choose_by_global_coverage(
    *,
    normed_flat: torch.Tensor,
    candidates: torch.Tensor,
    token_weight: torch.Tensor,
    candidate_quality: torch.Tensor | None,
    count: int,
    sim_threshold: float,
    initial_coverage: torch.Tensor | None = None,
    already_selected: torch.Tensor | None = None,
    intrinsic_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lazy greedy safe facility-location over all tokens.

    The coverage state has length L (all visual tokens), matching the WAVE
    objective sum_i q_i max_p s(x_i, p). A heap gives us lazy submodular
    greedy: stale gains are upper bounds, so only promising candidates are
    recomputed after each coverage update.
    """
    device = normed_flat.device
    total_tokens = int(normed_flat.shape[0])
    count = min(max(0, int(count)), int(candidates.numel()))
    coverage = (
        initial_coverage.float().clone()
        if initial_coverage is not None
        else torch.zeros((total_tokens,), dtype=torch.float32, device=device)
    )
    if count <= 0 or candidates.numel() == 0:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return empty, coverage

    candidates = candidates.long()
    candidate_normed = normed_flat[candidates].float()
    all_normed = normed_flat.float()
    token_weight = token_weight.float().clamp_min(0.0)
    safe_sim = torch.matmul(all_normed, candidate_normed.transpose(0, 1)).clamp_min(0.0)
    safe_sim = safe_sim * (safe_sim >= sim_threshold).float()

    num_candidates = int(candidates.numel())
    blocked = torch.zeros((num_candidates,), dtype=torch.bool, device=device)
    if already_selected is not None and already_selected.numel() > 0:
        already_mask = torch.zeros((total_tokens,), dtype=torch.bool, device=device)
        already_mask[already_selected.long()] = True
        blocked = already_mask[candidates]

    if candidate_quality is None:
        intrinsic = torch.zeros((num_candidates,), dtype=torch.float32, device=device)
    else:
        intrinsic = candidate_quality[candidates].float().clamp_min(0.0) * max(0.0, float(intrinsic_weight))

    initial_gain = (token_weight.unsqueeze(1) * (safe_sim - coverage.unsqueeze(1)).clamp_min(0.0)).sum(dim=0)
    initial_gain = initial_gain + intrinsic
    initial_gain = initial_gain.masked_fill(blocked, -float("inf"))

    versions = torch.zeros((num_candidates,), dtype=torch.long, device=device)
    heap: list[tuple[float, int, int]] = [
        (-float(initial_gain[pos].item()), pos, 0)
        for pos in range(num_candidates)
        if torch.isfinite(initial_gain[pos])
    ]
    heapq.heapify(heap)

    def _next_upper_bound() -> float:
        while heap:
            neg_gain, pos, version = heap[0]
            if bool(blocked[pos]) or version != int(versions[pos].item()):
                heapq.heappop(heap)
                continue
            return -float(neg_gain)
        return -float("inf")

    selected_positions: list[int] = []
    for _ in range(count):
        accepted_pos: int | None = None
        while heap:
            neg_gain, pos, version = heapq.heappop(heap)
            if bool(blocked[pos]) or version != int(versions[pos].item()):
                continue
            exact_gain = (token_weight * (safe_sim[:, pos] - coverage).clamp_min(0.0)).sum() + intrinsic[pos]
            exact_value = float(exact_gain.item())
            if not torch.isfinite(exact_gain):
                blocked[pos] = True
                continue
            next_bound = _next_upper_bound()
            if exact_value >= next_bound - 1.0e-8:
                accepted_pos = pos
                break
            versions[pos] += 1
            heapq.heappush(heap, (-exact_value, pos, int(versions[pos].item())))
        if accepted_pos is None:
            break
        selected_positions.append(accepted_pos)
        blocked[accepted_pos] = True
        coverage = torch.maximum(coverage, safe_sim[:, accepted_pos])

    if not selected_positions:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return empty, coverage
    selected_tensor = candidates[torch.tensor(selected_positions[:count], dtype=torch.long, device=device)]
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


def _merge_unique_indices(
    *indices: torch.Tensor,
    max_count: int | None = None,
) -> torch.Tensor:
    parts = [idx.long().reshape(-1) for idx in indices if idx is not None and idx.numel() > 0]
    if not parts:
        device = indices[0].device if indices else torch.device("cpu")
        return torch.empty((0,), dtype=torch.long, device=device)
    merged = torch.cat(parts, dim=0)
    keep: list[int] = []
    seen: set[int] = set()
    for value in merged.detach().cpu().tolist():
        idx = int(value)
        if idx in seen:
            continue
        seen.add(idx)
        keep.append(idx)
        if max_count is not None and len(keep) >= int(max_count):
            break
    return torch.tensor(keep, dtype=torch.long, device=merged.device)


def wavevault_segment_compression(
    segment_features: torch.Tensor,
    segment_global_indices: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
    question_features: torch.Tensor | None = None,
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
    intrinsic_weight = max(0.0, _cfg_float(flashvid_config, "wave_intrinsic_weight", 0.01))
    vault_intrinsic_weight = max(0.0, _cfg_float(flashvid_config, "wave_vault_intrinsic_weight", 0.0))
    q_floor = max(0.0, _cfg_float(flashvid_config, "wave_q_floor", 0.03))
    target_tokens = min(total_tokens, max(1, int(round(per_frame_budget * num_frames * budget_scale))))
    anchor_ratio = min(max(_cfg_float(flashvid_config, "wave_anchor_ratio", 0.80), 0.0), 1.0)
    anchor_tokens = min(target_tokens, max(0, int(round(target_tokens * anchor_ratio))))
    vault_tokens = max(0, target_tokens - anchor_tokens)
    if anchor_tokens <= 0 and target_tokens > 0:
        anchor_tokens = 1
        vault_tokens = max(0, target_tokens - 1)
    sim_threshold = min(max(_cfg_float(flashvid_config, "wave_sim_threshold", 0.55), 0.0), 1.0)

    utility_map, evidence_map, _bridge, _surprise, _background, normed = _compute_pivot_utility(
        segment_features,
        cls_attention,
        flashvid_config,
    )
    learn_active = False
    learn_blend = min(max(float(getattr(flashvid_config, "learn_score_blend", 0.35)), 0.0), 1.0)
    learn_score_mean: float | None = None
    learn_score_std: float | None = None
    selector = _get_wave_selector(flashvid_config, segment_features.device)
    if selector is not None and learn_blend > 0.0:
        scalar_features, aux = build_scalar_token_features(
            segment_features,
            cls_attention,
            question_features if bool(getattr(flashvid_config, "learn_qaware", True)) else None,
            density_topk=int(getattr(flashvid_config, "learn_density_topk", 8) or 8),
        )
        learned_score = score_with_selector(
            selector,
            scalar_features,
            aux,
            blend=learn_blend,
            q_weight=float(getattr(flashvid_config, "learn_q_relevance_weight", 0.35)),
        )
        learn_active = True
        learn_score_mean = float(learned_score.float().mean().item())
        learn_score_std = float(learned_score.float().std(unbiased=False).item())
        utility_map = _minmax_per_frame((1.0 - learn_blend) * utility_map.float() + learn_blend * learned_score.float())
        evidence_map = _minmax_per_frame((1.0 - learn_blend) * evidence_map.float() + learn_blend * learned_score.float())
    features_flat = segment_features.reshape(total_tokens, feat_dim)
    globals_flat = segment_global_indices.reshape(total_tokens)
    normed_flat = normed.reshape(total_tokens, feat_dim)
    utility = utility_map.reshape(total_tokens)
    evidence = evidence_map.reshape(total_tokens)
    q_raw = evidence.clamp_min(0.0)
    q_floor = min(max(q_floor, 0.0), 0.5)
    if bool(torch.isfinite(q_raw).all()) and float(q_raw.sum().item()) > 1.0e-6:
        q_evidence = q_raw / q_raw.sum().clamp_min(1.0e-6)
    else:
        q_evidence = torch.full_like(q_raw, 1.0 / max(1, total_tokens))
    q_uniform = torch.full_like(q_evidence, 1.0 / max(1, total_tokens))
    q = (1.0 - q_floor) * q_evidence + q_floor * q_uniform
    q = q / q.sum().clamp_min(1.0e-6)
    base_score = (q * utility.clamp_min(1.0e-6)).clamp_min(1.0e-9)

    candidate_factor = max(1.0, _cfg_float(flashvid_config, "wave_candidate_factor", 2.0))
    max_candidates = max(target_tokens, _cfg_int(flashvid_config, "wave_max_candidates", 1024))
    candidate_count = min(total_tokens, max(target_tokens, min(max_candidates, int(candidate_factor * target_tokens))))
    uniform_count = min(total_tokens, max(1, target_tokens // 4))
    top_count = max(1, min(total_tokens, max(1, candidate_count - uniform_count)))
    top_candidates = _safe_topk(base_score, top_count, largest=True)
    uniform_idx = torch.linspace(
        0,
        max(0, total_tokens - 1),
        steps=uniform_count,
        device=segment_features.device,
    ).long()
    candidates = _merge_unique_indices(top_candidates, uniform_idx, max_count=max_candidates)
    if candidates.numel() == 0:
        candidates = torch.arange(total_tokens, dtype=torch.long, device=segment_features.device)

    anchors, coverage_after_anchor = _choose_by_global_coverage(
        normed_flat=normed_flat,
        candidates=candidates,
        token_weight=q,
        candidate_quality=utility,
        count=anchor_tokens,
        sim_threshold=sim_threshold,
        intrinsic_weight=intrinsic_weight,
    )
    residual_score = q * (1.0 - coverage_after_anchor).clamp_min(0.0)
    if anchors.numel() > 0:
        residual_score[anchors] = -float("inf")
    vault_candidate_count = min(total_tokens, max(vault_tokens, min(max_candidates, int(candidate_factor * max(1, vault_tokens)))))
    vault_candidates = _safe_topk(residual_score, vault_candidate_count, largest=True)
    vaults, coverage_after_vault = _choose_by_global_coverage(
        normed_flat=normed_flat,
        candidates=vault_candidates,
        token_weight=q,
        candidate_quality=residual_score.clamp_min(0.0),
        count=vault_tokens,
        sim_threshold=sim_threshold,
        initial_coverage=coverage_after_anchor,
        already_selected=anchors,
        intrinsic_weight=vault_intrinsic_weight,
    )
    selected = torch.cat([anchors, vaults], dim=0) if vaults.numel() > 0 else anchors
    if selected.numel() < target_tokens:
        selected_mask = torch.zeros((total_tokens,), dtype=torch.bool, device=segment_features.device)
        if selected.numel() > 0:
            selected_mask[selected] = True
        current_coverage = coverage_after_vault if selected.numel() > 0 else torch.zeros_like(q)
        fill_score = q * (1.0 - current_coverage).clamp_min(0.0)
        fill_score[selected_mask] = -float("inf")
        fill = _safe_topk(fill_score, target_tokens - int(selected.numel()), largest=True)
        selected = torch.cat([selected, fill], dim=0)
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
        intrinsic_weight=intrinsic_weight,
        vault_intrinsic_weight=vault_intrinsic_weight,
        q_floor=q_floor,
        learn_active=learn_active,
        learn_blend=learn_blend if learn_active else 0.0,
        coverage_mean=float(final_coverage.mean().item()) if final_coverage.numel() > 0 else None,
        residual_mean=float((1.0 - final_coverage).mean().item()) if final_coverage.numel() > 0 else None,
        anchor_utility_mean=float(utility[anchors].mean().item()) if anchors.numel() > 0 else None,
        vault_residual_mean=float(residual_score[vaults].mean().item()) if vaults.numel() > 0 else None,
        learn_score_mean=learn_score_mean,
        learn_score_std=learn_score_std,
    )
    return selected_tokens.to(dtype=segment_features.dtype), selected_globals
