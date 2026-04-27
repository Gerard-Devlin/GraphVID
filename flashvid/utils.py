from typing import Optional, Tuple, Union, List, Iterable, Dict

import math
import torch
from torch.nn import functional as F
from .configuration_flashvid import FlashVidConfig
from .token_selection import (
    attn_based_token_selection,
    attn_div_based_token_selection,
    attn_div_v2_based_token_selection,
    div_based_token_selection,
    TokenSelectionMethod,
)

ALL_TOKEN_SELECTION_METHOD = {
    TokenSelectionMethod.ATTN: attn_based_token_selection,
    TokenSelectionMethod.ADTS_v2: attn_div_v2_based_token_selection,
    TokenSelectionMethod.ADTS: attn_div_based_token_selection,
    TokenSelectionMethod.DIV: div_based_token_selection,
}


def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    scores = scores.float()
    if scores.numel() == 0:
        return scores
    min_score = scores.min()
    max_score = scores.max()
    return (scores - min_score) / (max_score - min_score + 1e-6)


def extract_question_features(
    input_ids: Optional[torch.Tensor],
    inputs_embeds: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    invalid_token_ids: Optional[Iterable[int]] = None,
    batch_index: int = 0,
) -> Optional[torch.Tensor]:
    """Extract text/question token embeddings for question-aware token reweighting."""
    if input_ids is None or inputs_embeds is None:
        return None
    if input_ids.ndim != 2 or inputs_embeds.ndim != 3:
        return None
    if batch_index >= input_ids.shape[0] or batch_index >= inputs_embeds.shape[0]:
        return None

    token_ids = input_ids[batch_index]
    token_mask = torch.ones_like(token_ids, dtype=torch.bool)

    if attention_mask is not None and attention_mask.ndim == 2:
        attn = attention_mask[batch_index]
        token_mask &= attn.bool() if attn.dtype == torch.bool else attn > 0

    if invalid_token_ids is not None:
        invalid_mask = torch.zeros_like(token_mask)
        for token_id in invalid_token_ids:
            if token_id is None:
                continue
            invalid_mask |= token_ids == int(token_id)
        token_mask &= ~invalid_mask

    if not token_mask.any():
        return None
    return inputs_embeds[batch_index, token_mask]


def _estimate_video_complexity(video_features: torch.Tensor) -> float:
    """Estimate video complexity in [0, 1] from temporal change + spatial dispersion."""
    num_frames = video_features.shape[0]
    normed_tokens = F.normalize(video_features.float(), p=2, dim=-1, eps=1e-6)

    frame_centers = F.normalize(normed_tokens.mean(dim=1), p=2, dim=-1, eps=1e-6)
    if num_frames > 1:
        temporal_sim = torch.sum(frame_centers[:-1] * frame_centers[1:], dim=-1)
        temporal_complexity = ((1.0 - temporal_sim).clamp(min=0.0, max=2.0) * 0.5).mean()
    else:
        temporal_complexity = torch.tensor(0.0, device=video_features.device)

    center_per_token = frame_centers.unsqueeze(1).expand_as(normed_tokens)
    spatial_sim = torch.sum(normed_tokens * center_per_token, dim=-1)
    spatial_complexity = ((1.0 - spatial_sim).clamp(min=0.0, max=2.0) * 0.5).mean()

    return float((0.6 * temporal_complexity + 0.4 * spatial_complexity).item())


def _estimate_question_difficulty(question_features: Optional[torch.Tensor]) -> float:
    """Estimate question difficulty in [0, 1] from length + semantic dispersion."""
    if question_features is None or question_features.numel() == 0:
        return 0.5

    q = F.normalize(question_features.float(), p=2, dim=-1, eps=1e-6)
    q_center = F.normalize(q.mean(dim=0), p=2, dim=-1, eps=1e-6)
    q_dispersion = 1.0 - torch.matmul(q, q_center).clamp(min=-1.0, max=1.0)
    semantic_difficulty = (q_dispersion.clamp(min=0.0, max=2.0) * 0.5).mean()
    length_difficulty = min(1.0, q.shape[0] / 32.0)

    return float(0.5 * semantic_difficulty.item() + 0.5 * length_difficulty)


def _resolve_effective_retention_ratio(
    video_features: torch.Tensor,
    question_features: Optional[torch.Tensor],
    flashvid_config: FlashVidConfig,
) -> float:
    base_ratio = float(flashvid_config.retention_ratio)
    if not bool(getattr(flashvid_config, "adaptive_token_budget", False)):
        flashvid_config.last_adaptive_retention_ratio = base_ratio
        return base_ratio

    candidate_ratios = sorted(
        [
            max(0.01, min(1.0, float(getattr(flashvid_config, "adaptive_budget_low", 0.10)))),
            max(0.01, min(1.0, float(getattr(flashvid_config, "adaptive_budget_mid", 0.15)))),
            max(0.01, min(1.0, float(getattr(flashvid_config, "adaptive_budget_high", 0.20)))),
        ]
    )

    video_complexity = _estimate_video_complexity(video_features)
    question_difficulty = _estimate_question_difficulty(question_features)
    complexity_score = 0.7 * video_complexity + 0.3 * question_difficulty
    level = min(len(candidate_ratios) - 1, int(complexity_score * len(candidate_ratios)))
    adaptive_ratio = candidate_ratios[level]

    flashvid_config.last_adaptive_retention_ratio = adaptive_ratio
    return adaptive_ratio


def _question_aware_scores(
    flat_features: torch.Tensor,
    flat_attention: torch.Tensor,
    question_features: Optional[torch.Tensor],
    flashvid_config: FlashVidConfig,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    visual_scores = _normalize_scores(flat_attention)
    if not bool(getattr(flashvid_config, "question_aware_reweighting", False)):
        return visual_scores, None
    if question_features is None or question_features.numel() == 0:
        return visual_scores, None

    token_features = F.normalize(flat_features.float(), p=2, dim=-1, eps=1e-6)
    question_proto = F.normalize(question_features.float().mean(dim=0), p=2, dim=-1, eps=1e-6)
    question_scores = _normalize_scores(torch.matmul(token_features, question_proto))

    beta = float(getattr(flashvid_config, "question_reweight_beta", 0.35))
    beta = min(max(beta, 0.0), 1.0)
    fused_scores = (1.0 - beta) * visual_scores + beta * question_scores
    return fused_scores, question_scores


def _resolve_memory_budget(
    num_tokens: int,
    target_budget: int,
    flashvid_config: FlashVidConfig,
) -> int:
    dropped_tokens = max(0, num_tokens - target_budget)
    if dropped_tokens == 0:
        return 0

    ratio = max(0.0, float(getattr(flashvid_config, "memory_token_ratio", 0.10)))
    memory_min = max(0, int(getattr(flashvid_config, "memory_token_min", 1)))
    memory_max = max(memory_min, int(getattr(flashvid_config, "memory_token_max", 8)))

    memory_budget = int(round(target_budget * ratio))
    memory_budget = max(memory_min, memory_budget)
    memory_budget = min(memory_budget, memory_max, dropped_tokens)
    memory_budget = min(memory_budget, max(0, target_budget - 1))
    return max(0, memory_budget)


def _build_residual_memory_tokens(
    flat_features: torch.Tensor,
    dropped_indices: torch.Tensor,
    residual_vectors: torch.Tensor,
    question_scores: Optional[torch.Tensor],
    memory_budget: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    feat_dim = flat_features.shape[-1]
    if memory_budget <= 0 or dropped_indices.numel() == 0:
        return (
            flat_features.new_zeros((0, feat_dim)),
            torch.empty((0,), dtype=torch.long, device=flat_features.device),
        )

    priorities = residual_vectors.float().norm(p=2, dim=-1)
    if question_scores is not None:
        priorities = priorities * (1.0 + question_scores[dropped_indices].float())

    order = torch.argsort(priorities, descending=True)
    sorted_dropped = dropped_indices[order]
    sorted_priorities = priorities[order]
    sorted_features = flat_features[sorted_dropped]

    num_mem_tokens = min(memory_budget, sorted_dropped.numel())
    split_indices = torch.chunk(torch.arange(sorted_dropped.numel(), device=flat_features.device), num_mem_tokens)

    memory_tokens = []
    memory_indices = []
    for split in split_indices:
        if split.numel() == 0:
            continue
        split_weights = sorted_priorities[split].to(flat_features.dtype).clamp_min(1e-6)
        split_features = sorted_features[split]
        merged_token = torch.sum(split_features * split_weights.unsqueeze(-1), dim=0) / split_weights.sum()
        memory_tokens.append(merged_token)
        memory_indices.append(sorted_dropped[split[0]])

    if not memory_tokens:
        return (
            flat_features.new_zeros((0, feat_dim)),
            torch.empty((0,), dtype=torch.long, device=flat_features.device),
        )
    return torch.stack(memory_tokens, dim=0), torch.stack(memory_indices, dim=0)


def _resolve_grid_hw(num_visual_tokens: int, flashvid_config: FlashVidConfig) -> Tuple[int, int]:
    h = int(getattr(flashvid_config, "H", 0) or 0)
    w = int(getattr(flashvid_config, "W", 0) or 0)
    if h > 0 and w > 0 and h * w >= num_visual_tokens:
        return h, w
    h = max(1, int(round(math.sqrt(max(1, num_visual_tokens)))))
    w = max(1, int(math.ceil(num_visual_tokens / h)))
    return h, w


def _resolve_slot_role_names(flashvid_config: FlashVidConfig) -> List[str]:
    predefined = ["scene", "motion", "interaction", "background", "detail"]
    role_count = max(1, int(getattr(flashvid_config, "slot_base_roles", 5)))
    if role_count <= len(predefined):
        return predefined[:role_count]
    extras = [f"extra_{i + 1}" for i in range(role_count - len(predefined))]
    return predefined + extras


def _resolve_slot_priority(role_names: List[str], flashvid_config: FlashVidConfig) -> List[str]:
    # Precision-oriented default priority.
    default_priority = ["motion", "interaction", "detail", "scene", "background"]
    raw = str(getattr(flashvid_config, "slot_role_allocation", "") or "")
    parsed = [item.strip().lower() for item in raw.split(",") if item.strip()]
    ordered: List[str] = []
    for role in parsed + default_priority + role_names:
        if role in role_names and role not in ordered:
            ordered.append(role)
    return ordered


def _allocate_slot_roles(
    slot_budget: int,
    flashvid_config: FlashVidConfig,
) -> Tuple[List[str], List[str]]:
    role_names = _resolve_slot_role_names(flashvid_config)
    if slot_budget <= 0:
        return role_names, []

    max_slots = int(getattr(flashvid_config, "slot_max_per_segment", slot_budget))
    if max_slots > 0:
        slot_budget = min(slot_budget, max_slots)
    slot_budget = max(1, slot_budget)

    priority = _resolve_slot_priority(role_names, flashvid_config)
    counts: Dict[str, int] = {role: 0 for role in role_names}

    # One slot per role first when budget allows.
    for role in priority:
        if sum(counts.values()) >= slot_budget:
            break
        counts[role] = 1

    remaining = slot_budget - sum(counts.values())
    if remaining > 0:
        # Precision-first weighted allocation:
        # motion > interaction > detail > scene > background (and then extras).
        # We avoid round-robin here because it over-allocates low-priority roles.
        weight_map: Dict[str, float] = {}
        rank_count = len(priority)
        for rank, role in enumerate(priority):
            weight_map[role] = float(rank_count - rank)

        for _ in range(remaining):
            best_role = None
            best_gain = None
            for role in priority:
                # Diminishing return: keep growing top roles, but prevent single-role collapse.
                gain = weight_map[role] / (1.0 + float(counts[role]))
                if best_gain is None or gain > best_gain:
                    best_gain = gain
                    best_role = role
            if best_role is None:
                break
            counts[best_role] += 1

    slot_roles: List[str] = []
    for role in priority:
        slot_roles.extend([role] * counts[role])
    return role_names, slot_roles


def _estimate_motion_score(
    segment_features: torch.Tensor,
    motion_window: int,
) -> torch.Tensor:
    num_frames, num_visual_tokens, _ = segment_features.shape
    if num_frames <= 1:
        return segment_features.new_zeros((num_frames * num_visual_tokens,))

    normed = F.normalize(segment_features.float(), p=2, dim=-1, eps=1e-6)
    motion = torch.zeros((num_frames, num_visual_tokens), dtype=torch.float32, device=segment_features.device)
    counts = torch.zeros_like(motion)
    max_window = max(1, min(int(motion_window), num_frames - 1))

    for delta in range(1, max_window + 1):
        sim = torch.sum(normed[delta:] * normed[:-delta], dim=-1)
        change = ((1.0 - sim).clamp(min=0.0, max=2.0) * 0.5).float()
        motion[delta:] += change
        counts[delta:] += 1.0
        motion[:-delta] += change
        counts[:-delta] += 1.0

    motion = motion / counts.clamp_min(1.0)
    return motion.reshape(-1)


def _build_slot_role_scores(
    segment_features: torch.Tensor,
    flat_features: torch.Tensor,
    flat_attention: torch.Tensor,
    fused_scores: torch.Tensor,
    flashvid_config: FlashVidConfig,
) -> Dict[str, torch.Tensor]:
    num_frames, num_visual_tokens, _ = segment_features.shape
    normed_segment = F.normalize(segment_features.float(), p=2, dim=-1, eps=1e-6)
    normed_flat = normed_segment.reshape(num_frames * num_visual_tokens, -1)

    frame_centers = F.normalize(normed_segment.mean(dim=1), p=2, dim=-1, eps=1e-6)
    frame_ids = torch.arange(num_frames, device=flat_features.device).repeat_interleave(num_visual_tokens)
    frame_center_per_token = frame_centers[frame_ids]
    scene_center = F.normalize(normed_flat.mean(dim=0), p=2, dim=-1, eps=1e-6)

    scene_sim = torch.sum(normed_flat * scene_center.unsqueeze(0), dim=-1)
    frame_sim = torch.sum(normed_flat * frame_center_per_token, dim=-1)
    dispersion = ((1.0 - frame_sim).clamp(min=0.0, max=2.0) * 0.5).float()
    motion = _estimate_motion_score(
        segment_features=segment_features,
        motion_window=int(getattr(flashvid_config, "slot_motion_window", 1)),
    )

    visual = _normalize_scores(flat_attention.float())
    scene_core = _normalize_scores(scene_sim)
    background_core = _normalize_scores(frame_sim)
    motion_core = _normalize_scores(motion)
    interaction_core = _normalize_scores(0.7 * dispersion + 0.3 * motion_core)
    detail_core = _normalize_scores(dispersion)

    # Base role-specific scores (non-learning deterministic composition).
    role_scores: Dict[str, torch.Tensor] = {
        "scene": _normalize_scores(0.70 * scene_core + 0.30 * visual),
        "motion": _normalize_scores(0.65 * motion_core + 0.35 * visual),
        "interaction": _normalize_scores(0.50 * interaction_core + 0.30 * motion_core + 0.20 * visual),
        "background": _normalize_scores(0.70 * background_core + 0.30 * (1.0 - motion_core)),
        "detail": _normalize_scores(0.60 * detail_core + 0.40 * visual),
    }

    # Inject question-aware signal as a secondary term (never dominant).
    for role, score in role_scores.items():
        role_scores[role] = _normalize_scores(0.85 * score + 0.15 * fused_scores)
    return role_scores


def _select_slot_anchor_indices(
    flat_features: torch.Tensor,
    slot_roles: List[str],
    role_scores: Dict[str, torch.Tensor],
    fused_scores: torch.Tensor,
    flashvid_config: FlashVidConfig,
) -> torch.Tensor:
    num_tokens = flat_features.shape[0]
    if num_tokens == 0 or not slot_roles:
        return torch.empty((0,), dtype=torch.long, device=flat_features.device)

    normed_features = F.normalize(flat_features.float(), p=2, dim=-1, eps=1e-6)
    used = torch.zeros(num_tokens, dtype=torch.bool, device=flat_features.device)
    selected: List[int] = []
    coverage_gain = min(0.45, 0.08 * max(1, int(getattr(flashvid_config, "graph_topk", 4))))
    min_dist_to_selected: Optional[torch.Tensor] = None

    for role in slot_roles:
        score = role_scores.get(role, fused_scores).clone()
        score = score + 0.20 * fused_scores
        if min_dist_to_selected is not None:
            score = score + coverage_gain * _normalize_scores(min_dist_to_selected)
        score = score.masked_fill(used, -1e9)
        if torch.all(score <= -1e8):
            # Fallback: allow reused anchors only when no available candidates remain.
            score = role_scores.get(role, fused_scores) + 0.20 * fused_scores
        anchor_idx = int(torch.argmax(score).item())
        selected.append(anchor_idx)
        used[anchor_idx] = True

        anchor_feature = normed_features[anchor_idx]
        dist_new = (1.0 - torch.matmul(normed_features, anchor_feature)).clamp(min=0.0)
        if min_dist_to_selected is None:
            min_dist_to_selected = dist_new
        else:
            min_dist_to_selected = torch.minimum(min_dist_to_selected, dist_new)
    return torch.tensor(selected, dtype=torch.long, device=flat_features.device)


def _assign_tokens_to_slots(
    flat_features: torch.Tensor,
    slot_anchor_indices: torch.Tensor,
    slot_roles: List[str],
    role_scores: Dict[str, torch.Tensor],
    fused_scores: torch.Tensor,
    num_frames: int,
    num_visual_tokens: int,
    flashvid_config: FlashVidConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens = flat_features.shape[0]
    num_slots = int(slot_anchor_indices.numel())
    if num_slots == 0:
        return (
            torch.empty((0,), dtype=torch.long, device=flat_features.device),
            flat_features.new_zeros((0,)),
        )

    token_frame_ids = torch.arange(num_frames, device=flat_features.device).repeat_interleave(num_visual_tokens)
    token_local_ids = torch.arange(num_visual_tokens, device=flat_features.device).repeat(num_frames)
    _, grid_w = _resolve_grid_hw(num_visual_tokens, flashvid_config)
    token_y = token_local_ids // max(1, grid_w)
    token_x = token_local_ids % max(1, grid_w)

    slot_frame_ids = token_frame_ids[slot_anchor_indices]
    slot_y = token_y[slot_anchor_indices]
    slot_x = token_x[slot_anchor_indices]

    score_matrix = torch.stack([role_scores.get(role, fused_scores) for role in slot_roles], dim=1)
    score_matrix = score_matrix + 0.15 * fused_scores.unsqueeze(1)

    overlap_radius = max(0, int(getattr(flashvid_config, "slot_overlap_radius", 1)))
    temporal_radius = max(
        1,
        overlap_radius,
        int(getattr(flashvid_config, "graph_temporal_radius", 1)),
    )
    dt = torch.abs(token_frame_ids.unsqueeze(1) - slot_frame_ids.unsqueeze(0))

    # Hard temporal continuity mask: reduce position-token mismatch and improve answer stability.
    candidate_mask = dt <= temporal_radius
    has_candidate = candidate_mask.any(dim=1, keepdim=True)
    candidate_mask = torch.where(has_candidate, candidate_mask, torch.ones_like(candidate_mask))
    score_matrix = score_matrix.masked_fill(~candidate_mask, -1e9)

    if overlap_radius > 0:
        dy = torch.abs(token_y.unsqueeze(1) - slot_y.unsqueeze(0))
        dx = torch.abs(token_x.unsqueeze(1) - slot_x.unsqueeze(0))
        temporal_bonus = (dt <= overlap_radius).float() * 0.12
        spatial_bonus = ((dy <= overlap_radius) & (dx <= overlap_radius)).float() * 0.10
        drift_penalty = (dt.float() / max(1, num_frames - 1)) * 0.02
        score_matrix = score_matrix + temporal_bonus + spatial_bonus - drift_penalty

    k = min(4, num_slots)
    top_vals, top_indices = torch.topk(score_matrix, k=k, dim=1)
    assigned_slots = top_indices[:, 0]

    if num_slots > 1:
        tie_eps = max(0.0, float(getattr(flashvid_config, "slot_tiebreak_eps", 1e-4)))
        tie_mask = (top_vals[:, 0] - top_vals[:, 1]).abs() <= tie_eps if k > 1 else torch.zeros_like(top_vals[:, 0], dtype=torch.bool)
        if tie_mask.any():
            tied_rows = torch.where(tie_mask)[0]
            normed_features = F.normalize(flat_features.float(), p=2, dim=-1, eps=1e-6)
            anchor_features = normed_features[slot_anchor_indices]
            row_top_vals = top_vals[tied_rows]
            row_top_idx = top_indices[tied_rows]
            row_best = row_top_vals[:, :1]
            near_mask = (row_best - row_top_vals) <= tie_eps
            token_feat = normed_features[tied_rows].unsqueeze(1)  # (m,1,d)
            cand_feat = anchor_features[row_top_idx]  # (m,k,d)
            sims = torch.sum(token_feat * cand_feat, dim=-1)  # (m,k)
            sims = sims.masked_fill(~near_mask, -1e9)
            best_local = torch.argmax(sims, dim=-1)
            assigned_slots[tied_rows] = row_top_idx[torch.arange(best_local.shape[0], device=flat_features.device), best_local]

    selected_scores = score_matrix.gather(1, assigned_slots.unsqueeze(-1)).squeeze(-1)
    token_weights = (0.40 + 0.60 * _normalize_scores(selected_scores)).to(flat_features.dtype)
    return assigned_slots, token_weights


def _build_role_memory_tokens(
    flat_features: torch.Tensor,
    dropped_indices: torch.Tensor,
    residual_vectors: torch.Tensor,
    question_scores: Optional[torch.Tensor],
    memory_budget: int,
    token_role_ids: torch.Tensor,
    num_roles: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    feat_dim = flat_features.shape[-1]
    if memory_budget <= 0 or dropped_indices.numel() == 0:
        return (
            flat_features.new_zeros((0, feat_dim)),
            torch.empty((0,), dtype=torch.long, device=flat_features.device),
        )

    priorities = residual_vectors.float().norm(p=2, dim=-1)
    if question_scores is not None:
        priorities = priorities * (1.0 + question_scores[dropped_indices].float())
    role_ids = token_role_ids[dropped_indices]

    role_priority_sum = torch.zeros((num_roles,), dtype=torch.float32, device=flat_features.device)
    role_priority_sum.scatter_add_(0, role_ids, priorities)
    active_roles = torch.where(role_priority_sum > 0)[0]
    if active_roles.numel() == 0:
        return (
            flat_features.new_zeros((0, feat_dim)),
            torch.empty((0,), dtype=torch.long, device=flat_features.device),
        )

    alloc = torch.zeros((num_roles,), dtype=torch.long, device=flat_features.device)
    remaining = int(memory_budget)

    # Ensure role diversity first.
    top_roles = active_roles[torch.argsort(role_priority_sum[active_roles], descending=True)]
    for rid in top_roles:
        if remaining <= 0:
            break
        alloc[rid] += 1
        remaining -= 1

    # Allocate extra memory slots by residual priority.
    while remaining > 0:
        gains = role_priority_sum / (alloc.float() + 1.0)
        rid = int(torch.argmax(gains).item())
        alloc[rid] += 1
        remaining -= 1

    memory_tokens = []
    memory_indices = []
    for rid in range(num_roles):
        role_mem = int(alloc[rid].item())
        if role_mem <= 0:
            continue
        role_mask = role_ids == rid
        role_indices = dropped_indices[role_mask]
        if role_indices.numel() == 0:
            continue
        role_priorities = priorities[role_mask]
        order = torch.argsort(role_priorities, descending=True)
        role_indices = role_indices[order]
        role_priorities = role_priorities[order]
        role_features = flat_features[role_indices]

        num_chunks = min(role_mem, role_indices.numel())
        chunk_indices = torch.chunk(torch.arange(role_indices.numel(), device=flat_features.device), num_chunks)
        for split in chunk_indices:
            if split.numel() == 0:
                continue
            weights = role_priorities[split].to(flat_features.dtype).clamp_min(1e-6)
            feats = role_features[split]
            merged = torch.sum(feats * weights.unsqueeze(-1), dim=0) / weights.sum()
            memory_tokens.append(merged)
            memory_indices.append(role_indices[split[0]])

    if not memory_tokens:
        return (
            flat_features.new_zeros((0, feat_dim)),
            torch.empty((0,), dtype=torch.long, device=flat_features.device),
        )
    return torch.stack(memory_tokens, dim=0), torch.stack(memory_indices, dim=0)


def _segment_slot_memory_compression(
    segment_features: torch.Tensor,
    segment_global_indices: torch.Tensor,
    cls_attention: torch.Tensor,
    retention_ratio: float,
    flashvid_config: FlashVidConfig,
    question_features: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Slot-memory token->slot aggregation with role-aware residual memory tokens."""
    num_frames, num_visual_tokens, feat_dim = segment_features.shape
    num_tokens = num_frames * num_visual_tokens
    device = segment_features.device

    if num_tokens == 0:
        return (
            segment_features.new_zeros((0, feat_dim)),
            torch.empty((0,), dtype=torch.long, device=device),
        )

    flat_features = segment_features.reshape(num_tokens, feat_dim)
    flat_attention = cls_attention.reshape(num_tokens).float()
    flat_global_indices = segment_global_indices.reshape(num_tokens)

    effective_ratio = min(1.0, max(0.01, retention_ratio * float(getattr(flashvid_config, "expansion", 1.0))))
    target_budget = max(1, min(num_tokens, math.ceil(num_tokens * effective_ratio)))
    memory_budget = _resolve_memory_budget(
        num_tokens=num_tokens,
        target_budget=target_budget,
        flashvid_config=flashvid_config,
    )
    slot_budget = max(1, target_budget - memory_budget)
    requested_slot_cap = int(getattr(flashvid_config, "slot_max_per_segment", 0) or 0)
    if requested_slot_cap > 0:
        # Soft anti-collapse guard:
        # keep user-provided cap, but avoid forcing very large segments into tiny slot budgets.
        soft_fraction = float(getattr(flashvid_config, "slot_soft_cap_fraction", 0.35))
        soft_fraction = min(max(soft_fraction, 0.0), 1.0)
        soft_floor = max(1, int(math.ceil(slot_budget * soft_fraction)))
        effective_cap = max(requested_slot_cap, soft_floor)
        slot_budget = min(slot_budget, effective_cap)

    role_names, slot_roles = _allocate_slot_roles(
        slot_budget=slot_budget,
        flashvid_config=flashvid_config,
    )
    if not slot_roles:
        slot_roles = [role_names[0]]

    fused_scores, question_scores = _question_aware_scores(
        flat_features=flat_features,
        flat_attention=flat_attention,
        question_features=question_features,
        flashvid_config=flashvid_config,
    )
    role_scores = _build_slot_role_scores(
        segment_features=segment_features,
        flat_features=flat_features,
        flat_attention=flat_attention,
        fused_scores=fused_scores,
        flashvid_config=flashvid_config,
    )
    slot_anchor_indices = _select_slot_anchor_indices(
        flat_features=flat_features,
        slot_roles=slot_roles,
        role_scores=role_scores,
        fused_scores=fused_scores,
        flashvid_config=flashvid_config,
    )
    if slot_anchor_indices.numel() == 0:
        top1 = torch.topk(fused_scores, k=1, dim=0).indices
        slot_anchor_indices = top1.to(dtype=torch.long, device=device)
        slot_roles = [slot_roles[0]]

    assigned_slots, token_weights = _assign_tokens_to_slots(
        flat_features=flat_features,
        slot_anchor_indices=slot_anchor_indices,
        slot_roles=slot_roles,
        role_scores=role_scores,
        fused_scores=fused_scores,
        num_frames=num_frames,
        num_visual_tokens=num_visual_tokens,
        flashvid_config=flashvid_config,
    )

    num_slots = int(slot_anchor_indices.numel())
    slot_sum = torch.zeros((num_slots, feat_dim), dtype=flat_features.dtype, device=device)
    slot_weight = torch.zeros((num_slots, 1), dtype=flat_features.dtype, device=device)
    slot_sum.scatter_add_(
        dim=0,
        index=assigned_slots.unsqueeze(-1).expand(-1, feat_dim),
        src=flat_features * token_weights.unsqueeze(-1),
    )
    slot_weight.scatter_add_(
        dim=0,
        index=assigned_slots.unsqueeze(-1),
        src=token_weights.unsqueeze(-1),
    )
    slot_tokens = slot_sum / slot_weight.clamp_min(1e-6)
    # Precision-first: preserve slot anchors as a strong semantic prior.
    anchor_blend = float(getattr(flashvid_config, "slot_anchor_blend", 0.65))
    anchor_blend = min(max(anchor_blend, 0.0), 1.0)
    if anchor_blend > 0.0:
        anchor_tokens = flat_features[slot_anchor_indices]
        slot_tokens = (1.0 - anchor_blend) * slot_tokens + anchor_blend * anchor_tokens
    slot_global_indices = flat_global_indices[slot_anchor_indices]

    reconstructed = slot_tokens[assigned_slots]
    residual_vectors = flat_features - reconstructed
    anchor_mask = torch.zeros((num_tokens,), dtype=torch.bool, device=device)
    anchor_mask[slot_anchor_indices.unique()] = True
    dropped_indices = torch.where(~anchor_mask)[0]

    role_to_id = {role: idx for idx, role in enumerate(role_names)}
    slot_role_ids = torch.tensor([role_to_id.get(role, 0) for role in slot_roles], dtype=torch.long, device=device)
    token_role_ids = slot_role_ids[assigned_slots]

    memory_tokens, memory_token_indices = _build_role_memory_tokens(
        flat_features=flat_features,
        dropped_indices=dropped_indices,
        residual_vectors=residual_vectors[dropped_indices],
        question_scores=question_scores,
        memory_budget=memory_budget,
        token_role_ids=token_role_ids,
        num_roles=max(1, len(role_names)),
    )

    all_tokens = [slot_tokens]
    all_indices = [slot_global_indices]
    if memory_tokens.shape[0] > 0:
        all_tokens.append(memory_tokens)
        all_indices.append(flat_global_indices[memory_token_indices])

    return torch.cat(all_tokens, dim=0), torch.cat(all_indices, dim=0)


def _segment_graph_compression(
    segment_features: torch.Tensor,
    segment_global_indices: torch.Tensor,
    cls_attention: torch.Tensor,
    retention_ratio: float,
    flashvid_config: FlashVidConfig,
    question_features: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Legacy entrypoint kept for compatibility; graph is now an alias of slot-memory.
    return _segment_slot_memory_compression(
        segment_features=segment_features,
        segment_global_indices=segment_global_indices,
        cls_attention=cls_attention,
        retention_ratio=retention_ratio,
        flashvid_config=flashvid_config,
        question_features=question_features,
    )


def flashvid_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
    question_features: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_frames, num_visual_tokens, feat_dim = video_features.shape
    compression_variant = str(getattr(flashvid_config, "compression_variant", "flashvid")).lower()
    if compression_variant == "graph":
        compression_variant = "slot"
    retention_ratio = _resolve_effective_retention_ratio(
        video_features=video_features,
        question_features=question_features,
        flashvid_config=flashvid_config,
    )

    # 1. Partition the video frames into segments based on transition similarities.
    if flashvid_config.do_segment:
        segment_lengths = segment(
            video_features=video_features.mean(1),
            segment_threshold=flashvid_config.segment_threshold,
            min_segment_num=flashvid_config.min_segment_num,
            complementary_segment=flashvid_config.complementary_segment,
        )
    else:
        # Treat the whole video as a single segment.
        segment_lengths = torch.tensor([num_frames], dtype=torch.long, device=video_features.device)

    num_segments = segment_lengths.shape[0]
    global_indices = torch.arange(num_frames * num_visual_tokens, dtype=torch.long, device=video_features.device)

    # 2. Apply Attention and Diversity-based Token Selection(ADTS).
    token_budget = math.ceil(num_visual_tokens * retention_ratio * flashvid_config.expansion)
    num_attn_div_tokens = math.ceil(token_budget * flashvid_config.alpha)
    num_sttm_tokens = token_budget - num_attn_div_tokens
    # store in the config.
    flashvid_config.num_attn_div_tokens = num_attn_div_tokens
    flashvid_config.num_sttm_tokens = num_sttm_tokens

    all_segment_features = []
    all_segment_indices = []
    offset = 0
    for seg_idx in range(num_segments):
        seg_len = int(segment_lengths[seg_idx].item())
        segment_features = video_features[offset : offset + seg_len]
        segment_cls_attention = cls_attention[offset : offset + seg_len]
        segment_global_indices = global_indices.view(num_frames, num_visual_tokens)[offset : offset + seg_len]
        if compression_variant == "slot":
            segment_features, segment_global_indices = _segment_slot_memory_compression(
                segment_features=segment_features,
                segment_global_indices=segment_global_indices,
                cls_attention=segment_cls_attention,
                retention_ratio=retention_ratio,
                flashvid_config=flashvid_config,
                question_features=question_features,
            )
        else:
            segment_features, segment_global_indices = segment_compression(
                segment_features=segment_features,
                segment_global_indices=segment_global_indices,
                cls_attention=segment_cls_attention,
                flashvid_config=flashvid_config,
            )
        all_segment_features.append(segment_features)
        all_segment_indices.append(segment_global_indices)
        offset += seg_len
    final_tokens = torch.cat(all_segment_features, dim=0)  # (num_final_tokens, feat_dim)
    final_global_indices = torch.cat(all_segment_indices, dim=0)  # (num_final_tokens,)

    sorted_indices = final_global_indices.argsort()
    sorted_tokens = final_tokens[sorted_indices]  # Sort by global indices.
    # Store the final token length in the `flashvid_config`.
    flashvid_config.visual_token_length = sorted_tokens.shape[0]
    # print(f"#Visual Tokens After Vision-Side Compression : {flashvid_config.visual_token_length}")
    return sorted_tokens, final_global_indices[sorted_indices]


def segment_compression(
    segment_features: torch.Tensor,
    segment_global_indices: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compress the segment features by applying Temporal Average Merging (TAM) and Spatial Merging.

    Args:
        segment_features (torch.Tensor): The features of the video segment, of shape (num_frames, num_visual_tokens, feat_dim).
        segment_global_indices (torch.Tensor): The global indices of the video segment, of shape (num_frames, num_visual_tokens).
        cls_attention (torch.Tensor): [CLS] attentions used for per-frame token selection, of shape (num_frames, num_visual_tokens).
        flashvid_config (FlashVidConfig): The configuration for FlashVid.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The final tokens and their global indices after compression.
    """
    num_frames, num_visual_tokens, feat_dim = segment_features.shape

    # 1. Apply Attention and Diversity-based Token Selection (ADTS).
    if flashvid_config.alpha > 0:
        additional_kwargs = {"cls_attention": cls_attention} if "attn" in flashvid_config.token_selection_method else {}
        selected_features, selected_indices = ALL_TOKEN_SELECTION_METHOD[flashvid_config.token_selection_method](
            features=segment_features,
            num_retained_tokens=flashvid_config.num_attn_div_tokens,
            **additional_kwargs,
        )
        selected_global_indices = segment_global_indices.gather(1, index=selected_indices).view(-1)
    else:
        # No token selection
        selected_features = torch.tensor([]).to(segment_features)
        selected_indices = torch.tensor([]).to(segment_global_indices)
        selected_global_indices = torch.tensor([]).to(segment_global_indices)

    mask = torch.ones(num_frames, num_visual_tokens, dtype=torch.bool, device=segment_features.device)
    mask.scatter_(1, selected_indices, False)

    num_other_tokens = flashvid_config.num_sttm_tokens * num_frames
    # 1. Apply Temporal Average Merging (TAM) to the segment features.
    if num_other_tokens > 0 and flashvid_config.temporal_threshold < 1.0:
        if num_frames > 1:
            temp_merged_token_list, temp_merged_indices_list = spatiotemporal_compression(
                video_features=segment_features,
                temporal_threshold=flashvid_config.temporal_threshold,
                token_mask=mask,
                flashvid_config=flashvid_config,
            )
            temp_merged_global_indices_list = [segment_global_indices.view(num_frames, -1)[i][temp_merged_indices] for i, temp_merged_indices in enumerate(temp_merged_indices_list)]
        else:
            # Single-frame segment, no temporal merging needed.
            temp_merged_token_list = [segment_features[0]]
            temp_merged_global_indices_list = [segment_global_indices[0]]
    else:
        # No spatial-temporal merging needed.
        temp_merged_token_list = []
        temp_merged_global_indices_list = []

    all_tokens = [selected_features.view(-1, feat_dim)]
    all_global_indices = [selected_global_indices]
    # 2. Apply Spatial Merging to the tokens after temporal merging.
    if num_other_tokens > 0: ## Only apply spatial merging when there are STTM tokens.
        # Calculate adaptive contextual ratio.
        num_current_retained_tokens = sum(len(tokens) for tokens in temp_merged_token_list)
        adapative_contextual_ratio = num_other_tokens / num_current_retained_tokens
        if adapative_contextual_ratio < 1.0:
            num_frames_in_segment = len(temp_merged_token_list)
            max_num_tokens = max(len(tokens) for tokens in temp_merged_token_list)
            batched_tokens = torch.zeros((num_frames_in_segment, max_num_tokens, feat_dim), dtype=segment_features.dtype, device=segment_features.device)
            valid_token_mask = torch.zeros((num_frames_in_segment, max_num_tokens), dtype=torch.bool, device=segment_features.device)
            num_clusters_list = []
            k_list = []
            for i, temp_merged_tokens in enumerate(temp_merged_token_list):
                num_tokens = len(temp_merged_tokens)
                batched_tokens[i, :num_tokens] = temp_merged_tokens
                valid_token_mask[i, :num_tokens] = True
                num_clusters = math.ceil(num_tokens * adapative_contextual_ratio)
                num_clusters_list.append(num_clusters)
                k_list.append(min(num_clusters, 7))
            cluster_indices_list, cluster_center_indices_list = dpc_knn(
                features=batched_tokens,
                num_clusters=num_clusters_list,
                k=k_list,
                valid_token_mask=valid_token_mask,
            )
            for i, (temp_merged_tokens, temp_merged_global_indices) in enumerate(zip(temp_merged_token_list, temp_merged_global_indices_list)):
                num_clusters = num_clusters_list[i]
                if num_clusters > 0:
                    cluster_indices = cluster_indices_list[i][:len(temp_merged_tokens)]
                    cluster_center_indices = cluster_center_indices_list[i]
                    aggregated_tokens = torch.zeros((num_clusters, feat_dim), dtype=segment_features.dtype, device=segment_features.device)
                    aggregated_tokens.scatter_add_(0, cluster_indices.unsqueeze(-1).expand(-1, feat_dim), temp_merged_tokens)
                    cluster_counts = torch.bincount(cluster_indices, minlength=num_clusters).unsqueeze(-1).to(segment_features.dtype)
                    aggregated_tokens = aggregated_tokens / cluster_counts.clamp_min(1)

                    # Guard against rare invalid center indices from padded/batched clustering.
                    if temp_merged_global_indices.numel() > 0:
                        max_valid_center = temp_merged_global_indices.shape[0] - 1
                        cluster_center_indices = cluster_center_indices.clamp(min=0, max=max_valid_center)
                    else:
                        global_token_indices = torch.zeros((num_clusters,), dtype=torch.long, device=segment_features.device)
                        all_tokens.append(aggregated_tokens)
                        all_global_indices.append(global_token_indices)
                        continue

                    if cluster_center_indices.numel() < num_clusters:
                        pad_num = num_clusters - cluster_center_indices.numel()
                        pad_value = cluster_center_indices[-1] if cluster_center_indices.numel() > 0 else torch.tensor(0, device=segment_features.device)
                        pad_tensor = pad_value.repeat(pad_num)
                        cluster_center_indices = torch.cat([cluster_center_indices, pad_tensor], dim=0)
                    elif cluster_center_indices.numel() > num_clusters:
                        cluster_center_indices = cluster_center_indices[:num_clusters]

                    global_token_indices = temp_merged_global_indices[cluster_center_indices]
                else:
                    aggregated_tokens = temp_merged_tokens
                    global_token_indices = temp_merged_global_indices
                    
                all_tokens.append(aggregated_tokens)
                all_global_indices.append(global_token_indices)
        else:
            for temp_merged_tokens, temp_merged_global_indices in zip(temp_merged_token_list, temp_merged_global_indices_list):
                all_tokens.append(temp_merged_tokens)
                all_global_indices.append(temp_merged_global_indices)

    segment_final_tokens = torch.cat(all_tokens, dim=0)  # (num_final_tokens, feat_dim)
    segment_final_global_indices = torch.cat(all_global_indices, dim=0)  # (num_final_tokens,)
    return segment_final_tokens, segment_final_global_indices


def segment(
    video_features: torch.Tensor,
    segment_threshold: float,
    min_segment_num: int,
    complementary_segment: bool = True,
) -> torch.Tensor:
    """Segments the video features into distinct segments based on similarity.

    Args:
        video_features (torch.Tensor): The video features to segment.
        segment_threshold (float): The threshold for segmenting.
        min_segment_num (int): The minimum number of segments required.
        complementary_segment (int): Use complementary segmentation to ensure `min_segment_num` constraint.

    Returns:
        torch.Tensor: The lengths of the segments.
    """
    num_frames, feat_dim = video_features.shape

    # 0. Calculate transition similarities
    normed_video_features = video_features / video_features.norm(p=2, dim=-1, keepdim=True)
    transition_similarities = torch.sum(normed_video_features[:-1] * normed_video_features[1:], dim=-1)

    # 1. Find cut indices based on the segment threshold
    cut_indices = torch.where(transition_similarities < segment_threshold)[0]

    # 2. Ensure at least `min_segment_num` segments (Top-K or Uniform complementary segment)
    segment_lengths = additional_segment(
        cut_indices=cut_indices,
        num_frames=num_frames,
        min_segment_num=min_segment_num,
        transition_similarities=transition_similarities,
        segment_threshold=segment_threshold,
        complementary_segment=complementary_segment,
    )
    return segment_lengths


def additional_segment(
    cut_indices: torch.Tensor,
    num_frames: int,
    min_segment_num: int,
    transition_similarities: torch.Tensor,
    segment_threshold: float,
    complementary_segment: bool = True,
):
    num_segments = cut_indices.numel() + 1
    if num_segments < min_segment_num and complementary_segment:
        num_remaining_cut_indices = min_segment_num - num_segments
        transition_similarities[transition_similarities < segment_threshold] = 1.0
        complementary_cut_indices = torch.topk(transition_similarities, k=min(num_remaining_cut_indices, transition_similarities.shape[0]), largest=False).indices
        cut_indices = torch.cat([cut_indices, complementary_cut_indices]).sort().values

    padded_cut_indices = F.pad(cut_indices, (1, 1), value=0)
    padded_cut_indices[0] = -1
    padded_cut_indices[-1] = num_frames - 1
    segment_lengths = torch.diff(padded_cut_indices, n=1, dim=0)
    # print(f"segment lengths: {segment_lengths}")
    return segment_lengths


@torch.no_grad()
def dpc_knn(features: torch.Tensor, num_clusters: Union[int, List[int]], k: Union[int, List[int]] = 7, valid_token_mask: Optional[torch.Tensor] = None) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Union[torch.Tensor, List[torch.Tensor]]]:
    """Apply DPC-kNN clustering algorithm to the pooled image features, generating preliminary clustering result.

    Args:
        features (torch.Tensor): Pooled image features (temporal features), of shape (batch_size, seq_len, feat_dim).
        num_clusters (int or List[int]): The number of clusters. If a list, specifies the number of clusters for each batch element.
        k (int or List[int]): The number of nearest neighbors to consider for local density. Default is 7.
        valid_token_mask (Optional[torch.Tensor]): Boolean Mask indicating valid tokens, of shape (batch_size, seq_len). Default is None.

    Returns:
        Tuple[Union[torch.Tensor, List[torch.Tensor]], Union[torch.Tensor, List[torch.Tensor]]]: 
            Cluster indices and cluster center indices. If num_clusters is a list, returns lists of tensors.
    """
    invalid_token_mask = ~valid_token_mask if valid_token_mask is not None else None
    bsz, seq_len, feat_dim = features.shape

    # Calculate euclidean distance and local density
    dists = torch.cdist(features.float(), features.float()) / math.sqrt(feat_dim)

    # Mask out invalid tokens
    if valid_token_mask is not None:
        dists = torch.masked_fill(dists, invalid_token_mask.unsqueeze(1).expand(-1, seq_len, -1), dists.max() + 1)
        
    max_k = max(k) if isinstance(k, list) else k
    nearest_dist = torch.topk(dists, k=max_k, dim=-1, largest=False).values
    
    if isinstance(k, list):
        density = torch.empty((bsz, seq_len), device=features.device, dtype=features.dtype)
        for i in range(bsz):
            density[i] = torch.mean(-(nearest_dist[i, :, :k[i]]**2), dim=-1).exp()
    else:
        density = torch.mean(-(nearest_dist**2), dim=-1).exp()

    # ! [DISABLED] Add little random noise to ensure no tokens have the same density.
    # density = density + torch.rand_like(density, device=density.device, dtype=density.dtype) * 1e-6

    # Ensure the density of the empty token be 0
    if valid_token_mask is not None:
        density = torch.masked_fill(density, invalid_token_mask, 0.0)

    # Obtain the minimum distance to the point with higher density.
    mask = density[:, None, :] > density[:, :, None]
    max_dist = dists.view(bsz, -1).max(dim=-1)[0].view(-1, 1, 1)
    modified_dists = torch.where(mask, dists, max_dist)
    dist, _ = torch.min(modified_dists, dim=-1)

    # Calculate clustering score (clustering centers have the highest score)
    score = dist * density
    if isinstance(num_clusters, int):
        cluster_center_indices = torch.topk(score, k=num_clusters, dim=-1).indices
        # Obtain the distance matrix w.r.t cluster centers (batch_size, seq_len, num_clusters)
        dists = torch.gather(dists, dim=-1, index=cluster_center_indices.unsqueeze(1).expand(-1, seq_len, -1))
        cluster_indices = torch.argmin(dists, dim=-1)
        # Ensure each cluster center to merge with itself
        cluster_indices.scatter_(
            dim=-1,
            index=cluster_center_indices,
            src=torch.arange(num_clusters).to(cluster_indices).unsqueeze(0).expand(bsz, -1),
        )
        return cluster_indices, cluster_center_indices
    else:
        cluster_indices_list = []
        cluster_center_indices_list = []
        for i in range(bsz):
            k_i = num_clusters[i]
            if k_i == 0:
                cluster_center_indices_list.append(torch.tensor([], dtype=torch.long, device=features.device))
                cluster_indices_list.append(torch.zeros(seq_len, dtype=torch.long, device=features.device))
                continue
            cc_idx = torch.topk(score[i], k=k_i, dim=-1).indices
            cluster_center_indices_list.append(cc_idx)
            dists_i = torch.gather(dists[i], dim=-1, index=cc_idx.unsqueeze(0).expand(seq_len, -1))
            c_idx = torch.argmin(dists_i, dim=-1)
            c_idx.scatter_(
                dim=-1,
                index=cc_idx,
                src=torch.arange(k_i).to(c_idx),
            )
            cluster_indices_list.append(c_idx)
        return cluster_indices_list, cluster_center_indices_list


def spatiotemporal_compression(
    video_features: torch.Tensor,
    temporal_threshold: float,
    token_mask: torch.Tensor,
    flashvid_config: FlashVidConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_frames, num_visual_tokens, feat_dim = video_features.shape
    # since we pass the whole segment features, the lower bound should contain ADTS tokens.
    lower_bound = (flashvid_config.num_attn_div_tokens + flashvid_config.num_sttm_tokens) * num_frames
    normed_video_features = video_features / video_features.norm(p=2, dim=-1, keepdim=True)
    cosine_similarities = torch.bmm(normed_video_features[1:], normed_video_features[:-1].transpose(1, 2))
    # Mask out the selected tokens.
    cosine_similarities[~token_mask[1:].unsqueeze(-1).expand(-1, -1, num_visual_tokens)] = -1.0
    cosine_similarities[~token_mask[:-1].unsqueeze(1).expand(-1, num_visual_tokens, -1)] = -1.0

    max_sims, max_sim_indices = torch.max(cosine_similarities, dim=-1)

    padded_max_sims = F.pad(max_sims, (0, 0, 1, 0), value=-1)
    padded_max_sim_indices = F.pad(max_sim_indices, (0, 0, 1, 0), value=-1)

    token_counts = torch.ones(num_frames, num_visual_tokens).to(video_features)
    mask = padded_max_sims > temporal_threshold
    retaining_token_mask = ~mask

    # Ensure the number of retained tokens after TAM does not exceed the lower bound.
    if retaining_token_mask.int().sum() < lower_bound:
        soft_threshold = padded_max_sims.view(-1).topk(k=(num_frames * num_visual_tokens) - lower_bound).values[-1]
        soft_threshold = max(soft_threshold, -1.0 + 1e-6)
        mask = padded_max_sims > soft_threshold
        retaining_token_mask = ~mask

    for frame_idx in range(num_frames - 1, -1, -1):
        frame_features = video_features[frame_idx]
        frame_token_counts = token_counts[frame_idx]
        frame_max_sim_indices = padded_max_sim_indices[frame_idx]

        # Apply spatiotemporal average merging.
        tokens_to_merge = frame_features[~mask[frame_idx]]
        to_merge_token_counts = frame_token_counts[~mask[frame_idx]]
        if tokens_to_merge.numel() > 0:
            aggregated_tokens = tokens_to_merge / to_merge_token_counts.unsqueeze(-1).to(tokens_to_merge.dtype)
            video_features[frame_idx][~mask[frame_idx]] = aggregated_tokens
            token_counts[frame_idx][~mask[frame_idx]] = 1

        # other tokens are connected to the previous frame's tokens
        other_tokens = frame_features[mask[frame_idx]]
        if other_tokens.numel() > 0:
            # Distribute other tokens to the previous frame's tokens (anchor tokens)
            anchor_token_indices = frame_max_sim_indices[mask[frame_idx]]
            aggregated_tokens = torch.zeros((num_visual_tokens, feat_dim), dtype=video_features.dtype, device=video_features.device)
            aggregated_tokens.scatter_add_(0, anchor_token_indices.unsqueeze(-1).expand(-1, feat_dim), other_tokens)  # (num_visual_tokens, feat_dim)
            aggregated_token_counts = torch.bincount(anchor_token_indices, minlength=num_visual_tokens).to(video_features.dtype)  # (num_visual_tokens,)
            video_features[frame_idx - 1] += aggregated_tokens
            token_counts[frame_idx - 1] += aggregated_token_counts
            token_counts[frame_idx][mask[frame_idx]] = 0

    # Filter final tokens
    final_tokens = []
    retained_token_indices = []
    for i in range(num_frames):
        frame_mask = retaining_token_mask[i] & token_mask[i]
        frame_retained_tokens = video_features[i][frame_mask]  # (frame_retained_tokens_num, feat_dim)
        frame_retained_indices = torch.where(frame_mask)[0]  # (frame_retained_tokens_num,)
        final_tokens.append(frame_retained_tokens)
        retained_token_indices.append(frame_retained_indices)

    return final_tokens, retained_token_indices


def maybe_apply_decode_policy(
    hidden_states: torch.Tensor,
    causal_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    cache_position: Optional[torch.Tensor],
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    flashvid_config: FlashVidConfig,
    layer_idx: int,
    is_prefill: bool,
):
    """Decode-stage scaffold for Route3; default policy keeps behavior unchanged."""
    if is_prefill:
        return hidden_states, causal_mask, position_ids, cache_position, position_embeddings

    policy = str(getattr(flashvid_config, "decode_policy", "none") or "none").strip().lower()
    if policy in ("none", "off", "disabled"):
        return hidden_states, causal_mask, position_ids, cache_position, position_embeddings

    # Reserved policy name for future implementation; currently no-op with strict validation.
    if policy == "static_kv_budget":
        budget = float(getattr(flashvid_config, "decode_kv_budget_ratio", 1.0))
        update_interval = int(getattr(flashvid_config, "decode_update_interval", 4))
        start_layer = int(getattr(flashvid_config, "decode_start_layer", 0))
        if not (0.0 < budget <= 1.0):
            raise ValueError(f"decode_kv_budget_ratio must be in (0, 1], got {budget}")
        if update_interval <= 0:
            raise ValueError(f"decode_update_interval must be > 0, got {update_interval}")
        if start_layer < 0:
            raise ValueError(f"decode_start_layer must be >= 0, got {start_layer}")
        return hidden_states, causal_mask, position_ids, cache_position, position_embeddings

    raise ValueError(
        f"Unsupported decode_policy={policy!r}. "
        "Supported policies: none, static_kv_budget (scaffold/no-op)."
    )


def fastv_prune(
    hidden_states: torch.Tensor,
    causal_mask: Optional[torch.Tensor],
    attentions: Optional[torch.Tensor],
    cache_position: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    flashvid_config: FlashVidConfig,
    visual_pos_masks: Optional[torch.BoolTensor] = None,
):
    bsz, seq_length, _ = hidden_states.shape
    device = hidden_states.device
    # No-op shortcut: disable inner-LLM pruning when retention ratio is >= 1.
    if float(getattr(flashvid_config, "llm_retention_ratio", 1.0)) >= 0.9999:
        keep_indices = torch.arange(seq_length, device=device, dtype=torch.long)
        return hidden_states, causal_mask, position_ids, cache_position, position_embeddings, keep_indices

    # Obtain FlashVid arguments.
    visual_token_start_index = flashvid_config.visual_token_start_index
    visual_token_length = flashvid_config.visual_token_length
    visual_token_end_index = visual_token_start_index + visual_token_length

    retention_ratio = flashvid_config.llm_retention_ratio

    # Compatible to LLaVA-OneVision.
    if visual_pos_masks is None:
        visual_pos_masks = torch.zeros((bsz, seq_length), dtype=torch.bool, device=device)
        visual_pos_masks[:, visual_token_start_index:visual_token_end_index] = True
    non_visual_pos_masks = ~visual_pos_masks

    visual_features = hidden_states[visual_pos_masks, :]
    visual_global_indices = torch.where(visual_pos_masks[0])[0]
    non_visual_global_indices = torch.where(non_visual_pos_masks[0])[0]
    if visual_features.numel() == 0 or visual_global_indices.numel() == 0:
        return hidden_states, causal_mask, position_ids, cache_position, position_embeddings, torch.arange(seq_length, device=device)

    num_retained_tokens = math.ceil(visual_token_length * retention_ratio)
    num_retained_tokens = min(max(1, num_retained_tokens), int(visual_global_indices.numel()))
    attn = torch.mean(attentions[:, :, -1, :], dim=1)[visual_pos_masks]

    _, topk_indices = attn_based_token_selection(
        features=visual_features.unsqueeze(0),
        cls_attention=attn.unsqueeze(0),
        num_retained_tokens=num_retained_tokens,
    )
    topk_indices = topk_indices.squeeze(0)
    topk_indices = topk_indices.clamp(min=0, max=max(0, int(visual_global_indices.numel()) - 1))
    all_global_indices = [non_visual_global_indices, visual_global_indices[topk_indices]]
    keep_indices = torch.sort(torch.cat(all_global_indices).unique()).values
    keep_indices = keep_indices[keep_indices < seq_length]
    if keep_indices.numel() == 0:
        keep_indices = torch.arange(seq_length, device=device)

    # Filter
    hidden_states = hidden_states[:, keep_indices]
    # Keep RoPE positions by selecting with keep_indices, but reset cache positions to
    # contiguous range to avoid out-of-range indexing in cache update paths.
    if cache_position is None:
        cache_position = torch.arange(hidden_states.shape[1], device=device, dtype=torch.long)
    else:
        cache_position = torch.arange(hidden_states.shape[1], device=device, dtype=cache_position.dtype)
    position_ids = keep_indices.unsqueeze(0) if position_ids is None else position_ids[..., keep_indices].contiguous()
    position_embeddings = (
        position_embeddings[0][..., keep_indices, :].contiguous(),
        position_embeddings[1][..., keep_indices, :].contiguous(),
    )

    new_seq_length = hidden_states.shape[1]
    if causal_mask is not None:
        # Use index-select instead of naive truncation because keep_indices is a sparse subset.
        causal_mask = causal_mask.index_select(2, keep_indices).index_select(3, keep_indices)
    # Update flashvid config.
    flashvid_config.visual_token_length = num_retained_tokens
    return hidden_states, causal_mask, position_ids, cache_position, position_embeddings, keep_indices
