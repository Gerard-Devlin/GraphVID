from __future__ import annotations

from typing import List, Tuple

import math
import torch
from torch.nn import functional as F

from .configuration_flashvid import FlashVidConfig
from .graphvid import _choose_position_node, _grid_hw, _neighbor_table, _normalize_on_mask, _spatial_detail_score


def _bool_config(config: FlashVidConfig, name: str, default: bool) -> bool:
    value = getattr(config, name, None)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


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


def _score_preset_code(preset: str) -> int:
    name = str(preset or "event_v2").strip().lower()
    if name in ("base", "legacy"):
        return 0
    if name in ("event_v1", "event"):
        return 1
    return 2


def _component_radius(node_features: torch.Tensor, members: list[int]) -> float:
    if len(members) <= 1:
        return 0.0
    ids = torch.tensor(members, dtype=torch.long, device=node_features.device)
    feats = node_features[ids].float()
    center = F.normalize(feats.mean(dim=0), p=2, dim=0, eps=1e-6)
    radius = (1.0 - torch.sum(feats * center.unsqueeze(0), dim=-1)).clamp_min(0.0).max()
    return float(radius.item())


def _medoid_node(node_features: torch.Tensor, members: list[int]) -> int:
    if len(members) <= 1:
        return members[0]
    ids = torch.tensor(members, dtype=torch.long, device=node_features.device)
    feats = node_features[ids].float()
    sims = feats @ feats.transpose(0, 1)
    return members[int(torch.argmax(sims.sum(dim=1)).item())]


def _weighted_mean_feature(
    node_raw_features: torch.Tensor,
    members: list[int],
    protection_cpu: list[float],
) -> torch.Tensor:
    ids = torch.tensor(members, dtype=torch.long, device=node_raw_features.device)
    weights = torch.tensor(
        [max(1.0e-4, float(protection_cpu[m])) for m in members],
        dtype=node_raw_features.dtype,
        device=node_raw_features.device,
    )
    feats = node_raw_features[ids]
    return (feats * weights.unsqueeze(-1)).sum(dim=0) / weights.sum().clamp_min(1.0e-6)


def _mean_feature(node_raw_features: torch.Tensor, members: list[int]) -> torch.Tensor:
    ids = torch.tensor(members, dtype=torch.long, device=node_raw_features.device)
    return node_raw_features[ids].mean(dim=0)


def _reset_graft_metrics(config: FlashVidConfig) -> None:
    public_defaults = {
        "last_graft_num_nodes": 0.0,
        "last_graft_target_components": 0.0,
        "last_graft_protected_count": 0.0,
        "last_graft_entries_before_budget": 0.0,
        "last_graft_entries_after_budget": 0.0,
        "last_graft_scene_threshold": 0.0,
        "last_graft_global_topk": 0.0,
        "last_graft_anchor_ratio": None,
        "last_graft_input_is_residual": 1.0,
        "last_graft_budget_diversity_weight": 0.0,
        "last_graft_score_preset_code": 2.0,
        "last_graft_budget_correction_active": 1.0,
        "last_graft_protected_kept_count": 0.0,
        "last_graft_component_count": 0.0,
        "last_graft_avg_component_size": None,
        "last_graft_max_component_size": 0.0,
        "last_graft_radius_mean": None,
        "last_graft_radius_max": 0.0,
        "last_graft_edges_considered": 0.0,
        "last_graft_edges_accepted": 0.0,
        "last_graft_mutual_rejected": 0.0,
        "last_graft_radius_rejected": 0.0,
        "last_graft_capacity_rejected": 0.0,
        "last_graft_same_frame_rejected": 0.0,
    }
    for key, value in public_defaults.items():
        setattr(config, key, value)
    private_defaults = {
        "_graft_component_count": 0.0,
        "_graft_size_sum": 0.0,
        "_graft_radius_sum": 0.0,
        "_graft_radius_count": 0.0,
        "_graft_max_component_size": 0.0,
        "_graft_radius_max": 0.0,
        "_graft_edges_considered": 0.0,
        "_graft_edges_accepted": 0.0,
        "_graft_mutual_rejected": 0.0,
        "_graft_radius_rejected": 0.0,
        "_graft_capacity_rejected": 0.0,
        "_graft_same_frame_rejected": 0.0,
    }
    for key, value in private_defaults.items():
        setattr(config, key, value)


def _ensure_graft_metrics(config: FlashVidConfig) -> None:
    if not hasattr(config, "_graft_component_count"):
        _reset_graft_metrics(config)


def _accumulate_graft_metrics(
    config: FlashVidConfig,
    *,
    component_sizes: list[int],
    radii: list[float],
    edges_considered: int,
    edges_accepted: int,
    mutual_rejected: int,
    radius_rejected: int,
    capacity_rejected: int,
    same_frame_rejected: int,
) -> None:
    _ensure_graft_metrics(config)
    setattr(config, "_graft_component_count", float(getattr(config, "_graft_component_count", 0.0)) + float(len(component_sizes)))
    setattr(config, "_graft_size_sum", float(getattr(config, "_graft_size_sum", 0.0)) + float(sum(component_sizes)))
    setattr(config, "_graft_radius_sum", float(getattr(config, "_graft_radius_sum", 0.0)) + float(sum(radii)))
    setattr(config, "_graft_radius_count", float(getattr(config, "_graft_radius_count", 0.0)) + float(len(radii)))
    setattr(
        config,
        "_graft_max_component_size",
        max(float(getattr(config, "_graft_max_component_size", 0.0)), float(max(component_sizes) if component_sizes else 0)),
    )
    setattr(
        config,
        "_graft_radius_max",
        max(float(getattr(config, "_graft_radius_max", 0.0)), float(max(radii) if radii else 0.0)),
    )
    for key, value in (
        ("_graft_edges_considered", edges_considered),
        ("_graft_edges_accepted", edges_accepted),
        ("_graft_mutual_rejected", mutual_rejected),
        ("_graft_radius_rejected", radius_rejected),
        ("_graft_capacity_rejected", capacity_rejected),
        ("_graft_same_frame_rejected", same_frame_rejected),
    ):
        setattr(config, key, float(getattr(config, key, 0.0)) + float(value))

    component_count = float(getattr(config, "_graft_component_count", 0.0))
    radius_count = float(getattr(config, "_graft_radius_count", 0.0))
    setattr(config, "last_graft_component_count", component_count)
    setattr(
        config,
        "last_graft_avg_component_size",
        float(getattr(config, "_graft_size_sum", 0.0)) / component_count if component_count > 0 else None,
    )
    setattr(config, "last_graft_max_component_size", float(getattr(config, "_graft_max_component_size", 0.0)))
    setattr(
        config,
        "last_graft_radius_mean",
        float(getattr(config, "_graft_radius_sum", 0.0)) / radius_count if radius_count > 0 else None,
    )
    setattr(config, "last_graft_radius_max", float(getattr(config, "_graft_radius_max", 0.0)))
    setattr(config, "last_graft_edges_considered", float(getattr(config, "_graft_edges_considered", 0.0)))
    setattr(config, "last_graft_edges_accepted", float(getattr(config, "_graft_edges_accepted", 0.0)))
    setattr(config, "last_graft_mutual_rejected", float(getattr(config, "_graft_mutual_rejected", 0.0)))
    setattr(config, "last_graft_radius_rejected", float(getattr(config, "_graft_radius_rejected", 0.0)))
    setattr(config, "last_graft_capacity_rejected", float(getattr(config, "_graft_capacity_rejected", 0.0)))
    setattr(config, "last_graft_same_frame_rejected", float(getattr(config, "_graft_same_frame_rejected", 0.0)))


def _collect_candidate_pairs(
    normed: torch.Tensor,
    token_mask: torch.Tensor,
    node_id: torch.Tensor,
    neighbor_idx: torch.Tensor,
    neighbor_valid: torch.Tensor,
    src_frame: int,
    dst_frame: int,
    topk: int,
    global_topk: int,
) -> dict[tuple[int, int], float]:
    """Collect local+global top-k candidate pairs as (src_node, dst_node) -> sim."""
    pairs: dict[tuple[int, int], float] = {}
    src_valid = torch.where(token_mask[src_frame])[0]
    dst_valid = torch.where(token_mask[dst_frame])[0]
    if src_valid.numel() == 0 or dst_valid.numel() == 0:
        return pairs

    neigh = neighbor_idx[src_valid]
    valid = neighbor_valid[src_valid] & token_mask[dst_frame, neigh]
    sims = torch.sum(normed[dst_frame, neigh] * normed[src_frame, src_valid].unsqueeze(1), dim=-1)
    sims = sims.masked_fill(~valid, -1.0)
    k = min(topk, int(sims.shape[1]))
    if k > 0:
        vals, ids = torch.topk(sims, k=k, dim=1, largest=True)
        src_nodes = node_id[src_frame, src_valid].unsqueeze(1).expand_as(vals)
        dst_nodes = node_id[dst_frame, neigh.gather(1, ids)]
        edge_valid = (vals > -0.5) & (src_nodes >= 0) & (dst_nodes >= 0)
        for sim, src, dst in zip(
            vals[edge_valid].detach().cpu().tolist(),
            src_nodes[edge_valid].detach().cpu().tolist(),
            dst_nodes[edge_valid].detach().cpu().tolist(),
        ):
            pairs[(int(src), int(dst))] = max(float(sim), pairs.get((int(src), int(dst)), -1.0))

    gk = min(max(0, global_topk), int(dst_valid.numel()))
    if gk > 0:
        global_sims = torch.matmul(normed[src_frame, src_valid].float(), normed[dst_frame, dst_valid].float().transpose(0, 1))
        vals, ids = torch.topk(global_sims, k=gk, dim=1, largest=True)
        src_nodes = node_id[src_frame, src_valid].unsqueeze(1).expand_as(vals)
        dst_nodes = node_id[dst_frame, dst_valid[ids]]
        edge_valid = (vals > -0.5) & (src_nodes >= 0) & (dst_nodes >= 0)
        for sim, src, dst in zip(
            vals[edge_valid].detach().cpu().tolist(),
            src_nodes[edge_valid].detach().cpu().tolist(),
            dst_nodes[edge_valid].detach().cpu().tolist(),
        ):
            pairs[(int(src), int(dst))] = max(float(sim), pairs.get((int(src), int(dst)), -1.0))
    return pairs


def graft_spatiotemporal_compression(
    video_features: torch.Tensor,
    temporal_threshold: float,
    token_mask: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """GRAFT-VID constrained temporal forest compression.

    This path uses graph edges only as candidates. The final structure is a
    constrained temporal forest: no protected-node merging, optional mutual-kNN,
    one token per frame in each component, capacity-limited parents, and a hard
    component-radius check before union.
    """
    num_frames, num_visual_tokens, feat_dim = video_features.shape
    device = video_features.device
    candidate_positions = torch.where(token_mask)
    num_nodes = int(candidate_positions[0].numel())
    if num_nodes == 0:
        _accumulate_graft_metrics(
            flashvid_config,
            component_sizes=[],
            radii=[],
            edges_considered=0,
            edges_accepted=0,
            mutual_rejected=0,
            radius_rejected=0,
            capacity_rejected=0,
            same_frame_rejected=0,
        )
        empty_tokens = torch.empty((0, feat_dim), dtype=video_features.dtype, device=device)
        empty_indices = torch.empty((0,), dtype=torch.long, device=device)
        return [empty_tokens for _ in range(num_frames)], [empty_indices for _ in range(num_frames)]

    num_sttm_tokens_value = getattr(flashvid_config, "num_sttm_tokens", None)
    if num_sttm_tokens_value is None:
        fallback_ratio = min(max(_cfg_float(flashvid_config, "retention_ratio", 0.10), 0.0), 1.0)
        target_components = int(math.ceil(num_nodes * fallback_ratio))
    else:
        per_frame_budget = int(num_sttm_tokens_value)
        if per_frame_budget <= 0:
            fallback_ratio = min(max(_cfg_float(flashvid_config, "retention_ratio", 0.10), 0.0), 1.0)
            target_components = int(math.ceil(num_nodes * fallback_ratio))
        else:
            target_components = per_frame_budget * num_frames
    target_components = min(num_nodes, max(1, target_components))

    node_id = torch.full((num_frames, num_visual_tokens), -1, dtype=torch.long, device=device)
    node_id[candidate_positions] = torch.arange(num_nodes, dtype=torch.long, device=device)
    node_frames = candidate_positions[0].detach().cpu().tolist()
    node_tokens = candidate_positions[1].detach().cpu().tolist()

    normed = F.normalize(video_features.float(), p=2, dim=-1, eps=1e-6)
    node_features = normed[candidate_positions]
    node_raw_features = video_features[candidate_positions]

    attn = cls_attention.float()
    attn_norm = _normalize_on_mask(attn, token_mask)
    h, w = _grid_hw(num_visual_tokens, flashvid_config)
    radius = max(0, _cfg_int(flashvid_config, "graft_temporal_radius", 1))
    topk = max(1, _cfg_int(flashvid_config, "graft_temporal_topk", 3))
    global_topk = max(0, _cfg_int(flashvid_config, "graft_global_topk", topk))
    temporal_skip = max(1, _cfg_int(flashvid_config, "graft_temporal_skip", 1))
    neighbor_idx, neighbor_valid = _neighbor_table(num_visual_tokens, h, w, radius, device)

    frame_embeds: list[torch.Tensor] = []
    for frame_idx in range(num_frames):
        valid = token_mask[frame_idx]
        if bool(valid.any().item()):
            frame_embeds.append(normed[frame_idx, valid].float().mean(dim=0))
        else:
            frame_embeds.append(normed[frame_idx].float().mean(dim=0))
    frame_embeds_t = F.normalize(torch.stack(frame_embeds, dim=0), p=2, dim=-1, eps=1e-6)

    novelty = torch.ones((num_frames, num_visual_tokens), dtype=torch.float32, device=device)
    for lag in range(1, temporal_skip + 1):
        for frame_idx in range(lag, num_frames):
            prev_frame = frame_idx - lag
            cur_valid = torch.where(token_mask[frame_idx])[0]
            if cur_valid.numel() == 0:
                continue
            neigh = neighbor_idx[cur_valid]
            valid = neighbor_valid[cur_valid] & token_mask[prev_frame, neigh]
            sims = torch.sum(normed[prev_frame, neigh] * normed[frame_idx, cur_valid].unsqueeze(1), dim=-1)
            sims = sims.masked_fill(~valid, -1.0)
            max_sim = sims.max(dim=1).values
            prev_valid = torch.where(token_mask[prev_frame])[0]
            if global_topk > 0 and prev_valid.numel() > 0:
                global_sims = torch.matmul(
                    normed[frame_idx, cur_valid].float(),
                    normed[prev_frame, prev_valid].float().transpose(0, 1),
                )
                max_sim = torch.maximum(max_sim, global_sims.max(dim=1).values)
            cur_novelty = (1.0 - max_sim).clamp(0.0, 2.0) * 0.5
            novelty[frame_idx, cur_valid] = torch.minimum(novelty[frame_idx, cur_valid], cur_novelty)
    novelty_norm = _normalize_on_mask(novelty, token_mask)
    detail = _spatial_detail_score(normed, neighbor_idx, neighbor_valid)
    detail_norm = _normalize_on_mask(detail, token_mask)
    event_rel = torch.einsum("fnd,gd->fng", normed.float(), frame_embeds_t.float()).mean(dim=-1)
    event_rel_norm = _normalize_on_mask(event_rel, token_mask)
    score_preset = str(getattr(flashvid_config, "graft_score_preset", "event_v2") or "event_v2").strip().lower()
    score_preset_code = _score_preset_code(score_preset)
    if score_preset_code == 0:
        protection_map = (0.65 * attn_norm + 0.25 * novelty_norm + 0.10 * detail_norm).clamp(0.0, 1.0)
    elif score_preset_code == 1:
        protection_map = (
            0.45 * attn_norm
            + 0.25 * event_rel_norm
            + 0.20 * novelty_norm
            + 0.10 * detail_norm
        ).clamp(0.0, 1.0)
    else:
        protection_map = (
            0.50 * attn_norm
            + 0.30 * event_rel_norm
            + 0.15 * novelty_norm
            + 0.05 * detail_norm
        ).clamp(0.0, 1.0)
    protection = protection_map[token_mask]
    protection_cpu = protection.detach().float().cpu().tolist()

    protected = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    input_is_residual = _bool_config(flashvid_config, "graft_input_is_residual", True)
    default_anchor_ratio = 0.15 if input_is_residual else 0.65
    anchor_ratio = min(max(_cfg_float(flashvid_config, "graft_anchor_ratio", default_anchor_ratio), 0.0), 0.95)
    protect_budget = min(num_nodes, int(math.ceil(target_components * anchor_ratio)))
    valid_frames = [frame_idx for frame_idx in range(num_frames) if bool(token_mask[frame_idx].any().item())]
    used_protect = 0
    if protect_budget > 0 and valid_frames:
        base_budget = protect_budget // len(valid_frames)
        extra_budget = protect_budget % len(valid_frames)
        for offset, frame_idx in enumerate(valid_frames):
            frame_budget = base_budget + (1 if offset < extra_budget else 0)
            if frame_budget <= 0:
                continue
            frame_nodes = node_id[frame_idx, token_mask[frame_idx]]
            frame_nodes = frame_nodes[frame_nodes >= 0]
            if frame_nodes.numel() == 0:
                continue
            k = min(int(frame_nodes.numel()), frame_budget)
            local_scores = protection[frame_nodes]
            chosen = frame_nodes[torch.topk(local_scores, k=k, largest=True).indices]
            protected[chosen] = True
            used_protect += int(chosen.numel())
        if used_protect < protect_budget:
            remaining = torch.where(~protected)[0]
            if remaining.numel() > 0:
                k = min(int(remaining.numel()), protect_budget - used_protect)
                chosen = remaining[torch.topk(protection[remaining], k=k, largest=True).indices]
                protected[chosen] = True
    protected_cpu = protected.detach().cpu().tolist()
    merge_mask = token_mask.clone()
    merge_mask[candidate_positions] = ~protected

    edge_threshold = _cfg_float(flashvid_config, "graft_edge_threshold", 0.80)
    if edge_threshold <= 0.0:
        edge_threshold = float(temporal_threshold)
    mutual_knn = _bool_config(flashvid_config, "graft_mutual_knn", True)
    one_token_per_frame = _bool_config(flashvid_config, "graft_one_token_per_frame", True)
    component_radius_eps = max(0.0, _cfg_float(flashvid_config, "graft_component_radius_eps", 0.12))
    split_radius_eps = max(component_radius_eps, _cfg_float(flashvid_config, "graft_split_radius_eps", 0.20))
    parent_capacity = max(1, _cfg_int(flashvid_config, "graft_parent_capacity", 1))
    spatial_penalty = max(0.0, _cfg_float(flashvid_config, "graft_spatial_penalty", 0.10))
    importance_penalty = max(0.0, _cfg_float(flashvid_config, "graft_importance_penalty", 0.05))
    hub_penalty = max(0.0, _cfg_float(flashvid_config, "graft_hub_penalty", 0.05))
    adaptive_aggregation = _bool_config(flashvid_config, "graft_adaptive_aggregation", True)
    scene_threshold = max(0.0, _cfg_float(flashvid_config, "graft_scene_threshold", 0.0))
    min_tokens_per_frame = max(0, _cfg_int(flashvid_config, "graft_min_tokens_per_frame", 0))
    budget_correction = _bool_config(flashvid_config, "graft_budget_correction", True)
    budget_diversity_weight = max(0.0, _cfg_float(flashvid_config, "graft_budget_diversity_weight", 0.35))
    union_radius_eps = split_radius_eps if adaptive_aggregation else component_radius_eps

    raw_edges: list[tuple[float, int, int, float, float, float]] = []
    mutual_rejected = 0
    for lag in range(1, temporal_skip + 1):
        for frame_idx in range(lag, num_frames):
            prev_frame = frame_idx - lag
            if scene_threshold > 0.0:
                scene_sim = float(torch.sum(frame_embeds_t[frame_idx] * frame_embeds_t[prev_frame]).item())
                if scene_sim < scene_threshold:
                    continue
            forward_pairs = _collect_candidate_pairs(
                normed,
                merge_mask,
                node_id,
                neighbor_idx,
                neighbor_valid,
                frame_idx,
                prev_frame,
                topk,
                global_topk,
            )
            if not forward_pairs:
                continue
            reverse_pairs = (
                _collect_candidate_pairs(
                    normed,
                    merge_mask,
                    node_id,
                    neighbor_idx,
                    neighbor_valid,
                    prev_frame,
                    frame_idx,
                    topk,
                    global_topk,
                )
                if mutual_knn
                else {}
            )
            for (src_i, dst_i), sim in forward_pairs.items():
                if sim < edge_threshold:
                    continue
                if mutual_knn and (dst_i, src_i) not in reverse_pairs:
                    mutual_rejected += 1
                    continue
                src_row, src_col = divmod(node_tokens[src_i], w)
                dst_row, dst_col = divmod(node_tokens[dst_i], w)
                pos_dist = float((src_row - dst_row) ** 2 + (src_col - dst_col) ** 2)
                pos_norm = min(pos_dist / float(max(1, radius) ** 2), 4.0)
                imp_delta = abs(float(protection_cpu[src_i]) - float(protection_cpu[dst_i]))
                raw_edges.append((float(sim), src_i, dst_i, pos_norm, imp_delta, 0.0))

    candidate_degree: dict[int, int] = {}
    for _, _, dst, _, _, _ in raw_edges:
        candidate_degree[dst] = candidate_degree.get(dst, 0) + 1
    edges = []
    for sim, src, dst, pos_norm, imp_delta, _ in raw_edges:
        score = sim - spatial_penalty * pos_norm - importance_penalty * imp_delta - hub_penalty * math.log1p(candidate_degree.get(dst, 0))
        edges.append((score, src, dst))
    edges.sort(key=lambda item: item[0], reverse=True)

    parent = list(range(num_nodes))
    members: list[list[int]] = [[idx] for idx in range(num_nodes)]
    frame_sets: list[set[int]] = [{node_frames[idx]} for idx in range(num_nodes)]
    size = [1] * num_nodes
    accepted_in_degree = [0] * num_nodes
    component_count = num_nodes
    edges_accepted = 0
    radius_rejected = 0
    capacity_rejected = 0
    same_frame_rejected = 0

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, src, dst in edges:
        if component_count <= target_components:
            break
        if protected_cpu[src] or protected_cpu[dst]:
            continue
        if accepted_in_degree[dst] >= parent_capacity:
            capacity_rejected += 1
            continue
        src_root = find(src)
        dst_root = find(dst)
        if src_root == dst_root:
            continue
        if one_token_per_frame and frame_sets[src_root].intersection(frame_sets[dst_root]):
            same_frame_rejected += 1
            continue
        merged_members = members[src_root] + members[dst_root]
        if _component_radius(node_features, merged_members) > union_radius_eps:
            radius_rejected += 1
            continue
        if size[src_root] < size[dst_root]:
            src_root, dst_root = dst_root, src_root
        parent[dst_root] = src_root
        size[src_root] += size[dst_root]
        members[src_root].extend(members[dst_root])
        frame_sets[src_root].update(frame_sets[dst_root])
        accepted_in_degree[dst] += 1
        component_count -= 1
        edges_accepted += 1

    root_to_members: dict[int, list[int]] = {}
    for idx in range(num_nodes):
        root_to_members.setdefault(find(idx), []).append(idx)
    components = list(root_to_members.values())

    representative_mode = str(getattr(flashvid_config, "graph_merge_representative", "medoid") or "medoid").strip().lower()
    position_mode = str(getattr(flashvid_config, "graph_representative_position", "protection") or "protection").strip().lower()
    entries: list[dict[str, object]] = []
    radii: list[float] = []

    def component_entry_score(part: list[int], radius_value: float) -> float:
        scores = [float(protection_cpu[node]) for node in part]
        s_max = max(scores) if scores else 0.0
        s_mean = sum(scores) / max(1, len(scores))
        size_gain = math.log1p(len(part)) / math.log1p(max(2, num_frames))
        time_span = len({node_frames[node] for node in part})
        time_gain = math.log1p(time_span) / math.log1p(max(2, num_frames))
        return float(
            0.60 * s_max
            + 0.20 * s_mean
            + 0.15 * size_gain
            + 0.05 * time_gain
            - 0.10 * float(radius_value)
        )

    def emit(part: list[int], feature_mode: str | None = None, radius_value: float | None = None) -> None:
        if not part:
            return
        part_radius = _component_radius(node_features, part) if radius_value is None else float(radius_value)
        mode = feature_mode or representative_mode
        if mode in ("weighted_mean", "attn_mean", "protection_mean"):
            feature = _weighted_mean_feature(node_raw_features, part, protection_cpu)
            pos_node = _choose_position_node(part, node_frames, node_tokens, protection_cpu, w, position_mode)
        elif mode == "mean":
            feature = _mean_feature(node_raw_features, part)
            pos_node = _choose_position_node(part, node_frames, node_tokens, protection_cpu, w, position_mode)
        else:
            pos_node = _medoid_node(node_features, part)
            feature = node_raw_features[pos_node]
        entries.append(
            {
                "frame": node_frames[pos_node],
                "token": node_tokens[pos_node],
                "feature": feature,
                "node": pos_node,
                "score": component_entry_score(part, part_radius),
                "raw_score": max(float(protection_cpu[node]) for node in part),
                "size": len(part),
                "radius": part_radius,
                "time_span": len({node_frames[node] for node in part}),
                "is_protected": any(bool(protected_cpu[node]) for node in part),
            }
        )

    for comp_members in components:
        comp_members = sorted(comp_members, key=lambda node: (node_frames[node], node_tokens[node]))
        radius_value = _component_radius(node_features, comp_members)
        radii.append(radius_value)
        if adaptive_aggregation and radius_value <= component_radius_eps:
            safe_mode = "weighted_mean" if representative_mode == "medoid" else representative_mode
            emit(comp_members, safe_mode, radius_value)
        elif adaptive_aggregation and radius_value <= split_radius_eps and len(comp_members) >= 2:
            midpoint = len(comp_members) // 2
            left = comp_members[:midpoint]
            right = comp_members[midpoint:]
            emit(left, "medoid", _component_radius(node_features, left))
            emit(right, "medoid", _component_radius(node_features, right))
        else:
            emit(comp_members, None, radius_value)

    entries_before_budget_count = len(entries)
    if budget_correction and entries:
        target_total = min(max(1, target_components), num_nodes)
        min_pf = min_tokens_per_frame
        if min_pf * num_frames > target_total:
            min_pf = max(0, target_total // max(1, num_frames))

        def entry_key(entry: dict[str, object]) -> tuple[int, int]:
            return int(entry["frame"]), int(entry["token"])

        def append_diverse_entries(
            selected: list[dict[str, object]],
            selected_keys: set[tuple[int, int]],
            candidates: list[dict[str, object]],
            target: int,
        ) -> None:
            remaining = [entry for entry in candidates if entry_key(entry) not in selected_keys]
            if not remaining:
                return
            if len(selected) >= target:
                return

            scores = torch.tensor(
                [float(entry["score"]) for entry in remaining],
                dtype=torch.float32,
                device=device,
            )
            score_min = scores.min()
            score_range = (scores.max() - score_min).clamp_min(1.0e-6)
            scores = (scores - score_min) / score_range
            feats = torch.stack([entry["feature"].detach().float() for entry in remaining], dim=0)  # type: ignore[union-attr]
            feats = F.normalize(feats, p=2, dim=-1, eps=1.0e-6)
            alive = torch.ones(len(remaining), dtype=torch.bool, device=device)

            if selected:
                selected_feats = torch.stack([entry["feature"].detach().float() for entry in selected], dim=0)  # type: ignore[union-attr]
                selected_feats = F.normalize(selected_feats, p=2, dim=-1, eps=1.0e-6)
                diversity = (1.0 - torch.matmul(feats, selected_feats.transpose(0, 1)).max(dim=1).values).clamp(0.0, 2.0) * 0.5
            else:
                diversity = torch.ones(len(remaining), dtype=torch.float32, device=device)

            while len(selected) < target and bool(alive.any().item()):
                values = scores + budget_diversity_weight * diversity
                values = values.masked_fill(~alive, -1.0e9)
                idx = int(torch.argmax(values).item())
                entry = remaining[idx]
                selected.append(entry)
                selected_keys.add(entry_key(entry))
                alive[idx] = False
                if not bool(alive.any().item()):
                    break
                sim_to_new = torch.matmul(feats, feats[idx].unsqueeze(-1)).squeeze(-1)
                diversity = torch.minimum(diversity, (1.0 - sim_to_new).clamp(0.0, 2.0) * 0.5)

        if len(entries) > target_total:
            frame_sorted: dict[int, list[dict[str, object]]] = {}
            for entry in entries:
                frame_sorted.setdefault(int(entry["frame"]), []).append(entry)
            for frame_entries in frame_sorted.values():
                frame_entries.sort(key=lambda item: (-float(item["score"]), int(item["token"])))

            selected: list[dict[str, object]] = []
            selected_keys: set[tuple[int, int]] = set()
            protected_entries = [entry for entry in entries if bool(entry.get("is_protected", False))]
            protected_entries.sort(key=lambda item: (-float(item["score"]), int(item["frame"]), int(item["token"])))
            for entry in protected_entries:
                if len(selected) >= target_total:
                    break
                key = entry_key(entry)
                if key not in selected_keys:
                    selected.append(entry)
                    selected_keys.add(key)
            if min_pf > 0:
                for frame_idx in range(num_frames):
                    for entry in frame_sorted.get(frame_idx, [])[:min_pf]:
                        key = entry_key(entry)
                        if key not in selected_keys and len(selected) < target_total:
                            selected.append(entry)
                            selected_keys.add(key)
            append_diverse_entries(selected, selected_keys, entries, target_total)
            entries = selected

        if len(entries) < target_total:
            used_keys = {entry_key(entry) for entry in entries}

            def add_rescue(node: int) -> bool:
                key = (node_frames[node], node_tokens[node])
                if key in used_keys:
                    return False
                entries.append(
                    {
                        "frame": node_frames[node],
                        "token": node_tokens[node],
                        "feature": node_raw_features[node],
                        "node": node,
                        "score": component_entry_score([node], 0.0),
                        "raw_score": float(protection_cpu[node]),
                        "size": 1,
                        "radius": 0.0,
                        "time_span": 1,
                        "is_protected": bool(protected_cpu[node]),
                    }
                )
                used_keys.add(key)
                return True

            if min_pf > 0:
                for frame_idx in range(num_frames):
                    while len(entries) < target_total and sum(1 for entry in entries if int(entry["frame"]) == frame_idx) < min_pf:
                        frame_nodes = [node for node in range(num_nodes) if node_frames[node] == frame_idx and (node_frames[node], node_tokens[node]) not in used_keys]
                        if not frame_nodes:
                            break
                        frame_nodes.sort(key=lambda node: (-float(protection_cpu[node]), node_tokens[node]))
                        if not add_rescue(frame_nodes[0]):
                            break
            if len(entries) < target_total:
                rescue_nodes = [node for node in range(num_nodes) if (node_frames[node], node_tokens[node]) not in used_keys]
                rescue_nodes.sort(key=lambda node: (-float(protection_cpu[node]), node_frames[node], node_tokens[node]))
                for node in rescue_nodes:
                    if len(entries) >= target_total:
                        break
                    add_rescue(node)

    setattr(flashvid_config, "last_graft_num_nodes", float(num_nodes))
    setattr(flashvid_config, "last_graft_target_components", float(target_components))
    setattr(flashvid_config, "last_graft_protected_count", float(int(protected.sum().item())))
    setattr(flashvid_config, "last_graft_entries_before_budget", float(entries_before_budget_count))
    setattr(flashvid_config, "last_graft_entries_after_budget", float(len(entries)))
    setattr(flashvid_config, "last_graft_scene_threshold", float(scene_threshold))
    setattr(flashvid_config, "last_graft_global_topk", float(global_topk))
    setattr(flashvid_config, "last_graft_anchor_ratio", float(anchor_ratio))
    setattr(flashvid_config, "last_graft_input_is_residual", float(bool(input_is_residual)))
    setattr(flashvid_config, "last_graft_budget_diversity_weight", float(budget_diversity_weight))
    setattr(flashvid_config, "last_graft_score_preset_code", float(score_preset_code))
    setattr(flashvid_config, "last_graft_budget_correction_active", float(bool(budget_correction)))
    setattr(
        flashvid_config,
        "last_graft_protected_kept_count",
        float(sum(1 for entry in entries if bool(entry.get("is_protected", False)))),
    )

    component_sizes = [len(comp) for comp in components]
    _accumulate_graft_metrics(
        flashvid_config,
        component_sizes=component_sizes,
        radii=radii,
        edges_considered=len(raw_edges),
        edges_accepted=edges_accepted,
        mutual_rejected=mutual_rejected,
        radius_rejected=radius_rejected,
        capacity_rejected=capacity_rejected,
        same_frame_rejected=same_frame_rejected,
    )

    token_lists: list[torch.Tensor] = []
    index_lists: list[torch.Tensor] = []
    out_tokens_by_frame: list[list[torch.Tensor]] = [[] for _ in range(num_frames)]
    out_indices_by_frame: list[list[int]] = [[] for _ in range(num_frames)]
    for entry in sorted(entries, key=lambda item: (int(item["frame"]), int(item["token"]))):
        frame_idx = int(entry["frame"])
        out_tokens_by_frame[frame_idx].append(entry["feature"])  # type: ignore[arg-type]
        out_indices_by_frame[frame_idx].append(int(entry["token"]))
    for frame_idx in range(num_frames):
        if out_tokens_by_frame[frame_idx]:
            frame_tokens = torch.stack(out_tokens_by_frame[frame_idx], dim=0).to(dtype=video_features.dtype)
            frame_indices = torch.tensor(out_indices_by_frame[frame_idx], dtype=torch.long, device=device)
            order = torch.argsort(frame_indices)
            token_lists.append(frame_tokens[order])
            index_lists.append(frame_indices[order])
        else:
            token_lists.append(torch.empty((0, feat_dim), dtype=video_features.dtype, device=device))
            index_lists.append(torch.empty((0,), dtype=torch.long, device=device))
    return token_lists, index_lists
