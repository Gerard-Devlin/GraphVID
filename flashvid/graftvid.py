from __future__ import annotations

from typing import List, Tuple

import math
import torch
from torch.nn import functional as F

from .configuration_flashvid import FlashVidConfig
from .graphvid import _choose_position_node, _grid_hw, _neighbor_table, _normalize_on_mask, _spatial_detail_score


def _bool_config(config: FlashVidConfig, name: str, default: bool) -> bool:
    value = getattr(config, name, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


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


def _reverse_mutual_pairs(
    normed: torch.Tensor,
    token_mask: torch.Tensor,
    node_id: torch.Tensor,
    neighbor_idx: torch.Tensor,
    neighbor_valid: torch.Tensor,
    prev_frame: int,
    cur_frame: int,
    topk: int,
) -> set[tuple[int, int]]:
    reverse_pairs: set[tuple[int, int]] = set()
    prev_valid = torch.where(token_mask[prev_frame])[0]
    if prev_valid.numel() == 0:
        return reverse_pairs
    neigh = neighbor_idx[prev_valid]
    valid = neighbor_valid[prev_valid] & token_mask[cur_frame, neigh]
    sims = torch.sum(normed[cur_frame, neigh] * normed[prev_frame, prev_valid].unsqueeze(1), dim=-1)
    sims = sims.masked_fill(~valid, -1.0)
    k = min(topk, int(sims.shape[1]))
    if k <= 0:
        return reverse_pairs
    vals, ids = torch.topk(sims, k=k, dim=1, largest=True)
    prev_nodes = node_id[prev_frame, prev_valid].unsqueeze(1).expand_as(vals)
    cur_nodes = node_id[cur_frame, neigh.gather(1, ids)]
    edge_valid = (vals > -0.5) & (prev_nodes >= 0) & (cur_nodes >= 0)
    for cur_node, prev_node in zip(cur_nodes[edge_valid].detach().cpu().tolist(), prev_nodes[edge_valid].detach().cpu().tolist()):
        reverse_pairs.add((int(cur_node), int(prev_node)))
    return reverse_pairs


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

    target_components = max(1, int(getattr(flashvid_config, "num_sttm_tokens", 0) or 0) * num_frames)
    target_components = min(num_nodes, target_components)

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
    radius = max(0, int(getattr(flashvid_config, "graft_temporal_radius", 1) or 1))
    topk = max(1, int(getattr(flashvid_config, "graft_temporal_topk", 3) or 3))
    temporal_skip = max(1, int(getattr(flashvid_config, "graft_temporal_skip", 1) or 1))
    neighbor_idx, neighbor_valid = _neighbor_table(num_visual_tokens, h, w, radius, device)

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
            cur_novelty = (1.0 - max_sim).clamp(0.0, 2.0) * 0.5
            novelty[frame_idx, cur_valid] = torch.minimum(novelty[frame_idx, cur_valid], cur_novelty)
    novelty_norm = _normalize_on_mask(novelty, token_mask)
    detail = _spatial_detail_score(normed, neighbor_idx, neighbor_valid)
    detail_norm = _normalize_on_mask(detail, token_mask)
    protection_map = (0.65 * attn_norm + 0.25 * novelty_norm + 0.10 * detail_norm).clamp(0.0, 1.0)
    protection = protection_map[token_mask]
    protection_cpu = protection.detach().float().cpu().tolist()

    protected = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    anchor_ratio = min(max(float(getattr(flashvid_config, "graft_anchor_ratio", 0.65) or 0.65), 0.0), 0.95)
    protect_count = min(num_nodes, int(math.ceil(target_components * anchor_ratio)))
    if protect_count > 0:
        protected[torch.topk(protection, k=protect_count, largest=True).indices] = True
    protected_cpu = protected.detach().cpu().tolist()

    edge_threshold = float(getattr(flashvid_config, "graft_edge_threshold", 0.80) or 0.80)
    if edge_threshold <= 0.0:
        edge_threshold = float(temporal_threshold)
    mutual_knn = _bool_config(flashvid_config, "graft_mutual_knn", True)
    one_token_per_frame = _bool_config(flashvid_config, "graft_one_token_per_frame", True)
    component_radius_eps = max(0.0, float(getattr(flashvid_config, "graft_component_radius_eps", 0.12) or 0.12))
    split_radius_eps = max(component_radius_eps, float(getattr(flashvid_config, "graft_split_radius_eps", 0.20) or 0.20))
    parent_capacity = max(1, int(getattr(flashvid_config, "graft_parent_capacity", 1) or 1))
    spatial_penalty = max(0.0, float(getattr(flashvid_config, "graft_spatial_penalty", 0.10) or 0.10))
    importance_penalty = max(0.0, float(getattr(flashvid_config, "graft_importance_penalty", 0.05) or 0.05))
    hub_penalty = max(0.0, float(getattr(flashvid_config, "graft_hub_penalty", 0.05) or 0.05))

    raw_edges: list[tuple[float, int, int, float, float, float]] = []
    mutual_rejected = 0
    for lag in range(1, temporal_skip + 1):
        for frame_idx in range(lag, num_frames):
            prev_frame = frame_idx - lag
            cur_valid = torch.where(token_mask[frame_idx])[0]
            if cur_valid.numel() == 0:
                continue
            reverse_pairs = (
                _reverse_mutual_pairs(normed, token_mask, node_id, neighbor_idx, neighbor_valid, prev_frame, frame_idx, topk)
                if mutual_knn
                else set()
            )
            neigh = neighbor_idx[cur_valid]
            valid = neighbor_valid[cur_valid] & token_mask[prev_frame, neigh]
            sims = torch.sum(normed[prev_frame, neigh] * normed[frame_idx, cur_valid].unsqueeze(1), dim=-1)
            sims = sims.masked_fill(~valid, -1.0)
            k = min(topk, int(sims.shape[1]))
            if k <= 0:
                continue
            vals, ids = torch.topk(sims, k=k, dim=1, largest=True)
            cur_nodes = node_id[frame_idx, cur_valid].unsqueeze(1).expand_as(vals)
            prev_nodes = node_id[prev_frame, neigh.gather(1, ids)]
            edge_valid = (vals >= edge_threshold) & (cur_nodes >= 0) & (prev_nodes >= 0)
            for sim, src, dst in zip(vals[edge_valid].detach().cpu().tolist(), cur_nodes[edge_valid].detach().cpu().tolist(), prev_nodes[edge_valid].detach().cpu().tolist()):
                src_i = int(src)
                dst_i = int(dst)
                if mutual_knn and (src_i, dst_i) not in reverse_pairs:
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
        if _component_radius(node_features, merged_members) > component_radius_eps:
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
    adaptive_aggregation = _bool_config(flashvid_config, "graft_adaptive_aggregation", True)
    out_tokens_by_frame: list[list[torch.Tensor]] = [[] for _ in range(num_frames)]
    out_indices_by_frame: list[list[int]] = [[] for _ in range(num_frames)]
    radii: list[float] = []

    def emit(part: list[int]) -> None:
        if not part:
            return
        if representative_mode in ("weighted_mean", "attn_mean", "protection_mean"):
            feature = _weighted_mean_feature(node_raw_features, part, protection_cpu)
            pos_node = _choose_position_node(part, node_frames, node_tokens, protection_cpu, w, position_mode)
        elif representative_mode == "mean":
            feature = _mean_feature(node_raw_features, part)
            pos_node = _choose_position_node(part, node_frames, node_tokens, protection_cpu, w, position_mode)
        else:
            pos_node = _medoid_node(node_features, part)
            feature = node_raw_features[pos_node]
        out_tokens_by_frame[node_frames[pos_node]].append(feature)
        out_indices_by_frame[node_frames[pos_node]].append(node_tokens[pos_node])

    for comp_members in components:
        comp_members = sorted(comp_members, key=lambda node: (node_frames[node], node_tokens[node]))
        radius_value = _component_radius(node_features, comp_members)
        radii.append(radius_value)
        if adaptive_aggregation and radius_value >= split_radius_eps and len(comp_members) >= 2:
            midpoint = len(comp_members) // 2
            emit(comp_members[:midpoint])
            emit(comp_members[midpoint:])
        else:
            emit(comp_members)

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
