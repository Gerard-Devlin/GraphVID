from typing import List, Tuple

import math
import torch
from torch.nn import functional as F

from .configuration_flashvid import FlashVidConfig


def _grid_hw(num_visual_tokens: int, flashvid_config: FlashVidConfig) -> Tuple[int, int]:
    h = int(getattr(flashvid_config, "H", 0) or 0)
    w = int(getattr(flashvid_config, "W", 0) or 0)
    if h > 0 and w > 0 and h * w == num_visual_tokens:
        return h, w
    h = int(math.sqrt(num_visual_tokens))
    while h > 1 and num_visual_tokens % h != 0:
        h -= 1
    w = max(1, num_visual_tokens // max(1, h))
    return max(1, h), w


def _local_neighbors(index: int, h: int, w: int, radius: int, device: torch.device) -> torch.Tensor:
    row, col = divmod(int(index), w)
    ids = []
    for rr in range(max(0, row - radius), min(h, row + radius + 1)):
        for cc in range(max(0, col - radius), min(w, col + radius + 1)):
            idx = rr * w + cc
            if idx < h * w:
                ids.append(idx)
    return torch.tensor(ids, dtype=torch.long, device=device)


def _neighbor_table(num_visual_tokens: int, h: int, w: int, radius: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    rows = []
    max_len = 0
    for idx in range(num_visual_tokens):
        neighbors = _local_neighbors(idx, h, w, radius, device="cpu").tolist()
        rows.append(neighbors)
        max_len = max(max_len, len(neighbors))
    table = torch.zeros((num_visual_tokens, max_len), dtype=torch.long)
    valid = torch.zeros((num_visual_tokens, max_len), dtype=torch.bool)
    for idx, neighbors in enumerate(rows):
        if not neighbors:
            table[idx, 0] = idx
            valid[idx, 0] = True
            continue
        n = len(neighbors)
        table[idx, :n] = torch.tensor(neighbors, dtype=torch.long)
        valid[idx, :n] = True
    return table.to(device=device), valid.to(device=device)


def _normalize_on_mask(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = values[mask]
    if valid.numel() == 0:
        return torch.zeros_like(values)
    min_val = valid.min()
    max_val = valid.max()
    return (values - min_val) / (max_val - min_val + 1e-6)


def _spatial_detail_score(
    normed_features: torch.Tensor,
    neighbor_idx: torch.Tensor,
    neighbor_valid: torch.Tensor,
) -> torch.Tensor:
    """Estimate local detail from same-frame neighborhood uniqueness."""
    num_frames, num_visual_tokens, _ = normed_features.shape
    device = normed_features.device
    token_ids = torch.arange(num_visual_tokens, dtype=torch.long, device=device)
    neigh = neighbor_idx[token_ids]
    valid = neighbor_valid[token_ids] & (neigh != token_ids.unsqueeze(1))
    detail = torch.zeros((num_frames, num_visual_tokens), dtype=torch.float32, device=device)
    if valid.numel() == 0 or not valid.any():
        return detail

    for frame_idx in range(num_frames):
        cur = normed_features[frame_idx, token_ids]
        neigh_feats = normed_features[frame_idx, neigh]
        sims = torch.sum(neigh_feats * cur.unsqueeze(1), dim=-1)
        sims = sims.masked_fill(~valid, -1.0)
        max_sim = sims.max(dim=1).values.clamp(min=-1.0, max=1.0)
        detail[frame_idx] = ((1.0 - max_sim) * 0.5).clamp(0.0, 1.0)
    return detail


def _top_fraction_mean(values: torch.Tensor, fraction: float) -> torch.Tensor:
    if values.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=values.device)
    fraction = min(max(float(fraction), 0.0), 1.0)
    k = max(1, int(math.ceil(values.numel() * fraction)))
    return torch.topk(values.float(), k=k, largest=True).values.mean()


def _split_by_temporal_span(members: list[int], node_frames: list[int], max_span: int) -> list[list[int]]:
    if max_span <= 0 or len(members) <= 1:
        return [members]
    chunks: list[list[int]] = []
    cur: list[int] = []
    start_frame: int | None = None
    for node in sorted(members, key=lambda item: (node_frames[item], item)):
        frame = int(node_frames[node])
        if start_frame is None or frame - start_frame + 1 <= max_span:
            if start_frame is None:
                start_frame = frame
            cur.append(node)
            continue
        if cur:
            chunks.append(cur)
        cur = [node]
        start_frame = frame
    if cur:
        chunks.append(cur)
    return chunks


def _component_spatial_radius(members: list[int], node_tokens: list[int], rep_token: int, w: int) -> int:
    rep_row, rep_col = divmod(int(rep_token), w)
    max_radius = 0
    for node in members:
        row, col = divmod(int(node_tokens[node]), w)
        max_radius = max(max_radius, abs(row - rep_row), abs(col - rep_col))
    return max_radius


def graph_spatiotemporal_compression(
    video_features: torch.Tensor,
    temporal_threshold: float,
    token_mask: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Graph-based replacement for FlashVID's tree-style temporal merging.

    The original TSTM path attaches each token to one previous-frame parent. This
    path builds sparse local temporal edges, groups redundant nodes with
    union-find, and emits raw medoid representatives for each component.
    """
    num_frames, num_visual_tokens, feat_dim = video_features.shape
    device = video_features.device
    candidate_positions = torch.where(token_mask)
    num_nodes = int(candidate_positions[0].numel())
    if num_nodes == 0:
        empty_tokens = torch.empty((0, feat_dim), dtype=video_features.dtype, device=device)
        empty_indices = torch.empty((0,), dtype=torch.long, device=device)
        return [empty_tokens for _ in range(num_frames)], [empty_indices for _ in range(num_frames)]

    target_context = max(1, int(getattr(flashvid_config, "num_sttm_tokens", 0) or 0) * num_frames)
    # Ratios below 1.0 deliberately make GraphVID more aggressive than FlashVID's
    # STTM budget, enabling real token reduction instead of an equal-budget swap.
    target_ratio = min(max(float(getattr(flashvid_config, "graph_merge_target_ratio", 0.65) or 0.65), 0.05), 4.0)
    target_components = min(num_nodes, max(1, int(math.ceil(target_context * target_ratio))))

    node_id = torch.full((num_frames, num_visual_tokens), -1, dtype=torch.long, device=device)
    node_id[candidate_positions] = torch.arange(num_nodes, dtype=torch.long, device=device)
    node_frames = candidate_positions[0].tolist()
    node_tokens = candidate_positions[1].tolist()

    normed = F.normalize(video_features.float(), p=2, dim=-1, eps=1e-6)
    attn = cls_attention.float()
    valid_attn = attn[token_mask]
    if valid_attn.numel() > 0:
        attn_min = valid_attn.min()
        attn_max = valid_attn.max()
        attn_norm = (attn - attn_min) / (attn_max - attn_min + 1e-6)
    else:
        attn_norm = torch.zeros_like(attn)

    h, w = _grid_hw(num_visual_tokens, flashvid_config)
    radius = max(0, int(getattr(flashvid_config, "graph_temporal_radius", 1) or 1))
    temporal_skip = max(1, int(getattr(flashvid_config, "graph_temporal_skip", 1) or 1))
    topk = max(1, int(getattr(flashvid_config, "graph_temporal_topk", 3) or 3))
    neighbor_idx, neighbor_valid = _neighbor_table(num_visual_tokens, h, w, radius, device)

    novelty = torch.ones((num_frames, num_visual_tokens), dtype=torch.float32, device=device)
    edge_score_parts: list[torch.Tensor] = []
    edge_src_parts: list[torch.Tensor] = []
    edge_dst_parts: list[torch.Tensor] = []
    for lag in range(1, temporal_skip + 1):
        for frame_idx in range(lag, num_frames):
            prev_frame = frame_idx - lag
            cur_valid = torch.where(token_mask[frame_idx])[0]
            if cur_valid.numel() == 0:
                continue
            neigh = neighbor_idx[cur_valid]
            valid = neighbor_valid[cur_valid] & token_mask[prev_frame, neigh]
            prev_feats = normed[prev_frame, neigh]
            cur_feats = normed[frame_idx, cur_valid]
            sims = torch.sum(prev_feats * cur_feats.unsqueeze(1), dim=-1)
            sims = sims.masked_fill(~valid, -1.0)
            max_sim = sims.max(dim=1).values
            cur_novelty = (1.0 - max_sim).clamp(0.0, 2.0) * 0.5
            novelty[frame_idx, cur_valid] = torch.minimum(novelty[frame_idx, cur_valid], cur_novelty)

            k = min(topk, int(sims.shape[1]))
            vals, ids = torch.topk(sims, k=k, dim=1, largest=True)
            cur_nodes = node_id[frame_idx, cur_valid]
            prev_nodes = node_id[prev_frame, neigh.gather(1, ids)]
            edge_valid = (vals > -0.5) & (cur_nodes.unsqueeze(1) >= 0) & (prev_nodes >= 0)
            if edge_valid.any():
                edge_score_parts.append(vals[edge_valid].detach())
                edge_src_parts.append(cur_nodes.unsqueeze(1).expand_as(vals)[edge_valid].detach())
                edge_dst_parts.append(prev_nodes[edge_valid].detach())

    novelty_vals = novelty[token_mask]
    novelty_min = novelty_vals.min()
    novelty_max = novelty_vals.max()
    novelty_norm = (novelty - novelty_min) / (novelty_max - novelty_min + 1e-6)
    detail_weight = max(0.0, float(getattr(flashvid_config, "graph_protection_detail_weight", 0.0) or 0.0))
    adaptive_detail = bool(getattr(flashvid_config, "graph_adaptive_detail_protection", False))
    detail_norm = torch.zeros_like(attn_norm)
    detail_pressure = 0.0
    if detail_weight > 0.0 or adaptive_detail:
        detail = _spatial_detail_score(normed, neighbor_idx, neighbor_valid)
        detail_norm = _normalize_on_mask(detail, token_mask)
        if adaptive_detail:
            valid_detail = detail_norm[token_mask]
            valid_attn = attn_norm[token_mask]
            detail_tail = _top_fraction_mean(valid_detail, 0.15)
            attn_tail = _top_fraction_mean(valid_attn, 0.15)
            detail_pressure = float((0.65 * detail_tail + 0.35 * attn_tail).clamp(0.0, 1.0).item())
            detail_boost = max(0.0, float(getattr(flashvid_config, "graph_adaptive_detail_boost", 0.22) or 0.0))
            detail_weight = max(detail_weight, detail_boost * detail_pressure)

    attn_weight = max(0.0, float(getattr(flashvid_config, "graph_protection_attn_weight", 0.70) or 0.0))
    novelty_weight = max(0.0, float(getattr(flashvid_config, "graph_protection_novelty_weight", 0.30) or 0.0))
    total_weight = attn_weight + novelty_weight + detail_weight
    if total_weight <= 0.0:
        attn_weight, novelty_weight, detail_weight, total_weight = 0.70, 0.30, 0.0, 1.0
    protection_map = (
        attn_weight * attn_norm
        + novelty_weight * novelty_norm
        + detail_weight * detail_norm
    ) / total_weight
    protection = protection_map[token_mask]

    protected = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    protect_ratio = min(max(float(getattr(flashvid_config, "graph_merge_protect_ratio", 0.15)), 0.0), 0.80)
    if adaptive_detail and detail_pressure > 0.0:
        protect_boost = max(0.0, float(getattr(flashvid_config, "graph_adaptive_protect_boost", 0.10) or 0.0))
        protect_ratio = min(0.80, protect_ratio + protect_boost * detail_pressure)
    protect_count = min(num_nodes, int(math.ceil(target_components * protect_ratio)))
    if protect_count > 0:
        protected[torch.topk(protection, k=protect_count, largest=True).indices] = True
    protected_cpu = protected.detach().cpu().tolist()
    protection_cpu = protection.detach().float().cpu().tolist()

    parent = list(range(num_nodes))
    size = [1] * num_nodes
    component_count = num_nodes

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        nonlocal component_count
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if protected_cpu[ra] or protected_cpu[rb] or protected_cpu[a] or protected_cpu[b]:
            return False
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
        component_count -= 1
        return True

    if edge_score_parts:
        edge_scores = torch.cat(edge_score_parts).float().cpu()
        edge_src = torch.cat(edge_src_parts).long().cpu()
        edge_dst = torch.cat(edge_dst_parts).long().cpu()
        importance_penalty = max(0.0, float(getattr(flashvid_config, "graph_merge_importance_penalty", 0.0) or 0.0))
        if importance_penalty > 0.0:
            edge_importance = torch.tensor(
                [
                    max(protection_cpu[int(src)], protection_cpu[int(dst)])
                    for src, dst in zip(edge_src.tolist(), edge_dst.tolist())
                ],
                dtype=edge_scores.dtype,
            )
            edge_sort_scores = edge_scores - importance_penalty * edge_importance
        else:
            edge_sort_scores = edge_scores
        edge_order = torch.argsort(edge_sort_scores, descending=True)
        respect_threshold = bool(getattr(flashvid_config, "graph_respect_temporal_threshold", False))
        for edge_idx in edge_order.tolist():
            if component_count <= target_components:
                break
            sim = float(edge_scores[edge_idx].item())
            if respect_threshold and sim < temporal_threshold:
                continue
            union(int(edge_src[edge_idx].item()), int(edge_dst[edge_idx].item()))

    groups: dict[int, list[int]] = {}
    for node in range(num_nodes):
        groups.setdefault(find(node), []).append(node)

    representative_mode = str(
        getattr(flashvid_config, "graph_merge_representative", "medoid") or "medoid"
    ).strip().lower()
    blend_alpha = min(
        max(float(getattr(flashvid_config, "graph_representative_blend_alpha", 0.20) or 0.0), 0.0),
        1.0,
    )
    spatial_guard_radius = max(0, int(getattr(flashvid_config, "graph_spatial_spread_guard_radius", 0) or 0))
    temporal_span_guard = max(0, int(getattr(flashvid_config, "graph_temporal_span_guard", 0) or 0))
    frame_tokens: list[list[torch.Tensor]] = [[] for _ in range(num_frames)]
    frame_indices: list[list[int]] = [[] for _ in range(num_frames)]
    for raw_members in groups.values():
        for members in _split_by_temporal_span(raw_members, node_frames, temporal_span_guard):
            if not members:
                continue
            rep_node = max(members, key=lambda node: protection_cpu[node])
            rep_frame = int(node_frames[rep_node])
            rep_token = int(node_tokens[rep_node])
            anchor = video_features[rep_frame, rep_token]
            spread_too_wide = (
                spatial_guard_radius > 0
                and _component_spatial_radius(members, node_tokens, rep_token, w) > spatial_guard_radius
            )

            if representative_mode in ("hybrid_anchor", "anchor_blend", "hybrid", "weighted_anchor"):
                if len(members) == 1 or blend_alpha <= 0.0 or spread_too_wide:
                    out_token = anchor
                else:
                    member_feats = torch.stack([video_features[node_frames[m], node_tokens[m]] for m in members], dim=0)
                    weights = torch.tensor(
                        [max(1.0e-4, protection_cpu[m]) for m in members],
                        dtype=torch.float32,
                        device=device,
                    )
                    weights = weights / weights.sum().clamp_min(1.0e-6)
                    weighted = torch.sum(member_feats.float() * weights.unsqueeze(-1), dim=0)
                    out_token = (anchor.float() + blend_alpha * (weighted - anchor.float())).to(dtype=video_features.dtype)
            elif representative_mode in ("weighted_mean", "attn_mean", "protection_mean"):
                if len(members) == 1 or spread_too_wide:
                    out_token = anchor
                else:
                    member_feats = torch.stack([video_features[node_frames[m], node_tokens[m]] for m in members], dim=0)
                    weights = torch.tensor(
                        [max(1.0e-4, protection_cpu[m]) for m in members],
                        dtype=torch.float32,
                        device=device,
                    )
                    weights = weights / weights.sum().clamp_min(1.0e-6)
                    out_token = torch.sum(member_feats.float() * weights.unsqueeze(-1), dim=0).to(dtype=video_features.dtype)
            elif representative_mode == "mean":
                if len(members) == 1 or spread_too_wide:
                    out_token = anchor
                else:
                    member_feats = torch.stack([video_features[node_frames[m], node_tokens[m]] for m in members], dim=0)
                    out_token = member_feats.mean(dim=0).to(dtype=video_features.dtype)
            else:
                out_token = anchor
            frame_tokens[rep_frame].append(out_token)
            frame_indices[rep_frame].append(rep_token)

    final_tokens: list[torch.Tensor] = []
    retained_token_indices: list[torch.Tensor] = []
    for frame_idx in range(num_frames):
        if not frame_tokens[frame_idx]:
            final_tokens.append(torch.empty((0, feat_dim), dtype=video_features.dtype, device=device))
            retained_token_indices.append(torch.empty((0,), dtype=torch.long, device=device))
            continue
        idx_tensor = torch.tensor(frame_indices[frame_idx], dtype=torch.long, device=device)
        order = idx_tensor.argsort()
        token_tensor = torch.stack(frame_tokens[frame_idx], dim=0)[order]
        final_tokens.append(token_tensor)
        retained_token_indices.append(idx_tensor[order])

    return final_tokens, retained_token_indices
