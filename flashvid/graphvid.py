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
    protection_map = 0.7 * attn_norm + 0.3 * novelty_norm
    protection = protection_map[token_mask]

    protected = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    protect_ratio = min(max(float(getattr(flashvid_config, "graph_merge_protect_ratio", 0.15)), 0.0), 0.80)
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
        edge_order = torch.argsort(edge_scores, descending=True)
        for edge_idx in edge_order.tolist():
            if component_count <= target_components:
                break
            sim = float(edge_scores[edge_idx].item())
            if sim < temporal_threshold and component_count <= target_components:
                break
            union(int(edge_src[edge_idx].item()), int(edge_dst[edge_idx].item()))

    groups: dict[int, list[int]] = {}
    for node in range(num_nodes):
        groups.setdefault(find(node), []).append(node)

    representative_mode = str(
        getattr(flashvid_config, "graph_merge_representative", "medoid") or "medoid"
    ).strip().lower()
    frame_tokens: list[list[torch.Tensor]] = [[] for _ in range(num_frames)]
    frame_indices: list[list[int]] = [[] for _ in range(num_frames)]
    for members in groups.values():
        rep_node = max(members, key=lambda node: protection_cpu[node])
        rep_frame = int(node_frames[rep_node])
        rep_token = int(node_tokens[rep_node])
        if representative_mode == "mean":
            member_feats = torch.stack([video_features[node_frames[m], node_tokens[m]] for m in members], dim=0)
            out_token = member_feats.mean(dim=0).to(dtype=video_features.dtype)
        else:
            out_token = video_features[rep_frame, rep_token]
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
