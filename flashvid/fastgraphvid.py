from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F

from .configuration_flashvid import FlashVidConfig
from .token_selection import TokenSelectionMethod
from .utils import ALL_TOKEN_SELECTION_METHOD


def _cfg_float(config: FlashVidConfig, name: str, default: float) -> float:
    value = getattr(config, name, None)
    return float(default if value is None else value)


def _cfg_int(config: FlashVidConfig, name: str, default: int) -> int:
    value = getattr(config, name, None)
    return int(default if value is None else value)


def _effective_ratio(config: FlashVidConfig) -> float:
    ratio = float(getattr(config, "retention_ratio", 0.10))
    uses_expansion = bool(
        getattr(
            config,
            "fastgraph_budget_uses_expansion",
            getattr(
                config,
                "adapter_budget_uses_expansion",
                getattr(config, "external_budget_uses_expansion", True),
            ),
        )
    )
    if uses_expansion:
        ratio *= float(getattr(config, "expansion", 1.0))
    return max(0.0, min(1.0, ratio))


def _record_fastgraph_metrics(config: FlashVidConfig, *, output_tokens: int, raw_tokens: int) -> None:
    setattr(config, "last_fastgraph_output_tokens", float(output_tokens))
    setattr(config, "last_fastgraph_raw_tokens", float(raw_tokens))
    # Keep the adapter metrics populated for old result collectors.
    setattr(config, "last_adapter_variant", "fastgraphvid")
    setattr(config, "last_adapter_output_tokens", float(output_tokens))
    setattr(config, "last_adapter_raw_tokens", float(raw_tokens))


def _grid_hw(num_visual_tokens: int, config: FlashVidConfig) -> tuple[int, int]:
    h = int(getattr(config, "H", 0) or 0)
    w = int(getattr(config, "W", 0) or 0)
    if h > 0 and w > 0 and h * w == num_visual_tokens:
        return h, w
    h = int(math.sqrt(num_visual_tokens))
    while h > 1 and num_visual_tokens % h != 0:
        h -= 1
    return max(1, h), max(1, num_visual_tokens // max(1, h))


def _neighbor_table(num_visual_tokens: int, h: int, w: int, radius: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[list[int]] = []
    max_len = 1
    for idx in range(num_visual_tokens):
        row, col = divmod(int(idx), w)
        neighbors: list[int] = []
        for rr in range(max(0, row - radius), min(h, row + radius + 1)):
            for cc in range(max(0, col - radius), min(w, col + radius + 1)):
                pos = rr * w + cc
                if pos < num_visual_tokens:
                    neighbors.append(pos)
        if not neighbors:
            neighbors = [idx]
        rows.append(neighbors)
        max_len = max(max_len, len(neighbors))

    table = torch.zeros((num_visual_tokens, max_len), dtype=torch.long, device=device)
    valid = torch.zeros((num_visual_tokens, max_len), dtype=torch.bool, device=device)
    for idx, neighbors in enumerate(rows):
        n = len(neighbors)
        table[idx, :n] = torch.tensor(neighbors, dtype=torch.long, device=device)
        valid[idx, :n] = True
    return table, valid


def _normalize(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    values = values.float()
    if mask is None:
        valid = values.reshape(-1)
    else:
        valid = values[mask]
    if valid.numel() == 0:
        return torch.zeros_like(values, dtype=torch.float32)
    lo = valid.min()
    hi = valid.max()
    return ((values - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)


def _density_score(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    frame_num, frame_token_len, _ = features.shape
    density = torch.zeros((frame_num, frame_token_len), dtype=torch.float32, device=features.device)
    for frame_idx in range(frame_num):
        idx = torch.where(mask[frame_idx])[0]
        if idx.numel() == 0:
            continue
        cur = features[frame_idx, idx].float()
        dist = torch.cdist(cur, cur)
        k = min(4, int(dist.shape[-1]))
        nearest = torch.topk(dist, k=k, dim=-1, largest=False).values
        rho = (-(nearest**2).mean(dim=-1)).exp()
        if rho.numel() > 1:
            rho = rho + torch.arange(rho.numel(), device=features.device, dtype=rho.dtype) * 1e-7
        density[frame_idx, idx] = rho.float()
    return _normalize(density, mask)


def _temporal_novelty(
    normed: torch.Tensor,
    mask: torch.Tensor,
    *,
    radius: int,
    temporal_skip: int,
    config: FlashVidConfig,
) -> torch.Tensor:
    frame_num, frame_token_len, _ = normed.shape
    h, w = _grid_hw(frame_token_len, config)
    neighbor_idx, neighbor_valid = _neighbor_table(frame_token_len, h, w, radius, normed.device)
    novelty = torch.ones((frame_num, frame_token_len), dtype=torch.float32, device=normed.device)

    for lag in range(1, max(1, temporal_skip) + 1):
        for frame_idx in range(lag, frame_num):
            prev_frame = frame_idx - lag
            cur_tokens = torch.where(mask[frame_idx])[0]
            if cur_tokens.numel() == 0:
                continue
            neigh = neighbor_idx[cur_tokens]
            valid = neighbor_valid[cur_tokens] & mask[prev_frame, neigh]
            if not bool(valid.any()):
                continue
            sims = (normed[prev_frame, neigh] * normed[frame_idx, cur_tokens].unsqueeze(1)).sum(dim=-1)
            sims = sims.masked_fill(~valid, -1.0)
            max_sim = sims.max(dim=1).values
            cur_novelty = (1.0 - max_sim.clamp(-1.0, 1.0)).clamp(0.0, 2.0) * 0.5
            novelty[frame_idx, cur_tokens] = torch.minimum(novelty[frame_idx, cur_tokens], cur_novelty)
    return novelty


def _resolve_token_selection_method(config: FlashVidConfig) -> TokenSelectionMethod:
    method = getattr(config, "token_selection_method", TokenSelectionMethod.ADTS_v2)
    try:
        return TokenSelectionMethod(method)
    except ValueError:
        return TokenSelectionMethod.ADTS_STABLE


def _select_graphstm_medoids(
    *,
    video_features: torch.Tensor,
    normed: torch.Tensor,
    candidate_mask: torch.Tensor,
    quality: torch.Tensor,
    global_indices: torch.Tensor,
    target_count: int,
    config: FlashVidConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = video_features.device
    feat_dim = int(video_features.shape[-1])
    positions = torch.where(candidate_mask)
    num_nodes = int(positions[0].numel())
    target_count = min(max(0, int(target_count)), num_nodes)
    if target_count <= 0 or num_nodes <= 0:
        return (
            torch.empty((0, feat_dim), dtype=video_features.dtype, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
        )

    if num_nodes <= target_count:
        tokens = video_features[positions[0], positions[1]]
        indices = global_indices[positions[0], positions[1]]
        order = torch.argsort(indices)
        return tokens[order], indices[order]

    frame_num, frame_token_len, _ = video_features.shape
    local_node = torch.full((frame_num, frame_token_len), -1, dtype=torch.long, device=device)
    local_node[positions] = torch.arange(num_nodes, dtype=torch.long, device=device)

    node_frames = positions[0].detach().cpu().tolist()
    node_tokens = positions[1].detach().cpu().tolist()
    node_quality = quality[positions].detach().float().cpu().tolist()

    h, w = _grid_hw(frame_token_len, config)
    radius = max(0, _cfg_int(config, "fastgraph_temporal_radius", 1))
    temporal_skip = max(1, _cfg_int(config, "fastgraph_temporal_skip", 1))
    topk = max(1, _cfg_int(config, "fastgraph_temporal_topk", 2))
    edge_threshold = _cfg_float(config, "fastgraph_edge_threshold", 0.0)
    protect_ratio = min(max(_cfg_float(config, "fastgraph_protect_ratio", 0.15), 0.0), 0.8)
    neighbor_idx, neighbor_valid = _neighbor_table(frame_token_len, h, w, radius, device)

    edge_scores: list[torch.Tensor] = []
    edge_src: list[torch.Tensor] = []
    edge_dst: list[torch.Tensor] = []
    for lag in range(1, temporal_skip + 1):
        for frame_idx in range(lag, frame_num):
            prev_frame = frame_idx - lag
            cur_tokens = torch.where(candidate_mask[frame_idx])[0]
            if cur_tokens.numel() == 0:
                continue
            neigh = neighbor_idx[cur_tokens]
            valid = neighbor_valid[cur_tokens] & candidate_mask[prev_frame, neigh]
            if not bool(valid.any()):
                continue
            sims = (normed[prev_frame, neigh] * normed[frame_idx, cur_tokens].unsqueeze(1)).sum(dim=-1)
            sims = sims.masked_fill(~valid, -1.0)
            k = min(topk, int(sims.shape[1]))
            vals, ids = torch.topk(sims, k=k, dim=1, largest=True)
            src = local_node[frame_idx, cur_tokens]
            dst = local_node[prev_frame, neigh.gather(1, ids)]
            ok = (vals >= edge_threshold) & (src.unsqueeze(1) >= 0) & (dst >= 0)
            if bool(ok.any()):
                edge_scores.append(vals[ok].detach())
                edge_src.append(src.unsqueeze(1).expand_as(vals)[ok].detach())
                edge_dst.append(dst[ok].detach())

    protected = [False] * num_nodes
    protect_count = min(num_nodes, int(math.ceil(float(target_count) * protect_ratio)))
    if protect_count > 0:
        q = torch.tensor(node_quality, dtype=torch.float32)
        for node in torch.topk(q, k=protect_count, largest=True).indices.tolist():
            protected[int(node)] = True

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
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return False
        if protected[a] or protected[b] or protected[ra] or protected[rb]:
            return False
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
        protected[ra] = protected[ra] or protected[rb]
        component_count -= 1
        return True

    if edge_scores:
        scores = torch.cat(edge_scores).float().cpu()
        srcs = torch.cat(edge_src).long().cpu()
        dsts = torch.cat(edge_dst).long().cpu()
        for edge_idx in torch.argsort(scores, descending=True).tolist():
            if component_count <= target_count:
                break
            union(int(srcs[edge_idx].item()), int(dsts[edge_idx].item()))

    groups: dict[int, list[int]] = {}
    for node in range(num_nodes):
        groups.setdefault(find(node), []).append(node)

    reps: list[int] = []
    for members in groups.values():
        reps.append(max(members, key=lambda node: node_quality[node]))

    if len(reps) > target_count:
        reps = sorted(reps, key=lambda node: node_quality[node], reverse=True)[:target_count]
    elif len(reps) < target_count:
        selected = set(reps)
        remaining = sorted(
            (node for node in range(num_nodes) if node not in selected),
            key=lambda node: node_quality[node],
            reverse=True,
        )
        reps.extend(remaining[: target_count - len(reps)])

    rep_frames = torch.tensor([node_frames[node] for node in reps], dtype=torch.long, device=device)
    rep_tokens = torch.tensor([node_tokens[node] for node in reps], dtype=torch.long, device=device)
    out_tokens = video_features[rep_frames, rep_tokens]
    out_indices = global_indices[rep_frames, rep_tokens]
    order = torch.argsort(out_indices)
    return out_tokens[order], out_indices[order]


def fastgraphvid_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ATS + GraphSTM compression for Qwen3 visual tokens.

    This sidecar method follows FastVID's STPrune spirit: ATS preserves salient
    per-frame details, while a GraphVID-style spatiotemporal graph keeps raw
    residual medoids as the contextual branch. It does not call FastVID DySeg or
    synthesize averaged DTM tokens.
    """
    frame_num, frame_token_len, feat_dim = video_features.shape
    device = video_features.device
    ratio = _effective_ratio(flashvid_config)
    ats_ratio = max(0.0, min(1.0, _cfg_float(flashvid_config, "fastgraph_ats_ratio", 0.60)))

    frame_retain_num = max(1, min(frame_token_len, int(frame_token_len * ratio)))
    frame_ats_num = max(0, min(frame_token_len, int(round(frame_retain_num * ats_ratio))))
    frame_graphstm_num = max(0, frame_retain_num - frame_ats_num)

    all_indices = torch.arange(frame_num * frame_token_len, dtype=torch.long, device=device).view(frame_num, frame_token_len)
    attn = cls_attention.float()
    ats_mask = torch.zeros((frame_num, frame_token_len), dtype=torch.bool, device=device)
    keep_indices: list[torch.Tensor] = []

    if frame_ats_num > 0:
        selection_method = _resolve_token_selection_method(flashvid_config)
        additional_kwargs = {"cls_attention": attn} if "attn" in selection_method.value else {}
        _, ats_idx = ALL_TOKEN_SELECTION_METHOD[selection_method](
            features=video_features,
            num_retained_tokens=frame_ats_num,
            **additional_kwargs,
        )
        batch_idx = torch.arange(frame_num, dtype=torch.long, device=device).unsqueeze(1).expand(-1, frame_ats_num)
        ats_mask[batch_idx, ats_idx] = True
        keep_indices.append(all_indices.gather(1, ats_idx).reshape(-1))

    residual_mask = ~ats_mask
    normed = F.normalize(video_features.float(), p=2, dim=-1, eps=1e-6)
    novelty = _temporal_novelty(
        normed,
        residual_mask,
        radius=max(0, _cfg_int(flashvid_config, "fastgraph_temporal_radius", 1)),
        temporal_skip=max(1, _cfg_int(flashvid_config, "fastgraph_temporal_skip", 1)),
        config=flashvid_config,
    )
    density = _density_score(video_features, residual_mask)
    attn_norm = _normalize(attn, residual_mask)

    attn_weight = _cfg_float(flashvid_config, "fastgraph_attn_weight", 0.55)
    novelty_weight = _cfg_float(flashvid_config, "fastgraph_novelty_weight", 0.30)
    density_weight = _cfg_float(flashvid_config, "fastgraph_density_weight", 0.15)
    quality = attn_weight * attn_norm + novelty_weight * novelty + density_weight * density

    graphstm_target = int(frame_graphstm_num * frame_num)
    if graphstm_target > 0:
        _, graphstm_indices = _select_graphstm_medoids(
            video_features=video_features,
            normed=normed,
            candidate_mask=residual_mask,
            quality=quality,
            global_indices=all_indices,
            target_count=graphstm_target,
            config=flashvid_config,
        )
        if graphstm_indices.numel() > 0:
            keep_indices.append(graphstm_indices)

    if not keep_indices:
        hidden_states = video_features.reshape(-1, feat_dim)[:1]
        selected = torch.zeros((1,), dtype=torch.long, device=device)
    else:
        selected = torch.cat(keep_indices, dim=0)
        selected = torch.unique(selected, sorted=True)
        # Re-gather raw tokens after de-duplication to guarantee original-token output.
        frame_idx = selected // frame_token_len
        token_idx = selected % frame_token_len
        hidden_states = video_features[frame_idx, token_idx]

    flashvid_config.vision_token_length = int(hidden_states.shape[0])
    flashvid_config.llm_token_length = None
    flashvid_config.visual_token_length = int(hidden_states.shape[0])
    setattr(flashvid_config, "last_fastgraph_ats_ratio", float(ats_ratio))
    setattr(flashvid_config, "last_fastgraph_frame_retain_num", float(frame_retain_num))
    setattr(flashvid_config, "last_fastgraph_frame_ats_num", float(frame_ats_num))
    setattr(flashvid_config, "last_fastgraph_frame_graphstm_num", float(frame_graphstm_num))
    _record_fastgraph_metrics(
        flashvid_config,
        output_tokens=int(hidden_states.shape[0]),
        raw_tokens=int(frame_num * frame_token_len),
    )
    return hidden_states, selected
