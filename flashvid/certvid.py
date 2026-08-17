from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .configuration_flashvid import FlashVidConfig


@dataclass
class CertVidPlan:
    """Sparse assignment shared by base and Qwen3 DeepStack features."""

    anchor_indices: torch.Tensor
    assignment_indices: torch.Tensor
    assignment_weights: torch.Tensor
    source_mass: torch.Tensor
    fusion_alpha: torch.Tensor
    raw_token_count: int


def _cfg_float(config: FlashVidConfig, name: str, default: float) -> float:
    try:
        return float(getattr(config, name, default))
    except (TypeError, ValueError):
        return float(default)


def _cfg_int(config: FlashVidConfig, name: str, default: int) -> int:
    try:
        return int(getattr(config, name, default))
    except (TypeError, ValueError):
        return int(default)


def _effective_ratio(config: FlashVidConfig) -> float:
    ratio = _cfg_float(config, "retention_ratio", 0.10)
    if bool(getattr(config, "cert_budget_uses_expansion", True)):
        ratio *= _cfg_float(config, "expansion", 1.0)
    return min(1.0, max(0.0, ratio))


def _grid_hw(tokens_per_frame: int, config: FlashVidConfig) -> tuple[int, int]:
    height = _cfg_int(config, "H", 0)
    width = _cfg_int(config, "W", 0)
    if height > 0 and width > 0 and height * width == tokens_per_frame:
        return height, width
    height = max(1, int(math.sqrt(tokens_per_frame)))
    while height > 1 and tokens_per_frame % height != 0:
        height -= 1
    return height, max(1, tokens_per_frame // height)


def _minmax(values: torch.Tensor, dim: int = -1) -> torch.Tensor:
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
    lo = values.amin(dim=dim, keepdim=True)
    hi = values.amax(dim=dim, keepdim=True)
    return ((values - lo) / (hi - lo + 1e-6)).clamp_(0.0, 1.0)


def _rank_normalize(values: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] <= 1:
        return torch.ones_like(values, dtype=torch.float32)
    order = torch.argsort(values.float(), dim=-1, stable=True)
    ranks = torch.argsort(order, dim=-1, stable=True).float()
    return ranks / float(values.shape[-1] - 1)


def _metric_features(features: torch.Tensor, metric_dim: int) -> torch.Tensor:
    flat = features.reshape(-1, features.shape[-1]).float()
    if metric_dim > 0 and flat.shape[-1] > metric_dim:
        flat = F.adaptive_avg_pool1d(flat.unsqueeze(1), metric_dim).squeeze(1)
    return F.normalize(flat, p=2, dim=-1, eps=1e-6)


def _spatial_layout(
    tokens_per_frame: int,
    height: int,
    width: int,
    spatial_bins: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_ids = torch.arange(tokens_per_frame, device=device)
    rows = torch.div(token_ids, width, rounding_mode="floor").clamp_max(height - 1)
    cols = torch.remainder(token_ids, width).clamp_max(width - 1)
    coords = torch.stack(
        [rows.float() / max(1, height - 1), cols.float() / max(1, width - 1)],
        dim=-1,
    )
    row_bin = torch.div(rows * spatial_bins, max(1, height), rounding_mode="floor").clamp_max(spatial_bins - 1)
    col_bin = torch.div(cols * spatial_bins, max(1, width), rounding_mode="floor").clamp_max(spatial_bins - 1)
    cells = row_bin * spatial_bins + col_bin
    return coords, cells.long()


def _temporal_signals(
    metric_frames: torch.Tensor,
    coords: torch.Tensor,
    spatial_penalty: float,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    frame_count, tokens_per_frame, _ = metric_frames.shape
    device = metric_frames.device
    frame_reps = F.normalize(metric_frames.mean(dim=1), p=2, dim=-1, eps=1e-6)
    frame_event = torch.zeros(frame_count, dtype=torch.float32, device=device)
    token_novelty = torch.ones((frame_count, tokens_per_frame), dtype=torch.float32, device=device)
    matches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    spatial_distance = torch.cdist(coords.float(), coords.float(), p=2)
    for frame_idx in range(1, frame_count):
        current = metric_frames[frame_idx]
        previous = metric_frames[frame_idx - 1]
        raw_similarity = current @ previous.transpose(0, 1)
        match_score = raw_similarity - float(max(0.0, spatial_penalty)) * spatial_distance
        best_score, best_previous = match_score.max(dim=1)
        best_current = match_score.argmax(dim=0)
        current_ids = torch.arange(tokens_per_frame, device=device)
        mutual = best_current[best_previous] == current_ids
        matched_similarity = raw_similarity[current_ids, best_previous]
        token_novelty[frame_idx] = (1.0 - matched_similarity).clamp(0.0, 2.0) * 0.5
        frame_event[frame_idx] = 1.0 - torch.sum(frame_reps[frame_idx] * frame_reps[frame_idx - 1]).clamp(-1.0, 1.0)
        matches.append((best_previous, mutual, matched_similarity))

    if frame_count > 2:
        incoming = frame_reps[1:-1] - frame_reps[:-2]
        outgoing = frame_reps[2:] - frame_reps[1:-1]
        curvature = 1.0 - F.cosine_similarity(incoming, outgoing, dim=-1, eps=1e-6)
        frame_event[1:-1] = torch.maximum(frame_event[1:-1], curvature.clamp(0.0, 2.0))
    frame_event = _minmax(frame_event, dim=0)
    token_novelty[0] = frame_event[0]
    return frame_event, _minmax(token_novelty, dim=-1), matches


def _build_components(
    frame_count: int,
    tokens_per_frame: int,
    frame_event: torch.Tensor,
    matches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_tokens = frame_count * tokens_per_frame
    parent = list(range(total_tokens))
    size = [1] * total_tokens

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            return
        if size[root_left] < size[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        size[root_left] += size[root_right]

    if frame_count > 1:
        cut_threshold = float(torch.quantile(frame_event.float(), 0.90).item())
        frame_event_cpu = frame_event.detach().float().cpu().tolist()
        current_links: list[torch.Tensor] = []
        previous_links: list[torch.Tensor] = []
        for frame_idx, (best_previous, mutual, similarity) in enumerate(matches, start=1):
            if float(frame_event_cpu[frame_idx]) > max(0.65, cut_threshold):
                continue
            valid = mutual & (similarity >= float(threshold))
            current_tokens = torch.where(valid)[0]
            if current_tokens.numel() == 0:
                continue
            current_links.append(current_tokens + frame_idx * tokens_per_frame)
            previous_links.append(
                best_previous[current_tokens] + (frame_idx - 1) * tokens_per_frame
            )

        if current_links:
            current_cpu = torch.cat(current_links).detach().cpu().tolist()
            previous_cpu = torch.cat(previous_links).detach().cpu().tolist()
            for current_token, previous_token in zip(current_cpu, previous_cpu):
                union(int(current_token), int(previous_token))

    roots = [find(node) for node in range(total_tokens)]
    root_to_component: dict[int, int] = {}
    component_ids: list[int] = []
    for root in roots:
        if root not in root_to_component:
            root_to_component[root] = len(root_to_component)
        component_ids.append(root_to_component[root])
    component_count = len(root_to_component)
    component_sizes = torch.bincount(torch.tensor(component_ids, dtype=torch.long), minlength=component_count)
    return torch.tensor(component_ids, dtype=torch.long), component_sizes


def _question_atoms(
    question_features: Optional[torch.Tensor],
    max_atoms: int,
    metric_dim: int,
) -> torch.Tensor:
    if question_features is None or question_features.numel() == 0 or max_atoms <= 0:
        device = question_features.device if question_features is not None else torch.device("cpu")
        return torch.empty((0, max(1, metric_dim)), dtype=torch.float32, device=device)
    question = question_features.float()
    if metric_dim > 0 and question.shape[-1] > metric_dim:
        question = F.adaptive_avg_pool1d(question.unsqueeze(1), metric_dim).squeeze(1)
    question = F.normalize(question, p=2, dim=-1, eps=1e-6)
    atom_count = min(max_atoms, int(question.shape[0]))
    if atom_count <= 0:
        return question[:0]

    center = F.normalize(question.mean(dim=0), p=2, dim=-1, eps=1e-6)
    selected = [int(torch.argmin(question @ center).item())]
    min_distance = 1.0 - question @ question[selected[0]]
    for _ in range(1, atom_count):
        min_distance[selected] = -1.0
        next_idx = int(torch.argmax(min_distance).item())
        if float(min_distance[next_idx].item()) < 0.01:
            break
        selected.append(next_idx)
        distance = 1.0 - question @ question[next_idx]
        min_distance = torch.minimum(min_distance, distance)
    return question[torch.tensor(selected, dtype=torch.long, device=question.device)]


def _question_relevance(
    atoms: torch.Tensor,
    metric_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    if atoms.numel() == 0:
        return (
            metric_features.new_empty((0, metric_features.shape[0])),
            metric_features.new_empty((0,)),
            0.0,
        )
    raw = atoms @ metric_features.transpose(0, 1)
    relevance = _minmax(raw, dim=1)
    top_count = max(1, int(math.ceil(metric_features.shape[0] * 0.05)))
    top_mean = torch.topk(relevance, k=top_count, dim=1).values.mean(dim=1)
    median = relevance.median(dim=1).values
    spread = relevance.std(dim=1, unbiased=False).clamp_min(1e-6)
    confidence = (((top_mean - median) / spread - 0.75) / 2.0).clamp(0.0, 1.0)
    if float(confidence.sum().item()) <= 1e-6:
        atom_weights = torch.full_like(confidence, 1.0 / max(1, confidence.numel()))
    else:
        atom_weights = confidence / confidence.sum()
    return relevance, atom_weights, float(confidence.mean().item())


def _local_detail(video_features: torch.Tensor, height: int, width: int) -> torch.Tensor:
    frame_count, tokens_per_frame, _ = video_features.shape
    if height * width != tokens_per_frame:
        return torch.zeros((frame_count, tokens_per_frame), dtype=torch.float32, device=video_features.device)
    normed = F.normalize(video_features.float(), p=2, dim=-1, eps=1e-6)
    grid = normed.view(frame_count, height, width, -1).permute(0, 3, 1, 2)
    local_mean = F.avg_pool2d(grid, kernel_size=3, stride=1, padding=1, count_include_pad=False)
    local_mean = F.normalize(local_mean, p=2, dim=1, eps=1e-6)
    detail = 1.0 - torch.sum(grid * local_mean, dim=1)
    return _minmax(detail.flatten(1), dim=-1)


def _candidate_pool(
    budget: int,
    quality: torch.Tensor,
    component_ids: torch.Tensor,
    temporal_ids: torch.Tensor,
    cell_ids: torch.Tensor,
    query_relevance: torch.Tensor,
    query_confidence: float,
    multiplier: float,
) -> tuple[torch.Tensor, list[int], list[int]]:
    total_tokens = int(quality.numel())
    candidate_limit = min(total_tokens, max(budget, int(math.ceil(budget * max(1.0, multiplier)))))
    candidate_set: set[int] = set()
    mandatory: list[int] = []
    query_seeds: list[int] = []

    # Temporal certificates are the only hard coverage constraint.
    for temporal_id in torch.unique(temporal_ids).detach().cpu().tolist():
        members = torch.where(temporal_ids == int(temporal_id))[0]
        if members.numel() == 0:
            continue
        best = int(members[torch.argmax(quality[members])].item())
        candidate_set.add(best)
        mandatory.append(best)

    # Keep two temporally separated peaks for every reliable query atom.
    if query_relevance.numel() > 0 and query_confidence >= 0.10:
        temporal_count = int(temporal_ids.max().item()) + 1
        for atom_scores in query_relevance:
            per_bin_best: list[tuple[float, int]] = []
            for temporal_id in range(temporal_count):
                members = torch.where(temporal_ids == temporal_id)[0]
                if members.numel() == 0:
                    continue
                local = int(torch.argmax(atom_scores[members]).item())
                token_idx = int(members[local].item())
                per_bin_best.append((float(atom_scores[token_idx].item()), token_idx))
            per_bin_best.sort(reverse=True)
            for _, token_idx in per_bin_best[:2]:
                candidate_set.add(token_idx)
                mandatory.append(token_idx)
                query_seeds.append(token_idx)

    # Add the strongest representative from each semantic trajectory.
    component_cpu = component_ids.detach().cpu().tolist()
    quality_cpu = quality.detach().float().cpu().tolist()
    representatives: dict[int, int] = {}
    for token_idx, component_id in enumerate(component_cpu):
        previous = representatives.get(component_id)
        if previous is None or quality_cpu[token_idx] > quality_cpu[previous]:
            representatives[component_id] = token_idx
    component_reps = sorted(representatives.values(), key=lambda idx: quality_cpu[idx], reverse=True)
    for token_idx in component_reps:
        if len(candidate_set) >= candidate_limit:
            break
        candidate_set.add(token_idx)

    # Spatial-temporal cells contribute candidates, but are not forced into output.
    for cell_id in torch.unique(cell_ids).detach().cpu().tolist():
        members = torch.where(cell_ids == int(cell_id))[0]
        if members.numel() == 0:
            continue
        best = int(members[torch.argmax(quality[members])].item())
        candidate_set.add(best)
        if len(candidate_set) >= candidate_limit:
            break

    if len(candidate_set) < candidate_limit:
        for token_idx in torch.argsort(quality, descending=True).detach().cpu().tolist():
            candidate_set.add(int(token_idx))
            if len(candidate_set) >= candidate_limit:
                break

    mandatory = list(dict.fromkeys(mandatory))[:budget]
    query_seeds = list(dict.fromkeys(query_seeds))
    candidates = torch.tensor(sorted(candidate_set), dtype=torch.long, device=quality.device)
    return candidates, mandatory, query_seeds


def _stochastic_coreset(
    *,
    budget: int,
    main_budget: int,
    metric_features: torch.Tensor,
    candidates: torch.Tensor,
    mandatory: list[int],
    demand_weight: torch.Tensor,
    query_relevance: torch.Tensor,
    atom_weights: torch.Tensor,
    temporal_ids: torch.Tensor,
    temporal_value: torch.Tensor,
    cell_ids: torch.Tensor,
    cell_value: torch.Tensor,
    component_ids: torch.Tensor,
    component_sizes: torch.Tensor,
    query_weight: float,
    temporal_weight: float,
    detail_weight: float,
) -> list[int]:
    device = metric_features.device
    total_tokens = int(metric_features.shape[0])
    candidate_list = candidates.detach().cpu().tolist()
    candidate_to_column = {token_idx: col for col, token_idx in enumerate(candidate_list)}
    metric_dtype = torch.float16 if device.type == "cuda" else torch.float32
    similarity = metric_features.to(metric_dtype) @ metric_features[candidates].to(metric_dtype).transpose(0, 1)
    similarity = ((similarity + 1.0) * 0.5).clamp_(0.0, 1.0)

    temporal_count = int(temporal_ids.max().item()) + 1
    cell_count = int(cell_ids.max().item()) + 1
    component_count = int(component_sizes.numel())
    visual_coverage = torch.zeros(total_tokens, dtype=torch.float32, device=device)
    query_coverage = torch.zeros(query_relevance.shape[0], dtype=torch.float32, device=device)
    temporal_coverage = torch.zeros(temporal_count, dtype=torch.float32, device=device)
    cell_coverage = torch.zeros(cell_count, dtype=torch.float32, device=device)
    component_coverage = torch.zeros(component_count, dtype=torch.float32, device=device)
    component_weight = torch.sqrt(component_sizes.to(device=device, dtype=torch.float32))
    component_weight = component_weight / component_weight.sum().clamp_min(1e-6)

    q_weight = min(max(float(query_weight), 0.0), 0.45)
    t_weight = min(max(float(temporal_weight), 0.0), 0.35)
    d_weight = min(max(float(detail_weight), 0.0), 0.25)
    visual_weight = max(0.20, 1.0 - q_weight - t_weight - d_weight)

    selected: list[int] = []
    selected_set: set[int] = set()

    def update(token_idx: int) -> None:
        nonlocal visual_coverage, query_coverage
        if token_idx in selected_set:
            return
        selected.append(token_idx)
        selected_set.add(token_idx)
        column = candidate_to_column.get(token_idx)
        if column is None:
            token_similarity = ((metric_features @ metric_features[token_idx]) + 1.0) * 0.5
        else:
            token_similarity = similarity[:, column].float()
        visual_coverage = torch.maximum(visual_coverage, token_similarity)
        if query_relevance.numel() > 0:
            query_coverage = torch.maximum(query_coverage, query_relevance[:, token_idx])
        temporal_id = int(temporal_ids[token_idx].item())
        cell_id = int(cell_ids[token_idx].item())
        component_id = int(component_ids[token_idx].item())
        temporal_coverage[temporal_id] = torch.maximum(temporal_coverage[temporal_id], temporal_value[token_idx])
        cell_coverage[cell_id] = torch.maximum(cell_coverage[cell_id], cell_value[token_idx])
        component_coverage[component_id] = torch.maximum(component_coverage[component_id], cell_value[token_idx])

    for token_idx in mandatory:
        if len(selected) >= budget:
            break
        update(int(token_idx))

    remaining = [idx for idx in candidate_list if idx not in selected_set]
    rng = random.Random(0)
    sample_size = max(12, int(math.ceil(max(1.0, len(candidate_list) / max(1, budget)) * math.log(10.0))))
    main_budget = max(len(selected), min(budget, main_budget))

    while len(selected) < main_budget and remaining:
        subset = rng.sample(remaining, k=min(sample_size, len(remaining)))
        subset_tensor = torch.tensor(subset, dtype=torch.long, device=device)
        subset_columns = torch.tensor([candidate_to_column[idx] for idx in subset], dtype=torch.long, device=device)
        subset_similarity = similarity[:, subset_columns].float()
        visual_gain = ((subset_similarity - visual_coverage.unsqueeze(1)).clamp_min(0.0) * demand_weight.unsqueeze(1)).sum(dim=0)

        if query_relevance.numel() > 0:
            query_gain = (
                (query_relevance[:, subset_tensor] - query_coverage.unsqueeze(1)).clamp_min(0.0)
                * atom_weights.unsqueeze(1)
            ).sum(dim=0)
        else:
            query_gain = torch.zeros_like(visual_gain)

        subset_temporal = temporal_ids[subset_tensor]
        subset_cells = cell_ids[subset_tensor]
        subset_components = component_ids[subset_tensor]
        temporal_gain = (
            temporal_value[subset_tensor] - temporal_coverage[subset_temporal]
        ).clamp_min(0.0) / max(1, temporal_count)
        spatial_gain = (
            cell_value[subset_tensor] - cell_coverage[subset_cells]
        ).clamp_min(0.0) / max(1, cell_count)
        track_gain = (
            cell_value[subset_tensor] - component_coverage[subset_components]
        ).clamp_min(0.0) * component_weight[subset_components]
        gain = (
            visual_weight * visual_gain
            + q_weight * query_gain
            + t_weight * temporal_gain
            + d_weight * (0.5 * spatial_gain + 0.5 * track_gain)
        )
        best_local = int(torch.argmax(gain).item())
        best_token = int(subset[best_local])
        update(best_token)
        remaining.remove(best_token)

    # Refill from the full pool using weighted reconstruction error. This lets
    # raw, previously filtered details re-enter when the coreset misses them.
    available = torch.ones(total_tokens, dtype=torch.bool, device=device)
    if selected:
        available[torch.tensor(selected, dtype=torch.long, device=device)] = False
    query_token_score = (
        (query_relevance * atom_weights.unsqueeze(1)).sum(dim=0)
        if query_relevance.numel() > 0
        else torch.zeros(total_tokens, dtype=torch.float32, device=device)
    )
    while len(selected) < budget and bool(available.any()):
        residual = demand_weight * (1.0 - visual_coverage).clamp_min(0.0)
        residual = residual + q_weight * query_token_score * (1.0 - visual_coverage).clamp_min(0.0)
        residual = residual.masked_fill(~available, -1.0)
        token_idx = int(torch.argmax(residual).item())
        update(token_idx)
        available[token_idx] = False

    return selected[:budget]


def _build_plan(
    *,
    selected: torch.Tensor,
    metric_features: torch.Tensor,
    demand_weight: torch.Tensor,
    attention: torch.Tensor,
    query_score: torch.Tensor,
    temporal_ids: torch.Tensor,
    component_ids: torch.Tensor,
    fusion_alpha: float,
    temperature: float,
) -> CertVidPlan:
    total_tokens = int(metric_features.shape[0])
    budget = int(selected.numel())
    similarity = metric_features @ metric_features[selected].transpose(0, 1)
    source_temporal = temporal_ids.unsqueeze(1)
    anchor_temporal = temporal_ids[selected].unsqueeze(0)
    temporal_valid = (source_temporal - anchor_temporal).abs() <= 1
    similarity = similarity.masked_fill(~temporal_valid, -2.0)
    same_component = component_ids.unsqueeze(1) == component_ids[selected].unsqueeze(0)
    similarity = similarity + 0.08 * same_component.float()

    topk = min(2, budget)
    values, assignment = torch.topk(similarity, k=topk, dim=1, largest=True)
    weights = torch.softmax(values.float() / max(1e-4, float(temperature)), dim=1)

    # Every anchor keeps an identity path; otherwise two nearby anchors could
    # collapse into each other during the shared DeepStack aggregation.
    anchor_positions = torch.arange(budget, dtype=torch.long, device=selected.device)
    assignment[selected, 0] = anchor_positions
    weights[selected] = 0.0
    weights[selected, 0] = 1.0

    source_mass = (0.5 + 0.5 * demand_weight * float(total_tokens)).clamp(0.25, 2.0)
    protection = torch.maximum(attention[selected], query_score[selected])
    protected_count = min(budget, max(1, int(math.ceil(0.15 * budget))))
    protected = torch.zeros(budget, dtype=torch.bool, device=selected.device)
    protected[torch.topk(protection, k=protected_count, largest=True).indices] = True
    alpha = torch.full((budget,), min(max(float(fusion_alpha), 0.0), 0.75), dtype=torch.float32, device=selected.device)
    alpha = alpha * (1.0 - 0.65 * protection.clamp(0.0, 1.0))
    alpha[protected] = 0.0
    return CertVidPlan(
        anchor_indices=selected,
        assignment_indices=assignment,
        assignment_weights=weights,
        source_mass=source_mass,
        fusion_alpha=alpha,
        raw_token_count=total_tokens,
    )


def apply_certvid_plan(flat_features: torch.Tensor, plan: CertVidPlan) -> torch.Tensor:
    """Apply a CertVID sparse aggregation plan to one feature hierarchy."""
    if flat_features.ndim != 2:
        raise ValueError(f"expected flat features [N, D], got {tuple(flat_features.shape)}")
    if int(flat_features.shape[0]) != int(plan.raw_token_count):
        raise ValueError(
            f"CertVID plan expects {plan.raw_token_count} tokens, got {int(flat_features.shape[0])}"
        )
    budget = int(plan.anchor_indices.numel())
    feature_dim = int(flat_features.shape[-1])
    anchor_indices = plan.anchor_indices.to(device=flat_features.device, dtype=torch.long)
    assignment_indices = plan.assignment_indices.to(device=flat_features.device, dtype=torch.long)
    assignment_weights = plan.assignment_weights.to(device=flat_features.device, dtype=torch.float32)
    source_mass = plan.source_mass.to(device=flat_features.device, dtype=torch.float32)
    fusion_alpha = plan.fusion_alpha.to(device=flat_features.device, dtype=torch.float32)
    accumulation = torch.zeros((budget, feature_dim), dtype=torch.float32, device=flat_features.device)
    mass = torch.zeros((budget,), dtype=torch.float32, device=flat_features.device)
    source = flat_features.float()
    for neighbor in range(assignment_indices.shape[1]):
        target = assignment_indices[:, neighbor]
        weight = assignment_weights[:, neighbor] * source_mass
        accumulation.index_add_(0, target, source * weight.unsqueeze(1))
        mass.index_add_(0, target, weight)
    pooled = accumulation / mass.clamp_min(1e-6).unsqueeze(1)
    anchors = source[anchor_indices]
    alpha = fusion_alpha.unsqueeze(1)
    output = anchors + alpha * (pooled - anchors)
    return torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).to(dtype=flat_features.dtype)


def certvid_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
    question_features: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct a question-aware temporal evidence coreset under one budget."""
    if video_features.ndim != 3:
        raise ValueError(f"expected video_features [T, HW, D], got {tuple(video_features.shape)}")
    frame_count, tokens_per_frame, _ = video_features.shape
    total_tokens = int(frame_count * tokens_per_frame)
    budget = max(1, min(total_tokens, int(round(total_tokens * _effective_ratio(flashvid_config)))))
    if budget >= total_tokens:
        indices = torch.arange(total_tokens, dtype=torch.long, device=video_features.device)
        plan = CertVidPlan(
            anchor_indices=indices,
            assignment_indices=indices.unsqueeze(1),
            assignment_weights=torch.ones((total_tokens, 1), dtype=torch.float32, device=video_features.device),
            source_mass=torch.ones(total_tokens, dtype=torch.float32, device=video_features.device),
            fusion_alpha=torch.zeros(total_tokens, dtype=torch.float32, device=video_features.device),
            raw_token_count=total_tokens,
        )
        setattr(flashvid_config, "_certvid_plan", plan)
        return video_features.reshape(total_tokens, -1), indices

    metric_dim = max(32, _cfg_int(flashvid_config, "cert_metric_dim", 256))
    metric_flat = _metric_features(video_features, metric_dim)
    metric_frames = metric_flat.view(frame_count, tokens_per_frame, -1)
    height, width = _grid_hw(tokens_per_frame, flashvid_config)
    spatial_bins = max(1, _cfg_int(flashvid_config, "cert_spatial_bins", 3))
    coords, spatial_cells = _spatial_layout(tokens_per_frame, height, width, spatial_bins, video_features.device)
    frame_event, token_novelty, matches = _temporal_signals(
        metric_frames,
        coords,
        _cfg_float(flashvid_config, "cert_spatial_penalty", 0.08),
    )
    component_ids_cpu, component_sizes_cpu = _build_components(
        frame_count,
        tokens_per_frame,
        frame_event,
        matches,
        _cfg_float(flashvid_config, "cert_track_threshold", 0.82),
    )
    component_ids = component_ids_cpu.to(video_features.device)
    component_sizes = component_sizes_cpu.to(video_features.device)

    temporal_bins = min(frame_count, max(1, _cfg_int(flashvid_config, "cert_temporal_bins", 8)))
    frame_ids = torch.arange(frame_count, device=video_features.device).repeat_interleave(tokens_per_frame)
    temporal_ids = torch.div(frame_ids * temporal_bins, max(1, frame_count), rounding_mode="floor").clamp_max(temporal_bins - 1)
    cell_ids = temporal_ids * (spatial_bins * spatial_bins) + spatial_cells.repeat(frame_count)

    attention = _rank_normalize(cls_attention.float()).reshape(-1)
    novelty = token_novelty.reshape(-1)
    detail = _local_detail(video_features, height, width).reshape(-1)
    event = frame_event.repeat_interleave(tokens_per_frame)
    atoms = _question_atoms(
        question_features,
        max(0, _cfg_int(flashvid_config, "cert_query_atoms", 6)),
        metric_dim,
    ).to(video_features.device)
    query_relevance, atom_weights, query_confidence = _question_relevance(atoms, metric_flat)
    query_score = (
        (query_relevance * atom_weights.unsqueeze(1)).sum(dim=0)
        if query_relevance.numel() > 0
        else torch.zeros(total_tokens, dtype=torch.float32, device=video_features.device)
    )

    demand_weight = 0.40 + 0.22 * attention + 0.18 * novelty + 0.12 * detail + 0.08 * event
    demand_weight = demand_weight / demand_weight.sum().clamp_min(1e-6)
    quality = _minmax(
        0.28 * attention + 0.24 * novelty + 0.20 * detail + 0.14 * event + 0.14 * query_score,
        dim=0,
    )
    cell_value = _minmax(0.55 * detail + 0.30 * attention + 0.15 * novelty, dim=0)
    temporal_value = _minmax(0.55 * event + 0.30 * novelty + 0.15 * query_score, dim=0)

    candidates, mandatory, query_seeds = _candidate_pool(
        budget,
        quality,
        component_ids,
        temporal_ids,
        cell_ids,
        query_relevance,
        query_confidence,
        _cfg_float(flashvid_config, "cert_candidate_multiplier", 3.0),
    )
    repair_ratio = min(max(_cfg_float(flashvid_config, "cert_repair_ratio", 0.20), 0.0), 0.50)
    main_budget = max(len(mandatory), budget - int(round(budget * repair_ratio)))
    selected_list = _stochastic_coreset(
        budget=budget,
        main_budget=main_budget,
        metric_features=metric_flat,
        candidates=candidates,
        mandatory=mandatory,
        demand_weight=demand_weight,
        query_relevance=query_relevance,
        atom_weights=atom_weights,
        temporal_ids=temporal_ids,
        temporal_value=temporal_value,
        cell_ids=cell_ids,
        cell_value=cell_value,
        component_ids=component_ids,
        component_sizes=component_sizes,
        query_weight=_cfg_float(flashvid_config, "cert_query_weight", 0.20) * query_confidence,
        temporal_weight=_cfg_float(flashvid_config, "cert_temporal_weight", 0.20),
        detail_weight=_cfg_float(flashvid_config, "cert_detail_weight", 0.10),
    )
    selected = torch.tensor(sorted(selected_list), dtype=torch.long, device=video_features.device)
    plan = _build_plan(
        selected=selected,
        metric_features=metric_flat,
        demand_weight=demand_weight,
        attention=attention,
        query_score=query_score,
        temporal_ids=temporal_ids,
        component_ids=component_ids,
        fusion_alpha=_cfg_float(flashvid_config, "cert_fusion_alpha", 0.25),
        temperature=_cfg_float(flashvid_config, "cert_assignment_temperature", 0.07),
    )
    output = apply_certvid_plan(video_features.reshape(total_tokens, -1), plan)
    setattr(flashvid_config, "_certvid_plan", plan)
    flashvid_config.vision_token_length = int(output.shape[0])
    flashvid_config.visual_token_length = int(output.shape[0])
    flashvid_config.llm_token_length = None
    setattr(flashvid_config, "last_adapter_variant", "certvid")
    setattr(flashvid_config, "last_adapter_raw_tokens", float(total_tokens))
    setattr(flashvid_config, "last_adapter_output_tokens", float(output.shape[0]))
    setattr(flashvid_config, "last_cert_target_tokens", float(budget))
    setattr(flashvid_config, "last_cert_candidate_tokens", float(candidates.numel()))
    setattr(flashvid_config, "last_cert_component_count", float(component_sizes.numel()))
    setattr(flashvid_config, "last_cert_query_confidence", float(query_confidence))
    setattr(flashvid_config, "last_cert_query_seed_count", float(len(query_seeds)))
    setattr(flashvid_config, "last_cert_repair_tokens", float(max(0, budget - main_budget)))
    return output, selected
