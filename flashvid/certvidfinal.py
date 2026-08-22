"""Self-contained no-certificate, no-trajectory CertVID final method."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, MutableMapping, Optional

import torch
import torch.nn.functional as F

from .configuration_flashvid import FlashVidConfig


# V1 primitives required by V3.

@dataclass
class CertVidPlan:
    """Sparse assignment shared by base and Qwen3 DeepStack features."""

    anchor_indices: torch.Tensor
    assignment_indices: torch.Tensor
    assignment_weights: torch.Tensor
    source_mass: torch.Tensor
    fusion_alpha: torch.Tensor
    raw_token_count: int

def _final_v1_cfg_float(config: FlashVidConfig, name: str, default: float) -> float:
    try:
        return float(getattr(config, name, default))
    except (TypeError, ValueError):
        return float(default)

def _final_v1_cfg_int(config: FlashVidConfig, name: str, default: int) -> int:
    try:
        return int(getattr(config, name, default))
    except (TypeError, ValueError):
        return int(default)

def _final_v1_grid_hw(tokens_per_frame: int, config: FlashVidConfig) -> tuple[int, int]:
    height = _final_v1_cfg_int(config, "H", 0)
    width = _final_v1_cfg_int(config, "W", 0)
    if height > 0 and width > 0 and height * width == tokens_per_frame:
        return height, width
    height = max(1, int(math.sqrt(tokens_per_frame)))
    while height > 1 and tokens_per_frame % height != 0:
        height -= 1
    return height, max(1, tokens_per_frame // height)

def _final_v1_minmax(values: torch.Tensor, dim: int = -1) -> torch.Tensor:
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
    lo = values.amin(dim=dim, keepdim=True)
    hi = values.amax(dim=dim, keepdim=True)
    return ((values - lo) / (hi - lo + 1e-6)).clamp_(0.0, 1.0)

def _final_v1_rank_normalize(values: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] <= 1:
        return torch.ones_like(values, dtype=torch.float32)
    order = torch.argsort(values.float(), dim=-1, stable=True)
    ranks = torch.argsort(order, dim=-1, stable=True).float()
    return ranks / float(values.shape[-1] - 1)

def _final_v1_metric_features(features: torch.Tensor, metric_dim: int) -> torch.Tensor:
    flat = features.reshape(-1, features.shape[-1]).float()
    if metric_dim > 0 and flat.shape[-1] > metric_dim:
        flat = F.adaptive_avg_pool1d(flat.unsqueeze(1), metric_dim).squeeze(1)
    return F.normalize(flat, p=2, dim=-1, eps=1e-6)

def _final_v1_spatial_layout(
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

def _final_v1_question_atoms(
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

def _final_v1_question_relevance(
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
    relevance = _final_v1_minmax(raw, dim=1)
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

def _final_v1_local_detail(video_features: torch.Tensor, height: int, width: int) -> torch.Tensor:
    frame_count, tokens_per_frame, _ = video_features.shape
    if height * width != tokens_per_frame:
        return torch.zeros((frame_count, tokens_per_frame), dtype=torch.float32, device=video_features.device)
    normed = F.normalize(video_features.float(), p=2, dim=-1, eps=1e-6)
    grid = normed.view(frame_count, height, width, -1).permute(0, 3, 1, 2)
    local_mean = F.avg_pool2d(grid, kernel_size=3, stride=1, padding=1, count_include_pad=False)
    local_mean = F.normalize(local_mean, p=2, dim=1, eps=1e-6)
    detail = 1.0 - torch.sum(grid * local_mean, dim=1)
    return _final_v1_minmax(detail.flatten(1), dim=-1)

def _final_v1_build_plan(
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

def _final_v1apply_certvid_plan(flat_features: torch.Tensor, plan: CertVidPlan) -> torch.Tensor:
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

# V3 selection, refinement, fusion, and diagnostics.

_PROFILE_ENV = "CERTV3_PROFILE_PHASES"

_EXACT_OPTIMIZATION_AUDIT_ENV = "CERTV3_AUDIT_EXACT_OPTIMIZATIONS"

_EXACT_CUDA_GRAPH_ENV = "CERTV3_USE_EXACT_CUDA_GRAPHS"

_DOPT_GRAPH_CACHE: dict[tuple[object, ...], dict[str, object]] = {}

_FEDOROV_GRAPH_CACHE: dict[tuple[object, ...], dict[str, object]] = {}

def _profile_enabled(features: torch.Tensor) -> bool:
    value = os.environ.get(_PROFILE_ENV, "").strip().lower()
    return features.is_cuda and value in {"1", "true", "yes", "on"}

def _exact_optimization_audit_enabled() -> bool:
    value = os.environ.get(_EXACT_OPTIMIZATION_AUDIT_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "on"}

def _exact_cuda_graph_enabled(tensor: torch.Tensor) -> bool:
    value = os.environ.get(_EXACT_CUDA_GRAPH_ENV, "").strip().lower()
    return tensor.is_cuda and value in {"1", "true", "yes", "on"}

def _profile_record(
    events: Optional[dict[str, tuple[torch.cuda.Event, torch.cuda.Event]]],
    name: str,
    boundary: int,
) -> None:
    """Record one phase boundary without synchronizing the CUDA stream."""
    if events is None:
        return
    pair = events.get(name)
    if pair is None:
        pair = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        events[name] = pair
    pair[boundary].record()

def _cfg_bool(config: FlashVidConfig, name: str, default: bool) -> bool:
    value = getattr(config, name, default)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"{name} must be a boolean, got {value!r}")
    return bool(value)

def _write_certv3_diagnostics(
    config: FlashVidConfig,
    diagnostics: dict[str, Any],
) -> None:
    """Append scalar diagnostics without changing the V3 selection path."""
    template = os.environ.get("CERTV3_DIAGNOSTICS_JSONL", "").strip()
    if not template:
        return

    rank = os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))
    path = template.replace("{rank}", rank).replace("{pid}", str(os.getpid()))
    if "{rank}" not in template and "{pid}" not in template:
        root, extension = os.path.splitext(path)
        path = f"{root}.rank{rank}{extension or '.jsonl'}"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    record = dict(diagnostics)
    record["sample_id"] = str(getattr(config, "_debug_sample_id", "unknown"))
    record["task"] = getattr(config, "_certvid_task_name", None)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")

def _effective_ratio(config: FlashVidConfig) -> float:
    ratio = _final_v1_cfg_float(config, "retention_ratio", 0.10)
    if bool(getattr(config, "certv3_budget_uses_expansion", True)):
        ratio *= _final_v1_cfg_float(config, "expansion", 1.0)
    return min(1.0, max(0.0, ratio))

def _identity_plan(total_tokens: int, device: torch.device) -> CertVidPlan:
    indices = torch.arange(total_tokens, dtype=torch.long, device=device)
    return CertVidPlan(
        anchor_indices=indices,
        assignment_indices=indices.unsqueeze(1),
        assignment_weights=torch.ones((total_tokens, 1), dtype=torch.float32, device=device),
        source_mass=torch.ones(total_tokens, dtype=torch.float32, device=device),
        fusion_alpha=torch.zeros(total_tokens, dtype=torch.float32, device=device),
        raw_token_count=total_tokens,
    )

def _whiten_features(features: torch.Tensor, strength: float) -> torch.Tensor:
    """Shrinkage whitening exposes complementary directions without amplifying noise."""
    strength = min(1.0, max(0.0, float(strength)))
    centered = features.float() - features.float().mean(dim=0, keepdim=True)
    if strength <= 1e-6 or centered.shape[0] <= 1:
        return F.normalize(centered, p=2, dim=-1, eps=1e-6)

    covariance = centered.transpose(0, 1) @ centered
    covariance = covariance / float(max(1, centered.shape[0] - 1))
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    mean_eigenvalue = eigenvalues.mean().clamp_min(1e-6)
    eigenvalues = eigenvalues.clamp_min(mean_eigenvalue * 1e-4)
    # strength=0 leaves the spectrum unchanged; strength=1 fully whitens it.
    scales = torch.pow(eigenvalues / mean_eigenvalue, -0.5 * strength)
    whitened = (centered @ eigenvectors) * scales.unsqueeze(0)
    return F.normalize(
        torch.nan_to_num(whitened, nan=0.0, posinf=0.0, neginf=0.0),
        p=2,
        dim=-1,
        eps=1e-6,
    )

def _design_features(
    *,
    metric_features: torch.Tensor,
    quality: torch.Tensor,
    temporal_ids: torch.Tensor,
    spatial_ids: torch.Tensor,
    attention: torch.Tensor,
    novelty: torch.Tensor,
    curvature: torch.Tensor,
    event: torch.Tensor,
    detail: torch.Tensor,
    component_support: torch.Tensor,
    query_relevance: torch.Tensor,
    atom_weights: torch.Tensor,
    query_confidence: float,
    temporal_count: int,
    spatial_count: int,
    structural_weight: float,
    use_spatiotemporal_design: bool,
    whitening_strength: float,
    quality_floor: float,
) -> torch.Tensor:
    visual = _whiten_features(metric_features, whitening_strength)
    temporal = F.one_hot(temporal_ids, num_classes=temporal_count).float()
    spatial = F.one_hot(spatial_ids, num_classes=spatial_count).float()
    signals = torch.stack(
        [attention, novelty, curvature, event, detail, component_support],
        dim=1,
    )
    signals = F.normalize(signals, p=2, dim=-1, eps=1e-6)

    structural_weight = min(0.80, max(0.0, float(structural_weight)))
    visual_weight = max(0.20, 1.0 - structural_weight)
    query_gate = min(1.0, max(0.0, float(query_confidence)))
    query_share = 0.20 * structural_weight * query_gate if query_relevance.numel() > 0 else 0.0
    structural_remainder = max(0.0, structural_weight - query_share)
    parts = [visual * math.sqrt(visual_weight)]
    if use_spatiotemporal_design:
        parts.extend(
            [
                temporal * math.sqrt(0.45 * structural_remainder),
                spatial * math.sqrt(0.25 * structural_remainder),
                signals * math.sqrt(0.30 * structural_remainder),
            ]
        )
    else:
        # Preserve total structural mass so this ablation removes only the
        # explicit temporal/spatial coordinate axes, not their budget.
        parts.append(signals * math.sqrt(structural_remainder))
    if query_relevance.numel() > 0 and query_share > 0.0:
        query_axes = query_relevance.transpose(0, 1) * torch.sqrt(
            atom_weights.clamp_min(1e-6)
        ).unsqueeze(0)
        parts.append(query_axes * math.sqrt(query_share))

    design = F.normalize(torch.cat(parts, dim=1), p=2, dim=-1, eps=1e-6)
    quality_floor = min(1.0, max(1e-4, float(quality_floor)))
    row_mass = quality_floor + (1.0 - quality_floor) * quality.clamp(0.0, 1.0)
    return design * torch.sqrt(row_mass).unsqueeze(1)

def _stable_score_token_order(
    scores: torch.Tensor,
    tokens: torch.Tensor,
) -> torch.Tensor:
    """Match Python's ``(-score, token)`` ordering without a CPU sync."""
    token_order = torch.argsort(tokens, stable=True)
    return token_order[
        torch.argsort(scores[token_order], descending=True, stable=True)
    ]

def _best_token_per_group(
    scores: torch.Tensor,
    groups: torch.Tensor,
    token_ids: torch.Tensor,
    token_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return each group's highest-score, lowest-id token entirely on device."""
    unique_groups, inverse = torch.unique(groups, sorted=True, return_inverse=True)
    group_count = int(unique_groups.numel())
    if group_count == 0:
        return token_ids[:0], scores[:0]

    maxima = torch.full(
        (group_count,),
        float("-inf"),
        dtype=scores.dtype,
        device=scores.device,
    )
    maxima.scatter_reduce_(0, inverse, scores, reduce="amax", include_self=True)
    sentinel = int(token_count)
    eligible = torch.where(
        scores == maxima[inverse],
        token_ids,
        torch.full_like(token_ids, sentinel),
    )
    best = torch.full(
        (group_count,),
        sentinel,
        dtype=torch.long,
        device=scores.device,
    )
    best.scatter_reduce_(0, inverse, eligible, reduce="amin", include_self=True)
    valid = best != sentinel
    return best[valid], maxima[valid]

def _candidate_pool_without_certificates(
    *,
    quality: torch.Tensor,
    temporal_ids: torch.Tensor,
    spatial_ids: torch.Tensor,
    query_relevance: torch.Tensor,
    limit: int,
) -> torch.Tensor:
    """Exact no-certificate offer stream without host transfers or Python sets."""
    total_tokens = int(quality.numel())
    token_ids = torch.arange(total_tokens, dtype=torch.long, device=quality.device)
    offer_streams: list[torch.Tensor] = []

    if query_relevance.numel() > 0:
        atom_count = int(query_relevance.shape[0])
        query_tokens = token_ids.unsqueeze(0).expand(atom_count, -1).reshape(-1)
        query_groups = (
            torch.arange(atom_count, device=quality.device).unsqueeze(1)
            * total_tokens
            + temporal_ids.unsqueeze(0)
        ).reshape(-1)
        best, scores = _best_token_per_group(
            query_relevance.reshape(-1),
            query_groups,
            query_tokens,
            total_tokens,
        )
        offer_streams.append(best[_stable_score_token_order(scores, best)])

    joint_cells = temporal_ids * total_tokens + spatial_ids
    best, scores = _best_token_per_group(
        quality,
        joint_cells,
        token_ids,
        total_tokens,
    )
    offer_streams.append(best[_stable_score_token_order(scores, best)])
    offer_streams.append(torch.argsort(quality, descending=True, stable=True))

    offers = torch.cat(offer_streams)
    positions = torch.arange(offers.numel(), dtype=torch.long, device=quality.device)
    first = torch.full(
        (total_tokens,),
        offers.numel(),
        dtype=torch.long,
        device=quality.device,
    )
    first.scatter_reduce_(0, offers, positions, reduce="amin", include_self=True)
    candidates = torch.argsort(first, stable=True)[:limit]
    return torch.sort(candidates).values

def _score_only_select(
    *,
    quality: torch.Tensor,
    candidates: torch.Tensor,
    mandatory: list[int],
    budget: int,
) -> torch.Tensor:
    """Fill the fixed budget by existing scalar scores without D-optimal design."""
    candidate_set = set(int(token) for token in candidates.detach().cpu().tolist())
    selected = list(dict.fromkeys(int(token) for token in mandatory if int(token) in candidate_set))
    selected_set = set(selected)
    ranked = sorted(
        candidate_set,
        key=lambda token: (-float(quality[token].item()), token),
    )
    for token in ranked:
        if len(selected) >= budget:
            break
        if token not in selected_set:
            selected.append(token)
            selected_set.add(token)
    if len(selected) != budget:
        raise RuntimeError(f"score-only selection produced {len(selected)} tokens for budget {budget}")
    return torch.tensor(selected, dtype=torch.long, device=candidates.device)

def _selection_logdet(
    design: torch.Tensor,
    selected: torch.Tensor,
    ridge: float,
) -> float:
    rows = design[selected].float()
    identity = torch.eye(rows.shape[1], dtype=torch.float32, device=rows.device)
    information = max(1e-4, float(ridge)) * identity + rows.transpose(0, 1) @ rows
    sign, logabsdet = torch.linalg.slogdet(information)
    return float(logabsdet.item()) if float(sign.item()) > 0.0 else float("-inf")

def _d_optimal_unconstrained_columns(
    rows: torch.Tensor,
    budget: int,
    ridge: float,
) -> torch.Tensor:
    """Original greedy updates for the no-certificate path."""
    candidate_count, design_dim = rows.shape
    inverse = torch.eye(
        design_dim,
        dtype=torch.float32,
        device=rows.device,
    ) / ridge
    leverage = rows.square().sum(dim=1) / ridge
    active = torch.ones(candidate_count, dtype=torch.bool, device=rows.device)
    columns = torch.empty(budget, dtype=torch.long, device=rows.device)

    for step in range(budget):
        score = torch.log1p(leverage.clamp_min(0.0)).masked_fill(
            ~active,
            float("-inf"),
        )
        column = torch.argmax(score)
        remaining = torch.where(active)[0]
        column = torch.where(
            torch.isfinite(score[column]),
            column,
            remaining[0],
        )

        row = rows[column]
        direction = inverse @ row
        denominator = (1.0 + torch.dot(row, direction)).clamp_min(1e-6)
        projection = rows @ direction
        leverage = (
            leverage - projection.square() / denominator
        ).clamp_min(0.0)
        inverse = inverse - torch.outer(direction, direction) / denominator
        inverse = 0.5 * (inverse + inverse.transpose(0, 1))
        active[column] = False
        leverage[column] = -1.0
        columns[step] = column
    return columns

def _d_optimal_unconstrained_columns_capturable(
    rows: torch.Tensor,
    budget: int,
    ridge: float,
) -> torch.Tensor:
    """The eager D-optimal program with a fixed-shape finite fallback."""
    candidate_count, design_dim = rows.shape
    inverse = torch.eye(
        design_dim,
        dtype=torch.float32,
        device=rows.device,
    ) / ridge
    leverage = rows.square().sum(dim=1) / ridge
    active = torch.ones(candidate_count, dtype=torch.bool, device=rows.device)
    all_columns = torch.arange(candidate_count, dtype=torch.long, device=rows.device)
    selected_columns: list[torch.Tensor] = []

    for step in range(budget):
        score = torch.log1p(leverage.clamp_min(0.0)).masked_fill(
            ~active,
            float("-inf"),
        )
        column = torch.argmax(score)
        # This is the fixed-shape equivalent of torch.where(active)[0][0].
        fallback = torch.argmax(active.to(torch.int8))
        selected_score = torch.gather(score, 0, column.reshape(1)).squeeze(0)
        column = torch.where(torch.isfinite(selected_score), column, fallback)

        row = torch.index_select(rows, 0, column.reshape(1)).squeeze(0)
        direction = inverse @ row
        denominator = (1.0 + torch.dot(row, direction)).clamp_min(1e-6)
        projection = rows @ direction
        leverage = (
            leverage - projection.square() / denominator
        ).clamp_min(0.0)
        inverse = inverse - torch.outer(direction, direction) / denominator
        inverse = 0.5 * (inverse + inverse.transpose(0, 1))
        chosen = all_columns == column
        active = active & ~chosen
        leverage = torch.where(chosen, -torch.ones_like(leverage), leverage)
        selected_columns.append(column)
    return torch.stack(selected_columns)

def _d_optimal_unconstrained_columns_graph(
    rows: torch.Tensor,
    budget: int,
    ridge: float,
) -> torch.Tensor:
    """Replay the numerically unchanged greedy loop as one CUDA launch."""
    if not _exact_cuda_graph_enabled(rows):
        return _d_optimal_unconstrained_columns(rows, budget, ridge)

    key = (
        rows.device.index,
        tuple(rows.shape),
        rows.dtype,
        int(budget),
        float(ridge),
    )
    cached = _DOPT_GRAPH_CACHE.get(key)
    if cached is None:
        static_rows = rows.detach().clone()
        warmup_stream = torch.cuda.Stream(device=rows.device)
        warmup_stream.wait_stream(torch.cuda.current_stream(rows.device))
        with torch.cuda.stream(warmup_stream):
            for _ in range(2):
                _d_optimal_unconstrained_columns_capturable(
                    static_rows,
                    budget,
                    ridge,
                )
        torch.cuda.current_stream(rows.device).wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_columns = _d_optimal_unconstrained_columns_capturable(
                static_rows,
                budget,
                ridge,
            )
        cached = {
            "graph": graph,
            "rows": static_rows,
            "columns": static_columns,
        }
        _DOPT_GRAPH_CACHE[key] = cached
        print(
            "[CertVID V3] captured exact D-optimal CUDA graph "
            f"for rows={tuple(rows.shape)} budget={budget}"
        )

    static_rows = cached["rows"]
    assert isinstance(static_rows, torch.Tensor)
    static_rows.copy_(rows)
    graph = cached["graph"]
    assert isinstance(graph, torch.cuda.CUDAGraph)
    graph.replay()
    static_columns = cached["columns"]
    assert isinstance(static_columns, torch.Tensor)

    if _exact_optimization_audit_enabled():
        reference = _d_optimal_unconstrained_columns(rows, budget, ridge)
        if not torch.equal(static_columns, reference):
            raise RuntimeError("CUDA-graph D-optimal selection differs from eager reference")
    return static_columns

def _d_optimal_greedy(
    *,
    design: torch.Tensor,
    candidates: torch.Tensor,
    mandatory: list[int],
    budget: int,
    ridge: float,
) -> torch.Tensor:
    """Greedy regularized D-optimal design with exact rank-one leverage updates."""
    rows = design[candidates].float()
    candidate_count, design_dim = rows.shape
    if candidate_count < budget:
        raise RuntimeError(f"D-optimal pool has {candidate_count} candidates for budget {budget}")

    ridge = max(1e-4, float(ridge))
    if not mandatory:
        columns = _d_optimal_unconstrained_columns_graph(
            rows,
            budget,
            ridge,
        )
        return candidates[columns]

    inverse = torch.eye(design_dim, dtype=torch.float32, device=rows.device) / ridge
    leverage = rows.square().sum(dim=1) / ridge
    active = torch.ones(candidate_count, dtype=torch.bool, device=rows.device)
    selected_columns: list[int] = []

    def add(column: int) -> None:
        nonlocal inverse, leverage
        if not bool(active[column]):
            return
        row = rows[column]
        direction = inverse @ row
        denominator = (1.0 + torch.dot(row, direction)).clamp_min(1e-6)
        projection = rows @ direction
        leverage = (leverage - projection.square() / denominator).clamp_min(0.0)
        inverse = inverse - torch.outer(direction, direction) / denominator
        inverse = 0.5 * (inverse + inverse.transpose(0, 1))
        active[column] = False
        leverage[column] = -1.0
        selected_columns.append(column)

    if mandatory:
        token_to_column = {
            int(token): column
            for column, token in enumerate(candidates.detach().cpu().tolist())
        }
        for token in mandatory:
            column = token_to_column.get(int(token))
            if column is not None and len(selected_columns) < budget:
                add(column)

    remaining_steps = budget - len(selected_columns)
    greedy_columns = torch.empty(
        remaining_steps,
        dtype=torch.long,
        device=rows.device,
    )
    for step in range(remaining_steps):
        score = torch.log1p(leverage.clamp_min(0.0)).masked_fill(~active, float("-inf"))
        column = torch.argmax(score)
        remaining = torch.where(active)[0]
        column = torch.where(torch.isfinite(score[column]), column, remaining[0])

        row = rows[column]
        direction = inverse @ row
        denominator = (1.0 + torch.dot(row, direction)).clamp_min(1e-6)
        projection = rows @ direction
        leverage = (leverage - projection.square() / denominator).clamp_min(0.0)
        inverse = inverse - torch.outer(direction, direction) / denominator
        inverse = 0.5 * (inverse + inverse.transpose(0, 1))
        active[column] = False
        leverage[column] = -1.0
        greedy_columns[step] = column

    if selected_columns:
        mandatory_columns = torch.tensor(
            selected_columns,
            dtype=torch.long,
            device=rows.device,
        )
        columns = torch.cat((mandatory_columns, greedy_columns))
    else:
        columns = greedy_columns
    if int(columns.numel()) != budget:
        raise RuntimeError(f"D-optimal selector produced {len(selected_columns)} tokens for budget {budget}")
    return candidates[columns]

def _swap_refine_eager(
    *,
    selected: torch.Tensor,
    candidates: torch.Tensor,
    design: torch.Tensor,
    mandatory: list[int],
    ridge: float,
    steps: int,
    pool_size: int,
    margin: float,
) -> tuple[torch.Tensor, int, float]:
    """Fedorov-style exchanges improve log-det while preserving hard certificates."""
    steps = max(0, int(steps))
    if steps == 0 or selected.numel() == 0:
        return selected, 0, 0.0

    rows = design[candidates].float()
    ridge = max(1e-4, float(ridge))
    dimension = int(rows.shape[1])
    identity = torch.eye(dimension, dtype=torch.float32, device=rows.device)
    selected_columns = torch.searchsorted(candidates, selected).to(
        device=rows.device,
        dtype=torch.long,
    )
    mandatory_columns = torch.zeros(
        int(candidates.numel()),
        dtype=torch.bool,
        device=rows.device,
    )
    if mandatory:
        mandatory_tokens = torch.tensor(
            mandatory,
            dtype=torch.long,
            device=rows.device,
        )
        mandatory_columns[torch.searchsorted(candidates, mandatory_tokens)] = True
    swaps = 0

    for _ in range(steps):
        selected_rows = rows[selected_columns]
        information = ridge * identity + selected_rows.transpose(0, 1) @ selected_rows
        inverse = torch.linalg.inv(information)
        selected_leverage = torch.sum((selected_rows @ inverse) * selected_rows, dim=1).clamp(0.0, 1.0 - 1e-5)
        removal_loss = -torch.log1p(-selected_leverage)

        removable_positions = torch.where(~mandatory_columns[selected_columns])[0]
        if int(removable_positions.numel()) == 0:
            break
        # Match Python's (removal_loss, token_id) ordering with two stable
        # sorts, while keeping the values resident on the GPU.
        token_order = torch.argsort(
            candidates[selected_columns[removable_positions]],
            stable=True,
        )
        removable_positions = removable_positions[token_order]
        loss_order = torch.argsort(
            removal_loss[removable_positions],
            stable=True,
        )
        removable_positions = removable_positions[loss_order]
        removable_positions = removable_positions[: max(1, int(pool_size))]

        outside_mask = torch.ones(
            int(candidates.numel()),
            dtype=torch.bool,
            device=rows.device,
        )
        outside_mask[selected_columns] = False
        outside_tensor = torch.where(outside_mask)[0]
        if int(outside_tensor.numel()) == 0:
            break
        outside_rows = rows[outside_tensor]
        outside_leverage = torch.sum((outside_rows @ inverse) * outside_rows, dim=1)
        outside_order = torch.argsort(outside_leverage, descending=True, stable=True)
        outside_order = outside_order[: max(1, int(pool_size))]
        outside_tensor = outside_tensor[outside_order]
        outside_rows = rows[outside_tensor]

        local_delta_tensors: list[torch.Tensor] = []
        local_add_tensors: list[torch.Tensor] = []
        for position in removable_positions.unbind():
            removed = selected_rows[position]
            direction = inverse @ removed
            remove_denominator = (1.0 - torch.dot(removed, direction)).clamp_min(1e-5)
            inverse_without = inverse + torch.outer(direction, direction) / remove_denominator
            add_leverage = torch.sum((outside_rows @ inverse_without) * outside_rows, dim=1).clamp_min(0.0)
            delta = torch.log(remove_denominator) + torch.log1p(add_leverage)
            local = torch.argmax(delta)
            local_delta_tensors.append(delta[local])
            local_add_tensors.append(outside_tensor[local])

        local_deltas = torch.stack(local_delta_tensors)
        local_add_columns = torch.stack(local_add_tensors)
        best_local = torch.argmax(local_deltas)
        if float(local_deltas[best_local].item()) <= float(margin):
            break
        best_remove = removable_positions[best_local]
        selected_columns[best_remove] = local_add_columns[best_local]
        swaps += 1

    final_rows = rows[selected_columns]
    final_information = ridge * identity + final_rows.transpose(0, 1) @ final_rows
    sign, logabsdet = torch.linalg.slogdet(final_information)
    final_logdet = float(logabsdet.item()) if float(sign.item()) > 0.0 else float("-inf")
    return candidates[selected_columns], swaps, final_logdet

def _swap_refine_no_certificate_capturable(
    rows: torch.Tensor,
    selected_columns: torch.Tensor,
    ridge: float,
    steps: int,
    pool_size: int,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed-shape form of the original no-certificate Fedorov loop."""
    candidate_count, dimension = rows.shape
    budget = int(selected_columns.numel())
    identity = torch.eye(dimension, dtype=torch.float32, device=rows.device)
    all_columns = torch.arange(candidate_count, dtype=torch.long, device=rows.device)
    selected_columns = selected_columns.clone()
    selected_positions = torch.arange(budget, dtype=torch.long, device=rows.device)
    running = torch.ones((), dtype=torch.bool, device=rows.device)
    swaps = torch.zeros((), dtype=torch.long, device=rows.device)
    remove_limit = min(budget, max(1, int(pool_size)))
    outside_count = candidate_count - budget
    add_limit = min(outside_count, max(1, int(pool_size)))

    for _ in range(steps):
        selected_rows = torch.index_select(rows, 0, selected_columns)
        information = ridge * identity + selected_rows.transpose(0, 1) @ selected_rows
        # inv() synchronizes CUDA to check solver status, which is forbidden
        # during graph capture. inv_ex uses the same solver without that sync;
        # ridge regularization guarantees this information matrix is invertible.
        inverse, _ = torch.linalg.inv_ex(information, check_errors=False)
        selected_leverage = torch.sum(
            (selected_rows @ inverse) * selected_rows,
            dim=1,
        ).clamp(0.0, 1.0 - 1e-5)
        removal_loss = -torch.log1p(-selected_leverage)

        # candidates is sorted, so sorting selected column ids is exactly the
        # original token-id tie break used by the eager implementation.
        token_order = torch.argsort(selected_columns, stable=True)
        ordered_loss = torch.index_select(removal_loss, 0, token_order)
        removable_positions = torch.index_select(
            token_order,
            0,
            torch.argsort(ordered_loss, stable=True),
        )[:remove_limit]

        outside_mask = torch.ones(
            candidate_count,
            dtype=torch.bool,
            device=rows.device,
        )
        outside_mask = outside_mask.scatter(
            0,
            selected_columns,
            torch.zeros_like(selected_columns, dtype=torch.bool),
        )
        # Stable mask sorting yields the same ascending list as where(mask),
        # but its output shape is fixed and therefore CUDA-graph capturable.
        outside_tensor = torch.index_select(
            all_columns,
            0,
            torch.argsort(outside_mask.to(torch.int8), descending=True, stable=True),
        )[:outside_count]
        outside_rows = torch.index_select(rows, 0, outside_tensor)
        outside_leverage = torch.sum(
            (outside_rows @ inverse) * outside_rows,
            dim=1,
        )
        outside_order = torch.argsort(
            outside_leverage,
            descending=True,
            stable=True,
        )[:add_limit]
        outside_tensor = torch.index_select(outside_tensor, 0, outside_order)
        outside_rows = torch.index_select(rows, 0, outside_tensor)

        local_delta_tensors: list[torch.Tensor] = []
        local_add_tensors: list[torch.Tensor] = []
        for position in removable_positions.unbind():
            removed = torch.index_select(
                selected_rows,
                0,
                position.reshape(1),
            ).squeeze(0)
            direction = inverse @ removed
            remove_denominator = (
                1.0 - torch.dot(removed, direction)
            ).clamp_min(1e-5)
            inverse_without = (
                inverse
                + torch.outer(direction, direction) / remove_denominator
            )
            add_leverage = torch.sum(
                (outside_rows @ inverse_without) * outside_rows,
                dim=1,
            ).clamp_min(0.0)
            delta = torch.log(remove_denominator) + torch.log1p(add_leverage)
            local = torch.argmax(delta)
            local_delta_tensors.append(
                torch.gather(delta, 0, local.reshape(1)).squeeze(0)
            )
            local_add_tensors.append(
                torch.gather(outside_tensor, 0, local.reshape(1)).squeeze(0)
            )

        local_deltas = torch.stack(local_delta_tensors)
        local_add_columns = torch.stack(local_add_tensors)
        best_local = torch.argmax(local_deltas)
        best_delta = torch.gather(
            local_deltas,
            0,
            best_local.reshape(1),
        ).squeeze(0)
        improved = running & (best_delta > float(margin))
        best_remove = torch.gather(
            removable_positions,
            0,
            best_local.reshape(1),
        ).squeeze(0)
        best_add = torch.gather(
            local_add_columns,
            0,
            best_local.reshape(1),
        ).squeeze(0)
        old_column = torch.gather(
            selected_columns,
            0,
            best_remove.reshape(1),
        ).squeeze(0)
        replacement = torch.where(
            improved,
            best_add,
            old_column,
        )
        selected_columns = torch.where(
            selected_positions == best_remove,
            replacement,
            selected_columns,
        )
        swaps = swaps + improved.to(torch.long)
        running = improved

    return selected_columns, swaps

def _swap_refine(
    *,
    selected: torch.Tensor,
    candidates: torch.Tensor,
    design: torch.Tensor,
    mandatory: list[int],
    ridge: float,
    steps: int,
    pool_size: int,
    margin: float,
) -> tuple[torch.Tensor, int, float]:
    """Run the exact Fedorov program eagerly or as one static CUDA graph."""
    steps = max(0, int(steps))
    if (
        mandatory
        or steps == 0
        or selected.numel() == 0
        or not _exact_cuda_graph_enabled(design)
    ):
        return _swap_refine_eager(
            selected=selected,
            candidates=candidates,
            design=design,
            mandatory=mandatory,
            ridge=ridge,
            steps=steps,
            pool_size=pool_size,
            margin=margin,
        )

    rows = design[candidates].float()
    if int(rows.shape[0]) <= int(selected.numel()):
        return _swap_refine_eager(
            selected=selected,
            candidates=candidates,
            design=design,
            mandatory=mandatory,
            ridge=ridge,
            steps=steps,
            pool_size=pool_size,
            margin=margin,
        )
    ridge = max(1e-4, float(ridge))
    selected_columns = torch.searchsorted(candidates, selected).to(
        device=rows.device,
        dtype=torch.long,
    )
    key = (
        rows.device.index,
        tuple(rows.shape),
        rows.dtype,
        int(selected_columns.numel()),
        float(ridge),
        int(steps),
        int(pool_size),
        float(margin),
    )
    cached = _FEDOROV_GRAPH_CACHE.get(key)
    if cached is None:
        static_rows = rows.detach().clone()
        static_selected = selected_columns.detach().clone()
        warmup_stream = torch.cuda.Stream(device=rows.device)
        warmup_stream.wait_stream(torch.cuda.current_stream(rows.device))
        with torch.cuda.stream(warmup_stream):
            for _ in range(2):
                _swap_refine_no_certificate_capturable(
                    static_rows,
                    static_selected,
                    ridge,
                    steps,
                    pool_size,
                    margin,
                )
        torch.cuda.current_stream(rows.device).wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = _swap_refine_no_certificate_capturable(
                static_rows,
                static_selected,
                ridge,
                steps,
                pool_size,
                margin,
            )
        cached = {
            "graph": graph,
            "rows": static_rows,
            "selected": static_selected,
            "output": static_output,
        }
        _FEDOROV_GRAPH_CACHE[key] = cached
        print(
            "[CertVID V3] captured exact Fedorov CUDA graph "
            f"for rows={tuple(rows.shape)} budget={selected_columns.numel()}"
        )

    static_rows = cached["rows"]
    static_selected = cached["selected"]
    assert isinstance(static_rows, torch.Tensor)
    assert isinstance(static_selected, torch.Tensor)
    static_rows.copy_(rows)
    static_selected.copy_(selected_columns)
    graph = cached["graph"]
    assert isinstance(graph, torch.cuda.CUDAGraph)
    graph.replay()
    output = cached["output"]
    assert isinstance(output, tuple)
    graph_columns, graph_swaps_tensor = output
    graph_selected = candidates[graph_columns]
    graph_swaps = int(graph_swaps_tensor.item())
    final_rows = torch.index_select(rows, 0, graph_columns)
    identity = torch.eye(
        int(rows.shape[1]),
        dtype=torch.float32,
        device=rows.device,
    )
    final_information = ridge * identity + final_rows.transpose(0, 1) @ final_rows
    graph_sign, graph_logdet_tensor = torch.linalg.slogdet(final_information)
    graph_logdet = (
        float(graph_logdet_tensor.item())
        if float(graph_sign.item()) > 0.0
        else float("-inf")
    )

    if _exact_optimization_audit_enabled():
        reference_selected, reference_swaps, reference_logdet = _swap_refine_eager(
            selected=selected,
            candidates=candidates,
            design=design,
            mandatory=mandatory,
            ridge=ridge,
            steps=steps,
            pool_size=pool_size,
            margin=margin,
        )
        if (
            not torch.equal(graph_selected, reference_selected)
            or graph_swaps != reference_swaps
            or graph_logdet != reference_logdet
        ):
            raise RuntimeError("CUDA-graph Fedorov refinement differs from eager reference")
    return graph_selected, graph_swaps, graph_logdet

def _certvidfinal_v3_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
    question_features: Optional[torch.Tensor] = None,
    *,
    analysis_sink: Optional[MutableMapping[str, Any]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the paper's no-certificate, no-trajectory D-optimal coreset."""
    phase_events: Optional[
        dict[str, tuple[torch.cuda.Event, torch.cuda.Event]]
    ] = {} if _profile_enabled(video_features) else None
    setattr(flashvid_config, "_certv3_profile_events", phase_events)
    if analysis_sink is not None:
        analysis_sink.clear()
    if video_features.ndim != 3:
        raise ValueError(f"expected video_features [T, HW, D], got {tuple(video_features.shape)}")
    frame_count, tokens_per_frame, _ = video_features.shape
    total_tokens = int(frame_count * tokens_per_frame)
    ratio = _effective_ratio(flashvid_config)
    strict_budget = bool(
        str(getattr(flashvid_config, "compression_variant", "")).strip().lower()
        == "certvid_v3"
        and getattr(flashvid_config, "strict_token_budget", False)
    )
    budget_value = (
        math.floor(total_tokens * ratio + 1e-9)
        if strict_budget
        else round(total_tokens * ratio)
    )
    budget = max(1, min(total_tokens, int(budget_value)))
    flat_features = video_features.reshape(total_tokens, -1)
    backbone = str(
        getattr(flashvid_config, "_baseline_backbone", "")
    ).strip().lower()
    selection_objective = str(
        getattr(flashvid_config, "certv3_selection_objective", "d_optimal")
    ).strip().lower()
    # Keep old visualization commands readable while naming the paper
    # ablation by what it removes rather than by the resulting ranking rule.
    if selection_objective == "quality_topk":
        selection_objective = "score_only"
    if selection_objective not in {"d_optimal", "score_only"}:
        raise ValueError(
            "certv3_selection_objective must be 'd_optimal' or 'score_only', "
            f"got {selection_objective!r}"
        )
    use_query = _cfg_bool(flashvid_config, "certv3_use_query", True)
    use_candidate_pool = _cfg_bool(
        flashvid_config,
        "certv3_use_candidate_pool",
        True,
    )

    if budget >= total_tokens:
        plan = _identity_plan(total_tokens, video_features.device)
        output = flat_features
        candidates = total_tokens
        components = total_tokens
        query_confidence = 0.0
        swaps = 0
        logdet = 0.0
        if analysis_sink is not None:
            analysis_sink["identity"] = True
    else:
        _profile_record(phase_events, "metric_layout", 0)
        metric_dim = max(32, _final_v1_cfg_int(flashvid_config, "certv3_metric_dim", 96))
        metric_flat = _final_v1_metric_features(video_features, metric_dim)
        height, width = _final_v1_grid_hw(tokens_per_frame, flashvid_config)
        spatial_bins = max(1, _final_v1_cfg_int(flashvid_config, "certv3_spatial_bins", 3))
        _, frame_spatial_ids = _final_v1_spatial_layout(
            tokens_per_frame,
            height,
            width,
            spatial_bins,
            video_features.device,
        )
        _profile_record(phase_events, "metric_layout", 1)

        # Final intentionally excludes all cross-frame trajectory/event signals.
        frame_event = torch.zeros(
            frame_count,
            dtype=torch.float32,
            device=video_features.device,
        )
        novelty_2d = torch.zeros(
            (frame_count, tokens_per_frame),
            dtype=torch.float32,
            device=video_features.device,
        )
        curvature_2d = torch.zeros_like(novelty_2d)
        component_ids = torch.arange(
            total_tokens,
            dtype=torch.long,
            device=video_features.device,
        )
        frame_ids = torch.arange(
            frame_count,
            device=video_features.device,
        ).repeat_interleave(tokens_per_frame)
        component_value = torch.zeros(
            total_tokens,
            dtype=torch.float32,
            device=video_features.device,
        )

        _profile_record(phase_events, "quality_signals", 0)
        temporal_count = min(frame_count, max(1, _final_v1_cfg_int(flashvid_config, "certv3_temporal_bins", 12)))
        temporal_ids = torch.div(
            frame_ids * temporal_count,
            max(1, frame_count),
            rounding_mode="floor",
        ).clamp_max(temporal_count - 1)
        spatial_ids = frame_spatial_ids.repeat(frame_count)
        spatial_count = spatial_bins * spatial_bins

        attention = _final_v1_rank_normalize(cls_attention.float()).reshape(-1)
        novelty = novelty_2d.reshape(-1)
        curvature = curvature_2d.reshape(-1)
        detail = _final_v1_local_detail(video_features, height, width).reshape(-1)
        event = frame_event.repeat_interleave(tokens_per_frame)
        atoms = _final_v1_question_atoms(
            question_features if use_query else None,
            max(0, _final_v1_cfg_int(flashvid_config, "certv3_query_atoms", 8)),
            metric_dim,
        ).to(video_features.device)
        query_relevance, atom_weights, query_confidence = _final_v1_question_relevance(atoms, metric_flat)
        query_score = (
            (query_relevance * atom_weights.unsqueeze(1)).sum(dim=0)
            if query_relevance.numel() > 0
            else torch.zeros(total_tokens, dtype=torch.float32, device=video_features.device)
        )

        query_weight = min(
            0.30,
            max(0.0, _final_v1_cfg_float(flashvid_config, "certv3_query_weight", 0.18) * query_confidence),
        )
        visual_weights = [
            _final_v1_cfg_float(flashvid_config, "certv3_visual_attention_weight", 0.28),
            _final_v1_cfg_float(flashvid_config, "certv3_visual_novelty_weight", 0.20),
            _final_v1_cfg_float(flashvid_config, "certv3_visual_curvature_weight", 0.14),
            _final_v1_cfg_float(flashvid_config, "certv3_visual_event_weight", 0.12),
            _final_v1_cfg_float(flashvid_config, "certv3_visual_detail_weight", 0.12),
            _final_v1_cfg_float(flashvid_config, "certv3_visual_component_weight", 0.14),
        ]
        visual_weight_sum = sum(max(0.0, weight) for weight in visual_weights)
        if visual_weight_sum <= 0.0:
            raise ValueError("CertVID V3 visual weights must have a positive sum")
        visual_weights = [max(0.0, weight) / visual_weight_sum for weight in visual_weights]
        visual_quality = _final_v1_minmax(
            visual_weights[0] * attention
            + visual_weights[1] * novelty
            + visual_weights[2] * curvature
            + visual_weights[3] * event
            + visual_weights[4] * detail
            + visual_weights[5] * component_value,
            dim=0,
        )
        quality = _final_v1_minmax((1.0 - query_weight) * visual_quality + query_weight * query_score, dim=0)
        event_weights = [
            _final_v1_cfg_float(flashvid_config, "certv3_event_novelty_weight", 0.34),
            _final_v1_cfg_float(flashvid_config, "certv3_event_curvature_weight", 0.28),
            _final_v1_cfg_float(flashvid_config, "certv3_event_frame_weight", 0.18),
            _final_v1_cfg_float(flashvid_config, "certv3_event_detail_weight", 0.10),
            _final_v1_cfg_float(flashvid_config, "certv3_event_query_weight", 0.10),
        ]
        event_weight_sum = sum(max(0.0, weight) for weight in event_weights)
        if event_weight_sum <= 0.0:
            raise ValueError("CertVID V3 event weights must have a positive sum")
        event_weights = [max(0.0, weight) / event_weight_sum for weight in event_weights]
        event_score = _final_v1_minmax(
            event_weights[0] * novelty
            + event_weights[1] * curvature
            + event_weights[2] * event
            + event_weights[3] * detail
            + event_weights[4] * query_score,
            dim=0,
        )
        demand_weight = 0.20 + 0.42 * quality + 0.20 * event_score + 0.18 * component_value
        demand_weight = demand_weight / demand_weight.sum().clamp_min(1e-6)
        _profile_record(phase_events, "quality_signals", 1)

        _profile_record(phase_events, "certificates_candidates", 0)
        mandatory: list[int] = []
        candidate_multiplier = _final_v1_cfg_float(
            flashvid_config,
            "certv3_candidate_multiplier",
            2.5,
        )
        candidate_limit = min(
            total_tokens,
            max(
                budget,
                int(math.ceil(budget * max(1.0, candidate_multiplier))),
            ),
        )
        if use_candidate_pool:
            candidate_indices = _candidate_pool_without_certificates(
                quality=quality,
                temporal_ids=temporal_ids,
                spatial_ids=spatial_ids,
                query_relevance=query_relevance,
                limit=candidate_limit,
            )
        else:
            candidate_indices = torch.arange(
                total_tokens,
                dtype=torch.long,
                device=video_features.device,
            )
        _profile_record(phase_events, "certificates_candidates", 1)

        _profile_record(phase_events, "design_whitening", 0)
        design = _design_features(
            metric_features=metric_flat,
            quality=quality,
            temporal_ids=temporal_ids,
            spatial_ids=spatial_ids,
            attention=attention,
            novelty=novelty,
            curvature=curvature,
            event=event,
            detail=detail,
            component_support=component_value,
            query_relevance=query_relevance,
            atom_weights=atom_weights,
            query_confidence=query_confidence,
            temporal_count=temporal_count,
            spatial_count=spatial_count,
            structural_weight=_final_v1_cfg_float(flashvid_config, "certv3_structural_weight", 0.32),
            use_spatiotemporal_design=_cfg_bool(
                flashvid_config,
                "certv3_use_spatiotemporal_design",
                True,
            ),
            whitening_strength=_final_v1_cfg_float(flashvid_config, "certv3_whitening_strength", 0.50),
            quality_floor=_final_v1_cfg_float(flashvid_config, "certv3_quality_floor", 0.15),
        )
        design_mass = getattr(
            flashvid_config,
            "_certvid_design_mass_multiplier",
            None,
        )
        if design_mass is not None:
            design_mass = torch.as_tensor(
                design_mass,
                dtype=torch.float32,
                device=design.device,
            ).flatten()
            if design_mass.numel() != total_tokens:
                raise ValueError(
                    "CertVID design mass must contain one value per visual token"
                )
            if not bool(torch.isfinite(design_mass).all()) or bool(
                (design_mass <= 0).any()
            ):
                raise ValueError("CertVID design mass must be finite and positive")
            design = design * design_mass.sqrt().unsqueeze(1)
        _profile_record(phase_events, "design_whitening", 1)

        ridge = _final_v1_cfg_float(flashvid_config, "certv3_ridge", 0.50)
        if selection_objective == "score_only":
            _profile_record(phase_events, "d_optimal", 0)
            selected = _score_only_select(
                quality=quality,
                candidates=candidate_indices,
                mandatory=mandatory,
                budget=budget,
            )
            swaps = 0
            logdet = _selection_logdet(design, selected, ridge)
            _profile_record(phase_events, "d_optimal", 1)
        else:
            _profile_record(phase_events, "d_optimal", 0)
            selected = _d_optimal_greedy(
                design=design,
                candidates=candidate_indices,
                mandatory=mandatory,
                budget=budget,
                ridge=ridge,
            )
            _profile_record(phase_events, "d_optimal", 1)

            _profile_record(phase_events, "fedorov", 0)
            selected, swaps, logdet = _swap_refine(
                selected=selected,
                candidates=candidate_indices,
                design=design,
                mandatory=mandatory,
                ridge=ridge,
                steps=_final_v1_cfg_int(flashvid_config, "certv3_swap_steps", 6),
                pool_size=_final_v1_cfg_int(flashvid_config, "certv3_swap_pool", 24),
                margin=_final_v1_cfg_float(flashvid_config, "certv3_swap_margin", 1e-4),
            )
            _profile_record(phase_events, "fedorov", 1)

        _profile_record(phase_events, "fusion_plan", 0)
        selected = torch.sort(selected).values
        plan = _final_v1_build_plan(
            selected=selected,
            metric_features=metric_flat,
            demand_weight=demand_weight,
            attention=attention,
            query_score=query_score,
            temporal_ids=temporal_ids,
            component_ids=component_ids,
            fusion_alpha=_final_v1_cfg_float(flashvid_config, "certv3_fusion_alpha", 0.12),
            temperature=_final_v1_cfg_float(flashvid_config, "certv3_assignment_temperature", 0.07),
        )
        _profile_record(phase_events, "fusion_plan", 1)

        _profile_record(phase_events, "fusion_apply", 0)
        output = _final_v1apply_certvid_plan(flat_features, plan)
        _profile_record(phase_events, "fusion_apply", 1)
        candidates = int(candidate_indices.numel())
        components = total_tokens
        if analysis_sink is not None:
            # CertVID-HR consumes these tensors immediately and never stores
            # them on the persistent model config. Existing V3 callers keep
            # the exact same path because analysis_sink defaults to None.
            analysis_sink.update(
                {
                    "metric_flat": metric_flat,
                    "design": design,
                    "quality": quality,
                    "demand_weight": demand_weight,
                    "attention": attention,
                    "query_score": query_score,
                    "query_relevance": query_relevance,
                    "query_confidence": float(query_confidence),
                    "component_ids": component_ids,
                    "frame_ids": frame_ids,
                    "temporal_ids": temporal_ids,
                    "candidate_indices": candidate_indices,
                    "ridge": float(ridge),
                }
            )

    setattr(flashvid_config, "_certvid_plan", plan)
    flashvid_config.vision_token_length = int(output.shape[0])
    flashvid_config.visual_token_length = int(output.shape[0])
    flashvid_config.llm_token_length = None
    setattr(flashvid_config, "last_adapter_variant", "certvid_v3")
    setattr(flashvid_config, "last_adapter_raw_tokens", float(total_tokens))
    setattr(flashvid_config, "last_adapter_output_tokens", float(output.shape[0]))
    setattr(flashvid_config, "last_certv3_target_tokens", float(budget))
    setattr(flashvid_config, "last_certv3_candidate_tokens", float(candidates))
    setattr(flashvid_config, "last_certv3_component_count", float(components))
    # Preserve V3's public telemetry without retaining dead certificate logic.
    qwen_certificate_policy = backbone in {"qwen2_5_vl", "qwen3_vl"}
    certificate_telemetry = {
        "last_certv3_certificate_count": 0.0,
        "last_certv3_original_certificate_count": 0.0,
        "last_certv3_certificate_budget_ratio": 0.0,
        "last_certv3_certificate_limit": 0.0,
        "last_certv3_certificate_cap_active": True,
        "last_certv3_qwen_certificate_cap_active": qwen_certificate_policy,
        "last_certv3_qwen_certificate_limit": (
            0.0 if qwen_certificate_policy else float(budget)
        ),
        "last_certv3_certificate_pressure": 0.0,
        "last_certv3_query_seed_count": 0.0,
    }
    for name, value in certificate_telemetry.items():
        setattr(flashvid_config, name, value)
    # With no immutable anchors, every selected slot is available to D-optimal.
    setattr(flashvid_config, "last_certv3_free_dopt_slots", float(budget))
    setattr(flashvid_config, "last_certv3_dopt_slot_ratio", 1.0)
    setattr(flashvid_config, "last_certv3_query_confidence", float(query_confidence))
    setattr(flashvid_config, "last_certv3_swap_count", float(swaps))
    setattr(flashvid_config, "last_certv3_logdet", float(logdet))
    diagnostics = {
        "identity": bool(budget >= total_tokens),
        "retention_ratio": float(
            _final_v1_cfg_float(flashvid_config, "retention_ratio", 0.10)
        ),
        "effective_outer_ratio": float(ratio),
        "expansion": float(_final_v1_cfg_float(flashvid_config, "expansion", 1.0)),
        "pruning_layer": int(_final_v1_cfg_int(flashvid_config, "pruning_layer", 0)),
        "llm_retention_ratio": float(
            _final_v1_cfg_float(flashvid_config, "llm_retention_ratio", 1.0)
        ),
        "fusion_alpha": float(
            _final_v1_cfg_float(flashvid_config, "certv3_fusion_alpha", 0.12)
        ),
        "raw_tokens": int(total_tokens),
        "target_tokens": int(budget),
        "output_tokens": int(output.shape[0]),
        "backbone": backbone,
        "certificate_count": 0,
        "original_certificate_count": 0,
        "certificate_pressure": 0.0,
        "original_certificate_pressure": 0.0,
        "free_dopt_slots": int(budget),
        "dopt_slot_ratio": 1.0,
        "certificate_budget_ratio": 0.0,
        "configured_certificate_budget_ratio": 0.0,
        "certificate_limit": 0,
        "certificate_cap_active": True,
        "qwen_certificate_policy": qwen_certificate_policy,
        "qwen_certificate_cap_active": qwen_certificate_policy,
        "qwen_certificate_budget_ratio": 0.0,
        "qwen_certificate_limit": (
            0 if qwen_certificate_policy else int(budget)
        ),
        "candidate_count": int(candidates),
        "component_count": int(components),
        "query_seed_count": 0,
        "query_confidence": float(query_confidence),
        "swap_count": int(swaps),
        "logdet": float(logdet),
        "selection_objective": selection_objective,
        "use_trajectory": False,
        "use_query": bool(use_query),
        "use_candidate_pool": bool(use_candidate_pool),
    }
    setattr(flashvid_config, "last_certv3_diagnostics", diagnostics)
    _write_certv3_diagnostics(flashvid_config, diagnostics)
    return output, plan.anchor_indices

_MISSING_FINAL_CONFIG_VALUE = object()


def _restore_final_config_value(
    config: FlashVidConfig,
    name: str,
    previous: object,
) -> None:
    if previous is _MISSING_FINAL_CONFIG_VALUE:
        try:
            delattr(config, name)
        except AttributeError:
            pass
    else:
        setattr(config, name, previous)


def certvidfinal_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
    question_features: Optional[torch.Tensor] = None,
    *,
    analysis_sink: Optional[MutableMapping[str, Any]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the fixed no-certificate, no-trajectory final selection path."""
    variant = getattr(
        flashvid_config,
        "compression_variant",
        _MISSING_FINAL_CONFIG_VALUE,
    )
    try:
        # V3 keys strict token-budget rounding on this variant name.
        setattr(flashvid_config, "compression_variant", "certvid_v3")
        return _certvidfinal_v3_compression(
            video_features=video_features,
            cls_attention=cls_attention,
            flashvid_config=flashvid_config,
            question_features=question_features,
            analysis_sink=analysis_sink,
        )
    finally:
        _restore_final_config_value(
            flashvid_config,
            "compression_variant",
            variant,
        )
