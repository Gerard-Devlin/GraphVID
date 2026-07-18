from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .certvid import (
    CertVidPlan,
    _build_components,
    _cfg_float,
    _cfg_int,
    _grid_hw,
    _local_detail,
    _metric_features,
    _minmax,
    _question_atoms,
    _question_relevance,
    _spatial_layout,
    apply_certvid_plan,
)
from .certvid_v2 import _component_support, _trajectory_signals
from .configuration_flashvid import FlashVidConfig


_CERTIFICATE_CATEGORIES = ("query", "frame", "temporal", "spatial")
_CERTIFICATE_SHARES = {"query": 0.20, "frame": 0.35, "temporal": 0.20, "spatial": 0.25}
_CANDIDATE_SOURCES = ("query", "trajectory", "spatial", "global")
_CANDIDATE_SHARES = {"query": 0.20, "trajectory": 0.30, "spatial": 0.25, "global": 0.25}
_QUERY_MODES = {"certificates_only", "design_only", "certificates_and_design", "off"}


@dataclass(frozen=True)
class _CertificateRequest:
    category: str
    request_id: str
    token: int
    score: float


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


def _tie_safe_rank_normalize(values: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, bool, str]:
    """Return mid-ranks so tied or degenerate inputs cannot encode token position."""
    tensor = values.float()
    flat = tensor.reshape(-1)
    if flat.numel() == 0:
        return tensor, False, "empty"
    if not bool(torch.isfinite(flat).all()):
        raise ValueError("cls_attention contains NaN or Inf")
    if tensor.ndim == 0 or tensor.shape[-1] <= 1:
        return torch.zeros_like(tensor), False, "single_value"

    eps = max(0.0, float(eps))
    rows = tensor.reshape(-1, tensor.shape[-1])
    normalized = torch.zeros_like(rows)
    valid_rows = 0
    for row_idx, row in enumerate(rows):
        spread = float((row.max() - row.min()).item())
        standard_deviation = float(row.std(unbiased=False).item())
        if spread < eps or standard_deviation < eps:
            continue
        _, inverse, counts = torch.unique(row, sorted=True, return_inverse=True, return_counts=True)
        counts_float = counts.float()
        starts = torch.cumsum(counts_float, dim=0) - counts_float
        mid_ranks = starts + 0.5 * (counts_float - 1.0)
        normalized[row_idx] = mid_ranks[inverse] / float(row.numel() - 1)
        valid_rows += 1

    if valid_rows == 0:
        return normalized.reshape_as(tensor), False, "degenerate"
    reason = "validated" if valid_rows == rows.shape[0] else "validated_partial"
    return normalized.reshape_as(tensor), True, reason


def _validated_attention(
    cls_attention: torch.Tensor,
    frame_count: int,
    tokens_per_frame: int,
    config: FlashVidConfig,
) -> tuple[torch.Tensor, dict[str, object]]:
    expected = (int(frame_count), int(tokens_per_frame))
    if tuple(cls_attention.shape) != expected:
        raise ValueError(f"cls_attention must have shape {expected}, got {tuple(cls_attention.shape)}")
    raw = cls_attention.float()
    if not bool(torch.isfinite(raw).all()):
        raise ValueError("cls_attention contains NaN or Inf")

    policy = str(getattr(config, "certv4_attention_policy", "validated")).strip().lower()
    if policy not in {"validated", "strict", "off"}:
        raise ValueError(f"unsupported certv4_attention_policy={policy!r}")
    source = str(getattr(config, "_certvid_attention_source", "missing")).strip().lower()
    diagnostic: dict[str, object] = {
        "policy": policy,
        "source": source,
        "used": False,
        "reason": "policy_off" if policy == "off" else "unvalidated_source",
    }
    if policy == "strict" and source != "manual_qk":
        raise ValueError(
            "certv4_attention_policy='strict' requires attention provenance 'manual_qk', "
            f"got {source!r}"
        )
    if policy == "off" or source != "manual_qk":
        return torch.zeros(frame_count * tokens_per_frame, dtype=torch.float32, device=raw.device), diagnostic

    normalized, used, reason = _tie_safe_rank_normalize(
        raw,
        _cfg_float(config, "certv4_attention_eps", 1e-6),
    )
    diagnostic["used"] = bool(used)
    diagnostic["reason"] = reason
    diagnostic["raw_std"] = float(raw.std(unbiased=False).item())
    diagnostic["raw_range"] = float((raw.max() - raw.min()).item())
    return normalized.reshape(-1), diagnostic


def _resolve_budget(
    config: FlashVidConfig,
    total_tokens: int,
) -> tuple[int, dict[str, object]]:
    mode = str(getattr(config, "certv4_budget_mode", "layer_average")).strip().lower()
    if mode not in {"layer_average", "outer_only"}:
        raise ValueError(f"unsupported certv4_budget_mode={mode!r}")

    nominal = _cfg_float(config, "retention_ratio", 0.10)
    expansion = _cfg_float(config, "expansion", 1.0)
    pruning_layer = _cfg_int(config, "pruning_layer", 0)
    inner_retention = _cfg_float(config, "llm_retention_ratio", 1.0)
    layers = _cfg_int(config, "certv4_num_hidden_layers", 0)
    tolerance = 1e-4
    if not (0.0 < nominal <= 1.0):
        raise ValueError(f"retention_ratio must be in (0, 1], got {nominal}")

    if mode == "outer_only":
        if abs(expansion - 1.0) > tolerance or abs(inner_retention - 1.0) > tolerance:
            raise ValueError(
                "certv4 outer_only requires expansion=1 and llm_retention_ratio=1; "
                f"got expansion={expansion}, llm_retention_ratio={inner_retention}"
            )
        outer_retention = nominal
        post_inner_retention = nominal
        layer_multiplier = 1.0
        average_retention = nominal
    else:
        if layers <= 1:
            raise ValueError("certv4 layer_average requires certv4_num_hidden_layers from the model config")
        if not (0 < pruning_layer < layers):
            raise ValueError(
                f"pruning_layer must satisfy 0 < K < L, got K={pruning_layer}, L={layers}"
            )
        if not (0.0 < inner_retention < 1.0):
            raise ValueError(
                "certv4 layer_average requires 0 < llm_retention_ratio < 1, "
                f"got {inner_retention}"
            )
        if not bool(getattr(config, "certv4_inner_hook_enabled", False)):
            raise ValueError("certv4 layer_average requires an installed inner-pruning hook")
        if nominal * expansion > 1.0 + tolerance:
            raise ValueError(
                f"outer retention R*E must not exceed 1, got {nominal * expansion:.8f}"
            )
        layer_multiplier = expansion * (
            pruning_layer + (layers - pruning_layer) * inner_retention
        ) / float(layers)
        if abs(layer_multiplier - 1.0) > tolerance:
            raise ValueError(
                "certv4 layer_average budget is not aligned: "
                f"E*(K+(L-K)*r)/L={layer_multiplier:.8f}, expected 1 within {tolerance}"
            )
        outer_retention = nominal * expansion
        post_inner_retention = outer_retention * inner_retention
        average_retention = nominal * layer_multiplier

    budget = max(1, int(round(total_tokens * outer_retention)))
    if budget > total_tokens:
        raise ValueError(f"certv4 outer budget {budget} exceeds raw token count {total_tokens}")
    post_inner_tokens = (
        max(1, int(round(budget * inner_retention)))
        if mode == "layer_average"
        else budget
    )
    average_layer_tokens = (
        (pruning_layer * budget + (layers - pruning_layer) * post_inner_tokens) / float(layers)
        if mode == "layer_average"
        else float(budget)
    )
    diagnostics: dict[str, object] = {
        "mode": mode,
        "nominal_retention": nominal,
        "outer_retention": outer_retention,
        "post_inner_retention": post_inner_retention,
        "average_retention": average_retention,
        "average_layer_multiplier": layer_multiplier,
        "expansion": expansion,
        "pruning_layer": pruning_layer,
        "inner_retention": inner_retention,
        "num_hidden_layers": layers,
        "raw_tokens": total_tokens,
        "target_tokens": budget,
        "post_inner_tokens": post_inner_tokens,
        "average_layer_tokens": average_layer_tokens,
    }
    return budget, diagnostics


def _whiten_features(features: torch.Tensor, strength: float) -> torch.Tensor:
    strength = min(1.0, max(0.0, float(strength)))
    centered = features.float() - features.float().mean(dim=0, keepdim=True)
    if strength <= 1e-6 or centered.shape[0] <= 1:
        return F.normalize(centered, p=2, dim=-1, eps=1e-6)
    covariance = centered.transpose(0, 1) @ centered / float(max(1, centered.shape[0] - 1))
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    mean_eigenvalue = eigenvalues.mean().clamp_min(1e-6)
    eigenvalues = eigenvalues.clamp_min(mean_eigenvalue * 1e-4)
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
    novelty: torch.Tensor,
    curvature: torch.Tensor,
    event: torch.Tensor,
    detail: torch.Tensor,
    component_support: torch.Tensor,
    query_relevance: torch.Tensor,
    atom_weights: torch.Tensor,
    query_confidence: float,
    query_mode: str,
    temporal_count: int,
    spatial_count: int,
    structural_weight: float,
    whitening_strength: float,
    quality_floor: float,
) -> torch.Tensor:
    visual = _whiten_features(metric_features, whitening_strength)
    temporal = F.one_hot(temporal_ids, num_classes=temporal_count).float()
    spatial = F.one_hot(spatial_ids, num_classes=spatial_count).float()
    signals = F.normalize(
        torch.stack([novelty, curvature, event, detail, component_support], dim=1),
        p=2,
        dim=-1,
        eps=1e-6,
    )

    structural_weight = min(0.80, max(0.0, float(structural_weight)))
    query_enabled = query_mode in {"design_only", "certificates_and_design"}
    query_share = (
        0.20 * structural_weight * min(1.0, max(0.0, float(query_confidence)))
        if query_enabled and query_relevance.numel() > 0
        else 0.0
    )
    structural_remainder = max(0.0, structural_weight - query_share)
    parts = [
        visual * math.sqrt(max(0.20, 1.0 - structural_weight)),
        temporal * math.sqrt(0.45 * structural_remainder),
        spatial * math.sqrt(0.25 * structural_remainder),
        signals * math.sqrt(0.30 * structural_remainder),
    ]
    if query_share > 0.0:
        query_axes = query_relevance.transpose(0, 1) * torch.sqrt(
            atom_weights.clamp_min(1e-6)
        ).unsqueeze(0)
        parts.append(query_axes * math.sqrt(query_share))

    design = F.normalize(torch.cat(parts, dim=1), p=2, dim=-1, eps=1e-6)
    quality_floor = min(1.0, max(1e-4, float(quality_floor)))
    row_mass = quality_floor + (1.0 - quality_floor) * quality.clamp(0.0, 1.0)
    return design * torch.sqrt(row_mass).unsqueeze(1)


def _evenly_spaced_ids(count: int, keep: int, device: torch.device) -> torch.Tensor:
    if keep >= count:
        return torch.arange(count, dtype=torch.long, device=device)
    if keep <= 1:
        return torch.tensor([count // 2], dtype=torch.long, device=device)
    return torch.round(torch.linspace(0, count - 1, steps=keep, device=device)).long().unique()


def _largest_remainder(total: int, shares: dict[str, float], names: list[str]) -> dict[str, int]:
    if total <= 0 or not names:
        return {name: 0 for name in names}
    weight_sum = sum(max(0.0, shares[name]) for name in names)
    if weight_sum <= 0.0:
        weight_sum = float(len(names))
        normalized = {name: 1.0 / len(names) for name in names}
    else:
        normalized = {name: max(0.0, shares[name]) / weight_sum for name in names}
    raw = {name: total * normalized[name] for name in names}
    output = {name: int(math.floor(raw[name])) for name in names}
    remaining = total - sum(output.values())
    order = sorted(names, key=lambda name: (-(raw[name] - output[name]), names.index(name)))
    for name in order[:remaining]:
        output[name] += 1
    return output


def _build_certificate_requests(
    *,
    quality: torch.Tensor,
    event_score: torch.Tensor,
    frame_ids: torch.Tensor,
    temporal_ids: torch.Tensor,
    spatial_ids: torch.Tensor,
    query_relevance: torch.Tensor,
    atom_weights: torch.Tensor,
    query_confidence: float,
    query_mode: str,
    frame_count: int,
    temporal_count: int,
    spatial_count: int,
    frame_coverage_ratio: float,
    cell_coverage_ratio: float,
    query_threshold: float,
    query_per_atom: int,
) -> list[_CertificateRequest]:
    requests: list[_CertificateRequest] = []
    frame_keep = max(1, int(math.ceil(frame_count * min(1.0, max(0.0, frame_coverage_ratio)))))
    for frame_id in _evenly_spaced_ids(frame_count, frame_keep, quality.device).tolist():
        members = torch.where(frame_ids == int(frame_id))[0]
        if members.numel() > 0:
            token = int(members[torch.argmax(quality[members])].item())
            requests.append(_CertificateRequest("frame", f"frame:{frame_id}", token, float(quality[token])))

    for temporal_id in range(temporal_count):
        members = torch.where(temporal_ids == temporal_id)[0]
        if members.numel() > 0:
            token = int(members[torch.argmax(event_score[members])].item())
            requests.append(
                _CertificateRequest("temporal", f"temporal:{temporal_id}", token, float(event_score[token]))
            )

    cell_keep = min(
        spatial_count,
        max(0, int(math.ceil(spatial_count * min(1.0, max(0.0, cell_coverage_ratio))))),
    )
    for temporal_id in range(temporal_count):
        representatives: list[tuple[float, int, int]] = []
        for spatial_id in range(spatial_count):
            members = torch.where((temporal_ids == temporal_id) & (spatial_ids == spatial_id))[0]
            if members.numel() == 0:
                continue
            token = int(members[torch.argmax(quality[members])].item())
            representatives.append((float(quality[token]), token, spatial_id))
        representatives.sort(key=lambda item: (-item[0], item[1]))
        for score, token, spatial_id in representatives[:cell_keep]:
            requests.append(
                _CertificateRequest("spatial", f"cell:{temporal_id}:{spatial_id}", token, score)
            )

    if (
        query_mode in {"certificates_only", "certificates_and_design"}
        and query_relevance.numel() > 0
        and query_confidence >= float(query_threshold)
    ):
        per_atom = max(1, int(query_per_atom))
        for atom_idx, atom_scores in enumerate(query_relevance):
            alternatives: list[tuple[float, int, int]] = []
            for temporal_id in range(temporal_count):
                members = torch.where(temporal_ids == temporal_id)[0]
                if members.numel() == 0:
                    continue
                token = int(members[torch.argmax(atom_scores[members])].item())
                score = float(atom_scores[token] * atom_weights[atom_idx].clamp_min(1e-6))
                alternatives.append((score, token, temporal_id))
            alternatives.sort(key=lambda item: (-item[0], item[1]))
            for rank, (score, token, temporal_id) in enumerate(alternatives[:per_atom]):
                requests.append(
                    _CertificateRequest(
                        "query",
                        f"query:{atom_idx}:{rank}:{temporal_id}",
                        token,
                        score,
                    )
                )
    return requests


def _admit_certificates(
    requests: list[_CertificateRequest],
    budget: int,
    budget_ratio: float,
) -> tuple[list[int], dict[str, object]]:
    cap = min(budget, max(0, int(math.floor(budget * min(1.0, max(0.0, budget_ratio))))))
    grouped = {name: [] for name in _CERTIFICATE_CATEGORIES}
    for request in requests:
        grouped[request.category].append(request)
    for category in _CERTIFICATE_CATEGORIES:
        grouped[category].sort(key=lambda request: (-request.score, request.token, request.request_id))

    active = [category for category in _CERTIFICATE_CATEGORIES if grouped[category]]
    quotas = _largest_remainder(cap, _CERTIFICATE_SHARES, active)
    selected: set[int] = set()
    admitted_by_source = {category: 0 for category in _CERTIFICATE_CATEGORIES}
    pointers = {category: 0 for category in _CERTIFICATE_CATEGORIES}

    def offer(category: str) -> bool:
        entries = grouped[category]
        while pointers[category] < len(entries):
            request = entries[pointers[category]]
            pointers[category] += 1
            if request.token in selected:
                continue
            if len(selected) >= cap:
                return False
            selected.add(request.token)
            admitted_by_source[category] += 1
            return True
        return False

    for category in active:
        while admitted_by_source[category] < quotas[category]:
            if not offer(category):
                break

    cycle: list[str] = []
    for category in _CERTIFICATE_CATEGORIES:
        cycle.extend([category] * max(1, int(round(_CERTIFICATE_SHARES[category] * 20))))
    while len(selected) < cap:
        progress = False
        for category in cycle:
            if len(selected) >= cap:
                break
            progress = offer(category) or progress
        if not progress:
            break

    selected_sorted = sorted(selected)
    selected_set = set(selected_sorted)
    category_stats: dict[str, dict[str, int]] = {}
    for category in _CERTIFICATE_CATEGORIES:
        entries = grouped[category]
        unique_requested = len({entry.token for entry in entries})
        admitted_requests = sum(entry.token in selected_set for entry in entries)
        admitted_unique = len({entry.token for entry in entries if entry.token in selected_set})
        category_stats[category] = {
            "requested": len(entries),
            "deduplicated": unique_requested,
            "admitted": admitted_requests,
            "admitted_unique": admitted_unique,
            "truncated_or_unmet": len(entries) - admitted_requests,
            "quota": quotas.get(category, 0),
            "contributed_unique": admitted_by_source[category],
        }
    diagnostics: dict[str, object] = {
        "cap": cap,
        "admitted_unique": len(selected_sorted),
        "ratio": len(selected_sorted) / float(max(1, budget)),
        "requested": len(requests),
        "deduplicated": len({request.token for request in requests}),
        "categories": category_stats,
    }
    return selected_sorted, diagnostics


def _unique_ranked(tokens: list[tuple[float, int]]) -> list[int]:
    tokens.sort(key=lambda item: (-item[0], item[1]))
    output: list[int] = []
    seen: set[int] = set()
    for _, token in tokens:
        if token not in seen:
            output.append(token)
            seen.add(token)
    return output


def _candidate_pool(
    *,
    budget: int,
    quality: torch.Tensor,
    component_ids: torch.Tensor,
    temporal_ids: torch.Tensor,
    spatial_ids: torch.Tensor,
    query_relevance: torch.Tensor,
    atom_weights: torch.Tensor,
    query_mode: str,
    locked: list[int],
    multiplier: float,
) -> tuple[torch.Tensor, dict[str, object]]:
    total_tokens = int(quality.numel())
    limit = min(total_tokens, max(budget, int(math.ceil(budget * max(1.0, multiplier)))))
    offers: dict[str, list[int]] = {name: [] for name in _CANDIDATE_SOURCES}

    if query_mode in {"design_only", "certificates_and_design"} and query_relevance.numel() > 0:
        ranked: list[tuple[float, int]] = []
        for atom_idx, scores in enumerate(query_relevance):
            for temporal_id in torch.unique(temporal_ids).tolist():
                members = torch.where(temporal_ids == int(temporal_id))[0]
                if members.numel() > 0:
                    token = int(members[torch.argmax(scores[members])].item())
                    ranked.append((float(scores[token] * atom_weights[atom_idx].clamp_min(1e-6)), token))
        offers["query"] = _unique_ranked(ranked)

    quality_cpu = quality.detach().float().cpu().tolist()
    representatives: dict[int, int] = {}
    for token, component in enumerate(component_ids.detach().cpu().tolist()):
        previous = representatives.get(int(component))
        if previous is None or quality_cpu[token] > quality_cpu[previous]:
            representatives[int(component)] = token
    offers["trajectory"] = sorted(
        representatives.values(),
        key=lambda token: (-quality_cpu[token], token),
    )

    joint_cells = temporal_ids * (int(spatial_ids.max().item()) + 1) + spatial_ids
    cells: list[tuple[float, int]] = []
    for cell_id in torch.unique(joint_cells).tolist():
        members = torch.where(joint_cells == int(cell_id))[0]
        token = int(members[torch.argmax(quality[members])].item())
        cells.append((float(quality[token]), token))
    offers["spatial"] = _unique_ranked(cells)
    offers["global"] = torch.argsort(quality, descending=True, stable=True).detach().cpu().tolist()

    selected: set[int] = set(int(token) for token in locked)
    contributed = {name: 0 for name in _CANDIDATE_SOURCES}
    pointers = {name: 0 for name in _CANDIDATE_SOURCES}
    remaining = max(0, limit - len(selected))
    active = [name for name in _CANDIDATE_SOURCES if offers[name]]
    quotas = _largest_remainder(remaining, _CANDIDATE_SHARES, active)

    def offer(source: str) -> bool:
        source_tokens = offers[source]
        while pointers[source] < len(source_tokens):
            token = int(source_tokens[pointers[source]])
            pointers[source] += 1
            if token in selected:
                continue
            if len(selected) >= limit:
                return False
            selected.add(token)
            contributed[source] += 1
            return True
        return False

    for source in active:
        while contributed[source] < quotas[source]:
            if not offer(source):
                break

    cycle: list[str] = []
    for source in _CANDIDATE_SOURCES:
        cycle.extend([source] * max(1, int(round(_CANDIDATE_SHARES[source] * 20))))
    while len(selected) < limit:
        progress = False
        for source in cycle:
            if len(selected) >= limit:
                break
            progress = offer(source) or progress
        if not progress:
            break
    if len(selected) < limit:
        for token in offers["global"]:
            selected.add(token)
            if len(selected) >= limit:
                break
    if len(selected) < budget:
        raise RuntimeError(f"candidate pool has {len(selected)} candidates for budget {budget}")

    diagnostics: dict[str, object] = {
        "limit": limit,
        "locked": len(locked),
        "sources": {
            source: {
                "offered": len(offers[source]),
                "unique": len(set(offers[source])),
                "quota": quotas.get(source, 0),
                "admitted": contributed[source],
                "covered": len(set(offers[source]) & selected),
            }
            for source in _CANDIDATE_SOURCES
        },
    }
    return torch.tensor(sorted(selected), dtype=torch.long, device=quality.device), diagnostics


def _d_optimal_greedy(
    *,
    design: torch.Tensor,
    candidates: torch.Tensor,
    locked: list[int],
    budget: int,
    ridge: float,
) -> torch.Tensor:
    rows = design[candidates].float()
    candidate_count, design_dim = rows.shape
    if candidate_count < budget:
        raise RuntimeError(f"D-optimal pool has {candidate_count} candidates for budget {budget}")
    ridge = max(1e-4, float(ridge))
    inverse = torch.eye(design_dim, dtype=torch.float32, device=rows.device) / ridge
    leverage = rows.square().sum(dim=1) / ridge
    active = torch.ones(candidate_count, dtype=torch.bool, device=rows.device)
    token_to_column = {int(token): idx for idx, token in enumerate(candidates.detach().cpu().tolist())}
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

    for token in locked:
        if len(selected_columns) >= budget:
            break
        column = token_to_column.get(int(token))
        if column is not None:
            add(column)
    while len(selected_columns) < budget:
        score = torch.log1p(leverage.clamp_min(0.0)).masked_fill(~active, float("-inf"))
        column = int(torch.argmax(score).item())
        if not math.isfinite(float(score[column])):
            remaining = torch.where(active)[0]
            if remaining.numel() == 0:
                break
            column = int(remaining[0])
        add(column)
    if len(selected_columns) != budget:
        raise RuntimeError(f"D-optimal selector produced {len(selected_columns)} tokens for budget {budget}")
    return candidates[torch.tensor(selected_columns, dtype=torch.long, device=candidates.device)]


def _factor_information(
    rows: torch.Tensor,
    ridge: float,
) -> tuple[torch.Tensor, float, float]:
    dimension = int(rows.shape[1])
    identity = torch.eye(dimension, dtype=torch.float32, device=rows.device)
    base = max(1e-4, float(ridge)) * identity + rows.float().transpose(0, 1) @ rows.float()
    for jitter in (0.0, 1e-6, 1e-5, 1e-4):
        chol, info = torch.linalg.cholesky_ex(base + jitter * identity, check_errors=False)
        if int(info.max().item()) == 0:
            logdet = float((2.0 * torch.log(torch.diagonal(chol).clamp_min(1e-20))).sum().item())
            return chol, logdet, jitter
    raise RuntimeError("CertVID V4 information matrix is not positive definite after jitter")


def _leverage(rows: torch.Tensor, chol: torch.Tensor) -> torch.Tensor:
    if rows.numel() == 0:
        return rows.new_empty((0,), dtype=torch.float32)
    solved = torch.cholesky_solve(rows.float().transpose(0, 1), chol)
    return torch.sum(rows.float().transpose(0, 1) * solved, dim=0)


def _swap_refine(
    *,
    selected: torch.Tensor,
    candidates: torch.Tensor,
    design: torch.Tensor,
    locked: list[int],
    ridge: float,
    steps: int,
    pool_size: int,
    margin: float,
) -> tuple[torch.Tensor, int, float, float]:
    locked_set = set(int(token) for token in locked)
    candidate_tokens = candidates.detach().cpu().tolist()
    token_to_column = {int(token): idx for idx, token in enumerate(candidate_tokens)}
    selected_tokens = [int(token) for token in selected.detach().cpu().tolist()]
    selected_columns = [token_to_column[token] for token in selected_tokens]
    rows = design[candidates].float()
    swaps = 0
    max_jitter = 0.0
    current_rows = rows[torch.tensor(selected_columns, dtype=torch.long, device=rows.device)]
    _, current_logdet, jitter = _factor_information(current_rows, ridge)
    max_jitter = max(max_jitter, jitter)

    for _ in range(max(0, int(steps))):
        selected_tensor = torch.tensor(selected_columns, dtype=torch.long, device=rows.device)
        selected_rows = rows[selected_tensor]
        chol, iteration_logdet, jitter = _factor_information(selected_rows, ridge)
        max_jitter = max(max_jitter, jitter)
        current_logdet = iteration_logdet
        selected_leverage = _leverage(selected_rows, chol).clamp(0.0, 1.0 - 1e-6)
        removal_loss = -torch.log1p(-selected_leverage)
        removable = [idx for idx, token in enumerate(selected_tokens) if token not in locked_set]
        if not removable:
            break
        removable.sort(key=lambda idx: (float(removal_loss[idx]), selected_tokens[idx]))
        removable = removable[: max(1, int(pool_size))]

        selected_set = set(selected_columns)
        outside_columns = [idx for idx in range(len(candidate_tokens)) if idx not in selected_set]
        if not outside_columns:
            break
        outside_tensor = torch.tensor(outside_columns, dtype=torch.long, device=rows.device)
        outside_rows = rows[outside_tensor]
        outside_leverage = _leverage(outside_rows, chol).clamp_min(0.0)
        outside_order = torch.argsort(outside_leverage, descending=True, stable=True)
        outside_order = outside_order[: max(1, int(pool_size))]
        outside_tensor = outside_tensor[outside_order]
        outside_rows = rows[outside_tensor]
        outside_leverage = outside_leverage[outside_order]

        best_delta = float(margin)
        best_remove = -1
        best_add = -1
        for position in removable:
            removed = selected_rows[position]
            direction = torch.cholesky_solve(removed.unsqueeze(1), chol).squeeze(1)
            remove_denominator = 1.0 - torch.dot(removed, direction)
            if float(remove_denominator) <= 1e-6:
                continue
            cross = outside_rows @ direction
            add_without = (outside_leverage + cross.square() / remove_denominator).clamp_min(0.0)
            delta = torch.log(remove_denominator) + torch.log1p(add_without)
            local = int(torch.argmax(delta))
            local_delta = float(delta[local])
            add_column = int(outside_tensor[local])
            if local_delta > best_delta or (
                abs(local_delta - best_delta) <= 1e-12
                and best_add >= 0
                and candidate_tokens[add_column] < candidate_tokens[best_add]
            ):
                best_delta = local_delta
                best_remove = position
                best_add = add_column
        if best_remove < 0:
            break

        proposed_columns = list(selected_columns)
        proposed_columns[best_remove] = best_add
        proposed_rows = rows[torch.tensor(proposed_columns, dtype=torch.long, device=rows.device)]
        _, proposed_logdet, jitter = _factor_information(proposed_rows, ridge)
        max_jitter = max(max_jitter, jitter)
        if proposed_logdet + 1e-6 < current_logdet:
            raise RuntimeError(
                f"CertVID V4 swap reduced log-det from {current_logdet:.8f} to {proposed_logdet:.8f}"
            )
        if proposed_logdet <= current_logdet + float(margin):
            break
        selected_columns = proposed_columns
        selected_tokens[best_remove] = int(candidate_tokens[best_add])
        current_logdet = proposed_logdet
        swaps += 1

    if not locked_set.issubset(set(selected_tokens)):
        raise RuntimeError("CertVID V4 swap refinement removed a locked certificate")
    final_tensor = torch.tensor(selected_columns, dtype=torch.long, device=rows.device)
    _, final_logdet, jitter = _factor_information(rows[final_tensor], ridge)
    max_jitter = max(max_jitter, jitter)
    return candidates[final_tensor], swaps, final_logdet, max_jitter


def _design_protected_tokens(
    selected: torch.Tensor,
    design: torch.Tensor,
    ridge: float,
    ratio: float,
) -> tuple[set[int], float]:
    rows = design[selected].float()
    chol, _, jitter = _factor_information(rows, ridge)
    leverage = _leverage(rows, chol)
    count = min(int(selected.numel()), max(0, int(math.ceil(selected.numel() * max(0.0, ratio)))))
    if count <= 0:
        return set(), jitter
    positions = torch.argsort(leverage, descending=True, stable=True)[:count]
    return {int(selected[position]) for position in positions}, jitter


def _build_plan(
    *,
    selected: torch.Tensor,
    metric_features: torch.Tensor,
    demand_weight: torch.Tensor,
    temporal_ids: torch.Tensor,
    component_ids: torch.Tensor,
    fusion_alpha: float,
    temperature: float,
    protected_tokens: set[int],
) -> CertVidPlan:
    total_tokens = int(metric_features.shape[0])
    budget = int(selected.numel())
    similarity = metric_features @ metric_features[selected].transpose(0, 1)
    temporal_valid = (temporal_ids.unsqueeze(1) - temporal_ids[selected].unsqueeze(0)).abs() <= 1
    similarity = similarity.masked_fill(~temporal_valid, -2.0)
    same_component = component_ids.unsqueeze(1) == component_ids[selected].unsqueeze(0)
    similarity = similarity + 0.08 * same_component.float()
    values, assignment = torch.topk(similarity, k=min(2, budget), dim=1, largest=True)
    weights = torch.softmax(values.float() / max(1e-4, float(temperature)), dim=1)

    positions = torch.arange(budget, dtype=torch.long, device=selected.device)
    assignment[selected, 0] = positions
    weights[selected] = 0.0
    weights[selected, 0] = 1.0
    source_mass = (0.5 + 0.5 * demand_weight * float(total_tokens)).clamp(0.25, 2.0)
    alpha = torch.full(
        (budget,),
        min(max(float(fusion_alpha), 0.0), 0.75),
        dtype=torch.float32,
        device=selected.device,
    )
    if protected_tokens:
        protected = torch.tensor(sorted(protected_tokens), dtype=torch.long, device=selected.device)
        alpha[torch.isin(selected, protected)] = 0.0
    return CertVidPlan(
        anchor_indices=selected,
        assignment_indices=assignment,
        assignment_weights=weights,
        source_mass=source_mass,
        fusion_alpha=alpha,
        raw_token_count=total_tokens,
    )


def _publish_diagnostics(config: FlashVidConfig, diagnostics: dict[str, object]) -> None:
    setattr(config, "last_certv4_diagnostics", diagnostics)
    budget = diagnostics["budget"]
    certificates = diagnostics["certificates"]
    d_optimal = diagnostics["d_optimal"]
    attention = diagnostics["attention"]
    setattr(config, "last_certv4_target_tokens", float(budget["target_tokens"]))
    setattr(config, "last_certv4_nominal_retention", float(budget["nominal_retention"]))
    setattr(config, "last_certv4_outer_retention", float(budget["outer_retention"]))
    setattr(config, "last_certv4_post_inner_retention", float(budget["post_inner_retention"]))
    setattr(config, "last_certv4_average_layer_multiplier", float(budget["average_layer_multiplier"]))
    setattr(config, "last_certv4_post_inner_tokens", float(budget["post_inner_tokens"]))
    setattr(config, "last_certv4_average_layer_tokens", float(budget["average_layer_tokens"]))
    setattr(config, "last_certv4_certificate_count", float(certificates["admitted_unique"]))
    setattr(config, "last_certv4_candidate_tokens", float(diagnostics["candidate_count"]))
    setattr(config, "last_certv4_component_count", float(diagnostics["component_count"]))
    setattr(config, "last_certv4_query_confidence", float(diagnostics["query_confidence"]))
    setattr(config, "last_certv4_swap_count", float(d_optimal["swaps"]))
    setattr(config, "last_certv4_logdet", float(d_optimal["logdet"]))
    setattr(config, "last_certv4_attention_used", float(bool(attention["used"])))
    if bool(getattr(config, "certv4_debug", False)):
        print(f"[certvid-v4] {diagnostics}")


def certvid_v4_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
    question_features: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a verifiable budget-constrained D-optimal visual coreset."""
    if video_features.ndim != 3:
        raise ValueError(f"expected video_features [T, HW, D], got {tuple(video_features.shape)}")
    frame_count, tokens_per_frame, _ = video_features.shape
    total_tokens = int(frame_count * tokens_per_frame)
    budget, budget_diagnostics = _resolve_budget(flashvid_config, total_tokens)
    attention, attention_diagnostics = _validated_attention(
        cls_attention,
        frame_count,
        tokens_per_frame,
        flashvid_config,
    )
    flat_features = video_features.reshape(total_tokens, -1)
    query_mode = str(getattr(flashvid_config, "certv4_query_mode", "certificates_and_design")).strip().lower()
    if query_mode not in _QUERY_MODES:
        raise ValueError(f"unsupported certv4_query_mode={query_mode!r}")

    if budget >= total_tokens:
        plan = _identity_plan(total_tokens, video_features.device)
        output = flat_features
        diagnostics: dict[str, object] = {
            "budget": budget_diagnostics,
            "attention": attention_diagnostics,
            "query_mode": query_mode,
            "query_confidence": 0.0,
            "certificates": {
                "cap": int(math.floor(total_tokens * min(1.0, max(0.0, _cfg_float(flashvid_config, "certv4_certificate_budget_ratio", 0.40))))),
                "admitted_unique": 0,
                "ratio": 0.0,
                "requested": 0,
                "deduplicated": 0,
                "categories": {},
            },
            "candidates": {"limit": total_tokens, "locked": 0, "sources": {}},
            "candidate_count": total_tokens,
            "component_count": 0,
            "design_protected_count": 0,
            "fusion_protected_count": 0,
            "d_optimal": {"swaps": 0, "logdet": 0.0, "cholesky_jitter": 0.0},
        }
    else:
        metric_dim = max(32, _cfg_int(flashvid_config, "certv4_metric_dim", 96))
        metric_flat = _metric_features(video_features, metric_dim)
        metric_frames = metric_flat.view(frame_count, tokens_per_frame, -1)
        height, width = _grid_hw(tokens_per_frame, flashvid_config)
        spatial_bins = max(1, _cfg_int(flashvid_config, "certv4_spatial_bins", 3))
        coords, frame_spatial_ids = _spatial_layout(
            tokens_per_frame,
            height,
            width,
            spatial_bins,
            video_features.device,
        )
        frame_event, _, novelty_2d, curvature_2d, matches = _trajectory_signals(
            metric_frames,
            coords,
            _cfg_float(flashvid_config, "certv4_spatial_penalty", 0.08),
        )
        component_ids_cpu, component_sizes_cpu = _build_components(
            frame_count,
            tokens_per_frame,
            frame_event,
            matches,
            _cfg_float(flashvid_config, "certv4_track_threshold", 0.82),
        )
        component_ids = component_ids_cpu.to(video_features.device)
        component_sizes = component_sizes_cpu.to(video_features.device)
        frame_ids = torch.arange(frame_count, device=video_features.device).repeat_interleave(tokens_per_frame)
        component_value = _component_support(
            metric_flat,
            component_ids,
            component_sizes,
            frame_ids,
            frame_count,
        )
        temporal_count = min(frame_count, max(1, _cfg_int(flashvid_config, "certv4_temporal_bins", 12)))
        temporal_ids = torch.div(
            frame_ids * temporal_count,
            max(1, frame_count),
            rounding_mode="floor",
        ).clamp_max(temporal_count - 1)
        spatial_ids = frame_spatial_ids.repeat(frame_count)
        spatial_count = spatial_bins * spatial_bins

        novelty = novelty_2d.reshape(-1)
        curvature = curvature_2d.reshape(-1)
        detail = _local_detail(video_features, height, width).reshape(-1)
        event = frame_event.repeat_interleave(tokens_per_frame)
        atoms = _question_atoms(
            question_features,
            max(0, _cfg_int(flashvid_config, "certv4_query_atoms", 8)),
            metric_dim,
        ).to(video_features.device)
        query_relevance, atom_weights, query_confidence = _question_relevance(atoms, metric_flat)

        quality = _minmax(
            0.20 * attention
            + 0.24 * novelty
            + 0.18 * curvature
            + 0.14 * event
            + 0.12 * detail
            + 0.12 * component_value,
            dim=0,
        )
        event_score = _minmax(
            0.38 * novelty + 0.31 * curvature + 0.20 * event + 0.11 * detail,
            dim=0,
        )
        demand_weight = 0.20 + 0.44 * quality + 0.22 * event_score + 0.14 * component_value
        demand_weight = demand_weight / demand_weight.sum().clamp_min(1e-6)

        requests = _build_certificate_requests(
            quality=quality,
            event_score=event_score,
            frame_ids=frame_ids,
            temporal_ids=temporal_ids,
            spatial_ids=spatial_ids,
            query_relevance=query_relevance,
            atom_weights=atom_weights,
            query_confidence=query_confidence,
            query_mode=query_mode,
            frame_count=frame_count,
            temporal_count=temporal_count,
            spatial_count=spatial_count,
            frame_coverage_ratio=_cfg_float(flashvid_config, "certv4_frame_coverage_ratio", 1.0),
            cell_coverage_ratio=_cfg_float(flashvid_config, "certv4_cell_coverage_ratio", 0.50),
            query_threshold=_cfg_float(flashvid_config, "certv4_query_threshold", 0.10),
            query_per_atom=_cfg_int(flashvid_config, "certv4_query_per_atom", 1),
        )
        locked, certificate_diagnostics = _admit_certificates(
            requests,
            budget,
            _cfg_float(flashvid_config, "certv4_certificate_budget_ratio", 0.40),
        )
        candidates, candidate_diagnostics = _candidate_pool(
            budget=budget,
            quality=quality,
            component_ids=component_ids,
            temporal_ids=temporal_ids,
            spatial_ids=spatial_ids,
            query_relevance=query_relevance,
            atom_weights=atom_weights,
            query_mode=query_mode,
            locked=locked,
            multiplier=_cfg_float(flashvid_config, "certv4_candidate_multiplier", 2.5),
        )
        design = _design_features(
            metric_features=metric_flat,
            quality=quality,
            temporal_ids=temporal_ids,
            spatial_ids=spatial_ids,
            novelty=novelty,
            curvature=curvature,
            event=event,
            detail=detail,
            component_support=component_value,
            query_relevance=query_relevance,
            atom_weights=atom_weights,
            query_confidence=query_confidence,
            query_mode=query_mode,
            temporal_count=temporal_count,
            spatial_count=spatial_count,
            structural_weight=_cfg_float(flashvid_config, "certv4_structural_weight", 0.32),
            whitening_strength=_cfg_float(flashvid_config, "certv4_whitening_strength", 0.50),
            quality_floor=_cfg_float(flashvid_config, "certv4_quality_floor", 0.15),
        )
        ridge = _cfg_float(flashvid_config, "certv4_ridge", 0.50)
        selected = _d_optimal_greedy(
            design=design,
            candidates=candidates,
            locked=locked,
            budget=budget,
            ridge=ridge,
        )
        selected, swaps, logdet, swap_jitter = _swap_refine(
            selected=selected,
            candidates=candidates,
            design=design,
            locked=locked,
            ridge=ridge,
            steps=_cfg_int(flashvid_config, "certv4_swap_steps", 6),
            pool_size=_cfg_int(flashvid_config, "certv4_swap_pool", 24),
            margin=_cfg_float(flashvid_config, "certv4_swap_margin", 1e-4),
        )
        selected = torch.sort(selected).values
        design_protected, protection_jitter = _design_protected_tokens(
            selected,
            design,
            ridge,
            _cfg_float(flashvid_config, "certv4_design_protect_ratio", 0.15),
        )
        protected_tokens = set(locked) | design_protected
        plan = _build_plan(
            selected=selected,
            metric_features=metric_flat,
            demand_weight=demand_weight,
            temporal_ids=temporal_ids,
            component_ids=component_ids,
            fusion_alpha=_cfg_float(flashvid_config, "certv4_fusion_alpha", 0.12),
            temperature=_cfg_float(flashvid_config, "certv4_assignment_temperature", 0.07),
            protected_tokens=protected_tokens,
        )
        output = apply_certvid_plan(flat_features, plan)
        diagnostics = {
            "budget": budget_diagnostics,
            "attention": attention_diagnostics,
            "query_mode": query_mode,
            "query_confidence": float(query_confidence),
            "certificates": certificate_diagnostics,
            "candidates": candidate_diagnostics,
            "candidate_count": int(candidates.numel()),
            "component_count": int(component_sizes.numel()),
            "design_protected_count": len(design_protected),
            "fusion_protected_count": len(protected_tokens),
            "d_optimal": {
                "swaps": int(swaps),
                "logdet": float(logdet),
                "cholesky_jitter": float(max(swap_jitter, protection_jitter)),
            },
        }

    setattr(flashvid_config, "_certvid_plan", plan)
    flashvid_config.vision_token_length = int(output.shape[0])
    flashvid_config.visual_token_length = int(output.shape[0])
    flashvid_config.llm_token_length = None
    setattr(flashvid_config, "last_adapter_variant", "certvid_v4")
    setattr(flashvid_config, "last_adapter_raw_tokens", float(total_tokens))
    setattr(flashvid_config, "last_adapter_output_tokens", float(output.shape[0]))
    _publish_diagnostics(flashvid_config, diagnostics)
    return output, plan.anchor_indices
