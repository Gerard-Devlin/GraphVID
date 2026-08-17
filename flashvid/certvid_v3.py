from __future__ import annotations

import json
import math
import os
from typing import Any, MutableMapping, Optional

import torch
import torch.nn.functional as F

from .certvid import (
    CertVidPlan,
    _build_components,
    _build_plan,
    _cfg_float,
    _cfg_int,
    _grid_hw,
    _local_detail,
    _metric_features,
    _minmax,
    _question_atoms,
    _question_relevance,
    _rank_normalize,
    _spatial_layout,
    apply_certvid_plan,
)
from .certvid_v2 import _component_support, _trajectory_signals
from .configuration_flashvid import FlashVidConfig


_PROFILE_ENV = "CERTV3_PROFILE_PHASES"


def _profile_enabled(features: torch.Tensor) -> bool:
    value = os.environ.get(_PROFILE_ENV, "").strip().lower()
    return features.is_cuda and value in {"1", "true", "yes", "on"}


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
    ratio = _cfg_float(config, "retention_ratio", 0.10)
    if bool(getattr(config, "certv3_budget_uses_expansion", True)):
        ratio *= _cfg_float(config, "expansion", 1.0)
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


def _evenly_spaced_ids(count: int, keep: int, device: torch.device) -> torch.Tensor:
    if keep >= count:
        return torch.arange(count, dtype=torch.long, device=device)
    if keep <= 1:
        return torch.tensor([count // 2], dtype=torch.long, device=device)
    return torch.round(torch.linspace(0, count - 1, steps=keep, device=device)).long().unique()


def _hard_certificates(
    *,
    budget: int,
    quality: torch.Tensor,
    event_score: torch.Tensor,
    frame_ids: torch.Tensor,
    temporal_ids: torch.Tensor,
    spatial_ids: torch.Tensor,
    query_relevance: torch.Tensor,
    atom_weights: torch.Tensor,
    query_confidence: float,
    frame_count: int,
    temporal_count: int,
    spatial_count: int,
    frame_coverage_ratio: float,
    cell_coverage_ratio: float,
    query_threshold: float,
    query_per_atom: int,
    use_spatiotemporal: bool = True,
) -> tuple[list[int], list[int]]:
    """Build immutable temporal, spatial, event, and query evidence constraints."""
    entries: list[tuple[float, float, int, bool]] = []

    if use_spatiotemporal:
        frame_keep = max(1, int(math.ceil(frame_count * min(1.0, max(0.0, frame_coverage_ratio)))))
        for frame_id in _evenly_spaced_ids(frame_count, frame_keep, quality.device).tolist():
            members = torch.where(frame_ids == int(frame_id))[0]
            if members.numel() > 0:
                token = int(members[torch.argmax(quality[members])].item())
                entries.append((5.0, float(quality[token].item()), token, False))

        # Every temporal interval keeps its strongest transition or curved trajectory.
        for temporal_id in range(temporal_count):
            members = torch.where(temporal_ids == temporal_id)[0]
            if members.numel() > 0:
                token = int(members[torch.argmax(event_score[members])].item())
                entries.append((4.5, float(event_score[token].item()), token, False))

        cells_per_interval = max(
            0,
            min(spatial_count, int(math.ceil(spatial_count * min(1.0, max(0.0, cell_coverage_ratio))))),
        )
        for temporal_id in range(temporal_count):
            cell_representatives: list[tuple[float, int]] = []
            for spatial_id in range(spatial_count):
                members = torch.where(
                    (temporal_ids == temporal_id) & (spatial_ids == spatial_id)
                )[0]
                if members.numel() == 0:
                    continue
                token = int(members[torch.argmax(quality[members])].item())
                cell_representatives.append((float(quality[token].item()), token))
            cell_representatives.sort(key=lambda item: (-item[0], item[1]))
            for score, token in cell_representatives[:cells_per_interval]:
                entries.append((3.0, score, token, False))

    query_seeds: list[int] = []
    if query_relevance.numel() > 0 and query_confidence >= float(query_threshold):
        per_atom = max(1, int(query_per_atom))
        for atom_idx, atom_scores in enumerate(query_relevance):
            per_interval: list[tuple[float, int]] = []
            for temporal_id in range(temporal_count):
                members = torch.where(temporal_ids == temporal_id)[0]
                if members.numel() == 0:
                    continue
                token = int(members[torch.argmax(atom_scores[members])].item())
                per_interval.append((float(atom_scores[token].item()), token))
            per_interval.sort(key=lambda item: (-item[0], item[1]))
            atom_priority = 6.0 + float(atom_weights[atom_idx].item())
            for score, token in per_interval[:per_atom]:
                entries.append((atom_priority, score, token, True))
                query_seeds.append(token)

    best_by_token: dict[int, tuple[float, float, int, bool]] = {}
    for entry in entries:
        previous = best_by_token.get(entry[2])
        if previous is None or (entry[0], entry[1]) > (previous[0], previous[1]):
            best_by_token[entry[2]] = entry
    ordered = sorted(best_by_token.values(), key=lambda item: (-item[0], -item[1], item[2]))
    mandatory = [entry[2] for entry in ordered[:budget]]
    mandatory_set = set(mandatory)
    query_seeds = list(dict.fromkeys(token for token in query_seeds if token in mandatory_set))
    return mandatory, query_seeds


def _budget_aware_certificates(
    *,
    budget: int,
    certificate_ratio: float,
    quality: torch.Tensor,
    event_score: torch.Tensor,
    frame_ids: torch.Tensor,
    temporal_ids: torch.Tensor,
    spatial_ids: torch.Tensor,
    query_relevance: torch.Tensor,
    atom_weights: torch.Tensor,
    query_confidence: float,
    frame_count: int,
    temporal_count: int,
    spatial_count: int,
    frame_coverage_ratio: float,
    cell_coverage_ratio: float,
    query_threshold: float,
    query_per_atom: int,
    use_spatiotemporal: bool = True,
) -> tuple[list[int], list[int], dict[str, int]]:
    """Build a category-balanced certificate subset under a hard budget cap."""
    Request = tuple[float, float, int, bool]
    requests: dict[str, list[Request]] = {
        "query": [],
        "frame": [],
        "temporal": [],
        "spatial": [],
    }

    def materialize_pairs(
        token_tensors: list[torch.Tensor],
        score_tensors: list[torch.Tensor],
    ) -> list[tuple[float, int]]:
        if not token_tensors:
            return []
        tokens_cpu = torch.stack(token_tensors).detach().cpu().tolist()
        scores_cpu = torch.stack(score_tensors).detach().float().cpu().tolist()
        return [
            (float(score), int(token))
            for score, token in zip(scores_cpu, tokens_cpu)
        ]

    if use_spatiotemporal:
        frame_keep = max(
            1,
            int(
                math.ceil(
                    frame_count
                    * min(1.0, max(0.0, frame_coverage_ratio))
                )
            ),
        )
        frame_tokens: list[torch.Tensor] = []
        frame_scores: list[torch.Tensor] = []
        for frame_id in _evenly_spaced_ids(
            frame_count,
            frame_keep,
            quality.device,
        ).tolist():
            members = torch.where(frame_ids == int(frame_id))[0]
            if members.numel() > 0:
                token = members[torch.argmax(quality[members])]
                frame_tokens.append(token)
                frame_scores.append(quality[token])
        requests["frame"].extend(
            (5.0, score, token, False)
            for score, token in materialize_pairs(frame_tokens, frame_scores)
        )

        temporal_tokens: list[torch.Tensor] = []
        temporal_scores: list[torch.Tensor] = []
        for temporal_id in range(temporal_count):
            members = torch.where(temporal_ids == temporal_id)[0]
            if members.numel() > 0:
                token = members[torch.argmax(event_score[members])]
                temporal_tokens.append(token)
                temporal_scores.append(event_score[token])
        requests["temporal"].extend(
            (4.5, score, token, False)
            for score, token in materialize_pairs(
                temporal_tokens,
                temporal_scores,
            )
        )

        cells_per_interval = max(
            0,
            min(
                spatial_count,
                int(
                    math.ceil(
                        spatial_count
                        * min(1.0, max(0.0, cell_coverage_ratio))
                    )
                ),
            ),
        )
        cell_groups: list[list[int]] = [[] for _ in range(temporal_count)]
        cell_tokens: list[torch.Tensor] = []
        cell_scores: list[torch.Tensor] = []
        for temporal_id in range(temporal_count):
            for spatial_id in range(spatial_count):
                members = torch.where(
                    (temporal_ids == temporal_id) & (spatial_ids == spatial_id)
                )[0]
                if members.numel() == 0:
                    continue
                token = members[torch.argmax(quality[members])]
                cell_groups[temporal_id].append(len(cell_tokens))
                cell_tokens.append(token)
                cell_scores.append(quality[token])
        cell_pairs = materialize_pairs(cell_tokens, cell_scores)
        for temporal_id in range(temporal_count):
            cell_representatives = [
                cell_pairs[position]
                for position in cell_groups[temporal_id]
            ]
            cell_representatives.sort(key=lambda item: (-item[0], item[1]))
            for score, token in cell_representatives[:cells_per_interval]:
                requests["spatial"].append((3.0, score, token, False))

    if query_relevance.numel() > 0 and query_confidence >= float(query_threshold):
        per_atom = max(1, int(query_per_atom))
        query_groups: list[list[int]] = [
            [] for _ in range(int(query_relevance.shape[0]))
        ]
        query_tokens: list[torch.Tensor] = []
        query_scores: list[torch.Tensor] = []
        for atom_idx, atom_scores in enumerate(query_relevance):
            for temporal_id in range(temporal_count):
                members = torch.where(temporal_ids == temporal_id)[0]
                if members.numel() == 0:
                    continue
                token = members[torch.argmax(atom_scores[members])]
                query_groups[atom_idx].append(len(query_tokens))
                query_tokens.append(token)
                query_scores.append(atom_scores[token])
        query_pairs = materialize_pairs(query_tokens, query_scores)
        atom_weights_cpu = atom_weights.detach().float().cpu().tolist()
        for atom_idx in range(int(query_relevance.shape[0])):
            per_interval = [
                query_pairs[position]
                for position in query_groups[atom_idx]
            ]
            per_interval.sort(key=lambda item: (-item[0], item[1]))
            atom_priority = 6.0 + float(atom_weights_cpu[atom_idx])
            for score, token in per_interval[:per_atom]:
                requests["query"].append((atom_priority, score, token, True))

    # Keep frame requests temporally spread; rank all other categories by evidence.
    category_order = ("query", "frame", "temporal", "spatial")
    for category in category_order:
        if category != "frame":
            requests[category].sort(
                key=lambda item: (-item[0], -item[1], item[2])
            )

    ratio = min(1.0, max(0.0, float(certificate_ratio)))
    certificate_limit = min(budget, int(math.floor(budget * ratio)))
    category_weights = {"query": 2, "frame": 2, "temporal": 2, "spatial": 1}
    active = [category for category in category_order if requests[category]]
    total_weight = sum(category_weights[category] for category in active)
    quotas = {category: 0 for category in category_order}
    if total_weight > 0:
        exact = {
            category: certificate_limit
            * category_weights[category]
            / float(total_weight)
            for category in active
        }
        for category in active:
            quotas[category] = int(math.floor(exact[category]))
        remainder = certificate_limit - sum(quotas.values())
        remainder_order = sorted(
            active,
            key=lambda category: (
                -(exact[category] - quotas[category]),
                category_order.index(category),
            ),
        )
        for category in remainder_order[:remainder]:
            quotas[category] += 1

    selected: list[int] = []
    selected_set: set[int] = set()
    covered_by_category = {category: 0 for category in category_order}

    def admit(entry: Request) -> None:
        token = int(entry[2])
        if token not in selected_set and len(selected) < certificate_limit:
            selected.append(token)
            selected_set.add(token)

    for category in category_order:
        category_requests = requests[category]
        quota = quotas[category]
        if quota <= 0:
            continue
        if category == "frame" and len(category_requests) > quota:
            positions = _evenly_spaced_ids(
                len(category_requests),
                quota,
                quality.device,
            ).tolist()
            position_set = set(positions)
            ordered_requests = [category_requests[position] for position in positions]
            ordered_requests.extend(
                entry
                for position, entry in enumerate(category_requests)
                if position not in position_set
            )
        else:
            ordered_requests = category_requests

        covered_tokens: set[int] = set()
        for entry in ordered_requests:
            token = int(entry[2])
            if token in covered_tokens:
                continue
            covered_tokens.add(token)
            covered_by_category[category] += 1
            admit(entry)
            if covered_by_category[category] >= quota:
                break

    # Duplicate requests and empty categories return their unused quota to the
    # strongest remaining requests while preserving the global certificate cap.
    all_requests = sorted(
        (entry for category in category_order for entry in requests[category]),
        key=lambda item: (-item[0], -item[1], item[2]),
    )
    for entry in all_requests:
        if len(selected) >= certificate_limit:
            break
        admit(entry)

    query_tokens = {int(entry[2]) for entry in requests["query"]}
    query_seeds = [token for token in selected if token in query_tokens]
    stats = {
        "limit": int(certificate_limit),
        "requested_unique": int(
            len({int(entry[2]) for entry in all_requests})
        ),
        **{
            f"{category}_requested": int(
                len({int(entry[2]) for entry in requests[category]})
            )
            for category in category_order
        },
        **{
            f"{category}_quota": int(quotas[category])
            for category in category_order
        },
        **{
            f"{category}_covered": int(covered_by_category[category])
            for category in category_order
        },
    }
    return selected, query_seeds, stats


def _candidate_pool(
    *,
    budget: int,
    quality: torch.Tensor,
    component_ids: torch.Tensor,
    temporal_ids: torch.Tensor,
    spatial_ids: torch.Tensor,
    query_relevance: torch.Tensor,
    mandatory: list[int],
    multiplier: float,
    use_trajectory: bool = True,
) -> torch.Tensor:
    total_tokens = int(quality.numel())
    limit = min(total_tokens, max(budget, int(math.ceil(budget * max(1.0, multiplier)))))
    candidates: set[int] = set(int(token) for token in mandatory)

    def offer(tokens: list[int]) -> None:
        for token in tokens:
            if len(candidates) >= limit:
                return
            candidates.add(int(token))

    # Expose query alternatives across time before filling with generic quality peaks.
    if query_relevance.numel() > 0:
        query_offer_tokens: list[torch.Tensor] = []
        query_offer_scores: list[torch.Tensor] = []
        for scores in query_relevance:
            for temporal_id in torch.unique(temporal_ids).tolist():
                members = torch.where(temporal_ids == int(temporal_id))[0]
                if members.numel() == 0:
                    continue
                token = members[torch.argmax(scores[members])]
                query_offer_tokens.append(token)
                query_offer_scores.append(scores[token])
        query_tokens_cpu = torch.stack(query_offer_tokens).detach().cpu().tolist()
        query_scores_cpu = torch.stack(query_offer_scores).detach().float().cpu().tolist()
        query_offers = list(zip(query_scores_cpu, query_tokens_cpu))
        query_offers.sort(key=lambda item: (-item[0], item[1]))
        offer([token for _, token in query_offers])

    # Semantic trajectory representatives make rare persistent objects eligible.
    quality_cpu = quality.detach().float().cpu().tolist()
    if use_trajectory:
        component_cpu = component_ids.detach().cpu().tolist()
        representatives: dict[int, int] = {}
        for token, component in enumerate(component_cpu):
            previous = representatives.get(component)
            if previous is None or quality_cpu[token] > quality_cpu[previous]:
                representatives[component] = token
        component_tokens = sorted(
            representatives.values(),
            key=lambda token: (-quality_cpu[token], token),
        )
        offer(component_tokens)

    # Add one strong alternative for every temporal-spatial cell.
    cell_token_tensors: list[torch.Tensor] = []
    cell_score_tensors: list[torch.Tensor] = []
    joint_cells = temporal_ids * (int(spatial_ids.max().item()) + 1) + spatial_ids
    for cell_id in torch.unique(joint_cells).tolist():
        members = torch.where(joint_cells == int(cell_id))[0]
        token = members[torch.argmax(quality[members])]
        cell_token_tensors.append(token)
        cell_score_tensors.append(quality[token])
    cell_tokens_cpu = torch.stack(cell_token_tensors).detach().cpu().tolist()
    cell_scores_cpu = torch.stack(cell_score_tensors).detach().float().cpu().tolist()
    cell_tokens = list(zip(cell_scores_cpu, cell_tokens_cpu))
    cell_tokens.sort(key=lambda item: (-item[0], item[1]))
    offer([token for _, token in cell_tokens])

    offer(torch.argsort(quality, descending=True, stable=True).detach().cpu().tolist())
    if len(candidates) < budget:
        offer(list(range(total_tokens)))
    return torch.tensor(sorted(candidates), dtype=torch.long, device=quality.device)


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
    inverse = torch.eye(design_dim, dtype=torch.float32, device=rows.device) / ridge
    leverage = rows.square().sum(dim=1) / ridge
    active = torch.ones(candidate_count, dtype=torch.bool, device=rows.device)
    token_to_column = {
        int(token): column for column, token in enumerate(candidates.detach().cpu().tolist())
    }
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
    """Fedorov-style exchanges improve log-det while preserving hard certificates."""
    steps = max(0, int(steps))
    if steps == 0 or selected.numel() == 0:
        return selected, 0, 0.0

    mandatory_set = set(int(token) for token in mandatory)
    candidate_tokens = candidates.detach().cpu().tolist()
    token_to_column = {int(token): column for column, token in enumerate(candidate_tokens)}
    selected_tokens = selected.detach().cpu().tolist()
    selected_columns = [token_to_column[int(token)] for token in selected_tokens]
    rows = design[candidates].float()
    ridge = max(1e-4, float(ridge))
    dimension = int(rows.shape[1])
    identity = torch.eye(dimension, dtype=torch.float32, device=rows.device)
    swaps = 0

    for _ in range(steps):
        selected_tensor = torch.tensor(selected_columns, dtype=torch.long, device=rows.device)
        selected_rows = rows[selected_tensor]
        information = ridge * identity + selected_rows.transpose(0, 1) @ selected_rows
        inverse = torch.linalg.inv(information)
        selected_leverage = torch.sum((selected_rows @ inverse) * selected_rows, dim=1).clamp(0.0, 1.0 - 1e-5)
        removal_loss = -torch.log1p(-selected_leverage)

        removable_positions = [
            position for position, token in enumerate(selected_tokens) if int(token) not in mandatory_set
        ]
        if not removable_positions:
            break
        removal_loss_cpu = removal_loss.detach().float().cpu().tolist()
        removable_positions.sort(
            key=lambda position: (
                float(removal_loss_cpu[position]),
                selected_tokens[position],
            )
        )
        removable_positions = removable_positions[: max(1, int(pool_size))]

        selected_column_set = set(selected_columns)
        outside_columns = [column for column in range(len(candidate_tokens)) if column not in selected_column_set]
        if not outside_columns:
            break
        outside_tensor = torch.tensor(outside_columns, dtype=torch.long, device=rows.device)
        outside_rows = rows[outside_tensor]
        outside_leverage = torch.sum((outside_rows @ inverse) * outside_rows, dim=1)
        outside_order = torch.argsort(outside_leverage, descending=True, stable=True)
        outside_order = outside_order[: max(1, int(pool_size))]
        outside_tensor = outside_tensor[outside_order]
        outside_rows = rows[outside_tensor]

        best_delta = float(margin)
        best_remove = -1
        best_add_column = -1
        local_delta_tensors: list[torch.Tensor] = []
        local_add_tensors: list[torch.Tensor] = []
        for position in removable_positions:
            removed = selected_rows[position]
            direction = inverse @ removed
            remove_denominator = (1.0 - torch.dot(removed, direction)).clamp_min(1e-5)
            inverse_without = inverse + torch.outer(direction, direction) / remove_denominator
            add_leverage = torch.sum((outside_rows @ inverse_without) * outside_rows, dim=1).clamp_min(0.0)
            delta = torch.log(remove_denominator) + torch.log1p(add_leverage)
            local = torch.argmax(delta)
            local_delta_tensors.append(delta[local])
            local_add_tensors.append(outside_tensor[local])

        local_deltas = torch.stack(local_delta_tensors).detach().float().cpu().tolist()
        local_add_columns = torch.stack(local_add_tensors).detach().cpu().tolist()
        for position, local_delta, local_add_column in zip(
            removable_positions,
            local_deltas,
            local_add_columns,
        ):
            local_delta = float(local_delta)
            if local_delta > best_delta:
                best_delta = local_delta
                best_remove = position
                best_add_column = int(local_add_column)

        if best_remove < 0:
            break
        selected_columns[best_remove] = best_add_column
        selected_tokens[best_remove] = int(candidate_tokens[best_add_column])
        swaps += 1

    selected_tensor = torch.tensor(selected_columns, dtype=torch.long, device=rows.device)
    final_rows = rows[selected_tensor]
    final_information = ridge * identity + final_rows.transpose(0, 1) @ final_rows
    sign, logabsdet = torch.linalg.slogdet(final_information)
    final_logdet = float(logabsdet.item()) if float(sign.item()) > 0.0 else float("-inf")
    return candidates[selected_tensor], swaps, final_logdet


def certvid_v3_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    flashvid_config: FlashVidConfig,
    question_features: Optional[torch.Tensor] = None,
    *,
    analysis_sink: Optional[MutableMapping[str, Any]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select a certified D-optimal visual evidence coreset under one budget."""
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
    compression_variant = str(
        getattr(flashvid_config, "compression_variant", "")
    ).strip().lower()
    qwen_v3_certificate_policy = (
        backbone in {"qwen2_5_vl", "qwen3_vl"}
        and compression_variant == "certvid_v3"
    )
    qwen_certificate_ratio = min(
        1.0,
        max(
            0.0,
            _cfg_float(
                flashvid_config,
                "certv3_qwen_certificate_budget_ratio",
                0.35,
            ),
        ),
    )
    certificate_ratio = min(
        1.0,
        max(
            0.0,
            _cfg_float(
                flashvid_config,
                "certv3_certificate_budget_ratio",
                1.0,
            ),
        ),
    )
    effective_certificate_ratio = (
        min(certificate_ratio, qwen_certificate_ratio)
        if qwen_v3_certificate_policy
        else certificate_ratio
    )
    certificate_limit = min(
        budget,
        int(math.floor(budget * effective_certificate_ratio)),
    )
    certificate_cap_active = False
    certificate_stats: dict[str, int] = {}
    qwen_certificate_limit = (
        certificate_limit if qwen_v3_certificate_policy else budget
    )
    qwen_certificate_cap_active = False
    original_certificate_count = budget
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
    use_spatiotemporal_certificates = _cfg_bool(
        flashvid_config, "certv3_use_spatiotemporal_certificates", True
    )
    use_trajectory = _cfg_bool(flashvid_config, "certv3_use_trajectory", True)
    use_query = _cfg_bool(flashvid_config, "certv3_use_query", True)

    if budget >= total_tokens:
        plan = _identity_plan(total_tokens, video_features.device)
        output = flat_features
        candidates = total_tokens
        components = total_tokens
        mandatory: list[int] = list(range(total_tokens))
        query_seeds: list[int] = []
        query_confidence = 0.0
        swaps = 0
        logdet = 0.0
        if analysis_sink is not None:
            analysis_sink["identity"] = True
    else:
        _profile_record(phase_events, "metric_layout", 0)
        metric_dim = max(32, _cfg_int(flashvid_config, "certv3_metric_dim", 96))
        metric_flat = _metric_features(video_features, metric_dim)
        metric_frames = metric_flat.view(frame_count, tokens_per_frame, -1)
        height, width = _grid_hw(tokens_per_frame, flashvid_config)
        spatial_bins = max(1, _cfg_int(flashvid_config, "certv3_spatial_bins", 3))
        coords, frame_spatial_ids = _spatial_layout(
            tokens_per_frame,
            height,
            width,
            spatial_bins,
            video_features.device,
        )
        _profile_record(phase_events, "metric_layout", 1)

        _profile_record(phase_events, "trajectory_components", 0)
        if use_trajectory:
            frame_event, _, novelty_2d, curvature_2d, matches = _trajectory_signals(
                metric_frames,
                coords,
                _cfg_float(flashvid_config, "certv3_spatial_penalty", 0.08),
            )
            component_ids_cpu, component_sizes_cpu = _build_components(
                frame_count,
                tokens_per_frame,
                frame_event,
                matches,
                _cfg_float(flashvid_config, "certv3_track_threshold", 0.82),
            )
        else:
            # The complete trajectory-dynamics ablation removes every signal
            # derived from cross-frame matching, including frame-level events.
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
            component_ids_cpu = torch.arange(total_tokens, dtype=torch.long)
            component_sizes_cpu = torch.ones(total_tokens, dtype=torch.long)
        component_ids = component_ids_cpu.to(video_features.device)
        component_sizes = component_sizes_cpu.to(video_features.device)
        frame_ids = torch.arange(frame_count, device=video_features.device).repeat_interleave(tokens_per_frame)
        component_value = (
            _component_support(
                metric_flat,
                component_ids,
                component_sizes,
                frame_ids,
                frame_count,
            )
            if use_trajectory
            else torch.zeros(total_tokens, dtype=torch.float32, device=video_features.device)
        )
        _profile_record(phase_events, "trajectory_components", 1)

        _profile_record(phase_events, "quality_signals", 0)
        temporal_count = min(frame_count, max(1, _cfg_int(flashvid_config, "certv3_temporal_bins", 12)))
        temporal_ids = torch.div(
            frame_ids * temporal_count,
            max(1, frame_count),
            rounding_mode="floor",
        ).clamp_max(temporal_count - 1)
        spatial_ids = frame_spatial_ids.repeat(frame_count)
        spatial_count = spatial_bins * spatial_bins

        attention = _rank_normalize(cls_attention.float()).reshape(-1)
        novelty = novelty_2d.reshape(-1)
        curvature = curvature_2d.reshape(-1)
        detail = _local_detail(video_features, height, width).reshape(-1)
        event = frame_event.repeat_interleave(tokens_per_frame)
        atoms = _question_atoms(
            question_features if use_query else None,
            max(0, _cfg_int(flashvid_config, "certv3_query_atoms", 8)),
            metric_dim,
        ).to(video_features.device)
        query_relevance, atom_weights, query_confidence = _question_relevance(atoms, metric_flat)
        query_score = (
            (query_relevance * atom_weights.unsqueeze(1)).sum(dim=0)
            if query_relevance.numel() > 0
            else torch.zeros(total_tokens, dtype=torch.float32, device=video_features.device)
        )

        query_weight = min(
            0.30,
            max(0.0, _cfg_float(flashvid_config, "certv3_query_weight", 0.18) * query_confidence),
        )
        visual_weights = [
            _cfg_float(flashvid_config, "certv3_visual_attention_weight", 0.28),
            _cfg_float(flashvid_config, "certv3_visual_novelty_weight", 0.20),
            _cfg_float(flashvid_config, "certv3_visual_curvature_weight", 0.14),
            _cfg_float(flashvid_config, "certv3_visual_event_weight", 0.12),
            _cfg_float(flashvid_config, "certv3_visual_detail_weight", 0.12),
            _cfg_float(flashvid_config, "certv3_visual_component_weight", 0.14),
        ]
        visual_weight_sum = sum(max(0.0, weight) for weight in visual_weights)
        if visual_weight_sum <= 0.0:
            raise ValueError("CertVID V3 visual weights must have a positive sum")
        visual_weights = [max(0.0, weight) / visual_weight_sum for weight in visual_weights]
        visual_quality = _minmax(
            visual_weights[0] * attention
            + visual_weights[1] * novelty
            + visual_weights[2] * curvature
            + visual_weights[3] * event
            + visual_weights[4] * detail
            + visual_weights[5] * component_value,
            dim=0,
        )
        quality = _minmax((1.0 - query_weight) * visual_quality + query_weight * query_score, dim=0)
        event_weights = [
            _cfg_float(flashvid_config, "certv3_event_novelty_weight", 0.34),
            _cfg_float(flashvid_config, "certv3_event_curvature_weight", 0.28),
            _cfg_float(flashvid_config, "certv3_event_frame_weight", 0.18),
            _cfg_float(flashvid_config, "certv3_event_detail_weight", 0.10),
            _cfg_float(flashvid_config, "certv3_event_query_weight", 0.10),
        ]
        event_weight_sum = sum(max(0.0, weight) for weight in event_weights)
        if event_weight_sum <= 0.0:
            raise ValueError("CertVID V3 event weights must have a positive sum")
        event_weights = [max(0.0, weight) / event_weight_sum for weight in event_weights]
        event_score = _minmax(
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
        certificate_kwargs = {
            "budget": budget,
            "quality": quality,
            "event_score": event_score,
            "frame_ids": frame_ids,
            "temporal_ids": temporal_ids,
            "spatial_ids": spatial_ids,
            "query_relevance": query_relevance,
            "atom_weights": atom_weights,
            "query_confidence": query_confidence,
            "frame_count": frame_count,
            "temporal_count": temporal_count,
            "spatial_count": spatial_count,
            "frame_coverage_ratio": _cfg_float(
                flashvid_config,
                "certv3_frame_coverage_ratio",
                1.0,
            ),
            "cell_coverage_ratio": _cfg_float(
                flashvid_config,
                "certv3_cell_coverage_ratio",
                0.50,
            ),
            "query_threshold": _cfg_float(
                flashvid_config,
                "certv3_query_threshold",
                0.10,
            ),
            "query_per_atom": _cfg_int(
                flashvid_config,
                "certv3_query_per_atom",
                1,
            ),
            "use_spatiotemporal": use_spatiotemporal_certificates,
        }
        if certificate_limit == 0 and use_spatiotemporal_certificates:
            # A zero cap always discards every requested certificate. Build the
            # capped request set once and recover the uncapped count exactly.
            mandatory, query_seeds, certificate_stats = (
                _budget_aware_certificates(
                    certificate_ratio=effective_certificate_ratio,
                    **certificate_kwargs,
                )
            )
            original_certificate_count = min(
                budget,
                int(certificate_stats["requested_unique"]),
            )
            certificate_cap_active = True
            if qwen_v3_certificate_policy:
                qwen_certificate_cap_active = True
        else:
            mandatory, query_seeds = _hard_certificates(**certificate_kwargs)
            original_certificate_count = len(mandatory)
            if original_certificate_count > certificate_limit:
                mandatory, query_seeds, certificate_stats = (
                    _budget_aware_certificates(
                        certificate_ratio=effective_certificate_ratio,
                        **certificate_kwargs,
                    )
                )
                certificate_cap_active = True
                if qwen_v3_certificate_policy:
                    qwen_certificate_cap_active = True
        candidate_indices = _candidate_pool(
            budget=budget,
            quality=quality,
            component_ids=component_ids,
            temporal_ids=temporal_ids,
            spatial_ids=spatial_ids,
            query_relevance=query_relevance,
            mandatory=mandatory,
            multiplier=_cfg_float(flashvid_config, "certv3_candidate_multiplier", 2.5),
            use_trajectory=use_trajectory,
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
            structural_weight=_cfg_float(flashvid_config, "certv3_structural_weight", 0.32),
            use_spatiotemporal_design=_cfg_bool(
                flashvid_config,
                "certv3_use_spatiotemporal_design",
                True,
            ),
            whitening_strength=_cfg_float(flashvid_config, "certv3_whitening_strength", 0.50),
            quality_floor=_cfg_float(flashvid_config, "certv3_quality_floor", 0.15),
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

        ridge = _cfg_float(flashvid_config, "certv3_ridge", 0.50)
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
                steps=_cfg_int(flashvid_config, "certv3_swap_steps", 6),
                pool_size=_cfg_int(flashvid_config, "certv3_swap_pool", 24),
                margin=_cfg_float(flashvid_config, "certv3_swap_margin", 1e-4),
            )
            _profile_record(phase_events, "fedorov", 1)

        _profile_record(phase_events, "fusion_plan", 0)
        selected = torch.sort(selected).values
        plan = _build_plan(
            selected=selected,
            metric_features=metric_flat,
            demand_weight=demand_weight,
            attention=attention,
            query_score=query_score,
            temporal_ids=temporal_ids,
            component_ids=component_ids,
            fusion_alpha=_cfg_float(flashvid_config, "certv3_fusion_alpha", 0.12),
            temperature=_cfg_float(flashvid_config, "certv3_assignment_temperature", 0.07),
        )
        _profile_record(phase_events, "fusion_plan", 1)

        _profile_record(phase_events, "fusion_apply", 0)
        if mandatory:
            certificate_indices = torch.tensor(mandatory, dtype=torch.long, device=selected.device)
            plan.fusion_alpha[torch.isin(selected, certificate_indices)] = 0.0
        output = apply_certvid_plan(flat_features, plan)
        _profile_record(phase_events, "fusion_apply", 1)
        candidates = int(candidate_indices.numel())
        components = int(component_sizes.numel())
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
            if bool(
                getattr(
                    flashvid_config,
                    "_capture_visualization_trajectory",
                    False,
                )
            ):
                match_previous = (
                    torch.stack([entry[0] for entry in matches], dim=0)
                    if matches
                    else torch.empty(
                        (0, tokens_per_frame),
                        dtype=torch.long,
                        device=video_features.device,
                    )
                )
                match_mutual = (
                    torch.stack([entry[1] for entry in matches], dim=0)
                    if matches
                    else torch.empty(
                        (0, tokens_per_frame),
                        dtype=torch.bool,
                        device=video_features.device,
                    )
                )
                match_similarity = (
                    torch.stack([entry[2] for entry in matches], dim=0)
                    if matches
                    else torch.empty(
                        (0, tokens_per_frame),
                        dtype=torch.float32,
                        device=video_features.device,
                    )
                )
                analysis_sink.update(
                    {
                        "component_sizes": component_sizes,
                        "component_support": component_value,
                        "frame_event": frame_event,
                        "novelty": novelty,
                        "curvature": curvature,
                        "spatial_coords": coords,
                        "match_previous": match_previous,
                        "match_mutual": match_mutual,
                        "match_similarity": match_similarity,
                        "frame_count": int(frame_count),
                        "tokens_per_frame": int(tokens_per_frame),
                        "grid_height": int(height),
                        "grid_width": int(width),
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
    setattr(flashvid_config, "last_certv3_certificate_count", float(len(mandatory)))
    setattr(
        flashvid_config,
        "last_certv3_original_certificate_count",
        float(original_certificate_count),
    )
    setattr(
        flashvid_config,
        "last_certv3_certificate_budget_ratio",
        float(effective_certificate_ratio),
    )
    setattr(
        flashvid_config,
        "last_certv3_certificate_limit",
        float(certificate_limit),
    )
    setattr(
        flashvid_config,
        "last_certv3_certificate_cap_active",
        bool(certificate_cap_active),
    )
    setattr(
        flashvid_config,
        "last_certv3_qwen_certificate_cap_active",
        bool(qwen_certificate_cap_active),
    )
    setattr(
        flashvid_config,
        "last_certv3_qwen_certificate_limit",
        float(qwen_certificate_limit),
    )
    free_dopt_slots = max(0, int(budget) - len(mandatory))
    certificate_pressure = len(mandatory) / float(max(1, budget))
    dopt_slot_ratio = free_dopt_slots / float(max(1, budget))
    setattr(
        flashvid_config,
        "last_certv3_certificate_pressure",
        float(certificate_pressure),
    )
    setattr(
        flashvid_config,
        "last_certv3_free_dopt_slots",
        float(free_dopt_slots),
    )
    setattr(
        flashvid_config,
        "last_certv3_dopt_slot_ratio",
        float(dopt_slot_ratio),
    )
    setattr(flashvid_config, "last_certv3_query_seed_count", float(len(query_seeds)))
    setattr(flashvid_config, "last_certv3_query_confidence", float(query_confidence))
    setattr(flashvid_config, "last_certv3_swap_count", float(swaps))
    setattr(flashvid_config, "last_certv3_logdet", float(logdet))
    diagnostics = {
        "identity": bool(budget >= total_tokens),
        "retention_ratio": float(
            _cfg_float(flashvid_config, "retention_ratio", 0.10)
        ),
        "effective_outer_ratio": float(ratio),
        "expansion": float(_cfg_float(flashvid_config, "expansion", 1.0)),
        "pruning_layer": int(_cfg_int(flashvid_config, "pruning_layer", 0)),
        "llm_retention_ratio": float(
            _cfg_float(flashvid_config, "llm_retention_ratio", 1.0)
        ),
        "fusion_alpha": float(
            _cfg_float(flashvid_config, "certv3_fusion_alpha", 0.12)
        ),
        "raw_tokens": int(total_tokens),
        "target_tokens": int(budget),
        "output_tokens": int(output.shape[0]),
        "backbone": backbone,
        "certificate_count": int(len(mandatory)),
        "original_certificate_count": int(original_certificate_count),
        "certificate_pressure": float(certificate_pressure),
        "original_certificate_pressure": float(
            original_certificate_count / float(max(1, budget))
        ),
        "free_dopt_slots": int(free_dopt_slots),
        "dopt_slot_ratio": float(dopt_slot_ratio),
        "certificate_budget_ratio": float(effective_certificate_ratio),
        "configured_certificate_budget_ratio": float(certificate_ratio),
        "certificate_limit": int(certificate_limit),
        "certificate_cap_active": bool(certificate_cap_active),
        "qwen_certificate_policy": bool(qwen_v3_certificate_policy),
        "qwen_certificate_cap_active": bool(qwen_certificate_cap_active),
        "qwen_certificate_budget_ratio": float(qwen_certificate_ratio),
        "qwen_certificate_limit": int(qwen_certificate_limit),
        "candidate_count": int(candidates),
        "component_count": int(components),
        "query_seed_count": int(len(query_seeds)),
        "query_confidence": float(query_confidence),
        "swap_count": int(swaps),
        "logdet": float(logdet),
        "selection_objective": selection_objective,
        "use_spatiotemporal_certificates": bool(use_spatiotemporal_certificates),
        "use_trajectory": bool(use_trajectory),
        "use_query": bool(use_query),
    }
    diagnostics.update(
        {
            f"certificate_{key}": int(value)
            for key, value in certificate_stats.items()
        }
    )
    if qwen_v3_certificate_policy:
        diagnostics.update(
            {
                f"qwen_certificate_{key}": int(value)
                for key, value in certificate_stats.items()
            }
        )
    setattr(flashvid_config, "last_certv3_diagnostics", diagnostics)
    _write_certv3_diagnostics(flashvid_config, diagnostics)
    return output, plan.anchor_indices
