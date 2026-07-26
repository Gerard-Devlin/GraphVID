from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F


@dataclass
class V3PlusOuterMetadata:
    """V3 anchor attributes aligned with the flattened outer-token sequence."""

    global_indices: torch.Tensor
    frame_ids: torch.Tensor
    temporal_ids: torch.Tensor
    spatial_ids: torch.Tensor
    component_ids: torch.Tensor
    demand_weight: torch.Tensor
    certificate_mask: torch.Tensor
    raw_token_count: int
    frame_count: int
    tokens_per_frame: int


def clear_v3plus_runtime(config: Any) -> None:
    """Release sample-local tensors without deleting completed diagnostics."""
    setattr(config, "_v3plus_outer_metadata", None)
    setattr(config, "_v3plus_inner_attention", None)
    setattr(config, "_v3plus_attention_diagnostics", None)
    setattr(config, "_v3plus_query_prefix_tokens", 0)


def _cfg_float(config: Any, name: str, default: float) -> float:
    try:
        return float(getattr(config, name, default))
    except (TypeError, ValueError):
        return float(default)


def _cfg_int(config: Any, name: str, default: int) -> int:
    try:
        return int(getattr(config, name, default))
    except (TypeError, ValueError):
        return int(default)


def _minmax(values: torch.Tensor) -> torch.Tensor:
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if values.numel() <= 1:
        return torch.ones_like(values)
    lo = values.min()
    hi = values.max()
    if float((hi - lo).abs().item()) < 1e-6:
        return torch.zeros_like(values)
    return ((values - lo) / (hi - lo)).clamp_(0.0, 1.0)


def capture_v3plus_multitext_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    config: Any,
    *,
    scaling: float,
    num_key_value_groups: int,
) -> None:
    """Store a compact multi-text-to-visual relevance vector for inner pruning."""
    if str(getattr(config, "compression_variant", "")).strip().lower() != "certvid_v3plus":
        return
    if str(getattr(config, "v3plus_inner_mode", "structured")).strip().lower() != "structured":
        return

    started = time.perf_counter()
    setattr(config, "_v3plus_inner_attention", None)
    diagnostics = {
        "source": "legacy_fallback",
        "query_rows": 0,
        "specificity_mean": 0.0,
        "specificity_max": 0.0,
        "attention_entropy": 1.0,
        "qk_host_ms": 0.0,
        "fallback_reason": None,
    }
    try:
        if query_states.ndim != 4 or key_states.ndim != 4:
            raise ValueError("unexpected Q/K rank")
        if int(query_states.shape[0]) != 1:
            raise ValueError("V3Plus currently supports batch size 1")

        visual_start = int(getattr(config, "visual_token_start_index", -1))
        visual_length = int(getattr(config, "visual_token_length", 0))
        visual_end = visual_start + visual_length
        prefix_tokens = max(0, int(getattr(config, "_v3plus_query_prefix_tokens", 0)))
        query_start = visual_end + prefix_tokens
        sequence_length = int(query_states.shape[2])
        if visual_start < 0 or visual_length <= 0 or visual_end > sequence_length:
            raise ValueError("invalid visual span")
        if query_start >= sequence_length:
            raise ValueError("no post-visual text query rows")

        repeated_keys = key_states
        if int(repeated_keys.shape[1]) != int(query_states.shape[1]):
            from transformers.models.qwen2.modeling_qwen2 import repeat_kv

            repeated_keys = repeat_kv(repeated_keys, num_key_value_groups)

        text_queries = query_states[:, :, query_start:, :].float()
        visual_keys = repeated_keys[:, :, visual_start:visual_end, :].float()
        logits = torch.matmul(text_queries, visual_keys.transpose(2, 3))
        probabilities = torch.softmax(logits * float(scaling), dim=-1)
        probabilities = probabilities.mean(dim=1)
        if not torch.isfinite(probabilities).all():
            raise ValueError("non-finite multi-text attention")

        visual_count = int(probabilities.shape[-1])
        entropy = -torch.sum(
            probabilities * torch.log(probabilities.clamp_min(1e-12)),
            dim=-1,
        )
        entropy = entropy / max(math.log(max(2, visual_count)), 1e-6)
        specificity = (1.0 - entropy).clamp_(0.0, 1.0)
        finite_rows = torch.isfinite(specificity[0]) & (specificity[0] > 1e-6)
        valid_rows = torch.where(finite_rows)[0]
        if valid_rows.numel() == 0:
            raise ValueError("all text attention rows are non-specific")

        max_rows = max(1, _cfg_int(config, "v3plus_query_rows", 32))
        if valid_rows.numel() > max_rows:
            local_scores = specificity[0, valid_rows]
            chosen = torch.topk(local_scores, k=max_rows, sorted=False).indices
            valid_rows = valid_rows[chosen]
        valid_rows = torch.sort(valid_rows).values

        row_specificity = specificity[0, valid_rows]
        row_weights = row_specificity / row_specificity.sum().clamp_min(1e-8)
        row_probabilities = probabilities[0, valid_rows]
        weighted_mean = torch.sum(row_probabilities * row_weights.unsqueeze(1), dim=0)
        maximum = row_probabilities.max(dim=0).values
        mean_weight = min(
            1.0,
            max(0.0, _cfg_float(config, "v3plus_attention_mean_weight", 0.75)),
        )
        relevance = mean_weight * weighted_mean + (1.0 - mean_weight) * maximum
        relevance = relevance / relevance.sum().clamp_min(1e-8)
        if not torch.isfinite(relevance).all():
            raise ValueError("non-finite aggregated relevance")

        setattr(config, "_v3plus_inner_attention", relevance.detach())
        diagnostics.update(
            {
                "source": "multi_text_qk",
                "query_rows": int(valid_rows.numel()),
                "specificity_mean": float(row_specificity.mean().item()),
                "specificity_max": float(row_specificity.max().item()),
                "attention_entropy": float(entropy[0, valid_rows].mean().item()),
                "fallback_reason": None,
            }
        )
    except (RuntimeError, TypeError, ValueError) as error:
        diagnostics["fallback_reason"] = str(error)
    finally:
        diagnostics["qk_host_ms"] = (time.perf_counter() - started) * 1000.0
        setattr(config, "_v3plus_attention_diagnostics", diagnostics)


def _frame_counts(frame_ids: torch.Tensor, selected_mask: torch.Tensor, frame_count: int) -> list[int]:
    counts = torch.bincount(frame_ids[selected_mask], minlength=frame_count)
    return [int(value) for value in counts.detach().cpu().tolist()]


def _temporal_entropy(counts: list[int]) -> float:
    total = float(sum(counts))
    if total <= 0.0:
        return 0.0
    active = max(1, sum(value > 0 for value in counts))
    if active <= 1:
        return 0.0
    probabilities = torch.tensor(counts, dtype=torch.float64)
    probabilities = probabilities[probabilities > 0] / total
    entropy = -torch.sum(probabilities * torch.log(probabilities)).item()
    return float(entropy / math.log(active))


def _frame_cv(counts: list[int]) -> float:
    if not counts:
        return 0.0
    values = torch.tensor(counts, dtype=torch.float32)
    mean = float(values.mean().item())
    if mean <= 1e-8:
        return 0.0
    return float(values.std(unbiased=False).item() / mean)


def _relation_pairs(
    features: torch.Tensor,
    quality: torch.Tensor,
    attention: torch.Tensor,
    metadata: V3PlusOuterMetadata,
    budget: int,
    config: Any,
) -> list[tuple[int, int]]:
    max_endpoints = int(
        math.floor(
            budget
            * min(1.0, max(0.0, _cfg_float(config, "v3plus_pair_budget_ratio", 0.10)))
        )
    )
    max_endpoints -= max_endpoints % 2
    if max_endpoints < 2:
        return []

    frame_ids = metadata.frame_ids.long()
    temporal_ids = metadata.temporal_ids.long()
    component_ids = metadata.component_ids.long()
    frame_count = int(metadata.frame_count)
    frame_medians = torch.zeros(frame_count, dtype=torch.float32, device=features.device)
    for frame in torch.unique(frame_ids, sorted=True).tolist():
        frame_mask = frame_ids == int(frame)
        frame_medians[int(frame)] = attention[frame_mask].median()

    normalized = F.normalize(features.float(), p=2, dim=-1, eps=1e-6)
    max_bin = max(1, int(temporal_ids.max().item()) if temporal_ids.numel() else 1)
    proposals: list[tuple[float, int, int]] = []
    for component in torch.unique(component_ids, sorted=True).tolist():
        members = torch.where(component_ids == int(component))[0]
        if members.numel() < 2:
            continue
        eligible = attention[members] >= frame_medians[frame_ids[members]]
        members = members[eligible]
        if members.numel() < 2:
            continue
        if members.numel() > 24:
            top = torch.topk(quality[members], k=20, sorted=False).indices
            extrema = torch.stack(
                [
                    torch.argmin(temporal_ids[members]),
                    torch.argmax(temporal_ids[members]),
                ]
            )
            members = torch.unique(torch.cat([members[top], members[extrema]]), sorted=True)

        pair_rows = torch.triu_indices(
            int(members.numel()),
            int(members.numel()),
            offset=1,
            device=features.device,
        )
        left = members[pair_rows[0]]
        right = members[pair_rows[1]]
        spans = (temporal_ids[left] - temporal_ids[right]).abs()
        valid = spans >= 2
        if not bool(valid.any()):
            continue
        left = left[valid]
        right = right[valid]
        spans = spans[valid].float()
        state_difference = 1.0 - torch.sum(normalized[left] * normalized[right], dim=-1)
        pair_attention = 0.5 * (attention[left] + attention[right])
        score = (
            0.45 * pair_attention
            + 0.35 * state_difference.clamp_(0.0, 2.0)
            + 0.20 * (spans / float(max_bin))
        )
        for pair_score, first, second in zip(
            score.detach().cpu().tolist(),
            left.detach().cpu().tolist(),
            right.detach().cpu().tolist(),
        ):
            proposals.append((float(pair_score), int(first), int(second)))

    proposals.sort(key=lambda item: (-item[0], item[1], item[2]))
    protected: set[int] = set()
    accepted: list[tuple[int, int]] = []
    for _, first, second in proposals:
        updated = protected | {first, second}
        if len(updated) > max_endpoints:
            continue
        accepted.append((first, second))
        protected = updated
        if len(protected) >= max_endpoints:
            break
    return accepted


def _allocate_frame_quotas(
    frame_ids: torch.Tensor,
    mandatory: torch.Tensor,
    attention: torch.Tensor,
    demand: torch.Tensor,
    budget: int,
    frame_count: int,
    cap_multiplier: float,
) -> torch.Tensor:
    availability = torch.bincount(frame_ids, minlength=frame_count).long()
    mandatory_counts = torch.bincount(frame_ids[mandatory], minlength=frame_count).long()
    quota = mandatory_counts.clone()
    active = availability > 0

    attention_mass = torch.zeros(frame_count, dtype=torch.float32, device=frame_ids.device)
    demand_mass = torch.zeros_like(attention_mass)
    attention_mass.scatter_add_(0, frame_ids, attention)
    demand_mass.scatter_add_(0, frame_ids, demand)
    attention_mass = attention_mass / attention_mass.sum().clamp_min(1e-8)
    demand_mass = demand_mass / demand_mass.sum().clamp_min(1e-8)
    weights = 0.75 * attention_mass + 0.25 * demand_mass
    weights = torch.where(active, weights + 1e-6, torch.zeros_like(weights))

    active_count = max(1, int(active.sum().item()))
    base_cap = max(1, int(math.ceil(float(budget) / active_count * max(1.0, cap_multiplier))))
    caps = torch.minimum(availability, torch.full_like(availability, base_cap))
    caps = torch.maximum(caps, mandatory_counts)

    while int(caps.sum().item()) < budget:
        room = torch.where(caps < availability)[0]
        if room.numel() == 0:
            break
        order = sorted(
            room.detach().cpu().tolist(),
            key=lambda frame: (-float(weights[frame].item()), int(frame)),
        )
        for frame in order:
            if int(caps.sum().item()) >= budget:
                break
            caps[frame] += 1

    remaining = budget - int(quota.sum().item())
    while remaining > 0:
        room = caps - quota
        eligible = torch.where(room > 0)[0]
        if eligible.numel() == 0:
            break
        eligible_weights = weights[eligible]
        if float(eligible_weights.sum().item()) <= 1e-8:
            eligible_weights = torch.ones_like(eligible_weights)
        raw = remaining * eligible_weights / eligible_weights.sum()
        base = torch.minimum(torch.floor(raw).long(), room[eligible])
        if int(base.sum().item()) > 0:
            quota[eligible] += base
            remaining -= int(base.sum().item())
            if remaining <= 0:
                break

        room = caps - quota
        remainder_frames = torch.where(room > 0)[0]
        if remainder_frames.numel() == 0:
            break
        fraction_by_frame = {
            int(frame): float(fraction)
            for frame, fraction in zip(
                eligible.detach().cpu().tolist(),
                (raw - torch.floor(raw)).detach().cpu().tolist(),
            )
        }
        order = sorted(
            remainder_frames.detach().cpu().tolist(),
            key=lambda frame: (
                -fraction_by_frame.get(int(frame), 0.0),
                -float(weights[frame].item()),
                int(frame),
            ),
        )
        progressed = False
        for frame in order:
            if remaining <= 0:
                break
            if quota[frame] >= caps[frame]:
                continue
            quota[frame] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            break

    return quota


def _write_diagnostics(config: Any, diagnostics: dict[str, Any]) -> None:
    setattr(config, "last_v3plus_diagnostics", diagnostics)
    setattr(config, "last_v3plus_keep_overlap", float(diagnostics.get("legacy_keep_overlap", 1.0)))
    setattr(config, "last_v3plus_query_rows", int(diagnostics.get("query_rows", 0)))
    setattr(config, "last_v3plus_selection_ms", float(diagnostics.get("selection_host_ms", 0.0)))
    template = os.environ.get("V3PLUS_DIAGNOSTICS_JSONL", "").strip()
    if not template:
        return
    rank = os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))
    path = template.replace("{rank}", rank).replace("{pid}", str(os.getpid()))
    if int(os.environ.get("WORLD_SIZE", "1") or "1") > 1 and "{rank}" not in template and "{pid}" not in template:
        root, extension = os.path.splitext(path)
        path = f"{root}.rank{rank}{extension or '.jsonl'}"
    record = dict(diagnostics)
    record["sample_id"] = str(getattr(config, "_debug_sample_id", "unknown"))
    record["question"] = str(getattr(config, "_certvid_query_text", "") or "")
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError as error:
        diagnostics["diagnostics_io_error"] = str(error)


def record_v3plus_fallback(config: Any, reason: str, budget: int) -> None:
    diagnostics = {
        **(getattr(config, "_v3plus_attention_diagnostics", None) or {}),
        "fallback_reason": str(reason),
        "attention_source": "legacy_last_query",
        "inner_tokens": int(budget),
        "legacy_keep_overlap": 1.0,
        "selection_host_ms": 0.0,
    }
    _write_diagnostics(config, diagnostics)


def select_v3plus_inner_tokens(
    visual_features: torch.Tensor,
    legacy_attention: torch.Tensor,
    budget: int,
    config: Any,
) -> torch.Tensor:
    """Select an exact-budget, frame-structured subset of V3 outer tokens."""
    started = time.perf_counter()
    metadata: Optional[V3PlusOuterMetadata] = getattr(config, "_v3plus_outer_metadata", None)
    visual_count = int(visual_features.shape[0])
    budget = min(max(1, int(budget)), visual_count)
    legacy_attention = legacy_attention.float().reshape(-1)
    legacy_keep = torch.topk(legacy_attention, k=budget, dim=-1).indices.sort().values
    attention_info = getattr(config, "_v3plus_attention_diagnostics", None) or {}

    if metadata is None:
        diagnostics = {
            **attention_info,
            "fallback_reason": "missing_outer_metadata",
            "inner_tokens": budget,
            "legacy_keep_overlap": 1.0,
            "selection_host_ms": (time.perf_counter() - started) * 1000.0,
        }
        _write_diagnostics(config, diagnostics)
        return legacy_keep
    tensor_fields = (
        metadata.global_indices,
        metadata.frame_ids,
        metadata.temporal_ids,
        metadata.spatial_ids,
        metadata.component_ids,
        metadata.demand_weight,
        metadata.certificate_mask,
    )
    if any(int(field.numel()) != visual_count for field in tensor_fields):
        diagnostics = {
            **attention_info,
            "fallback_reason": "outer_metadata_shape_mismatch",
            "inner_tokens": budget,
            "legacy_keep_overlap": 1.0,
            "selection_host_ms": (time.perf_counter() - started) * 1000.0,
        }
        _write_diagnostics(config, diagnostics)
        return legacy_keep

    structured_attention = getattr(config, "_v3plus_inner_attention", None)
    attention_source = "multi_text_qk"
    if (
        structured_attention is None
        or int(structured_attention.numel()) != visual_count
        or not bool(torch.isfinite(structured_attention).all())
    ):
        structured_attention = legacy_attention
        attention_source = "legacy_last_query"
    attention = _minmax(structured_attention.reshape(-1))
    demand = _minmax(metadata.demand_weight.to(visual_features.device))
    certificate = metadata.certificate_mask.to(visual_features.device).float()
    attention_weight = _cfg_float(config, "v3plus_attention_weight", 0.70)
    demand_weight = _cfg_float(config, "v3plus_outer_demand_weight", 0.20)
    certificate_weight = _cfg_float(config, "v3plus_certificate_weight", 0.10)
    weight_sum = max(1e-8, attention_weight + demand_weight + certificate_weight)
    quality = (
        attention_weight * attention
        + demand_weight * demand
        + certificate_weight * certificate
    ) / weight_sum

    frame_ids = metadata.frame_ids.to(visual_features.device).long()
    spatial_ids = metadata.spatial_ids.to(visual_features.device).long()
    frame_count = int(metadata.frame_count)
    selected = torch.zeros(visual_count, dtype=torch.bool, device=visual_features.device)

    frame_floor = max(0, _cfg_int(config, "v3plus_frame_floor", 1))
    active_frames = torch.unique(frame_ids, sorted=True)
    if frame_floor > 0 and budget >= int(active_frames.numel()) * frame_floor:
        for frame in active_frames.detach().cpu().tolist():
            members = torch.where(frame_ids == int(frame))[0]
            keep = min(frame_floor, int(members.numel()))
            chosen = torch.topk(quality[members], k=keep, sorted=False).indices
            selected[members[chosen]] = True
    elif frame_floor > 0:
        frame_best: list[tuple[float, int]] = []
        for frame in active_frames.detach().cpu().tolist():
            members = torch.where(frame_ids == int(frame))[0]
            best_local = int(torch.argmax(quality[members]).item())
            token = int(members[best_local].item())
            frame_best.append((float(quality[token].item()), token))
        frame_best.sort(key=lambda item: (-item[0], item[1]))
        for _, token in frame_best[:budget]:
            selected[token] = True

    pairs = _relation_pairs(
        visual_features,
        quality,
        attention,
        metadata,
        budget,
        config,
    )
    accepted_pairs: list[tuple[int, int]] = []
    for first, second in pairs:
        additions = int(not bool(selected[first])) + int(not bool(selected[second]))
        if int(selected.sum().item()) + additions > budget:
            continue
        selected[first] = True
        selected[second] = True
        accepted_pairs.append((first, second))

    quotas = _allocate_frame_quotas(
        frame_ids=frame_ids,
        mandatory=selected,
        attention=attention,
        demand=demand,
        budget=budget,
        frame_count=frame_count,
        cap_multiplier=_cfg_float(config, "v3plus_frame_cap_multiplier", 2.0),
    )
    normalized_features = F.normalize(visual_features.float(), p=2, dim=-1, eps=1e-6)
    diversity_weight = max(0.0, _cfg_float(config, "v3plus_diversity_weight", 0.15))
    spatial_bonus_weight = max(0.0, _cfg_float(config, "v3plus_spatial_bonus", 0.05))

    for frame in active_frames.detach().cpu().tolist():
        members = torch.where(frame_ids == int(frame))[0]
        target = int(quotas[int(frame)].item())
        current_count = int(selected[members].sum().item())
        while current_count < target:
            candidates = members[~selected[members]]
            if candidates.numel() == 0:
                break
            chosen_members = members[selected[members]]
            if chosen_members.numel() > 0:
                redundancy = (
                    normalized_features[candidates]
                    @ normalized_features[chosen_members].transpose(0, 1)
                ).max(dim=1).values
                covered_cells = spatial_ids[chosen_members]
                spatial_bonus = (~torch.isin(spatial_ids[candidates], covered_cells)).float()
            else:
                redundancy = torch.zeros(candidates.numel(), device=visual_features.device)
                spatial_bonus = torch.ones_like(redundancy)
            score = (
                quality[candidates]
                - diversity_weight * redundancy
                + spatial_bonus_weight * spatial_bonus
            )
            selected[candidates[torch.argmax(score)]] = True
            current_count += 1

    selected_count = int(selected.sum().item())
    if selected_count < budget:
        remaining = torch.where(~selected)[0]
        fill = torch.topk(
            quality[remaining],
            k=budget - selected_count,
            sorted=False,
        ).indices
        selected[remaining[fill]] = True
        selected_count = budget
    if selected_count > budget:
        raise RuntimeError("V3Plus mandatory selection exceeded the inner budget")

    keep = torch.where(selected)[0].sort().values
    if keep.numel() != budget or torch.unique(keep).numel() != budget:
        raise RuntimeError("V3Plus produced an invalid inner keep set")

    legacy_mask = torch.zeros(visual_count, dtype=torch.bool, device=visual_features.device)
    legacy_mask[legacy_keep] = True
    structured_counts = _frame_counts(frame_ids, selected, frame_count)
    legacy_counts = _frame_counts(frame_ids, legacy_mask, frame_count)
    outer_counts = [
        int(value)
        for value in torch.bincount(frame_ids, minlength=frame_count).detach().cpu().tolist()
    ]
    certificate_count = int(metadata.certificate_mask.sum().item())
    pair_endpoints = sorted({token for pair in accepted_pairs for token in pair})
    relation_endpoint_survival = (
        float(selected[pair_endpoints].float().mean().item()) if pair_endpoints else 1.0
    )
    demand_total = float(metadata.demand_weight.float().sum().item())
    diagnostics = {
        **attention_info,
        "fallback_reason": None,
        "attention_source": attention_source,
        "inner_tokens": budget,
        "outer_tokens": visual_count,
        "legacy_keep_overlap": float(legacy_mask[keep].float().mean().item()),
        "outer_per_frame": outer_counts,
        "legacy_inner_per_frame": legacy_counts,
        "structured_inner_per_frame": structured_counts,
        "legacy_empty_frames": int(sum(value == 0 for value in legacy_counts)),
        "structured_empty_frames": int(sum(value == 0 for value in structured_counts)),
        "legacy_temporal_entropy": _temporal_entropy(legacy_counts),
        "structured_temporal_entropy": _temporal_entropy(structured_counts),
        "legacy_frame_cv": _frame_cv(legacy_counts),
        "structured_frame_cv": _frame_cv(structured_counts),
        "certificate_survival": (
            float(
                metadata.certificate_mask.to(selected.device)[selected].sum().item()
                / certificate_count
            )
            if certificate_count
            else 1.0
        ),
        "relation_pair_count": len(accepted_pairs),
        "relation_endpoint_survival": relation_endpoint_survival,
        "outer_demand_mass_survival": float(
            metadata.demand_weight.to(selected.device)[selected].sum().item()
            / max(1e-8, demand_total)
        ),
        "selection_host_ms": (time.perf_counter() - started) * 1000.0,
    }
    _write_diagnostics(config, diagnostics)
    return keep
