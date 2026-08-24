"""Canonical fixed-budget baselines used only by CertVID V3 ablations.

These selectors intentionally operate on the raw visual metric features.  They
must not inherit CertVID's quality weighting, whitening, structural axes, or
candidate pool; otherwise they cease to be independent selector baselines.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F


def _mandatory_columns(candidates: torch.Tensor, mandatory: list[int]) -> list[int]:
    if not mandatory:
        return []
    token_to_column = {
        int(token): column
        for column, token in enumerate(candidates.detach().cpu().tolist())
    }
    return list(
        dict.fromkeys(
            token_to_column[int(token)]
            for token in mandatory
            if int(token) in token_to_column
        )
    )


def _validate_inputs(
    features: torch.Tensor,
    candidates: torch.Tensor,
    mandatory: list[int],
    budget: int,
) -> tuple[torch.Tensor, list[int]]:
    rows = torch.nan_to_num(
        features[candidates].float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    budget = int(budget)
    if budget < 0 or int(candidates.numel()) < budget:
        raise RuntimeError(
            f"ablation selector has {int(candidates.numel())} candidates for budget {budget}"
        )
    mandatory_columns = _mandatory_columns(candidates, mandatory)
    if len(mandatory_columns) > budget:
        mandatory_columns = mandatory_columns[:budget]
    return rows, mandatory_columns


def _kdpp_map_select(
    *,
    features: torch.Tensor,
    candidates: torch.Tensor,
    mandatory: list[int],
    budget: int,
) -> torch.Tensor:
    """Greedy k-DPP MAP with an RBF kernel over raw visual features."""
    rows, mandatory_columns = _validate_inputs(features, candidates, mandatory, budget)
    count = int(rows.shape[0])
    budget = int(budget)
    if budget == 0:
        return candidates[:0]

    rows = F.normalize(rows, p=2, dim=-1, eps=1e-6)

    # Estimate the RBF scale from an evenly spaced subset without materializing
    # the full N x N kernel.  A pure RBF kernel has unit diagonal, so singleton
    # gains do not smuggle CertVID quality scores into this baseline.
    sample_count = min(count, 256)
    sample_ids = torch.linspace(
        0,
        count - 1,
        steps=sample_count,
        device=rows.device,
    ).round().long().unique()
    sample = rows[sample_ids]
    sample_distance = torch.cdist(sample, sample, p=2).square()
    positive = sample_distance[sample_distance > 1e-8]
    bandwidth_sq = (
        positive.median()
        if int(positive.numel()) > 0
        else torch.tensor(1.0, device=rows.device)
    ).clamp_min(1e-6)

    residual_diagonal = torch.ones(count, dtype=torch.float32, device=rows.device)
    factors = torch.zeros((budget, count), dtype=torch.float32, device=rows.device)
    active = torch.ones(count, dtype=torch.bool, device=rows.device)
    selected_columns: list[torch.Tensor] = []

    def add(column: torch.Tensor) -> None:
        step = len(selected_columns)
        squared_distance = (rows - rows[column]).square().sum(dim=1)
        cross = torch.exp(-0.5 * squared_distance / bandwidth_sq)
        if step:
            previous = factors[:step]
            cross = cross - previous[:, column] @ previous
        denominator = residual_diagonal[column].clamp_min(1e-12).sqrt()
        update = cross / denominator
        factors[step] = update
        residual_diagonal.sub_(update.square()).clamp_(min=0.0)
        active[column] = False
        residual_diagonal[column] = -1.0
        selected_columns.append(column)

    for column in mandatory_columns:
        add(torch.tensor(column, dtype=torch.long, device=rows.device))

    if not selected_columns:
        # All singleton RBF gains are equal; index order is the deterministic
        # tie-break used by this baseline.
        add(torch.zeros((), dtype=torch.long, device=rows.device))

    while len(selected_columns) < budget:
        gains = residual_diagonal.masked_fill(~active, float("-inf"))
        column = torch.argmax(gains)
        if not bool(torch.isfinite(gains[column])):
            column = torch.argmax(active.to(torch.int8))
        add(column)

    columns = torch.stack(selected_columns)
    return candidates[columns]


def _max_min_select(
    *,
    rows: torch.Tensor,
    candidates: torch.Tensor,
    mandatory_columns: list[int],
    budget: int,
    seed_score: torch.Tensor,
    distance_to: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    count = int(rows.shape[0])
    active = torch.ones(count, dtype=torch.bool, device=rows.device)
    selected: list[torch.Tensor] = []
    min_distance = torch.full(
        (count,),
        float("inf"),
        dtype=torch.float32,
        device=rows.device,
    )

    def add(column: torch.Tensor) -> None:
        min_distance.copy_(torch.minimum(min_distance, distance_to(column)))
        active[column] = False
        min_distance[column] = -1.0
        selected.append(column)

    for column in mandatory_columns:
        add(torch.tensor(column, dtype=torch.long, device=rows.device))

    if not selected and int(budget) > 0:
        first = torch.argmax(seed_score)
        add(first)

    while len(selected) < int(budget):
        score = min_distance.masked_fill(~active, float("-inf"))
        column = torch.argmax(score)
        if not bool(torch.isfinite(score[column])):
            column = torch.argmax(active.to(torch.int8))
        add(column)

    return candidates[torch.stack(selected)] if selected else candidates[:0]


def _farthest_first_select(
    *,
    features: torch.Tensor,
    candidates: torch.Tensor,
    mandatory: list[int],
    budget: int,
) -> torch.Tensor:
    """Cosine farthest-first traversal over raw visual features."""
    rows, mandatory_columns = _validate_inputs(features, candidates, mandatory, budget)
    unit_rows = F.normalize(rows, p=2, dim=-1, eps=1e-6)
    # Classical farthest-first allows an arbitrary initial center.  Candidate
    # order gives a deterministic seed without using CertVID quality.
    seed_score = torch.zeros(int(candidates.numel()), device=rows.device)
    if int(seed_score.numel()) > 0:
        seed_score[0] = 1.0
    return _max_min_select(
        rows=unit_rows,
        candidates=candidates,
        mandatory_columns=mandatory_columns,
        budget=budget,
        seed_score=seed_score,
        distance_to=lambda column: (1.0 - unit_rows @ unit_rows[column]).clamp_min(0.0),
    )


def _fps_kcenter_select(
    *,
    features: torch.Tensor,
    candidates: torch.Tensor,
    mandatory: list[int],
    budget: int,
) -> torch.Tensor:
    """Euclidean FPS/k-center over unnormalized raw visual features."""
    rows, mandatory_columns = _validate_inputs(features, candidates, mandatory, budget)
    centroid = rows.mean(dim=0, keepdim=True)
    seed_score = (rows - centroid).square().sum(dim=1)
    squared_norm = rows.square().sum(dim=1)

    def squared_distance(column: torch.Tensor) -> torch.Tensor:
        return (
            squared_norm
            + squared_norm[column]
            - 2.0 * (rows @ rows[column])
        ).clamp_min(0.0)

    return _max_min_select(
        rows=rows,
        candidates=candidates,
        mandatory_columns=mandatory_columns,
        budget=budget,
        seed_score=seed_score,
        distance_to=squared_distance,
    )


_SELECTORS = {
    "k_dpp_map": _kdpp_map_select,
    "farthest_first": _farthest_first_select,
    "fps_kcenter": _fps_kcenter_select,
}


def select_ablation_objective(
    objective: str,
    *,
    features: torch.Tensor,
    candidates: torch.Tensor,
    mandatory: list[int],
    budget: int,
) -> torch.Tensor:
    """Dispatch a selector that is never used by the default V3 objective."""
    try:
        selector = _SELECTORS[objective]
    except KeyError as error:
        raise ValueError(f"unknown CertVID V3 ablation selector: {objective!r}") from error
    selected = selector(
        features=features,
        candidates=candidates,
        mandatory=mandatory,
        budget=budget,
    )
    if int(selected.numel()) != int(budget):
        raise RuntimeError(
            f"{objective} produced {int(selected.numel())} tokens for budget {int(budget)}"
        )
    if int(torch.unique(selected).numel()) != int(selected.numel()):
        raise RuntimeError(f"{objective} produced duplicate token indices")
    return selected
