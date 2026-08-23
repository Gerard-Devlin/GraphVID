"""Alternative fixed-budget selectors used only by CertVID V3 ablations."""

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
    design: torch.Tensor,
    candidates: torch.Tensor,
    mandatory: list[int],
    budget: int,
) -> tuple[torch.Tensor, list[int]]:
    rows = design[candidates].float()
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
    design: torch.Tensor,
    candidates: torch.Tensor,
    mandatory: list[int],
    budget: int,
    quality: torch.Tensor,
) -> torch.Tensor:
    """Greedy fixed-cardinality DPP MAP via conditional kernel diagonals."""
    del quality
    rows, mandatory_columns = _validate_inputs(design, candidates, mandatory, budget)
    count = int(rows.shape[0])
    budget = int(budget)
    if budget == 0:
        return candidates[:0]

    # The V3 design rows already contain the quality mass. This kernel therefore
    # changes only the set objective, not any upstream feature or weight.
    residual_diagonal = rows.square().sum(dim=1).clamp_min(1e-12)
    factors = torch.zeros((budget, count), dtype=torch.float32, device=rows.device)
    active = torch.ones(count, dtype=torch.bool, device=rows.device)
    selected_columns: list[torch.Tensor] = []

    def add(column: torch.Tensor) -> None:
        step = len(selected_columns)
        row = rows[column]
        cross = rows @ row
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
    design: torch.Tensor,
    candidates: torch.Tensor,
    mandatory: list[int],
    budget: int,
    quality: torch.Tensor,
) -> torch.Tensor:
    """Cosine farthest-first traversal seeded by the strongest quality token."""
    rows, mandatory_columns = _validate_inputs(design, candidates, mandatory, budget)
    unit_rows = F.normalize(rows, p=2, dim=-1, eps=1e-6)
    candidate_quality = quality[candidates].float()
    return _max_min_select(
        rows=unit_rows,
        candidates=candidates,
        mandatory_columns=mandatory_columns,
        budget=budget,
        seed_score=candidate_quality,
        distance_to=lambda column: (1.0 - unit_rows @ unit_rows[column]).clamp_min(0.0),
    )


def _fps_kcenter_select(
    *,
    design: torch.Tensor,
    candidates: torch.Tensor,
    mandatory: list[int],
    budget: int,
    quality: torch.Tensor,
) -> torch.Tensor:
    """Deterministic Euclidean FPS/k-center with a centroid-farthest seed."""
    del quality
    rows, mandatory_columns = _validate_inputs(design, candidates, mandatory, budget)
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
    design: torch.Tensor,
    quality: torch.Tensor,
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
        design=design,
        quality=quality,
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
