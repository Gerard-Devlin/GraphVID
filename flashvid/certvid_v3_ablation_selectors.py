"""Fixed-budget alternatives to CertVID's D-optimal selector."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _rows(features: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(features[candidates].float(), nan=0.0, posinf=0.0, neginf=0.0)


def _kdpp_map(features: torch.Tensor, candidates: torch.Tensor, budget: int) -> torch.Tensor:
    """Greedy MAP for a linear k-DPP on the shared CertVID design."""
    rows = _rows(features, candidates)
    residual = torch.ones(rows.shape[0], device=rows.device)
    selected: list[torch.Tensor] = []
    basis: list[torch.Tensor] = []
    for _ in range(budget):
        column = torch.argmax(residual)
        selected.append(column)
        vector = rows[column]
        if basis:
            matrix = torch.stack(basis)
            vector = vector - (matrix @ vector) @ matrix
        vector = F.normalize(vector, dim=0, eps=1e-6)
        basis.append(vector)
        residual.sub_((rows @ vector).square()).clamp_(min=0.0)
        residual[torch.stack(selected)] = -1.0
    return candidates[torch.stack(selected)]


def _max_min(
    rows: torch.Tensor,
    candidates: torch.Tensor,
    budget: int,
    first: torch.Tensor,
) -> torch.Tensor:
    selected = [first]
    distance = (rows - rows[first]).square().sum(dim=1)
    distance[first] = -1.0
    while len(selected) < budget:
        column = torch.argmax(distance)
        selected.append(column)
        distance = torch.minimum(distance, (rows - rows[column]).square().sum(dim=1))
        distance[torch.stack(selected)] = -1.0
    return candidates[torch.stack(selected)]


def _farthest_first(
    features: torch.Tensor,
    candidates: torch.Tensor,
    budget: int,
) -> torch.Tensor:
    """Cosine farthest-first on the shared CertVID design."""
    rows = F.normalize(_rows(features, candidates), dim=-1, eps=1e-6)
    first = torch.zeros((), dtype=torch.long, device=rows.device)
    return _max_min(rows, candidates, budget, first)


def _fps_kcenter(
    features: torch.Tensor,
    candidates: torch.Tensor,
    budget: int,
) -> torch.Tensor:
    """Euclidean farthest-point sampling on the shared CertVID design."""
    rows = _rows(features, candidates)
    first = torch.argmax((rows - rows.mean(dim=0)).square().sum(dim=1))
    return _max_min(rows, candidates, budget, first)


_SELECTORS = {
    "k_dpp_map": _kdpp_map,
    "farthest_first": _farthest_first,
    "fps_kcenter": _fps_kcenter,
}


def select_ablation_objective(
    objective: str,
    *,
    features: torch.Tensor,
    candidates: torch.Tensor,
    mandatory: list[int],
    budget: int,
) -> torch.Tensor:
    """Replace only D-optimal selection while preserving the shared pipeline."""
    if budget < 1 or budget > candidates.numel():
        raise ValueError(f"invalid selector budget {budget}/{candidates.numel()}")
    mandatory = list(dict.fromkeys(int(token) for token in mandatory))
    if len(mandatory) > budget:
        raise ValueError(f"mandatory set exceeds selector budget: {len(mandatory)}/{budget}")
    mandatory_tensor = torch.tensor(mandatory, dtype=torch.long, device=candidates.device)
    if mandatory and not bool(torch.isin(mandatory_tensor, candidates).all()):
        raise ValueError("mandatory tokens must belong to the candidate pool")
    remaining_candidates = (
        candidates[~torch.isin(candidates, mandatory_tensor)]
        if mandatory
        else candidates
    )
    remaining_budget = budget - len(mandatory)
    if remaining_budget == 0:
        selected = torch.empty(0, dtype=torch.long, device=candidates.device)
    else:
        try:
            selected = _SELECTORS[objective](features, remaining_candidates, remaining_budget)
        except KeyError as error:
            raise ValueError(f"unknown ablation selector: {objective!r}") from error
    if mandatory:
        selected = torch.cat((mandatory_tensor, selected))
    if selected.numel() != budget or torch.unique(selected).numel() != budget:
        raise RuntimeError(f"{objective} failed its exact unique-token budget")
    return selected
