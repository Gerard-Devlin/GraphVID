"""Small, independent fixed-budget selectors for CertVID ablations only."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _rows(features: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(features[candidates].float(), nan=0.0, posinf=0.0, neginf=0.0)


def _kdpp_map(features: torch.Tensor, candidates: torch.Tensor, budget: int) -> torch.Tensor:
    """Greedy MAP for a linear k-DPP on normalized raw token embeddings."""
    rows = F.normalize(_rows(features, candidates), dim=-1, eps=1e-6)
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
        residual[column] = -1.0
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
    """Cosine farthest-first with the first token as its deterministic seed."""
    rows = F.normalize(_rows(features, candidates), dim=-1, eps=1e-6)
    first = torch.zeros((), dtype=torch.long, device=rows.device)
    return _max_min(rows, candidates, budget, first)


def _fps_kcenter(
    features: torch.Tensor,
    candidates: torch.Tensor,
    budget: int,
) -> torch.Tensor:
    """Euclidean farthest-point sampling on unnormalized raw token embeddings."""
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
    """Select raw tokens without any CertVID quality or design-space machinery."""
    if mandatory:
        raise ValueError(f"pure {objective} ablation does not accept certificates")
    if budget < 1 or budget > candidates.numel():
        raise ValueError(f"invalid selector budget {budget}/{candidates.numel()}")
    try:
        selected = _SELECTORS[objective](features, candidates, int(budget))
    except KeyError as error:
        raise ValueError(f"unknown ablation selector: {objective!r}") from error
    if selected.numel() != budget or torch.unique(selected).numel() != budget:
        raise RuntimeError(f"{objective} failed its exact unique-token budget")
    return selected
