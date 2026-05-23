from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


LEARN_FEATURE_NAMES = [
    "attn",
    "event",
    "novelty",
    "density",
    "detail",
    "question",
    "frame_pos",
    "y_pos",
    "x_pos",
]


def _minmax_per_frame(values: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    values = values.float()
    lo = values.amin(dim=1, keepdim=True)
    hi = values.amax(dim=1, keepdim=True)
    return ((values - lo) / (hi - lo + eps)).clamp(0.0, 1.0)


def _safe_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), p=2, dim=-1, eps=1e-6)


def _temporal_novelty(normed: torch.Tensor) -> torch.Tensor:
    num_frames, num_tokens, _ = normed.shape
    if num_frames <= 1:
        return torch.zeros((num_frames, num_tokens), dtype=torch.float32, device=normed.device)
    novelty = torch.ones((num_frames, num_tokens), dtype=torch.float32, device=normed.device)
    prev_sim = torch.bmm(normed[1:], normed[:-1].transpose(1, 2)).amax(dim=-1)
    next_sim = torch.bmm(normed[:-1], normed[1:].transpose(1, 2)).amax(dim=-1)
    novelty[1:] = torch.minimum(novelty[1:], 1.0 - prev_sim.float().clamp(-1.0, 1.0))
    novelty[:-1] = torch.minimum(novelty[:-1], 1.0 - next_sim.float().clamp(-1.0, 1.0))
    return _minmax_per_frame(novelty)


def _density_score(normed: torch.Tensor, topk: int = 8) -> torch.Tensor:
    """FastVID-style DPC density score computed frame-wise."""
    num_frames, num_tokens, dim = normed.shape
    out = torch.zeros((num_frames, num_tokens), dtype=torch.float32, device=normed.device)
    if num_tokens <= 1:
        return out
    k = max(1, min(int(topk), num_tokens - 1))
    scale = math.sqrt(max(1, dim))
    for frame_idx in range(num_frames):
        dist = torch.cdist(normed[frame_idx].float(), normed[frame_idx].float()) / scale
        nearest = torch.topk(dist, k=k + 1, dim=-1, largest=False).values[:, 1:]
        density = torch.exp(-(nearest.square().mean(dim=-1)))
        higher = density.unsqueeze(0) > density.unsqueeze(1)
        max_dist = dist.max().clamp_min(1e-6)
        parent_dist = torch.where(higher, dist, torch.full_like(dist, float(max_dist))).amin(dim=-1)
        peak = int(torch.argmax(density).item())
        parent_dist[peak] = max_dist
        out[frame_idx] = density * parent_dist
    return _minmax_per_frame(out)


def _spatial_positions(num_frames: int, num_tokens: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frame_pos = torch.linspace(0.0, 1.0, steps=max(1, num_frames), device=device).view(num_frames, 1).expand(num_frames, num_tokens)
    grid_h = max(1, int(round(math.sqrt(max(1, num_tokens)))))
    grid_w = max(1, int(math.ceil(num_tokens / grid_h)))
    ids = torch.arange(num_tokens, device=device)
    y = (ids // grid_w).float() / max(1, grid_h - 1)
    x = (ids % grid_w).float() / max(1, grid_w - 1)
    return frame_pos, y.view(1, -1).expand(num_frames, -1), x.view(1, -1).expand(num_frames, -1)


def build_scalar_token_features(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    question_features: torch.Tensor | None = None,
    *,
    density_topk: int = 8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Build cheap per-token features for learned QA-aware selection.

    Returns:
        features: [num_frames, num_tokens, len(LEARN_FEATURE_NAMES)]
        aux: named feature maps, each [num_frames, num_tokens]
    """
    num_frames, num_tokens, _ = video_features.shape
    device = video_features.device
    normed = _safe_normalize(video_features)
    attn = _minmax_per_frame(cls_attention.float())

    frame_proto = _safe_normalize(normed.mean(dim=1))
    event = torch.einsum("fnd,gd->fng", normed.float(), frame_proto.float()).mean(dim=-1).clamp_min(0.0)
    event = _minmax_per_frame(event)

    novelty = _temporal_novelty(normed)
    density = _density_score(normed, topk=density_topk)

    video_proto = _safe_normalize(normed.mean(dim=(0, 1), keepdim=True)).view(-1)
    detail = 1.0 - torch.einsum("fnd,d->fn", normed.float(), video_proto.float()).clamp(-1.0, 1.0)
    detail = _minmax_per_frame(detail)

    if question_features is not None and question_features.numel() > 0 and question_features.shape[-1] == video_features.shape[-1]:
        q = _safe_normalize(question_features.reshape(-1, question_features.shape[-1]).mean(dim=0))
        q_rel = torch.einsum("fnd,d->fn", normed.float(), q.float()).clamp(-1.0, 1.0)
        q_rel = _minmax_per_frame(q_rel)
    else:
        q_rel = torch.zeros((num_frames, num_tokens), dtype=torch.float32, device=device)

    frame_pos, y_pos, x_pos = _spatial_positions(num_frames, num_tokens, device)
    parts = [attn, event, novelty, density, detail, q_rel, frame_pos, y_pos, x_pos]
    features = torch.stack(parts, dim=-1).float()
    aux = {
        "attn": attn,
        "event": event,
        "novelty": novelty,
        "density": density,
        "detail": detail,
        "question": q_rel,
        "frame_pos": frame_pos,
        "y_pos": y_pos,
        "x_pos": x_pos,
    }
    return features, aux


def heuristic_learn_score(aux: dict[str, torch.Tensor], *, q_weight: float = 0.20) -> torch.Tensor:
    score = (
        0.36 * aux["attn"]
        + 0.24 * aux["event"]
        + 0.16 * aux["novelty"]
        + 0.14 * aux["density"]
        + 0.10 * aux["detail"]
        + float(q_weight) * aux["question"]
    )
    return _minmax_per_frame(score)


class LearnedTokenSelector(nn.Module):
    def __init__(self, input_dim: int = len(LEARN_FEATURE_NAMES), hidden_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_selector_checkpoint(path: str | Path, device: torch.device) -> LearnedTokenSelector | None:
    if not path:
        return None
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        return None
    payload = torch.load(str(ckpt_path), map_location=device)
    if isinstance(payload, dict) and "model" in payload:
        input_dim = int(payload.get("input_dim", len(LEARN_FEATURE_NAMES)))
        hidden_dim = int(payload.get("hidden_dim", 128))
        state = payload["model"]
    else:
        input_dim = len(LEARN_FEATURE_NAMES)
        hidden_dim = 128
        state = payload
    model = LearnedTokenSelector(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def score_with_selector(
    selector: LearnedTokenSelector | None,
    features: torch.Tensor,
    aux: dict[str, torch.Tensor],
    *,
    blend: float = 0.50,
    q_weight: float = 0.20,
) -> torch.Tensor:
    heuristic = heuristic_learn_score(aux, q_weight=q_weight)
    if selector is None:
        return heuristic
    flat = features.reshape(-1, features.shape[-1]).to(next(selector.parameters()).device)
    logits = selector(flat).reshape(features.shape[:2]).float()
    learned = _minmax_per_frame(logits.to(features.device))
    blend = min(max(float(blend), 0.0), 1.0)
    return _minmax_per_frame(blend * learned + (1.0 - blend) * heuristic)


def topk_per_frame(score: torch.Tensor, k: int, exclude: torch.Tensor | None = None) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    num_frames, num_tokens = score.shape
    k = max(0, min(int(k), num_tokens))
    for frame_idx in range(num_frames):
        if k <= 0:
            out.append(torch.empty((0,), dtype=torch.long, device=score.device))
            continue
        local = score[frame_idx]
        if exclude is not None:
            local = local.masked_fill(exclude[frame_idx], -1e9)
        out.append(torch.topk(local, k=k, dim=-1).indices)
    return out


def make_selection_mask(num_frames: int, num_tokens: int, indices: Iterable[torch.Tensor], device: torch.device) -> torch.Tensor:
    mask = torch.zeros((num_frames, num_tokens), dtype=torch.bool, device=device)
    for frame_idx, idx in enumerate(indices):
        if idx.numel() > 0:
            mask[frame_idx, idx.to(device=device, dtype=torch.long)] = True
    return mask
