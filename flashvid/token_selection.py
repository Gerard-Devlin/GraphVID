from typing import Tuple

from enum import Enum
import torch


class TokenSelectionMethod(str, Enum):
    ATTN = "attn"
    DIV = "div"
    ADTS = "attn_div"
    ADTS_v2 = "attn_div_v2"
    ADTS_STABLE = "attn_div_stable"


def pairwise_cosine_distances(image_features: torch.Tensor) -> torch.Tensor:
    """Calculate pairwise cosine distances for a batch of feature vectors.

    Args:
        image_features (torch.Tensor): Visual features, of shape (bsz, num_visual_tokens, feat_dim)

    Returns:
        torch.Tensor: Pairwise cosine distances, of shape (bsz, num_visual_tokens, num_visual_tokens)
    """
    normed_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    similarities = torch.bmm(normed_features, normed_features.transpose(-1, -2))
    return 1.0 - similarities


def _minmax_normalize(scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    min_val = scores.amin(dim=dim, keepdim=True)
    max_val = scores.amax(dim=dim, keepdim=True)
    return (scores - min_val) / (max_val - min_val + 1e-6)


def attn_div_based_token_selection(
    features: torch.Tensor,
    cls_attention: torch.Tensor,
    num_retained_tokens: int,
) -> torch.Tensor:
    """Select visual tokens based on attention and diversity.

    Args:
        features (torch.Tensor): Visual features, of shape (bsz, num_visual_tokens, feat_dim)
        cls_attention (torch.Tensor): [CLS] attention, of shape (bsz, num_visual_tokens)
        num_retained_tokens (int): Number of tokens to retain

    Returns:
        torch.Tensor: Pruned features, of shape (bsz, num_retained_tokens, feat_dim)
    """
    # Convert features and cls_attention to torch.float32
    original_features = features
    features = features.float()
    cls_attention = cls_attention.float() * 1e6  # Scale attention to avoid numerical issues
    bsz, num_visual_tokens, feat_dim = features.shape
    dist_matrix = pairwise_cosine_distances(features)

    # Calibrate by [CLS] attention.
    dist_matrix = dist_matrix * cls_attention.unsqueeze(1)  # (bsz, num_visual_tokens, num_visual_tokens)

    # Initialize keeping indices.
    keep_indices = torch.zeros(bsz, num_retained_tokens, dtype=torch.long, device=features.device)  # (bsz, num_retained_tokens)

    # select the first token.
    min_dist = torch.topk(dist_matrix, k=2, dim=1, largest=False).values[:, 1, :]  # (bsz, num_visual_tokens)
    keep_indices[:, 0] = torch.argmax(min_dist, dim=-1)  # (bsz,)

    # Select the rest of the tokens.
    for i in range(1, num_retained_tokens):
        # Get the distances to the already selected tokens.
        dist_sub_matrix = torch.gather(
            dist_matrix,
            dim=1,
            index=keep_indices[:, :i].unsqueeze(-1).expand(-1, -1, num_visual_tokens),
        )
        min_dist = torch.min(dist_sub_matrix, dim=1).values
        # Prevent select the same token again.
        min_dist.scatter_(1, keep_indices[:, :i], -1)  # (bsz, num_visual_tokens)
        keep_indices[:, i] = torch.argmax(min_dist, dim=-1)

    keep_indices = keep_indices.sort().values
    selected_features = torch.gather(original_features, dim=1, index=keep_indices.unsqueeze(-1).expand(-1, -1, feat_dim))  # (bsz, num_retained_tokens, feat_dim)

    return selected_features, keep_indices


def attn_div_v2_based_token_selection(
    features: torch.Tensor,
    cls_attention: torch.Tensor,
    num_retained_tokens: int,
) -> torch.Tensor:
    """Select visual tokens based on attention and diversity.

    Args:
        features (torch.Tensor): Visual features, of shape (bsz, num_visual_tokens, feat_dim)
        cls_attention (torch.Tensor): [CLS] attention, of shape (bsz, num_visual_tokens)
        num_retained_tokens (int): Number of tokens to retain

    Returns:
        torch.Tensor: Pruned features, of shape (bsz, num_retained_tokens, feat_dim)
    """
    original_features = features
    features = features.float()
    pooled_features = features.mean(1) # (num_frames, feat_dim)
    global_cls_attention = cls_attention.float() * 1e6  # Scale attention to avoid numerical issues
    bsz, num_visual_tokens, feat_dim = features.shape
    dist_matrix = pairwise_cosine_distances(features)

    # (1) [CLS] attention calibration term (bsz, 1, num_visual_tokens).
    calibration_term1 = global_cls_attention.unsqueeze(1)
    # (2) Event relevance calibration term (bsz, 1, num_visual_tokens).
    local_cls_attention = torch.einsum("b n d, c d -> b c n", features, pooled_features).mean(1)
    calibration_term2 = local_cls_attention.unsqueeze(1)
    # Calibrate distance matrix by [cls] attention and event relevance (bsz, num_visual_tokens, num_visual_tokens)
    dist_matrix = dist_matrix * calibration_term1 * calibration_term2

    # Initialize keeping indices (bsz, num_retained_tokens).
    keep_indices = torch.zeros(bsz, num_retained_tokens, dtype=torch.long, device=features.device)

    # select the first token.
    min_dist = torch.topk(dist_matrix, k=2, dim=1, largest=False).values[:, 1, :]  # (bsz, num_visual_tokens)
    keep_indices[:, 0] = torch.argmax(min_dist, dim=-1)  # (bsz,)

    # Select the rest of the tokens.
    for i in range(1, num_retained_tokens):
        # Get the distances to the already selected tokens.
        dist_sub_matrix = torch.gather(
            dist_matrix,
            dim=1,
            index=keep_indices[:, :i].unsqueeze(-1).expand(-1, -1, num_visual_tokens),
        )
        min_dist = torch.min(dist_sub_matrix, dim=1).values
        keep_indices[:, i] = torch.argmax(min_dist, dim=-1)

    keep_indices = keep_indices.sort().values
    selected_features = torch.gather(original_features, dim=1, index=keep_indices.unsqueeze(-1).expand(-1, -1, feat_dim))  # (bsz, num_retained_tokens, feat_dim)

    return selected_features, keep_indices


def attn_div_stable_based_token_selection(
    features: torch.Tensor,
    cls_attention: torch.Tensor,
    num_retained_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """A numerically stable ADTS variant with explicit duplicate suppression.

    It keeps the original ADTS idea, but calibrates candidate diversity with
    normalized attention and normalized frame-level relevance instead of raw
    dot-product magnitudes. The output remains raw visual tokens.
    """
    original_features = features
    features = features.float()
    bsz, num_visual_tokens, feat_dim = features.shape
    num_retained_tokens = min(max(1, int(num_retained_tokens)), num_visual_tokens)

    normed = features / features.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-6)
    dist_matrix = 1.0 - torch.bmm(normed, normed.transpose(-1, -2))

    attn = torch.nan_to_num(cls_attention.float(), nan=0.0, posinf=0.0, neginf=0.0)
    attn = _minmax_normalize(attn, dim=-1)

    frame_proto = normed.mean(dim=1)
    frame_proto = frame_proto / frame_proto.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-6)
    event_relevance = torch.bmm(normed, frame_proto.unsqueeze(-1)).squeeze(-1)
    event_relevance = _minmax_normalize(event_relevance, dim=-1)

    candidate_weight = 0.15 + 0.85 * (0.65 * attn + 0.35 * event_relevance)
    weighted_dist = dist_matrix * candidate_weight.unsqueeze(1)

    keep_indices = torch.zeros(bsz, num_retained_tokens, dtype=torch.long, device=features.device)
    nearest = torch.topk(weighted_dist, k=min(2, num_visual_tokens), dim=1, largest=False).values[:, -1, :]
    keep_indices[:, 0] = torch.argmax(nearest + 0.05 * candidate_weight, dim=-1)

    for i in range(1, num_retained_tokens):
        dist_sub_matrix = torch.gather(
            weighted_dist,
            dim=1,
            index=keep_indices[:, :i].unsqueeze(-1).expand(-1, -1, num_visual_tokens),
        )
        score = torch.min(dist_sub_matrix, dim=1).values + 0.05 * candidate_weight
        score.scatter_(1, keep_indices[:, :i], -1.0)
        keep_indices[:, i] = torch.argmax(score, dim=-1)

    keep_indices = keep_indices.sort().values
    selected_features = torch.gather(
        original_features,
        dim=1,
        index=keep_indices.unsqueeze(-1).expand(-1, -1, feat_dim),
    )
    return selected_features, keep_indices


def attn_based_token_selection(
    features: torch.Tensor,
    cls_attention: torch.Tensor,
    num_retained_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select visual tokens based on attention.

    Args:
        features (torch.Tensor): Visual features, of shape (bsz, num_visual_tokens, feat_dim)
        cls_attention (torch.Tensor): [CLS] attention, of shape (bsz, num_visual_tokens)
        num_retained_tokens (int): Number of tokens to retain

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Pruned features and their indices.
    """
    bsz, num_visual_tokens, feat_dim = features.shape
    topk_indices = torch.topk(cls_attention, k=num_retained_tokens, dim=-1).indices.sort().values
    selected_features = torch.gather(features, dim=1, index=topk_indices.unsqueeze(-1).expand(-1, -1, feat_dim))
    return selected_features, topk_indices


def div_based_token_selection(
    features: torch.Tensor,
    num_retained_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select visual tokens based on diversity.

    Args:
        features (torch.Tensor): Visual features, of shape (bsz, num_visual_tokens, feat_dim)
        num_retained_tokens (int): Number of tokens to retain

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Pruned features and their indices.
    """
    original_features = features
    features = features.float()
    bsz, num_visual_tokens, feat_dim = features.shape
    dist_matrix = pairwise_cosine_distances(features)
    min_dist = torch.topk(dist_matrix, k=2, dim=1, largest=False).values[:, 1, :]  # (bsz, num_visual_tokens)

    keep_indices = torch.zeros(bsz, num_retained_tokens, dtype=torch.long, device=features.device)  # (bsz, num_retained_tokens)
    keep_indices[:, 0] = torch.argmax(min_dist, dim=-1)  # (bsz,)

    for i in range(1, num_retained_tokens):
        dist_sub_matrix = torch.gather(
            dist_matrix,
            dim=1,
            index=keep_indices[:, :i].unsqueeze(-1).expand(-1, -1, num_visual_tokens),
        )
        min_dist = torch.min(dist_sub_matrix, dim=1).values
        # Prevent select the same token again.
        min_dist.scatter_(1, keep_indices[:, :i], -1)  # (bsz, num_visual_tokens)
        keep_indices[:, i] = torch.argmax(min_dist, dim=-1)

    keep_indices = keep_indices.sort().values
    selected_features = torch.gather(original_features, dim=1, index=keep_indices.unsqueeze(-1).expand(-1, -1, feat_dim))  # (bsz, num_retained_tokens, feat_dim)

    return selected_features, keep_indices
