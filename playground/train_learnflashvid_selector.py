from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flashvid.learned_selector import LEARN_FEATURE_NAMES, LearnedTokenSelector


class TeacherTokenDataset(Dataset):
    def __init__(self, teacher_dir: str | Path, *, max_files: int = 0):
        root = Path(teacher_dir)
        files = sorted(root.rglob("*.pt"))
        if max_files and max_files > 0:
            files = files[: int(max_files)]
        if not files:
            raise FileNotFoundError(f"no .pt teacher files found under {root}")

        features = []
        labels = []
        for path in files:
            payload = torch.load(str(path), map_location="cpu")
            x = payload.get("features")
            y = payload.get("labels")
            if x is None or y is None:
                continue
            x = torch.as_tensor(x, dtype=torch.float32)
            y = torch.as_tensor(y, dtype=torch.float32).view(-1)
            if x.ndim != 2 or x.shape[0] != y.shape[0]:
                continue
            features.append(x)
            labels.append(y)
        if not features:
            raise RuntimeError(f"teacher files under {root} did not contain usable features/labels")

        self.features = torch.cat(features, dim=0).float()
        self.labels = torch.cat(labels, dim=0).float()
        self.num_files = len(files)

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def _sample_ranking_loss(logits: torch.Tensor, labels: torch.Tensor, max_pairs: int) -> torch.Tensor:
    pos = torch.where(labels > 0.5)[0]
    neg = torch.where(labels <= 0.5)[0]
    if pos.numel() == 0 or neg.numel() == 0 or max_pairs <= 0:
        return logits.new_tensor(0.0)
    count = min(int(max_pairs), int(pos.numel()), int(neg.numel()))
    pos_idx = pos[torch.randint(0, pos.numel(), (count,), device=logits.device)]
    neg_idx = neg[torch.randint(0, neg.numel(), (count,), device=logits.device)]
    # Pairwise soft margin: teacher-kept tokens should score higher than dropped tokens.
    return F.softplus(-(logits[pos_idx] - logits[neg_idx])).mean()


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = TeacherTokenDataset(args.teacher_dir, max_files=args.max_files)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_tokens),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = LearnedTokenSelector(
        input_dim=dataset.features.shape[-1],
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
    ).to(device)

    positives = float(dataset.labels.sum().item())
    negatives = float(dataset.labels.numel() - positives)
    pos_weight = torch.tensor([max(1.0, negatives / max(1.0, positives))], dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    print(
        json.dumps(
            {
                "teacher_dir": str(args.teacher_dir),
                "num_files": dataset.num_files,
                "num_tokens": len(dataset),
                "positive_ratio": positives / max(1.0, positives + negatives),
                "pos_weight": float(pos_weight.item()),
                "feature_names": LEARN_FEATURE_NAMES,
            },
            ensure_ascii=False,
        )
    )

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total_loss = 0.0
        total_bce = 0.0
        total_rank = 0.0
        total = 0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            rank = _sample_ranking_loss(logits, y, int(args.rank_pairs))
            loss = bce + float(args.rank_weight) * rank
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            opt.step()
            batch = int(y.numel())
            total += batch
            total_loss += float(loss.item()) * batch
            total_bce += float(bce.item()) * batch
            total_rank += float(rank.item()) * batch
        denom = max(1, total)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "loss": total_loss / denom,
                    "bce": total_bce / denom,
                    "rank": total_rank / denom,
                    "lr": opt.param_groups[0]["lr"],
                },
                ensure_ascii=False,
            )
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.cpu().state_dict(),
            "input_dim": int(dataset.features.shape[-1]),
            "hidden_dim": int(args.hidden_dim),
            "feature_names": list(LEARN_FEATURE_NAMES),
            "positive_ratio": positives / max(1.0, positives + negatives),
            "num_tokens": int(len(dataset)),
            "num_files": int(dataset.num_files),
        },
        str(out),
    )
    print(f"[saved] {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the tiny LearnFlashVID selector from teacher token labels.")
    parser.add_argument("--teacher_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_tokens", type=int, default=65536)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--rank_weight", type=float, default=0.15)
    parser.add_argument("--rank_pairs", type=int, default=4096)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=42)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
