"""User-disjoint validation for a short-trajectory multi-scale network.

The 60-second GeoLife protocol contains only five or six observations per
window.  This model therefore uses temporal kernels 1/3/5 instead of a deep
long-sequence encoder, and deliberately excludes absolute location/time.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/geolife_five_rate_fixed_60s_features"
FEATURE_INDICES = (2, 3, 4, 5, 7, 8)
MAX_LENGTH = 6


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def finite(values):
    return np.nan_to_num(
        np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )


def signed_log(values):
    return np.sign(values) * np.log1p(np.abs(values))


def tensorize(feature_samples, trajectory_samples):
    """Return robust relative-motion tensors with shape [N, 12, 6]."""
    output = np.zeros((len(feature_samples), 12, MAX_LENGTH), dtype=np.float32)
    mask = np.zeros((len(feature_samples), MAX_LENGTH), dtype=np.float32)
    for row in range(len(feature_samples)):
        length = min(
            MAX_LENGTH,
            len(finite(feature_samples[row, FEATURE_INDICES[0]]).reshape(-1)),
            len(finite(trajectory_samples[row, 0]).reshape(-1)),
            len(finite(trajectory_samples[row, 1]).reshape(-1)),
        )
        if length == 0:
            continue
        mask[row, :length] = 1.0
        robust_channels = [
            finite(feature_samples[row, index]).reshape(-1)[:length]
            for index in FEATURE_INDICES
        ]
        heading = finite(feature_samples[row, 6]).reshape(-1)[:length]
        heading_radians = np.deg2rad(heading)
        latitude = finite(trajectory_samples[row, 0]).reshape(-1)[:length]
        longitude = finite(trajectory_samples[row, 1]).reshape(-1)[:length]
        mean_latitude = np.deg2rad(float(latitude.mean()))
        relative_y = (latitude - latitude[0]) * 111_320.0
        relative_x = (
            (longitude - longitude[0]) * 111_320.0 * math.cos(mean_latitude)
        )
        step_x = np.diff(relative_x, prepend=relative_x[0])
        step_y = np.diff(relative_y, prepend=relative_y[0])
        robust_channels.extend([relative_x, relative_y, step_x, step_y])
        output[row, :10, :length] = signed_log(
            np.stack(robust_channels, axis=0)
        )
        output[row, 10, :length] = np.sin(heading_radians)
        output[row, 11, :length] = np.cos(heading_radians)
    return output, mask


def load_training():
    base = DATA_ROOT / "train"
    clean_features = np.load(
        base / "clean_multi_feature_segs.npy", allow_pickle=True
    )
    noise_features = np.load(
        base / "noise_multi_feature_segs.npy", allow_pickle=True
    )
    clean_trajectories = np.load(base / "clean_trj_segs.npy", allow_pickle=True)
    noise_trajectories = np.load(base / "noise_trj_segs.npy", allow_pickle=True)
    labels = np.load(base / "clean_multi_feature_seg_labels.npy").astype(np.int64)
    users = np.load(base / "segment_user_ids.npy").astype(np.int64)
    clean, mask = tensorize(clean_features, clean_trajectories)
    noise, noise_mask = tensorize(noise_features, noise_trajectories)
    if not np.array_equal(mask, noise_mask):
        raise ValueError("Clean/noise masks are not aligned")
    return clean, noise, mask, labels, users


def normalization(values, mask):
    valid = mask.astype(bool)
    mean = np.empty((values.shape[1],), dtype=np.float32)
    std = np.empty_like(mean)
    for channel in range(values.shape[1]):
        selected = values[:, channel, :][valid]
        mean[channel] = selected.mean()
        std[channel] = max(float(selected.std()), 1e-4)
    return mean, std


def normalize(values, mask, mean, std):
    result = (values - mean[None, :, None]) / std[None, :, None]
    result *= mask[:, None, :]
    return result.astype(np.float32)


def user_weights(users, power=0.5):
    unique, counts = np.unique(users, return_counts=True)
    table = {int(user): float(count) ** (-power) for user, count in zip(unique, counts)}
    weights = np.asarray([table[int(user)] for user in users], dtype=np.float32)
    return weights / weights.mean()


class TemporalBlock(nn.Module):
    def __init__(self, width: int, kernels=(1, 3, 5), dropout=0.15):
        super().__init__()
        branch_width = width // len(kernels)
        self.branches = nn.ModuleList([
            nn.Conv1d(width, branch_width, kernel, padding=kernel // 2)
            for kernel in kernels
        ])
        self.merge = nn.Conv1d(branch_width * len(kernels), width, 1)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, mask):
        residual = values
        values = torch.cat([branch(values) for branch in self.branches], dim=1)
        values = self.merge(F.gelu(values))
        values = self.norm(values.transpose(1, 2)).transpose(1, 2)
        values = self.dropout(F.gelu(values))
        return (values + residual) * mask


class MultiScaleShortTrajectoryNet(nn.Module):
    def __init__(self, channels=12, width=96, classes=5, kernels=(1, 3, 5),
                 summary_pool=False):
        super().__init__()
        self.summary_pool = summary_pool
        self.input_projection = nn.Conv1d(channels, width, 1)
        self.input_norm = nn.LayerNorm(width)
        self.blocks = nn.ModuleList([
            TemporalBlock(width, kernels=kernels, dropout=0.15),
            TemporalBlock(width, kernels=kernels, dropout=0.15),
        ])
        pooled_width = width * 3 + (channels * 7 if summary_pool else 0)
        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_width),
            nn.Linear(pooled_width, 128),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(128, classes),
        )

    def forward(self, values, mask):
        mask_3d = mask[:, None, :]
        raw_values = values
        values = self.input_projection(values)
        values = self.input_norm(values.transpose(1, 2)).transpose(1, 2)
        values = F.gelu(values) * mask_3d
        for block in self.blocks:
            values = block(values, mask_3d)
        count = mask_3d.sum(dim=2).clamp_min(1.0)
        mean = (values * mask_3d).sum(dim=2) / count
        variance = ((values - mean[:, :, None]) ** 2 * mask_3d).sum(dim=2) / count
        maximum = values.masked_fill(mask_3d == 0, -torch.inf).max(dim=2).values
        pooled = [mean, variance.sqrt(), maximum]
        if self.summary_pool:
            raw_count = mask_3d.sum(dim=2).clamp_min(1.0)
            raw_mean = (raw_values * mask_3d).sum(dim=2) / raw_count
            raw_variance = (
                (raw_values - raw_mean[:, :, None]) ** 2 * mask_3d
            ).sum(dim=2) / raw_count
            raw_minimum = raw_values.masked_fill(
                mask_3d == 0, torch.inf
            ).min(dim=2).values
            raw_maximum = raw_values.masked_fill(
                mask_3d == 0, -torch.inf
            ).max(dim=2).values
            first = raw_values[:, :, 0]
            last_index = (mask.sum(dim=1).long() - 1).clamp_min(0)
            last = raw_values.gather(
                2, last_index[:, None, None].expand(-1, raw_values.shape[1], 1)
            ).squeeze(2)
            pooled.extend([
                raw_mean, raw_variance.sqrt(), raw_minimum, raw_maximum,
                first, last, last - first,
            ])
        return self.classifier(torch.cat(pooled, dim=1))


def evaluate(model, loader, device):
    model.eval()
    predictions, targets = [], []
    total_loss, total = 0.0, 0
    with torch.no_grad():
        for values, mask, labels, _ in loader:
            values, mask, labels = values.to(device), mask.to(device), labels.to(device)
            logits = model(values, mask)
            total_loss += F.cross_entropy(logits, labels).item() * len(labels)
            total += len(labels)
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            targets.append(labels.cpu().numpy())
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    return {
        "loss": total_loss / total,
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
    }


def make_loader(values, mask, labels, weights, batch_size, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(values), torch.from_numpy(mask),
        torch.from_numpy(labels), torch.from_numpy(weights),
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
        pin_memory=True, persistent_workers=False,
    )


def run_fold(clean, noise, mask, labels, users, train_index, val_index,
             fold, args, output_dir):
    seed_everything(args.seed + fold)
    mean, std = normalization(
        np.concatenate([clean[train_index], noise[train_index]]),
        np.concatenate([mask[train_index], mask[train_index]]),
    )
    train_values = normalize(
        np.concatenate([clean[train_index], noise[train_index]]),
        np.concatenate([mask[train_index], mask[train_index]]), mean, std,
    )
    train_mask = np.concatenate([mask[train_index], mask[train_index]])
    train_labels = np.concatenate([labels[train_index], labels[train_index]])
    train_users = np.concatenate([users[train_index], users[train_index]])
    train_weight = user_weights(train_users, args.user_weight_power)

    generator = np.random.default_rng(args.seed + fold)
    use_noise = generator.random(len(val_index)) < 0.5
    val_values = clean[val_index].copy()
    val_values[use_noise] = noise[val_index][use_noise]
    val_values = normalize(val_values, mask[val_index], mean, std)

    train_loader = make_loader(
        train_values, train_mask, train_labels, train_weight,
        args.batch_size, True,
    )
    val_loader = make_loader(
        val_values, mask[val_index], labels[val_index],
        np.ones(len(val_index), dtype=np.float32), args.batch_size, False,
    )
    kernels = (1, 3, 5) if args.model != "single3" else (3,)
    model = MultiScaleShortTrajectoryNet(
        kernels=kernels, summary_pool=args.model == "hybrid"
    ).to(args.device)
    class_counts = np.bincount(train_labels, minlength=5).astype(np.float32)
    class_weight = np.power(class_counts, -args.class_weight_power)
    class_weight /= class_weight.mean()
    class_weight = torch.from_numpy(class_weight).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )
    best, best_state, bad_epochs = None, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for values, batch_mask, batch_labels, weights in train_loader:
            values = values.to(args.device, non_blocking=True)
            batch_mask = batch_mask.to(args.device, non_blocking=True)
            batch_labels = batch_labels.to(args.device, non_blocking=True)
            weights = weights.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(values, batch_mask)
            loss = F.cross_entropy(
                logits, batch_labels, weight=class_weight,
                reduction="none", label_smoothing=0.03
            )
            loss = (loss * weights).sum() / weights.sum()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss in fold {fold}, epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        scheduler.step()
        score = evaluate(model, val_loader, args.device)
        rank = (min(score["accuracy"], score["macro_f1"]),
                score["accuracy"] + score["macro_f1"])
        if best is None or rank > best[0]:
            best = (rank, epoch, score)
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        print(
            f"fold={fold} epoch={epoch:03d} acc={score['accuracy']:.4f} "
            f"f1={score['macro_f1']:.4f} best={best[1]:03d}", flush=True,
        )
        if bad_epochs >= args.patience:
            break
    checkpoint = output_dir / f"fold{fold}_{args.model}_best.pth"
    torch.save({
        "state_dict": best_state, "epoch": best[1], "metrics": best[2],
        "mean": mean, "std": std, "kernels": kernels,
    }, checkpoint)
    return {
        "fold": fold,
        "model": args.model,
        "train_users": int(len(np.unique(users[train_index]))),
        "val_users": int(len(np.unique(users[val_index]))),
        "val_samples": int(len(val_index)),
        "best_epoch": int(best[1]),
        **best[2],
        "checkpoint": str(checkpoint.relative_to(ROOT)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument(
        "--model", choices=("hybrid", "multiscale", "single3"),
        default="hybrid"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--user-weight-power", type=float, default=0.5)
    parser.add_argument("--class-weight-power", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    clean, noise, mask, labels, users = load_training()
    splits = list(StratifiedGroupKFold(
        n_splits=3, shuffle=True, random_state=args.seed
    ).split(clean, labels, groups=users))
    output_dir = ROOT / f"experiments/geolife_multiscale_short_user_cv_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = range(3) if args.all_folds else [args.fold]
    results = [
        run_fold(clean, noise, mask, labels, users, *splits[fold],
                 fold, args, output_dir)
        for fold in folds
    ]
    summary = {
        "protocol": "geolife-60s-short-trajectory-user-cv-v1",
        "seed": args.seed,
        "training": {
            "model": args.model,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "user_weight_power": args.user_weight_power,
            "class_weight_power": args.class_weight_power,
        },
        "input_channels": [
            "distance", "velocity", "acceleration", "jerk",
            "heading_change", "heading_change_rate", "relative_x",
            "relative_y", "step_x", "step_y", "heading_sin", "heading_cos",
        ],
        "folds": results,
        "accuracy_mean": float(np.mean([item["accuracy"] for item in results])),
        "macro_f1_mean": float(np.mean([item["macro_f1"] for item in results])),
    }
    result_path = output_dir / f"results_{args.model}_{'all' if args.all_folds else f'fold{args.fold}'}.json"
    result_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
