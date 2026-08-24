"""Observation-drop consistency training for the V64 sparse relation encoder."""

from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from scripts.train_pair_relation_sparse_v64 import (
    ROOT, PairRelationSparseNet, evaluate, load_split, metrics, normalize,
    normalization, seed_everything, user_weights, weighted,
)


def select_aligned(data, seed, mean, std):
    clean, noisy, mask, labels, users = data
    generator = random.Random(seed)
    choose_noisy = np.fromiter((generator.random() < .5 for _ in range(len(labels))),
                               dtype=bool, count=len(labels))
    values = clean.copy()
    values[choose_noisy] = noisy[choose_noisy]
    return normalize(values, mask, mean, std), mask, labels, users


def point_drop(mask, probability):
    result = mask.clone()
    apply = torch.rand(len(mask), device=mask.device) < probability
    lengths = mask.sum(1).long()
    for row in torch.nonzero(apply & (lengths > 3), as_tuple=False).flatten():
        # Preserve endpoints; remove one genuinely internal observation.
        index = torch.randint(1, int(lengths[row]) - 1, (1,), device=mask.device)
        result[row, index] = 0.0
    return result



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--name", default="pointdrop050")
    parser.add_argument("--drop-probability", type=float, default=.5)
    parser.add_argument("--drop-weight", type=float, default=.35)
    parser.add_argument("--consistency-weight", type=float, default=.10)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(args.seed)
    train, validation = load_split("train"), load_split("val")
    mean, std = normalization(np.concatenate((train[0], train[1])),
                              np.concatenate((train[2], train[2])))
    clean, noisy = (normalize(train[0], train[2], mean, std),
                    normalize(train[1], train[2], mean, std))
    weights = user_weights(train[4], .35)
    data = TensorDataset(torch.from_numpy(clean), torch.from_numpy(noisy),
                         torch.from_numpy(train[2]), torch.from_numpy(train[3]),
                         torch.from_numpy(weights))
    loader = DataLoader(data, batch_size=512, shuffle=True, num_workers=0,
                        pin_memory=True)
    val_x, val_mask, val_y, _ = select_aligned(validation, args.seed, mean, std)
    model = PairRelationSparseNet(width=96, layers=2).to(args.device)
    counts = np.bincount(train[3], minlength=5).astype(np.float32)
    class_weights = counts ** (-.35)
    class_weights /= class_weights.mean()
    class_weights = torch.from_numpy(class_weights).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=3e-4*.05)
    best, state, bad = None, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for clean_x, noisy_x, mask, labels, sample_weight in loader:
            clean_x, noisy_x = clean_x.to(args.device), noisy_x.to(args.device)
            mask, labels = mask.to(args.device), labels.to(args.device)
            sample_weight = sample_weight.to(args.device)
            dropped_mask = point_drop(mask, args.drop_probability)
            optimizer.zero_grad(set_to_none=True)
            clean_logits, clean_embedding = model(clean_x, mask)
            noisy_logits, noisy_embedding = model(noisy_x, mask)
            drop_logits, drop_embedding = model(noisy_x, dropped_mask)
            clean_ce = F.cross_entropy(clean_logits, labels, weight=class_weights,
                                       reduction="none", label_smoothing=.03)
            noisy_ce = F.cross_entropy(noisy_logits, labels, weight=class_weights,
                                       reduction="none", label_smoothing=.03)
            drop_ce = F.cross_entropy(drop_logits, labels, weight=class_weights,
                                      reduction="none", label_smoothing=.03)
            loss = (.5*(weighted(clean_ce, sample_weight)
                         + weighted(noisy_ce, sample_weight))
                    + args.drop_weight*weighted(drop_ce, sample_weight))
            target = .5*(torch.softmax(clean_logits.detach(), 1)
                          + torch.softmax(noisy_logits.detach(), 1))
            consistency = F.kl_div(F.log_softmax(drop_logits, 1), target,
                                   reduction="batchmean")
            alignment = (1.0-F.cosine_similarity(drop_embedding,
                                                  noisy_embedding.detach(), dim=1))
            loss = (loss + args.consistency_weight*consistency
                    + args.consistency_weight*weighted(alignment, sample_weight))
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite V74 loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        scheduler.step()
        current, _ = evaluate(model, val_x, val_mask, val_y, args.device)
        rank = (min(current.values()), sum(current.values()))
        if best is None or rank > best[0]:
            best = (rank, epoch, current)
            state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        print(f"epoch={epoch:03d} acc={current['accuracy']:.5f} "
              f"f1={current['macro_f1']:.5f} best={best[1]:03d}", flush=True)
        if bad >= args.patience:
            break
    model.load_state_dict(state)
    _, expert_val = evaluate(model, val_x, val_mask, val_y, args.device)
    saved = np.load(ROOT/f"experiments/geolife_nohistory_v58_seed{args.seed}/v53_predictions.npz")
    base_val, base_score = saved["validation"], metrics(saved["validation"], val_y)
    trials = []
    for weight in np.linspace(0, .7, 71):
        fused = (1-weight)*base_val + weight*expert_val
        current = metrics(fused, val_y)
        delta = {key: current[key]-base_score[key] for key in base_score}
        trials.append({"weight":float(weight),"metrics":current,"delta":delta})
    trials.sort(key=lambda row:(min(row["delta"].values()),sum(row["delta"].values()),-row["weight"]),reverse=True)
    winner = trials[0]
    result = {"protocol":"V74-current-segment-observation-drop-consistency",
              "settings":vars(args),"best_epoch":best[1],
              "expert_validation":best[2],"v53_validation":base_score,
              "fusion":winner,"test_gate":"both validation deltas >=0.003"}
    if min(winner["delta"].values()) >= .003:
        test_x,test_mask,test_y,_=select_aligned(load_split("test"),args.seed,mean,std)
        _,expert_test=evaluate(model,test_x,test_mask,test_y,args.device)
        fused=(1-winner["weight"])*saved["test"]+winner["weight"]*expert_test
        result["test"]={"v53":metrics(saved["test"],test_y),
                        "expert":metrics(expert_test,test_y),
                        "v74":metrics(fused,test_y)}
        result["test"]["delta"]={key:result["test"]["v74"][key]-result["test"]["v53"][key]
                                    for key in ("accuracy","macro_f1")}
    output=ROOT/f"experiments/geolife_relation_pointdrop_v74_seed{args.seed}/{args.name}"
    output.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":state,"mean":mean,"std":std,"settings":vars(args)},
               output/"model_best.pth")
    (output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
