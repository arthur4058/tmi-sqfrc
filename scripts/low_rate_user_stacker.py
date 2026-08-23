"""User-disjoint stacking for sparse (60-second) GeoLife trajectories.

The selector fits only on the validation users.  Test evaluation is a separate
command that loads the frozen stacker and never refits on test samples.
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from scripts.evaluate_dual_expert_consensus import metrics, softmax
from scripts.evaluate_short_pair_v24 import user_robust_score
from scripts.evaluate_v24_bias_v25 import calibrate, collect_v24
from scripts.test_statistical_motion_expert import load_v2_test
from scripts.train_multiscale_short_official import load_split as load_sequence_split
from scripts.train_multiscale_short_trajectory import ROOT
from scripts.train_statistical_motion_expert import load_split, load_v2_validation


DEFAULT_PARAMETERS = {
    "C": 0.001,
    "class_power": 0.5,
    "user_power": 0.0,
    "alpha": 0.5,
}


def probability_features(probabilities):
    return np.concatenate(
        [np.log(np.clip(value, 1e-8, 1.0)) for value in probabilities],
        axis=1,
    )


def sample_weights(labels, users, class_power, user_power):
    labels = np.asarray(labels, dtype=np.int64)
    users = np.asarray(users)
    class_count = np.bincount(labels)
    class_weight = (
        labels.size / (len(class_count) * class_count)
    ) ** class_power
    unique_users, user_count = np.unique(users, return_counts=True)
    count_by_user = dict(zip(unique_users.tolist(), user_count.tolist()))
    user_weight = np.asarray([
        labels.size / (len(unique_users) * count_by_user[user])
        for user in users
    ]) ** user_power
    return class_weight[labels] * user_weight


def fit_stacker(features, labels, users, parameters):
    model = LogisticRegression(
        C=float(parameters["C"]), solver="lbfgs", multi_class="multinomial",
        max_iter=1200, random_state=10086,
    )
    model.fit(
        features, labels,
        sample_weight=sample_weights(
            labels, users, parameters["class_power"], parameters["user_power"]
        ),
    )
    return model


def collect_experts(split, seed, device):
    root = ROOT / f"experiments/geolife_short_variants_v23_seed{seed}"
    protocol = json.loads((root / "v24_validation.json").read_text())
    bias_path = root / "v25_validation.json"
    if not bias_path.exists():
        bias_path = root / "v25_frozen_test.json"
    bias = json.loads(bias_path.read_text())["bias"]
    if split == "val":
        v2, labels, temperatures, v2_weights = load_v2_validation(seed)
    elif split == "test":
        logits, labels = load_v2_test(seed, root)
        _, _, temperatures, v2_weights = load_v2_validation(seed)
        v2 = sum(
            weight * softmax(item, temperature)
            for weight, item, temperature in zip(v2_weights, logits, temperatures)
        )
    else:
        raise ValueError(f"unsupported split: {split}")
    short, short_labels = collect_v24(split, seed, protocol, device)
    if not np.array_equal(labels, short_labels):
        raise ValueError("short expert labels are not aligned")
    winner = protocol["winner"]
    weight_a, weight_b = winner["weight_a"], winner["weight_b"]
    v24 = (
        (1.0 - weight_a - weight_b) * v2
        + weight_a * short[0] + weight_b * short[1]
    )
    v25 = calibrate(v24, bias)
    statistical = joblib.load(
        ROOT / f"experiments/geolife_statistical_motion_seed{seed}/model.joblib"
    )
    values, candidate_labels = load_split(split, seed, training=False)
    if not np.array_equal(labels, candidate_labels):
        raise ValueError("statistical expert labels are not aligned")
    statistical_probability = softmax(
        np.log(statistical.predict_proba(values).clip(1e-9)), 0.9
    )
    features = probability_features(
        [v2, short[0], short[1], statistical_probability, v25]
    )
    return features, labels, v25


def oof_probability(features, labels, users, parameters):
    probability = np.zeros((labels.size, int(labels.max()) + 1), dtype=float)
    for heldout_user in np.unique(users):
        train_index = users != heldout_user
        heldout_index = ~train_index
        model = fit_stacker(
            features[train_index], labels[train_index], users[train_index],
            parameters,
        )
        probability[heldout_index] = model.predict_proba(features[heldout_index])
    return probability


def candidate_rank(item):
    score = item["score"]
    complexity = item["C"] + item["class_power"] + item["user_power"]
    return (
        score["users_both_nonnegative"], score["q25_joint_delta"],
        min(score["global_delta"].values()),
        sum(score["global_delta"].values()), -complexity,
    )


def select(seed, device):
    output = ROOT / f"experiments/geolife_user_stacker_seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    features, labels, base = collect_experts("val", seed, device)
    users = load_sequence_split("val")[-1]
    trials = []
    for c_value in (0.001, 0.003, 0.01, 0.03, 0.1):
        for class_power in (0.0, 0.25, 0.5):
            for user_power in (0.0, 0.5, 1.0):
                parameters = {
                    "C": c_value, "class_power": class_power,
                    "user_power": user_power,
                }
                oof = oof_probability(features, labels, users, parameters)
                for alpha in (0.25, 0.5, 0.75, 1.0):
                    fused = (1.0 - alpha) * base + alpha * oof
                    score = user_robust_score(base, fused, labels, users)
                    trials.append({
                        **parameters, "alpha": alpha,
                        "metrics": metrics(fused, labels), "score": score,
                    })
    winner = max(trials, key=candidate_rank)
    model = fit_stacker(features, labels, users, winner)
    joblib.dump(model, output / "stacker.joblib")
    result = {
        "protocol": "geolife-user-loou-regularized-stacker",
        "seed": seed,
        "selection_split": "leave-one-validation-user-out",
        "test_used_during_selection": False,
        "winner": winner,
        "base_validation": metrics(base, labels),
    }
    (output / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def frozen(seed, device, parameters, test):
    output = ROOT / f"experiments/geolife_user_stacker_seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    features, labels, base = collect_experts("val", seed, device)
    users = load_sequence_split("val")[-1]
    oof = oof_probability(features, labels, users, parameters)
    fused = (1.0 - parameters["alpha"]) * base + parameters["alpha"] * oof
    result = {
        "protocol": "geolife-user-loou-regularized-stacker-frozen",
        "seed": seed, "parameters": parameters,
        "validation": {
            "base": metrics(base, labels), "fusion": metrics(fused, labels),
            "robust_score": user_robust_score(base, fused, labels, users),
        },
    }
    if test:
        model = fit_stacker(features, labels, users, parameters)
        test_features, test_labels, test_base = collect_experts(
            "test", seed, device
        )
        stack = model.predict_proba(test_features)
        test_fused = (
            (1.0 - parameters["alpha"]) * test_base
            + parameters["alpha"] * stack
        )
        result["test"] = {
            "base": metrics(test_base, test_labels),
            "fusion": metrics(test_fused, test_labels),
        }
    name = "frozen_test.json" if test else "frozen_validation.json"
    (output / name).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def test_selected(seed, device):
    output = ROOT / f"experiments/geolife_user_stacker_seed{seed}"
    selection = json.loads((output / "selection.json").read_text())
    winner = selection["winner"]
    model = joblib.load(output / "stacker.joblib")
    features, labels, base = collect_experts("test", seed, device)
    stack = model.predict_proba(features)
    alpha = winner["alpha"]
    fused = (1.0 - alpha) * base + alpha * stack
    result = {
        "protocol": selection["protocol"], "seed": seed,
        "test_fit_performed": False,
        "base": metrics(base, labels), "fusion": metrics(fused, labels),
    }
    result["delta"] = {
        name: result["fusion"][name] - result["base"][name]
        for name in ("accuracy", "macro_f1")
    }
    (output / "test.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("select", "test", "frozen"))
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test-frozen", action="store_true")
    args = parser.parse_args()
    if args.command == "select":
        select(args.seed, args.device)
    elif args.command == "test":
        test_selected(args.seed, args.device)
    else:
        frozen(args.seed, args.device, DEFAULT_PARAMETERS, args.test_frozen)
