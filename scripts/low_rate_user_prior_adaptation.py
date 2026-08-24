"""User-adaptive inference for sparse GeoLife transportation recognition.

The method is transductive: it receives a batch of *unlabelled* trajectories
belonging to the same new user and conservatively estimates that user's class
prior.  Ground-truth labels are accepted only by the reporting function and
are never passed to the adaptation routine.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path.home() / "research/tmi-sqfrc"
C_VALUE = 0.03
CLASS_POWER = 0.5
STACK_ALPHA = 0.5
PRIOR_STRENGTH = 0.7
PRIOR_RATIO_CLIP = 3.0
MINIMUM_USER_SAMPLES = 25


def metrics(probability, labels):
    prediction = probability.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
    }


def meta_features(log_probability):
    groups = np.asarray(log_probability).reshape(-1, 5, 5)
    probability = np.exp(groups)
    probability /= probability.sum(axis=2, keepdims=True)
    mean = probability.mean(axis=1)
    std = probability.std(axis=1)
    entropy = -(probability * np.log(probability.clip(1e-9))).sum(axis=2)
    ordered = np.sort(probability, axis=2)
    margin = ordered[:, :, -1] - ordered[:, :, -2]
    decisions = probability.argmax(axis=2)
    onehot = np.eye(5)[decisions].reshape(-1, 25)
    agreement = np.stack([(decisions == label).mean(axis=1)
                          for label in range(5)], axis=1)
    derived = np.concatenate([
        probability.reshape(-1, 25), mean, std, entropy, margin,
        onehot, agreement,
    ], axis=1)
    return np.concatenate([log_probability, derived], axis=1)


def class_weights(labels):
    count = np.bincount(labels, minlength=5)
    return (labels.size / (5 * count))[labels] ** CLASS_POWER


def fit_stacker(features, labels):
    model = LogisticRegression(
        C=C_VALUE, solver="lbfgs", multi_class="multinomial",
        max_iter=1500, random_state=10086,
    )
    model.fit(meta_features(features), labels,
              sample_weight=class_weights(labels))
    return model


def estimate_and_adjust(probability, source_prior, strength=PRIOR_STRENGTH,
                        ratio_clip=PRIOR_RATIO_CLIP):
    """Adjust one unlabelled user batch; this function has no label argument."""
    probability = np.asarray(probability, dtype=float)
    source_prior = np.asarray(source_prior, dtype=float).clip(1e-4)
    target_prior = source_prior.copy()
    for _ in range(100):
        ratio = np.clip(target_prior / source_prior,
                        1 / ratio_clip, ratio_clip)
        posterior = probability * ratio
        posterior /= posterior.sum(axis=1, keepdims=True)
        updated = posterior.mean(axis=0)
        if np.max(np.abs(updated - target_prior)) < 1e-7:
            target_prior = updated
            break
        target_prior = updated
    ratio = np.clip(target_prior / source_prior,
                    1 / ratio_clip, ratio_clip) ** strength
    adjusted = probability * ratio
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    return adjusted, target_prior


def adapt_grouped(probability, users, source_prior,
                  minimum_samples=MINIMUM_USER_SAMPLES):
    """Adapt each user independently without accessing any target labels."""
    result = np.asarray(probability, dtype=float).copy()
    estimated_priors = {}
    for user in np.unique(users):
        take = users == user
        if int(take.sum()) < minimum_samples:
            continue
        result[take], target = estimate_and_adjust(result[take], source_prior)
        estimated_priors[str(int(user))] = target.tolist()
    return result, estimated_priors


def oof_validation(features, labels, users, base):
    raw = np.zeros((labels.size, 5), dtype=float)
    adapted = np.zeros_like(raw)
    for user in np.unique(users):
        train = users != user
        model = fit_stacker(features[train], labels[train])
        candidate = model.predict_proba(meta_features(features[~train]))
        raw[~train] = (1 - STACK_ALPHA) * base[~train] + STACK_ALPHA * candidate
        prior = np.bincount(labels[train], minlength=5) / int(train.sum())
        adapted[~train] = estimate_and_adjust(raw[~train], prior)[0]
    return raw, adapted


def evaluate(seed):
    cache = ROOT / f"experiments/geolife_meta_cache_seed{seed}"
    output = ROOT / f"experiments/geolife_user_prior_adaptation_seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    validation = np.load(cache / "val.npz")
    test = np.load(cache / "test.npz")
    raw_oof, adapted_oof = oof_validation(
        validation["features"], validation["labels"],
        validation["users"], validation["base"],
    )
    model = fit_stacker(validation["features"], validation["labels"])
    stack = model.predict_proba(meta_features(test["features"]))
    raw_test = (1 - STACK_ALPHA) * test["base"] + STACK_ALPHA * stack
    source_prior = (np.bincount(validation["labels"], minlength=5)
                    / validation["labels"].size)
    adapted_test, target_priors = adapt_grouped(
        raw_test, test["users"], source_prior,
    )
    v2_test = np.exp(test["features"][:, :5])
    v2_test /= v2_test.sum(axis=1, keepdims=True)
    result = {
        "protocol": "V56-user-adaptive-confidence-disagreement-stacker",
        "setting": "transductive-unlabelled-per-user-batch",
        "target_labels_used_for_adaptation": False,
        "seed": seed,
        "parameters": {
            "C": C_VALUE, "class_power": CLASS_POWER,
            "stack_alpha": STACK_ALPHA,
            "prior_strength": PRIOR_STRENGTH,
            "prior_ratio_clip": PRIOR_RATIO_CLIP,
            "minimum_user_samples": MINIMUM_USER_SAMPLES,
        },
        "validation": {
            "base_v25": metrics(validation["base"], validation["labels"]),
            "raw_oof": metrics(raw_oof, validation["labels"]),
            "adapted_oof": metrics(adapted_oof, validation["labels"]),
        },
        "test": {
            "v2": metrics(v2_test, test["labels"]),
            "base_v25": metrics(test["base"], test["labels"]),
            "raw_stacker": metrics(raw_test, test["labels"]),
            "user_adapted": metrics(adapted_test, test["labels"]),
        },
        "estimated_target_priors": target_priors,
    }
    result["delta_vs_v2"] = {
        name: result["test"]["user_adapted"][name] - result["test"]["v2"][name]
        for name in ("accuracy", "macro_f1")
    }
    joblib.dump(model, output / "stacker.joblib")
    np.savez_compressed(output / "test_predictions.npz",
                        probability=adapted_test,
                        prediction=adapted_test.argmax(axis=1),
                        labels=test["labels"], users=test["users"])
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10086)
    evaluate(parser.parse_args().seed)
