"""Past-only online user adaptation using the frozen V56 stacker."""

import argparse
import json

import numpy as np

from scripts.low_rate_user_prior_adaptation import (
    PRIOR_RATIO_CLIP, ROOT, STACK_ALPHA, fit_stacker, meta_features, metrics,
)


def estimate_prior(history, source_prior):
    target = source_prior.copy()
    for _ in range(100):
        ratio = np.clip(target / source_prior, 1 / PRIOR_RATIO_CLIP, PRIOR_RATIO_CLIP)
        posterior = history * ratio
        posterior /= posterior.sum(axis=1, keepdims=True)
        updated = posterior.mean(axis=0)
        if np.max(np.abs(updated - target)) < 1e-7:
            return updated
        target = updated
    return target


def adapt_online(probability, users, source_prior, strength, warmup=25,
                 update_interval=10):
    result = probability.copy()
    for user in np.unique(users):
        indices = np.flatnonzero(users == user)
        target = source_prior.copy()
        for position, index in enumerate(indices):
            if position >= warmup:
                if position == warmup or (position - warmup) % update_interval == 0:
                    target = estimate_prior(probability[indices[:position]], source_prior)
                ratio = np.clip(target / source_prior,
                                1 / PRIOR_RATIO_CLIP, PRIOR_RATIO_CLIP) ** strength
                result[index] *= ratio
                result[index] /= result[index].sum()
    return result


def raw_probabilities(validation, target):
    model = fit_stacker(validation["features"], validation["labels"])
    stack = model.predict_proba(meta_features(target["features"]))
    return (1 - STACK_ALPHA) * target["base"] + STACK_ALPHA * stack


def oof_raw(validation):
    labels, users = validation["labels"], validation["users"]
    result = np.zeros((labels.size, 5))
    for user in np.unique(users):
        train = users != user
        model = fit_stacker(validation["features"][train], labels[train])
        stack = model.predict_proba(meta_features(validation["features"][~train]))
        result[~train] = ((1 - STACK_ALPHA) * validation["base"][~train]
                          + STACK_ALPHA * stack)
    return result


def main(seed):
    cache = ROOT / f"experiments/geolife_meta_cache_seed{seed}"
    validation, test = np.load(cache / "val.npz"), np.load(cache / "test.npz")
    source = np.bincount(validation["labels"], minlength=5) / validation["labels"].size
    val_raw = oof_raw(validation)
    trials = []
    for strength in np.linspace(.1, 1.0, 10):
        candidate = adapt_online(val_raw, validation["users"], source, strength)
        value = metrics(candidate, validation["labels"])
        trials.append({"strength": float(strength), "metrics": value,
                       "objective": value["accuracy"] + .2*value["macro_f1"]})
    winner = max(trials, key=lambda x: x["objective"])
    test_raw = raw_probabilities(validation, test)
    test_adapted = adapt_online(test_raw, test["users"], source, winner["strength"])
    result = {"protocol": "V57-past-only-online-user-adaptation",
              "seed": seed, "future_predictions_used": False,
              "selection": winner, "validation_trials": trials,
              "test": {"raw": metrics(test_raw, test["labels"]),
                       "online_adapted": metrics(test_adapted, test["labels"])}}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10086)
    main(parser.parse_args().seed)
