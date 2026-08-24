"""Controls for V56: true user grouping versus shuffled pseudo-users."""

import argparse
import json

import numpy as np

from scripts.low_rate_user_prior_adaptation import (
    ROOT, STACK_ALPHA, adapt_grouped, fit_stacker, meta_features, metrics,
)


def main(seed):
    cache = ROOT / f"experiments/geolife_meta_cache_seed{seed}"
    val, test = np.load(cache / "val.npz"), np.load(cache / "test.npz")
    model = fit_stacker(val["features"], val["labels"])
    raw = ((1 - STACK_ALPHA) * test["base"] + STACK_ALPHA *
           model.predict_proba(meta_features(test["features"])))
    source = np.bincount(val["labels"], minlength=5) / val["labels"].size
    true_probability = adapt_grouped(raw, test["users"], source)[0]
    rng = np.random.default_rng(20260824)
    shuffled = []
    for _ in range(50):
        pseudo_users = rng.permutation(test["users"])
        probability = adapt_grouped(raw, pseudo_users, source)[0]
        shuffled.append(metrics(probability, test["labels"]))
    result = {
        "seed": seed,
        "raw": metrics(raw, test["labels"]),
        "true_user_grouping": metrics(true_probability, test["labels"]),
        "shuffled_grouping_mean": {
            key: float(np.mean([item[key] for item in shuffled]))
            for key in ("accuracy", "macro_f1")
        },
        "shuffled_grouping_std": {
            key: float(np.std([item[key] for item in shuffled]))
            for key in ("accuracy", "macro_f1")
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10086)
    main(parser.parse_args().seed)
