"""Order-sensitivity audit for past-only online adaptation."""

import json
import numpy as np

from scripts.low_rate_user_prior_adaptation import ROOT, metrics
from scripts.online_user_adaptation_v57 import adapt_online, raw_probabilities


def main():
    seed = 10086
    cache = ROOT / f"experiments/geolife_meta_cache_seed{seed}"
    val, test = np.load(cache / "val.npz"), np.load(cache / "test.npz")
    raw = raw_probabilities(val, test)
    source = np.bincount(val["labels"], minlength=5) / val["labels"].size
    rng = np.random.default_rng(20260824)
    outcomes = []
    unique_users = np.unique(test["users"])
    for _ in range(50):
        indices = np.concatenate([
            rng.permutation(np.flatnonzero(test["users"] == user))
            for user in unique_users
        ])
        probability = adapt_online(raw[indices], test["users"][indices], source, .4)
        outcomes.append(metrics(probability, test["labels"][indices]))
    result = {
        "repetitions": len(outcomes),
        "mean": {key: float(np.mean([x[key] for x in outcomes]))
                 for key in ("accuracy", "macro_f1")},
        "std": {key: float(np.std([x[key] for x in outcomes]))
                for key in ("accuracy", "macro_f1")},
        "minimum": {key: float(np.min([x[key] for x in outcomes]))
                    for key in ("accuracy", "macro_f1")},
        "maximum": {key: float(np.max([x[key] for x in outcomes]))
                    for key in ("accuracy", "macro_f1")},
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
