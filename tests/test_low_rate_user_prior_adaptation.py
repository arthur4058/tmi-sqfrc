import unittest

import numpy as np

from scripts.low_rate_user_prior_adaptation import (
    adapt_grouped, estimate_and_adjust, meta_features,
)


class UserPriorAdaptationTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        raw = rng.uniform(0.02, 1.0, size=(40, 25))
        groups = raw.reshape(-1, 5, 5)
        groups /= groups.sum(axis=2, keepdims=True)
        self.features = np.log(groups.reshape(-1, 25))
        probability = rng.uniform(0.02, 1.0, size=(40, 5))
        self.probability = probability / probability.sum(axis=1, keepdims=True)
        self.prior = np.full(5, 0.2)

    def test_meta_features_are_finite(self):
        transformed = meta_features(self.features)
        self.assertEqual(transformed.shape, (40, 100))
        self.assertTrue(np.isfinite(transformed).all())

    def test_adjustment_is_normalized_and_label_free(self):
        adjusted, prior = estimate_and_adjust(self.probability, self.prior)
        np.testing.assert_allclose(adjusted.sum(axis=1), 1.0)
        self.assertEqual(prior.shape, (5,))

    def test_grouping_does_not_mix_users(self):
        users = np.repeat([1, 2], 20)
        first, _ = adapt_grouped(self.probability, users, self.prior,
                                 minimum_samples=1)
        changed = self.probability.copy()
        changed[20:] = np.roll(changed[20:], 1, axis=1)
        second, _ = adapt_grouped(changed, users, self.prior,
                                  minimum_samples=1)
        np.testing.assert_allclose(first[:20], second[:20])


if __name__ == "__main__":
    unittest.main()
