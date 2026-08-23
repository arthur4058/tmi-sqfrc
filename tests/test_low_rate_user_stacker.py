import unittest

import numpy as np

from scripts.low_rate_user_stacker import probability_features, sample_weights


class LowRateUserStackerTests(unittest.TestCase):
    def test_probability_features_are_finite(self):
        first = np.asarray([[0.0, 1.0], [0.25, 0.75]])
        second = np.asarray([[0.5, 0.5], [1.0, 0.0]])
        features = probability_features([first, second])
        self.assertEqual(features.shape, (2, 4))
        self.assertTrue(np.isfinite(features).all())

    def test_zero_weight_powers_return_unit_weights(self):
        labels = np.asarray([0, 0, 1, 1])
        users = np.asarray([10, 10, 20, 30])
        weights = sample_weights(labels, users, 0.0, 0.0)
        np.testing.assert_allclose(weights, np.ones(4))

    def test_user_weighting_upweights_small_user(self):
        labels = np.asarray([0, 0, 1, 1])
        users = np.asarray([10, 10, 10, 20])
        weights = sample_weights(labels, users, 0.0, 1.0)
        self.assertGreater(weights[-1], weights[0])


if __name__ == "__main__":
    unittest.main()
