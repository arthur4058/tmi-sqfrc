"""Unit tests for global three-expert weight selection."""

import unittest

import numpy as np

from scripts.evaluate_three_expert_consensus import choose_simplex_mix


class ThreeExpertConsensusTest(unittest.TestCase):
    def test_weights_are_on_probability_simplex(self):
        targets = np.array([0, 1, 2, 0, 1, 2])
        probabilities = [
            np.eye(3)[np.array([0, 1, 0, 0, 2, 2])] * 0.8 + 0.2 / 3,
            np.eye(3)[np.array([0, 0, 2, 1, 1, 2])] * 0.8 + 0.2 / 3,
            np.eye(3)[targets] * 0.8 + 0.2 / 3,
        ]
        weights = choose_simplex_mix(probabilities, targets, step=0.1)
        self.assertTrue(np.all(weights >= 0.0))
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertGreater(weights[2], 0.0)


if __name__ == "__main__":
    unittest.main()
