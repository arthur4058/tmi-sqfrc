"""Unit tests for low-rate motion-expert consensus selection."""

import unittest

import numpy as np

from scripts.evaluate_dual_expert_consensus import (
    choose_temperature,
    choose_uniform_mix,
    metrics,
    softmax,
)


class MotionExpertConsensusTest(unittest.TestCase):
    def test_softmax_is_normalized(self):
        logits = np.array([[2.0, 1.0], [-1.0, 3.0]])
        probabilities = softmax(logits, temperature=1.5)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_validation_selection_returns_supported_values(self):
        targets = np.array([0, 1, 0, 1])
        b0_logits = np.array([[3, 0], [0, 3], [2, 1], [2, 1]], dtype=float)
        expert_logits = np.array([[2, 0], [0, 2], [0, 2], [0, 2]], dtype=float)
        t_b0 = choose_temperature(b0_logits, targets)
        t_expert = choose_temperature(expert_logits, targets)
        alpha = choose_uniform_mix(
            softmax(b0_logits, t_b0),
            softmax(expert_logits, t_expert),
            targets,
        )
        self.assertGreaterEqual(t_b0, 0.5)
        self.assertLessEqual(t_expert, 3.0)
        self.assertGreaterEqual(alpha, 0.0)
        self.assertLessEqual(alpha, 1.0)

    def test_metrics_exposes_paper_metrics(self):
        probabilities = np.array([[0.9, 0.1], [0.2, 0.8]])
        result = metrics(probabilities, np.array([0, 1]))
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["macro_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
