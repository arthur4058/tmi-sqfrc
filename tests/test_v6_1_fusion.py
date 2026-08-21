import unittest

import numpy as np

from scripts.calibrate_v6_1_v2_fusion import choose_residual_mix


class V61FusionTest(unittest.TestCase):
    def test_residual_mix_prefers_zero_for_identical_probabilities(self):
        probability = np.array([[0.8, 0.2], [0.2, 0.8]])
        targets = np.array([0, 1])

        weight, base_score, fusion_score = choose_residual_mix(
            probability, probability, targets,
        )

        self.assertEqual(weight, 0.0)
        self.assertEqual(fusion_score, base_score)

    def test_residual_mix_can_improve_both_metrics(self):
        base = np.array([
            [0.6, 0.4], [0.4, 0.6], [0.4, 0.6], [0.6, 0.4],
        ])
        candidate = np.array([
            [0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9],
        ])
        targets = np.array([0, 1, 0, 1])

        weight, base_score, fusion_score = choose_residual_mix(
            base, candidate, targets,
        )

        self.assertGreater(weight, 0.0)
        self.assertGreater(fusion_score["accuracy"], base_score["accuracy"])
        self.assertGreater(fusion_score["macro_f1"], base_score["macro_f1"])


if __name__ == "__main__":
    unittest.main()
