import unittest

import numpy as np

from scripts.evaluate_user_robust_v2_v16 import choose_user_robust_weight


class UserRobustFusionTest(unittest.TestCase):
    def test_zero_weight_wins_when_expert_does_not_change_predictions(self):
        v2 = np.asarray([
            [0.8, 0.2],
            [0.7, 0.3],
            [0.2, 0.8],
            [0.3, 0.7],
        ])
        selected, _ = choose_user_robust_weight(
            v2,
            v2.copy(),
            np.asarray([0, 0, 1, 1]),
            np.asarray([10, 10, 20, 20]),
        )
        self.assertEqual(selected["weight"], 0.0)
        self.assertEqual(selected["users_both_nonnegative"], 1.0)

    def test_rule_tracks_user_level_non_degradation(self):
        labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
        users = np.asarray([10, 10, 10, 10, 20, 20, 20, 20])
        v2 = np.asarray([
            [0.8, 0.2], [0.6, 0.4], [0.4, 0.6], [0.2, 0.8],
            [0.8, 0.2], [0.4, 0.6], [0.6, 0.4], [0.2, 0.8],
        ])
        short = np.asarray([
            [0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9],
            [0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9],
        ])
        selected, trials = choose_user_robust_weight(v2, short, labels, users)
        self.assertGreater(selected["weight"], 0.0)
        self.assertEqual(selected["users_both_nonnegative"], 1.0)
        self.assertEqual(len(trials), 51)


if __name__ == "__main__":
    unittest.main()
