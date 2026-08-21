import unittest

import numpy as np

from scripts.build_dense_behavior_targets import (
    TARGET_NAMES,
    build_window_intervals,
    select_dev_users,
)


class BehaviorTargetAlignmentTests(unittest.TestCase):
    def test_dense_edges_are_selected_by_timestamp(self):
        timestamps = np.arange(0.0, 65.0, 5.0)
        dense = np.column_stack((
            timestamps,
            np.zeros_like(timestamps),
            np.linspace(0.0, 0.006, len(timestamps)),
        ))
        sparse = dense[[0, 6, 12]]
        inputs, targets, valid = build_window_intervals(dense, sparse)
        self.assertEqual(inputs.shape, (2, 7))
        self.assertEqual(targets.shape, (2, len(TARGET_NAMES)))
        self.assertTrue(np.all(valid))
        self.assertTrue(np.allclose(inputs[:, 0], 30.0))
        self.assertTrue(np.all(np.isfinite(targets)))

    def test_interval_without_two_dense_points_is_invalid(self):
        dense = np.asarray([[0.0, 0.0, 0.0], [120.0, 0.0, 0.001]])
        sparse = np.asarray([[0.0, 0.0, 0.0], [60.0, 0.0, 0.0005]])
        _, targets, valid = build_window_intervals(dense, sparse)
        self.assertFalse(valid[0])
        self.assertTrue(np.all(np.isnan(targets[0])))

    def test_train_a_and_dev_a_are_user_disjoint_and_cover_classes(self):
        users = np.repeat(np.arange(10), 5)
        labels = np.tile(np.arange(5), 10)
        train, dev = select_dev_users(labels, users, seed=7, trials=100)
        self.assertFalse(set(train) & set(dev))
        self.assertEqual(set(labels[np.isin(users, train)]), set(range(5)))
        self.assertEqual(set(labels[np.isin(users, dev)]), set(range(5)))


if __name__ == "__main__":
    unittest.main()
