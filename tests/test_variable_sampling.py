import unittest

import numpy as np

from tmi.data_preprocess.variable_sampling import (
    DEFAULT_CONDITIONS,
    apply_condition,
    build_paired_views,
    fixed_interval_sample,
    physical_windows,
)


def trajectory(duration=600, step=1):
    timestamps = np.arange(0, duration + 1, step, dtype=float)
    return np.column_stack((
        timestamps,
        39.9 + timestamps * 1e-6,
        116.3 + timestamps * 1e-6,
    ))


class VariableSamplingTest(unittest.TestCase):
    def test_physical_windows_and_fixed_sampling_select_original_points(self):
        trj = trajectory()
        windows = physical_windows(trj, 300, 150)
        self.assertEqual(len(windows), 3)
        sampled = fixed_interval_sample(windows[0][1], 60)
        self.assertEqual(len(sampled), 6)
        self.assertTrue(np.all(np.isin(sampled[:, 0], trj[:, 0])))
        self.assertTrue(np.all(np.diff(sampled[:, 0]) > 0))

    def test_random_views_are_deterministic_and_keep_timestamps(self):
        trj = trajectory(300)
        first = apply_condition(trj, "random_drop_70", 42, "pair")
        second = apply_condition(trj, "random_drop_70", 42, "pair")
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(np.isin(first[:, 0], trj[:, 0])))

    def test_paired_views_have_identical_metadata_and_minimum_points(self):
        trjs = np.asarray([trajectory(600), trajectory(600)], dtype=object)
        labels = np.asarray([0, 1])
        users = np.asarray([1, 2])
        sources = np.asarray(["a", "b"])
        views, labels_out, users_out, _, pairs, _ = build_paired_views(
            trjs, labels, users, sources, DEFAULT_CONDITIONS, seed=42)
        expected = len(labels_out)
        self.assertEqual(expected, len(users_out))
        self.assertEqual(expected, len(pairs))
        self.assertGreater(expected, 0)
        for condition, condition_trjs in views.items():
            self.assertEqual(len(condition_trjs), expected, condition)
            self.assertGreaterEqual(min(map(len, condition_trjs)), 5)
            for sampled in condition_trjs:
                self.assertLessEqual(len(sampled), 61)


if __name__ == "__main__":
    unittest.main()
