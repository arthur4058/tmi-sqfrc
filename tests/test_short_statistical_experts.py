import unittest

import numpy as np
import torch

from scripts.train_short_trajectory_mlp import ShortTrajectoryMLP
from scripts.train_statistical_motion_expert import (
    FEATURE_INDICES,
    _channel_summary,
    _trajectory_summary,
)


class StatisticalMotionFeatureTests(unittest.TestCase):
    def test_feature_set_excludes_time_and_location_channels(self):
        self.assertEqual(FEATURE_INDICES, (2, 3, 4, 5, 6, 7, 8))

    def test_channel_summary_is_finite(self):
        summary = _channel_summary([1.0, np.nan, np.inf, 4.0])
        self.assertEqual(summary.shape, (13,))
        self.assertTrue(np.isfinite(summary).all())

    def test_trajectory_summary_ignores_absolute_longitude(self):
        trajectory = np.asarray([
            np.asarray([39.0, 39.001, 39.002]),
            np.asarray([116.0, 116.001, 116.003]),
        ], dtype=object)
        translated = np.asarray([
            trajectory[0].copy(),
            trajectory[1] - 20.0,
        ], dtype=object)
        np.testing.assert_allclose(
            _trajectory_summary(trajectory),
            _trajectory_summary(translated),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_short_mlp_output_shape(self):
        model = ShortTrajectoryMLP(101)
        self.assertEqual(tuple(model(torch.zeros(4, 101)).shape), (4, 5))


if __name__ == "__main__":
    unittest.main()
