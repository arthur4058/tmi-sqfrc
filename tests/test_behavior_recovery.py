import unittest

import numpy as np
import torch

from scripts.evaluate_behavior_recovery import inverse_targets, transform_targets
from tmi.models.behavior_recovery import IntervalBehaviorRecovery


class BehaviorRecoveryTests(unittest.TestCase):
    def test_shape_and_finite_output(self):
        model = IntervalBehaviorRecovery()
        output = model(torch.randn(11, 7))
        self.assertEqual(output.shape, (11, 6))
        self.assertTrue(torch.isfinite(output).all())

    def test_target_transform_round_trip(self):
        values = np.asarray([
            [3.0, 0.5, 4.0, 0.25, 0.1, 12.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float32)
        restored = inverse_targets(transform_targets(values))
        np.testing.assert_allclose(restored, values, rtol=1e-6, atol=1e-6)

    def test_model_accepts_sparse_features_only(self):
        model = IntervalBehaviorRecovery(input_dim=7)
        names = [name for name, _ in model.named_parameters()]
        self.assertFalse(any("dense" in name for name in names))


if __name__ == "__main__":
    unittest.main()
