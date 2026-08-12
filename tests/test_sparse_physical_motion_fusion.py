import unittest

import numpy as np
import pandas as pd
import torch

from tmi.datasets.data import SPARSE_PHYSICAL_FEATURES, append_sparse_physical_features
from tmi.models.models import DualTSTransformerEncoderClassifier


def _hyperparams(feat_dim, d_model):
    return {
        "feat_dim": feat_dim,
        "max_len": 12,
        "d_model": d_model,
        "n_heads": 2,
        "num_layers": 1,
        "dim_feedforward": 16,
        "dropout": 0.0,
        "pos_encoding": "fixed",
        "activation": "gelu",
        "norm": "LayerNorm",
        "freeze": False,
    }


def _model(enabled=False, interval=60.0):
    return DualTSTransformerEncoderClassifier(
        _hyperparams(2, 8),
        _hyperparams(4, 8),
        num_classes=5,
        dropout=0.0,
        sparse_physical_motion_fusion=enabled,
        sparse_physical_edge_hidden_dim=8,
        sampling_interval_seconds=interval,
        physical_window_seconds=300.0,
    )


class SparsePhysicalMotionFusionTest(unittest.TestCase):
    def test_engineered_physical_features_are_finite(self):
        frame = pd.DataFrame({
            0: [0.0, 5.0, 60.0],
            2: [0.0, 25.0, 600.0],
            6: [0.0, 90.0, 180.0],
            7: [0.0, 45.0, -90.0],
        })
        result = append_sparse_physical_features(frame)
        values = result[list(SPARSE_PHYSICAL_FEATURES)].to_numpy()
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue(np.allclose(result["v5_physical_speed"], [0.0, 5.0, 10.0]))

    def test_v5_preserves_seeded_backbone_and_initial_logits(self):
        torch.manual_seed(10086)
        baseline = _model(enabled=False)
        torch.manual_seed(10086)
        v5 = _model(enabled=True)

        v5_state = v5.state_dict()
        for name, value in baseline.state_dict().items():
            self.assertIn(name, v5_state)
            self.assertTrue(torch.equal(value, v5_state[name]), name)

        baseline.eval()
        v5.eval()
        x1 = torch.randn(3, 12, 2)
        motion = torch.randn(3, 12, 4)
        physical = torch.randn(3, 12, 6)
        mask = torch.ones(3, 12, dtype=torch.bool)
        with torch.no_grad():
            baseline_logits = baseline(x1, mask, motion, mask)
            v5_logits = v5(x1, mask, torch.cat([motion, physical], dim=-1), mask)
        self.assertTrue(torch.equal(baseline_logits, v5_logits))

    def test_v5_is_finite_and_residual_receives_gradient(self):
        model = _model(enabled=True)
        model.train()
        x1 = torch.randn(4, 12, 2)
        x2 = torch.randn(4, 12, 10)
        mask = torch.ones(4, 12, dtype=torch.bool)
        logits = model(x1, mask, x2, mask)
        self.assertTrue(torch.isfinite(logits).all())
        logits.square().mean().backward()
        grad = model.sparse_physical_residual[-1].weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0.0)

    def test_fusion_budget_increases_with_sampling_interval(self):
        mask = torch.ones(1, 6, dtype=torch.bool)
        alphas = [
            _model(enabled=True, interval=interval)._sparse_physical_alpha(mask).item()
            for interval in (5.0, 10.0, 20.0, 30.0, 60.0)
        ]
        self.assertEqual(alphas, sorted(alphas))
        self.assertGreater(alphas[-1], alphas[0])

    def test_v5_rejects_missing_physical_channels(self):
        model = _model(enabled=True)
        mask = torch.ones(2, 12, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "motion \\+ physical"):
            model(torch.randn(2, 12, 2), mask, torch.randn(2, 12, 4), mask)

    def test_v5_rejects_m1_combination(self):
        with self.assertRaisesRegex(ValueError, "cannot be enabled together"):
            DualTSTransformerEncoderClassifier(
                _hyperparams(2, 8),
                _hyperparams(4, 8),
                num_classes=5,
                sampling_quality_reliability=True,
                sparse_physical_motion_fusion=True,
            )


if __name__ == "__main__":
    unittest.main()
