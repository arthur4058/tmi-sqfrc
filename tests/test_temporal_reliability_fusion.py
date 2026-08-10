import unittest

import torch

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


class TemporalReliabilityFusionTest(unittest.TestCase):
    def _inputs(self):
        x1 = torch.randn(3, 12, 2)
        motion = torch.randn(3, 12, 4)
        delta_t = torch.randn(3, 12, 1)
        x2 = torch.cat([motion, delta_t], dim=-1)
        mask = torch.ones(3, 12, dtype=torch.bool)
        mask[1, 8:] = False
        mask[2, 5:] = False
        return x1, motion, x2, mask

    def test_identity_initialization_matches_b0(self):
        torch.manual_seed(7)
        baseline = DualTSTransformerEncoderClassifier(
            _hyperparams(2, 8), _hyperparams(4, 8), 5, dropout=0.0)
        torch.manual_seed(7)
        calibrated = DualTSTransformerEncoderClassifier(
            _hyperparams(2, 8),
            _hyperparams(4, 8),
            5,
            dropout=0.0,
            temporal_reliability_fusion=True,
            sampling_interval_seconds=60.0,
        )
        calibrated_state = calibrated.state_dict()
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(calibrated_state[name], value)
        baseline.eval()
        calibrated.eval()
        x1, motion, x2, mask = self._inputs()
        expected = baseline(x1, mask, motion, mask)
        actual = calibrated(x1, mask, x2, mask)
        torch.testing.assert_close(actual, expected)

    def test_gate_backpropagates_with_finite_values(self):
        model = DualTSTransformerEncoderClassifier(
            _hyperparams(2, 8),
            _hyperparams(4, 8),
            5,
            dropout=0.0,
            temporal_reliability_fusion=True,
            temporal_reliability_strength=0.25,
            sampling_interval_seconds=60.0,
        )
        x1, _, x2, mask = self._inputs()
        logits = model(x1, mask, x2, mask)
        self.assertTrue(torch.isfinite(logits).all())
        logits.sum().backward()
        grad = model.temporal_reliability_gate[-1].weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())

    def test_missing_delta_t_is_rejected(self):
        model = DualTSTransformerEncoderClassifier(
            _hyperparams(2, 8),
            _hyperparams(4, 8),
            5,
            temporal_reliability_fusion=True,
        )
        x1 = torch.randn(2, 12, 2)
        x2 = torch.randn(2, 12, 4)
        mask = torch.ones(2, 12, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "motion \\+ delta_t"):
            model(x1, mask, x2, mask)

    def test_incompatible_reliability_modes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot be enabled together"):
            DualTSTransformerEncoderClassifier(
                _hyperparams(2, 8),
                _hyperparams(4, 8),
                5,
                sampling_quality_reliability=True,
                temporal_reliability_fusion=True,
            )


if __name__ == "__main__":
    unittest.main()
