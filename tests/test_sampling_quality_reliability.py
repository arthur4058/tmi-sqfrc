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


class SamplingQualityReliabilityTest(unittest.TestCase):
    def test_quality_gate_is_identity_initialized_and_backpropagates(self):
        model = DualTSTransformerEncoderClassifier(
            _hyperparams(2, 8),
            _hyperparams(4, 8),
            num_classes=5,
            dropout=0.0,
            sampling_quality_reliability=True,
            sampling_quality_hidden_dim=8,
        )
        model.eval()

        x1 = torch.randn(3, 12, 2)
        motion = torch.randn(3, 12, 4)
        delta_t = torch.tensor([5.0, 30.0, 60.0]).view(3, 1, 1).expand(3, 12, 1)
        x2 = torch.cat([motion, delta_t], dim=-1)
        mask = torch.ones(3, 12, dtype=torch.bool)

        logits = model(x1, mask, x2, mask)
        self.assertEqual(logits.shape, (3, 5))
        self.assertTrue(torch.isfinite(logits).all())

        logits.sum().backward()
        grad = model.sampling_quality_gate[-1].weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())

    def test_quality_gate_rejects_missing_delta_t_channel(self):
        model = DualTSTransformerEncoderClassifier(
            _hyperparams(2, 8),
            _hyperparams(4, 8),
            num_classes=5,
            sampling_quality_reliability=True,
        )
        x1 = torch.randn(2, 12, 2)
        x2 = torch.randn(2, 12, 4)
        mask = torch.ones(2, 12, dtype=torch.bool)

        with self.assertRaisesRegex(ValueError, "motion \\+ delta_t"):
            model(x1, mask, x2, mask)


if __name__ == "__main__":
    unittest.main()
