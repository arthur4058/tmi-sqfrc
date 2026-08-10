import unittest

import torch

from tmi.models.models import (
    DualTSTransformerEncoderClassifier,
    PhysicalTimeMultiScaleAdapter,
)


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


class PhysicalTimeMultiScaleAdapterTest(unittest.TestCase):
    def test_zero_residual_preserves_baseline_features(self):
        adapter = PhysicalTimeMultiScaleAdapter(
            feature_dim=4, windows_seconds=(30, 60, 120), hidden_dim=8)
        motion = torch.randn(2, 6, 4)
        timestamps = torch.tensor([
            [0, 5, 10, 15, 20, 25],
            [0, 60, 120, 180, 240, 300],
        ], dtype=torch.float32)
        mask = torch.ones(2, 6, dtype=torch.bool)

        output = adapter(motion, timestamps, mask)
        self.assertTrue(torch.equal(output, motion))

        output.sum().backward()
        self.assertIsNotNone(adapter.residual_alpha.grad)
        self.assertTrue(torch.isfinite(adapter.residual_alpha.grad))

    def test_context_uses_seconds_not_point_offsets(self):
        adapter = PhysicalTimeMultiScaleAdapter(
            feature_dim=1, windows_seconds=(30,), hidden_dim=4)
        motion = torch.tensor([[[1.0], [2.0], [9.0]]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        dense_time = torch.tensor([[0.0, 10.0, 20.0]])
        sparse_time = torch.tensor([[0.0, 60.0, 120.0]])

        dense, _ = adapter._physical_contexts(motion, dense_time, mask)
        sparse, _ = adapter._physical_contexts(motion, sparse_time, mask)
        self.assertAlmostEqual(dense[0, 2, 0, 0].item(), 4.0)
        self.assertAlmostEqual(sparse[0, 2, 0, 0].item(), 9.0)

    def test_dual_model_forward_and_backpropagation(self):
        model = DualTSTransformerEncoderClassifier(
            _hyperparams(2, 8),
            _hyperparams(4, 8),
            num_classes=5,
            dropout=0.0,
            physical_time_multiscale=True,
            physical_time_windows=(30, 60, 120),
            physical_time_hidden_dim=8,
        )
        model.eval()
        x1 = torch.randn(3, 12, 2)
        motion = torch.randn(3, 12, 4)
        timestamps = torch.arange(12, dtype=torch.float32).view(
            1, 12, 1).expand(3, 12, 1) * 10.0
        x2 = torch.cat([motion, timestamps], dim=-1)
        mask = torch.ones(3, 12, dtype=torch.bool)

        logits = model(x1, mask, x2, mask)
        self.assertEqual(logits.shape, (3, 5))
        self.assertTrue(torch.isfinite(logits).all())
        logits.sum().backward()
        self.assertIsNotNone(model.physical_time_adapter.residual_alpha.grad)

    def test_dual_model_requires_timestamp_channel(self):
        model = DualTSTransformerEncoderClassifier(
            _hyperparams(2, 8),
            _hyperparams(4, 8),
            num_classes=5,
            physical_time_multiscale=True,
        )
        x1 = torch.randn(2, 12, 2)
        x2 = torch.randn(2, 12, 4)
        mask = torch.ones(2, 12, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "motion \\+ timestamp"):
            model(x1, mask, x2, mask)


if __name__ == "__main__":
    unittest.main()
