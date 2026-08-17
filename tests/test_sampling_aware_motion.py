import unittest

import torch

from tmi.models.models import (
    DualTSTransformerEncoderClassifier,
    ECATemporalChannelAttention,
)


def hyperparams(feat_dim, d_model):
    return {
        "feat_dim": feat_dim, "max_len": 16, "d_model": d_model,
        "n_heads": 2, "num_layers": 1, "dim_feedforward": 32,
        "dropout": 0.0, "pos_encoding": "fixed", "activation": "gelu",
        "norm": "LayerNorm", "freeze": False,
    }


class SamplingAwareMotionTests(unittest.TestCase):
    def make_model(self):
        torch.manual_seed(7)
        return DualTSTransformerEncoderClassifier(
            hyperparams(2, 8), hyperparams(4, 8), num_classes=5,
            dropout=0.0, sampling_aware_multiscale_motion=True,
            sampling_interval_seconds=60,
            sampling_motion_horizons_seconds=(60, 120, 300),
            sampling_motion_hidden_dim=8,
            sampling_motion_gate_max=0.5,
            sampling_motion_gate_init=0.1,
            sampling_motion_base_indices=(2, 3, 4, 6),
        )

    def test_physical_horizons_become_rate_specific_kernels(self):
        model = self.make_model()
        self.assertEqual(model.sampling_motion_kernel_sizes, (1, 2, 5))

    def test_forward_shape_and_finite_values(self):
        model = self.make_model().eval()
        trajectory = torch.randn(3, 6, 2)
        motion = torch.randn(3, 6, 7)
        mask = torch.tensor([
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 0],
        ], dtype=torch.bool)
        with torch.no_grad():
            output = model(trajectory, mask, motion, mask)
        self.assertEqual(tuple(output.shape), (3, 5))
        self.assertTrue(torch.isfinite(output).all())
        self.assertAlmostEqual(model.last_sampling_motion_gate_mean, 0.1, places=5)

    def test_padding_values_do_not_change_motion_summary(self):
        model = self.make_model().eval()
        motion = torch.randn(2, 6, 7)
        mask = torch.tensor([
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
        ], dtype=torch.bool)
        motion[1, :3] = motion[0, :3]
        motion[0, 3:] = 0
        motion[1, 3:] = 1000
        with torch.no_grad():
            residual, gate = model._sampling_aware_motion_summary(motion, mask)
        self.assertTrue(torch.allclose(residual[0], residual[1], atol=1e-5))
        self.assertTrue(torch.allclose(gate[0], gate[1], atol=1e-6))

    def test_eca_preserves_shape(self):
        module = ECATemporalChannelAttention(kernel_size=3)
        sequence = torch.randn(4, 8, 6)
        mask = torch.ones(4, 6, dtype=torch.bool)
        output = module(sequence, mask)
        self.assertEqual(output.shape, sequence.shape)

    def test_invalid_motion_width_is_rejected(self):
        model = self.make_model().eval()
        trajectory = torch.randn(2, 6, 2)
        motion = torch.randn(2, 6, 6)
        mask = torch.ones(2, 6, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "seven channels"):
            model(trajectory, mask, motion, mask)


if __name__ == "__main__":
    unittest.main()
