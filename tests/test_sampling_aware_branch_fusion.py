import unittest

import torch

from tmi.models.models import DualTSTransformerEncoderClassifier


def hyperparams(feat_dim, d_model):
    return {
        'feat_dim': feat_dim,
        'max_len': 6,
        'd_model': d_model,
        'n_heads': 2,
        'num_layers': 1,
        'dim_feedforward': 16,
        'dropout': 0.0,
        'pos_encoding': 'fixed',
        'activation': 'gelu',
        'norm': 'LayerNorm',
        'freeze': False,
    }


def build(enabled=False, frozen=False, use_activation_stats=True):
    return DualTSTransformerEncoderClassifier(
        hyperparams(2, 8),
        hyperparams(4, 8),
        num_classes=5,
        dropout=0.0,
        sampling_aware_branch_fusion=enabled,
        branch_fusion_hidden_dim=4,
        branch_fusion_use_activation_stats=use_activation_stats,
        branch_fusion_freeze_backbone=frozen,
    )


class SamplingAwareBranchFusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10086)
        self.trajectory = torch.randn(3, 6, 2)
        self.motion = torch.randn(3, 6, 4)
        self.delta_t = torch.randn(3, 6, 1)
        self.mask = torch.tensor([
            [True, True, True, True, True, False],
            [True, True, True, True, True, True],
            [True, True, True, True, False, False],
        ])

    def test_identity_initialization_matches_b0_logits_and_rng(self):
        torch.manual_seed(42)
        b0 = build(enabled=False)
        rng_after_b0 = torch.get_rng_state().clone()
        torch.manual_seed(42)
        gated = build(enabled=True)
        rng_after_gate = torch.get_rng_state().clone()

        self.assertTrue(torch.equal(rng_after_b0, rng_after_gate))
        for name, parameter in b0.state_dict().items():
            self.assertTrue(torch.equal(parameter, gated.state_dict()[name]), name)

        b0.eval()
        gated.eval()
        with torch.no_grad():
            baseline = b0(self.trajectory, self.mask, self.motion, self.mask)
            augmented = torch.cat([self.motion, self.delta_t], dim=-1)
            candidate = gated(self.trajectory, self.mask, augmented, self.mask)
        self.assertTrue(torch.equal(baseline, candidate))

    def test_frozen_mode_only_trains_gate_and_keeps_backbone_in_eval(self):
        model = build(enabled=True, frozen=True)
        trainable = [name for name, value in model.named_parameters() if value.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith('branch_fusion_gate.') for name in trainable))

        model.train()
        self.assertTrue(model.branch_fusion_gate.training)
        self.assertFalse(model.trajectory_branch.training)
        self.assertFalse(model.feature_branch.training)
        self.assertFalse(model.conv_layers.training)
        self.assertFalse(model.classifier.training)

    def test_gate_receives_finite_gradients(self):
        model = build(enabled=True, frozen=True)
        augmented = torch.cat([self.motion, self.delta_t], dim=-1)
        logits = model(self.trajectory, self.mask, augmented, self.mask)
        logits.sum().backward()
        gradient = model.branch_fusion_gate[-1].weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_scales_are_bounded(self):
        model = build(enabled=True)
        with torch.no_grad():
            model.branch_fusion_gate[-1].bias.copy_(torch.tensor([100.0, -100.0]))
        scales = model.branch_fusion_scales(torch.zeros(2, 10))
        self.assertTrue(torch.allclose(scales[:, 0], torch.full((2,), 1.5)))
        self.assertTrue(torch.allclose(scales[:, 1], torch.full((2,), 0.85)))

    def test_physical_only_gate_uses_five_descriptors(self):
        model = build(enabled=True, use_activation_stats=False)
        self.assertEqual(model.branch_fusion_gate[0].in_features, 5)
        augmented = torch.cat([self.motion, self.delta_t], dim=-1)
        model.eval()
        with torch.no_grad():
            logits = model(self.trajectory, self.mask, augmented, self.mask)
        self.assertEqual(logits.shape, (3, 5))

    def test_missing_delta_t_is_rejected(self):
        model = build(enabled=True)
        with self.assertRaisesRegex(ValueError, 'motion \\+ delta_t'):
            model(self.trajectory, self.mask, self.motion, self.mask)


if __name__ == '__main__':
    unittest.main()
