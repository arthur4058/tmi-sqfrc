import unittest

import torch

from tmi.models.models import DualTSTransformerEncoderClassifier


def hyperparams(feat_dim):
    return {
        "feat_dim": feat_dim, "max_len": 6, "d_model": 8,
        "n_heads": 2, "num_layers": 1, "dim_feedforward": 16,
        "dropout": 0.0, "pos_encoding": "fixed",
        "activation": "gelu", "norm": "LayerNorm", "freeze": False,
    }


def make_model(enabled=True, freeze=False, scale=0.5):
    return DualTSTransformerEncoderClassifier(
        hyperparams(2), hyperparams(4), 5, dropout=0.0,
        kinematic_summary_residual=enabled,
        kinematic_summary_hidden_dim=8,
        kinematic_summary_scale=scale,
        freeze_base_model_for_sparse=freeze,
    )


class KinematicSummaryResidualTests(unittest.TestCase):
    def test_zero_initialization_preserves_b0_logits(self):
        torch.manual_seed(31)
        baseline = make_model(enabled=False).eval()
        torch.manual_seed(31)
        candidate = make_model(enabled=True).eval()
        x1 = torch.randn(3, 6, 2)
        x2 = torch.randn(3, 6, 4)
        mask = torch.ones(3, 6, dtype=torch.bool)
        with torch.no_grad():
            expected = baseline(x1, mask, x2, mask)
            actual = candidate(x1, mask, x2, mask)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_padding_values_do_not_change_summary(self):
        model = make_model().eval()
        features = torch.randn(2, 6, 4)
        mask = torch.tensor([
            [1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]
        ], dtype=torch.bool)
        changed = features.clone()
        changed[0, 3:] = 10000.0
        with torch.no_grad():
            first = model._kinematic_window_summary(features, mask)
            second = model._kinematic_window_summary(changed, mask)
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    def test_single_point_and_empty_windows_are_finite(self):
        model = make_model().eval()
        features = torch.randn(2, 1, 4)
        mask = torch.tensor([[1], [0]], dtype=torch.bool)
        summary = model._kinematic_window_summary(features, mask)
        self.assertEqual(summary.shape, (2, 26))
        self.assertTrue(torch.isfinite(summary).all())

    def test_frozen_b0_stays_in_eval_mode(self):
        model = make_model(freeze=True).train()
        self.assertFalse(model.trajectory_branch.training)
        self.assertFalse(model.feature_branch.training)
        self.assertFalse(model.conv_layers.training)
        self.assertFalse(model.classifier.training)
        self.assertTrue(model.kinematic_summary_head.training)
        self.assertFalse(any(
            parameter.requires_grad
            for module in model._base_modules()
            for parameter in module.parameters()
        ))
        self.assertTrue(any(
            parameter.requires_grad
            for parameter in model.kinematic_summary_head.parameters()
        ))

    def test_optional_module_preserves_rng_stream(self):
        torch.manual_seed(37)
        make_model(enabled=False)
        expected = torch.rand(8)
        torch.manual_seed(37)
        make_model(enabled=True)
        actual = torch.rand(8)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_scale_is_bounded(self):
        with self.assertRaisesRegex(ValueError, "kinematic_summary_scale"):
            make_model(scale=1.1)


if __name__ == "__main__":
    unittest.main()
