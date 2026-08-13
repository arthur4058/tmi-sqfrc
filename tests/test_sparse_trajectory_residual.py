import unittest

import torch

from tmi.models.models import DualTSTransformerEncoderClassifier


def _hyperparams(feat_dim, d_model, max_len=6):
    return {
        "feat_dim": feat_dim,
        "max_len": max_len,
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


def _model(masked=False, residual=False, logits=False, freeze_base=False,
           correction_scale=1.0, physical_only=False):
    return DualTSTransformerEncoderClassifier(
        _hyperparams(2, 8),
        _hyperparams(10, 8),
        num_classes=5,
        dropout=0.0,
        masked_fusion_pooling=masked,
        sparse_trajectory_residual=residual,
        sparse_trajectory_logit_correction=logits,
        freeze_base_model_for_sparse=freeze_base,
        sparse_trajectory_correction_scale=correction_scale,
        sparse_physical_only=physical_only,
        sparse_trajectory_hidden_dim=8,
        sparse_trajectory_gate_init=0.1,
    )


class SparseTrajectoryResidualTest(unittest.TestCase):
    def test_masked_pooling_and_residual_forward_backward(self):
        model = _model(masked=True, residual=True)
        model.train()
        x1 = torch.randn(4, 6, 2)
        x2 = torch.randn(4, 6, 10)
        mask = torch.tensor([
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
        ], dtype=torch.bool)

        logits = model(x1, mask, x2, mask)
        self.assertEqual(logits.shape, (4, 5))
        self.assertTrue(torch.isfinite(logits).all())
        logits.square().mean().backward()
        grad = model.sparse_residual_projector[-1].weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())

    def test_zero_residual_preserves_baseline_logits(self):
        torch.manual_seed(7)
        baseline = _model(masked=False, residual=False).eval()
        torch.manual_seed(7)
        candidate = _model(masked=False, residual=True).eval()
        x1 = torch.randn(3, 6, 2)
        x2 = torch.randn(3, 6, 10)
        mask = torch.ones(3, 6, dtype=torch.bool)

        with torch.no_grad():
            expected = baseline(x1, mask, x2, mask)
            actual = candidate(x1, mask, x2, mask)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_optional_module_preserves_global_rng_stream(self):
        torch.manual_seed(19)
        _model(masked=False, residual=False)
        expected = torch.rand(8)
        torch.manual_seed(19)
        _model(masked=False, residual=True)
        actual = torch.rand(8)

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_zero_logit_correction_preserves_baseline_logits(self):
        torch.manual_seed(23)
        baseline = _model().eval()
        torch.manual_seed(23)
        candidate = _model(logits=True).eval()
        x1 = torch.randn(3, 6, 2)
        x2 = torch.randn(3, 6, 10)
        mask = torch.ones(3, 6, dtype=torch.bool)

        with torch.no_grad():
            expected = baseline(x1, mask, x2, mask)
            actual = candidate(x1, mask, x2, mask)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_residual_modes_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "either"):
            _model(residual=True, logits=True)

    def test_frozen_base_stays_eval_while_correction_trains(self):
        model = _model(logits=True, freeze_base=True).train()

        self.assertFalse(model.trajectory_branch.training)
        self.assertFalse(model.feature_branch.training)
        self.assertFalse(model.conv_layers.training)
        self.assertFalse(model.classifier.training)
        self.assertTrue(model.sparse_motion_encoder.training)
        self.assertFalse(any(
            parameter.requires_grad
            for module in model._base_modules()
            for parameter in module.parameters()
        ))
        self.assertTrue(any(
            parameter.requires_grad
            for parameter in model.sparse_residual_projector.parameters()
        ))

    def test_zero_correction_scale_returns_base_logits(self):
        torch.manual_seed(29)
        baseline = _model().eval()
        torch.manual_seed(29)
        candidate = _model(logits=True, correction_scale=0.0).eval()
        torch.nn.init.normal_(candidate.sparse_residual_projector[-1].weight)
        x1 = torch.randn(3, 6, 2)
        x2 = torch.randn(3, 6, 10)
        mask = torch.ones(3, 6, dtype=torch.bool)

        with torch.no_grad():
            expected = baseline(x1, mask, x2, mask)
            actual = candidate(x1, mask, x2, mask)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_correction_scale_is_bounded(self):
        with self.assertRaisesRegex(ValueError, "\[0, 1\]"):
            _model(logits=True, correction_scale=1.1)

    def test_physical_only_correction_ignores_transformer_states(self):
        model = _model(logits=True, physical_only=True).eval()
        torch.nn.init.normal_(model.sparse_residual_projector[-1].weight)
        x1 = torch.randn(2, 6, 2)
        encoded_a = torch.randn(2, 6, model.trajectory_d_model)
        encoded_b = torch.randn(2, 6, model.trajectory_d_model)
        mask = torch.ones(2, 6, dtype=torch.bool)

        with torch.no_grad():
            residual_a, gate_a = model._sparse_trajectory_summary(
                x1, mask, encoded_a
            )
            residual_b, gate_b = model._sparse_trajectory_summary(
                x1, mask, encoded_b
            )
        torch.testing.assert_close(residual_a, residual_b, rtol=0.0, atol=0.0)
        torch.testing.assert_close(gate_a, gate_b, rtol=0.0, atol=0.0)

    def test_relative_motion_is_translation_invariant(self):
        model = _model(residual=True).eval()
        torch.nn.init.normal_(model.sparse_residual_projector[-1].weight)
        x1 = torch.randn(2, 6, 2)
        translated = x1 + torch.tensor([20.0, -35.0])
        encoded = torch.randn(2, 6, model.trajectory_d_model)
        mask = torch.ones(2, 6, dtype=torch.bool)

        with torch.no_grad():
            residual_a, gate_a = model._sparse_trajectory_summary(x1, mask, encoded)
            residual_b, gate_b = model._sparse_trajectory_summary(
                translated, mask, encoded
            )
        torch.testing.assert_close(residual_a, residual_b, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(gate_a, gate_b, rtol=1e-5, atol=1e-5)

    def test_single_valid_point_summary_is_finite(self):
        model = _model(residual=True).eval()
        x1 = torch.randn(2, 1, 2)
        encoded = torch.randn(2, 1, model.trajectory_d_model)
        mask = torch.ones(2, 1, dtype=torch.bool)

        with torch.no_grad():
            residual, gate = model._sparse_trajectory_summary(x1, mask, encoded)
        self.assertTrue(torch.isfinite(residual).all())
        self.assertTrue(torch.isfinite(gate).all())

    def test_masked_pooling_requires_aligned_masks(self):
        model = _model(masked=True).eval()
        x1 = torch.randn(2, 6, 2)
        x2 = torch.randn(2, 6, 10)
        mask1 = torch.ones(2, 6, dtype=torch.bool)
        mask2 = torch.ones(2, 5, dtype=torch.bool)

        with self.assertRaisesRegex(ValueError, "aligned"):
            model(x1, mask1, x2[:, :5], mask2)


if __name__ == "__main__":
    unittest.main()
