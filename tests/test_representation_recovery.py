import unittest

import torch

from tmi.models.models import DualTSTransformerEncoderClassifier
from tmi.paired_consistency import representation_recovery_loss


def _hyperparams(feat_dim):
    return {
        "feat_dim": feat_dim,
        "max_len": 12,
        "d_model": 8,
        "n_heads": 2,
        "num_layers": 1,
        "dim_feedforward": 16,
        "dropout": 0.0,
        "pos_encoding": "fixed",
        "activation": "gelu",
        "norm": "LayerNorm",
        "freeze": False,
    }


def _model(recovery):
    return DualTSTransformerEncoderClassifier(
        _hyperparams(2), _hyperparams(4), num_classes=5, dropout=0.0,
        low_rate_representation_recovery=recovery,
        representation_recovery_hidden_dim=16,
        representation_recovery_gate_init=0.2,
    )


class RepresentationRecoveryTest(unittest.TestCase):
    def test_zero_initialized_adapter_preserves_baseline(self):
        torch.manual_seed(41)
        baseline = _model(False).eval()
        torch.manual_seed(41)
        candidate = _model(True).eval()
        x1 = torch.randn(3, 6, 2)
        x2 = torch.randn(3, 6, 4)
        mask = torch.ones(3, 6, dtype=torch.bool)

        with torch.no_grad():
            expected = baseline(x1, mask, x2, mask)
            actual = candidate(x1, mask, x2, mask)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_forward_exposes_recovered_representation(self):
        model = _model(True).train()
        x1 = torch.randn(4, 6, 2)
        x2 = torch.randn(4, 6, 4)
        mask = torch.ones(4, 6, dtype=torch.bool)

        logits, recovered, base = model(
            x1, mask, x2, mask, return_representation=True)
        self.assertEqual(logits.shape, (4, 5))
        self.assertEqual(recovered.shape, (4, 128))
        self.assertEqual(base.shape, (4, 128))
        logits.square().mean().backward()
        gradient = model.representation_recovery_adapter[-1].weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_recovery_loss_detaches_dense_target(self):
        sparse = torch.randn(4, 16, requires_grad=True)
        dense = torch.randn(4, 16, requires_grad=True)
        dense_logits = torch.tensor([
            [5.0, 0.0], [5.0, 0.0], [0.0, 5.0], [0.0, 5.0]
        ], requires_grad=True)
        loss, coverage = representation_recovery_loss(
            sparse, dense, dense_logits, confidence_threshold=0.5)
        loss.backward()

        self.assertGreater(coverage.item(), 0.0)
        self.assertIsNotNone(sparse.grad)
        self.assertIsNone(dense.grad)
        self.assertIsNone(dense_logits.grad)

    def test_low_confidence_teacher_disables_alignment(self):
        sparse = torch.randn(3, 8, requires_grad=True)
        dense = torch.randn(3, 8)
        dense_logits = torch.zeros(3, 5)
        loss, coverage = representation_recovery_loss(
            sparse, dense, dense_logits, confidence_threshold=0.9)

        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(coverage.item(), 0.0)

    def test_invalid_representation_shapes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "equal shape"):
            representation_recovery_loss(
                torch.randn(2, 8), torch.randn(2, 7), torch.randn(2, 5))


if __name__ == "__main__":
    unittest.main()
