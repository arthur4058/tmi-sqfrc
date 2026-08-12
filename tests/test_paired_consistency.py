import unittest

import torch

from tmi.paired_consistency import (
    collate_paired_multirate,
    paired_consistency_terms,
)


class PairedConsistencyTest(unittest.TestCase):
    @staticmethod
    def loss_module(logits, targets):
        return torch.nn.functional.cross_entropy(
            logits, targets.squeeze(-1).long(), reduction="none")

    def test_identical_views_have_zero_consistency(self):
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
        targets = torch.tensor([[0], [1]])
        terms = paired_consistency_terms(
            logits, logits, targets, self.loss_module, epoch=10,
            consistency_weight=0.2, ramp_epochs=10)
        self.assertAlmostEqual(terms.consistency.item(), 0.0, places=7)
        self.assertAlmostEqual(terms.ramp, 1.0)
        terms.total.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_consistency_is_symmetric_and_positive(self):
        sparse = torch.tensor([[3.0, 0.0], [0.5, 1.5]], requires_grad=True)
        dense = torch.tensor([[0.0, 3.0], [1.5, 0.5]], requires_grad=True)
        targets = torch.tensor([[0], [1]])
        forward = paired_consistency_terms(
            sparse, dense, targets, self.loss_module, epoch=5,
            ramp_epochs=10)
        reverse = paired_consistency_terms(
            dense, sparse, targets, self.loss_module, epoch=5,
            sparse_supervised_weight=0.3, ramp_epochs=10)
        self.assertGreater(forward.consistency.item(), 0.0)
        self.assertAlmostEqual(
            forward.consistency.item(), reverse.consistency.item(), places=7)
        self.assertAlmostEqual(forward.ramp, 0.5)
        self.assertTrue(torch.isfinite(forward.total))

    def test_collate_preserves_rate_specific_lengths(self):
        batch = [
            (
                torch.randn(3, 2), torch.randn(3, 4),
                torch.randn(7, 2), torch.randn(7, 4),
                torch.tensor([1]), torch.tensor(11),
            ),
            (
                torch.randn(4, 2), torch.randn(4, 4),
                torch.randn(6, 2), torch.randn(6, 4),
                torch.tensor([2]), torch.tensor(12),
            ),
        ]
        result = collate_paired_multirate(batch)
        sx1, sx2, sm1, sm2, dx1, dx2, dm1, dm2, targets, _ = result
        self.assertEqual(sx1.shape[1], 4)
        self.assertEqual(sx2.shape[1], 4)
        self.assertEqual(dx1.shape[1], 7)
        self.assertEqual(dx2.shape[1], 7)
        self.assertEqual(sm1.shape[1], 4)
        self.assertEqual(dm1.shape[1], 7)
        self.assertEqual(targets.shape, (2, 1))

    def test_invalid_weights_are_rejected(self):
        logits = torch.zeros(2, 2)
        targets = torch.zeros(2, 1, dtype=torch.long)
        with self.assertRaises(ValueError):
            paired_consistency_terms(
                logits, logits, targets, self.loss_module, epoch=0,
                sparse_supervised_weight=1.1)


if __name__ == "__main__":
    unittest.main()
