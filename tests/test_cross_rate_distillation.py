import unittest

import torch

from tmi.distillation import (
    collate_cross_rate_superv,
    cross_rate_distillation_terms,
)
from tmi.models.loss import NoFussCrossEntropyLoss


class CrossRateDistillationTest(unittest.TestCase):
    def test_collate_keeps_dense_and_sparse_lengths_separate(self):
        batch = []
        for sample_id, sparse_len, dense_len in ((1, 3, 8), (2, 4, 10)):
            batch.append((
                torch.randn(sparse_len, 2, dtype=torch.float64),
                torch.randn(sparse_len, 4, dtype=torch.float64),
                torch.randn(dense_len, 2, dtype=torch.float64),
                torch.randn(dense_len, 4, dtype=torch.float64),
                torch.tensor([sample_id % 2]),
                torch.tensor(sample_id),
            ))
        output = collate_cross_rate_superv(batch)
        self.assertEqual(output[0].shape, (2, 4, 2))
        self.assertEqual(output[4].shape, (2, 10, 2))
        self.assertEqual(output[2].sum(dim=1).tolist(), [3, 4])
        self.assertEqual(output[6].sum(dim=1).tolist(), [8, 10])
        self.assertEqual(output[0].dtype, torch.float32)
        self.assertEqual(output[4].dtype, torch.float32)

    def test_distillation_is_finite_and_backpropagates(self):
        student_logits = torch.randn(3, 5, requires_grad=True)
        teacher_logits = torch.randn(3, 5)
        student_features = torch.randn(3, 8, requires_grad=True)
        teacher_features = torch.randn(3, 8)
        targets = torch.tensor([[0], [1], [2]])
        student_mask = torch.tensor([
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 0, 0, 0],
        ], dtype=torch.bool)
        teacher_mask = torch.ones(3, 8, dtype=torch.bool)
        terms = cross_rate_distillation_terms(
            student_logits,
            teacher_logits,
            student_features,
            teacher_features,
            targets,
            NoFussCrossEntropyLoss(reduction='none'),
            student_mask,
            teacher_mask,
        )
        self.assertTrue(torch.isfinite(terms.total))
        self.assertLess(terms.density_ratio.item(), 1.0)
        terms.total.backward()
        self.assertIsNotNone(student_logits.grad)
        self.assertIsNotNone(student_features.grad)

    def test_teacher_confidence_term_is_finite(self):
        student_logits = torch.zeros(2, 5, requires_grad=True)
        teacher_logits = torch.tensor([
            [8.0, -2.0, -2.0, -2.0, -2.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        features = torch.randn(2, 8)
        mask = torch.ones(2, 4, dtype=torch.bool)
        terms = cross_rate_distillation_terms(
            student_logits,
            teacher_logits,
            features,
            features,
            torch.tensor([[0], [0]]),
            NoFussCrossEntropyLoss(reduction='none'),
            mask,
            mask,
        )
        self.assertGreater(terms.teacher_confidence.item(), 0.2)
        self.assertTrue(torch.isfinite(terms.logits))

    def test_density_capped_reduces_sparse_teacher_pressure(self):
        student_logits = torch.tensor([
            [2.0, 0.0, -1.0],
            [1.0, 0.5, -0.5],
        ], requires_grad=True)
        teacher_logits = torch.tensor([
            [0.0, 3.0, -1.0],
            [0.0, 2.0, -1.0],
        ])
        features = torch.randn(2, 6)
        student_mask = torch.tensor([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
        ], dtype=torch.bool)
        teacher_mask = torch.ones(2, 8, dtype=torch.bool)
        common = dict(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            student_features=features,
            teacher_features=-features,
            targets=torch.tensor([[0], [1]]),
            loss_module=NoFussCrossEntropyLoss(reduction='none'),
            student_padding_mask=student_mask,
            teacher_padding_mask=teacher_mask,
        )
        sparse_weighted = cross_rate_distillation_terms(
            **common, density_mode='sparsity')
        density_capped = cross_rate_distillation_terms(
            **common, density_mode='density_capped')
        self.assertLess(
            density_capped.logits.item(), sparse_weighted.logits.item())
        self.assertLess(
            density_capped.features.item(), sparse_weighted.features.item())

    def test_unknown_density_mode_is_rejected(self):
        logits = torch.randn(1, 3, requires_grad=True)
        features = torch.randn(1, 4)
        mask = torch.ones(1, 2, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "Unknown density mode"):
            cross_rate_distillation_terms(
                logits,
                logits.detach(),
                features,
                features,
                torch.tensor([[0]]),
                NoFussCrossEntropyLoss(reduction='none'),
                mask,
                mask,
                density_mode='invalid',
            )


if __name__ == '__main__':
    unittest.main()
