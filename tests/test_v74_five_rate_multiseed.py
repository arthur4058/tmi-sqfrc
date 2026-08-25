import unittest

import numpy as np
import torch

from scripts.run_v74_five_rate_multiseed import (
    RateGeneralizedRelationNet,
    all_pairs,
    anchor_count,
    anchor_indices,
    choose_fusion,
    point_drop,
)


class V74FiveRateTest(unittest.TestCase):
    def test_anchor_counts_preserve_60_second_architecture(self):
        self.assertEqual(anchor_count(60), 6)
        self.assertEqual(anchor_count(30), 11)
        self.assertEqual(anchor_count(20), 16)
        self.assertEqual(anchor_count(10), 16)
        self.assertEqual(anchor_count(5), 16)

    def test_anchor_selection_uses_real_endpoints(self):
        indices = anchor_indices(61, 16)
        self.assertEqual(len(indices), 16)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 60)
        self.assertTrue(np.all(np.diff(indices) > 0))

    def test_six_points_still_use_all_fifteen_pairs(self):
        left, right = all_pairs(6)
        self.assertEqual(len(left), 15)
        self.assertEqual(len(set(zip(left, right))), 15)

    def test_forward_shapes(self):
        model = RateGeneralizedRelationNet(11, width=32, embedding=20, layers=1)
        values = torch.randn(4, 12, 11)
        mask = torch.zeros(4, 11)
        mask[:, :7] = 1
        logits, embedding = model(values, mask)
        self.assertEqual(tuple(logits.shape), (4, 5))
        self.assertEqual(tuple(embedding.shape), (4, 20))
        self.assertTrue(torch.isfinite(logits).all())

    def test_point_drop_preserves_endpoints(self):
        torch.manual_seed(7)
        mask = torch.ones(32, 6)
        dropped = point_drop(mask, 1.0)
        self.assertTrue(torch.equal(dropped[:, 0], torch.ones(32)))
        self.assertTrue(torch.equal(dropped[:, -1], torch.ones(32)))
        self.assertTrue(torch.equal(dropped.sum(1), torch.full((32,), 5.0)))

    def test_fusion_can_fall_back_to_baseline(self):
        labels = np.array([0, 1, 2, 0])
        base = np.eye(5)[labels] * .9 + .02
        expert = np.roll(base, 1, axis=1)
        selected = choose_fusion(base, expert, labels)
        self.assertEqual(selected["weight"], 0.0)


if __name__ == "__main__":
    unittest.main()
