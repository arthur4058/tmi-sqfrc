import unittest

import numpy as np
import torch

from scripts.train_pair_relation_sparse_v64 import PairRelationSparseNet
from scripts.train_relation_pointdrop_v74 import point_drop, select_aligned


class SparseObservationRecoveryTests(unittest.TestCase):
    def test_point_drop_preserves_endpoints_and_drops_at_most_one(self):
        torch.manual_seed(7)
        mask = torch.ones(128, 6)
        dropped = point_drop(mask, 1.0)
        self.assertTrue(torch.equal(dropped[:, 0], torch.ones(128)))
        self.assertTrue(torch.equal(dropped[:, -1], torch.ones(128)))
        self.assertTrue(torch.equal((mask-dropped).sum(1), torch.ones(128)))

    def test_short_segments_are_not_dropped(self):
        mask = torch.tensor([[1., 1., 1., 0., 0., 0.]])
        self.assertTrue(torch.equal(point_drop(mask, 1.0), mask))

    def test_masked_points_cannot_change_prediction(self):
        torch.manual_seed(11)
        model = PairRelationSparseNet(width=32, embedding=24, layers=1, dropout=0.0)
        model.eval()
        mask = torch.tensor([[1., 1., 1., 0., 0., 0.]])
        first = torch.randn(1, 12, 6)
        second = first.clone()
        second[:, :, 3:] = 1000*torch.randn(1, 12, 3)
        with torch.no_grad():
            logits_first, _ = model(first, mask)
            logits_second, _ = model(second, mask)
        torch.testing.assert_close(logits_first, logits_second)

    def test_aligned_view_uses_repeatable_python_rng(self):
        clean = np.zeros((8, 2, 3), dtype=np.float32)
        noisy = np.ones_like(clean)
        mask = np.ones((8, 3), dtype=np.float32)
        labels = np.arange(8) % 5
        users = np.arange(8)
        mean = np.zeros(2, dtype=np.float32)
        std = np.ones(2, dtype=np.float32)
        first = select_aligned((clean, noisy, mask, labels, users), 10086, mean, std)[0]
        second = select_aligned((clean, noisy, mask, labels, users), 10086, mean, std)[0]
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
