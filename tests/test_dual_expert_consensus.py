import unittest

import numpy as np
import torch

from scripts.evaluate_dual_expert_consensus import (
    choose_uniform_mix,
    pad_for_expert,
    softmax,
)


class DualExpertConsensusTest(unittest.TestCase):
    def test_pad_for_expert_preserves_values_and_marks_padding(self):
        features = torch.arange(24).reshape(2, 3, 4).float()
        masks = torch.tensor([[True, True, True], [True, True, False]])

        padded, padded_masks = pad_for_expert(features, masks, max_len=5)

        self.assertEqual(padded.shape, (2, 5, 4))
        self.assertEqual(padded_masks.shape, (2, 5))
        torch.testing.assert_close(padded[:, :3], features)
        self.assertFalse(padded_masks[:, 3:].any())

    def test_pad_for_expert_clips_long_batches(self):
        features = torch.ones(2, 7, 4)
        masks = torch.ones(2, 7, dtype=torch.bool)

        clipped, clipped_masks = pad_for_expert(features, masks, max_len=5)

        self.assertEqual(clipped.shape, (2, 5, 4))
        self.assertEqual(clipped_masks.shape, (2, 5))

    def test_mix_weight_is_selected_only_from_given_targets(self):
        targets = np.array([0, 1, 0, 1])
        baseline = softmax(np.array([[3, 0], [3, 0], [3, 0], [3, 0]]))
        expert = softmax(np.array([[0, 3], [0, 3], [0, 3], [0, 3]]))

        alpha = choose_uniform_mix(baseline, expert, targets)

        self.assertGreaterEqual(alpha, 0.0)
        self.assertLessEqual(alpha, 1.0)


if __name__ == "__main__":
    unittest.main()
