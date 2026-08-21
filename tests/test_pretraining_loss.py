import unittest

import torch

from tmi.models.loss import MaskedMSELoss


class PretrainingLossTests(unittest.TestCase):
    def test_masked_mse_accepts_runner_padding_mask(self):
        prediction = torch.tensor([[[1.0], [3.0], [9.0]]])
        target = torch.tensor([[[2.0], [1.0], [0.0]]])
        objective_mask = torch.tensor([[[True], [True], [False]]])
        padding_mask = torch.tensor([[True, True, False]])

        loss = MaskedMSELoss(reduction='none')(
            prediction,
            target,
            objective_mask,
            padding_mask,
        )

        self.assertTrue(torch.equal(loss, torch.tensor([1.0, 4.0])))

    def test_padding_is_not_selected_by_objective_mask(self):
        prediction = torch.tensor([[[1.0], [999.0]]])
        target = torch.tensor([[[0.0], [0.0]]])
        objective_mask = torch.tensor([[[True], [False]]])
        padding_mask = torch.tensor([[True, False]])

        loss = MaskedMSELoss(reduction='none')(
            prediction, target, objective_mask, padding_mask)
        self.assertTrue(torch.equal(loss, torch.tensor([1.0])))


if __name__ == '__main__':
    unittest.main()
