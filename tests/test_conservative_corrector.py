import unittest

import torch

from tmi.models.conservative_corrector import (
    ConservativeBusCarCorrector,
    sparsity_gate,
    uncertainty_gate,
)


class ConservativeCorrectorTests(unittest.TestCase):
    def test_sparse_gate_endpoints(self):
        values = sparsity_gate(torch.tensor([5.0, 10.0, 15.0]))
        torch.testing.assert_close(values, torch.tensor([1.0, 0.5, 0.0]))

    def test_uncertainty_gate_decreases_with_margin(self):
        values = uncertainty_gate(torch.tensor([0.0, 1.0, 4.0]))
        self.assertGreater(float(values[0]), float(values[1]))
        self.assertGreater(float(values[1]), float(values[2]))

    def test_only_bus_and_car_are_changed_and_delta_is_bounded(self):
        model = ConservativeBusCarCorrector(input_dim=19, alpha=0.4)
        with torch.no_grad():
            model.network[-1].bias.fill_(10.0)
        base = torch.randn(8, 5)
        corrected, delta, _, _ = model(
            torch.randn(8, 19), base, torch.full((8,), 5.0))
        torch.testing.assert_close(corrected[:, 0], base[:, 0])
        torch.testing.assert_close(corrected[:, 1], base[:, 1])
        torch.testing.assert_close(corrected[:, 4], base[:, 4])
        torch.testing.assert_close(corrected[:, 2] - base[:, 2], delta)
        torch.testing.assert_close(corrected[:, 3] - base[:, 3], -delta)
        self.assertLessEqual(float(delta.abs().max()), 0.4 + 1e-6)

    def test_high_rate_is_invariant(self):
        model = ConservativeBusCarCorrector(input_dim=19, alpha=1.0)
        base = torch.randn(4, 5)
        corrected, delta, sparse, _ = model(
            torch.randn(4, 19), base, torch.full((4,), 20.0))
        torch.testing.assert_close(sparse, torch.zeros_like(sparse))
        torch.testing.assert_close(delta, torch.zeros_like(delta))
        torch.testing.assert_close(corrected, base)


if __name__ == "__main__":
    unittest.main()
