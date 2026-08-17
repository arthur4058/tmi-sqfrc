import unittest

import numpy as np

from scripts.evaluate_class_aware_stacker import (
    class_power_weights,
    cross_validated_stacker,
    predict_from_parameters,
    probability_features,
)


class ClassAwareStackerTests(unittest.TestCase):
    def test_probability_features_are_finite(self):
        probabilities = [
            np.array([[1.0, 0.0], [0.25, 0.75]]),
            np.array([[0.9, 0.1], [0.4, 0.6]]),
        ]
        features = probability_features(probabilities)
        self.assertEqual(features.shape, (2, 4))
        self.assertTrue(np.isfinite(features).all())

    def test_class_power_weights(self):
        targets = np.array([0, 0, 0, 1])
        unweighted = class_power_weights(targets, 0.0)
        weighted = class_power_weights(targets, 0.5)
        self.assertTrue(np.allclose(unweighted, 1.0))
        self.assertGreater(weighted[-1], weighted[0])

    def test_parameter_prediction_is_normalized(self):
        features = np.array([[1.0, -1.0], [-1.0, 1.0]])
        coefficients = np.array([[1.0, 0.0], [0.0, 1.0]])
        probability = predict_from_parameters(
            features, coefficients, np.zeros(2)
        )
        self.assertTrue(np.allclose(probability.sum(axis=1), 1.0))
        self.assertEqual(probability.argmax(axis=1).tolist(), [0, 1])

    def test_cross_validation_is_deterministic(self):
        rng = np.random.default_rng(7)
        targets = np.repeat(np.arange(3), 30)
        signal = np.eye(3)[targets]
        probabilities = []
        for noise in (0.15, 0.25, 0.35):
            logits = signal + rng.normal(0.0, noise, signal.shape)
            logits = np.exp(logits)
            probabilities.append(logits / logits.sum(axis=1, keepdims=True))
        features = probability_features(probabilities)
        first, _, first_model = cross_validated_stacker(
            features, targets, c_values=(0.01,),
            balance_powers=(0.0,), folds=3,
        )
        second, _, second_model = cross_validated_stacker(
            features, targets, c_values=(0.01,),
            balance_powers=(0.0,), folds=3,
        )
        self.assertEqual(first, second)
        self.assertTrue(np.allclose(first_model.coef_, second_model.coef_))

    def test_invalid_balance_power_is_rejected(self):
        with self.assertRaises(ValueError):
            class_power_weights(np.array([0, 1]), 1.1)


if __name__ == "__main__":
    unittest.main()
