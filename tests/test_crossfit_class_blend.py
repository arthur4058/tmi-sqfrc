import unittest

import numpy as np

from scripts.evaluate_crossfit_class_blend import (
    blend_by_class,
    build_crossfit_stacker,
    ensemble_predict,
    select_class_alphas,
)


class CrossfitClassBlendTests(unittest.TestCase):
    def test_class_blend_is_normalized(self):
        stable = np.array([[0.8, 0.2], [0.3, 0.7]])
        adaptive = np.array([[0.6, 0.4], [0.5, 0.5]])
        mixed = blend_by_class(stable, adaptive, [0.25, 0.75])
        self.assertTrue(np.allclose(mixed.sum(axis=1), 1.0))
        self.assertTrue((mixed >= 0.0).all())

    def test_zero_and_one_alphas_select_sources(self):
        stable = np.array([[0.8, 0.2]])
        adaptive = np.array([[0.6, 0.4]])
        self.assertTrue(np.allclose(
            blend_by_class(stable, adaptive, [0.0, 0.0]), stable
        ))
        self.assertTrue(np.allclose(
            blend_by_class(stable, adaptive, [1.0, 1.0]), adaptive
        ))

    def test_alpha_selection_is_deterministic(self):
        targets = np.repeat(np.arange(2), 20)
        stable = np.full((40, 2), 0.25)
        stable[np.arange(40), targets] = 0.75
        adaptive = stable[:, ::-1]
        first = select_class_alphas(stable, adaptive, targets)
        second = select_class_alphas(stable, adaptive, targets)
        self.assertTrue(np.array_equal(first, second))

    def test_crossfit_ensemble_shapes(self):
        rng = np.random.default_rng(11)
        targets = np.repeat(np.arange(3), 20)
        features = rng.normal(size=(60, 9))
        features[:, :3] += np.eye(3)[targets]
        oof, parameters = build_crossfit_stacker(
            features, targets, folds=3
        )
        prediction = ensemble_predict(features, parameters)
        self.assertEqual(oof.shape, (60, 3))
        self.assertEqual(prediction.shape, (60, 3))
        self.assertEqual(len(parameters), 3)
        self.assertTrue(np.allclose(prediction.sum(axis=1), 1.0))


if __name__ == "__main__":
    unittest.main()
