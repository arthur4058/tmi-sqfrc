import unittest

import numpy as np

from tmi.data_preprocess.utils import generate_mask_for_trj_using_KDE_SABM


def line_trajectory(n_points):
    return np.column_stack((
        np.linspace(39.0, 39.01, n_points),
        np.linspace(116.0, 116.01, n_points),
    ))


class SamplingAwareBehaviorMaskTests(unittest.TestCase):
    def test_target_ratio_and_visible_point_are_preserved(self):
        mask, metadata = generate_mask_for_trj_using_KDE_SABM(
            line_trajectory(5),
            np.full(5, 60.0),
            target_mask_ratio=0.30,
            mask_duration_seconds=30.0,
            return_metadata=True,
        )
        self.assertEqual(np.sum(mask == 0), 2)
        self.assertGreater(np.sum(mask == 1), 0)
        self.assertEqual(metadata['block_length_points'], 1)
        self.assertAlmostEqual(metadata['actual_mask_ratio'], 0.4)

    def test_physical_duration_changes_block_length(self):
        _, dense = generate_mask_for_trj_using_KDE_SABM(
            line_trajectory(20),
            np.full(20, 5.0),
            target_mask_ratio=0.30,
            mask_duration_seconds=30.0,
            return_metadata=True,
        )
        _, sparse = generate_mask_for_trj_using_KDE_SABM(
            line_trajectory(20),
            np.full(20, 60.0),
            target_mask_ratio=0.30,
            mask_duration_seconds=30.0,
            return_metadata=True,
        )
        self.assertEqual(dense['block_length_points'], 6)
        self.assertEqual(sparse['block_length_points'], 1)
        self.assertEqual(dense['target_mask_points'], 6)
        self.assertEqual(sparse['target_mask_points'], 6)

    def test_degenerate_coordinates_do_not_create_complete_mask(self):
        trajectory = np.repeat([[39.0, 116.0]], repeats=7, axis=0)
        mask = generate_mask_for_trj_using_KDE_SABM(
            trajectory,
            np.full(7, 60.0),
            target_mask_ratio=0.30,
            mask_duration_seconds=30.0,
        )
        self.assertEqual(np.sum(mask == 0), 2)
        self.assertEqual(np.sum(mask == 1), 5)

    def test_invalid_delta_time_length_is_rejected(self):
        with self.assertRaises(ValueError):
            generate_mask_for_trj_using_KDE_SABM(
                line_trajectory(5),
                np.full(4, 60.0),
            )

    def test_empty_segment_is_supported(self):
        mask, metadata = generate_mask_for_trj_using_KDE_SABM(
            np.empty((0, 2)),
            np.empty(0),
            return_metadata=True,
        )
        self.assertEqual(mask.size, 0)
        self.assertEqual(metadata['target_mask_points'], 0)


if __name__ == '__main__':
    unittest.main()
