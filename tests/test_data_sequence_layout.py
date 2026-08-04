import unittest

import numpy as np

from tmi.datasets.data import _as_object_sequence_matrix


class SequenceArrayLayoutTest(unittest.TestCase):
    def test_dense_multichannel_array_becomes_object_matrix(self):
        dense = np.arange(3 * 2 * 5).reshape(3, 2, 5).astype(object)
        actual = _as_object_sequence_matrix(dense, channels=2)

        self.assertEqual(actual.shape, (3, 2))
        self.assertEqual(actual.dtype, object)
        self.assertNotEqual(actual[1, 0].dtype, object)
        np.testing.assert_array_equal(actual[1, 0], dense[1, 0])

    def test_dense_single_channel_array_becomes_object_column(self):
        dense = np.arange(3 * 5).reshape(3, 5).astype(object)
        actual = _as_object_sequence_matrix(dense, channels=1)

        self.assertEqual(actual.shape, (3, 1))
        self.assertEqual(actual.dtype, object)
        self.assertNotEqual(actual[2, 0].dtype, object)
        np.testing.assert_array_equal(actual[2, 0], dense[2])

    def test_existing_object_matrix_is_unchanged(self):
        existing = np.empty((2, 2), dtype=object)
        existing[0, 0] = np.array([1, 2])
        existing[0, 1] = np.array([3, 4])
        existing[1, 0] = np.array([5])
        existing[1, 1] = np.array([6])

        actual = _as_object_sequence_matrix(existing, channels=2)

        self.assertIs(actual, existing)

    def test_object_vector_becomes_single_channel_matrix(self):
        existing = np.empty(2, dtype=object)
        existing[0] = np.array([True, False])
        existing[1] = np.array([False])

        actual = _as_object_sequence_matrix(existing, channels=1)

        self.assertEqual(actual.shape, (2, 1))
        np.testing.assert_array_equal(actual[0, 0], existing[0])


if __name__ == "__main__":
    unittest.main()
