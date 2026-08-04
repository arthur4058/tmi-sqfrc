import unittest

import numpy as np

from tmi.data_preprocess.s2_user_disjoint_split import (
    find_user_split,
    validate_user_disjoint_split,
)


class UserDisjointSplitTest(unittest.TestCase):
    def test_user_split_is_disjoint_deterministic_and_complete(self):
        users = np.repeat(np.arange(20), 10)
        labels = np.tile(np.arange(5), 40)
        first = find_user_split(labels, users, seed=42, trials=500)
        second = find_user_split(labels, users, seed=42, trials=500)
        self.assertTrue(validate_user_disjoint_split(first, labels, users))
        for split in ("train", "val", "test"):
            np.testing.assert_array_equal(first[split], second[split])


if __name__ == "__main__":
    unittest.main()
