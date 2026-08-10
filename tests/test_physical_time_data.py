import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

import tmi.datasets.dataset as dataset_module
from tmi.datasets.data import Normalizer
from tmi.datasets.dataset import DualBranchClassificationDataset


class PhysicalTimeDataTest(unittest.TestCase):
    def test_normalizer_preserves_raw_timestamp_channel(self):
        timestamps = [1_600_000_000.0, 1_600_000_005.0, 1_600_000_015.0]
        frame = pd.DataFrame({
            3: [1.0, 2.0, 3.0],
            4: [2.0, 4.0, 8.0],
            9: timestamps,
        })
        result = Normalizer("standardization_except_last").normalize(frame)
        np.testing.assert_array_equal(result.iloc[:, -1], timestamps)
        np.testing.assert_allclose(
            result.iloc[:, :-1].mean().to_numpy(), 0.0, atol=1e-7)

    def test_dataset_converts_epoch_timestamp_before_float32_collate(self):
        index = pd.Index([0, 0, 0])
        trajectory = pd.DataFrame(
            [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]], index=index)
        feature = pd.DataFrame([
            [0.1, 0.2, 0.3, 0.4, 1_600_000_000.0],
            [0.2, 0.3, 0.4, 0.5, 1_600_000_005.0],
            [0.3, 0.4, 0.5, 0.6, 1_600_000_015.0],
        ], index=index)
        labels = pd.DataFrame([1], index=[0])
        branch = lambda frame: SimpleNamespace(
            noise_feature_df=frame,
            clean_feature_df=frame,
            labels_df=labels,
        )
        data = SimpleNamespace(
            trajectory_data=branch(trajectory),
            feature_data=branch(feature),
        )
        previous = dataset_module.config
        dataset_module.config = {
            "input_type": "0%noise",
            "physical_time_multiscale": True,
        }
        try:
            dataset = DualBranchClassificationDataset(data, np.asarray([0]))
            _, x2, _, _ = dataset[0]
        finally:
            dataset_module.config = previous
        np.testing.assert_array_equal(
            x2[:, -1].numpy(), np.asarray([0.0, 5.0, 15.0]))


if __name__ == "__main__":
    unittest.main()
