import unittest

from scripts.analyze_behavior_masks import evaluate_hypothesis, summarize_samples


def sample(ts_ratio, fs_ratio, duration, length=5, label=0):
    return {
        "label": label,
        "sequence_length": length,
        "segment_duration": 300.0,
        "ts_masked_points": ts_ratio * length,
        "ts_mask_ratio": ts_ratio,
        "ts_mask_duration": duration,
        "ts_fully_masked": float(ts_ratio == 1.0),
        "fs_mask_ratio": fs_ratio,
        "fs_mask_duration": duration,
        "fs_fully_masked_fraction": float(fs_ratio == 1.0),
    }


class BehaviorMaskDiagnosisTests(unittest.TestCase):
    def test_summarize_samples(self):
        result = summarize_samples([
            sample(0.2, 0.3, 60.0),
            sample(1.0, 0.5, 300.0),
        ])
        self.assertEqual(result["num_samples"], 2)
        self.assertAlmostEqual(result["avg_ts_mask_ratio"], 0.6)
        self.assertAlmostEqual(result["avg_fs_mask_ratio"], 0.4)
        self.assertAlmostEqual(result["fully_masked_ts_sample_ratio"], 0.5)

    def test_hypothesis_goes_on_low_rate_ratio_increase(self):
        overall = []
        for rate, ratio in ((5, 0.10), (10, 0.12), (30, 0.30), (60, 0.40)):
            row = summarize_samples([sample(ratio, ratio, ratio * 300)])
            row["rate"] = rate
            overall.append(row)
        result = evaluate_hypothesis(overall, [], 300.0)
        self.assertEqual(result["verdict"], "GO")
        self.assertTrue(result["criteria"]["A"])

    def test_hypothesis_stops_without_evidence(self):
        overall = []
        for rate in (5, 10, 30, 60):
            row = summarize_samples([sample(0.10, 0.10, 20.0)])
            row["rate"] = rate
            overall.append(row)
        result = evaluate_hypothesis(overall, [], 300.0)
        self.assertEqual(result["verdict"], "STOP")


if __name__ == "__main__":
    unittest.main()
