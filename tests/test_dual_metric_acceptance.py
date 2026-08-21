import copy
import unittest

from scripts.verify_dual_metric_acceptance import verify_acceptance


def paired_result():
    return {
        "protocol": "geolife-class-aware-stacking-v8",
        "sampling_interval_seconds": 60,
        "test_fit_performed": False,
        "test_samples": 4918,
        "baseline": {"accuracy": 0.70, "macro_f1": 0.60},
        "class_aware_stacker": {"accuracy": 0.71, "macro_f1": 0.62},
        "delta_vs_baseline": {"accuracy": 0.01, "macro_f1": 0.02},
    }


class DualMetricAcceptanceTests(unittest.TestCase):
    def test_both_metrics_must_pass(self):
        decision = verify_acceptance(
            paired_result(), expected_samples=4918
        )
        self.assertTrue(decision["accepted"])
        self.assertEqual(
            decision["metric_passed"],
            {"accuracy": True, "macro_f1": True},
        )

    def test_accuracy_failure_rejects_result(self):
        result = paired_result()
        result["class_aware_stacker"]["accuracy"] = 0.709
        result["delta_vs_baseline"]["accuracy"] = 0.009
        self.assertFalse(verify_acceptance(result)["accepted"])

    def test_macro_f1_failure_rejects_result(self):
        result = paired_result()
        result["class_aware_stacker"]["macro_f1"] = 0.609
        result["delta_vs_baseline"]["macro_f1"] = 0.009
        self.assertFalse(verify_acceptance(result)["accepted"])

    def test_test_time_fit_is_rejected(self):
        result = paired_result()
        result["test_fit_performed"] = True
        with self.assertRaisesRegex(ValueError, "no test-time fit"):
            verify_acceptance(result)

    def test_mixed_or_stale_delta_is_rejected(self):
        result = paired_result()
        result["delta_vs_baseline"]["accuracy"] = 0.02
        with self.assertRaisesRegex(ValueError, "not paired consistently"):
            verify_acceptance(result)

    def test_wrong_rate_or_split_is_rejected(self):
        result = paired_result()
        with self.assertRaisesRegex(ValueError, "sampling interval"):
            verify_acceptance(result, expected_rate=30)
        with self.assertRaisesRegex(ValueError, "sample count"):
            verify_acceptance(result, expected_samples=5000)


if __name__ == "__main__":
    unittest.main()
