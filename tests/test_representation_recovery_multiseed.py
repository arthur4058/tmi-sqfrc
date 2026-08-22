import unittest

from scripts.summarize_representation_recovery_multiseed import summary


class RepresentationRecoveryMultiseedTest(unittest.TestCase):
    def test_summary_uses_sample_standard_deviation(self):
        result = summary([1.0, 2.0, 3.0])

        self.assertEqual(result["mean"], 2.0)
        self.assertEqual(result["sample_std"], 1.0)


if __name__ == "__main__":
    unittest.main()
