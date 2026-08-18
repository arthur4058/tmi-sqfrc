import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_v2_multiseed


class V2MultiseedTests(unittest.TestCase):
    def test_generated_configs_change_only_seed_specific_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            with patch.object(run_v2_multiseed, "CONFIG_DIR", temporary):
                configs = run_v2_multiseed.build_seed_configs(42)
            self.assertEqual(set(configs), {"b0", "four_feature", "motion"})
            for config_path in configs.values():
                config = json.loads(config_path.read_text(encoding="utf-8"))
                self.assertEqual(config["seed"], 42)
                self.assertIn("seed42", config["experiment_name"])
                self.assertIn("seed42", config["output_dir"])
                self.assertEqual(config["data_name"], "geolife_five_rate_fixed_60s")
                self.assertTrue(config["use_separate_val"])
                self.assertEqual(config["val_ratio"], 0.0)

    def test_b0_template_matches_v2_protocol(self):
        template = run_v2_multiseed.B0_TEMPLATE
        self.assertEqual(template["epochs"], 200)
        self.assertEqual(template["patience"], 40)
        self.assertEqual(template["input_type"], "50%noise")
        self.assertEqual(template["motion_features"], [3, 4, 5, 8])
        self.assertFalse(template["sampling_quality_reliability"])


if __name__ == "__main__":
    unittest.main()
