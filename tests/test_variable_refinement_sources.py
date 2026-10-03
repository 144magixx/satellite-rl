"""Frozen initial-allocation identity and common feasible baseline checks."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from satellite_rl.workbench.refinement_sources import source_rows
from satellite_rl.workbench.storage import digest
from satellite_rl.workbench.baselines import BaselinePolicy
from satellite_rl.power_refinement.config import default_config, validate_config, VARIABLE_BEAM_METHODS


class SavedAllocationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "results.json"
        self.row = {"scene": {"source_sha256": "a" * 64}, "repeat_index": 0,
                    "action_seed": 53001, "beam_count": 1, "beams": [{"beam_id": 0}],
                    "initial_allocation_seconds": 0.7, "inference_seconds": 0.2}

    def write(self, rows):
        self.path.write_text(json.dumps({"scenes": rows}), encoding="utf-8")
        return {"path": str(self.path), "sha256": digest(self.path), "group": "model",
                "scene_count": 1, "repeats": 1, "action_seed": 53001}

    def test_saved_full_time_and_identity(self):
        rows = source_rows(self.write([self.row]))
        row = rows[("a" * 64, 0)]
        self.assertEqual(row["inference_seconds"], 0.7)
        self.assertEqual(row["policy_inference_seconds"], 0.2)
        self.assertEqual(self.row["inference_seconds"], 0.2)

    def test_corruption_missing_time_and_duplicate_rejected(self):
        reference = self.write([self.row])
        self.path.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA"):
            source_rows(reference)
        invalid = deepcopy(self.row)
        del invalid["initial_allocation_seconds"]
        with self.assertRaisesRegex(ValueError, "耗时"):
            source_rows(self.write([invalid]))
        with self.assertRaisesRegex(ValueError, "重复"):
            source_rows(self.write([self.row, self.row]))

    def test_saved_seed_and_complete_grid_required(self):
        invalid = deepcopy(self.row)
        invalid["action_seed"] = 53002
        with self.assertRaisesRegex(ValueError, "种子"):
            source_rows(self.write([invalid]))
        with self.assertRaisesRegex(ValueError, "数量"):
            source_rows(self.write([]))

    def test_variable_route_keeps_frozen_budget(self):
        config = default_config()
        config["methods"] = list(VARIABLE_BEAM_METHODS)
        self.assertEqual(validate_config(config)["maxeval"], 2000)
        config["maxeval"] = 4000
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_resumed_copy_cannot_substitute_another_self_consistent_initial(self):
        from satellite_rl.power_refinement.runner import _verify_saved_initial
        reference = {"sha256": "frozen-source"}
        original = {"beams": [{"power_w": 10}], "metrics": {"final_si_mean": 0.5},
                    "initial_allocation_seconds": 0.7, "action_seed": 53001}
        copied = {**deepcopy(original), "source_evaluation": reference,
                  "initial_allocation_reused": True, "physics_setup_seconds": 0.2}
        _verify_saved_initial(copied, original, reference)
        copied["beams"][0]["power_w"] = 20
        copied["metrics"]["final_si_mean"] = 0.8
        with self.assertRaisesRegex(ValueError, "原始case"):
            _verify_saved_initial(copied, original, reference)


class VariableBaselineTests(unittest.TestCase):
    def test_out_of_domain_fixed_request_is_explicitly_projected(self):
        from types import SimpleNamespace
        env = SimpleNamespace(raw_env=object(), config={"environment": {
            "beam_count_max": 200, "power_levels_w": list(range(5, 51, 5))}})
        policy = BaselinePolicy("fixed", env)
        mask = {"frequency_slots": np.zeros((100, 10), dtype=bool),
                "power": np.zeros(10, dtype=bool)}
        mask["frequency_slots"][42, 2] = True
        mask["power"][1] = True
        with patch.dict("satellite_rl.workbench.baselines.POLICIES", {
                "fixed": lambda raw: [0, 192, 8, 10 * np.log10(30)]}):
            self.assertEqual(policy.act(None, action_mask=mask), (42, 2, 1))
        self.assertEqual(policy.last_action_info["source_frequency"], 192)
        self.assertEqual(policy.last_action_info["source_slots"], 8)
        self.assertTrue(policy.last_action_info["feasible_projection"])
        self.assertEqual(policy.metadata()["variant"], "workbench_feasible_fixed_v2")


if __name__ == "__main__":
    unittest.main()
