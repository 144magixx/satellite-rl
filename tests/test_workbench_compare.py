"""Scene pairing distinguishes an action-mask treatment from physical changes."""

from copy import deepcopy
import tempfile
import unittest
from unittest.mock import patch

from satellite_rl.workbench.config import default_config
from satellite_rl.workbench.service import Application


class WorkbenchCompareTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.application = Application(temporary.name, source_root=temporary.name)
        self.off = default_config()
        self.on = deepcopy(self.off)
        self.on["environment"]["action_mask"] = "joint_feasible"

    @staticmethod
    def scenes(mask_on, count=100):
        rows = []
        for index in range(count):
            scores = ([0.4, 0.6, 0.8] if index % 2 == 0 else [0.2]) if mask_on else [0.1, 0.3]
            for score in scores:
                rows.append({"scene": {"sha256": f"{index:064x}"},
                             "metrics": {"final_si_mean": score, "final_sgm_mean": score / 2}})
        return rows

    def compare(self, *, off=None, on=None, off_metadata=None, on_metadata=None, count=100):
        configs = {"off": off if off is not None else self.off,
                   "on": on if on is not None else self.on}
        results = {
            "off": {"summary": {"final_si_mean": 0.2},
                    "metadata": {"action_mode": "stochastic"} if off_metadata is None else off_metadata,
                    "scenes": self.scenes(False, count)},
            "on": {"summary": {"final_si_mean": 0.5},
                   "metadata": {"action_mode": "stochastic"} if on_metadata is None else on_metadata,
                   "scenes": self.scenes(True, count)},
        }
        details = {run_id: {"config": config, "job": {"name": run_id}}
                   for run_id, config in configs.items()}
        with patch.object(self.application, "run_detail", side_effect=details.__getitem__), \
             patch.object(self.application, "results", side_effect=results.__getitem__):
            return self.application.compare(["off", "on"])

    def test_known_mask_treatment_pairs_100_hashes_after_averaging_repeats(self):
        result = self.compare()
        self.assertEqual(len(result["paired_scene_hashes"]), 100)
        self.assertEqual(result["rows"][1]["paired_scenes"], 100)
        self.assertAlmostEqual(result["rows"][1]["final_si_mean_paired_delta"], 0.2)
        self.assertEqual(result["treatment_factors"]["action_mask"],
                         {"off": "none", "on": "joint_feasible"})
        self.assertTrue(any("最低资源预留" in warning and "配套熵" in warning and "不能全部归因" in warning
                            for warning in result["warnings"]))

    def test_missing_action_mask_is_equivalent_to_explicit_none(self):
        off = deepcopy(self.off)
        off["environment"].pop("action_mask")
        result = self.compare(off=off, on=self.off, count=2)
        self.assertEqual(result["rows"][1]["paired_scenes"], 2)
        self.assertEqual(result["treatment_factors"], {})

    def test_physical_environment_accounting_and_action_mode_still_block_pairing(self):
        mutations = {
            "power_budget": lambda c: c["environment"].update(power_budget_w=2500),
            "power_levels": lambda c: c["environment"].update(power_levels_w=[5, 10]),
            "basic_fraction": lambda c: c["environment"].update(basic_fraction=0.2),
            "projection": lambda c: c["environment"].update(action_mode="feasible_projection"),
            "power_accounting": lambda c: c["reward"].update(preset="legacy_original"),
            "unknown_environment_field": lambda c: c["environment"].update(new_physical_rule=True),
            "unknown_action_mask": lambda c: c["environment"].update(action_mask="unrecognized"),
            "unknown_mask_type": lambda c: c["environment"].update(action_mask=["none"]),
        }
        for name, mutate in mutations.items():
            with self.subTest(condition=name):
                changed = deepcopy(self.on)
                mutate(changed)
                result = self.compare(on=changed, count=2)
                self.assertEqual(result["rows"][1]["paired_scenes"], 0)
                self.assertEqual(result["paired_scene_hashes"], [])
                self.assertNotIn("final_si_mean_paired_delta", result["rows"][1])
        # An unknown mask remains unsafe even when both records use the same value.
        unknown = deepcopy(self.off)
        unknown["environment"]["action_mask"] = "unrecognized"
        self.assertEqual(self.compare(off=unknown, on=unknown, count=2)["rows"][1]["paired_scenes"], 0)

    def test_evaluation_metadata_takes_precedence_and_sample_alias_pairs(self):
        self.on["evaluation"]["action_mode"] = "deterministic"
        paired = self.compare(off_metadata={"action_mode": "sample"},
                              on_metadata={"action_mode": "stochastic"}, count=2)
        self.assertEqual(paired["rows"][1]["paired_scenes"], 2)
        blocked = self.compare(off_metadata={"action_mode": "deterministic"},
                               on_metadata={"action_mode": "stochastic"}, count=2)
        self.assertEqual(blocked["rows"][1]["paired_scenes"], 0)

    def test_evaluation_config_fallback_does_not_mix_deterministic_and_sampled(self):
        self.off["evaluation"]["action_mode"] = "deterministic"
        self.on["evaluation"]["action_mode"] = "stochastic"
        blocked = self.compare(off_metadata={}, on_metadata={}, count=2)
        self.assertEqual(blocked["rows"][1]["paired_scenes"], 0)
        self.on["evaluation"]["action_mode"] = "deterministic"
        paired = self.compare(off_metadata={}, on_metadata={}, count=2)
        self.assertEqual(paired["rows"][1]["paired_scenes"], 2)
        unknown = self.compare(off_metadata={"action_mode": "unrecognized"},
                               on_metadata={"action_mode": "unrecognized"}, count=2)
        self.assertEqual(unknown["rows"][1]["paired_scenes"], 0)


if __name__ == "__main__":
    unittest.main()
