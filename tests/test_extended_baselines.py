"""CPU-only checks for the saved-scene extended baseline evaluation."""

import math
from types import SimpleNamespace
import unittest

import numpy as np

from satellite_rl.evaluation.extended_baselines import (
    SUMMARY_FIELDS, random_action, summarize, validate_row,
)


class ExtendedBaselinesTests(unittest.TestCase):
    def test_random_action_uses_saved_seed_and_original_power_grid(self):
        env = SimpleNamespace(beam_idx=17)
        first = random_action(env, np.random.RandomState(91001))
        again = random_action(env, np.random.RandomState(91001))
        self.assertEqual(first, again)
        self.assertEqual(first[:3], [1, 32, 6])
        self.assertAlmostEqual(10 ** (first[3] / 10), 15.0)

    def test_random_summary_uses_two_scenes_not_six_actions_as_units(self):
        rows = []
        for policy, scene, seeds, values in (
            ("four_color", 1, (None,), (1,)),
            ("four_color", 2, (None,), (3,)),
            ("random", 1, (10, 20, 30), (0, 0, 6)),
            ("random", 2, (10, 20, 30), (4, 4, 4)),
        ):
            for seed, value in zip(seeds, values):
                rows.append({"policy": policy, "scene_id": scene,
                             "action_seed": seed,
                             **{field: value for field in SUMMARY_FIELDS}})
        result = summarize(rows, scene_ids=(1, 2), action_seeds=(10, 20, 30))
        self.assertEqual(result["random"]["si_mean"]["mean"], 3.0)
        self.assertAlmostEqual(result["random"]["si_mean"]["std"], math.sqrt(2.0))
        self.assertEqual(result["four_color"]["si_mean"]["mean"], 2.0)
        with self.assertRaisesRegex(ValueError, "Incomplete or duplicated"):
            summarize(rows + rows[:1], scene_ids=(1, 2), action_seeds=(10, 20, 30))

    def test_terminal_and_resource_accounts_are_independent_of_macro_values(self):
        row = {
            "final_rate_bps_per_beam": [0.0, 10.0, 20.0],
            "required_rate_bps_per_beam": [10.0, 10.0, 10.0],
            "satisfaction_per_beam": [0.0, 1.0, 1.0],
            "sgm_per_beam": [0.0, 0.5, 0.5],
            "si_sum": 2.0, "si_mean": 2 / 3, "sgm_mean": 1 / 3,
            "throughput_bps": 30.0, "total_power_w": 30.0,
            "requested_power_w_total": 30.0,
            "requested_slots_total": 4, "actual_slots_total": 3,
            "out_slots_total": 1, "occupied_slots_total": 0,
            "allocated_slots": 3, "steps": 3, "served_beams": 3,
            "completed": True, "power_exceeded": False,
        }
        validate_row(row, total_beams=3)
        wrong = {**row, "si_mean": 1.0}
        with self.assertRaisesRegex(ValueError, "SI mean"):
            validate_row(wrong, total_beams=3)
        wrong = {**row, "requested_slots_total": 5}
        with self.assertRaisesRegex(ValueError, "conserve slots"):
            validate_row(wrong, total_beams=3)


if __name__ == "__main__":
    unittest.main()
