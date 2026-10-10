"""Bounded actual geometry/physical-record replay for archived D12A inputs."""

import copy
import tempfile
from pathlib import Path
import unittest

import numpy as np

from scripts.stage1_drl_r04.strict_audit import audit_scene_row, scenes_from_npz
from satellite_rl.coverage.drl_r04_core import DrlR04Env
from satellite_rl.coverage.drl_r04_runner import _terminal_physics
from satellite_rl.coverage.scenes import CoverageDataset, CoverageScene
from satellite_rl.coverage.variable_dataset import save_variable_dataset


class StrictAuditTests(unittest.TestCase):
    def test_compressed_real_small_scene_and_service_algebra(self):
        points = np.asarray([[45.0, 10.0], [45.01, 10.01],
                             [46.0, 11.0], [46.01, 11.01]], dtype=float)
        scene = CoverageScene("strict-small", points,
                              np.asarray([120e6, 180e6, 210e6, 170e6]),
                              np.empty((0, 2)), ["a", "b", "c", "d"], {}, "source-small")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.npz"
            save_variable_dataset(CoverageDataset([scene], {"partition_role": "test"}), path)
            restored = scenes_from_npz(path.read_bytes())
        self.assertEqual(len(restored), 1)
        env = DrlR04Env(scene)
        env.reset()
        env.step(0)
        row = env.result()
        row.update(scene_id=scene.scene_id, scene_sha256=scene.source_hash,
                   demand_multiplier=1.0, algorithm_seconds=.1)
        row["terminal_physics"] = _terminal_physics(env)
        physical = sum(value["timing"]["total_seconds"] for value in
                       row["terminal_physics"]["layouts_by_sha256"].values())
        row["total_seconds"] = row["algorithm_seconds"] + physical
        errors, facts = audit_scene_row(restored[0], row)
        self.assertEqual(errors, [])
        self.assertTrue(facts["best"]["N_lt_50"])
        self.assertEqual(facts["best"]["N_eq_200"], False)
        self.assertIsNotNone(facts["best"]["aircraft_TDMA_carried_fraction"])
        measured = row["terminal_physics"]["layouts_by_sha256"][
            row["terminal_physics"]["layout_aliases"]["best"]]
        demands = [beam["demand_bps"] for beam in measured["beams"] if beam["demand_bps"] > 0]
        mean_demand = sum(demands) / len(demands)
        self.assertEqual(facts["best"]["positive_band_0p8_1p2"],
                         sum(.8 * mean_demand <= d <= 1.2 * mean_demand for d in demands))
        self.assertEqual(facts["best"]["capacity_demand_ratio_band_0p8_1p2"],
                         sum(.8 <= beam["capacity_bps"] / beam["demand_bps"] <= 1.2
                             for beam in measured["beams"] if beam["demand_bps"] > 0))
        broken = dict(row)
        broken["best_layout"] = dict(row["best_layout"])
        broken["best_layout"]["assignment"] = [-1] * len(scene.demands)
        errors, _ = audit_scene_row(restored[0], broken)
        self.assertTrue(any("best_" in error for error in errors))
        altered = copy.deepcopy(row)
        physical = altered["terminal_physics"]
        best = physical["layouts_by_sha256"][physical["layout_aliases"]["best"]]
        best["beams"][0]["sgm"] += .1
        errors, _ = audit_scene_row(restored[0], altered)
        self.assertIn("best_beam_SGM_formula_differs", errors)
        altered = copy.deepcopy(row)
        best = altered["terminal_physics"]["layouts_by_sha256"][
            altered["terminal_physics"]["layout_aliases"]["best"]]
        best["aircraft_service"]["airtime_fraction"][0] += .1
        errors, _ = audit_scene_row(restored[0], altered)
        self.assertIn("aircraft_TDMA_share_not_one", errors)
        altered = copy.deepcopy(row)
        altered["total_seconds"] += .1
        errors, _ = audit_scene_row(restored[0], altered)
        self.assertIn("algorithm_plus_physical_total_time_differs", errors)
        training = dict(row)
        training.pop("algorithm_seconds")
        training.pop("total_seconds")
        errors, facts = audit_scene_row(restored[0], training, require_total_timing=False)
        self.assertEqual(errors, [])
        self.assertEqual(facts["end_to_end_timing_scope"],
                         "not_recorded_for_training_terminal_not_filled")
        errors, _ = audit_scene_row(restored[0], training)
        self.assertIn("algorithm_plus_physical_total_time_differs", errors)


if __name__ == "__main__":
    unittest.main()
