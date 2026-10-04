"""Small gates for read-only S1-02 B result supplementation."""

import copy
import unittest

from scripts import report_stage1_balance_b_supplement as supplement


def row(scene_id="scene-0", source_sha="source-0"):
    return {
        "scene_id": scene_id, "scene_sha256": source_sha,
        "input": {"aircraft_ids": ["aircraft-0"], "positions": [[45.0, 5.0]],
                  "demands_bps": [1.0], "initial_centers": [[45.0, 5.0]]},
        "status": "completed", "metrics": {"coverage_users": 1.0},
    }


class BalanceBSupplementTests(unittest.TestCase):
    def test_raw_view_is_separate_and_never_mutates_best_result(self):
        saved = row()
        saved.update(final_positions=[[1.0, 1.0]], final_associations=[0],
                     beam_loads_bps=[1.0],
                     raw_terminal_metrics={"coverage_users": 0.5},
                     raw_terminal_positions=[[2.0, 2.0]],
                     raw_terminal_associations=[-1],
                     raw_terminal_beam_loads_bps=[0.0],
                     balance_diagnostics={"certified_regions": [],
                                          "qualified_edge_beam_ids": [7]})
        original = copy.deepcopy(saved)
        raw = supplement.raw_audit_view(saved)
        self.assertEqual(raw["metrics"]["coverage_users"], 0.5)
        self.assertEqual(raw["final_positions"], [[2.0, 2.0]])
        self.assertEqual(raw["balance_diagnostics"]["qualified_edge_beam_ids"], [])
        self.assertEqual(saved, original)

    def test_missing_raw_fields_are_rejected_for_adjusted_policy(self):
        saved = row()
        with self.assertRaisesRegex(ValueError, "lacks raw fields"):
            supplement.raw_audit_view(saved)
        saved["PPO_adjustment_performed"] = False
        self.assertIsNone(supplement.raw_audit_view(saved))

    def test_partial_requires_complete_menu_pair_and_same_input(self):
        a = row()
        b = row()
        aligned = supplement.align_results({"X0_search": [a], "X0_ppo": [b]}, 1,
                                           allow_partial=True)
        self.assertEqual(len(aligned), 2)
        with self.assertRaisesRegex(ValueError, "must be paired"):
            supplement.align_results({"X0_search": [a]}, 1, allow_partial=True)
        with self.assertRaisesRegex(ValueError, "exactly"):
            supplement.align_results({"X0_search": [a], "X0_ppo": [b]}, 1,
                                     allow_partial=False)

    def test_scene_source_or_aircraft_input_mismatch_is_rejected(self):
        a = row()
        for mutation in ({"scene_sha256": "different-source"},
                         {"input": {**a["input"], "demands_bps": [2.0]}},
                         {"scene_id": "different-scene"}):
            changed = {**copy.deepcopy(a), **mutation}
            with self.subTest(mutation=mutation):
                with self.assertRaisesRegex(ValueError, "differs"):
                    supplement.align_results({"X0_search": [a], "X0_ppo": [changed]},
                                             1, allow_partial=True)


if __name__ == "__main__":
    unittest.main()
