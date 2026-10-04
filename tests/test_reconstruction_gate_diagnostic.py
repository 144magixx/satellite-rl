"""Small raw-log mechanics for stopped S1-03 gate diagnosis."""
import unittest

from scripts.stage1_reconstruction import report_gate_diagnostic as diagnostic


def event(step, operation, accepted, reason, serial=0, identity=None):
    return {"env_step": step, "row": 0, "episode_serial": serial,
            "scene_index": 7, "requested_action_fields": {"operation": operation},
            "accepted": accepted, "reason": reason,
            "action_identity": identity or [2, 1, 2, 0, 0],
            "candidate_identity_sha256": "a" * 64}


class GateDiagnosticTests(unittest.TestCase):
    def test_prior_accepted_strata_and_reasons_are_distinct(self):
        rows = [event(1, "repartition2", True, "accepted"),
                event(2, "repartition2", False, "association outside actual satellite footprint"),
                event(3, "repartition2", False, "no_change"),
                event(4, "stop", False, "stop")]
        result = diagnostic.action_diagnosis(rows, maximum=4)
        self.assertEqual(result["windows"]["all"]["non_stop"], 3)
        self.assertEqual(result["windows"]["all"]["accepted_non_stop"], 1)
        strata = result["prior_acceptance_strata"]["all"]
        self.assertEqual(strata["no_prior_accepted"]["non_stop"], 1)
        self.assertEqual(strata["prior_accepted"]["non_stop"], 2)
        self.assertEqual(strata["prior_accepted"]["reason:no_change"], 1)
        self.assertEqual(strata["prior_accepted"][
            "category:geometry_or_budget_infeasible"], 1)
        self.assertEqual(result["repeats"]["same_menu_same_action_within_episode"], 3)

    def test_episode_change_resets_prior_and_repeat_identity(self):
        rows = [event(1, "assign", True, "accepted", serial=0),
                event(2, "assign", False, "no_change", serial=1)]
        result = diagnostic.action_diagnosis(rows, maximum=2)
        self.assertEqual(result["prior_acceptance_strata"]["all"][
            "no_prior_accepted"]["non_stop"], 2)
        self.assertEqual(result["repeats"].get(
            "same_menu_same_action_within_episode", 0), 0)

    def test_missing_metric_is_not_zero_and_zero_step_is_separate(self):
        rows = [{"terminal_env_step": 1, "macro_steps": 0, "stop_reason": "stop",
                 "metrics": {"quality": 1.2}, "initial_metrics": {"quality": 1.0},
                 "episode_return": .2},
                {"terminal_env_step": 2, "macro_steps": 2, "stop_reason": "macro_budget",
                 "metrics": {"quality": None}, "initial_metrics": {"quality": 1.0},
                 "episode_return": None}]
        result = diagnostic.episode_diagnosis(rows, maximum=2)["bins"][0]
        self.assertEqual(result["zero_step"], 1)
        self.assertEqual(result["quality_defined_count"], 1)
        self.assertIsNone(result["quality_mean"])
        summary = diagnostic.numerical_summary([1, None], 2)
        self.assertEqual(summary["missing_count"], 1)
        self.assertEqual(summary["median"], 1)
        self.assertEqual(summary["p95"], 1)

    def test_duplicate_or_missing_action_step_stops_analysis(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            diagnostic.action_diagnosis([event(1, "stop", False, "stop"),
                                         event(1, "stop", False, "stop")], maximum=2)


if __name__ == "__main__":
    unittest.main()
