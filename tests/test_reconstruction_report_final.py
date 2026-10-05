"""Small, source-free checks of S1-03-R1 final analysis semantics."""

import unittest

from satellite_rl.coverage.report_final import analyze_formal, distribution, flatten


def row(route, quality, initial=1.0):
    def metrics(q):
        return {"quality": q, "coverage_users": 1.0, "coverage_demand": 1.0,
                "subject_cv": .2, "load_cv": .2, "subject_in_band_beams": 90,
                "subject_beams": 98, "edge_beam_count": 2, "empty_beams": 0,
                "target_met": True, "coverage_100": True}
    return {"scene_id": "scene", "scene_sha256": "hash", "scene_index": 0,
            "training_seed": 51001, "model_step": 262144, "initialization_valid": True,
            "stop_reason": "macro_budget", "macro_steps": 2, "metrics": metrics(quality),
            "initial_metrics": metrics(initial), "initialization_seconds": None,
            "beam_load_demand_bps": [1.0] * 100, "best_quality_diagnostic": quality + .1,
            "policy_seconds": .1, "reconstruction_seconds": .2,
            "final_check_seconds": .01, "total_seconds": .31}


class FinalReportTests(unittest.TestCase):
    def test_missing_initialization_time_remains_missing(self):
        keyed = {}
        for route in ("R0", "R1", "R2", "R3", "P1"):
            keyed[route] = {("scene", "hash"): flatten(row(route, 2.0 if route == "P1" else 1.5), route)}
        _, pairs, result = analyze_formal(keyed)
        self.assertEqual(result["routes"]["P1"]["timing"]["initialization_seconds"]["n"], 0)
        self.assertIsNone(result["routes"]["P1"]["timing"]["initialization_seconds"]["mean"])
        self.assertEqual(pairs[0]["P1_minus_R3_quality"], .5)
        self.assertEqual(pairs[0]["P1_minus_initial_quality"], 1.0)
        self.assertEqual(result["routes"]["P1"]["target_failures_all_scenes"], 0)

    def test_distribution_linear_p95(self):
        self.assertEqual(distribution([0, 10])["p95"], 9.5)
        self.assertEqual(distribution([None, float("nan")])["n"], 0)


if __name__ == "__main__":
    unittest.main()
