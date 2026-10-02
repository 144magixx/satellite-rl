"""Finite-budget optimization behavior on the frozen link model."""

import unittest

import numpy as np

from satellite_rl.power_refinement import FixedAllocationPhysics, optimize_power
from test_power_refinement_physics import replay_scene


class OptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.problem, cls.initial = replay_scene()

    def test_none_is_identical(self):
        result = optimize_power(self.problem, self.initial, "none")
        np.testing.assert_array_equal(result["power_w"], self.initial)
        self.assertEqual(result["metrics"], result["initial_metrics"])

    def test_discrete_budget_and_stop_return_feasible_fallback(self):
        result = optimize_power(self.problem, self.initial, "discrete",
                                {"power_cap_w": float(self.initial.sum()),
                                 "max_evaluations": 30})
        self.assertLessEqual(result["metrics"]["power_w"], self.initial.sum() + 1e-7)
        self.assertGreaterEqual(result["metrics"]["final_sgm_mean"],
                                result["initial_metrics"]["final_sgm_mean"] - 1e-8)
        self.assertLessEqual(result["evaluations"], 30)
        self.assertEqual(result["termination_reason"], "evaluation_limit")

    def test_rejects_overspent_or_unknown_options(self):
        with self.assertRaises(ValueError):
            optimize_power(self.problem, self.initial, "none", {"power_cap_w": 1})
        with self.assertRaises(ValueError):
            optimize_power(self.problem, self.initial, "none", {"unknown": 1})

    def test_requested_stop_returns_unchanged_feasible_initial_solution(self):
        result = optimize_power(self.problem, self.initial, "discrete",
                                stop_requested=lambda: True)
        self.assertEqual(result["termination_reason"], "stop_requested")
        self.assertTrue(result["fallback_used"])
        self.assertFalse(result["converged"])
        np.testing.assert_array_equal(result["power_w"], self.initial)

    def test_slsqp_returns_real_feasible_result_under_short_budget(self):
        result = optimize_power(self.problem, self.initial, "slsqp",
                                {"power_cap_w": float(self.initial.sum()),
                                 "maxiter": 2, "max_evaluations": 80,
                                 "time_limit_seconds": 15})
        power = np.asarray(result["power_w"])
        ev = self.problem.evaluate(power)
        initial_ev = self.problem.evaluate(self.initial)
        self.assertLessEqual(power.sum(), self.initial.sum() + 1e-7)
        self.assertGreaterEqual(ev.si, initial_ev.si - 1e-8)
        self.assertGreaterEqual(ev.sgm, initial_ev.sgm - 1e-8)
        np.testing.assert_array_less(
            np.minimum(initial_ev.capacity_bps, 0.1 * self.problem.demand_bps)
            - ev.capacity_bps,
            np.full(100, 1e-8 * self.problem.demand_bps.max()))
        self.assertTrue(np.isfinite(power).all())

    def test_discrete_preserves_improvement_when_evaluation_limit_hits_mid_scan(self):
        problem = FixedAllocationPhysics(
            np.full(2, 40e6), np.array([0, 1]), np.zeros(2, dtype=int),
            np.ones(2, dtype=int), np.full(2, 1e-14), np.zeros((2, 2)),
            np.full(2, 1e-12), np.arange(2), 3000)
        initial = np.array([5.0, 5.0])
        result = optimize_power(problem, initial, "discrete", {
            "max_evaluations": 2, "power_cap_w": 20})
        self.assertEqual(result["termination_reason"], "evaluation_limit")
        self.assertFalse(result["converged"])
        self.assertGreater(result["metrics"]["final_si_mean"],
                           result["initial_metrics"]["final_si_mean"])

    def test_slsqp_can_use_more_than_50_w_when_only_total_budget_applies(self):
        problem = FixedAllocationPhysics(
            np.array([40e6]), np.array([0]), np.array([0]), np.array([1]),
            np.array([1e-14]), np.zeros((1, 1)), np.array([1e-12]),
            np.array([0]), 300)
        result = optimize_power(problem, np.array([50.0]), "slsqp", {
            "power_cap_w": 300, "maxiter": 80, "max_evaluations": 500})
        self.assertGreater(result["power_w"][0], 50)
        self.assertGreater(result["metrics"]["final_si_mean"],
                           result["initial_metrics"]["final_si_mean"])
        self.assertLessEqual(result["power_w"][0], 300)

    def test_slsqp_reduces_oversupply_and_raises_sgm(self):
        problem = FixedAllocationPhysics(
            np.array([20e6]), np.array([0]), np.array([0]), np.array([1]),
            np.array([1e-14]), np.zeros((1, 1)), np.array([1e-12]),
            np.array([0]), 300)
        result = optimize_power(problem, np.array([200.0]), "slsqp", {
            "power_cap_w": 300, "maxiter": 80, "max_evaluations": 500})
        self.assertLess(result["power_w"][0], 200)
        self.assertGreater(result["metrics"]["final_sgm_mean"],
                           result["initial_metrics"]["final_sgm_mean"])
        self.assertGreaterEqual(result["metrics"]["final_si_mean"], 1 - 1e-10)
        self.assertTrue(result["converged"])
        self.assertEqual(result["solver_power_scale_w"], 300.0)
        demand_matching_power = (2 ** (20e6 / 25e6) - 1) * 1e-12 / 1e-14
        self.assertAlmostEqual(result["power_w"][0], demand_matching_power, delta=0.5)
        self.assertGreater(result["metrics"]["final_sgm_mean"], 0.999)

    def test_scaled_power_chain_rule_matches_finite_difference(self):
        problem = FixedAllocationPhysics(
            np.array([40e6]), np.array([0]), np.array([0]), np.array([1]),
            np.array([1e-14]), np.zeros((1, 1)), np.array([1e-12]),
            np.array([0]), 300)
        scale = 300.0
        x = 100.0 / scale
        delta = 1e-5
        ev = problem.evaluate(np.array([scale * x]), with_jacobian=True)
        plus = problem.evaluate(np.array([scale * (x + delta)]))
        minus = problem.evaluate(np.array([scale * (x - delta)]))
        self.assertAlmostEqual((plus.capacity_bps[0] - minus.capacity_bps[0]) / (2 * delta),
                               ev.capacity_jacobian[0, 0] * scale, delta=1e-2)
        self.assertAlmostEqual((plus.sgm - minus.sgm) / (2 * delta),
                               ev.sgm_jacobian[0] * scale, delta=1e-7)

    def test_matching_warm_start_counts_scalar_calls_and_retains_true_feasible_points(self):
        problem = FixedAllocationPhysics(
            np.full(2, 20e6), np.array([0, 2]), np.zeros(2, dtype=int),
            np.ones(2, dtype=int), np.full(2, 1e-14),
            np.array([[0, 1e-16], [1e-16, 0]]), np.full(2, 1e-12),
            np.arange(2), 3000)
        initial = np.array([200.0, 200.0])
        result = optimize_power(problem, initial, "slsqp", {
            "maxiter": 80, "max_evaluations": 500})
        self.assertIn("demand_matching_warm_start", result["algorithm_components"])
        self.assertGreater(result["warm_start_status"]["accepted_candidates"], 0)
        self.assertGreater(result["scalar_capacity_evaluations"], 0)
        self.assertEqual(result["evaluations"], result["full_scene_evaluations"]
                         + result["scalar_capacity_evaluations"])
        self.assertLessEqual(result["evaluations"], 500)
        final = problem.evaluate(result["power_w"])
        base = problem.evaluate(initial)
        self.assertLess(sum(result["power_w"]), sum(initial))
        self.assertGreater(final.sgm, base.sgm)
        self.assertGreaterEqual(final.si, base.si - 1e-10)
        np.testing.assert_array_less(
            np.minimum(base.capacity_bps, 0.1 * problem.demand_bps)
            - final.capacity_bps, np.full(2, 1e-3))

    def test_stop_during_matching_returns_last_real_feasible_candidate(self):
        problem = FixedAllocationPhysics(
            np.full(2, 20e6), np.array([0, 2]), np.zeros(2, dtype=int),
            np.ones(2, dtype=int), np.full(2, 1e-14),
            np.array([[0, 1e-16], [1e-16, 0]]), np.full(2, 1e-12),
            np.arange(2), 3000)
        stopped = [False]
        def on_candidate(entry):
            if entry["phase"] == "demand_matching_warm_start":
                stopped[0] = True
        result = optimize_power(problem, np.array([200.0, 200.0]), "slsqp",
                                {"maxiter": 80, "max_evaluations": 500},
                                callback=on_candidate,
                                stop_requested=lambda: stopped[0])
        self.assertTrue(stopped[0])
        self.assertEqual(result["termination_reason"], "stop_requested")
        self.assertFalse(result["converged"])
        self.assertFalse(result["fallback_used"])
        self.assertLessEqual(result["evaluations"], 500)
        self.assertGreater(problem.evaluate(result["power_w"]).sgm,
                           problem.evaluate([200.0, 200.0]).sgm)


if __name__ == "__main__":
    unittest.main()
