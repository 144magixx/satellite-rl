"""Small deterministic checks of the two distinct model-based proposals."""

import unittest

import numpy as np

from satellite_rl.power_refinement.model_based import (
    _wmmse_proposal, optimize_model_based,
)
from satellite_rl.power_refinement.physics import FixedAllocationPhysics


def tiny_problem(demand=(40e6, 40e6), interference=0.0):
    return FixedAllocationPhysics(
        np.asarray(demand, dtype=float), np.array([0, 2]),
        np.zeros(2, dtype=int), np.ones(2, dtype=int),
        np.full(2, 1e-14),
        np.array([[0, interference], [interference, 0]], dtype=float),
        np.full(2, 1e-12), np.arange(2), 300.0)


class ModelBasedTests(unittest.TestCase):
    def test_wmmse_one_link_quadratic_step_and_shared_budget(self):
        problem = FixedAllocationPhysics(
            np.array([100e6]), np.array([0]), np.array([0]), np.array([1]),
            np.array([1e-13]), np.zeros((1, 1)), np.array([1e-12]),
            np.array([0]), 300.0)
        p0 = np.array([100.0])
        ev = problem.evaluate(p0)
        # For one link, fixed receiver and MSE weight give
        # x_new=x_old*(1+g*p_old)/(g*p_old)=11, hence p_new=121 W.
        unconstrained = _wmmse_proposal(problem, p0, ev, 300.0, "si")
        constrained = _wmmse_proposal(problem, p0, ev, 105.0, "si")
        self.assertAlmostEqual(unconstrained[0], 121.0, places=8)
        self.assertAlmostEqual(constrained[0], 105.0, places=8)

    def test_wmmse_block_step_improves_its_rate_surrogate(self):
        problem = tiny_problem((100e6, 100e6), 1e-16)
        p0 = np.array([5.0, 10.0])
        ev = problem.evaluate(p0)
        proposal = _wmmse_proposal(problem, p0, ev, 100.0, "si")
        self.assertIsNotNone(proposal)
        self.assertLessEqual(proposal.sum(), 100 + 1e-8)
        old = np.sum(ev.capacity_bps / problem.demand_bps)
        new = np.sum(problem.evaluate(proposal).capacity_bps / problem.demand_bps)
        self.assertGreaterEqual(new, old - 1e-8)

    def test_sca_solves_lp_and_rechecks_true_metrics(self):
        problem = tiny_problem((40e6, 40e6), 1e-16)
        initial = np.array([5.0, 5.0])
        result = optimize_model_based(problem, initial, "sca", {
            "power_cap_w": 40.0, "maxiter": 12, "maxeval": 80,
            "max_time_seconds": 5.0,
        })
        final_power = np.asarray(result["power_w"])
        initial_eval = problem.evaluate(initial)
        final = problem.evaluate(final_power)
        self.assertGreater(result["lp_subproblems"], 0)
        self.assertLessEqual(result["evaluations"], 80)
        self.assertLessEqual(final_power.sum(), 40 + 1e-7)
        self.assertGreaterEqual(final.si, initial_eval.si - 1e-10)
        self.assertGreaterEqual(final.sgm, initial_eval.sgm - 1e-10)
        np.testing.assert_array_less(
            np.minimum(initial_eval.capacity_bps, 0.1 * problem.demand_bps)
            - final.capacity_bps, np.full(2, 1e-2))

    def test_wmmse_true_acceptance_and_empty_slot(self):
        problem = FixedAllocationPhysics(
            np.array([100e6, 10e6]), np.array([0, 1]),
            np.zeros(2, dtype=int), np.array([1, 0]),
            np.array([1e-14]), np.zeros((1, 2)), np.array([1e-12]),
            np.array([0]), 300.0)
        initial = np.array([5.0, 0.0])
        result = optimize_model_based(problem, initial, "wmmse", {
            "power_cap_w": 100.0, "maxiter": 8, "maxeval": 50,
            "max_time_seconds": 5.0,
        })
        self.assertGreater(result["metrics"]["final_si_mean"],
                           result["initial_metrics"]["final_si_mean"])
        self.assertEqual(result["power_w"][1], 0.0)
        self.assertLessEqual(sum(result["power_w"]), 100 + 1e-7)
        self.assertGreaterEqual(result["metrics"]["final_sgm_mean"],
                                result["initial_metrics"]["final_sgm_mean"] - 1e-10)
        self.assertIn("weighted sum-rate", result["surrogate_weight_rule"])

    def test_stop_retains_initial_and_never_exceeds_budget(self):
        problem = tiny_problem()
        for method in ("sca", "wmmse"):
            with self.subTest(method=method):
                result = optimize_model_based(problem, np.array([5.0, 5.0]),
                                              method, stop_requested=lambda: True)
                self.assertEqual(result["termination_reason"], "stop_requested")
                self.assertEqual(result["evaluations"], 1)
                self.assertTrue(result["fallback_used"])

    def test_sca_above_50_and_zero_initial_power_are_allowed(self):
        problem = FixedAllocationPhysics(
            np.array([40e6]), np.array([0]), np.array([0]), np.array([1]),
            np.array([1e-14]), np.zeros((1, 1)), np.array([1e-12]),
            np.array([0]), 300.0)
        for initial in (np.array([0.0]), np.array([50.0])):
            with self.subTest(initial=initial[0]):
                result = optimize_model_based(problem, initial, "sca", {
                    "power_cap_w": 300.0, "maxiter": 20, "maxeval": 100,
                    "max_time_seconds": 5.0,
                })
                self.assertGreater(result["power_w"][0], 50.0)
                self.assertGreaterEqual(result["metrics"]["final_sgm_mean"],
                                        result["initial_metrics"]["final_sgm_mean"])
                self.assertLessEqual(result["power_w"][0], 300.0)
                self.assertGreater(result["solver_status"][0]["iterations"], 0)
                self.assertIn("stop_reason", result["solver_status"][0])

    def test_sca_sgm_guard_and_one_evaluation_fallback(self):
        problem = tiny_problem((20e6, 20e6), 1e-16)
        initial = np.array([100.0, 100.0])
        result = optimize_model_based(problem, initial, "sca", {
            "power_cap_w": 300.0, "maxiter": 10, "maxeval": 1,
        })
        self.assertEqual(result["termination_reason"], "evaluation_limit")
        self.assertEqual(result["evaluations"], 1)
        np.testing.assert_array_equal(result["power_w"], initial)
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["solver_status"][0]["stop_reason"],
                         "evaluation_limit")
        self.assertFalse(result["converged"])
        self.assertIn("not certified", result["convergence_scope"])

    def test_callback_stop_keeps_last_accepted_sca_point(self):
        problem = tiny_problem((100e6, 100e6), 0.0)
        initial = np.array([5.0, 5.0])
        stopped = [False]
        result = optimize_model_based(problem, initial, "sca", {
            "power_cap_w": 100.0, "maxiter": 20, "maxeval": 60,
            "max_time_seconds": 5.0,
        }, callback=lambda _: stopped.__setitem__(0, True),
            stop_requested=lambda: stopped[0])
        self.assertTrue(stopped[0])
        self.assertEqual(result["termination_reason"], "stop_requested")
        self.assertFalse(result["fallback_used"])
        self.assertLessEqual(result["evaluations"], 60)
        self.assertGreater(result["metrics"]["final_si_mean"],
                           result["initial_metrics"]["final_si_mean"])


if __name__ == "__main__":
    unittest.main()
