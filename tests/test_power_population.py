"""Deterministic and accounting checks for bounded population power search."""

import unittest

import numpy as np

from satellite_rl.power_refinement.physics import FixedAllocationPhysics
from satellite_rl.power_refinement.population import optimize_population
from satellite_rl.power_refinement.runner import _optimizer_seed


class CountingPhysics(FixedAllocationPhysics):
    def evaluate(self, power_w, with_jacobian=False):
        self.calls += 1
        return super().evaluate(power_w, with_jacobian=with_jacobian)


def toy_problem(demand=40e6, initial_power=50.0):
    problem = CountingPhysics(
        np.array([demand]), np.array([0]), np.array([0]), np.array([1]),
        np.array([1e-14]), np.zeros((1, 1)), np.array([1e-12]),
        np.array([0]), 300)
    problem.calls = 0
    return problem, np.array([initial_power])


class PopulationTests(unittest.TestCase):
    def test_de_and_pso_improve_single_beam_and_allow_above_50_w(self):
        for method in ("de", "pso"):
            with self.subTest(method=method):
                problem, initial = toy_problem()
                result = optimize_population(problem, initial, method, {
                    "optimizer_seed": 63001, "power_cap_w": 300,
                    "maxiter": 30, "maxeval": 200, "max_time_seconds": 10})
                self.assertGreater(result["power_w"][0], 50)
                self.assertGreater(result["metrics"]["final_si_mean"],
                                   result["initial_metrics"]["final_si_mean"])
                self.assertEqual(problem.calls, result["evaluations"])
                self.assertLessEqual(result["evaluations"], 200)
                self.assertLessEqual(result["power_w"][0], 300)

    def test_oversupply_can_drop_below_5_w_without_fabricated_gain(self):
        for method in ("de", "pso"):
            with self.subTest(method=method):
                problem, initial = toy_problem(demand=0.5e6, initial_power=50)
                result = optimize_population(problem, initial, method, {
                    "optimizer_seed": 63001, "power_cap_w": 50,
                    "maxiter": 80, "maxeval": 500, "max_time_seconds": 10})
                self.assertLess(result["power_w"][0], 5)
                self.assertGreater(result["metrics"]["final_sgm_mean"],
                                   result["initial_metrics"]["final_sgm_mean"])
                self.assertEqual(problem.calls, result["evaluations"])

    def test_identical_seed_and_budget_repeat_exact_power_and_history(self):
        for method in ("de", "pso"):
            with self.subTest(method=method):
                runs = []
                for _ in range(2):
                    problem, initial = toy_problem(demand=20e6, initial_power=200)
                    runs.append(optimize_population(problem, initial, method, {
                        "optimizer_seed": 63001, "maxiter": 12, "maxeval": 100}))
                self.assertEqual(runs[0]["power_w"], runs[1]["power_w"])
                self.assertEqual(runs[0]["evaluations"], runs[1]["evaluations"])
                self.assertEqual([e["evaluation"] for e in runs[0]["history"]],
                                 [e["evaluation"] for e in runs[1]["history"]])

    def test_budget_stop_and_fallback_count_initial_once(self):
        for method in ("de", "pso"):
            with self.subTest(method=method):
                problem, initial = toy_problem()
                result = optimize_population(problem, initial, method, {
                    "optimizer_seed": 63001, "maxeval": 1})
                self.assertEqual(result["termination_reason"], "evaluation_limit")
                self.assertEqual(result["evaluations"], 1)
                self.assertEqual(problem.calls, 1)
                self.assertEqual(result["power_w"], initial.tolist())
                self.assertTrue(result["fallback_used"])
                problem, initial = toy_problem()
                result = optimize_population(problem, initial, method, {
                    "optimizer_seed": 63001}, stop_requested=lambda: True)
                self.assertEqual(result["termination_reason"], "stop_requested")
                self.assertEqual(problem.calls, 1)

    def test_empty_slot_and_basic_service_guard(self):
        problem = CountingPhysics(
            np.array([20e6, 20e6, 20e6]), np.array([0, 1, 2]),
            np.zeros(3, dtype=int), np.array([1, 0, 1]),
            np.array([1e-14, 1e-14]), np.zeros((2, 3)),
            np.array([1e-12, 1e-12]), np.array([0, 2]), 300)
        problem.calls = 0
        initial = np.array([50.0, 0.0, 50.0])
        baseline = problem.evaluate(initial)
        for method in ("de", "pso"):
            result = optimize_population(problem, initial, method, {
                "optimizer_seed": 63001, "maxiter": 20, "maxeval": 200})
            final = problem.evaluate(result["power_w"])
            self.assertEqual(result["power_w"][1], 0)
            self.assertGreaterEqual(final.si, baseline.si - 1e-8)
            self.assertGreaterEqual(final.sgm, baseline.sgm - 1e-8)
            self.assertTrue(np.all(final.capacity_bps >=
                                   np.minimum(baseline.capacity_bps,
                                              0.1 * problem.demand_bps) - 1e-3))

    def test_seed_is_required(self):
        problem, initial = toy_problem()
        with self.assertRaises(ValueError):
            optimize_population(problem, initial, "de")

    def test_runner_streams_are_distinct_at_30_repeats_boundary(self):
        seeds = {_optimizer_seed(63001, scene, repeat, method)
                 for scene in range(3) for repeat in range(30)
                 for method in ("de", "pso")}
        self.assertEqual(len(seeds), 3 * 30 * 2)
        self.assertEqual(_optimizer_seed(63001, 0, 0, "de"), 63002)
        self.assertEqual(_optimizer_seed(63001, 0, 29, "pso"), 63293)
        self.assertEqual(_optimizer_seed(63001, 1, 0, "de"), 64002)
        for repeat in (-1, 30):
            with self.assertRaises(ValueError):
                _optimizer_seed(63001, 0, repeat, "de")


if __name__ == "__main__":
    unittest.main()
