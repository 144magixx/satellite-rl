"""Numerical and acceptance checks for the SI-priority power refiners."""

import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

from satellite_rl.power_refinement.physics import (FixedAllocationPhysics,
                                                   PhysicalEvaluation)
from satellite_rl.power_refinement.priority import (
    _capacity_lower_bound, _proposal_lb, _proposal_lp, optimize_priority,
)


def problem_two(demand=(100e6, 100e6), interference=2e-15):
    return FixedAllocationPhysics(
        np.asarray(demand, dtype=float), np.array([0, 2]),
        np.zeros(2, dtype=int), np.ones(2, dtype=int),
        np.array([1e-14, 1.5e-14]),
        np.array([[0.0, interference], [interference, 0.0]]),
        np.full(2, 1e-12), np.arange(2), 300.0)


class PriorityTests(unittest.TestCase):
    def test_log_capacity_bound_is_global_tight_and_tangent(self):
        problem = problem_two()
        centre = np.array([30.0, 45.0])
        actual = problem.evaluate(centre, with_jacobian=True)
        bound, jac = _capacity_lower_bound(problem, centre, centre)
        np.testing.assert_allclose(bound, actual.capacity_bps, rtol=1e-12, atol=1e-6)
        np.testing.assert_allclose(jac, actual.capacity_jacobian, rtol=1e-11, atol=1e-5)
        for trial in (np.array([0.0, 0.0]), np.array([100.0, 15.0]),
                      np.array([12.0, 250.0])):
            lower, _ = _capacity_lower_bound(problem, centre, trial)
            true = problem.evaluate(trial).capacity_bps
            self.assertTrue(np.all(lower <= true + 1e-6))
        eps = 1e-4
        for j in range(2):
            delta = np.eye(2)[j] * eps
            plus, _ = _capacity_lower_bound(problem, centre, centre + delta)
            minus, _ = _capacity_lower_bound(problem, centre, centre - delta)
            np.testing.assert_allclose((plus - minus) / (2 * eps), jac[:, j],
                                       rtol=1e-6, atol=1e-3)

    @unittest.skipUnless(importlib.util.find_spec("cvxpy"), "cvxpy unavailable")
    def test_lb_solves_curved_model_and_true_capacity_accepts(self):
        problem = problem_two((100e6, 140e6))
        initial = np.array([10.0, 10.0])
        ev = problem.evaluate(initial, with_jacobian=True)
        floor = np.minimum(ev.capacity_bps / problem.demand_bps, 0.1)
        lp, _ = _proposal_lp(problem, initial, ev, 100.0, "si", ev.si,
                             floor, 20.0, 3.0)
        lb, status = _proposal_lb(problem, initial, ev, 100.0, "si", ev.si,
                                  floor, 20.0, 3.0)
        self.assertIsNotNone(lp)
        self.assertIsNotNone(lb, status)
        self.assertLessEqual(lb.sum(), 100.0 + 1e-5)
        self.assertTrue(np.isfinite(lb).all())
        lower, _ = _capacity_lower_bound(problem, initial, lb)
        self.assertTrue(np.all(lower / problem.demand_bps >= floor - 1e-6))
        self.assertTrue(np.all(lower <= problem.evaluate(lb).capacity_bps + 1e-6))

    def test_stop_callback_and_budget_include_reference(self):
        problem = problem_two((100e6, 100e6), 0.0)
        initial = np.array([5.0, 5.0])
        accepted = []
        with patch("satellite_rl.power_refinement.priority._proposal_lp",
                   return_value=(np.array([10.0, 10.0]), "fixed")):
            result = optimize_priority(
                problem, initial, options={"power_cap_w": 100.0,
                                           "maxiter": 10, "maxeval": 2,
                                           "time_limit_seconds": 5.0},
                callback=accepted.append,
                stop_requested=lambda: bool(accepted))
        self.assertEqual(result["evaluations"], 2)
        self.assertEqual(result["termination_reason"], "stop_requested")
        self.assertEqual(len(accepted), 1)
        self.assertGreater(result["metrics"]["final_si_mean"],
                           result["initial_metrics"]["final_si_mean"])
        self.assertFalse(result["fallback_used"])
        self.assertEqual(result["phase1_best_si_mean"],
                         result["metrics"]["final_si_mean"])

    def test_si_gain_is_accepted_even_when_sgm_falls(self):
        problem = FixedAllocationPhysics(
            np.array([25e6, 25e6]), np.array([0, 2]),
            np.zeros(2, dtype=int), np.ones(2, dtype=int),
            np.full(2, 1e-14), np.zeros((2, 2)),
            np.full(2, 1e-12), np.arange(2), 110.0)
        initial = np.array([40.0, 40.0])
        candidate = np.array([8.0, 100.0])
        first = problem.evaluate(initial)
        moved = problem.evaluate(candidate)
        self.assertGreater(moved.si, first.si)
        self.assertLess(moved.sgm, first.sgm)
        for method, fn in (("sca_si_lp", "_proposal_lp"),
                           ("sca_si_lb", "_proposal_lb")):
            with self.subTest(method=method), patch(
                f"satellite_rl.power_refinement.priority.{fn}",
                return_value=(candidate, "fixed")):
                result = optimize_priority(problem, initial, method=method,
                                           options={"maxiter": 3, "maxeval": 20,
                                                    "time_limit_seconds": 5.0})
                self.assertGreater(result["metrics"]["final_si_mean"], first.si)
                self.assertLess(result["metrics"]["final_sgm_mean"], first.sgm)
                np.testing.assert_array_equal(result["power_w"], candidate)

    def test_service_threshold_strict_and_empty_slot_remains_zero(self):
        problem = FixedAllocationPhysics(
            np.array([40e6, 100e6, 100e6]), np.array([0, 2, 4]),
            np.zeros(3, dtype=int), np.array([1, 1, 0]),
            np.array([1e-14, 1e-14]), np.zeros((2, 3)),
            np.full(2, 1e-12), np.array([0, 1]), 300.0)
        initial = np.array([12.0, 1.0, 0.0])
        self.assertGreater(problem.evaluate(initial).capacity_bps[0] /
                           problem.demand_bps[0], 0.1)
        with patch("satellite_rl.power_refinement.priority._proposal_lp",
                   return_value=(np.array([0.0, 250.0, 0.0]), "fixed")):
            result = optimize_priority(problem, initial, options={
                "power_cap_w": 300.0, "maxiter": 3, "maxeval": 25,
                "time_limit_seconds": 5.0})
        final = problem.evaluate(result["power_w"])
        self.assertGreaterEqual(final.capacity_bps[0] / problem.demand_bps[0], 0.1)
        self.assertEqual(result["power_w"][2], 0.0)

    def test_t0_floor_does_not_gain_another_tolerance_after_a(self):
        problem = FixedAllocationPhysics(
            np.array([25e6, 25e6]), np.array([0, 2]),
            np.zeros(2, dtype=int), np.ones(2, dtype=int),
            np.full(2, 1e-14), np.zeros((2, 2)),
            np.full(2, 1e-12), np.arange(2), 110.0)

        def power_for_ratio(ratio):
            return (2.0 ** ratio - 1.0) * 100.0

        initial = np.array([power_for_ratio(0.1 - 5e-9), 10.0])
        proposal = np.array([power_for_ratio(0.1 - 2e-8), 100.0])
        with patch("satellite_rl.power_refinement.priority._proposal_lp",
                   return_value=(proposal, "fixed")):
            result = optimize_priority(problem, initial, options={
                "power_cap_w": 110.0, "maxiter": 2, "maxeval": 20,
                "time_limit_seconds": 5.0,
                "baseline_service_floor_bps": [2.5e6, 0.0]})
        ratio = problem.evaluate(result["power_w"]).capacity_bps[0] / 25e6
        self.assertGreaterEqual(ratio, 0.1 - 1e-8)

    def test_two_sgm_steps_cannot_accumulate_si_tolerance(self):
        # Controlled physical response isolates the acceptance rule: two SGM
        # gains each lose a little SI, and a third would cross the fixed floor.
        problem = FixedAllocationPhysics(
            np.array([1.0]), np.array([0]), np.array([0]), np.array([1]),
            np.array([1.0]), np.zeros((1, 1)), np.array([1.0]),
            np.array([0]), 10.0)

        def evaluate(power, with_jacobian=False):
            p = float(power[0])
            if p <= 1.0:
                si, sgm = 0.5, 0.1
            elif p <= 2.0:
                si, sgm = 0.6, 0.0
            elif p <= 3.0:
                si, sgm = 0.6 - 5e-9, 0.2
            elif p <= 4.0:
                si, sgm = 0.6 - 9e-9, 0.3
            else:
                si, sgm = 0.6 - 15e-9, 0.4
            jac = np.zeros((1, 1)) if with_jacobian else None
            grad = np.zeros(1) if with_jacobian else None
            return PhysicalEvaluation(np.array([si]), si, sgm,
                                      np.array([si]), np.array([sgm]),
                                      jac, grad, grad)

        problem.evaluate = evaluate
        phase2_calls = [0]

        def proposal(_problem, power, _ev, _cap, phase, anchor, *_args):
            self.assertEqual(anchor, 0.5 if phase == "si" and power[0] <= 1
                             else 0.6)
            if phase == "si":
                return np.array([2.0]), "fixed"
            phase2_calls[0] += 1
            return np.array([min(5.0, 2.0 + phase2_calls[0])]), "fixed"

        with patch("satellite_rl.power_refinement.priority._proposal_lp", proposal):
            result = optimize_priority(problem, np.array([1.0]), options={
                "power_cap_w": 10.0, "maxiter": 4, "maxeval": 30,
                "time_limit_seconds": 5.0})
        self.assertEqual(result["phase1_best_si_mean"], 0.6)
        self.assertEqual(result["phase2_si_floor"], 0.6 - 1e-8)
        self.assertEqual(result["power_w"], [4.0])
        self.assertGreaterEqual(result["metrics"]["final_si_mean"],
                                result["phase2_si_floor"])

    def test_time_reserve_switches_with_few_capacity_calls(self):
        problem = problem_two((100e6, 100e6), 0.0)
        clock = [0.0]

        def proposal(_problem, _power, _ev, _cap, phase, _anchor, *_args):
            if phase == "si":
                clock[0] = 0.86
                return np.array([10.0, 10.0]), "fixed"
            return None, "none"

        with patch("satellite_rl.power_refinement.priority.perf_counter",
                   lambda: clock[0]), patch(
                "satellite_rl.power_refinement.priority._proposal_lp", proposal):
            result = optimize_priority(problem, np.array([5.0, 5.0]), options={
                "power_cap_w": 100.0, "maxiter": 3, "maxeval": 2000,
                "time_limit_seconds": 1.0})
        self.assertEqual(result["solver_status"][0]["stop_reason"],
                         "phase_reserve")
        self.assertEqual(result["solver_status"][1]["phase"], "sgm")
        self.assertLess(result["evaluations"], 200)

    def test_higher_si_anchor_keeps_better_sgm_when_old_point_still_valid(self):
        problem = FixedAllocationPhysics(
            np.array([1.0]), np.array([0]), np.array([0]), np.array([1]),
            np.array([1.0]), np.zeros((1, 1)), np.array([1.0]),
            np.array([0]), 10.0)

        def evaluate(power, with_jacobian=False):
            p = float(power[0])
            if p <= 1:
                si, sgm = 0.5, 0.1
            elif p <= 2:
                si, sgm = 0.6, 0.3
            elif p <= 3:
                si, sgm = 0.6 - 5e-9, 0.4
            else:
                si, sgm = 0.6 + 2e-9, 0.1
            jac = np.zeros((1, 1)) if with_jacobian else None
            grad = np.zeros(1) if with_jacobian else None
            return PhysicalEvaluation(np.array([si]), si, sgm,
                                      np.array([si]), np.array([sgm]),
                                      jac, grad, grad)

        problem.evaluate = evaluate

        def proposal(_problem, power, _ev, _cap, phase, _anchor, *_args):
            if phase == "si":
                return np.array([2.0]), "fixed"
            return np.array([3.0 if power[0] <= 2 else 4.0]), "fixed"

        with patch("satellite_rl.power_refinement.priority._proposal_lp", proposal):
            result = optimize_priority(problem, np.array([1.0]), options={
                "power_cap_w": 10.0, "maxiter": 3, "maxeval": 30,
                "time_limit_seconds": 5.0})
        self.assertEqual(result["power_w"], [3.0])
        self.assertEqual(result["best_feasible_si_mean"], 0.6 + 2e-9)
        self.assertEqual(result["highest_si_power_w"], [4.0])
        self.assertTrue(any(row.get("anchor_changed") and
                            not row.get("selected_changed")
                            for row in result["history"]))

    def test_no_effective_candidate_preserves_exact_a_allocation(self):
        problem = problem_two()
        initial = np.array([10.0, 10.0])
        for method, fn in (("sca_si_lp", "_proposal_lp"),
                           ("sca_si_lb", "_proposal_lb")):
            with self.subTest(method=method), patch(
                f"satellite_rl.power_refinement.priority.{fn}",
                return_value=(None, "no solution")):
                result = optimize_priority(problem, initial, method=method,
                                           options={"maxiter": 2, "maxeval": 5,
                                                    "time_limit_seconds": 5.0})
                np.testing.assert_array_equal(result["power_w"], initial)
                self.assertTrue(result["fallback_used"])
                self.assertEqual(result["evaluations"], 1)


if __name__ == "__main__":
    unittest.main()
