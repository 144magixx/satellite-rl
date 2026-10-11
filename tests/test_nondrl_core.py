"""Exact common-contract checks; CPU micro examples are not training results."""
import unittest
import numpy as np

from satellite_rl.coverage.scenes import CoverageScene
from satellite_rl.coverage.variable_design import Layout
from satellite_rl.coverage.geometry import satellite_rays_numpy
from satellite_rl.coverage.nondrl import NondrlConfig, Problem, metrics_all, solve
from satellite_rl.coverage.nondrl.core import BudgetStop, layout_sha256, ray_to_location


def small_scene(demands=(90., 10., 10., 10.)):
    positions = np.array([[45., 13.], [45.05, 13.], [45., 13.05], [45.05, 13.05]])
    scene = CoverageScene("micro", positions, np.asarray(demands), np.empty((0, 2)),
                          [f"a{i}" for i in range(4)])
    initial = Layout([[45., 13.], [45.05, 13.05]], [0.5, 0.5], [0, 0, 1, 1], tuple(scene.aircraft_ids))
    return scene, initial


def constant_physical(scene, layout):
    return {"aircraft_service": {"demand_weighted_carried_fraction":
            1.0 if np.sum(scene.demands) else None},
            "metrics": {"total_power_w": 0.0, "complete_allocation": True}}


class NondrlCoreTests(unittest.TestCase):
    def test_all_beam_cv_keeps_zero_demand_serving_beam(self):
        scene, initial = small_scene((10., 10., 0., 0.))
        result = metrics_all(scene, initial)
        self.assertEqual(result["beam_count"], 2)
        self.assertEqual(result["zero_demand_beams"], 1)
        self.assertEqual(result["load_cv_all"], 1.)
        self.assertEqual(result["load_cv_positive"], 0.)
        scene.demands[:] = 0
        self.assertIsNone(metrics_all(scene, initial)["load_cv_all"])

    def test_identity_and_empty_beams_rejected(self):
        scene, initial = small_scene()
        initial.assignment[:] = 0
        with self.assertRaisesRegex(ValueError, "empty"):
            metrics_all(scene, initial)
        initial.assignment = np.array([0, 0, 1, 1])
        initial.aircraft_ids = ("wrong", "a1", "a2", "a3")
        with self.assertRaisesRegex(ValueError, "IDs"):
            metrics_all(scene, initial)

    def test_ray_inverse_roundtrip_and_sky_ray(self):
        positions = np.array([[45., 13.], [30., -5.], [60., 20.]])
        for point, ray in zip(positions, satellite_rays_numpy(positions)):
            np.testing.assert_allclose(ray_to_location(ray), point, atol=2e-10)
        self.assertIsNone(ray_to_location([1., 0., 0.]))

    def test_rejected_proposal_has_atomic_full_rollback(self):
        scene, initial = small_scene()
        problem = Problem(scene, {"online_seconds": 5.}, initial, constant_physical)
        before = layout_sha256(problem.current)
        bad = Layout([[45., 30.], [45., 13.]], [0.5, 0.5], [0, 0, 1, 1], tuple(scene.aircraft_ids))
        trial = problem.consider(bad)
        self.assertFalse(problem.commit(trial))
        self.assertEqual(layout_sha256(problem.current), before)
        self.assertGreater(problem.calls["geometry_calls"], 0)

    def test_carry_gate_is_anchored_to_I0(self):
        scene, initial = small_scene()
        def declining(scene, layout):
            value = 1. if layout.assignment.tolist() == [0, 0, 1, 1] else 0.99
            return {"aircraft_service": {"demand_weighted_carried_fraction": value},
                    "metrics": {"total_power_w": 0.}}
        problem = Problem(scene, {"online_seconds": 5.}, initial, declining)
        action = next(a for a in problem.menu() if a["kind"] == "transfer" and a["aircraft"] == 1)
        trial = problem.preview(action)
        self.assertFalse(trial["accepted"])
        self.assertEqual(trial["reason"], "I0_aircraft_carry_protection")

    def test_candidate_limit_charges_last_candidate(self):
        scene, initial = small_scene()
        problem = Problem(scene, {"online_seconds": 5., "max_candidates": 1}, initial, constant_physical)
        trial = problem.preview({"kind": "transfer", "aircraft": 1, "target": 1})
        self.assertTrue(trial["accepted"])
        self.assertEqual(problem.calls["candidates"], 1)
        self.assertEqual(problem.calls["physical_calls"], 1)
        with self.assertRaises(BudgetStop):
            problem.preview({"kind": "transfer", "aircraft": 2, "target": 0})

    def test_BH_uses_same_N_and_exact_loads(self):
        scene, initial = small_scene()
        result = solve(scene, "BH", {"online_seconds": 5.}, initial,
                       physical_evaluator=constant_physical)
        self.assertEqual(len(result["layout"].centers), 2)
        self.assertLess(result["metrics_all"]["load_cv_all"], result["initial_metrics_all"]["load_cv_all"])
        self.assertEqual(result["metrics_all"]["total_demand_bps"], 120.)
        self.assertGreater(result["timing"]["total_seconds"], result["timing"]["online_seconds"])

    def test_indivisible_physical_overrun_is_recorded(self):
        from time import sleep
        scene, initial = small_scene()
        problem = Problem(scene, {"online_seconds": .005}, initial, constant_physical)
        def delayed(scene, layout):
            sleep(.015)
            return constant_physical(scene, layout)
        problem._physical_evaluator = delayed
        trial = problem.preview({"kind": "transfer", "aircraft": 1, "target": 1})
        self.assertTrue(trial["accepted"])
        self.assertTrue(trial["finished_after_deadline"])
        self.assertTrue(problem.commit(trial))
        result = problem.result("BH")
        self.assertGreater(result["online_overrun_seconds"], 0)
        self.assertEqual(problem.calls["physical_calls"], 1)

    def test_unknown_method_fails_closed(self):
        scene, initial = small_scene()
        with self.assertRaisesRegex(ValueError, "unsupported"):
            solve(scene, "06", {"online_seconds": 2.}, initial, method="made_up",
                  physical_evaluator=constant_physical)

    def test_actual_200_beams_are_allowed_and_201_rejected(self):
        for count in (200, 201):
            positions = np.tile([45., 13.], (count, 1))
            scene = CoverageScene("count", positions, np.ones(count), np.empty((0, 2)),
                                  [f"a{i}" for i in range(count)])
            layout = Layout(positions, np.full(count, .5), np.arange(count), tuple(scene.aircraft_ids))
            if count == 200:
                result = metrics_all(scene, layout)
                self.assertEqual(result["beam_count"], 200)
                self.assertEqual(result["load_cv_all"], 0.)
            else:
                with self.assertRaisesRegex(ValueError, "beam count"):
                    metrics_all(scene, layout)

    def test_nonfinite_physical_result_cannot_be_published(self):
        scene, initial = small_scene()
        problem = Problem(scene, {"online_seconds": 2.}, initial, constant_physical)
        problem._physical_evaluator = lambda scene, layout: {
            "aircraft_service": {"demand_weighted_carried_fraction": float("nan")},
            "metrics": {"total_power_w": 0.}}
        before = layout_sha256(problem.current)
        trial = problem.preview({"kind": "transfer", "aircraft": 1, "target": 1})
        self.assertFalse(problem.commit(trial))
        self.assertIn("physical", trial["reason"])
        self.assertEqual(layout_sha256(problem.current), before)

    def test_expired_menu_reports_truncation_without_exposure(self):
        scene, initial = small_scene()
        problem = Problem(scene, {"online_seconds": 2.}, initial, constant_physical)
        problem.deadline = problem.online_started - 1
        with self.assertRaises(BudgetStop):
            problem.menu()
        self.assertTrue(problem.trace[-1]["truncated"])
        self.assertEqual(problem.trace[-1]["returned_actions"], 0)


if __name__ == "__main__":
    unittest.main()
