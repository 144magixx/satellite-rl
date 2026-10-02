"""Fixed-slot physical parity and derivative checks (no training)."""

import unittest

import numpy as np

from satellite_rl.power_refinement import FixedAllocationPhysics
from satellite_rl.workbench.config import default_config
from satellite_rl.workbench.environment import WorkbenchEnv


def replay_scene():
    config = default_config()
    config["environment"]["profile"] = "europe_v1"
    env = WorkbenchEnv(config)
    env.reset(seed=51001)
    for i in range(100):
        # Same polarization overlaps across groups; same-group slots do not.
        _, _, done, _ = env.step([(i // 8) * 2, 1, 3])
        if i < 99:
            assert not done
    beams = env.beam_results()
    initial = np.array([row["power_w"] for row in beams])
    problem = FixedAllocationPhysics.from_scene(env.scene(), config, beams)
    return env, problem, initial


class PhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env, cls.problem, cls.initial = replay_scene()

    def test_complete_replay_matches_legacy_final_capacity_and_metrics(self):
        result = self.problem.evaluate(self.initial)
        observed = np.array([row["capacity_bps"] for row in self.env.beam_results()])
        np.testing.assert_allclose(result.capacity_bps, observed, rtol=3e-10, atol=1e-5)
        self.assertAlmostEqual(result.si, self.env.metrics()["final_si_mean"], places=10)
        self.assertAlmostEqual(result.sgm, self.env.metrics()["final_sgm_mean"], places=10)

    def test_capacity_and_sgm_gradients_match_central_difference(self):
        power = self.initial.copy()
        power[::7] = 2.0
        power[1::11] = 65.0
        ev = self.problem.evaluate(power, with_jacobian=True)
        for j in (0, 1, 8, 18, 61):
            delta = 1e-3
            plus, minus = power.copy(), power.copy()
            plus[j] += delta
            minus[j] -= delta
            e_plus = self.problem.evaluate(plus)
            e_minus = self.problem.evaluate(minus)
            np.testing.assert_allclose(
                (e_plus.capacity_bps - e_minus.capacity_bps) / (2 * delta),
                ev.capacity_jacobian[:, j], rtol=2e-5, atol=2e-2)
            self.assertAlmostEqual((e_plus.sgm - e_minus.sgm) / (2 * delta),
                                   ev.sgm_jacobian[j], delta=2e-7)

    def test_zero_power_and_over_50_w_are_physical(self):
        power = self.initial.copy()
        power[0], power[1] = 0, 75
        ev = self.problem.evaluate(power, with_jacobian=True)
        self.assertEqual(ev.capacity_bps[0], 0)
        self.assertEqual(ev.beam_sgm[0], 0)
        self.assertTrue(np.isfinite(ev.capacity_jacobian).all())
        self.assertTrue(np.isfinite(ev.sgm_jacobian).all())

    def test_exact_demand_has_unit_sgm_and_finite_stationary_gradient(self):
        problem = FixedAllocationPhysics(
            np.array([25e6]), np.array([0]), np.array([0]), np.array([1]),
            np.array([1e-14]), np.zeros((1, 1)), np.array([1e-12]),
            np.array([0]), 3000)
        at_demand = problem.evaluate(np.array([100.0]), with_jacobian=True)
        self.assertAlmostEqual(at_demand.capacity_bps[0], 25e6, delta=1e-5)
        self.assertAlmostEqual(at_demand.sgm, 1.0, places=12)
        self.assertTrue(np.isfinite(at_demand.sgm_jacobian).all())
        self.assertAlmostEqual(at_demand.sgm_jacobian[0], 0.0, places=10)

    def test_rejects_invalid_power_or_actual_allocations(self):
        with self.assertRaises(ValueError):
            self.problem.evaluate(np.full(100, -1.0))
        with self.assertRaises(ValueError):
            self.problem.evaluate(np.full(100, np.nan))
        beams = self.env.beam_results()
        beams[8]["freq"] = beams[0]["freq"]
        with self.assertRaises(ValueError):
            FixedAllocationPhysics.from_scene(self.env.scene(), self.env.config, beams)

    def test_empty_slot_cannot_radiate(self):
        beams = self.env.beam_results()
        beams[0]["slots"] = 0
        beams[0]["freq"] = None
        beams[0]["power_w"] = 0.0
        problem = FixedAllocationPhysics.from_scene(self.env.scene(), self.env.config, beams)
        power = self.initial.copy()
        power[0] = 0
        self.assertEqual(problem.evaluate(power).capacity_bps[0], 0)
        power[0] = 1
        with self.assertRaises(ValueError):
            problem.evaluate(power)


if __name__ == "__main__":
    unittest.main()
