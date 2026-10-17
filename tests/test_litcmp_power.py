import unittest
import numpy as np
from test_s2fix_physics import scene
from satellite_rl.s2fix.greedy import greedy_allocation
from satellite_rl.s2fix.evaluate import fixed_problem
from satellite_rl.litcmp.moves import Alloc, legal
from satellite_rl.litcmp.objective import Probe
from satellite_rl.litcmp.power_lit import B, C1, C2, noise_limited, waterfill
from satellite_rl.litcmp.convex_relax import convex_relax


class LiteraturePowerTest(unittest.TestCase):
    def test_waterfill_demand_and_budget(self):
        gamma, k, demand = np.array([.1, .2]), np.array([2, 3]), np.array([1e7, 2e7])
        expected = k * np.expm1(demand / (k * B) * np.log(2)) / gamma
        np.testing.assert_allclose(waterfill(gamma, k, demand), expected)
        gamma = np.linspace(.25, .5, 100)
        k, demand = np.full(100, 10), np.full(100, 1e10)
        p = waterfill(gamma, k, demand)
        self.assertAlmostEqual(p.sum(), 3000, places=5)
        self.assertTrue(np.all((p > 0) & (p < 50)))
        marginal = B * gamma / ((1 + gamma * p / k) * np.log(2))
        np.testing.assert_allclose(marginal, marginal.mean(), rtol=1e-3)

    def test_physical_units_and_convex_relax(self):
        s = scene()
        a = Alloc(*greedy_allocation(s))
        problem = fixed_problem(s, *a)
        gamma = noise_limited(problem, a)
        problem.interference = np.zeros_like(problem.interference)
        p = np.array([12., 15.])
        capacity = a.counts * B * np.minimum(np.log2(1 + gamma * p / a.counts), 4.5)
        np.testing.assert_allclose(capacity, problem.evaluate(p).capacity_bps, rtol=1e-13)
        # Mean interference belongs inside the denominator, rather than
        # averaging per-slot inverse denominators (a different surrogate).
        interference = np.arange(len(problem.slot_owner)) * problem.noise
        frozen_gamma = noise_limited(problem, a, interference)
        for beam in range(len(a.counts)):
            slots = problem.slot_owner == beam
            np.testing.assert_array_equal(problem.signal[slots], np.full(slots.sum(), problem.signal[slots][0]))
            np.testing.assert_array_equal(problem.noise[slots], np.full(slots.sum(), problem.noise[slots][0]))
            expected = (problem.signal[slots].mean() * a.counts[beam]
                        / (problem.noise[slots].mean() + interference[slots].mean())
                        * 10 ** (-problem.implementation_loss_dB / 10))
            self.assertAlmostEqual(frozen_gamma[beam], expected)
        result = convex_relax(s, a)
        self.assertTrue(legal(result))
        self.assertLessEqual(result.counts.sum(), 800)
        self.assertLessEqual(result.powers.sum(), 3000.01)
        def relaxed(x):
            c = x.counts * B * np.minimum(np.log2(1 + gamma * x.powers / x.counts), 4.5)
            return np.minimum(c, problem.demand_bps).sum()
        self.assertGreaterEqual(relaxed(result) + 1, relaxed(a))
        self.assertTrue(np.all(C1(s, a) <= 50))

    def test_c2_frozen_objective(self):
        # Fine-stage guarantee concerns its frozen approximation, not the
        # true coupled interference after both beams change power.
        s = scene()
        a = Alloc(np.array([0, 2]), np.array([0, 0]), np.array([4, 4]), np.array([31., 43.]))
        probe = Probe(s)
        p = C2(s, a, 51001, probe)
        self.assertEqual(probe.calls, 320)
        coarse = probe.best
        problem = fixed_problem(s, *coarse)
        gamma = noise_limited(problem, coarse, problem.interference @ coarse.powers)
        omega = .001 * problem.demand_bps.sum() / 3000
        def frozen_objective(power):
            capacity = coarse.counts * B * np.minimum(np.log2(1 + gamma * power / coarse.counts), 4.5)
            return np.maximum(problem.demand_bps - capacity, 0).sum() + omega * power.sum()
        self.assertLessEqual(frozen_objective(p), frozen_objective(coarse.powers) + 1)


if __name__ == '__main__':
    unittest.main()
