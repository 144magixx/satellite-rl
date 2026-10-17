import unittest
from unittest.mock import patch
import numpy as np
from test_litcmp_moves import scene
from satellite_rl.s2fix.evaluate import metrics
from satellite_rl.s2fix.greedy import greedy_allocation
from satellite_rl.litcmp.moves import Alloc, legal
from satellite_rl.litcmp.objective import Probe, j_gap, j_w
from satellite_rl.litcmp.anneal import anneal
from satellite_rl.litcmp.genetic import genetic
from satellite_rl.litcmp.convex_relax import convex_relax
from satellite_rl.litcmp.power_lit import noise_limited, B
from satellite_rl.s2fix.evaluate import fixed_problem


class SearchTest(unittest.TestCase):
    def test_convex_relax_bounds_and_noise_limited_service(self):
        stage = scene()
        start = Alloc(*greedy_allocation(stage))
        result = convex_relax(stage, start)
        self.assertTrue(legal(result))
        self.assertLessEqual(result.counts.sum(), 800)
        self.assertLessEqual(result.powers.sum(), 3000.01)
        self.assertTrue(np.all((result.powers >= 0) & (result.powers <= 50.00001)))

        def served(alloc):
            problem = fixed_problem(stage, *alloc)
            gamma = noise_limited(problem, alloc)
            capacity = alloc.counts * B * np.minimum(np.log2(1 + gamma * alloc.powers / alloc.counts), 4.5)
            return np.minimum(capacity, problem.demand_bps).sum()

        self.assertGreaterEqual(served(result) + 1., served(start))

    def test_true_evaluation_counts_and_incumbent(self):
        stage = scene()
        start = Alloc(*greedy_allocation(stage))
        for method, calls in ((anneal, 32), (genetic, 31)):
            for objective in (j_gap, j_w):
                probe = Probe(stage, objective)
                baseline = objective(probe(start, count=False))
                with patch('satellite_rl.litcmp.objective.metrics', wraps=metrics) as evaluate:
                    result = method(stage, start, probe, 32, 51001, objective)
                self.assertEqual(probe.calls, calls)
                self.assertEqual(evaluate.call_count, calls + 1)
                self.assertEqual(len(probe.history), calls)
                self.assertTrue(legal(result))
                self.assertLessEqual(probe.best_value, baseline)
                if method is anneal:
                    self.assertIn(32, probe.curve)
                np.testing.assert_array_equal(result.groups, probe.best.groups)


if __name__ == '__main__':
    unittest.main()
