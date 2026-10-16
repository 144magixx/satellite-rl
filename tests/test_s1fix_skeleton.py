import itertools
from types import SimpleNamespace
import unittest

import numpy as np

from satellite_rl.coverage.s1fix.geom import Layout, angles, envelope, rays, validate_layout
from satellite_rl.coverage.s1fix.skeleton import (
    assign, initial_cover, init_baseline, repair_spacing, solve,
)


class SkeletonTests(unittest.TestCase):
    def test_coincident_centers_fill_budget_without_raising_peak(self):
        frame = SimpleNamespace(positions=np.repeat([[45., 10.]], 8, axis=0))
        for function in (solve, init_baseline):
            result = function(frame, 4, spacing=False, max_iters=1)
            self.assertTrue(result.feasible)
            self.assertEqual(len(result.layout.widths), 4)
            np.testing.assert_array_equal(np.bincount(result.layout.owner), [2, 2, 2, 2])

    def test_flow_matches_exhaustive_peak_and_balance(self):
        centers = rays(np.array([[45., 9.], [45., 11.], [45., 13.]]))
        members = rays(np.array([[45., x] for x in [8., 9., 10., 11., 12., 13., 14.]]))
        eligible = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 0], [0, 1, 1],
                             [0, 1, 1], [0, 0, 1], [0, 0, 1]], dtype=bool)
        options = [np.flatnonzero(row) for row in eligible]
        brute = min((max(np.bincount(x, minlength=3)), sum(np.bincount(x, minlength=3) ** 2))
                    for x in itertools.product(*options))
        peak = assign(centers, members, eligible_mask=eligible)
        balanced = assign(centers, members, "balance", eligible_mask=eligible)
        self.assertEqual(np.bincount(peak, minlength=3).max(), brute[0])
        loads = np.bincount(balanced, minlength=3)
        self.assertEqual((loads.max(), sum(loads ** 2)), brute)
        self.assertTrue(eligible[np.arange(len(members)), balanced].all())

    def test_assignment_cost_and_impossible(self):
        centers = rays(np.array([[45., 10.], [45., 10.1]]))
        members = rays(np.array([[45., 10.02], [45., 10.08]]))
        previous = np.array([1, 0])
        owner = assign(centers, members, "balance", prev_owner=previous, kappa=10)
        np.testing.assert_array_equal(owner, previous)
        self.assertIsNone(assign(centers, members, eligible_mask=np.array([[0, 0], [1, 1]], bool)))
        self.assertIsNone(assign(centers, rays(np.array([[70., -20.]]))))

    def test_small_solve_and_baseline(self):
        positions = np.array([[45., 10.], [45.2, 10.2], [45.4, 10.4],
                              [50., 20.], [50.1, 20.1], [50.2, 20.2]])
        frame = SimpleNamespace(positions=positions)
        members = rays(positions)
        initial = initial_cover(members)
        self.assertTrue((angles(members, initial).min(axis=1) <= .5 + 1e-9).all())
        for function in (solve, init_baseline):
            result = function(frame, 4, spacing=False, max_iters=2)
            self.assertTrue(result.feasible)
            self.assertTrue(validate_layout(result.layout, members, 4, spacing=False))
            self.assertEqual(len(result.layout.owner), len(positions))
            self.assertGreaterEqual(result.t_total, result.t_init)
        result = solve(frame, 1, max_iters=1)
        self.assertFalse(result.feasible)
        self.assertGreater(result.N0, 1)
        initial = init_baseline(frame, 4, spacing=False)
        same_n = solve(frame, len(initial.layout.widths), spacing=False,
                       init_layout=initial.layout, preserve_N=True, max_iters=2)
        self.assertTrue(same_n.feasible)
        self.assertEqual(len(same_n.layout.widths), len(initial.layout.widths))

    def test_spacing_and_fixed_width(self):
        frame = SimpleNamespace(positions=np.array([[45., 10.], [45.1, 10.1], [50., 20.]]))
        result = solve(frame, 3, spacing=True, fixed_width=.5, max_iters=1)
        self.assertTrue(result.feasible)
        self.assertTrue(validate_layout(result.layout, rays(frame.positions), 3, fixed_width=.5))
        np.testing.assert_allclose(result.layout.widths, .5)

    def test_spacing_merge_keeps_ids_and_rolls_back(self):
        positions = np.array([[45., 10.], [45.01, 10.01], [50., 20.], [50.01, 20.01]])
        members = rays(positions)
        centers = positions[[0, 1, 2]]
        owner = np.array([0, 1, 2, 2])
        layout = Layout(centers, envelope(rays(centers), members, owner), owner, np.array([4, 8, 12]))
        saved = layout.owner.copy()
        repaired = repair_spacing(layout, members, 3)
        self.assertIsNotNone(repaired)
        self.assertTrue(validate_layout(repaired, members, 3))
        self.assertEqual(set(repaired.beam_ids), {4, 12})
        np.testing.assert_array_equal(layout.owner, saved)


if __name__ == "__main__":
    unittest.main()
