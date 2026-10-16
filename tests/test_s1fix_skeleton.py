import itertools
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from satellite_rl.coverage.s1fix import skeleton
from satellite_rl.coverage.s1fix.geom import angles, rays, validate_layout
from satellite_rl.coverage.s1fix.skeleton import (
    assign, initial_cover, init_baseline, coverage_radii, centers_separated, solve, _spread_split, _edges,
)


class SkeletonTests(unittest.TestCase):
    def test_coincident_split_is_rejected(self):
        frame = SimpleNamespace(positions=np.repeat([[45., 10.]], 8, axis=0))
        for function in (solve, init_baseline):
            result = function(frame, 4, spacing=False, max_iters=1)
            self.assertTrue(result.feasible)
            self.assertEqual(len(result.layout.widths), 1)
            self.assertEqual(np.bincount(result.layout.owner).max(), 8)

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

    def test_later_growth_uses_full_budget_and_updates_best(self):
        frame = SimpleNamespace(positions=np.array([
            [45., 10.], [45.2, 10.2], [45.4, 10.4],
            [50., 20.], [50.1, 20.1], [50.2, 20.2]]))
        original = skeleton._add_beams
        calls = []
        def delayed_growth(layout, members, budget, *args):
            calls.append(budget)
            if len(calls) == 1:
                return layout, args[2]
            return original(layout, members, budget, *args)
        with patch.object(skeleton, "_add_beams", side_effect=delayed_growth):
            result = solve(frame, 4, spacing=False, max_iters=1)
        self.assertTrue(result.feasible)
        self.assertEqual(len(result.layout.widths), 4)
        self.assertEqual(calls[0], 4)
        self.assertEqual(calls[-1], 4)

    def test_spacing_and_fixed_width(self):
        frame = SimpleNamespace(positions=np.array([[45., 10.], [45.1, 10.1], [50., 20.]]))
        result = solve(frame, 3, spacing=True, fixed_width=.5, max_iters=1)
        self.assertTrue(result.feasible)
        self.assertTrue(validate_layout(result.layout, rays(frame.positions), 3, fixed_width=.5))
        np.testing.assert_allclose(result.layout.widths, .5)

    def test_radius_limits_and_eligible_nearest(self):
        centers = rays(np.array([[45., 10.], [45., 12.], [50., 20.]]))
        distance = angles(centers, centers)
        np.fill_diagonal(distance, np.inf)
        np.testing.assert_allclose(coverage_radii(centers), np.minimum(.5, distance.min(axis=1)))
        np.testing.assert_allclose(coverage_radii(centers, False), .5)
        np.testing.assert_allclose(coverage_radii(centers, fixed_width=.5), .25)
        self.assertFalse(centers_separated(np.repeat(centers[:1], 2, axis=0), False))
        members = rays(np.array([[45., 10.1], [45., 11.9]]))
        owner = assign(centers, members, radii=np.array([.001, .5, .5]))
        self.assertIsNotNone(owner)
        self.assertNotIn(0, owner)

    def test_outward_split_stops_at_first_legal_step(self):
        def directions(degrees):
            theta = np.deg2rad(degrees)
            return np.column_stack((np.sin(theta), np.zeros(len(theta)), np.cos(theta)))
        split = directions([-.02, .02])
        for minimum in (.15, .25):
            spread = next(_spread_split(split, np.empty((0, 3)), minimum))
            separation = angles(spread[:1], spread[1:])[0, 0]
            self.assertGreaterEqual(separation + 1e-9, minimum)
            self.assertLess(separation - .01, minimum)
        others = directions([.1])
        spread = next(_spread_split(split, others, .15))
        self.assertTrue((angles(spread, others) >= .15 - 1e-9).all())
        self.assertEqual(list(_spread_split(split, directions([-.2, .2]), .25)), [])

    def test_split_retries_after_legal_candidate_assignment_fails(self):
        frame = SimpleNamespace(positions=np.array([[45., 10.], [45., 10.2], [45., 10.4]]))
        members = rays(frame.positions)
        owner = np.zeros(3, dtype=np.int64)
        layout = skeleton._layout(members[:1], members, owner, np.array([0]))
        first, second = members[[0, 1]], members[[0, 2]]
        with patch.object(skeleton, "_spread_split", return_value=iter([first, second])), \
             patch.object(skeleton, "assign", side_effect=[None, np.array([0, 0, 1])]) as assignment:
            result, _ = skeleton._add_beams(layout, members, 2, True, None, 1)
        self.assertEqual(assignment.call_count, 2)
        self.assertEqual(len(result.widths), 2)

    def test_sparse_edges_match_per_beam_radii(self):
        centers = rays(np.array([[45., 10.], [45., 12.], [50., 20.]]))
        members = rays(np.array([[45., x] for x in np.linspace(8, 14, 20)]))
        radii = np.array([.05, .3, .5])
        expected = np.nonzero(angles(members, centers) <= radii + 1e-9)
        actual = _edges(centers, members, radii)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])

    def test_initial_net_has_separated_centers(self):
        members = rays(np.array([[45., x] for x in np.linspace(0, 30, 100)]))
        for radius in (.5, .25):
            centers = initial_cover(members, radius)
            distance = angles(centers, centers)
            np.fill_diagonal(distance, np.inf)
            self.assertTrue((distance > radius).all())
            self.assertTrue((angles(members, centers).min(axis=1) <= radius + 1e-9).all())


if __name__ == "__main__":
    unittest.main()
