import itertools
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from satellite_rl.coverage.s1fix import skeleton
from satellite_rl.coverage.s1fix.geom import angles, rays, ray_to_latlon, validate_layout
from satellite_rl.coverage.s1fix.skeleton import (
    adaptive_net, assign, initial_cover, init_baseline, coverage_radii, centers_separated,
    solve, _spread_split, _edges,
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

    def test_blocked_recenter_does_not_discard_load_neutral_tightening(self):
        axis = rays(np.array([[45., 10.]]))[0]
        tangent = np.cross(axis, np.array([0., 0., 1.]))
        tangent /= np.linalg.norm(tangent)

        def directions(degrees):
            theta = np.deg2rad(degrees)[:, None]
            return axis * np.cos(theta) + tangent * np.sin(theta)

        centers = directions([0., .2, 1.])
        members = directions([.08, .08, .2, .2, 1.12, 1.16])
        owner = np.repeat(np.arange(3), 2)
        initial = skeleton._layout(centers, members, owner, np.arange(3))
        self.assertTrue(validate_layout(initial, members, 3))

        # The first beam's ideal center enters its neighbor's minimum footprint.
        blocked = centers.copy()
        blocked[0] = skeleton.cone(members[owner == 0])[0]
        self.assertFalse(validate_layout(
            skeleton._layout(blocked, members, owner, np.arange(3)), members, 3))

        frame = SimpleNamespace(positions=ray_to_latlon(members))
        result = solve(frame, 3, init_layout=initial, preserve_N=True, max_iters=1)
        self.assertTrue(result.feasible)
        np.testing.assert_array_equal(np.bincount(result.layout.owner), [2, 2, 2])
        final_centers = rays(result.layout.centers_latlon)
        np.testing.assert_allclose(final_centers[0], centers[0], atol=1e-10)
        np.testing.assert_allclose(final_centers[2], skeleton.cone(members[owner == 2])[0],
                                   atol=1e-10)
        self.assertLess(result.layout.widths[2], initial.widths[2] - .01)

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
             patch.object(skeleton, "_at_peak", side_effect=[None, np.array([0, 0, 1])]) as feasibility:
            result, _ = skeleton._add_beams(layout, members, 2, True, None, 1)
        self.assertEqual(feasibility.call_count, 2)
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

    def test_adaptive_net_covers_dense_and_sparse_regions_under_radius_caps(self):
        theta = np.deg2rad([0., .01, .02, .16, .17, .3, .7, 1.3, 2., 2.45])
        members = np.column_stack((np.sin(theta), np.zeros(len(theta)), np.cos(theta)))
        for k in (1, 2, len(members)):
            for fixed in (None, .5):
                centers = adaptive_net(members, k, fixed)
                self.assertTrue(centers_separated(centers, fixed_width=fixed))
                eligible = angles(members, centers) <= coverage_radii(centers, fixed_width=fixed) + 1e-9
                self.assertTrue(eligible.any(axis=1).all())
                self.assertEqual(len(centers), len(np.unique(centers, axis=0)))
        # Sparse endpoints open before the dense cluster; ties use aircraft order.
        np.testing.assert_array_equal(adaptive_net(members, 2)[0], members[6])
        np.testing.assert_array_equal(adaptive_net(members, 2), adaptive_net(members, 2))

    def test_init_uses_adaptive_net_without_flow(self):
        frame = SimpleNamespace(positions=np.array([[45., 10.], [45.1, 10.1], [50., 20.]]))
        with patch.object(skeleton, "assign", side_effect=AssertionError("INIT must use nearest")), \
                patch.object(skeleton, "_add_beams", side_effect=AssertionError("INIT must use initial centers")), \
                patch.object(skeleton, "adaptive_net", wraps=adaptive_net) as net:
            result = init_baseline(frame, 3)
        self.assertTrue(result.feasible)
        self.assertTrue(net.called)
        self.assertEqual(result.N0, len(initial_cover(rays(frame.positions))))


if __name__ == "__main__":
    unittest.main()
