"""Small FLOW-AO v2 geometry/capacity and legacy regression examples."""
import unittest
from unittest.mock import patch
import numpy as np
from satellite_rl.coverage.s1fix.data import Frame
from satellite_rl.coverage.s1fix.geom import rays, validate_layout, cone
from satellite_rl.coverage.s1fix import skeleton as sk
from satellite_rl.coverage.s1fix.skeleton import assign, solve, coverage_radii


class FlowTests(unittest.TestCase):
    def frame(self):
        return Frame("2024-01-12", 0., 0, np.arange(30),
                     np.tile([50., 10.], (30, 1)), np.zeros(30), np.zeros(30))

    def test_dense_stack(self):
        frame = self.frame()
        result = solve(frame, 10, per_beam_cap=8, objective="narrow", stacking=True, time_limit=2)
        self.assertTrue(result.feasible)
        self.assertEqual(len(result.layout.widths), 4)
        self.assertEqual(result.over_cap_beams, 0)
        self.assertTrue(validate_layout(result.layout, rays(frame.positions), 10, per_beam_cap=8))
        self.assertTrue((result.layout.stack_id == 0).all())
        self.assertTrue((coverage_radii(rays(result.layout.centers_latlon), stack_id=result.layout.stack_id) == .5).all())

    def test_flow_cap_and_narrow(self):
        points = rays(np.tile([50., 10.], (12, 1)))
        centers = points[:2]
        self.assertIsNone(assign(centers, points, per_beam_cap=5))
        owner = assign(centers, points, per_beam_cap=6, objective="narrow")
        np.testing.assert_array_equal(np.bincount(owner), [6, 6])

    def test_narrow_keeps_current_peak_budget(self):
        centers = rays(np.array([[50., 0.], [50., .3]]))
        members = rays(np.array([[50., 0.], [50., .01], [50., .02], [50., .29]]))
        owner = assign(centers, members, per_beam_cap=3, objective="narrow", current_peak=3)
        np.testing.assert_array_equal(np.bincount(owner), [3, 1])

    def test_legacy_default(self):
        frame = self.frame()
        result = solve(frame, 20)
        np.testing.assert_array_equal(result.layout.owner, np.zeros(30, dtype=np.int64))
        np.testing.assert_array_equal(result.layout.beam_ids, [0])
        np.testing.assert_array_equal(result.layout.widths, [.300000001])
        self.assertIsNone(result.layout.stack_id)

    def test_global_flow_reassigns_shared_aircraft(self):
        # Aircraft 0 fits both sites; aircraft 1 only fits site 0. A local
        # partition would strand aircraft 1 although the global flow fits both.
        owner = sk._flow(2, 2, np.array([0, 0, 1]), np.array([0, 1, 0]), 1, partial=True)
        np.testing.assert_array_equal(owner, [1, 0])

    def test_insufficient_budget_still_adds_partial_stack(self):
        members = rays(self.frame().positions[:10])
        base = sk._layout(members[:1], members, np.zeros(10, dtype=int), [8])
        layout, over = sk._stack_layout(base, members, 3, 3)
        self.assertEqual(len(layout.widths), 3)
        self.assertEqual(len(layout.owner), 10)
        # Fallback minimizes peak, without a secondary count-of-overloads goal.
        self.assertEqual(np.bincount(layout.owner).max(), 4)
        self.assertEqual(over, int(np.count_nonzero(np.bincount(layout.owner) > 3)))
        self.assertTrue(validate_layout(layout, members, 3))
        self.assertTrue((layout.stack_id == 0).all())

    def test_narrow_moves_complete_stack_as_one_site(self):
        positions = np.array([[50., 9.9], [50., 10.], [50., 10.1],
                              [50., 10.2], [50., 14.], [50., 14.1]])
        members = rays(positions)
        centers = rays(np.array([[50., 9.8], [50., 9.8], [50., 14.]]))
        base = sk._layout(centers, members, np.array([0, 0, 1, 1, 2, 2]),
                          [0, 1, 2], stack_id=np.array([0, 0, -1]))
        layout = sk._narrow_layout(base, members, True, None, np.inf, 3, 2)
        np.testing.assert_allclose(layout.centers_latlon[0], layout.centers_latlon[1], atol=1e-10)
        np.testing.assert_allclose(rays(layout.centers_latlon[:1])[0], cone(members[:4])[0], atol=1e-10)
        self.assertTrue(validate_layout(layout, members, 3, per_beam_cap=2))

    def test_stack_schedule_uses_two_full_phase_budgets(self):
        frame = self.frame()
        members = rays(frame.positions)
        base = sk._layout(members[:1], members, np.zeros(30, dtype=int), [0])
        clock = [100.]
        calls = []
        def single(*args, **kwargs):
            calls.append((args[1], args[4], kwargs.get('per_beam_cap')))
            clock[0] += 120.
            return sk.Result(base, 1, True, 0., 120., 0., 120.)
        with patch.object(sk, 'perf_counter', side_effect=lambda: clock[0]), patch.object(sk, '_single', side_effect=single), patch.object(sk, '_narrow_layout', side_effect=lambda layout, *args: layout) as narrow:
            result = solve(frame, 20, per_beam_cap=8, objective='narrow', stacking=True)
        self.assertEqual(calls, [(20, 220., None), (12, 340., None)])
        self.assertEqual(narrow.call_args.args[4], 400.)
        self.assertEqual(result.over_cap_beams, 0)

    def test_no_deficit_skips_reserved_skeleton(self):
        frame = self.frame()
        members = rays(frame.positions)
        base = sk._layout(members[:1], members, np.zeros(30, dtype=int), [0])
        with patch.object(sk, '_single', return_value=sk.Result(base, 1, True, 0., 0., 0., 0.)) as single, patch.object(sk, '_stack_layout') as stack:
            result = solve(frame, 20, per_beam_cap=30, stacking=True)
        single.assert_called_once()
        stack.assert_not_called()
        self.assertEqual(result.reserve, 0)
        self.assertTrue(result.feasible)



if __name__ == "__main__": unittest.main()
