"""Small FLOW-AO v2 geometry/capacity and legacy regression examples."""
import unittest
import numpy as np
from satellite_rl.coverage.s1fix.data import Frame
from satellite_rl.coverage.s1fix.geom import rays, validate_layout
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



if __name__ == "__main__": unittest.main()
