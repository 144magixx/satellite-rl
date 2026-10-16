import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from satellite_rl.coverage.s1fix.bound import compute_bound, grid_edges


def separated_clusters(counts):
    longitude = np.deg2rad(np.repeat(np.arange(len(counts)) * 3., counts))
    return np.column_stack((np.cos(longitude), np.sin(longitude), np.zeros(len(longitude))))


class BoundTests(unittest.TestCase):
    def test_average_and_geometric_lower_bounds(self):
        result = compute_bound(separated_clusters([5, 1]), 2, 5, milp_seconds=0)
        self.assertEqual(result["LB_avg"], 3)
        self.assertEqual(result["LB_lp"], 5)
        self.assertEqual(result["LB_milp"], 5)

    def test_integer_strengthening(self):
        result = compute_bound(separated_clusters([3, 3]), 3, 3, milp_seconds=5)
        self.assertEqual(result["LB_avg"], 2)
        self.assertEqual(result["LB_lp"], 2)
        self.assertEqual(result["LB_milp"], 3)

    def test_unknown_milp_does_not_prove_infeasibility(self):
        timed = SimpleNamespace(status=1, fun=4., mip_dual_bound=2.)
        with patch("satellite_rl.coverage.s1fix.bound.milp", return_value=timed):
            result = compute_bound(separated_clusters([3, 3]), 3, 3, milp_seconds=1)
        self.assertEqual(result["LB_milp"], 2)

    def test_timed_milp_dual_bound_is_used(self):
        timed = SimpleNamespace(status=1, fun=None, mip_dual_bound=3.01)
        with patch("satellite_rl.coverage.s1fix.bound.milp", return_value=timed):
            result = compute_bound(separated_clusters([3, 3]), 3, 3, milp_seconds=1)
        self.assertEqual(result["LB_milp"], 3)

    def test_enlarged_grid_contains_original_disk(self):
        # A true .5-degree beam shifted off the grid must still fit one grid disk.
        center = np.deg2rad([.037, .043])
        longitude = center[0] + np.deg2rad(.5) * np.cos(np.linspace(0, 2*np.pi, 24))
        latitude = center[1] + np.deg2rad(.5) * np.sin(np.linspace(0, 2*np.pi, 24))
        directions = np.column_stack((np.cos(latitude)*np.cos(longitude),
                                      np.cos(latitude)*np.sin(longitude), np.sin(latitude)))
        candidates, aircraft, candidate = grid_edges(directions)
        coverage = np.bincount(candidate, minlength=len(candidates))
        self.assertEqual(coverage.max(), len(directions))
        self.assertEqual(len(np.unique(aircraft)), len(directions))


if __name__ == "__main__":
    unittest.main()
