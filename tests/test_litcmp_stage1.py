import json
import unittest
import numpy as np
from satellite_rl.coverage.s1fix import geom
from satellite_rl.coverage.s1fix.data import Frame
from satellite_rl.litcmp.geom_plane import from_plane, fit_budget, to_plane
from satellite_rl.litcmp.layouts import solve_layout, to_stage2, frame_from_case
from satellite_rl.shared_interface.stages import make_case, UniformStage2Input
from satellite_rl.shared_interface.spherical_gaussian import PHYSICS_V2


class Stage1Tests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        c0 = geom.rays(np.array([[45., 10.]]))[0]
        xy = np.concatenate([rng.normal(center, .055, (100, 2))
                             for center in [(-.6, -.3), (.5, -.2), (0., .6)]])
        xy = np.vstack((xy, [[2., 0.], [-2., 0.], [0., -2.]]))
        positions = geom.ray_to_latlon(from_plane(xy, c0))
        self.frame = Frame("2024-01-12", 0., 0, np.arange(303), positions, np.zeros(303), np.zeros(303))

    def test_methods_and_downstream(self):
        frame = self.frame
        metadata = dict(case_id="synthetic", actual_role="development", source_role="development",
                        original_mother_id="synthetic", UTC="2024-01-12T00:00:00Z",
                        source_asset_SHA="synthetic", source=dict(frame_index=0))
        case = make_case(metadata, [str(i) for i in frame.icao], frame.positions, plain=True)
        self.assertEqual(frame_from_case(case).index, 0)
        timed_case = dict(case, UTC="2024-01-12T00:37:00Z", source=dict(original_case_index=0))
        self.assertEqual(frame_from_case(timed_case).index, 7)
        for method in ("A1", "A2", "A3"):
            with self.subTest(method=method):
                result = solve_layout(frame, 20, method)
                layout = result.layout
                fixed = float(layout.widths[0]) if method == "A1" else None
                valid = geom.validate_layout(layout, geom.rays(frame.positions), 20,
                                             spacing=False, fixed_width=fixed)
                self.assertEqual(result.feasible, valid)
                self.assertTrue(result.feasible)
                if method == "A1":
                    self.assertTrue(np.all(layout.widths == fixed))
                converted_case, converted_layout = to_stage2(result, case, 86400000)
                json.dumps([converted_case, converted_layout])
                UniformStage2Input(converted_case, converted_layout, None, PHYSICS_V2, plain=True)
                self.assertEqual(sum(converted_layout["request_bps_by_beam_int64"]), 303 * 86400000)

    def test_split_merge_and_projection(self):
        c0 = geom.rays(np.array([[45., 10.]]))[0]
        xy = np.array([[-.8, 0.], [-.7, 0.], [.7, 0.], [.8, 0.]])
        points = from_plane(xy, c0)
        np.testing.assert_allclose(to_plane(points, c0), xy, atol=1e-12)
        clusters, centers, feasible = fit_budget(points, [np.arange(4)], 2)
        self.assertTrue(feasible)
        self.assertEqual(len(clusters), 2)
        self.assertTrue(all(geom.angles(center, points[cluster]).max() <= .5
                            for cluster, center in zip(clusters, centers)))
        points = from_plane(xy / 4, c0)
        clusters, _, feasible = fit_budget(points, [np.array([i]) for i in range(4)], 2)
        self.assertTrue(feasible)
        self.assertEqual(len(clusters), 2)
        original = from_plane(np.array([[.05, 0.]]), c0)
        _, centers, _ = fit_budget(points, [np.arange(4)], 2, "keep", original)
        np.testing.assert_array_equal(centers, original)
        displaced = from_plane(np.array([[1.5, 0.]]), c0)
        coincident = np.repeat(points[:1], 2, axis=0)
        _, centers, feasible = fit_budget(coincident, [np.array([0, 1])], 2, "keep", displaced)
        self.assertTrue(feasible)
        np.testing.assert_allclose(centers, points[:1])

    def test_repeated_positions_keep_all_aircraft(self):
        positions = np.repeat(self.frame.positions[:3], 4, axis=0)
        frame = Frame("2024-01-12", 0., 0, np.arange(12), positions, np.zeros(12), np.zeros(12))
        result = solve_layout(frame, 10, "A2")
        self.assertTrue(result.feasible)
        self.assertEqual(len(result.layout.widths), 3)
        self.assertEqual(len(result.layout.owner), 12)


if __name__ == "__main__":
    unittest.main()
