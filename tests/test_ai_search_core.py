import unittest
import numpy as np
from satellite_rl.coverage.ai_search.config import GeometryConfig
from satellite_rl.coverage.ai_search.core import (Scene, Layout, initial_layout, legal_matrix,
    validate_layout, metrics, minimum_enclosing_width, layout_sha256)
from satellite_rl.coverage.geometry import ray_angle_numpy, satellite_rays_numpy


class AI17CoreTest(unittest.TestCase):
    def test_full_half_width_and_independent_angle(self):
        scene = Scene([[50, 10], [51, 10]], [1, 0], ("hot", "edge"))
        angle = float(ray_angle_numpy(scene.rays[0], scene.rays[1]))
        cfg = GeometryConfig(width_min_deg=.0001, width_max_deg=2., width_menu_deg=(.0001, .5, 2.))
        narrow = 2 * angle - 1e-5
        wide = 2 * angle + 1e-5
        self.assertFalse(legal_matrix(scene, [[50, 10]], [narrow], cfg)[1, 0])
        self.assertTrue(legal_matrix(scene, [[50, 10]], [wide], cfg)[1, 0])
        required = minimum_enclosing_width(scene, [0, 1], [50, 10], cfg)
        self.assertGreaterEqual(required, 2 * angle)
        self.assertTrue(legal_matrix(scene, [[50, 10]], [required], cfg).all())

    def test_variable_width_changes_actual_coverage(self):
        scene = Scene([[50, 10], [54, 10]], [5, 0])
        narrow = legal_matrix(scene, [[50, 10]], [.3])
        wide = legal_matrix(scene, [[50, 10]], [1.])
        self.assertTrue(narrow[0, 0]); self.assertFalse(narrow[1, 0]); self.assertTrue(wide.all())

    def test_zero_demand_aircraft_kept_and_conserved(self):
        scene = Scene([[50, 10], [50.1, 10], [65, 30]], [10, 0, 0], ("a", "b", "c"))
        layout = initial_layout(scene)
        evidence = validate_layout(scene, layout)
        self.assertTrue(evidence["valid"]); self.assertEqual(len(layout.assignment), 3)
        self.assertEqual(sum(evidence["loads"]), 10)
        self.assertEqual(metrics(scene, layout)["aircraft_count"], 3)

    def test_empty_and_zero_quality_undefined(self):
        empty = Scene([], [])
        layout = initial_layout(empty)
        self.assertTrue(validate_layout(empty, layout)["valid"])
        self.assertEqual(metrics(empty, layout)["N"], 0)
        scene = Scene([[50, 10]], [0])
        self.assertIsNone(metrics(scene, initial_layout(scene))["CV"])

    def test_immutability_identity_and_fractional_owner(self):
        scene = Scene([[50, 10]], [2], ("stable",))
        layout = initial_layout(scene)
        with self.assertRaises(ValueError): scene.demands[0] = 5
        with self.assertRaises(ValueError): layout.widths[0] = .3
        with self.assertRaises(ValueError): Layout([[50, 10]], [.5], [.5])
        with self.assertRaises(ValueError): Scene([[50, 10], [50, 11]], [1, 2], ("x", "x"))
        self.assertEqual(layout_sha256(layout), layout_sha256(Layout.from_record(layout.to_record())))
        self.assertIn("widths_deg", layout.to_record())

    def test_invalid_width_visibility_owner_empty_and_N200(self):
        scene = Scene([[50, 10]], [1])
        for layout in (Layout([[50, 10]], [.29], [0]), Layout([[50, 10]], [1.01], [0]),
                       Layout([[5, 10]], [.5], [0]), Layout([[50, 10]], [.5], [-1]),
                       Layout([[50, 10], [50, 11]], [.5, .5], [0])):
            self.assertFalse(validate_layout(scene, layout)["valid"])
        many = Scene(np.tile([50, 10], (200, 1)), np.ones(200))
        self.assertTrue(validate_layout(many, Layout(many.positions, np.full(200, .3), np.arange(200)))["valid"])
        too_many = Scene(np.tile([50, 10], (201, 1)), np.ones(201))
        self.assertFalse(validate_layout(too_many, Layout(too_many.positions, np.full(201, .3), np.arange(201)))["valid"])

    def test_solid_angle_exact_and_scale_invariance(self):
        scene = Scene([[50, 10], [50.1, 10]], [1, 3])
        layout = Layout(scene.positions, [.3, 1], [0, 1])
        value = metrics(scene, layout)
        self.assertAlmostEqual(value["solid_angle_sum_sr"], float(sum(2*np.pi*(1-np.cos(np.deg2rad(np.array([.3, 1])/2))))))
        scaled = metrics(Scene(scene.positions, scene.demands*7), layout)
        self.assertEqual(value["P"], scaled["P"])
        self.assertAlmostEqual(value["Q"], 1+value["CV"]**2)


if __name__ == "__main__": unittest.main()
