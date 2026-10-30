import unittest

import numpy as np

from satellite_rl.coverage.s1fix.geom import (
    Layout, angles, cone, envelope, forecast, metrics, ray_to_latlon, rays,
    validate_layout,
)


class GeometryTests(unittest.TestCase):
    def test_ray_roundtrip_and_pairwise(self):
        positions = np.array([[45., 10.], [55., -5.], [35., 20.]])
        directions = rays(positions)
        np.testing.assert_allclose(ray_to_latlon(directions), positions, atol=1e-10)
        self.assertEqual(angles(directions, directions).shape, (3, 3))
        np.testing.assert_allclose(np.diag(angles(directions, directions)), 0, atol=1e-12)

    def test_cone_and_widths(self):
        directions = rays(np.array([[45., 10.], [46., 11.], [45., 11.]]))
        center, half = cone(directions)
        self.assertLessEqual(half, angles(directions[0], directions).max())
        np.testing.assert_allclose(np.linalg.norm(center), 1.)
        self.assertTrue(np.all(angles(center, directions) <= half + 1e-10))
        self.assertAlmostEqual(envelope(center[None], directions, np.zeros(3, dtype=int))[0],
                               max(.3, 2 * half) + 1e-9)
        center = np.array([[1., 0., 0.]])
        edge = np.array([[np.cos(np.deg2rad(.5)), np.sin(np.deg2rad(.5)), 0.]])
        self.assertEqual(envelope(center, edge, np.array([0]))[0], 1.)

    def test_validate_envelope_spacing_and_nonempty(self):
        positions = np.array([[45., 10.], [45.1, 10.1], [50., 20.]])
        directions = rays(positions)
        centers = positions[[0, 2]]
        owner = np.array([0, 0, 1])
        widths = envelope(rays(centers), directions, owner)
        layout = Layout(centers, widths, owner, np.array([7, 11]))
        self.assertTrue(validate_layout(layout, directions, 2))
        self.assertFalse(validate_layout(layout, directions, 1))
        wide = Layout(centers, np.full(2, 1.), owner, layout.beam_ids)
        self.assertFalse(validate_layout(wide, directions, 2))  # not member envelope
        overlap = Layout(np.repeat(positions[:1], 2, axis=0), np.full(2, .3 + 1e-9),
                         np.array([0, 1]), np.array([0, 1]))
        self.assertFalse(validate_layout(overlap, directions[:2], 2))
        self.assertTrue(validate_layout(overlap, directions[:2], 2, spacing=False))
        self.assertEqual(metrics(layout)["peak"], 2)

    def test_forecast_missing_and_eastward(self):
        result = forecast(np.array([45., 45.]), np.array([10., 10.]),
                          np.array([360., np.nan]), np.array([90., 90.]), 60.)
        self.assertGreater(result[0, 1], 10.)
        np.testing.assert_array_equal(result[1], [45., 10.])
        np.testing.assert_array_equal(forecast(45.123, 10.456, 400., np.nan, 60.), [45.123, 10.456])


if __name__ == "__main__":
    unittest.main()
