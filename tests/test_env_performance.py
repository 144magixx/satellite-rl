"""Geometry-cache correctness and invalidation in both environment variants."""
import importlib
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


def uncached_angle(module, beam_lat, beam_lon, current_lat, current_lon):
    """Scalar geometry oracle retaining the original arithmetic order."""
    sat_lon_rad = np.deg2rad(module.satellite_lon)
    x_sat = module.R * np.cos(sat_lon_rad)
    y_sat = module.R * np.sin(sat_lon_rad)
    vec_beam = module.geo_to_cartesian(beam_lat, beam_lon) - [x_sat, y_sat, 0.0]
    vec_current = module.geo_to_cartesian(current_lat, current_lon) - [x_sat, y_sat, 0.0]
    cos_theta = np.dot(vec_beam, vec_current) / (
        np.linalg.norm(vec_beam) * np.linalg.norm(vec_current)
    )
    return np.rad2deg(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


class EnvironmentPerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output_root = tempfile.TemporaryDirectory(prefix='satellite_env_perf_test_')
        cls.output_environment = patch.dict(os.environ, {
            'SATELLITE_RL_OUTPUT_ROOT': cls.output_root.name,
            'MPLBACKEND': 'Agg',
        })
        cls.output_environment.start()
        cls.modules = tuple(importlib.import_module('satellite_rl.envs.' + name) for name in (
            'satellite', 'satellite_variant_103',
        ))

    @classmethod
    def tearDownClass(cls):
        cls.output_environment.stop()
        cls.output_root.cleanup()

    def setUp(self):
        self.random_state = np.random.get_state()
        self.addCleanup(np.random.set_state, self.random_state)

    def make_env(self, module, seed=20260924):
        np.random.seed(seed)
        scene = module.BeamInfo()
        env = module.Env(module.SatelliteFreqState(), scene,
                         module.CurrentBeamState(scene.beam_info))
        env.reset(scene)
        return env, scene

    def test_repeated_geometry_reuses_only_coordinate_work(self):
        for module in self.modules:
            with self.subTest(environment=module.__name__):
                env, _ = self.make_env(module)
                points = (np.float64(30.123), np.float64(110.234),
                          np.float64(31.345), np.float64(112.456))
                expected = uncached_angle(module, *points)
                random_before = np.random.get_state()
                with patch.object(module, 'geo_to_cartesian', wraps=module.geo_to_cartesian) as convert:
                    self.assertEqual(env.calculate_beam_angle(*points), expected)
                    self.assertEqual(env.calculate_beam_angle(*points), expected)
                    self.assertEqual(convert.call_count, 2)
                for before, after in zip(random_before, np.random.get_state()):
                    np.testing.assert_array_equal(before, after)

    def test_equal_coordinates_with_different_dtypes_do_not_share_rounding(self):
        for module in self.modules:
            with self.subTest(environment=module.__name__):
                env, _ = self.make_env(module)
                # All four values are exactly representable in both dtypes; the
                # trigonometric input conversion still has different precision.
                for dtype in (np.float32, np.float64, float):
                    points = tuple(dtype(value) for value in (30.125, 110.25, 31.375, 112.5))
                    self.assertEqual(env.calculate_beam_angle(*points), uncached_angle(module, *points))

    def test_geometry_constants_and_scene_reset_invalidate_cached_values(self):
        for module in self.modules:
            with self.subTest(environment=module.__name__):
                env, _ = self.make_env(module)
                points = tuple(env.beam_info['beam_location'][:2].flatten())
                env.calculate_beam_angle(*points)
                for constant, value in (('R', 43000), ('r', 6400), ('satellite_lon', 123.4)):
                    with patch.object(module, constant, value):
                        self.assertEqual(env.calculate_beam_angle(*points), uncached_angle(module, *points))
                self.assertEqual(env.calculate_beam_angle(*points), uncached_angle(module, *points))
                np.random.seed(20260925)
                new_scene = module.BeamInfo()
                env.reset(new_scene)
                self.assertEqual(len(env._beam_geometry_cache), 0)
                new_points = tuple(new_scene.beam_info['beam_location'][:2].flatten())
                self.assertEqual(env.calculate_beam_angle(*new_points), uncached_angle(module, *new_points))

    def test_arbitrary_geometry_queries_remain_bounded_and_correct(self):
        for module in self.modules:
            with self.subTest(environment=module.__name__):
                env, _ = self.make_env(module)
                for index in range(2 * env.beam_num + 5):
                    points = (30.0 + index / 1000, 110.0, 31.0, 111.0)
                    self.assertEqual(env.calculate_beam_angle(*points), uncached_angle(module, *points))
                    self.assertLessEqual(len(env._beam_geometry_cache), 2 * env.beam_num)


if __name__ == '__main__':
    unittest.main()
