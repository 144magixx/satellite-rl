"""Correctness of separate Europe geometry; these are not learning experiments."""

from dataclasses import FrozenInstanceError
from unittest.mock import patch
import unittest

import numpy as np

from satellite_rl.envs import satellite as legacy
from satellite_rl.envs.regions import (
    elevation_degrees, geo_distances_km, get_profile, profile_metadata,
)
from satellite_rl.workbench.config import catalog, config_summary, default_config, validate_config
from satellite_rl.workbench.environment import WorkbenchEnv, validate_scene


def europe_config():
    config = default_config()
    config["environment"]["profile"] = "europe_v1"
    config["environment"]["action_mask"] = "joint_feasible"
    return config


def step_first_legal(env):
    mask = env.action_mask()
    frequency, slot_index = np.argwhere(mask["frequency_slots"])[0]
    power_index = np.flatnonzero(mask["power"])[0]
    return env.step([int(frequency), int(slot_index), int(power_index)])


class EuropeEnvironmentTests(unittest.TestCase):
    def test_profile_validation_and_versioned_state_semantics(self):
        self.assertEqual(validate_config({})["environment"]["profile"], "china_legacy")
        with self.assertRaises(ValueError):
            validate_config({"environment": {"profile": "europe"}})
        with self.assertRaises(ValueError):
            validate_config({"environment": {"profile": []}})
        with self.assertRaises(ValueError):
            validate_config({"environment": {"profile": get_profile("europe_v1")}})
        self.assertEqual({p["id"] for p in catalog()["environment_profiles"]},
                         {"china_legacy", "europe_v1"})
        p = get_profile("europe_v1")
        with self.assertRaises(FrozenInstanceError):
            p.satellite_longitude_deg = 122.2
        summary = config_summary(europe_config())
        self.assertEqual(summary["state_dim"], 204)
        self.assertEqual(summary["environment_metadata"]["position_normalization"]["version"],
                         "europe_v1_latlon_affine_v1")
        self.assertEqual(summary["environment_metadata"]["coordinate_order"], ["latitude", "longitude"])
        self.assertIn("工程约定", summary["environment_metadata"]["parameter_basis"])

    def test_nadir_range_elevation_and_independent_direction_oracle(self):
        profile = get_profile("europe_v1")
        self.assertAlmostEqual(geo_distances_km([[0, 13]], profile)[0], 42164 - 6371)
        self.assertAlmostEqual(elevation_degrees([[0, 13]], profile)[0], 90)
        self.assertLess(elevation_degrees([[0, -167]], profile)[0], 0)
        env = WorkbenchEnv(europe_config())
        env.reset(seed=710)
        coordinates = [(40.0, 10.0), (41.0, 14.0)]
        sat = np.array([42164 * np.cos(np.radians(13)), 42164 * np.sin(np.radians(13)), 0])
        directions = []
        for lat, lon in coordinates:
            ground = np.array([6371 * np.cos(np.radians(lat)) * np.cos(np.radians(lon)),
                               6371 * np.cos(np.radians(lat)) * np.sin(np.radians(lon)),
                               6371 * np.sin(np.radians(lat))])
            directions.append(ground - sat)
        expected = np.degrees(np.arccos(np.dot(*directions) /
                                       (np.linalg.norm(directions[0]) * np.linalg.norm(directions[1]))))
        self.assertAlmostEqual(env.raw_env.calculate_beam_angle(*coordinates[0], *coordinates[1]), expected, places=12)

    def test_import_rejects_wrong_region_swapped_coordinates_and_low_elevation(self):
        env = WorkbenchEnv(europe_config())
        env.reset(seed=711)
        scene = env.scene()
        validate_scene(scene, "europe_v1")
        for position, message in [([40, 122.2], "地区"), ([13, 40], "纬度"), ([79, 13], "仰角")]:
            invalid = {key: value.copy() for key, value in scene.items()}
            invalid["beam_location"][0] = position
            invalid["distances"] = geo_distances_km(invalid["beam_location"], "europe_v1")
            with self.assertRaisesRegex(ValueError, message):
                env.reset(scene=invalid)
        inconsistent = {key: value.copy() for key, value in scene.items()}
        inconsistent["distances"] = geo_distances_km(scene["beam_location"], "china_legacy")
        with self.assertRaisesRegex(ValueError, "距离"):
            env.reset(scene=inconsistent)

    def test_synthetic_reference_provenance_and_affine_normalization(self):
        env = WorkbenchEnv(europe_config())
        state = env.reset(seed=712)
        scene = env.scene()
        p = get_profile("europe_v1")
        lat, lon = scene["beam_location"][0]
        self.assertAlmostEqual(float(state[0]), 2 * (lat - 52.5) / 55, places=6)
        self.assertAlmostEqual(float(state[1]), 2 * (lon - 5) / 90, places=6)
        self.assertGreaterEqual(elevation_degrees(scene["beam_location"], p).min(), 10)
        self.assertEqual(len(np.unique(scene["beam_location"], axis=0)), 100)
        self.assertFalse(env.scene_metadata["measured_traffic"])
        self.assertEqual(env.scene_metadata["kind"], "europe_synthetic_reference_grid")
        self.assertGreater(env.scene_metadata["excluded_below_elevation_count"], 0)
        second = WorkbenchEnv(europe_config())
        np.testing.assert_array_equal(second.reset(seed=712), state)
        for key in scene:
            np.testing.assert_array_equal(scene[key], second.scene()[key])
        self.assertEqual(profile_metadata(p)["satellite_longitude_deg"], 13)

    def test_alternating_regions_and_global_mutation_do_not_change_instances(self):
        europe = WorkbenchEnv(europe_config())
        config = default_config()
        config["environment"]["action_mask"] = "joint_feasible"
        china = WorkbenchEnv(config)
        europe.reset(seed=713)
        china.reset(seed=713)
        europe_control, china_control = WorkbenchEnv(europe_config()), WorkbenchEnv(config)
        europe_control.reset(scene=europe.scene())
        china_control.reset(scene=china.scene())
        initial_lon = legacy.satellite_lon
        for _ in range(12):
            eu_expected = step_first_legal(europe_control)
            cn_expected = step_first_legal(china_control)
            with patch.object(legacy, "satellite_lon", -120.0):
                eu_actual = step_first_legal(europe)
                cn_actual = step_first_legal(china)
            for actual, expected in [(eu_actual, eu_expected), (cn_actual, cn_expected)]:
                np.testing.assert_array_equal(actual[0], expected[0])
                self.assertEqual(actual[1:3], expected[1:3])
                self.assertEqual(actual[3]["actual_action"], expected[3]["actual_action"])
        self.assertEqual(legacy.satellite_lon, initial_lon)
        np.testing.assert_array_equal(europe.reset(seed=713), europe_control.reset(seed=713))
        np.testing.assert_array_equal(china.reset(seed=713), china_control.reset(seed=713))

    def test_complete_europe_episode_resource_and_final_ledger(self):
        env = WorkbenchEnv(europe_config())
        env.reset(seed=714)
        for index in range(100):
            state, reward, done, info = step_first_legal(env)
            self.assertTrue(np.isfinite(state).all())
            self.assertTrue(np.isfinite(reward))
            self.assertEqual(done, index == 99)
            self.assertEqual(info["actual_action"]["slots"], 1)
            self.assertEqual(info["out"], 0)
            self.assertEqual(info["occupy"], 0)
        metrics = env.metrics()
        self.assertTrue(metrics["complete"])
        self.assertFalse(metrics["overload"])
        self.assertEqual(metrics["actual_slots"], 100)
        self.assertAlmostEqual(metrics["power_w"], 500, places=6)
        self.assertEqual(metrics["processed_count"], 100)
        final_rates = np.asarray(env.raw_env.beam_bps)
        expected_si = np.minimum(final_rates / env.scene()["rate"].ravel(), 1).mean()
        self.assertAlmostEqual(metrics["final_si_mean"], expected_si, places=12)


if __name__ == "__main__":
    unittest.main()
