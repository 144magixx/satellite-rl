"""Small deterministic CPU references; formal computation runs on the remote host."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from satellite_rl.coverage import CoverageBatchEnv, CoverageDataset, CoverageScene, load_dataset, save_dataset
from satellite_rl.coverage.geometry import assign_numpy, ray_angle_numpy, satellite_rays_numpy
from satellite_rl.workbench.aviation import _cover
from satellite_rl.envs.regions import get_profile


def reference_scene():
    lat, lon = np.meshgrid(np.linspace(40, 60, 10), np.linspace(-5, 20, 10), indexing="ij")
    positions = np.column_stack((lat.ravel(), lon.ravel()))
    demands = np.arange(1, 101, dtype=np.float64) * 1e6
    return CoverageScene("small-reference", positions, demands, positions.copy(),
                         [f"ac-{i}" for i in range(100)], {"initialization_valid": True}, "source-hash")


class CoverageEnvironmentTests(unittest.TestCase):
    def test_rule_matches_existing_aviation_medoids(self):
        scene = reference_scene()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "source_observations_00000.jsonl.gz"
            import gzip
            rows = [dict(aircraft_id=scene.aircraft_ids[i], lat=float(p[0]), lon=float(p[1]),
                         demand_bps=float(scene.demands[i]), timestamp_utc="2024-01-08T00:00:00Z",
                         source_date="2024-01-08") for i, p in enumerate(scene.positions)]
            path.write_bytes(gzip.compress("\n".join(json.dumps(r) for r in rows).encode()))
            rule = load_dataset(path, "rule")
            cluster = load_dataset(path, "cluster")
            expected, _, _, _ = _cover(scene.positions, scene.demands, get_profile("europe_v1"))
            np.testing.assert_array_equal(rule.scenes[0].centers, expected)
            self.assertEqual(rule.scenes[0].scene_id, cluster.scenes[0].scene_id)
            self.assertEqual(rule.scenes[0].source_hash, hashlib.sha256(path.read_bytes()).hexdigest())
            prepared = Path(temp) / "prepared.npz"
            record = save_dataset(rule, prepared)
            restored = load_dataset(prepared, "rule")
            self.assertEqual(record["partition_role"], None)
            np.testing.assert_array_equal(restored.scenes[0].positions, scene.positions)
            self.assertEqual(restored.scenes[0].scene_id, rule.scenes[0].scene_id)

    def test_cluster_iterations_follow_config_and_invalidate_saved_centers(self):
        scene = reference_scene()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "source_observations_00000.jsonl.gz"
            import gzip
            rows = [dict(aircraft_id=scene.aircraft_ids[i], lat=float(p[0]), lon=float(p[1]),
                         demand_bps=float(scene.demands[i])) for i, p in enumerate(scene.positions)]
            path.write_bytes(gzip.compress("\n".join(json.dumps(r) for r in rows).encode()))
            one = load_dataset(path, "cluster", {"cluster_iterations": 1})
            self.assertEqual(one.scenes[0].metadata["cluster_iterations"], 1)
            prepared = Path(temp) / "prepared" / "aircraft.npz"
            record = save_dataset(one, prepared)
            self.assertEqual(record["cluster_iterations"], 1)
            three = load_dataset(prepared, "cluster", {"cluster_iterations": 3})
            self.assertEqual(three.scenes[0].metadata["cluster_iterations"], 3)
            self.assertNotEqual(one.metadata["initializer_sha256"], three.metadata["initializer_sha256"])
            self.assertEqual(one.scenes[0].scene_id, three.scenes[0].scene_id)
            with self.assertRaises(ValueError):
                load_dataset(path, "cluster", {"cluster_iterations": 0})

    def test_float64_reference_conservation_and_unique_association(self):
        scene = reference_scene()
        # One overlapping user is assigned once to the nearest stable beam.
        scene.positions = np.concatenate((scene.positions, scene.positions[:1]), axis=0)
        scene.demands = np.concatenate((scene.demands, [7e6]))
        scene.aircraft_ids.append("extra")
        env = CoverageBatchEnv(CoverageDataset([scene]), {"coverage": {"sweeps": 1}},
                               num_envs=1, device="cpu", auto_reset=False)
        association, loads = assign_numpy(scene.positions, scene.demands, scene.centers)
        actual, mask = env.associations()
        np.testing.assert_array_equal(actual[0][mask[0]].numpy(), association)
        np.testing.assert_allclose(env.loads[0].numpy(), loads, rtol=1e-12, atol=1e-6)
        self.assertEqual(int((association == 0).sum()), 2)
        self.assertAlmostEqual(float(env.loads.sum() + env.metrics()["uncovered_demand_bps"][0]),
                               float(scene.demands.sum()), delta=1e-5)
        self.assertTrue(bool(env.metrics()["feasible"][0]))

    def test_changed_center_reassigns_every_user_against_dot_acos_reference(self):
        scene = reference_scene()
        scene.centers[0] = [50.0, 5.0]
        scene.centers[1] = [50.0, 5.08]
        scene.positions = np.vstack((scene.positions, [50.0, 5.035]))
        scene.demands = np.r_[scene.demands, 9e6]
        scene.aircraft_ids.append("between-beams")
        env = CoverageBatchEnv(CoverageDataset([scene]), {"coverage": {"sweeps": 1}},
                               num_envs=1, device="cpu", auto_reset=False)
        before = int(env.associations()[0][0, -1])
        env.centers[0, 0] = torch.tensor([50.0, 5.1], dtype=torch.float64)
        env._recompute()
        after = int(env.associations()[0][0, -1])
        self.assertNotEqual(before, after)
        # This uses the original dot/acos formulation, independent of the
        # environment's squared-chord nearest-neighbor calculation.
        user_rays = satellite_rays_numpy(scene.positions)
        beam_rays = satellite_rays_numpy(env.layouts()[0].numpy())
        dots = np.clip(user_rays @ beam_rays.T, -1.0, 1.0)
        nearest = np.argmax(dots, axis=1)
        degrees = np.rad2deg(np.arccos(dots[np.arange(len(nearest)), nearest]))
        reference = np.where(degrees <= 0.25 + 1e-10, nearest, -1)
        actual, mask = env.associations()
        np.testing.assert_array_equal(actual[0][mask[0]].numpy(), reference)
        expected_loads = np.bincount(reference[reference >= 0],
                                     weights=scene.demands[reference >= 0], minlength=100)
        np.testing.assert_allclose(env.loads[0].numpy(), expected_loads, atol=1e-6)

    def test_action_mask_movement_is_satellite_angle_and_checkpoint_exact(self):
        scene = reference_scene()
        config = {"coverage": {"sweeps": 1, "coordinate_step_deg": 0.1,
                               "max_movement_deg": 0.002}}
        env = CoverageBatchEnv(CoverageDataset([scene]), config, num_envs=2,
                               device="cpu", auto_reset=False, seed=123)
        self.assertTrue(env.action_mask()[:, 0].all())
        self.assertFalse(env.action_mask()[:, 1:].any())
        start = env.state_dict()
        _, reward, done, _ = env.step(torch.tensor([0, 0]))
        self.assertFalse(done.any())
        self.assertTrue(torch.equal(reward, torch.zeros_like(reward)))
        env.load_state_dict(start)
        self.assertTrue(torch.equal(env.step_count, start["step_count"]))
        angle = ray_angle_numpy(satellite_rays_numpy(scene.centers[0]),
                                satellite_rays_numpy(scene.centers[0] + [0.1, 0.0]))
        self.assertGreater(float(angle), 0)
        self.assertNotAlmostEqual(float(angle), 0.1, places=3)

    def test_terminal_is_horizon_not_collection_batch(self):
        scene = reference_scene()
        env = CoverageBatchEnv(CoverageDataset([scene]), {"coverage": {"sweeps": 1}},
                               num_envs=1, device="cpu", auto_reset=False)
        cumulative = 0.0
        initial_score = env.metrics()["score"][0].item()
        for step in range(100):
            _, reward, done, info = env.step(torch.tensor([0]))
            cumulative += reward.item()
            self.assertEqual(bool(done[0]), step == 99)
        self.assertAlmostEqual(cumulative, env.metrics()["score"][0].item() - initial_score, places=6)
        self.assertEqual(info["terminal_layouts"].shape, (1, 100, 2))
        self.assertEqual(info["terminal_associations"].shape[1], 100)
        with self.assertRaises(RuntimeError):
            env.step(torch.tensor([0]))


if __name__ == "__main__":
    unittest.main()
