"""Small deterministic CPU references; formal computation runs on the remote host."""

import hashlib
import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from satellite_rl.coverage import (
    CoverageBatchEnv, CoverageDataset, CoverageScene, load_dataset, run_baseline, save_dataset)
from satellite_rl.coverage.geometry import (
    assign_numpy, ray_angle_numpy, ray_angle_torch, satellite_rays_numpy,
    satellite_rays_torch, squared_ray_limit)
from satellite_rl.workbench.aviation import _cover
from satellite_rl.envs.regions import get_profile
from satellite_rl.coverage.scenes import EMPTY_PARKING_LATLON, positive_position_candidates


def reference_scene():
    lat, lon = np.meshgrid(np.linspace(40, 60, 10), np.linspace(-5, 20, 10), indexing="ij")
    positions = np.column_stack((lat.ravel(), lon.ravel()))
    demands = np.arange(1, 101, dtype=np.float64) * 1e6
    return CoverageScene("small-reference", positions, demands, positions.copy(),
                         [f"ac-{i}" for i in range(100)], {"initialization_valid": True}, "source-hash")


def full_distance_reference(env):
    """Original unblocked [N,U,100,3] float64 geometry and metric arithmetic."""
    users = env._user_rays[env.scene_index]
    rays = satellite_rays_torch(env.centers)
    squared = (users[:, :, None, :] - rays[:, None, :, :]).square().sum(-1)
    nearest_distance, nearest = squared.min(dim=-1)
    valid = env._user_mask[env.scene_index]
    covered = valid & (nearest_distance <= squared_ray_limit(0.25) + 1e-14)
    assignment = torch.where(covered, nearest, -1)
    demand = env._demands[env.scene_index]
    loads = torch.zeros_like(env.loads)
    loads.scatter_add_(1, nearest, torch.where(covered, demand, 0.0))
    total = torch.where(valid, demand, 0.0).sum(1)
    covered_demand = torch.where(covered, demand, 0.0).sum(1)
    uncovered = torch.where(valid & ~covered, demand, 0.0).sum(1)
    coverage_demand = torch.where(total > 0, covered_demand / total, torch.zeros_like(total))
    coverage_users = covered.sum(1) / valid.sum(1).clamp_min(1)
    mean = loads.mean(1)
    cv = torch.where(mean > 0, loads.std(1, unbiased=False) / mean, torch.zeros_like(mean))
    movement = ray_angle_torch(rays, satellite_rays_torch(env.initial_centers))
    score = (env.options["score_coverage_demand_weight"] * coverage_demand +
             env.options["score_coverage_users_weight"] * coverage_users -
             env.options["score_load_cv_weight"] * cv -
             env.options["score_movement_weight"] * movement.mean(1) / env.options["max_movement_deg"])
    return {"distance": squared, "assignment": assignment, "loads": loads,
            "coverage_demand": coverage_demand, "coverage_users": coverage_users,
            "uncovered_demand_bps": uncovered, "score": score}


class CoverageEnvironmentTests(unittest.TestCase):
    def test_empty_start_places_only_original_unique_users_and_preserves_checkpoint(self):
        scene = reference_scene()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "source_observations_00000.jsonl.gz"
            records = [{"aircraft_id": scene.aircraft_ids[i], "lat": float(scene.positions[i, 0]),
                        "lon": float(scene.positions[i, 1]), "demand_bps": float(scene.demands[i])}
                       for i in reversed(range(100))]
            path.write_bytes(gzip.compress("\n".join(json.dumps(row) for row in records).encode()))
            config = {"coverage": {"initialization": "empty", "candidate_capacity": 2048,
                                   "sweeps": 2}}
            data = load_dataset(path, "empty", config)
            candidate_positions = positive_position_candidates(data.scenes[0].positions,
                                                               data.scenes[0].demands)
            self.assertEqual(data.scenes[0].metadata["candidate_count"], 100)
            self.assertEqual(data.scenes[0].metadata["initial_active_count"], 0)
            np.testing.assert_array_equal(data.scenes[0].centers,
                                          np.tile(EMPTY_PARKING_LATLON, (100, 1)))
            prepared = Path(temp) / "prepared" / "empty.npz"
            save_dataset(data, prepared)
            reloaded = load_dataset(prepared, "empty", config)
            self.assertEqual(reloaded.scenes[0].metadata["candidate_sha256"],
                             data.scenes[0].metadata["candidate_sha256"])
            env = CoverageBatchEnv(reloaded, config, num_envs=1, device="cpu", auto_reset=False)
            observation, mask = env.reset(scene_indices=[0])
            self.assertEqual((env.obs_dim, env.action_dim, env.horizon), (964, 2048, 100))
            self.assertEqual(observation.shape, (1, 964))
            self.assertEqual(env.candidate_features().shape, (1, 2048, 4))
            self.assertTrue(mask[0, :100].all())
            self.assertFalse(mask[0, 100:].any())
            self.assertFalse(env.active_masks().any())
            self.assertEqual(int(env.metrics()["active_beams"][0]), 0)
            self.assertEqual(env.metrics()["coverage_demand"][0].item(), 0.0)
            self.assertEqual(env.metrics()["uncovered_demand_bps"][0].item(),
                             float(data.scenes[0].demands.sum()))
            self.assertEqual(env.loads.sum().item(), 0.0)
            self.assertTrue(torch.isinf(env._distance_cache).all())
            self.assertTrue((env.assignment == -1).all())
            self.assertFalse(bool(env.metrics()["movement_applicable"][0]))
            self.assertEqual(env.metrics()["score"][0].item(), 0.0)
            self.assertTrue(torch.equal(env.candidate_features()[0, :100, 3], torch.zeros(100)))
            self.assertTrue(torch.equal(observation[0, :400], torch.zeros(400)))
            self.assertTrue(torch.equal(observation[0, 508:708], torch.zeros(200)))
            self.assertTrue(torch.isclose(observation[0, 708:].sum(), torch.tensor(1.0)))

            _, first_reward, first_done, _ = env.step(torch.tensor([0]))
            self.assertFalse(bool(first_done[0]))
            self.assertAlmostEqual(first_reward[0].item(), env.metrics()["score"][0].item(), places=5)
            self.assertTrue(bool(env.active_masks()[0, 0]))
            self.assertFalse(bool(env.active_masks()[0, 1:].any()))
            self.assertFalse(bool(env.action_mask()[0, 0]))
            np.testing.assert_array_equal(env.layouts()[0, 0].numpy(), candidate_positions[0])
            self.assertTrue(torch.isinf(env._distance_cache[0, :, 1:]).all())
            user_rays = env._user_rays[env.scene_index]
            placed_ray = satellite_rays_torch(env.centers[:, 0])
            full_first_column = (user_rays - placed_ray[:, None, :]).square().sum(-1)
            torch.testing.assert_close(env._distance_cache[:, :, 0], full_first_column, rtol=1e-12, atol=1e-14)
            first_covered = full_first_column[0] <= squared_ray_limit(0.25) + 1e-14
            expected_assignment = torch.where(first_covered, 0, -1)
            self.assertTrue(torch.equal(env.assignment[0], expected_assignment))
            expected_load = torch.where(first_covered, env._demands[0], 0.0).sum()
            self.assertEqual(env.loads[0, 0].item(), expected_load.item())
            saved = env.state_dict()
            self.assertNotIn("_distance_cache", saved)
            self.assertEqual(saved["candidate_hashes"], env._candidate_hashes)
            restored = CoverageBatchEnv(reloaded, config, num_envs=1, device="cpu", auto_reset=False)
            restored.load_state_dict(saved)
            self.assertTrue(torch.equal(restored.action_mask(), env.action_mask()))
            torch.testing.assert_close(restored.candidate_features(), env.candidate_features(), rtol=0, atol=0)
            torch.testing.assert_close(restored._distance_cache, env._distance_cache, rtol=0, atol=0)
            second_action = torch.tensor([1])
            expected = env.step(second_action)
            actual = restored.step(second_action)
            torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
            torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
            altered = dict(saved)
            altered["candidate_hashes"] = ["wrong"]
            with self.assertRaisesRegex(ValueError, "checkpoint configuration/scene mismatch"):
                restored.load_state_dict(altered)
            with self.assertRaisesRegex(ValueError, "no approved"):
                run_baseline(data.scenes[0], config, method="none")

    def test_empty_start_finishes_100_distinct_placements(self):
        scene = reference_scene()
        scene.centers = np.tile(EMPTY_PARKING_LATLON, (100, 1))
        scene.metadata = {"initialization_valid": True, "initial_active_count": 0}
        config = {"coverage": {"initialization": "empty", "candidate_capacity": 2048}}
        env = CoverageBatchEnv(CoverageDataset([scene], {"initialization": "empty"}),
                               config, num_envs=1, device="cpu", auto_reset=False)
        total_reward = 0.0
        for turn in range(100):
            action = torch.tensor([turn])
            _, reward, done, _ = env.step(action)
            total_reward += reward.item()
            self.assertEqual(bool(done[0]), turn == 99)
        self.assertEqual(int(env.metrics()["active_beams"][0]), 100)
        self.assertEqual(int(env.candidate_used[0].sum()), 100)
        self.assertTrue(bool(env.metrics()["feasible"][0]))
        self.assertEqual(env.metrics()["uncovered_demand_bps"][0].item(), 0.0)
        self.assertEqual(env.metrics()["movement_mean_deg"][0].item(), 0.0)
        self.assertAlmostEqual(total_reward, env.metrics()["score"][0].item(), places=3)
        self.assertEqual(len(np.unique(env.layouts()[0].numpy(), axis=0)), 100)

    def test_empty_start_capacity_and_insufficient_unique_candidates_are_explicit(self):
        scene = reference_scene()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "source_observations_00000.jsonl.gz"
            rows = [{"aircraft_id": scene.aircraft_ids[i], "lat": float(scene.positions[i, 0]),
                     "lon": float(scene.positions[i, 1]), "demand_bps": float(scene.demands[i])}
                    for i in range(100)]
            rows.append({"aircraft_id": "extra", "lat": 50.2, "lon": 10.3, "demand_bps": 1e6})
            path.write_bytes(gzip.compress("\n".join(json.dumps(row) for row in rows).encode()))
            with self.assertRaisesRegex(ValueError, "increase candidate_capacity"):
                load_dataset(path, "empty", {"candidate_capacity": 100})
            path.write_bytes(gzip.compress("\n".join(json.dumps(row) for row in rows[:99]).encode()))
            insufficient = load_dataset(path, "empty", {"candidate_capacity": 2048})
            self.assertEqual(insufficient.scenes[0].metadata["candidate_count"], 99)
            self.assertFalse(insufficient.scenes[0].metadata["initialization_valid"])
            with self.assertRaisesRegex(ValueError, "no representable 100-beam layout"):
                CoverageBatchEnv(insufficient, {"coverage": {"initialization": "empty"}},
                                 num_envs=1, device="cpu")

    def test_blocked_distance_cache_matches_full_geometry_actions_and_restore(self):
        first = reference_scene()
        second = reference_scene()
        second.scene_id = "more-aircraft"
        second.positions = np.vstack((second.positions, [50.2, 10.3], [65.0, 25.0], [30.0, -35.0]))
        second.demands = np.r_[second.demands, 1.25e6, 2.5e6, 3.75e6]
        second.aircraft_ids.extend(("extra-covered", "extra-a", "extra-b"))
        dataset = CoverageDataset([first, second])
        config = {"coverage": {"sweeps": 1}}
        env = CoverageBatchEnv(dataset, config, num_envs=3, device="cpu", auto_reset=False)
        env._distance_env_chunk = 2
        env._distance_user_chunk = 17
        env._column_env_chunk = 2
        env.reset(scene_indices=[0, 1, 0])
        initial = full_distance_reference(env)
        torch.testing.assert_close(env._distance_cache, initial["distance"], rtol=1e-12, atol=1e-14)
        torch.testing.assert_close(env.loads, initial["loads"], rtol=1e-12, atol=1e-6)
        self.assertTrue(torch.equal(env.assignment, initial["assignment"]))

        actions_and_results = []
        snapshot = None
        for turn in range(8):
            mask = env.action_mask()
            actions = torch.tensor([int(mask[i].nonzero()[((turn + i) % int(mask[i].sum()))][0])
                                    for i in range(env.num_envs)])
            before = full_distance_reference(env)
            observation, reward, done, _ = env.step(actions)
            after = full_distance_reference(env)
            torch.testing.assert_close(env._distance_cache, after["distance"], rtol=1e-12, atol=1e-14)
            torch.testing.assert_close(env.loads, after["loads"], rtol=1e-12, atol=1e-6)
            self.assertTrue(torch.equal(env.assignment, after["assignment"]))
            for key in ("coverage_demand", "coverage_users", "uncovered_demand_bps", "score"):
                torch.testing.assert_close(env.metrics()[key], after[key], rtol=1e-12, atol=1e-10)
            torch.testing.assert_close(reward, (after["score"] - before["score"]).float(),
                                       rtol=1e-6, atol=1e-6)
            actions_and_results.append((actions, observation.clone(), reward.clone(), done.clone()))
            if turn == 3:
                snapshot = env.state_dict()
                self.assertNotIn("distance_cache", snapshot)
                self.assertNotIn("_distance_cache", snapshot)
        restored = CoverageBatchEnv(dataset, config, num_envs=3, device="cpu", auto_reset=False)
        restored._distance_env_chunk = 2
        restored._distance_user_chunk = 17
        restored._column_env_chunk = 2
        restored.load_state_dict(snapshot)
        torch.testing.assert_close(restored._distance_cache, full_distance_reference(restored)["distance"],
                                   rtol=1e-12, atol=1e-14)
        for actions, observation, reward, done in actions_and_results[4:]:
            actual_observation, actual_reward, actual_done, _ = restored.step(actions)
            torch.testing.assert_close(actual_observation, observation, rtol=0, atol=0)
            torch.testing.assert_close(actual_reward, reward, rtol=0, atol=0)
            self.assertTrue(torch.equal(actual_done, done))
        torch.testing.assert_close(restored.layouts(), env.layouts(), rtol=0, atol=0)
        torch.testing.assert_close(restored._distance_cache, env._distance_cache, rtol=0, atol=0)

    def test_source_rejects_invalid_coordinates_visibility_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "source_observations_00000.jsonl.gz"
            valid = {"aircraft_id": "ABC001", "lat": 50.0, "lon": 13.0, "demand_bps": 1e6}
            bad_rows = [
                ({**valid, "lat": 91.0}, "geographic coordinates"),
                ({**valid, "lon": 51.0}, "outside europe_v1"),
                ({**valid, "lat": 70.0, "lon": -35.0}, "below 10 degree elevation"),
            ]
            for row, reason in bad_rows:
                with self.subTest(reason=reason):
                    path.write_bytes(gzip.compress((json.dumps(row) + "\n").encode()))
                    with self.assertRaisesRegex(ValueError, reason):
                        load_dataset(path, "rule")
            rows = [valid, {**valid, "aircraft_id": "abc001", "lon": 13.1}]
            path.write_bytes(gzip.compress("\n".join(json.dumps(r) for r in rows).encode()))
            with self.assertRaisesRegex(ValueError, "duplicate aircraft ID"):
                load_dataset(path, "rule")

    def test_prepared_rechecks_lengths_business_and_visibility(self):
        scene = reference_scene()
        with tempfile.TemporaryDirectory() as temp:
            prepared = Path(temp) / "valid" / "aircraft.npz"
            save_dataset(CoverageDataset([scene], {"initialization": "rule"}), prepared)
            with np.load(prepared, allow_pickle=False) as original:
                payload = {key: original[key] for key in original.files}
            corruptions = (
                ("fractional_length", "lengths", np.array([99.5]), "integer vector"),
                ("excess_length", "lengths", np.array([101]), "length exceeds padded"),
                ("outside", "positions", np.where(
                    np.arange(100)[None, :, None] == 0,
                    np.array([[[50.0, 51.0]]]), payload["positions"]), "outside europe_v1"),
                ("negative_demand", "demands", np.r_[-1.0, scene.demands[1:]][None],
                 "invalid nonnegative demand"),
                ("duplicate_id", "aircraft_ids_json", json.dumps(
                    [["ac-0", "ac-0", *scene.aircraft_ids[2:]]]), "duplicate aircraft ID"),
            )
            for label, key, value, reason in corruptions:
                with self.subTest(label=label):
                    target = Path(temp) / f"{label}.npz"
                    np.savez_compressed(target, **{**payload, key: value})
                    with self.assertRaisesRegex(ValueError, reason):
                        load_dataset(target, "rule")

    def test_prepared_preserves_failed_initializer_without_fake_centers(self):
        scene = reference_scene()
        scene.positions = scene.positions[:99]
        scene.demands = scene.demands[:99]
        scene.aircraft_ids = scene.aircraft_ids[:99]
        scene.centers = np.full((100, 2), np.nan)
        scene.metadata = {"initialization_valid": False,
                          "initialization_error": "unique_positive_positions_below_100"}
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "failed.npz"
            save_dataset(CoverageDataset([scene], {"initialization": "rule"}), target)
            restored = load_dataset(target, "rule")
            self.assertFalse(restored.scenes[0].metadata["initialization_valid"])
            self.assertTrue(np.isnan(restored.scenes[0].centers).all())

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
        observation, mask_actions = env.reset(scene_indices=[0])
        self.assertEqual(observation.shape, (1, 964))
        self.assertEqual(mask_actions.shape, (1, 9))
        self.assertEqual(env.candidate_features().shape, (1, 9, 4))
        self.assertTrue(env.active_masks().all())
        self.assertTrue(torch.equal(observation[0, 508:708], torch.ones(200)))
        self.assertTrue(torch.isclose(observation[0, 708:].sum(), torch.tensor(1.0)))
        association, loads = assign_numpy(scene.positions, scene.demands, scene.centers)
        actual, mask = env.associations()
        np.testing.assert_array_equal(actual[0][mask[0]].numpy(), association)
        np.testing.assert_allclose(env.loads[0].numpy(), loads, rtol=1e-12, atol=1e-6)
        self.assertEqual(int((association == 0).sum()), 2)
        self.assertAlmostEqual(float(env.loads.sum() + env.metrics()["uncovered_demand_bps"][0]),
                               float(scene.demands.sum()), delta=1e-5)
        self.assertTrue(bool(env.metrics()["feasible"][0]))

    def test_uncovered_demand_uses_direct_aircraft_sum(self):
        scene = reference_scene()
        scene.demands = scene.demands + np.linspace(0.0123, 0.9876, 100)
        env = CoverageBatchEnv(CoverageDataset([scene]), {"coverage": {"sweeps": 1}},
                               num_envs=1, device="cpu", auto_reset=False)
        self.assertEqual(env.metrics()["uncovered_demand_bps"][0].item(), 0.0)
        self.assertEqual(int(env.metrics()["uncovered_users"][0]), 0)
        self.assertTrue(np.isclose(env.loads[0].sum().item(), scene.demands.sum(), rtol=1e-12))

        scene.positions = np.vstack((scene.positions, [65.0, 25.0], [30.0, -35.0]))
        scene.demands = np.r_[scene.demands, 5_123_456.125, 7_765_432.875]
        scene.aircraft_ids.extend(("uncovered-one", "uncovered-two"))
        partial = CoverageBatchEnv(CoverageDataset([scene]), {"coverage": {"sweeps": 1}},
                                   num_envs=1, device="cpu", auto_reset=False)
        assignments, mask = partial.associations()
        unserved = assignments[0][mask[0]].numpy() < 0
        direct_uncovered = scene.demands[unserved].sum()
        self.assertEqual(partial.metrics()["uncovered_demand_bps"][0].item(), direct_uncovered)
        self.assertEqual(int(partial.metrics()["uncovered_users"][0]), int(unserved.sum()))
        self.assertTrue(np.isclose(partial.metrics()["coverage_demand"][0].item(),
                                   scene.demands[~unserved].sum() / scene.demands.sum(), rtol=1e-12))
        self.assertTrue(np.isclose(partial.loads[0].sum().item() + direct_uncovered,
                                   scene.demands.sum(), rtol=1e-12, atol=1e-6))

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
