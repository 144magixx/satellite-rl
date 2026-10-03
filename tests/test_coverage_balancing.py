"""Deterministic checks for S1-02 geometry, archived starts and CUDA-shaped env."""

import unittest
from dataclasses import replace

import numpy as np
import torch

from satellite_rl.coverage.balancing import (
    ACTION_COUNTS, BalanceStart, GeometryState, _state_record, balance_metrics,
    balance_options, certify_edge_regions, post_move_assignment, _layout_sha,
    prepare_common_a2, run_balance_route, action_proposals,
)
from satellite_rl.coverage.balancing_env import BalanceBatchEnv
from satellite_rl.coverage.config import balance_default_config
from satellite_rl.coverage.scenes import CoverageDataset, CoverageScene


def scene_100():
    lat, lon = np.meshgrid(np.linspace(40, 60, 10), np.linspace(-5, 20, 10), indexing="ij")
    points = np.column_stack((lat.ravel(), lon.ravel()))
    return CoverageScene("balance-small", points, np.arange(1, 101, dtype=np.float64) * 1e6,
                         points.copy(), [f"ac-{i:03d}" for i in range(100)],
                         {"initialization_valid": True}, "frozen-source-sha")


def frozen_start(scene, config):
    options = balance_options(config)
    state = GeometryState(scene, scene.centers)
    regions = certify_edge_regions(scene, options)
    metric = balance_metrics(state, regions, options, scene.centers)
    snapshot = _state_record(state, metric)
    return BalanceStart(scene.scene_id, scene.source_hash, scene.aircraft_ids,
                        state.centers, state.associations, state.loads, regions, metric,
                        {"c1": {"seconds": 1.0,
                                 "initial_layout_sha256": _layout_sha(scene.centers),
                                 "counts": {"candidate_evaluations": 1800}},
                         "edge_preprocess": {"seconds": 0.2, "counts": {}},
                         "repair": {"seconds": 0.1, "counts": {}},
                         "assignment": {"seconds": 0.1, "counts": {}}},
                        "completed", snapshot, snapshot)


class BalanceCoreTests(unittest.TestCase):
    def setUp(self):
        self.scene = scene_100()
        self.config = balance_default_config()
        self.start = frozen_start(self.scene, self.config)

    def test_start_sha_and_demand_ledger_detect_tampering(self):
        record = self.start.to_record()
        checked = BalanceStart.from_record(self.scene, record)
        np.testing.assert_array_equal(checked.associations, self.start.associations)
        altered = dict(record)
        altered["beam_loads_bps"] = list(record["beam_loads_bps"])
        altered["beam_loads_bps"][0] += 1
        with self.assertRaises(ValueError):
            BalanceStart.from_record(self.scene, altered)

    def test_certification_is_conservative_and_stable(self):
        first = certify_edge_regions(self.scene, balance_options(self.config))
        second = certify_edge_regions(self.scene, balance_options(self.config))
        self.assertEqual(first, second)
        self.assertTrue(all(region["demand_bps"] < self.scene.demands.sum() / 200
                            for region in first))
        self.assertTrue(all(region["witness_aircraft_id"] in region["aircraft_ids"]
                            for region in first))

    def test_post_move_preserves_whole_user_demand(self):
        state = GeometryState(self.scene, self.scene.centers)
        after, info = post_move_assignment(state, state.copy(), [])
        self.assertEqual(after.associations.shape, (100,))
        self.assertAlmostEqual(after.loads.sum(), self.scene.demands.sum())
        self.assertLessEqual(info["transfers"], 1)

    def test_all_menus_batched_env_shapes_step_and_resume(self):
        dataset = CoverageDataset([self.scene], {"initialization": "cluster"})
        for menu, action_count in ACTION_COUNTS.items():
            with self.subTest(menu=menu):
                config = balance_default_config()
                config["coverage"]["balance"]["action_menu"] = menu
                env = BalanceBatchEnv(dataset, config["coverage"], [self.start.to_record()],
                                      num_envs=2, device="cpu", auto_reset=False)
                obs, mask = env.reset(scene_indices=[0, 0])
                self.assertEqual(obs.shape, (2, 1120))
                self.assertEqual(mask.shape, (2, action_count))
                self.assertEqual(env.candidate_features().shape, (2, action_count, 8))
                self.assertTrue(bool(mask[:, 0].all()))
                proposed = action_proposals(GeometryState(self.scene, self.start.centers,
                                                          self.start.associations),
                                            0, menu, balance_options(config))
                menu_primary, menu_secondary, _, exists, _ = env._build_menu()
                self.assertEqual(len(proposed), action_count)
                for action in proposed:
                    index = action["action_id"]
                    self.assertEqual(bool(exists[0, index]), action.get("exists", True))
                    if not action.get("exists", True):
                        continue
                    np.testing.assert_allclose(
                        menu_primary[0, index].numpy(),
                        action["changes"].get(0, self.start.centers[0]),
                        atol=1e-6, rtol=0)
                    for other, center in action["changes"].items():
                        if other != 0:
                            np.testing.assert_allclose(menu_secondary[0, index].numpy(),
                                                       center, atol=1e-6, rtol=0)
                _, reward, done, info = env.step(torch.zeros(2, dtype=torch.long))
                self.assertEqual(reward.shape, (2,))
                self.assertEqual(done.shape, (2,))
                self.assertIn("finished_rows", info)
                self.assertIn("best_associations", info)
                self.assertIn("best_metrics", info)
                self.assertTrue(torch.allclose(
                    info["best_beam_loads_bps"].sum(-1),
                    info["best_metrics"]["coverage_demand"] * float(self.scene.demands.sum()),
                    atol=1e-5, rtol=1e-12))
                self.assertTrue(torch.allclose(env.loads.sum(-1),
                                               torch.full((2,), float(self.scene.demands.sum()),
                                                          dtype=torch.float64)))
                saved = env.state_dict()
                twin = BalanceBatchEnv(dataset, config["coverage"], [self.start.to_record()],
                                       num_envs=2, device="cpu", auto_reset=False)
                twin.load_state_dict(saved)
                self.assertTrue(torch.equal(twin.assignment, env.assignment))
                self.assertTrue(torch.allclose(twin.loads, env.loads))
                self.assertTrue(torch.allclose(twin.best_metrics["score"],
                                                env.best_metrics["score"]))

    def test_frozen_assignment_attempt_limit_stops_without_extra_work(self):
        dataset = CoverageDataset([self.scene], {"initialization": "cluster"})
        self.start.stages["assignment"]["counts"]["attempts"] = 1
        config = balance_default_config()
        config["coverage"]["balance"]["max_assignment_attempts"] = 1
        env = BalanceBatchEnv(dataset, config["coverage"], [self.start.to_record()],
                              num_envs=1, device="cpu", auto_reset=False)
        _, mask = env.reset(scene_indices=[0])
        self.assertTrue(bool(mask[0, 0]))
        self.assertFalse(bool(mask[0, 1:].any()))
        _, reward, done, info = env.step(torch.tensor([0]))
        self.assertTrue(bool(done[0]))
        self.assertEqual(int(info["stop_reason_code"][0]), 5)
        self.assertEqual(int(info["post_move_attempts"][0]), 0)
        self.assertEqual(int(env.assignment_attempts[0]), 1)
        self.assertEqual(float(reward[0]), 0.0)

    def test_time_limit_snapshot_preserves_raw_and_best_without_action(self):
        dataset = CoverageDataset([self.scene], {"initialization": "cluster"})
        env = BalanceBatchEnv(dataset, self.config["coverage"],
                              [self.start.to_record()], num_envs=1,
                              device="cpu", auto_reset=False)
        env.reset(scene_indices=[0])
        before = env.state_dict()
        info = env.terminal_snapshot("time_limit")
        self.assertTrue(bool(info["finished_rows"][0]))
        self.assertEqual(int(info["stop_reason_code"][0]), 6)
        self.assertEqual(int(info["requested_action"][0]), -1)
        self.assertEqual(int(env.step_count[0]), int(before["step_count"][0]))
        self.assertEqual(int(env.assignment_attempts[0]),
                         int(before["assignment_attempts"][0]))
        self.assertTrue(torch.equal(info["best_associations"],
                                    info["terminal_associations"]))
        self.assertFalse(bool(env.terminal_snapshot("time_limit")["finished_rows"][0]))

    def test_explicit_eligible_subset_keeps_original_scene_indices(self):
        failed_scene = replace(self.scene, scene_id="failed-before-eligible")
        dataset = CoverageDataset([failed_scene, self.scene],
                                  {"initialization": "cluster"})
        failed_start = {"status": "initialization_failed",
                        "scene_id": failed_scene.scene_id}
        env = BalanceBatchEnv(dataset, self.config["coverage"],
                              [failed_start, self.start.to_record()],
                              eligible_indices=[1], num_envs=1,
                              device="cpu", auto_reset=False)
        env.reset(scene_indices=[0])  # Internal index 0 represents original index 1.
        _, _, _, info = env.step(torch.tensor([0]))
        self.assertEqual(int(info["episode_metrics"]["scene_index"][0]), 1)
        self.assertEqual(env.state_dict()["eligible_indices"], [1])
        with self.assertRaises(ValueError):
            BalanceBatchEnv(dataset, self.config["coverage"],
                            [failed_start, self.start.to_record()],
                            num_envs=1, device="cpu")

    def test_failed_initializer_is_archived_without_fake_layout(self):
        failed = replace(self.scene, centers=np.full((100, 2), np.nan),
                         metadata={"initialization_valid": False,
                                   "initialization_error": "test_failure"})
        record = prepare_common_a2(failed, self.config, device="cpu")
        self.assertEqual(record["status"], "initialization_failed")
        self.assertGreaterEqual(record["stages"]["c1"]["seconds"], 0.0)
        row = run_balance_route(failed, self.config, "A0", common=record)
        self.assertIsNone(row["final_positions"])
        self.assertIsNone(row["metrics"])
        self.assertEqual(row["elapsed_seconds"],
                         record["stages"]["c1"]["seconds"])


if __name__ == "__main__":
    unittest.main()
