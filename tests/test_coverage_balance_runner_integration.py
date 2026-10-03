"""A short real CUDA path through the schema-3 policy evaluator."""
import unittest
from pathlib import Path
import tempfile
import subprocess
import sys

import numpy as np
import torch

from satellite_rl.coverage.balance_runner import (_checkpoint, _evaluate_policy,
                                                  _restore)
from satellite_rl.coverage.runner import _write_json
from satellite_rl.coverage.balancing import (BalanceStart, GeometryState, _layout_sha,
                                            _state_record,
                                            balance_metrics, balance_options,
                                            certify_edge_regions)
from satellite_rl.coverage.config import balance_default_config
from satellite_rl.coverage.ppo import CoveragePolicy, Rollout, ppo_update
from satellite_rl.coverage.scenes import CoverageDataset, CoverageScene


@unittest.skipUnless(torch.cuda.is_available(), "requires an isolated CUDA smoke check")
class BalanceRunnerCudaTests(unittest.TestCase):
    def test_tiny_refinement_deadline_ends_with_unmodified_snapshot(self):
        lat, lon = np.meshgrid(np.linspace(40, 60, 10),
                               np.linspace(-5, 20, 10), indexing="ij")
        centers = np.column_stack((lat.ravel(), lon.ravel()))
        scene = CoverageScene("deadline", centers.copy(),
                              np.arange(1, 101, dtype=np.float64) * 1e6,
                              centers.copy(), [f"ac-{i}" for i in range(100)],
                              {"initialization_valid": True}, "deadline-sha")
        dataset = CoverageDataset([scene], {"initialization": "cluster"})
        config = balance_default_config()
        config["coverage"]["balance"].update(route="ppo",
                                               max_refinement_seconds=.001)
        config["coverage"]["method"] = "ppo"
        options = balance_options(config)
        state = GeometryState(scene, centers)
        regions = certify_edge_regions(scene, options)
        metric = balance_metrics(state, regions, options, centers)
        snapshot = _state_record(state, metric)
        start = BalanceStart(scene.scene_id, scene.source_hash, scene.aircraft_ids,
                             state.centers, state.associations, state.loads,
                             regions, metric,
                             {"c1": {"seconds": 1.0,
                                     "initial_layout_sha256": _layout_sha(scene.centers)},
                              "repair": {"seconds": .0005},
                              "assignment": {"seconds": .0005}},
                             "completed", snapshot, snapshot)
        policy = CoveragePolicy(1120, 5, candidate_dim=8).to("cuda")
        rows, _ = _evaluate_policy(config, policy, dataset, [start], role="test")
        row = rows[0]
        self.assertEqual(row["stop_reason_code"], 6)
        self.assertEqual(row["action_trace"], [])
        self.assertEqual(row["metrics"]["decision_steps"], 0)
        np.testing.assert_array_equal(row["final_positions"], centers)
        self.assertEqual(row["balance_diagnostics"]["stage_seconds"]["repair"], .0005)

    def test_real_update_row_early_stop_and_complete_checkpoint_restore(self):
        lat, lon = np.meshgrid(np.linspace(40, 60, 10),
                               np.linspace(-5, 20, 10), indexing="ij")
        centers = np.column_stack((lat.ravel(), lon.ravel()))
        config = balance_default_config()
        config["coverage"]["balance"].update(route="ppo", max_rounds=1,
                                               max_beam_decisions=100,
                                               max_edge_beams=100,
                                               min_subject_beams_in_range=0)
        config["coverage"].update(method="ppo", sweeps=1)
        scenes, starts = [], []
        for index, demands in enumerate((np.ones(100) * 1e6,
                                         np.arange(1, 101, dtype=np.float64) * 1e6)):
            scene = CoverageScene(f"update-{index}", centers.copy(), demands,
                                  centers.copy(), [f"ac-{i}" for i in range(100)],
                                  {"initialization_valid": True}, f"sha-{index}")
            options = balance_options(config)
            state = GeometryState(scene, centers)
            regions = certify_edge_regions(scene, options)
            metric = balance_metrics(state, regions, options, centers)
            snapshot = _state_record(state, metric)
            starts.append(BalanceStart(scene.scene_id, scene.source_hash, scene.aircraft_ids,
                                       state.centers, state.associations, state.loads,
                                       regions, metric, {"c1": {"seconds": 0.1,
                                         "initial_layout_sha256": _layout_sha(scene.centers)}},
                                       "completed", snapshot, snapshot))
            scenes.append(scene)
        from satellite_rl.coverage.balancing_env import BalanceBatchEnv
        dataset = CoverageDataset(scenes, {"initialization": "cluster"})
        env = BalanceBatchEnv(dataset, config["coverage"], starts, num_envs=2,
                              device="cuda", auto_reset=True)
        policy = CoveragePolicy(env.obs_dim, env.action_dim,
                                candidate_dim=env.candidate_dim).to("cuda")
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.0003)
        obs, mask = env.reset(scene_indices=[0, 1])
        candidates = env.candidate_features()
        saved_parameters = [parameter.detach().clone() for parameter in policy.parameters()]
        batches = [[] for _ in range(8)]
        finished_patterns = []
        for _ in range(4):
            with torch.no_grad():
                action, logp, value = policy.act(obs, mask, candidates)
            next_obs, reward, done, info = env.step(action)
            finished_patterns.append(info["finished_rows"].cpu().tolist())
            for bucket, item in zip(batches, (obs, mask, candidates, action,
                                              logp, value, reward * .01, done)):
                bucket.append(item)
            obs = next_obs
            mask = info.get("mask", env.action_mask())
            candidates = env.candidate_features()
        self.assertTrue(finished_patterns[0][0])
        self.assertFalse(finished_patterns[0][1])
        self.assertFalse(any(pair[1] for pair in finished_patterns))
        with torch.no_grad():
            bootstrap = policy.value(obs)
        rollout = Rollout(*(torch.stack(items) for items in batches), bootstrap)
        diagnostics = ppo_update(policy, optimizer, rollout, gamma=1.0,
            gae_lambda=.95, clip_ratio=.2, entropy_coef=.01, value_coef=.5,
            max_grad_norm=.5, epochs=2, minibatch_size=4)
        self.assertGreater(diagnostics["updates"], 0)
        self.assertTrue(any(not torch.equal(before, after) for before, after in
                            zip(saved_parameters, policy.parameters())))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            job = {"git_commit": "fixture", "source_root": str(directory)}
            returns = torch.zeros(2, device="cuda")
            traces = [[{"requested_action": 0}], [{"requested_action": 1}]]
            _checkpoint(directory, job, config, policy, optimizer, env, obs, mask,
                        candidates, step=8, update=diagnostics["updates"], episode=1,
                        returns=returns, traces=traces, elapsed=1.5,
                        last_completed_validation_threshold=0)
            twin = BalanceBatchEnv(dataset, config["coverage"], starts, num_envs=2,
                                   device="cuda", auto_reset=True)
            twin_policy = CoveragePolicy(env.obs_dim, env.action_dim,
                                         candidate_dim=env.candidate_dim).to("cuda")
            twin_optimizer = torch.optim.Adam(twin_policy.parameters(), lr=.0003)
            restored = _restore(directory, job, config, twin_policy, twin_optimizer,
                                twin)
            self.assertEqual(restored["episode_traces"], traces)
            self.assertEqual(restored["last_completed_validation_threshold"], 0)
            self.assertTrue(torch.equal(twin.assignment_attempts, env.assignment_attempts))
            self.assertTrue(torch.equal(twin.centers, env.centers))
            self.assertTrue(torch.equal(twin.assignment, env.assignment))
            for left, right in zip(policy.parameters(), twin_policy.parameters()):
                torch.testing.assert_close(left, right)

    def test_single_scene_finishes_with_full_terminal_and_real_actions(self):
        lat, lon = np.meshgrid(np.linspace(40, 60, 10),
                               np.linspace(-5, 20, 10), indexing="ij")
        centers = np.column_stack((lat.ravel(), lon.ravel()))
        scene = CoverageScene("runner-small", centers.copy(),
                              np.arange(1, 101, dtype=np.float64) * 1e6,
                              centers.copy(), [f"ac-{i}" for i in range(100)],
                              {"initialization_valid": True}, "runner-scene-sha")
        dataset = CoverageDataset([scene], {"initialization": "cluster"})
        config = balance_default_config()
        config["coverage"]["balance"].update(route="ppo", max_rounds=1,
                                               max_beam_decisions=100)
        config["coverage"].update(method="ppo", sweeps=1)
        options = balance_options(config)
        state = GeometryState(scene, centers)
        regions = certify_edge_regions(scene, options)
        metric = balance_metrics(state, regions, options, centers)
        snapshot = _state_record(state, metric)
        start = BalanceStart(scene.scene_id, scene.source_hash, scene.aircraft_ids,
                             state.centers, state.associations, state.loads,
                             regions, metric, {"c1": {"seconds": 1.0,
                                 "initial_layout_sha256": _layout_sha(scene.centers)}},
                             "completed", snapshot, snapshot)
        policy = CoveragePolicy(1120, 5, candidate_dim=8).to("cuda")
        rows, summary = _evaluate_policy(config, policy, dataset, [start], role="test")
        self.assertEqual(summary["scene_count"], 1)
        self.assertEqual(len(rows[0]["action_trace"]), 100)
        self.assertEqual(len(rows[0]["beam_loads_bps"]), 100)
        self.assertGreaterEqual(rows[0]["elapsed_seconds"], 1.0)
        self.assertIsNone(rows[0]["physical_metrics_not_applicable"]["SI"])
        with tempfile.TemporaryDirectory() as temporary:
            result_path = Path(temporary) / "results.json"
            _write_json(result_path, {"scenes": rows, "metadata": {"config": config}})
            audit = subprocess.run([sys.executable,
                str(Path(__file__).resolve().parents[1] / "scripts" / "audit_stage1_balance.py"),
                "--result", f"ppo={result_path}", "--expected-scenes", "1",
                "--output", str(Path(temporary) / "audit")],
                capture_output=True, text=True)
            self.assertEqual(audit.returncode, 0, audit.stdout + audit.stderr)


if __name__ == "__main__":
    unittest.main()
