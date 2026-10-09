"""Targeted variable-set learning and same-task recovery checks."""

from __future__ import annotations

import ast
import copy
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import multiprocessing
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from satellite_rl.coverage.scenes import CoverageScene
from satellite_rl.coverage.variable_config import VariableConfig
from satellite_rl.coverage.variable_design import apply_action, initial_layout, metrics
from satellite_rl.coverage.variable_observation import candidate_digest, observation
from satellite_rl.coverage.variable_ppo import (Transition, VariableSetPolicy,
                                                collate, ppo_update,
                                                returns_and_advantages)
from satellite_rl.coverage.variable_runner import (_checkpoint_parallel, _load_parallel,
                                                   _fresh_stream, _train,
                                                   _single_evaluation, _evaluate_rows,
                                                   _check_training,
                                                   _parallel_validation_if_due,
                                                   _cpu_worker_init, _cpu_observe,
                                                   _cpu_execute,
                                                   THROUGHPUT_REPAIR_ID,
                                                   THROUGHPUT_REPAIR_ORIGIN,
                                                   THROUGHPUT_REPAIR_RUNS,
                                                   THROUGHPUT_REPAIR_CRITERIA)
from satellite_rl.workbench.coverage_variable import variable_default_config


class VariableLearningTests(unittest.TestCase):
    def setUp(self):
        points = np.asarray([[45, 10], [45.01, 10.01], [46, 11],
                             [46.01, 11.01], [47, 12], [47.01, 12.01]], dtype=float)
        self.scene = CoverageScene("small", points, np.asarray([80, 40, 90, 30, 70, 60]) * 1e6,
                                   np.empty((0, 2)), [f"u-{i}" for i in range(6)],
                                   source_hash="synthetic")
        self.cfg = VariableConfig(max_beams=6, max_actions=4, max_candidates=16)

    def test_policy_module_has_no_runtime_geometry_import(self):
        source = Path(__file__).resolve().parents[1] / "src" / "satellite_rl" / "coverage" / "variable_ppo.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        self.assertFalse(any(name and ("variable_design" in name or "geometry" in name)
                             for name in imports))

    def test_variable_candidates_and_stop_gate(self):
        for route in ("R08", "R09"):
            with self.subTest(route=route):
                layout, _ = initial_layout(self.scene, self.cfg, route)
                obs, menu = observation(self.scene, layout, self.cfg,
                                        route=route, steps=0, geometry_calls=0)
                self.assertEqual(obs["beam"].shape[0], len(layout.centers))
                self.assertEqual(obs["candidate"].shape[1], 16)
                self.assertEqual(menu[0][0].operation, "stop")
                self.assertEqual(obs["candidate_mask"][0], metrics(
                    self.scene, layout, self.cfg)["coverage_100"])
                self.assertEqual(candidate_digest(menu), candidate_digest(menu))
                batch = collate([obs], "cpu")
                policy = VariableSetPolicy()
                choice, logp, value = policy.act(batch)
                self.assertTrue(bool(batch["candidate_mask"][0, choice]))
                self.assertTrue(torch.isfinite(logp).all())
                self.assertTrue(torch.isfinite(value).all())
                if route == "R09":
                    added, info = apply_action(self.scene, layout, self.cfg, menu[1][0])
                    self.assertTrue(info["accepted"])
                    self.assertGreater(len(added.centers), 0)

    def test_real_ppo_update_changes_weights_and_set_order_is_invariant(self):
        torch.manual_seed(7)
        policy = VariableSetPolicy()
        optimizer = torch.optim.Adam(policy.parameters(), lr=.001)
        layout, _ = initial_layout(self.scene, self.cfg, "R08")
        obs, _ = observation(self.scene, layout, self.cfg,
                             route="R08", steps=0, geometry_calls=0)
        reversed_obs = copy.deepcopy(obs)
        reversed_obs["beam"] = reversed_obs["beam"][::-1].copy()
        with torch.no_grad():
            logits = policy(collate([obs, reversed_obs], "cpu"))[0]
        self.assertTrue(torch.allclose(logits[0], logits[1], atol=1e-6))
        transitions = []
        for index in range(8):
            batch = collate([obs], "cpu")
            with torch.no_grad():
                choice, logp, value = policy.act(batch)
            transitions.append(Transition(obs, int(choice), float(logp), float(value),
                                          float(index % 3 - 1), index == 7))
        before = [item.detach().clone() for item in policy.parameters()]
        result = ppo_update(policy, optimizer, transitions, 0.0, device="cpu",
                            epochs=2, minibatch_size=4)
        self.assertEqual(result["minibatch_updates"], 4)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, policy.parameters())))

    def test_batch_padding_supports_one_to_two_hundred_beams(self):
        small = {"beam": np.ones((1, 12), dtype=np.float32),
                 "global": np.zeros(8, dtype=np.float32),
                 "candidate": np.ones((2, 16), dtype=np.float32),
                 "candidate_mask": np.asarray([True, True])}
        large = {"beam": np.ones((200, 12), dtype=np.float32),
                 "global": np.zeros(8, dtype=np.float32),
                 "candidate": np.ones((17, 16), dtype=np.float32),
                 "candidate_mask": np.asarray([True] * 17)}
        batch = collate([small, large], "cpu")
        logits, values = VariableSetPolicy()(batch)
        self.assertEqual(tuple(logits.shape), (2, 17))
        self.assertEqual(tuple(values.shape), (2,))
        self.assertTrue(bool(batch["beam_mask"][0, 0]))
        self.assertFalse(bool(batch["beam_mask"][0, 1]))
        self.assertFalse(bool(batch["candidate_mask"][0, 2]))
        self.assertTrue(torch.isfinite(logits[:, :2]).all())

    def test_advantages_keep_parallel_streams_independent(self):
        observations = [{"beam": np.zeros((1, 12), dtype=np.float32),
                         "global": np.zeros(8, dtype=np.float32),
                         "candidate": np.zeros((1, 16), dtype=np.float32),
                         "candidate_mask": np.ones(1, dtype=bool)}] * 4
        rows = [Transition(observations[i], 0, 0, 0, reward, done)
                for i, (reward, done) in enumerate(((1, False), (10, True),
                                                     (2, True), (20, True)))]
        returns, advantages = returns_and_advantages(
            rows, np.asarray([100.0, 100.0]), num_envs=2,
            gamma=1.0, gae_lambda=1.0)
        self.assertEqual(advantages.tolist(), [3.0, 10.0, 2.0, 20.0])
        self.assertEqual(returns.tolist(), advantages.tolist())

    def test_parallel_checkpoint_restores_all_streams_and_truncates_uncommitted_log(self):
        config = variable_default_config()
        config["coverage"]["variable"]["route"] = "R09"
        config["coverage"]["method"] = "ppo"
        policy = VariableSetPolicy()
        optimizer = torch.optim.Adam(policy.parameters(), lr=.001)
        from satellite_rl.coverage.scenes import CoverageDataset
        dataset = CoverageDataset([self.scene])
        streams = [_fresh_stream(dataset, self.cfg, "R09", 100 + i,
                                 [.75, 1, 1.25]) for i in range(32)]
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            job = {"run_id": "parallel", "git_commit": "frozen"}
            counts = {name: 0 for name in ("actions.jsonl", "episodes.jsonl",
                                           "updates.jsonl", "evaluations.jsonl")}
            _checkpoint_parallel(directory, job, config, policy, optimizer,
                                 streams=streams, step=2048, updates=128,
                                 episode=100, last_validation=0,
                                 ledger_counts=counts)
            (directory / "actions.jsonl").write_text('{"orphan":true}\n', encoding="utf-8")
            saved = _load_parallel(directory, job, config, policy, optimizer)
            self.assertEqual(saved["step"], 2048)
            self.assertEqual(len(saved["streams"]), 32)
            self.assertEqual((directory / "actions.jsonl").read_text(encoding="utf-8"), "")
            self.assertEqual([x["level"] for x in saved["streams"]],
                             [x["level"] for x in streams])

    def test_only_exact_approved_b311_repair_can_load_old_source_checkpoint(self):
        import hashlib
        from satellite_rl.coverage.scenes import CoverageDataset
        from satellite_rl.coverage.config import config_hash

        config = variable_default_config()
        config["coverage"]["variable"]["route"] = "R08"
        config["coverage"]["method"] = "ppo"
        config["datasets"] = {
            "train": {"path": "isolated-train", "sha256": "a" * 64,
                      "partition_role": "train"},
            "validation": {"path": "isolated-validation", "sha256": "b" * 64,
                           "partition_role": "validation"}}
        dataset = CoverageDataset([self.scene])
        streams = [_fresh_stream(dataset, self.cfg, "R08", 800 + i,
                                 [.75, 1, 1.25]) for i in range(32)]
        policy = VariableSetPolicy()
        optimizer = torch.optim.Adam(policy.parameters(), lr=.001)
        optimizer.zero_grad()
        sum(parameter.sum() for parameter in policy.parameters()).backward()
        optimizer.step()
        run_id = THROUGHPUT_REPAIR_RUNS["R08"]
        repair_commit = "c" * 40
        request_id = "s1v10_train_R08_20418_s51001"
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            old_root, new_root = directory / "old-source", directory / "new-source"
            old_root.mkdir()
            new_root.mkdir()
            (old_root / "source_manifest.json").write_text('{"source":"old"}', encoding="utf-8")
            (new_root / "source_manifest.json").write_text('{"source":"new"}', encoding="utf-8")
            config_path = directory / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            old_job = {"run_id": run_id, "git_commit": THROUGHPUT_REPAIR_ORIGIN}
            counts = {name: 0 for name in ("actions.jsonl", "episodes.jsonl",
                                           "updates.jsonl", "evaluations.jsonl")}
            _checkpoint_parallel(directory, old_job, config, policy, optimizer,
                                 streams=streams, step=2048, updates=128,
                                 episode=3, last_validation=0, ledger_counts=counts)
            for name in counts:
                (directory / name).write_text("", encoding="utf-8")
            checkpoint_sha = hashlib.sha256((directory / "checkpoint.pt").read_bytes()).hexdigest()
            config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
            archive = directory / "recoveries" / f"{THROUGHPUT_REPAIR_ID}_{repair_commit[:12]}"
            (archive / "original").mkdir(parents=True)
            shutil.copyfile(directory / "checkpoint.pt", archive / "original" / "checkpoint.pt")
            shutil.copyfile(config_path, archive / "original" / "config.json")
            for name in ("job.json", "status.json", "launch.json", "worker.log"):
                text = (json.dumps({"run_id": run_id, "request_id": request_id,
                                    "git_commit": THROUGHPUT_REPAIR_ORIGIN,
                                    "source_root": str(old_root)})
                        if name == "job.json" else "{}")
                (archive / "original" / name).write_text(text, encoding="utf-8")
            backups = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (archive / "original").iterdir()}
            evidences = []
            for i in range(5):
                evidence = archive / f"numeric_evidence_{i}.json"
                evidence.write_text('{"passed":true}', encoding="utf-8")
                evidences.append({"path": str(evidence),
                                  "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest()})
            receipt = {"format": "s1v10_throughput_repair_validation_v1",
                       "repair_id": THROUGHPUT_REPAIR_ID,
                       "origin_commit": THROUGHPUT_REPAIR_ORIGIN,
                       "repair_commit": repair_commit, "run_id": run_id,
                       "request_id": request_id, "route": "R08",
                       "actual_checkpoint": {"sha256": checkpoint_sha,
                                             "step": 2048, "updates": 128, "episode": 3},
                       "config_sha256": config_sha, "config_hash": config_hash(config),
                       "dataset_sha256": {"train": "a" * 64, "validation": "b" * 64},
                       "stream_count": 32, "tests_passed": True,
                       "approved": True,
                       "criteria": {key: True for key in THROUGHPUT_REPAIR_CRITERIA},
                       "old_source_root": str(old_root), "new_source_root": str(new_root),
                       "evidence_files": evidences}
            receipt_path = directory / "recoveries" / "validation_receipt.json"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            shutil.copyfile(receipt_path, archive / "validation_receipt.json")
            proof = {"repair_id": THROUGHPUT_REPAIR_ID,
                     "origin_commit": THROUGHPUT_REPAIR_ORIGIN,
                     "repair_commit": repair_commit, "run_id": run_id,
                     "request_id": request_id, "checkpoint_sha256": checkpoint_sha,
                     "checkpoint_step": 2048, "checkpoint_updates": 128,
                     "checkpoint_episode": 3, "route": "R08",
                     "config_sha256": config_sha,
                     "config_hash": config_hash(config),
                     "validation_receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                     "validation_receipt_path": str(receipt_path),
                     "archive_path": str(archive), "old_source_root": str(old_root),
                     "new_source_root": str(new_root),
                     "old_manifest_sha256": hashlib.sha256(
                         (old_root / "source_manifest.json").read_bytes()).hexdigest(),
                     "new_manifest_sha256": hashlib.sha256(
                         (new_root / "source_manifest.json").read_bytes()).hexdigest(),
                     "backup_files_sha256": backups, "evidence_files": evidences,
                     "approved": True}
            (archive / "repair_proof.json").write_text(json.dumps(proof), encoding="utf-8")
            new_job = {"run_id": run_id, "git_commit": repair_commit,
                       "original_git_commit": THROUGHPUT_REPAIR_ORIGIN,
                       "source_root": str(new_root), "source_history": [proof],
                       "experiment_id": "S1-V10", "kind": "train",
                       "experiment_role": "formal", "request_id": request_id,
                       "variable_route": "R08",
                       "execution_target": "u2024110224@10.160.17.102:20418",
                       "archive_root": str(directory),
                       "variable_throughput_repair": proof}
            restored = _load_parallel(directory, new_job, config, policy, optimizer)
            self.assertEqual(restored["step"], 2048)
            self.assertEqual(len(restored["streams"]), 32)
            bad_job = dict(new_job, variable_throughput_repair={
                **proof, "checkpoint_sha256": "0" * 64})
            with self.assertRaisesRegex(ValueError, "repair"):
                _load_parallel(directory, bad_job, config, policy, optimizer)
            bad_job = dict(new_job, run_id="unrelated-training")
            with self.assertRaisesRegex(ValueError, "run_id|provenance|repair"):
                _load_parallel(directory, bad_job, config, policy, optimizer)
            false_receipt = {**receipt, "criteria": {
                **receipt["criteria"], "same_input_action_gpu_update_equal": False}}
            receipt_path.write_text(json.dumps(false_receipt), encoding="utf-8")
            shutil.copyfile(receipt_path, archive / "validation_receipt.json")
            false_proof = {**proof, "validation_receipt_sha256": hashlib.sha256(
                receipt_path.read_bytes()).hexdigest()}
            (archive / "repair_proof.json").write_text(json.dumps(false_proof), encoding="utf-8")
            false_job = {**new_job, "variable_throughput_repair": false_proof,
                         "source_history": [false_proof]}
            with self.assertRaisesRegex(ValueError, "validation receipt"):
                _load_parallel(directory, false_job, config, policy, optimizer)
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            shutil.copyfile(receipt_path, archive / "validation_receipt.json")
            (archive / "repair_proof.json").write_text(json.dumps(proof), encoding="utf-8")
            next_sha = _checkpoint_parallel(directory, new_job, config, policy, optimizer,
                                            streams=restored["streams"], step=4096,
                                            updates=256, episode=3, last_validation=0,
                                            ledger_counts=counts)
            next_payload = torch.load(directory / "checkpoint.pt", map_location="cpu",
                                      weights_only=False)
            self.assertEqual(next_payload["source_commit"], repair_commit)
            self.assertEqual(next_payload["throughput_repair"]["origin_commit"],
                             THROUGHPUT_REPAIR_ORIGIN)
            self.assertEqual(next_sha, hashlib.sha256(
                (directory / "checkpoint.pt").read_bytes()).hexdigest())

    def test_parallel_runner_short_real_update_and_same_id_resume(self):
        from satellite_rl.coverage.scenes import CoverageDataset
        from satellite_rl.coverage import variable_runner

        config = variable_default_config()
        config["coverage"]["variable"]["route"] = "R09"
        config["coverage"]["method"] = "ppo"
        short = dict(config["training"], total_steps=128, rollout_steps=2,
                     checkpoint_interval=64, validation_interval=64,
                     epochs=1, minibatch_size=32, device="cpu")
        dataset = CoverageDataset([self.scene])
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            job = {"run_id": "short-parallel", "git_commit": "frozen",
                   "model_dir": str(directory / "models")}
            (directory / "stop.request").write_text("", encoding="utf-8")
            with patch.object(variable_runner, "_check_training", return_value=short), \
                 patch.object(variable_runner, "_core_config", return_value=self.cfg), \
                 patch.object(variable_runner, "_dataset",
                              side_effect=lambda _config, role: dataset if role == "train"
                              else CoverageDataset([self.scene] * 20)), \
                 patch.object(variable_runner, "config_hash", return_value="short-test"):
                _train(directory, job, config, False, {"gpu": "local_cpu_test"})
                first = torch.load(directory / "checkpoint.pt", map_location="cpu",
                                   weights_only=False)
                self.assertEqual(first["step"], 64)
                self.assertGreater(first["updates"], 0)
                (directory / "stop.request").unlink()
                _train(directory, job, config, True, {"gpu": "local_cpu_test"})
            final = torch.load(directory / "checkpoint.pt", map_location="cpu",
                               weights_only=False)
            self.assertEqual(final["step"], 128)
            action_rows = [json.loads(line) for line in (directory / "actions.jsonl").read_text(
                encoding="utf-8").splitlines()]
            self.assertEqual(len(action_rows), 128)
            self.assertTrue((directory / "models" / "final" / "policy.pt").is_file())
            metadata = json.loads((directory / "models" / "final" / "model.json").read_text(
                encoding="utf-8"))
            self.assertEqual(metadata["validation_downstream_allocation_call_budget"], 960)
            episode_rows = [json.loads(line) for line in (directory / "episodes.jsonl").read_text(
                encoding="utf-8").splitlines()]
            completed = [row for row in episode_rows if row["completed_episode"]]
            self.assertEqual(metadata["training_terminal_downstream_allocation_calls"],
                             len(completed))
            self.assertTrue(completed)
            for row in completed:
                self.assertEqual(row["terminal_physics_status"],
                                 "evaluated_all_actual_beams_and_users")
                self.assertNotIn("actions", row)
                self.assertEqual(row["macro_steps"], row["action_trace"]["count"])
                matches = [action for action in action_rows
                           if action["stream_id"] == row["action_trace"]["stream_id"] and
                           action["stream_episode_serial"] ==
                           row["action_trace"]["stream_episode_serial"]]
                self.assertEqual(len(matches), row["action_trace"]["count"])
                self.assertGreaterEqual(row["geometry_calls"], sum(
                    action["observation_actual_geometry_calls"] +
                    action["execution_actual_geometry_calls"] for action in matches))
                self.assertTrue(all(beam["capacity_bps"] is not None for beam in row["beams"]))
                self.assertTrue(all("capacity_bps" in user for user in row["users"]))
                self.assertGreater(row["geometry_calls"], 0)
                self.assertGreater(row["ray_pair_checks"], 0)
            for row in episode_rows:
                if not row["completed_episode"]:
                    self.assertEqual(row["terminal_physics_status"],
                                     "not_run_incomplete_budget_tail")

    def test_all_routes_share_real_downstream_beam_ledger(self):
        config = variable_default_config()
        for route in [f"R{i:02d}" for i in range(1, 13)]:
            with self.subTest(route=route):
                config["coverage"]["variable"]["route"] = route
                config["coverage"]["method"] = "ppo" if route in ("R08", "R09") else "local_search"
                policy = VariableSetPolicy().eval() if route in ("R08", "R09") else None
                row = _single_evaluation(self.scene, self.cfg, route, policy,
                                         config=config)
                self.assertTrue(row["coverage_complete"])
                self.assertEqual(len(row["beams"]), row["metrics"]["beam_count"])
                self.assertEqual(len(row["users"]), len(self.scene.demands))
                self.assertTrue(all(beam["capacity_bps"] is not None
                                    for beam in row["beams"]))
                self.assertGreaterEqual(row["total_seconds"],
                                        row["downstream"]["timing"]["total_seconds"])
                if route == "R10":
                    self.assertEqual(row["geometry_calls"],
                                     row["initialization_info"]["actual_geometry_calls"])
                    self.assertEqual(row["geometry_budget_charges"],
                                     row["initialization_info"]["geometry_budget_charges"])

    def test_role_budget_gate_distinguishes_verification_and_formal(self):
        config = variable_default_config()
        config["datasets"] = {"train": {}, "development": {}}
        config["training"]["total_steps"] = 2048
        with patch.object(torch.cuda, "is_available", return_value=True):
            self.assertEqual(_check_training(config, {"experiment_role": "verification"})["total_steps"], 2048)
            with self.assertRaisesRegex(ValueError, "decision budget"):
                _check_training(config, {"experiment_role": "formal"})
            config["training"]["total_steps"] = 262144
            with self.assertRaisesRegex(ValueError, "partition"):
                _check_training(config, {"experiment_role": "formal"})

    def test_spawn_cpu_worker_matches_threads_on_same_aircraft_packet(self):
        from satellite_rl.coverage.scenes import CoverageDataset
        from satellite_rl.coverage.variable_dataset import save_variable_dataset

        config = variable_default_config()
        config["coverage"]["variable"]["route"] = "R09"
        config["coverage"]["method"] = "ppo"
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp) / "aircraft" / "inputs.npz"
            save_variable_dataset(CoverageDataset([self.scene]), data)
            import hashlib
            record = {"path": str(data), "sha256": hashlib.sha256(data.read_bytes()).hexdigest()}
            layout, _ = initial_layout(self.scene, self.cfg, "R09")
            packet = {"scene_index": 0, "level": 1, "layout": layout.to_record(),
                      "episode_steps": 0, "geometry_calls": 0}
            _cpu_worker_init(record, config, "R09", [.75, 1, 1.25])
            with ThreadPoolExecutor(max_workers=2) as pool:
                threaded = list(pool.map(_cpu_observe, [packet, packet]))
            with ProcessPoolExecutor(
                    max_workers=2, mp_context=multiprocessing.get_context("spawn"),
                    initializer=_cpu_worker_init,
                    initargs=(record, config, "R09", [.75, 1, 1.25])) as pool:
                spawned = list(pool.map(_cpu_observe, [packet, packet]))
                action = threaded[0][1][1]
                process_step = pool.submit(_cpu_execute,
                                           (packet, action, threaded[0][3])).result()
            for left, right in zip(threaded, spawned):
                for name in left[0]:
                    self.assertTrue(np.array_equal(left[0][name], right[0][name]))
                self.assertEqual(left[1:], right[1:])
            thread_step = _cpu_execute((packet, action, threaded[0][3]))
            self.assertEqual(thread_step[0], process_step[0])
            self.assertEqual({k: v for k, v in thread_step[1].items() if k != "elapsed_s"},
                             {k: v for k, v in process_step[1].items() if k != "elapsed_s"})
            self.assertEqual(thread_step[2:], process_step[2:])

    def test_r12_static_cold_and_sequence_warm_are_distinct(self):
        from dataclasses import replace
        from satellite_rl.coverage.scenes import CoverageDataset
        from satellite_rl.coverage import variable_runner

        config = variable_default_config()
        config["coverage"]["variable"]["route"] = "R12"
        first = replace(self.scene, scene_id="frame-1", source_hash="sha-1",
                        metadata={"sequence_id": "s", "selected_time_utc": "2024-02-06T00:00:00Z"})
        second = replace(self.scene, scene_id="frame-2", source_hash="sha-2",
                         metadata={"sequence_id": "s", "selected_time_utc": "2024-02-06T00:01:00Z"})
        job = {"run_id": "r12-test"}
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with patch.object(variable_runner, "_dataset",
                              return_value=CoverageDataset([first, second])), \
                 patch.object(variable_runner, "_core_config", return_value=self.cfg):
                static, total = _evaluate_rows(directory, job, config, role="test",
                                               route="R12", policy=None,
                                               output=directory / "static.jsonl", resume=False)
                self.assertEqual((len(static), total), (6, 6))
                self.assertTrue(all(row["temporal_mode"] == "cold_init" for row in static))
                sequence, total = _evaluate_rows(directory, job, config, role="sequence",
                                                 route="R12", policy=None,
                                                 output=directory / "sequence.jsonl", resume=False)
                self.assertEqual((len(sequence), total), (6, 6))
                self.assertEqual([row["temporal_mode"] for row in sequence],
                                 ["sequence_first_frame"] * 3 + ["sequence_warm_start"] * 3)
                resumed, total = _evaluate_rows(directory, job, config, role="sequence",
                                                route="R12", policy=None,
                                                output=directory / "sequence.jsonl", resume=True)
                self.assertEqual(len(resumed), total)
                config["coverage"]["variable"]["route"] = "R04"
                independent, _ = _evaluate_rows(directory, job, config, role="sequence",
                                                route="R04", policy=None,
                                                output=directory / "independent.jsonl", resume=False)
                self.assertTrue(all(row["temporal_mode"] == "independent_scene"
                                    for row in independent))

    def test_verification_endpoint_writes_sixty_development_rows(self):
        from satellite_rl.coverage.scenes import CoverageDataset
        from satellite_rl.coverage import variable_runner

        config = variable_default_config()
        config["coverage"]["variable"]["route"] = "R08"
        config["coverage"]["method"] = "ppo"
        config["training"].update(total_steps=2048, validation_interval=2048)
        config["datasets"] = {"train": {}, "development": {}}
        policy = VariableSetPolicy().eval()
        optimizer = torch.optim.Adam(policy.parameters(), lr=.001)
        mothers = CoverageDataset([self.scene] * 20)
        streams = [_fresh_stream(mothers, self.cfg, "R08", 300 + i,
                                 [.75, 1, 1.25]) for i in range(32)]
        job = {"run_id": "verification", "git_commit": "frozen",
               "experiment_role": "verification"}
        counts = {name: 0 for name in ("actions.jsonl", "episodes.jsonl",
                                       "updates.jsonl", "evaluations.jsonl")}
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with patch.object(variable_runner, "_dataset", return_value=mothers), \
                 patch.object(variable_runner, "_core_config", return_value=self.cfg), \
                 patch.object(variable_runner, "config_hash", return_value="verification-test"):
                due, interrupted = _parallel_validation_if_due(
                    directory, job, config, policy, optimizer, streams,
                    resume=False, step=2048, updates=4, episode=32,
                    last_validation=0, ledgers=counts)
            self.assertFalse(interrupted)
            self.assertEqual(due, 2048)
            self.assertEqual(len((directory / "validation_2048.jsonl").read_text(
                encoding="utf-8").splitlines()), 60)
            self.assertEqual(counts["evaluations.jsonl"], 1)
            validation = [json.loads(line) for line in (directory / "validation_2048.jsonl").read_text(
                encoding="utf-8").splitlines()]
            self.assertTrue(all(row["downstream"]["calls"]["layout_allocation_calls"] == 1
                                and all(beam["capacity_bps"] is not None and
                                        beam["carried_bps"] is not None and
                                        (beam["demand_bps"] == 0 or
                                         (beam["unclipped_capacity_demand_ratio"] is not None and
                                          beam["si"] is not None and beam["sgm"] is not None))
                                        for beam in row["beams"]) for row in validation))
            ledger = [json.loads(line) for line in (directory / "evaluations.jsonl").read_text(
                encoding="utf-8").splitlines()]
            self.assertEqual(ledger[0]["downstream_allocation_calls"], 60)

if __name__ == "__main__":
    unittest.main()
