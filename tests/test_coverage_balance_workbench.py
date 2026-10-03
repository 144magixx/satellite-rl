"""S1-02 workbench configuration and frozen A2 provenance contracts."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from satellite_rl.coverage.config import balance_default_config, default_config, validate_config
from satellite_rl.coverage.balance_runner import (_due_validation_threshold, _partial_rows,
                                                  _sync_validation_records,
                                                  _terminal_row,
                                                  _training_eligibility)
from satellite_rl.coverage.config import config_hash
from satellite_rl.coverage.scenes import CoverageDataset, CoverageScene
from satellite_rl.workbench.coverage import (CoverageApplication,
                                             _same_balance_preparation,
                                             freeze_balance_start)
from satellite_rl.workbench.service import Application
from satellite_rl.workbench.runtime import Scheduler
from satellite_rl.workbench.storage import digest


class BalanceWorkbenchContractTests(unittest.TestCase):
    def test_validation_node_recovery_rebuilds_complete_rows_once(self):
        scene = CoverageScene("val-0", np.array([[45., 5.]]), np.array([1.]),
                              np.zeros((100, 2)), ["ac"], {}, "val-sha")
        dataset = CoverageDataset([scene])
        config = balance_default_config()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            nodes = directory / "validation_nodes"
            nodes.mkdir()
            for target, actual in ((20480, 24576), (40960, 40960)):
                node = {"scheduled_step": target, "actual_env_step": actual,
                        "scene_count": 1, "config_hash": config_hash(config),
                        "rows": [{"scene_index": 0, "scene_id": scene.scene_id,
                                  "scene_sha256": scene.source_hash,
                                  "input": {"demands_bps": [1.]}, "metrics": {}}],
                        "summary": {"coverage_users": 1.0},
                        "completed_at": "2026-10-04T00:00:00+00:00"}
                (nodes / f"{target}.json").write_text(json.dumps(node))
            _sync_validation_records(directory, config, dataset)
            _sync_validation_records(directory, config, dataset)
            rows = [json.loads(line) for line in
                    (directory / "evaluations.jsonl").read_text().splitlines()]
            metrics = [json.loads(line) for line in
                       (directory / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([(r["scheduled_step"], r["step"]) for r in rows],
                             [(20480, 24576), (40960, 40960)])
            self.assertEqual(len(metrics), 2)

    def test_training_eligibility_keeps_original_indices_and_failure_reason(self):
        scenes = [CoverageScene(f"scene-{i}", np.array([[45., 5.]]),
                                np.array([1.]), np.zeros((100, 2)), ["ac"],
                                {}, f"sha-{i}") for i in range(2)]
        dataset = CoverageDataset(scenes)
        starts = [{"status": "initialization_failed", "failure_reason": "C1 failed",
                   "start_sha256": "a" * 64},
                  SimpleNamespace(status="completed", start_sha256="b" * 64)]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first = _training_eligibility(directory, dataset, starts, resume=False)
            self.assertEqual(first["eligible_original_indices"], [1])
            self.assertEqual(first["source_scene_count"], 2)
            self.assertEqual(first["scenes"][0]["failure_reason"], "C1 failed")
            self.assertEqual(_training_eligibility(directory, dataset, starts, resume=True), first)
            (directory / "training_eligibility.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "资格"):
                _training_eligibility(directory, dataset, starts, resume=True)

    def test_fifty_batches_cross_twenty_validation_targets_and_resume_exactly(self):
        training = balance_default_config()["training"]
        step = 0
        last_completed = 0
        nodes = []
        interactions = []
        while step < training["total_steps"]:
            interactions.append(training["rollout_steps"] * training["num_envs"])
            step += interactions[-1]
            due = _due_validation_threshold(step, last_completed,
                                            training["validation_interval"])
            if due is not None:
                nodes.append((due, step))
                last_completed = due
                # A restored checkpoint resumes from the actual interaction
                # step and remembers which scheduled target was already done.
                self.assertIsNone(_due_validation_threshold(
                    step, last_completed, training["validation_interval"]))
            if len(interactions) == 25:
                resumed = json.loads(json.dumps({"step": step,
                    "last_completed_validation_threshold": last_completed}))
                step = resumed["step"]
                last_completed = resumed["last_completed_validation_threshold"]
        expected = [(target, ((target + 8191) // 8192) * 8192)
                    for target in range(20480, 409601, 20480)]
        self.assertEqual(nodes, expected)
        self.assertEqual(len(interactions), 50)
        self.assertEqual(sum(interactions), 409600)
        self.assertEqual(sum(x * training["epochs"] // training["minibatch_size"]
                             for x in interactions), 1600)

    def test_terminal_keeps_raw_and_best_layout_with_full_beam_ledger(self):
        scene = CoverageScene("fixture", np.array([[45., 5.], [46., 5.]]),
                              np.array([10., 20.]), np.zeros((100, 2)),
                              ["a", "b"], {}, "fixture-sha")
        raw = torch.zeros((1, 100, 2))
        best = torch.ones((1, 100, 2))
        info = {"episode_metrics": {"coverage_users": torch.tensor([1.]),
                                    "scene_index": torch.tensor([0]),
                                    "stop_reason_code": torch.tensor([2])},
                "terminal_layouts": raw, "best_layouts": best,
                "terminal_associations": torch.tensor([[0, -1]]),
                "best_associations": torch.tensor([[0, 1]]),
                "terminal_aircraft_mask": torch.tensor([[True, True]]),
                "best_beam_loads_bps": torch.zeros((1, 100))}
        start = SimpleNamespace(stages={"c1": {"seconds": 2.0},
                                        "repair": {"seconds": 3.0}}, regions=[],
                                to_record=lambda: {"start_sha256": "a" * 64})
        row = _terminal_row(scene, 0, info, [{"requested_action": 1,
                                              "actual_action": 1}],
                            step=9, group="test", elapsed=4.0, start=start)
        self.assertEqual(len(row["beam_loads_bps"]), 100)
        self.assertEqual(row["final_positions"][0, 0], 1)
        self.assertEqual(row["raw_terminal_positions"][0, 0], 0)
        self.assertEqual(row["final_associations"].tolist(), [0, 1])
        self.assertEqual(row["elapsed_seconds"], 9.0)
        self.assertIsNone(row["physical_metrics_not_applicable"]["SI"])

    def test_new_switches_are_validated_and_old_schema_is_unchanged(self):
        old = validate_config(default_config())
        self.assertEqual(old["schema_version"], 2)
        self.assertNotIn("balance", old["coverage"])
        config = balance_default_config()
        self.assertEqual(validate_config(config)["coverage"]["balance"]["action_menu"], "X0")
        for bad in (
            lambda c: c["coverage"]["balance"].update(route="X4", action_menu="X0"),
            lambda c: c["coverage"]["balance"].update(post_move_transfers=2),
            lambda c: c["coverage"]["balance"].update(coverage_goal="near_99_9"),
            lambda c: c["coverage"]["balance"].update(max_beam_decisions=1001),
            lambda c: c["coverage"]["balance"].update(unknown_science_switch=True),
        ):
            candidate = deepcopy(config)
            bad(candidate)
            with self.assertRaises(ValueError):
                validate_config(candidate)
        candidate = deepcopy(config)
        candidate["coverage"]["balance"].update(coverage_goal="near_99_9")
        candidate["coverage"]["min_coverage_demand"] = .999
        candidate["coverage"]["min_coverage_users"] = .999
        self.assertEqual(validate_config(candidate)["coverage"]["balance"]["coverage_goal"], "near_99_9")

    def test_a2_cache_reuse_requires_same_scientific_settings(self):
        prepared = balance_default_config()["coverage"]
        prepared["balance"]["route"] = "prepare"
        prepared["method"] = "none"
        selected = deepcopy(prepared)
        selected["balance"].update(route="X4", action_menu="X4")
        selected["method"] = "local_search"
        self.assertTrue(_same_balance_preparation(prepared, selected))
        selected["balance"]["max_assignment_attempts"] -= 1
        self.assertFalse(_same_balance_preparation(prepared, selected))

    def test_start_file_freeze_preserves_sha(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "prepared.jsonl"
            source.write_text('{"scene_id":"fixture"}\n', encoding="utf-8")
            reference = {"path": str(source), "sha256": digest(source),
                         "dataset_sha256": "a" * 64, "source_run_id": "prepare_1"}
            frozen = freeze_balance_start(reference, root / "frozen")
            source.write_text("changed", encoding="utf-8")
            self.assertEqual(digest(frozen["path"]), reference["sha256"])
            with self.assertRaises(ValueError):
                freeze_balance_start(reference, root / "bad")

    def test_stopped_evaluation_resumes_only_from_partial_scene_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            application = Application(temporary)
            run_id = "balance_eval_fixture"
            directory = application.store.job_dir(run_id)
            directory.mkdir()
            (directory / "job.json").write_text(json.dumps({
                "run_id": run_id, "kind": "evaluate", "research_stage": "coverage"}))
            (directory / "config.json").write_text(json.dumps({
                "schema_version": 3, "research_stage": "coverage",
                "coverage": {"balance": {"route": "X0"}}}))
            (directory / "status.json").write_text(json.dumps({
                "run_id": run_id, "stage": "stopped", "pid": 99999999}))
            scheduler = Scheduler(application.store)
            with mock.patch("satellite_rl.workbench.runtime.pid_alive", return_value=False):
                with self.assertRaisesRegex(ValueError, "逐场景"):
                    scheduler.resume(run_id)
                (directory / "results.partial.jsonl").write_text('{"scene_index":0}\n')
                resumed = scheduler.resume(run_id)
            self.assertEqual(resumed["stage"], "queued")
            self.assertTrue(json.loads((directory / "job.json").read_text())["resume"])

    def test_torn_last_scene_is_removed_before_evaluation_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.partial.jsonl"
            self.assertEqual(_partial_rows(path, resume=False), [])
            path.write_text('{"scene_index":0,"scene_id":"a"}\n{"scene_index":1,',
                            encoding="utf-8")
            self.assertEqual([r["scene_id"] for r in _partial_rows(path, resume=True)], ["a"])
            self.assertEqual(path.read_text(encoding="utf-8"),
                             '{"scene_index": 0, "scene_id": "a"}\n')

    @unittest.skipUnless(os.name == "posix", "20418工作台采用Linux解释器路径")
    def test_submission_requires_completed_matching_prepare_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            application = Application(temporary)
            coverage = CoverageApplication(application)
            scene_sha = "b" * 64
            application.store.save_record("scenes", {
                "id": "test_scene", "research_stage": "coverage", "partition_role": "test",
                "dates": ["2024-01-28"], "initialization": "cluster",
                "cluster_iterations": 12, "path": "fixture", "sha256": scene_sha})
            config = balance_default_config()
            config["coverage"]["balance"].update(route="X0", action_menu="X0")
            source_id = "prepare_fixture"
            source_dir = application.store.job_dir(source_id)
            source_dir.mkdir()
            start_path = source_dir / "starts.jsonl"
            start_path.write_text('{"scene_id":"fixture"}\n', encoding="utf-8")
            (source_dir / "job.json").write_text(json.dumps({"research_stage": "coverage"}))
            (source_dir / "status.json").write_text(json.dumps({"stage": "completed"}))
            prepared = balance_default_config()
            prepared["coverage"]["balance"]["route"] = "prepare"
            prepared["coverage"]["method"] = "none"
            prepared["datasets"] = {"test": {"path": "fixture", "sha256": scene_sha}}
            (source_dir / "config.json").write_text(json.dumps(prepared))
            (source_dir / "artifacts.json").write_text(json.dumps({
                "balance_starts": str(start_path), "balance_starts_sha256": digest(start_path)}))
            remote_prefix = str(Path("/home/u2024110224/first_stage_satellite_rl/envs/DQN").resolve())
            patches = [mock.patch("satellite_rl.workbench.coverage.sys.prefix", remote_prefix),
                       mock.patch("satellite_rl.workbench.coverage.getpass.getuser", return_value="u2024110224"),
                       mock.patch("torch.cuda.is_available", return_value=True),
                       mock.patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 4090 D")]
            body = {"config": config, "kind": "evaluate", "dataset_ids": {"test": "test_scene"}}
            with patches[0], patches[1], patches[2], patches[3], mock.patch.object(application.store, "create_job", return_value={"run_id": "created"}) as create:
                with self.assertRaisesRegex(ValueError, "A2"):
                    coverage.submit(body)
                result = coverage.submit({**body, "start_run_ids": {"test": source_id}})
                self.assertEqual(result["run_id"], "created")
                starts = create.call_args.args[0]["balance_starts"]
                self.assertEqual(starts["test"]["source_run_id"], source_id)
                self.assertEqual(starts["test"]["dataset_sha256"], scene_sha)
                start_path.write_text("tampered", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "SHA"):
                    coverage.submit({**body, "start_run_ids": {"test": source_id}})


if __name__ == "__main__":
    unittest.main()
