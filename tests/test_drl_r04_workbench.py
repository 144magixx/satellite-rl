"""Contract checks that protect D12-A budget, role isolation and fair test timing."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from satellite_rl.workbench import coverage_drl_r04 as d12
from satellite_rl.workbench import drl_r04_gate
from satellite_rl.workbench.coverage import CoverageApplication
from satellite_rl.workbench.storage import digest
from satellite_rl.training.run_monitor import atomic_json


class _Store:
    def __init__(self, records):
        self.records = records
        self.created = None

    def get_record(self, kind, identifier):
        assert kind == "scenes"
        return self.records[identifier]

    def create_job(self, config, *, kind, request_id, extra):
        self.created = {"config": config, "kind": kind,
                        "request_id": request_id, "extra": extra}
        return self.created


class DrlR04WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.records = {}
        for role in ("train", "validation", "test"):
            file = Path(self.temp.name) / (role + ".npz")
            file.write_bytes(role.encode())
            self.records[role] = {"id": role, "path": str(file), "sha256": digest(file),
                                  "schema_version": 6,
                                  "partition_role": role,
                                  "record_type": "drl_r04_aircraft_dataset",
                                  "dates": list(d12.ROLE_DATES[role]),
                                  "count": d12.ROLE_COUNTS[role]}
        self.store = _Store(self.records)
        self.application = type("Application", (), {"store": self.store})()

    def test_frozen_config_rejects_budget_and_physics_changes(self):
        valid = d12.default_drl_r04_config()
        self.assertEqual(d12.validate_drl_r04_config(valid)["schema_version"], 6)
        for section, key, value in (("training", "seed", 51002),
                                    ("training", "cpu_workers", 8),
                                    ("coverage", "environment_profile", "china_legacy")):
            altered = deepcopy(valid)
            altered[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                d12.validate_drl_r04_config(altered)

    def test_test_shard_freezes_same_25_mothers_for_every_method(self):
        with patch.object(d12, "deployment", return_value="20418"):
            for variant, method in (("I0", "none"), ("H1", "local_search"),
                                    ("H2", "local_search")):
                config = d12.default_drl_r04_config()
                config["coverage"]["method"] = method
                body = {"kind": "evaluate", "variant": variant, "dataset_role": "test",
                        "dataset_ids": {"test": "test"}, "shard_id": 0,
                        "scene_indices": list(range(0, 100, 4)),
                        "experiment_role": "formal", "request_id": variant}
                job = d12.submit(self.application, body, config)
                self.assertEqual(job["extra"]["scene_indices"], list(range(0, 100, 4)))
                self.assertEqual(job["extra"]["expected_result_rows"], 75)
                self.assertEqual(job["extra"]["shard_source_dataset_sha256"],
                                 self.records["test"]["sha256"])
                body["scene_indices"] = list(range(1, 100, 4))
                with self.assertRaisesRegex(ValueError, "shard"):
                    d12.submit(self.application, body, config)

    def test_api_test_shards_accept_learned_models_with_frozen_source(self):
        root = Path(self.temp.name)
        coverage = CoverageApplication(self.application)
        with patch.object(d12, "PARENT", root), patch.object(d12, "deployment", return_value="20418"):
            for variant in ("BC", "P0", "P1"):
                run_id = f"train_{variant.lower()}"
                source = root / "20424" / "archive" / "outputs" / "workbench" / "jobs" / run_id
                source.mkdir(parents=True)
                endpoint = "bc_endpoint" if variant == "BC" else "final"
                model_root = root / f"model_{variant.lower()}"
                model_dir = model_root / endpoint
                model_dir.mkdir(parents=True)
                policy = model_dir / "policy.pt"
                policy.write_bytes(variant.encode())
                atomic_json(model_dir / "model.json", {
                    "env_step": 0 if variant == "BC" else 262144,
                    "selection": "bc_endpoint" if variant == "BC" else "fixed_budget_endpoint",
                    "policy_sha256": digest(policy)})
                atomic_json(source / "job.json", {"run_id": run_id, "experiment_id": d12.GROUP,
                            "kind": "train", "variant": variant, "experiment_role": "formal",
                            "model_dir": str(model_root)})
                atomic_json(source / "status.json", {"stage": "completed",
                            "env_step": 0 if variant == "BC" else 262144})
                for shard in range(4):
                    config = d12.default_drl_r04_config()
                    body = {"kind": "evaluate", "variant": variant, "dataset_role": "test",
                            "dataset_ids": {"test": "test"}, "shard_id": shard,
                            "scene_indices": list(range(shard, 100, 4)),
                            "experiment_role": "formal", "source_port": "20424",
                            "source_run_id": run_id, "request_id": f"{variant}_{shard}"}
                    job = coverage.api("POST", ["runs"], {}, {"config": config, **body})
                    self.assertEqual(set(job["config"]["datasets"]), {"test"})
                    self.assertEqual(job["extra"]["expected_result_rows"], 75)
                    self.assertEqual(job["extra"]["model_source"]["source_run_id"], run_id)
                    self.assertEqual(job["extra"]["model_source"]["model_files"][0]["sha256"],
                                     digest(policy))
                policy.write_bytes(b"tampered endpoint")
                with self.assertRaisesRegex(ValueError, "metadata/budget"):
                    coverage.api("POST", ["runs"], {}, {"config": config, **body})

    def test_api_training_still_requires_train_dataset(self):
        coverage = CoverageApplication(self.application)
        with patch.object(d12, "deployment", return_value="20418"):
            for variant, datasets in (("BC", {}), ("P0", {"validation": "validation"}),
                                      ("P1", {"validation": "validation"})):
                with self.subTest(variant=variant), self.assertRaisesRegex(ValueError, "datasets"):
                    coverage.api("POST", ["runs"], {}, {
                        "config": d12.default_drl_r04_config(), "kind": "train",
                        "variant": variant, "dataset_ids": datasets,
                        "experiment_role": "formal", "request_id": f"train_{variant}"})

    def test_verification_cannot_expand_formal_training_budget(self):
        body = {"kind": "train", "variant": "P0", "dataset_ids": {
                "train": "train", "validation": "validation"},
                "experiment_role": "verification", "request_id": "verify_p0"}
        config = d12.default_drl_r04_config()
        with patch.object(d12, "deployment", return_value="20418"):
            with self.assertRaisesRegex(ValueError, "budget"):
                d12.submit(self.application, body, config)
            config["training"]["total_steps"] = 2048
            config["training"]["validation_interval"] = 2048
            with patch.object(d12, "_reachability_contract", return_value={
                    "path": "frozen", "sha256": "a" * 64,
                    "validation_dataset_sha256": self.records["validation"]["sha256"],
                    "reachable_mother_indices": [0]}):
                job = d12.submit(self.application, body, config)
            self.assertEqual(job["extra"]["scientific_phase"], "pilot")
            self.assertEqual(job["config"]["training"]["total_steps"], 2048)

    def test_pilot_gate_requires_committed_validation_and_stops_all_stop(self):
        root = Path(self.temp.name)
        run_id = "train_d12a_gate_1"
        directory = root / "outputs" / "workbench" / "jobs" / run_id
        validation = directory / "validation"
        validation.mkdir(parents=True)
        checkpoint = directory / "checkpoint.pt"
        checkpoint.write_bytes(b"fixed-checkpoint")
        checkpoint_sha = digest(checkpoint)
        config = d12.default_drl_r04_config()
        config["datasets"]["validation"] = self.records["validation"]
        atomic_json(directory / "job.json", {"run_id": run_id, "experiment_id": "S1-D12A",
                    "kind": "train", "train_mode": "P0", "experiment_role": "formal",
                    "git_commit": "a" * 40})
        atomic_json(directory / "config.json", config)
        atomic_json(directory / "status.json", {"stage": "pilot_pending",
                    "stop_reason": "pilot_gate", "env_step": 65536,
                    "checkpoint_sha256": checkpoint_sha})
        ledger = []
        for step in (16384, 32768, 49152, 65536):
            path = validation / f"step_{step:06d}.jsonl"
            rows = [{"dataset_index": i // 3,
                     "demand_multiplier": d12.MULTIPLIERS[i % 3],
                     "initial_j": 1.0, "best_j": 1.0,
                     "reward_sum": 0.0,
                     "geometry_calls": 0, "trials": 0, "commits": 1,
                     "matrix_calls": 0, "ray_pair_checks": 0,
                     "best_layout": {"centers": [[0.0, 0.0]], "widths": [.5],
                                     "assignment": [0], "aircraft_ids": ["A"]},
                     "terminal_physics": {"status": "evaluated_all_actual_beams_and_users",
                                          "layout_aliases": {"best": "fixed"},
                                          "layouts_by_sha256": {"fixed": {
                                              "metrics": {"total_demand_bps": 1.,
                                                          "uncovered_demand_bps": 0.,
                                                          "total_power_w": 0.},
                                              "beams": [{"demand_bps": 1.}]}}},
                     "ledger": [{"action": {"operation": "stop"},
                                 "accepted": True, "best_j": 1.0}]}
                    for i in range(60)]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            ledger.append({"env_step": step, "rows": 60, "path": str(path),
                           "sha256": digest(path), "first_step_stop_count": 60,
                           "best_j_improved_count": 0,
                           "three_level_improved_mother_count": 0,
                           "nonstop_best_refresh_rows": 0,
                           "feasible_row_count": 60,
                           "all_60_first_step_stop_zero_progress": True})
        (directory / "evaluations.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in ledger), encoding="utf-8")
        store = type("GateStore", (), {"archive_root": root,
                                     "job_dir": lambda self, value: directory})()
        app = type("GateApp", (), {"store": store})()
        outcome = drl_r04_gate.approve_pilot_gate(app, run_id, {"mode": "audit"})
        self.assertTrue(outcome["audit"]["zero_progress_stop"])
        self.assertEqual(json.loads((directory / "status.json").read_text())["stop_reason"],
                         "consecutive_validation_all_stop_zero_progress")


if __name__ == "__main__":
    unittest.main()
