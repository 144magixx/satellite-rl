"""Strict contracts protect ND12 budgets, scientific stop and date separation."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.coverage_nondrl import (default_nondrl_config, validate_nondrl_config,
    partition_contract, resume_job, METHODS, merge_datasets)
from satellite_rl.workbench.storage import Store, digest
from satellite_rl.workbench.nondrl_preparation import bind_parser_source


class NondrlWorkbenchTests(unittest.TestCase):
    def test_cross_lane_merge_keeps_source_task_lineage(self):
        import numpy as np
        from satellite_rl.coverage.scenes import CoverageDataset, CoverageScene
        from satellite_rl.coverage.variable_dataset import save_variable_dataset, load_variable_dataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "partition.json"
            roles = {r: {"dates": [f"2024-01-{i:02d}"], "count": 1}
                     for i, r in enumerate(("train", "development", "validation"), 1)}
            roles["test"] = {"dates": ["2024-02-11", "2024-02-12"], "count": 2}
            atomic_json(manifest_path, {"group": "S1-ND12", "roles": roles})
            refs = []
            for port, day in (("20418", "2024-02-11"), ("20474", "2024-02-12")):
                store = Store(root / port / "archive")
                identifier = "source_" + port
                directory = store.record_dir("scenes", identifier)
                path = directory / "inputs.npz"
                scene = CoverageScene(day, np.array([[45., 13.]]), np.array([1.e6]), np.empty((0, 2)), ["aircraft"],
                                      {"source_dates": [day], "selected_time_utc": day + "T00:00:00Z"}, "rawsha")
                save_variable_dataset(CoverageDataset([scene], {"partition_role": "test", "source_dates": [day]}), path)
                record = {"id": identifier, "path": str(path), "sha256": digest(path), "schema_version": 7,
                          "partition_role": "test", "dates": [day], "count": 1, "source_run_id": "prep_" + port,
                          "partition_manifest_path": str(manifest_path), "partition_manifest_sha256": digest(manifest_path)}
                store.save_record("scenes", record)
                jobdir = store.job_dir(record["source_run_id"])
                atomic_json(jobdir / "job.json", {"experiment_id": "S1-ND12", "preparation_mode": "dataset", "dataset_role": "test", "source_date": day})
                atomic_json(jobdir / "status.json", {"stage": "completed"})
                atomic_json(jobdir / "artifacts.json", {"prepared_dataset": str(path), "prepared_dataset_sha256": digest(path)})
                refs.append({"port": port, "id": identifier})
            application = type("Application", (), {"store": Store(root / "20418" / "archive")})()
            with patch("satellite_rl.workbench.coverage_nondrl.PARENT", root), patch("satellite_rl.workbench.coverage_nondrl.deployment", return_value="20418"):
                result = merge_datasets(application, {"source_datasets": refs})
                self.assertEqual(result["count"], 2)
                self.assertEqual(result["dates"], roles["test"]["dates"])
                merged = load_variable_dataset(result["path"])
                self.assertEqual({r["source_port"] for r in merged.metadata["source_datasets"]}, {"20418", "20474"})
                self.assertTrue(all(r["source_run_id"] for r in merged.metadata["source_datasets"]))

    def test_catalog_methods_match_executors(self):
        from satellite_rl.coverage.nondrl.classical import CLASSICAL_METHODS
        from satellite_rl.coverage.nondrl.learning import LEARNING_METHODS
        for route, methods in CLASSICAL_METHODS.items():
            if route != "BH":
                self.assertEqual(set(METHODS[route]), set(methods))
        for route, methods in LEARNING_METHODS.items():
            self.assertEqual(set(METHODS[route]), set(methods) | {"main"})

    def test_all_routes_budget_and_reject_unknown(self):
        for malformed in ([], {"coverage": []}, {"coverage": {"nondrl": []}}):
            with self.assertRaises(ValueError):
                validate_nondrl_config(malformed)
        for i in range(1, 13):
            route = f"{i:02d}"
            config = default_nondrl_config(route)
            self.assertEqual(validate_nondrl_config(config), config)
            self.assertEqual(config["training"]["total_steps"], 0)
            self.assertEqual(config["training"]["updates"] > 0, i <= 5)
        bad = default_nondrl_config("01")
        bad["coverage"]["nondrl"]["extra_unimplemented"] = True
        with self.assertRaises(ValueError):
            validate_nondrl_config(bad)

    def test_cannot_expand_seed_updates_or_search(self):
        for mutate in (lambda c: c["training"].update(seed=51002),
                       lambda c: c["training"].update(updates=10001),
                       lambda c: c["coverage"]["nondrl"]["limits"].update(online_seconds=3),
                       lambda c: c["coverage"]["nondrl"].update(track="variable_count")):
            bad = default_nondrl_config("01")
            mutate(bad)
            with self.assertRaises(ValueError):
                validate_nondrl_config(bad)

    def test_partition_collision_and_hash_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "partition.json"
            roles = {r: {"dates": [f"2024-01-{i:02d}"], "count": 1}
                     for i, r in enumerate(("train", "development", "validation", "test"), 1)}
            atomic_json(path, {"group": "S1-ND12", "roles": roles})
            sha = digest(path)
            self.assertEqual(partition_contract(path, sha)["roles"], roles)
            roles["test"]["dates"] = roles["train"]["dates"]
            atomic_json(path, {"group": "S1-ND12", "roles": roles})
            with self.assertRaises(ValueError):
                partition_contract(path, digest(path))
            with self.assertRaises(ValueError):
                partition_contract(path, sha)

    def test_scientific_stop_not_resumable(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp)
            directory = store.job_dir("nd12_test")
            directory.mkdir()
            atomic_json(directory / "job.json", {"experiment_id": "S1-ND12", "kind": "evaluate"})
            atomic_json(directory / "status.json", {"stage": "stopped", "scientific_stop": True})
            scheduler = type("Scheduler", (), {"store": store})()
            with self.assertRaises(ValueError):
                resume_job(scheduler, "nd12_test")

    def test_parser_uses_frozen_prod_source_and_never_default_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work"
            (work / "raw").mkdir(parents=True)
            name = "v2024.02.12-planes-readsb-prod-0.tar"
            archive = root / name
            archive.write_bytes(b"isolated short source fixture")
            (work / "raw" / name).hardlink_to(archive)
            reference = type("Reference", (), {"ROOT": work, "SOURCE_INSTANCE": "staging-0"})()
            prep = {"date_utc": "2024-02-12", "archive_name": name}
            bind_parser_source(reference, work, prep, {"source_instance": "planes-readsb-prod-0"}, archive)
            self.assertEqual(reference.SOURCE_INSTANCE, "prod-0")
            with self.assertRaises(ValueError):
                bind_parser_source(reference, work, prep, {"source_instance": "staging-0"}, archive)
            with self.assertRaises(ValueError):
                bind_parser_source(reference, work, prep, {}, archive)
            (work / "raw" / name).unlink()
            (work / "raw" / name).write_bytes(b"wrong original source")
            with self.assertRaises(ValueError):
                bind_parser_source(reference, work, prep, {"source_instance": "prod-0"}, archive)


if __name__ == "__main__":
    unittest.main()
