"""Strict contracts protect ND12 budgets, scientific stop and date separation."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.coverage_nondrl import (default_nondrl_config, validate_nondrl_config,
    partition_contract, resume_job, METHODS)
from satellite_rl.workbench.storage import Store, digest
from satellite_rl.workbench.nondrl_preparation import bind_parser_source


class NondrlWorkbenchTests(unittest.TestCase):
    def test_catalog_methods_match_executors(self):
        from satellite_rl.coverage.nondrl.classical import CLASSICAL_METHODS
        from satellite_rl.coverage.nondrl.learning import LEARNING_METHODS
        for route, methods in CLASSICAL_METHODS.items():
            if route != "BH":
                self.assertEqual(set(METHODS[route]), set(methods))
        for route, methods in LEARNING_METHODS.items():
            self.assertEqual(set(METHODS[route]), set(methods) | {"main"})

    def test_all_routes_budget_and_reject_unknown(self):
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
