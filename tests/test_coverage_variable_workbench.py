"""S1-V10 schema and aircraft-level input contract checks."""
from __future__ import annotations

import json
import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np

from satellite_rl.coverage.config import validate_config
from satellite_rl.coverage.scenes import CoverageDataset, CoverageScene
from satellite_rl.coverage.variable_dataset import load_variable_dataset, save_variable_dataset
from satellite_rl.workbench.coverage_variable import (_validate_dataset_lineage,
    _canonical_contract, HISTORICAL_SOURCE_SHA, PLAN_REVISION, ROLE_COUNTS,
    ROLE_DATES, SOURCE_CONTRACT_SHA, SOURCE_PREFLIGHT_SHA,
    submit_variable, variable_default_config)
from satellite_rl.workbench.variable_preparation import _scene_from_rows, run_variable_preparation_job
from satellite_rl.workbench.runtime import Scheduler
from satellite_rl.workbench.storage import Store, digest


class VariableWorkbenchContractTests(unittest.TestCase):
    def test_all_routes_and_training_method_gate(self):
        for index in range(1, 13):
            route = f"R{index:02d}"
            config = variable_default_config()
            config["coverage"]["variable"]["route"] = route
            config["coverage"]["method"] = "ppo" if route in ("R08", "R09") else "local_search"
            validated = validate_config(config)
            self.assertEqual(validated["coverage"]["variable"]["route"], route)
            self.assertEqual(validated["training"]["total_steps"], 262144)
            self.assertEqual(validated["training"]["num_envs"], 32)

    def test_reject_unfrozen_science_or_extra_fields(self):
        for mutate in (
                lambda c: c["coverage"]["variable"].update(max_beams=201),
                lambda c: c["coverage"]["variable"].update(route="R13"),
                lambda c: c["training"].update(seed=51002),
                lambda c: c["evaluation"].update(demand_multipliers=[1.0]),
                lambda c: c["coverage"].update(beam_count=100)):
            config = variable_default_config()
            mutate(config)
            with self.assertRaises(ValueError):
                validate_config(config)

    def test_short_verification_budget_has_explicit_validation_node(self):
        config = variable_default_config()
        config["coverage"]["variable"]["route"] = "R08"
        config["coverage"]["method"] = "ppo"
        config["training"].update(total_steps=2048, validation_interval=2048)
        self.assertEqual(validate_config(config)["training"]["total_steps"], 2048)
        config["training"]["validation_interval"] = 16384
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_aircraft_roundtrip_has_no_legacy_centers_or_multiplier(self):
        scene = CoverageScene("mother", np.array([[45., 13.], [46., 14.]]),
                              np.array([1e8, 2e8]), np.full((100, 2), 0.),
                              ["plane-1", "plane-2"],
                              {"source_dates": ["2024-02-03"]}, "source-hash")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inputs.npz"
            record = save_variable_dataset(CoverageDataset([scene], {
                "partition_role": "validation", "source_dates": ["2024-02-03"],
                "demand_multipliers_applied": [1.]}), path)
            loaded = load_variable_dataset(path)
            self.assertEqual(loaded.scenes[0].centers.shape, (0, 2))
            np.testing.assert_array_equal(loaded.scenes[0].demands, scene.demands)
            self.assertEqual(loaded.scenes[0].aircraft_ids, scene.aircraft_ids)
            self.assertEqual(record["prepared_npz_sha256"],
                             json.loads((path.parent / "record.json").read_text())["prepared_npz_sha256"])

    def test_empty_aircraft_scene_is_retained(self):
        scene = CoverageScene("empty", np.empty((0, 2)), np.empty(0),
                              np.empty((0, 2)), [], {"source_dates": ["2024-02-06"]}, "empty-source")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inputs.npz"
            save_variable_dataset(CoverageDataset([scene], {"source_dates": ["2024-02-06"]}), path)
            loaded = load_variable_dataset(path)
            self.assertEqual(len(loaded.scenes[0].demands), 0)

    def test_preparation_from_frozen_trace_keeps_all_aircraft(self):
        stamp = "2024-02-03T12:00:00.000Z"
        base = datetime(2024, 2, 3, 11, 59, 30, tzinfo=timezone.utc).timestamp()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            archive = source / "v2024.02.03-planes-readsb-staging-0.tar"
            with tarfile.open(archive, "w") as tar:
                for index, lat in enumerate((45., 46.)):
                    payload = json.dumps({"icao": f"abc{index:03d}", "timestamp": base,
                                          "trace": [[0, lat, 13. + index, 30000, 0, 0, 0]]}).encode()
                    item = tarfile.TarInfo(f"./traces/{index}.json")
                    item.size = len(payload)
                    tar.addfile(item, io.BytesIO(payload))
            pin = source / "source_preflight.json"
            contract = source / "selection_contract.json"
            release_api = source / "release_api.json"
            preferred = source / "PREFERRED_RELEASES.txt"
            license_text = source / "LICENSE-ODbL-official.txt"
            release_api.write_text("{}")
            preferred.write_text("official test release")
            license_text.write_text("ODbL")
            reference = Path(__file__).resolve().parents[1] / "scripts" / "stage1_reconstruction" / "prepare_snapshots_reference.py"
            pin.write_text(json.dumps({"group": "S1-V10", "date_utc": "2024-02-03",
                                       "asset_id": 1, "archive_name": archive.name,
                                       "release_api_sha256": hashlib.sha256(release_api.read_bytes()).hexdigest(),
                                       "preferred_sha256": hashlib.sha256(preferred.read_bytes()).hexdigest(),
                                       "license_sha256": hashlib.sha256(license_text.read_bytes()).hexdigest()}))
            contract.write_text(json.dumps({"group": "S1-V10", "role": "validation",
                                            "date_utc": "2024-02-03", "count": 1,
                                            "selected_times_utc": [stamp]}))
            def ref(path):
                return {"path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            directory = root / "job"
            directory.mkdir()
            (directory / "status.json").write_text("{}")
            job = {"run_id": "test_prepare", "archive_root": str(root / "archive"),
                   "source_root": str(Path(__file__).resolve().parents[1]),
                   "variable_preparation": {"role": "validation", "date_utc": "2024-02-03",
                       "selected_times_sha256": hashlib.sha256(json.dumps([stamp],
                           separators=(",", ":")).encode()).hexdigest(),
                       "selected_time_count": 1,
                       "archive": {**ref(archive), "bytes": archive.stat().st_size, "asset_id": 1},
                       "source_preflight": ref(pin), "selection_contract": ref(contract),
                       "preparation_reference": ref(reference),
                       "release_api": ref(release_api),
                       "preferred_releases": ref(preferred), "license": ref(license_text)}}
            run_variable_preparation_job(directory, job, variable_default_config(),
                                         preflight={"cuda": "synthetic-test-only"})
            result = json.loads((directory / "results.json").read_text())
            self.assertEqual(result["summary"]["count"], 1)
            prepared = load_variable_dataset(directory / "preparation" / "inputs.npz")
            self.assertEqual(prepared.scenes[0].aircraft_ids, ["abc000", "abc001"])
            self.assertGreater(float(prepared.scenes[0].demands.sum()), 0)
            first_artifacts = json.loads((directory / "artifacts.json").read_text())
            recovery = directory / "recovery_history" / "attempt_001"
            recovery.mkdir(parents=True)
            (recovery / "status_before_resume.json").write_text('{"stage":"failed"}')
            (directory / "failure.json").write_text('{"error":"interrupted after publish"}')
            job["resume_attempt"] = 1
            run_variable_preparation_job(directory, job, variable_default_config(),
                                         resume=True, preflight={"cuda": "synthetic-test-only"})
            second_artifacts = json.loads((directory / "artifacts.json").read_text())
            self.assertEqual(first_artifacts["prepared_dataset_id"],
                             second_artifacts["prepared_dataset_id"])
            self.assertTrue(json.loads((directory / "status.json").read_text())["resume_reused_complete_dataset"])
            self.assertTrue((recovery / "failure_active_before_retry.json").is_file())
            (directory / "preparation" / "inputs.npz").unlink()
            second_recovery = directory / "recovery_history" / "attempt_002"
            second_recovery.mkdir()
            (second_recovery / "status_before_resume.json").write_text('{"stage":"failed"}')
            (directory / "failure.json").write_text('{"error":"partial file"}')
            job["resume_attempt"] = 2
            run_variable_preparation_job(directory, job, variable_default_config(),
                                         resume=True, preflight={"cuda": "synthetic-test-only"})
            self.assertTrue((second_recovery / "partial_preparation" / "record.json").is_file())
            self.assertEqual(json.loads((directory / "artifacts.json").read_text())[
                "prepared_dataset_id"], first_artifacts["prepared_dataset_id"])

    def test_submission_binds_roles_budget_and_dataset_sha(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = {}
            for role, day in (("train", "2024-01-08"),
                              ("development", "2024-01-12"),
                              ("validation", "2024-02-03")):
                directory = root / role
                directory.mkdir()
                scene = CoverageScene(role, np.array([[45., 13.]]), np.array([1e8]),
                                      np.empty((0, 2)), [role],
                                      {"source_dates": [day]}, role)
                path = directory / "inputs.npz"
                prepared = save_variable_dataset(CoverageDataset([scene],
                    {"partition_role": role, "source_dates": [day]}), path)
                records[role] = {"id": role, "path": str(path),
                    "sha256": prepared["prepared_npz_sha256"], "dates": [day],
                    "partition_role": role, "research_stage": "coverage",
                    "schema_version": 5, "record_type": "variable_aircraft_dataset"}
            class Store:
                def get_record(self, category, identifier):
                    return records[identifier]
                def create_job(self, config, **kwargs):
                    return {"config": config, **kwargs}
            class App:
                store = Store()
            config = variable_default_config()
            config["coverage"]["variable"]["route"] = "R08"
            config["coverage"]["method"] = "ppo"
            config["training"].update(total_steps=2048, validation_interval=2048)
            body = {"kind": "train", "config": config, "experiment_role": "verification",
                    "dataset_ids": {"train": "train", "development": "development"}}
            contract = {"datasets": {role: {"sha256": record["sha256"]}
                                     for role, record in records.items()}}
            with patch("satellite_rl.workbench.coverage_variable._deployment", return_value="20418"), \
                 patch("satellite_rl.workbench.coverage_variable._canonical_contract",
                       return_value=(contract, {"path": "contract.json", "sha256": "a" * 64})), \
                 patch("satellite_rl.workbench.coverage_variable._validate_dataset_lineage"):
                submitted = submit_variable(App(), body, config)
                self.assertEqual(set(submitted["config"]["datasets"]), {"train", "development"})
                self.assertEqual(submitted["extra"]["execution_target"],
                                 "u2024110224@10.160.17.102:20418")
                self.assertEqual(submitted["extra"]["experiment_role"], "verification")
                body["experiment_role"] = "formal"
                with self.assertRaises(ValueError):
                    submit_variable(App(), body, config)

    def test_formal_dataset_lineage_rejects_wrong_mother_dates(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "inputs.npz"
            scene = CoverageScene("old-test", np.array([[45., 13.]]), np.array([1e8]),
                                  np.empty((0, 2)), ["aircraft"],
                                  {"source_dates": ["2024-01-31"]}, "old")
            saved = save_variable_dataset(CoverageDataset([scene], {
                "partition_role": "test", "source_dates": ["2024-01-31"]}), path)
            record = {"path": str(path), "count": 1, "dates": ["2024-01-31"],
                      "sha256": saved["prepared_npz_sha256"]}
            with self.assertRaisesRegex(ValueError, "mother dates/count"):
                _validate_dataset_lineage("test", record, {"datasets": {"test": {}}})

    def test_adapter_cannot_silently_drop_parser_selected_aircraft(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "adapter changed selected parser population"):
                _scene_from_rows(Path(temporary), 0, "2024-02-03T12:00:00.000Z", [
                    ("abc123", "2024-02-03T12:00:00.000Z", 45., 13., 9000.),
                    ("def456", "2024-02-03T11:55:00.000Z", 46., 14., 9000.)],
                    "validation", "synthetic")

    def test_canonical_contract_requires_fixed_dates_and_exact_test_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            source_days = {day: {"selection_contract_sha256": sha,
                                 "source_preflight_sha256": SOURCE_PREFLIGHT_SHA[day],
                                 "archive_sha256": "c" * 64,
                                 "prepared_dataset_sha256": day.replace("-", "") * 8}
                           for day, sha in SOURCE_CONTRACT_SHA.items()}
            datasets = {role: {"sha256": "a" * 64, "count": ROLE_COUNTS[role],
                               "dates": ROLE_DATES[role], "source_datasets": None}
                        for role in ROLE_DATES}
            for role in ("validation", "sequence"):
                datasets[role]["sha256"] = source_days[ROLE_DATES[role][0]]["prepared_dataset_sha256"]
            datasets["test"]["source_datasets"] = [
                {"id": f"source_{day}", "dates": [day],
                 "sha256": source_days[day]["prepared_dataset_sha256"]}
                for day in ROLE_DATES["test"]]
            contract = {"experiment_id": "S1-V10", "plan_revision": PLAN_REVISION,
                        "scientific_commit": "a" * 40,
                        "datasets": datasets,
                        "source_dates": source_days,
                        "historical_source_sha256": {
                            "train": HISTORICAL_SOURCE_SHA["train"],
                            "validation": HISTORICAL_SOURCE_SHA["development"]}}
            path = source / "canonical_dataset_contract.json"
            path.write_text(json.dumps(contract), encoding="utf-8")
            fake_store = type("Store", (), {"source_root": root})()
            with patch("satellite_rl.workbench.coverage_variable.ARCHIVE_PARENT", root), \
                 patch("satellite_rl.workbench.coverage_variable.subprocess.run",
                       return_value=type("Result", (), {"stdout": "a" * 40})()):
                self.assertEqual(_canonical_contract(fake_store)[0]["plan_revision"], PLAN_REVISION)
                contract["datasets"]["test"]["dates"] = ["2024-01-31"]
                path.write_text(json.dumps(contract), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "date/count"):
                    _canonical_contract(fake_store)

    def test_throughput_contract_only_allows_matching_learned_evaluation(self):
        from satellite_rl.workbench.variable_throughput_recovery import (
            DATA_CONTRACT_SHA256, ORIGIN_COMMIT, REPAIR_ID)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            original = {"experiment_id": "S1-V10", "plan_revision": PLAN_REVISION,
                        "scientific_commit": ORIGIN_COMMIT,
                        "datasets": {role: {"sha256": "a" * 64,
                                            "count": ROLE_COUNTS[role],
                                            "dates": ROLE_DATES[role],
                                            "source_datasets": None}
                                     for role in ROLE_DATES},
                        "source_dates": {day: {"selection_contract_sha256": sha,
                                              "source_preflight_sha256": SOURCE_PREFLIGHT_SHA[day],
                                              "archive_sha256": "b" * 64,
                                              "prepared_dataset_sha256": "c" * 64}
                                         for day, sha in SOURCE_CONTRACT_SHA.items()},
                        "historical_source_sha256": {
                            "train": HISTORICAL_SOURCE_SHA["train"],
                            "validation": HISTORICAL_SOURCE_SHA["development"]}}
            for role in ("validation", "sequence"):
                original["datasets"][role]["sha256"] = "c" * 64
            original["datasets"]["test"]["source_datasets"] = [
                {"id": day, "dates": [day], "sha256": "c" * 64}
                for day in ROLE_DATES["test"]]
            primary = source / "canonical_dataset_contract.json"
            primary.write_text(json.dumps(original), encoding="utf-8")
            new_commit = "d" * 40
            continuation = {**original, "scientific_commit": new_commit,
                            "format": "s1v10_throughput_canonical_continuation_v1",
                            "repair_id": REPAIR_ID, "origin_commit": ORIGIN_COMMIT,
                            "repair_commit": new_commit,
                            "origin_contract_sha256": DATA_CONTRACT_SHA256,
                            "allowed_eval_only": True,
                            "allowed_route_port": {"R08": "20418", "R09": "20424"}}
            sidecar = source / f"canonical_dataset_contract_throughput_{new_commit}.json"
            sidecar.write_text(json.dumps(continuation), encoding="utf-8")
            fake_store = type("Store", (), {"source_root": root,
                                             "archive_root": root / "20418"})()
            real_digest = digest
            def pinned_digest(value):
                return DATA_CONTRACT_SHA256 if Path(value) == primary else real_digest(value)
            with patch("satellite_rl.workbench.coverage_variable.ARCHIVE_PARENT", root), \
                 patch("satellite_rl.workbench.coverage_variable.subprocess.run",
                       return_value=type("Result", (), {"stdout": new_commit})()), \
                 patch("satellite_rl.workbench.coverage_variable.digest",
                       side_effect=pinned_digest):
                self.assertEqual(_canonical_contract(fake_store, route="R08",
                                                     kind="evaluate")[0]["scientific_commit"], new_commit)
                with self.assertRaisesRegex(ValueError, "learned evaluation"):
                    _canonical_contract(fake_store, route="R01", kind="evaluate")
                with self.assertRaisesRegex(ValueError, "learned evaluation"):
                    _canonical_contract(fake_store, route="R08", kind="train")

    def test_same_id_preparation_resume_preserves_failed_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(temporary)
            run_id = "prepare_s1v10_fixture"
            directory = store.job_dir(run_id)
            directory.mkdir()
            (directory / "job.json").write_text(json.dumps({
                "run_id": run_id, "kind": "prepare", "research_stage": "coverage",
                "experiment_id": "S1-V10"}))
            (directory / "config.json").write_text(json.dumps({
                "schema_version": 5, "research_stage": "coverage"}))
            (directory / "status.json").write_text(json.dumps({
                "run_id": run_id, "stage": "failed", "pid": 99999999,
                "error": "synthetic interruption"}))
            (directory / "failure.json").write_text('{"error":"synthetic interruption"}')
            scheduler = Scheduler(store)
            with patch("satellite_rl.workbench.runtime.pid_alive", return_value=False):
                resumed = scheduler.resume(run_id)
            self.assertEqual(resumed["stage"], "queued")
            self.assertEqual(json.loads((directory / "job.json").read_text())["resume_attempt"], 1)
            history = directory / "recovery_history" / "attempt_001"
            self.assertEqual(json.loads((history / "status_before_resume.json").read_text())["error"],
                             "synthetic interruption")
            self.assertTrue((history / "failure_before_resume.json").is_file())


if __name__ == "__main__":
    unittest.main()
