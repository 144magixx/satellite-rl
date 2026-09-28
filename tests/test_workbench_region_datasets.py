"""Actual scene freezing, region guards and explicit evaluation partitions."""

from copy import deepcopy
import csv
import json
from pathlib import Path
import random
import shutil
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from satellite_rl.envs.regions import profile_metadata
from satellite_rl.workbench.config import default_config
from satellite_rl.workbench.datasets import csv_scene, generate_scenes
from satellite_rl.workbench.environment import validate_scene
from satellite_rl.workbench.legacy import legacy_config
from satellite_rl.workbench.runner import _evaluate_job, _fixed_evaluate, _training
from satellite_rl.workbench.service import Application
from satellite_rl.workbench.storage import Store, digest, read_json


class RegionDatasetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.app = Application(self.root, source_root=self.root)
        self.snapshot = self.root / "frozen"
        self.snapshot.mkdir()
        self.freeze = patch.object(Store, "freeze_source", return_value=(self.snapshot, "f" * 40))
        self.freeze.start()
        self.addCleanup(self.freeze.stop)
        self.config = default_config()
        self.config["environment"]["profile"] = "europe_v1"
        self.config["evaluation"].update(validation_count=1, test_count=1, repeats=1)

    def scene(self, seed=19, profile="europe_v1"):
        return self.app.api("POST", ["scenes"], {}, {"seed": seed, "count": 1, "profile": profile})

    def submit(self, train, validation, test, **options):
        return self.app.api("POST", ["runs"], {}, {"config": self.config,
            "scene_id": train["id"], "validation_scene_id": validation["id"],
            "test_scene_id": test["id"], **options})

    def test_europe_generated_and_csv_arrays_bind_full_profile(self):
        record = self.scene()
        self.assertEqual(record["profile"], "europe_v1")
        self.assertEqual(record["environment_profile"], profile_metadata("europe_v1"))
        with np.load(record["files"][0]["path"], allow_pickle=False) as arrays:
            scene = {key: arrays[key].copy() for key in arrays.files}
        validate_scene(scene, profile="europe_v1")
        with self.assertRaises(ValueError):
            validate_scene(scene, profile="china_legacy")
        path = self.root / "europe.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["beam_id", "lat", "lon", "demand_mbps"])
            for index in range(100):
                writer.writerow([(index + 17) % 100, 45 + index / 100, 13, 70 + index])
        imported = self.app.api("POST", ["scenes", "import"], {}, {"path": str(path), "profile": "europe_v1"})
        self.assertEqual(imported["profile"], "europe_v1")
        scene = csv_scene(path, profile="europe_v1")
        self.assertEqual(scene["original_indices"][0], 17)
        self.assertEqual(scene["rate"][1, 0], 71e6)
        validate_scene(scene, profile="europe_v1")

    def test_training_freezes_three_actual_sets_before_publication(self):
        train, validation, test = (self.scene(seed) for seed in (19, 20, 21))
        job = self.submit(train, validation, test, request_id="europe_freeze")
        self.assertEqual(job["environment_profile"], profile_metadata("europe_v1"))
        for role, original in (("train", train), ("validation", validation), ("test", test)):
            frozen = job["scene_sets"][role]
            self.assertEqual(frozen["id"], original["id"])
            self.assertEqual(digest(frozen["files"][0]["path"]), original["files"][0]["sha256"])
            self.assertTrue(Path(frozen["path"]).is_relative_to(self.app.store.job_dir(job["run_id"])))
            self.assertEqual(read_json(Path(frozen["path"]) / "record.json"), frozen)
        self.assertEqual(job["training_scene_path"], job["scene_sets"]["train"]["path"])
        self.assertEqual(job["run_id"], self.submit(train, validation, test, request_id="europe_freeze")["run_id"])
        # Source changes cannot change an already frozen task.
        Path(train["files"][0]["path"]).write_bytes(b"changed")
        self.assertEqual(digest(job["scene_sets"]["train"]["files"][0]["path"]), train["files"][0]["sha256"])

    def test_custom_europe_train_requires_counts_and_independent_partitions(self):
        train, validation, test = (self.scene(seed) for seed in (19, 20, 21))
        with self.assertRaisesRegex(ValueError, "显式"):
            self.app.api("POST", ["runs"], {}, {"config": self.config, "scene_id": train["id"]})
        with self.assertRaisesRegex(ValueError, "复用"):
            self.submit(train, train, test)
        self.config["evaluation"]["validation_count"] = 2
        with self.assertRaisesRegex(ValueError, "场景数"):
            self.submit(train, validation, test)
        self.config["evaluation"]["validation_count"] = 1
        for role, record in (("train", train), ("validation", validation), ("test", test)):
            record["source"].update(partition_role=role, source_dates=["2026-08-03"])
            self.app.store.save_record("scenes", record)
        with self.assertRaisesRegex(ValueError, "日期"):
            self.submit(train, validation, test)
        self.assertEqual(self.app.store.jobs(), [])

    def test_missing_profile_is_legacy_china_and_cross_region_rejected(self):
        china = self.scene(profile="china_legacy")
        for key in ("profile", "environment_version", "environment_profile"):
            china.pop(key)
        self.app.store.save_record("scenes", china)
        with self.assertRaisesRegex(ValueError, "profile"):
            self.app.api("POST", ["runs"], {}, {"config": self.config, "scene_id": china["id"],
                "validation_scene_id": china["id"], "test_scene_id": china["id"]})
        old = default_config()
        old["environment"].pop("profile", None)
        job = self.app.api("POST", ["runs"], {}, {"config": old, "scene_id": china["id"]})
        self.assertEqual(job["profile"], "china_legacy")
        self.assertIn("train", job["scene_sets"])

    def test_europe_metadata_and_hashes_cannot_be_filled_or_changed_silently(self):
        train, validation, test = (self.scene(seed) for seed in (19, 20, 21))
        complete = deepcopy(train)
        train.pop("environment_version")
        self.app.store.save_record("scenes", train)
        with self.assertRaisesRegex(ValueError, "缺少完整"):
            self.submit(train, validation, test)
        train = deepcopy(complete)
        train["environment_profile"]["satellite_longitude_deg"] = 99
        self.app.store.save_record("scenes", train)
        with self.assertRaisesRegex(ValueError, "参数"):
            self.submit(train, validation, test)
        self.app.store.save_record("scenes", complete)
        Path(complete["files"][0]["path"]).write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "校验"):
            self.submit(complete, validation, test)
        self.assertEqual(self.app.store.jobs(), [])

    def test_evaluation_rejects_model_region_before_loading_weights(self):
        china = self.scene(profile="china_legacy")
        model = {"id": "europe_model", "name": "europe", "path": str(self.root / "policy"), "config": self.config}
        Path(model["path"]).mkdir()
        (Path(model["path"]) / "policy.pt").write_bytes(b"queued weights")
        (Path(model["path"]) / "model.json").write_text("{}")
        with patch.object(self.app, "get_model", return_value=model), \
             patch("satellite_rl.workbench.checkpoints.load_policy") as loader:
            with self.assertRaisesRegex(ValueError, "profile"):
                self.app.start_evaluation({"model_id": model["id"], "scene_id": china["id"]})
            loader.assert_not_called()
            test = self.scene(seed=22)
            job = self.app.start_evaluation({"model_id": model["id"], "scene_id": test["id"]})
        self.assertEqual(job["scene_path"], job["scene_sets"]["test"]["path"])
        self.assertEqual(job["profile"], "europe_v1")
        frozen_hash = digest(Path(job["model_path"]) / "policy.pt")
        (Path(model["path"]) / "policy.pt").write_bytes(b"updated later")
        self.assertEqual(digest(Path(job["model_path"]) / "policy.pt"), frozen_hash)
        # A task-local frozen weight is also checked before any worker load.
        (Path(job["model_path"]) / "policy.pt").write_bytes(b"tampered task copy")
        with patch("satellite_rl.workbench.runner.load_policy") as worker_loader:
            with self.assertRaisesRegex(ValueError, "冻结评价模型"):
                _evaluate_job(self.app.store.job_dir(job["run_id"]), job, self.config, Mock(), device="cpu")
            worker_loader.assert_not_called()

    def test_model_changes_during_submission_never_publish_an_evaluation(self):
        test = self.scene(57)
        directory = self.root / "moving_policy"
        directory.mkdir()
        (directory / "policy.pt").write_bytes(b"first weights")
        (directory / "model.json").write_text("{}")
        model = {"id": "moving_model", "name": "moving", "path": str(directory), "config": self.config}
        def concurrent_export(*args, **kwargs):
            (directory / "policy.pt").write_bytes(b"changed weights")
            return SimpleNamespace(metadata={})
        with patch.object(self.app, "get_model", return_value=model), \
             patch("satellite_rl.workbench.checkpoints.load_policy", side_effect=concurrent_export):
            with self.assertRaisesRegex(ValueError, "提交期间"):
                self.app.start_evaluation({"model_id": model["id"], "scene_id": test["id"]})
        self.assertEqual(self.app.store.jobs(), [])

    def test_evaluation_checks_loaded_policy_provenance_when_catalog_is_stale(self):
        test = self.scene(63)
        test["source"].update(source_dates=["2026-09-20"], partition_role="test")
        self.app.store.save_record("scenes", test)
        directory = self.root / "latest_policy"
        directory.mkdir()
        (directory / "policy.pt").write_bytes(b"weights")
        (directory / "model.json").write_text("{}")
        model = {"id": "latest_model", "name": "latest", "path": str(directory), "config": self.config}
        loaded = SimpleNamespace(metadata={"provenance": {"scene_sets": {
            "train": {"source": {"source_dates": ["2026-09-20"]}}}}})
        with patch.object(self.app, "get_model", return_value=model), \
             patch("satellite_rl.workbench.checkpoints.load_policy", return_value=loaded):
            with self.assertRaisesRegex(ValueError, "日期"):
                self.app.start_evaluation({"model_id": model["id"], "scene_id": test["id"]})
        self.assertEqual(self.app.store.jobs(), [])

    def test_imported_policy_retains_dataset_provenance(self):
        train = self.scene(58)
        source = self.root / "import_policy"
        source.mkdir()
        (source / "policy.pt").write_bytes(b"weights")
        (source / "model.json").write_text("{}")
        provenance = {"scene_sets": {"train": train}}
        policy = SimpleNamespace(config=self.config, metadata={"variant": {}, "provenance": provenance})
        with patch("satellite_rl.workbench.checkpoints.load_policy", return_value=policy):
            model = self.app.import_model({"path": str(source)})
        self.assertEqual(model["provenance"], provenance)
        self.assertEqual(model["environment_profile"], profile_metadata("europe_v1"))

    def test_legacy_weights_are_frozen_and_cannot_be_relabelled_europe(self):
        test = self.scene(61, profile="china_legacy")
        weight = self.root / "actor.pth"
        weight.write_bytes(b"old weights")
        config = legacy_config("sac_mlp")
        model = {"id": "legacy_model", "name": "legacy", "path": str(weight),
                 "legacy_type": "sac_mlp", "config": config}
        with patch.object(self.app, "get_model", return_value=model), \
             patch("satellite_rl.workbench.legacy.load_legacy_policy"):
            job = self.app.start_evaluation({"model_id": model["id"], "scene_id": test["id"]})
        self.assertEqual(Path(job["model_path"]).read_bytes(), b"old weights")
        weight.write_bytes(b"changed source")
        self.assertEqual(Path(job["model_path"]).read_bytes(), b"old weights")
        european = deepcopy(config)
        european["environment"]["profile"] = "europe_v1"
        with self.assertRaisesRegex(ValueError, "欧洲模型"):
            self.app.import_model({"path": str(weight), "legacy_type": "sac_mlp", "config": european})

    def test_fixed_evaluate_uses_frozen_custom_arrays_and_rejects_tampering(self):
        train, validation, test = (self.scene(seed) for seed in (19, 20, 21))
        job = self.submit(train, validation, test)
        directory = self.app.store.job_dir(job["run_id"])
        agent = Mock(rng=random.Random(123))
        observed = []
        def evaluate(env, policy, scene, *args, **kwargs):
            observed.append(deepcopy(scene))
            return {"scene_id": args[0], "metrics": {"final_si_mean": 0.5}}
        ledger = Mock()
        with patch("satellite_rl.workbench.runner._evaluation_episode", side_effect=evaluate):
            rows, _ = _fixed_evaluate(self.config, agent, directory, "test", 1, 9000, 0, ledger,
                                      scene_set=job["scene_sets"]["test"])
        with np.load(test["files"][0]["path"], allow_pickle=False) as arrays:
            for key in arrays.files:
                np.testing.assert_array_equal(observed[0][key], arrays[key])
        self.assertEqual(rows[0]["scene"]["source_scene_id"], test["id"])
        Path(job["scene_sets"]["test"]["files"][0]["path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "校验"):
            _fixed_evaluate(self.config, agent, directory, "test", 1, 9000, 0, ledger,
                             scene_set=job["scene_sets"]["test"])

    def test_cross_region_comparison_never_pairs_even_identical_hash_labels(self):
        china = default_config()
        records = [{"config": c, "job": {"name": "fixture"}} for c in (china, self.config)]
        result = {"summary": {}, "scenes": [{"scene": {"sha256": "f" * 64}, "metrics": {"final_si_mean": 0.5}}],
                  "metadata": {"action_mode": "deterministic"}}
        with patch.object(self.app, "run_detail", side_effect=records), patch.object(self.app, "results", return_value=result):
            comparison = self.app.compare(["china", "europe"])
        self.assertEqual(comparison["paired_scene_hashes"], [])

    def test_training_runner_routes_validation_test_and_exports_dataset_provenance(self):
        train, validation, test = (self.scene(seed) for seed in (19, 20, 21))
        self.config["training"]["total_steps"] = 1
        self.config["evaluation"]["interval_steps"] = 1
        job = self.submit(train, validation, test)
        directory = self.app.store.job_dir(job["run_id"])
        agent = SimpleNamespace(update_step=0, rng=random.Random(123),
            act=lambda state, **kw: [0, 0, 0], observe=lambda *a, **kw: None,
            ready=lambda *a: False, last_action_info={})
        ledger = Mock(status={"preflight": {"gpu": "CPU interface fixture"}})
        with patch("satellite_rl.workbench.runner.build_agent", return_value=agent), \
             patch("satellite_rl.workbench.runner.save_checkpoint") as checkpoint, \
             patch("satellite_rl.workbench.runner.export_policy") as export, \
             patch("satellite_rl.workbench.runner._ensure_finite_agent"):
            _training(directory, job, self.config, ledger, resume=False)
        result = read_json(directory / "results.json")
        self.assertEqual(result["metadata"]["profile"], "europe_v1")
        self.assertEqual(result["metadata"]["scene_sets"], job["scene_sets"])
        self.assertEqual(result["scenes"][0]["scene"]["source_scene_id"], test["id"])
        evaluations = [json.loads(line) for line in (directory / "evaluations.jsonl").read_text().splitlines()]
        self.assertEqual([row["group"] for row in evaluations], ["validation", "test"])
        for call in export.call_args_list:
            self.assertEqual(call.args[3]["scene_sets"], job["scene_sets"])
        for call in checkpoint.call_args_list:
            self.assertEqual(call.args[4]["scene_sets"], job["scene_sets"])

    def test_aviation_api_records_explicit_units_filters_and_requested_rate_model(self):
        path = self.root / "aircraft.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["aircraft_id", "timestamp", "lat", "lon", "altitude_m", "persons"])
            for index in range(100):
                writer.writerow([f"plane{index:03d}", "2026-09-20T10:00:00Z", 45 + index / 100, 13, 11000, 100])
            writer.writerow(["ground", "2026-09-20T10:00:00Z", 45, 13, "ground", 100])
        model = {"persons_per_aircraft": 180, "satellite_access_fraction": 0.5,
                 "active_fraction": 0.3, "rate_mbps_per_active_person": 2,
                 "aircraft_base_mbps": 1, "parameter_source": "test engineering assumption"}
        body = {"path": str(path), "selected_times": ["2026-09-20T10:00:00Z"],
                "partition_role": "train", "seed": 41, "window_seconds": 60, "demand_model": model}
        record = self.app.api("POST", ["scenes", "aviation"], {}, body)
        self.assertEqual(record["profile"], "europe_v1")
        self.assertEqual(record["source"]["partition_role"], "train")
        self.assertEqual(record["source"]["source_dates"], ["2026-09-20"])
        snapshot = record["source"]["snapshots"][0]
        self.assertEqual(snapshot["counts"]["ground_aircraft"], 1)
        self.assertAlmostEqual(snapshot["total_demand_bps"], 100 * 31e6)
        self.assertIn("synthetic", record["source"]["demand_interpretation"])
        self.assertEqual(record["source"]["input_units"]["altitude_m"], "m")
        with self.assertRaisesRegex(ValueError, "格式"):
            self.app.api("POST", ["scenes", "aviation"], {}, {**body, "source_format": "wrong"})
        with self.assertRaisesRegex(ValueError, "重复"):
            self.app.api("POST", ["scenes", "aviation"], {}, {**body, "selected_times": body["selected_times"] * 2})

    def test_midnight_observation_reuse_and_reimport_cannot_erase_partition(self):
        path = self.root / "midnight.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["aircraft_id", "timestamp", "lat", "lon", "altitude_m"])
            for index in range(100):
                writer.writerow([f"plane{index:03d}", "2026-09-20T23:59:50Z", 45 + index / 100, 13, 11000])
        model = {"persons_per_aircraft": 100, "satellite_access_fraction": 0.5,
                 "active_fraction": 0.3, "rate_mbps_per_active_person": 2,
                 "lognormal_sigma": 0.2, "parameter_source": "test engineering assumption"}
        body = {"path": str(path), "selected_times": ["2026-09-21T00:00:30Z"],
                "partition_role": "train", "seed": 41, "demand_model": model}
        train = self.app.api("POST", ["scenes", "aviation"], {}, body)
        test = self.app.api("POST", ["scenes", "aviation"], {}, {**body,
            "selected_times": ["2026-09-21T00:01:30Z"], "partition_role": "test"})
        self.assertNotEqual(train["files"][0]["sha256"], test["files"][0]["sha256"])
        self.assertEqual(train["source"]["source_dates"], ["2026-09-20"])
        validation = self.scene(48)
        with self.assertRaisesRegex(ValueError, "日期"):
            self.submit(train, validation, test)
        reimport = self.app.api("POST", ["scenes", "import"], {}, {"path": train["path"], "profile": "europe_v1"})
        self.assertEqual(reimport["source"]["partition_role"], "train")
        self.assertEqual(reimport["source"]["source_dates"], train["source"]["source_dates"])
        with self.assertRaisesRegex(ValueError, "日期"):
            self.submit(reimport, validation, test)
        model_record = {"id": "trained_model", "name": "trained", "config": self.config,
                        "path": str(self.root / "model"), "provenance": {"scene_sets": {"train": train}}}
        with patch.object(self.app, "get_model", return_value=model_record), \
             patch("satellite_rl.workbench.checkpoints.load_policy") as loader:
            with self.assertRaisesRegex(ValueError, "日期"):
                self.app.start_evaluation({"model_id": model_record["id"], "scene_id": test["id"]})
            loader.assert_not_called()

    def test_reimport_binds_sidecar_hashes_after_directory_move(self):
        original = self.scene()
        staged = self.root / "moved-staged"
        shutil.copytree(original["path"], staged)
        sidecar = staged / "record.json"
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        record["source"].update(partition_role="train", source_dates=["2024-01-08"])
        sidecar.write_text(json.dumps(record), encoding="utf-8")
        imported = self.app.api("POST", ["scenes", "import"], {},
                                {"path": str(staged), "profile": "europe_v1"})
        self.assertEqual(imported["source"]["partition_role"], "train")
        self.assertEqual(imported["source"]["source_dates"], ["2024-01-08"])
        scene_file = next(staged.glob("*.npz"))
        scene_file.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA-256校验失败"):
            self.app.api("POST", ["scenes", "import"], {},
                         {"path": str(staged), "profile": "europe_v1"})
        shutil.copyfile(original["files"][0]["path"], scene_file)
        record.pop("files")
        sidecar.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "缺少数组文件"):
            self.app.api("POST", ["scenes", "import"], {},
                         {"path": str(staged), "profile": "europe_v1"})
        record["files"] = original["files"]
        record["files"][0].pop("sha256")
        sidecar.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "缺少完整SHA-256"):
            self.app.api("POST", ["scenes", "import"], {},
                         {"path": str(staged), "profile": "europe_v1"})

    def test_reimport_rejects_unknown_or_conflicting_sidecar_files(self):
        original = self.app.api("POST", ["scenes"], {},
                                {"seed": 19, "count": 2, "profile": "europe_v1"})
        staged = self.root / "staged-pair"
        shutil.copytree(original["path"], staged)
        sidecar = staged / "record.json"
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        record["files"][1]["path"] = record["files"][0]["path"]
        sidecar.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "文件名重复"):
            self.app.api("POST", ["scenes", "import"], {},
                         {"path": str(staged), "profile": "europe_v1"})
        record = json.loads((Path(original["path"]) / "record.json").read_text(encoding="utf-8"))
        record["files"][1]["path"] = "elsewhere/unknown.npz"
        sidecar.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "实际文件不一致"):
            self.app.api("POST", ["scenes", "import"], {},
                         {"path": str(staged), "profile": "europe_v1"})

    def test_legacy_china_npz_without_sidecar_still_imports(self):
        original = self.scene(profile="china_legacy")
        directory = self.root / "legacy-npz"
        directory.mkdir()
        shutil.copyfile(original["files"][0]["path"], directory / "scene_00000.npz")
        imported = self.app.api("POST", ["scenes", "import"], {},
                                {"path": str(directory), "profile": "china_legacy"})
        self.assertEqual(imported["count"], 1)
        self.assertEqual(imported["profile"], "china_legacy")


if __name__ == "__main__":
    unittest.main()
