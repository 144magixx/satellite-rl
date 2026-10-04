"""S1-03 workbench provenance, role and schema contracts."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from satellite_rl.coverage.config import config_hash, validate_config
from satellite_rl.coverage.reconstruction_config import reconstruction_default_config
from satellite_rl.workbench.coverage import CoverageApplication
from satellite_rl.workbench.service import Application
from satellite_rl.workbench.storage import digest


class ReconstructionWorkbenchTests(unittest.TestCase):
    def test_schema4_rejects_unknown_budget_and_mismatched_provenance(self):
        config = reconstruction_default_config()
        self.assertEqual(validate_config(config)["schema_version"], 4)
        self.assertEqual(config_hash(config), config_hash(validate_config(config)))
        for mutate in (
            lambda c: c["coverage"]["reconstruction"].update(route="P1"),
            lambda c: c["coverage"]["reconstruction"].update(split_templates=[0, 1]),
            lambda c: c["coverage"]["reconstruction"].update(max_geometry_calls_total=1),
            lambda c: c["training"].update(total_steps=524288),
            lambda c: c["network"].update(hidden_size=256),
            lambda c: c["coverage"].update(max_movement_deg=1.),
        ):
            changed = deepcopy(config)
            mutate(changed)
            with self.assertRaises(ValueError):
                validate_config(changed)

    def test_submit_freezes_prepare_start_and_rejects_false_target_or_stale_sha(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            application = Application(root / "archive")
            coverage = CoverageApplication(application)
            scene_ids, start_ids = {}, {}
            for role, date in (("train", "2024-01-08"), ("validation", "2024-01-12")):
                source = root / role
                source.mkdir()
                scene = source / "inputs.npz"
                scene.write_bytes((role + " dataset").encode())
                (source / "record.json").write_text(json.dumps({"source_dates": [date]}))
                scene_id = "scene_" + role
                scene_ids[role] = scene_id
                scene_sha = digest(scene)
                application.store.save_record("scenes", {
                    "id": scene_id, "research_stage": "coverage", "partition_role": role,
                    "dates": [date], "initialization": "cluster", "cluster_iterations": 12,
                    "path": str(scene), "sha256": scene_sha})
                start_id = "prepare_" + role
                start_ids[role] = start_id
                directory = application.store.job_dir(start_id)
                directory.mkdir()
                starts = directory / "starts.jsonl"
                starts.write_text(json.dumps({"scene_id": role}) + "\n")
                prepared = reconstruction_default_config()
                prepared["datasets"] = {role: {"path": str(scene), "sha256": scene_sha,
                                               "partition_role": role}}
                (directory / "job.json").write_text(json.dumps({
                    "research_stage": "coverage", "execution_target": "fixture:20418"}))
                (directory / "status.json").write_text(json.dumps({"stage": "completed"}))
                (directory / "config.json").write_text(json.dumps(prepared))
                (directory / "artifacts.json").write_text(json.dumps({
                    "reconstruction_starts": str(starts),
                    "reconstruction_starts_sha256": digest(starts)}))
            config = reconstruction_default_config()
            config["coverage"]["reconstruction"]["route"] = "P1"
            config["coverage"]["method"] = "ppo"
            body = {"config": config, "kind": "train", "dataset_ids": scene_ids,
                    "start_run_ids": start_ids, "request_id": "s103_train_fixture"}
            with mock.patch("satellite_rl.workbench.coverage._require_remote_dqn"), \
                 mock.patch("satellite_rl.workbench.coverage._reconstruction_target", return_value="fixture:20418"), \
                 mock.patch.object(application.store, "freeze_source", return_value=(root, "fixture-commit")):
                with self.assertRaisesRegex(ValueError, "执行目标"):
                    coverage.submit({**body, "execution_target": "fixture:20424"})
                job = coverage.submit(body)
                frozen = json.loads((application.store.job_dir(job["run_id"]) / "config.json").read_text(encoding="utf-8"))
                self.assertEqual(set(frozen["reconstruction_starts"]), {"train", "validation"})
                for role in ("train", "validation"):
                    self.assertEqual(frozen["reconstruction_starts"][role]["source_run_id"], start_ids[role])
                    self.assertEqual(digest(frozen["reconstruction_starts"][role]["path"]),
                                     frozen["reconstruction_starts"][role]["sha256"])
                (application.store.job_dir(start_ids["validation"]) / "starts.jsonl").write_text("changed")
                with self.assertRaisesRegex(ValueError, "SHA"):
                    coverage.submit({**body, "request_id": "s103_train_stale"})
                job_dir = application.store.job_dir(job["run_id"])
                checkpoint = job_dir / "checkpoint.pt"
                checkpoint.write_bytes(b"fixture full checkpoint")
                checkpoint_sha = digest(checkpoint)
                status = json.loads((job_dir / "status.json").read_text(encoding="utf-8"))
                status.update(stage="stopped", stop_reason="mechanism_gate_pending",
                              env_step=65536, checkpoint_sha256=checkpoint_sha)
                (job_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "机制闸门"):
                    application.scheduler.resume(job["run_id"])
                self.assertEqual(json.loads((job_dir / "status.json").read_text(encoding="utf-8"))["stage"], "stopped")
                audit_path = job_dir / "mechanism_audit.json"
                audit_path.write_text(json.dumps({"passed": True, "run_id": job["run_id"],
                                                  "env_step": 65536, "checkpoint_sha256": checkpoint_sha}),
                                      encoding="utf-8")
                proof = coverage.approve_reconstruction_gate(job["run_id"], {
                    "audit_path": str(audit_path), "audit_sha256": digest(audit_path)})
                self.assertEqual(proof["checkpoint_sha256"], checkpoint_sha)
                self.assertEqual(application.scheduler.resume(job["run_id"])["stage"], "queued")

    def test_cross_instance_import_keeps_original_dataset_and_prepare_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "s1_03"
            source_app = Application(parent / "20424")
            target_app = Application(parent / "20418")
            source_id = "prepared_train"
            scene_dir = source_app.store.record_dir("scenes", source_id)
            scene_dir.mkdir(parents=True)
            scene = scene_dir / "inputs.npz"
            scene.write_bytes(b"fixed Jan08 scene input")
            source_app.store.save_record("scenes", {
                "id": source_id, "research_stage": "coverage", "partition_role": "train",
                "initialization": "cluster", "cluster_iterations": 12,
                "dates": ["2024-01-08"], "count": 1, "path": str(scene),
                "sha256": digest(scene)})
            prepare_id = "prepare_train"
            prepare_dir = source_app.store.job_dir(prepare_id)
            prepare_dir.mkdir()
            starts = prepare_dir / "starts.jsonl"
            starts.write_text('{"scene_id":"one"}\n')
            prepared = reconstruction_default_config()
            prepared["datasets"] = {"train": {"path": str(scene), "sha256": digest(scene),
                                               "partition_role": "train"}}
            (prepare_dir / "job.json").write_text(json.dumps({
                "run_id": prepare_id, "research_stage": "coverage",
                "execution_target": "u2024110224@10.160.17.102:20424"}))
            (prepare_dir / "status.json").write_text(json.dumps({"stage": "completed"}))
            (prepare_dir / "config.json").write_text(json.dumps(prepared))
            (prepare_dir / "artifacts.json").write_text(json.dumps({
                "reconstruction_starts": str(starts), "reconstruction_starts_sha256": digest(starts)}))
            train_id = "train_policy"
            train_dir = source_app.store.job_dir(train_id)
            train_dir.mkdir()
            trained = reconstruction_default_config()
            trained["coverage"]["reconstruction"]["route"] = "P1"
            trained["coverage"]["method"] = "ppo"
            trained["datasets"] = prepared["datasets"]
            trained["reconstruction_starts"] = {"train": {
                "path": str(starts), "sha256": digest(starts),
                "dataset_sha256": digest(scene), "source_run_id": prepare_id}}
            (train_dir / "job.json").write_text(json.dumps({
                "run_id": train_id, "kind": "train", "research_stage": "coverage",
                "execution_target": "u2024110224@10.160.17.102:20424",
                "git_commit": "fixture-source"}))
            (train_dir / "status.json").write_text(json.dumps({"stage": "completed"}))
            (train_dir / "config.json").write_text(json.dumps(trained))
            final = source_app.store.models_root / train_id / "final"
            final.mkdir(parents=True)
            (final / "policy.pt").write_bytes(b"fixed terminal PPO weights")
            (final / "model.json").write_text(json.dumps({
                "research_stage": "coverage", "schema_version": 4, "run_id": train_id,
                "execution_target": "u2024110224@10.160.17.102:20424",
                "source_commit": "fixture-source", "env_step": 262144,
                "config_hash": config_hash(trained), "policy_sha256": digest(final / "policy.pt")}))
            coverage = CoverageApplication(target_app)
            with mock.patch("satellite_rl.workbench.coverage.RECONSTRUCTION_ARCHIVE_PARENT", parent), \
                 mock.patch("satellite_rl.workbench.coverage._require_remote_dqn"), \
                 mock.patch("satellite_rl.workbench.coverage._reconstruction_target",
                            return_value="u2024110224@10.160.17.102:20418"):
                imported_dataset = coverage.import_reconstruction_dataset({
                    "source_target": "20424", "source_dataset_id": source_id,
                    "partition_role": "train"})
                imported_start = coverage.import_reconstruction_start({
                    "source_target": "20424", "source_run_id": prepare_id,
                    "partition_role": "train"})
                imported_model = coverage.import_reconstruction_model({
                    "source_target": "20424", "source_run_id": train_id})
            self.assertEqual(imported_dataset["sha256"], digest(scene))
            self.assertEqual(imported_start["dataset_sha256"], imported_dataset["sha256"])
            self.assertEqual(digest(imported_start["path"]), digest(starts))
            self.assertEqual(imported_start["source_run_id"], prepare_id)
            self.assertEqual(imported_start["source_target"], "u2024110224@10.160.17.102:20424")
            self.assertEqual(imported_model["sha256"], digest(final / "policy.pt"))
            self.assertEqual(imported_model["source_run_id"], train_id)
            self.assertIn(imported_model["id"], {row["id"] for row in coverage.models()})


if __name__ == "__main__":
    unittest.main()
