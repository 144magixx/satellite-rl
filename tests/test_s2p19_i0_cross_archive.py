"""A foreign I0 endpoint is accepted only with its complete frozen provenance."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from satellite_rl.sac_improvement import DEFAULT_CONFIG
from satellite_rl.workbench.s2p19_service import (
    I0_SOURCE_COMMIT, I0_SOURCE_RUN, S2P19Application, external_i0_endpoint)
from satellite_rl.workbench.storage import Store, digest


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class FakeStore:
    def __init__(self, archive_root):
        self.archive_root = archive_root
        self.created = []

    def create_job(self, config, **kwargs):
        self.created.append((config, kwargs))
        return {"run_id": "evaluate_sac_test", "job": kwargs}


class CrossArchiveI0Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.source = root / "20418" / "archive"
        self.target = root / "20474" / "archive"
        self.job_dir = self.source / "outputs" / "workbench" / "jobs" / I0_SOURCE_RUN
        self.model_dir = self.source / "models" / "runs" / "workbench" / I0_SOURCE_RUN / "final"
        self.model_dir.mkdir(parents=True)
        (self.job_dir / "checkpoint.pt").parent.mkdir(parents=True)
        (self.job_dir / "checkpoint.pt").write_bytes(b"frozen checkpoint")
        (self.model_dir / "policy.pt").write_bytes(b"frozen policy")
        self.config = {"research_stage": "sac_improvement", "i0": DEFAULT_CONFIG}
        write_json(self.job_dir / "config.json", self.config)
        self.job = {"run_id": I0_SOURCE_RUN, "archive_root": str(self.source),
            "model_dir": str(self.model_dir.parent), "experiment_id": "S2-P19",
            "kind": "train", "research_stage": "sac_improvement", "mode": "train",
            "git_commit": I0_SOURCE_COMMIT, "seed": 51001, "action_seed": 53001,
            "repeats": 1, "total_steps": 400000, "i0_config": DEFAULT_CONFIG}
        write_json(self.job_dir / "job.json", self.job)
        self.shas = {name: digest((self.model_dir if name in ("policy.pt", "model.json")
                                   else self.job_dir) / name)
                     for name in ("config.json", "checkpoint.pt", "policy.pt")}
        self.status = {"run_id": I0_SOURCE_RUN, "stage": "completed",
            "completed_due_to": "episode_cap", "env_step": 397112, "episodes": 12500,
            "updates": 395113, "budget_env_steps": 400000,
            "checkpoint_sha256": self.shas["checkpoint.pt"],
            "model_path": str(self.model_dir / "policy.pt"),
            "model_sha256": self.shas["policy.pt"]}
        write_json(self.job_dir / "status.json", self.status)
        self.model = {"run_id": I0_SOURCE_RUN, "env_step": 397112,
            "episodes": 12500, "updates": 395113, "git_commit": I0_SOURCE_COMMIT,
            "format": "satellite_rl_s2p19_improvement_v1",
            "config_sha256": self.shas["config.json"],
            "policy_sha256": self.shas["policy.pt"],
            "provenance": {"run_id": I0_SOURCE_RUN, "env_step": 397112}}
        write_json(self.model_dir / "model.json", self.model)
        self.shas["model.json"] = digest(self.model_dir / "model.json")
        self.store = FakeStore(self.target)
        constants = patch.multiple("satellite_rl.workbench.s2p19_service",
            I0_SOURCE_ARCHIVE=self.source, I0_EVALUATION_ARCHIVE=self.target,
            I0_SOURCE_SHAS=self.shas)
        constants.start()
        self.addCleanup(constants.stop)

    def audit(self):
        return external_i0_endpoint(self.store, self.source, I0_SOURCE_RUN, DEFAULT_CONFIG)

    def test_accepts_exact_endpoint_and_preserves_original_model_run(self):
        path, files, provenance = self.audit()
        self.assertEqual(path, str(self.model_dir))
        self.assertEqual({Path(row["path"]).name for row in files}, {"policy.pt", "model.json"})
        self.assertEqual(provenance["source_run_id"], I0_SOURCE_RUN)
        app = S2P19Application(type("App", (), {"store": self.store})())
        with patch.object(app, "scene", return_value={"id": "test", "count": 880}):
            app.improvement({"mode": "evaluate", "scene_id": "test", "model_run_id": I0_SOURCE_RUN,
                             "model_source_archive": str(self.source), "repeats": 3,
                             "action_seed": 53001})
        config, kwargs = self.store.created[0]
        self.assertEqual(config["training"]["total_steps"], 2640)
        self.assertEqual(kwargs["extra"]["model_run_id"], I0_SOURCE_RUN)
        self.assertEqual(kwargs["extra"]["model_files"], files)
        self.assertEqual(kwargs["extra"]["external_model_source"]["source_env_step"], 397112)

    def test_store_freezes_audited_pair_without_importing_training_job(self):
        real_store = Store(self.target)
        app = S2P19Application(type("App", (), {"store": real_store})())
        scene = {"id": "test", "count": 880, "path": str(self.target / "test_scenes")}
        with patch.object(app, "scene", return_value=scene), patch.object(
                real_store, "freeze_source", return_value=(self.target, "13ea-test")), patch(
                "satellite_rl.workbench.datasets.freeze_scene_set", return_value=scene):
            job = app.improvement({"mode": "evaluate", "scene_id": "test",
                "model_run_id": I0_SOURCE_RUN, "model_source_archive": str(self.source),
                "repeats": 3, "action_seed": 53001})
        job_dir = real_store.job_dir(job["run_id"])
        frozen = job_dir / "model_input"
        self.assertEqual(digest(frozen / "policy.pt"), self.shas["policy.pt"])
        self.assertEqual(digest(frozen / "model.json"), self.shas["model.json"])
        self.assertEqual(job["model_run_id"], I0_SOURCE_RUN)
        self.assertEqual(job["model_path"], str(frozen))
        self.assertFalse(real_store.job_dir(I0_SOURCE_RUN).exists())

    def test_existing_local_completed_run_still_uses_original_entry(self):
        local = self.target / "outputs" / "workbench" / "jobs" / I0_SOURCE_RUN
        write_json(local / "job.json", {"experiment_id": "S2-P19",
            "model_dir": str(self.model_dir.parent)})
        write_json(local / "status.json", {"stage": "completed"})
        app_obj = type("App", (), {"store": self.store,
            "resolve_run": lambda _, run_id: (local, False)})()
        app = S2P19Application(app_obj)
        with patch.object(app, "scene", return_value={"id": "test", "count": 880}):
            app.improvement({"mode": "evaluate", "scene_id": "test",
                "model_run_id": I0_SOURCE_RUN})
        self.assertNotIn("external_model_source", self.store.created[0][1]["extra"])

    def test_rejects_foreign_destination_and_source(self):
        self.store.archive_root = self.source
        with self.assertRaises(ValueError):
            self.audit()
        self.store.archive_root = self.target
        with self.assertRaises(ValueError):
            external_i0_endpoint(self.store, self.target, I0_SOURCE_RUN, DEFAULT_CONFIG)
        with self.assertRaises(ValueError):
            external_i0_endpoint(self.store, self.source, "another_run", DEFAULT_CONFIG)

    def test_rejects_incomplete_or_modified_endpoint(self):
        for status_change in ({"stage": "training"}, {"env_step": 400000},
                              {"updates": 395112}, {"checkpoint_sha256": "0" * 64}):
            write_json(self.job_dir / "status.json", {**self.status, **status_change})
            with self.assertRaises(ValueError):
                self.audit()
        write_json(self.job_dir / "status.json", self.status)
        (self.model_dir / "policy.pt").write_bytes(b"changed policy")
        with self.assertRaises(ValueError):
            self.audit()
        (self.model_dir / "policy.pt").write_bytes(b"frozen policy")
        (self.model_dir / "model.json").unlink()
        with self.assertRaises(ValueError):
            self.audit()

    def test_rejects_forged_metadata_and_config(self):
        write_json(self.model_dir / "model.json", {**self.model,
            "provenance": {"run_id": "forged", "env_step": 397112}})
        with patch("satellite_rl.workbench.s2p19_service.I0_SOURCE_SHAS",
                   {**self.shas, "model.json": digest(self.model_dir / "model.json")}):
            with self.assertRaises(ValueError):
                self.audit()
        write_json(self.model_dir / "model.json", self.model)
        write_json(self.job_dir / "config.json", {"research_stage": "sac_improvement", "i0": {}})
        with self.assertRaises(ValueError):
            self.audit()


if __name__ == "__main__":
    unittest.main()
