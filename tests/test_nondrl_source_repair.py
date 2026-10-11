import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from satellite_rl.workbench import nondrl_source_repair as repair
from satellite_rl.workbench.runtime import Scheduler
from satellite_rl.workbench.storage import Store, digest, read_json
from satellite_rl.training.run_monitor import atomic_json


class SourceRepairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.repo = self.base / "20473/ops/repair/repo"
        self.repo.mkdir(parents=True)
        self.git("init", "-q")
        self.git("config", "user.name", "unit test")
        self.git("config", "user.email", "unit@example.invalid")
        file = self.repo / "src/satellite_rl/coverage/nondrl/classical.py"
        file.parent.mkdir(parents=True)
        file.write_text("original=True\n")
        (self.repo / "data").mkdir()
        (self.repo / "data/a.json").write_text("{}")
        (self.repo / "pyproject.toml").write_text("[project]\nname='test'\nversion='0.0.0'\n")
        self.git("add", "src", "data", "pyproject.toml")
        self.git("commit", "-qm", "origin")
        self.origin = self.git("rev-parse", "HEAD").strip()
        self.origin_patch = patch.object(repair, "ORIGIN_COMMIT", self.origin)
        self.origin_patch.start()
        self.store = Store(self.base / "20473/archive", self.repo)
        self.old_snapshot, _ = self.store.freeze_source()
        file.write_text("original=True\nbounded_native=True\n")
        self.git("add", "src")
        self.git("commit", "-qm", "repair")
        self.execution = self.git("rev-parse", "HEAD").strip()
        self.run_id = "prepare_nondrl_20261012_045228_1d87b8e0"
        self.directory = self.store.job_dir(self.run_id)
        self.directory.mkdir()
        self.job = {"experiment_id": "S1-ND12", "run_id": self.run_id, "git_commit": self.origin,
                    "source_root": str(self.old_snapshot), "kind": "prepare", "variant": "02",
                    "preparation_mode": "teacher_probe"}
        atomic_json(self.directory / "job.json", self.job)
        atomic_json(self.directory / "config.json", {"frozen_budget": 600})
        atomic_json(self.directory / "status.json", {"stage": "completed", "pid": 991111})
        atomic_json(self.directory / "launch.json", {"pid": 991111, "process_start_time": 100.})
        raw = self.directory / "evidence/manifest.json"
        raw.parent.mkdir()
        atomic_json(raw, {"provenance": {"run_id": self.run_id, "git_commit": self.origin,
            "source_sha256": digest(self.old_snapshot / "source_manifest.json")}, "problems": 60})
        (self.directory / "worker.log").write_text("original completed\n")
        self.proof = self.base / "proof.json"
        atomic_json(self.proof, {"passed": True})
        self.contract_path = self.base / "contract.json"
        self.contract = {
            "format": repair.FORMAT, "repair_id": repair.REPAIR_ID, "origin_commit": self.origin,
            "execution_commit": self.execution, "execution_source_repo": str(self.repo),
            "allowed_run_ids": [self.run_id], "allowed_future_roles": [
                {"route": "02", "kind": "train", "preparation_mode": None}],
            "runs": {self.run_id: repair.pin_run(self.store, self.run_id)},
            "diff_files": repair.pin_diff(self.store, self.repo, self.execution),
            "validation_receipts": [{"path": str(self.proof), "sha256": digest(self.proof)}]}
        self.publish()
        self.scheduler = Scheduler(self.store)

    def tearDown(self):
        self.origin_patch.stop()
        self.temp.cleanup()

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.repo, text=True, check=True,
                              capture_output=True).stdout

    def publish(self):
        atomic_json(self.contract_path, self.contract)
        self.body = {"contract_path": str(self.contract_path), "contract_sha256": digest(self.contract_path)}

    def do_repair(self):
        with patch("satellite_rl.workbench.runtime.pid_alive", return_value=False):
            return repair.repair_job(self.scheduler, self.run_id, self.body)

    def test_completed_probe_preserves_original_bytes_and_marks_actual_execution(self):
        original = (self.directory / "job.json").read_bytes()
        old_raw = (self.directory / "evidence/manifest.json").read_bytes()
        result = self.do_repair()
        self.assertEqual(result["stage"], "queued")
        current = read_json(self.directory / "job.json")
        self.assertEqual(current["git_commit"], self.execution)
        self.assertEqual(current["origin_commit"], self.origin)
        self.assertEqual((self.directory / "source_lineage/origin/job.json").read_bytes(), original)
        self.assertEqual((self.directory / "source_lineage/origin/artifacts/evidence/manifest.json").read_bytes(), old_raw)
        atomic_json(self.directory / "evidence/manifest.json", {"new_committed_progress": True})
        repair.validate_execution_job(self.store, current)
        self.assertEqual(self.do_repair()["stage"], "queued")
        repair.validate_job_lineage(current, Path("/not/on/this/host"), verify_source_snapshot=False,
            read_artifact=lambda relative: (self.directory / relative).read_bytes())

    def test_live_original_birth_is_rejected_without_any_mutation(self):
        before = (self.directory / "job.json").read_bytes()
        with patch("satellite_rl.workbench.runtime.pid_alive", return_value=True):
            with self.assertRaisesRegex(ValueError, "birth"):
                repair.repair_job(self.scheduler, self.run_id, self.body)
        self.assertEqual((self.directory / "job.json").read_bytes(), before)

    def test_tampered_original_config_is_rejected(self):
        (self.directory / "config.json").write_text("{\"frozen_budget\":601}")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            self.do_repair()

    def test_scientific_stop_cannot_be_revived(self):
        atomic_json(self.directory / "status.json", {"stage": "completed", "scientific_stop": True})
        self.contract["runs"][self.run_id] = repair.pin_run(self.store, self.run_id)
        self.publish()
        with self.assertRaisesRegex(ValueError, "terminal"):
            self.do_repair()

    def test_missing_worker_manifest_pin_is_rejected(self):
        new = self.directory / "workers/problem_000003/manifest.json"
        new.parent.mkdir(parents=True)
        atomic_json(new, {"old_active": True})
        with self.assertRaisesRegex(ValueError, "must all be pinned"):
            self.do_repair()

    def test_unapproved_future_route_and_git_diff_are_rejected(self):
        self.contract["allowed_future_roles"].append({"route": "04", "kind": "train"})
        self.publish()
        with self.assertRaisesRegex(ValueError, "scope"):
            repair.validate_contract(self.store, **{"path": self.contract_path, "sha256": self.body["contract_sha256"]})
        self.contract["allowed_future_roles"].pop()
        self.contract["diff_files"] = []
        self.publish()
        with self.assertRaisesRegex(ValueError, "diff"):
            repair.validate_contract(self.store, self.contract_path, self.body["contract_sha256"])

    def test_future_task_binds_exact_config_and_source_without_reusing_old_provenance(self):
        extra = {"experiment_id": "S1-ND12", "variant": "02", "execution_source": self.body}
        snapshot, commit = repair.select_source(self.store, extra, "train")
        future = {**extra, "kind": "train", "run_id": "new_train", "source_root": str(snapshot), "git_commit": commit}
        directory = self.store.job_dir("new_train")
        directory.mkdir()
        atomic_json(directory / "config.json", {"budget": 10000})
        repair.bind_new_job(self.store, future, directory)
        self.assertIsNone(future["source_repair_contract"]["origin_provenance"])
        repair.validate_job_lineage(future, directory)
        from satellite_rl.coverage.nondrl import source_lineage
        from satellite_rl.coverage.nondrl.runner import RowStore
        provenance = {"experiment_id": "S1-ND12", "run_id": "new_train", "git_commit": commit,
                      "source_sha256": future["source_history"][-1]["source_manifest_sha256"]}
        with patch.object(source_lineage, "ORIGIN_COMMIT", self.origin):
            source_lineage.validate_repair(future["source_repair_contract"], provenance)
            rows = RowStore(directory / "evidence", provenance, repair_contract=future["source_repair_contract"])
            self.assertEqual(rows.manifest["execution_segments"][0]["execution_source"]["git_commit"], commit)
        future["variant"] = "04"
        with self.assertRaisesRegex(ValueError, "role"):
            repair.validate_job_lineage(future, directory)

    def test_mixed_inline_contract_and_mutated_snapshot_fail_closed(self):
        self.do_repair()
        current = read_json(self.directory / "job.json")
        current["source_repair_contract"]["origin_provenance"]["git_commit"] = self.execution
        with self.assertRaisesRegex(ValueError, "inline"):
            repair.validate_execution_job(self.store, current)
        current = read_json(self.directory / "job.json")
        path = Path(current["source_root"]) / "src/satellite_rl/coverage/nondrl/classical.py"
        path.write_text("tampered=True")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            repair.validate_execution_job(self.store, current)

    def test_partial_activation_cannot_launch_and_same_contract_finishes_transaction(self):
        self.do_repair()
        current = read_json(self.directory / "job.json")
        current["source_repair_pending"] = True
        atomic_json(self.directory / "job.json", current)
        atomic_json(self.directory / "status.json", {"stage": "completed"})
        with self.assertRaisesRegex(ValueError, "not yet committed"):
            repair.validate_execution_job(self.store, current)
        self.assertEqual(self.do_repair()["stage"], "queued")
        self.assertFalse(read_json(self.directory / "job.json")["source_repair_pending"])

    def test_source_consumer_cannot_use_an_unrepaired_teacher(self):
        with self.assertRaisesRegex(ValueError, "consumer"):
            repair.validate_dependency_source(self.store, self.body, self.job)
        self.do_repair()
        current = read_json(self.directory / "job.json")
        repair.validate_dependency_source(self.store, self.body, current)
        with self.assertRaisesRegex(ValueError, "consumer"):
            repair.validate_dependency_source(self.store, None, current)

    def test_explicit_api_route_and_worker_launch_fail_closed(self):
        from satellite_rl.workbench.coverage_nondrl import api
        application = SimpleNamespace(store=self.store, app=SimpleNamespace(scheduler=self.scheduler))
        with patch("satellite_rl.workbench.coverage_nondrl.deployment"), patch(
                "satellite_rl.workbench.runtime.pid_alive", return_value=False):
            result = api(application, "POST", ["runs", self.run_id, "source-repair"], self.body)
        self.assertEqual(result["stage"], "queued")
        current = read_json(self.directory / "job.json")
        current["execution_commit"] = self.origin
        with patch("satellite_rl.workbench.runtime.subprocess.Popen") as spawn:
            self.scheduler._launch(current, result)
            spawn.assert_not_called()
        self.assertTrue(read_json(self.directory / "status.json")["attention_required"])


if __name__ == "__main__":
    unittest.main()
