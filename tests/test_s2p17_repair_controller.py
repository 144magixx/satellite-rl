"""The resumed controller separates origin science from repair runtime state."""

import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "variable_beams"))
import controller  # noqa: E402
import remote_server  # noqa: E402


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ControllerRepairTests(unittest.TestCase):
    def test_switch_contract_checks_failed_first_switch_and_allows_bound_reuse(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            commit = "b" * 40
            old_repo = base / "repo"
            new_repo = base / "recovery" / commit / "repo"
            old_repo.mkdir(parents=True)
            new_repo.mkdir(parents=True)
            ops = base / "ops"
            validation = (base / "archive" / "outputs/workbench/jobs" / controller.FAILED_RUN /
                          "recoveries/validation_receipt.json")
            write(validation, {"tests_passed": True})
            status = validation.parent.parent / "status.json"
            write(status, {"stage": "failed", "env_step": 299})
            contract_path = ops / "recovery_switch_contract.json"
            contract = {"format": "s2p17_server_repair_switch_v1",
                        "repair_id": controller.REPAIR_ID,
                        "origin_science_commit": controller.ORIGIN_COMMIT,
                        "repair_commit": commit,
                        "origin_repo": str(old_repo), "repair_repo": str(new_repo),
                        "failed_run_id": controller.FAILED_RUN,
                        "validation_receipt_path": str(validation),
                        "validation_receipt_sha256": sha(validation),
                        "old_server_pid": 71017, "old_server_start_ticks": 1234}
            write(contract_path, contract)
            old = {"commit": controller.ORIGIN_COMMIT, "repo": str(old_repo),
                   "pid": 71017, "start_ticks": 1234, "max_workers": 1}
            with patch.object(remote_server, "BASE", base), \
                    patch.object(remote_server, "REPO", old_repo), \
                    patch.object(remote_server, "ARCHIVE", base / "archive"), \
                    patch.object(remote_server, "OPS", ops), \
                    patch.object(remote_server, "LAUNCH", ops / "workbench_8781_launch.json"):
                self.assertEqual(remote_server._repair_repo_contract(
                    new_repo, commit, contract_path, sha(contract_path), old), contract)
                write(status, {"stage": "queued", "env_step": 299})
                self.assertEqual(remote_server._repair_repo_contract(
                    new_repo, commit, contract_path, sha(contract_path), None), contract)
                proof = {"repair_id": controller.REPAIR_ID,
                         "origin_science_commit": controller.ORIGIN_COMMIT,
                         "repair_commit": commit,
                         "new_source_root": str(base / "archive/source_snapshots" / commit),
                         "fallback_contract_sha256": "c" * 64}
                write(status.parent / "job.json", {"git_commit": commit,
                     "source_root": proof["new_source_root"], "boundary_repair": proof})
                launch = {"commit": commit, "repo": str(new_repo), "pid": 999,
                          "start_ticks": 1234, "max_workers": 1,
                          "repair_switch_contract_sha256": sha(contract_path)}
                write(ops / "workbench_8781_launch.json", launch)
                system = {"source_root": str(new_repo), "archive_root": str(base / "archive"),
                          "package_file": str(new_repo / "src/satellite_rl/__init__.py"),
                          "cuda": True}
                with patch.object(remote_server, "preflight", return_value={}), \
                        patch.object(remote_server, "port_open", return_value=True), \
                        patch.object(remote_server, "proc_birth", return_value=1234), \
                        patch.object(remote_server, "api", return_value=system):
                    self.assertEqual(remote_server.ensure_server(
                        commit, "evaluation", repo=new_repo,
                        repair_switch_contract=contract_path,
                        repair_switch_sha256=sha(contract_path))["status"],
                        "already_running")

    def test_repair_binding_clears_stale_error_and_step_returns_running_sca(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            ops = base / "ops"
            repair_commit = "b" * 40
            runtime_repo = base / "recovery" / repair_commit / "repo"
            runtime_repo.mkdir(parents=True)
            proof = {"repair_id": controller.REPAIR_ID,
                     "origin_science_commit": controller.ORIGIN_COMMIT,
                     "repair_commit": repair_commit,
                     "new_source_root": str(base / "archive" / "source_snapshots" / repair_commit),
                     "fallback_contract_sha256": "c" * 64,
                     "archive_path": str(base / "archive" / "recoveries" / "one")}
            proof_path = Path(proof["archive_path"]) / "repair_proof.json"
            write(proof_path, proof)
            switch_path = ops / "recovery_switch_contract.json"
            write(switch_path, {"repair_commit": repair_commit})
            job_path = base / "archive" / "jobs" / controller.FAILED_RUN / "job.json"
            write(job_path, {"run_id": controller.FAILED_RUN,
                             "git_commit": repair_commit,
                             "scientific_origin_commit": controller.ORIGIN_COMMIT,
                             "source_root": proof["new_source_root"],
                             "boundary_repair": proof})
            jobs = {}
            for key in controller.REQUESTS:
                jobs[key] = {"run_id": key, "stage": "completed"}
            jobs["V0_sca"] = {"run_id": controller.FAILED_RUN,
                              "stage": "queued", "job_path": str(job_path)}
            manifest = {"scientific_commit": controller.ORIGIN_COMMIT,
                        "datasets": {name: {} for name in ("train", "validation", "test")},
                        "source": {"h0_model_id": "model-h0"},
                        "jobs": jobs, "phase": "attention_required",
                        "last_error": "V0_sca failed under old source"}
            with patch.object(controller, "BASE", base), \
                    patch.object(controller, "OPS", ops), \
                    patch.object(controller, "MANIFEST", ops / "controller_manifest.json"), \
                    patch.object(controller, "scope_contract", return_value={}), \
                    patch.object(controller, "validate_system"), \
                    patch.object(controller, "load_manifest", return_value=manifest), \
                    patch.object(controller, "refresh_job", side_effect=lambda m, k: m["jobs"][k]), \
                    patch.object(controller, "final_model"), \
                    patch.object(controller, "ensure_server"), \
                    patch.object(controller, "check_server_mode"):
                result = controller.step(
                    controller.ORIGIN_COMMIT, "scope-sha",
                    runtime_commit=repair_commit, runtime_repo=runtime_repo,
                    repair_switch_contract=switch_path,
                    repair_switch_sha256=sha(switch_path))
            self.assertEqual(result["phase"], "sca")
            self.assertIsNone(result["last_error"])
            self.assertEqual(result["boundary_repair"]["repair_commit"], repair_commit)


if __name__ == "__main__":
    unittest.main()
