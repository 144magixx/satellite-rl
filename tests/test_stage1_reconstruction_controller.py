"""Controller ordering and idempotence without SSH or scientific execution."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.stage1_reconstruction.controller import Attention, Controller, GATES, S2_SCA_JOBS, load_plan
from scripts.stage1_reconstruction import remote


SHA = "a" * 64
TARGET = "u2024110224@10.160.17.102:"


def plan(root):
    return {"science_commit": "a" * 40,
            "config_template": {"schema_version": 4, "research_stage": "coverage",
                                "algorithm": "ppo", "name": "S1-03",
                                "coverage": {"initialization": "cluster", "method": "none",
                                             "reconstruction": {"route": "prepare"}},
                                "training": {"seed": 51001, "total_steps": 262144,
                                             "mechanism_gate_step": 65536}},
            "ports": [20418, 20424],
            "hostroot": "/home/u2024110224/first_stage_satellite_rl/s1_03",
            "local_output_root": str(root),
            "raw_inputs": {"train": {"path": "/raw/train", "sha256": SHA,
                                      "dates": ["2024-01-08", "2024-01-09"]},
                           "validation": {"path": "/raw/validation", "sha256": SHA,
                                          "dates": ["2024-01-12"]},
                           "test": {"path": "/raw/test"}},
            "test_prepare_port": 20418,
            "gates": deepcopy(GATES), "audit_commands": {},
            "s2_release": {"controller_manifest": "/s2/controller_manifest.json"}}


class MemoryController(Controller):
    def __init__(self, root):
        super().__init__(plan(root), SHA, root / "state.json")
        self.datasets = {20418: [], 20424: []}
        self.jobs = {20418: [], 20424: []}
        self.models = []
        self.imported_starts = []
        self.release = False
        self.test_ready = False
        self.post_count = {}

    def code(self, *_args, **_kwargs):
        return {"active": []}

    def _s2_released(self):
        if self.release:
            self.state["s2_release"] = {"mode": "verified_terminal"}
        return self.release

    def _verify_raw_npz(self, role, port):
        self.state["observed"][f"input_source_{port}:{role}"] = {"input_source_sha256": SHA}

    def _test_source_ready(self):
        if self.test_ready:
            self.state["test_release"] = {"record_sha256": SHA, "scene_count": 100}
        return self.test_ready

    def _run_audit(self, mode, port, checkpoint_sha=None):
        if mode == "development":
            self.assert_development_jobs()
        if mode == "formal":
            self.assert_formal_jobs()
        proof = {"mode": mode, "passed": True, "audit_path": "/audit/" + mode,
                 "audit_sha256": SHA}
        self.state["audits"][mode] = proof
        return proof

    def assert_development_jobs(self):
        assert all(f"dev_{route}" in self.state["jobs"] for route in ("R0", "R1", "R2", "R3"))

    def assert_formal_jobs(self):
        assert all(f"formal_{route}" in self.state["jobs"] for route in ("R0", "R1", "R2", "R3", "P1"))

    def api(self, port, endpoint, body=None, **kwargs):
        if endpoint == "system" and body is None:
            return {"device": "fixture GPU", "gpu": {}, "python": "/dqn", "torch": "fixture",
                    "cuda_build": "fixture", "server_time": "fixture UTC"}
        if endpoint == "coverage/datasets" and body is None:
            return {"datasets": self.datasets[port]}
        if endpoint == "coverage/datasets/import":
            role = body["partition_role"]
            row = {"id": f"dataset_{port}_{role}", "name": body["name"],
                   "source_path": body["path"], "partition_role": role,
                   "research_stage": "coverage", "initialization": "cluster",
                   "sha256": SHA, "count": {"train": 576, "validation": 20, "test": 100}[role]}
            self.datasets[port].append(row)
            return row
        if endpoint == "coverage/reconstruction/import-dataset":
            row = {"id": "imported_test", "source_dataset_id": body["source_dataset_id"],
                   "source_target": TARGET + "20418", "partition_role": "test",
                   "sha256": SHA, "count": 100}
            self.datasets[port].append(row)
            return row
        if endpoint == "coverage/reconstruction/imports" and body is None:
            return {"starts": self.imported_starts, "models": self.models}
        if endpoint == "coverage/reconstruction/import-start":
            row = {"id": "imported_start_test", "source_run_id": body["source_run_id"],
                   "source_target": TARGET + "20418", "partition_role": "test",
                   "sha256": SHA, "dataset_sha256": SHA}
            self.imported_starts.append(row)
            return row
        if endpoint == "coverage/runs" and body is None:
            return {"runs": self.jobs[port]}
        if endpoint == "coverage/runs" and body is not None:
            label = body["config"]["name"].split()[-1]
            self.post_count[label] = self.post_count.get(label, 0) + 1
            job = {"run_id": "job_" + label, "kind": body["kind"],
                   "request_id": body["request_id"], "git_commit": "a" * 40,
                   "execution_target": TARGET + str(port),
                   "reconstruction_route": body["config"]["coverage"]["reconstruction"]["route"],
                   "stage": "completed"}
            self.jobs[port].append(job)
            return job
        if endpoint.startswith("runs/") and body is None:
            run_id = endpoint.split("/")[1]
            job = next(row for row in self.jobs[port] if row["run_id"] == run_id)
            return {"job": job, "status": {"stage": job["stage"], "heartbeat_at": "verified"},
                    "artifacts": {"reconstruction_starts_sha256": SHA}}
        if endpoint == "coverage/models" and body is None:
            return {"models": self.models}
        if endpoint == "coverage/reconstruction/import-model":
            row = {"id": "imported_P1", "source_run_id": body["source_run_id"],
                   "source_target": TARGET + "20418", "sha256": SHA}
            self.models.append(row)
            return row
        raise AssertionError((port, endpoint, body))


class ControllerTests(unittest.TestCase):
    def test_test_contract_sha_is_distinct_from_selected_times_sha(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = plan(Path(temporary))
            source["raw_inputs"]["test"].update(path="/raw/test", record_path="/raw/test/record.json",
                                                manifest_path="/raw/test/manifest.json",
                                                selection_contract_path="/raw/test/selection_contract.json",
                                                selection_sha256="b" * 64,
                                                selected_times_sha256="c" * 64)
            controller = Controller(source, SHA, Path(temporary) / "state.json")
            observed = {"ready": True, "record_sha256": SHA, "manifest_sha256": SHA,
                        "selection_contract_sha256": "b" * 64,
                        "selected_times_sha256": "c" * 64,
                        "scene_files_verified": True,
                        "record": {"path": "/raw/test", "count": 100},
                        "manifest": {"status": "prepared_not_imported_or_evaluated",
                                     "count": 100, "staged_record_sha256": SHA}}
            controller.code = lambda *_args, **_kwargs: observed
            self.assertTrue(controller._test_source_ready())
            other = Controller(source, SHA, Path(temporary) / "other.json")
            other.code = lambda *_args, **_kwargs: {**observed,
                "selection_contract_sha256": "c" * 64}
            with self.assertRaisesRegex(Attention, "选择SHA"):
                other._test_source_ready()

    def test_original_train_npz_sha_role_and_dates_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = Controller(plan(Path(temporary)), SHA, Path(temporary) / "state.json")
            controller.code = lambda *_args, **_kwargs: {
                "ready": True, "input_source_sha256": SHA, "record_sha256": SHA,
                "record": {"partition_role": "train", "dates": ["2024-01-08", "2024-01-09"]}}
            controller._verify_raw_npz("train", 20418)
            self.assertEqual(controller.state["observed"]["input_source_20418:train"]["input_source_sha256"], SHA)
            other = Controller(plan(Path(temporary)), SHA, Path(temporary) / "other.json")
            other.code = lambda *_args, **_kwargs: {
                "ready": True, "input_source_sha256": SHA, "record_sha256": SHA,
                "record": {"partition_role": "validation", "dates": ["2024-01-08"]}}
            with self.assertRaisesRegex(Attention, "用途/日期"):
                other._verify_raw_npz("train", 20418)

    def test_mechanism_gate_waits_for_same_worker_and_rejects_pid_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = Controller(plan(Path(temporary)), SHA, Path(temporary) / "state.json")
            controller.state["jobs"]["train_P1"] = "train_P1"
            controller.code = lambda *_args, **_kwargs: {"sha256": SHA}
            controller._run_audit = lambda *_args, **_kwargs: {
                "audit_path": "/archive/ops/audits/mechanism_audit.json", "audit_sha256": SHA}
            requests = []
            controller.api = lambda port, endpoint, body=None, **kwargs: requests.append((port, endpoint))
            detail = {"status": {"stage": "stopped", "env_step": 65536,
                                  "checkpoint_sha256": SHA}}
            controller._worker_process = lambda *_args: {"state": "same_process_alive", "pid": 123}
            self.assertFalse(controller._mechanism_gate("train_P1", detail))
            self.assertFalse(requests)
            controller._worker_process = lambda *_args: {"state": "pid_reused", "pid": 123}
            with self.assertRaisesRegex(Attention, "PID"):
                controller._mechanism_gate("train_P1", detail)
            self.assertFalse(requests)
            controller._worker_process = lambda *_args: {"state": "exited", "pid": 123}
            self.assertFalse(controller._mechanism_gate("train_P1", detail))
            self.assertEqual([name for _, name in requests],
                             ["coverage/runs/train_P1/mechanism-gate", "runs/train_P1/resume"])

    def test_plan_freezes_gate_values_and_audit_script_sha(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = plan(root)
            source["raw_inputs"]["test"].update(record_path="/raw/test/record.json",
                                                manifest_path="/raw/test/manifest.json",
                                                selection_contract_path="/raw/test/selection_contract.json",
                                                selection_sha256=SHA, selected_times_sha256=SHA)
            source["audit_commands"] = {mode: {"argv": [remote.DQN, "/audit.py", "--context", "{audit_context_path}"],
                                               "script_sha256": SHA}
                                        for mode in ("development", "mechanism", "formal")}
            path = root / "plan.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            loaded, digest = load_plan(path)
            self.assertEqual(loaded["gates"], GATES)
            self.assertEqual(len(digest), 64)
            source["gates"]["mechanism_step"] += 1
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "门槛"):
                load_plan(path)
            source["gates"] = deepcopy(GATES)
            source["s2_release"]["allow_file"] = "/unverified/allow.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "手工文件旁路"):
                load_plan(path)

    def test_audit_running_waits_and_completed_file_must_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = plan(Path(temporary))
            source["audit_commands"] = {"development": {"argv": ["/dqn", "/audit.py", "--context", "{audit_context_path}"],
                                                         "script_sha256": SHA}}
            controller = Controller(source, SHA, Path(temporary) / "state.json")
            controller.state["jobs"].update({f"dev_{route}": f"job_{route}"
                                             for route in ("R0", "R1", "R2", "R3")})
            controller._write_audit_context = lambda *_args: None
            controller._audit_process = lambda *_args: {"state": "running", "launch": {"pid": 123}}
            self.assertIsNone(controller._run_audit("development", 20418))
            self.assertNotIn("development", controller.state["audits"])
            controller._audit_process = lambda mode, port, argv, path, context_sha, script_sha: {
                "state": "completed", "exit": {"returncode": 0,
                                               "context_sha256": context_sha,
                                               "script_sha256": script_sha,
                                               "preflight_path": "/audit/development_preflight.json",
                                               "preflight_sha256": SHA}}
            frozen_root = controller.archive[20418] + "/outputs/workbench/source_snapshots/" + "a" * 40
            controller._read_remote_json = lambda _port, path: (
                {"exists": True, "sha256": SHA,
                 "value": {"source_root": frozen_root, "git_commit": "a" * 40,
                           "ssh_port": 20418, "gpu": "RTX 4090 D",
                           "package_file": frozen_root + "/src/satellite_rl/__init__.py"}}
                if path.endswith("_preflight.json") else
                {"exists": True, "sha256": SHA,
                 "value": {"passed": True, "mode": "development", "plan_sha256": SHA}})
            self.assertTrue(controller._run_audit("development", 20418)["passed"])

    def test_two_lanes_wait_for_test_release_and_submit_each_job_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = MemoryController(Path(temporary))
            controller.tick()
            self.assertNotIn("train_P1", controller.state["jobs"])
            self.assertFalse(any(row["reconstruction_route"] == "P1" for row in controller.jobs[20418]))
            controller.test_ready = True
            controller.tick()
            self.assertIn("train_P1", controller.state["jobs"])
            self.assertEqual(controller.state["observed"]["prepare_test"]["stage"], "completed")
            self.assertFalse(controller.jobs[20424])
            controller.release = True
            controller.tick()
            self.assertEqual(controller.state["phase"], "awaiting_final_report")
            self.assertEqual(len(controller.jobs[20418]), 8)  # 3 prepare + 4 development + P1 train
            self.assertEqual(len(controller.jobs[20424]), 5)  # 4 baselines + P1 eval
            self.assertEqual(controller.state["imports"]["test_start_20424"], "imported_start_test")
            self.assertEqual(controller.post_count["train_P1"], 1)
            self.assertEqual(controller.post_count["formal_P1"], 1)
            controller.tick()
            self.assertEqual(controller.post_count["train_P1"], 1)
            controller.state["jobs"].pop("dev_R1")
            self.assertTrue(controller._job("dev_R1", 20418, "R1", "evaluate", "validation"))
            self.assertEqual(controller.post_count["dev_R1"], 1)

    def test_s2_release_requires_terminal_manifest_and_no_active_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = Controller(plan(Path(temporary)), SHA, Path(temporary) / "state.json")
            completed = {key: {"run_id": key, "manifest_stage": "completed",
                               "status_stage": "completed", "status_path": "/s2/" + key,
                               "identity_ok": True}
                         for key in S2_SCA_JOBS}
            snapshots = [
                {"ready": True, "phase": "sca", "jobs": completed, "live": [], "api_error": None},
                {"ready": True, "phase": "awaiting_report_and_archive", "jobs": completed,
                 "live": [{"run_id": "other", "stage": "evaluating"}], "api_error": None},
                {"ready": True, "phase": "awaiting_report_and_archive", "jobs": completed,
                 "live": [], "api_error": None},
            ]
            controller.code = lambda *_args, **_kwargs: snapshots.pop(0)
            self.assertFalse(controller._s2_released())
            self.assertFalse(controller._s2_released())
            self.assertTrue(controller._s2_released())
            self.assertEqual(controller.state["s2_release"]["mode"],
                             "S2_controller_terminal_and_live_API")

    def test_s2_completed_status_with_wrong_job_identity_does_not_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = Controller(plan(Path(temporary)), SHA, Path(temporary) / "state.json")
            completed = {key: {"run_id": key, "manifest_stage": "completed",
                               "status_stage": "completed", "identity_ok": True}
                         for key in S2_SCA_JOBS}
            completed["H0_sca"]["identity_ok"] = False
            controller.code = lambda *_args, **_kwargs: {
                "ready": True, "phase": "awaiting_report_and_archive", "jobs": completed,
                "identity_errors": ["H0_sca: status/job run_id differs"],
                "live": [], "api_error": None}
            self.assertFalse(controller._s2_released())
            self.assertIn("identity mismatch", controller.state["observed"]["s2_release"]["blocked_reason"])

    def test_s2_generated_probe_rejects_cross_job_status_path_and_run_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "20424"
            manifest_path = base / "ops" / "controller_manifest.json"
            manifest_path.parent.mkdir(parents=True)
            archive = base / "archive"
            jobs = {}
            for key in S2_SCA_JOBS:
                job_dir = archive / "outputs" / "workbench" / "jobs" / key
                job_dir.mkdir(parents=True)
                (job_dir / "status.json").write_text(json.dumps({"run_id": key, "stage": "completed"}))
                (job_dir / "job.json").write_text(json.dumps({"run_id": key, "request_id": key + "_request"}))
                jobs[key] = {"run_id": key, "request_id": key + "_request",
                             "stage": "completed", "status_path": str(job_dir / "status.json")}
            manifest = {"phase": "awaiting_report_and_archive", "experiment_id": "S2-P17",
                        "target": {"archive": str(archive), "ssh_port": 20424,
                                   "workbench_port": 8781}, "jobs": jobs}
            source = plan(Path(temporary))
            source["s2_release"]["controller_manifest"] = str(manifest_path)
            controller = Controller(source, SHA, Path(temporary) / "state.json")

            def execute_probe(_port, code, **_kwargs):
                output = io.StringIO()
                with redirect_stdout(output):
                    try:
                        exec(compile(code, "s2_probe", "exec"), {})
                    except SystemExit as exit_code:
                        self.assertEqual(exit_code.code, 0)
                return json.loads(output.getvalue())

            controller.code = execute_probe
            jobs["H0_sca"]["status_path"] = jobs["V0_sca"]["status_path"]
            manifest_path.write_text(json.dumps(manifest))
            self.assertFalse(controller._s2_released())
            self.assertTrue(controller.state["observed"]["s2_release"]["identity_errors"])
            jobs["H0_sca"]["status_path"] = str(archive / "outputs/workbench/jobs/H0_sca/status.json")
            (archive / "outputs/workbench/jobs/H0_sca/status.json").write_text(json.dumps({
                "run_id": "wrong_job", "stage": "completed"}))
            manifest_path.write_text(json.dumps(manifest))
            self.assertFalse(controller._s2_released())
            self.assertTrue(controller.state["observed"]["s2_release"]["identity_errors"])

    def test_s2_terminal_identity_with_8781_timeout_waits_without_attention(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "20424"
            manifest_path = base / "ops" / "controller_manifest.json"
            manifest_path.parent.mkdir(parents=True)
            archive = base / "archive"
            jobs = {}
            for key in S2_SCA_JOBS:
                job_dir = archive / "outputs" / "workbench" / "jobs" / key
                job_dir.mkdir(parents=True)
                (job_dir / "status.json").write_text(json.dumps({"run_id": key, "stage": "completed"}))
                (job_dir / "job.json").write_text(json.dumps({"run_id": key, "request_id": key + "_request"}))
                jobs[key] = {"run_id": key, "request_id": key + "_request",
                             "stage": "completed", "status_path": str(job_dir / "status.json")}
            manifest_path.write_text(json.dumps({
                "phase": "awaiting_report_and_archive", "experiment_id": "S2-P17",
                "target": {"archive": str(archive), "ssh_port": 20424, "workbench_port": 8781},
                "jobs": jobs}))
            source = plan(Path(temporary))
            source["s2_release"]["controller_manifest"] = str(manifest_path)
            controller = Controller(source, SHA, Path(temporary) / "state.json")

            def execute_probe(_port, code, **_kwargs):
                output = io.StringIO()
                with mock.patch("urllib.request.urlopen", side_effect=TimeoutError("8781 timed out")):
                    with redirect_stdout(output):
                        exec(compile(code, "s2_probe", "exec"), {})
                return json.loads(output.getvalue())

            controller.code = execute_probe
            self.assertFalse(controller._s2_released())
            observed = controller.state["observed"]["s2_release"]
            self.assertEqual(observed["identity_errors"], [])
            self.assertIn("timed out", observed["api_error"])
            self.assertIn("keep 20424 queued", observed["blocked_reason"])
            self.assertIsNone(controller.state["attention"])

    def test_unavailable_s2_dependency_keeps_peer_queued_without_halting_primary(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = Controller(plan(Path(temporary)), SHA, Path(temporary) / "state.json")
            controller.code = lambda *_args, **_kwargs: {
                "ready": True, "phase": "awaiting_report_and_archive",
                "jobs": {}, "live": [], "api_error": "timed out"}
            self.assertFalse(controller._s2_released())
            self.assertIsNone(controller.state["attention"])
            self.assertIsNone(controller.state["s2_release"])
            self.assertIn("20418 continues", controller.state["observed"]["s2_release"]["blocked_reason"])
            def unavailable(*_args, **_kwargs):
                raise Attention("20424 unavailable")
            controller.code = unavailable
            self.assertFalse(controller._s2_released())
            self.assertIsNone(controller.state["attention"])
            self.assertIsNone(controller.state["s2_release"])

    def test_busy_s2_generated_probe_does_not_query_api(self):
        import contextlib
        import io
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "s2.json"
            manifest.write_text(json.dumps({"phase": "sca", "jobs": {}}), encoding="utf-8")
            source = plan(root)
            source["s2_release"]["controller_manifest"] = str(manifest)
            controller = Controller(source, SHA, root / "state.json")
            def execute_probe(_port, code):
                output = io.StringIO()
                with contextlib.redirect_stdout(output), patch("urllib.request.urlopen", side_effect=AssertionError("busy S2 must not query API")):
                    try:
                        exec(code, {})
                    except SystemExit as error:
                        self.assertEqual(error.code, 0)
                return json.loads(output.getvalue())
            controller.code = execute_probe
            self.assertFalse(controller._s2_released())
            self.assertEqual(controller.state["observed"]["s2_release"]["phase"], "sca")


if __name__ == "__main__":
    unittest.main()
