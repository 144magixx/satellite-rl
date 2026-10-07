"""The final timing switch cannot precede real training/B completion."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.s2p19 import prepare_final_timing as timing


def fixture():
    catalogs = {20418: [], 20424: []}
    details = {}
    for arm, (port, run_id, request_id, commit) in timing.ARMS.items():
        checkpoint = arm.lower() * 32
        catalogs[port].append({"run_id": run_id, "request_id": request_id})
        status = {"run_id": run_id, "stage": "completed",
                  "env_step": 380000 if arm == "I0" else 400000,
                  "episodes": 12500 if arm == "I0" else 3300,
                  "completed_due_to": "episode_cap" if arm == "I0" else None,
                  "checkpoint_sha256": checkpoint}
        details[run_id] = {"job": {"run_id": run_id, "request_id": request_id,
                                   "git_commit": commit, "kind": "train",
                                   "experiment_id": "S2-P19"}, "status": status}
        b_id = f"refine_{arm.lower()}_endpoint"
        b_request = f"s2p19-{arm.lower()}-endpoint-B"
        catalogs[port].append({"run_id": b_id, "request_id": b_request})
        details[b_id] = {"job": {"run_id": b_id, "request_id": b_request,
                                  "kind": "refine", "experiment_id": "S2-P19",
                                  "initial_source": {"run_id": run_id, "group": "model",
                                     "partition": "validation", "validation_step": status["env_step"],
                                     "source_checkpoint_sha256": checkpoint,
                                     "scene_count": 72, "case_files": [{}] * 72}},
                         "status": {"run_id": b_id, "stage": "completed",
                                    "env_step": 216, "budget_env_steps": 216}}

    def api(port, endpoint, *_args, **_kwargs):
        if endpoint == "/api/s2p19/runs":
            return {"runs": catalogs[port]}
        return details[endpoint.rsplit("/", 1)[-1]]

    return catalogs, details, api


class FinalTimingTests(unittest.TestCase):
    def test_active_training_waits_and_execute_does_nothing(self):
        catalogs, details, api = fixture()
        details[timing.ARMS["C0"][1]]["status"]["stage"] = "training"
        with TemporaryDirectory() as temporary:
            with (patch.object(timing.ops, "api", side_effect=api),
                  patch.object(timing, "SWITCH_RECEIPT", Path(temporary) / "switch.json"),
                  patch.object(timing, "_stop_controller") as stop,
                  patch.object(timing, "_stop_old_service") as stop_remote,
                  patch.object(timing, "_launch_one_worker") as launch):
                result = timing.execute()
        self.assertEqual(result["status"], "waiting")
        self.assertEqual(result["mutation_count"], 0)
        self.assertIn("C0 training has not completed", result["reasons"])
        stop.assert_not_called()
        stop_remote.assert_not_called()
        launch.assert_not_called()

    def test_unfinished_endpoint_B_waits_without_switch(self):
        catalogs, details, api = fixture()
        details["refine_c1_endpoint"]["status"]["stage"] = "running"
        with TemporaryDirectory() as temporary:
            with (patch.object(timing.ops, "api", side_effect=api),
                  patch.object(timing, "SWITCH_RECEIPT", Path(temporary) / "switch.json"),
                  patch.object(timing, "_controller_state_ready", return_value=None),
                  patch.object(timing, "_launch_one_worker") as launch):
                result = timing.execute()
        self.assertEqual(result["status"], "waiting")
        self.assertIn("C1 endpoint B has not completed", result["reasons"])
        launch.assert_not_called()

    def test_wrong_training_source_or_B_step_is_hard_block(self):
        catalogs, details, api = fixture()
        details[timing.ARMS["I0"][1]]["job"]["git_commit"] = "0" * 40
        with patch.object(timing.ops, "api", side_effect=api):
            with self.assertRaisesRegex(timing.TimingGateError, "identity/source"):
                timing.inspect()
        details[timing.ARMS["I0"][1]]["job"]["git_commit"] = timing.ARMS["I0"][3]
        details["refine_i0_endpoint"]["job"]["initial_source"]["validation_step"] = 300000
        with patch.object(timing.ops, "api", side_effect=api):
            with self.assertRaisesRegex(timing.TimingGateError, "exact final model/step"):
                timing.inspect()

    def test_existing_identical_service_is_recognized_without_new_launch(self):
        with TemporaryDirectory() as temporary:
            saved = Path(temporary) / "switch.json"
            saved.write_text(json.dumps({"new_server_pid": 123,
                                         "new_server_start_ticks": 456}), encoding="utf-8")
            observed = {"pid": 123, "start_ticks": 456, "max_workers": "1",
                        "environment": {key: "1" for key in timing.THREAD_NAMES}}
            with (patch.object(timing, "SWITCH_RECEIPT", saved),
                  patch.object(timing, "_service_process", return_value=observed),
                  patch.object(timing, "inspect") as inspect,
                  patch.object(timing, "_launch_one_worker") as launch):
                result = timing.execute()
            self.assertEqual(result["status"], "already_switched")
            self.assertEqual(result["mutation_count"], 0)
            inspect.assert_not_called()
            launch.assert_not_called()

    def test_unknown_response_adopts_exact_target_without_relaunch(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "original.json"
            service = root / "service.json"
            intent = root / "intent.json"
            switch = root / "switch.json"
            origin = {"instances": {"20418": {"service_repo": "/s2p19/20418/repo",
                                                 "new_server_pid": 11,
                                                 "new_server_start_ticks": 22}}}
            original.write_text(json.dumps(origin), encoding="utf-8")
            service.write_text(json.dumps(origin), encoding="utf-8")
            from scripts.s2p19 import controller
            intent.write_text(json.dumps({"original_receipt_sha256": controller.digest(original),
                                          "old_service": {"pid": 11},
                                          "controller": {"pid": 99}}), encoding="utf-8")
            observed = {"pid": 33, "start_ticks": 44, "max_workers": "1",
                        "cwd": "/s2p19/20418/repo", "source_commit": timing.SERVICE_COMMIT,
                        "environment": {key: "1" for key in timing.THREAD_NAMES}}
            with (patch.object(timing, "SWITCH_RECEIPT", switch),
                  patch.object(timing, "ORIGINAL_RECEIPT", original),
                  patch.object(timing, "SERVICE_RECEIPT", service),
                  patch.object(timing, "INTENT", intent),
                  patch.object(timing, "_service_process", return_value=observed),
                  patch("psutil.process_iter", return_value=[]),
                  patch.object(timing, "_update_deployment_receipt"),
                  patch.object(timing.controller, "_service_identity", return_value={}),
                  patch.object(timing.plan_builder, "capture_final_timing_isolation",
                               return_value={"path": "audit.json", "sha256": "a" * 64}),
                  patch.object(timing, "_launch_one_worker") as launch):
                result = timing._adopt_existing_switch()
            self.assertEqual(result["status"], "adopted_same_target")
            self.assertTrue(switch.is_file())
            launch.assert_not_called()

    def test_same_source_launch_snippet_compiles_without_starting_service(self):
        with TemporaryDirectory() as temporary:
            receipt = Path(temporary) / "service.json"
            receipt.write_text(json.dumps({"instances": {"20418": {
                "service_repo": "/s2p19/20418/ops/service_13ea/repo"}}}), encoding="utf-8")
            scripts = []

            def capture(code, port, timeout=120):
                self.assertEqual(port, 20418)
                compile(code, "remote_service_launcher", "exec")
                scripts.append(code)
                return {"exit_code": 0, "stdout": json.dumps({"pid": 123,
                    "start_ticks": 456, "repo": "/s2p19/20418/ops/service_13ea/repo",
                    "archive": "/archive", "service_commit": timing.SERVICE_COMMIT,
                    "max_workers": 1}) + "\n", "stderr": ""}

            with (patch.object(timing, "SERVICE_RECEIPT", receipt),
                  patch.object(timing.ops, "remote_python", side_effect=capture)):
                result = timing._launch_one_worker()
            self.assertEqual(result["pid"], 123)
            self.assertEqual(len(scripts), 1)
            self.assertIn("--max-workers','1'", scripts[0])


if __name__ == "__main__":
    unittest.main()
