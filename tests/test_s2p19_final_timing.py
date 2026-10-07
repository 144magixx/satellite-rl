"""S2-P19 final matrix source pins and single-instance timing gate."""

from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.s2p19 import controller, plan_builder


RUN = "train_sac_20261007_153047_1faf6d10"


def lane(port):
    return {"observed_utc": datetime.now(timezone.utc).isoformat(),
            "server": [{"pid": port, "start_ticks": 123456,
                        "max_workers": "1" if port == 20418 else "2",
                        "thread_environment": {key: "1" for key in (
                            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                            "NUMEXPR_NUM_THREADS")}}],
            "active_jobs": [], "gpu_uuids": [f"GPU-{port}"],
            "external_gpu_pids": []}


class FinalTimingTests(unittest.TestCase):
    def test_each_timed_dispatch_rechecks_pinned_idle_service(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "timing.json"
            observations = {str(port): lane(port) for port in (20418, 20424)}
            path.write_text(json.dumps({"passed": True, "target_port": 20418,
                "observations": observations}), encoding="utf-8")
            plan = {"timing_isolation_receipt": {"path": str(path),
                                                "sha256": controller.digest(path)}}
            with patch.object(plan_builder, "_timing_lane_observation",
                              side_effect=lambda port: observations[str(port)]):
                controller._final_timing_dispatch_gate(plan)
            busy = json.loads(json.dumps(observations))
            busy["20418"]["external_gpu_pids"] = [991]
            with patch.object(plan_builder, "_timing_lane_observation",
                              side_effect=lambda port: busy[str(port)]):
                with self.assertRaises(RuntimeError):
                    controller._final_timing_dispatch_gate(plan)
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                controller._final_timing_dispatch_gate(plan)

    def test_retained_completion_is_read_from_workbench_not_only_state(self):
        receipts = {"C1_pilot_20424": {"state": "completed", "run_id": RUN}}
        valid = {"job": {"run_id": RUN, "experiment_id": "S2-P19",
                         "git_commit": "a" * 40, "model_dir": "/frozen/model"},
                 "status": {"stage": "completed", "env_step": 400000}}
        with patch.object(plan_builder.ops, "api", return_value=valid) as api:
            observed = plan_builder._training_completion_observation(["C1"], receipts)
        self.assertEqual(observed["C1"]["env_step"], 400000)
        api.assert_called_once_with(20424, f"/api/runs/{RUN}")
        invalid = json.loads(json.dumps(valid))
        invalid["status"]["stage"] = "training"
        with patch.object(plan_builder.ops, "api", return_value=invalid):
            with self.assertRaises(RuntimeError):
                plan_builder._training_completion_observation(["C1"], receipts)

    def test_capture_writes_only_current_observed_idle_lane(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            state = root / "controller/state.json"
            state.parent.mkdir(parents=True)
            state.write_text(json.dumps({"tasks": {
                "C0_pilot_20418": {"state": "completed", "run_id": "train_c0"}}}),
                encoding="utf-8")
            record = root / "controller/gates/final_timing_isolation_receipt.json"
            with (patch.object(plan_builder.ops, "LOCAL", root),
                  patch.object(plan_builder.controller, "STATE", state),
                  patch.object(plan_builder.controller, "PLAN", root / "controller/plan.json"),
                  patch.object(plan_builder, "_training_completion_observation",
                               return_value={"C0": {"run_id": "train_c0", "env_step": 400000}}),
                  patch.object(plan_builder, "_timing_lane_observation", side_effect=lane)):
                result = plan_builder.capture_final_timing_isolation(["C0"])
            saved = json.loads(record.read_text(encoding="utf-8"))
            self.assertEqual(result["path"], str(record))
            self.assertEqual(saved["retained_runs"], {"C0": "train_c0"})
            record.unlink()
            busy = lane(20418)
            busy["external_gpu_pids"] = [9876]
            with (patch.object(plan_builder.ops, "LOCAL", root),
                  patch.object(plan_builder.controller, "STATE", state),
                  patch.object(plan_builder.controller, "PLAN", root / "controller/plan.json"),
                  patch.object(plan_builder, "_training_completion_observation",
                               return_value={"C0": {"run_id": "train_c0", "env_step": 400000}}),
                  patch.object(plan_builder, "_timing_lane_observation",
                               side_effect=lambda port: busy if port == 20418 else lane(port))):
                with self.assertRaises(RuntimeError):
                    plan_builder.capture_final_timing_isolation(["C0"])
            self.assertFalse(record.exists())

    def test_missing_stale_busy_or_two_worker_receipt_blocks_append_gate(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            receipts = {"C1_pilot_20424": {"run_id": RUN}}
            with patch.object(plan_builder.ops, "LOCAL", root):
                with self.assertRaises(RuntimeError):
                    plan_builder._require_final_timing_receipt(["C1"], receipts)
                path = root / "controller/gates/final_timing_isolation_receipt.json"
                path.parent.mkdir(parents=True)
                observations = {str(port): lane(port) for port in (20418, 20424)}
                record = {"format": "s2p19_final_timing_isolation_v1",
                          "experiment_id": "S2-P19", "passed": True,
                          "target_port": 20418, "retained_runs": {"C1": RUN},
                          "training_completion": {"C1": {"run_id": RUN, "env_step": 400000}},
                          "observed_utc": datetime.now(timezone.utc).isoformat(),
                          "observations": observations}
                path.write_text(json.dumps(record), encoding="utf-8")
                with (patch.object(plan_builder, "_timing_lane_observation",
                                   side_effect=lambda port: observations[str(port)]),
                      patch.object(plan_builder, "_training_completion_observation",
                                   return_value=record["training_completion"])):
                    self.assertEqual(plan_builder._require_final_timing_receipt(
                        ["C1"], receipts)["path"], str(path))
                    bad = json.loads(json.dumps(record))
                    bad["observations"]["20418"]["server"][0]["max_workers"] = "2"
                    path.write_text(json.dumps(bad), encoding="utf-8")
                    with self.assertRaises(RuntimeError):
                        plan_builder._require_final_timing_receipt(["C1"], receipts)
                    bad = json.loads(json.dumps(record))
                    bad["observations"]["20424"]["active_jobs"] = [{
                        "run_id": "other", "stage": "training"}]
                    path.write_text(json.dumps(bad), encoding="utf-8")
                    with self.assertRaises(RuntimeError):
                        plan_builder._require_final_timing_receipt(["C1"], receipts)
                    bad = json.loads(json.dumps(record))
                    bad["retained_runs"] = {"C1": "other"}
                    path.write_text(json.dumps(bad), encoding="utf-8")
                    with self.assertRaises(RuntimeError):
                        plan_builder._require_final_timing_receipt(["C1"], receipts)

    def test_c1_source_requires_exact_20424_final_and_both_file_hashes(self):
        archive = plan_builder.ops.instance(20424)["archive"]
        source = f"{archive}/models/runs/workbench/{RUN}/final"
        model = {"id": "model_c1", "path": source,
                 "format": "satellite_rl_workbench_policy_v1",
                 "provenance": {"run_id": RUN, "env_step": 400000}}
        observed = {"files": {"policy.pt": "1" * 64, "model.json": "2" * 64},
                    "format": "satellite_rl_workbench_policy_v1",
                    "provenance": model["provenance"], "config_algorithm": "sac"}
        with (patch.object(plan_builder.ops, "api", return_value={"models": [model]}),
              patch.object(plan_builder.ops, "remote_python", return_value={
                  "exit_code": 0, "stdout": json.dumps(observed), "stderr": ""})):
            pinned = plan_builder._pin_c1_final_model(RUN)
            self.assertEqual(pinned["policy.pt"], "1" * 64)
            self.assertEqual(pinned["model.json"], "2" * 64)
        for changed in ({"path": source.replace("/20424/", "/20418/")},
                        {"provenance": {"run_id": RUN, "env_step": 100000}}):
            invalid = {**model, **changed}
            with patch.object(plan_builder.ops, "api", return_value={"models": [invalid]}):
                with self.assertRaises(RuntimeError):
                    plan_builder._pin_c1_final_model(RUN)

    def test_imported_c1_copy_requires_same_model_json(self):
        archive = controller.ops.instance(20418)["archive"]
        source = (plan_builder.ops.instance(20424)["archive"] +
                  f"/models/runs/workbench/{RUN}/final")
        task = {"id": "import_C1_final_model_20418", "port": 20418,
                "source_path": source, "expected_sha256": "1" * 64,
                "expected_model_json_sha256": "2" * 64,
                "expected_provenance_run_id": RUN}
        model = {"id": "model_copy", "source_path": source,
                 "path": f"{archive}/outputs/workbench/models/model_copy",
                 "sha256": "1" * 64,
                 "provenance": {"run_id": RUN, "env_step": 400000}}
        with patch.object(controller.ops, "remote_python", return_value={
                "exit_code": 0, "stdout": json.dumps({"model_json_sha256": "2" * 64}),
                "stderr": ""}):
            controller._verify_imported_model(task, model)
        with patch.object(controller.ops, "remote_python", return_value={
                "exit_code": 0, "stdout": json.dumps({"model_json_sha256": "3" * 64}),
                "stderr": ""}):
            with self.assertRaises(RuntimeError):
                controller._verify_imported_model(task, model)


if __name__ == "__main__":
    unittest.main()
