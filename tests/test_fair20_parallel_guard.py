"""Pure executor/lease/readonly source checks; no resource probes or science."""
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from satellite_rl.workbench.fair20_guard import (executor_port, release_reasons, admission_reasons, OLD_API_PORTS,
                                              _idle_completed_service_matches)
from satellite_rl.workbench.fair20_source_repair import readonly_source_path, ORIGIN_COMMIT, FORMAL_EXECUTION_COMMIT
from satellite_rl.workbench.storage import digest


class ParallelGuardTests(unittest.TestCase):
    def test_actual_executor_config_and_old_api_mapping(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict("os.environ", {}, clear=True):
            group = Path(temporary)/"s2_f20"
            for port in (20418, 20474):
                self.assertEqual(executor_port(group/str(port)/"archive"), port)
            with self.assertRaises(ValueError):
                executor_port(group/"20418"/"archive", 20474)
            with self.assertRaises(ValueError):
                executor_port(group/"20424"/"archive", 20424)
            self.assertEqual(OLD_API_PORTS, {20418: 8940, 20474: 8943})

    def test_readonly_peer_exact_science_only_and_group_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            group = Path(temporary)/"s2_f20"
            own, peer = group/"20418", group/"20474"
            for revision in (ORIGIN_COMMIT, FORMAL_EXECUTION_COMMIT):
                self.assertEqual(readonly_source_path(own, peer/"repo", revision), (peer/"repo").resolve())
            for path, revision in ((group/"20424"/"repo", ORIGIN_COMMIT),
                                   (Path(temporary)/"s1_nd12"/"20474"/"repo", ORIGIN_COMMIT),
                                   (peer/"repo", "b474"), (own/"server_repo", "R3control")):
                with self.assertRaises(ValueError):
                    readonly_source_path(own, path, revision)

    def test_completed223_release_no_fabricated_exclusion_and_reused_pid_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def pin(name, value):
                path = root/name
                path.write_text(json.dumps(value), encoding="utf-8")
                return {"path": str(path), "sha256": digest(path)}
            identity = {"pid": 111, "expected_birth": 123., "same_identity_alive": False, "pid_exists": False}
            receipt = {"released": True, "released_ssh_port": 20418,
                "release_basis": "nd12_scientific_complete_and_persistent_followups_stopped",
                "scientific_completed_tasks": 223, "scientific_completion_verified": True,
                "completion_evidence": pin("completion.json", {"experiment_id": "S1-ND12", "scientific_complete": True,
                    "scientific_completed_tasks": 223, "active_jobs": [], "pending_jobs": []}),
                "persistent_coordinator_stopped_verified": True,
                "coordinator_evidence": pin("coordinator.json", {"stopped": True, "future_launch_disabled": True, "identities": [identity]}),
                "heartbeat_stopped_verified": True,
                "heartbeat_evidence": pin("heartbeat.json", {"stopped": True, "active": False, "automation_id": "nd12"}),
                "active_jobs": [], "pending_jobs": [], "future_lane_reserved": False}
            self.assertEqual(release_reasons(receipt, 20418), [])
            self.assertTrue(release_reasons(receipt, 20474))
            identity["pid_exists"] = True
            receipt["coordinator_evidence"] = pin("coordinator2.json", {"stopped": True, "future_launch_disabled": True, "identities": [identity]})
            self.assertTrue(release_reasons(receipt, 20418))
            identity.update(expected_birth=1791760304.581377, actual_birth=1791782565.114439,
                            current_command="python unrelated_application.py")
            receipt["coordinator_evidence"] = pin("reused_distinct.json", {"stopped": True, "future_launch_disabled": True, "identities": [identity]})
            self.assertEqual(release_reasons(receipt, 20418), [])
            for actual, command in ((identity["expected_birth"], "python unrelated_application.py"),
                                     (None, "python unrelated_application.py"),
                                     (float("nan"), "python unrelated_application.py"),
                                     (1791782565.114439, "python s1_nd12 controller.py"),
                                     (1791782565.114439, "")):
                identity.update(actual_birth=actual, current_command=command)
                receipt["coordinator_evidence"] = pin("reused_invalid.json", {"stopped": True, "future_launch_disabled": True, "identities": [identity]})
                self.assertTrue(release_reasons(receipt, 20418))

    def test_exact_idle_completed_dispatch_capable_service_identity_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def pin(name, value):
                path = root/name
                path.write_text(json.dumps(value), encoding="utf-8")
                return {"path": str(path), "sha256": digest(path)}
            receipt = {
                "completion_evidence": pin("complete.json", {"experiment_id": "S1-ND12", "scientific_complete": True,
                    "scientific_completed_tasks": 223, "active_jobs": [], "pending_jobs": []}),
                "coordinator_evidence": pin("coordinator.json", {"stopped": True, "future_launch_disabled": True,
                    "identities": [{"pid": 111, "expected_birth": 123., "same_identity_alive": False, "pid_exists": False}]}),
                "heartbeat_evidence": pin("heartbeat.json", {"stopped": True, "active": False, "automation_id": "nd12"})}
            source = "/home/u2024110224/first_stage_satellite_rl/s1_nd12/20418/ops/datafeed_service_252e05a9cc02.py"
            command = "/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python -B "+source
            source_sha = "9c8d1f212108bfdd645907e953590cfccc5dbf17a08e6c1d8020067bcd44bc71"
            evidence = {"port": 20418, "pid": 32504, "birth_ticks": 835345015, "exact_command": command,
                "service_source_path": source, "service_source_sha256": source_sha,
                "role": "idle_completed_nd12_workbench_service", "independent_idle_service_audit_verified": True,
                "audit_source_reference": pin("audit.json", {"source_sha256": source_sha,
                    "service_has_scheduler_capability": True, "scientific_group_complete": True,
                    "pending_jobs": [], "future_coordinator_stopped": True, "heartbeat_paused": True})}
            receipt["inert_service_evidence"] = pin("inert.json", evidence)
            process = {"pid": 32504, "birth_ticks": 835345015, "command": command+" "}
            real_digest = digest
            def pinned_source(path):
                return source_sha if str(path) == source else real_digest(path)
            old_tasks = [{"stage": "completed", "status": "已完成"}]
            with patch("satellite_rl.workbench.storage.digest", side_effect=pinned_source):
                self.assertTrue(_idle_completed_service_matches(process, receipt, 20418, old_tasks))
                self.assertFalse(_idle_completed_service_matches(process, receipt, 20474, old_tasks))
                self.assertFalse(_idle_completed_service_matches(process, receipt, 20418, [{"stage": "queued"}]))
                self.assertFalse(_idle_completed_service_matches(process, receipt, 20418, {"unavailable": "error"}))
                for key, value in (("pid", 32505), ("birth_ticks", 835345016), ("command", command+" --dispatch")):
                    changed = {**process, key: value}
                    self.assertFalse(_idle_completed_service_matches(changed, receipt, 20418, old_tasks))
                evidence["service_source_sha256"] = "b"*64
                receipt["inert_service_evidence"] = pin("inert_bad.json", evidence)
                self.assertFalse(_idle_completed_service_matches(process, receipt, 20418, old_tasks))

    def test_live_admission_real_port_uuid_temperature_and_processes(self):
        release = {"port": 20418, "executor_port": 20418, "scientific_commit": ORIGIN_COMMIT,
                   "old_lane_all_work_terminal": True, "old_future_training_reserved": False, "gpu_uuid": "GPU-one"}
        observed = {"observed_utc": datetime.now(timezone.utc).isoformat(), "port": 20418, "executor_port": 20418,
            "gpu": "GPU-one, RTX4090D, 0, 24000, 0, 40", "gpu_query_exit": 0, "compute_query_exit": 0,
            "compute_apps": "", "disk_free": 20*1024**3, "memory": {"MemAvailable": 16*1024**3},
            "cpu_affinity": [0], "cgroup": {"attempted": True}, "old_tasks": [], "processes": []}
        self.assertEqual(admission_reasons(observed, release, ORIGIN_COMMIT, executor_port=20418), [])
        observed["old_tasks"] = [{"status": "已完成", "stage": "completed"}]
        self.assertEqual(admission_reasons(observed, release, ORIGIN_COMMIT, executor_port=20418), [])
        observed["old_tasks"] = [{"status": "已完成", "stage": "running"}]
        self.assertTrue(admission_reasons(observed, release, ORIGIN_COMMIT, executor_port=20418))
        observed["old_tasks"] = []
        self.assertTrue(admission_reasons(observed, release, ORIGIN_COMMIT, executor_port=20474))
        observed["processes"] = [{"pid": 111, "birth_ticks": 999, "command": "python nondrl12 controller"}]
        self.assertTrue(admission_reasons(observed, release, ORIGIN_COMMIT, executor_port=20418))


if __name__ == "__main__":
    unittest.main()
