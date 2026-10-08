"""Local-only fixtures for the explicit S2-P19 archive waiter."""

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.s2p19 import archive, archive_core, archive_waiter


LEGACY = tuple(list(archive_core.FINAL_UNITS)[:5])


def inputs():
    active_tasks = []
    state_tasks = {}
    old_tasks = []
    completed_legacy = {}
    for task_id in archive_core.FINAL_UNITS:
        port = int(task_id.rsplit("_", 1)[-1])
        task = {"id": task_id, "port": port, "request_id": "request_" + task_id,
                "expected_commit": "a" * 40}
        if task_id in LEGACY:
            old_tasks.append(task)
            completed_legacy[task_id] = "run_" + task_id
        else:
            active_tasks.append(task)
            state_tasks[task_id] = {"state": "completed", "run_id": "run_" + task_id,
                                    "request_id": task["request_id"]}
    active = {"schema": "s2p19.four_lane.v1", "experiment_id": "S2-P19",
              "final_mothers": 40, "final_scenes": 880,
              "complete_B_budget": 14960,
              "completed_legacy": completed_legacy, "tasks": active_tasks}
    historical = {"experiment_id": "S2-P19", "tasks": old_tasks}
    manifest = {"experiment_id": "S2-P19", "tasks": {
        arm: {"port": port, "run_id": "run_" + arm,
              "request_id": "request_" + arm, "git_commit": "b" * 40,
              "env_step": step, "budget_env_steps": budget}
        for arm, (port, step, budget) in archive_core.TRAINING.items()}}
    state = {"plan_sha256": "a" * 64, "tasks": state_tasks}
    return active, state, historical, manifest


def fake_inventory(item, *, members=None, remote_free=100 * 1024 ** 3):
    members = members if members is not None else [
        {"member": "job/job.json", "bytes": 100},
        {"member": "source/source_manifest.json", "bytes": 200}]
    return {"schema": "s2p19.core_inventory.v1", "port": item["port"],
            "run_id": item["run_id"], "members": members,
            "member_count": len(members),
            "total_input_bytes": sum(row["bytes"] for row in members),
            "remote_statvfs_free_bytes": remote_free,
            "remote_statvfs_scope": "filesystem, not quota",
            "observed_utc": "2026-10-08T00:00:00+00:00"}


class ArchiveWaiterTests(unittest.TestCase):
    def test_local_state_waits_for_all_seven_before_any_remote_gate(self):
        active, state, historical, manifest = inputs()
        task_id = active["tasks"][-1]["id"]
        state["tasks"][task_id]["state"] = "evaluating"
        jobs, gate = archive_waiter.local_gate(active, state, historical,
            manifest, plan_sha256=state["plan_sha256"])
        self.assertEqual(len(jobs), 15)
        self.assertFalse(gate["ready"])
        self.assertEqual(gate["pending"], [task_id])
        state["tasks"][task_id]["state"] = "completed"
        _, ready = archive_waiter.local_gate(active, state, historical,
            manifest, plan_sha256=state["plan_sha256"])
        self.assertTrue(ready["ready"])
        self.assertEqual(ready["legacy_bound"], 5)

    def test_inventory_exact_member_bounds_volume_count_and_space(self):
        active, state, historical, manifest = inputs()
        jobs, _ = archive_waiter.local_gate(active, state, historical,
            manifest, plan_sha256=state["plan_sha256"])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            okay = archive_waiter.inventory_gate(jobs, probe=fake_inventory,
                                                 destination=root)
            self.assertTrue(okay["ready"])
            self.assertEqual(okay["jobs"][jobs[0]["task_id"]]["predicted_volume_count"], 1)
            self.assertTrue(okay["remote_statvfs_is_not_quota"])
            huge = [{"member": "job/results.json", "bytes": 257 * 1024 ** 3}]
            bound = archive_waiter.inventory_gate(jobs,
                probe=lambda item: fake_inventory(item, members=huge),
                destination=root)
            self.assertFalse(bound["ready"])
            self.assertTrue(any("archive input/file/volume bound" in x
                                for x in bound["pending"]))
            many = [{"member": f"job/cases/{i}.json", "bytes": 1024 ** 3}
                    for i in range(201)]
            bound = archive_waiter.inventory_gate(jobs,
                probe=lambda item: fake_inventory(item, members=many),
                destination=root)
            self.assertFalse(bound["ready"])
            self.assertEqual(bound["jobs"][jobs[0]["task_id"]]
                             ["predicted_volume_count"], 201)
            zero_then_large = [{"member": "job/empty", "bytes": 0},
                               {"member": "job/large", "bytes": 2 * 1024 ** 3}]
            split = archive_waiter.inventory_gate(jobs,
                probe=lambda item: fake_inventory(item, members=zero_then_large),
                destination=root)
            self.assertEqual(split["jobs"][jobs[0]["task_id"]]
                             ["predicted_volume_count"], 2)
            with patch.object(archive_waiter.shutil, "disk_usage",
                              return_value=type("Usage", (), {"free": 7 * 1024 ** 3})()):
                low = archive_waiter.inventory_gate(jobs, probe=fake_inventory,
                                                    destination=root)
            self.assertFalse(low["ready"])
            self.assertIn("local compressed ZIP reserve unavailable", low["pending"])

    def test_nonblocking_waiter_lock_refuses_second_owner(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "waiter.lock"
            with archive_waiter.exclusive_lock(path):
                with self.assertRaises(OSError):
                    with archive_waiter.exclusive_lock(path):
                        pass
            with archive_waiter.exclusive_lock(path):
                pass

    def test_waiter_persists_attention_without_remote_when_local_pending(self):
        active, state, historical, manifest = inputs()
        state["tasks"][active["tasks"][0]["id"]]["state"] = "evaluating"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = {}
            for name, value in (("plan", active), ("historical", historical),
                                ("manifest", manifest)):
                path = root / (name + ".json")
                path.write_text(json.dumps(value), encoding="utf-8")
                files[name] = path
            state["plan_sha256"] = archive.digest(files["plan"])
            files["state"] = root / "state.json"
            files["state"].write_text(json.dumps(state), encoding="utf-8")
            with patch.object(archive_waiter, "source_commit", return_value="c" * 40), \
                    patch.object(archive_core, "live_gate", side_effect=AssertionError):
                result = archive_waiter.run_waiter(
                    plan_path=files["plan"], state_path=files["state"],
                    historical_path=files["historical"], manifest_path=files["manifest"],
                    status_path=root / "status.json", lock_path=root / "lock",
                    inventory_dir=root / "inventory", ledger_path=root / "ledger",
                    source_index=root / "source_index", destination=root / "copies",
                    deadline_utc="2020-01-01T00:00:00+00:00", poll_seconds=5)
            self.assertEqual(result["phase"], "attention")
            self.assertEqual(result["source_commit"], "c" * 40)
            self.assertEqual(result["plan_sha256"], state["plan_sha256"])
            self.assertEqual(result["supplemental_archive"], "pending_separate")
            self.assertEqual(result["formal_report"], "pending_separate")
            self.assertTrue((root / "status.json").is_file())

    def test_ready_waiter_runs_gate_inventory_then_core_with_pinned_options(self):
        active, state, historical, manifest = inputs()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = {}
            for name, value in (("plan", active), ("historical", historical),
                                ("manifest", manifest)):
                path = root / (name + ".json")
                path.write_text(json.dumps(value), encoding="utf-8")
                paths[name] = path
            state["plan_sha256"] = archive.digest(paths["plan"])
            paths["state"] = root / "state.json"
            paths["state"].write_text(json.dumps(state), encoding="utf-8")
            fake_gate = {"ready": True, "completed": 15, "pending": [], "jobs": {}}
            fake_inventory_result = {"ready": True, "jobs": {
                task_id: {"member_count": 2, "total_input_bytes": 300}
                for task_id in archive_core.FINAL_UNITS},
                "pending": [], "remote_statvfs_is_not_quota": True}
            with patch.object(archive_waiter, "source_commit", return_value="d" * 40), \
                    patch.object(archive_core, "live_gate", return_value=fake_gate) as live, \
                    patch.object(archive_waiter, "inventory_gate",
                                 return_value=fake_inventory_result) as inventory, \
                    patch.object(archive_core, "execute",
                                 return_value={"completed": {str(i): {} for i in range(15)}}) as execute:
                result = archive_waiter.run_waiter(
                    plan_path=paths["plan"], state_path=paths["state"],
                    historical_path=paths["historical"], manifest_path=paths["manifest"],
                    status_path=root / "status.json", lock_path=root / "lock",
                    inventory_dir=root / "inventory", ledger_path=root / "ledger",
                    source_index=root / "source_index", destination=root / "copies",
                    deadline_utc="2020-01-01T00:00:00+00:00", poll_seconds=5)
            self.assertEqual(result["phase"], "core_complete")
            self.assertEqual(result["core_count"], 15)
            self.assertEqual(set(result["phase_times"]),
                             {"waiting", "gating", "inventory", "archiving", "core_complete"})
            live.assert_called_once()
            inventory.assert_called_once()
            execute.assert_called_once()
            self.assertEqual(execute.call_args.kwargs["archive_options"],
                {"part_mib": 1024, "max_gib": 256, "compression_level": 6,
                 "remote_timeout_seconds": 43200})
            self.assertIsInstance(execute.call_args.kwargs["deadline_utc"], datetime)


if __name__ == "__main__":
    unittest.main()
