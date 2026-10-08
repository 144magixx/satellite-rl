"""No-network checks for S2-P19 core archive completion and serial replay."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import zipfile

from scripts.s2p19 import archive, archive_core, ops


def fixture_jobs():
    return [{"task_id": key, "port": int(key.rsplit("_", 1)[-1]),
             "run_id": "run_" + key, "request_id": "request_" + key,
             "git_commit": "a" * 40, "kind": "final", "env_step": units,
             "budget_env_steps": units}
            for key, units in archive_core.FINAL_UNITS.items()] + [
           {"task_id": arm, "port": port, "run_id": "run_" + arm,
            "request_id": "request_" + arm, "git_commit": "b" * 40,
            "kind": "training", "env_step": step, "budget_env_steps": budget}
           for arm, (port, step, budget) in archive_core.TRAINING.items()]


def fake_detail(item, *, stage="completed"):
    status = {"run_id": item["run_id"], "stage": stage,
              "env_step": item["env_step"],
              "budget_env_steps": item["budget_env_steps"]}
    if item["kind"] == "final":
        status.update(completed_cases=item["env_step"], total_cases=item["env_step"])
    if item["task_id"] == "I0":
        status.update(completed_due_to="episode_cap", episodes=12500)
    return {"job": {"run_id": item["run_id"], "experiment_id": "S2-P19",
                    "archive_root": ops.instance(item["port"])["archive"],
                    "request_id": item["request_id"],
                    "git_commit": item["git_commit"]}, "status": status}


class CoreArchiveTests(unittest.TestCase):
    def test_legacy_smoke_port_enrichment_requires_receipt_and_zip_sha(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            volume = root / "part_0001.zip"
            with zipfile.ZipFile(volume, "w") as bundle:
                bundle.writestr("job/status.json", b"completed")
            run_id = "evaluate_smoke"
            inventory = "a" * 64
            receipt_path = root / "receipt.json"
            receipt_path.write_text(json.dumps({
                "schema": "s2p19.archive.v1", "experiment_id": "S2-P19",
                "run_id": run_id, "port": 20418, "job_stage": "completed",
                "inventory_sha256": inventory,
                "volumes": [{"name": volume.name, "sha256": archive.digest(volume)}]}),
                encoding="utf-8")
            (root / "download_receipt.json").write_text(json.dumps({
                "task_id": "validation_greedy_direct", "run_id": run_id,
                "port": 20418, "all_volume_and_member_shas_verified": True,
                "remote_receipt_sha256": archive.digest(receipt_path)}), encoding="utf-8")
            prior = {"run_id": run_id, "archive_inventory_sha256": inventory,
                     "archive_receipt_path": str(receipt_path),
                     "results_sha256": "b" * 64}
            incoming = {**prior, "port": 20418}
            index = root / "source_index.json"

            def reset():
                index.write_text(json.dumps({"schema": "s2p19.source_index.v1",
                    "jobs": {"validation_greedy_direct": prior,
                             "another_smoke": {"run_id": "other"}},
                    "training": {}}), encoding="utf-8")

            reset()
            archive.merge_index(index, "validation_greedy_direct", incoming, None)
            saved = archive_core.read_json(index)["jobs"]
            self.assertEqual(saved["validation_greedy_direct"]["port"], 20418)
            self.assertEqual(saved["validation_greedy_direct"]["archive_receipt_path"],
                             str(receipt_path))
            self.assertIn("another_smoke", saved)
            reset()
            with self.assertRaises(ValueError):
                archive.merge_index(index, "validation_greedy_direct",
                                    {**incoming, "port": 20424}, None)
            reset()
            volume.write_bytes(volume.read_bytes() + b"corrupt")
            with self.assertRaises(ValueError):
                archive.merge_index(index, "validation_greedy_direct", incoming, None)
            reset()
            with self.assertRaises(ValueError):
                archive.merge_index(index, "validation_greedy_direct",
                                    {**incoming, "archive_inventory_sha256": "c" * 64}, None)
            self.assertIsNone(archive_core.read_json(index)["jobs"]
                              ["validation_greedy_direct"].get("port"))
            reset()
            receipt_path.unlink()
            with self.assertRaises(ValueError):
                archive.merge_index(index, "validation_greedy_direct", incoming, None)
            index.write_text(json.dumps({"schema": "s2p19.source_index.v1",
                "jobs": {}, "training": {"C0": {"run_id": "run_c0",
                    "archive_inventory_sha256": inventory}}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                archive.merge_index(index, "C0", None,
                    {"run_id": "run_c0", "port": 20418,
                     "archive_inventory_sha256": inventory})

    def test_live_gate_blocks_every_copy_until_all_fifteen_complete(self):
        jobs = fixture_jobs()
        by_run = {item["run_id"]: item for item in jobs}
        calls = []

        def api(port, path):
            item = by_run[path.rsplit("/", 1)[-1]]
            calls.append((port, path))
            return fake_detail(item, stage="evaluating" if item["task_id"] ==
                               "I0_final_B_20474" else "completed")

        gate = archive_core.live_gate(jobs, api_get=api)
        self.assertEqual(len(calls), 15)
        self.assertFalse(gate["ready"])
        self.assertEqual(gate["completed"], 14)
        self.assertTrue(all(row.get("observed_utc") for row in gate["jobs"].values()))
        copier = Mock()
        with self.assertRaises(ValueError):
            archive_core.execute(jobs, gate, archive_job=copier)
        copier.assert_not_called()
        complete = archive_core.live_gate(jobs, api_get=lambda p, path:
            fake_detail(by_run[path.rsplit("/", 1)[-1]]))
        self.assertTrue(complete["ready"])
        self.assertEqual(complete["completed"], 15)

    def test_index_preserves_smoke_and_rejects_run_port_inventory_conflicts(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "source_index.json"
            path.write_text(json.dumps({"schema": "s2p19.source_index.v1",
                "jobs": {"validation_greedy_direct": {"run_id": "smoke"}},
                "training": {}}), encoding="utf-8")
            spec = {"run_id": "run_c0", "port": 20473,
                    "archive_inventory_sha256": "a" * 64}
            archive.merge_index(path, "C0_final_direct_20473", spec, None)
            saved = archive_core.read_json(path)
            self.assertEqual(saved["jobs"]["validation_greedy_direct"]["run_id"], "smoke")
            archive.merge_index(path, "C0_final_direct_20473", spec, None)
            for change in ({"run_id": "other"}, {"port": 20424},
                           {"archive_inventory_sha256": "b" * 64}):
                with self.assertRaises(ValueError):
                    archive.merge_index(path, "C0_final_direct_20473",
                                        {**spec, **change}, None)

    def test_serial_ledger_replays_verified_jobs_without_recopy(self):
        jobs = fixture_jobs()
        gate = archive_core.live_gate(jobs, api_get=lambda p, path:
            fake_detail(next(item for item in jobs if item["run_id"] ==
                             path.rsplit("/", 1)[-1])))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            index = root / "source_index.json"
            index.write_text(json.dumps({"schema": "s2p19.source_index.v1",
                "jobs": {"validation_greedy_direct": {"run_id": "smoke"}},
                "training": {}}), encoding="utf-8")
            called = []

            def copier(port, run_id, task_id, *, destination, source_index):
                called.append(task_id)
                folder = destination / str(port) / run_id
                folder.mkdir(parents=True)
                volume = folder / "part_0001.zip"
                with zipfile.ZipFile(volume, "w") as bundle:
                    bundle.writestr("job/status.json", b"completed")
                package = folder / "receipt.json"
                package.write_text(json.dumps({"port": port, "run_id": run_id,
                    "job_stage": "completed", "inventory_sha256": "a" * 64,
                    "volumes": [{"name": volume.name, "sha256": archive.digest(volume)}]}),
                    encoding="utf-8")
                local = folder / "download_receipt.json"
                local.write_text(json.dumps({"task_id": task_id, "port": port,
                    "run_id": run_id, "all_volume_and_member_shas_verified": True,
                    "local_receipt_path": str(package),
                    "remote_receipt_sha256": archive.digest(package)}), encoding="utf-8")
                spec = {"run_id": run_id, "port": port,
                        "archive_inventory_sha256": "a" * 64}
                archive.merge_index(source_index, task_id,
                                    spec if task_id not in archive_core.TRAINING else None,
                                    spec if task_id in archive_core.TRAINING else None)
                return {"receipt": str(local)}

            ledger_path = root / "ledger.json"
            result = archive_core.execute(jobs, gate, ledger_path=ledger_path,
                source_index=index, destination=root / "copies", archive_job=copier)
            self.assertEqual(len(called), 15)
            self.assertEqual(len(result["completed"]), 15)
            self.assertIn("validation_greedy_direct", archive_core.read_json(index)["jobs"])
            archive_core.execute(jobs, gate, ledger_path=ledger_path,
                source_index=index, destination=root / "copies", archive_job=Mock())

    def test_transport_failure_is_recorded_and_never_skips_gate(self):
        jobs = fixture_jobs()
        gate = archive_core.live_gate(jobs, api_get=lambda p, path:
            fake_detail(next(item for item in jobs if item["run_id"] ==
                             path.rsplit("/", 1)[-1])))
        with tempfile.TemporaryDirectory() as temp:
            ledger_path = Path(temp) / "ledger.json"
            copier = Mock(side_effect=OSError("fixture transport interrupted"))
            with self.assertRaisesRegex(OSError, "transport interrupted"):
                archive_core.execute(jobs, gate, ledger_path=ledger_path,
                                     archive_job=copier)
            saved = archive_core.read_json(ledger_path)
            self.assertEqual(saved["completed"], {})
            self.assertEqual(saved["failures"][0]["task_id"], jobs[0]["task_id"])
            self.assertEqual(copier.call_count, 1)


if __name__ == "__main__":
    unittest.main()
