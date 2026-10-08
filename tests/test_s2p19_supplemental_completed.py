"""No-network checks for completed-only supplemental S2-P19 archives."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

from scripts.s2p19 import archive, archive_core, archive_waiter, ops
from scripts.s2p19 import supplemental_completed as supplemental


def contract():
    return archive_core.read_json(supplemental.DEFAULT_CONTRACT)


def detail(item, *, stage="completed", request_id=None):
    return {"read_only": False,
            "job": {"run_id": item["run_id"], "request_id": request_id or item["request_id"],
                    "git_commit": item["git_commit"], "kind": item["kind"],
                    "experiment_id": "S2-P19",
                    "archive_root": ops.instance(item["port"])["archive"]},
            "status": {"run_id": item["run_id"], "stage": stage,
                       "env_step": item["env_step"],
                       "budget_env_steps": item["budget_env_steps"]}}


def package(root, item):
    folder = root / item["run_id"]
    folder.mkdir(parents=True)
    volume = folder / "part_0001.zip"
    with zipfile.ZipFile(volume, "w") as bundle:
        bundle.writestr("job/status.json", b"completed")
    receipt = folder / "receipt.json"
    receipt.write_text(json.dumps({"schema": "s2p19.archive.v1",
        "port": item["port"], "run_id": item["run_id"],
        "job_stage": "completed", "job_git_commit": item["git_commit"],
        "job_kind": item["kind"], "inventory_sha256": "a" * 64,
        "volumes": [{"name": volume.name, "sha256": archive.digest(volume)}]}),
        encoding="utf-8")
    download = folder / "download_receipt.json"
    download.write_text(json.dumps({"task_id": item["task_id"],
        "run_id": item["run_id"], "port": item["port"],
        "all_volume_and_member_shas_verified": True,
        "local_receipt_path": str(receipt),
        "remote_receipt_sha256": archive.digest(receipt)}), encoding="utf-8")
    return download, receipt


class SupplementalCompletedTests(unittest.TestCase):
    @unittest.skipUnless(supplemental.RAW_PLAN.is_file(), "local experiment evidence unavailable")
    def test_contract_is_rebuilt_from_frozen_plan_audits_and_identity_snapshot(self):
        saved = contract()
        self.assertEqual(saved, supplemental.derive_contract())
        self.assertEqual(len(saved["jobs"]), 16)
        self.assertEqual(len({x["run_id"] for x in saved["jobs"]}), 16)
        self.assertEqual(len({x["request_id"] for x in saved["jobs"]}), 16)
        pilots = {x["run_id"]: x for x in saved["jobs"]
                  if x["identity_basis"] == "pilot_audit+frozen_pilot_request"}
        self.assertEqual(set(pilots), set(supplemental.PILOT_REQUESTS))
        self.assertTrue(all(x["git_commit"] == supplemental.SOURCE_13EA
                            for x in pilots.values()))
        self.assertEqual(sum(x["reuse_existing"] for x in saved["jobs"]), 1)

    def test_live_get_checks_all_sixteen_exact_frozen_fields(self):
        jobs = contract()["jobs"]
        by_run = {x["run_id"]: x for x in jobs}
        calls = []

        def api(port, endpoint):
            item = by_run[endpoint.rsplit("/", 1)[-1]]
            calls.append((port, endpoint))
            if item["run_id"] == next(iter(by_run)):
                return detail(item, request_id="wrong-request")
            return detail(item)

        gate = supplemental.live_gate(jobs, api_get=api)
        self.assertEqual(len(calls), 16)
        self.assertFalse(gate["ready"])
        self.assertEqual(gate["completed"], 15)
        self.assertTrue(all(row.get("observed_utc") for row in gate["jobs"].values()))
        full = supplemental.live_gate(jobs, api_get=lambda port, endpoint:
            detail(by_run[endpoint.rsplit("/", 1)[-1]]))
        self.assertTrue(full["ready"])

    def test_metadata_capacity_gate_checks_only_fifteen_new_copies(self):
        jobs = contract()["jobs"]
        smoke = next(x for x in jobs if x["reuse_existing"])
        calls = []
        def probe(item):
            calls.append(item["run_id"])
            return {"schema": "s2p19.core_inventory.v1", "port": item["port"],
                    "run_id": item["run_id"],
                    "members": [{"member": "job/job.json", "bytes": 100}],
                    "member_count": 1, "total_input_bytes": 100,
                    "remote_statvfs_free_bytes": 100 * 1024 ** 3,
                    "observed_utc": "2026-10-08T14:24:54+00:00"}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, receipt = package(root, smoke)
            index = root / "source_index.json"
            index.write_text(json.dumps({"schema": "s2p19.source_index.v1",
                "jobs": {smoke["task_id"]: {"run_id": smoke["run_id"],
                    "archive_receipt_path": str(receipt),
                    "archive_inventory_sha256": "a" * 64}}, "training": {}}),
                encoding="utf-8")
            result = supplemental.inventory_gate(jobs, probe=probe,
                source_index=index, destination=root)
            self.assertTrue(result["ready"])
            self.assertEqual(result["existing_references"], 1)
            self.assertEqual(len(calls), 15)
            self.assertNotIn(smoke["run_id"], calls)
            self.assertEqual(result["jobs"][smoke["task_id"]]["source"],
                             "existing_verified_local_ZIP_no_new_copy")

    def test_core_incomplete_and_held_lock_refuse_supplemental_start(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            status, ledger = root / "status.json", root / "ledger.json"
            status.write_text(json.dumps({"phase": "archiving"}), encoding="utf-8")
            ledger.write_text(json.dumps({"completed": {str(i): {} for i in range(5)}}),
                              encoding="utf-8")
            gate = supplemental.core_local_gate(status_path=status, ledger_path=ledger,
                                                core_lock=root / "core.lock")
            self.assertFalse(gate["ready_snapshot"])
            status.write_text(json.dumps({"phase": "core_complete", "core_count": 15}),
                              encoding="utf-8")
            ledger.write_text(json.dumps({"completed": {str(i): {} for i in range(15)}}),
                              encoding="utf-8")
            self.assertTrue(supplemental.core_local_gate(status_path=status,
                ledger_path=ledger, core_lock=root / "core.lock")["ready_snapshot"])
            with archive_waiter.exclusive_lock(root / "core.lock"):
                with self.assertRaises(OSError):
                    with archive_waiter.exclusive_lock(root / "core.lock"):
                        pass

    def test_smoke_is_verified_and_referenced_without_new_archive(self):
        item = next(x for x in contract()["jobs"] if x["reuse_existing"])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            download, receipt = package(root, item)
            index = root / "source_index.json"
            index.write_text(json.dumps({"schema": "s2p19.source_index.v1",
                "jobs": {item["task_id"]: {"run_id": item["run_id"],
                    "archive_receipt_path": str(receipt),
                    "archive_inventory_sha256": "a" * 64}}, "training": {}}),
                encoding="utf-8")
            record = supplemental.verify_smoke(item, source_index=index)
            self.assertEqual(record["mode"], "existing_verified_no_recopy")
            self.assertEqual(record["receipt"], str(download))
            receipt.write_bytes(receipt.read_bytes() + b"corrupt")
            with self.assertRaises(ValueError):
                supplemental.verify_smoke(item, source_index=index)

    def test_serial_ledger_gate_rejects_partial_and_replays_same_receipts(self):
        frozen = contract()
        jobs = frozen["jobs"]
        gate = {"ready": True, "completed": 16, "pending": [],
                "jobs": {x["task_id"]: {"run_id": x["run_id"],
                    "port": x["port"], "reasons": []} for x in jobs}}
        inventory = {"ready": True, "pending": [],
                     "jobs": {x["task_id"]: {"run_id": x["run_id"],
                         "port": x["port"], "reasons": []} for x in jobs}}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            index = root / "source_index.json"
            smoke = next(x for x in jobs if x["reuse_existing"])
            _, smoke_receipt = package(root, smoke)
            index.write_text(json.dumps({"schema": "s2p19.source_index.v1",
                "jobs": {smoke["task_id"]: {"run_id": smoke["run_id"],
                    "archive_receipt_path": str(smoke_receipt),
                    "archive_inventory_sha256": "a" * 64}}, "training": {}}),
                encoding="utf-8")
            copier = Mock()
            with self.assertRaises(ValueError):
                supplemental.execute(frozen, {**gate, "ready": False}, inventory,
                    source_index=index, archive_job=copier)
            copier.assert_not_called()

            def fake_copy(port, run_id, task_id, *, destination, source_index,
                          part_mib, max_gib, compression_level,
                          remote_timeout_seconds):
                item = next(x for x in jobs if x["task_id"] == task_id)
                download, _ = package(root, item)
                if item["kind"] != "prepare":
                    archive.merge_index(source_index, task_id,
                        {"run_id": run_id, "port": port,
                         "archive_inventory_sha256": "a" * 64}, None)
                return {"receipt": str(download)}

            copier.side_effect = fake_copy
            ledger_path = root / "ledger.json"
            result = supplemental.execute(frozen, gate, inventory,
                source_index=index, destination=root / "copies",
                ledger_path=ledger_path, archive_job=copier)
            self.assertEqual(len(result["completed"]), 16)
            self.assertEqual(copier.call_count, 15)
            self.assertEqual(result["completed"][smoke["task_id"]]["mode"],
                             "existing_verified_no_recopy")
            supplemental.execute(frozen, gate, inventory, source_index=index,
                destination=root / "copies", ledger_path=ledger_path,
                archive_job=Mock(side_effect=AssertionError("unexpected recopy")))


if __name__ == "__main__":
    unittest.main()
