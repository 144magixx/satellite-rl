"""Small package gates and immutable resume behavior; no remote connection."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.stage1_drl_r04 import delivery


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value), encoding="utf-8")


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / "repo"
        self.base = self.repo / "outputs/experiments/first_stage_s1_d12a_20261010"
        report = self.repo / delivery.REPORT.relative_to(delivery.REPO)
        write(report, b"formal report")
        final = self.base / "analysis/final_report_verified"
        for name in delivery.REPORT_FILES:
            write(final / name, b"formal figure or statistic")
        write(final / "audit.json", {
            "experiment_id": "S1-D12A", "scientific_commit": delivery.SCIENTIFIC_COMMIT,
            "formal_ready": True, "issues": [], "test_rows": 1800,
            "strict_replayed_rows": 1800,
            "method_test_counts": {key: 300 for key in
                                   ("I0", "H1", "H2", "BC", "P0", "P1")}})
        self.tasks = {f"task_{i:02d}": {"port": 20424 if i == 1 else 20418,
                                       "run_id": f"run-{i:02d}",
                                       "git_commit": delivery.SCIENTIFIC_COMMIT}
                      for i in range(37)}
        write(self.base / "run_manifest.json", {"tasks": self.tasks})
        write(self.base / "analysis/final_input/state.json", {
            "experiment_id": "S1-D12A", "scientific_commit": delivery.SCIENTIFIC_COMMIT,
            "tasks": dict(list(self.tasks.items())[:34])})
        for key, row in self.tasks.items():
            write(self.base / "job_archives" / str(row["port"]) /
                  row["run_id"] / "receipt.json", {
                      **row, "source_commit": delivery.SCIENTIFIC_COMMIT,
                      "volumes": [{"file": "raw.zip", "bytes": 999999999,
                                   "sha256": "a" * 64,
                                   "members": {"raw": {"sha256": "a" * 64}}}]})
        write(self.base / "job_archives/20418/shared_sources/receipt.json", {
            "experiment_id": "S1-D12A", "source_commit": delivery.SCIENTIFIC_COMMIT,
            "volumes": [{"file": f"source-{i}.zip", "bytes": 100,
                         "sha256": "b" * 64,
                         "members": {"source": {"sha256": "b" * 64}}}
                        for i in range(3)]})
        checks = {key: True for key in (
            "completed_budget", "source", "seed", "config_hash", "learner_hash",
            "streams_and_optimizer_rng", "final_policy", "same_checkpoint_policy",
            "finite", "worker_exited", "ledger_budget_tails.jsonl", "validations",
            "observation_shards")}
        learning = {str(port): {"passed": True, "run_id": run,
                               "checks": checks, "checkpoint_sha256": "c" * 64,
                               "ledger_counts": {"actions.jsonl": 262144,
                                                 "budget_tails.jsonl": 32}}
                    for port, run in ((20418, "run-00"), (20424, "run-01"))}
        write(self.base / "preflight/final_learning_audit.json", learning)
        exports = {}
        for key, port, run in (("P0", 20418, "run-00"),
                               ("P1", 20424, "run-01"),
                               ("BC", 20418, "run-02")):
            folder = (self.repo / "models/runs/first_stage_s1_d12a_20261010" /
                      key)
            write(folder / "policy.pt", key.encode())
            write(folder / "model.json", {"key": key})
            hashes = {name: delivery.file_sha(folder / name)
                      for name in ("policy.pt", "model.json")}
            receipt_path = (self.base / "job_archives" / str(port) /
                            run / "receipt.json")
            archived = delivery.read(receipt_path)
            archived["volumes"][0]["members"].update({
                name: {"sha256": value} for name, value in hashes.items()})
            write(receipt_path, archived)
            exports[key] = {"source": delivery.SCIENTIFIC_COMMIT,
                            "run_id": run, "port": port,
                            "env_step": 0 if key == "BC" else 262144,
                            "path": str(folder), "files_sha256": hashes}
            if key != "BC":
                learning[str(port)]["policy_sha256"] = hashes["policy.pt"]
                learning[str(port)]["model_sha256"] = hashes["model.json"]
        write(self.base / "preflight/final_learning_audit.json", learning)
        write(self.base / "preflight/model_exports.json", {"exports": exports})
        fixed = {
            "deployment/s1d12a_" + delivery.SCIENTIFIC_COMMIT + ".bundle": b"science",
            "analysis/threshold_audit.py": b"print('audit')",
            "analysis/threshold_audit.json": b"{}",
            "analysis/threshold_audit.csv": b"a,b\n",
            "analysis/threshold_audit_mothers.csv": b"a,b\n",
            "analysis/result_interpretation.md": b"memo"}
        for name, value in fixed.items():
            write(self.base / name, value)
        for name in ("report.py", "strict_audit.py"):
            write(self.repo / "scripts/stage1_drl_r04" / name, b"source")

    def test_plan_gate_and_small_immutable_package(self):
        gates = delivery.gate(self.repo, self.base)
        self.assertTrue(gates["ready"], gates["errors"])
        receipt = delivery.package(self.repo, self.base)
        self.assertFalse(receipt["raw_job_zips_embedded"])
        self.assertFalse(any(row["name"].endswith(".zip") for row in receipt["members"]))
        delivery.verify_zip(Path(receipt["zip_path"]), receipt)
        with self.assertRaises(FileExistsError):
            delivery.package(self.repo, self.base)
        with mock.patch.object(delivery.ops, "remote_python") as remote:
            write(self.base / "analysis/final_report_verified/audit.json", {"formal_ready": False})
            with self.assertRaises(RuntimeError):
                delivery.upload(Path(receipt["zip_path"]).parent /
                                "local_package_receipt.json", self.repo, self.base)
            remote.assert_not_called()

    def test_missing_receipt_and_unverified_learning_block_package(self):
        (self.base / "job_archives/20418/run-00/receipt.json").unlink()
        self.assertFalse(delivery.gate(self.repo, self.base)["ready"])
        with self.assertRaises(RuntimeError):
            delivery.package(self.repo, self.base)
        write(self.base / "job_archives/20418/run-00/receipt.json", {
            "port": 20418, "run_id": "run-00",
            "source_commit": delivery.SCIENTIFIC_COMMIT,
            "volumes": [{"file": "raw.zip", "bytes": 100,
                         "sha256": "a" * 64,
                         "members": {"raw": {"sha256": "a" * 64}}}]})
        learning_path = self.base / "preflight/final_learning_audit.json"
        learning = delivery.read(learning_path)
        learning["20418"]["checks"]["same_checkpoint_policy"] = False
        write(learning_path, learning)
        self.assertFalse(delivery.gate(self.repo, self.base)["ready"])

    def test_upload_and_resume_read_same_package_on_four_lanes(self):
        receipt = delivery.package(self.repo, self.base)
        local_receipt = (Path(receipt["zip_path"]).parent /
                         "local_package_receipt.json")
        remote_path = (delivery.ops.BASE + "/delivery/final_" +
                       receipt["package_id"] + "/" +
                       Path(receipt["zip_path"]).name)
        calls = []

        def remote(code, port, timeout=120):
            if "folder.mkdir" in code:
                exists = len(calls) > 0
                return {"exit_code": 0, "stdout": json.dumps({
                    "exists": exists,
                    "sha256": receipt["zip_sha256"] if exists else None})}
            calls.append(port)
            return {"exit_code": 0, "stdout": json.dumps({
                "port": port, "path": remote_path,
                "zip_sha256": receipt["zip_sha256"],
                "zip_bytes": receipt["zip_bytes"],
                "member_count": len(receipt["members"]),
                "all_member_shas_verified": True, "device": 7, "inode": 11})}

        with (mock.patch.object(delivery.ops, "remote_python", side_effect=remote),
              mock.patch.object(delivery.ops, "upload", return_value={
                  "exit_code": 0, "local_sha256": receipt["zip_sha256"],
                  "remote": receipt["zip_sha256"]}) as sender):
            first = delivery.upload(local_receipt, self.repo, self.base)
            second = delivery.upload(local_receipt, self.repo, self.base)
        self.assertEqual(first, second)
        self.assertEqual(calls, list(delivery.ops.SSH_PORTS) * 2)
        sender.assert_called_once()
        self.assertTrue(first["single_shared_physical_copy"])


if __name__ == "__main__":
    unittest.main()
