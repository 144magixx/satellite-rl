"""Final S1-V10 delivery refuses partial science and large raw ZIP copies."""
from __future__ import annotations

import json
import base64
import hashlib
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.stage1_variable import final_delivery


class FinalDeliveryTests(unittest.TestCase):
    def test_remote_receipt_retry_ignores_timestamp_but_not_member_identity(self):
        base = {"schema": "s1v10.final_delivery_remote.v1", "package_id": "abc",
                "shared_remote_path": "/home/x/delivery/final_abc/package.zip",
                "zip_sha256": "a" * 64, "zip_bytes": 123,
                "member_count": 7, "single_shared_physical_copy": True,
                "raw_job_zips_embedded": False,
                "four_lane_access": {str(port): {
                    "remote_path": "/home/x/delivery/final_abc/package.zip",
                    "zip_sha256": "a" * 64, "zip_bytes": 123,
                    "member_count": 7, "all_member_shas_verified": True,
                    "verified_utc": "yesterday"} for port in (20418, 20424, 20473, 20474)}}
        later = deepcopy(base)
        later["verified_utc"] = "today"
        later["four_lane_access"]["20418"]["verified_utc"] = "today"
        self.assertEqual(final_delivery._remote_receipt_identity(base),
                         final_delivery._remote_receipt_identity(later))
        later["four_lane_access"]["20418"]["zip_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "lane receipt differs"):
            final_delivery._remote_receipt_identity(later)

    def test_formal_gate_binds_32_archives_strict_report_and_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            report = base / "docs/report.md"
            report.parent.mkdir()
            report.write_text("formal verified report\n" * 100, encoding="utf-8")
            final = base / "analysis/final_report"
            (final / "science_audit").mkdir(parents=True)
            publication = base / "analysis/publication_figures/final"
            publication.mkdir(parents=True)
            script = base / "analysis/publication_figures/build_figures.py"
            script.write_text("# frozen plot script\n", encoding="utf-8")
            (final / "per_scene.csv").write_text("scene_id,value\na,1\n", encoding="utf-8")
            (final / "plots.json").write_text("{}", encoding="utf-8")
            (final / "summary.json").write_text(json.dumps({
                "formal_ready": True, "issues": [], "task_count": 28,
                "audited_task_count": 28}), encoding="utf-8")
            (final / "science_audit/science_audit.json").write_text(json.dumps({
                "formal_ready": True, "static_rows": 3600,
                "temporal_rows": 720}), encoding="utf-8")
            (publication / "figure.png").write_bytes(b"frozen figure")
            manifest = base / "synced_jobs/manifest.json"
            manifest.parent.mkdir()
            science = {f"task_{index}": {} for index in range(28)}
            tasks = {**science, **{f"prepare_{port}": {}
                                 for port in (20418, 20424, 20473, 20474)}}
            manifest.write_text(json.dumps({"tasks": {key: {"archive_receipt": "receipt"}
                                                     for key in tasks},
                                            "shared_source_dependencies_closed": True}),
                                encoding="utf-8")
            controller = base / "controller/state.json"
            controller.parent.mkdir()
            controller.write_text(json.dumps({"tasks": science}), encoding="utf-8")
            acceptance = base / "preflight/acceptance.json"
            acceptance.parent.mkdir()
            (base / "preflight/final_learning_audit.json").write_text(json.dumps({
                "all_checks_passed": True, "issues": []}), encoding="utf-8")
            acceptance.write_text(json.dumps({
                "passed": True, "package_count": 32, "science_packages": 28,
                "preparation_packages": 4, "manifest_sha256": final_delivery.sha(manifest),
                "all_job_zip_and_member_shas_reverified": True,
                "all_shared_source_member_shas_reverified": True}), encoding="utf-8")
            root_check = base / "preflight/root.json"
            root_check.write_text(json.dumps({
                "acceptance_sha256": final_delivery.sha(acceptance),
                "acceptance_manifest_matches_actual_and_report_snapshot": True,
                "package_count": 32}), encoding="utf-8")
            gate = base / "analysis/gate.json"
            gate.write_text(json.dumps({
                "stage": "complete", "formal_ready": True,
                "source_commit": final_delivery.REPORT_COMMIT,
                "manifest_sha256": final_delivery.sha(manifest),
                "summary_sha256": final_delivery.sha(final / "summary.json"),
                "per_scene_sha256": final_delivery.sha(final / "per_scene.csv"),
                "plots_sha256": final_delivery.sha(final / "plots.json"),
                "science_audit_sha256": final_delivery.sha(
                    final / "science_audit/science_audit.json")}), encoding="utf-8")
            publication_receipt = publication / "publication_receipt.json"
            publication_receipt.write_text(json.dumps({
                "formal_ready_input": True, "test_rows": 3600,
                "sequence_rows": 720, "static_sequence_rows_compared": 4320,
                "input_per_scene_sha256": final_delivery.sha(final / "per_scene.csv"),
                "input_summary_sha256": final_delivery.sha(final / "summary.json"),
                "script_sha256": final_delivery.sha(script),
                "outputs": {"figure.png": {"path": str(publication / "figure.png"),
                                            "sha256": final_delivery.sha(publication / "figure.png")}}
            }), encoding="utf-8")
            with patch.object(final_delivery, "BASE", base), \
                 patch.object(final_delivery, "FINAL_REPORT", final), \
                 patch.object(final_delivery, "GATE", gate), \
                 patch.object(final_delivery, "ACCEPTANCE", acceptance), \
                 patch.object(final_delivery, "ROOT_VERIFY", root_check), \
                 patch.object(final_delivery, "MANIFEST", manifest):
                self.assertTrue(final_delivery.gate_state(report, publication)["ready"])
                changed = json.loads(acceptance.read_text(encoding="utf-8"))
                changed["manifest_sha256"] = "0" * 64
                acceptance.write_text(json.dumps(changed), encoding="utf-8")
                self.assertFalse(final_delivery.gate_state(report, publication)["ready"])

    def test_partial_report_gate_fails_before_package_or_upload(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            with patch.object(final_delivery, "REPORT_MD", base / "missing_report.md"), \
                 patch.object(final_delivery, "FINAL_REPORT", base / "missing_strict"), \
                 patch.object(final_delivery, "PUBLICATION", base / "missing_publication"), \
                 patch.object(final_delivery, "GATE", base / "missing_gate.json"), \
                 patch.object(final_delivery, "ACCEPTANCE", base / "missing_acceptance.json"), \
                 patch.object(final_delivery, "ROOT_VERIFY", base / "missing_root.json"), \
                 patch.object(final_delivery, "MANIFEST", base / "missing_manifest.json"), \
                 patch.object(final_delivery, "DELIVERY", base / "delivery"):
                gate = final_delivery.gate_state(final_delivery.REPORT_MD,
                                                 final_delivery.PUBLICATION)
                self.assertFalse(gate["ready"])
                self.assertIn("report_markdown absent", " ".join(gate["errors"]))
                with self.assertRaisesRegex(RuntimeError, "formal delivery gates incomplete"):
                    final_delivery.package(final_delivery.REPORT_MD,
                                           final_delivery.PUBLICATION)
                self.assertFalse((base / "delivery").exists())

    def test_small_zip_contains_hash_inventory_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            report = base / "report.md"
            report.write_text("formal evidence\n" * 120, encoding="utf-8")
            figure = base / "figure.png"
            figure.write_bytes(b"PNG fixture")
            publication = base / "publication"
            publication.mkdir()
            (publication / "publication_receipt.json").write_text("{}", encoding="utf-8")
            manifest = base / "manifest.json"
            manifest.write_text('{"tasks":{}}', encoding="utf-8")
            acceptance = base / "acceptance.json"
            acceptance.write_text("{}", encoding="utf-8")
            gate = base / "gate.json"
            gate.write_text("{}", encoding="utf-8")
            with patch.object(final_delivery, "DELIVERY", base / "delivery"), \
                 patch.object(final_delivery, "MANIFEST", manifest), \
                 patch.object(final_delivery, "ACCEPTANCE", acceptance), \
                 patch.object(final_delivery, "GATE", gate), \
                 patch.object(final_delivery, "gate_state",
                              return_value={"ready": True, "errors": [],
                                            "manifest_sha256": final_delivery.sha(manifest)}), \
                 patch.object(final_delivery, "_archive_index",
                              return_value={"schema": "index", "tasks": {}}), \
                 patch.object(final_delivery, "_git_index",
                              return_value={"schema": "source_index"}), \
                 patch.object(final_delivery, "_members",
                              return_value={"report/report.md": report,
                                            "analysis/publication_final/figure.png": figure}):
                first = final_delivery.package(report, publication)
                second = final_delivery.package(report, publication)
            self.assertEqual(first["zip_sha256"], second["zip_sha256"])
            self.assertEqual(first["member_count"], 3)
            with zipfile.ZipFile(first["zip_path"]) as bundle:
                self.assertIn("indexes/delivery_inventory.json", bundle.namelist())
                inventory = json.loads(bundle.read("indexes/delivery_inventory.json"))
                self.assertEqual(inventory["package_id"], first["package_id"])
                self.assertNotIn("raw_job.zip", bundle.namelist())
            wrong = {"report/report.md": {"bytes": report.stat().st_size,
                                          "sha256": "0" * 64}}
            with self.assertRaisesRegex(ValueError, "member list differs"):
                final_delivery._verify_zip(Path(first["zip_path"]), wrong,
                                           first["zip_sha256"])

    def test_raw_zip_and_symlink_are_not_delivery_members(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            raw = base / "raw_job.zip"
            raw.write_bytes(b"not a small report input")
            with self.assertRaisesRegex(ValueError, "refuses raw"):
                final_delivery._file(raw)

    def test_resume_upload_uses_existing_local_receipt_without_packaging(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            publication = base / "publication"
            publication.mkdir()
            (publication / "publication_receipt.json").write_bytes(b"publication")
            report = base / "report.md"
            report.write_bytes(b"formal report")
            acceptance, gate = base / "acceptance.json", base / "gate.json"
            acceptance.write_bytes(b"acceptance")
            gate.write_bytes(b"gate")
            package = base / "delivery/final_abc"
            package.mkdir(parents=True)
            inventory = package / "inventory.json"
            inventory.write_text('{"archive_manifest_sha256":"' + "a" * 64 + '"}',
                                 encoding="utf-8")
            receipt = {"schema": "s1v10.final_delivery_local.v1",
                       "package_id": "abc", "zip_path": str(package / "package.zip"),
                       "inventory_path": str(inventory),
                       "archive_acceptance_sha256": final_delivery.sha(acceptance),
                       "report_gate_sha256": final_delivery.sha(gate),
                       "publication_receipt_sha256": final_delivery.sha(
                           publication / "publication_receipt.json"),
                       "formal_report_sha256": final_delivery.sha(report)}
            receipt_path = package / "local_package_receipt.json"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with patch.object(final_delivery, "DELIVERY", base / "delivery"), \
                 patch.object(final_delivery, "ACCEPTANCE", acceptance), \
                 patch.object(final_delivery, "GATE", gate), \
                 patch.object(final_delivery, "gate_state", return_value={
                     "ready": True, "errors": [], "manifest_sha256": "a" * 64}), \
                 patch.object(final_delivery, "package", side_effect=AssertionError(
                     "resume must not repackage")), \
                 patch.object(final_delivery, "upload", return_value=(receipt_path,
                     {"resumed": True})) as uploader:
                _, result = final_delivery.resume_upload(receipt_path, report,
                                                         publication)
                self.assertTrue(result["resumed"])
                uploader.assert_called_once_with(receipt)
                receipt["formal_report_sha256"] = "0" * 64
                receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "no longer binds"):
                    final_delivery.resume_upload(receipt_path, report, publication)

    def test_upload_reuses_same_identity_remote_receipt_with_old_timestamp(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            report = base / "report.md"
            report.write_text("formal evidence\n" * 120, encoding="utf-8")
            publication = base / "publication"
            publication.mkdir()
            (publication / "publication_receipt.json").write_text("{}", encoding="utf-8")
            manifest = base / "manifest.json"
            manifest.write_text('{"tasks":{}}', encoding="utf-8")
            acceptance = base / "acceptance.json"
            acceptance.write_text("{}", encoding="utf-8")
            gate = base / "gate.json"
            gate.write_text("{}", encoding="utf-8")
            with patch.object(final_delivery, "DELIVERY", base / "delivery"), \
                 patch.object(final_delivery, "MANIFEST", manifest), \
                 patch.object(final_delivery, "ACCEPTANCE", acceptance), \
                 patch.object(final_delivery, "GATE", gate), \
                 patch.object(final_delivery, "gate_state", return_value={
                     "ready": True, "errors": [],
                     "manifest_sha256": final_delivery.sha(manifest)}), \
                 patch.object(final_delivery, "_archive_index",
                              return_value={"schema": "index", "tasks": {}}), \
                 patch.object(final_delivery, "_git_index",
                              return_value={"schema": "sources"}), \
                 patch.object(final_delivery, "_members",
                              return_value={"report/report.md": report}):
                local = final_delivery.package(report, publication)
                remote_zip = (final_delivery.ops.BASE + "/delivery/final_" +
                              local["package_id"] + "/" + Path(local["zip_path"]).name)
                lanes = {str(port): {"port": port, "remote_path": remote_zip,
                                     "zip_sha256": local["zip_sha256"],
                                     "zip_bytes": local["zip_bytes"],
                                     "member_count": local["member_count"],
                                     "all_member_shas_verified": True,
                                     "verified_utc": "today", "device": 1, "inode": 2}
                         for port in final_delivery.ops.SSH_PORTS}
                saved = {"schema": "s1v10.final_delivery_remote.v1",
                         "package_id": local["package_id"],
                         "shared_remote_path": remote_zip,
                         "zip_sha256": local["zip_sha256"],
                         "zip_bytes": local["zip_bytes"],
                         "member_count": local["member_count"],
                         "single_shared_physical_copy": True,
                         "same_device_inode_observed": True,
                         "four_lane_access": deepcopy(lanes),
                         "raw_job_zips_embedded": False,
                         "verified_utc": "yesterday"}
                for row in saved["four_lane_access"].values():
                    row["verified_utc"] = "yesterday"
                remote_bytes = (json.dumps(saved, ensure_ascii=False) + "\n").encode("utf-8")
                remote_sha = hashlib.sha256(remote_bytes).hexdigest()
                observations = iter([
                    {"exists": True, "sha256": local["zip_sha256"]},
                    {"exists": True, "sha256": remote_sha,
                     "base64": base64.b64encode(remote_bytes).decode("ascii")},
                    *({"sha256": remote_sha} for _ in final_delivery.ops.SSH_PORTS)])
                def remote(*args, **kwargs):
                    return {"exit_code": 0, "stderr": "",
                            "stdout": json.dumps(next(observations)) + "\n"}
                with patch.object(final_delivery.ops, "remote_python", side_effect=remote), \
                     patch.object(final_delivery, "_remote_validate",
                                  side_effect=lambda port, *args: lanes[str(port)]), \
                     patch.object(final_delivery.ops, "upload") as uploader:
                    path, result = final_delivery.upload(local)
                uploader.assert_not_called()
                self.assertTrue(result["all_four_lanes_readback_verified"])
                self.assertEqual(Path(path).parent.joinpath(
                    "remote_delivery_receipt.json").read_bytes(), remote_bytes)
                self.assertTrue(Path(path).parent.joinpath(
                    "remote_delivery_receipt_previous.json").is_file())

    def test_learning_evidence_is_explicit_and_excludes_draft_or_raw_logs(self):
        required = {"scripts/build_report08_text.py",
                    "evidence/report08_text_receipt.json",
                    "diagnostics/final_learning_diagnostics.md",
                    "diagnostics/final_learning_diagnostics_R08.json",
                    "diagnostics/final_learning_diagnostics_R09.json",
                    "diagnostics/final_learning_validation_pair_R08.json",
                    "diagnostics/final_learning_validation_pair_R09.json",
                    "diagnostics/reward_objective_audit.md",
                    "scripts/final_learning_actions_remote.py",
                    "scripts/final_learning_stream_remote.py",
                    "evidence/final_learning_audit.json",
                    "evidence/final_learning_audit_api.json",
                    "evidence/final_learning_audit_manifest.json"}
        self.assertTrue(required <= set(final_delivery.LEARNING_EVIDENCE))
        self.assertFalse(any("draft" in name or name.endswith((".zip", ".jsonl"))
                             for name in final_delivery.LEARNING_EVIDENCE))


if __name__ == "__main__":
    unittest.main()
