"""Final S1-V10 delivery refuses partial science and large raw ZIP copies."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.stage1_variable import final_delivery


class FinalDeliveryTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
