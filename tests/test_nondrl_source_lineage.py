"""Byte preservation and same-ID repair evidence; no remote tasks."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from satellite_rl.coverage.nondrl.runner import RowStore
from satellite_rl.coverage.nondrl.source_lineage import FORMAT, REPAIR_ID, ORIGIN_COMMIT, file_sha
from satellite_rl.coverage.nondrl.teachers import EvidenceWriter, generate_teacher
from test_nondrl_learning import fixture


def provenance(commit):
    return {"experiment_id": "S1-ND12", "run_id": "same-id", "git_commit": commit,
            "source_sha256": ("a" if commit == ORIGIN_COMMIT else "b") * 64,
            "config_sha256": "c" * 64, "dataset_sha256": {"development": "d" * 64}}


def contract(root, paths):
    config = root / "config.json"
    if not config.exists():
        config.write_text('{"note":"原冻结配置"}', encoding="utf-8")
    artifacts = [{"path": str(config), "sha256": file_sha(config), "kind": "config"}]
    artifacts += [{"path": str(p), "sha256": file_sha(p), "kind": "teacher_manifest"} for p in paths]
    return {"format": FORMAT, "repair_id": REPAIR_ID, "origin_commit": ORIGIN_COMMIT,
            "execution_commit": "f" * 40, "contract_sha256": "e" * 64,
            "run_id": "same-id", "config_path": str(config), "config_sha256": file_sha(config),
            "origin_provenance": provenance(ORIGIN_COMMIT), "artifacts": artifacts,
            "reason": "normal timeout recovery", "comparison_scope": "version-stratified evidence"}


class SourceLineageTests(unittest.TestCase):
    def test_scene_prefix_bytes_and_origin_survive_two_resumes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); old = RowStore(root / "raw", provenance(ORIGIN_COMMIT))
            old.append("0", {"result": {"valid": True}, "label": "旧行"})
            raw = (old.root / old.manifest["rows"][0]["file"]).read_bytes()
            c = contract(root, [old.path]); initial_manifest = old.path.read_bytes()
            new = RowStore(old.root, provenance("f" * 40), resume=True, repair_contract=c)
            new.append("1", {"result": {"valid": True}})
            again = RowStore(old.root, provenance("f" * 40), resume=True, repair_contract=c)
            self.assertEqual(again.manifest["provenance"], provenance(ORIGIN_COMMIT))
            self.assertEqual((old.root / old.manifest["rows"][0]["file"]).read_bytes(), raw)
            seg = again.manifest["execution_segments"][0]
            self.assertEqual((old.root / seg["origin_manifest_path"]).read_bytes(), initial_manifest)
            self.assertEqual(again.load(again.manifest["rows"][1])["execution_source"]["git_commit"], "f" * 40)
            self.assertEqual(len(again.manifest["execution_segments"]), 1)

    def test_uncontracted_change_and_changed_configuration_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); old = RowStore(root / "raw", provenance(ORIGIN_COMMIT)); old.append("0", {})
            with self.assertRaisesRegex(ValueError, "without source repair"):
                RowStore(old.root, provenance("f" * 40), resume=True)
            c = contract(root, [old.path]); (root / "config.json").write_text('{}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "configuration bytes"):
                RowStore(old.root, provenance("f" * 40), resume=True, repair_contract=c)

    def test_teacher_new_worker_and_sample_bytes_have_real_execution_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); target = root / "teacher"
            old = EvidenceWriter(target, "01", provenance(ORIGIN_COMMIT)); old.append({"visible": [1]}); old.flush()
            raw = (target / old.manifest["shards"][0]["path"]).read_bytes()
            c = contract(root, [target / "manifest.json"])
            new = EvidenceWriter(target, "01", provenance("f" * 40), resume=True, repair_contract=c)
            new.append({"visible": [-1]}); new.flush()
            child = EvidenceWriter(target / "workers" / "problem_000043", "01", provenance("f" * 40), repair_contract=c)
            child.append({"visible": [0]}); child.flush()
            self.assertTrue(child.manifest["execution_segments"][0]["created_after_repair"])
            self.assertEqual((target / old.manifest["shards"][0]["path"]).read_bytes(), raw)
            samples = gzip.decompress((child.folder / child.manifest["shards"][0]["path"]).read_bytes()).decode("utf-8")
            self.assertEqual(json.loads(samples)["execution_source"]["git_commit"], "f" * 40)
            again = EvidenceWriter(child.folder, "01", provenance("f" * 40), resume=True, repair_contract=c)
            self.assertEqual(again.rows, 1)

    def test_changed_committed_origin_prefix_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); old = RowStore(root / "raw", provenance(ORIGIN_COMMIT)); old.append("0", {})
            c = contract(root, [old.path]); new = RowStore(old.root, provenance("f" * 40), resume=True, repair_contract=c)
            new.append("1", {})
            value = copy.deepcopy(new.manifest); value["rows"] = value["rows"][1:]
            old.path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "committed origin prefix"):
                RowStore(old.root, provenance("f" * 40), resume=True, repair_contract=c)

    def test_parallel_resume_continues_into_a_new_worker_with_new_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); target = root / "teacher"
            limits = {"workers": 2, "max_trials": 10, "max_seconds": 10., "max_steps": 1, "max_problems": 3}
            old = generate_teacher([fixture()], "02", target, limits, provenance=provenance(ORIGIN_COMMIT))
            original_shards = copy.deepcopy(old["shards"])
            raw = [(target / r["path"]).read_bytes() for r in original_shards]
            pins = list(target.rglob("manifest.json"))
            c = contract(root, pins)
            new = generate_teacher([fixture() for _ in range(3)], "02", target, limits,
                    provenance=provenance("f" * 40), resume=True, repair_contract=c)
            self.assertEqual(new["problems"], 3)
            self.assertEqual(new["shards"][:len(original_shards)], original_shards)
            self.assertEqual([(target / r["path"]).read_bytes() for r in original_shards], raw)
            child = json.loads((target / "workers/problem_000001/manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(child["provenance"], provenance("f" * 40))
            self.assertTrue(child["execution_segments"][0]["created_after_repair"])
            self.assertEqual(new["provenance"], provenance(ORIGIN_COMMIT))

    def test_completed_truncated_probe_only_adds_unknown_I0_without_trials(self):
        for route in ("02", "03"):
            with self.subTest(route=route), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); target = root / "teacher"
                limits = {"workers": 1, "max_trials": 1, "max_seconds": 10., "max_steps": 1, "max_problems": 3}
                old = generate_teacher([fixture()], route, target, limits, provenance=provenance(ORIGIN_COMMIT))
                prefix = copy.deepcopy(old["shards"]); receipt = copy.deepcopy(old["problem_receipts"])
                old_bytes = [(target / r["path"]).read_bytes() for r in prefix + receipt]
                c = contract(root, [target / "manifest.json"])
                new = generate_teacher([fixture() for _ in range(3)], route, target, limits,
                        provenance=provenance("f" * 40), resume=True, repair_contract=c)
                self.assertEqual(new["trials"], old["trials"]); self.assertEqual(new["rows"], old["rows"])
                self.assertEqual(new["problems"], 3)
                self.assertEqual([(target / r["path"]).read_bytes() for r in prefix + receipt], old_bytes)
                for entry in new["problem_receipts"][1:]:
                    raw = json.loads(gzip.decompress((target / entry["path"]).read_bytes()))
                    self.assertEqual(raw["search_scope"], {"searched": False, "reference_only": True,
                                                          "unknown": True, "trials": 0, "complete_search": False})
                    self.assertEqual(raw["teacher_elapsed_seconds"], 0.)
                    self.assertEqual(raw["provenance"], provenance(ORIGIN_COMMIT))
                    self.assertEqual(raw["execution_source"]["git_commit"], "f" * 40)
                self.assertFalse(new["search_accounting"]["all_problems_strongly_searched"])

    def test_active_cursor_rng_and_trial_budget_survive_source_change(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); target = root / "teacher"
            limits = {"workers": 1, "max_trials": 10, "max_seconds": 10., "max_steps": 1}
            problem = fixture()
            first = generate_teacher([problem], "02", target, limits,
                     provenance=provenance(ORIGIN_COMMIT),
                     stop_callback=lambda: problem.calls["candidates"] >= 2)
            self.assertEqual(first["status"], "stopped")
            self.assertIsNotNone(first["active"])
            c = contract(root, [target / "manifest.json"])
            writer = EvidenceWriter(target, "02", provenance("f" * 40), resume=True, repair_contract=c)
            self.assertEqual(writer.manifest["active"], first["active"])
            self.assertEqual(writer.manifest["rng"], first["rng"])
            self.assertEqual(writer.manifest["trials"], first["trials"])
            resumed = generate_teacher([fixture()], "02", target, limits,
                      provenance=provenance("f" * 40), resume=True, repair_contract=c)
            full = generate_teacher([fixture()], "02", root / "full", limits,
                                     provenance=provenance("f" * 40))
            self.assertEqual(resumed["trials"], full["trials"])
            self.assertEqual(resumed["rows"], full["rows"])
            self.assertEqual(resumed["rng"], full["rng"])
            self.assertIsNone(resumed["active"])


if __name__ == "__main__":
    unittest.main()
