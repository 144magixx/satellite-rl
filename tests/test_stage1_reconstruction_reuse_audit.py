"""Independent reuse audit checks, without an old remote task or training."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/stage1_reconstruction"))
import audit_reuse

from satellite_rl.coverage.reconstruction import ReconstructionState
from tests.test_coverage_reconstruction import scene_fixture


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ReuseAuditTests(unittest.TestCase):
    def setUp(self):
        self.scene = scene_fixture()
        self.state = ReconstructionState.from_scene(self.scene)
        self.state.initialization_source = "scene_centers_seed+coverage_repair"
        self.state.initialization_candidates = [
            {"source": name, "valid": True, "seconds": .1}
            for name in ("scene_centers_seed", "recomputed_cluster", "recomputed_rule")]
        self.state.elapsed_initialization_s = .5
        self.start = {"scene_id": self.scene.scene_id,
                      "scene_sha256": self.scene.source_hash,
                      "initialization_valid": True, "initialization_seconds": .6,
                      "initialization_repair_geometry_calls": 0,
                      "initialization_repair_ray_pair_checks": 0,
                      "initialization_repair_attempts": 0,
                      "state": self.state.to_record()}
        self.input = {"scene_id": self.scene.scene_id,
                      "source_hash": self.scene.source_hash,
                      "aircraft_ids": self.scene.aircraft_ids,
                      "positions": self.scene.positions,
                      "demands": self.scene.demands}

    def test_independent_geometry_and_timing_accept_complete_start(self):
        row = audit_reuse.audit_scene(self.input, self.start, edge_ratio=.5)
        self.assertTrue(row["initialization_valid"])
        self.assertEqual(row["total_users"], len(self.scene.demands))
        self.assertEqual(row["positive_beams"], 100)
        self.assertEqual(row["initialization_seconds"], .6)
        self.assertEqual(row["initialization_state_seconds"], .5)

    def test_covered_unassigned_and_fabricated_timing_are_rejected(self):
        start = json.loads(json.dumps(self.start))
        start["state"]["assignment"][0] = -1
        with self.assertRaisesRegex(ValueError, "covered users"):
            audit_reuse.audit_scene(self.input, start, edge_ratio=.5)
        start = json.loads(json.dumps(self.start))
        start.pop("initialization_seconds")
        with self.assertRaisesRegex(ValueError, "duration missing"):
            audit_reuse.audit_scene(self.input, start, edge_ratio=.5)

    def test_only_cache_line_is_permitted_between_frozen_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old, new = root / "old", root / "new"
            deployed_commit = "a" * 40  # Includes operational route changes elsewhere.
            for directory, commit in ((old, audit_reuse.OLD_COMMIT),
                                      (new, deployed_commit)):
                files = {}
                for name in audit_reuse.SCIENTIFIC_FILES:
                    path = directory / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"unchanged initializer dependency\n")
                    files[name] = file_sha(path)
                core = directory / "src/satellite_rl/coverage/reconstruction.py"
                core.write_bytes(b"old core\n" + (
                    b"                    next_state._distance_cache = proposal._distance_cache\n"
                    if directory == new else b""))
                files["src/satellite_rl/coverage/reconstruction.py"] = file_sha(core)
                (directory / "source_manifest.json").write_text(
                    json.dumps({"git_commit": commit, "files": files}), encoding="utf-8")
            result = audit_reuse.audit_source_compatibility(old, new, deployed_commit)
            self.assertEqual(result["only_scientific_change"],
                             "accepted_branch_propagates_verified_distance_cache")
            with self.assertRaisesRegex(ValueError, "manifest commit identity"):
                audit_reuse.audit_source_compatibility(old, new, audit_reuse.NEW_COMMIT)
            with self.assertRaisesRegex(ValueError, "full 40-digit"):
                audit_reuse.audit_source_compatibility(old, new, "short")
            dependency = new / audit_reuse.SCIENTIFIC_FILES[0]
            dependency.write_bytes(b"changed initializer")
            new_manifest = new / "source_manifest.json"
            manifest = json.loads(new_manifest.read_text(encoding="utf-8"))
            manifest["files"][audit_reuse.SCIENTIFIC_FILES[0]] = file_sha(dependency)
            new_manifest.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "initialization/physical dependency changed"):
                audit_reuse.audit_source_compatibility(old, new, deployed_commit)
            dependency.write_bytes(b"unchanged initializer dependency\n")
            manifest["files"][audit_reuse.SCIENTIFIC_FILES[0]] = file_sha(dependency)
            core = new / "src/satellite_rl/coverage/reconstruction.py"
            core.write_bytes(core.read_bytes() + b"extra science change\n")
            manifest["files"]["src/satellite_rl/coverage/reconstruction.py"] = file_sha(core)
            new_manifest.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "scientific core differs"):
                audit_reuse.audit_source_compatibility(old, new, deployed_commit)

    def test_one_frozen_prepare_checks_npz_and_start_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "outputs/workbench/source_snapshots" / audit_reuse.OLD_COMMIT
            source.mkdir(parents=True)
            run_id = audit_reuse.PREPARES["train"][0]
            job_dir = root / "outputs/workbench/jobs" / run_id
            input_dir = job_dir / "scene_inputs/train"
            input_dir.mkdir(parents=True)
            npz = input_dir / "inputs.npz"
            n = len(self.scene.demands)
            np.savez_compressed(npz, format_version="coverage_aircraft_v1",
                                lengths=np.array([n], dtype=np.int64),
                                positions=self.scene.positions[None],
                                demands=self.scene.demands[None],
                                centers=self.scene.centers[None],
                                scene_ids=np.array([self.scene.scene_id]),
                                source_hashes=np.array([self.scene.source_hash]),
                                aircraft_ids_json=json.dumps([self.scene.aircraft_ids]),
                                scene_metadata_json=json.dumps([{}]),
                                metadata_json=json.dumps({"initializer_sha256": "fixture"}))
            (input_dir / "record.json").write_text(json.dumps({
                "prepared_npz": "inputs.npz", "prepared_npz_sha256": file_sha(npz),
                "partition_role": "train", "count": 1,
                "scenes": [{"scene_id": self.scene.scene_id,
                            "source_hash": self.scene.source_hash}]}), encoding="utf-8")
            starts = job_dir / "starts.jsonl"
            starts.write_text(json.dumps(self.start) + "\n", encoding="utf-8")
            config = {"datasets": {"train": {"id": "dataset-fixture", "path": str(npz),
                       "sha256": file_sha(npz), "count": 1, "partition_role": "train",
                       "initializer_sha256": "fixture"}},
                      "coverage": {"method": "none", "reconstruction": {
                          "route": "prepare", "edge_demand_ratio": .5}}}
            for name, body in (
                ("job", {"run_id": run_id, "git_commit": audit_reuse.OLD_COMMIT,
                         "source_root": str(source), "kind": "evaluate",
                         "reconstruction_route": "prepare"}),
                ("config", config),
                ("status", {"stage": "completed", "prepared_scenes": 1,
                            "preparation_total": 1}),
                ("artifacts", {"run_id": run_id, "reconstruction_starts": str(starts),
                               "reconstruction_starts_sha256": file_sha(starts),
                               "dataset_sha256": file_sha(npz), "source_root": str(source)})):
                (job_dir / f"{name}.json").write_text(json.dumps(body), encoding="utf-8")
            result = audit_reuse.audit_prepare(job_dir, root, source, "train", run_id, 1)
            self.assertEqual(result["qualified_count"], 1)
            self.assertEqual(result["dataset_sha256"], file_sha(npz))
            config["datasets"]["train"]["sha256"] = "0" * 64
            (job_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "NPZ/start SHA"):
                audit_reuse.audit_prepare(job_dir, root, source, "train", run_id, 1)

    def test_receipt_records_deployed_commit_and_independent_catalog(self):
        deployed_commit = "b" * 40
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old_plan = root / "old_plan.json"
            old_plan.write_text(json.dumps({"experiment_id": "S1-03",
                                            "science_commit": audit_reuse.OLD_COMMIT}),
                                encoding="utf-8")
            catalog_roles = {}
            prepared = {}
            for role, (run_id, count) in audit_reuse.PREPARES.items():
                qualified = 572 if role == "train" else count
                record = {"prepare_run_id": run_id, "source_run_id": run_id,
                          "source_dataset_id": "dataset-" + role,
                          "dataset_sha256": role[0] * 64,
                          "starts_sha256": role[-1] * 64,
                          "scene_count": count, "qualified_count": qualified,
                          "failed_count": count - qualified, "initializer_config": {},
                          "scenes": [], "issues": []}
                prepared[role] = record
                catalog_roles[role] = {key: record[key] for key in (
                    "source_dataset_id", "dataset_sha256", "source_run_id",
                    "starts_sha256", "scene_count", "qualified_count")}
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({"source_group": "s1_03",
                                           "source_target": "20418",
                                           "old_science_commit": audit_reuse.OLD_COMMIT,
                                           "old_plan_sha256": file_sha(old_plan),
                                           "roles": catalog_roles}), encoding="utf-8")
            with (patch.object(audit_reuse, "audit_source_compatibility",
                               return_value={"old_source_manifest_sha256": "0" * 64,
                                             "new_source_manifest_sha256": "1" * 64}),
                  patch.object(audit_reuse, "under", side_effect=lambda path, root: path),
                  patch.object(audit_reuse, "audit_prepare",
                               side_effect=lambda _dir, _root, _source, role, _run, _count:
                               prepared[role])):
                result = audit_reuse.run(old_plan, root / "old", root / "new",
                                         root, deployed_commit, catalog)
                self.assertTrue(result["passed"])
                self.assertEqual(result["new_science_commit"], deployed_commit)
                self.assertEqual(result["source_catalog_sha256"], file_sha(catalog))
                catalog_roles["train"]["qualified_count"] = 571
                catalog.write_text(json.dumps({"source_group": "s1_03",
                                               "source_target": "20418",
                                               "old_science_commit": audit_reuse.OLD_COMMIT,
                                               "old_plan_sha256": file_sha(old_plan),
                                               "roles": catalog_roles}), encoding="utf-8")
                rejected = audit_reuse.run(old_plan, root / "old", root / "new",
                                           root, deployed_commit, catalog)
                self.assertFalse(rejected["passed"])
                self.assertTrue(any("catalog role differs" in issue["reason"]
                                    for issue in rejected["issues"]))


if __name__ == "__main__":
    unittest.main()
