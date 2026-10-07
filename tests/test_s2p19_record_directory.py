"""Prepared record.json references must import the containing NPZ directory."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from scripts.s2p19 import controller, plan_builder
from satellite_rl.envs.regions import profile_metadata
from satellite_rl.workbench.datasets import _environment, import_scenes
from satellite_rl.workbench.storage import Store, digest


class PreparedSceneImportTests(unittest.TestCase):
    def test_future_bootstrap_imports_directory_but_mother_keeps_record_file(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            contracts = root / "contracts"
            contracts.mkdir()
            for name in ("api_prepare_dev_20418.json", "api_prepare_mothers_20424.json"):
                (contracts / name).write_text(json.dumps({"body": {"preparation":
                    {"preparation_type": "dev_validation"}}}), encoding="utf-8")
            commit = "a" * 40
            launch = root / "preflight" / f"workbench_{commit[:12]}.json"
            launch.parent.mkdir()
            launch.write_text(json.dumps({"scientific_commit": commit,
                "instances": {"20418": {}, "20424": {}}}), encoding="utf-8")
            (root / "preflight/p18_v2_model.json").write_text(json.dumps({
                "source_dir": "/frozen/v2", "provenance": {
                    "run_id": "train_sac_20261006_084454_365eff71", "env_step": 400000},
                "files": {"policy.pt": {"sha256":
                    "78ed38b164141af3968c4498dc4a462b8baf060764f133076ea7c02c7d868cea"}}}),
                encoding="utf-8")
            c0 = {"training": {"total_steps": 400000, "pilot_stop_steps": 100000,
                               "demonstrations": {}}}
            c1 = json.loads(json.dumps(c0))
            defaults = {"C0": c0, "C1": c1,
                        "I0": {"total_steps": 400000, "pilot_stop_steps": 100000}}
            with (patch.object(plan_builder, "CONTRACTS", contracts),
                  patch.object(plan_builder, "_source_commit", return_value=commit),
                  patch.object(plan_builder.ops, "LOCAL", root),
                  patch.object(plan_builder.ops, "api", return_value={
                      "experiment_id": "S2-P19", "defaults": defaults}),
                  patch.object(controller, "PLAN", root / "controller/plan.json")):
                plan_builder.bootstrap()
            tasks = {row["id"]: row for row in json.loads(
                (root / "controller/plan.json").read_text(encoding="utf-8"))["tasks"]}
            for port in (20418, 20424):
                for partition in ("development", "validation", "final"):
                    self.assertIn("$record_directory",
                                  tasks[f"import_{partition}_{port}"]["body"]["path"])
            mother = tasks["prepare_final_20424"]["body"]["preparation"]["source_record"]["path"]
            self.assertIn("$task", mother)
            self.assertNotIn("$record_directory", mother)

    def test_real_resolve_then_workbench_import_uses_sidecar_and_npz(self):
        with TemporaryDirectory() as temporary:
            archive = Path(temporary)
            run_id = "prepare_prepare_20261007_171407_3c757c98"
            source = archive / "outputs/workbench/jobs" / run_id / "prepared/final"
            source.mkdir(parents=True)
            environment = _environment("europe_v1", max_beams=200)
            environment.reset(seed=51001)
            scene = source / "scene_00000.npz"
            np.savez_compressed(scene, **environment.scene())
            profile = profile_metadata("europe_v1")
            record = {"id": "s2p19_final_20261007", "name": "final fixture",
                      "path": source.as_posix(), "count": 1, "profile": "europe_v1",
                      "environment_version": profile["environment_version"],
                      "environment_profile": profile,
                      "source": {"type": "aviation_variable_beams_s2p19",
                                 "partition_role": "test", "source_dates": ["2024-02-01"]},
                      "files": [{"path": scene.as_posix(), "sha256": digest(scene),
                                 "beam_count": len(environment.scene()["rate"])}]}
            sidecar = source / "record.json"
            sidecar.write_text(json.dumps(record), encoding="utf-8")
            receipts = {"prepare_final_20424": {"state": "completed", "run_id": run_id,
                "result": {"summary": {"outputs": {"final": {
                    "path": sidecar.as_posix(), "sha256": digest(sidecar)}}}}}}
            body = {"path": plan_builder.record_directory("prepare_final_20424", "final", 20424),
                    "name": "imported final", "profile": "europe_v1"}
            with patch.object(controller.ops, "instance",
                              return_value={"archive": archive.as_posix()}):
                resolved = controller._resolve(body, receipts)
                self.assertEqual(resolved["path"], source.as_posix())
                imported = import_scenes(Store(archive), **resolved)
                self.assertEqual(imported["count"], 1)
                self.assertEqual(imported["source"]["reimport"]["record_id"],
                                 "s2p19_final_20261007")
                damaged = json.loads(json.dumps(receipts))
                damaged["prepare_final_20424"]["result"]["summary"]["outputs"]["final"]["path"] = (
                    (archive / "outside/record.json").as_posix())
                with self.assertRaises(ValueError):
                    controller._resolve(body, damaged)
                damaged = json.loads(json.dumps(receipts))
                damaged["prepare_final_20424"]["result"]["summary"]["outputs"]["final"]["path"] = (
                    (archive / "outputs/workbench/jobs" / "another_run" /
                     "prepared/final/record.json").as_posix())
                with self.assertRaises(ValueError):
                    controller._resolve(body, damaged)
                with self.assertRaises(ValueError):
                    controller._resolve(plan_builder.record_directory(
                        "prepare_final_20424", "development", 20424), receipts)


if __name__ == "__main__":
    unittest.main()
