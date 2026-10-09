"""Report material must remain tied to verified raw rows and full denominators."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.stage1_variable import controller, report
from satellite_rl.coverage.scenes import CoverageScene
from satellite_rl.coverage.variable_config import VariableConfig
from satellite_rl.coverage.variable_runner import _single_evaluation
from satellite_rl.workbench.coverage_variable import variable_default_config


class VariableReportTests(unittest.TestCase):
    def setUp(self):
        self.scene = CoverageScene(
            "mother-1", np.asarray([[45., 10.], [45.01, 10.01], [46., 11.]]),
            np.asarray([80., 40., 90.]) * 1e6, np.empty((0, 2)),
            ["u1", "u2", "u3"], source_hash="aircraft-source-sha")
        self.cfg = VariableConfig(max_beams=4, max_actions=3)
        self.config = variable_default_config()

    def _raw(self, route="R01", factor=1.0, scene=None):
        scene = scene or self.scene
        scene = replace(scene, demands=scene.demands * factor,
                        metadata={**scene.metadata, "demand_multiplier": factor})
        self.config["coverage"]["variable"]["route"] = route
        self.config["coverage"]["method"] = "local_search"
        return _single_evaluation(scene, self.cfg, route, None, config=self.config)

    def _fixture(self, temp, rows, *, key="test_R01"):
        base = Path(temp)
        state_path = base / "controller" / "state.json"
        mirror = base / "synced_jobs"
        task = controller.TASKS[key] | {"run_id": "run-1"}
        state = {"scientific_commit": "a" * 40, "phase": "main",
                 "tasks": {key: task}, "audits": {}}
        report._save(state_path, state)
        path = mirror / str(task["port"]) / task["run_id"] / "results.partial.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                        encoding="utf-8")
        manifest = {"scientific_commit": state["scientific_commit"],
                    "tasks": {key: {"port": task["port"], "run_id": task["run_id"]}},
                    "files": [{"port": task["port"], "run_id": task["run_id"],
                               "path": str(path), "sha256": report._sha(path),
                               "bytes": path.stat().st_size}]}
        report._save(mirror / "manifest.json", manifest)
        return state_path, mirror, base / "analysis", path

    def test_partial_report_uses_only_three_real_rows_and_source_sha(self):
        rows = [self._raw(factor=factor) for factor in report.MULTIPLIERS]
        with tempfile.TemporaryDirectory() as temp:
            state, mirror, output, _ = self._fixture(temp, rows)
            result = report.build_report(state, mirror, output)
            self.assertFalse(result["formal_ready"])
            self.assertEqual(result["grids"]["test_R01"]["observed_rows"], 3)
            self.assertEqual(result["groups"][0]["scene_rows"], 1)
            self.assertEqual(sum(x["scene_rows"] for x in result["groups"]), 3)
            self.assertEqual(len((output / "per_scene.csv").read_text(
                encoding="utf-8-sig").splitlines()), 4)
            self.assertTrue((output / "静态方法三档对照.png").is_file())
            inputs = report._json(output / "inputs.json")
            self.assertEqual(inputs["scientific_commit"], "a" * 40)
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                report.build_report(state, mirror, output, strict=True)

    def test_duplicate_scene_tier_and_missing_capacity_fail_closed(self):
        row = self._raw()
        with tempfile.TemporaryDirectory() as temp:
            state, mirror, output, _ = self._fixture(temp, [row, row])
            with self.assertRaisesRegex(ValueError, "duplicate mother"):
                report.build_report(state, mirror, output)
        missing = self._raw()
        missing["beams"][0]["capacity_bps"] = None
        with tempfile.TemporaryDirectory() as temp:
            state, mirror, output, _ = self._fixture(temp, [missing])
            with self.assertRaisesRegex(ValueError, "capacity"):
                report.build_report(state, mirror, output)

    def test_synced_manifest_sha_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            state, mirror, output, path = self._fixture(temp, [self._raw()])
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA"):
                report.build_report(state, mirror, output)

    def test_sequence_pairing_uses_sequence_as_statistical_unit(self):
        rows = []
        for frame, timestamp in enumerate(("2024-02-06T00:00:00Z",
                                           "2024-02-06T00:01:00Z")):
            for route, carried in (("R04", .4 + .1 * frame),
                                   ("R12", .5 + .1 * frame)):
                rows.append({"role": "sequence", "route": route,
                             "sequence_id": "one-real-sequence",
                             "selected_time_utc": timestamp,
                             "demand_multiplier": 1.0,
                             "scene_sha256": f"frame-{frame}",
                             "user_tdma_carried_fraction": carried,
                             "load_cv": .3, "beam_count": 2,
                             "total_seconds": 1.0})
        paired = report._paired_sequences(rows)
        self.assertEqual(paired["matched_frames"], 2)
        self.assertEqual(len(paired["sequence_units"]), 1)
        self.assertAlmostEqual(paired["sequence_units"][0][
            "mean_delta_user_tdma_carried_fraction"], .1)


if __name__ == "__main__":
    unittest.main()
