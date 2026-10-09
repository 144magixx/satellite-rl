"""Report material must remain tied to verified raw rows and full denominators."""

from __future__ import annotations

from dataclasses import replace
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

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
            with (output / "per_scene.csv").open(encoding="utf-8-sig", newline="") as stream:
                first = next(csv.DictReader(stream))
            self.assertEqual(first["initialization_seconds"], "")
            self.assertEqual(first["algorithm_seconds"], "")
            self.assertEqual(first["timing_granularity"], "combined_initialization_and_search")
            self.assertGreaterEqual(float(first["method_seconds_before_common_downstream"]), 0)
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

    def test_sequence_plot_groups_twenty_mothers_by_three_tiers_with_id_map(self):
        units = [{"sequence_id": f"2024-02-06_segment_{i:02d}_long_source_identity",
                  "demand_multiplier": factor,
                  "mean_delta_user_tdma_carried_fraction": (i - 10) / 100,
                  "mean_delta_total_seconds": (i - 10) / 5}
                 for i in range(20) for factor in report.MULTIPLIERS]
        with tempfile.TemporaryDirectory() as temp:
            result = report._sequence_plot(Path(temp), {"sequence_units": units,
                                                          "matched_frames": 360})
            self.assertEqual(result["sequence_count"], 20)
            self.assertEqual(result["sequence_units"], 60)
            self.assertEqual(result["statistical_unit"], "mother_sequence")
            self.assertTrue(Path(result["path"]).is_file())
            with Path(result["sequence_label_map_path"]).open(encoding="utf-8-sig", newline="") as stream:
                mapping = list(csv.DictReader(stream))
            self.assertEqual([row["short_id"] for row in mapping],
                             [f"S{i:02d}" for i in range(1, 21)])
            self.assertEqual(mapping[0]["sequence_id"], units[0]["sequence_id"])

    def test_state_snapshot_uses_absolute_registry_and_verifies_sha(self):
        with tempfile.TemporaryDirectory() as temp:
            state_path, mirror, _, _ = self._fixture(temp, [self._raw()])
            base = Path(temp)
            registry = base / "frozen" / "dataset_registry.json"
            report._save(registry, {"scientific_commit": "a" * 40,
                                    "canonical_sha256": {"test": "test-source-sha"}})
            state = report._json(state_path)
            state["dataset_registry"] = str(registry.resolve())
            state["dataset_registry_file_sha256"] = report._sha(registry)
            state["dataset_sha256"] = {"test": "test-source-sha"}
            snapshot = base / "preflight" / "state_snapshot.json"
            report._save(snapshot, state)
            result = report.build_report(snapshot, mirror, base / "analysis")
            self.assertTrue(result["dataset_registry_verified"])
            inputs = report._json(base / "analysis" / "inputs.json")
            self.assertEqual(inputs["dataset_registry"]["path"], str(registry.resolve()))
            report._save(registry, {"scientific_commit": "a" * 40,
                                    "canonical_sha256": {"test": "changed"}})
            with self.assertRaisesRegex(ValueError, "registry file SHA differs"):
                report.build_report(snapshot, mirror, base / "analysis2")

    def test_repaired_learning_source_does_not_relabel_legacy_baseline(self):
        with tempfile.TemporaryDirectory() as temp:
            state = {"scientific_commit": "a" * 40,
                     "throughput_source_overrides": {
                         "train_R08": {"repair_commit": "b" * 40}},
                     "tasks": {
                         "train_R08": controller.TASKS["train_R08"] | {"run_id": "learned"},
                         "test_R01": controller.TASKS["test_R01"] | {"run_id": "baseline"}}}
            mapped = report._task_sources(state, Path(temp), None)
            self.assertEqual(mapped["train_R08"]["task"]["source_commit"], "b" * 40)
            self.assertEqual(mapped["test_R01"]["task"]["source_commit"], "a" * 40)

    def test_jsonl_is_streamed_without_path_read_text_and_bad_line_is_located(self):
        rows = [self._raw(factor=factor) for factor in report.MULTIPLIERS]
        with tempfile.TemporaryDirectory() as temp:
            state, mirror, output, _ = self._fixture(temp, rows)
            original = Path.read_text

            def guarded(path, *args, **kwargs):
                if path.suffix == ".jsonl":
                    raise AssertionError("large JSONL must not call read_text")
                return original(path, *args, **kwargs)

            with patch.object(Path, "read_text", guarded):
                result = report.build_report(state, mirror, output)
            self.assertEqual(result["grids"]["test_R01"]["observed_rows"], 3)
            broken = Path(temp) / "bad.jsonl"
            broken.write_text('{"ok":1}\nnot-json\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"bad\.jsonl:2: malformed JSONL"):
                list(report._iter_jsonl(broken))

    def test_terminal_user_arrays_and_beam_sgm_cannot_be_tampered(self):
        task = controller.TASKS["test_R01"] | {"key": "test_R01", "run_id": "run-1",
                                               "source_commit": "a" * 40}
        original = self._raw()
        self.assertTrue(report._scene_row(task, original)["demand_conserved"])
        changed_user = deepcopy(original)
        changed_user["downstream"]["aircraft_service"]["capacity_bps"][0] += 1e6
        with self.assertRaisesRegex(ValueError, "TDMA capacity"):
            report._scene_row(task, changed_user)
        changed_share = deepcopy(original)
        changed_share["downstream"]["aircraft_service"]["airtime_fraction"][0] = 0.9
        with self.assertRaisesRegex(ValueError, "airtime|TDMA capacity"):
            report._scene_row(task, changed_share)
        changed_sgm = deepcopy(original)
        changed_sgm["beams"][0]["sgm"] *= 0.5
        with self.assertRaisesRegex(ValueError, "SGM"):
            report._scene_row(task, changed_sgm)
        missing_user = deepcopy(original)
        missing_user["downstream"]["aircraft_service"].pop("full_time_rate_bps")
        with self.assertRaisesRegex(ValueError, "terminal arrays missing"):
            report._scene_row(task, missing_user)

    def test_compact_completed_episode_audited_and_unfinished_tail_remains_missing(self):
        base = self._raw()
        base["route"] = "R08"
        service = base["downstream"]["aircraft_service"]
        for field in ("capacity_bps", "carried_bps", "si", "airtime_fraction"):
            for user, value in zip(base["users"], service[field]):
                user[field] = value
        base["downstream"] = {"metrics": base["downstream"]["metrics"],
                              "timing": base["downstream"]["timing"],
                              "calls": base["downstream"]["calls"],
                              "user_tdma_carried_fraction": service["demand_weighted_carried_fraction"],
                              "aircraft_service_model": service["model"]}
        base.update(terminal_physics_status="evaluated_all_actual_beams_and_users",
                    completed_episode=True, terminal_env_step=1, episode_return=1.25,
                    action_trace={"file": "actions.jsonl", "stream_id": 0,
                                  "stream_episode_serial": 1, "first_env_step": 1,
                                  "last_env_step": 1, "count": 1})
        task = controller.TASKS["verify_R08"] | {"run_id": "run-1",
                                                  "source_commit": "a" * 40}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = {}
            data = {"actions.jsonl": [{"env_step": 1, "stream_id": 0,
                                       "stream_episode_serial": 1}],
                    "updates.jsonl": [{"env_step": 2048, "policy_loss": .1, "entropy": .2}],
                    "episodes.jsonl": [base]}
            for name, rows in data.items():
                path = root / name
                path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                files[name] = {"path": str(path), "sha256": report._sha(path)}
            source = {"task": task, "files": files}
            original = Path.read_text

            def guarded(path, *args, **kwargs):
                if path.suffix == ".jsonl":
                    raise AssertionError("training JSONL must stream")
                return original(path, *args, **kwargs)

            with patch.object(Path, "read_text", guarded):
                result = report._train_rows("verify_R08", source)
            self.assertEqual(result["completed_episodes"], 1)
            self.assertEqual(set(result["episodes_raw"][0]),
                             {"terminal_env_step", "episode_return", "coverage_100", "completed_episode"})
            altered = deepcopy(base)
            altered["users"][0]["carried_bps"] += 1e6
            (root / "episodes.jsonl").write_text(json.dumps(altered) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "min\(C,D\)"):
                report._train_rows("verify_R08", source)
            tail = deepcopy(base)
            tail["completed_episode"] = False
            tail["terminal_physics_status"] = "not_run_incomplete_budget_tail"
            for beam in tail["beams"]:
                beam["capacity_bps"] = None
            (root / "episodes.jsonl").write_text(json.dumps(tail) + "\n", encoding="utf-8")
            partial = report._train_rows("verify_R08", source)
            self.assertEqual(partial["partial_episodes"], 1)

    def test_validation_node_streams_once_and_rejects_tampered_sgm_at_line(self):
        originals = {factor: self._raw(factor=factor) for factor in report.MULTIPLIERS}
        rows = []
        for mother in range(20):
            for factor in report.MULTIPLIERS:
                row = deepcopy(originals[factor])
                row["route"] = "R08"
                row["scene_id"] = f"mother-{mother}"
                row["scene_sha256"] = f"source-{mother}"
                rows.append(row)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            node = root / "validation_2048.jsonl"
            node.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            update = root / "updates.jsonl"
            update.write_text(json.dumps({"env_step": 2048, "policy_loss": .1,
                                          "entropy": .2}) + "\n", encoding="utf-8")
            ledger = root / "evaluations.jsonl"
            ledger.write_text(json.dumps({"env_step": 2048,
                                          "summary": {"mean_coverage_users": 1., "mean_load_cv": .1},
                                          "rows_sha256": report._sha(node),
                                          "downstream_allocation_calls": 60}) + "\n", encoding="utf-8")
            source = {"task": controller.TASKS["verify_R08"] |
                      {"run_id": "run-1", "source_commit": "a" * 40},
                      "files": {name: {"path": str(path), "sha256": report._sha(path)}
                                for name, path in ((node.name, node), (update.name, update),
                                                   (ledger.name, ledger))}}
            original = Path.read_text

            def guarded(path, *args, **kwargs):
                if path.suffix == ".jsonl":
                    raise AssertionError("validation JSONL must stream")
                return original(path, *args, **kwargs)

            with patch.object(Path, "read_text", guarded):
                info = report._train_rows("verify_R08", source)
            self.assertEqual(info["validation_nodes"], [2048])
            self.assertEqual(set(info["evaluations_raw"][0]["summary"]),
                             {"mean_coverage_users", "mean_load_cv"})
            rows[1]["beams"][0]["sgm"] *= .5
            node.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"validation_2048\.jsonl:2:.*SGM"):
                report._train_rows("verify_R08", source)


if __name__ == "__main__":
    unittest.main()
