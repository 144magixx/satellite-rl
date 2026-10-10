"""Offline report checks use synthetic records solely to test audit logic."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

from scripts.stage1_drl_r04.report import Evidence, _audit_row, _layout_sha, _stats, analyze


def row():
    beam = {"demand_bps": 100.0, "capacity_bps": 80.0,
            "carried_bps": 80.0, "unclipped_capacity_demand_ratio": .8,
            "si": .8, "sgm": .7}
    layout = {"centers": [[45.0, 10.0]], "widths": [.5], "assignment": [0]}
    digest = _layout_sha(layout)
    return {"initialization": {"metrics": {"coverage_100": True,
                                              "demand_conserved": True,
                                              "total_demand_bps": 100.0}},
            "commits": 1, "trials": 1, "geometry_calls": 6,
            "initial_sha256": digest, "initial_layout": layout,
            "best_sha256": digest, "best_layout": layout,
            "algorithm_seconds": .2, "total_seconds": .3,
            "terminal_physics": {"layout_aliases": {"best": digest},
                                 "layouts_by_sha256": {digest: {
                                     "beams": [beam],
                                     "aircraft_service": {"demand_weighted_carried_fraction": .75},
                                     "metrics": {"carried_bps": 80.0,
                                                                   "total_demand_bps": 100.0,
                                                                   "demand_weighted_carried_fraction": .8,
                                                                   "beam_mean_si": .8,
                                                                   "beam_mean_sgm": .7}}}},
            "initial_j": .2, "best_j": .1, "reward_sum": 10.0,
            "ledger": [{"reward": 10.0}]}


class ReportTests(unittest.TestCase):
    def test_beam_and_reward_algebra_and_missing_values(self):
        valid, _ = _audit_row(row())
        self.assertEqual(valid, [])
        changed = row()
        del changed["terminal_physics"]["layouts_by_sha256"][changed["best_sha256"]]["beams"][0]["sgm"]
        errors, _ = _audit_row(changed)
        self.assertIn("per_beam_field_missing", errors)
        changed = row()
        changed["terminal_physics"]["layouts_by_sha256"][changed["best_sha256"]]["beams"][0]["carried_bps"] = 90.0
        errors, _ = _audit_row(changed)
        self.assertIn("per_beam_carried_not_min_capacity_demand", errors)
        changed = row()
        changed["reward_sum"] = 0.0
        errors, _ = _audit_row(changed)
        self.assertIn("reward_not_telescoping", errors)

    def test_statistics_keep_absence_as_null(self):
        self.assertEqual(_stats([None, float("nan")]),
                         {"count": 0, "mean": None, "median": None, "p95": None})
        self.assertEqual(_stats([1, 2, 3])["p95"], 3)

    def test_directory_and_zip_are_read_without_extracting(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            name = "20418/run-1/results.jsonl"
            target = root / name
            target.parent.mkdir(parents=True)
            target.write_text('{"value": 1}\n', encoding="utf-8")
            bundle = root / "bundle.zip"
            with zipfile.ZipFile(bundle, "w") as archive:
                archive.write(target, name)
            for source in (root, bundle):
                evidence = Evidence(source)
                try:
                    result, error = evidence.job_jsonl(20418, "run-1", "results.jsonl")
                    self.assertIsNone(error)
                    self.assertEqual(result, [{"value": 1}])
                    digest, error = evidence.sha(20418, "run-1", "results.jsonl")
                    self.assertEqual(digest, hashlib.sha256(target.read_bytes()).hexdigest())
                finally:
                    evidence.close()

    def test_partial_bundle_never_becomes_formal_ready(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            jobs = root / "jobs"
            jobs.mkdir()
            state = {"experiment_id": "S1-D12A", "scientific_commit": "a" * 40,
                     "tasks": {}}
            for source in (jobs, root / "jobs.zip"):
                if source.suffix == ".zip":
                    with zipfile.ZipFile(source, "w") as archive:
                        archive.writestr("placeholder.txt", "partial")
                audit = analyze(state, source, root / ("zip-report" if source.suffix else "dir-report"))
                self.assertFalse(audit["formal_ready"])
                self.assertEqual(audit["test_rows"], 0)
                self.assertEqual(audit["scope"], "partial QA only")
                self.assertTrue((root / ("zip-report" if source.suffix else "dir-report") /
                                 "audit.json").is_file())

    def test_single_real_schema_shard_is_visible_but_only_partial_qa(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            jobdir = root / "jobs" / "20418" / "eval-1"
            jobdir.mkdir(parents=True)
            data = row()
            data.update(dataset_index=0, demand_multiplier=.75,
                        scene_id="mother-0", scene_sha256="scene-sha",
                        checkpoint_step=0,
                        algorithm_seconds=.2, total_seconds=.3,
                        initialization_seconds=.1, menu_seconds=.01, trial_seconds=.02)
            result_bytes = (json.dumps(data) + "\n").encode()
            (jobdir / "results.jsonl").write_bytes(result_bytes)
            (jobdir / "job.json").write_text(json.dumps({"run_id": "eval-1",
                "variant": "I0", "git_commit": "a" * 40,
                "experiment_id": "S1-D12A", "shard_id": 0,
                "scene_indices": list(range(0, 100, 4))}), encoding="utf-8")
            (jobdir / "status.json").write_text('{"stage":"completed"}', encoding="utf-8")
            (jobdir / "config.json").write_text('{"schema_version":6,"datasets":{"test":{"sha256":"missing"}}}',
                                                 encoding="utf-8")
            (jobdir / "summary.json").write_text(json.dumps({"scene_count": 1,
                "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
                "model": {"env_step": 0}, "method": "I0"}), encoding="utf-8")
            state = {"experiment_id": "S1-D12A", "scientific_commit": "a" * 40,
                     "tasks": {"test_I0_20418": {"port": 20418,
                                                    "run_id": "eval-1", "variant": "I0"}}}
            with mock.patch("scripts.stage1_drl_r04.report._plot", return_value=([], None)):
                audit = analyze(state, root / "jobs", root / "report")
            self.assertFalse(audit["formal_ready"])
            self.assertEqual(audit["test_rows"], 1)
            self.assertTrue((root / "report" / "per_scene.csv").is_file())
            summary = json.loads((root / "report" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["I0@0.75"]["scene_count"], 1)
            self.assertEqual(summary["I0@0.75"]["beam_carried_fraction"]["mean"], .8)
            self.assertEqual(summary["I0@0.75"]["aircraft_tdma_carried_fraction"]["mean"], .75)

    def test_training_episode_audit_streams_and_keeps_only_plot_pairs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            jobdir = root / "jobs" / "20418" / "train-p0"
            jobdir.mkdir(parents=True)
            checkpoint = b"fixed-checkpoint"
            (jobdir / "checkpoint.pt").write_bytes(checkpoint)
            (jobdir / "job.json").write_text(json.dumps({
                "run_id": "train-p0", "variant": "P0", "git_commit": "a" * 40,
                "experiment_id": "S1-D12A"}), encoding="utf-8")
            (jobdir / "status.json").write_text(json.dumps({
                "stage": "completed", "env_step": 2,
                "checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest()}), encoding="utf-8")
            (jobdir / "config.json").write_text('{"schema_version":6}', encoding="utf-8")
            events = [{"stream_id": 0, "stream_episode_serial": serial,
                       "env_step": serial, "reward": 1.0,
                       "best_j_before": .2, "best_j_after": .19,
                       "info": {"commits": 1, "trials": 1, "geometry_calls": 6}}
                      for serial in (1, 2)]
            episodes = [{"stream_id": 0, "stream_episode_serial": serial,
                         "completed_episode": True, "terminal_env_step": serial,
                         "episode_return": 1.0,
                         "action_trace": {"count": 1, "first_env_step": serial,
                                          "last_env_step": serial},
                         "large_terminal_payload": "x" * 10000}
                        for serial in (1, 2)]
            for filename, records in (("actions.jsonl", events),
                                      ("episodes.jsonl", episodes),
                                      ("budget_tails.jsonl", []),
                                      ("evaluations.jsonl", [])):
                (jobdir / filename).write_text(
                    "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
            state = {"experiment_id": "S1-D12A", "scientific_commit": "a" * 40,
                     "tasks": {"P0": {"port": 20418, "run_id": "train-p0", "variant": "P0"}}}
            original = Evidence.job_jsonl
            def guarded(instance, port, run_id, filename):
                if filename == "episodes.jsonl":
                    raise AssertionError("episodes must be streamed")
                return original(instance, port, run_id, filename)
            with (mock.patch.object(Evidence, "job_jsonl", guarded),
                  mock.patch("scripts.stage1_drl_r04.report._plot", return_value=([], None)) as plot):
                audit = analyze(state, root / "jobs", root / "report")
            training = plot.call_args.args[2]
            self.assertEqual(training["P0"], [(1, 1.0), (2, 1.0)])
            task = json.loads((root / "report" / "task_audit.json").read_text(encoding="utf-8"))["P0"]
            self.assertEqual(task["action_rows"], 2)
            self.assertEqual(task["completed_episodes"], 2)
            self.assertFalse(any("terminal action trace or reward differs" in issue
                                 for issue in audit["issues"]))


if __name__ == "__main__":
    unittest.main()
