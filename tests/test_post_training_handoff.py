"""Read-only and filesystem-only checks; never start a GPU worker."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from datetime import datetime, timedelta, timezone

from satellite_rl.evaluation.post_training_checks import EXPECTED_RUN_IDS, TRAIN_COMMIT
from satellite_rl.evaluation.post_training_handoff import (
    _capture_supervisor, _command, _exclusive_manager_lock, _marker_valid, _outputs_complete,
    _pause, _resume_request, _run_step, _supervisor_exited, run_handoff,
)
from satellite_rl.training import experiment_runner as runner


class PostTrainingHandoffTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="satellite_handoff_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_supervisor_identity_uses_pid_and_creation_token(self):
        (self.root / "supervisor_launch.json").write_text(json.dumps({"pid": 111}), encoding="utf-8")
        (self.root / "suite.lock").write_text(json.dumps({
            "pid": 111, "process_token": 222,
        }), encoding="utf-8")
        with mock.patch("satellite_rl.evaluation.post_training_handoff._process_token",
                        return_value=222):
            identity = _capture_supervisor(self.root)
            self.assertFalse(_supervisor_exited(identity))
        with mock.patch("satellite_rl.evaluation.post_training_handoff._process_token",
                        return_value=333):
            self.assertTrue(_supervisor_exited(identity))
            with self.assertRaisesRegex(RuntimeError, "身份已变化"):
                _capture_supervisor(self.root)

    def test_manager_lock_is_exclusive_and_released_without_unlinking(self):
        path = self.root / "manager.lock"
        with _exclusive_manager_lock(path):
            with self.assertRaisesRegex(RuntimeError, "独占锁"):
                with _exclusive_manager_lock(path):
                    pass
        self.assertTrue(path.exists())
        with _exclusive_manager_lock(path):
            pass

    def test_second_manager_cannot_overwrite_initial_state_before_lock(self):
        handoff = self.root / "outputs" / "experiments" / "sac_comparison_20260924_r2" / "post_training_checks" / "handoff"
        handoff.mkdir(parents=True)
        with _exclusive_manager_lock(handoff / "manager.lock"), \
             mock.patch("satellite_rl.evaluation.post_training_handoff.source_guard",
                        return_value={"post_commit": "post", "post_source_root": str(self.root)}):
            with self.assertRaisesRegex(RuntimeError, "独占锁"):
                run_handoff(self.root, self.root, poll_seconds=1, max_hours=1)
        self.assertFalse((handoff / "state.json").exists())

    def test_restarted_supervisor_requires_explicit_new_token(self):
        (self.root / "supervisor_launch.json").write_text(json.dumps({"pid": 333}), encoding="utf-8")
        prior = {"pid": 111, "process_token": 222, "captured_at": "original",
                 "launch_record": str(self.root / "supervisor_launch.json")}
        with mock.patch("satellite_rl.evaluation.post_training_handoff._process_token",
                        return_value=None):
            self.assertEqual(_capture_supervisor(self.root, prior)["pid"], 111)
        with mock.patch("satellite_rl.evaluation.post_training_handoff._process_token",
                        return_value=444):
            refreshed = _capture_supervisor(self.root, explicit_token=444)
            self.assertEqual((refreshed["pid"], refreshed["process_token"]), (333, 444))

    def test_late_start_accepts_previously_captured_explicit_token(self):
        (self.root / "supervisor_launch.json").write_text(json.dumps({"pid": 111}), encoding="utf-8")
        with mock.patch("satellite_rl.evaluation.post_training_handoff._process_token",
                        return_value=None):
            identity = _capture_supervisor(self.root, explicit_token=222)
            self.assertTrue(_supervisor_exited(identity))

    def test_timing_step_needs_all_18_results(self):
        post_root = self.root / "post_training_checks"
        runs = post_root / "timing" / "runs"
        runs.mkdir(parents=True)
        manifest = post_root / "timing" / "manifest.json"
        manifest.write_text(json.dumps({"training_commit": TRAIN_COMMIT}), encoding="utf-8")
        indexed = {}
        for run_id in EXPECTED_RUN_IDS:
            path = runs / (run_id + ".json")
            path.write_text("{}", encoding="utf-8")
            indexed[run_id] = {"path": str(path), "sha256": runner.sha256_file(path)}
        (post_root / "timing" / "result_index.json").write_text(
            json.dumps({"manifest": str(manifest),
                        "manifest_sha256": runner.sha256_file(manifest),
                        "runs": indexed}), encoding="utf-8")
        self.assertTrue(_outputs_complete("timing", self.root, post_root))
        next(iter(indexed.values()))["path"] += ".missing"
        (post_root / "timing" / "result_index.json").write_text(
            json.dumps({"manifest": str(manifest),
                        "manifest_sha256": runner.sha256_file(manifest),
                        "runs": indexed}), encoding="utf-8")
        self.assertFalse(_outputs_complete("timing", self.root, post_root))

    def test_final_analysis_recovery_checks_commit_and_input_hash(self):
        analysis = self.root / "analysis"
        analysis.mkdir()
        source = self.root / "input.json"
        source.write_text("input", encoding="utf-8")
        report = analysis / "中文补充分析.md"
        report.write_text("done", encoding="utf-8")
        manifest = {"training_frozen_commit": TRAIN_COMMIT,
                    "analysis_commit": "post-sha",
                    "generated_outputs": [str(report)],
                    "input_sha256": {str(source): runner.sha256_file(source)}}
        (analysis / "analysis_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.assertTrue(_outputs_complete("final_analysis", self.root, self.root,
                                          post_commit="post-sha"))
        source.write_text("changed", encoding="utf-8")
        self.assertFalse(_outputs_complete("final_analysis", self.root, self.root,
                                           post_commit="post-sha"))

    def test_completion_marker_and_resume_request_are_explicit(self):
        state = {"post_commit": "post-sha", "failed_step": "conditions",
                 "post_checks_completed_at": "2026-09-24T00:00:00Z"}
        marker = self.root / "all_tasks_complete.json"
        marker.write_text(json.dumps({"all_tasks_complete": True,
                                      "training_commit": TRAIN_COMMIT,
                                      "post_commit": "post-sha", "written_at": "2026-09-24T00:00:01Z"}),
                          encoding="utf-8")
        self.assertTrue(_marker_valid(marker, state))
        marker.write_text(json.dumps({"all_tasks_complete": True,
                                      "training_commit": TRAIN_COMMIT,
                                      "post_commit": "post-sha", "written_at": "2026-09-23T23:59:59Z"}),
                          encoding="utf-8")
        self.assertFalse(_marker_valid(marker, state))
        marker.write_text(json.dumps({"all_tasks_complete": True,
                                      "training_commit": "wrong", "post_commit": "post-sha"}),
                          encoding="utf-8")
        self.assertFalse(_marker_valid(marker, state))
        request = self.root / "resume_request.json"
        request.write_text(json.dumps({"request_id": "one", "resume": True,
                                       "step": "conditions", "post_commit": "post-sha"}),
                           encoding="utf-8")
        self.assertEqual(_resume_request(self.root, state)["request_id"], "one")
        state["last_resume_request_id"] = "one"
        self.assertIsNone(_resume_request(self.root, state))

    def test_sleep_is_bounded_by_remaining_deadline(self):
        with mock.patch("satellite_rl.evaluation.post_training_handoff.time.sleep") as sleep:
            _pause(datetime.now(timezone.utc) + timedelta(milliseconds=100), 600)
        self.assertGreaterEqual(sleep.call_args.args[0], 0)
        self.assertLessEqual(sleep.call_args.args[0], 0.1)

    def test_fixed_commands_include_no_training_entry(self):
        command = _command("conditions", "DQN-python", self.root, self.root,
                           self.root, self.root)
        self.assertIn("satellite_rl.evaluation.post_training_checks", command)
        self.assertNotIn("satellite_rl.training.experiment_runner", command)
        analysis = _command("final_analysis", "DQN-python", self.root, self.root,
                            self.root, self.root)
        self.assertIn("satellite_rl.evaluation.final_analysis", analysis)

    def test_failed_step_records_exit_and_does_not_mark_completed(self):
        child = mock.Mock(pid=999, returncode=1)
        child.poll.return_value = 1
        state = {"python": "DQN-python", "post_source_root": str(self.root),
                 "stage": "ready_for_post_checks",
                 "steps": {"conditions": {"state": "pending", "attempts": 0}}}
        with mock.patch("satellite_rl.evaluation.post_training_handoff.subprocess.Popen",
                        return_value=child), \
             mock.patch("satellite_rl.evaluation.post_training_handoff._process_token",
                        return_value=123), \
             mock.patch("satellite_rl.evaluation.post_training_handoff._outputs_complete",
                        return_value=False):
            with self.assertRaisesRegex(RuntimeError, "退出码1"):
                _run_step("conditions", state, self.root, self.root, self.root,
                          self.root, self.root,
                          datetime.now(timezone.utc) + timedelta(hours=1), 1)
        self.assertEqual(state["steps"]["conditions"]["state"], "failed")
        self.assertEqual(state["steps"]["conditions"]["exit_code"], 1)

    def test_launch_intent_is_durable_before_child_creation(self):
        state = {"python": "DQN-python", "post_source_root": str(self.root),
                 "stage": "ready_for_post_checks",
                 "steps": {"conditions": {"state": "pending", "attempts": 0}}}
        with mock.patch("satellite_rl.evaluation.post_training_handoff.subprocess.Popen",
                        side_effect=OSError("cannot start")):
            with self.assertRaisesRegex(OSError, "cannot start"):
                _run_step("conditions", state, self.root, self.root, self.root,
                          self.root, self.root,
                          datetime.now(timezone.utc) + timedelta(hours=1), 1)
        persisted = json.loads((self.root / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["steps"]["conditions"]["state"], "launching")
        self.assertIsNone(persisted["steps"]["conditions"]["pid"])


if __name__ == "__main__":
    unittest.main()
