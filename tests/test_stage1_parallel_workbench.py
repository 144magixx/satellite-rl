"""Dispatch safety for the external S1-02 two-train workbench launcher."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.runtime import Scheduler
from satellite_rl.workbench.storage import Store, read_json


LAUNCHER = (Path(__file__).resolve().parents[1] / "scripts" /
            "stage1_balance" / "parallel_workbench.py")
SPEC = importlib.util.spec_from_file_location("stage1_parallel_workbench", LAUNCHER)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ParallelDispatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = Store(Path(self.temporary.name))
        self.scheduler = MODULE.ParallelCoverageScheduler(self.store)
        self.alive = set()
        self.launched = []

    def job(self, run_id, kind, stage="queued", pid=None, order=0):
        directory = self.store.job_dir(run_id)
        directory.mkdir()
        record = {"run_id": run_id, "kind": kind,
                  "created_at": f"2026-10-04T00:00:{order:02d}Z"}
        atomic_json(directory / "job.json", record)
        atomic_json(directory / "status.json", {"stage": stage, "pid": pid})
        if pid:
            atomic_json(directory / "launch.json", {"pid": pid,
                        "process_start_time": float(pid)})
            self.alive.add(pid)
        return record

    def tick(self):
        def fake_launch(scheduler, job, status):
            pid = 900 + len(self.launched)
            self.launched.append(job["run_id"])
            self.alive.add(pid)
            directory = self.store.job_dir(job["run_id"])
            atomic_json(directory / "status.json", {"stage": "starting", "pid": pid})
            atomic_json(directory / "launch.json", {"pid": pid,
                        "process_start_time": float(pid)})

        with mock.patch.object(Scheduler, "_launch", fake_launch), \
             mock.patch("satellite_rl.workbench.runtime.pid_alive",
                        side_effect=lambda pid, expected_start_time=None: pid in self.alive), \
             mock.patch.object(MODULE, "pid_alive",
                        side_effect=lambda pid, expected_start_time=None: pid in self.alive), \
             mock.patch("satellite_rl.workbench.runtime.gpu_snapshot",
                        return_value={"memory_total_mb": 24576, "memory_used_mb": 1000,
                                      "utilization_percent": 10, "compute_pids": []}):
            self.scheduler.tick()

    def test_two_training_jobs_can_launch_in_one_tick(self):
        self.job("train_a", "train", order=1)
        self.job("train_b", "train", order=2)
        self.tick()
        self.assertEqual(self.launched, ["train_a", "train_b"])

    def test_two_evaluations_launch_only_first(self):
        self.job("eval_a", "evaluate", order=1)
        self.job("eval_b", "evaluate", order=2)
        self.tick()
        self.assertEqual(self.launched, ["eval_a"])

    def test_last_train_to_evaluation_race_stays_exclusive(self):
        self.job("active_train", "train", stage="training", pid=701, order=0)
        self.job("queued_eval", "evaluate", order=1)
        self.job("queued_train", "train", order=2)
        self.tick()
        self.assertEqual(self.launched, ["queued_train"])
        self.alive.remove(701)
        atomic_json(self.store.job_dir("active_train") / "status.json",
                    {"stage": "completed", "pid": 701})
        self.tick()
        self.assertEqual(self.launched, ["queued_train"])
        self.alive.remove(900)
        atomic_json(self.store.job_dir("queued_train") / "status.json",
                    {"stage": "completed", "pid": 900})
        self.tick()
        self.assertEqual(self.launched, ["queued_train", "queued_eval"])

    def test_restart_takes_over_live_pid_without_relaunch(self):
        self.job("existing_train", "train", stage="training", pid=731, order=1)
        self.job("next_train", "train", order=2)
        self.tick()
        self.assertEqual(self.launched, ["next_train"])
        self.assertEqual(read_json(self.store.job_dir("existing_train") /
                                   "status.json")["stage"], "training")

    def test_evaluation_wins_when_last_train_finishes_before_tick(self):
        self.job("last_train", "train", stage="training", pid=741, order=0)
        self.job("waiting_eval", "evaluate", order=1)
        self.job("later_train", "train", order=2)
        self.alive.remove(741)
        atomic_json(self.store.job_dir("last_train") / "status.json",
                    {"stage": "completed", "pid": 741})
        self.tick()
        self.assertEqual(self.launched, ["waiting_eval"])


if __name__ == "__main__":
    unittest.main()
