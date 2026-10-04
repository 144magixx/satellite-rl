"""Workbench launcher: bounded parallel training or one evaluation worker.

The HTTP API, frozen jobs, and scientific worker code are the existing
workbench implementation.  Only this process's dispatch decision changes.
Run with ``--server-marker satellite_rl.workbench.server`` so operations can
identify the server process by its exact argv before stopping it.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import signal
import sys
from datetime import datetime, timezone

from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.runtime import Scheduler, pid_alive, process_start_time
from satellite_rl.workbench.server import WorkbenchServer
from satellite_rl.workbench.service import Application
from satellite_rl.workbench.storage import read_json


DEFAULT_ROOT = Path("/home/u2024110224/first_stage_satellite_rl/s1_02/20418")
DEFAULT_PYTHON = Path("/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python")
POLICY = "stage1_parallel_train_exclusive_evaluation_v2"
MARKER = "satellite_rl.workbench.server"


class ParallelCoverageScheduler(Scheduler):
    """Recheck live ownership at launch, after the base tick selects a queue."""

    def __init__(self, store, python_executable=sys.executable, max_training_workers=2):
        if type(max_training_workers) is not int or not 1 <= max_training_workers <= 5:
            raise ValueError("Training concurrency must be an integer from 1 to 5")
        super().__init__(store, python_executable=python_executable,
                         max_workers=max_training_workers)

    def _live_jobs(self):
        live = []
        for job, status in self.store.jobs():
            directory = self.store.job_dir(job["run_id"])
            launch = read_json(directory / "launch.json", {})
            expected = launch.get("process_start_time")
            pids = {status.get("pid"), launch.get("pid")}
            child = self.children.get(job["run_id"])
            owned = child is not None and child.poll() is None
            if owned or any(pid_alive(pid, expected) for pid in pids if pid):
                live.append(job)
        return live

    def _launch(self, job, status):
        live = self._live_jobs()
        if any(active["run_id"] == job["run_id"] for active in live):
            return  # A still-running PID cannot be launched again after restart.
        if job["kind"] == "train":
            if any(active["kind"] != "train" for active in live) or len(live) >= self.max_workers:
                return
        elif live:
            # Base tick inspects only as many queued jobs as free slots.  If
            # an older evaluation is blocked behind a live train, use the
            # spare slot for the oldest queued training job instead.
            if len(live) < self.max_workers and all(active["kind"] == "train" for active in live):
                for candidate, candidate_status in reversed(self.store.jobs()):
                    if candidate_status.get("stage") == "queued" and candidate["kind"] == "train":
                        if all(active["run_id"] != candidate["run_id"] for active in live):
                            super()._launch(candidate, candidate_status)
                        return
            return  # All evaluation/refinement work is exclusive on this GPU.
        super()._launch(job, status)


@contextmanager
def exclusive_ops_lock(path):
    import fcntl  # The approved S1-02 workbench host is Linux.

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another S1-02 workbench owns {path}") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--port", type=int, default=8892)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--max-training-workers", type=int, choices=range(1, 6), default=2)
    parser.add_argument("--server-marker", choices=(MARKER,), required=True)
    args = parser.parse_args(argv)
    archive = args.archive_root.resolve()
    source = args.source_root.resolve()
    python = args.python.resolve()
    if not python.is_file():
        raise ValueError(f"DQN interpreter missing: {python}")
    ops = archive / "ops"
    with exclusive_ops_lock(ops / "parallel_workbench.lock"):
        app = Application(archive, source_root=source, python_executable=python,
                          max_workers=args.max_training_workers)
        app.scheduler = ParallelCoverageScheduler(
            app.store, python_executable=python,
            max_training_workers=args.max_training_workers)
        server = WorkbenchServer(("127.0.0.1", args.port), app)
        state = ops / "parallel_workbench_state.json"
        metadata = {"policy": POLICY, "status": "running", "pid": os.getpid(),
                    "process_start_time": process_start_time(os.getpid()),
                    "port": server.server_port, "archive_root": str(archive),
                    "source_root": str(source), "python": str(python),
                    "max_training_workers": args.max_training_workers, "max_nontraining_workers": 1,
                    "server_marker": args.server_marker,
                    "started_at": datetime.now(timezone.utc).isoformat()}
        atomic_json(state, metadata)

        def stop_server(_signum, _frame):
            raise KeyboardInterrupt

        old_handler = signal.signal(signal.SIGTERM, stop_server)
        try:
            app.scheduler.start()
            print(f"S1-02 workbench http://127.0.0.1:{server.server_port}/", flush=True)
            server.serve_forever(poll_interval=0.5)
        except KeyboardInterrupt:
            pass
        finally:
            app.scheduler.stop_scheduler()  # Does not signal any worker PID.
            server.server_close()
            signal.signal(signal.SIGTERM, old_handler)
            atomic_json(state, {**metadata, "status": "stopped",
                                "stopped_at": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
