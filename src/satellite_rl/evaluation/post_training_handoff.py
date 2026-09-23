"""One bounded background handoff after the approved 18-run SAC queue.

The process waits without GPU work, keeps Windows awake, then runs four fixed
post-training commands in order. A failed step waits for an explicit local
resume request; it never starts the next step automatically after failure.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

from satellite_rl.evaluation.post_training_checks import (
    EXPERIMENT_ID, EXPECTED_RUN_IDS, TRAIN_COMMIT,
    require_completed, source_guard, verify_cached_post_result,
)
from satellite_rl.training import experiment_runner as runner
from satellite_rl.training.experiment_suite import (
    CREATE_NEW_PROCESS_GROUP, CREATE_NO_WINDOW, _process_token,
    keep_system_awake,
)
from satellite_rl.training.run_monitor import atomic_json, utc_now


STEPS = ("conditions", "environment", "timing", "final_analysis")
MAX_HOURS = 72


@contextmanager
def _exclusive_manager_lock(path):
    """A kernel-held lock prevents two restarts from racing over a stale file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("后台衔接管理进程已持有独占锁") from error
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError("后台衔接管理进程已持有独占锁") from error
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _utc(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _event(directory, kind, **details):
    with (Path(directory) / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"at": utc_now(), "kind": kind, **details},
                                ensure_ascii=False) + "\n")


def _capture_supervisor(exp_dir, prior=None, explicit_token=None):
    """Bind this handoff to the exact queue process, never to a recycled PID."""
    launch = _read(Path(exp_dir) / "supervisor_launch.json")
    pid = int(launch["pid"])
    if prior is not None:
        if prior["pid"] != pid and _process_token(prior["pid"]) == prior["process_token"]:
            raise RuntimeError("原监督仍在运行，但启动记录已指向另一PID")
        token = int(prior["process_token"])
        pid = prior["pid"]
    elif explicit_token is not None:
        token = int(explicit_token)
    else:
        lock_path = Path(exp_dir) / "suite.lock"
        if not lock_path.exists():
            raise RuntimeError("首次启动须在训练监督运行时读取PID与创建时间，或提供已核验token")
        lock = _read(lock_path)
        if int(lock["pid"]) != pid:
            raise RuntimeError("队列锁的监督PID与启动记录不一致")
        token = int(lock["process_token"])
    if explicit_token is not None and int(explicit_token) != token:
        raise RuntimeError("显式监督创建时间与已存身份不一致")
    current = _process_token(pid)
    if (prior is None and current != token
            and not (explicit_token is not None and current is None)):
        raise RuntimeError("首次启动时监督进程身份已变化或退出")
    return {"pid": pid, "process_token": token,
            "launch_record": str(Path(exp_dir) / "supervisor_launch.json"),
            "captured_at": prior["captured_at"] if prior else utc_now()}


def _supervisor_exited(identity):
    """A different creation token means the original process has exited."""
    return _process_token(identity["pid"]) != identity["process_token"]


def _same_process(pid, token):
    return pid is not None and token is not None and _process_token(pid) == token


def _pause(deadline, poll_seconds):
    remaining = max(0.0, (deadline - datetime.now(timezone.utc)).total_seconds())
    time.sleep(min(poll_seconds, remaining))


def _suite_complete(exp_dir):
    summary = _read(Path(exp_dir) / "suite_status.json")
    counts = summary.get("counts", {})
    return (counts.get("completed") == 18 and counts.get("pending") == 0
            and counts.get("running") == 0 and counts.get("failed") == 0)


def _outputs_complete(step, exp_dir, post_root, *, post_commit=None):
    """Check committed outputs when a child exit code is unavailable."""
    try:
        if step == "conditions":
            for condition in ("demand_075", "demand_125", "within_group_order"):
                scene_manifest = _read(post_root / "scenes" / condition / "manifest.json")
                if post_commit is not None and scene_manifest["post_commit"] != post_commit:
                    return False
                refs = scene_manifest["scenes"]
                for run_id in EXPECTED_RUN_IDS:
                    task = post_root / "conditions" / condition / run_id
                    status = _read(task / "status.json")
                    config = _read(task / "manifest.json")
                    if (status.get("stage") != "completed"
                            or (post_commit is not None and config["post_commit"] != post_commit)):
                        return False
                    verify_cached_post_result(status["result_path"], refs)
            return True
        if step == "environment":
            source = _read(post_root / "environment_comparison" / "source_manifest.json")
            if post_commit is not None and source["post_commit"] != post_commit:
                return False
            refs = source["scenes"]
            for label in ("before_c4963ba", "corrected_r2"):
                task = post_root / "environment_comparison" / label
                status = _read(task / "status.json")
                manifest = _read(task / "manifest.json")
                if (status.get("stage") != "completed"
                        or manifest.get("weight_training_steps") != "unknown"
                        or (post_commit is not None and manifest["post_commit"] != post_commit)):
                    return False
                verify_cached_post_result(status["result_path"], refs,
                                          expected_step="unknown")
            return True
        if step == "timing":
            index = _read(post_root / "timing" / "result_index.json")
            manifest_path = Path(index["manifest"])
            timing_manifest = _read(manifest_path)
            if (runner.sha256_file(manifest_path) != index["manifest_sha256"]
                    or timing_manifest["training_commit"] != TRAIN_COMMIT
                    or (post_commit is not None
                        and timing_manifest["post_commit"] != post_commit)):
                return False
            indexed = index["runs"]
            return set(indexed) == EXPECTED_RUN_IDS and all(
                runner.sha256_file(value["path"]) == value["sha256"]
                for value in indexed.values())
        if step == "final_analysis":
            output = Path(exp_dir) / "analysis"
            manifest = _read(output / "analysis_manifest.json")
            if (manifest["training_frozen_commit"] != TRAIN_COMMIT
                    or (post_commit is not None and manifest["analysis_commit"] != post_commit)
                    or not all(Path(path).is_file() for path in manifest["generated_outputs"])):
                return False
            return all(runner.sha256_file(path) == digest
                       for path, digest in manifest["input_sha256"].items())
        raise ValueError(step)
    except (OSError, ValueError, KeyError, RuntimeError):
        return False


def _command(step, python, exp_dir, archive_root, training_root, post_root):
    if step == "final_analysis":
        return [python, "-u", "-m", "satellite_rl.evaluation.final_analysis", str(exp_dir)]
    return [python, "-u", "-m", "satellite_rl.evaluation.post_training_checks",
            step, "--archive-root", str(archive_root),
            "--frozen-root", str(training_root)]


def _save_failure(handoff_dir, state, error):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = handoff_dir / f"failure_{stamp}_{uuid.uuid4().hex[:8]}.json"
    atomic_json(path, {"at": utc_now(), "stage": state["stage"],
                       "type": type(error).__name__, "message": str(error),
                       "traceback": traceback.format_exc()})
    atomic_json(handoff_dir / "failure.json", {"latest_failure": str(path),
                                                  "at": utc_now()})
    state["latest_failure"] = str(path)
    return path


def _marker_valid(path, state):
    if not Path(path).is_file():
        return False
    try:
        marker = _read(path)
        newer = _utc(marker["written_at"]) > _utc(state["post_checks_completed_at"])
    except (OSError, KeyError, ValueError, TypeError):
        return False
    return (marker.get("all_tasks_complete") is True
            and marker.get("training_commit") == TRAIN_COMMIT
            and marker.get("post_commit") == state["post_commit"]
            and newer)


def _resume_request(handoff_dir, state):
    path = handoff_dir / "resume_request.json"
    if not path.is_file():
        return None
    request = _read(path)
    request_id = request.get("request_id")
    if not request_id or request_id == state.get("last_resume_request_id"):
        return None
    if (request.get("resume") is not True
            or request.get("step") != state.get("failed_step")
            or request.get("post_commit") != state["post_commit"]):
        raise RuntimeError("恢复请求与当前失败步骤/后处理提交不匹配")
    return request


def _run_step(step, state, handoff_dir, exp_dir, archive_root, training_root,
              post_root, deadline, poll_seconds):
    python = state["python"]
    command = _command(step, python, exp_dir, archive_root, training_root, post_root)
    environment = os.environ.copy()
    environment.update({"PYTHONPATH": str(Path(state["post_source_root"]) / "src"),
                        "SATELLITE_RL_OUTPUT_ROOT": str(archive_root),
                        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                        "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
                        "MPLBACKEND": "Agg"})
    record = state["steps"][step]
    attempt = record["attempts"] + 1
    log_path = handoff_dir / f"{step}_attempt{attempt}_{uuid.uuid4().hex[:8]}.log"
    record.update(state="launching", attempts=attempt, pid=None,
                  process_token=None, log=str(log_path), command=command,
                  launch_intent_at=utc_now(), exit_code=None)
    state["stage"] = "running_" + step
    atomic_json(handoff_dir / "state.json", state)
    _event(handoff_dir, "step_launch_intent", step=step, attempt=attempt,
           log=str(log_path))
    with log_path.open("wb") as output:
        child = subprocess.Popen(
            command, cwd=state["post_source_root"], env=environment,
            stdout=output, stderr=subprocess.STDOUT,
            creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
        )
    record.update(state="running", pid=child.pid,
                  process_token=_process_token(child.pid), log=str(log_path),
                  command=command, started_at=utc_now(), exit_code=None)
    atomic_json(handoff_dir / "state.json", state)
    _event(handoff_dir, "step_started", step=step, pid=child.pid,
           attempt=attempt, log=str(log_path))
    while True:
        code = child.poll()
        if code is not None:
            record.update(pid=None, process_token=None, exit_code=code,
                          ended_at=utc_now())
            if code != 0 or not _outputs_complete(
                    step, exp_dir, post_root, post_commit=state["post_commit"]):
                record["state"] = "failed"
                raise RuntimeError(f"{step} 退出码{code}或输出不完整；日志：{log_path}")
            record["state"] = "completed"
            atomic_json(handoff_dir / "state.json", state)
            _event(handoff_dir, "step_completed", step=step, exit_code=code,
                   log=str(log_path))
            return
        if datetime.now(timezone.utc) >= deadline:
            if _same_process(child.pid, record["process_token"]):
                child.terminate()
                child.wait(timeout=30)
            record.update(state="timed_out", pid=None, process_token=None,
                          exit_code=child.returncode, ended_at=utc_now())
            raise TimeoutError(f"72小时时限到达，已停止当前{step}子进程")
        state["heartbeat_at"] = utc_now()
        atomic_json(handoff_dir / "state.json", state)
        _pause(deadline, poll_seconds)


def run_handoff(archive_root, training_root, *, supervisor_token=None,
                poll_seconds=60, max_hours=72):
    if not 1 <= max_hours <= MAX_HOURS or not 1 <= poll_seconds <= 600:
        raise ValueError("最长驻留须为1–72小时，轮询为1–600秒")
    archive_root, training_root = Path(archive_root).resolve(), Path(training_root).resolve()
    exp_dir = archive_root / "outputs" / "experiments" / EXPERIMENT_ID
    post_root = exp_dir / "post_training_checks"
    handoff_dir = post_root / "handoff"
    handoff_dir.mkdir(parents=True, exist_ok=True)
    provenance = source_guard(training_root)
    python = str(Path(sys.executable).resolve())
    if "dqn" not in (part.lower() for part in Path(python).parts):
        raise RuntimeError(f"后台衔接必须由DQN解释器运行：{python}")
    with _exclusive_manager_lock(handoff_dir / "manager.lock"):
        state_path = handoff_dir / "state.json"
        prior = _read(state_path) if state_path.exists() else None
        identity = _capture_supervisor(exp_dir,
                                       prior["supervisor"] if prior else None,
                                       supervisor_token)
        if prior:
            state = prior
            if (state["post_commit"] != provenance["post_commit"]
                    or Path(state["post_source_root"]).resolve() != Path(provenance["post_source_root"])
                    or state["supervisor"] != identity):
                raise RuntimeError("现有衔接状态的源码或监督身份不匹配")
            if state["stage"].startswith("running_"):
                state["stage"] = "ready_for_post_checks"
        else:
            started = datetime.now(timezone.utc)
            state = {"format_version": 1, "stage": "waiting_training",
                     "started_at": started.isoformat(),
                     "deadline_at": (started + timedelta(hours=max_hours)).isoformat(),
                     "heartbeat_at": utc_now(), "manager_pid": os.getpid(),
                     "manager_process_token": _process_token(os.getpid()),
                     "post_commit": provenance["post_commit"],
                     "post_source_root": provenance["post_source_root"],
                     "training_commit": TRAIN_COMMIT, "training_root": str(training_root),
                     "archive_root": str(archive_root), "python": python,
                     "supervisor": identity,
                     "steps": {name: {"state": "pending", "attempts": 0}
                               for name in STEPS}}
            atomic_json(state_path, state)
        deadline = _utc(state["deadline_at"])
        with keep_system_awake():
            state["manager_pid"] = os.getpid()
            state["manager_process_token"] = _process_token(os.getpid())
            atomic_json(state_path, state)
            _event(handoff_dir, "manager_started", pid=os.getpid(),
                   supervisor_pid=identity["pid"], deadline_at=state["deadline_at"])
            if state["stage"] == "complete":
                return 0
            while datetime.now(timezone.utc) < deadline:
                if state["stage"] == "waiting_training":
                    if _supervisor_exited(state["supervisor"]):
                        if not _suite_complete(exp_dir):
                            state["stage"] = "waiting_for_repair"
                            state["failed_step"] = "training_supervisor"
                            state["failure_reason"] = "原监督身份已退出，但18组未全部完成"
                        else:
                            try:
                                require_completed(archive_root, training_root)
                                state["stage"] = "ready_for_post_checks"
                                _event(handoff_dir, "training_completed_and_supervisor_exited",
                                       supervisor=state["supervisor"])
                            except Exception as error:
                                state["stage"] = "waiting_for_repair"
                                state["failed_step"] = "training_supervisor"
                                _save_failure(handoff_dir, state, error)
                        atomic_json(state_path, state)
                    else:
                        state["heartbeat_at"] = utc_now()
                        atomic_json(state_path, state)
                        _pause(deadline, poll_seconds)
                        continue
                if state["stage"] == "waiting_for_repair":
                    orphan = state.get("orphan_monitor")
                    if orphan:
                        record = state["steps"][orphan]
                        if _same_process(record.get("pid"), record.get("process_token")):
                            state["heartbeat_at"] = utc_now()
                            atomic_json(state_path, state)
                            _pause(deadline, poll_seconds)
                            continue
                        record.update(pid=None, process_token=None, ended_at=utc_now())
                        state.pop("orphan_monitor", None)
                        if _outputs_complete(orphan, exp_dir, post_root,
                                             post_commit=state["post_commit"]):
                            record["state"] = "completed"
                            state["stage"] = "ready_for_post_checks"
                            state.pop("failed_step", None)
                            state.pop("failure_reason", None)
                            atomic_json(state_path, state)
                            _event(handoff_dir, "orphan_step_completed", step=orphan)
                            continue
                        record["state"] = "failed"
                        state["failure_reason"] = "已知子进程退出但输出不完整，等待明确恢复请求"
                        atomic_json(state_path, state)
                    try:
                        request = _resume_request(handoff_dir, state)
                    except Exception as error:
                        if state.get("last_resume_error") != str(error):
                            _save_failure(handoff_dir, state, error)
                            state["last_resume_error"] = str(error)
                        request = None
                    if request:
                        try:
                            if state["failed_step"] == "training_supervisor":
                                token = request.get("supervisor_token")
                                if token is None:
                                    raise RuntimeError("训练监督恢复请求缺少新创建时间token")
                                new_identity = _capture_supervisor(
                                    exp_dir, explicit_token=token)
                                state["supervisor"] = new_identity
                                state["stage"] = "waiting_training"
                            else:
                                record = state["steps"][state["failed_step"]]
                                record.update(state="pending", pid=None, process_token=None)
                                state["stage"] = "ready_for_post_checks"
                        except Exception as error:
                            if state.get("last_resume_error") != str(error):
                                _save_failure(handoff_dir, state, error)
                                state["last_resume_error"] = str(error)
                            state["heartbeat_at"] = utc_now()
                            atomic_json(state_path, state)
                            _pause(deadline, poll_seconds)
                            continue
                        state["last_resume_request_id"] = request["request_id"]
                        state.pop("last_resume_error", None)
                        state.pop("failed_step", None)
                        state.pop("failure_reason", None)
                        atomic_json(state_path, state)
                        _event(handoff_dir, "resume_requested", request_id=request["request_id"])
                        continue
                    state["heartbeat_at"] = utc_now()
                    atomic_json(state_path, state)
                    _pause(deadline, poll_seconds)
                    continue
                if state["stage"] == "ready_for_post_checks":
                    pending = next((name for name in STEPS
                                    if state["steps"][name]["state"] != "completed"), None)
                    if pending is None:
                        state["stage"] = "awaiting_state_diagnosis"
                        state["post_checks_completed_at"] = utc_now()
                        atomic_json(state_path, state)
                        _event(handoff_dir, "post_checks_completed")
                        continue
                    if state["steps"][pending]["state"] == "launching":
                        state["stage"] = "waiting_for_repair"
                        state["failed_step"] = pending
                        state["failure_reason"] = "已记录启动意图但未记录子进程PID，拒绝可能的重复派发"
                        atomic_json(state_path, state)
                        continue
                    if state["steps"][pending]["state"] == "running":
                        record = state["steps"][pending]
                        if _same_process(record.get("pid"), record.get("process_token")):
                            state["stage"] = "waiting_for_repair"
                            state["failed_step"] = pending
                            state["orphan_monitor"] = pending
                            state["failure_reason"] = "管理进程重启后等待已知子作业结束，拒绝重复启动"
                            atomic_json(state_path, state)
                            continue
                        if _outputs_complete(pending, exp_dir, post_root,
                                             post_commit=state["post_commit"]):
                            record.update(state="completed", pid=None, process_token=None,
                                          ended_at=utc_now(), recovered_after_manager_restart=True)
                            atomic_json(state_path, state)
                            continue
                        state["stage"] = "waiting_for_repair"
                        state["failed_step"] = pending
                        state["failure_reason"] = "管理进程曾中断，子作业已退出但输出不完整；拒绝自动重复运行"
                        record["state"] = "failed"
                        atomic_json(state_path, state)
                        continue
                    try:
                        _run_step(pending, state, handoff_dir, exp_dir, archive_root,
                                  training_root, post_root, deadline, poll_seconds)
                        state["stage"] = "ready_for_post_checks"
                        atomic_json(state_path, state)
                    except TimeoutError as error:
                        state["stage"] = "timed_out"
                        state["failed_step"] = pending
                        _save_failure(handoff_dir, state, error)
                        atomic_json(state_path, state)
                        break
                    except Exception as error:
                        state["stage"] = "waiting_for_repair"
                        state["failed_step"] = pending
                        _save_failure(handoff_dir, state, error)
                        atomic_json(state_path, state)
                    continue
                if state["stage"] == "awaiting_state_diagnosis":
                    marker = handoff_dir / "all_tasks_complete.json"
                    if _marker_valid(marker, state):
                        state["stage"] = "complete"
                        state["completed_at"] = utc_now()
                        atomic_json(state_path, state)
                        _event(handoff_dir, "all_tasks_complete", marker=str(marker))
                        return 0
                    state["heartbeat_at"] = utc_now()
                    atomic_json(state_path, state)
                    _pause(deadline, poll_seconds)
                    continue
                if state["stage"] == "complete":
                    return 0
                state["stage"] = "waiting_for_repair"
                state["failed_step"] = state.get("failed_step", "manager_state")
                state["failure_reason"] = "未识别管理状态，等待人工检查"
                atomic_json(state_path, state)
            if state["stage"] != "complete":
                state["stage"] = "timed_out"
                state["timed_out_at"] = utc_now()
                atomic_json(state_path, state)
                _event(handoff_dir, "deadline_reached", deadline_at=state["deadline_at"])
            return 3


def main():
    parser = argparse.ArgumentParser(description="18组训练结束后的单进程后台衔接")
    parser.add_argument("--archive-root", default="D:/paper")
    parser.add_argument("--training-root", default="D:/paper_worktrees/frozen-sac-20260924-r2")
    parser.add_argument("--supervisor-token", type=int)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-hours", type=int, default=72)
    args = parser.parse_args()
    raise SystemExit(run_handoff(args.archive_root, args.training_root,
                                 supervisor_token=args.supervisor_token,
                                 poll_seconds=args.poll_seconds, max_hours=args.max_hours))


if __name__ == "__main__":
    main()
