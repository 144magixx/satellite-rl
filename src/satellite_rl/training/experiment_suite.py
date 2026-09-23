"""Durable supervisor for the 18 paired SAC CUDA experiments.

Prepare once in a frozen checkout, then run the queue from that same checkout.
Each worker writes its own checkpoints; the supervisor only launches and observes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from satellite_rl.training.experiment_runner import (
    ACTION_SEEDS, MODELS, TEST_SCENE_SEEDS, TRAIN_SEEDS,
    VALIDATION_SCENE_SEEDS, prepare_fixed_scenes,
    strict_cuda_preflight,
)
from satellite_rl.training.run_monitor import atomic_json, utc_now


FORMULAS = ("legacy", "corrected")
DEFAULT_EXPERIMENT_ID = "sac_comparison_20260924"
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)


def suite_dir(archive_root, experiment_id):
    return Path(archive_root).resolve() / "outputs" / "experiments" / experiment_id


def deterministic_jobs():
    """Keep paired formulas adjacent and cycle models within each seed."""
    return [
        {"run_id": f"{model}_{formula}_s{seed}", "model": model,
         "formula": formula, "seed": seed, "state": "pending",
         "attempts": 0, "pid": None, "process_token": None,
         "started_at": None, "ended_at": None, "last_exit_code": None,
         "last_error": None}
        for seed in TRAIN_SEEDS for model in MODELS for formula in FORMULAS
    ]


def _process_token(pid):
    """Windows creation time distinguishes a restarted worker from a recycled PID."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = (
        wintypes.HANDLE, ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME))
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return None
    try:
        exit_code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, exit_code) or exit_code.value != 259:
            return None
        created = wintypes.FILETIME()
        exited = wintypes.FILETIME()
        kernel_time = wintypes.FILETIME()
        user_time = wintypes.FILETIME()
        if not kernel.GetProcessTimes(handle, created, exited, kernel_time, user_time):
            return None
        return (created.dwHighDateTime << 32) | created.dwLowDateTime
    finally:
        kernel.CloseHandle(handle)


def process_alive(pid, token=None):
    if not pid:
        return False
    if os.name == "nt":
        current = _process_token(pid)
        return current is not None and (token is None or current == token)
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


class SuiteLock:
    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        if self.path.exists():
            try:
                old = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                raise RuntimeError(f"队列锁文件损坏，需检查：{self.path}") from None
            if process_alive(old.get("pid"), old.get("process_token")):
                raise RuntimeError(f"队列监督进程仍在运行：PID {old['pid']}")
            self.path.unlink()
        payload = {"pid": os.getpid(), "process_token": _process_token(os.getpid()),
                   "created_at": utc_now()}
        with self.path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream)
        return self

    def __exit__(self, *_):
        try:
            old = json.loads(self.path.read_text(encoding="utf-8"))
            if old.get("pid") == os.getpid():
                self.path.unlink()
        except FileNotFoundError:
            pass


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _event(directory, kind, **details):
    with (directory / "suite_events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"at": utc_now(), "kind": kind, **details},
                                ensure_ascii=False, allow_nan=False) + "\n")


def _source_root():
    return Path(__file__).resolve().parents[3]


def _verify_source(plan):
    runtime = strict_cuda_preflight()
    if runtime["git_commit"] != plan["source"]["git_commit"]:
        raise RuntimeError("冻结源码提交已改变，拒绝继续启动作业")
    if runtime["source_root"] != plan["source"]["source_root"]:
        raise RuntimeError("监督进程没有运行在原冻结源码目录")
    if runtime["python"] != plan["source"]["python"]:
        raise RuntimeError("DQN 解释器已改变，拒绝混用运行环境")
    if runtime["cities_data"]["sha256"] != plan["source"]["cities_data"]["sha256"]:
        raise RuntimeError("输入城市数据已改变，拒绝继续启动作业")


def _validate_plan(plan):
    expected = deterministic_jobs()
    expected_ids = [job["run_id"] for job in expected]
    actual_ids = [job["run_id"] for job in plan["jobs"]]
    if actual_ids != expected_ids or len(set(actual_ids)) != 18:
        raise RuntimeError("实验队列不是约定的 18 组配置")
    if plan["steps"] != 400000:
        raise RuntimeError("正式对照须保持 400000 环境交互步")
    if plan["validation_interval"] != 10000 or plan["checkpoint_interval"] != 10000:
        raise RuntimeError("正式验证与断点间隔须保持 10000 步")
    for actual, wanted in zip(plan["jobs"], expected):
        if any(actual[key] != wanted[key] for key in ("model", "formula", "seed")):
            raise RuntimeError(f"作业配置被更改：{actual['run_id']}")


def prepare(archive_root, experiment_id=DEFAULT_EXPERIMENT_ID):
    runtime = strict_cuda_preflight()
    directory = suite_dir(archive_root, experiment_id)
    directory.mkdir(parents=True, exist_ok=True)
    plan_path = directory / "suite_plan.json"
    if plan_path.exists():
        plan = _load_json(plan_path)
        _validate_plan(plan)
        _verify_source(plan)
        return plan
    # Create shared scene files serially, before concurrent workers inspect them.
    validation = prepare_fixed_scenes(directory, "validation", VALIDATION_SCENE_SEEDS)
    test = prepare_fixed_scenes(directory, "test", TEST_SCENE_SEEDS)
    plan = {"format_version": 1, "experiment_id": experiment_id,
            "archive_root": str(Path(archive_root).resolve()),
            "created_at": utc_now(), "source": runtime,
            "steps": 400000, "validation_interval": 10000,
            "checkpoint_interval": 10000,
            "validation_scenes": validation, "test_scenes": test,
            "action_seeds": list(ACTION_SEEDS), "jobs": deterministic_jobs()}
    atomic_json(plan_path, plan)
    _event(directory, "prepared", source_commit=runtime["git_commit"], jobs=18)
    write_summary(directory, plan, active_limit=None)
    return plan


def _worker_status(directory, job):
    path = directory / job["run_id"] / "status.json"
    try:
        return _load_json(path)
    except (FileNotFoundError, ValueError):
        return {}


def _worker_completed(directory, job, plan):
    status = _worker_status(directory, job)
    if status.get("stage") != "completed" or status.get("env_step") != plan["steps"]:
        return False
    manifest_path = directory / job["run_id"] / "manifest.json"
    result_path = directory / job["run_id"] / "evaluation" / f"final_test_{plan['steps']}.json"
    if not manifest_path.exists() or not result_path.exists():
        return False
    config = _load_json(manifest_path)["config"]
    return (config["model"] == job["model"] and
            config["formula"] == job["formula"] and
            config["train_seed"] == job["seed"] and
            config["runtime"]["git_commit"] == plan["source"]["git_commit"])


def write_summary(directory, plan, active_limit):
    counts = {state: sum(job["state"] == state for job in plan["jobs"])
              for state in ("pending", "running", "completed", "failed")}
    jobs = []
    for job in plan["jobs"]:
        worker = _worker_status(directory, job)
        jobs.append({key: job.get(key) for key in (
            "run_id", "model", "formula", "seed", "state", "attempts", "pid",
            "started_at", "ended_at", "last_exit_code", "last_error")}
            | {"stage": worker.get("stage"), "env_step": worker.get("env_step", 0),
               "update_step": worker.get("update_step", 0),
               "heartbeat_at": worker.get("heartbeat_at"),
               "eta_seconds": worker.get("eta_seconds")})
    snapshot = {"updated_at": utc_now(), "source_commit": plan["source"]["git_commit"],
                "experiment_id": plan["experiment_id"], "total_jobs": 18,
                "budget_env_steps_per_job": plan["steps"],
                "active_limit": active_limit, "counts": counts,
                "all_completed": counts["completed"] == 18,
                "report_path": str(directory / "中文实验对照报告.md")
                if (directory / "中文实验对照报告.md").exists() else None,
                "jobs": jobs}
    atomic_json(directory / "suite_status.json", snapshot)
    return snapshot


def _worker_command(plan, job, resume):
    command = [plan["source"]["python"], "-u", "-m",
               "satellite_rl.training.experiment_runner",
               "--model", job["model"], "--formula", job["formula"],
               "--seed", str(job["seed"]), "--steps", str(plan["steps"]),
               "--archive-root", plan["archive_root"],
               "--experiment-id", plan["experiment_id"],
               "--validation-interval", str(plan["validation_interval"]),
               "--checkpoint-interval", str(plan["checkpoint_interval"]),
               "--run-id", job["run_id"]]
    if resume:
        command.append("--resume")
    return command


def _launch(directory, plan, job):
    run_dir = directory / job["run_id"]
    checkpoint = (Path(plan["archive_root"]) / "models" / "runs" /
                  plan["experiment_id"] / job["run_id"] / "latest.pt")
    resume = (run_dir / "manifest.json").exists() and checkpoint.exists()
    if run_dir.exists() and not resume:
        raise RuntimeError("运行目录已有内容但缺少可恢复断点；保留现场待检查")
    log_dir = directory / "supervisor_logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / (job["run_id"] + ".log")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(plan["source"]["source_root"]) / "src")
    env["SATELLITE_RL_OUTPUT_ROOT"] = plan["archive_root"]
    command = _worker_command(plan, job, resume)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{utc_now()}] {'RESUME' if resume else 'START'} {command!r}\n")
        log.flush()
        child = subprocess.Popen(
            command, cwd=plan["source"]["source_root"], env=env,
            stdout=log, stderr=subprocess.STDOUT,
            creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
        )
    job.update(state="running", attempts=job["attempts"] + 1,
               pid=child.pid, process_token=_process_token(child.pid),
               started_at=utc_now(), ended_at=None, last_exit_code=None,
               last_error=None)
    _event(directory, "launched", run_id=job["run_id"], pid=child.pid,
           attempt=job["attempts"], resume=resume, log=str(log_path))
    return child


def _failure(directory, plan, job, error, exit_code, max_attempts):
    job.update(pid=None, process_token=None, ended_at=utc_now(),
               last_exit_code=exit_code, last_error=str(error))
    job["state"] = "pending" if job["attempts"] < max_attempts else "failed"
    _event(directory, "job_failed", run_id=job["run_id"],
           attempt=job["attempts"], next_state=job["state"],
           exit_code=exit_code, error=str(error))


def _refresh(directory, plan, processes, max_attempts):
    for job in plan["jobs"]:
        if job["state"] != "running":
            continue
        child = processes.get(job["run_id"])
        exit_code = child.poll() if child else None
        alive = (exit_code is None) if child else process_alive(
            job["pid"], job.get("process_token"))
        if alive:
            continue
        processes.pop(job["run_id"], None)
        if _worker_completed(directory, job, plan):
            job.update(state="completed", pid=None, process_token=None,
                       ended_at=utc_now(), last_exit_code=exit_code or 0)
            _event(directory, "job_completed", run_id=job["run_id"])
        else:
            status = _worker_status(directory, job)
            reason = f"worker exited; status={status.get('stage', 'missing')}"
            _failure(directory, plan, job, reason, exit_code, max_attempts)


def run_queue(archive_root, experiment_id=DEFAULT_EXPERIMENT_ID,
              *, max_concurrent=3, retries=3, poll_seconds=5):
    if not 1 <= max_concurrent <= 3:
        raise ValueError("max_concurrent 必须在 1 至 3 之间")
    if retries < 0 or poll_seconds <= 0:
        raise ValueError("重试次数须非负，轮询间隔须为正")
    directory = suite_dir(archive_root, experiment_id)
    plan = _load_json(directory / "suite_plan.json")
    _validate_plan(plan)
    if str(Path(archive_root).resolve()) != plan["archive_root"]:
        raise RuntimeError("归档根目录与已准备的队列不一致")
    _verify_source(plan)
    processes = {}
    max_attempts = retries + 1
    with SuiteLock(directory / "suite.lock"):
        from satellite_rl.evaluation.experiment_report import generate_progress_report

        _event(directory, "supervisor_started", max_concurrent=max_concurrent,
               max_attempts=max_attempts)
        last_progress_report = 0.0
        previous_counts = None
        while True:
            _verify_source(plan)
            _refresh(directory, plan, processes, max_attempts)
            running = sum(job["state"] == "running" for job in plan["jobs"])
            for job in plan["jobs"]:
                if running >= max_concurrent:
                    break
                if job["state"] != "pending":
                    continue
                try:
                    processes[job["run_id"]] = _launch(directory, plan, job)
                    running += 1
                except Exception as error:
                    # A broken partial run must not silently turn into a fresh run.
                    job["attempts"] += 1
                    _failure(directory, plan, job, error, None, max_attempts)
            atomic_json(directory / "suite_plan.json", plan)
            summary = write_summary(directory, plan, max_concurrent)
            if (summary["counts"] != previous_counts or
                    time.monotonic() - last_progress_report >= 60):
                generate_progress_report(directory)
                previous_counts = summary["counts"]
                last_progress_report = time.monotonic()
            if summary["counts"]["completed"] == 18:
                from satellite_rl.evaluation.experiment_report import generate_report
                try:
                    report = generate_report(directory)
                    _event(directory, "report_completed", path=str(report))
                    return 0
                except Exception as error:
                    atomic_json(directory / "report_failure.json", {
                        "at": utc_now(), "type": type(error).__name__,
                        "message": str(error), "traceback": traceback.format_exc()})
                    _event(directory, "report_failed", error=str(error))
                    return 2
            if summary["counts"]["pending"] == 0 and summary["counts"]["running"] == 0:
                _event(directory, "suite_incomplete", failed=summary["counts"]["failed"])
                return 1
            time.sleep(poll_seconds)


def build_parser():
    parser = argparse.ArgumentParser(description="18 组 SAC 实验的后台监督队列")
    parser.add_argument("command", choices=("prepare", "run", "status"))
    parser.add_argument("--archive-root", default=str(_source_root()))
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--max-concurrent", type=int, default=3)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=5)
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "prepare":
        plan = prepare(args.archive_root, args.experiment_id)
        print(f"已准备 {len(plan['jobs'])} 组作业：{suite_dir(args.archive_root, args.experiment_id)}")
        return
    if args.command == "status":
        path = suite_dir(args.archive_root, args.experiment_id) / "suite_status.json"
        print(json.dumps(_load_json(path), ensure_ascii=False, indent=2))
        return
    raise SystemExit(run_queue(args.archive_root, args.experiment_id,
                               max_concurrent=args.max_concurrent,
                               retries=args.retries, poll_seconds=args.poll_seconds))


if __name__ == "__main__":
    main()
