"""Persistent local scheduler; worker lifetimes are independent of browser requests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from satellite_rl.training.run_monitor import atomic_json
from .storage import now, read_json

TERMINAL = {"completed", "failed", "stopped", "cancelled", "interrupted"}
_SYSTEM_CACHE = (0, None)


def process_start_time(pid):
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_bool, ctypes.c_ulong]
    kernel.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return None
    times = [wintypes.FILETIME() for _ in range(4)]
    try:
        if kernel.GetProcessTimes(handle, *[ctypes.byref(value) for value in times]):
            return (((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime) / 10000000) - 11644473600
    finally:
        kernel.CloseHandle(handle)
    return None


def pid_alive(pid, expected_start_time=None):
    if not pid:
        return False
    try:
        if os.name == "nt":
            import ctypes
            kernel = ctypes.windll.kernel32
            kernel.OpenProcess.restype = ctypes.c_void_p
            kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_bool, ctypes.c_ulong]
            kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
            kernel.CloseHandle.argtypes = [ctypes.c_void_p]
            handle = kernel.OpenProcess(0x1000, False, int(pid))
            if not handle:
                return False
            code = ctypes.c_ulong()
            valid = kernel.GetExitCodeProcess(handle, ctypes.byref(code))
            kernel.CloseHandle(handle)
            if not valid or code.value != 259:
                return False
            actual_start = process_start_time(pid)
            return expected_start_time is None or (actual_start is not None and abs(actual_start - expected_start_time) < 0.01)
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def gpu_snapshot():
    try:
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                timeout=5, creationflags=flags, check=True)
        name, used, total, utilization = result.stdout.splitlines()[0].split(",", 3)
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=5, creationflags=flags)
        pids = []
        for line in apps.stdout.splitlines():
            pieces = line.split(",", 1)
            if len(pieces) == 2 and pieces[0].strip().isdigit():
                name_lower = pieces[1].strip().replace("\\", "/").rsplit("/", 1)[-1].lower()
                if name_lower in ("python", "python3", "python.exe", "pythonw.exe") and int(pieces[0]) != os.getpid():
                    pids.append(int(pieces[0]))
        return {"name": name.strip(), "memory_used_mb": float(used), "memory_total_mb": float(total),
                "utilization_percent": float(utilization), "compute_pids": pids}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as error:
        return {"error": str(error)}


def system_info(store):
    global _SYSTEM_CACHE
    if time.monotonic() - _SYSTEM_CACHE[0] > 8 or _SYSTEM_CACHE[1] is None:
        import torch
        import satellite_rl
        _SYSTEM_CACHE = (time.monotonic(), {
            "python": sys.executable, "torch": torch.__version__, "cuda_build": torch.version.cuda,
            "cuda": torch.cuda.is_available(), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu": gpu_snapshot(), "package_file": satellite_rl.__file__,
        })
    return {**_SYSTEM_CACHE[1], "source_root": str(store.source_root),
            "archive_root": str(store.archive_root), "server_time": now()}


class Scheduler:
    def __init__(self, store, python_executable=sys.executable, max_workers=1):
        self.store = store
        self.python = str(Path(python_executable).resolve())
        self.max_workers = max(1, min(3, int(max_workers)))
        self.children = {}
        self.lock = threading.RLock()
        self.shutdown = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True, name="workbench-scheduler")

    def start(self):
        self.thread.start()

    def stop_scheduler(self):
        self.shutdown.set()
        self.thread.join(timeout=5)

    def _status(self, directory, status, **changes):
        atomic_json(directory / "status.json", {**status, **changes, "heartbeat_at": now()})

    def tick(self):
        with self.lock:
            jobs = list(reversed(self.store.jobs()))
            active = []
            for job, status in jobs:
                stage = status.get("stage")
                if stage in TERMINAL or stage == "queued":
                    old_child = self.children.get(job["run_id"])
                    if old_child is not None and old_child.poll() is not None:
                        self.children.pop(job["run_id"], None)
                    continue
                run_id = job["run_id"]
                child = self.children.get(run_id)
                launch = read_json(self.store.job_dir(run_id) / "launch.json", {})
                alive = child.poll() is None if child else pid_alive(status.get("pid"), launch.get("process_start_time"))
                if alive:
                    active.append(run_id)
                    continue
                # A worker may finish between reading status and inspecting PID.
                directory = self.store.job_dir(run_id)
                latest = read_json(directory / "status.json", status)
                if latest.get("stage") in TERMINAL:
                    continue
                exit_code = child.returncode if child else None
                exit_hex = f"0x{exit_code & 0xffffffff:08X}" if isinstance(exit_code, int) else None
                failure = {"at": now(), "run_id": run_id, "pid": status.get("pid"),
                           "exit_code": exit_code, "exit_hex": exit_hex,
                           "source_root": job.get("source_root"), "git_commit": job.get("git_commit"),
                           "last_normal_checkpoint": str(directory / "checkpoint.pt") if (directory / "checkpoint.pt").exists() else None,
                           "error": "进程在没有Python异常记录的情况下结束；原生退出应结合Windows事件和驱动日志排查"}
                if not (directory / "failure.json").is_file():
                    atomic_json(directory / "failure.json", failure)
                self._status(directory, latest, stage="interrupted",
                             exit_code=exit_code, exit_hex=exit_hex,
                             error="后台进程已结束但未写入完成状态" + (f"（{exit_hex}）" if exit_hex else "") + "；检查日志后可从正常断点恢复")
                self.children.pop(run_id, None)
            queued = [(j, s) for j, s in jobs if s.get("stage") == "queued"]
            if not queued or len(active) >= self.max_workers:
                return
            gpu = gpu_snapshot()
            available = gpu.get("memory_total_mb", 0) - gpu.get("memory_used_mb", 0)
            external = [pid for pid in gpu.get("compute_pids", [])
                        if pid not in [read_json(self.store.job_dir(r) / "status.json", {}).get("pid") for r in active]]
            reason = None
            if gpu.get("error"):
                reason = "无法读取GPU状态：" + gpu["error"]
            elif available < 1536:
                reason = "GPU可用显存不足1.5GB，等待资源"
            elif len(external) >= 3 or (len(external) >= 2 and gpu.get("utilization_percent", 0) >= 95):
                reason = "现有GPU作业繁忙，等待资源，不中断其他实验"
            if reason:
                for job, status in queued:
                    if status.get("queue_reason") != reason:
                        self._status(self.store.job_dir(job["run_id"]), status, queue_reason=reason)
                return
            for job, status in queued[:self.max_workers - len(active)]:
                self._launch(job, status)

    def _launch(self, job, status):
        directory = self.store.job_dir(job["run_id"])
        command = [self.python, "-u", "-m", "satellite_rl.workbench.runner", "--job-dir", str(directory)]
        if job.get("resume"):
            command.append("--resume")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(job["source_root"]) / "src")
        env["SATELLITE_RL_OUTPUT_ROOT"] = job["archive_root"]
        env["PYTHONUNBUFFERED"] = "1"
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self._status(directory, status, stage="starting", queue_reason=None)
        try:
            with (directory / "worker.log").open("ab", buffering=0) as log:
                child = subprocess.Popen(command, cwd=job["source_root"], env=env,
                                         stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                         creationflags=flags, close_fds=True, start_new_session=os.name != "nt")
            self.children[job["run_id"]] = child
            latest = read_json(directory / "status.json", status)
            if latest.get("stage") in ("queued", "starting"):
                self._status(directory, latest, stage="starting", pid=child.pid, command=command, started_at=now())
            atomic_json(directory / "launch.json", {"pid": child.pid, "command": command,
                        "python": self.python, "source_root": job["source_root"], "at": now(),
                        "process_start_time": process_start_time(child.pid)})
        except Exception as error:
            self._status(directory, status, stage="failed", error=str(error))

    def request_stop(self, run_id):
        with self.lock:
            directory = self.store.job_dir(run_id)
            job = read_json(directory / "job.json")
            if not job:
                raise FileNotFoundError("只能停止工作台创建的任务")
            status = read_json(directory / "status.json", {})
            if status.get("stage") == "queued":
                self._status(directory, status, stage="cancelled", queue_reason="用户取消排队")
            elif status.get("stage") not in TERMINAL:
                (directory / "stop.request").write_text(now(), encoding="utf-8")
            return read_json(directory / "status.json", {})

    def resume(self, run_id):
        with self.lock:
            directory = self.store.job_dir(run_id)
            job = read_json(directory / "job.json")
            status = read_json(directory / "status.json", {})
            if not job or job["kind"] != "train":
                raise ValueError("只有工作台训练任务支持恢复")
            if status.get("stage") not in ("stopped", "failed", "interrupted"):
                raise ValueError("该任务当前不能恢复；已完成任务请复制配置新建")
            if not (directory / "checkpoint.pt").is_file():
                raise ValueError("没有可恢复的训练断点，推理权重不能代替断点")
            if pid_alive(status.get("pid")):
                raise ValueError("进程仍在运行，不能重复启动")
            (directory / "stop.request").unlink(missing_ok=True)
            job["resume"] = True
            atomic_json(directory / "job.json", job)
            self._status(directory, status, stage="queued", queue_reason="从正常断点恢复", pid=None)
            return read_json(directory / "status.json")

    def _loop(self):
        while not self.shutdown.is_set():
            try:
                self.tick()
            except Exception as error:
                atomic_json(self.store.root / "scheduler_error.json", {"at": now(), "error": repr(error)})
            self.shutdown.wait(4)
