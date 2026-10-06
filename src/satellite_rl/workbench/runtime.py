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
from .experiments import is_variable_beam_experiment
from .storage import now, read_json

TERMINAL = {"completed", "failed", "stopped", "cancelled", "interrupted"}
_SYSTEM_CACHE = (0, None)


def process_start_time(pid):
    if os.name != "nt":
        try:
            stat = Path(f"/proc/{int(pid)}/stat").read_text()
            # The parenthesized process name can contain whitespace.
            ticks = int(stat.rsplit(")", 1)[1].split()[19])
            boot = next(int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines()
                        if line.startswith("btime "))
            return boot + ticks / os.sysconf("SC_CLK_TCK")
        except (OSError, ValueError, IndexError, StopIteration):
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
        if expected_start_time is None:
            return True
        actual = process_start_time(pid)
        return actual is not None and abs(actual - expected_start_time) < 0.01
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
        self.max_workers = max(1, min(5, int(max_workers)))
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

    def dispatch_hold(self):
        with self.lock:
            path = self.store.root / "dispatch_hold.json"
            record = read_json(path, {})
            active = isinstance(record, dict) and record.get("type") == "temporary_dispatch_hold_v1" and \
                isinstance(record.get("expires_at_unix"), (int, float)) and \
                time.time() < record["expires_at_unix"]
            if not active:
                path.unlink(missing_ok=True)
                return {"active": False}
            return {**record, "active": True, "remaining_seconds": max(0, record["expires_at_unix"] - time.time())}

    def hold_dispatch(self, ttl_seconds, reason):
        with self.lock:
            if type(ttl_seconds) is not int or not 1 <= ttl_seconds <= 600:
                raise ValueError("临时派发保留须为1至600秒整数")
            if not isinstance(reason, str) or not reason.strip() or len(reason) > 120:
                raise ValueError("临时派发保留原因无效")
            record = {"type": "temporary_dispatch_hold_v1", "reason": reason.strip(),
                      "created_at": now(), "expires_at_unix": time.time() + ttl_seconds}
            atomic_json(self.store.root / "dispatch_hold.json", record)
            return {**record, "active": True, "remaining_seconds": ttl_seconds}

    def release_dispatch(self):
        with self.lock:
            (self.store.root / "dispatch_hold.json").unlink(missing_ok=True)
            return {"active": False}

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
            if self.dispatch_hold()["active"]:
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
        # Only SI-priority refinement jobs may use their frozen, isolated
        # solver dependencies. The frozen scientific source stays first and
        # unrelated workers keep the original single-source PYTHONPATH.
        from satellite_rl.power_refinement.config import SI_PRIORITY_METHODS
        if (job.get("kind") == "refine" and
                (job.get("power_refinement") or {}).get("methods") == list(SI_PRIORITY_METHODS)):
            dependency_root = job.get("priority_solver_deps_path")
            if (not dependency_root or not Path(dependency_root).is_absolute()
                    or not Path(dependency_root).is_dir()):
                self._status(directory, status, stage="failed",
                             error="冻结的SI优先求解依赖目录缺失或无效")
                return
            env["PYTHONPATH"] += os.pathsep + str(Path(dependency_root).resolve())
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
                        "priority_solver_deps_path": job.get("priority_solver_deps_path"),
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

    def resume(self, run_id, *, source_repair=None, repair_validation_path=None,
               repair_validation_sha256=None, expected_origin_commit=None,
               expected_repair_commit=None):
        with self.lock:
            from .recovery import (S2P17_BOUNDARY_REPAIR_ID, S2P17_ORIGIN_COMMIT,
                                   S2P17_FAILED_RUN)
            if source_repair not in (None, "atomic_io_retry_v1", S2P17_BOUNDARY_REPAIR_ID):
                raise ValueError("不支持的恢复源码修复类型")
            directory = self.store.job_dir(run_id)
            job = read_json(directory / "job.json")
            status = read_json(directory / "status.json", {})
            config = read_json(directory / "config.json", {})
            if (source_repair == S2P17_BOUNDARY_REPAIR_ID and
                    status.get("stage") == "queued" and job and job.get("resume") is True and
                    job.get("run_id") == S2P17_FAILED_RUN and
                    expected_origin_commit == S2P17_ORIGIN_COMMIT and
                    (job.get("boundary_repair") or {}).get("repair_commit") ==
                    expected_repair_commit and
                    (job.get("boundary_repair") or {}).get("validation_receipt_sha256") ==
                    repair_validation_sha256):
                return status  # Unknown API outcome: the same job is already queued.
            saved_evaluation = bool(job and job.get("kind") == "evaluate" and
                                    is_variable_beam_experiment(job.get("experiment_id")))
            resumable_balance_eval = bool(job and job.get("kind") == "evaluate" and
                                          config.get("research_stage") == "coverage" and
                                          config.get("schema_version") == 3)
            resumable_reconstruction_eval = bool(job and job.get("kind") == "evaluate" and
                                                 config.get("research_stage") == "coverage" and
                                                 config.get("schema_version") == 4)
            if not job or (job["kind"] not in ("train", "refine") and
                           not (saved_evaluation or resumable_balance_eval or resumable_reconstruction_eval)):
                raise ValueError("只有工作台训练或功率优化任务支持恢复")
            if status.get("stage") not in ("stopped", "failed", "interrupted"):
                raise ValueError("该任务当前不能恢复；已完成任务请复制配置新建")
            if job["kind"] == "train" and not (directory / "checkpoint.pt").is_file():
                raise ValueError("没有可恢复的训练断点，推理权重不能代替断点")
            if resumable_balance_eval:
                route = config.get("coverage", {}).get("balance", {}).get("route")
                partial_name = "starts.partial.jsonl" if route == "prepare" else "results.partial.jsonl"
                if not (directory / partial_name).is_file():
                    raise ValueError("没有可恢复的逐场景记录；评价不能跳过已完成场景")
            if resumable_reconstruction_eval:
                route = config.get("coverage", {}).get("reconstruction", {}).get("route")
                partial_name = "starts.partial.jsonl" if route == "prepare" else "results.partial.jsonl"
                if not (directory / partial_name).is_file():
                    raise ValueError("没有可恢复的S1-03逐场景记录；评价不能伪重跑")
            boundary_repair = source_repair == S2P17_BOUNDARY_REPAIR_ID
            if source_repair is not None and (
                    (job["kind"] == "refine" and not boundary_repair) or
                    saved_evaluation or resumable_balance_eval or resumable_reconstruction_eval):
                raise ValueError("功率优化按冻结源码及完整场景记录恢复，不支持训练源码替换")
            if boundary_repair and (job.get("run_id") != S2P17_FAILED_RUN or
                    expected_origin_commit != S2P17_ORIGIN_COMMIT or
                    not repair_validation_path or not repair_validation_sha256 or
                    not expected_repair_commit):
                raise ValueError("S2-P17边界修复缺少精确任务/原提交/新提交/验证回执")
            if not boundary_repair and any(value is not None for value in
                    (repair_validation_path, repair_validation_sha256,
                     expected_origin_commit, expected_repair_commit)):
                raise ValueError("普通恢复不能携带S2-P17修复参数")
            if (config.get("research_stage") == "coverage" and config.get("schema_version") == 4 and
                    job["kind"] == "train" and status.get("stop_reason") == "mechanism_gate_pending"):
                from satellite_rl.workbench.storage import digest
                checkpoint_sha = digest(directory / "checkpoint.pt")
                gate = read_json(directory / "mechanism_gate.json", {})
                audit_path = Path(gate.get("audit_path", ""))
                audit = read_json(audit_path, {})
                allowed = (directory.resolve(), (self.store.archive_root / "ops" / "audits").resolve())
                if (gate.get("approved") is not True or gate.get("run_id") != run_id or
                        gate.get("env_step") != status.get("env_step") or
                        gate.get("checkpoint_sha256") != checkpoint_sha or
                        status.get("checkpoint_sha256") != checkpoint_sha or
                        not audit_path.is_file() or not any(audit_path.resolve().is_relative_to(root) for root in allowed) or
                        digest(audit_path) != gate.get("audit_sha256") or
                        audit.get("passed") is not True or audit.get("run_id") != run_id or
                        audit.get("env_step") != status.get("env_step") or
                        audit.get("checkpoint_sha256") != checkpoint_sha):
                    raise ValueError("S1-03机制闸门缺少与当前断点一致的工作台审计证明；保持停止状态")
            if pid_alive(status.get("pid")):
                raise ValueError("进程仍在运行，不能重复启动")
            if boundary_repair:
                from .recovery import prepare_s2p17_boundary_recovery
                job = prepare_s2p17_boundary_recovery(
                    self.store, job, repair_validation_path,
                    repair_validation_sha256, expected_repair_commit)
            elif source_repair is not None:
                from .recovery import prepare_io_retry_recovery
                job = prepare_io_retry_recovery(self.store, job)
            (directory / "stop.request").unlink(missing_ok=True)
            job["resume"] = True
            atomic_json(directory / "job.json", job)
            reason = ("从已保存完整评价案例恢复" if saved_evaluation else
                      "从已保存完整优化记录恢复" if job["kind"] == "refine" else
                      "从S1-02逐场景记录恢复" if resumable_balance_eval else "从正常断点恢复")
            if resumable_reconstruction_eval:
                reason = "从S1-03逐场景记录恢复"
            fields = {"stage": "queued", "queue_reason": reason, "pid": None,
                      "error": None, "traceback": None, "failure_path": None,
                      "exit_code": None, "exit_hex": None}
            if boundary_repair:
                proof = job["boundary_repair"]
                fields.update(queue_reason="S2-P17原0a源码保留；按已核验数值边界修复快照继续同ID细调",
                              previous_failure_archive=proof["archive_path"],
                              source_repair=source_repair,
                              scientific_origin_commit=proof["origin_science_commit"],
                              repair_commit=proof["repair_commit"],
                              repair_profile=proof["repair_id"],
                              failed_attempt_fallback_sha256=proof["fallback_contract_sha256"])
            elif source_repair is not None:
                proof = job["io_repair"]
                fields.update(queue_reason="使用已核验的原子I/O重试修复，从正常断点恢复",
                              env_step=proof["checkpoint_step"], update_step=proof["checkpoint_update_step"],
                              previous_failure_archive=proof["archive_path"],
                              source_repair=source_repair)
            self._status(directory, status, **fields)
            if boundary_repair:
                with (directory / "events.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"time": now(),
                        "message": "S2-P17数值边界修复：原科学0a保留，实际worker使用已核验repair快照；第300条失败B按原A回退",
                        "origin_science_commit": proof["origin_science_commit"],
                        "repair_commit": proof["repair_commit"],
                        "repair_profile": proof["repair_id"],
                        "fallback_contract_sha256": proof["fallback_contract_sha256"]},
                        ensure_ascii=False) + "\n")
            return read_json(directory / "status.json")

    def upgrade(self, run_id, *, upgrade_type, receipt_path):
        """Rebind a stopped/queued G08 job; never stop or start a worker here."""
        from .performance_upgrade import UPGRADE_TYPE, commit_upgrade, prepare_upgrade
        with self.lock:
            if upgrade_type != UPGRADE_TYPE or not self.dispatch_hold()["active"]:
                raise ValueError("性能升级需要已核验类型与有效的临时派发保留")
            directory = self.store.job_dir(run_id)
            job = read_json(directory / "job.json")
            status = read_json(directory / "status.json", {})
            if not job or status.get("stage") not in ("stopped", "queued"):
                raise ValueError("性能升级仅允许已停止或排队的工作台训练任务")
            launch = read_json(directory / "launch.json", {})
            child = self.children.get(run_id)
            if (pid_alive(status.get("pid"), launch.get("process_start_time")) or
                    pid_alive(launch.get("pid"), launch.get("process_start_time")) or
                    (child is not None and child.poll() is None)):
                raise ValueError("原工作进程仍在运行，不能升级源码")
            checkpoint = directory / "checkpoint.pt"
            if status["stage"] == "stopped" and not checkpoint.is_file():
                raise ValueError("已停止任务缺少完整断点")
            if status["stage"] == "queued" and checkpoint.is_file() and job.get("resume") is not True:
                raise ValueError("排队任务有断点但未标记从断点恢复，不能升级")
            if (status["stage"] == "queued" and not checkpoint.is_file() and
                    (any(status.get(key, 0) != 0 for key in ("env_step", "update_step", "episode")) or
                     job.get("resume"))):
                raise ValueError("排队任务记录非零进度却没有完整断点")
            new_root, new_commit = self.store.freeze_source()
            upgraded_config, proof = prepare_upgrade(self.store, job, receipt_path, new_root, new_commit)
            if not self.dispatch_hold()["active"]:
                raise ValueError("验证期间临时派发保留已过期，请重新设置")
            return commit_upgrade(self.store, job, status, upgraded_config, proof)

    def _loop(self):
        while not self.shutdown.is_set():
            try:
                self.tick()
            except Exception as error:
                atomic_json(self.store.root / "scheduler_error.json", {"at": now(), "error": repr(error)})
            self.shutdown.wait(4)
