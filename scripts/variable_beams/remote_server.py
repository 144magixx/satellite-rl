"""Own only the isolated 20424:8781 workbench lifecycle.

Training mode permits two workers. Evaluation mode switches to one worker only
after every workbench job is inactive, preserving fair timing. This module is
called on 20424 by launch_workbench.py and the persistent controller.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request


BASE = Path("/home/u2024110224/second_stage_satellite_rl/variable_beams_20261004/20424")
REPO = BASE / "repo"
ORIGIN_COMMIT = "0a4742a9a15af93478501497fac5cd36579fa075"
REPAIR_ID = "sca_basic_threshold_exact_v1"
FAILED_RUN = "refine_sac_20261004_215930_e814197c"
ARCHIVE = BASE / "archive"
OPS = BASE / "ops"
PORT = 8781
DQN = Path("/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python")
DEPS = Path("/home/u2024110224/third_stage_satellite_rl/s3_si_priority_20261003/shared_data/deps/cvxpy-1.6.5-clarabel-0.11.1")
LAUNCH = OPS / "workbench_8781_launch.json"
THREAD_ENV = {name: "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                      "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
ACTIVE = {"queued", "starting", "initializing", "warmup", "training",
          "running", "validating", "evaluating", "stopping"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def proc_birth(pid: int) -> int | None:
    try:
        raw = (Path("/proc") / str(pid) / "stat").read_text()
        return int(raw[raw.rfind(")") + 2:].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def api(path: str):
    request = urllib.request.Request(f"http://127.0.0.1:{PORT}/api{path}",
                                     headers={"Cache-Control": "no-cache"})
    with urllib.request.urlopen(request, timeout=8) as response:
        return json.load(response)


def port_open() -> bool:
    with socket.socket() as sock:
        sock.settimeout(2)
        return sock.connect_ex(("127.0.0.1", PORT)) == 0


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repair_repo_contract(repo: Path, commit: str, path: Path | None,
                          expected_sha: str | None, old_launch: dict | None) -> dict | None:
    if repo.resolve() == REPO.resolve():
        if path is not None or expected_sha is not None:
            raise ValueError("原科学repo不能携带修复切换合同")
        return None
    expected_repo = BASE / "recovery" / commit / "repo"
    if repo.resolve() != expected_repo.resolve() or not path or not expected_sha:
        raise ValueError("修复repo须位于20424隔离恢复目录并带精确合同")
    path = Path(path).resolve()
    if path != OPS / "recovery_switch_contract.json" or _file_sha(path) != expected_sha:
        raise ValueError("工作台修复切换合同路径或SHA不符")
    contract = json.loads(path.read_text(encoding="utf-8"))
    expected_validation = (ARCHIVE / "outputs/workbench/jobs" / FAILED_RUN /
                           "recoveries/validation_receipt.json").resolve()
    if (contract.get("format") != "s2p17_server_repair_switch_v1" or
            contract.get("repair_id") != REPAIR_ID or
            contract.get("origin_science_commit") != ORIGIN_COMMIT or
            contract.get("repair_commit") != commit or
            Path(contract.get("origin_repo", "")).resolve() != REPO.resolve() or
            Path(contract.get("repair_repo", "")).resolve() != repo.resolve() or
            contract.get("failed_run_id") != FAILED_RUN or
            Path(contract.get("validation_receipt_path", "")).resolve() != expected_validation or
            _file_sha(Path(contract["validation_receipt_path"])) !=
            contract.get("validation_receipt_sha256")):
        raise ValueError("工作台修复切换合同缺来源、失败任务或验证回执")
    if old_launch is not None:
        status_path = ARCHIVE / "outputs/workbench/jobs" / FAILED_RUN / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if (status.get("stage") != "failed" or status.get("env_step") != 299 or
                (isinstance(status.get("pid"), int) and proc_birth(status["pid"]) is not None)):
            raise ValueError("修复切换只允许原V0细调失败299现场")
    if old_launch is not None and (old_launch.get("commit") != ORIGIN_COMMIT or
            Path(old_launch.get("repo", "")).resolve() != REPO.resolve() or
            contract.get("old_server_pid") != old_launch.get("pid") or
            contract.get("old_server_start_ticks") != old_launch.get("start_ticks")):
        raise ValueError("原0a工作台PID/出生与切换合同不一致")
    return contract


def env(repo=REPO) -> dict:
    result = os.environ.copy()
    result.update(PYTHONPATH=str(Path(repo) / "src") + os.pathsep + str(DEPS),
                  SATELLITE_RL_PRIORITY_DEPS=str(DEPS),
                  PYTHONDONTWRITEBYTECODE="1", **THREAD_ENV)
    return result


def preflight(commit: str, repo=REPO) -> dict:
    repo = Path(repo)
    if not (repo / ".git").exists() or not DEPS.is_dir() or not DQN.is_file():
        raise RuntimeError("Frozen repo, DQN or pinned solver dependencies missing")
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                           capture_output=True, text=True, check=True).stdout.strip()
    if head != commit or dirty:
        raise RuntimeError("Workbench repo differs from clean scientific commit")
    probe = subprocess.run([str(DQN), "-B", "-c",
                            "import json,sys,torch,cvxpy,clarabel,satellite_rl;"
                            "print(json.dumps({'python':sys.executable,'torch':torch.__version__,"
                            "'cuda_build':torch.version.cuda,'cuda':torch.cuda.is_available(),"
                            "'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"
                            "'cvxpy':cvxpy.__version__,'clarabel':clarabel.__version__,"
                            "'package':satellite_rl.__file__}))"],
                           env=env(repo), capture_output=True, text=True, check=True, timeout=90)
    runtime = json.loads(probe.stdout.splitlines()[-1])
    if (runtime["cuda"] is not True or runtime["cuda_build"] != "12.1"
            or Path(runtime["package"]).resolve() != repo / "src/satellite_rl/__init__.py"
            or runtime["cvxpy"] != "1.6.5" or runtime["clarabel"] != "0.11.1"):
        raise RuntimeError("DQN/CUDA/frozen package/solver preflight mismatch")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout.strip()
    return {"commit": head, "runtime": runtime, "gpu_uuid": gpu}


def ensure_server(commit: str, mode: str, *, repo=REPO,
                  repair_switch_contract=None, repair_switch_sha256=None) -> dict:
    if mode not in ("training", "evaluation"):
        raise ValueError("Server mode must be training or evaluation")
    repo = Path(repo).resolve()
    checked = preflight(commit, repo)
    workers = 2 if mode == "training" else 1
    if repo != REPO.resolve() and mode != "evaluation":
        raise ValueError("修复repo仅用于单worker的既定SCA评价")
    previous_launch = json.loads(LAUNCH.read_text(encoding="utf-8")) if LAUNCH.is_file() else None
    previous_launch_backup = None
    switch = _repair_repo_contract(repo, commit, repair_switch_contract,
                                   repair_switch_sha256,
                                   previous_launch if previous_launch and
                                   previous_launch.get("commit") != commit else None)
    if (switch and previous_launch and previous_launch.get("commit") == commit and
            previous_launch.get("repair_switch_contract_sha256") != repair_switch_sha256):
        raise ValueError("已运行修复工作台与切换合同SHA不一致")
    if switch and previous_launch and previous_launch.get("commit") == commit:
        repair_job = ARCHIVE / "outputs/workbench/jobs" / FAILED_RUN / "job.json"
        bound = json.loads(repair_job.read_text(encoding="utf-8"))
        proof = bound.get("boundary_repair") or {}
        if (bound.get("git_commit") != commit or
                proof.get("repair_id") != REPAIR_ID or
                proof.get("origin_science_commit") != ORIGIN_COMMIT or
                proof.get("repair_commit") != commit or
                bound.get("source_root") != proof.get("new_source_root") or
                not proof.get("fallback_contract_sha256")):
            raise ValueError("已切换修复工作台缺同ID任务来源证明")
    if port_open():
        if not LAUNCH.is_file():
            raise RuntimeError("8781 is occupied without this experiment's launch receipt")
        old = json.loads(LAUNCH.read_text(encoding="utf-8"))
        if proc_birth(old["pid"]) != old["start_ticks"]:
            raise RuntimeError("8781 server identity or process birth differs")
        system = api("/system")
        old_repo = Path(old.get("repo", str(REPO))).resolve()
        if (Path(system.get("source_root", "")).resolve() != old_repo
                or Path(system.get("archive_root", "")).resolve() != ARCHIVE
                or Path(system.get("package_file", "")).resolve() != old_repo / "src/satellite_rl/__init__.py"
                or system.get("cuda") is not True):
            raise RuntimeError("8781 live workbench source/archive/CUDA differs")
        if old.get("commit") == commit and old_repo == repo and old.get("max_workers") == workers:
            return {"status": "already_running", **old}
        switching_repair = bool(switch and old.get("commit") == ORIGIN_COMMIT and
                                old_repo == REPO.resolve() and old.get("max_workers") == 1)
        if not switching_repair and (old.get("commit") != commit or
                                     old_repo != repo or mode != "evaluation" or
                                     old.get("max_workers") != 2):
            raise RuntimeError("工作台切换只允许训练后单worker或已核验0a→repair")
        runs = api("/runs").get("runs", [])
        blocking = [item for item in runs if item.get("stage") in ACTIVE]
        if blocking:
            raise RuntimeError("Cannot switch to one worker while jobs are active or queued")
        if switching_repair:
            previous_launch_backup = OPS / "workbench_8781_launch_origin0a.json"
            if previous_launch_backup.exists():
                if json.loads(previous_launch_backup.read_text(encoding="utf-8")) != old:
                    raise RuntimeError("原0a工作台启动回执已有不同备份")
            else:
                previous_launch_backup.write_text(
                    json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8")
        os.kill(old["pid"], signal.SIGTERM)
        for _ in range(60):
            if proc_birth(old["pid"]) != old["start_ticks"] and not port_open():
                break
            time.sleep(1)
        else:
            raise RuntimeError("Old isolated workbench did not stop cleanly")
    elif LAUNCH.is_file():
        old = json.loads(LAUNCH.read_text(encoding="utf-8"))
        if old.get("commit") != commit and switch is None:
            raise RuntimeError("Old launch receipt belongs to another commit")
        if old.get("max_workers") == 2 and mode == "training":
            # A dead server may be restarted only after explicit status inspection.
            raise RuntimeError("Training server exited; inspect jobs before restarting")
    OPS.mkdir(parents=True, exist_ok=True)
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    log_path = OPS / "workbench_8781.log"
    with log_path.open("ab", buffering=0) as output:
        process = subprocess.Popen([str(DQN), "-B", "-m", "satellite_rl.workbench.server",
                                    "--archive-root", str(ARCHIVE), "--port", str(PORT),
                                    "--python", str(DQN), "--max-workers", str(workers)],
                                   cwd=repo, env=env(repo), stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True, close_fds=True)
    for _ in range(45):
        if process.poll() is not None:
            raise RuntimeError("Isolated workbench exited during startup; inspect log")
        try:
            system = api("/system")
            break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError("Isolated workbench did not become healthy")
    if (Path(system.get("source_root", "")).resolve() != repo
            or Path(system.get("archive_root", "")).resolve() != ARCHIVE
            or Path(system.get("package_file", "")).resolve() != repo / "src/satellite_rl/__init__.py"
            or Path(system.get("python", "")).resolve() != DQN.resolve()
            or system.get("cuda") is not True):
        raise RuntimeError("New workbench source/Python/CUDA/archive mismatch")
    receipt = {"status": "running_no_new_jobs", "checked_utc": now(),
               "target": "u2024110224@10.160.17.102:20424", "workbench_port": PORT,
               "pid": process.pid, "start_ticks": proc_birth(process.pid),
               "commit": commit, "max_workers": workers, "mode": mode,
               "repo": str(repo), "archive": str(ARCHIVE), "log": str(log_path),
               "origin_science_commit": ORIGIN_COMMIT if switch else commit,
               "repair_profile": REPAIR_ID if switch else None,
               "repair_switch_contract_sha256": repair_switch_sha256 if switch else None,
               "previous_launch_path": str(previous_launch_backup) if previous_launch_backup else None,
               "previous_launch_sha256": (_file_sha(previous_launch_backup)
                                           if previous_launch_backup else None),
               "gpu_uuid": checked["gpu_uuid"], "runtime": checked["runtime"]}
    pending = LAUNCH.with_suffix(".tmp")
    pending.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(LAUNCH)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--mode", choices=("training", "evaluation"), required=True)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--repair-switch-contract", type=Path)
    parser.add_argument("--repair-switch-sha256")
    args = parser.parse_args()
    print(json.dumps(ensure_server(args.commit, args.mode, repo=args.repo,
        repair_switch_contract=args.repair_switch_contract,
        repair_switch_sha256=args.repair_switch_sha256), ensure_ascii=False))


if __name__ == "__main__":
    main()
