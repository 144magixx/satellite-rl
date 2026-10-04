"""Own only the isolated 20424:8781 workbench lifecycle.

Training mode permits two workers. Evaluation mode switches to one worker only
after every workbench job is inactive, preserving fair timing. This module is
called on 20424 by launch_workbench.py and the persistent controller.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
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


def env() -> dict:
    result = os.environ.copy()
    result.update(PYTHONPATH=str(REPO / "src") + os.pathsep + str(DEPS),
                  SATELLITE_RL_PRIORITY_DEPS=str(DEPS),
                  PYTHONDONTWRITEBYTECODE="1", **THREAD_ENV)
    return result


def preflight(commit: str) -> dict:
    if not (REPO / ".git").exists() or not DEPS.is_dir() or not DQN.is_file():
        raise RuntimeError("Frozen repo, DQN or pinned solver dependencies missing")
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
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
                           env=env(), capture_output=True, text=True, check=True, timeout=90)
    runtime = json.loads(probe.stdout.splitlines()[-1])
    if (runtime["cuda"] is not True or runtime["cuda_build"] != "12.1"
            or Path(runtime["package"]).resolve() != REPO / "src/satellite_rl/__init__.py"
            or runtime["cvxpy"] != "1.6.5" or runtime["clarabel"] != "0.11.1"):
        raise RuntimeError("DQN/CUDA/frozen package/solver preflight mismatch")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout.strip()
    return {"commit": head, "runtime": runtime, "gpu_uuid": gpu}


def ensure_server(commit: str, mode: str) -> dict:
    if mode not in ("training", "evaluation"):
        raise ValueError("Server mode must be training or evaluation")
    checked = preflight(commit)
    workers = 2 if mode == "training" else 1
    if port_open():
        if not LAUNCH.is_file():
            raise RuntimeError("8781 is occupied without this experiment's launch receipt")
        old = json.loads(LAUNCH.read_text(encoding="utf-8"))
        if old.get("commit") != commit or proc_birth(old["pid"]) != old["start_ticks"]:
            raise RuntimeError("8781 server identity or process birth differs")
        system = api("/system")
        if (Path(system.get("source_root", "")).resolve() != REPO
                or Path(system.get("archive_root", "")).resolve() != ARCHIVE
                or Path(system.get("package_file", "")).resolve() != REPO / "src/satellite_rl/__init__.py"
                or system.get("cuda") is not True):
            raise RuntimeError("8781 live workbench source/archive/CUDA differs")
        if old.get("max_workers") == workers:
            return {"status": "already_running", **old}
        if mode != "evaluation" or old.get("max_workers") != 2:
            raise RuntimeError("Worker limit change is only allowed after training")
        runs = api("/runs").get("runs", [])
        blocking = [item for item in runs if item.get("stage") in ACTIVE]
        if blocking:
            raise RuntimeError("Cannot switch to one worker while jobs are active or queued")
        os.kill(old["pid"], signal.SIGTERM)
        for _ in range(60):
            if proc_birth(old["pid"]) != old["start_ticks"] and not port_open():
                break
            time.sleep(1)
        else:
            raise RuntimeError("Old isolated workbench did not stop cleanly")
    elif LAUNCH.is_file():
        old = json.loads(LAUNCH.read_text(encoding="utf-8"))
        if old.get("commit") != commit:
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
                                   cwd=REPO, env=env(), stdin=subprocess.DEVNULL,
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
    if (Path(system.get("source_root", "")).resolve() != REPO
            or Path(system.get("archive_root", "")).resolve() != ARCHIVE
            or Path(system.get("package_file", "")).resolve() != REPO / "src/satellite_rl/__init__.py"
            or Path(system.get("python", "")).resolve() != DQN.resolve()
            or system.get("cuda") is not True):
        raise RuntimeError("New workbench source/Python/CUDA/archive mismatch")
    receipt = {"status": "running_no_new_jobs", "checked_utc": now(),
               "target": "u2024110224@10.160.17.102:20424", "workbench_port": PORT,
               "pid": process.pid, "start_ticks": proc_birth(process.pid),
               "commit": commit, "max_workers": workers, "mode": mode,
               "repo": str(REPO), "archive": str(ARCHIVE), "log": str(log_path),
               "gpu_uuid": checked["gpu_uuid"], "runtime": checked["runtime"]}
    pending = LAUNCH.with_suffix(".tmp")
    pending.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(LAUNCH)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--mode", choices=("training", "evaluation"), required=True)
    args = parser.parse_args()
    print(json.dumps(ensure_server(args.commit, args.mode), ensure_ascii=False))


if __name__ == "__main__":
    main()
