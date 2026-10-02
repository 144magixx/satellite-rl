"""Start hidden local stage 3 tunnels and the persistent completion watcher."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import urllib.request


HERE = Path(__file__).resolve().parent
PREFLIGHT = Path("D:/paper/outputs/experiments/stage3_power_20261003/preflight")
MAPPINGS = ((20418, 8787, 8772), (20424, 8788, 8773))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def listening(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(1)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def system(port: int) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/system", timeout=10) as response:
        return json.load(response)


def detached(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        flags = 0
        kwargs = {}
        if os.name == "nt":
            flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
        else:
            kwargs["start_new_session"] = True
        process = subprocess.Popen(command, cwd=HERE.parents[1],
                                   stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, close_fds=True,
                                   creationflags=flags, **kwargs)
    return process.pid


def start_tunnels(commit: str) -> dict:
    items = {}
    for ssh_port, local_port, remote_port in MAPPINGS:
        launch = json.loads((PREFLIGHT / f"workbench_launch_{ssh_port}.json").read_text(encoding="utf-8"))
        if launch.get("commit") != commit or launch.get("workbench_port") != remote_port:
            raise RuntimeError("Frozen workbench launch receipt differs from mapping")
        if listening(local_port):
            existing = system(local_port)
            expected = f"/home/u2024110224/third_stage_satellite_rl/s3_power_20261003/{ssh_port}/repo"
            if existing.get("source_root") != expected:
                raise RuntimeError(f"Local port {local_port} is occupied by another service")
            items[str(local_port)] = {"status": "existing_verified", "ssh_port": ssh_port,
                                      "remote_port": remote_port, "system": existing}
            continue
        log = PREFLIGHT / f"tunnel_{local_port}.log"
        pid = detached([sys.executable, str(HERE / "remote_tunnel.py"),
                        "--ssh-port", str(ssh_port), "--local-port", str(local_port),
                        "--remote-port", str(remote_port)], log)
        for _ in range(20):
            try:
                current = system(local_port)
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError(f"Tunnel {local_port} did not become healthy; inspect {log}")
        expected = f"/home/u2024110224/third_stage_satellite_rl/s3_power_20261003/{ssh_port}/repo"
        if current.get("source_root") != expected:
            raise RuntimeError(f"Tunnel {local_port} routed to the wrong source")
        items[str(local_port)] = {"status": "started_verified", "pid": pid,
                                  "ssh_port": ssh_port, "remote_port": remote_port,
                                  "log": str(log), "system": current}
    receipt = {"status": "stage3_tunnels_ready", "checked_utc": now(),
               "commit": commit, "mappings": items}
    path = PREFLIGHT / "tunnel_launch.json"
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"receipt": str(path), "status": receipt["status"],
            "local_ports": sorted(items)}


def start_watcher(commit: str) -> dict:
    for ssh_port, _, _ in MAPPINGS:
        submitted = json.loads((PREFLIGHT / f"jan25_submission_{ssh_port}.json").read_text(encoding="utf-8"))
        if submitted.get("commit") != commit or not submitted.get("job", {}).get("run_id"):
            raise RuntimeError("Formal submission receipt missing")
    if (PREFLIGHT.parent / "completion/receipt.json").exists():
        raise RuntimeError("Completion receipt already exists; no new watcher needed")
    log = PREFLIGHT / "watcher.log"
    pid = detached([sys.executable, str(HERE / "watch_stage3.py"), "--commit", commit], log)
    receipt = {"status": "stage3_watcher_started", "started_utc": now(),
               "commit": commit, "pid": pid, "log": str(log),
               "state": str(PREFLIGHT / "watch_state.json")}
    path = PREFLIGHT / "watcher_launch.json"
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"receipt": str(path), "status": receipt["status"], "pid": pid}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("tunnels", "watcher"))
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    commit = args.commit.lower()
    if not re.fullmatch("[0-9a-f]{40}", commit):
        raise ValueError("Full frozen commit required")
    result = start_tunnels(commit) if args.action == "tunnels" else start_watcher(commit)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
