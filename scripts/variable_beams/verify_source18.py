"""Verify which source snapshot each live S2-P18 job is actually running."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ops18  # noqa: E402

CHECK = '''
import hashlib,json,pathlib
root=pathlib.Path({source!r})
files={{"config.py":"src/satellite_rl/workbench/config.py",
 "algorithms.py":"src/satellite_rl/agents/workbench_algorithms.py",
 "networks.py":"src/satellite_rl/agents/workbench_networks.py",
 "environment.py":"src/satellite_rl/workbench/environment.py"}}
out={{}}
for key,rel in files.items():
    path=root/rel
    out[key]={{"exists":path.is_file(),
               "sha256":hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.is_file() else None}}
print(json.dumps(out))
'''

LOCAL = {
    "config.py": r"D:\paper\src\satellite_rl\workbench\config.py",
    "algorithms.py": r"D:\paper\src\satellite_rl\agents\workbench_algorithms.py",
    "networks.py": r"D:\paper\src\satellite_rl\agents\workbench_networks.py",
    "environment.py": r"D:\paper\src\satellite_rl\workbench\environment.py",
}


def local_hashes() -> dict:
    import hashlib
    return {key: hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
            for key, path in LOCAL.items()}


def main() -> None:
    report = {"local_main": local_hashes(), "jobs": {}}
    for port in ops18.PORTS:
        runs = ops18.api(port, "/runs", timeout=60).get("runs", [])
        for item in runs:
            if not str(item.get("request_id", "")).startswith("s2p18-"):
                continue
            detail = ops18.api(port, "/runs/" + item["run_id"], timeout=60)
            job = detail.get("job") or {}
            source = job.get("source_root") or job.get("source")
            entry = {"port": port, "request_id": item.get("request_id"),
                     "run_id": item.get("run_id"), "stage": item.get("stage"),
                     "git_commit": job.get("git_commit"), "source_root": source}
            if source:
                result = ops18.remote_python(CHECK.format(source=source), port, timeout=180)
                if result["exit_code"] == 0:
                    entry["hashes"] = json.loads(result["stdout"].strip().splitlines()[-1])
                else:
                    entry["hashes"] = {"error": result["stderr"][-200:]}
            report["jobs"][item["run_id"]] = entry
    ops18.write_json(ops18.LOCAL / "preflight" / "running_source_identity.json",
                     {"checked_utc": ops18.utc_now(), **report})
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
