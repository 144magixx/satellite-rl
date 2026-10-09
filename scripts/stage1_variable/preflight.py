"""Read-only, pinned-SSH preflight for the four S1-V10 workbench lanes."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.s2p19.ops import SSH_PORTS, remote_python


OUTPUT = Path("D:/paper/outputs/experiments/first_stage_s1_v10_20261009/preflight")

PROBE = r'''
import json, os, subprocess, sys, torch, satellite_rl
from datetime import datetime, timezone
from pathlib import Path
def command(args):
    result = subprocess.run(args, text=True, capture_output=True, timeout=20)
    return {"code": result.returncode, "stdout": result.stdout[-12000:], "stderr": result.stderr[-2000:]}
print(json.dumps({
    "observed_utc": datetime.now(timezone.utc).isoformat(),
    "python": sys.executable, "prefix": sys.prefix,
    "torch": torch.__version__, "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "satellite_rl_file": satellite_rl.__file__,
    "cwd": os.getcwd(),
    "cpu_count": os.cpu_count(),
    "cpu_affinity_count": len(os.sched_getaffinity(0)),
    "cpu_affinity": sorted(os.sched_getaffinity(0)),
    "load_average": os.getloadavg(),
    "cgroup_cpu_max": Path('/sys/fs/cgroup/cpu.max').read_text().strip()
        if Path('/sys/fs/cgroup/cpu.max').is_file() else None,
    "cgroup_cpu_quota": Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read_text().strip()
        if Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').is_file() else None,
    "nvidia_smi": command(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"]),
    "gpu_util": command(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"]),
    "ports": command(["ss", "-ltnp"]),
    "processes": command(["ps", "-eo", "pid,lstart,etime,args", "--sort=-start_time"]),
    "disk": command(["df", "-h", "/home/u2024110224"]),
    "existing_root": Path("/home/u2024110224/first_stage_satellite_rl/s1_v10").exists(),
}, ensure_ascii=False))
'''


def probe(port: int) -> dict:
    response = remote_python(PROBE, port, timeout=90)
    if response["exit_code"]:
        return {"port": port, "error": response["stderr"][-2000:],
                "exit_code": response["exit_code"]}
    try:
        observed = json.loads(response["stdout"].splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        return {"port": port, "error": str(error),
                "stdout_tail": response["stdout"][-1000:]}
    observed["port"] = port
    observed["target"] = f"u2024110224@10.160.17.102:{port}"
    return observed


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = {"probe_started_utc": datetime.now(timezone.utc).isoformat(), "lanes": {}}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(probe, port): port for port in SSH_PORTS}
        for future in as_completed(futures):
            port = futures[future]
            try:
                result["lanes"][str(port)] = future.result()
            except Exception as error:
                result["lanes"][str(port)] = {"port": port, "error": str(error)}
    result["probe_finished_utc"] = datetime.now(timezone.utc).isoformat()
    path = OUTPUT / "four_lane_readonly.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({port: {key: value for key, value in row.items()
                             if key in ("error", "python", "torch", "torch_cuda", "cuda_available", "gpu", "satellite_rl_file", "observed_utc")}
                      for port, row in result["lanes"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
