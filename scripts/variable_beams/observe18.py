"""Read both instances' live training state and GPU usage (read-only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ops18  # noqa: E402

BODY = '''
import json,os,subprocess
def run(a):
    p=subprocess.run(a,capture_output=True,text=True)
    return p.stdout.strip()
gpu=run(["nvidia-smi","--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw","--format=csv,noheader"])
procs=run(["nvidia-smi","--query-compute-apps=pid,used_memory","--format=csv,noheader"])
print(json.dumps({"gpu":gpu,"compute_apps":procs,"cpus":len(os.sched_getaffinity(0))},ensure_ascii=False))
'''


def main() -> None:
    report = {}
    for port in ops18.PORTS:
        runs = ops18.api(port, "/runs", timeout=60).get("runs", [])
        live = []
        for item in runs:
            if not str(item.get("request_id", "")).startswith("s2p18-"):
                continue
            detail = ops18.api(port, "/runs/" + item["run_id"], timeout=60)
            status = detail.get("status") or {}
            live.append({
                "request_id": item.get("request_id"),
                "run_id": item.get("run_id"),
                "kind": item.get("kind"),
                "stage": status.get("stage") or item.get("stage"),
                "env_step": status.get("env_step"),
                "budget_env_steps": status.get("budget_env_steps"),
                "update_step": status.get("update_step"),
                "episode": status.get("episode"),
                "steps_per_second": status.get("steps_per_second"),
                "updates_per_second": status.get("updates_per_second"),
                "beam_count_episodes": status.get("beam_count_episodes"),
                "beam_bucket_episodes": status.get("beam_bucket_episodes"),
                "replay_sampled_by_bucket": status.get("replay_sampled_by_bucket"),
                "spread_by_bucket": status.get("progress_strata_sampled"),
                "error": status.get("error"),
                "heartbeat_at": status.get("heartbeat_at"),
            })
        host = ops18.remote_python(BODY, port, timeout=120)
        report[str(port)] = {
            "jobs": live,
            "host": json.loads(host["stdout"].strip().splitlines()[-1]) if host["exit_code"] == 0 else host["stderr"][-200:],
        }
    ops18.write_json(ops18.LOCAL / "preflight" / "training_live.json",
                     {"checked_utc": ops18.utc_now(), "instances": report})
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
