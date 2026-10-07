"""Bounded S2-P19 disk observation and dispatch forecast.

The guard never deletes or compresses files. Controller calls it before each
typed workbench submission and pauses new submissions if the bound fails.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from . import ops


GIB = 1024 ** 3
RESERVE_BYTES = 12 * GIB
PREDICTED_BYTES = {
    "mother_sources": 5 * GIB,  # two 1.7GB official tar archives plus CSV/checkpoints
    "dev_validation": 2 * GIB,
    "final": 6 * GIB,
    "demonstrations": 6 * GIB,
    "C0_pilot": 12 * GIB,  # includes checkpoint and immutable pilot_artifacts
    "C1_pilot": 12 * GIB,
    "I0_pilot": 12 * GIB,
    "C0_continuation": 15 * GIB,
    "C1_continuation": 15 * GIB,
    "I0_continuation": 15 * GIB,
    "gate_B_216": 216 * 10 * 1024 ** 2,
    "validation_B_504": 504 * 10 * 1024 ** 2,
    "final_direct_model_2640": 5 * GIB,
    "final_direct_baselines_4400": 8 * GIB,
    "final_B_one_stochastic_arm_2640": 2640 * 10 * 1024 ** 2,
    "final_B_one_deterministic_arm_880": 880 * 10 * 1024 ** 2,
}


def sample(port: int) -> dict:
    info = ops.instance(port)
    code = f'''
import datetime,json,os,pathlib,shutil
root=pathlib.Path({info['root']!r})
def size(path):
 if not path.exists():return 0
 total=0
 for parent,_,names in os.walk(path):
  for name in names:
   p=pathlib.Path(parent)/name
   try:total+=p.stat().st_size
   except FileNotFoundError:pass
 return total
jobs=root/'archive/outputs/workbench/jobs'
children=[]
if jobs.is_dir():
 for path in jobs.iterdir():
  if path.is_dir():children.append({{'run_id':path.name,'bytes':size(path)}})
children.sort(key=lambda x:x['bytes'],reverse=True)
disk=shutil.disk_usage('/home')
print(json.dumps({{'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'target':{info['target']!r},'root':str(root),'home_free_bytes':disk.free,
 'home_total_bytes':disk.total,'s2p19_root_bytes':size(root),
 'jobs_largest':children[:15]}},ensure_ascii=False))
'''
    result = ops.remote_python(code, port, timeout=180)
    if result["exit_code"]:
        raise RuntimeError(f"{port} disk sample failed: " + result["stderr"][-600:])
    return json.loads(result["stdout"].splitlines()[-1])


def gate(snapshot: dict, task_class: str, outstanding_reserved_bytes: int = 0) -> dict:
    if task_class not in PREDICTED_BYTES:
        raise ValueError("unknown bounded task class")
    predicted = PREDICTED_BYTES[task_class]
    free = snapshot["home_free_bytes"]
    return {"task_class": task_class, "observed_utc": snapshot["observed_utc"],
            "home_free_bytes": free, "predicted_increment_bytes": predicted,
            "reserve_bytes": RESERVE_BYTES,
            "outstanding_reserved_bytes": outstanding_reserved_bytes,
            "admit": free >= predicted + outstanding_reserved_bytes + RESERVE_BYTES,
            "estimated_free_after_bytes": free - predicted - outstanding_reserved_bytes}


def write_snapshot(port: int) -> dict:
    observed = sample(port)
    output = ops.LOCAL / "preflight" / f"disk_{port}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(observed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return observed


if __name__ == "__main__":
    for target in ops.SSH_PORTS:
        result = write_snapshot(target)
        print(target, result["observed_utc"], result["home_free_bytes"],
              result["s2p19_root_bytes"], result["jobs_largest"][:3])
