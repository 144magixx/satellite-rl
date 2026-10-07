"""Bounded S2-P19 disk observation and dispatch forecast.

The guard never deletes or compresses files. Controller calls it before each
typed workbench submission and pauses new submissions if the bound fails.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from . import ops


GIB = 1024 ** 3
RESERVE_BYTES = 12 * GIB
# NFS statvfs reports server-wide space, not a verified per-user quota. This
# 256 GiB is an archive-planning signal for this experiment, never a hard stop.
EXPERIMENT_ENGINEERING_BUDGET_BYTES = 256 * GIB
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
import datetime,json,os,pathlib,shutil,subprocess
root=pathlib.Path({info['root']!r})
archive=pathlib.Path({info['archive']!r})
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
disk=shutil.disk_usage(archive)
stat=archive.stat();vfs=os.statvfs(archive)
mounted=subprocess.run(['findmnt','-T',str(archive),'-n','-o','SOURCE,TARGET,FSTYPE'],
 capture_output=True,text=True,check=True).stdout.strip().split()
if len(mounted)!=3:raise RuntimeError('actual archive mount identity unavailable')
mount_source,mount_target,filesystem_type=mounted
print(json.dumps({{'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'target':{info['target']!r},'root':str(root),'archive':str(archive),
 'archive_free_bytes':disk.free,'archive_total_bytes':disk.total,
 'archive_device_id':stat.st_dev,'archive_fsid':vfs.f_fsid,
 'mount_source':mount_source,'mount_target':mount_target,'filesystem_type':filesystem_type,
 'mount_identity':mount_source+'|'+mount_target+'|'+filesystem_type,
 's2p19_root_bytes':size(root),
 'jobs_largest':children[:15]}},ensure_ascii=False))
'''
    result = ops.remote_python(code, port, timeout=180)
    if result["exit_code"]:
        raise RuntimeError(f"{port} disk sample failed: " + result["stderr"][-600:])
    return json.loads(result["stdout"].splitlines()[-1])


def gate(snapshot: dict, task_class: str, outstanding_reserved_bytes: int = 0,
         shared_root_bytes: int | None = None) -> dict:
    if task_class not in PREDICTED_BYTES:
        raise ValueError("unknown bounded task class")
    predicted = PREDICTED_BYTES[task_class]
    free = snapshot["archive_free_bytes"]
    used = snapshot["s2p19_root_bytes"] if shared_root_bytes is None else shared_root_bytes
    physical_ok = free >= predicted + outstanding_reserved_bytes + RESERVE_BYTES
    projected = used + predicted + outstanding_reserved_bytes
    archive_warning = projected >= EXPERIMENT_ENGINEERING_BUDGET_BYTES
    return {"task_class": task_class, "observed_utc": snapshot["observed_utc"],
            "archive_free_bytes": free, "mount_identity": snapshot["mount_identity"],
            "shared_root_bytes": used, "predicted_increment_bytes": predicted,
            "reserve_bytes": RESERVE_BYTES,
            "experiment_engineering_budget_bytes": EXPERIMENT_ENGINEERING_BUDGET_BYTES,
            "outstanding_reserved_bytes": outstanding_reserved_bytes,
            "physical_admit": physical_ok,
            "archive_schedule_warning": archive_warning,
            "admit": physical_ok,
            "estimated_archive_free_after_bytes": free - predicted - outstanding_reserved_bytes,
            "estimated_engineering_budget_headroom_after_bytes":
            EXPERIMENT_ENGINEERING_BUDGET_BYTES - projected}


def write_snapshot(port: int) -> dict:
    observed = sample(port)
    output = ops.LOCAL / "preflight" / f"disk_{port}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(observed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return observed


if __name__ == "__main__":
    for target in ops.SSH_PORTS:
        result = write_snapshot(target)
        print(target, result["observed_utc"], result["archive_free_bytes"],
              result["mount_identity"], result["s2p19_root_bytes"], result["jobs_largest"][:3])
