"""Shared pure lease checks and fresh local resource inventory for F20."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request

TERMINAL = {"completed", "failed", "stopped", "cancelled", "invalidated"}


def cgroup_inventory(root=Path("/sys/fs/cgroup"), proc_self=Path("/proc/self/cgroup")):
    """Record v1/v2 availability without inventing an unexposed CPU quota."""
    root,proc_self=Path(root),Path(proc_self)
    errors=[]
    def read(path):
        try:return path.read_text().strip()
        except OSError as error:
            errors.append({"path":str(path),"reason":type(error).__name__});return None
    proc_metadata=read(proc_self)
    v2={name:read(root/name) for name in ("cpu.max","cpuset.cpus.effective","memory.max","memory.current") if (root/name).is_file()}
    v1={}
    paths={"cpu.cfs_quota_us":("cpu/cpu.cfs_quota_us","cpu,cpuacct/cpu.cfs_quota_us"),
           "cpu.cfs_period_us":("cpu/cpu.cfs_period_us","cpu,cpuacct/cpu.cfs_period_us"),
           "memory.limit_in_bytes":("memory/memory.limit_in_bytes",),
           "memory.usage_in_bytes":("memory/memory.usage_in_bytes",),
           "cpuset.cpus":("cpuset/cpuset.cpus",)}
    for name,choices in paths.items():
        for choice in choices:
            path=root/choice
            if path.is_file():v1[name]=read(path);break
    limits={**v2,**v1}
    return {"attempted":True,"version":2 if v2 else 1 if v1 else "unexposed",
            "proc_self_cgroup":proc_metadata,"limits":limits,"read_errors":errors,
            "limits_exposed":bool(limits),"timing_cpu_quota_control_verified":False,
            "limitation":None if limits else "cgroup limit files are unexposed; CPU affinity and actual memory availability are observed, quota is not asserted"}


def release_reasons(receipt):
    reasons=[]
    if receipt.get("released") is not True or receipt.get("released_ssh_port") != 20474:
        reasons.append("old lane release is not verified")
    if 20474 not in receipt.get("old_coordinator_excluded_ports", []):
        reasons.append("old coordinator does not exclude physical 20474")
    if receipt.get("persistent_coordinator_exclusion_verified") is not True or not receipt.get("coordinator_evidence"):
        reasons.append("persistent old coordinator exclusion evidence missing")
    if receipt.get("active_jobs") or receipt.get("pending_jobs") or receipt.get("future_lane_reserved") is not False:
        reasons.append("old lane has active, pending or reserved future work")
    if "active_jobs" not in receipt or "pending_jobs" not in receipt:
        reasons.append("old lane task inventory absent")
    return reasons


def admission_reasons(observation, release, commit, receipt=None, max_age=60):
    reasons = release_reasons(receipt) if receipt is not None else []
    try:
        stamp = observation.get("checked_at_utc", observation.get("observed_utc"))
        age=(datetime.now(timezone.utc)-datetime.fromisoformat(stamp.replace("Z", "+00:00"))).total_seconds()
        if not 0 <= age <= max_age:reasons.append("stale resource observation")
    except (AttributeError, TypeError, ValueError):reasons.append("invalid resource observation time")
    if release.get("port") != 20474 or observation.get("port") != 20474 or release.get("scientific_commit") != commit:
        reasons.append("lane or scientific source identity mismatch")
    if release.get("old_lane_all_work_terminal") is not True or release.get("old_future_training_reserved") is not False:
        reasons.append("old scheduler still owns lane")
    if not release.get("gpu_uuid") or release["gpu_uuid"] not in observation.get("gpu", ""):
        reasons.append("GPU identity missing or changed")
    if observation.get("gpu_query_exit") != 0 or observation.get("compute_query_exit") != 0:
        reasons.append("GPU or compute inventory failed")
    if "compute_apps" not in observation or str(observation.get("compute_apps", "")).strip():
        reasons.append("GPU compute owner exists or inventory absent")
    if observation.get("disk_free", 0) < 12*1024**3:reasons.append("disk reserve below 12 GiB")
    if observation.get("memory", {}).get("MemAvailable", 0) < 8*1024**3:reasons.append("RAM reserve below 8 GiB")
    if not observation.get("cpu_affinity") or not observation.get("cgroup"):
        reasons.append("CPU or cgroup inventory missing")
    try:
        fields = observation["gpu"].splitlines()[0].split(",")
        if int(fields[-1].strip()) >= 83:reasons.append("GPU temperature at admission limit")
    except (KeyError, ValueError, IndexError):reasons.append("GPU temperature missing")
    old = observation.get("old_tasks")
    if not isinstance(old, (list, dict)):
        reasons.append("old API inventory missing")
    elif isinstance(old, dict) and "unavailable" in old:
        retired = receipt.get("old_service_retired_verified") if receipt is not None else release.get("old_service_retired_verified")
        if retired is not True:reasons.append("old API unavailable without verified retirement receipt")
    else:
        jobs = old if isinstance(old, list) else old.get("tasks", old.get("jobs", old.get("runs")))
        if not isinstance(jobs, list):reasons.append("old task schema unknown")
        else:
            for row in jobs:
                status=row.get("status",row)
                if not isinstance(status,dict) or status.get("stage") not in TERMINAL:
                    reasons.append("old API has pending or active work");break
    if "processes" not in observation:reasons.append("old process inventory missing")
    for process in observation.get("processes", []):
        if process.get("pid") == observation.get("probe_pid"):continue
        command=process.get("command", "")
        if ("s1_nd12" in command or "nondrl12" in command) and not ("workbench.server" in command or "tunnel" in command):
            reasons.append("old lane worker or coordinator remains: "+str(process.get("pid")))
    return reasons


def local_observation(archive_root):
    """Taken by the scheduler immediately before Popen, independent of proof booleans."""
    observed={"observed_utc":datetime.now(timezone.utc).isoformat(), "port":20474, "probe_pid":os.getpid()}
    for key, query in (("gpu", "--query-gpu=uuid,name,memory.used,memory.free,utilization.gpu,temperature.gpu"),
                       ("compute_apps", "--query-compute-apps=pid,gpu_uuid,used_memory")):
        result=subprocess.run(["nvidia-smi",query,"--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=15)
        observed[key]=result.stdout.strip()
        observed["gpu_query_exit" if key=="gpu" else "compute_query_exit"]=result.returncode
    try:
        with urllib.request.urlopen("http://127.0.0.1:8943/api/coverage/nondrl/runs",timeout=10) as response:
            observed["old_tasks"]=json.load(response)
    except Exception as error:
        observed["old_tasks"]={"unavailable":type(error).__name__}
    observed["processes"]=[]
    proc=Path("/proc")
    for directory in proc.iterdir():
        if not directory.name.isdigit() or int(directory.name)==os.getpid():continue
        try:
            command=(directory/"cmdline").read_bytes().replace(b"\0",b" ").decode("utf-8","replace")
            if "satellite_rl" in command or "nondrl12" in command:
                raw=(directory/"stat").read_text()
                observed["processes"].append({"pid":int(directory.name), "birth_ticks":int(raw[raw.rfind(")")+2:].split()[19]), "command":command})
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    observed["cpu_affinity"]=sorted(os.sched_getaffinity(0))
    observed["cgroup"]=cgroup_inventory()
    observed["memory"]={}
    for line in (proc/"meminfo").read_text().splitlines():
        if line.startswith(("MemAvailable:","MemTotal:")):
            observed["memory"][line.split(":")[0]]=int(line.split()[1])*1024
    observed["disk_free"]=shutil.disk_usage(archive_root).free
    return observed
