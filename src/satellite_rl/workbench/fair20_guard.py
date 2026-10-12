"""Shared pure lease checks and fresh local resource inventory for F20."""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request

TERMINAL = {"completed", "failed", "stopped", "cancelled", "invalidated"}
EXECUTOR_PORTS = {20418, 20474}
OLD_API_PORTS = {20418: 8940, 20474: 8943}


def executor_port(archive_root=None, configured=None):
    """Bind configured executor to the actual isolated archive lane."""
    setting = configured if configured is not None else os.environ.get("SATELLITE_RL_F20_EXECUTOR_PORT")
    physical = None
    if archive_root is not None:
        try:
            physical = int(Path(archive_root).resolve().parent.name)
        except ValueError:
            if setting is None:
                raise ValueError("F20 archive has no actual executor-port lane") from None
    port = int(setting) if setting is not None else physical
    if port not in EXECUTOR_PORTS or (physical is not None and port != physical):
        raise ValueError("F20 executor must match physical20418/20474 archive lane")
    return port


_resolve_executor_port = executor_port


def _evidence_reference_valid(reference):
    from .storage import digest
    try:
        return isinstance(reference, dict) and digest(Path(reference["path"])) == reference["sha256"]
    except (OSError, KeyError, TypeError):
        return False


def _completed_release_evidence(receipt):
    """Read SHA-bound completed-group/stop records, not just proof booleans."""
    from .storage import read_json
    references = [receipt.get(name) for name in ("completion_evidence", "coordinator_evidence", "heartbeat_evidence")]
    if not all(_evidence_reference_valid(reference) for reference in references):
        return False
    completion, coordinator, heartbeat = [read_json(reference["path"], {}) for reference in references]
    if (completion.get("experiment_id") != "S1-ND12" or completion.get("scientific_complete") is not True
            or completion.get("scientific_completed_tasks") != 223 or completion.get("active_jobs")
            or completion.get("pending_jobs") or "active_jobs" not in completion or "pending_jobs" not in completion):
        return False
    identities = coordinator.get("identities")
    if (coordinator.get("stopped") is not True or coordinator.get("future_launch_disabled") is not True
            or not isinstance(identities, list) or not identities):
        return False
    for identity in identities:
        if (not isinstance(identity.get("pid"), int) or identity["pid"] <= 0
                or not isinstance(identity.get("expected_birth"), (int, float))
                or not math.isfinite(identity["expected_birth"]) or identity["expected_birth"] < 0
                or identity.get("same_identity_alive") is not False):
            return False
        if identity.get("pid_exists") is True:
            actual = identity.get("actual_birth")
            command = identity.get("current_command")
            if (not isinstance(actual, (int, float)) or not math.isfinite(actual) or actual < 0
                    or actual == identity["expected_birth"] or not isinstance(command, str) or not command.strip()
                    or any(marker in command.lower() for marker in ("s1_nd12", "nondrl12", "coverage_nondrl"))):
                return False
        elif identity.get("pid_exists") is not False:
            return False
    return heartbeat.get("stopped") is True and heartbeat.get("active") is False and bool(heartbeat.get("automation_id"))


def _idle_completed_service_matches(process, receipt, port, old_tasks):
    """One dispatch-capable but completed/idle workbench, exactly identified."""
    if port != 20418 or not receipt or not _completed_release_evidence(receipt):
        return False
    reference = receipt.get("inert_service_evidence")
    if not _evidence_reference_valid(reference):
        return False
    from .storage import read_json, digest
    try:
        if not isinstance(old_tasks, (list, dict)):
            return False
        jobs = old_tasks if isinstance(old_tasks, list) else old_tasks.get("tasks", old_tasks.get("jobs", old_tasks.get("runs")))
        if not isinstance(jobs, list) or not jobs:
            return False
        for row in jobs:
            if not isinstance(row, dict):
                return False
            status = row.get("status") if isinstance(row.get("status"), dict) else row
            if status.get("stage") not in TERMINAL:
                return False
        evidence = read_json(reference["path"], {})
        audit_reference = evidence.get("audit_source_reference")
        if not _evidence_reference_valid(audit_reference):
            return False
        audit = read_json(audit_reference["path"], {})
        command = "/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python -B /home/u2024110224/first_stage_satellite_rl/s1_nd12/20418/ops/datafeed_service_252e05a9cc02.py"
        source = "/home/u2024110224/first_stage_satellite_rl/s1_nd12/20418/ops/datafeed_service_252e05a9cc02.py"
        source_sha = "9c8d1f212108bfdd645907e953590cfccc5dbf17a08e6c1d8020067bcd44bc71"
        return (evidence.get("port") == 20418 and evidence.get("pid") == 32504
            and evidence.get("birth_ticks") == 835345015 and evidence.get("exact_command") == command
            and evidence.get("role") == "idle_completed_nd12_workbench_service"
            and evidence.get("service_source_path") == source
            and evidence.get("independent_idle_service_audit_verified") is True
            and audit.get("source_sha256") == source_sha
            and audit.get("service_has_scheduler_capability") is True
            and audit.get("scientific_group_complete") is True
            and audit.get("pending_jobs") == []
            and audit.get("future_coordinator_stopped") is True
            and audit.get("heartbeat_paused") is True
            and digest(source) == evidence.get("service_source_sha256") == source_sha
            and process.get("pid") == evidence["pid"]
            and process.get("birth_ticks") == evidence["birth_ticks"]
            and process.get("command", "").strip() == command)
    except (OSError, KeyError, TypeError, ValueError):
        return False


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


def release_reasons(receipt, expected_port=None):
    reasons=[]
    port = expected_port if expected_port is not None else receipt.get("released_ssh_port")
    if port not in EXECUTOR_PORTS or receipt.get("released") is not True or receipt.get("released_ssh_port") != port:
        reasons.append("old lane release is not verified")
    if port == 20418:
        if (receipt.get("release_basis") != "nd12_scientific_complete_and_persistent_followups_stopped"
                or receipt.get("scientific_completed_tasks") != 223
                or receipt.get("scientific_completion_verified") is not True
                or not _evidence_reference_valid(receipt.get("completion_evidence"))):
            reasons.append("223-task scientific completion evidence missing")
        if (receipt.get("persistent_coordinator_stopped_verified") is not True
                or not _evidence_reference_valid(receipt.get("coordinator_evidence"))
                or receipt.get("heartbeat_stopped_verified") is not True
                or not _evidence_reference_valid(receipt.get("heartbeat_evidence"))):
            reasons.append("persistent old coordinator/heartbeat stop evidence missing")
        if not _completed_release_evidence(receipt):
            reasons.append("completed-group stopped identities or heartbeat source records contradicted")
    elif port == 20474:
        if port not in receipt.get("old_coordinator_excluded_ports", []):
            reasons.append("old coordinator does not exclude physical20474")
        if receipt.get("persistent_coordinator_exclusion_verified") is not True or not receipt.get("coordinator_evidence"):
            reasons.append("persistent old coordinator exclusion evidence missing")
    if receipt.get("active_jobs") or receipt.get("pending_jobs") or receipt.get("future_lane_reserved") is not False:
        reasons.append("old lane has active, pending or reserved future work")
    if "active_jobs" not in receipt or "pending_jobs" not in receipt:
        reasons.append("old lane task inventory absent")
    return reasons


def admission_reasons(observation, release, commit, receipt=None, max_age=60, executor_port=None):
    port = executor_port if executor_port is not None else release.get("port")
    reasons = release_reasons(receipt, port) if receipt is not None else []
    if port not in EXECUTOR_PORTS:
        reasons.append("unapproved executor port")
    try:
        stamp = observation.get("checked_at_utc", observation.get("observed_utc"))
        age=(datetime.now(timezone.utc)-datetime.fromisoformat(stamp.replace("Z", "+00:00"))).total_seconds()
        if not 0 <= age <= max_age:reasons.append("stale resource observation")
    except (AttributeError, TypeError, ValueError):reasons.append("invalid resource observation time")
    if (release.get("port") != port or observation.get("port") != port or
            release.get("executor_port", port) != port or observation.get("executor_port", port) != port
            or release.get("scientific_commit") != commit):
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
                if not isinstance(row, dict):
                    reasons.append("old task row schema unknown");break
                # Coverage API includes a display status string alongside its
                # actual stage; only the actual stage determines task ownership.
                status=row.get("status") if isinstance(row.get("status"), dict) else row
                if not isinstance(status,dict) or status.get("stage") not in TERMINAL:
                    reasons.append("old API has pending or active work");break
    if "processes" not in observation:reasons.append("old process inventory missing")
    for process in observation.get("processes", []):
        if process.get("pid") == observation.get("probe_pid"):continue
        command=process.get("command", "")
        if ("s1_nd12" in command or "nondrl12" in command) and not ("workbench.server" in command or "tunnel" in command):
            if not _idle_completed_service_matches(process, receipt, port, observation.get("old_tasks")):
                reasons.append("old lane worker or coordinator remains: "+str(process.get("pid")))
    return reasons


def local_observation(archive_root, executor_port=None):
    """Taken by the scheduler immediately before Popen, independent of proof booleans."""
    port = _resolve_executor_port(archive_root, executor_port)
    observed={"observed_utc":datetime.now(timezone.utc).isoformat(), "port":port,
              "executor_port":port, "old_api_port":OLD_API_PORTS[port], "probe_pid":os.getpid()}
    for key, query in (("gpu", "--query-gpu=uuid,name,memory.used,memory.free,utilization.gpu,temperature.gpu"),
                       ("compute_apps", "--query-compute-apps=pid,gpu_uuid,used_memory")):
        result=subprocess.run(["nvidia-smi",query,"--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=15)
        observed[key]=result.stdout.strip()
        observed["gpu_query_exit" if key=="gpu" else "compute_query_exit"]=result.returncode
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{OLD_API_PORTS[port]}/api/coverage/nondrl/runs",timeout=10) as response:
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
