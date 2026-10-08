"""Explicit S2-P19 core archive waiter; default commands are local-only.

--run alone enables polling, workbench GETs, remote stat-only inventories and
the completed-only serial core archive. It never deletes original/partial
assets and never marks supplemental evidence or Report 15 complete.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from . import archive, archive_core, ops


DEFAULT_STATUS = ops.LOCAL / "remote_archive/core_archive_waiter_status.json"
DEFAULT_LOCK = ops.LOCAL / "remote_archive/core_archive_waiter.lock"
DEFAULT_INVENTORY = ops.LOCAL / "remote_archive/core_inventory"
WAIT_DEADLINE_UTC = "2026-10-09T17:51:53+00:00"
ARCHIVE_TIMEOUT = timedelta(hours=72)
POLL_SECONDS = 60
MAX_GATE_RETRIES = 10
MAX_NETWORK_RETRIES = 3
MAX_GIB = 256
PART_MIB = 1024
COMPRESSION_LEVEL = 6
REMOTE_TIMEOUT_SECONDS = 43200
REMOTE_RESERVE_GIB = 12
LOCAL_RESERVE_GIB = archive.LOCAL_RESERVE_GIB


REMOTE_METADATA = r'''
import json,os,pathlib
request=json.loads(REQUEST_JSON)
port=request['port'];run_id=request['run_id']
base=pathlib.Path('/home/u2024110224/second_stage_satellite_rl/sac_routes_s2p19')/str(port)
archive=(base/'archive').resolve()
jobdir=archive/'outputs/workbench/jobs'/run_id
job=json.loads((jobdir/'job.json').read_text(encoding='utf-8'))
status=json.loads((jobdir/'status.json').read_text(encoding='utf-8'))
if (job.get('run_id')!=run_id or status.get('run_id')!=run_id or
    job.get('experiment_id')!='S2-P19' or job.get('archive_root')!=str(archive) or
    status.get('stage')!='completed'):
 raise RuntimeError('inventory requires completed same-archive S2-P19 job')
def safe(path):
 path=pathlib.Path(path)
 if path.is_symlink():raise RuntimeError('symlink refused: '+str(path))
 resolved=path.resolve(strict=True)
 if os.path.commonpath((str(archive),str(resolved)))!=str(archive):
  raise RuntimeError('inventory source outside owned archive: '+str(path))
 return resolved
items={}
def add(member,path):
 if member.startswith('/') or '..' in pathlib.PurePosixPath(member).parts or member in items:
  raise RuntimeError('duplicate or unsafe inventory member: '+member)
 path=safe(path)
 if not path.is_file():raise RuntimeError('missing inventory file: '+str(path))
 items[member]={'member':member,'bytes':path.stat().st_size}
def tree(prefix,root):
 root=safe(root)
 if not root.is_dir():raise RuntimeError('missing inventory root: '+str(root))
 for file in sorted(root.rglob('*')):
  if file.is_symlink():raise RuntimeError('symlink under inventory root: '+str(file))
  if file.is_file():add(prefix+'/'+file.relative_to(root).as_posix(),file)
tree('job',jobdir)
source=safe(job['source_root'])
source_manifest=json.loads((source/'source_manifest.json').read_text(encoding='utf-8'))
if source_manifest.get('git_commit')!=job.get('git_commit') or not source_manifest.get('files'):
 raise RuntimeError('frozen source manifest/commit differs')
for relative in source_manifest['files']:
 file=safe(source/relative)
 if os.path.commonpath((str(source),str(file)))!=str(source) or not file.is_file():
  raise RuntimeError('declared source file missing/outside source: '+relative)
tree('source',source)
for role,record in sorted((job.get('scene_sets') or {}).items()):
 if role not in ('train','development','validation','test') or not isinstance(record,dict):
  raise RuntimeError('unexpected scene role')
 refs=record.get('files') or []
 if record.get('count')!=len(refs) or not refs:
  raise RuntimeError('frozen scene set incomplete: '+role)
 names=set()
 for ref in refs:
  path=safe(ref['path'])
  if path.suffix!='.npz' or path.name in names:
   raise RuntimeError('scene NPZ name missing or duplicate: '+role)
  names.add(path.name)
  add('scenes/'+role+'/'+path.name,path)
 record_file=pathlib.Path(record['path'])/'record.json'
 if record_file.is_file():add('scenes/'+role+'/record.json',record_file)
model_dir=job.get('model_dir')
if model_dir and pathlib.Path(model_dir).is_dir():tree('weights/model_dir',model_dir)
for idx,ref in enumerate(job.get('model_files') or []):
 add('weights/model_input/'+str(idx).zfill(3)+'_'+pathlib.Path(ref['path']).name,ref['path'])
stat=os.statvfs(archive)
ordered=[items[name] for name in sorted(items)]
print(json.dumps({'schema':'s2p19.core_inventory.v1','port':port,'run_id':run_id,
 'member_count':len(ordered),'total_input_bytes':sum(x['bytes'] for x in ordered),
 'members':ordered,'remote_statvfs_free_bytes':stat.f_bavail*stat.f_frsize,
 'remote_statvfs_scope':'filesystem available blocks; not a user quota'}))
'''


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_commit() -> str:
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True, check=True)
    return result.stdout.strip()


@contextmanager
def exclusive_lock(path: Path):
    """Nonblocking OS lock; a dead process leaves no held lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def local_gate(active: dict, state: dict, historical: dict, manifest: dict,
               *, plan_sha256: str) -> tuple[list[dict], dict]:
    if state.get("plan_sha256") != plan_sha256:
        raise ValueError("four-lane state plan SHA differs")
    jobs = archive_core.planned_jobs(active, state, historical, manifest)
    final_ids = {task["id"] for task in active["tasks"]}
    pending = [task_id for task_id in sorted(final_ids)
               if (state.get("tasks") or {}).get(task_id, {}).get("state") != "completed"]
    return jobs, {"ready": not pending, "completed_new": 7 - len(pending),
                  "new_total": 7, "legacy_bound": len(active["completed_legacy"]),
                  "pending": pending, "observed_utc": utc_now(),
                  "observation_source": "local four_lane_state only"}


def probe_inventory(item: dict) -> dict:
    request = {"port": item["port"], "run_id": item["run_id"]}
    code = "REQUEST_JSON=" + repr(json.dumps(request, separators=(",", ":"))) + "\n" + REMOTE_METADATA
    response = ops.remote_python(code, item["port"], timeout=300)
    if response["exit_code"]:
        raise RuntimeError("remote metadata inventory failed: " + response["stderr"][-1500:])
    result = json.loads(response["stdout"].splitlines()[-1])
    result["observed_utc"] = utc_now()
    return result


def inventory_gate(jobs: list[dict], probe=probe_inventory, *,
                   destination: Path = archive.DEFAULT_DESTINATION) -> dict:
    """Mirror archive member names/bytes without hashing or compression."""
    observed, pending = {}, []
    for item in jobs:
        result = probe(item)
        task_id = item["task_id"]
        members = result.get("members") or []
        names = [entry.get("member") for entry in members]
        valid_sizes = all(type(entry.get("bytes")) is int and entry["bytes"] >= 0
                          for entry in members)
        total = sum(entry["bytes"] for entry in members) if valid_sizes else -1
        part_bytes = PART_MIB * 1024 ** 2
        groups, amount, current = 0, 0, False
        for entry in members:
            size = entry["bytes"] if valid_sizes else 0
            if current and amount + size > part_bytes:
                groups += 1
                amount = 0
                current = False
            amount += size
            current = True
        groups += current
        reasons = []
        if (result.get("schema") != "s2p19.core_inventory.v1" or
                result.get("port") != item["port"] or result.get("run_id") != item["run_id"]):
            reasons.append("inventory run/port identity")
        if (not members or len(names) != len(set(names)) or
                any(not isinstance(name, str) or name.startswith("/") or
                    ".." in name.split("/") for name in names) or
                not valid_sizes or
                result.get("member_count") != len(members) or
                result.get("total_input_bytes") != total):
            reasons.append("inventory member/byte integrity")
        if (total > MAX_GIB * 1024 ** 3 or len(members) > archive.MAX_FILES or
                groups > archive.MAX_VOLUMES):
            reasons.append("archive input/file/volume bound")
        if result.get("remote_statvfs_free_bytes", 0) < REMOTE_RESERVE_GIB * 1024 ** 3:
            reasons.append("remote statvfs reserve (not quota)")
        observed[task_id] = {**result, "predicted_volume_count": groups,
                             "reasons": reasons}
        if reasons:
            pending.append(task_id + ": " + ", ".join(reasons))
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    local_free = shutil.disk_usage(destination).free
    if local_free < LOCAL_RESERVE_GIB * 1024 ** 3:
        pending.append("local compressed ZIP reserve unavailable")
    return {"ready": not pending and len(observed) == 15,
            "checked_utc": utc_now(), "jobs": observed, "pending": pending,
            "local_free_bytes": local_free,
            "local_reserve_bytes": LOCAL_RESERVE_GIB * 1024 ** 3,
            "remote_statvfs_is_not_quota": True,
            "compression_ratio_unknown": True}


def save_status(path: Path, status: dict, phase: str, **fields) -> dict:
    status = dict(status)
    now = utc_now()
    status["phase"] = phase
    status["updated_utc"] = now
    status.setdefault("phase_times", {}).setdefault(phase, now)
    status.update(fields)
    archive_core.write_json(path, status)
    return status


def _transient(error: Exception) -> bool:
    return isinstance(error, (ConnectionError, TimeoutError, OSError)) or any(
        word in str(error).lower() for word in ("ssh", "transport", "timeout", "banner", "socket"))


def run_waiter(*, plan_path: Path, state_path: Path, historical_path: Path,
               manifest_path: Path, status_path: Path, lock_path: Path,
               inventory_dir: Path, ledger_path: Path, source_index: Path,
               destination: Path, poll_seconds: int = POLL_SECONDS,
               deadline_utc: str = WAIT_DEADLINE_UTC) -> dict:
    plan_path = Path(plan_path)
    active = archive_core.read_json(plan_path)
    historical = archive_core.read_json(historical_path)
    manifest = archive_core.read_json(manifest_path)
    plan_sha = archive.digest(plan_path)
    deadline = datetime.fromisoformat(deadline_utc)
    if deadline.tzinfo is None:
        raise ValueError("wait deadline requires UTC offset")
    with exclusive_lock(lock_path):
        current_commit = source_commit()
        if Path(status_path).is_file():
            status = archive_core.read_json(status_path)
            if (status.get("schema") != "s2p19.archive_waiter.v1" or
                    status.get("plan_sha256") != plan_sha or
                    status.get("source_commit") != current_commit):
                raise ValueError("existing waiter status source/plan identity differs")
            if status.get("phase") == "core_complete":
                return status
            status["restart_count"] = status.get("restart_count", 0) + 1
            status["restarted_utc"] = utc_now()
            status["pid"] = os.getpid()
        else:
            status = {"schema": "s2p19.archive_waiter.v1", "pid": os.getpid(),
                      "source_commit": current_commit, "plan_sha256": plan_sha,
                      "started_utc": utc_now(), "wait_deadline_utc": deadline_utc,
                      "supplemental_archive": "pending_separate",
                      "formal_report": "pending_separate"}
        status = save_status(status_path, status, "waiting")
        while True:
            state = archive_core.read_json(state_path)
            jobs, local = local_gate(active, state, historical, manifest,
                                     plan_sha256=plan_sha)
            status = save_status(status_path, status, "waiting", local_gate=local)
            if local["ready"]:
                break
            if datetime.now(timezone.utc) >= deadline:
                return save_status(status_path, status, "attention",
                                   reason="wait deadline reached before seven local completions")
            time.sleep(poll_seconds)
        for attempt in range(1, MAX_GATE_RETRIES + 1):
            status = save_status(status_path, status, "gating", gate_attempt=attempt)
            gate = archive_core.live_gate(jobs)
            status = save_status(status_path, status, "gating", live_gate=gate)
            if gate["ready"]:
                break
            if datetime.now(timezone.utc) >= deadline or attempt == MAX_GATE_RETRIES:
                return save_status(status_path, status, "attention",
                                   reason="live core completion gate pending")
            time.sleep(poll_seconds)
        for attempt in range(1, MAX_NETWORK_RETRIES + 1):
            status = save_status(status_path, status, "inventory",
                                 inventory_attempt=attempt)
            try:
                inventory = inventory_gate(jobs, destination=destination)
                inventory_dir = Path(inventory_dir)
                inventory_dir.mkdir(parents=True, exist_ok=True)
                for task_id, item in inventory["jobs"].items():
                    archive_core.write_json(inventory_dir / (task_id + ".json"), item)
                summary = {key: value for key, value in inventory.items() if key != "jobs"}
                status = save_status(status_path, status, "inventory",
                                     inventory=summary, inventory_directory=str(inventory_dir))
                if not inventory["ready"]:
                    return save_status(status_path, status, "attention",
                                       reason="metadata inventory/capacity gate failed")
                break
            except Exception as error:
                status = save_status(status_path, status, "inventory",
                                     last_error=str(error))
                if not _transient(error) or attempt == MAX_NETWORK_RETRIES:
                    return save_status(status_path, status, "attention",
                                       reason="metadata inventory failed: " + str(error))
                time.sleep(poll_seconds)
        archive_deadline = (datetime.fromisoformat(status["archive_deadline_utc"])
                            if status.get("archive_deadline_utc") else
                            datetime.now(timezone.utc) + ARCHIVE_TIMEOUT)
        status = save_status(status_path, status, "archiving",
                             archive_deadline_utc=archive_deadline.isoformat())
        for attempt in range(1, MAX_NETWORK_RETRIES + 1):
            if datetime.now(timezone.utc) >= archive_deadline:
                return save_status(status_path, status, "attention",
                                   reason="bounded archive duration exceeded")
            try:
                ledger = archive_core.execute(jobs, gate, ledger_path=ledger_path,
                    source_index=source_index, destination=destination,
                    deadline_utc=archive_deadline,
                    archive_options={"part_mib": PART_MIB, "max_gib": MAX_GIB,
                                     "compression_level": COMPRESSION_LEVEL,
                                     "remote_timeout_seconds": REMOTE_TIMEOUT_SECONDS})
                return save_status(status_path, status, "core_complete",
                                   core_count=len(ledger["completed"]),
                                   core_ledger=str(ledger_path))
            except Exception as error:
                status = save_status(status_path, status, "archiving",
                                     archive_attempt=attempt, last_error=str(error))
                if not _transient(error) or attempt == MAX_NETWORK_RETRIES:
                    return save_status(status_path, status, "attention",
                                       reason="core archive failed: " + str(error))
                time.sleep(poll_seconds)
                gate = archive_core.live_gate(jobs)
                status = save_status(status_path, status, "gating", live_gate=gate)
                if not gate["ready"]:
                    return save_status(status_path, status, "attention",
                                       reason="live core gate changed during retry")
                status = save_status(status_path, status, "archiving")
    raise RuntimeError("unreachable archive waiter state")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=archive_core.DEFAULT_PLAN)
    parser.add_argument("--state", type=Path, default=archive_core.DEFAULT_STATE)
    parser.add_argument("--historical-plan", type=Path,
                        default=archive_core.DEFAULT_HISTORICAL_PLAN)
    parser.add_argument("--manifest", type=Path, default=archive_core.DEFAULT_MANIFEST)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--inventory-dir", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--ledger", type=Path, default=archive_core.DEFAULT_LEDGER)
    parser.add_argument("--source-index", type=Path, default=archive.SOURCE_INDEX)
    parser.add_argument("--destination", type=Path, default=archive.DEFAULT_DESTINATION)
    parser.add_argument("--deadline-utc", default=WAIT_DEADLINE_UTC)
    parser.add_argument("--poll-seconds", type=int, default=POLL_SECONDS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="local task list only")
    mode.add_argument("--check", action="store_true", help="local state check only (default)")
    mode.add_argument("--run", action="store_true", help="explicit persistent remote workflow")
    args = parser.parse_args()
    if not 5 <= args.poll_seconds <= 3600:
        parser.error("poll interval must be 5..3600 seconds")
    active = archive_core.read_json(args.plan)
    state = archive_core.read_json(args.state)
    jobs, local = local_gate(active, state,
        archive_core.read_json(args.historical_plan),
        archive_core.read_json(args.manifest), plan_sha256=archive.digest(args.plan))
    if args.list:
        print(json.dumps({"jobs": jobs, "local_gate": local,
                          "remote_called": False}, ensure_ascii=False))
    elif not args.run:
        print(json.dumps({"local_gate": local, "remote_called": False,
                          "saved_status": (archive_core.read_json(args.status)
                                           if args.status.is_file() else None)},
                         ensure_ascii=False))
    else:
        print(json.dumps(run_waiter(plan_path=args.plan, state_path=args.state,
            historical_path=args.historical_plan, manifest_path=args.manifest,
            status_path=args.status, lock_path=args.lock,
            inventory_dir=args.inventory_dir, ledger_path=args.ledger,
            source_index=args.source_index, destination=args.destination,
            poll_seconds=args.poll_seconds, deadline_utc=args.deadline_utc),
            ensure_ascii=False))


if __name__ == "__main__":
    main()
