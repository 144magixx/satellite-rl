"""Read-only, SHA-verified local mirror of completed S1-03 workbench jobs.

The controller owns scientific scheduling. This watcher never submits, stops or
resumes a job, and never treats an active checkpoint or partial JSONL as final.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import time
import uuid

try:
    from . import remote
except ImportError:
    import remote


LOCAL = Path("D:/paper/outputs/experiments/first_stage_s1_03_20261005")
MODELS = Path("D:/paper/models/runs/first_stage_s1_03_20261005")
STATE = LOCAL / "archive_state.json"
LEDGER = LOCAL / "archive_ledger.jsonl"
GROUP = "s1_03"


def configure_group(group):
    global GROUP, LOCAL, MODELS, STATE, LEDGER
    profile = remote.group_profile(group)
    GROUP = group
    LOCAL, MODELS = profile["local"], profile["models"]
    STATE, LEDGER = LOCAL / "archive_state.json", LOCAL / "archive_ledger.jsonl"


ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
LABEL_PORT = {"prepare_validation": 20418, "prepare_train": 20418,
              "train_P1": 20418, "prepare_test": 20418,
              **{f"dev_{name}": 20418 for name in ("R0", "R1", "R2", "R3")},
              **{f"formal_{name}": 20424 for name in ("R0", "R1", "R2", "R3", "P1")}}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    os.replace(temp, path)


def append(value):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


@contextmanager
def single_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"1")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("Another S1-03 archiver holds the lock") from error
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError("Another S1-03 archiver holds the lock") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def verified_inputs():
    plan_path, control_path = LOCAL / "plan.json", LOCAL / "controller_state.json"
    raw = plan_path.read_bytes()
    plan = json.loads(raw.decode("utf-8-sig"))
    controller = read(control_path)
    plan_sha = hashlib.sha256(raw).hexdigest()
    profile = remote.group_profile(GROUP)
    if (plan.get("experiment_id") != profile["experiment_id"] or
            plan.get("hostroot") != profile["remote_base"] or
            plan.get("ports") != [20418, 20424] or
            not COMMIT.fullmatch(plan.get("science_commit", "")) or
            not isinstance(controller, dict) or
            controller.get("plan_sha256") != plan_sha or
            controller.get("science_commit") != plan["science_commit"]):
        raise ValueError("Local S1-03 plan/controller identity differs")
    jobs = controller.get("jobs", {})
    if not isinstance(jobs, dict) or set(jobs) - set(LABEL_PORT):
        raise ValueError("Controller lists an unauthorized S1-03 job label")
    seen = {}
    for label, run_id in jobs.items():
        if not isinstance(run_id, str) or not ID.fullmatch(run_id):
            raise ValueError("Controller job ID is not a safe original ID")
        if run_id in seen and seen[run_id] != LABEL_PORT[label]:
            raise ValueError("One run ID belongs to both instances")
        seen[run_id] = LABEL_PORT[label]
    return plan, controller, plan_sha


def archive_root(port):
    if port not in (20418, 20424):
        raise ValueError("Only authorized S1-03 instances can be archived")
    return f"{remote.group_profile(GROUP)['remote_base']}/{port}"


def remote_json(sftp, path):
    with sftp.open(path, "rb") as stream:
        return json.loads(stream.read().decode("utf-8-sig"))


def job_paths(port, run_id):
    if not ID.fullmatch(run_id):
        raise ValueError("Unsafe job ID")
    root = archive_root(port)
    return (f"{root}/outputs/workbench/jobs/{run_id}",
            f"{root}/models/runs/workbench/{run_id}/final",
            f"{root}/outputs/workbench/source_snapshots")


def completed_job(sftp, plan, label, port, run_id):
    directory, model, source_parent = job_paths(port, run_id)
    job = remote_json(sftp, directory + "/job.json")
    status = remote_json(sftp, directory + "/status.json")
    config = remote_json(sftp, directory + "/config.json")
    if (job.get("run_id") != run_id or job.get("git_commit") != plan["science_commit"] or
            job.get("research_stage") != "coverage" or
            job.get("execution_target") != f"u2024110224@10.160.17.102:{port}" or
            config.get("schema_version") != 4 or
            config.get("research_stage") != "coverage" or
            config.get("coverage", {}).get("reconstruction", {}).get("route") !=
            (label[4:] if label.startswith("dev_") else
             label[7:] if label.startswith("formal_") else
             "P1" if label == "train_P1" else "prepare") or
            job.get("kind") != ("train" if label == "train_P1" else "evaluate")):
        raise ValueError(f"{label} remote job identity/route/source differs")
    source = job.get("source_root")
    expected = source_parent + "/" + plan["science_commit"]
    if source != expected:
        raise ValueError(f"{label} source snapshot is outside frozen S1-03 root")
    if status.get("stage") != "completed":
        return None
    return {"label": label, "port": port, "run_id": run_id,
            "job_dir": directory, "model_dir": model if label == "train_P1" else None,
            "source_root": source, "status": status, "job": job, "config": config}


def inventory(client, roots):
    """Remote pre-download SHA inventory under explicit completed-job roots."""
    for root in roots:
        port = root["port"]
        base = archive_root(port)
        path = root["path"]
        if not (path.startswith(base + "/outputs/workbench/jobs/") or
                path.startswith(base + "/models/runs/workbench/") or
                path == base + "/outputs/workbench/source_snapshots/" +
                root.get("commit", "") or
                path == base + "/ops/audits" or
                path == base + "/ops/reports"):
            raise ValueError("Remote archive inventory escaped S1-03 job/source/report roots")
    code = f'''
import hashlib,json
from pathlib import Path
roots={roots!r};out=[]
for item in roots:
 p=Path(item['path']);scope=Path({remote.group_profile(GROUP)['remote_base']!r})/str(item['port'])
 if not p.resolve().is_relative_to(scope.resolve()):raise RuntimeError('S1-03 path escape')
 if not p.exists():
  if item.get('required',True):raise FileNotFoundError(str(p))
  continue
 files=[p] if p.is_file() else sorted(p.rglob('*'))
 for file in files:
  if file.is_symlink():raise RuntimeError('Symlink in frozen S1-03 archive: '+str(file))
  if not file.is_file():continue
  if item['group'].startswith('source:') and ('__pycache__' in file.relative_to(p).parts or file.suffix in ('.pyc','.pyo')):continue
  h=hashlib.sha256()
  with file.open('rb') as stream:
   for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
  out.append({{'port':item['port'],'group':item['group'],'root':str(p),
              'relative':str(file.relative_to(p)) if p.is_dir() else file.name,
              'remote_path':str(file),'sha256':h.hexdigest(),
              'bytes':file.stat().st_size}})
print(json.dumps(out))
'''
    return remote.run_code(client, code, timeout=300)


def validate_inventory(entries, roots, jobs):
    allowed = {(item["port"], item["group"], item["path"]): item for item in roots}
    result = []
    names = set()
    for item in entries:
        key = (item.get("port"), item.get("group"), item.get("root"))
        if key not in allowed:
            raise ValueError("Remote inventory returned an unrequested source")
        rel = PurePosixPath(item.get("relative", ""))
        if (rel.is_absolute() or not rel.parts or ".." in rel.parts or
                not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", "")) or
                type(item.get("bytes")) is not int or item["bytes"] < 0):
            raise ValueError("Unsafe remote archive file identity")
        remote_path = item["root"].rstrip("/") + "/" + str(rel)
        if item.get("remote_path") != remote_path:
            raise ValueError("Remote file path differs from confined inventory root")
        if key + (str(rel),) in names:
            raise ValueError("Duplicate remote archive file")
        names.add(key + (str(rel),))
        result.append(item)
    for job in jobs:
        job_files = {item["relative"] for item in result if item["port"] == job["port"] and
                     item["root"] == job["job_dir"]}
        required = {"job.json", "config.json", "status.json", "artifacts.json"}
        if job["label"].startswith("prepare_"):
            required.add("starts.jsonl")
        elif job["label"] == "train_P1":
            required |= {"checkpoint.pt", "updates.jsonl", "episodes.jsonl",
                         "evaluations.jsonl"}
            model_files = {item["relative"] for item in result if item["port"] == job["port"]
                           and item["root"] == job["model_dir"]}
            if model_files != {"policy.pt", "model.json"}:
                raise ValueError("Completed P1 model must contain exactly final policy and metadata")
        else:
            required |= {"results.json", "results.partial.jsonl"}
        if not required <= job_files:
            raise ValueError(f"Completed {job['label']} lacks final evidence: {sorted(required-job_files)}")
    return result


def verify_source_manifest(sftp, source_root, entries, commit):
    source = [item for item in entries if item["root"] == source_root]
    by_name = {item["relative"]: item for item in source}
    if "source_manifest.json" not in by_name:
        raise ValueError("Frozen source snapshot lacks source_manifest.json")
    manifest = remote_json(sftp, source_root + "/source_manifest.json")
    if manifest.get("git_commit") != commit or manifest.get("format_version") != 1:
        raise ValueError("Frozen source manifest commit/format differs")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(by_name) - {"source_manifest.json"}:
        raise ValueError("Frozen source manifest is not a complete file inventory")
    for name, expected in files.items():
        if (PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts or
                by_name[name]["sha256"] != expected):
            raise ValueError("Frozen source file differs from source_manifest SHA")
    return manifest


def target(item, jobs):
    group = item["group"]
    port = item["port"]
    rel = Path(*PurePosixPath(item["relative"]).parts)
    if group.startswith("job:"):
        run_id = group[4:]
        if run_id not in jobs or jobs[run_id]["port"] != port:
            raise ValueError("Job destination lacks original ID/instance ownership")
        base = LOCAL / "jobs" / run_id
    elif group.startswith("model:"):
        run_id = group[6:]
        if run_id not in jobs or jobs[run_id]["port"] != port or jobs[run_id]["label"] != "train_P1":
            raise ValueError("Model destination is not the completed S1-03 P1 task")
        base = MODELS / run_id / "final"
    elif group.startswith("source:"):
        commit = group[7:]
        if not COMMIT.fullmatch(commit):
            raise ValueError("Source destination requires frozen commit")
        base = LOCAL / "source_snapshots" / str(port) / commit
    elif group == "audits":
        base = LOCAL / "ops" / f"remote_{port}" / "audits"
    elif group == "reports":
        base = LOCAL / "ops" / f"remote_{port}" / "reports"
    else:
        raise ValueError("Unknown S1-03 archive destination group")
    destination = base / rel
    if not destination.resolve().is_relative_to(base.resolve()):
        raise ValueError("Local archive destination escaped S1-03")
    return destination


def after_hash(client, paths):
    result = {}
    for start in range(0, len(paths), 80):
        batch = paths[start:start + 80]
        code = f'''
import hashlib,json
from pathlib import Path
paths={batch!r};out={{}}
for name in paths:
 p=Path(name)
 if not p.is_file() or p.is_symlink():out[name]=None;continue
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 out[name]={{'sha256':h.hexdigest(),'bytes':p.stat().st_size}}
print(json.dumps(out))
'''
        result.update(remote.run_code(client, code, timeout=300))
    return result


def quarantine(temporary):
    if temporary is None or not temporary.exists():
        return None
    destination = (LOCAL / "archive_conflicts" / now().replace(":", "-") /
                   (uuid.uuid4().hex + ".bin"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, destination)
    return destination


def transfer_group(client, sftp, entries, jobs, prior):
    staged = []
    events = []
    for item in entries:
        destination = target(item, jobs)
        if destination.exists():
            local_sha = sha(destination)
            if local_sha != item["sha256"] or destination.stat().st_size != item["bytes"]:
                raise RuntimeError("Existing S1-03 archive conflicts with remote SHA: " + str(destination))
            staged.append((item, destination, None, "verified_existing"))
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + f".incoming.{os.getpid()}.{uuid.uuid4().hex}")
        try:
            sftp.get(item["remote_path"], str(temporary))
            if temporary.stat().st_size != item["bytes"] or sha(temporary) != item["sha256"]:
                raise RuntimeError("Local temporary download differs from remote before-SHA")
        except BaseException:
            quarantine(temporary)
            raise
        staged.append((item, destination, temporary, "copied_verified"))
    if not staged:
        return events
    try:
        after = after_hash(client, [item["remote_path"] for item, _, _, _ in staged])
        for item, destination, temporary, action in staged:
            live = after.get(item["remote_path"])
            if live != {"sha256": item["sha256"], "bytes": item["bytes"]}:
                raise RuntimeError("Remote after-SHA differs; no changed file published: " +
                                   item["remote_path"])
    except BaseException:
        for _, _, temporary, _ in staged:
            quarantine(temporary)
        raise
    for item, destination, temporary, action in staged:
        if temporary is not None:
            if destination.exists():
                if sha(destination) != item["sha256"]:
                    raise RuntimeError("Local destination appeared with another SHA")
                temporary.unlink()
                action = "verified_existing"
            else:
                os.replace(temporary, destination)
        event = {"checked_utc": now(), "remote_port": item["port"],
                 "remote_path": item["remote_path"], "local_path": str(destination),
                 "sha256": item["sha256"], "bytes": item["bytes"], "action": action}
        events.append(event)
        append(event)
    return events


def archive_once():
    plan, controller, plan_sha = verified_inputs()
    old = read(STATE, {})
    if old and (old.get("plan_sha256") != plan_sha or
                old.get("science_commit") != plan["science_commit"]):
        raise ValueError("Archive ledger belongs to another frozen S1-03 plan")
    state = {**old, "plan_sha256": plan_sha, "science_commit": plan["science_commit"],
             "checked_utc": now(), "controller_phase": controller.get("phase"),
             "controller_attention": controller.get("attention"),
             "pid": os.getpid(), "jobs": dict(old.get("jobs", {})),
             "sources": dict(old.get("sources", {})), "last_errors": []}
    all_events = []
    for port in (20418, 20424):
        selected = [(label, run_id) for label, run_id in controller["jobs"].items()
                    if LABEL_PORT[label] == port]
        if not selected:
            continue
        client = None
        try:
            client = remote.connect(port)
            with client.open_sftp() as sftp:
                jobs = []
                for label, run_id in selected:
                    job = completed_job(sftp, plan, label, port, run_id)
                    if job is not None:
                        jobs.append(job)
                if not jobs:
                    continue
                by_id = {job["run_id"]: job for job in jobs}
                roots = []
                for job in jobs:
                    roots.append({"port": port, "group": "job:" + job["run_id"],
                                  "path": job["job_dir"], "required": True})
                    if job["model_dir"]:
                        roots.append({"port": port, "group": "model:" + job["run_id"],
                                      "path": job["model_dir"], "required": True})
                source = jobs[0]["source_root"]
                source_key = f"{port}:{source}"
                prior_source = state["sources"].get(source_key)
                if prior_source is not None:
                    manifest_path = Path(prior_source["source_manifest"])
                    if (not manifest_path.is_file() or
                            sha(manifest_path) != prior_source.get("source_manifest_sha256")):
                        raise RuntimeError("Previously archived source manifest changed or vanished")
                if prior_source is None:
                    roots.append({"port": port, "group": "source:" + plan["science_commit"],
                                  "path": source, "commit": plan["science_commit"],
                                  "required": True})
                entries = validate_inventory(inventory(client, roots), roots, jobs)
                if prior_source is None:
                    verify_source_manifest(sftp, source, entries, plan["science_commit"])
                events = transfer_group(client, sftp, entries, by_id, state)
                all_events.extend(events)
                for job in jobs:
                    state["jobs"][job["run_id"]] = {"port": port, "label": job["label"],
                        "remote_job_dir": job["job_dir"], "source_root": job["source_root"],
                        "completed_status_checked_utc": now(),
                        "archived_files": sum(item["group"] == "job:" + job["run_id"]
                                              for item in entries)}
                if prior_source is None:
                    source_entries = [item for item in entries if item["group"] ==
                                      "source:" + plan["science_commit"]]
                    state["sources"][source_key] = {"port": port,
                        "remote_source_root": source, "git_commit": plan["science_commit"],
                        "source_manifest": str(LOCAL / "source_snapshots" / str(port) /
                                               plan["science_commit"] / "source_manifest.json"),
                        "source_manifest_sha256": next(item["sha256"] for item in source_entries
                            if item["relative"] == "source_manifest.json"),
                        "file_count": len(source_entries), "verified_utc": now()}
                passed_modes = [mode for mode, receipt in controller.get("audits", {}).items()
                                if receipt.get("passed") is True and
                                receipt.get("port") == port and mode in
                                ("development", "mechanism", "formal")]
                report_ready = controller.get("phase") == "awaiting_final_report"
                optional_roots = []
                if passed_modes:
                    optional_roots.append({"port": port, "group": "audits",
                        "path": archive_root(port) + "/ops/audits", "required": False})
                if report_ready:
                    optional_roots.append({"port": port, "group": "reports",
                        "path": archive_root(port) + "/ops/reports", "required": False})
                if optional_roots:
                    optional = validate_inventory(inventory(client, optional_roots),
                                                  optional_roots, [])
                    optional = [item for item in optional if
                        not item["relative"].endswith((".tmp", ".partial")) and
                        (item["group"] == "reports" or
                         any(item["relative"].startswith(mode + "_")
                             for mode in passed_modes))]
                    try:
                        all_events.extend(transfer_group(client, sftp, optional, by_id, state))
                    except Exception as error:
                        state["last_errors"].append({"port": port, "checked_utc": now(),
                            "group": "optional_report_assets",
                            "error": type(error).__name__ + ": " + str(error)})
        except BaseException as error:
            state["last_errors"].append({"port": port, "checked_utc": now(),
                                         "error": type(error).__name__ + ": " + str(error)})
        finally:
            if client is not None:
                client.close()
    state["last_cycle_files"] = len(all_events)
    state["last_cycle_copied_bytes"] = sum(event["bytes"] for event in all_events
                                           if event["action"] == "copied_verified")
    save(STATE, state)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--daemon", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=90)
    parser.add_argument("--group", choices=remote.GROUPS, default="s1_03")
    args = parser.parse_args(argv)
    configure_group(args.group)
    if not 60 <= args.interval_seconds <= 120:
        raise ValueError("S1-03 archive interval must be 60–120 seconds")
    with single_lock(LOCAL / "archive.lock"):
        while True:
            result = archive_once()
            print(json.dumps({"checked_utc": result["checked_utc"],
                              "completed_jobs": len(result["jobs"]),
                              "files": result["last_cycle_files"],
                              "errors": result["last_errors"]}, ensure_ascii=False), flush=True)
            if args.once:
                return 0 if not result["last_errors"] else 1
            time.sleep(args.interval_seconds)


if __name__ == "__main__":
    sys.exit(main())
