"""Final S1-V10 SHA inventory, small receipt sync and verified per-lane ZIPs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil

from scripts.stage1_variable import ops
from scripts.stage1_variable.controller import STATE, save
from scripts.stage1_variable.data_controller import REGISTRY


LOCAL = ops.LOCAL / "delivery"


def file_sha(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def bundle_local_path(port, remote_path):
    if port not in ops.SSH_PORTS:
        raise ValueError("unknown S1-V10 archive lane")
    name = PurePosixPath(remote_path).name
    if not name.startswith("s1v10_final_") or not name.endswith(".zip"):
        raise ValueError("unexpected S1-V10 final ZIP name")
    return LOCAL / "bundles" / str(port) / name


def _state(commit):
    state = json.loads(STATE.read_text(encoding="utf-8"))
    if (state["scientific_commit"] != commit or
            state["phase"] != "completed_pending_report_archive" or
            set(state["tasks"]) != set(state["audits"])):
        raise RuntimeError("final archive requires all exact frozen jobs to be independently audited")
    return state


def _remote(port, code, timeout=300):
    response = ops.remote_python(code, port, timeout=timeout)
    if response["exit_code"]:
        raise RuntimeError(f"{port}: {response['stderr'][-1000:]}")
    return json.loads(response["stdout"].splitlines()[-1])


def inventory_lane(port, commit, state, registry):
    info = ops.instance(port)
    run_ids = [row["run_id"] for row in state["tasks"].values() if row["port"] == port]
    preparation = json.loads((ops.LOCAL / "controller" / "data_preparation.json").read_text())
    run_ids += [preparation["prepares"][str(port)]["run_id"]]
    dataset_ids = [record["id"] for record in registry["lanes"][str(port)].values()]
    code = f'''
import hashlib,json,pathlib
root=pathlib.Path({info['root']!r}).resolve();commit={commit!r}
run_ids={run_ids!r};dataset_ids={dataset_ids!r}
targets=[]
for rid in run_ids:
 targets.append(root/'outputs/workbench/jobs'/rid)
 targets.append(root/'models/runs/workbench'/rid)
for did in dataset_ids:targets.append(root/'outputs/workbench/scenes'/did)
targets.append(root/'outputs/workbench/source_snapshots'/commit)
prep_commit={state.get('preparation_source_commit', commit)!r}
if prep_commit!=commit:targets.append(root/'outputs/workbench/source_snapshots'/prep_commit)
for repair_commit in {sorted({row.get('repair_commit') for row in state.get('throughput_source_overrides', {}).values() if row.get('repair_commit')})!r}:
 if repair_commit!=commit:targets.append(root/'outputs/workbench/source_snapshots'/repair_commit)
targets.append(root/'ops/service.json')
files=[]
for target in targets:
 if not target.exists():continue
 for path in ([target] if target.is_file() else target.rglob('*')):
  if not path.is_file() or path.is_symlink() or '__pycache__' in path.parts or path.suffix=='.pyc':continue
  path=path.resolve()
  if not path.is_relative_to(root):raise RuntimeError('artifact escaped instance root')
  h=hashlib.sha256()
  with path.open('rb') as stream:
   for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
  files.append({{'relative':path.relative_to(root).as_posix(),
                'bytes':path.stat().st_size,'sha256':h.hexdigest()}})
files.sort(key=lambda x:x['relative'])
if len(files)!=len(set(x['relative'] for x in files)):
 raise RuntimeError('duplicate artifact inventory path')
print(json.dumps({{'port':{port},'root':str(root),'commit':commit,'files':files,
 'total_bytes':sum(row['bytes'] for row in files)}}))
'''
    result = _remote(port, code, timeout=1800)
    if not result["files"]:
        raise RuntimeError(f"{port}: empty final artifact inventory")
    target = LOCAL / f"inventory_{port}.json"
    save(target, result)
    return {"port": port, "manifest": str(target), "files": len(result["files"]),
            "total_bytes": result["total_bytes"],
            "manifest_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}


def inventory(commit):
    state = _state(commit)
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    result = {str(port): inventory_lane(port, commit, state, registry)
              for port in ops.SSH_PORTS}
    save(LOCAL / "inventory_summary.json", result)
    return result


def fetch_small(commit, threshold=2_000_000):
    _state(commit)
    result = {}
    for port in ops.SSH_PORTS:
        manifest = json.loads((LOCAL / f"inventory_{port}.json").read_text(encoding="utf-8"))
        if manifest["commit"] != commit or manifest["root"] != ops.instance(port)["root"]:
            raise RuntimeError("inventory source or target differs")
        selected = [row for row in manifest["files"] if row["bytes"] <= threshold and
                    PurePosixPath(row["relative"]).suffix.lower() in
                    (".json", ".jsonl", ".md", ".log", ".csv", ".txt")]
        client = ops.connect(port)
        copied = 0
        try:
            with client.open_sftp() as sftp:
                for row in selected:
                    relative = PurePosixPath(row["relative"])
                    if ".." in relative.parts or relative.is_absolute():
                        raise RuntimeError("inventory path traversal")
                    target = LOCAL / "small" / str(port) / Path(*relative.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists() or file_sha(target) != row["sha256"]:
                        sftp.get(manifest["root"] + "/" + row["relative"], str(target))
                    if file_sha(target) != row["sha256"]:
                        raise RuntimeError(f"{port}: downloaded metadata SHA differs")
                    copied += 1
        finally:
            client.close()
        result[str(port)] = {"files_verified": copied, "limit_bytes": threshold}
    save(LOCAL / "small_receipt.json", result)
    return result


def fetch_analysis(commit):
    """Sync each audited run's text ledgers once for local plots and reporting."""
    state = _state(commit)
    allowed = {".json", ".jsonl", ".csv", ".md", ".txt", ".log"}
    result = {"scientific_commit": commit, "tasks": {}, "files": []}
    for port in ops.SSH_PORTS:
        inventory = json.loads((LOCAL / f"inventory_{port}.json").read_text(encoding="utf-8"))
        rows = {row["relative"]: row for row in inventory["files"]}
        run_ids = {task["run_id"] for task in state["tasks"].values() if task["port"] == port}
        for task in state["tasks"].values():
            if task["port"] == port:
                result["tasks"][task["key"]] = {"port": port, "run_id": task["run_id"]}
        client = ops.connect(port)
        try:
            with client.open_sftp() as sftp:
                for run_id in run_ids:
                    prefix = f"outputs/workbench/jobs/{run_id}/"
                    selected = [row for relative, row in rows.items()
                                if relative.startswith(prefix) and
                                PurePosixPath(relative).suffix.lower() in allowed and
                                "scene_inputs/" not in relative and
                                "source_snapshots/" not in relative]
                    for row in selected:
                        relative = row["relative"][len(prefix):]
                        parts = PurePosixPath(relative).parts
                        if not parts or ".." in parts:
                            raise RuntimeError("unsafe analysis relative path")
                        target = ops.LOCAL / "synced_jobs" / str(port) / run_id / Path(*parts)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if not target.exists() or file_sha(target) != row["sha256"]:
                            sftp.get(inventory["root"] + "/" + row["relative"], str(target))
                        if file_sha(target) != row["sha256"]:
                            raise RuntimeError(f"{port} {run_id}: analysis file SHA differs")
                        result["files"].append({"port": port, "run_id": run_id,
                                                "path": str(target),
                                                "source_relative": row["relative"],
                                                "sha256": row["sha256"],
                                                "bytes": row["bytes"]})
        finally:
            client.close()
        save(ops.LOCAL / "synced_jobs" / "manifest.json", result)
    return {"tasks": len(result["tasks"]), "files": len(result["files"]),
            "manifest": str(ops.LOCAL / "synced_jobs" / "manifest.json")}


def fetch_completed_analysis(commit):
    """Incrementally sync only immutable audited/completed task text ledgers."""
    from scripts.stage1_variable import controller

    state = json.loads(controller.STATE.read_text(encoding="utf-8"))
    if state.get("scientific_commit") != commit:
        raise RuntimeError("partial sync controller source differs")
    selected = {}
    for key, audit in state.get("audits", {}).items():
        task = state.get("tasks", {}).get(key)
        if not task or audit.get("run_id") != task["run_id"]:
            raise RuntimeError("partial sync audit and controller task differ")
        port, run_id = task["port"], task["run_id"]
        response = ops.api(port, "/api/runs/" + run_id)
        if response["status"] != 200:
            raise RuntimeError(f"{key}: live task API unavailable")
        detail = response["body"]
        if (detail["status"].get("stage") != "completed" or
                detail["status"].get("process_alive") is True or
                detail["job"].get("git_commit") != controller.expected_task_source(state, key) or
                detail["job"].get("run_id") != run_id):
            raise RuntimeError(f"{key}: task is not immutable/completed at its mapped source")
        if task["kind"] == "evaluate" and (
                detail["artifacts"].get("raw_results_sha256") != audit.get("sha256")):
            raise RuntimeError(f"{key}: live raw result SHA differs from controller audit")
        selected[key] = {"port": port, "run_id": run_id,
                         "source_commit": detail["job"]["git_commit"],
                         "archive_root": detail["job"]["archive_root"]}
    allowed = {"job.json", "config.json", "status.json", "artifacts.json",
               "results.json", "results.partial.jsonl", "actions.jsonl",
               "episodes.jsonl", "updates.jsonl", "evaluations.jsonl"}
    manifest = {"scientific_commit": commit,
                "sync_kind": "audited_completed_tasks_only",
                "created_utc": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc).isoformat(),
                "tasks": {key: {"port": row["port"], "run_id": row["run_id"],
                                "source_commit": row["source_commit"]}
                          for key, row in selected.items()},
                "files": []}
    for port in ops.SSH_PORTS:
        tasks = {key: row for key, row in selected.items() if row["port"] == port}
        if not tasks:
            continue
        info = ops.instance(port)
        run_ids = [row["run_id"] for row in tasks.values()]
        code = f'''
import hashlib,json,pathlib
root=pathlib.Path({info['root']!r}).resolve();runs={run_ids!r};rows=[]
allowed={sorted(allowed)!r}
for run_id in runs:
 directory=root/'outputs/workbench/jobs'/run_id
 for file in sorted(directory.iterdir()):
  if not file.is_file() or not (file.name in allowed or
      file.name.startswith('validation_') and file.name.endswith('.jsonl')):continue
  h=hashlib.sha256()
  with file.open('rb') as stream:
   for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
  rows.append({{'run_id':run_id,'name':file.name,'path':str(file),
               'sha256':h.hexdigest(),'bytes':file.stat().st_size}})
print(json.dumps(rows))
'''
        response = ops.remote_python(code, port, timeout=300)
        if response["exit_code"]:
            raise RuntimeError(f"{port}: partial analysis SHA inventory failed: {response['stderr'][-700:]}")
        rows = json.loads(response["stdout"].splitlines()[-1])
        client = ops.connect(port)
        try:
            with client.open_sftp() as sftp:
                for row in rows:
                    target = ops.LOCAL / "synced_jobs" / str(port) / row["run_id"] / row["name"]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists() or file_sha(target) != row["sha256"]:
                        sftp.get(row["path"], str(target))
                    if file_sha(target) != row["sha256"]:
                        raise RuntimeError(f"{port} {row['run_id']}: partial synced file SHA differs")
                    manifest["files"].append({"port": port, "run_id": row["run_id"],
                                              "path": str(target),
                                              "source_relative": row["path"].removeprefix(info["root"] + "/"),
                                              "sha256": row["sha256"], "bytes": row["bytes"]})
        finally:
            client.close()
        save(ops.LOCAL / "synced_jobs" / "manifest.json", manifest)
    save(ops.LOCAL / "synced_jobs" / "manifest.json", manifest)
    receipt = {"tasks": len(manifest["tasks"]), "files": len(manifest["files"]),
               "bytes": sum(row["bytes"] for row in manifest["files"]),
               "manifest": str(ops.LOCAL / "synced_jobs" / "manifest.json"),
               "formal_final_archive": False}
    save(LOCAL / "completed_analysis_receipt.json", receipt)
    return receipt


def bundle_download(commit):
    _state(commit)
    result = {}
    for port in ops.SSH_PORTS:
        manifest = json.loads((LOCAL / f"inventory_{port}.json").read_text(encoding="utf-8"))
        if manifest["commit"] != commit:
            raise RuntimeError("bundle commit differs")
        info = ops.instance(port)
        remote_manifest = f"{info['ops']}/inventory_{commit}.json"
        sent = ops.upload(LOCAL / f"inventory_{port}.json", remote_manifest, port)
        if sent["local_sha256"] != sent["remote"]:
            raise RuntimeError("remote inventory SHA differs")
        code = f'''
import hashlib,json,pathlib,shutil,zipfile
root=pathlib.Path({info['root']!r}).resolve();manifest=json.loads(pathlib.Path({remote_manifest!r}).read_text())
target=root/'ops'/('s1v10_final_{commit}.zip')
tmp=target.with_suffix('.zip.part')
required=sum(int(row['bytes']) for row in manifest['files'])+2*(1<<30)
if shutil.disk_usage(root).free<required:
 raise RuntimeError('remote final ZIP lacks uncompressed-size plus 2GiB free-space reserve')
text_suffix={{'.json','.jsonl','.csv','.log','.md','.txt','.py','.html','.js','.css'}}
with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5,allowZip64=True) as bundle:
 for row in manifest['files']:
  file=(root/row['relative']).resolve()
  if not file.is_relative_to(root) or not file.is_file():raise RuntimeError('inventory member absent')
  h=hashlib.sha256()
  with file.open('rb') as stream:
   for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
  h=h.hexdigest()
  if h!=row['sha256']:raise RuntimeError('inventory member SHA changed')
  mode=zipfile.ZIP_DEFLATED if file.suffix.lower() in text_suffix else zipfile.ZIP_STORED
  bundle.write(file,row['relative'],compress_type=mode,compresslevel=5 if mode==zipfile.ZIP_DEFLATED else None)
 bundle.write({remote_manifest!r},'inventory.json')
tmp.replace(target)
h=hashlib.sha256()
with target.open('rb') as stream:
 for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
print(json.dumps({{'path':str(target),'bytes':target.stat().st_size,'sha256':h.hexdigest(),
                  'file_count':len(manifest['files'])}}))
'''
        remote = _remote(port, code, timeout=3600)
        local = bundle_local_path(port, remote["path"])
        local.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(local.parent).free < remote["bytes"] + 2 * (1 << 30):
            raise RuntimeError(f"{port}: local ZIP needs payload plus 2GiB free-space reserve")
        client = ops.connect(port)
        try:
            with client.open_sftp() as sftp:
                sftp.get(remote["path"], str(local))
        finally:
            client.close()
        if local.stat().st_size != remote["bytes"] or file_sha(local) != remote["sha256"]:
            raise RuntimeError(f"{port}: final ZIP SHA differs")
        result[str(port)] = {**remote, "local_path": str(local)}
        save(LOCAL / "bundle_receipt.json", result)
    return result


def source_inventory(commit):
    """Audit the four shared raw daily archives once, outside per-lane ZIPs."""
    from scripts.stage1_variable.data_controller import (
        PREPARATION_ARCHIVE_SHA, SOURCE_CONTRACT_SHA, SOURCE_PREFLIGHT_SHA)

    _state(commit)
    code = f'''
import hashlib,json,pathlib
base=pathlib.Path({ops.BASE!r})/'source';days={PREPARATION_ARCHIVE_SHA!r};rows={{}}
for day,expected in days.items():
 folder=base/day;pin=json.loads((folder/'source_preflight.json').read_text());status=json.loads((folder/'download_status.json').read_text())
archive=folder/pin['archive_name'];h=hashlib.sha256()
with archive.open('rb') as stream:
 for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
if status.get('status')!='completed' or status.get('sha256')!=expected or h.hexdigest()!=expected:
 raise RuntimeError('original daily archive SHA/complete receipt differs: '+day)
files=[]
for path in sorted(folder.iterdir()):
 if path.is_file() and path!=archive and path.suffix!='.part':
  files.append({{'path':str(path),'bytes':path.stat().st_size,
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}})
rows[day]={{'archive':str(archive),'bytes':archive.stat().st_size,'sha256':expected,
            'asset_id':pin['asset_id'],'metadata_files':files}}
print(json.dumps(rows))
'''
    observed = _remote(20418, code, timeout=600)
    for day, row in observed.items():
        mapped = {Path(item["path"]).name: item["sha256"] for item in row["metadata_files"]}
        if (row["sha256"] != PREPARATION_ARCHIVE_SHA[day] or
                mapped.get("source_preflight.json") != SOURCE_PREFLIGHT_SHA[day] or
                mapped.get("selection_contract.json") != SOURCE_CONTRACT_SHA[day]):
            raise RuntimeError(f"{day}: shared raw source and official selection SHA differ")
    result = {"experiment_id": "S1-V10", "scientific_commit": commit,
              "shared_source_root": ops.BASE + "/source", "days": observed,
              "raw_archives_count": 4,
              "raw_bytes_once": sum(row["bytes"] for row in observed.values()),
              "per_lane_bundles_include_raw_archives": False}
    save(LOCAL / "source_inventory.json", result)
    return result


def source_bundle_download(commit):
    """Create one ZIP64 raw-source package, never four duplicate daily sets."""
    inventory = source_inventory(commit)
    local_inventory = LOCAL / "source_inventory.json"
    remote_inventory = f"{ops.BASE}/source/source_inventory_{commit}.json"
    uploaded = ops.upload(local_inventory, remote_inventory, 20418)
    if uploaded["remote"] != file_sha(local_inventory):
        raise RuntimeError("raw source inventory transfer SHA differs")
    estimated = inventory["raw_bytes_once"]
    target_local = LOCAL / "bundles" / "source" / f"s1v10_raw_source_{commit}.zip"
    target_local.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(target_local.parent).free < estimated + 2 * (1 << 30):
        raise RuntimeError("local raw-source ZIP requires daily payload plus 2GiB reserve")
    code = f'''
import hashlib,json,pathlib,shutil,zipfile
base=pathlib.Path({ops.BASE!r})/'source';manifest=json.loads(pathlib.Path({remote_inventory!r}).read_text())
required=manifest['raw_bytes_once']+2*(1<<30)
if shutil.disk_usage(base).free<required:raise RuntimeError('remote raw-source ZIP lacks free-space reserve')
target=base/('s1v10_raw_source_{commit}.zip');tmp=target.with_suffix('.zip.part')
with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5,allowZip64=True) as bundle:
 for day,row in sorted(manifest['days'].items()):
  raw=pathlib.Path(row['archive']);h=hashlib.sha256()
  with raw.open('rb') as stream:
   for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
  if h.hexdigest()!=row['sha256']:raise RuntimeError('raw archive changed before ZIP: '+day)
  bundle.write(raw,day+'/'+raw.name,compress_type=zipfile.ZIP_STORED)
  for item in row['metadata_files']:
   path=pathlib.Path(item['path'])
   if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
    raise RuntimeError('source metadata changed before ZIP')
   bundle.write(path,day+'/'+path.name,compress_type=zipfile.ZIP_DEFLATED,compresslevel=5)
 bundle.write({remote_inventory!r},'source_inventory.json',compress_type=zipfile.ZIP_DEFLATED,compresslevel=5)
tmp.replace(target)
h=hashlib.sha256()
with target.open('rb') as stream:
 for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
print(json.dumps({{'path':str(target),'bytes':target.stat().st_size,'sha256':h.hexdigest()}}))
'''
    remote = _remote(20418, code, timeout=3600)
    client = ops.connect(20418)
    try:
        with client.open_sftp() as sftp:
            sftp.get(remote["path"], str(target_local))
    finally:
        client.close()
    if target_local.stat().st_size != remote["bytes"] or file_sha(target_local) != remote["sha256"]:
        raise RuntimeError("single raw-source ZIP local SHA differs")
    receipt = {**remote, "local_path": str(target_local),
               "source_inventory_sha256": file_sha(local_inventory),
               "daily_archives_copied_once": 4}
    save(LOCAL / "source_bundle_receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("inventory", "fetch-small", "fetch-analysis",
                                           "fetch-completed-analysis", "bundle-download",
                                           "source-inventory", "source-bundle-download"))
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    result = (inventory(args.commit) if args.action == "inventory" else
              fetch_small(args.commit) if args.action == "fetch-small" else
              fetch_analysis(args.commit) if args.action == "fetch-analysis" else
              fetch_completed_analysis(args.commit) if args.action == "fetch-completed-analysis" else
              bundle_download(args.commit) if args.action == "bundle-download" else
              source_inventory(args.commit) if args.action == "source-inventory" else
              source_bundle_download(args.commit))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
