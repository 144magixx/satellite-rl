"""Final S1-V10 SHA inventory, small receipt sync and verified per-lane ZIPs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

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
import hashlib,json,pathlib,zipfile
root=pathlib.Path({info['root']!r}).resolve();manifest=json.loads(pathlib.Path({remote_manifest!r}).read_text())
target=root/'ops'/('s1v10_final_{commit}.zip')
tmp=target.with_suffix('.zip.part')
with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as bundle:
 for row in manifest['files']:
  file=(root/row['relative']).resolve()
  if not file.is_relative_to(root) or not file.is_file():raise RuntimeError('inventory member absent')
  h=hashlib.sha256()
  with file.open('rb') as stream:
   for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
  h=h.hexdigest()
  if h!=row['sha256']:raise RuntimeError('inventory member SHA changed')
  bundle.write(file,row['relative'])
 bundle.write({remote_manifest!r},'inventory.json')
tmp.replace(target)
h=hashlib.sha256()
with target.open('rb') as stream:
 for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
print(json.dumps({{'path':str(target),'bytes':target.stat().st_size,'sha256':h.hexdigest(),
                  'file_count':len(manifest['files'])}}))
'''
        remote = _remote(port, code, timeout=3600)
        local = LOCAL / "bundles" / Path(remote["path"]).name
        local.parent.mkdir(parents=True, exist_ok=True)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("inventory", "fetch-small", "fetch-analysis", "bundle-download"))
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    result = (inventory(args.commit) if args.action == "inventory" else
              fetch_small(args.commit) if args.action == "fetch-small" else
              fetch_analysis(args.commit) if args.action == "fetch-analysis" else
              bundle_download(args.commit))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
