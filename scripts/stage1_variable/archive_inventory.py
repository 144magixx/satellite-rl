"""Final S1-V10 SHA inventory, small receipt sync and verified per-lane ZIPs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

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
    """Archive all audited jobs as verified ZIP members for report streaming."""
    _state(commit)
    from scripts.stage1_variable.archive_jobs import archive_all_completed

    return archive_all_completed()


def fetch_completed_analysis(commit):
    """Archive only currently audited, completed and exited jobs."""
    state = json.loads(STATE.read_text(encoding="utf-8"))
    if state.get("scientific_commit") != commit:
        raise RuntimeError("partial archive controller source differs")
    from scripts.stage1_variable.archive_jobs import archive_all_completed

    return archive_all_completed()


def bundle_download(commit):
    """Prevent a second lane-wide copy of the same large job evidence."""
    _state(commit)
    raise RuntimeError("S1-V10 job ZIP64 packages are the final local archive; use archive_jobs")


def source_inventory(commit):
    """Audit the four shared raw daily archives once, outside per-lane ZIPs."""
    from scripts.stage1_variable.data_controller import (
        PREPARATION_ARCHIVE_SHA, SOURCE_CONTRACT_SHA, SOURCE_PREFLIGHT_SHA)

    state = json.loads(STATE.read_text(encoding="utf-8"))
    if (state.get("scientific_commit") != commit or
            set(state.get("prepares", {})) != {str(port) for port in ops.SSH_PORTS}):
        raise RuntimeError("shared S1-V10 source inventory lacks four frozen prepare jobs")
    for port in ops.SSH_PORTS:
        prepared = state["prepares"][str(port)]
        response = ops.api(port, "/api/runs/" + prepared["run_id"])
        if response["status"] != 200:
            raise RuntimeError("shared S1-V10 preparation API unavailable")
        job, status = response["body"]["job"], response["body"]["status"]
        if (job.get("run_id") != prepared["run_id"] or
                job.get("request_id") != prepared["request_id"] or
                job.get("git_commit") != prepared["git_commit"] or
                status.get("stage") != "completed" or
                status.get("process_alive") is not False):
            raise RuntimeError("shared S1-V10 source needs four completed exited preparations")
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
    if set(observed) != set(PREPARATION_ARCHIVE_SHA):
        raise RuntimeError("shared S1-V10 source inventory omits one of four pinned days")
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


def source_bundle_target(commit, remote):
    """Choose one verified D/C location from actual compressed ZIP bytes."""
    from scripts.stage1_variable import archive_jobs

    candidates = (LOCAL / "bundles" / "source",
                  archive_jobs.BACKUP_LOCAL / "_shared_source")
    for directory in candidates:
        directory.mkdir(parents=True, exist_ok=True)
    free = {str(path): shutil.disk_usage(path).free for path in candidates}
    if any(value < archive_jobs.LOCAL_RESERVE_BYTES for value in free.values()):
        raise RuntimeError("raw-source archive requires D and C each to retain 8 GiB")
    name = f"s1v10_raw_source_{commit}.zip"
    target = next((directory / name for directory in candidates
                   if (directory / name).is_file() and
                   (directory / name).stat().st_size == remote["bytes"] and
                   file_sha(directory / name) == remote["sha256"]), None)
    missing = 0 if target is not None else remote["bytes"]
    if target is None:
        target = next((directory / name for directory in candidates
                       if free[str(directory)] >= missing +
                       archive_jobs.LOCAL_RESERVE_BYTES), None)
    if target is None:
        raise RuntimeError("actual raw-source ZIP bytes do not fit D/C with 8 GiB reserves; "
                           "remote receipt remains at " + remote["path"])
    return target, {"free_bytes_before": free,
                    "missing_compressed_bytes": missing,
                    "reserved_bytes_each_drive": archive_jobs.LOCAL_RESERVE_BYTES}


def verify_shared_source_zip(path, inventory, inventory_path, expected_zip_sha):
    """Verify the sole raw-source ZIP and every uncompressed daily member."""
    from scripts.s2p19.archive_sources import safe_member

    if file_sha(path) != expected_zip_sha:
        raise RuntimeError("S1-V10 shared raw-source ZIP SHA differs")
    expected = {"source_inventory.json": {
        "sha256": file_sha(inventory_path), "bytes": Path(inventory_path).stat().st_size}}
    for day, row in inventory["days"].items():
        archive = Path(row["archive"])
        expected[safe_member(day + "/" + archive.name)] = {
            "sha256": row["sha256"], "bytes": row["bytes"]}
        for item in row["metadata_files"]:
            member = safe_member(day + "/" + Path(item["path"]).name)
            if member in expected:
                raise RuntimeError("S1-V10 shared source ZIP repeats a pinned member")
            expected[member] = {"sha256": item["sha256"], "bytes": item["bytes"]}
    with zipfile.ZipFile(path) as bundle:
        infos = bundle.infolist()
        if len(infos) != len(expected) or {info.filename for info in infos} != set(expected):
            raise RuntimeError("S1-V10 shared source ZIP member set differs")
        for info in infos:
            safe_member(info.filename)
            row = expected[info.filename]
            if info.is_dir() or info.file_size != row["bytes"]:
                raise RuntimeError("S1-V10 shared source ZIP member size differs")
            value = hashlib.sha256()
            with bundle.open(info) as stream:
                for block in iter(lambda: stream.read(8 << 20), b""):
                    value.update(block)
            if value.hexdigest() != row["sha256"]:
                raise RuntimeError("S1-V10 shared source ZIP member SHA differs")
    return [{"member": member, **record} for member, record in sorted(expected.items())]


def source_bundle_download(commit):
    """Create one ZIP64 raw-source package, never four duplicate daily sets."""
    inventory = source_inventory(commit)
    local_inventory = LOCAL / "source_inventory.json"
    remote_inventory = f"{ops.BASE}/source/source_inventory_{commit}.json"
    uploaded = ops.upload(local_inventory, remote_inventory, 20418)
    if uploaded["remote"] != file_sha(local_inventory):
        raise RuntimeError("raw source inventory transfer SHA differs")
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
    target_local, gate = source_bundle_target(commit, remote)
    client = ops.connect(20418)
    try:
        with client.open_sftp() as sftp:
            if gate["missing_compressed_bytes"]:
                pending = target_local.with_name(target_local.name + ".part")
                sftp.get(remote["path"], str(pending))
                if pending.stat().st_size != remote["bytes"] or file_sha(pending) != remote["sha256"]:
                    raise RuntimeError("single raw-source ZIP staging SHA differs")
                pending.replace(target_local)
    finally:
        client.close()
    if target_local.stat().st_size != remote["bytes"] or file_sha(target_local) != remote["sha256"]:
        raise RuntimeError("single raw-source ZIP local SHA differs")
    members = verify_shared_source_zip(target_local, inventory, local_inventory, remote["sha256"])
    receipt = {**remote, "schema": "s1v10.shared_source_bundle.v1",
               "scientific_commit": commit,
               "local_path": str(target_local),
               "local_space_gate": gate,
               "source_inventory_sha256": file_sha(local_inventory),
               "all_member_shas_verified": True, "members": members,
               "daily_archives_copied_once": 4}
    save(LOCAL / "source_bundle_receipt.json", receipt)
    return receipt


def relocate_shared_source_to_backup(commit):
    """Keep one verified source ZIP on C when it preserves D for large jobs."""
    from scripts.stage1_variable import archive_jobs

    receipt_path = LOCAL / "source_bundle_receipt.json"
    if not receipt_path.is_file():
        raise RuntimeError("verified shared source receipt absent")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    inventory_path = LOCAL / "source_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8")) if inventory_path.is_file() else {}
    name = f"s1v10_raw_source_{commit}.zip"
    original = (LOCAL / "bundles" / "source" / name).resolve()
    backup = (archive_jobs.BACKUP_LOCAL / "_shared_source" / name).resolve()
    if ((archive_jobs.LOCAL / "shared_source_closure_receipt.json").is_file() or
            receipt.get("schema") != "s1v10.shared_source_bundle.v1" or
            receipt.get("all_member_shas_verified") is not True or
            receipt.get("scientific_commit", commit) != commit or
            inventory.get("scientific_commit") != commit or
            receipt.get("source_inventory_sha256") != file_sha(inventory_path) or
            receipt.get("bytes", 0) <= 0 or
            Path(receipt.get("local_path") or "").resolve() not in (original, backup)):
        raise RuntimeError("shared source ZIP is already closed or has another identity")
    expected = receipt["sha256"]
    if (Path(receipt["local_path"]).resolve() == backup and backup.is_file() and
            file_sha(backup) == expected and not original.exists()):
        return receipt
    if not original.is_file() or original.is_symlink() or file_sha(original) != expected:
        raise RuntimeError("original D shared source ZIP SHA/path differs")
    backup.parent.mkdir(parents=True, exist_ok=True)
    source_free = shutil.disk_usage(original.parent).free
    backup_free = shutil.disk_usage(backup.parent).free
    if (source_free < archive_jobs.LOCAL_RESERVE_BYTES or
            backup_free < receipt["bytes"] + archive_jobs.LOCAL_RESERVE_BYTES):
        raise RuntimeError("C cannot hold shared source ZIP while retaining both 8 GiB reserves")
    if not backup.is_file() or file_sha(backup) != expected:
        pending = backup.with_name(backup.name + ".part")
        shutil.copy2(original, pending)
        if pending.stat().st_size != receipt["bytes"] or file_sha(pending) != expected:
            raise RuntimeError("shared source ZIP copy to C failed SHA/size verification")
        pending.replace(backup)
    if file_sha(backup) != expected or backup.stat().st_size != receipt["bytes"]:
        raise RuntimeError("verified C shared source ZIP differs")
    migration = {"schema": "s1v10.shared_source_relocation.v1",
                 "old_path": str(original), "new_path": str(backup),
                 "sha256": expected, "bytes": receipt["bytes"],
                 "D_free_before": source_free, "C_free_before": backup_free,
                 "verified_copy_utc": __import__("datetime").datetime.now(
                     __import__("datetime").timezone.utc).isoformat(),
                 "old_removed": False}
    receipt = {**receipt, "scientific_commit": commit,
               "local_path": str(backup), "relocation": migration}
    save(receipt_path, receipt)
    if (original.resolve() != (LOCAL / "bundles" / "source" / name).resolve() or
            file_sha(original) != expected):
        raise RuntimeError("D source ZIP changed before duplicate removal")
    original.unlink()
    migration.update({"old_removed": True,
                      "D_free_after": shutil.disk_usage(original.parent).free,
                      "C_free_after": shutil.disk_usage(backup.parent).free,
                      "completed_utc": __import__("datetime").datetime.now(
                          __import__("datetime").timezone.utc).isoformat()})
    receipt["relocation"] = migration
    save(receipt_path, receipt)
    save(LOCAL / "source_relocation_receipt.json", migration)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("inventory", "fetch-small", "fetch-analysis",
                                           "fetch-completed-analysis", "bundle-download",
                                           "source-inventory", "source-bundle-download",
                                           "source-relocate-backup"))
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    result = (inventory(args.commit) if args.action == "inventory" else
              fetch_small(args.commit) if args.action == "fetch-small" else
              fetch_analysis(args.commit) if args.action == "fetch-analysis" else
              fetch_completed_analysis(args.commit) if args.action == "fetch-completed-analysis" else
              bundle_download(args.commit) if args.action == "bundle-download" else
              source_inventory(args.commit) if args.action == "source-inventory" else
              source_bundle_download(args.commit) if args.action == "source-bundle-download" else
              relocate_shared_source_to_backup(args.commit))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
