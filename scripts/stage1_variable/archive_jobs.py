"""Completed S1-V10 jobs as SHA-verified ZIP64 volumes, without raw mirrors.

Only immutable, controller-audited workbench jobs may enter this path.  The
downloaded ZIPs are the local archive and the report's authoritative inputs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import zipfile

from scripts.s2p19 import archive_sources
from scripts.stage1_variable import controller, ops
from satellite_rl.workbench.variable_throughput_recovery import ORIGIN_COMMIT


LOCAL = ops.LOCAL / "job_archives"
BACKUP_LOCAL = Path("C:/paper_archives/outputs/experiments/first_stage_s1_v10_20261009/job_archives")
MODEL_EXPORT_ROOT = Path("D:/paper/models/runs/first_stage_s1_v10_20261009")
ANALYSIS_MANIFEST = ops.LOCAL / "synced_jobs" / "manifest.json"
MAX_FILES = 40000
MAX_VOLUMES = 1000
MAX_INPUT_GIB = 256
PART_MIB = 256
LOCAL_RESERVE_BYTES = 8 * (1 << 30)
SMALL_MIRROR_MEMBERS = {
    "job/job.json", "job/config.json", "job/status.json",
    "job/artifacts.json", "job/results.json", "source/source_manifest.json",
}


REMOTE_BUILD = r'''
import datetime,hashlib,json,os,pathlib,shutil,stat,sys,zipfile
request=json.loads(REQUEST_JSON)
root=pathlib.Path(request['archive_root']).resolve();run_id=request['run_id']
jobdir=root/'outputs/workbench/jobs'/run_id
job=json.loads((jobdir/'job.json').read_text(encoding='utf-8'))
status=json.loads((jobdir/'status.json').read_text(encoding='utf-8'))
launch=json.loads((jobdir/'launch.json').read_text(encoding='utf-8'))
if (job.get('run_id')!=run_id or job.get('request_id')!=request['request_id'] or
    job.get('experiment_id')!='S1-V10' or job.get('archive_root')!=str(root) or
    job.get('git_commit')!=request['source_commit'] or
    status.get('run_id')!=run_id or status.get('stage')!='completed'):
 raise RuntimeError('S1-V10 job identity, source or completed budget differs')
if request['expected_step'] is not None and status.get('env_step')!=request['expected_step']:
 raise RuntimeError('S1-V10 completed training budget differs')
pid=launch.get('pid');p=pathlib.Path('/proc')/str(pid)
if os.name=='posix' and p.exists() and isinstance(launch.get('process_start_time'),(int,float)):
 raw=(p/'stat').read_text();fields=raw[raw.rfind(')')+2:].split();ticks=int(fields[19])
 boot=next(int(x.split()[1]) for x in pathlib.Path('/proc/stat').read_text().splitlines() if x.startswith('btime '))
 if abs(boot+ticks/os.sysconf('SC_CLK_TCK')-launch['process_start_time'])<.02:
  raise RuntimeError('completed job worker of same birth still runs')

def digest(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
 return h.hexdigest()
def safe(path):
 path=pathlib.Path(path)
 if path.is_symlink():raise RuntimeError('symlink refused: '+str(path))
 resolved=path.resolve(strict=True)
 if not resolved.is_relative_to(root):raise RuntimeError('source escaped S1-V10 lane: '+str(path))
 return resolved
items={}
def add(member,path,expected=None):
 if (not member or '\\' in member or member.startswith('/') or
     any(x in ('','.','..') for x in member.split('/')) or member in items):
  raise RuntimeError('unsafe/duplicate S1-V10 member: '+member)
 path=safe(path)
 if not path.is_file():raise RuntimeError('archive member absent: '+str(path))
 size=path.stat().st_size
 if size>request['max_bytes'] or len(items)>=request['max_files']:
  raise RuntimeError('S1-V10 member size/count exceeds bound')
 actual=digest(path)
 if expected is not None and actual!=expected:
  raise RuntimeError('frozen S1-V10 member SHA differs: '+member)
 items[member]={'member':member,'remote_path':str(path),'bytes':size,'sha256':actual}
def tree(prefix,directory,skip=None):
 directory=safe(directory)
 if not directory.is_dir():raise RuntimeError('archive directory absent: '+str(directory))
 for file in sorted(directory.rglob('*')):
  if skip is not None and file==skip:continue
  if file.is_symlink():raise RuntimeError('symlink below archive tree')
  if file.is_file() and '__pycache__' not in file.parts and file.suffix!='.pyc':
   add(prefix+'/'+file.relative_to(directory).as_posix(),file)

external_dependency=None;skip_raw=None
if job.get('kind')=='prepare':
 preparation=job.get('variable_preparation') or {}
 archive=preparation.get('archive') or {}
 preflight=preparation.get('source_preflight') or {}
 day=preparation.get('date_utc')
 if (not isinstance(day,str) or pathlib.PurePosixPath(day).name!=day or
     len(day)!=10 or not isinstance(archive.get('sha256'),str) or
     not isinstance(archive.get('bytes'),int)):
  raise RuntimeError('preparation shared raw provenance incomplete')
 shared=pathlib.Path(request['shared_source_root']).resolve()
 pin=shared/day/'source_preflight.json'
 if (str(pin)!=preflight.get('source_path') or not pin.is_file() or
     digest(pin)!=preflight.get('sha256')):
  raise RuntimeError('preparation raw source preflight SHA/path differs')
 pin_record=json.loads(pin.read_text(encoding='utf-8'))
 archive_name=pin_record.get('archive_name')
 if (not isinstance(archive_name,str) or not archive_name or
     pathlib.PurePosixPath(archive_name).name!=archive_name or
     archive_name in ('.','..') or '\\' in archive_name):
  raise RuntimeError('preparation raw pinned archive name unsafe')
 shared_archive=shared/day/archive_name
 if (str(shared_archive)!=archive.get('path') or
     not shared_archive.is_file() or shared_archive.is_symlink() or
     not stat.S_ISREG(shared_archive.lstat().st_mode)):
  raise RuntimeError('preparation raw shared target differs from pinned daily archive')
 skip_raw=jobdir/'preparation/raw'/archive_name
 if not os.path.lexists(skip_raw):raise RuntimeError('preparation raw job member absent')
 raw_stat=skip_raw.lstat();source_stat=shared_archive.lstat()
 if stat.S_ISLNK(raw_stat.st_mode):
  if skip_raw.resolve(strict=True)!=shared_archive.resolve(strict=True):
   raise RuntimeError('preparation raw symlink points outside exact pinned source')
  link_kind='exact_source_symlink'
 elif stat.S_ISREG(raw_stat.st_mode):
  if (raw_stat.st_dev,raw_stat.st_ino)!=(source_stat.st_dev,source_stat.st_ino):
   raise RuntimeError('preparation raw regular file is not the pinned hardlink')
  link_kind='exact_source_hardlink'
 else:raise RuntimeError('preparation raw member has unsupported file type')
 if (skip_raw.stat().st_size!=archive['bytes'] or
     source_stat.st_size!=archive['bytes'] or
     digest(skip_raw)!=archive['sha256'] or
     digest(shared_archive)!=archive['sha256']):
  raise RuntimeError('preparation raw source SHA/bytes differ from frozen job')
 external_dependency={'kind':'shared_daily_aircraft_archive_v1',
  'job_member':'job/preparation/raw/'+archive_name,
  'source_bundle_member':day+'/'+archive_name,
  'source_path':str(shared_archive),'sha256':archive['sha256'],
  'bytes':archive['bytes'],'source_preflight_sha256':preflight['sha256'],
  'link_kind':link_kind}
tree('job',jobdir,skip=skip_raw)
config=json.loads((jobdir/'config.json').read_text(encoding='utf-8'))
for role,record in (config.get('datasets') or {}).items():
 if role not in ('train','development','validation','test','sequence'):
  raise RuntimeError('unknown frozen dataset role')
 file=safe(record['path'])
 if digest(file)!=record['sha256']:raise RuntimeError('frozen aircraft dataset SHA differs')
 if not file.is_relative_to(jobdir):add('datasets/'+role+'/'+file.name,file,record['sha256'])
source=safe(job['source_root'])
source_manifest=json.loads((source/'source_manifest.json').read_text(encoding='utf-8'))
if source_manifest.get('git_commit')!=job['git_commit'] or not source_manifest.get('files'):
 raise RuntimeError('frozen source manifest/commit differs')
for relative,expected in source_manifest['files'].items():
 file=safe(source/relative)
 if not file.is_relative_to(source) or digest(file)!=expected:
  raise RuntimeError('frozen source member SHA differs: '+relative)
tree('source',source)
model=job.get('model_dir')
if model and pathlib.Path(model).is_dir():tree('weights/model_dir',model)
model_path=job.get('model_path')
if model_path and pathlib.Path(model_path).is_dir():tree('weights/model_input',model_path)
elif model_path and pathlib.Path(model_path).is_file():add('weights/model_input/'+pathlib.Path(model_path).name,model_path)
for index,reference in enumerate(job.get('model_files') or []):
 add('weights/model_refs/'+str(index).zfill(3)+'_'+pathlib.Path(reference['path']).name,
     reference['path'],reference['sha256'])
if request['raw_result_sha256']:
 result=items.get('job/results.partial.jsonl')
 if result is None or result['sha256']!=request['raw_result_sha256']:
  raise RuntimeError('controller-audited raw evaluation SHA differs')
total=sum(x['bytes'] for x in items.values())
if not items or total>request['max_bytes']:raise RuntimeError('S1-V10 job input bound exceeded')
ordered=[items[name] for name in sorted(items)]
inventory_rows=[(x['member'],x['bytes'],x['sha256']) for x in ordered]
inventory_payload=inventory_rows if external_dependency is None else {
 'members':inventory_rows,'external_dependencies':[external_dependency]}
inventory=hashlib.sha256(json.dumps(inventory_payload,separators=(',',':')).encode()).hexdigest()
groups=[];current=[];amount=0
for item in ordered:
 if current and amount+item['bytes']>request['part_bytes']:
  groups.append(current);current=[];amount=0
 current.append(item);amount+=item['bytes']
if current:groups.append(current)
if len(groups)>request['max_volumes']:raise RuntimeError('too many S1-V10 archive volumes')
package_id=inventory[:20]+'_'+str(request['part_bytes'])+'_c'+str(request['compression_level'])
destination=root/'ops/archive_s1v10/jobs'/run_id/package_id
receipt_path=destination/'receipt.json'
if destination.exists():
 if not receipt_path.is_file():raise RuntimeError('incomplete S1-V10 package preserved; inspect before retry')
 receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
 if (receipt.get('inventory_sha256')!=inventory or receipt.get('run_id')!=run_id or
     receipt.get('source_commit')!=job['git_commit'] or
     receipt.get('external_dependencies')!=([] if external_dependency is None else [external_dependency])):
  raise RuntimeError('existing S1-V10 package inventory differs')
 for volume in receipt['volumes']:
  if digest(destination/volume['name'])!=volume['sha256']:
   raise RuntimeError('existing S1-V10 ZIP SHA differs')
 print(json.dumps({'receipt_path':str(receipt_path),'receipt_sha256':digest(receipt_path),
                   'reused_verified':True,'volume_count':len(receipt['volumes'])}))
 sys.exit(0)
destination.mkdir(parents=True,exist_ok=False)
volumes=[]
for index,group in enumerate(groups,1):
 free=shutil.disk_usage(destination).free
 if free<8*(1<<30)+min(sum(x['bytes'] for x in group),2*(1<<30)):
  raise RuntimeError('remote S1-V10 ZIP lacks safe working space')
 name='part_'+str(index).zfill(4)+'.zip';path=destination/name
 with zipfile.ZipFile(str(path)+'.part','w',compression=zipfile.ZIP_DEFLATED,
                      compresslevel=request['compression_level'],allowZip64=True) as output:
  for item in group:
   suffix=pathlib.Path(item['member']).suffix.lower()
   mode=zipfile.ZIP_STORED if suffix in ('.npz','.pt','.gz','.zip') else zipfile.ZIP_DEFLATED
   output.write(item['remote_path'],item['member'],compress_type=mode,
                compresslevel=None if mode==zipfile.ZIP_STORED else request['compression_level'])
 pathlib.Path(str(path)+'.part').replace(path)
 volumes.append({'name':name,'remote_path':str(path),'bytes':path.stat().st_size,
                 'sha256':digest(path),'member_count':len(group),
                 'input_bytes':sum(x['bytes'] for x in group)})
receipt={'schema':'s1v10.job_archive.v1',
         'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'experiment_id':'S1-V10','task_key':request['task_key'],'port':request['port'],
         'run_id':run_id,'request_id':job['request_id'],'job_kind':job['kind'],
         'job_stage':'completed','source_commit':job['git_commit'],
         'source_manifest_sha256':digest(source/'source_manifest.json'),
         'controller_audit_sha256':request['controller_audit_sha256'],
         'external_dependencies':[] if external_dependency is None else [external_dependency],
         'inventory_sha256':inventory,'compression_level':request['compression_level'],
         'total_input_bytes':total,'member_count':len(ordered),'members':ordered,
         'volumes':volumes,'remote_package_directory':str(destination)}
with (destination/'receipt.json.part').open('w',encoding='utf-8') as stream:
 json.dump(receipt,stream,ensure_ascii=False,indent=2)
(destination/'receipt.json.part').replace(receipt_path)
print(json.dumps({'receipt_path':str(receipt_path),'receipt_sha256':digest(receipt_path),
                  'reused_verified':False,'volume_count':len(volumes)}))
'''


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


@contextmanager
def _lock(path):
    """Use an independent S1-V10 advisory lock for each job or index."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _task_identity(key):
    state = json.loads(controller.STATE.read_text(encoding="utf-8"))
    if state.get("scientific_commit") != ORIGIN_COMMIT:
        raise RuntimeError("S1-V10 controller source differs")
    if key.startswith("prepare_"):
        port_name = key.removeprefix("prepare_")
        prepared = state.get("prepares", {}).get(port_name)
        if (port_name not in {str(port) for port in ops.SSH_PORTS} or
                not prepared or prepared.get("port") != int(port_name) or
                prepared.get("git_commit") != state.get("preparation_source_commit")):
            raise ValueError("S1-V10 preparation archive identity differs")
        task = {"port": int(port_name), "run_id": prepared["run_id"],
                "request_id": prepared["request_id"], "kind": "prepare"}
        audit = {field: prepared[field] for field in
                 ("run_id", "request_id", "role", "date", "git_commit")}
        source_commit = state["preparation_source_commit"]
    else:
        task = state.get("tasks", {}).get(key)
        audit = state.get("audits", {}).get(key)
        if not task or not audit or audit.get("run_id") != task.get("run_id"):
            raise ValueError("S1-V10 archive requires an audited controller task")
        source_commit = controller.expected_task_source(state, key)
    port, run_id = task["port"], task["run_id"]
    response = ops.api(port, "/api/runs/" + run_id)
    if response["status"] != 200:
        raise RuntimeError("S1-V10 completed task API unavailable")
    detail = response["body"]
    job, status = detail["job"], detail["status"]
    if (job.get("run_id") != run_id or job.get("request_id") != task["request_id"] or
            job.get("experiment_id") != "S1-V10" or
            job.get("kind") != task["kind"] or
            job.get("archive_root") != ops.instance(port)["root"] or
            job.get("git_commit") != source_commit or
            status.get("stage") != "completed" or status.get("process_alive") is not False):
        raise RuntimeError("S1-V10 archive requires completed, exited exact-source worker")
    if task["kind"] == "train" and status.get("env_step") != task["budget"]:
        raise RuntimeError("S1-V10 archived training budget differs")
    if task["kind"] == "evaluate" and detail["artifacts"].get("raw_results_sha256") != audit.get("sha256"):
        raise RuntimeError("S1-V10 controller-audited raw result SHA differs")
    audit_sha = hashlib.sha256(json.dumps(audit, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    return state, task, audit, detail, audit_sha, source_commit


def _verify_volume(path, expected, expected_sha):
    if digest(path) != expected_sha:
        raise ValueError("S1-V10 downloaded ZIP SHA differs")
    by_member = {item["member"]: item for item in expected}
    if len(by_member) != len(expected):
        raise ValueError("S1-V10 duplicate declared member")
    with zipfile.ZipFile(path) as bundle:
        infos = bundle.infolist()
        if len(infos) != len(expected) or {item.filename for item in infos} != set(by_member):
            raise ValueError("S1-V10 ZIP member list differs")
        for info in infos:
            archive_sources.safe_member(info.filename)
            item = by_member[info.filename]
            if info.is_dir() or info.file_size != item["bytes"]:
                raise ValueError("S1-V10 ZIP member size differs")
            h = hashlib.sha256()
            with bundle.open(info) as stream:
                for block in iter(lambda: stream.read(8 << 20), b""):
                    h.update(block)
            if h.hexdigest() != item["sha256"]:
                raise ValueError("S1-V10 ZIP member SHA differs")


def _require_local_capacity(directory, missing_bytes, small_mirror_bytes=0):
    """Use compressed volume sizes, leaving eight GiB after their download."""
    free = shutil.disk_usage(directory).free
    if free < missing_bytes + small_mirror_bytes + LOCAL_RESERVE_BYTES:
        raise RuntimeError("D: lacks actual compressed S1-V10 ZIP bytes plus 8 GiB reserve")
    return {"free_bytes": free, "missing_compressed_bytes": missing_bytes,
            "planned_small_mirror_bytes": small_mirror_bytes,
            "reserved_bytes": LOCAL_RESERVE_BYTES}


def _select_volume_locations(receipt, port, run_id, package_id, receipt_bytes):
    """Place each actual compressed volume on D or C with both reserves intact."""
    roots = (LOCAL, BACKUP_LOCAL)
    for root in roots:
        root.mkdir(parents=True, exist_ok=True)
    free = {root: shutil.disk_usage(root).free for root in roots}
    if any(value < LOCAL_RESERVE_BYTES for value in free.values()):
        raise RuntimeError("S1-V10 D and C archive drives must each retain 8 GiB")
    small_bytes = sum(item["bytes"] for item in receipt["members"]
                      if item["member"] in SMALL_MIRROR_MEMBERS and item["bytes"] <= 2_000_000)
    # Receipt and small mirrors live together. The extra MiB covers the
    # download receipt and atomic .part files without consuming the reserve.
    metadata_bytes = receipt_bytes + small_bytes + (1 << 20)
    metadata_root = next((root for root in roots
                          if free[root] >= LOCAL_RESERVE_BYTES + metadata_bytes), None)
    if metadata_root is None:
        raise RuntimeError("S1-V10 receipt/small mirrors lack 8 GiB free-space reserve")
    available = {root: free[root] - LOCAL_RESERVE_BYTES -
                 (metadata_bytes if root == metadata_root else 0) for root in roots}
    paths = {}
    for volume in sorted(receipt["volumes"], key=lambda row: row["bytes"], reverse=True):
        name = archive_sources.safe_member(volume["name"])
        existing = []
        for root in roots:
            path = root / str(port) / run_id / package_id / name
            if (path.is_file() and path.stat().st_size == volume["bytes"] and
                    digest(path) == volume["sha256"]):
                existing.append(path)
        if existing:
            paths[volume["name"]] = existing[0]
            continue
        fitting = [root for root in roots if available[root] >= volume["bytes"]]
        selected = min(fitting, key=lambda root: available[root] - volume["bytes"]) if fitting else None
        if selected is None:
            raise RuntimeError("S1-V10 actual ZIP volume exceeds D/C capacity plus 8 GiB reserves: " +
                               volume["name"] + " bytes=" + str(volume["bytes"]))
        available[selected] -= volume["bytes"]
        paths[volume["name"]] = selected / str(port) / run_id / package_id / name
    return (metadata_root / str(port) / run_id / package_id, paths,
            {"free_bytes_before": {str(root): free[root] for root in roots},
             "planned_new_bytes": {str(root): free[root] - LOCAL_RESERVE_BYTES -
                                   available[root] for root in roots},
             "reserved_bytes_each_drive": LOCAL_RESERVE_BYTES,
             "metadata_root": str(metadata_root)})


def _small_mirror(local, receipt, refs):
    members = {row["member"]: row for row in receipt["members"]}
    copied = []
    for name in sorted(SMALL_MIRROR_MEMBERS & members.keys()):
        row = members[name]
        if row["bytes"] > 2_000_000:
            continue
        ref = refs[name]
        target = local / "small" / Path(*name.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or digest(target) != row["sha256"]:
            pending = target.with_name(target.name + ".part")
            payload = archive_sources.read_bytes_source(ref)
            pending.write_bytes(payload)
            if digest(pending) != row["sha256"]:
                raise RuntimeError("S1-V10 small mirror SHA differs")
            pending.replace(target)
        copied.append({"name": name, "path": str(target), "sha256": row["sha256"]})
    return copied


def _export_final_model(task, run_id, refs):
    """Expose only the small formal endpoint weights after ZIP verification."""
    if task["kind"] != "train" or task.get("budget") != 262144:
        return None
    members = ("weights/model_dir/final/policy.pt",
               "weights/model_dir/final/model.json")
    if any(name not in refs or refs[name]["bytes"] > 64 * (1 << 20) for name in members):
        raise RuntimeError("S1-V10 formal final policy/model metadata absent or unexpectedly large")
    MODEL_EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    target = MODEL_EXPORT_ROOT / run_id / "final"
    missing = sum(refs[name]["bytes"] for name in members
                  if not (target / Path(name).name).is_file() or
                  digest(target / Path(name).name) != refs[name]["sha256"])
    _require_local_capacity(MODEL_EXPORT_ROOT, missing)
    target.mkdir(parents=True, exist_ok=True)
    exported = []
    for member in members:
        ref = refs[member]
        path = target / Path(member).name
        if not path.is_file() or digest(path) != ref["sha256"]:
            pending = path.with_name(path.name + ".part")
            pending.write_bytes(archive_sources.read_bytes_source(ref))
            if pending.stat().st_size != ref["bytes"] or digest(pending) != ref["sha256"]:
                raise RuntimeError("S1-V10 exported final model SHA differs")
            pending.replace(path)
        exported.append({"path": str(path), "sha256": ref["sha256"],
                         "bytes": ref["bytes"], "source_ref": ref})
    receipt = {"schema": "s1v10.final_model_export.v1", "run_id": run_id,
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "files": exported}
    pending = target / "export_receipt.json.part"
    pending.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(target / "export_receipt.json")
    return {"path": str(target), "receipt": str(target / "export_receipt.json"),
            "files": exported}


def _merge_analysis_manifest(key, port, run_id, source_commit, local, receipt, small, refs):
    small_by_name = {row["name"]: row["path"] for row in small}
    with _lock(ANALYSIS_MANIFEST.with_suffix(".lock")):
        original = json.loads(ANALYSIS_MANIFEST.read_text(encoding="utf-8")) if ANALYSIS_MANIFEST.is_file() else {
            "scientific_commit": receipt["controller_scientific_commit"], "tasks": {}, "files": []}
        if original.get("scientific_commit") != receipt["controller_scientific_commit"]:
            raise RuntimeError("S1-V10 analysis manifest scientific commit differs")
        old = {(row["port"], row["run_id"], row.get("name") or Path(row["path"]).name): row
               for row in original.get("files", [])}
        if len(old) != len(original.get("files", [])):
            raise RuntimeError("S1-V10 existing analysis manifest repeats a logical file")
        new_rows = []
        for member in receipt["members"]:
            name = member["member"].removeprefix("job/") if member["member"].startswith("job/") else member["member"]
            ref = refs.get(member["member"])
            if not ref or ref["sha256"] != member["sha256"] or ref["bytes"] != member["bytes"]:
                raise RuntimeError("S1-V10 ZIP source reference differs")
            row = {"port": port, "run_id": run_id, "name": name,
                   "source_relative": member["remote_path"].removeprefix(ops.instance(port)["root"] + "/"),
                   "source_ref": ref, "sha256": member["sha256"], "bytes": member["bytes"]}
            if member["member"] in small_by_name:
                row["path"] = small_by_name[member["member"]]
            new_rows.append(row)
        prior = [row for row in original.get("files", []) if (row["port"], row["run_id"]) != (port, run_id)]
        old_job = [row for row in original.get("files", []) if (row["port"], row["run_id"]) == (port, run_id)]
        new_names = {row["name"]: row for row in new_rows}
        if len(new_names) != len(new_rows):
            raise RuntimeError("S1-V10 archive repeats a logical file name")
        for previous in old_job:
            name = previous.get("name") or Path(previous["path"]).name
            current = new_names.get(name)
            if current is None or current["sha256"] != previous["sha256"] or current["bytes"] != previous["bytes"]:
                raise RuntimeError("S1-V10 ZIP conflicts with previously synchronized raw file")
        tasks = dict(original.get("tasks", {}))
        prior_task = tasks.get(key)
        if prior_task and (prior_task.get("port") != port or prior_task.get("run_id") != run_id):
            raise RuntimeError("S1-V10 logical task already bound to another workbench job")
        tasks[key] = {"port": port, "run_id": run_id, "source_commit": source_commit,
                      "archive_receipt": str(local / "receipt.json"),
                      "inventory_sha256": receipt["inventory_sha256"],
                      "external_dependencies": receipt.get("external_dependencies", []),
                      "external_dependencies_closed": not receipt.get("external_dependencies")}
        updated = {**original, "sync_kind": "verified_completed_job_zip_sources",
                   "created_utc": datetime.now(timezone.utc).isoformat(),
                   "tasks": tasks, "files": sorted(prior + new_rows,
                                                  key=lambda row: (row["port"], row["run_id"],
                                                                   row.get("name") or Path(row["path"]).name))}
        ANALYSIS_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        pending = ANALYSIS_MANIFEST.with_suffix(".json.part")
        pending.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        pending.replace(ANALYSIS_MANIFEST)
        # Remove only previously mirrored large files after the authoritative ZIP
        # and manifest are verified.  A completed job can still be re-downloaded.
        removed = []
        mirror_root = (ops.LOCAL / "synced_jobs" / str(port) / run_id).resolve()
        for previous in old_job:
            path = Path(previous.get("path") or "")
            if (not path.is_file() or path.stat().st_size <= 2_000_000 or
                    not path.resolve().is_relative_to(mirror_root)):
                continue
            name = previous.get("name") or path.name
            if digest(path) != new_names[name]["sha256"]:
                raise RuntimeError("S1-V10 old mirror changed before pruning")
            path.unlink()
            removed.append(str(path))
        return {"manifest": str(ANALYSIS_MANIFEST), "files": len(new_rows),
                "pruned_large_plain_mirrors": removed}


def archive_completed(key, *, part_mib=PART_MIB, compression_level=5,
                      remote_timeout_seconds=3600):
    if (not re.fullmatch(r"[A-Za-z0-9_]+", key) or
            not 1 <= part_mib <= 8192 or not 1 <= compression_level <= 9 or
            not 60 <= remote_timeout_seconds <= 43200):
        raise ValueError("S1-V10 archive task or bounds invalid")
    state, task, audit, detail, audit_sha, source_commit = _task_identity(key)
    port, run_id = task["port"], task["run_id"]
    with _lock(LOCAL / "locks" / f"{port}_{run_id}.lock"):
        request = {"archive_root": ops.instance(port)["root"], "port": port,
                   "shared_source_root": ops.BASE + "/source",
                   "task_key": key, "run_id": run_id,
                   "request_id": task["request_id"],
                   "source_commit": source_commit,
                   "expected_step": task.get("budget") if task["kind"] == "train" else None,
                   "raw_result_sha256": audit.get("sha256") if task["kind"] == "evaluate" else None,
                   "controller_audit_sha256": audit_sha,
                   "part_bytes": part_mib * (1 << 20),
                   "compression_level": compression_level,
                   "max_files": MAX_FILES, "max_volumes": MAX_VOLUMES,
                   "max_bytes": MAX_INPUT_GIB * (1 << 30)}
        code = "REQUEST_JSON=" + repr(json.dumps(request, separators=(",", ":"))) + "\n" + REMOTE_BUILD
        remote = ops.remote_python(code, port, timeout=remote_timeout_seconds)
        if remote["exit_code"]:
            raise RuntimeError("S1-V10 remote job package failed: " + remote["stderr"][-1500:])
        handoff = json.loads(remote["stdout"].splitlines()[-1])
        package_id = Path(handoff["receipt_path"]).parent.name
        if not re.fullmatch(r"[0-9a-f]{20}_[0-9]+_c[1-9]", package_id):
            raise RuntimeError("S1-V10 remote package identity invalid")
        client = ops.connect(port)
        try:
            with client.open_sftp() as sftp:
                with sftp.open(handoff["receipt_path"], "rb") as remote_receipt:
                    receipt_bytes = remote_receipt.read()
                if hashlib.sha256(receipt_bytes).hexdigest() != handoff["receipt_sha256"]:
                    raise RuntimeError("S1-V10 remote receipt SHA differs")
                receipt = json.loads(receipt_bytes)
                if (receipt.get("schema") != "s1v10.job_archive.v1" or
                        receipt.get("experiment_id") != "S1-V10" or
                        receipt.get("run_id") != run_id or receipt.get("port") != port or
                        receipt.get("task_key") != key or
                        receipt.get("request_id") != task["request_id"] or
                        receipt.get("source_commit") != request["source_commit"] or
                        receipt.get("job_stage") != "completed" or
                        receipt.get("controller_audit_sha256") != audit_sha or
                        receipt.get("member_count") != len(receipt.get("members") or []) or
                        receipt.get("member_count", 0) > MAX_FILES or
                        len(receipt.get("volumes") or []) > MAX_VOLUMES or
                        receipt.get("total_input_bytes", 0) > MAX_INPUT_GIB * (1 << 30)):
                    raise RuntimeError("S1-V10 remote receipt scope differs")
                dependencies = receipt.get("external_dependencies")
                archive_meta = (detail["job"].get("variable_preparation") or {}).get("archive") or {}
                if task["kind"] == "prepare":
                    if (not isinstance(dependencies, list) or len(dependencies) != 1 or
                            dependencies[0].get("kind") != "shared_daily_aircraft_archive_v1" or
                            dependencies[0].get("sha256") != archive_meta.get("sha256") or
                            dependencies[0].get("bytes") != archive_meta.get("bytes") or
                            dependencies[0].get("source_path") != archive_meta.get("path")):
                        raise RuntimeError("S1-V10 preparation shared raw dependency differs")
                elif dependencies != []:
                    raise RuntimeError("non-preparation S1-V10 package has external dependency")
                receipt["controller_scientific_commit"] = state["scientific_commit"]
                members = {item["member"]: item for item in receipt["members"]}
                if len(members) != receipt["member_count"] or not receipt["volumes"]:
                    raise RuntimeError("S1-V10 receipt member/volume identity differs")
                try:
                    local, volume_paths, capacity = _select_volume_locations(
                        receipt, port, run_id, package_id, len(receipt_bytes))
                except RuntimeError as error:
                    # The remote immutable ZIPs and receipt stay intact.  This
                    # local receipt makes a later split-volume decision auditable.
                    fallback = next((root for root in (LOCAL, BACKUP_LOCAL)
                                     if shutil.disk_usage(root).free >= LOCAL_RESERVE_BYTES +
                                     len(receipt_bytes) + (1 << 20)), None)
                    if fallback is not None:
                        failure_dir = fallback / "blocked_handoffs" / str(port) / run_id / package_id
                        failure_dir.mkdir(parents=True, exist_ok=True)
                        (failure_dir / "receipt.json").write_bytes(receipt_bytes)
                        (failure_dir / "space_failure.json").write_text(json.dumps({
                            "error": str(error), "remote_receipt_path": handoff["receipt_path"],
                            "remote_receipt_sha256": handoff["receipt_sha256"],
                            "volumes": receipt["volumes"],
                            "observed_utc": datetime.now(timezone.utc).isoformat()},
                            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    raise RuntimeError(str(error) + "; remote receipt preserved at " +
                                       handoff["receipt_path"]) from error
                local.mkdir(parents=True, exist_ok=True)
                receipt_path = local / "receipt.json"
                pending = local / "receipt.json.part"
                pending.write_bytes(receipt_bytes)
                pending.replace(receipt_path)
                seen = set()
                refs = {}
                for volume in receipt["volumes"]:
                    path = volume_paths[volume["name"]]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if not path.is_file() or digest(path) != volume["sha256"]:
                        pending = path.with_name(path.name + ".part")
                        sftp.get(volume["remote_path"], str(pending))
                        if pending.stat().st_size != volume["bytes"] or digest(pending) != volume["sha256"]:
                            raise RuntimeError("S1-V10 downloaded ZIP size/SHA differs")
                        pending.replace(path)
                    with zipfile.ZipFile(path) as bundle:
                        names = bundle.namelist()
                    if any(name in seen for name in names):
                        raise RuntimeError("S1-V10 member duplicated across volumes")
                    selected = [members[name] for name in names]
                    _verify_volume(path, selected, volume["sha256"])
                    for item in selected:
                        refs[item["member"]] = {
                            "zip_path": str(path.resolve()), "zip_sha256": volume["sha256"],
                            "member": item["member"], "sha256": item["sha256"],
                            "bytes": item["bytes"]}
                    seen.update(names)
                if seen != set(members):
                    raise RuntimeError("S1-V10 ZIP volumes omit declared members")
        finally:
            client.close()
        small = _small_mirror(local, receipt, refs)
        index = _merge_analysis_manifest(key, port, run_id,
                                         request["source_commit"], local, receipt, small, refs)
        model_export = _export_final_model(task, run_id, refs)
        result = {"schema": "s1v10.job_archive_download.v1",
                  "archived_utc": datetime.now(timezone.utc).isoformat(),
                  "task_key": key, "port": port, "run_id": run_id,
                  "source_commit": request["source_commit"],
                  "remote_receipt_path": handoff["receipt_path"],
                  "remote_receipt_sha256": handoff["receipt_sha256"],
                  "local_receipt_path": str(local / "receipt.json"),
                  "total_input_bytes": receipt["total_input_bytes"],
                  "total_compressed_bytes": sum(row["bytes"] for row in receipt["volumes"]),
                  "actual_compression_ratio": receipt["total_input_bytes"] /
                      max(sum(row["bytes"] for row in receipt["volumes"]), 1),
                  "all_volume_and_member_shas_verified": True,
                  "external_dependencies": dependencies,
                  "external_dependencies_closed": not dependencies,
                  "local_space_gate": capacity,
                  "local_volume_paths": {name: str(path.resolve())
                                         for name, path in volume_paths.items()},
                  "analysis_manifest": index, "small_mirrors": small,
                  "formal_model_export": model_export,
                  "remote_package_reused": handoff["reused_verified"]}
        pending = local / "download_receipt.json.part"
        pending.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        pending.replace(local / "download_receipt.json")
        return result


def archive_all_completed(*, limit=None, on_task=None):
    if limit is not None and limit <= 0:
        raise ValueError("positive completed-job archive limit required")
    state = json.loads(controller.STATE.read_text(encoding="utf-8"))
    manifest = json.loads(ANALYSIS_MANIFEST.read_text(encoding="utf-8")) if ANALYSIS_MANIFEST.is_file() else {}
    archived = set()
    for key, task in (manifest.get("tasks") or {}).items():
        receipt_path = Path(task.get("archive_receipt") or "")
        download = receipt_path.with_name("download_receipt.json")
        proof = json.loads(download.read_text(encoding="utf-8")) if download.is_file() else {}
        if (receipt_path.is_file() and proof.get("all_volume_and_member_shas_verified") is True and
                proof.get("task_key") == key and proof.get("run_id") == task.get("run_id") and
                proof.get("port") == task.get("port")):
            archived.add(key)
    audited_keys = [key for key in state.get("audits", {}) if key in state.get("tasks", {})]
    audited_keys.sort(key=lambda key: (state["tasks"][key].get("port") in (20418, 20424), key))
    prepared_keys = [f"prepare_{port}" for port in state.get("prepares", {})]
    keys = [key for key in [*prepared_keys, *audited_keys] if key not in archived]
    if limit is not None:
        keys = keys[:limit]
    results = {}
    for key in keys:
        results[key] = archive_completed(key)
        if on_task is not None:
            on_task(key, results[key], len(results), len(keys))
    remaining = [key for key in [*prepared_keys, *audited_keys]
                 if key not in archived and key not in results]
    return {"archived_new": len(results), "already_archived": len(archived),
            "remaining_audited": len(remaining), "tasks": results,
            "all_controller_tasks_audited": (
                set(state.get("tasks", {})) == set(controller.TASKS) and
                set(state.get("audits", {})) == set(controller.TASKS)),
            "expected_science_tasks": len(controller.TASKS),
            "expected_preparation_tasks": len(ops.SSH_PORTS)}


def close_shared_source_dependencies():
    """Bind four preparation raw references to one fully verified shared ZIP."""
    source_receipt_path = ops.LOCAL / "delivery" / "source_bundle_receipt.json"
    if not source_receipt_path.is_file():
        raise RuntimeError("S1-V10 shared raw-source ZIP receipt is absent")
    source_receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
    zip_path = Path(source_receipt.get("local_path") or "")
    if (source_receipt.get("schema") != "s1v10.shared_source_bundle.v1" or
            source_receipt.get("all_member_shas_verified") is not True or
            source_receipt.get("daily_archives_copied_once") != 4 or
            not zip_path.is_file() or digest(zip_path) != source_receipt.get("sha256")):
        raise RuntimeError("S1-V10 shared source ZIP/member receipt is not closed")
    indexed = {item["member"]: item for item in source_receipt["members"]}
    if len(indexed) != len(source_receipt["members"]):
        raise RuntimeError("S1-V10 shared source receipt repeats a member")
    with zipfile.ZipFile(zip_path) as bundle:
        if set(bundle.namelist()) != set(indexed):
            raise RuntimeError("S1-V10 shared source ZIP member set changed")
        for name, item in indexed.items():
            info = bundle.getinfo(archive_sources.safe_member(name))
            if info.is_dir() or info.file_size != item["bytes"]:
                raise RuntimeError("S1-V10 shared source ZIP member size changed")
        raw_members = [item for item in source_receipt["members"]
                       if Path(item["member"]).suffix == ".tar"]
        if len(raw_members) != 4:
            raise RuntimeError("S1-V10 shared source ZIP lacks exactly four daily archives")
        for item in raw_members:
            value = hashlib.sha256()
            with bundle.open(item["member"]) as stream:
                for block in iter(lambda: stream.read(8 << 20), b""):
                    value.update(block)
            if value.hexdigest() != item["sha256"]:
                raise RuntimeError("S1-V10 shared raw member SHA differs at dependency closure")
    with _lock(ANALYSIS_MANIFEST.with_suffix(".lock")):
        manifest = json.loads(ANALYSIS_MANIFEST.read_text(encoding="utf-8"))
        state = json.loads(controller.STATE.read_text(encoding="utf-8"))
        if manifest.get("scientific_commit") != state.get("scientific_commit"):
            raise RuntimeError("S1-V10 shared closure controller source differs")
        tasks = dict(manifest.get("tasks") or {})
        prepare_keys = {f"prepare_{port}" for port in ops.SSH_PORTS}
        if not prepare_keys <= set(tasks):
            raise RuntimeError("all four completed preparation job ZIPs required before shared closure")
        files = list(manifest.get("files") or [])
        keyed = {(row["port"], row["run_id"],
                  row.get("name") or Path(row["path"]).name): row for row in files}
        if len(keyed) != len(files):
            raise RuntimeError("S1-V10 analysis manifest repeats a source name")
        dependencies = {}
        for key in sorted(prepare_keys):
            task = tasks[key]
            receipt_path = Path(task["archive_receipt"])
            if not receipt_path.is_file():
                raise RuntimeError("S1-V10 preparation job ZIP receipt absent")
            job_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            dep_rows = job_receipt.get("external_dependencies")
            if (job_receipt.get("schema") != "s1v10.job_archive.v1" or
                    job_receipt.get("run_id") != task["run_id"] or
                    job_receipt.get("port") != task["port"] or
                    not isinstance(dep_rows, list) or len(dep_rows) != 1 or
                    task.get("external_dependencies") != dep_rows):
                raise RuntimeError("S1-V10 preparation external dependency receipt differs")
            dep = dep_rows[0]
            item = indexed.get(dep["source_bundle_member"])
            if (item is None or item["sha256"] != dep["sha256"] or
                    item["bytes"] != dep["bytes"] or
                    dep["job_member"] != "job/preparation/raw/" +
                    Path(dep["source_bundle_member"]).name):
                raise RuntimeError("S1-V10 shared ZIP lacks the exact pinned preparation raw member")
            ref = {"zip_path": str(zip_path.resolve()),
                   "zip_sha256": source_receipt["sha256"],
                   "member": dep["source_bundle_member"],
                   "sha256": dep["sha256"], "bytes": dep["bytes"]}
            logical = dep["job_member"].removeprefix("job/")
            row = {"port": task["port"], "run_id": task["run_id"],
                   "name": logical, "source_relative": dep["source_path"].removeprefix(
                       ops.BASE + "/"), "source_ref": ref,
                   "sha256": dep["sha256"], "bytes": dep["bytes"],
                   "external_dependency": dep["kind"]}
            identity = (row["port"], row["run_id"], row["name"])
            prior = keyed.get(identity)
            if prior is not None and prior != row:
                raise RuntimeError("S1-V10 shared raw source already bound to another member")
            if prior is None:
                files.append(row)
                keyed[identity] = row
            tasks[key] = {**task, "external_dependencies_closed": True,
                          "shared_source_zip_sha256": source_receipt["sha256"]}
            dependencies[key] = {"run_id": task["run_id"], "source_ref": ref,
                                 "job_receipt_path": str(receipt_path)}
        updated = {**manifest, "tasks": tasks,
                   "files": sorted(files, key=lambda row: (
                       row["port"], row["run_id"], row.get("name") or Path(row["path"]).name)),
                   "shared_source_bundle_receipt": str(source_receipt_path),
                   "shared_source_dependencies_closed": True}
        pending = ANALYSIS_MANIFEST.with_suffix(".json.part")
        pending.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        pending.replace(ANALYSIS_MANIFEST)
    result = {"schema": "s1v10.shared_source_closure.v1",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_bundle_receipt": str(source_receipt_path),
              "source_bundle_receipt_sha256": digest(source_receipt_path),
              "source_zip_path": str(zip_path.resolve()),
              "source_zip_sha256": source_receipt["sha256"],
              "all_four_preparation_dependencies_closed": True,
              "dependencies": dependencies,
              "analysis_manifest": str(ANALYSIS_MANIFEST)}
    target = LOCAL / "shared_source_closure_receipt.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_suffix(".json.part")
    pending.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(target)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("archive-task", "archive-completed",
                                           "close-shared-source"))
    parser.add_argument("--task-key")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.action == "archive-task":
        if not args.task_key:
            parser.error("--task-key required")
        result = archive_completed(args.task_key)
    elif args.action == "archive-completed":
        result = archive_all_completed(limit=args.limit)
    else:
        result = close_shared_source_dependencies()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
