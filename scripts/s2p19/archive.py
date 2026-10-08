"""Copy completed S2-P19 workbench evidence into verified ZIP volumes.

This is read-only with respect to the workbench archive. The remote ZIP copies
are written under this experiment's ops/archive_s2p19, then downloaded to the
local experiment outputs. No policy, environment, or optimizer is run.
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

from . import ops


DEFAULT_DESTINATION = ops.LOCAL / "remote_archive"
SOURCE_INDEX = ops.LOCAL / "analysis" / "source_index.json"
MAX_FILES = 40000
MAX_VOLUMES = 200
MIRROR_PREFIXES = ("job/cases/", "job/validation_cases/", "scenes/")
MIRROR_NAMES = {"job/job.json", "job/config.json", "job/status.json",
                "job/results.json", "job/episodes.jsonl", "job/evaluations.jsonl",
                "job/updates.jsonl",
                "source/source_manifest.json"}


REMOTE_BUILD = r'''
import datetime,hashlib,json,os,pathlib,sys,zipfile

request=json.loads(REQUEST_JSON)
instance=request['port'];run_id=request['run_id'];part_bytes=request['part_bytes']
base=pathlib.Path('/home/u2024110224/second_stage_satellite_rl/sac_routes_s2p19')/str(instance)
archive=(base/'archive').resolve()
jobdir=archive/'outputs/workbench/jobs'/run_id
job=json.loads((jobdir/'job.json').read_text(encoding='utf-8'))
status=json.loads((jobdir/'status.json').read_text(encoding='utf-8'))
if (job.get('run_id')!=run_id or job.get('experiment_id')!='S2-P19' or
    job.get('archive_root')!=str(archive) or status.get('run_id')!=run_id or
    status.get('stage')!='completed'):
 raise RuntimeError('only a completed same-experiment workbench job may be archived')

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def safe(path):
 path=pathlib.Path(path)
 if path.is_symlink():raise RuntimeError('symlink refused: '+str(path))
 resolved=path.resolve(strict=True)
 if os.path.commonpath((str(archive),str(resolved)))!=str(archive):
  raise RuntimeError('source is outside this workbench archive: '+str(path))
 return resolved
items={}
def add(member,path,expected=None):
 if member.startswith('/') or '..' in pathlib.PurePosixPath(member).parts or member in items:
  raise RuntimeError('duplicate or unsafe archive member: '+member)
 path=safe(path)
 if not path.is_file():raise RuntimeError('missing archive input: '+str(path))
 size=path.stat().st_size
 if size>request['max_bytes'] or len(items)>=request['max_files']:
  raise RuntimeError('archive member/count bound exceeded')
 actual=digest(path)
 if expected is not None and actual!=expected:
  raise RuntimeError('frozen member SHA mismatch: '+member)
 items[member]={'member':member,'remote_path':str(path),'bytes':size,'sha256':actual}
def tree(prefix,root):
 root=safe(root)
 if not root.is_dir():raise RuntimeError('missing archive root: '+str(root))
 for file in sorted(root.rglob('*')):
  if file.is_symlink():raise RuntimeError('symlink under archive root: '+str(file))
  if file.is_file():add(prefix+'/'+file.relative_to(root).as_posix(),file)

tree('job',jobdir)
source=safe(job['source_root'])
source_manifest=json.loads((source/'source_manifest.json').read_text(encoding='utf-8'))
if source_manifest.get('git_commit')!=job.get('git_commit') or not source_manifest.get('files'):
 raise RuntimeError('frozen source manifest/commit differs')
for relative,expected in source_manifest['files'].items():
 file=(source/relative).resolve(strict=True)
 if os.path.commonpath((str(source),str(file)))!=str(source) or digest(file)!=expected:
  raise RuntimeError('frozen source file SHA differs: '+relative)
tree('source',source)
scene_roles={}
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
  add('scenes/'+role+'/'+path.name,path,ref['sha256'])
 scene_roles[role]={'id':record.get('id'),'count':len(refs),'source':record.get('source')}
 record_file=pathlib.Path(record['path'])/'record.json'
 if record_file.is_file():add('scenes/'+role+'/record.json',record_file)
model_dir=job.get('model_dir')
if model_dir and pathlib.Path(model_dir).is_dir():tree('weights/model_dir',model_dir)
for idx,ref in enumerate(job.get('model_files') or []):
 add('weights/model_input/'+str(idx).zfill(3)+'_'+pathlib.Path(ref['path']).name,
     ref['path'],ref['sha256'])
total=sum(item['bytes'] for item in items.values())
if total>request['max_bytes'] or not items:raise RuntimeError('archive size/emptiness bound exceeded')
ordered=[items[name] for name in sorted(items)]
inventory=hashlib.sha256(json.dumps(
 [(x['member'],x['bytes'],x['sha256']) for x in ordered],
 separators=(',',':')).encode()).hexdigest()
package_id=inventory[:20]+'_'+str(part_bytes)
destination=base/'ops/archive_s2p19'/run_id/package_id
receipt_path=destination/'receipt.json'
if destination.exists():
 if not receipt_path.is_file():raise RuntimeError('preserved incomplete package; refusing overwrite')
 receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
 if receipt.get('inventory_sha256')!=inventory or receipt.get('run_id')!=run_id:
  raise RuntimeError('existing package identity differs')
 for volume in receipt['volumes']:
  if digest(destination/volume['name'])!=volume['sha256']:
   raise RuntimeError('existing remote volume SHA differs')
 print(json.dumps({'receipt_path':str(receipt_path),'receipt_sha256':digest(receipt_path),
                   'volume_count':len(receipt['volumes']),'total_input_bytes':total,
                   'reused_verified':True}))
 sys.exit(0)
groups=[];current=[];amount=0
for item in ordered:
 if current and amount+item['bytes']>part_bytes:
  groups.append(current);current=[];amount=0
 current.append(item);amount+=item['bytes']
if current:groups.append(current)
if len(groups)>request['max_volumes']:raise RuntimeError('too many archive volumes')
destination.mkdir(parents=True,exist_ok=False)
volumes=[]
for idx,group in enumerate(groups,1):
 name='part_'+str(idx).zfill(4)+'.zip'
 path=destination/name
 with zipfile.ZipFile(str(path)+'.part','w',compression=zipfile.ZIP_DEFLATED,
                      compresslevel=1,allowZip64=True) as output:
  for item in group:output.write(item['remote_path'],item['member'])
 pathlib.Path(str(path)+'.part').replace(path)
 volumes.append({'name':name,'remote_path':str(path),'bytes':path.stat().st_size,
                 'sha256':digest(path),'member_count':len(group),
                 'input_bytes':sum(x['bytes'] for x in group)})
receipt={'schema':'s2p19.archive.v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'port':instance,'run_id':run_id,'experiment_id':'S2-P19','job_stage':'completed',
         'job_git_commit':job['git_commit'],'job_kind':job['kind'],
         'source_manifest_sha256':digest(source/'source_manifest.json'),
         'inventory_sha256':inventory,'total_input_bytes':total,'member_count':len(ordered),
         'scene_roles':scene_roles,'members':ordered,'volumes':volumes,
         'remote_package_directory':str(destination)}
with (destination/'receipt.json.part').open('w',encoding='utf-8') as stream:
 json.dump(receipt,stream,ensure_ascii=False,indent=2)
(destination/'receipt.json.part').replace(receipt_path)
print(json.dumps({'receipt_path':str(receipt_path),'receipt_sha256':digest(receipt_path),
                  'volume_count':len(volumes),'total_input_bytes':total,
                  'reused_verified':False}))
'''


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def safe_member(name: str) -> Path:
    candidate = Path(name)
    if (not name or "\\" in name or candidate.is_absolute() or
            any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError("unsafe archive member: " + name)
    return candidate


def verify_volume(path: Path, expected: list[dict], expected_sha: str) -> None:
    """Check ZIP bytes, exact member set, decompressed size, and member SHA."""
    if digest(path) != expected_sha:
        raise ValueError("downloaded ZIP SHA differs: " + path.name)
    indexed = {item["member"]: item for item in expected}
    if len(indexed) != len(expected):
        raise ValueError("duplicate declared archive member")
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if {info.filename for info in infos} != set(indexed) or len(infos) != len(indexed):
            raise ValueError("ZIP member list differs from remote receipt")
        for info in infos:
            safe_member(info.filename)
            item = indexed[info.filename]
            if info.is_dir() or info.file_size != item["bytes"]:
                raise ValueError("ZIP member size/type differs: " + info.filename)
            value = hashlib.sha256()
            with archive.open(info) as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    value.update(block)
            if value.hexdigest() != item["sha256"]:
                raise ValueError("ZIP member SHA differs: " + info.filename)


def mirror_member(name: str) -> bool:
    return name in MIRROR_NAMES or name.startswith(MIRROR_PREFIXES)


def materialize_mirror(volume: Path, expected: list[dict], mirror: Path) -> int:
    count = 0
    with zipfile.ZipFile(volume) as archive:
        for item in expected:
            member = item["member"]
            if not mirror_member(member):
                continue
            target = mirror / safe_member(member)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file() and target.stat().st_size == item["bytes"] and digest(target) == item["sha256"]:
                continue
            partial = target.with_name(target.name + ".part")
            with archive.open(member) as source, partial.open("wb") as output:
                shutil.copyfileobj(source, output, 8 * 1024 * 1024)
            if partial.stat().st_size != item["bytes"] or digest(partial) != item["sha256"]:
                raise ValueError("report mirror SHA differs: " + member)
            partial.replace(target)
            count += 1
    return count


def source_spec(task_id: str, run_id: str, mirror: Path, receipt: dict) -> tuple[dict | None, dict | None]:
    members = {item["member"]: item for item in receipt["members"]}

    def bound(name: str):
        value = members.get(name)
        return ({"path": str(mirror / safe_member(name)), "sha256": value["sha256"]}
                if value is not None else None)

    job = bound("job/job.json")
    config = bound("job/config.json")
    status = bound("job/status.json")
    results = bound("job/results.json")
    manifest = bound("source/source_manifest.json")
    if not all((job, config, status, manifest)):
        raise ValueError("archived job/config/status/source manifest is incomplete")
    scene_roles = receipt["scene_roles"]
    primary = next((role for role in ("test", "validation", "development", "train")
                    if role in scene_roles), None)
    spec = None
    if results:
        spec = {"run_id": run_id, "port": receipt.get("port"),
                "results_path": results["path"],
                "results_sha256": results["sha256"],
                "job_path": job["path"], "job_sha256": job["sha256"],
                "config_path": config["path"], "config_sha256": config["sha256"],
                "status_path": status["path"], "status_sha256": status["sha256"],
                "source_manifest_path": manifest["path"],
                "source_manifest_sha256": manifest["sha256"],
                "scene_root": str(mirror / "scenes" / primary) if primary else None,
                "case_root": str(mirror / "job" / "cases") if (mirror / "job" / "cases").is_dir() else None,
                "archive_receipt_path": str(mirror.parent / "receipt.json"),
                "archive_inventory_sha256": receipt["inventory_sha256"]}
    training = None
    if task_id in ("C0", "C1", "I0"):
        training = {"run_id": run_id, "port": receipt.get("port"),
                    "archive_inventory_sha256": receipt["inventory_sha256"],
                    "archive_receipt_path": str(mirror.parent / "receipt.json")}
        for kind in ("episodes", "evaluations", "updates"):
            ref = bound(f"job/{kind}.jsonl")
            if ref:
                training[kind] = ref
    return spec, training


def merge_index(path: Path, task_id: str, spec: dict | None,
                training: dict | None) -> None:
    with index_lock(path):
        _merge_index_locked(path, task_id, spec, training)


@contextmanager
def index_lock(path: Path):
    """Cross-process lock for the single shared source index."""
    lock = Path(path).with_name(Path(path).name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+b") as handle:
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


def _merge_index_locked(path: Path, task_id: str, spec: dict | None,
                        training: dict | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    original = path.read_bytes() if path.is_file() else None
    data = json.loads(original) if original else {"schema": "s2p19.source_index.v1", "jobs": {}, "training": {}}
    if data.get("schema") != "s2p19.source_index.v1":
        raise ValueError("existing S2-P19 source index has unknown schema")
    if spec:
        prior = data.setdefault("jobs", {}).get(task_id)
        if prior and (prior.get("run_id") != spec["run_id"] or
                      prior.get("port") != spec.get("port") or
                      prior.get("archive_inventory_sha256") != spec.get("archive_inventory_sha256")):
            raise ValueError("task ID already bound to another run/port/inventory")
        data["jobs"][task_id] = spec
    if training:
        prior = data.setdefault("training", {}).get(task_id)
        if prior and (prior.get("run_id") != training["run_id"] or
                      prior.get("port") != training.get("port") or
                      prior.get("archive_inventory_sha256") != training.get("archive_inventory_sha256") or
                      any(prior.get(kind, {}).get("sha256") != training.get(kind, {}).get("sha256")
                          for kind in ("episodes", "evaluations", "updates"))):
            raise ValueError("training arm already bound to different archived evidence")
        data["training"][task_id] = training
    if path.is_file() and path.read_bytes() != original:
        raise RuntimeError("source index changed during archive merge")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def archive_completed(port: int, run_id: str, task_id: str, *,
                      destination: Path = DEFAULT_DESTINATION,
                      source_index: Path = SOURCE_INDEX,
                      part_mib: int = 256, max_gib: int = 20) -> dict:
    tagged_port = re.search(r"_(20418|20424|20473|20474)$", task_id)
    if (port not in ops.SSH_PORTS or not re.fullmatch(r"[a-z0-9_]+", run_id) or
            not re.fullmatch(r"[A-Za-z0-9_]+", task_id) or
            (tagged_port is not None and int(tagged_port.group(1)) != port) or
            not 1 <= part_mib <= 1024 or not 1 <= max_gib <= 64):
        raise ValueError("S2-P19 archive scope or bounds invalid")
    detail = ops.api(port, "/api/runs/" + run_id)
    job, status = detail.get("job") or {}, detail.get("status") or {}
    if (detail.get("read_only") or job.get("run_id") != run_id or
            job.get("experiment_id") != "S2-P19" or
            job.get("archive_root") != ops.instance(port)["archive"] or
            status.get("run_id") != run_id or status.get("stage") != "completed"):
        raise ValueError("only a completed same-instance S2-P19 workbench job may be archived")
    request = {"port": port, "run_id": run_id, "part_bytes": part_mib * 1024 ** 2,
               "max_bytes": max_gib * 1024 ** 3, "max_files": MAX_FILES,
               "max_volumes": MAX_VOLUMES}
    code = "REQUEST_JSON=" + repr(json.dumps(request, separators=(",", ":"))) + "\n" + REMOTE_BUILD
    response = ops.remote_python(code, port, timeout=1200)
    if response["exit_code"]:
        raise RuntimeError("remote archive copy failed: " + response["stderr"][-1800:])
    handoff = json.loads(response["stdout"].splitlines()[-1])
    package_id = Path(handoff["receipt_path"]).parent.name
    if not re.fullmatch(r"[0-9a-f]{20}_[0-9]+", package_id):
        raise ValueError("remote archive package identity invalid")
    local = Path(destination) / str(port) / run_id / package_id
    local.mkdir(parents=True, exist_ok=True)
    transport = ops.connect(port)
    try:
        import paramiko
        with paramiko.SFTPClient.from_transport(transport.get_transport()) as sftp:
            remote_receipt = handoff["receipt_path"]
            receipt_path = local / "receipt.json"
            staging = receipt_path.with_suffix(".json.part")
            sftp.get(remote_receipt, str(staging))
            if digest(staging) != handoff["receipt_sha256"]:
                raise ValueError("downloaded remote archive receipt SHA differs")
            staging.replace(receipt_path)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (receipt.get("schema") != "s2p19.archive.v1" or
                    receipt.get("port") != port or receipt.get("run_id") != run_id or
                    receipt.get("job_git_commit") != job.get("git_commit") or
                    receipt.get("job_stage") != "completed" or
                    receipt.get("member_count") != len(receipt.get("members") or []) or
                    receipt.get("member_count", 0) > MAX_FILES or
                    receipt.get("total_input_bytes", 0) > max_gib * 1024 ** 3 or
                    len(receipt.get("volumes") or []) > MAX_VOLUMES):
                raise ValueError("remote archive receipt scope differs")
            members = {item["member"]: item for item in receipt["members"]}
            if len(members) != receipt["member_count"]:
                raise ValueError("archive receipt has duplicate members")
            verified = set()
            mirrored = 0
            for volume in receipt["volumes"]:
                path = local / safe_member(volume["name"])
                if not path.is_file() or digest(path) != volume["sha256"]:
                    partial = path.with_name(path.name + ".part")
                    sftp.get(volume["remote_path"], str(partial))
                    if partial.stat().st_size != volume["bytes"] or digest(partial) != volume["sha256"]:
                        raise ValueError("downloaded ZIP size/SHA differs: " + volume["name"])
                    partial.replace(path)
                with zipfile.ZipFile(path) as bundle:
                    names = bundle.namelist()
                selected = [members[name] for name in names]
                if any(name in verified for name in names):
                    raise ValueError("archive member appears in two volumes")
                verify_volume(path, selected, volume["sha256"])
                verified.update(names)
                mirrored += materialize_mirror(path, selected, local / "mirror")
            if verified != set(members):
                raise ValueError("archive volumes omit declared members")
    finally:
        transport.close()
    spec, training = source_spec(task_id, run_id, local / "mirror", receipt)
    merge_index(Path(source_index), task_id, spec, training)
    local_receipt = {"archived_utc": datetime.now(timezone.utc).isoformat(),
                     "purpose": "copy_only_completed_workbench_job_no_new_evaluation",
                     "task_id": task_id, "port": port, "run_id": run_id,
                     "remote_receipt_path": handoff["receipt_path"],
                     "remote_receipt_sha256": handoff["receipt_sha256"],
                     "remote_reused_verified": handoff["reused_verified"],
                     "local_receipt_path": str(local / "receipt.json"),
                     "volume_count": len(receipt["volumes"]),
                     "member_count": receipt["member_count"],
                     "total_input_bytes": receipt["total_input_bytes"],
                     "mirror_files_written": mirrored,
                     "all_volume_and_member_shas_verified": True,
                     "source_index_path": str(source_index),
                     "source_index_sha256": digest(source_index)}
    output = local / "download_receipt.json"
    output.write_text(json.dumps(local_receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"receipt": str(output), "run_id": run_id, "port": port,
            "volume_count": local_receipt["volume_count"],
            "member_count": local_receipt["member_count"],
            "total_input_bytes": local_receipt["total_input_bytes"],
            "source_index": str(source_index)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True, choices=ops.SSH_PORTS)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--source-index", type=Path, default=SOURCE_INDEX)
    parser.add_argument("--part-mib", type=int, default=256)
    parser.add_argument("--max-gib", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(archive_completed(args.port, args.run_id, args.task_id,
                                       destination=args.destination,
                                       source_index=args.source_index,
                                       part_mib=args.part_mib, max_gib=args.max_gib),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
