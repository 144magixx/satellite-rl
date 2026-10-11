"""Canonical ND12 ZIP64 archiving. Offline plan/verify; exports require terminal jobs."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
import shutil
import zipfile
from scripts.stage1_nondrl12 import ops
LOCAL=ops.LOCAL / "job_archives"
BACKUP=Path("C:/paper_archives/outputs/experiments/first_stage_nondrl12_20261012/job_archives")
EXTRA=Path("E:/paper_archives/outputs/experiments/first_stage_nondrl12_20261012/job_archives")
RESERVE=8*(1<<30)
PART_BYTES=256*(1<<20)
MAX_FILES=40000
MAX_JOB_BYTES=16*(1<<30)


def plan(state, preparations=None):
    if state.get("experiment_id") != "S1-ND12":
        raise ValueError("wrong controller group")
    commit = state.get("scientific_commit") or (preparations or {}).get("scientific_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("explicit frozen scientific commit required")
    tasks = {}
    jobs = state.get("jobs", state.get("tasks", {}))
    for key, task in jobs.items():
        ops.instance(int(task["port"]))
        if not re.fullmatch(r"[A-Za-z0-9_-]+", task["run_id"]):
            raise ValueError("unsafe workbench run identity")
        tasks[key] = {"key": key, "port": int(task["port"]), "run_id": task["run_id"],
                      "request_id": task.get("request_id"), "source_commit": commit,
                      "kind": task.get("kind"), "route": task.get("route"), "method": task.get("method")}
    for key, task in (preparations or {}).get("tasks", {}).items():
        name = "prepare/" + key
        if name in tasks:
            raise ValueError("preparation task identity collision")
        tasks[name] = {"key": name, "port": int(task["port"]), "run_id": task["run_id"],
                       "request_id": task.get("request_id"), "source_commit": commit, "kind": "prepare"}
    identities = [(task["port"], task["run_id"]) for task in tasks.values()]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate workbench task ownership")
    return {"experiment_id": "S1-ND12", "scientific_commit": commit, "tasks": tasks,
            "sources": (preparations or {}).get("sources", state.get("sources", [])),
            "partition_manifest_path": (preparations or {}).get("partition_manifest_path", state.get("partition_manifest_path")),
            "partition_manifest_sha256": (preparations or {}).get("partition_manifest_sha256", state.get("partition_manifest_sha256")),
            "canonical_raw_source_copies": 1, "reserve_bytes_each_drive": RESERVE}


def verify_source_receipt(receipt, directory):
    if receipt.get("experiment_id") != "S1-ND12" or receipt.get("run_id") != "shared_sources":
        raise ValueError("wrong source receipt")
    members = {}
    for volume in receipt.get("volumes", []):
        verify_volume(volume_path(directory, volume), volume)
        for name, value in volume["members"].items():
            if name in members:
                raise ValueError("duplicate source member")
            members[name] = value
    sources = receipt.get("source_refs", [])
    if len(sources) != 3 or len({r["date_utc"] for r in sources}) != 3:
        raise ValueError("exactly three frozen source days required")
    if {(r["role"], r["date_utc"]) for r in sources} != {("validation", "2024-02-10"),
            ("test", "2024-02-11"), ("test", "2024-02-12")}:
        raise ValueError("ND12 source day/role differs")
    raw_members = {name for name in members if name.endswith(".tar")}
    if len(raw_members) != 3:
        raise ValueError("raw source duplicate or missing")
    if members.get("shared_sources/partition_manifest.json", {}).get("sha256") != receipt.get("partition_manifest_sha256"):
        raise ValueError("source partition manifest identity differs")
    for source in sources:
        member = source["source_bundle_member"]
        if members.get(member) != {"sha256": source["sha256"], "bytes": source["bytes"]}:
            raise ValueError("source raw member identity differs")
        prefix = member.rsplit("/", 1)[0]
        for name, sha in (("source_preflight.json", source["source_preflight_sha256"]),
                          ("selection_contract.json", source["selection_contract_sha256"])):
            if members.get(prefix + "/" + name, {}).get("sha256") != sha:
                raise ValueError("source metadata reference differs")
    return members


def verify_receipt(receipt, directory, source_receipt=None, source_directory=None, dependency_receipts=None):
    if receipt.get("experiment_id") != "S1-ND12" or not receipt.get("volumes"):
        raise ValueError("wrong or empty ND12 receipt")
    members = {}
    for volume in receipt["volumes"]:
        verify_volume(volume_path(directory, volume), volume)
        for name, value in volume["members"].items():
            if name in members:
                raise ValueError("member duplicated across volumes")
            members[name] = value
    prefix = f"{receipt['port']}/{receipt['run_id']}/"
    if not {prefix + name for name in ("job.json", "config.json", "status.json")}.issubset(members):
        raise ValueError("workbench identity evidence missing")
    if any(name.endswith(".tar") for name in members):
        raise ValueError("task duplicates canonical raw source")
    if receipt.get("source_refs"):
        if source_receipt is None or source_directory is None:
            raise ValueError("shared source receipt required")
        sources = verify_source_receipt(source_receipt, source_directory)
        for ref in receipt["source_refs"]:
            if sources.get(ref["source_bundle_member"]) != {"sha256": ref["sha256"], "bytes": ref["bytes"]}:
                raise ValueError("task raw source reference does not close")
    available = set()
    dependency_by_sha = {}
    for dep in dependency_receipts or []:
        if dep.get("experiment_id") != "S1-ND12" or dep.get("source_commit") != receipt["source_commit"]:
            raise ValueError("dependency archive group/source differs")
        for volume in dep.get("volumes", []):
            verify_volume(volume_path(LOCAL / str(dep["port"]) / dep["run_id"], volume), volume)
            for value in volume["members"].values():
                available.add(value["sha256"])
                dependency_by_sha.setdefault(value["sha256"], set()).add(dep["run_id"])
    dependencies = receipt.get("dependency_refs", [])
    if dependencies and dependency_receipts is not None and any(d["sha256"] not in available for d in dependencies):
        raise ValueError("teacher/model/initial dependency reference does not close")
    if dependency_receipts is not None and any(d.get("run_id") and d["run_id"] not in dependency_by_sha.get(d["sha256"], set()) for d in dependencies):
        raise ValueError("dependency model/teacher task ownership differs")
    return {"run_id": receipt["run_id"], "members": len(members), "source_refs": len(receipt.get("source_refs", [])),
            "dependency_refs": len(dependencies), "dependencies_closed": not dependencies or dependency_receipts is not None,
            "closed": not dependencies or dependency_receipts is not None}


REMOTE_BUILD = r'''
import hashlib,json,os,pathlib,shutil,sys,zipfile
request=json.loads(REQUEST_JSON);base=pathlib.Path(request['base']).resolve()
port,run_id=request['port'],request['run_id'];dest=pathlib.Path(request['destination'])
jobdir=pathlib.Path(request['archive'])/'outputs/workbench/jobs'/run_id
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as s:
  for b in iter(lambda:s.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def safe(path):
 p=pathlib.Path(path)
 if p.is_symlink() or not p.resolve(strict=True).is_relative_to(base):raise RuntimeError('unsafe source path')
 return p.resolve()
if not dest.resolve().is_relative_to(base):raise RuntimeError('destination escaped ND12')
dest.mkdir(parents=True,exist_ok=True);lock=dest/'.lock';lock.mkdir()
try:
 prior=dest/'receipt.json'
 if prior.exists():
  receipt=json.loads(prior.read_text())
  if receipt.get('source_commit')!=request['source_commit'] or receipt.get('run_id')!=run_id:raise RuntimeError('archive retry identity differs')
  for v in receipt['volumes']:
   if sha(pathlib.Path(v['remote_path']))!=v['sha256']:raise RuntimeError('archive retry SHA differs')
  print(json.dumps(receipt));sys.exit(0)
 items={};refs=[];deps=[]
 def add(name,path,expected=None):
  if name in items or name.startswith('/') or '\\' in name or any(x in ('','.','..') for x in name.split('/')):raise RuntimeError('unsafe duplicate member')
  p=safe(path);h=sha(p)
  if expected is not None and h!=expected:raise RuntimeError('pinned source SHA differs')
  if len(items)>=request['max_files']:raise RuntimeError('file bound')
  items[name]={'path':str(p),'sha256':h,'bytes':p.stat().st_size}
 def tree(prefix,root,skip=()):
  root=safe(root)
  for p in sorted(root.rglob('*')):
   if str(p) in skip:continue
   if p.is_symlink():raise RuntimeError('unexpected symlink')
   if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':
    if p.suffix=='.tar':raise RuntimeError('raw source must be a reference')
    add(prefix+'/'+p.relative_to(root).as_posix(),p)
 if request['mode']=='source':
  for entry in request['sources']:
   p=safe(entry['path']);h=sha(p)
   if h!=entry['sha256'] or p.stat().st_size!=entry['bytes']:raise RuntimeError('final prepared raw SHA/size differs')
   prefix='shared_sources/'+entry['role']+'/'+entry['date_utc']
   add(prefix+'/'+p.name,p,h)
   for name,key in [('source_preflight.json','source_preflight'),('selection_contract.json','selection_contract')]:
    add(prefix+'/'+name,entry[key+'_path'],entry[key+'_sha256'])
   for name in ['release_api.json','PREFERRED_RELEASES.txt','LICENSE-ODbL-official.txt']:
    extra=p.parent/name
    if extra.exists():add(prefix+'/'+name,extra)
   refs.append({**entry,'source_bundle_member':prefix+'/'+p.name})
  add('shared_sources/partition_manifest.json',request['partition_manifest_path'],request['partition_manifest_sha256'])
  status={'stage':'completed'}
 else:
  job=json.loads((jobdir/'job.json').read_text());status=json.loads((jobdir/'status.json').read_text())
  launch=json.loads((jobdir/'launch.json').read_text()) if (jobdir/'launch.json').exists() else {}
  if job.get('experiment_id')!='S1-ND12' or job.get('run_id')!=run_id or job.get('git_commit')!=request['source_commit'] or job.get('archive_root')!=request['archive']:raise RuntimeError('workbench group/source/identity differs')
  if request.get('request_id') and job.get('request_id')!=request['request_id']:raise RuntimeError('request identity differs')
  if status.get('stage') not in ('completed','failed','stopped'):raise RuntimeError('task is not terminal')
  if status['stage']=='failed' and not (jobdir/'failure.json').exists():raise RuntimeError('failed task lacks failure evidence')
  if status['stage']=='stopped' and status.get('scientific_stop') is not True:raise RuntimeError('unexplained stopped task')
  for pid in {launch.get('pid'),status.get('pid')}:
   if not isinstance(pid,int) or pid<=0:continue
   proc=pathlib.Path('/proc')/str(pid)/'stat'
   if proc.exists():
    born=launch.get('process_start_time')
    if not isinstance(born,(int,float)):raise RuntimeError('live PID birth unknown')
    raw=proc.read_text();ticks=int(raw[raw.rfind(')')+2:].split()[19])
    boot=next(int(x.split()[1]) for x in pathlib.Path('/proc/stat').read_text().splitlines() if x.startswith('btime '))
    if abs(boot+ticks/os.sysconf('SC_CLK_TCK')-born)<.02:raise RuntimeError('same-birth worker remains live')
  prefix=str(port)+'/'+run_id;skip=[]
  prep=job.get('dataset_preparation')
  if prep:
   p=base/'source'/prep['role']/prep['date_utc']/prep['archive_name'];h=sha(safe(p))
   artifact=json.loads((jobdir/'artifacts.json').read_text())
   if artifact['raw_archive_sha256']!=h or p.stat().st_size!=prep['expected_bytes']:raise RuntimeError('prepared raw source differs')
   linked=jobdir/'preparation/raw'/p.name
   if not linked.exists() or not linked.samefile(p):raise RuntimeError('raw source hardlink differs')
   skip.append(str(linked));refs.append({'source_bundle_member':'shared_sources/'+prep['role']+'/'+prep['date_utc']+'/'+p.name,'sha256':h,'bytes':p.stat().st_size})
  for entry in job.get('model_files',[]):
   skip.append(entry['path']);deps.append({'kind':'model','run_id':job.get('model_source_run_id'),'sha256':entry['sha256'],'path':entry.get('source_path',entry['path'])})
  for key in ['teacher_source','initial_source','validation_initial_source']:
   records=job.get(key,[]);records=records if isinstance(records,list) else [records]
   for record in records:
    p=safe(record['path'])
    if sha(p)!=record['sha256']:raise RuntimeError('dependency manifest SHA differs')
    deps.append({'kind':key,'run_id':record.get('run_id'),'sha256':record['sha256'],'path':record.get('source_path',record['path'])})
  tree(prefix,jobdir,skip)
  config=json.loads((jobdir/'config.json').read_text())
  for role,record in config.get('datasets',{}).items():
   p=safe(record['path'])
   if sha(p)!=record['sha256']:raise RuntimeError('dataset SHA differs')
   if not p.is_relative_to(jobdir):add(prefix+'/scene_inputs/'+role+'/inputs.npz',p,record['sha256'])
  source=safe(job['source_root']);manifest=json.loads((source/'source_manifest.json').read_text())
  if manifest.get('git_commit')!=request['source_commit']:raise RuntimeError('source manifest commit differs')
  for relative,h in manifest.get('files',{}).items():
   if sha(safe(source/relative))!=h:raise RuntimeError('frozen source member changed')
  tree(prefix+'/source',source)
  if job.get('kind')=='train' and job.get('model_dir'):tree(prefix+'/models',job['model_dir'])
 total=sum(x['bytes'] for x in items.values())
 if request['mode']=='task' and total>request['max_bytes']:raise RuntimeError('task export byte bound')
 if shutil.disk_usage(dest).free<request['reserve']+total+len(items)*2048:raise RuntimeError('8GiB remote reserve')
 volumes=[];group=[];size=0
 def flush():
  if not group:return
  name='volume_%03d.zip'%len(volumes);path=dest/name;temp=dest/(name+'.tmp')
  if path.exists() or temp.exists():raise RuntimeError('partial remote ZIP requires audit; no overwrite')
  with zipfile.ZipFile(temp,'w',allowZip64=True,compression=zipfile.ZIP_DEFLATED,compresslevel=6) as out:
   for name,r in group:out.write(r['path'],name)
  temp.replace(path);volumes.append({'file':path.name,'remote_path':str(path),'sha256':sha(path),'bytes':path.stat().st_size,'members':{n:{'sha256':r['sha256'],'bytes':r['bytes']} for n,r in group}})
 for name,r in sorted(items.items()):
  if group and size+r['bytes']>request['part_bytes']:flush();group=[];size=0
  group.append((name,r));size+=r['bytes']
 flush()
 receipt={'experiment_id':'S1-ND12','port':port,'run_id':run_id,'source_commit':request['source_commit'],'status':status['stage'],'source_refs':refs,'dependency_refs':deps,'volumes':volumes,'partition_manifest_sha256':request.get('partition_manifest_sha256')}
 (dest/'receipt.json').write_text(json.dumps(receipt));print(json.dumps(receipt))
finally:lock.rmdir()
'''


def one_task(entry):
    port = entry["port"]
    local = LOCAL / str(port) / entry["run_id"] / "receipt.json"
    if local.is_file():
        receipt = json.loads(local.read_text())
        if receipt["source_commit"] != entry["source_commit"]:
            raise ValueError("existing local receipt source differs")
        for volume in receipt["volumes"]:
            verify_volume(volume_path(local.parent, volume), volume)
        return receipt
    detail = ops.api(port, f"/api/runs/{entry['run_id']}")
    if detail.get("status", {}).get("stage") not in ("completed", "failed", "stopped"):
        raise ValueError("workbench task is not terminal")
    info = ops.instance(port)
    request = {**entry, "base": ops.BASE, "archive": info["archive"], "mode": "task",
               "destination": info["ops"] + "/archive_nd12/" + entry["run_id"],
               "reserve": RESERVE, "part_bytes": PART_BYTES, "max_files": MAX_FILES, "max_bytes": MAX_JOB_BYTES}
    remote = ops.remote_python("REQUEST_JSON=" + repr(json.dumps(request)) + "\n" + REMOTE_BUILD, port, timeout=1800)
    if remote["exit_code"]:
        raise RuntimeError(remote["stderr"][-2000:])
    receipt = json.loads(remote["stdout"].splitlines()[-1])
    directory = _download(port, receipt)
    (directory / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return receipt


def one_source(planned, *, port=20418):
    if len(planned.get("sources", [])) != 3:
        raise ValueError("three final prepared source SHA records required; pre-download hashes are insufficient")
    local = LOCAL / "20418/shared_sources/receipt.json"
    if local.exists():
        receipt = json.loads(local.read_text())
        if receipt["source_commit"] != planned["scientific_commit"]:
            raise ValueError("source receipt source differs")
        verify_source_receipt(receipt, local.parent)
        return receipt
    info = ops.instance(port)
    request = {"base": ops.BASE, "mode": "source", "port": 20418, "run_id": "shared_sources",
               "source_commit": planned["scientific_commit"], "sources": planned["sources"],
               "partition_manifest_path": planned["partition_manifest_path"],
               "partition_manifest_sha256": planned["partition_manifest_sha256"],
               "destination": info["ops"] + "/archive_nd12/shared_sources", "archive": info["archive"],
               "reserve": RESERVE, "part_bytes": PART_BYTES, "max_files": MAX_FILES, "max_bytes": MAX_JOB_BYTES}
    remote = ops.remote_python("REQUEST_JSON=" + repr(json.dumps(request)) + "\n" + REMOTE_BUILD, port, timeout=1800)
    if remote["exit_code"]:
        raise RuntimeError(remote["stderr"][-2000:])
    receipt = json.loads(remote["stdout"].splitlines()[-1])
    directory = _download(port, receipt)
    verify_source_receipt(receipt, directory)
    (directory / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "one-task", "one-source", "verify", "verify-source"))
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--preparations", type=Path)
    parser.add_argument("--key")
    parser.add_argument("--source-receipt", type=Path)
    parser.add_argument("--dependencies", type=Path, nargs="*")
    args = parser.parse_args()
    state = json.loads(args.state.read_text())
    preparation = json.loads(args.preparations.read_text()) if args.preparations else None
    planned = plan(state, preparation)
    if args.action == "plan":
        print(json.dumps({"experiment_id": "S1-ND12", "tasks": len(planned["tasks"]), "sources": len(planned["sources"])}));return
    if args.action == "one-source":
        result = one_source(planned)
    elif args.action == "verify-source":
        path = args.source_receipt or LOCAL / "20418/shared_sources/receipt.json"
        result = {"members": len(verify_source_receipt(json.loads(path.read_text()), path.parent)), "closed": True}
    else:
        if args.key not in planned["tasks"]:
            parser.error("--key must identify a frozen task")
        entry = planned["tasks"][args.key]
        if args.action == "one-task":
            result = one_task(entry)
        else:
            directory = LOCAL / str(entry["port"]) / entry["run_id"]
            source = json.loads(args.source_receipt.read_text()) if args.source_receipt else None
            dependencies = [json.loads(path.read_text()) for path in args.dependencies] if args.dependencies is not None else None
            result = verify_receipt(json.loads((directory / "receipt.json").read_text()), directory, source,
                                    args.source_receipt.parent if args.source_receipt else None, dependencies)
    print(json.dumps({key: result.get(key) for key in ("run_id", "closed", "members", "source_refs", "dependencies_closed")}))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()

def volume_path(directory, volume):
    """Resolve a single canonical local ZIP; old receipts remain readable."""
    canonical = volume.get("canonical_path")
    if canonical is not None:
        path = Path(canonical)
        if not path.is_absolute() or path.name != volume["file"]:
            raise ValueError("invalid canonical ZIP path")
        return path
    return Path(directory) / volume["file"]

def verify_volume(path, expected):
    """Stream each ZIP member; compare explicit member and ZIP byte SHA."""
    path = Path(path)
    if digest(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
        raise ValueError(f"ZIP byte SHA/size differs: {path}")
    manifest = expected["members"]
    if len(manifest) != len(set(manifest)):
        raise ValueError("duplicate archived member")
    with zipfile.ZipFile(path) as archive:
        actual = archive.namelist()
        if set(actual) != set(manifest) or len(actual) != len(manifest):
            raise ValueError("ZIP member inventory differs")
        for name in actual:
            if (name.startswith("/") or "\\" in name or
                    any(x in ("", ".", "..") for x in name.split("/"))):
                raise ValueError("unsafe ZIP member")
            record = manifest[name]
            h = hashlib.sha256()
            size = 0
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(8 << 20), b""):
                    h.update(block)
                    size += len(block)
            if h.hexdigest() != record["sha256"] or size != record["bytes"]:
                raise ValueError(f"ZIP member SHA/size differs: {name}")
    return len(manifest)

def _disk_free(path):
    path = Path(path)
    while not path.exists():
        if path == path.parent:
            raise FileNotFoundError(path)
        path = path.parent
    return shutil.disk_usage(path).free

def plan_canonical(port, receipt, free_by_root=None):
    """Choose one D/C/E location per ZIP, preserving 8 GiB on each drive."""
    roots = {"local": LOCAL / str(port) / receipt["run_id"],
             "backup": BACKUP / str(port) / receipt["run_id"]}
    # E is optional on other hosts. In tests, free_by_root explicitly selects
    # the drives under consideration; on this host, probe the E drive itself.
    if ((free_by_root is None and EXTRA.anchor and Path(EXTRA.anchor).exists()) or
            (free_by_root is not None and "extra" in free_by_root)):
        roots["extra"] = EXTRA / str(port) / receipt["run_id"]
    free = {key: (_disk_free(root) if free_by_root is None else free_by_root[key]) - RESERVE
            for key, root in roots.items()}
    choices = {}
    for volume in receipt["volumes"]:
        filename = volume["file"]
        if filename in choices or Path(filename).name != filename:
            raise ValueError("duplicate or unsafe archive volume name")
        existing = [root / filename for root in roots.values() if (root / filename).exists()]
        if len(existing) > 1:
            raise ValueError("ZIP already exists on both local drives")
        if existing:
            verify_volume(existing[0], volume)
            choices[filename] = existing[0]
    for volume in sorted(receipt["volumes"], key=lambda row: row["bytes"], reverse=True):
        if volume["file"] in choices:
            continue
        partials = [(key, root / (volume["file"] + ".part")) for key, root in roots.items()
                    if (root / (volume["file"] + ".part")).exists()]
        if partials:
            if len(partials) != 1:
                raise ValueError("partial ZIP exists on both drives")
            key, partial = partials[0]
            remaining = volume["bytes"] - partial.stat().st_size
            if remaining < 0 or free[key] < remaining:
                raise RuntimeError("partial ZIP is oversized or lacks 8GiB reserve")
            choices[volume["file"]] = roots[key] / volume["file"]
            free[key] -= remaining
            continue
        eligible = [key for key in roots if free[key] >= volume["bytes"]]
        if not eligible:
            raise RuntimeError("no canonical drive has ZIP bytes plus 8GiB reserve")
        # Preserve the existing D/C evidence and their small remaining
        # capacity. New exports use the verified E volume when available.
        key = ("extra" if "extra" in eligible else
               min(eligible, key=lambda item: (free[item], item != "local")))
        target = roots[key] / volume["file"]
        if target.with_suffix(target.suffix + ".part").exists():
            raise FileExistsError("interrupted local ZIP requires manual inspection")
        choices[volume["file"]] = target
        free[key] -= volume["bytes"]
    return {name: str(path.resolve()) for name, path in choices.items()}

def remote_blocks(remote, total_bytes, window_bytes=2 << 20, *, start_offset=0):
    """Use bounded public SFTP vector reads, avoiding one round trip per 32 KiB."""
    offset = start_offset
    while offset < total_bytes:
        size = min(window_bytes, total_bytes - offset)
        blocks = list(remote.readv([(offset, size)]))
        if len(blocks) != 1 or len(blocks[0]) != size:
            raise ValueError("remote ZIP vector read was short or out of order")
        yield blocks[0]
        offset += size

def _download(port, receipt):
    directory = LOCAL / str(port) / receipt["run_id"]
    choices = plan_canonical(port, receipt)
    # Receipts live in the small D-side index even when every ZIP goes to C.
    directory.mkdir(parents=True, exist_ok=True)
    client = ops.connect(port)
    try:
        with client.open_sftp() as sftp:
            for volume in receipt["volumes"]:
                target = Path(choices[volume["file"]])
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    verify_volume(target, volume)
                else:
                    partial = target.with_suffix(target.suffix + ".part")
                    h = hashlib.sha256()
                    size = 0
                    if partial.exists():
                        with partial.open("rb") as prior:
                            for block in iter(lambda: prior.read(8 << 20), b""):
                                h.update(block)
                                size += len(block)
                        if size > volume["bytes"]:
                            raise ValueError("partial ZIP exceeds immutable receipt size")
                    resume_offset = size
                    with sftp.open(volume["remote_path"], "rb") as remote, partial.open("ab") as local:
                        remote.settimeout(90)
                        for block in remote_blocks(remote, volume["bytes"], start_offset=resume_offset):
                            h.update(block)
                            size += len(block)
                            local.write(block)
                    if size != volume["bytes"] or h.hexdigest() != volume["sha256"]:
                        raise ValueError("remote ZIP stream SHA/size differs")
                    partial.replace(target)
                    verify_volume(target, volume)
                    volume["download_resumed_from_bytes"] = resume_offset
                volume["canonical_path"] = str(target)
    finally:
        client.close()
    return directory


if __name__ == "__main__":
    main()
