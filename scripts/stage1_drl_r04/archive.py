"""Independent S1-D12A ZIP64 export with SHA and shared-source closure.

The plan command is offline. one-task performs one bounded remote export and
streaming download after live workbench/worker checks. verify is offline.
No legacy S1-V10 archive state is read or written.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

from scripts.stage1_drl_r04 import ops
from scripts.stage1_drl_r04.data_controller import ASSIGNMENT, RECEIPT


LOCAL = ops.LOCAL / "job_archives"
BACKUP = Path("C:/paper_archives/outputs/experiments/first_stage_s1_d12a_20261010/job_archives")
RECOVERY_PROOF = ops.LOCAL / "preflight" / "source_recovery_cache_audit.json"
RESERVE = 8 * (1 << 30)
PART_BYTES = 256 * (1 << 20)
MAX_FILES = 40000
MAX_JOB_BYTES = 16 * (1 << 30)


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


def plan(state, preparations):
    if (state.get("experiment_id") != "S1-D12A" or
            preparations.get("experiment_id") != "S1-D12A" or
            state.get("scientific_commit") != preparations.get("scientific_commit") or
            not isinstance(state.get("scientific_commit"), str) or
            len(state["scientific_commit"]) != 40):
        raise ValueError("S1-D12A controller/preparation source identity differs")
    rows = {}
    for key, task in state.get("tasks", {}).items():
        port = int(task["port"])
        ops.instance(port)
        if task.get("variant") not in ("I0", "H1", "H2", "BC", "P0", "P1", "teacher"):
            raise ValueError(f"unexpected D12A scientific variant: {key}")
        rows[key] = {"key": key, "port": port, "run_id": task["run_id"],
                     "request_id": task.get("request_id"), "kind": task.get("kind"),
                     "variant": task["variant"], "source_commit": state["scientific_commit"],
                     "preparation_day": None}
    expected_days = {day for _, day in ASSIGNMENT.values()}
    if set(preparations.get("tasks", {})) != expected_days:
        raise ValueError("three official preparation task receipts required")
    for day, task in preparations["tasks"].items():
        port = int(task["port"])
        role, expected_day = ASSIGNMENT[port]
        if day != expected_day or task.get("role") != role or task.get("git_commit") != state["scientific_commit"]:
            raise ValueError("official preparation date/port/source differs")
        key = "prepare_" + day
        rows[key] = {"key": key, "port": port, "run_id": task["run_id"],
                     "request_id": task["request_id"], "kind": "prepare",
                     "variant": "dataset", "source_commit": state["scientific_commit"],
                     "preparation_day": day}
    identities = [(row["port"], row["run_id"]) for row in rows.values()]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate workbench run ownership")
    return {"experiment_id": "S1-D12A", "scientific_commit": state["scientific_commit"],
            "shared_source_days": sorted(expected_days), "tasks": rows}


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


def verify_source_receipt(receipt, directory):
    if receipt.get("experiment_id") != "S1-D12A" or receipt.get("run_id") != "shared_sources":
        raise ValueError("wrong shared source receipt")
    members = {}
    for volume in receipt.get("volumes", []):
        verify_volume(volume_path(directory, volume), volume)
        for name, record in volume["members"].items():
            if name in members:
                raise ValueError("shared source member duplicated")
            members[name] = record
    days = {day for _, day in ASSIGNMENT.values()}
    raw = {name.split("/")[1] for name in members
           if name.startswith("shared_sources/") and name.endswith(".tar") and len(name.split("/")) == 3}
    if raw != days or len(raw) != 3:
        raise ValueError("shared source bundle lacks exactly three pinned raw days")
    excluded = receipt.get("excluded_recovery_cache")
    if excluded:
        proof_member = "shared_sources/2024-02-07/recovery/source_recovery_cache_audit.json"
        manifest_member = "shared_sources/2024-02-07/recovery/chunks_manifest.json"
        raw_member = next((name for name in members if name.startswith(
            "shared_sources/2024-02-07/") and name.endswith(".tar")), None)
        if (excluded.get("source_bundle_member") != raw_member or
                excluded.get("canonical_sha256") != members.get(raw_member, {}).get("sha256") or
                excluded.get("proof_sha256") != members.get(proof_member, {}).get("sha256") or
                excluded.get("chunk_manifest_sha256") != members.get(manifest_member, {}).get("sha256") or
                len(excluded.get("files", [])) != 401 or
                len({item.get("path") for item in excluded["files"]}) != 401 or
                any(item.get("canonical_offset", -1) < 0 or
                    item.get("canonical_offset", 0) + item.get("bytes", 0) >
                    members.get(raw_member, {}).get("bytes", 0)
                    for item in excluded["files"])):
            raise ValueError("excluded recovery cache does not close against canonical source")
        volume = next((item for item in receipt["volumes"]
                       if proof_member in item["members"]), None)
        if volume is None:
            raise ValueError("excluded recovery cache proof ZIP missing")
        with zipfile.ZipFile(volume_path(directory, volume)) as bundle:
            proof = json.loads(bundle.read(proof_member))
        expected = [{"path": row["path"], "bytes": row["bytes"],
                     "sha256": row["sha256"], "canonical_offset": row["canonical_offset"]}
                    for row in proof.get("cache_files", [])]
        if (proof.get("all_reconstructable") is not True or
                proof.get("canonical_tar", {}).get("sha256") != excluded["canonical_sha256"] or
                expected != excluded["files"]):
            raise ValueError("excluded recovery cache receipt differs from archived proof")
    return members


def verify_receipt(receipt, directory, source_receipt=None, source_directory=None):
    if receipt.get("experiment_id") != "S1-D12A" or not receipt.get("volumes"):
        raise ValueError("wrong or empty D12A archive receipt")
    all_members = {}
    for volume in receipt["volumes"]:
        path = volume_path(directory, volume)
        verify_volume(path, volume)
        for name, record in volume["members"].items():
            if name in all_members:
                raise ValueError("member duplicated across volumes")
            all_members[name] = record
    required = {f"{receipt['port']}/{receipt['run_id']}/{name}"
                for name in ("job.json", "config.json", "status.json")}
    if not required.issubset(all_members):
        raise ValueError("workbench job/config/status evidence missing")
    refs = receipt.get("source_refs", [])
    if refs:
        if source_receipt is None or source_directory is None:
            raise ValueError("shared raw source receipt required for closure")
        source_members = verify_source_receipt(source_receipt, source_directory)
        for ref in refs:
            member = ref.get("source_bundle_member")
            if (member not in source_members or source_members[member].get("sha256") != ref.get("sha256") or
                    source_members[member].get("bytes") != ref.get("bytes")):
                raise ValueError("shared raw source reference does not close")
            pin = "/".join(member.split("/")[:2]) + "/source_preflight.json"
            if source_members.get(pin, {}).get("sha256") != ref.get("source_preflight_sha256"):
                raise ValueError("shared source preflight reference does not close")
    return {"run_id": receipt["run_id"], "members": len(all_members),
            "source_refs": len(refs), "closed": True}


REMOTE_BUILD = r'''
import hashlib,json,os,pathlib,shutil,stat,sys,zipfile
request=json.loads(REQUEST_JSON)
port=request['port'];run_id=request['run_id']
base=pathlib.Path(request['base']).resolve();archive=pathlib.Path(request['archive']).resolve()
jobdir=archive/'outputs/workbench/jobs'/run_id
job=json.loads((jobdir/'job.json').read_text(encoding='utf-8'))
status=json.loads((jobdir/'status.json').read_text(encoding='utf-8'))
launch=json.loads((jobdir/'launch.json').read_text(encoding='utf-8'))
if (job.get('run_id')!=run_id or job.get('request_id')!=request['request_id'] or
 job.get('experiment_id')!='S1-D12A' or job.get('git_commit')!=request['source_commit'] or
 job.get('archive_root')!=str(archive) or status.get('run_id')!=run_id or
 job.get('variant')!=request['variant'] or status.get('stage') not in ('completed','stopped') or
 (status.get('stage')=='stopped' and not (request['variant'] in ('P0','P1') and status.get('scientific_stop') is True))):
 raise RuntimeError('D12A identity, frozen source or terminal state differs')
for pid in {launch.get('pid'),status.get('pid')}:
 if not isinstance(pid,int) or pid<=0:continue
 proc=pathlib.Path('/proc')/str(pid)
 if proc.exists():
  if not isinstance(launch.get('process_start_time'),(int,float)):
   raise RuntimeError('live PID has no trustworthy launch birth record')
  raw=(proc/'stat').read_text();parts=raw[raw.rfind(')')+2:].split();ticks=int(parts[19])
  boot=next(int(x.split()[1]) for x in pathlib.Path('/proc/stat').read_text().splitlines() if x.startswith('btime '))
  born=boot+ticks/os.sysconf('SC_CLK_TCK')
  if abs(born-launch['process_start_time'])<.02:
   raise RuntimeError('same-birth worker still running')

dest=pathlib.Path(request['destination']);dest.mkdir(parents=True,exist_ok=True)
lock=dest/'.lock'
lock.mkdir()
try:
 def sha(path):
  h=hashlib.sha256()
  with path.open('rb') as stream:
   for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
  return h.hexdigest()
 saved=dest/'receipt.json'
 if saved.exists():
  prior=json.loads(saved.read_text())
  if (prior.get('run_id')!=run_id or prior.get('source_commit')!=request['source_commit'] or
   any(not pathlib.Path(v['remote_path']).is_file() or
       sha(pathlib.Path(v['remote_path']))!=v['sha256'] for v in prior.get('volumes',[]))):
   raise RuntimeError('existing remote archive receipt differs')
  print(json.dumps(prior));sys.exit(0)
 def safe(path):
  path=pathlib.Path(path)
  if path.is_symlink():raise RuntimeError('symlink refused: '+str(path))
  resolved=path.resolve(strict=True)
  if not resolved.is_relative_to(base):raise RuntimeError('source escaped D12A group')
  return resolved
 items={};source_refs=[]
 def add(member,path,expected=None):
  if (member.startswith('/') or '\\' in member or
   any(x in ('','.','..') for x in member.split('/')) or member in items):
   raise RuntimeError('unsafe or duplicate ZIP member')
  path=safe(path)
  if not path.is_file() or len(items)>=request['max_files']:
   raise RuntimeError('source member absent or file bound exceeded')
  actual=sha(path);size=path.stat().st_size
  if expected is not None and actual!=expected:raise RuntimeError('frozen source SHA differs')
  items[member]={'path':str(path),'sha256':actual,'bytes':size}
 def tree(prefix,root,skip=None):
  root=safe(root)
  for file in sorted(root.rglob('*')):
   if file.is_symlink() and file!=skip:raise RuntimeError('unexpected symlink')
   if file.is_file() and file!=skip and '__pycache__' not in file.parts and file.suffix!='.pyc':
    add(prefix+'/'+file.relative_to(root).as_posix(),file)
 prefix=str(port)+'/'+run_id
 skip=None
 if request['preparation_day']:
  prep=job.get('dataset_preparation') or {}
  if prep.get('date_utc')!=request['preparation_day']:
   raise RuntimeError('preparation day differs')
  pinfile=safe(prep['files']['source_preflight.json']['path'])
  if sha(pinfile)!=prep['files']['source_preflight.json']['sha256']:
   raise RuntimeError('source preflight metadata SHA differs')
  pin=json.loads(pinfile.read_text())
  raw=base/'source'/prep['role']/prep['date_utc']/prep['archive_name']
  skip=jobdir/'preparation/raw'/prep['archive_name']
  if not raw.is_file() or raw.is_symlink() or not skip.exists():
   raise RuntimeError('shared raw source/link absent')
  rawstat=raw.stat();linkstat=skip.stat()
  if skip.is_symlink():
   if skip.resolve()!=raw.resolve():raise RuntimeError('raw symlink differs')
  elif (rawstat.st_dev,rawstat.st_ino)!=(linkstat.st_dev,linkstat.st_ino):
   raise RuntimeError('raw hardlink differs')
  artifacts=json.loads((jobdir/'artifacts.json').read_text())
  rawsha=sha(raw)
  if (pin.get('archive_name')!=prep['archive_name'] or
   pin.get('expected_bytes')!=prep['expected_bytes'] or
   rawstat.st_size!=prep['expected_bytes'] or rawsha!=sha(skip) or
   rawsha!=artifacts.get('raw_archive_sha256')):
   raise RuntimeError('raw shared archive SHA/bytes differ')
  source_refs.append({'source_bundle_member':'shared_sources/'+prep['date_utc']+'/'+prep['archive_name'],
   'sha256':rawsha,'bytes':rawstat.st_size,
   'source_preflight_sha256':prep['files']['source_preflight.json']['sha256']})
 tree(prefix,jobdir,skip)
 config=json.loads((jobdir/'config.json').read_text())
 for role,record in (config.get('datasets') or {}).items():
  file=safe(record['path'])
  if sha(file)!=record['sha256']:raise RuntimeError('frozen dataset SHA differs')
  if not file.is_relative_to(jobdir):add(prefix+'/scene_inputs/'+role+'/inputs.npz',file,record['sha256'])
 source=safe(job['source_root']);manifest=json.loads((source/'source_manifest.json').read_text())
 if manifest.get('git_commit')!=job['git_commit']:raise RuntimeError('source manifest commit differs')
 for relative,expected in manifest.get('files',{}).items():
  if sha(safe(source/relative))!=expected:raise RuntimeError('source manifest member differs')
 tree(prefix+'/source',source)
 model_items={}
 for name in ('model_dir','model_path','bc_model_dir'):
  value=job.get(name)
  if not value:continue
  if not pathlib.Path(value).exists():
   if name=='model_dir' and request['variant'] in ('I0','H1','H2','teacher','dataset'):
    continue
   raise RuntimeError('required model source absent: '+name)
  path=safe(value)
  files=sorted(path.rglob('*')) if path.is_dir() else [path]
  for file in files:
   if file.is_file():
    member='models/'+str(port)+'/'+run_id+'/'+name+'/'+file.name
    model_items[member]={'path':str(file),'sha256':sha(file),'bytes':file.stat().st_size}
 if sum(x['bytes'] for x in items.values())>request['max_bytes']:
  raise RuntimeError('job archive byte bound exceeded')
 need=sum(x['bytes'] for x in items.values())+sum(x['bytes'] for x in model_items.values())
 if shutil.disk_usage(dest).free < request['reserve']+need+(len(items)+len(model_items))*2048:
  raise RuntimeError('remote free space below 8GiB reserve plus actual task export')
 def zip_group(name,entries):
  path=dest/name;temporary=dest/(name+'.tmp')
  if path.exists() or temporary.exists():raise RuntimeError('archive already exists; verify before retry')
  with zipfile.ZipFile(temporary,'w',allowZip64=True,compression=zipfile.ZIP_DEFLATED,compresslevel=6) as out:
   for member,record in entries:
    out.write(record['path'],member)
  temporary.replace(path)
  return {'file':name,'remote_path':str(path),'sha256':sha(path),'bytes':path.stat().st_size,
   'members':{member:{'sha256':rec['sha256'],'bytes':rec['bytes']} for member,rec in entries}}
 volumes=[];group=[];size=0
 for member,record in sorted(items.items()):
  if group and size+record['bytes']>request['part_bytes']:
   volumes.append(zip_group('job_%03d.zip'%len(volumes),group));group=[];size=0
  group.append((member,record));size+=record['bytes']
 if group:volumes.append(zip_group('job_%03d.zip'%len(volumes),group))
 if model_items:volumes.append(zip_group('models.zip',sorted(model_items.items())))
 receipt={'experiment_id':'S1-D12A','port':port,'run_id':run_id,
  'source_commit':request['source_commit'],'status':status['stage'],
  'source_refs':source_refs,'volumes':volumes}
 (dest/'receipt.json').write_text(json.dumps(receipt))
 print(json.dumps(receipt))
finally:
 lock.rmdir()
'''


REMOTE_SOURCE_BUILD = r'''
import hashlib,json,os,pathlib,shutil,sys,zipfile
request=json.loads(REQUEST_JSON)
base=pathlib.Path(request['base']).resolve();dest=pathlib.Path(request['destination'])
dest.mkdir(parents=True,exist_ok=True);lock=dest/'.lock';lock.mkdir()
owner={'pid':os.getpid(),'run_id':'shared_sources','source_commit':request['source_commit']}
proc=pathlib.Path('/proc')/str(os.getpid())/'stat'
if proc.is_file():
 raw=proc.read_text();owner['start_ticks']=int(raw[raw.rfind(')')+2:].split()[19])
(lock/'owner.json').write_text(json.dumps(owner))
try:
 def sha(path):
  h=hashlib.sha256()
  with path.open('rb') as stream:
   for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
  return h.hexdigest()
 saved=dest/'receipt.json'
 if saved.exists():
  prior=json.loads(saved.read_text())
  if (prior.get('run_id')!='shared_sources' or prior.get('source_commit')!=request['source_commit'] or
    len(prior.get('volumes',[]))!=3 or
   any(not pathlib.Path(v['remote_path']).is_file() or sha(pathlib.Path(v['remote_path']))!=v['sha256']
       for v in prior.get('volumes',[]))):raise RuntimeError('existing shared source receipt differs')
  print(json.dumps(prior));sys.exit(0)
 items={};excluded_recovery_cache=None
 for entry in request['preparations']:
  port,run_id,day,role=entry['port'],entry['run_id'],entry['day'],entry['role']
  root=base/str(port)/'archive'/'outputs/workbench/jobs'/run_id
  job=json.loads((root/'job.json').read_text());status=json.loads((root/'status.json').read_text())
  launch=json.loads((root/'launch.json').read_text())
  if (job.get('experiment_id')!='S1-D12A' or job.get('run_id')!=run_id or
   job.get('git_commit')!=request['source_commit'] or status.get('stage')!='completed' or
   job.get('source_date')!=day):raise RuntimeError('source preparation identity incomplete')
  for pid in {status.get('pid'),launch.get('pid')}:
   if not isinstance(pid,int) or pid<=0:continue
   proc=pathlib.Path('/proc')/str(pid)
   if proc.exists():
    start=launch.get('process_start_time')
    if not isinstance(start,(int,float)):raise RuntimeError('source worker birth unknown')
    raw=(proc/'stat').read_text();ticks=int(raw[raw.rfind(')')+2:].split()[19])
    boot=next(int(x.split()[1]) for x in pathlib.Path('/proc/stat').read_text().splitlines() if x.startswith('btime '))
    if abs(boot+ticks/os.sysconf('SC_CLK_TCK')-start)<.02:
     raise RuntimeError('source preparation worker still running')
  prep=job['dataset_preparation'];source=base/'source'/role/day
  pinfile=source/'source_preflight.json';pin=json.loads(pinfile.read_text())
  archive=source/prep['archive_name'];artifacts=json.loads((root/'artifacts.json').read_text())
  raw_sha=sha(archive)
  if (pin['archive_name']!=archive.name or pin['expected_bytes']!=archive.stat().st_size or
   sha(pinfile)!=prep['files']['source_preflight.json']['sha256'] or
   raw_sha!=artifacts['raw_archive_sha256']):
   raise RuntimeError('shared official raw archive differs from frozen preparation')
  if day=='2024-02-07' and role=='validation' and ((source/'chunks').exists() or (source/(archive.name+'.part')).exists()):
   proof_path=pathlib.Path(request['recovery_proof_path'])
   if (proof_path.is_symlink() or not proof_path.is_file() or
    not proof_path.resolve().is_relative_to(base/'20473'/'ops') or
    sha(proof_path)!=request['recovery_proof_sha256']):
    raise RuntimeError('Feb07 recovery proof source/SHA differs')
   proof=json.loads(proof_path.read_text())
   canonical=proof.get('canonical_tar') or {}
   if (proof.get('experiment_id')!='S1-D12A' or proof.get('role')!='validation' or
    proof.get('date_utc')!=day or proof.get('all_reconstructable') is not True or
    proof.get('source_preflight_sha256')!=sha(pinfile) or
    pin.get('asset_id')!=150440724 or
    canonical.get('path')!=str(archive) or canonical.get('bytes')!=archive.stat().st_size or
    canonical.get('sha256')!=raw_sha or canonical.get('expected_sha256')!=raw_sha):
    raise RuntimeError('Feb07 recovery proof not bound to frozen canonical tar')
   chunkdir=source/'chunks';manifest_file=chunkdir/'manifest.json'
   metadata=proof.get('metadata_files') or []
   if (chunkdir.is_symlink() or not chunkdir.is_dir() or len(metadata)!=1 or
    metadata[0].get('path')!=str(manifest_file) or
    metadata[0].get('sha256')!=sha(manifest_file) or
    metadata[0].get('record')!=json.loads(manifest_file.read_text())):
    raise RuntimeError('Feb07 recovery chunk manifest differs')
   manifest=metadata[0]['record'];count=manifest.get('chunk_count')
   if (count!=400 or manifest.get('expected_bytes')!=archive.stat().st_size or
    not isinstance(manifest.get('chunk_size'),int) or manifest['chunk_size']<=0 or
    len(manifest.get('chunks',[]))!=count or
    proof.get('checked_bytes')!=archive.stat().st_size):
    raise RuntimeError('Feb07 recovery chunk geometry differs')
   partial=source/(archive.name+'.part')
   expected_paths={str(chunkdir/('chunk_%04d.bin'%i)) for i in range(count)}|{str(partial)}
   cache=proof.get('cache_files') or []
   actual_chunks={str(path) for path in chunkdir.iterdir()}
   if (len(cache)!=count+1 or {row.get('path') for row in cache}!=expected_paths or
    actual_chunks!=(expected_paths-{str(partial)})|{str(manifest_file)} or
    proof.get('unexpected') or proof.get('chunk_count')!=count or
    sum(row.get('bytes',0) for row in cache if row.get('path')!=str(partial))!=archive.stat().st_size):
    raise RuntimeError('Feb07 recovery cache inventory differs')
   excluded=[]
   for row in cache:
    path=pathlib.Path(row['path'])
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(source):
     raise RuntimeError('Feb07 recovery cache path unsafe')
    if path==partial:index=None;offset=0
    else:
     index=int(path.stem.split('_')[1]);offset=index*manifest['chunk_size']
     if row.get('sha256')!=manifest['chunks'][index]['sha256']:
      raise RuntimeError('Feb07 recovery chunk manifest SHA differs')
    if (row.get('canonical_offset')!=offset or row.get('bytes')!=path.stat().st_size or
     row.get('sha256')!=sha(path) or row.get('matches_canonical_bytes') is not True or
     row.get('receipt_sha256_matches') is not True or
     row.get('receipt_bytes_matches') is not True or
     offset+path.stat().st_size>archive.stat().st_size):
     raise RuntimeError('Feb07 recovery cache receipt differs')
    with path.open('rb') as cached,archive.open('rb') as canonical_stream:
     canonical_stream.seek(offset)
     while block:=cached.read(8<<20):
      if canonical_stream.read(len(block))!=block:
       raise RuntimeError('Feb07 recovery cache differs from canonical bytes')
    excluded.append({'path':str(path),'bytes':row['bytes'],'sha256':row['sha256'],
     'canonical_offset':offset})
   excluded_recovery_cache={'source_bundle_member':'shared_sources/'+day+'/'+archive.name,
    'canonical_sha256':raw_sha,'proof_sha256':request['recovery_proof_sha256'],
    'chunk_manifest_sha256':metadata[0]['sha256'],'files':excluded}
   for name,path in (('source_recovery_cache_audit.json',proof_path),
                     ('chunks_manifest.json',manifest_file)):
    member='shared_sources/'+day+'/recovery/'+name
    items[member]={'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}
  allowed={prep['archive_name'],'source_preflight.json','selection_contract.json',
   'release_api.json','PREFERRED_RELEASES.txt','LICENSE-ODbL-official.txt'}
  for file in sorted(source.iterdir()):
   if day=='2024-02-07' and role=='validation' and excluded_recovery_cache and file.name in ('chunks',archive.name+'.part'):
    continue
   if file.name not in allowed:raise RuntimeError('unexpected shared source file')
   if file.is_symlink() or not file.is_file():raise RuntimeError('shared source member unsafe')
   member='shared_sources/'+day+'/'+file.name
   if member in items:raise RuntimeError('shared source duplicate')
   items[member]={'path':str(file),'bytes':file.stat().st_size,'sha256':sha(file)}
 total=sum(x['bytes'] for x in items.values())
 if shutil.disk_usage(dest).free < request['reserve']+total+len(items)*2048:
  raise RuntimeError('remote free space below source bundle plus 8GiB reserve')
 volumes=[]
 for day in sorted({entry['day'] for entry in request['preparations']}):
  day_items={name:row for name,row in items.items() if name.startswith('shared_sources/'+day+'/')}
  if not any(name.endswith('.tar') for name in day_items):raise RuntimeError('source day lacks raw archive')
  path=dest/('shared_sources_'+day+'.zip');temporary=dest/(path.name+'.tmp')
  if path.exists() or temporary.exists():raise RuntimeError('shared source volume already exists')
  with zipfile.ZipFile(temporary,'w',allowZip64=True,compression=zipfile.ZIP_DEFLATED,compresslevel=6) as out:
   for member,row in sorted(day_items.items()):out.write(row['path'],member)
  temporary.replace(path)
  volumes.append({'file':path.name,'remote_path':str(path),'bytes':path.stat().st_size,
   'sha256':sha(path),'members':{name:{'bytes':row['bytes'],'sha256':row['sha256']}
    for name,row in day_items.items()}})
 receipt={'experiment_id':'S1-D12A','run_id':'shared_sources','port':20418,
   'builder_ssh_port':request.get('builder_ssh_port',20418),
    'source_commit':request['source_commit'],'volumes':volumes,
    'excluded_recovery_cache':excluded_recovery_cache}
 (dest/'receipt.json').write_text(json.dumps(receipt))
 print(json.dumps(receipt))
finally:
 (lock/'owner.json').unlink(missing_ok=True)
 lock.rmdir()
'''


def _disk_free(path):
    path = Path(path)
    while not path.exists():
        if path == path.parent:
            raise FileNotFoundError(path)
        path = path.parent
    return shutil.disk_usage(path).free


def plan_canonical(port, receipt, free_by_root=None):
    """Choose one existing or new D/C location per ZIP, preserving 8 GiB on each."""
    roots = {"local": LOCAL / str(port) / receipt["run_id"],
             "backup": BACKUP / str(port) / receipt["run_id"]}
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
        eligible = [key for key in roots if free[key] >= volume["bytes"]]
        if not eligible:
            raise RuntimeError("no canonical drive has ZIP bytes plus 8GiB reserve")
        # Put each largest volume on the tighter fitting drive so another
        # large volume can still use the drive with more remaining space.
        key = min(eligible, key=lambda item: (free[item], item != "local"))
        target = roots[key] / volume["file"]
        if target.with_suffix(target.suffix + ".part").exists():
            raise FileExistsError("interrupted local ZIP requires manual inspection")
        choices[volume["file"]] = target
        free[key] -= volume["bytes"]
    return {name: str(path.resolve()) for name, path in choices.items()}


def _download(port, receipt):
    directory = LOCAL / str(port) / receipt["run_id"]
    choices = plan_canonical(port, receipt)
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
                    if partial.exists():
                        raise FileExistsError("interrupted local ZIP requires manual inspection")
                    h = hashlib.sha256()
                    size = 0
                    with sftp.open(volume["remote_path"], "rb") as remote, partial.open("wb") as local:
                        while block := remote.read(8 << 20):
                            h.update(block)
                            size += len(block)
                            local.write(block)
                    if size != volume["bytes"] or h.hexdigest() != volume["sha256"]:
                        raise ValueError("remote ZIP stream SHA/size differs")
                    partial.replace(target)
                    verify_volume(target, volume)
                volume["canonical_path"] = str(target)
    finally:
        client.close()
    return directory


def one_task(entry):
    port = entry["port"]
    existing = LOCAL / str(port) / entry["run_id"] / "receipt.json"
    if existing.is_file():
        receipt = json.loads(existing.read_text(encoding="utf-8"))
        if (receipt.get("run_id") != entry["run_id"] or
                receipt.get("source_commit") != entry["source_commit"]):
            raise ValueError("existing archive receipt identity differs")
        for volume in receipt.get("volumes", []):
            verify_volume(volume_path(existing.parent, volume), volume)
        return receipt
    detail = ops.api(port, f"/api/runs/{entry['run_id']}")
    if (detail.get("job", {}).get("run_id") != entry["run_id"] or
            detail.get("status", {}).get("stage") not in ("completed", "stopped")):
        raise ValueError("workbench task is not terminal")
    instance = ops.instance(port)
    request = {**entry, "base": ops.BASE, "archive": instance["archive"],
               "destination": instance["ops"] + "/archive_d12a/" + entry["run_id"],
               "reserve": RESERVE, "part_bytes": PART_BYTES,
               "max_files": MAX_FILES, "max_bytes": MAX_JOB_BYTES}
    code = "REQUEST_JSON=" + repr(json.dumps(request)) + "\n" + REMOTE_BUILD
    result = ops.remote_python(code, port, timeout=1800)
    if result["exit_code"]:
        raise RuntimeError(result["stderr"][-2000:])
    receipt = json.loads(result["stdout"].splitlines()[-1])
    directory = _download(port, receipt)
    path = directory / "receipt.json"
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return receipt


def one_source(planned):
    entries = []
    for port, (role, day) in ASSIGNMENT.items():
        task = planned["tasks"]["prepare_" + day]
        detail = ops.api(port, f"/api/runs/{task['run_id']}")
        if detail.get("status", {}).get("stage") != "completed":
            raise ValueError("all three official preparation workers must complete first")
        entries.append({"port": port, "role": role, "day": day, "run_id": task["run_id"]})
    existing = LOCAL / "20418" / "shared_sources" / "receipt.json"
    if existing.is_file():
        receipt = json.loads(existing.read_text(encoding="utf-8"))
        if receipt.get("source_commit") != planned["scientific_commit"]:
            raise ValueError("existing source archive receipt commit differs")
        verify_source_receipt(receipt, existing.parent)
        return receipt
    proof = json.loads(RECOVERY_PROOF.read_text(encoding="utf-8"))
    if (proof.get("experiment_id") != "S1-D12A" or
            proof.get("date_utc") != "2024-02-07" or
            proof.get("role") != "validation" or
            proof.get("all_reconstructable") is not True or
            len(proof.get("cache_files", [])) != 401):
        raise ValueError("Feb07 recovery cache proof incomplete")
    proof_sha = digest(RECOVERY_PROOF)
    remote_proof = ops.instance(20473)["ops"] + "/source_recovery_cache_audit.json"
    uploaded = ops.upload(RECOVERY_PROOF, remote_proof, 20473)
    if (uploaded.get("exit_code") != 0 or uploaded.get("local_sha256") != proof_sha or
            uploaded.get("remote") != proof_sha or uploaded.get("path") != remote_proof):
        raise RuntimeError("Feb07 recovery cache proof upload SHA differs")
    request = {"base": ops.BASE,
               "destination": ops.instance(20418)["ops"] + "/archive_d12a/shared_sources",
               "source_commit": planned["scientific_commit"],
               "preparations": entries, "reserve": RESERVE, "builder_ssh_port": 20473,
               "recovery_proof_path": remote_proof, "recovery_proof_sha256": proof_sha}
    code = "REQUEST_JSON=" + repr(json.dumps(request)) + "\n" + REMOTE_SOURCE_BUILD
    result = ops.remote_python(code, 20473, timeout=1800)
    if result["exit_code"]:
        raise RuntimeError(result["stderr"][-2000:])
    receipt = json.loads(result["stdout"].splitlines()[-1])
    directory = _download(20418, receipt)
    verify_source_receipt(receipt, directory)
    path = directory / "receipt.json"
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "one-task", "one-source", "verify", "verify-source"))
    parser.add_argument("--key")
    parser.add_argument("--state", type=Path, default=ops.LOCAL / "controller" / "state.json")
    parser.add_argument("--preparations", type=Path, default=RECEIPT)
    parser.add_argument("--source-receipt", type=Path)
    args = parser.parse_args()
    state = json.loads(args.state.read_text(encoding="utf-8"))
    preparation = json.loads(args.preparations.read_text(encoding="utf-8"))
    planned = plan(state, preparation)
    if args.action == "plan":
        print(json.dumps(planned, ensure_ascii=False, indent=2))
        return
    if args.action == "one-source":
        print(json.dumps(one_source(planned), ensure_ascii=False, indent=2))
        return
    if args.action == "verify-source":
        directory = LOCAL / "20418" / "shared_sources"
        receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
        result = {"members": len(verify_source_receipt(receipt, directory))}
        print(json.dumps(result))
        return
    if args.key not in planned["tasks"]:
        parser.error("--key must identify one planned D12A task")
    entry = planned["tasks"][args.key]
    if args.action == "one-task":
        result = one_task(entry)
    else:
        directory = LOCAL / str(entry["port"]) / entry["run_id"]
        receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
        source = (json.loads(args.source_receipt.read_text(encoding="utf-8"))
                  if args.source_receipt else None)
        result = verify_receipt(receipt, directory, source,
                                args.source_receipt.parent if args.source_receipt else None)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
