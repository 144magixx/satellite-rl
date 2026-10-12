"""Candidate bounded readv G16 archive transport; no live module replacement.

one-export requires explicit per-task ownership confirmation after the caller
excludes legacy exporters. The old exporter has no lock: this module cannot
assert mutual exclusion with a still-enabled legacy controller. Unknown tmp,
packer lock or live worker is refused, never deleted or restarted.
Existing remote bytes require a pinned receipt and archive manifest hash.
No ZIP compression/member changes are made when downloading existing archives.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import zipfile

SCIENCE='495bff976c59c40ddb33de6b5b37b20a894a1347'
RESERVE=8 << 30
WINDOWS=(2 << 20,4 << 20)


def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':')).encode()


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def blocks(remote,total,window=2 << 20,*,offset=0):
    if window not in WINDOWS or offset<0 or offset>total:raise ValueError('invalid bounded readv window/offset')
    while offset<total:
        count=min(window,total-offset)
        result=list(remote.readv([(offset,count)]))
        if len(result)!=1 or len(result[0])!=count:raise ValueError('short/out-of-order readv window')
        yield result[0];offset+=count


def binding(receipt):
    required=('path','bytes','sha256','manifest_sha256','run_id','port','scientific_commit')
    value={key:receipt[key] for key in required}
    if value['port'] not in (20424,20473) or value['scientific_commit']!=SCIENCE or not re.fullmatch('[A-Za-z0-9_-]+',value['run_id']):raise ValueError('wrong archive owner/source')
    if value['bytes']<1 or any(not re.fullmatch('[0-9a-f]{64}',value[key]) for key in ('sha256','manifest_sha256')):raise ValueError('size/fullSHA/manifest pin required')
    return value


def verify_local(path,receipt):
    if Path(path).stat().st_size!=receipt['bytes'] or digest(path)!=receipt['sha256']:raise ValueError('whole ZIP SHA/size differs')
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)):raise ValueError('duplicate ZIP member')
        data=archive.read('archive_manifest.json')
        if hashlib.sha256(data).hexdigest()!=receipt['manifest_sha256']:raise ValueError('archive manifest pin differs')
        manifest=json.loads(data)
        if manifest.get('group')!='S1-G16' or manifest.get('run_id')!=receipt['run_id'] or manifest.get('scientific_commit')!=SCIENCE:raise ValueError('archive manifest identity differs')
        if set(names)!=set(manifest['members'])|{'archive_manifest.json'}:raise ValueError('archive member set differs')


def download(receipt,destination,*,window=2 << 20,transport=None,free_bytes=None):
    """Bounded memory, persistent identity-bound partial, final prefix+tail SHA."""
    identity=binding(receipt)
    if window not in WINDOWS:raise ValueError('window must be 2MiB or 4MiB')
    if transport is None:
        from . import ops as transport
    target=Path(destination)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        verify_local(target,identity)
        return {**receipt,'local_path':str(target.resolve()),'verified':True,'download_reused':True}
    partial=target.with_suffix(target.suffix+'.part');sidecar=target.with_suffix(target.suffix+'.part.binding.json')
    if partial.is_symlink() or sidecar.is_symlink():raise ValueError('linked partial refused')
    offset=partial.stat().st_size if partial.exists() else 0
    if offset>identity['bytes']:raise ValueError('partial exceeds immutable receipt size')
    if partial.exists() and (not sidecar.exists() or read(sidecar)!=identity):raise ValueError('partial not bound to exact remote archive')
    if sidecar.exists() and read(sidecar)!=identity:raise ValueError('partial binding differs')
    free=shutil.disk_usage(target.parent).free if free_bytes is None else free_bytes
    if free<identity['bytes']-offset+RESERVE:raise RuntimeError('download lacks remaining bytes plus 8GiB reserve')
    if not sidecar.exists():sidecar.write_bytes(encoded(identity))
    h=hashlib.sha256()
    if offset:
        with partial.open('rb') as prior:
            for block in iter(lambda:prior.read(1 << 20),b''):h.update(block)
    start=time.perf_counter();count=offset;client=transport.connect(identity['port'])
    try:
        with client.open_sftp() as sftp:
            stat=sftp.stat(identity['path'])
            if stat.st_size!=identity['bytes']:raise ValueError('remote size changed')
            with sftp.open(identity['path'],'rb') as remote, partial.open('ab') as local:
                remote.settimeout(90)
                for block in blocks(remote,identity['bytes'],window,offset=offset):
                    local.write(block);h.update(block);count+=len(block)
    finally:client.close()
    if count!=identity['bytes'] or h.hexdigest()!=identity['sha256']:raise ValueError('prefix+download final fullSHA differs; partial retained')
    verify_local(partial,identity)
    partial.replace(target)
    # Binding stays as a small recovery/provenance record; no data duplicate.
    result={**receipt,'local_path':str(target.resolve()),'verified':True,
        'download_resumed_from_bytes':offset,'download_window_bytes':window,
        'download_seconds':time.perf_counter()-start,'timing_scope':'engineering_SFTP_only_not_algorithm_speed'}
    target.with_suffix('.receipt.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def remote_verify_code(receipt):
    value=binding(receipt)
    return f'''import pathlib,json,hashlib,zipfile
p=pathlib.Path({value['path']!r});h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
if p.stat().st_size!={value['bytes']} or h.hexdigest()!={value['sha256']!r}:raise ValueError('remote volume pin differs')
with zipfile.ZipFile(p) as z:
 raw=z.read('archive_manifest.json');m=json.loads(raw)
 if hashlib.sha256(raw).hexdigest()!={value['manifest_sha256']!r}:raise ValueError('remote manifest pin differs')
 if m.get('run_id')!={value['run_id']!r} or m.get('group')!='S1-G16' or m.get('scientific_commit')!={SCIENCE!r}:raise ValueError('remote owner/source differs')
 names=z.namelist()
 if len(names)!=len(set(names)) or set(names)!=set(m['members'])|{{'archive_manifest.json'}}:raise ValueError('remote member inventory differs')
print(json.dumps({value!r}))
'''


def attest_existing(receipt,port,run_id,*,transport=None):
    """Upgrade an original volume receipt only after full remote SHA verification."""
    if transport is None:
        from . import ops as transport
    transport.instance(port)
    if port not in (20424,20473) or not re.fullmatch('[A-Za-z0-9_-]+',run_id):raise ValueError('wrong existing volume owner')
    if receipt.get('port') not in (None,port) or receipt.get('run_id') not in (None,run_id):raise ValueError('old receipt owner differs')
    if not re.fullmatch('[0-9a-f]{64}',receipt.get('sha256','')) or not isinstance(receipt.get('bytes'),int) or receipt['bytes']<1:raise ValueError('existing full volume SHA/size required')
    value={**receipt,'port':port,'run_id':run_id,'scientific_commit':SCIENCE}
    code=f'''import pathlib,json,hashlib,zipfile
p=pathlib.Path({receipt['path']!r});h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
if p.stat().st_size!={receipt['bytes']} or h.hexdigest()!={receipt['sha256']!r}:raise ValueError('existing volume full SHA/size differs')
with zipfile.ZipFile(p) as z:
 raw=z.read('archive_manifest.json');m=json.loads(raw);names=z.namelist()
 if m.get('group')!='S1-G16' or m.get('run_id')!={run_id!r} or m.get('scientific_commit')!={SCIENCE!r}:raise ValueError('existing actual manifest identity differs')
 if len(names)!=len(set(names)) or set(names)!=set(m['members'])|{{'archive_manifest.json'}}:raise ValueError('existing archive inventory differs')
 job=json.loads(z.read('job/job.json'));status=json.loads(z.read('job/status.json'))
 if job.get('run_id')!={run_id!r} or job.get('scientific_commit')!={SCIENCE!r} or status.get('stage') not in ('completed','failed','stopped','cancelled'):raise ValueError('existing archived task/source/nonterminal differs')
value=json.loads({json.dumps(value)!r});value['manifest_sha256']=hashlib.sha256(raw).hexdigest()
print(json.dumps(value))
'''
    result=transport.remote_python(code,port,timeout=600)
    if result['exit_code']:raise RuntimeError(result['stderr'][-1000:])
    verified=json.loads(result['stdout'].splitlines()[-1]);binding(verified)
    return verified


def export_code(port,run_id,info):
    """No automatic takeover: retained tmp or existing unpinned ZIP is blocked."""
    directory=info['archive']+'/outputs/workbench/jobs/'+run_id
    target=info['ops']+'/archive_transport/'+run_id+'.zip'
    legacy=info['ops']+'/'+run_id+'.zip'
    return f'''import os,pathlib,json,hashlib,zipfile,fcntl
root=pathlib.Path({directory!r});target=pathlib.Path({target!r});legacy=pathlib.Path({legacy!r})
job=json.loads((root/'job.json').read_text());status=json.loads((root/'status.json').read_text())
if job.get('experiment_id')!='S1-G16' or job.get('scientific_commit')!={SCIENCE!r} or job.get('run_id')!={run_id!r} or status.get('stage') not in ('completed','failed','stopped','cancelled'):raise RuntimeError('wrong/nonterminal worker')
pid=status.get('pid')
if not isinstance(pid,int) or pathlib.Path('/proc/'+str(pid)).exists():raise RuntimeError('actual worker exit unknown/not exited; refuse packing')
target.parent.mkdir(parents=True,exist_ok=True)
lock=(target.parent/({run_id!r}+'.packer.lock')).open('a+b')
fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
worker=(root/'worker.lock').open('a+b');fcntl.flock(worker.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
if target.exists() or target.with_suffix('.tmp').exists() or legacy.exists() or legacy.with_suffix('.tmp').exists():raise RuntimeError('existing/unknown archive requires pinned receipt; no duplicate packer')
# Refuse another process with an open archive/job/model source handle.
for proc in pathlib.Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try:
  for fd in (proc/'fd').iterdir():
   try:linked=os.readlink(fd)
   except OSError:continue
   if linked.startswith(str(root)+'/') or linked.startswith(job['model_dir'].rstrip('/')+'/') or linked in (str(target),str(legacy),str(target.with_suffix('.tmp')),str(legacy.with_suffix('.tmp'))):raise RuntimeError('unknown packer/source reader active; no takeover')
 except (FileNotFoundError,PermissionError):pass
members={{}};temporary=target.with_suffix('.tmp')
with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as z:
 for prefix,source in [('job',root),('model',pathlib.Path(job['model_dir']))]:
  for p in sorted(source.rglob('*')):
   if not p.is_file() or p.is_symlink():continue
   name=prefix+'/'+p.relative_to(source).as_posix();h=hashlib.sha256();count=0
   with p.open('rb') as f,z.open(name,'w',force_zip64=True) as writer:
    for b in iter(lambda:f.read(1048576),b''):h.update(b);count+=len(b);writer.write(b)
   if count!=p.stat().st_size:raise RuntimeError('source changed while packing')
   members[name]={{'sha256':h.hexdigest(),'bytes':count}}
 raw=json.dumps({{'group':'S1-G16','run_id':job['run_id'],'scientific_commit':job['scientific_commit'],'members':members}},sort_keys=True).encode()
 z.writestr('archive_manifest.json',raw)
temporary.replace(target);h=hashlib.sha256()
with target.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
receipt={{'path':str(target),'sha256':h.hexdigest(),'bytes':target.stat().st_size,'members':len(members),'manifest_sha256':hashlib.sha256(raw).hexdigest(),'port':{port},'run_id':{run_id!r},'scientific_commit':{SCIENCE!r}}}
target.with_suffix('.remote.receipt.json').write_text(json.dumps(receipt,sort_keys=True))
print(json.dumps(receipt))
'''


def one_export(port,run_id,destination,*,existing_receipt=None,ownership_confirmed=False,window=2 << 20,transport=None):
    if transport is None:
        from . import ops as transport
    info=transport.instance(port)
    if port not in (20424,20473) or not re.fullmatch('[A-Za-z0-9_-]+',run_id):raise ValueError('unauthorized endpoint/task')
    if not ownership_confirmed:raise ValueError('caller must explicitly exclude legacy exporter for this task')
    if existing_receipt:
        receipt=read(existing_receipt)
        if 'manifest_sha256' not in receipt:receipt=attest_existing(receipt,port,run_id,transport=transport)
        if receipt['port']!=port or receipt['run_id']!=run_id:raise ValueError('existing receipt owner differs')
        code=remote_verify_code(receipt)
    else:code=export_code(port,run_id,info)
    result=transport.remote_python(code,port,timeout=600)
    if result['exit_code']:raise RuntimeError(result['stderr'][-1000:])
    receipt=json.loads(result['stdout'].splitlines()[-1]);binding(receipt)
    # Durable remote pin before download; interruption resumes exactly this ZIP.
    pin=Path(destination).with_suffix('.remote.receipt.json');pin.parent.mkdir(parents=True,exist_ok=True)
    if pin.exists() and binding(read(pin))!=binding(receipt):raise ValueError('local remote receipt differs')
    if not pin.exists():pin.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    return download(receipt,destination,window=window,transport=transport)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('one-export','download'))
    parser.add_argument('--port',type=int,choices=(20424,20473));parser.add_argument('--run-id')
    parser.add_argument('--receipt');parser.add_argument('--output',required=True)
    parser.add_argument('--window-mib',type=int,choices=(2,4),default=2)
    parser.add_argument('--exclusive-task-owner',action='store_true')
    args=parser.parse_args()
    if args.action=='download':
        if not args.receipt:parser.error('--receipt required')
        value=download(read(args.receipt),args.output,window=args.window_mib << 20)
    else:
        if args.port is None or not args.run_id:parser.error('--port and --run-id required')
        value=one_export(args.port,args.run_id,args.output,existing_receipt=args.receipt,
            ownership_confirmed=args.exclusive_task_owner,window=args.window_mib << 20)
    print(json.dumps(value,ensure_ascii=False))


if __name__=='__main__':main()
