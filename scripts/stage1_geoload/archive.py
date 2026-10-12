"""Stream terminal G16 job/model ZIP64 archives, keeping original assets as references."""
from pathlib import Path
import hashlib
import json
import shutil
from . import ops
from satellite_rl.workbench.storage import validate_id

def export_job(port,run_id,destination):
    info=ops.instance(port);validate_id(run_id)
    directory=info['archive']+'/outputs/workbench/jobs/'+run_id
    target=info['ops']+'/'+run_id+'.zip'
    code=f'''import hashlib,json,pathlib,zipfile
root=pathlib.Path({directory!r});job=json.loads((root/'job.json').read_text());status=json.loads((root/'status.json').read_text())
if job.get('experiment_id')!='S1-G16' or status.get('stage') not in ('completed','failed','stopped','cancelled'):raise RuntimeError('nonterminal or wrong group archive')
target=pathlib.Path({target!r});members={{}}
with zipfile.ZipFile(target.with_suffix('.tmp'), 'w',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as z:
 for prefix,source in [('job',root),('model',pathlib.Path(job['model_dir']))]:
  for p in sorted(source.rglob('*')):
   if not p.is_file() or p.is_symlink():continue
   name=prefix+'/'+p.relative_to(source).as_posix()
   h=hashlib.sha256()
   count=0
   with p.open('rb') as f,z.open(name,'w',force_zip64=True) as writer:
    for b in iter(lambda:f.read(1048576),b''):
     h.update(b);count+=len(b);writer.write(b)
   if count!=p.stat().st_size:raise RuntimeError('archive source changed while streaming')
   members[name]={{'sha256':h.hexdigest(),'bytes':p.stat().st_size}}
 z.writestr('archive_manifest.json',json.dumps({{'group':'S1-G16','run_id':job['run_id'],'scientific_commit':job['scientific_commit'],'members':members}},sort_keys=True))
target.with_suffix('.tmp').replace(target);h=hashlib.sha256()
with target.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
print(json.dumps({{'path':str(target),'sha256':h.hexdigest(),'bytes':target.stat().st_size,'members':len(members)}}))
'''
    result=ops.remote_python(code,port,timeout=600)
    if result['exit_code']:raise RuntimeError(result['stderr'][-1500:])
    receipt=json.loads(result['stdout'].splitlines()[-1])
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(destination.parent).free < receipt['bytes'] + 8*(1<<30):
        raise RuntimeError('canonical archive volume lacks 8GiB reserve; remote ZIP retained')
    temp=destination.with_suffix(destination.suffix+'.part');client=ops.connect(port)
    try:
        sftp=client.open_sftp();h=hashlib.sha256();count=0
        with sftp.open(receipt['path'],'rb') as source,temp.open('wb') as out:
            while True:
                block=source.read(1024*1024)
                if not block:break
                h.update(block);count+=len(block);out.write(block)
        sftp.close()
        if count!=receipt['bytes'] or h.hexdigest()!=receipt['sha256']:raise ValueError('archive transport SHA differs')
        temp.replace(destination)
    finally:client.close()
    receipt.update(local_path=str(destination),verified=True,port=port,run_id=run_id)
    destination.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    return receipt
