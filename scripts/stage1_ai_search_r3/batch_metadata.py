"""One immutable ZIP transfer, cache by exact SHA, publish original members."""
import hashlib,json,zipfile
from pathlib import Path,PurePosixPath
from . import ops


def validate_members(files):
    if len({r['remote'] for r in files})!=len(files):raise ValueError('duplicate metadata destination')
    for r in files:
        path=PurePosixPath(r['remote'])
        if not path.is_relative_to(PurePosixPath(ops.BASE+'/source')) or '..' in path.parts:raise ValueError('metadata destination outside isolated R3 source')
        if Path(r['local']).suffix!='.json':raise ValueError('small JSON metadata only')
        if Path(r['local']).stat().st_size>64*(1<<20):raise ValueError('metadata member too large')
        if hashlib.sha256(Path(r['local']).read_bytes()).hexdigest()!=r['sha256']:raise ValueError('metadata local SHA differs')


def upload(files,folder,port=20418):
    validate_members(files);folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    descriptor=[{'remote':r['remote'],'sha256':r['sha256'],'member':r['sha256']+'.json'} for r in files]
    key=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest();bundle=folder/(key+'.zip')
    if not bundle.exists():
        unique={r['sha256']:r for r in files}
        with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for sha,r in sorted(unique.items()):z.write(r['local'],sha+'.json')
    target=ops.BASE+'/source/metadata_bundles/'+key+'.zip';receipt=ops.upload(bundle,target,port)
    code=f'''import pathlib,json,hashlib,zipfile,os
records={descriptor!r};archive=pathlib.Path({target!r});scope=pathlib.Path({ops.BASE!r})/'source';results=[]
with zipfile.ZipFile(archive) as z:
 expected={{r['member'] for r in records}}
 if set(z.namelist())!=expected or any('/' in n or '\\\\' in n or '..' in n for n in z.namelist()):raise ValueError('metadata ZIP member mismatch')
 # Check every member before publication; no partial unvalidated extraction.
 for name in sorted(expected):
  data=z.read(name)
  if hashlib.sha256(data).hexdigest()!=name[:-5]:raise ValueError('metadata member SHA changed')
 for record in records:
  path=pathlib.Path(record['remote']);resolved=path.resolve()
  if not resolved.is_relative_to(scope.resolve()):raise ValueError('member destination escapes R3 source')
  reused=False
  if path.exists():
   if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('existing immutable metadata bytes differ')
   reused=True
  else:
   path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.part');data=z.read(record['member'])
   with temp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
   os.replace(temp,path)
  results.append({{'path':str(path),'sha256':record['sha256'],'bytes':path.stat().st_size,'reused_exact_SHA':reused}})
print(json.dumps(results))
'''
    result=ops.remote_python(code,port,timeout=100)
    if result['exit_code']:raise ValueError(result['stderr'][-1200:])
    return {'bundle':receipt,'members':json.loads(result['stdout'])}
