"""Exact Git-blob source shipment, shared once per account and verified per lane."""
import gzip,hashlib,io,json,subprocess,tarfile
from concurrent.futures import ThreadPoolExecutor
from . import remote_ops as r

def package(commit):
 folder=r.LOCAL/'deployment'/commit;folder.mkdir(parents=True,exist_ok=True)
 tree=subprocess.check_output(['git','ls-tree','-r','-z',commit,'src','scripts','pyproject.toml','tests'],cwd=r.REPO)
 members=[]
 for entry in tree.split(b'\0'):
  if not entry:continue
  header,name=entry.split(b'\t',1);mode,kind,oid=header.split();name=name.decode()
  if name.startswith('tests/')and not name.startswith(('tests/test_tsac_search27','tests/test_t0_width')):continue
  if kind!=b'blob'or mode not in(b'100644',b'100755'):raise ValueError('regular Git blobs only')
  members.append((name,mode,oid))
 output=subprocess.check_output(['git','cat-file','--batch'],input=b'\n'.join(x[2]for x in members)+b'\n',cwd=r.REPO)
 offset=0;blobs={}
 for name,mode,expected in members:
  end=output.index(b'\n',offset);oid,kind,size=output[offset:end].split();size=int(size);data=output[end+1:end+1+size];offset=end+size+2
  if oid!=expected or kind!=b'blob'or hashlib.sha1(b'blob '+str(size).encode()+b'\0'+data).hexdigest().encode()!=oid:raise ValueError('raw Git blob identity')
  blobs[name]=data
 assert offset==len(output)
 archive=folder/('source_'+commit+'.tar.gz')
 if not archive.exists():
  with archive.open('xb')as f:
   with gzip.GzipFile(fileobj=f,mode='wb',filename='',mtime=0)as gz:
    with tarfile.open(fileobj=gz,mode='w')as t:
     for name,mode,_ in members:
      info=tarfile.TarInfo(name);info.size=len(blobs[name]);info.mode=int(mode,8);info.mtime=0;t.addfile(info,io.BytesIO(blobs[name]))
 hashes={name:hashlib.sha256(data).hexdigest()for name,data in blobs.items()}
 with tarfile.open(archive,'r:gz')as t:
  actual={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest()for m in t if m.isfile()}
  assert actual==hashes
 manifest=dict(source_commit=commit,files=hashes,python_manifest={k:v for k,v in hashes.items()if k.endswith('.py')and k.startswith(('src/','scripts/'))},archive_SHA=r.sha(archive),archive_bytes=archive.stat().st_size)
 r.write_once(folder/'source_members.json',manifest);return archive,manifest

def install(commit,phase):
 archive,manifest=package(commit);roots={};readbacks={}
 for port in (20418,20627):
  node=r.NODES[port];source=node['base']+'/source/'+commit
  ref=r.publish(port,archive,node['base']+'/deployment/'+commit+'/'+archive.name)
  code='''
import pathlib,tarfile,hashlib,json
root=pathlib.Path(SOURCE);files=FILES;archive=pathlib.Path(ARCHIVE_FILE)
assert hashlib.sha256(archive.read_bytes()).hexdigest()==ARCHIVE_HASH
root.mkdir(parents=True,exist_ok=True)
with tarfile.open(archive,'r:gz')as t:
 seen=set()
 for m in t:
  if not m.isfile()or m.name not in files or m.name in seen:raise ValueError('source archive set')
  p=(root/m.name).resolve()
  if not p.is_relative_to(root.resolve()):raise ValueError('source path')
  raw=t.extractfile(m).read()
  if hashlib.sha256(raw).hexdigest()!=files[m.name]:raise ValueError('source byte SHA')
  if p.exists():
   if p.read_bytes()!=raw:raise ValueError('immutable source differs')
  else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
  seen.add(m.name)
assert seen==set(files)
print(json.dumps(dict(passed=True,files=len(files),source_root=str(root))))
'''.replace('SOURCE',repr(source)).replace('FILES',repr(manifest['files'])).replace('ARCHIVE_FILE',repr(ref['path'])).replace('ARCHIVE_HASH',repr(ref['sha256']))
  result=r.remote_python(port,code,timeout=90);r.write_once(r.LOCAL/'deployment'/phase/f'install_{port}.json',result)
  if result['exit_code']:raise RuntimeError(result['stderr'])
 for port,node in r.NODES.items():roots[port]=node['base']+'/source/'+commit
 def witness(port):
  code='''
import pathlib,hashlib,json,sys,torch
root=pathlib.Path(SOURCE);expected=FILES
actual={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()for p in root.rglob('*')if p.is_file()and'__pycache__'not in p.parts}
assert actual==expected
sys.path[:0]=[str(root/'src'),str(root)]
import satellite_rl.tsac_search27.agent as module
assert pathlib.Path(module.__file__).resolve().is_relative_to(root.resolve())
print(json.dumps(dict(passed=True,files=actual,python=sys.executable,torch=str(torch.__version__),cuda_available=torch.cuda.is_available(),actual_module=module.__file__)))
'''.replace('SOURCE',repr(roots[port])).replace('FILES',repr(manifest['files']))
  return port,r.remote_python(port,code,timeout=90)
 with ThreadPoolExecutor(3)as pool:
  for port,result in pool.map(witness,r.NODES):
   r.write_once(r.LOCAL/'deployment'/phase/f'witness_{port}.json',result)
   if result['exit_code']:raise RuntimeError(result['stderr'])
   readbacks[port]=json.loads(result['stdout'])
 return roots,manifest,readbacks
