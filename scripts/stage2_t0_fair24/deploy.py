"""Prepared S2-T24 deployment. Explicit mutation executor; only borrowed18/24.
Build from clean Git blob bytes, never mutable D:/paper modules. No science submit.
"""
from pathlib import Path
import argparse,base64,hashlib,json,os,subprocess,uuid
from scripts.stage2_t0_fair24 import ops
PORTS=(20418,20424)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def pin(ref):
 p=Path(ref['path'])
 if not p.is_file() or p.is_symlink() or sha(p)!=ref['sha256']:raise ValueError('actual source/dependency bytes mismatch')
 return p

def prepare(repo,output,dependencies):
 repo=Path(repo).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
 if subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip():raise ValueError('deployment source must be committed clean')
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip();prefix=output/commit;prefix.mkdir(exist_ok=True)
 bundle=prefix/'source.bundle';archive=prefix/'source.tar.gz'
 if not bundle.exists():subprocess.run(['git','bundle','create',str(bundle),'HEAD'],cwd=repo,check=True)
 if not archive.exists():subprocess.run(['git','archive','--format=tar.gz','--output='+str(archive),commit],cwd=repo,check=True)
 refs={role:{'path':str(pin(ref)),'sha256':ref['sha256']} for role,ref in dependencies.items()}
 mandatory={'reference_config','design_contract','historical_authority','human_authority','input_verification','normalized_handoff','raw_handoff','scanner_reference','converter_reference','official_metadata_feb21','official_metadata_feb22'}
 if not mandatory<=set(refs):raise ValueError('missing immutable deployment dependencies: '+str(sorted(mandatory-set(refs))))
 record={'format':'s2t24_deployment_prepared_v1','scientific_commit':commit,'allowed_ports':list(PORTS),'source_bundle':{'path':str(bundle),'sha256':sha(bundle)},'source_archive':{'path':str(archive),'sha256':sha(archive)},'dependencies':refs,'Git_blob_LF':True,'science_submissions':0,'GPU_calls':0,'remote_mutations':0,'source_roles':{'science':'new namespace with original85f environment/physics unchanged','data':'historical old converter and original scanner immutablebytes','control':'new independent workbench/controller'}}
 p=prefix/'prepared.json'
 with p.open('x',encoding='utf8') as f:json.dump(record,f,indent=2)
 return {'path':str(p),'sha256':sha(p)}

def execute_remote(port,code,payload):
 if port not in PORTS:raise ValueError('only borrowed18/24 deployment')
 client=ops.connect(port)
 try:
  encoded=base64.b64encode(code.encode()).decode();stdin,out,err=client.exec_command(ops.DQN+' -B -c "import base64;exec(base64.b64decode(\''+encoded+'\'))"',timeout=300)
  stdin.write(json.dumps(payload).encode());stdin.flush();stdin.channel.shutdown_write();stdout=out.read().decode();stderr=err.read().decode();status=out.channel.recv_exit_status()
  if status:raise RuntimeError('explicit deployment mutation failed '+str(status)+': '+stderr[-3000:])
  return json.loads(stdout)
 finally:client.close()

REMOTE=r'''import pathlib,json,sys,hashlib,subprocess,os
j=json.load(sys.stdin);root=pathlib.Path(j['root']);commit=j['commit'];repo=root/'source'/commit/'repo';bundle=root/'staging'/commit/'source.bundle'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
if sha(bundle)!=j['bundle_sha256']:raise ValueError('uploadedbundle bytes changed')
if not repo.exists():
 repo.parent.mkdir(parents=True,exist_ok=True);subprocess.run(['git','clone',str(bundle),str(repo)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE);subprocess.run(['git','checkout','--detach',commit],cwd=repo,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
if subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()!=commit or subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip():raise ValueError('receivedsource not clean exactcommit')
for ref in j['received_refs'].values():
 if sha(ref['path'])!=ref['sha256']:raise ValueError('receiveddependency bytes mismatch')
print(json.dumps({'status':'received_source_only_not_service_started','repo':str(repo),'git_commit':commit,'dependency_refs':j['received_refs'],'science_submissions':0,'GPU_calls':0}))
'''

def deploy(prepared_ref,approval_ref,port):
 prepared=json.loads(pin(prepared_ref).read_text(encoding='utf8'));approval=json.loads(pin(approval_ref).read_text(encoding='utf8'))
 if port not in PORTS or approval.get('format')!='s2t24_root_deployment_approval_v1' or approval.get('approved') is not True or approval.get('prepared_sha256')!=prepared_ref['sha256'] or approval.get('scientific_commit')!=prepared['scientific_commit'] or port not in approval.get('receiver_verified_ports',[]):raise ValueError('actual Root source/receiver approval absent')
 root=ops.BASE+'/'+str(port);commit=prepared['scientific_commit'];received={};client=ops.connect(port)
 try:
  sftp=client.open_sftp()
  def mkdirs(path):
   parts=Path(path).as_posix().split('/');current=''
   for part in parts:
    if not part:continue
    current+='/'+part
    try:sftp.mkdir(current)
    except OSError:sftp.stat(current)
  for role,ref in {'source.bundle':prepared['source_bundle'],**prepared['dependencies']}.items():
   source=pin(ref);target=ops.own_path(port,root+'/staging/'+commit+'/'+role+'/'+source.name) if role!='source.bundle' else ops.own_path(port,root+'/staging/'+commit+'/source.bundle');mkdirs(str(Path(target).parent));temporary=target+'.partial.'+uuid.uuid4().hex;sftp.put(str(source),temporary)
   # Exclusive SFTP destination creation prevents overwriting canonical dependencies.
   try:sftp.stat(target)
   except FileNotFoundError:sftp.rename(temporary,target)
   else:sftp.remove(temporary)
   if role!='source.bundle':received[role]={'path':target,'sha256':ref['sha256']}
  sftp.close()
 finally:client.close()
 return execute_remote(port,REMOTE,{'root':root,'commit':commit,'bundle_sha256':prepared['source_bundle']['sha256'],'received_refs':received})

def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path);p.add_argument('--output',type=Path);p.add_argument('--dependencies',type=Path);p.add_argument('--deploy',action='store_true');p.add_argument('--prepared-ref',type=Path);p.add_argument('--approval-ref',type=Path);p.add_argument('--port',type=int,choices=PORTS);a=p.parse_args()
 if a.deploy:result=deploy(json.loads(a.prepared_ref.read_text(encoding='utf8')),json.loads(a.approval_ref.read_text(encoding='utf8')),a.port)
 else:result=prepare(a.repo,a.output,json.loads(a.dependencies.read_text(encoding='utf8')))
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
