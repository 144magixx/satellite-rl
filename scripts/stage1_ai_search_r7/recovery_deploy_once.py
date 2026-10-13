"""Root-admitted endpoint deployment only; no HTTP POST or science dispatch."""
import argparse,base64,hashlib,json,os,subprocess
from pathlib import Path
import paramiko

RUN='r7_labels_20261013_234807_312cdce7'
OPS='bb7b84d4912e45ac1eac4f59b9fe78d6c7f5ce1a'
SCIENCE='13702ff5d4c2aac8c559ddf79bdd2d8202dec45f'
REMOTE=r'''
import base64,datetime,fcntl,hashlib,importlib.util,json,os,signal,socket,subprocess,sys,time,urllib.request
from pathlib import Path
payload=json.loads(sys.stdin.read());plan=payload['plan'];admission=payload['admission']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(ok,message):
 if not ok:raise ValueError(message)
def pin(ref):need(sha(ref['path'])==ref['sha256'],'reference SHA mismatch: '+ref['path']);return Path(ref['path'])
def proc(pid):
 p=Path('/proc')/str(pid)
 try:
  raw=(p/'stat').read_text();s=raw[raw.rfind(')')+2:].split()
  return dict(birth=int(s[19]),state=s[0],cmdline_sha256=sha(p/'cmdline'),cmdline=[v.decode() for v in (p/'cmdline').read_bytes().split(b'\0') if v],pythonpath=dict(v.split(b'=',1) for v in (p/'environ').read_bytes().split(b'\0') if b'=' in v).get(b'PYTHONPATH',b'').decode())
 except (FileNotFoundError,ProcessLookupError):return None
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
need(boot==plan['old_service']['boot_id'],'boot changed: new root service admission required')
archive=Path(plan['archive_root']);directory=archive/'outputs/workbench/jobs'/plan['run_id']
need(plan['ssh_port']==20474 and plan['api_port']==9015 and plan['run_id']=='r7_labels_20261013_234807_312cdce7','only original74 endpoint')
need(str(archive)=='/home/u2024110224/first_stage_satellite_rl/s1_ai17/20474/round7/archive','original74 archive only')
old=plan['old_service'];actual=proc(old['pid'])
need(actual is not None and all(actual[k]==old[k] for k in ('birth','cmdline','cmdline_sha256','pythonpath')),'old service PID/birth/command/PYTHONPATH changed')
need(actual['state'] not in ('Z','X'),'old service not live')
def listeners():
 result=[]
 for table in ('/proc/net/tcp','/proc/net/tcp6'):
  for line in Path(table).read_text().splitlines()[1:]:
   f=line.split()
   if int(f[1].split(':')[-1],16)==9015 and f[3]=='0A':result.append(f[9])
 return result
inodes=listeners();fds=[os.readlink(p) for p in (Path('/proc')/str(old['pid'])/'fd').iterdir()]
need(len(inodes)==1 and 'socket:['+inodes[0]+']' in fds,'9015 listener ownership changed')
for ref in plan['unchanged_refs'].values():pin(ref)
job=json.loads((directory/'job.json').read_text());status=json.loads((directory/'status.json').read_text())
need(status['stage']=='failed' and status['completed_cases']==147,'original failed147 endpoint required')
need(not (directory/'operations_recovery.json').exists(),'recovery already registered; never redeploy automatically')
manifest_ref=dict(path=str(Path(job['source_root'])/'source_manifest.json'),sha256=job['source_manifest_sha256']);manifest=json.loads(pin(manifest_ref).read_text())
need(manifest['git_commit']==plan['scientific_commit'] and len(manifest['files'])==698,'exact original698 science snapshot')
for name,expected in manifest['files'].items():need(sha(Path(job['source_root'])/name)==expected,'original science leaf changed: '+name)
for d in (archive/'outputs/workbench/jobs').iterdir():
 if (d/'status.json').is_file():need(json.loads((d/'status.json').read_text()).get('stage') in ('completed','failed','stopped','cancelled','interrupted'),'another active/queued job forbids endpoint deployment')
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==old['pid']:continue
 try:
  cmd=(p/'cmdline').read_bytes()
  need(str(archive).encode() not in cmd,'another process references74 archive: '+p.name)
 except (FileNotFoundError,ProcessLookupError):pass
v=os.statvfs(archive);free=v.f_bavail*v.f_frsize
mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.split(':')[0] in ('MemAvailable','MemTotal')}
need(free>=plan['resource_limits']['minimum_free_bytes'] and mem['MemAvailable']>=plan['resource_limits']['minimum_available_memory_bytes'],'resource guard failed')
gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10,check=True)
need(gpu.stdout.split(',')[0].strip()==plan['gpu_uuid'],'GPU identity changed')
compute=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10,check=True)
need(not compute.stdout.strip(),'GPU compute process present; root fresh allocation required')
contract=json.loads(base64.b64decode(payload['assets']['contract']['base64']));review=json.loads(base64.b64decode(payload['assets']['recovery_review']['base64']))
need(review.get('passed') is True and review['run_id']==plan['run_id'] and review['operational_commit']==plan['operational_commit'] and review['contract_sha256']==plan['contract_ref']['sha256'],'root recovery acceptance differs')
need(contract['operational_commit']==plan['operational_commit'],'ops contract commit differs')
index=json.loads((directory/'case_index.json').read_text())
need(set(map(int,index))==set(contract['preserved_case_indices']),'original147 index partition differs')
for i in contract['preserved_case_indices']:
 result=json.loads(pin(index[str(i)]).read_text())
 need(result['parent_job_sha256']==contract['original_job_ref']['sha256'] and result['case_index']==i,'original147 parent ancestry differs')
 pin(result['result_ref']);pin(result['result_ref']['cost_endpoint_ref'])
for failure in contract['failed_cases'].values():
 for ref in failure['evidence_refs'].values():pin(ref)
for i in contract['remaining_case_indices']:need(not (directory/'case_workers'/str(i)/'started.json').exists(),'remaining139 original attempt already started')
for observation in contract['original_pid_births']:
 if observation['boot_id']!=boot:continue
 state=proc(observation['pid'])
 if state is None or state['state'] in ('Z','X'):continue
 if observation['birth_kind']=='proc_start_ticks':alive=state['birth']==observation['birth']
 elif observation['birth_kind']=='unix_boot_seconds':
  btime=int(next(line.split()[1] for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime ')))
  alive=abs(btime+state['birth']/os.sysconf('SC_CLK_TCK')-observation['birth'])<.02
 else:raise ValueError('untyped PID birth')
 need(not alive,'original owned PID still live: '+str(observation['pid']))
opsdir=Path(plan['operations_dir']);allowed=Path('/home/u2024110224/first_stage_satellite_rl/s1_ai17/20474/round7/operations_recovery')
need(opsdir.is_relative_to(allowed) and not opsdir.exists(),'new independent operations epoch required')
opsdir.mkdir(parents=True)
lock=(opsdir/'deployment.lock').open('xb');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for key,item in payload['assets'].items():
 ref=item['ref'];target=Path(ref['path']);data=base64.b64decode(item['base64'])
 need(target.is_relative_to(allowed) and hashlib.sha256(data).hexdigest()==ref['sha256'],'asset target/source differs')
 if target.exists():need(sha(target)==ref['sha256'],'existing immutable asset differs')
 else:target.parent.mkdir(parents=True,exist_ok=True);target.open('xb').write(data)
 pin(ref)
for k,ref in contract['operations_source_refs'].items():need(ref==plan['operations_source_refs'][k],'ops3 paths/SHA differ');pin(ref)
sys.path[:0]=[str(Path(job['source_root'])/'src'),job['source_root']]
spec=importlib.util.spec_from_file_location('satellite_rl.workbench.r7_recovery_ops',pin(contract['operations_source_refs']['module']));ops=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ops;spec.loader.exec_module(ops)
ops.validate(job,dict(contract_ref=plan['contract_ref'],root_review_ref=plan['recovery_review_ref']))
record={'schema':'R7EndpointOperationalDeploymentV1','checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'plan_sha256':payload['plan_sha256'],'root_admission_sha256':payload['admission_sha256'],'original_service':old,'scientific_commit':plan['scientific_commit'],'operational_commit':plan['operational_commit'],'operations_source_refs':plan['operations_source_refs'],'source698_verified':True,'original_pid_observations_gone':len(contract['original_pid_births']),'free_bytes':free,'memory':mem,'gpu':gpu.stdout,'compute':compute.stdout,'actual_POST':0,'actual_new_science_calls':0}
(opsdir/'deployment_preflight.json').write_text(json.dumps(record,indent=2)+'\n')
for ref in plan['unchanged_refs'].values():pin(ref)
def same_old():
 state=proc(old['pid']);return state is not None and all(state[k]==old[k] for k in ('birth','cmdline','cmdline_sha256','pythonpath'))
need(same_old(),'old service identity changed before stop')
need(hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal'),'race-safe pidfd signal unavailable')
pidfd=os.pidfd_open(old['pid']);need(same_old(),'PID reuse before stop');signal.pidfd_send_signal(pidfd,signal.SIGTERM);os.close(pidfd)
deadline=time.monotonic()+10
while time.monotonic()<deadline:
 state=proc(old['pid'])
 if state is None or state['state'] in ('Z','X'):break
 time.sleep(.1)
else:raise ValueError('old service did not exit; no SIGKILL or new server')
need(not listeners(),'9015 occupied after old service exit')
command=[plan['python'],'-B','-u',plan['operations_source_refs']['server']['path'],'--original-source',job['source_root'],'--archive-root',str(archive),'--port','9015','--contract',plan['contract_ref']['path'],'--contract-sha256',plan['contract_ref']['sha256'],'--review',plan['recovery_review_ref']['path'],'--review-sha256',plan['recovery_review_ref']['sha256']]
env=os.environ.copy();env['PYTHONPATH']=str(Path(job['source_root'])/'src')+':'+job['source_root'];env['SATELLITE_RL_SSH_PORT']='20474';env['PYTHONDONTWRITEBYTECODE']='1'
env.pop('BUPT_PASSWORD',None)
with (opsdir/'server_stdout.bin').open('xb') as out,(opsdir/'server_stderr.bin').open('xb') as err:
 child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err,cwd=opsdir,env=env,start_new_session=True)
record.update(new_service_pid=child.pid,actual_command=command)
(opsdir/'actual_endpoint_launch.json').write_text(json.dumps(record,indent=2)+'\n')
health=None;deadline=time.monotonic()+8
while time.monotonic()<deadline:
 need(child.poll() is None,'new restricted server exited; inspect exact logs')
 try:
  with urllib.request.urlopen('http://127.0.0.1:9015/api/ai-search-r7/tasks/'+plan['run_id'],timeout=1) as r:health=dict(status=r.status,body=json.loads(r.read()))
  break
 except (ConnectionError,TimeoutError,urllib.error.URLError):time.sleep(.2)
need(health is not None and health['status']==200,'bounded readonly endpoint health failed')
need(health['body']['job']==job and health['body']['status']['stage']=='failed' and health['body']['status']['completed_cases']==147,'readonly health altered original task')
for ref in plan['unchanged_refs'].values():pin(ref)
need(not (directory/'operations_recovery.json').exists(),'unexpected recovery dispatch during endpoint deployment')
record.update(new_service_identity=proc(child.pid),health=health,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_records_unchanged=True)
(opsdir/'endpoint_deployment_receipt.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,ensure_ascii=False))
'''

def digest(data):return hashlib.sha256(data).hexdigest()
def validate_admission(plan,praw,admission,script_sha):
    if plan.get('schema')!='R7Labels74EndpointDeploymentPlanV1' or plan.get('operational_commit')!=OPS or plan.get('scientific_commit')!=SCIENCE or plan.get('run_id')!=RUN or plan.get('ssh_port')!=20474 or plan.get('api_port')!=9015:raise ValueError('only exact bb7 original74 endpoint plan')
    if plan.get('python')!='/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python':raise ValueError('original DQN interpreter only')
    if admission.get('schema')!='R7Labels74EndpointDeploymentAdmissionV1' or admission.get('passed') is not True or admission.get('plan_sha256')!=digest(praw) or admission.get('deployment_script_sha256')!=script_sha or admission.get('contract_ref')!=plan['contract_ref'] or admission.get('recovery_review_ref')!=plan['recovery_review_ref'] or admission.get('actual_POST_authorized') is not False:raise ValueError('exact root endpoint-only deployment admission required')

def assets(plan,repo):
    result={}
    for key,item in plan['assets'].items():
        if 'git_path' in item:data=subprocess.check_output(['git','show',OPS+':'+item['git_path']],cwd=repo)
        else:data=Path(item['local_path']).read_bytes()
        if digest(data)!=item['ref']['sha256']:raise ValueError('local immutable asset differs: '+key)
        result[key]=dict(ref=item['ref'],base64=base64.b64encode(data).decode())
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--admission',required=True);p.add_argument('--admission-sha256',required=True);p.add_argument('--output-root',required=True);args=p.parse_args()
    praw=Path(args.plan).read_bytes();araw=Path(args.admission).read_bytes()
    if digest(praw)!=args.plan_sha256 or digest(araw)!=args.admission_sha256:raise ValueError('local plan/admission SHA differs')
    plan=json.loads(praw);admission=json.loads(araw);validate_admission(plan,praw,admission,digest(Path(__file__).read_bytes()))
    payload=dict(plan=plan,admission=admission,plan_sha256=args.plan_sha256,admission_sha256=args.admission_sha256,assets=assets(plan,Path(__file__).resolve().parents[2]))
    out=Path(args.output_root);out.mkdir(parents=True,exist_ok=False)
    client=paramiko.SSHClient();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('10.160.17.102',port=20474,username='u2024110224',password=os.environ['BUPT_PASSWORD'],look_for_keys=False,allow_agent=False,timeout=20)
    try:
        code=REMOTE.replace("payload=json.loads(sys.stdin.read());",'payload=json.loads('+repr(json.dumps(payload))+');',1)
        stdin,stdout,stderr=client.exec_command(plan['python']+' -B -',timeout=90);stdin.write(code);stdin.channel.shutdown_write()
        raw=stdout.read();err=stderr.read();exitcode=stdout.channel.recv_exit_status();(out/'stdout.bin').write_bytes(raw);(out/'stderr.bin').write_bytes(err)
        (out/'transport_receipt.json').write_text(json.dumps(dict(exit_code=exitcode,stdout_sha256=digest(raw),stderr_sha256=digest(err),plan_sha256=args.plan_sha256,admission_sha256=args.admission_sha256,actual_POST=0))+'\n')
        if exitcode:raise RuntimeError('deployment failed closed; inspect saved stdout/stderr, do not retry blindly')
        (out/'endpoint_deployment_receipt.json').write_text(json.dumps(json.loads(raw),ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(dict(output_root=str(out),exit_code=exitcode,actual_POST=0,actual_new_science_calls=0)))
    finally:client.close()

if __name__=='__main__':main()
