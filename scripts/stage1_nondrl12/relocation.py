"""Reviewable logical74 -> physical73 ND12 service relocation.

Nothing executes without an explicit release/activate command and a pinned
client-release receipt. Stop uses the old workbench API first; its frozen API
has no force option. Optional exact-PID TERM/KILL is separately authorized and
never targets a process group. Shared source/archive/run IDs remain unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
from satellite_rl.coverage.nondrl.runner import worker_lock
from satellite_rl.workbench.runtime import pid_alive,process_start_time
from . import ops

SOURCE="105d5f57ec241cce0c484b95f6341d6dc418b3fe"
RUN_ID="evaluate_nondrl_20261012_034602_5f3a61a6"
LOGICAL=20474
OLD_PHYSICAL=20474
NEW_PHYSICAL=20473
API_PORT=8943
MIGRATION_ID="logical20474_to_physical20473_20261012"
REMOTE_ROOT=ops.BASE+"/20474"
REMOTE_RECEIPTS=REMOTE_ROOT+"/ops/relocation/"+MIGRATION_ID
LOCAL=ops.LOCAL/"preflight/relocation"/MIGRATION_ID


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
    temporary.replace(path)


def verify_client_release(path,expected_sha):
    if digest(path)!=expected_sha:raise ValueError("client release receipt SHA differs")
    record=read(path)
    if (record.get("experiment_id")!="S1-ND12" or record.get("exited") is not True or
            type(record.get("pid")) is not int or not isinstance(record.get("process_start_time"),(int,float)) or
            "stage1_nondrl12" not in " ".join(record.get("command",[]))):
        raise ValueError("precise ND12 client identity/release evidence required")
    if pid_alive(record["pid"],record["process_start_time"]):
        raise ValueError("same-birth old client remains alive")
    return record


COMMON=r'''
import datetime,fcntl,hashlib,json,os,pathlib,signal,socket,subprocess,time,urllib.request
r=json.loads(REQUEST_JSON);root=pathlib.Path(r['root']);archive=root/'archive';repo=root/'repo'
jobdir=archive/'outputs/workbench/jobs'/r['run_id'];service_file=root/'ops/service.json'
receipt_dir=root/'ops/relocation'/r['migration_id']
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as s:
  for b in iter(lambda:s.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def put(p,v):
 p=pathlib.Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');t.replace(p)
def proc(pid):
 p=pathlib.Path('/proc')/str(pid)
 if not p.exists():return {'pid':pid,'alive':False}
 try:
  a=(p/'stat').read_text(encoding='utf-8');v=a[a.rfind(')')+2:].split();ticks=int(v[19]);boot=next(int(x.split()[1]) for x in pathlib.Path('/proc/stat').read_text(encoding='utf-8').splitlines() if x.startswith('btime '));command=(p/'cmdline').read_bytes().decode('utf-8','replace').strip('\0').split('\0')
 except FileNotFoundError:return {'pid':pid,'alive':False,'exit_read_race':True}
 return {'pid':pid,'alive':v[0]!='Z','state':v[0],'start_ticks':ticks,'process_start_time':boot+ticks/os.sysconf('SC_CLK_TCK'),'cpu_seconds':(int(v[11])+int(v[12]))/os.sysconf('SC_CLK_TCK'),'rss_bytes':int(v[21])*os.sysconf('SC_PAGE_SIZE'),'command':command}
def identity(pid,birth,command=None,ticks=None):
 for attempt in range(5):
  p=proc(pid)
  if not p['alive']:return p
  if ticks is not None and p['start_ticks']!=ticks:raise RuntimeError('service PID birth differs')
  if birth is not None and abs(p['process_start_time']-birth)>=.01:raise RuntimeError('worker PID birth differs')
  if command is not None and p['command']!=command:
   if not any(p['command']) and attempt<4:time.sleep(.05);continue
   raise RuntimeError('worker exact command differs; no signal eligibility')
  return p
def send_exact(pid,birth,command,signo,ticks=None):
 p=identity(pid,birth,command,ticks)
 if not p['alive']:return
 if hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal'):
  fd=os.pidfd_open(pid)
  try:identity(pid,birth,command,ticks);signal.pidfd_send_signal(fd,signo)
  finally:os.close(fd)
 else:identity(pid,birth,command,ticks);os.kill(pid,signo)
def api(path,body=None):
 request=urllib.request.Request('http://127.0.0.1:'+str(r['api_port'])+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(request,timeout=20) as s:return json.load(s)
def listening():
 with socket.socket() as s:s.settimeout(1);return s.connect_ex(('127.0.0.1',r['api_port']))==0
def ledger():
 p=jobdir/'raw_scenes/manifest.json';m=read(p);rows=m['rows']
 if len({x['key'] for x in rows})!=len(rows):raise RuntimeError('duplicate committed row keys')
 for x in rows:
  q=(p.parent/x['file']).resolve()
  if not q.is_relative_to(p.parent.resolve()) or sha(q)!=x['sha256']:raise RuntimeError('committed RowStore member SHA differs')
 return {'manifest_sha256':sha(p),'provenance':m['provenance'],'rows':rows,'committed_count':len(rows)}
def nd12_processes():
 result=[]
 for p in pathlib.Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:
   v=proc(int(p.name));env=(p/'environ').read_bytes().split(b'\0')
   owned=(b'SATELLITE_RL_OUTPUT_ROOT='+str(archive).encode()) in env
   if v['alive'] and (owned or any(str(jobdir.parent) in x or str(archive) in x for x in v.get('command',[]))):result.append(v)
  except (OSError,ValueError):pass
 return result
def queue():
 return [p.parent.name for p in (archive/'outputs/workbench/jobs').glob('*/status.json') if read(p).get('stage')=='queued']
def facts():
 job=read(jobdir/'job.json');status=read(jobdir/'status.json');launch=read(jobdir/'launch.json');service=read(service_file)
 if job.get('experiment_id')!='S1-ND12' or job.get('run_id')!=r['run_id'] or job.get('git_commit')!=r['source_commit'] or job.get('archive_root')!=str(archive) or job.get('kind')!='evaluate':raise RuntimeError('exact frozen evaluation identity differs')
 if service.get('source_commit')!=r['source_commit'] or service.get('archive_root')!=str(archive) or service.get('api_port')!=r['api_port']:raise RuntimeError('logical service identity differs')
 worker=identity(launch['pid'],launch['process_start_time'],launch['command'])
 if worker['alive'] and ('--job-dir' not in worker['command'] or str(jobdir) not in worker['command']):raise RuntimeError('worker belongs to another jobdir')
 server=identity(service['pid'],None,ticks=service['start_ticks'])
 if server['alive'] and ('satellite_rl.workbench.server' not in server['command'] or str(archive) not in server['command'] or str(r['api_port']) not in server['command']):raise RuntimeError('old server command differs')
 return {'job':job,'status':status,'launch':launch,'service':service,'worker':worker,'server':server,'ledger':ledger(),'queued':queue(),'group_processes':nd12_processes(),'config_sha256':sha(jobdir/'config.json'),'observed_at':now()}
'''

INSPECT=COMMON+r'''
print(json.dumps(facts(),ensure_ascii=False))
'''

RELEASE=COMMON+r'''
receipt_dir.mkdir(parents=True,exist_ok=True)
lock=(root/'ops/relocation.lock').open('a+b');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
try:
 saved=receipt_dir/'release_receipt.json'
 if saved.exists():
  old=read(saved)
  if old.get('migration_id')!=r['migration_id'] or nd12_processes() or listening() or queue():raise RuntimeError('existing release no longer holds')
  print(json.dumps(old));raise SystemExit(0)
 signals=[];reconciled=False
 if (receipt_dir/'before.json').exists() or (receipt_dir/'api_stop.json').exists():
  if not r.get('reconcile_existing_stop'):raise RuntimeError('prior stop evidence exists; explicit reconciliation required, never overwrite before')
  before=read(receipt_dir/'before.json');prior_api=read(receipt_dir/'api_stop.json');live=facts();proof=r.get('reconciliation_proof') or {}
  if live['worker']['alive'] or live['status'].get('stage')!='interrupted':raise RuntimeError('prior worker exit/interrupted evidence not established')
  if before['job']['run_id']!=r['run_id'] or before['job']['git_commit']!=r['source_commit'] or before['config_sha256']!=live['config_sha256']:raise RuntimeError('prior stop identity/source differs')
  launch=before['launch']
  if proof.get('experiment_id')!='S1-ND12' or proof.get('run_id')!=r['run_id'] or proof.get('worker_pid')!=launch['pid'] or proof.get('worker_process_start_time')!=launch['process_start_time'] or proof.get('worker_absent_observed') is not True or proof.get('signal_delivery_confirmed') not in ('SIGTERM','SIGKILL'):raise RuntimeError('pinned actual signal/exit reconciliation proof required')
  signals=[{'signal':proof['signal_delivery_confirmed'],'pid':launch['pid'],'at':proof.get('signal_time_utc'),'observation_upper_bound':proof.get('observation_utc'),'signal_time_unknown':proof.get('signal_time_utc') is None,'source':'root pinned reconciliation proof','proof_sha256':r['reconciliation_sha256']}]
  put(receipt_dir/'reconciliation.json',{'proof':proof,'proof_sha256':r['reconciliation_sha256'],'live':live,'prior_api_stop_sha256':sha(receipt_dir/'api_stop.json'),'original_before_sha256':sha(receipt_dir/'before.json')});reconciled=True
 else:before=facts()
 if before['queued']:raise RuntimeError('pending work must be drained by root before relocation')
 unrelated=[p for p in before['group_processes'] if p['pid'] not in (before['worker']['pid'],before['server']['pid'])]
 if unrelated:raise RuntimeError('unexpected logical74 process; no broad signals allowed')
 if not reconciled:
  hold=api('/api/dispatch/hold',{'ttl_seconds':600,'reason':'S1-ND12 authorized logical74 relocation'})
  stopped=api('/api/runs/'+r['run_id']+'/stop',{})
  put(receipt_dir/'before.json',before);put(receipt_dir/'api_stop.json',{'at':now(),'response':stopped,'hold':hold,'force_supported':False})
 launch=before['launch'];until=time.monotonic()+r['cooperative_seconds']
 while identity(launch['pid'],launch['process_start_time'],launch['command'])['alive'] and time.monotonic()<until:time.sleep(.5)
 if identity(launch['pid'],launch['process_start_time'],launch['command'])['alive']:
  if not r['allow_force']:raise RuntimeError('cooperative native stop timed out; force not authorized; no release')
  send_exact(launch['pid'],launch['process_start_time'],launch['command'],signal.SIGTERM);signals.append({'signal':'SIGTERM','pid':launch['pid'],'at':now()});put(receipt_dir/'signals.json',signals)
  until=time.monotonic()+5
  while identity(launch['pid'],launch['process_start_time'],launch['command'])['alive'] and time.monotonic()<until:time.sleep(.5)
  if identity(launch['pid'],launch['process_start_time'],launch['command'])['alive']:
   send_exact(launch['pid'],launch['process_start_time'],launch['command'],signal.SIGKILL);signals.append({'signal':'SIGKILL','pid':launch['pid'],'at':now()});put(receipt_dir/'signals.json',signals)
 until=time.monotonic()+5
 while identity(launch['pid'],launch['process_start_time'],launch['command'])['alive'] and time.monotonic()<until:time.sleep(.5)
 if identity(launch['pid'],launch['process_start_time'],launch['command'])['alive']:raise RuntimeError('exact worker has not exited')
 time.sleep(4.5) # Give the original scheduler one tick to retain interrupted evidence.
 service=before['service'];server=identity(service['pid'],None,ticks=service['start_ticks'])
 if server['alive']:
  send_exact(service['pid'],None,server['command'],signal.SIGTERM,ticks=service['start_ticks'])
  until=time.monotonic()+5
  while identity(service['pid'],None,ticks=service['start_ticks'])['alive'] and time.monotonic()<until:time.sleep(.5)
  if identity(service['pid'],None,ticks=service['start_ticks'])['alive']:raise RuntimeError('old scheduler/service did not exit; no release')
 after=ledger();old_by_key={x['key']:x for x in before['ledger']['rows']};new_by_key={x['key']:x for x in after['rows']}
 if any(new_by_key.get(k)!=v for k,v in old_by_key.items()):raise RuntimeError('committed row changed or disappeared')
 if listening() or nd12_processes() or queue():raise RuntimeError('old physical74 not released: service/worker/queue remains')
 status=read(jobdir/'status.json')
 if status.get('stage') not in ('completed','stopped','interrupted','failed'):
  put(receipt_dir/'status_before_interrupted.json',status)
  status.update(stage='interrupted',error='Authorized physical relocation after exact worker exit',relocation_receipt=str(saved),heartbeat_at=now());put(jobdir/'status.json',status)
 receipt={'experiment_id':'S1-ND12','migration_id':r['migration_id'],'source_commit':r['source_commit'],'logical_ssh_port':20474,'old_physical_ssh_port':20474,'new_physical_ssh_port':20473,'api_port':r['api_port'],'run_id':r['run_id'],'archive_root':str(archive),'repo_root':str(repo),'before':before,'retained_ledger':after,'signals':signals,'config_sha256':sha(jobdir/'config.json'),'old_owner_absent':True,'old_service_closed':True,'old_worker_exited':True,'queued_count':0,'released_physical_port':20474,'released':True,'released_at':now(),'inflight_committed_after_stop':after['committed_count']-before['ledger']['committed_count'],'inflight_discarded':bool(signals),'engineering_inflight_wall_seconds_estimate':max(0.,time.time()-datetime.datetime.fromisoformat(before['status']['heartbeat_at']).timestamp()) if signals else 0.,'estimate_scope':'since last durable heartbeat, not exact solver runtime','uncommitted_files':[str(p.relative_to(jobdir)) for p in (jobdir/'raw_scenes').glob('*') if p.name!='manifest.json' and p.name not in {x['file'] for x in after['rows']}],'client_release_sha256':r['client_release_sha256']}
 if receipt['config_sha256']!=before['config_sha256']:raise RuntimeError('scientific config changed')
 put(saved,receipt);print(json.dumps(receipt,ensure_ascii=False))
finally:lock.close()
'''

ACTIVATE=COMMON+r'''
receipt_dir.mkdir(parents=True,exist_ok=True)
lock=(root/'ops/relocation.lock').open('a+b');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
try:
 release_path=receipt_dir/'release_receipt.json'
 if sha(release_path)!=r['release_sha256']:raise RuntimeError('release receipt SHA differs')
 released=read(release_path)
 if not all(released.get(k) is True for k in ['released','old_owner_absent','old_service_closed','old_worker_exited']):raise RuntimeError('release hard gate incomplete')
 if released.get('client_release_sha256')!=r['client_release_sha256'] or released.get('source_commit')!=r['source_commit']:raise RuntimeError('release source/client differs')
 current_ledger=ledger();old_by_key={x['key']:x for x in released['retained_ledger']['rows']};current_by_key={x['key']:x for x in current_ledger['rows']}
 if sha(jobdir/'config.json')!=released['config_sha256'] or current_ledger['provenance']!=released['retained_ledger']['provenance'] or any(current_by_key.get(k)!=v for k,v in old_by_key.items()):raise RuntimeError('source/committed RowStore prefix changed before activation')
 head=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
 dirty=subprocess.run(['git','-C',str(repo),'status','--porcelain'],capture_output=True,text=True,check=True).stdout.strip()
 if head!=r['source_commit'] or dirty:raise RuntimeError('shared original frozen repo differs')
 env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(repo/'src'),SATELLITE_RL_OUTPUT_ROOT=str(archive),SATELLITE_RL_SSH_PORT='20474',SATELLITE_RL_PHYSICAL_SSH_PORT='20473',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
 proof=subprocess.run([r['python'],'-c','import sys,json,torch,satellite_rl;print(json.dumps({"python":sys.executable,"cuda":torch.cuda.is_available(),"package":satellite_rl.__file__,"build":torch.version.cuda,"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))'],env=env,cwd=repo,capture_output=True,text=True,check=True,timeout=45)
 package=json.loads(proof.stdout)
 if not package['cuda'] or pathlib.Path(package['package']).resolve()!=repo/'src/satellite_rl/__init__.py':raise RuntimeError('destination DQN/CUDA/package differs')
 command=[r['python'],'-B','-m','satellite_rl.workbench.server','--archive-root',str(archive),'--port',str(r['api_port']),'--python',r['python'],'--max-workers',str(r['workers'])]
 reused_service=listening()
 if reused_service:
  record=read(service_file)
  if record.get('source_commit')!=head or record.get('logical_ssh_port')!=20474 or record.get('physical_ssh_port')!=20473 or record.get('relocation_receipt_sha256')!=r['release_sha256'] or record.get('archive_root')!=str(archive):raise RuntimeError('already listening service ownership differs')
  server=identity(record['pid'],record['process_start_time'],ticks=record['start_ticks'])
  if not server['alive'] or server['command']!=command:raise RuntimeError('reused new service birth/command differs')
 else:
  with (root/'ops/relocated_workbench.log').open('ab',buffering=0) as log:
   child=subprocess.Popen(command,env=env,cwd=repo,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
  server=proc(child.pid)
  record={'pid':child.pid,'start_ticks':server['start_ticks'],'process_start_time':server['process_start_time'],'source_commit':head,'source_root':str(repo),'archive_root':str(archive),'api_port':r['api_port'],'ssh_port':20474,'logical_ssh_port':20474,'physical_ssh_port':20473,'actual_execution_target':'u2024110224@10.160.17.102:20473','legacy_execution_target_scope':'logical submission only, not actual hardware after relocation','max_workers':r['workers'],'package':package,'relocation_receipt':str(release_path),'relocation_receipt_sha256':r['release_sha256'],'observed_utc':now()}
  put(service_file,record) # Save owner birth before the first potentially unknown network result.
  until=time.monotonic()+45
  while not listening() and time.monotonic()<until:
   if not identity(child.pid,server['process_start_time'])['alive']:raise RuntimeError('new service exited')
   time.sleep(.5)
  if not listening():raise RuntimeError('new service8943 did not listen')
  ready=identity(child.pid,server['process_start_time'],command)
  if not ready['alive'] or (pathlib.Path('/proc')/str(child.pid)/'cwd').resolve()!=repo.resolve():raise RuntimeError('ready service exact command/cwd differs')
 api('/api/coverage/nondrl/catalog')
 hold_release=api('/api/dispatch/hold/release',{})
 if hold_release.get('active') is not False:raise RuntimeError('new logical dispatch hold remains active')
 status=read(jobdir/'status.json');resume=None
 if status['stage'] in ('stopped','failed','interrupted'):
  resume=api('/api/runs/'+r['run_id']+'/resume',{})
 elif status['stage'] not in ('completed','queued','starting','initializing','evaluating'):raise RuntimeError('unexpected same-ID continuation stage')
 until=time.monotonic()+30;launch_after=None
 while status['stage']!='completed' and time.monotonic()<until:
  current=read(jobdir/'launch.json')
  if current.get('process_start_time')!=released['before']['launch'].get('process_start_time'):
   worker=identity(current['pid'],current['process_start_time'],current['command']);launch_after={'run_id':r['run_id'],'logical_ssh_port':20474,'physical_ssh_port':20473,'actual_execution_target':record['actual_execution_target'],'worker':worker,'source_commit':head,'config_sha256':sha(jobdir/'config.json'),'relocation_receipt_sha256':r['release_sha256'],'observed_at':now()};put(jobdir/'launch_placement.json',launch_after);break
  time.sleep(.5)
 if status['stage']!='completed' and launch_after is None:raise RuntimeError('same-ID continuation did not produce a new guarded launch')
 after=ledger();old_by_key={x['key']:x for x in released['retained_ledger']['rows']};new_by_key={x['key']:x for x in after['rows']}
 if any(new_by_key.get(k)!=v for k,v in old_by_key.items()):raise RuntimeError('resume changed a committed row')
 receipt={'experiment_id':'S1-ND12','migration_id':r['migration_id'],'stage':'activated_same_id','logical_ssh_port':20474,'physical_ssh_port':20473,'run_id':r['run_id'],'source_commit':head,'service':record,'launch_placement':launch_after,'resume_api':resume,'reused_service':reused_service,'dispatch_hold_release':hold_release,'committed_rows_retained':len(old_by_key),'committed_rows_after':after['committed_count'],'release_receipt_sha256':r['release_sha256'],'observed_at':now(),'other_logical_services_untouched':True,'timing_scope':'physical placement/concurrency changed; do not mix with original74 timings'}
 put(receipt_dir/'activation_receipt.json',receipt);print(json.dumps(receipt,ensure_ascii=False))
finally:lock.close()
'''


def request(**fields):
    return {"root":REMOTE_ROOT,"run_id":RUN_ID,"source_commit":SOURCE,"migration_id":MIGRATION_ID,
            "api_port":API_PORT,"python":ops.DQN,**fields}


def remote(physical_port,code,payload,timeout=120):
    # Raw old transport: placement_transport must not remap this safety probe.
    result=ops.transport.remote_python("REQUEST_JSON="+repr(json.dumps(payload))+"\n"+code,physical_port,timeout=timeout)
    if result['exit_code']:raise RuntimeError(result['stderr'][-1600:])
    return json.loads(result['stdout'].splitlines()[-1])


def plan():
    old=remote(OLD_PHYSICAL,INSPECT,request())
    return {"experiment_id":"S1-ND12","migration_id":MIGRATION_ID,"scope":"plan_only",
            "source_commit":SOURCE,"logical_ssh_port":LOGICAL,"old_physical_ssh_port":OLD_PHYSICAL,
            "new_physical_ssh_port":NEW_PHYSICAL,"old_facts":old,"api_stop_force_supported":False,
            "requires_old_client_release":True,"preserve_committed_rows":True,"released":False}


def validate_release(record):
    if (record.get('experiment_id')!='S1-ND12' or record.get('migration_id')!=MIGRATION_ID or
            record.get('source_commit')!=SOURCE or record.get('run_id')!=RUN_ID or
            record.get('logical_ssh_port')!=LOGICAL or record.get('released_physical_port')!=OLD_PHYSICAL or
            record.get('new_physical_ssh_port')!=NEW_PHYSICAL or record.get('queued_count')!=0 or
            not all(record.get(key) is True for key in ('released','old_owner_absent','old_service_closed','old_worker_exited'))):
        raise ValueError('completed exact old-owner release hard gate required')
    return record


def release(client_receipt,client_sha,*,allow_force=False,cooperative_seconds=10,reconcile_existing_stop=False,
            reconciliation_proof=None,reconciliation_sha256=None):
    if not 8<=cooperative_seconds<=600:raise ValueError("cooperative wait must be8..600s")
    verify_client_release(client_receipt,client_sha)
    proof=None
    if reconcile_existing_stop:
        if reconciliation_proof is None or digest(reconciliation_proof)!=reconciliation_sha256:
            raise ValueError('pinned reconciliation proof required')
        proof=read(reconciliation_proof)
    with worker_lock(LOCAL):
        result=remote(OLD_PHYSICAL,RELEASE,request(client_release_sha256=client_sha,allow_force=allow_force,
                      cooperative_seconds=cooperative_seconds,reconcile_existing_stop=reconcile_existing_stop,
                      reconciliation_proof=proof,reconciliation_sha256=reconciliation_sha256),timeout=cooperative_seconds+60)
        validate_release(result)
        save(LOCAL/'release_receipt.json',result)
        return result


def activate(client_receipt,client_sha,release_receipt,release_sha,*,workers=2):
    verify_client_release(client_receipt,client_sha)
    if digest(release_receipt)!=release_sha:raise ValueError("local release receipt SHA differs")
    released=validate_release(read(release_receipt))
    if workers not in (1,2,4):raise ValueError("bounded service workers1/2/4 only")
    # Fresh old-host proof prevents stale released=true from granting dual ownership.
    check=remote(OLD_PHYSICAL,COMMON+r"print(json.dumps({'listening':listening(),'processes':nd12_processes(),'queued':queue()}))",request())
    if check['listening'] or check['processes']:raise ValueError("old physical owner reappeared")
    with worker_lock(LOCAL):
        result=remote(NEW_PHYSICAL,ACTIVATE,request(client_release_sha256=client_sha,release_sha256=release_sha,workers=workers),timeout=120)
        save(LOCAL/'activation_receipt.json',result)
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('plan','release','activate'))
    parser.add_argument('--client-release-receipt',type=Path)
    parser.add_argument('--client-release-sha256')
    parser.add_argument('--allow-force-native-stop',action='store_true')
    parser.add_argument('--reconcile-existing-stop',action='store_true')
    parser.add_argument('--reconciliation-proof',type=Path)
    parser.add_argument('--reconciliation-sha256')
    parser.add_argument('--release-receipt',type=Path)
    parser.add_argument('--release-sha256')
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if args.action!='plan' and not(args.client_release_receipt and args.client_release_sha256):parser.error('pinned old-client release required')
    if args.action=='plan':value=plan()
    elif args.action=='release':value=release(args.client_release_receipt,args.client_release_sha256,allow_force=args.allow_force_native_stop,
        reconcile_existing_stop=args.reconcile_existing_stop,reconciliation_proof=args.reconciliation_proof,
        reconciliation_sha256=args.reconciliation_sha256)
    else:
        if not(args.release_receipt and args.release_sha256):parser.error('pinned old-owner release required')
        value=activate(args.client_release_receipt,args.client_release_sha256,args.release_receipt,args.release_sha256,workers=args.workers)
    print(json.dumps({k:value.get(k) for k in ('scope','stage','migration_id','released','logical_ssh_port','physical_ssh_port','run_id')},ensure_ascii=False))


if __name__=='__main__':main()
