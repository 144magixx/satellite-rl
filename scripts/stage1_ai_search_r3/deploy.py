"""Prepare independent committed R3 Git/workbench lanes; never submits tasks."""
from concurrent.futures import ThreadPoolExecutor
import argparse,json,subprocess
from pathlib import Path
from satellite_rl.training.run_monitor import atomic_json
from . import ops

SOURCE=Path(__file__).resolve().parents[2]


def git(*args):return subprocess.run(['git','-C',str(SOURCE),*args],capture_output=True,text=True,check=True).stdout.strip()


def remote(port,code):
    result=ops.remote_python(code,port,timeout=100)
    if result['exit_code']:raise RuntimeError(result['stderr'][-1200:])
    return json.loads(result['stdout'].splitlines()[-1])


def lane(port,commit,bundle):
    i=ops.instance(port);target=i['ops']+'/source_'+commit+'.bundle';uploaded=ops.upload(bundle,target,port)
    epoch_repo=i['root']+'/source_epochs/'+commit+'/repo'
    code=f'''import sys,json,pathlib,subprocess,os,socket,time,datetime,urllib.request
root=pathlib.Path({i['root']!r});repo=pathlib.Path({epoch_repo!r});api={i['api_port']};commit={commit!r}
with socket.socket() as sock:
 sock.settimeout(.5);occupied=sock.connect_ex(('127.0.0.1',api))==0
if occupied:
 record=json.loads((root/'ops/service.json').read_text(encoding='utf-8'));raw=(pathlib.Path('/proc')/str(record['pid'])/'stat').read_text(encoding='utf-8');birth=int(raw[raw.rfind(')')+2:].split()[19])
 command=(pathlib.Path('/proc')/str(record['pid'])/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
 if record['source_commit']!=commit or birth!=record['birth_ticks'] or str(root) not in command:raise ValueError('occupied R3 endpoint belongs to another source/process')
 with urllib.request.urlopen('http://127.0.0.1:'+str(api)+'/api/ai-search-r3/schema',timeout=3) as r:schema=json.load(r)
 if schema['group']!={ops.GROUP!r}:raise ValueError('existing API is another scientific group')
 print(json.dumps({{'reused_original_service':True,'registry':record,'schema':schema,'actual_command':command}}));sys.exit(0)
if repo.exists():
 actual=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
 dirty=subprocess.run(['git','-C',str(repo),'status','--porcelain'],capture_output=True,text=True,check=True).stdout.strip()
 if actual!=commit or dirty:raise ValueError('R3 checkout is another epoch or dirty; no hot checkout')
else:
 repo.parent.mkdir(parents=True,exist_ok=True)
 subprocess.run(['git','clone','--quiet','--no-checkout',{target!r},str(repo)],check=True)
 subprocess.run(['git','-C',str(repo),'checkout','--quiet',commit],check=True)
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(repo/'src'),SATELLITE_RL_OUTPUT_ROOT=str(root/'archive'),SATELLITE_RL_SSH_PORT={str(port)!r},OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
result=subprocess.run([{ops.DQN!r},'-B','-c','import sys,json,torch,satellite_rl;print(json.dumps({{"python":sys.executable,"package":satellite_rl.__file__,"cuda":torch.cuda.is_available(),"cuda_build":torch.version.cuda,"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}))'],cwd=repo,env=env,capture_output=True,text=True,check=True,timeout=40)
identity=json.loads(result.stdout)
if pathlib.Path(identity['package']).resolve()!=repo/'src/satellite_rl/__init__.py' or not identity['cuda']:raise ValueError('new R3 package/DQN/CUDA mismatch')
with (root/'ops/workbench.log').open('ab',buffering=0) as out:
 child=subprocess.Popen([{ops.DQN!r},'-B','-u','-m','satellite_rl.workbench.server','--archive-root',str(root/'archive'),'--port',str(api),'--python',{ops.DQN!r},'--max-workers','1'],cwd=repo,env=env,stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,close_fds=True)
raw=(pathlib.Path('/proc')/str(child.pid)/'stat').read_text(encoding='utf-8');birth=int(raw[raw.rfind(')')+2:].split()[19])
record={{'group':{ops.GROUP!r},'pid':child.pid,'birth_ticks':birth,'source_commit':commit,'source_root':str(repo),'archive_root':str(root/'archive'),'ssh_port':{port},'api_port':api,'max_workers':1,'identity':identity,'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'science_jobs_submitted':0}}
sys.path.insert(0,str(repo/'src'))
from satellite_rl.training.run_monitor import atomic_json
atomic_json(root/'ops/service.json',record)
for _ in range(35):
 if child.poll() is not None:raise ValueError('R3 service exited; original registry/log retained')
 try:
  with urllib.request.urlopen('http://127.0.0.1:'+str(api)+'/api/ai-search-r3/schema',timeout=1) as r:schema=json.load(r)
  if schema['group']=={ops.GROUP!r}:break
 except OSError:time.sleep(1)
else:raise ValueError('R3 API schema unavailable')
print(json.dumps({{'registry':record,'schema':schema}}))
'''
    return {'bundle':uploaded,'service':remote(port,code)}


def deploy(commit):
    if len(commit)!=40 or set(commit)-set('0123456789abcdef'):raise ValueError('exact science commit')
    if git('rev-parse','HEAD')!=commit or git('status','--porcelain','--','src','scripts','tests','pyproject.toml'):
        raise ValueError('R3 deployment requires root-verified committed clean source')
    folder=ops.LOCAL/'deployment';folder.mkdir(parents=True,exist_ok=True);bundle=folder/(commit+'.bundle')
    if not bundle.exists():subprocess.run(['git','-C',str(SOURCE),'bundle','create',str(bundle),'HEAD'],check=True)
    with ThreadPoolExecutor(max_workers=3) as pool:rows=dict(zip(map(str,ops.SSH_PORTS),pool.map(lambda p:lane(p,commit,bundle),ops.SSH_PORTS)))
    receipt={'group':ops.GROUP,'scientific_commit':commit,'lanes':rows,'science_submitted':0};atomic_json(folder/('deployed_'+commit+'.json'),receipt);return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--commit',required=True);args=parser.parse_args();print(json.dumps(deploy(args.commit)))
