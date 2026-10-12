"""Deploy explicit committed scientific epochs; never submits a science job."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
from satellite_rl.training.run_monitor import atomic_json
from . import ops

SOURCE = Path(__file__).resolve().parents[2]


def git(*args):
    return subprocess.run(['git', '-C', str(SOURCE), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def remote(port, code, timeout=90):
    result = ops.remote_python(code, port, timeout=timeout)
    if result['exit_code']: raise RuntimeError(f'{port}: '+result['stderr'][-1000:])
    return json.loads(result['stdout'].splitlines()[-1])


def lane(port, commit, bundle, workers):
    info = ops.instance(port)
    remote(port, f'import pathlib,json;pathlib.Path({info["ops"]!r}).mkdir(parents=True,exist_ok=True);print(json.dumps({{"prepared":True}}))')
    target = f'{info["ops"]}/ai17_{commit}.bundle'
    receipt = ops.upload(bundle, target, port)
    if receipt['local_sha256'] != receipt['remote']:raise ValueError('uploaded scientific bundle SHA differs')
    code = f'''import datetime,json,os,pathlib,socket,subprocess,sys,time,urllib.request
root=pathlib.Path({info['root']!r});repo=root/'repo';commit={commit!r};api={info['api_port']}
def git(*args):return subprocess.run(['git','-C',str(repo),*args],capture_output=True,text=True,check=True).stdout.strip()
with socket.socket() as s:
 s.settimeout(.5);busy=s.connect_ex(('127.0.0.1',api))==0
if busy:
 registry=json.loads((root/'ops/service.json').read_text())
 raw=(pathlib.Path('/proc')/str(registry['pid'])/'stat').read_text();birth=int(raw[raw.rfind(')')+2:].split()[19])
 cmd=(pathlib.Path('/proc')/str(registry['pid'])/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
 if registry.get('source_commit')!=commit or birth!=registry['birth_ticks'] or str(root) not in cmd:raise RuntimeError('occupied port is not the expected original service')
 print(json.dumps({{'reused_service':True,'registry':registry}}));sys.exit(0)
if repo.exists():
 if not (repo/'.git').exists():raise RuntimeError('existing target is not Git')
 if git('status','--porcelain'):raise RuntimeError('existing checkout is dirty')
 if git('rev-parse','HEAD')!=commit:raise RuntimeError('different epoch requires separate immutable checkout/service migration')
else:
 subprocess.run(['git','clone','--quiet','--no-checkout',{target!r},str(repo)],check=True)
 subprocess.run(['git','-C',str(repo),'checkout','--quiet',commit],check=True)
if git('rev-parse','HEAD')!=commit:raise RuntimeError('checkout SHA mismatch')
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(repo/'src'),SATELLITE_RL_SSH_PORT={str(port)!r},SATELLITE_RL_OUTPUT_ROOT=str(root/'archive'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
identity=subprocess.run([{ops.DQN!r},'-B','-c','import json,pathlib,torch,satellite_rl;print(json.dumps({{"python":__import__("sys").executable,"cuda":torch.cuda.is_available(),"build":torch.version.cuda,"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"package":str(pathlib.Path(satellite_rl.__file__).resolve())}}))'],cwd=repo,env=env,capture_output=True,text=True,check=True,timeout=35)
identity=json.loads(identity.stdout)
if identity['package']!=str(repo/'src/satellite_rl/__init__.py') or not identity['cuda'] or '4090 D' not in identity['gpu']:raise RuntimeError('new frozen package/DQN/CUDA mismatch')
with (root/'ops/workbench.log').open('ab',buffering=0) as stream:
 child=subprocess.Popen([{ops.DQN!r},'-B','-u','-m','satellite_rl.workbench.server','--archive-root',str(root/'archive'),'--port',str(api),'--python',{ops.DQN!r},'--max-workers',{str(workers)!r}],cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
raw=(pathlib.Path('/proc')/str(child.pid)/'stat').read_text();birth=int(raw[raw.rfind(')')+2:].split()[19])
registry={{'group':'S1-AI17','pid':child.pid,'birth_ticks':birth,'source_commit':commit,'source_root':str(repo),'archive_root':str(root/'archive'),'ssh_port':{port},'api_port':api,'max_workers':{workers},'identity':identity,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'science_jobs_submitted':0}}
from satellite_rl.training.run_monitor import atomic_json
atomic_json(root/'ops/service.json',registry)
for _ in range(35):
 if child.poll() is not None:raise RuntimeError('workbench exited; retained log and registry')
 try:
  with urllib.request.urlopen('http://127.0.0.1:'+str(api)+'/api/ai-search/schema',timeout=1) as r:schema=json.load(r)
  if schema.get('group')=='S1-AI17':break
 except OSError:time.sleep(1)
else:raise RuntimeError('new API not ready; retained live service registry')
print(json.dumps({{'registry':registry,'schema':schema}}))
'''
    # New deployment defaults to three free lanes; 74 needs a separate return proof.
    return {'bundle': receipt, 'service': remote(port, code, timeout=90)}


def deploy(commit, ports=(20418, 20424, 20473), workers=2):
    if len(commit) != 40 or any(x not in '0123456789abcdef' for x in commit): raise ValueError('exact commit required')
    if git('rev-parse', 'HEAD') != commit or git('status', '--porcelain', '--', 'src', 'scripts', 'tests', 'pyproject.toml'):
        raise ValueError('deployment requires verified committed clean scientific/ops code')
    if 20474 in ports: raise ValueError('74 deployment admission requires a verified final F20 return; use explicit handoff after proof')
    if not ports or set(ports)-set(ops.SSH_PORTS): raise ValueError('invalid lanes')
    bundle = ops.LOCAL / 'deployment' / f'ai17_{commit}.bundle'; bundle.parent.mkdir(parents=True, exist_ok=True)
    if not bundle.exists(): subprocess.run(['git', '-C', str(SOURCE), 'bundle', 'create', str(bundle), 'HEAD'], check=True)
    subprocess.run(['git', '-C', str(SOURCE), 'bundle', 'verify', str(bundle)], check=True, capture_output=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda p: (str(p), lane(p, commit, bundle, workers)), ports))
    receipt = {'scientific_commit': commit, 'lanes': dict(results), 'science_jobs_submitted': 0}
    atomic_json(ops.LOCAL / 'deployment' / f'deployed_{commit}.json', receipt)
    return receipt


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--commit',required=True)
    parser.add_argument('--ports',nargs='+',type=int,default=[20418,20424,20473]);parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();print(json.dumps(deploy(args.commit,tuple(args.ports),args.workers),ensure_ascii=False))
