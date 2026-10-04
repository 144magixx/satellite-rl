"""Stage immutable S1-03 Git sources and start only its isolated workbenches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from remote import DQN, HOST, LOCAL, PORTS, REMOTE_BASE, connect, run_code, save, upload_once


REPOSITORY = Path(__file__).resolve().parents[2]
OLD_SOURCE = "/home/u2024110224/first_stage_satellite_rl/s1_02/20418/repo"
OLD_COMMIT = "e22f449ccee11d525bce1c8011f297dc87bc86e1"
EXPECTED_GPUS = {20418: "GPU-638e7ef5-3af6-87fa-f240-f2be8161bd8a",
                 20424: "GPU-a59df44a-9d97-8c5a-3b74-ebf38d7a7e0b"}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPOSITORY)


def stage(client, port, commit):
    root = f"{REMOTE_BASE}/{port}"
    if git("status", "--porcelain", "--", "src", "data", "pyproject.toml").strip():
        raise RuntimeError("Scientific source must be clean and committed")
    base = git("merge-base", commit, OLD_COMMIT).decode().strip()
    bundle = LOCAL / "preflight" / ("source_" + commit + ".bundle")
    if not bundle.is_file():
        subprocess.run(["git", "bundle", "create", str(bundle), commit, "^"+base],
                       cwd=REPOSITORY, check=True)
    run_code(client, f"""
import pathlib,json
root=pathlib.Path({root!r})
for sub in ('setup','ops'): (root/sub).mkdir(parents=True,exist_ok=True)
print(json.dumps({{'root':str(root)}}))
""")
    incoming = root + "/setup/" + bundle.name
    with client.open_sftp() as sftp:
        transfer = upload_once(sftp, bundle, incoming)
    result = run_code(client, f"""
import pathlib,subprocess,json,datetime
root=pathlib.Path({root!r});repo=root/'repo';old=pathlib.Path({OLD_SOURCE!r})
def call(argv):
 p=subprocess.run(argv,text=True,capture_output=True,timeout=120)
 if p.returncode: raise RuntimeError(p.stderr[-2000:])
 return p.stdout.strip()
if not (repo/'.git').is_dir():
 call(['git','-C',str(old),'cat-file','-e',{base!r}+'^{{commit}}'])
 call(['git','clone','--no-hardlinks','--no-checkout',str(old),str(repo)])
current=call(['git','-C',str(repo),'rev-parse','HEAD'])
if current!={commit!r}:
 registry=root/'ops/current_workbench_service.json'
 if registry.exists():
  record=json.loads(registry.read_text());p=pathlib.Path('/proc')/str(record.get('pid'))
  if p.exists():
   raw=(p/'stat').read_text();birth=int(raw[raw.rfind(')')+2:].split()[19])
   if birth==record.get('start_ticks'): raise RuntimeError('Cannot change source under an active S1-03 service')
 if call(['git','-C',str(repo),'status','--porcelain']):raise RuntimeError('Existing isolated repo has local modifications')
 call(['git','-C',str(repo),'fetch',{incoming!r},{commit!r}])
 call(['git','-C',str(repo),'checkout','--detach',{commit!r}])
actual=call(['git','-C',str(repo),'rev-parse','HEAD'])
if actual!={commit!r} or call(['git','-C',str(repo),'status','--porcelain','--','src','data','pyproject.toml']):
 raise RuntimeError('Remote scientific source did not freeze cleanly')
print(json.dumps({{'commit':actual,'repo':str(repo),'base':{base!r},'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}}))
""", timeout=180)
    return {"transfer": transfer, "source": result}


def start(client, port, commit):
    root, api_port = f"{REMOTE_BASE}/{port}", {20418:8894,20424:8895}[port]
    code = f"""
import pathlib,subprocess,json,os,socket,time,datetime,hashlib
root=pathlib.Path({root!r});repo=root/'repo';registry=root/'ops/current_workbench_service.json'
def birth(pid):
 p=pathlib.Path('/proc')/str(pid)
 if not p.exists():return None
 raw=(p/'stat').read_text();return int(raw[raw.rfind(')')+2:].split()[19])
if registry.exists():
 old=json.loads(registry.read_text())
 if birth(old['pid'])==old['start_ticks']:
  if old['source_commit']!={commit!r}:raise RuntimeError('Active service uses another source; no hot replacement')
  print(json.dumps({{'status':'already_running','registry':old}}));raise SystemExit
with socket.socket() as sock:
 if sock.connect_ex(('127.0.0.1',{api_port}))==0:raise RuntimeError('Reserved workbench port is already in use')
env=os.environ.copy();env.update(PYTHONPATH=str(repo/'src'),SATELLITE_RL_SSH_PORT={str(port)!r},
 OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
 SATELLITE_RL_OUTPUT_ROOT=str(root),PYTHONUNBUFFERED='1')
probe="import json,sys,torch,satellite_rl;print(json.dumps({{'python':sys.executable,'prefix':sys.prefix,'torch':torch.__version__,'cuda':torch.cuda.is_available(),'cuda_build':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'package_file':satellite_rl.__file__}}))"
p=subprocess.run([{DQN!r},'-B','-c',probe],env=env,cwd=repo,text=True,capture_output=True,timeout=45)
if p.returncode:raise RuntimeError(p.stderr[-2000:])
preflight=json.loads(p.stdout)
gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],text=True,capture_output=True,check=True,timeout=15).stdout.strip()
if (not preflight['cuda'] or not preflight['package_file'].startswith(str(repo/'src')) or
    {EXPECTED_GPUS[port]!r} not in gpu):raise RuntimeError('DQN/CUDA/source/device identity mismatch')
actual=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
if actual!={commit!r}:raise RuntimeError('Source HEAD differs from approved revision')
argv=[{DQN!r},'-B','-m','satellite_rl.workbench.server','--archive-root',str(root),
 '--port',{str(api_port)!r},'--python',{DQN!r},'--max-workers','1']
with (root/'ops/workbench.log').open('ab',buffering=0) as stream:
 child=subprocess.Popen(argv,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=stream,
  stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
value={{'pid':child.pid,'start_ticks':birth(child.pid),'source_commit':actual,
 'source_root':str(repo),'archive_root':str(root),'ssh_port':{port},'api_port':{api_port},
 'python':{DQN!r},'preflight':preflight,'gpu_at_launch':gpu,
 'server_module_sha256':hashlib.sha256((repo/'src/satellite_rl/workbench/server.py').read_bytes()).hexdigest(),
 'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'max_workers':1,
 'schedule_note':'One approved PPO; parallelism across isolated hosts, nonlearning timing exclusive within each host'}}
if registry.exists():
 backup=registry.with_name('workbench_previous_'+str(int(time.time()))+'.json');backup.write_bytes(registry.read_bytes())
registry.write_text(json.dumps(value,indent=2)+'\\n')
print(json.dumps({{'status':'launched','registry':value}}))
"""
    return run_code(client, code, timeout=90)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("stage","start"))
    parser.add_argument("--port", type=int, choices=PORTS, required=True)
    parser.add_argument("--commit", required=True)
    args=parser.parse_args()
    commit=git("rev-parse",args.commit).decode().strip()
    client=connect(args.port)
    try:
        result=stage(client,args.port,commit) if args.mode=='stage' else start(client,args.port,commit)
    finally:
        client.close()
    path=LOCAL/'preflight'/f'{args.mode}_{args.port}_{commit[:12]}.json'
    save(path,result)
    print(json.dumps({'receipt':str(path),'result':result},ensure_ascii=False))


if __name__=='__main__':main()
