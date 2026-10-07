"""Plan, deploy and serve a clean S2-P19 commit on the two isolated instances.

No stage submits a scientific job. ``plan`` is read-only; ``deploy`` and
``serve`` require an explicitly supplied forty-character frozen commit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from . import ops


SOURCE = Path("C:/Users/Admin/.codex/worktrees/s2-p19-sac/paper")


def git(*args: str) -> str:
    result = subprocess.run(["git", "-C", str(SOURCE), *args], capture_output=True,
                            text=True, check=True)
    return result.stdout.strip()


def check_commit(commit: str) -> None:
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError("explicit frozen commit must be a forty-character SHA")


def plan() -> dict:
    report = {"purpose": "read_only_s2p19_deploy_plan", "local_head": git("rev-parse", "HEAD"),
              "local_dirty": bool(git("status", "--porcelain")), "instances": {}}
    for port in ops.SSH_PORTS:
        info = ops.instance(port)
        code = f'''
import datetime,json,os,pathlib,shutil,socket,subprocess
root=pathlib.Path({info['root']!r});repo=root/'repo';api={info['api_port']}
capacity_path=root if root.exists() else pathlib.Path('/home/u2024110224')
with socket.socket() as sock:
 sock.settimeout(2);busy=sock.connect_ex(('127.0.0.1',api))==0
head=None;dirty=None
if (repo/'.git').is_dir():
 head=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()
 dirty=subprocess.run(['git','-C',str(repo),'status','--porcelain'],capture_output=True,text=True).stdout.strip()
print(json.dumps({{'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'root_exists':root.exists(),'repo_exists':repo.exists(),'repo_head':head,'repo_dirty':dirty,
 'api_port':api,'api_port_busy':busy,
 'archive_mount_free_bytes':shutil.disk_usage(capacity_path).free,
 'archive_mount_device':capacity_path.stat().st_dev,
 'gpu':subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu',
  '--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip()}},ensure_ascii=False))
'''
        result = ops.remote_python(code, port)
        if result["exit_code"]:
            raise RuntimeError(f"{port} deploy preflight failed: " + result["stderr"][-500:])
        report["instances"][str(port)] = json.loads(result["stdout"].splitlines()[-1])
    ops.LOCAL.mkdir(parents=True, exist_ok=True)
    path = ops.LOCAL / "preflight" / "deployment_plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def deploy(commit: str) -> dict:
    check_commit(commit)
    if git("rev-parse", "HEAD") != commit or git("status", "--porcelain"):
        raise RuntimeError("local frozen source must be clean and at the supplied commit")
    preflight = plan()
    for port, status in preflight["instances"].items():
        if status["api_port_busy"] or status["repo_dirty"]:
            raise RuntimeError(f"{port} existing service or dirty checkout needs identity audit")
        if status["repo_head"] not in (None, commit):
            raise RuntimeError(f"{port} different checkout already exists; do not replace it")
    bundle = ops.LOCAL / "deployment" / f"s2p19_{commit}.bundle"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    if not bundle.exists():
        subprocess.run(["git", "-C", str(SOURCE), "bundle", "create", str(bundle), "HEAD"], check=True)
    subprocess.run(["git", "-C", str(SOURCE), "bundle", "verify", str(bundle)],
                   check=True, capture_output=True, text=True)
    bundle_sha = ops.transport.sha256_file(bundle)
    receipt = {"scientific_commit": commit, "bundle_sha256": bundle_sha, "instances": {}}
    for port in ops.SSH_PORTS:
        info = ops.instance(port)
        prepare = ops.remote_python(f'''
import pathlib
root=pathlib.Path({info['root']!r})
for name in ('ops','data'):(root/name).mkdir(parents=True,exist_ok=True)
print('ready')
''', port)
        if prepare["exit_code"]:
            raise RuntimeError(f"{port} isolated target create failed: " + prepare["stderr"][-500:])
        remote_bundle = info["ops"] + "/" + bundle.name
        transfer = ops.upload(bundle, remote_bundle, port)
        if transfer["local_sha256"] != transfer["remote"] or transfer["remote"] != bundle_sha:
            raise RuntimeError(f"{port} Git bundle SHA mismatch")
        code = f'''
import json,pathlib,subprocess
repo=pathlib.Path({info['repo']!r});bundle=pathlib.Path({remote_bundle!r});commit={commit!r}
if repo.exists() and not (repo/'.git').is_dir():
 if any(repo.iterdir()):raise RuntimeError('nonempty non-Git target checkout')
 repo.rmdir()
if not repo.exists():
 subprocess.run(['git','clone','--quiet',str(bundle),str(repo)],check=True)
head=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
if head!=commit:
 subprocess.run(['git','-C',str(repo),'fetch','--quiet',str(bundle),'HEAD'],check=True)
 subprocess.run(['git','-C',str(repo),'checkout','--quiet',commit],check=True)
head=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
dirty=subprocess.run(['git','-C',str(repo),'status','--porcelain'],capture_output=True,text=True,check=True).stdout.strip()
if head!=commit or dirty:raise RuntimeError('remote frozen checkout differs')
print(json.dumps({{'repo':str(repo),'head':head,'dirty':False}}))
'''
        result = ops.remote_python(code, port, timeout=600)
        if result["exit_code"]:
            raise RuntimeError(f"{port} checkout failed: " + result["stderr"][-700:])
        receipt["instances"][str(port)] = {
            "target": info["target"], "repo": info["repo"],
            "archive": info["archive"], "api_port": info["api_port"],
            "bundle": remote_bundle, "bundle_sha256": transfer["remote"],
            "checkout": json.loads(result["stdout"].splitlines()[-1])}
    path = ops.LOCAL / "preflight" / f"deployment_{commit[:12]}.json"
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


def serve(commit: str, workers_20418: int, workers_20424: int) -> dict:
    check_commit(commit)
    saved = ops.LOCAL / "preflight" / f"deployment_{commit[:12]}.json"
    if not saved.is_file() or json.loads(saved.read_text(encoding="utf-8"))["scientific_commit"] != commit:
        raise RuntimeError("missing verified deployment receipt")
    results = {}
    for port, workers in ((20418, workers_20418), (20424, workers_20424)):
        if workers not in (1, 2):
            raise ValueError("workbench max_workers must be 1 or 2 per GPU")
        info = ops.instance(port)
        code = f'''
import json,os,pathlib,socket,subprocess,time
repo=pathlib.Path({info['repo']!r});archive=pathlib.Path({info['archive']!r})
head=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
dirty=subprocess.run(['git','-C',str(repo),'status','--porcelain'],capture_output=True,text=True,check=True).stdout.strip()
if head!={commit!r} or dirty:raise RuntimeError('frozen repo changed')
with socket.socket() as sock:
 sock.settimeout(2)
 if sock.connect_ex(('127.0.0.1',{info['api_port']}))==0:
  raise RuntimeError('S2-P19 port already busy; inspect owner before serve')
archive.mkdir(parents=True,exist_ok=True)
log=pathlib.Path({info['ops']!r})/'workbench.log';log.parent.mkdir(parents=True,exist_ok=True)
deps=pathlib.Path({ops.PRIORITY_DEPS!r})
if not deps.is_dir():raise RuntimeError('pinned SCA solver dependency directory missing')
for package,version in (('cvxpy','1.6.5'),('clarabel','0.11.1')):
 if not (deps/(package+'-'+version+'.dist-info')/'METADATA').is_file():
  raise RuntimeError('pinned SCA solver package metadata missing: '+package)
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',
 PYTHONPATH=str(repo/'src')+os.pathsep+str(deps),SATELLITE_RL_PRIORITY_DEPS=str(deps),
 OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
identity=subprocess.run([{ops.DQN!r},'-c',
 'import json,pathlib,torch,cvxpy,clarabel,satellite_rl;print(json.dumps({{"package":str(pathlib.Path(satellite_rl.__file__).resolve()),"cuda":torch.cuda.is_available(),"build":torch.version.cuda,"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"cvxpy":cvxpy.__version__,"clarabel":clarabel.__version__}}))'],
 cwd=str(repo),env=env,capture_output=True,text=True,check=True)
package=json.loads(identity.stdout)
if (package['package']!=str((repo/'src/satellite_rl/__init__.py').resolve()) or not package['cuda']
 or package['cvxpy']!='1.6.5' or package['clarabel']!='0.11.1'):
 raise RuntimeError('DQN does not import frozen S2-P19 source on CUDA')
with log.open('ab',buffering=0) as output:
 process=subprocess.Popen([{ops.DQN!r},'-B','-m','satellite_rl.workbench.server',
  '--archive-root',str(archive),'--port',str({info['api_port']}),
  '--python',{ops.DQN!r},'--max-workers',str({workers})],
  cwd=str(repo),env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=output,
  start_new_session=True,close_fds=True)
stat=(pathlib.Path('/proc')/str(process.pid)/'stat').read_text()
ticks=int(stat[stat.rfind(')')+2:].split()[19])
for _ in range(60):
 time.sleep(1)
 with socket.socket() as sock:
  sock.settimeout(2)
  if sock.connect_ex(('127.0.0.1',{info['api_port']}))==0:break
else:raise RuntimeError('workbench did not listen within 60s')
print(json.dumps({{'status':'serving_no_jobs','pid':process.pid,'start_ticks':ticks,
 'repo_head':head,'archive':str(archive),'api_port':{info['api_port']},
 'max_workers':{workers},'package':package,'log':str(log)}}))
'''
        result = ops.remote_python(code, port, timeout=120)
        if result["exit_code"]:
            raise RuntimeError(f"{port} service failed: " + result["stderr"][-700:])
        results[str(port)] = json.loads(result["stdout"].splitlines()[-1])
    receipt = {"scientific_commit": commit, "instances": results}
    path = ops.LOCAL / "preflight" / f"workbench_{commit[:12]}.json"
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "deploy", "serve"))
    parser.add_argument("--commit")
    parser.add_argument("--workers-20418", type=int, default=1)
    parser.add_argument("--workers-20424", type=int, default=2)
    args = parser.parse_args()
    if args.action != "plan" and not args.commit:
        parser.error("--commit is required for deploy/serve")
    output = (plan() if args.action == "plan" else
              deploy(args.commit) if args.action == "deploy" else
              serve(args.commit, args.workers_20418, args.workers_20424))
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
