"""Deploy a clean frozen S1-V10 commit and start isolated four-lane workbenches.

This tool never submits scientific jobs. Source archives under the sibling
``source`` directory remain separate from each lane's repo, archive and lock.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess

from scripts.stage1_variable import ops


SOURCE = Path(__file__).resolve().parents[2]


def git(*args):
    return subprocess.run(["git", "-C", str(SOURCE), *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def check_commit(commit):
    if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        raise ValueError("explicit forty-character Git commit required")


def remote(port, code, timeout=120):
    result = ops.remote_python(code, port, timeout=timeout)
    if result["exit_code"]:
        raise RuntimeError(f"{port}: {result['stderr'][-900:]}")
    return json.loads(result["stdout"].splitlines()[-1])


def plan_lane(port):
    info = ops.instance(port)
    code = f'''
import datetime,json,pathlib,socket,subprocess
root=pathlib.Path({info['root']!r});repo=root/'repo';api={info['api_port']}
with socket.socket() as sock:
 sock.settimeout(2);busy=sock.connect_ex(('127.0.0.1',api))==0
head=None;dirty=None
if (repo/'.git').is_dir():
 head=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()
 dirty=subprocess.run(['git','-C',str(repo),'status','--porcelain'],capture_output=True,text=True).stdout.strip()
print(json.dumps({{'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'root_exists':root.exists(),'repo_head':head,'repo_dirty':dirty,
 'api_port':api,'api_port_busy':busy}}))
'''
    return remote(port, code)


def plan():
    result = {"local_head": git("rev-parse", "HEAD"),
              "local_dirty": bool(git("status", "--porcelain")), "lanes": {}}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(plan_lane, port): port for port in ops.SSH_PORTS}
        for future in as_completed(futures):
            result["lanes"][str(futures[future])] = future.result()
    path = ops.LOCAL / "preflight" / "deployment_plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def deploy_lane(port, commit, bundle):
    info = ops.instance(port)
    remote(port, f"import json,pathlib;p=pathlib.Path({info['ops']!r});p.mkdir(parents=True,exist_ok=True);print(json.dumps({{'ops':str(p)}}))")
    target = f"{info['ops']}/s1v10_{commit}.bundle"
    receipt = ops.upload(bundle, target, port)
    if receipt["local_sha256"] != receipt["remote"]:
        raise RuntimeError(f"{port}: source bundle SHA differs")
    code = f'''
import json,pathlib,subprocess
repo=pathlib.Path({info['repo']!r});bundle=pathlib.Path({target!r});commit={commit!r}
if repo.exists() and not (repo/'.git').is_dir():
 if any(repo.iterdir()):raise RuntimeError('nonempty non-Git target checkout')
 repo.rmdir()
if not repo.exists():subprocess.run(['git','clone','--quiet',str(bundle),str(repo)],check=True)
head=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
if head!=commit:
 subprocess.run(['git','-C',str(repo),'fetch','--quiet',str(bundle),'HEAD'],check=True)
 subprocess.run(['git','-C',str(repo),'checkout','--quiet',commit],check=True)
head=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
dirty=subprocess.run(['git','-C',str(repo),'status','--porcelain'],capture_output=True,text=True,check=True).stdout.strip()
if head!=commit or dirty:raise RuntimeError('frozen checkout differs')
print(json.dumps({{'repo':str(repo),'head':head,'dirty':False}}))
'''
    return {"source_bundle_sha256": receipt["remote"],
            "checkout": remote(port, code, timeout=600)}


def deploy(commit):
    check_commit(commit)
    if git("rev-parse", "HEAD") != commit or git("status", "--porcelain"):
        raise RuntimeError("frozen local checkout must be clean and at supplied commit")
    observed = plan()
    for port, state in observed["lanes"].items():
        if state["api_port_busy"] or state["repo_dirty"] or state["repo_head"] not in (None, commit):
            raise RuntimeError(f"{port}: active port or different checkout requires ownership audit")
    bundle = ops.LOCAL / "deployment" / f"s1v10_{commit}.bundle"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    if not bundle.exists():
        subprocess.run(["git", "-C", str(SOURCE), "bundle", "create", str(bundle), "HEAD"], check=True)
    subprocess.run(["git", "-C", str(SOURCE), "bundle", "verify", str(bundle)],
                   check=True, capture_output=True, text=True)
    result = {"scientific_commit": commit, "lanes": {}}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(deploy_lane, p, commit, bundle): p for p in ops.SSH_PORTS}
        for future in as_completed(futures):
            result["lanes"][str(futures[future])] = future.result()
    path = ops.LOCAL / "preflight" / f"deployment_{commit[:12]}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def serve_lane(port, commit, workers):
    info = ops.instance(port)
    code = f'''
import datetime,json,os,pathlib,socket,subprocess,time
repo=pathlib.Path({info['repo']!r});root=pathlib.Path({info['root']!r});registry=root/'ops/service.json'
head=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
dirty=subprocess.run(['git','-C',str(repo),'status','--porcelain'],capture_output=True,text=True,check=True).stdout.strip()
if head!={commit!r} or dirty:raise RuntimeError('frozen repo differs')
with socket.socket() as sock:
 sock.settimeout(2)
 if sock.connect_ex(('127.0.0.1',{info['api_port']}))==0:raise RuntimeError('reserved API port already used')
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(repo/'src'),
 SATELLITE_RL_SSH_PORT={str(port)!r},SATELLITE_RL_OUTPUT_ROOT=str(root),
 OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
identity=subprocess.run([{ops.DQN!r},'-c',
 'import json,pathlib,torch,satellite_rl;print(json.dumps({{"package":str(pathlib.Path(satellite_rl.__file__).resolve()),"cuda":torch.cuda.is_available(),"build":torch.version.cuda,"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}))'],
 cwd=repo,env=env,capture_output=True,text=True,check=True,timeout=45)
package=json.loads(identity.stdout)
if (package['package']!=str((repo/'src/satellite_rl/__init__.py').resolve()) or
 not package['cuda'] or '4090 D' not in package['gpu']):raise RuntimeError('DQN/CUDA/source/device mismatch')
log=root/'ops/workbench.log'
with log.open('ab',buffering=0) as stream:
 child=subprocess.Popen([{ops.DQN!r},'-B','-m','satellite_rl.workbench.server',
  '--archive-root',str(root),'--port',{str(info['api_port'])!r},
  '--python',{ops.DQN!r},'--max-workers',{str(workers)!r}],
  cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
  start_new_session=True,close_fds=True)
raw=(pathlib.Path('/proc')/str(child.pid)/'stat').read_text()
ticks=int(raw[raw.rfind(')')+2:].split()[19])
for _ in range(45):
 time.sleep(1)
 with socket.socket() as sock:
  sock.settimeout(2)
  if sock.connect_ex(('127.0.0.1',{info['api_port']}))==0:break
else:raise RuntimeError('workbench did not listen')
value={{'status':'serving_no_jobs','pid':child.pid,'start_ticks':ticks,
 'source_commit':head,'source_root':str(repo),'archive_root':str(root),
 'api_port':{info['api_port']},'ssh_port':{port},'max_workers':{workers},
 'package':package,'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}}
registry.write_text(json.dumps(value,indent=2)+'\\n')
print(json.dumps(value))
'''
    return remote(port, code, timeout=120)


def serve(commit, workers):
    check_commit(commit)
    receipt_path = ops.LOCAL / "preflight" / f"deployment_{commit[:12]}.json"
    if not receipt_path.is_file() or json.loads(receipt_path.read_text())["scientific_commit"] != commit:
        raise RuntimeError("verified four-lane deployment receipt required")
    if not 1 <= workers <= 4:
        raise ValueError("worker count must be 1–4 per GPU")
    result = {"scientific_commit": commit, "workers_per_lane": workers, "lanes": {}}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(serve_lane, p, commit, workers): p for p in ops.SSH_PORTS}
        for future in as_completed(futures):
            result["lanes"][str(futures[future])] = future.result()
    path = ops.LOCAL / "preflight" / f"workbench_{commit[:12]}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "deploy", "serve"))
    parser.add_argument("--commit")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.action != "plan" and not args.commit:
        parser.error("--commit required")
    result = plan() if args.action == "plan" else deploy(args.commit) if args.action == "deploy" else serve(args.commit, args.workers)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
