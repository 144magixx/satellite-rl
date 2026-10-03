"""Deploy one clean frozen stage 3 Git commit to two isolated remote checkouts.

The script prepares source only; it does not start a workbench or submit jobs.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


sys.path.insert(0, "D:/paper/outputs/verification/g11_remote_launch_20261001")
from remote_transport import connect, execute, paramiko  # noqa: E402


BASE = "/home/u2024110224/third_stage_satellite_rl/s3_si_priority_20261003"
DQN = "/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python"
PREFLIGHT = Path("D:/paper/outputs/experiments/stage3_si_priority_20261003/preflight")
REF = "refs/heads/codex/stage3-si-priority"


def git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *arguments], capture_output=True,
                            text=True, check=True)
    return result.stdout.strip()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def remote_python(transport, code: str, timeout: int = 300) -> dict:
    payload = base64.b64encode(code.encode()).decode()
    command = "python3 -c 'exec(__import__(\"base64\").b64decode(\"" + payload + "\"))'"
    result = execute(transport, command, timeout=timeout)
    if result["exit_code"]:
        raise RuntimeError(result["stderr"][-1500:])
    return json.loads(result["stdout"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--replace-clean-unstarted", action="store_true",
                        help="Advance a clean unused stage3 checkout to a descendant frozen commit")
    args = parser.parse_args()
    commit = args.commit.lower()
    if not re.fullmatch("[0-9a-f]{40}", commit):
        raise ValueError("Full frozen commit required")
    source = args.source_repo.resolve()
    if git(source, "rev-parse", "HEAD") != commit or git(source, "status", "--porcelain"):
        raise RuntimeError("Source worktree must be clean at the requested commit")
    PREFLIGHT.mkdir(parents=True, exist_ok=True)
    bundle = PREFLIGHT / f"stage3_{commit}.bundle"
    if not bundle.exists():
        subprocess.run(["git", "-C", str(source), "bundle", "create", str(bundle), "HEAD"], check=True)
    subprocess.run(["git", "-C", str(source), "bundle", "verify", str(bundle)],
                   capture_output=True, text=True, check=True)
    bundle_sha = sha(bundle)
    remote_bundle = BASE + f"/shared_data/ops/stage3_{commit}.bundle"
    transport = connect(20418)
    try:
        with paramiko.SFTPClient.from_transport(transport) as sftp:
            sftp.put(str(bundle), remote_bundle + ".part")
        code = f'''
import hashlib,json,pathlib
part=pathlib.Path({(remote_bundle + '.part')!r});target=pathlib.Path({remote_bundle!r})
h=hashlib.sha256()
with part.open('rb') as source:
 for block in iter(lambda:source.read(8*1024*1024),b''):h.update(block)
if h.hexdigest()!={bundle_sha!r}:raise RuntimeError('Git bundle upload SHA mismatch')
if target.exists():
 if hashlib.sha256(target.read_bytes()).hexdigest()!=h.hexdigest():raise RuntimeError('Existing bundle SHA differs')
 part.unlink()
else:part.rename(target)
print(json.dumps({{'path':str(target),'sha256':h.hexdigest()}}))
'''
        uploaded = remote_python(transport, code)
    finally:
        transport.close()
    installed = {}
    for port in (20418, 20424):
        code = f'''
import datetime,json,os,pathlib,socket,subprocess
root=pathlib.Path({BASE!r})/{str(port)!r}
bare=root/'receive.git';repo=root/'repo';bundle=pathlib.Path({remote_bundle!r})
commit={commit!r};ref={REF!r};dqn={DQN!r}
def run(args,**kwargs):return subprocess.run(args,capture_output=True,text=True,check=True,**kwargs).stdout.strip()
gpu=run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader']).strip()
if not gpu.startswith('GPU-'):raise RuntimeError('GPU UUID unavailable')
if not (bare/'HEAD').is_file() or not repo.is_dir():raise RuntimeError('Stage3 isolated Git directories not initialized')
if (repo/'.git').exists():
 head=run(['git','-C',str(repo),'rev-parse','HEAD'])
 dirty=run(['git','-C',str(repo),'status','--porcelain'])
 if dirty:raise RuntimeError('Existing stage3 checkout is dirty')
 if head!=commit:
  if not {args.replace_clean_unstarted!r}:raise RuntimeError('Existing stage3 checkout differs; explicit unused replacement flag required')
  wb_port={8776 if port==20418 else 8777}
  with socket.socket() as probe:
   probe.settimeout(1)
   if probe.connect_ex(('127.0.0.1',wb_port))==0:raise RuntimeError('Stage3 workbench is already listening')
  if list((root/'archive/outputs/workbench/jobs').glob('*/job.json')):
   raise RuntimeError('Stage3 formal job already exists; frozen checkout cannot be replaced')
  if list((root/'ops').glob('workbench_*_launch.json')):
   raise RuntimeError('Stage3 workbench launch receipt exists; inspect before replacement')
  incoming='refs/stage3/incoming_'+commit[:12]
  run(['git','-C',str(bare),'fetch',str(bundle),'HEAD:'+incoming])
  if subprocess.run(['git','-C',str(bare),'merge-base','--is-ancestor',head,commit]).returncode!=0:
   raise RuntimeError('Replacement commit does not descend from unused stage3 checkout')
  run(['git','-C',str(bare),'update-ref',ref,commit])
  run(['git','-C',str(repo),'fetch',str(bare),ref])
  run(['git','-C',str(repo),'checkout','--detach',commit])
else:
 run(['git','-C',str(bare),'fetch',str(bundle),'HEAD:'+ref])
 run(['git','-C',str(bare),'symbolic-ref','HEAD',ref])
 run(['git','clone','--no-checkout',str(bare),str(repo)])
 run(['git','-C',str(repo),'checkout','--detach',commit])
head=run(['git','-C',str(repo),'rev-parse','HEAD'])
dirty=run(['git','-C',str(repo),'status','--porcelain'])
if head!=commit or dirty:raise RuntimeError('Deployed checkout differs from frozen commit')
deps=pathlib.Path({(BASE + '/shared_data/deps/cvxpy-1.6.5-clarabel-0.11.1')!r})
environment=os.environ.copy();environment.update(PYTHONPATH=str(repo/'src')+os.pathsep+str(deps),PYTHONDONTWRITEBYTECODE='1')
probe=run([dqn,'-B','-c','import json,sys,scipy,torch,cvxpy,clarabel,satellite_rl;print(json.dumps({{"python":sys.executable,"scipy":scipy.__version__,"torch":torch.__version__,"cuda_build":torch.version.cuda,"cuda":torch.cuda.is_available(),"cvxpy":cvxpy.__version__,"clarabel":clarabel.__version__,"package":satellite_rl.__file__}}))'],env=environment)
info=json.loads(probe.splitlines()[-1])
if (not info['cuda'] or pathlib.Path(info['package']).resolve()!=repo/'src/satellite_rl/__init__.py'
 or info['cvxpy']!='1.6.5' or info['clarabel']!='0.11.1'):
 raise RuntimeError('DQN CUDA, isolated solver or frozen satellite_rl import failed')
print(json.dumps({{'port':{port},'gpu_uuid':gpu,'commit':head,'repo':str(repo),
 'archive':str(root/'archive'),'receive_git':str(bare),'clean':not dirty,'runtime':info,
 'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}}))
'''
        transport = connect(port)
        try:
            installed[str(port)] = remote_python(transport, code, timeout=600)
        finally:
            transport.close()
    if installed["20418"]["gpu_uuid"] == installed["20424"]["gpu_uuid"]:
        raise RuntimeError("Both SSH instances exposed the same GPU UUID")
    receipt = {"status": "frozen_git_deployed_no_workbench_or_jobs",
               "checked_utc": datetime.now(timezone.utc).isoformat(),
               "commit": commit, "source_repo": str(source), "bundle": str(bundle),
               "bundle_sha256": bundle_sha, "remote_bundle": uploaded,
               "installed": installed}
    path = PREFLIGHT / f"remote_deployment_{commit[:12]}.json"
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"receipt": str(path), "status": receipt["status"], "commit": commit},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
