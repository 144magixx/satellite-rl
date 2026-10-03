"""Deploy one clean scientific Git commit to the isolated 20424 checkout.

Default is a read-only plan. ``--execute`` uploads a verified bundle and creates
the repo; it never starts a server or submits a task.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from ops_common import (ARCHIVE, BASE, DQN, LOCAL, OPS, PRIORITY_DEPS, REPO,
                        THREAD_ENV, require_commit, remote_file, remote_python,
                        sha256, utc_now, write_json)


def git(source: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(source), *args], capture_output=True,
                            text=True, check=True)
    return result.stdout.strip()


def plan(source: Path, commit: str) -> dict:
    source = source.resolve()
    if git(source, "rev-parse", "HEAD") != commit or git(source, "status", "--porcelain"):
        raise RuntimeError("正式源码须为所指定提交且工作区干净")
    remote = remote_python(f'''
import datetime,json,os,pathlib,shutil,socket,subprocess
root=pathlib.Path({BASE!r});repo=pathlib.Path({REPO!r});archive=pathlib.Path({ARCHIVE!r})
def run(args):
 p=subprocess.run(args,capture_output=True,text=True)
 return {{'exit_code':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr[-300:]}}
gpu=run(['nvidia-smi','--query-gpu=name,uuid,memory.total,memory.used','--format=csv,noheader,nounits'])
existing={{'repo':repo.exists(),'archive':archive.exists(),
 'repo_head':run(['git','-C',str(repo),'rev-parse','HEAD']) if (repo/'.git').exists() else None,
 'repo_dirty':run(['git','-C',str(repo),'status','--porcelain']) if (repo/'.git').exists() else None}}
with socket.socket() as sock:
 sock.settimeout(2);port_busy=sock.connect_ex(('127.0.0.1',8781))==0
print(json.dumps({{'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'target':'u2024110224@10.160.17.102:20424','gpu':gpu,'home_free_bytes':shutil.disk_usage('/home').free,
 'cpu_affinity':len(os.sched_getaffinity(0)),'workbench_8781_busy':port_busy,
 'existing':existing}},ensure_ascii=False))
''')
    return {"status": "deployment_plan_only", "checked_utc": utc_now(),
            "scientific_commit": commit, "source_repo": str(source),
            "target": "u2024110224@10.160.17.102:20424", "remote": remote}


def deploy(source: Path, commit: str) -> dict:
    checked = plan(source, commit)
    existing = checked["remote"]["existing"]
    if checked["remote"]["workbench_8781_busy"]:
        raise RuntimeError("8781端口已占用，先核现有服务")
    if existing["repo"]:
        if (existing["repo_head"] or {}).get("stdout") != commit or (existing["repo_dirty"] or {}).get("stdout"):
            raise RuntimeError("现有独立repo提交不符或有改动；不得替换")
        status = "already_deployed_clean"
        bundle_info = None
    else:
        LOCAL.mkdir(parents=True, exist_ok=True)
        bundle = LOCAL / f"s2p17_{commit}.bundle"
        if not bundle.exists():
            subprocess.run(["git", "-C", str(source), "bundle", "create", str(bundle), "HEAD"], check=True)
        subprocess.run(["git", "-C", str(source), "bundle", "verify", str(bundle)],
                       capture_output=True, text=True, check=True)
        bundle_sha = sha256(bundle)
        remote_bundle = OPS + f"/s2p17_{commit}.bundle"
        remote_python(f'''
import json,pathlib
root=pathlib.Path({BASE!r})
for name in ('ops','archive','locks'):(root/name).mkdir(parents=True,exist_ok=True)
if (root/'repo').exists():raise RuntimeError('Repo appeared after deployment plan')
print(json.dumps({{'ready':True,'root':str(root)}}))
''')
        remote_file(bundle, remote_bundle + ".part")
        installed = remote_python(f'''
import datetime,hashlib,json,os,pathlib,subprocess
root=pathlib.Path({BASE!r});bundle=pathlib.Path({remote_bundle!r});part=pathlib.Path(str(bundle)+'.part')
repo=pathlib.Path({REPO!r});incoming=root/'repo.incoming';commit={commit!r}
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
if sha(part)!={bundle_sha!r}:raise RuntimeError('Uploaded Git bundle SHA differs')
if bundle.exists():
 if sha(bundle)!={bundle_sha!r}:raise RuntimeError('Existing Git bundle differs')
 part.unlink()
else:part.rename(bundle)
if repo.exists() or incoming.exists():raise RuntimeError('Repo or incomplete incoming checkout exists')
subprocess.run(['git','clone','--no-checkout',str(bundle),str(incoming)],capture_output=True,text=True,check=True)
subprocess.run(['git','-C',str(incoming),'checkout','--detach',commit],capture_output=True,text=True,check=True)
head=subprocess.run(['git','-C',str(incoming),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
dirty=subprocess.run(['git','-C',str(incoming),'status','--porcelain'],capture_output=True,text=True,check=True).stdout.strip()
if head!=commit or dirty:raise RuntimeError('Incoming checkout differs from clean scientific commit')
incoming.rename(repo)
deps=pathlib.Path({PRIORITY_DEPS!r})
if not deps.is_dir():raise RuntimeError('Pinned CVXPY/Clarabel dependency directory missing')
env=os.environ.copy();env.update(PYTHONPATH=str(repo/'src')+os.pathsep+str(deps),PYTHONDONTWRITEBYTECODE='1',**{THREAD_ENV!r})
cmd=[{DQN!r},'-B','-c','import json,sys,torch,cvxpy,clarabel,satellite_rl;print(json.dumps({{"python":sys.executable,"torch":torch.__version__,"cuda_build":torch.version.cuda,"cuda":torch.cuda.is_available(),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"cvxpy":cvxpy.__version__,"clarabel":clarabel.__version__,"package":satellite_rl.__file__}}))']
probe=subprocess.run(cmd,capture_output=True,text=True,timeout=90,check=True,env=env)
runtime=json.loads(probe.stdout.splitlines()[-1])
if (runtime['cuda'] is not True or runtime['cuda_build']!='12.1' or runtime['cvxpy']!='1.6.5'
 or runtime['clarabel']!='0.11.1' or pathlib.Path(runtime['package']).resolve()!=repo/'src/satellite_rl/__init__.py'):
 raise RuntimeError('Frozen DQN/CUDA/package/solver preflight failed')
gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
receipt={{'status':'deployed_clean_no_workbench_or_jobs',
 'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'target':'u2024110224@10.160.17.102:20424','commit':commit,'bundle_sha256':{bundle_sha!r},
 'repo':str(repo),'archive':{ARCHIVE!r},'gpu_uuid':gpu,'runtime':runtime}}
(root/'ops/deployment.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
print(json.dumps(receipt,ensure_ascii=False))
''', timeout=600)
        status = "deployed_clean_no_workbench_or_jobs"
        bundle_info = {"path": str(bundle), "sha256": bundle_sha, "remote": remote_bundle}
    result = {"status": status, "checked_utc": utc_now(), "scientific_commit": commit,
              "source_repo": str(source.resolve()), "bundle": bundle_info,
              "installed": installed if not existing["repo"] else checked["remote"]}
    write_json(LOCAL / f"deployment_{commit[:12]}.json", result)
    return {"receipt": str(LOCAL / f"deployment_{commit[:12]}.json"), **result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    commit = require_commit(args.commit)
    result = deploy(args.source_repo, commit) if args.execute else plan(args.source_repo, commit)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
