"""Deploy the frozen S1-02 Git commit and start its isolated 20418 workbench.

This starts no experiment job. Formal jobs are later submitted by controller.py.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import remote_stage1  # noqa: E402
from remote_sources import LOCAL, REMOTE, connect  # noqa: E402


DQN = "/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python"
PORT = 8892
GPU_UUID = "GPU-638e7ef5-3af6-87fa-f240-f2be8161bd8a"


def remote(client, code: str) -> dict:
    encoded = base64.b64encode(code.encode()).decode()
    command = DQN + " -B -c 'exec(__import__(\"base64\").b64decode(\"" + encoded + "\"))'"
    return json.loads(remote_stage1.execute(client, command))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--repository", type=Path,
                        default=Path("C:/Users/Admin/.codex/worktrees/stage1-balance-actions/paper"))
    parser.add_argument("--max-workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--receipt-name", choices=("workbench_launch.json",
                                                   "workbench_recovery_launch.json"),
                        default="workbench_launch.json")
    parser.add_argument("--sync-only", action="store_true",
                        help="Deploy frozen Git source for short checks without starting the workbench")
    args = parser.parse_args()
    commit = args.commit.lower()
    if not re.fullmatch("[0-9a-f]{40}", commit):
        raise ValueError("Full scientific commit is required")
    repo = args.repository.resolve()
    current = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                      text=True).strip()
    if current != commit:
        raise RuntimeError("Local development HEAD differs from frozen scientific commit")
    client = connect()
    try:
        from types import SimpleNamespace

        remote_stage1.sync(client, SimpleNamespace(repository=str(repo), remote_root=REMOTE))
        if args.sync_only:
            receipt = {"status": "frozen_git_synced_no_workbench_or_jobs",
                       "checked_utc": datetime.now(timezone.utc).isoformat(),
                       "commit": commit, "remote_repo": REMOTE + "/repo"}
            output = LOCAL / "preflight" / "source_deployment.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(receipt, ensure_ascii=False))
            return
        code = f'''
import datetime,json,os,pathlib,socket,subprocess,sys,time,urllib.request
root=pathlib.Path({REMOTE!r});repo=root/'repo';ops=root/'ops';python={DQN!r}
commit={commit!r};port={PORT};expected_gpu={GPU_UUID!r}
head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
dirty=subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--','src','data','pyproject.toml'],text=True).strip()
if head!=commit or dirty:raise RuntimeError('S1-02 remote repo differs from clean frozen commit')
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
if gpu!=expected_gpu:raise RuntimeError('GPU UUID changed since read-only preflight')
threads={{key:'1' for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')}}
env=dict(os.environ,PYTHONPATH=str(repo/'src'),PYTHONDONTWRITEBYTECODE='1',**threads)
probe=subprocess.run([python,'-B','-c',
 'import json,sys,torch,satellite_rl;print(json.dumps({{"python":sys.executable,"torch":torch.__version__,"cuda":torch.version.cuda,"available":torch.cuda.is_available(),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"package":satellite_rl.__file__}}))'],
 env=env,capture_output=True,text=True,timeout=45,check=True)
runtime=json.loads(probe.stdout)
if (runtime['cuda']!='12.1' or not runtime['available'] or '4090 D' not in runtime['gpu']
 or pathlib.Path(runtime['package']).resolve()!=repo/'src/satellite_rl/__init__.py'):
 raise RuntimeError('DQN CUDA or package import does not match frozen S1-02 source')
with socket.socket() as socket_probe:
 socket_probe.settimeout(2)
 if socket_probe.connect_ex(('127.0.0.1',port))==0:
  raise RuntimeError('Selected isolated workbench port is already occupied')
log=ops/f'workbench_{{port}}.log'
with log.open('ab',buffering=0) as stream:
 process=subprocess.Popen([python,'-B','-m','satellite_rl.workbench.server',
  '--archive-root',str(root),'--port',str(port),'--python',python,
  '--max-workers',str({args.max_workers})],cwd=repo,env=env,
  stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
  start_new_session=True,close_fds=True)
system=None
for _ in range(30):
 if process.poll() is not None:raise RuntimeError('S1-02 workbench exited early')
 try:
  with urllib.request.urlopen(f'http://127.0.0.1:{{port}}/api/system',timeout=3) as response:
   system=json.load(response)
  break
 except Exception:time.sleep(1)
if system is None:raise RuntimeError('S1-02 workbench did not become healthy')
if (system.get('cuda') is not True or system.get('cuda_build')!='12.1'
 or pathlib.Path(system.get('source_root','')).resolve()!=repo.resolve()
 or pathlib.Path(system.get('archive_root','')).resolve()!=root.resolve()
 or pathlib.Path(system.get('package_file','')).resolve()!=repo/'src/satellite_rl/__init__.py'):
 raise RuntimeError('Live workbench source/Python/CUDA/archive mismatch')
stat=(pathlib.Path('/proc')/str(process.pid)/'stat').read_text()
ticks=int(stat[stat.rfind(')')+2:].split()[19])
receipt={{'status':'workbench_running_no_experiment_jobs',
 'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'execution_target':'u2024110224@10.160.17.102:20418',
 'port':port,'pid':process.pid,'start_ticks':ticks,'repo':str(repo),'commit':commit,
 'archive_root':str(root),'log':str(log),'max_workers':{args.max_workers},
 'gpu_uuid':gpu,'runtime':runtime,'system':system}}
(ops/{args.receipt_name!r}).write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\\n')
print(json.dumps(receipt,ensure_ascii=False))
'''
        result = remote(client, code)
    finally:
        client.close()
    output = LOCAL / "preflight" / args.receipt_name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"receipt": str(output), "port": PORT, "pid": result["pid"],
                      "commit": commit, "verified_utc": datetime.now(timezone.utc).isoformat()},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
