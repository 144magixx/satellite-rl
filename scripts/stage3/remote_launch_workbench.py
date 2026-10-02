"""Start an isolated stage 3 workbench from its frozen Git checkout.

Launching the server creates no formal evaluation job.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import re
import sys


sys.path.insert(0, "D:/paper/outputs/verification/g11_remote_launch_20261001")
from remote_transport import connect, execute  # noqa: E402


BASE = "/home/u2024110224/third_stage_satellite_rl/s3_power_20261003"
DQN = "/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python"
PORTS = {20418: 8772, 20424: 8773}
PREFLIGHT = Path("D:/paper/outputs/experiments/stage3_power_20261003/preflight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssh-port", type=int, choices=PORTS, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--max-workers", type=int, choices=(1, 2, 3, 4), default=1)
    args = parser.parse_args()
    commit = args.commit.lower()
    if not re.fullmatch("[0-9a-f]{40}", commit):
        raise ValueError("Full frozen commit required")
    deployment = PREFLIGHT / f"remote_deployment_{commit[:12]}.json"
    receipt = json.loads(deployment.read_text(encoding="utf-8"))
    if receipt.get("commit") != commit or receipt.get("status") != "frozen_git_deployed_no_workbench_or_jobs":
        raise RuntimeError("Frozen deployment receipt differs")
    expected_gpu = receipt["installed"][str(args.ssh_port)]["gpu_uuid"]
    workbench_port = PORTS[args.ssh_port]
    code = f'''
import datetime,json,os,pathlib,socket,subprocess,sys,time,urllib.request
root=pathlib.Path({BASE!r})/{str(args.ssh_port)!r}
repo=root/'repo';archive=root/'archive';ops=root/'ops'
python={DQN!r};commit={commit!r};port={workbench_port};expected_gpu={expected_gpu!r}
head=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
dirty=subprocess.run(['git','-C',str(repo),'status','--porcelain'],capture_output=True,text=True,check=True).stdout.strip()
if head!=commit or dirty:raise RuntimeError('Stage3 checkout differs from clean frozen commit')
gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
if gpu!=expected_gpu:raise RuntimeError('GPU UUID differs from deployment receipt')
sys.path.insert(0,str(repo/'src'))
import satellite_rl,torch,scipy
if (pathlib.Path(satellite_rl.__file__).resolve()!=repo/'src/satellite_rl/__init__.py'
    or not torch.cuda.is_available() or torch.version.cuda!='12.1'):
 raise RuntimeError('Frozen package or DQN CUDA unavailable')
with socket.socket() as probe:
 probe.settimeout(2)
 if probe.connect_ex(('127.0.0.1',port))==0:raise RuntimeError('WorkBench port occupied; inspect process')
env=dict(os.environ,PYTHONPATH=str(repo/'src'))
log_path=ops/f'server_{{port}}.log'
with log_path.open('ab',buffering=0) as log:
 process=subprocess.Popen([python,'-m','satellite_rl.workbench.server',
  '--archive-root',str(archive),'--port',str(port),'--python',python,
  '--max-workers',str({args.max_workers})],cwd=repo,env=env,
  stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
system=None
for _ in range(30):
 if process.poll() is not None:raise RuntimeError('Stage3 workbench exited early; inspect server log')
 try:
  request=urllib.request.Request(f'http://127.0.0.1:{{port}}/api/system',headers={{'Cache-Control':'no-cache'}})
  with urllib.request.urlopen(request,timeout=3) as response:system=json.load(response)
  break
 except Exception:time.sleep(1)
if system is None:raise RuntimeError('Stage3 workbench did not become healthy')
if (system.get('cuda') is not True or system.get('cuda_build')!='12.1'
 or pathlib.Path(system.get('source_root','')).resolve()!=repo.resolve()
 or pathlib.Path(system.get('archive_root','')).resolve()!=archive.resolve()
 or pathlib.Path(system.get('package_file','')).resolve()!=repo/'src/satellite_rl/__init__.py'
 or pathlib.Path(system.get('python','')).resolve()!=pathlib.Path(python).resolve()):
 raise RuntimeError('Live workbench source/Python/CUDA/archive mismatch')
stat=(pathlib.Path('/proc')/str(process.pid)/'stat').read_text()
start_ticks=int(stat[stat.rfind(')')+2:].split()[19])
result={{'status':'stage3_workbench_running_no_formal_jobs',
 'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'ssh_port':{args.ssh_port},'workbench_port':port,'pid':process.pid,'start_ticks':start_ticks,
 'repo':str(repo),'commit':commit,'archive':str(archive),'log':str(log_path),
 'gpu_uuid':gpu,'scipy':scipy.__version__,'system':system}}
(ops/f'workbench_{{port}}_launch.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False))
'''
    encoded = base64.b64encode(code.encode()).decode()
    command = DQN + " -c 'exec(__import__(\"base64\").b64decode(\"" + encoded + "\"))'"
    transport = connect(args.ssh_port)
    try:
        response = execute(transport, command, timeout=180)
    finally:
        transport.close()
    if response["exit_code"]:
        raise RuntimeError(response["stderr"][-1200:])
    result = json.loads(response["stdout"])
    path = PREFLIGHT / f"workbench_launch_{args.ssh_port}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"receipt": str(path), "status": result["status"],
                      "ssh_port": args.ssh_port, "workbench_port": workbench_port}, ensure_ascii=False))


if __name__ == "__main__":
    main()
