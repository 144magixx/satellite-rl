"""Show the S3-P03 development UI with an empty isolated archive."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, "D:/paper/outputs/verification/g11_remote_launch_20261001")
from remote_transport import connect, execute  # noqa: E402


HERE = Path(__file__).resolve().parent
BASE = "/home/u2024110224/third_stage_satellite_rl/s3_si_priority_20261003"
DQN = "/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python"
OUTPUT = Path("D:/paper/outputs/verification/stage3_si_priority_20261003/remote_dev")
REMOTE = 8778
LOCAL = 8778


def start_remote(snapshot: str, old: dict | None) -> dict:
    if not snapshot.startswith(BASE + "/shared_data/dev_snapshots/"):
        raise ValueError("Expected isolated development snapshot")
    code = f'''
import datetime,json,os,pathlib,signal,socket,subprocess,time,urllib.request
base=pathlib.Path({BASE!r});root=pathlib.Path({snapshot!r})
archive=base/'20418/dev_archive';archive.mkdir(parents=True,exist_ok=True)
deps=base/'shared_data/deps/cvxpy-1.6.5-clarabel-0.11.1'
dqn={DQN!r};port={REMOTE}
if list((archive/'outputs/workbench/jobs').glob('*/job.json')):
 raise RuntimeError('Development archive unexpectedly contains jobs')
with socket.socket() as sock:
 sock.settimeout(1)
 if sock.connect_ex(('127.0.0.1',port))==0:
  old={old!r}
  if old is None:raise RuntimeError('Development workbench port occupied without prior receipt')
  pid=old['pid'];proc=pathlib.Path('/proc')/str(pid)
  if not proc.is_dir():raise RuntimeError('Old development workbench PID vanished while port occupied')
  stat=(proc/'stat').read_text();ticks=int(stat[stat.rfind(')')+2:].split()[19])
  cmd=(proc/'cmdline').read_bytes().decode('utf-8','replace')
  if (ticks!=old['start_ticks'] or 'satellite_rl.workbench.server' not in cmd
      or str(archive) not in cmd):
   raise RuntimeError('Port occupant differs from prior development workbench')
  with urllib.request.urlopen(f'http://127.0.0.1:{{port}}/api/system',timeout=3) as response:
   previous=json.load(response)
  if previous.get('source_root')!=old['snapshot'] or previous.get('archive_root')!=str(archive):
   raise RuntimeError('Previous development workbench source/archive differs')
  os.kill(pid,signal.SIGTERM)
  for _ in range(30):
   if not proc.is_dir():break
   time.sleep(0.1)
  with socket.socket() as again:
   again.settimeout(1)
   if again.connect_ex(('127.0.0.1',port))==0:
    raise RuntimeError('Previous development workbench did not release port')
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(root/'src')+os.pathsep+str(deps),
 SATELLITE_RL_PRIORITY_DEPS=str(deps),
 OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
log=base/'20418/ops/dev_workbench_8778.log'
with log.open('ab') as output:
 process=subprocess.Popen([dqn,'-B','-m','satellite_rl.workbench.server',
  '--archive-root',str(archive),'--port',str(port),'--python',dqn,'--max-workers','1'],
  cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
  start_new_session=True,close_fds=True)
system=None
for _ in range(40):
 if process.poll() is not None:raise RuntimeError('Development workbench exited early')
 try:
  with urllib.request.urlopen(f'http://127.0.0.1:{{port}}/api/system',timeout=3) as response:system=json.load(response)
  break
 except Exception:time.sleep(1)
if system is None or system.get('source_root')!=str(root) or system.get('archive_root')!=str(archive) or system.get('cuda') is not True:
 raise RuntimeError('Development workbench did not expose pinned source/archive/CUDA')
stat=(pathlib.Path('/proc')/str(process.pid)/'stat').read_text()
ticks=int(stat[stat.rfind(')')+2:].split()[19])
print(json.dumps({{'status':'dev_ui_running_empty_archive','checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'snapshot':str(root),'archive':str(archive),'pid':process.pid,'start_ticks':ticks,
 'remote_port':port,'system':system}}))
'''
    payload = base64.b64encode(code.encode()).decode()
    transport = connect(20418)
    try:
        result = execute(transport, "python3 -B -c 'exec(__import__(\"base64\").b64decode(\"" + payload + "\"))'", timeout=120)
    finally:
        transport.close()
    if result["exit_code"]:
        raise RuntimeError(result["stderr"][-1500:])
    return json.loads(result["stdout"])


def start_tunnel() -> dict:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{LOCAL}/api/system", timeout=3) as response:
            current = json.load(response)
        return {"local_port": LOCAL, "status": "existing_verified", "system": current}
    except Exception:
        pass
    log = OUTPUT / "dev_tunnel_8778.log"
    with log.open("ab") as stream:
        flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS if os.name == "nt" else 0
        proc = subprocess.Popen([sys.executable, str(HERE / "remote_tunnel.py"),
                                 "--ssh-port", "20418", "--local-port", str(LOCAL),
                                 "--remote-port", str(REMOTE)],
                                cwd=HERE.parents[1], env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                close_fds=True, creationflags=flags)
    for _ in range(30):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{LOCAL}/api/system", timeout=3) as response:
                system = json.load(response)
            return {"local_port": LOCAL, "pid": proc.pid, "system": system}
        except Exception:
            time.sleep(1)
    raise RuntimeError("Development UI tunnel did not become healthy")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-receipt", type=Path,
                        default=OUTPUT / "dev_test_20418_test_all_1456a49aa5c3c9bb.json")
    args = parser.parse_args()
    source = json.loads(args.snapshot_receipt.read_text(encoding="utf-8"))
    if source.get("exit_code") != 0:
        raise RuntimeError("Full development tests must pass before UI launch")
    previous = OUTPUT / "dev_workbench_launch.json"
    old = json.loads(previous.read_text(encoding="utf-8"))["remote"] if previous.is_file() else None
    remote = start_remote(source["snapshot_root"], old)
    tunnel = start_tunnel()
    if tunnel["system"].get("source_root") != source["snapshot_root"]:
        raise RuntimeError("Tunnel routed to wrong development source")
    result = {"remote": remote, "tunnel": tunnel,
              "url": f"http://127.0.0.1:{LOCAL}/", "formal_jobs_created": 0}
    path = OUTPUT / "dev_workbench_launch.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"receipt": str(path), "url": result["url"],
                      "source_root": source["snapshot_root"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
