"""Launch or inspect bounded S1-02 remote verification in isolated storage."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess

from remote_sources import LOCAL, REMOTE, connect, upload_once
import remote_stage1


DQN = "/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python"
CHECK = REMOTE + "/verification/candidate_checks"


def remote(client, code: str) -> dict:
    payload = base64.b64encode(code.encode()).decode()
    command = DQN + " -B -c 'exec(__import__(\"base64\").b64decode(\"" + payload + "\"))'"
    return json.loads(remote_stage1.execute(client, command))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "status"))
    parser.add_argument("--commit")
    parser.add_argument("--a2-local", type=Path,
                        default=Path("C:/Users/Admin/.codex/worktrees/stage1-balance-actions/paper/outputs/verification/s1_balance_core_20261004/common_a2.json"))
    args = parser.parse_args()
    client = connect()
    try:
        if args.action == "start":
            if not args.commit or not re.fullmatch(r"[0-9a-f]{40}", args.commit):
                raise ValueError("Full candidate science commit required")
            repo = Path("C:/Users/Admin/.codex/worktrees/stage1-balance-actions/paper")
            head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                           text=True).strip()
            if head != args.commit:
                raise RuntimeError("Local candidate checkout differs")
            remote_stage1.execute(client, f"mkdir -p {CHECK} {REMOTE}/ops")
            uploads = []
            with client.open_sftp() as sftp:
                uploads.append(upload_once(sftp, args.a2_local.resolve(), CHECK + "/common_a2.json"))
                for name in ("candidate_checks.py", "verify_real_scene.py"):
                    uploads.append(upload_once(
                        sftp, Path(__file__).with_name(name), REMOTE + "/ops/" + name))
            code = f'''
import datetime,json,os,pathlib,subprocess
root=pathlib.Path({REMOTE!r});repo=root/'repo';check=pathlib.Path({CHECK!r})
commit={args.commit!r};python={DQN!r};script=root/'ops/candidate_checks.py'
head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
if head!=commit:raise RuntimeError('Remote candidate repo differs')
receipt=check/'launch.json';state=check/'status.json'
if receipt.exists():
 old=json.loads(receipt.read_text());pid=old.get('pid');proc=pathlib.Path('/proc')/str(pid)
 if proc.exists():
  stat=(proc/'stat').read_text();ticks=int(stat[stat.rfind(')')+2:].split()[19])
  cmd=(proc/'cmdline').read_bytes().decode('utf-8','replace')
  if ticks==old.get('start_ticks') and str(script) in cmd:
   print(json.dumps({{'status':'already_running','previous':old}}));raise SystemExit
 if state.exists() and json.loads(state.read_text()).get('status')=='completed':
  print(json.dumps({{'status':'already_completed','previous':old}}));raise SystemExit
 raise RuntimeError('Existing candidate verification needs diagnosis; no silent rerun')
env=dict(os.environ,PYTHONPATH=str(repo/'src'),PYTHONDONTWRITEBYTECODE='1',
 OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
log=check/'controller.log'
with log.open('ab',buffering=0) as output:
 p=subprocess.Popen([python,'-B',str(script),'--repo',str(repo),'--commit',commit,
  '--output',str(check),'--a2-start',str(check/'common_a2.json')],cwd=repo,env=env,
  stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
  start_new_session=True,close_fds=True)
stat=(pathlib.Path('/proc')/str(p.pid)/'stat').read_text()
ticks=int(stat[stat.rfind(')')+2:].split()[19])
value={{'status':'verification_launched','pid':p.pid,'start_ticks':ticks,
 'commit':commit,'script':str(script),'log':str(log),
 'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}}
receipt.write_text(json.dumps(value,indent=2)+'\\n')
print(json.dumps(value))
'''
            result = remote(client, code)
            result["uploaded"] = uploads
        else:
            code = f'''
import datetime,json,pathlib
check=pathlib.Path({CHECK!r});launch=check/'launch.json';state=check/'status.json'
start=json.loads(launch.read_text()) if launch.is_file() else None
progress=json.loads(state.read_text()) if state.is_file() else None
live=False
if start:
 proc=pathlib.Path('/proc')/str(start['pid'])
 if proc.exists():
  stat=(proc/'stat').read_text();ticks=int(stat[stat.rfind(')')+2:].split()[19])
  cmd=(proc/'cmdline').read_bytes().decode('utf-8','replace')
  live=(ticks==start['start_ticks'] and start['script'] in cmd)
log=check/'controller.log'
print(json.dumps({{'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'launch':start,'live':live,'state':progress,
 'log_tail':log.read_text(errors='replace')[-1800:] if log.is_file() else ''}}))
'''
            result = remote(client, code)
    finally:
        client.close()
    output = LOCAL / "preflight" / ("candidate_checks_" + args.action + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"captured_utc": datetime.now(timezone.utc).isoformat(),
                                  "result": result}, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({"receipt": str(output), "result": result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
