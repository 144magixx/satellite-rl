"""Start the persistent S1-02 API controller after frozen workbench review."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

from remote_sources import LOCAL, REMOTE, connect, upload_once
import remote_stage1


DQN = "/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python"


def remote(client, code: str) -> dict:
    payload = base64.b64encode(code.encode()).decode()
    command = DQN + " -B -c 'exec(__import__(\"base64\").b64decode(\"" + payload + "\"))'"
    return json.loads(remote_stage1.execute(client, command))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=LOCAL / "plan_frozen.json")
    parser.add_argument("--repository", type=Path,
                        default=Path("C:/Users/Admin/.codex/worktrees/stage1-balance-actions/paper"))
    args = parser.parse_args()
    plan_file = args.plan.resolve()
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    commit = plan.get("source_commit", "")
    if (plan.get("experiment_id") != "S1-02" or
            plan.get("remote_repo") != REMOTE + "/repo" or
            plan.get("api_port") != 8892 or
            not re.fullmatch(r"[0-9a-f]{40}", commit)):
        raise ValueError("Frozen S1-02 plan target/commit differs")
    actual = subprocess.check_output(["git", "-C", str(args.repository), "rev-parse", "HEAD"],
                                     text=True).strip()
    if actual != commit:
        raise RuntimeError("Local scientific source differs from controller plan")
    client = connect()
    try:
        with client.open_sftp() as sftp:
            uploaded = [upload_once(sftp, plan_file, REMOTE + "/ops/controller_plan.json")]
            for name in ("controller.py", "postprocess.py"):
                uploaded.append(upload_once(
                    sftp, Path(__file__).with_name(name), REMOTE + "/ops/" + name))
        code = f'''
import datetime,hashlib,json,os,pathlib,subprocess,time,urllib.request
root=pathlib.Path({REMOTE!r});repo=root/'repo';ops=root/'ops';python={DQN!r}
plan=ops/'controller_plan.json';p=json.loads(plan.read_text())
if p['source_commit']!={commit!r} or p['api_port']!=8892:
 raise RuntimeError('Uploaded controller plan differs')
workbench=json.loads((ops/'workbench_launch.json').read_text())
if workbench['commit']!={commit!r} or workbench['port']!=8892:
 raise RuntimeError('WorkBench receipt does not match S1-02 plan')
wbproc=pathlib.Path('/proc')/str(workbench['pid'])
if not wbproc.exists():raise RuntimeError('WorkBench process is absent')
stat=(wbproc/'stat').read_text();ticks=int(stat[stat.rfind(')')+2:].split()[19])
if ticks!=workbench['start_ticks']:raise RuntimeError('WorkBench PID was reused')
with urllib.request.urlopen('http://127.0.0.1:8892/api/system',timeout=8) as response:
 system=json.load(response)
if pathlib.Path(system['source_root']).resolve()!=repo.resolve() or pathlib.Path(system['archive_root']).resolve()!=root.resolve():
 raise RuntimeError('Live WorkBench source/archive differs')
env=dict(os.environ,PYTHONPATH=str(repo/'src'),PYTHONDONTWRITEBYTECODE='1',
 OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
check=subprocess.run([python,'-B',str(ops/'controller.py'),'--plan',str(plan),
 '--state',str(ops/'controller_state.json'),'--check'],env=env,
 cwd=repo,capture_output=True,text=True,timeout=60)
if check.returncode:raise RuntimeError('Controller check failed: '+check.stderr[-1000:])
receipt=ops/'controller_launch.json'
if receipt.exists():
 old=json.loads(receipt.read_text());pid=old.get('pid');proc=pathlib.Path('/proc')/str(pid)
 if proc.exists():
  stat=(proc/'stat').read_text();ticks=int(stat[stat.rfind(')')+2:].split()[19])
  cmd=(proc/'cmdline').read_bytes().decode('utf-8','replace')
  if ticks==old.get('start_ticks') and str(ops/'controller.py') in cmd:
   print(json.dumps({{'status':'already_running','previous':old}}));raise SystemExit
 state=ops/'controller_state.json'
 if state.exists() and json.loads(state.read_text()).get('phase')!='complete':
  raise RuntimeError('Previous controller stopped; diagnose before explicit restart')
log=ops/'controller.log'
with log.open('ab',buffering=0) as stream:
 process=subprocess.Popen([python,'-B',str(ops/'controller.py'),'--plan',str(plan),
  '--state',str(ops/'controller_state.json')],cwd=repo,env=env,
  stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
  start_new_session=True,close_fds=True)
stat=(pathlib.Path('/proc')/str(process.pid)/'stat').read_text()
ticks=int(stat[stat.rfind(')')+2:].split()[19])
value={{'status':'controller_launched','pid':process.pid,'start_ticks':ticks,
 'plan_sha256':hashlib.sha256(plan.read_bytes()).hexdigest(),
 'commit':{commit!r},'workbench_port':8892,'log':str(log),
 'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}}
receipt.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\\n')
print(json.dumps(value))
'''
        result = remote(client, code)
    finally:
        client.close()
    output = LOCAL / "preflight" / "controller_launch.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"saved_utc": datetime.now(timezone.utc).isoformat(),
                                  "remote": result, "uploaded": uploaded},
                                 ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"receipt": str(output), "remote": result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
