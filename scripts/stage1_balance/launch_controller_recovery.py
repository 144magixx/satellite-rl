"""Start S1-02 recovery controller with a sole completed A2 dependency.

The original 77d plan/state/four zero-scene failed tasks remain unchanged.
Only call after the fixed science workbench and one-scene A0 gate pass.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

from remote_sources import LOCAL, REMOTE, connect, run_code, upload_once


REPO = Path(__file__).resolve().parents[2]
OLD_PLAN_SHA = "5895e5d2712eb8db84942bbdfd1719af2d75ccfd32e390ca72282f15d46ef8f4"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=LOCAL / "plan_recovery.json")
    args = parser.parse_args()
    plan_file = args.plan.resolve()
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    commit = plan.get("source_commit", "")
    if (plan.get("experiment_id") != "S1-02" or
            plan.get("remote_repo") != REMOTE + "/repo" or
            plan.get("api_port") != 8892 or
            not re.fullmatch(r"[0-9a-f]{40}", commit) or
            plan.get("recovery_context", {}).get("previous_plan_sha256") != OLD_PLAN_SHA):
        raise ValueError("Frozen recovery plan target/scope differs")
    head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                   text=True).strip()
    if head != commit:
        raise RuntimeError("Local recovery source commit differs")
    plan_sha = hashlib.sha256(plan_file.read_bytes()).hexdigest()
    client = connect()
    try:
        with client.open_sftp() as sftp:
            uploads = [upload_once(sftp, plan_file, REMOTE + "/ops/controller_plan_recovery.json"),
                       upload_once(sftp, Path(__file__).with_name("controller.py"),
                                   REMOTE + "/ops/controller_recovery.py")]
        code = r'''
import datetime,hashlib,json,os,pathlib,subprocess,urllib.request
root=pathlib.Path(__REMOTE__);ops=root/'ops';repo=root/'repo';commit=__COMMIT__
plan_path=ops/'controller_plan_recovery.json';state_path=ops/'controller_state_recovery.json'
plan=json.loads(plan_path.read_text())
if hashlib.sha256(plan_path.read_bytes()).hexdigest()!=__PLAN_SHA__:
 raise RuntimeError('Uploaded recovery plan SHA differs')
old_plan=ops/'controller_plan.json';old_state=ops/'controller_state.json'
if hashlib.sha256(old_plan.read_bytes()).hexdigest()!=__OLD_SHA__:
 raise RuntimeError('Original frozen plan changed')
if hashlib.sha256(old_state.read_bytes()).hexdigest()!=plan['recovery_context']['previous_state_sha256']:
 raise RuntimeError('Original failed controller state changed')
old_launch=json.loads((ops/'controller_launch.json').read_text())
proc=pathlib.Path('/proc')/str(old_launch['pid'])
if proc.exists():
 stat=(proc/'stat').read_text();ticks=int(stat[stat.rfind(')')+2:].split()[19])
 command=(proc/'cmdline').read_bytes().decode('utf-8','replace')
 if ticks==old_launch['start_ticks'] and 'controller.py' in command:
  raise RuntimeError('Old failed controller still running')
wb=json.loads((ops/'workbench_recovery_launch.json').read_text())
if wb['commit']!=commit or wb['port']!=8892:raise RuntimeError('Recovery workbench receipt differs')
wb_proc=pathlib.Path('/proc')/str(wb['pid'])
if not wb_proc.exists():raise RuntimeError('Recovery workbench is absent')
stat=wb_proc.joinpath('stat').read_text();ticks=int(stat[stat.rfind(')')+2:].split()[19])
if ticks!=wb['start_ticks']:raise RuntimeError('Recovery workbench PID reused')
with urllib.request.urlopen('http://127.0.0.1:8892/api/system',timeout=10) as response:system=json.load(response)
if (pathlib.Path(system['source_root']).resolve()!=repo or
    pathlib.Path(system['archive_root']).resolve()!=root or
    pathlib.Path(system['package_file']).resolve()!=repo/'src/satellite_rl/__init__.py'):
 raise RuntimeError('Live recovery workbench source/archive/package differs')
if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()!=commit:
 raise RuntimeError('Recovery source checkout moved')
route_path=ops/'current_controller_route.json'
route={'experiment_id':'S1-02','status':'active_recovery_route',
 'plan_relative':'ops/controller_plan_recovery.json',
 'state_relative':'ops/controller_state_recovery.json',
 'plan_sha256':__PLAN_SHA__,'source_commit':commit,
 'previous_plan_sha256':__OLD_SHA__,
 'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
if route_path.exists():
 old=json.loads(route_path.read_text())
 stable=lambda value:{k:v for k,v in value.items() if k!='created_utc'}
 if stable(old)!=stable(route):raise RuntimeError('Existing recovery route differs')
 route=old
if not state_path.exists():
 initial={'experiment_id':'S1-02','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'plan_sha256':__PLAN_SHA__,'phase':'preflight','datasets':{},'dataset_records':{},
  'jobs':{},'events':[],'replaces_run_ids':plan['recovery_context']['previous_failed_jobs']}
 state_path.write_text(json.dumps(initial,ensure_ascii=False,indent=2)+'\n')
env=dict(os.environ,PYTHONPATH=str(repo/'src'),PYTHONDONTWRITEBYTECODE='1',
 OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
python='/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python'
check=subprocess.run([python,'-B',str(ops/'controller_recovery.py'),
 '--plan',str(plan_path),'--state',str(state_path),'--check'],cwd=repo,env=env,
 capture_output=True,text=True,timeout=60)
if check.returncode:raise RuntimeError('Recovery controller precheck failed: '+check.stderr[-1200:])
receipt=ops/'controller_recovery_launch.json'
if receipt.exists():raise RuntimeError('Recovery controller already launched; inspect its identity')
tmp=route_path.with_suffix('.tmp');tmp.write_text(json.dumps(route,ensure_ascii=False,indent=2)+'\n');tmp.replace(route_path)
log=ops/'controller_recovery.log'
with log.open('ab',buffering=0) as stream:
 p=subprocess.Popen([python,'-B',str(ops/'controller_recovery.py'),
  '--plan',str(plan_path),'--state',str(state_path)],cwd=repo,env=env,
  stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
  start_new_session=True,close_fds=True)
stat=(pathlib.Path('/proc')/str(p.pid)/'stat').read_text();ticks=int(stat[stat.rfind(')')+2:].split()[19])
value={'status':'recovery_controller_launched','pid':p.pid,'start_ticks':ticks,
 'commit':commit,'plan_sha256':__PLAN_SHA__,'route':route,
 'script_sha256':hashlib.sha256((ops/'controller_recovery.py').read_bytes()).hexdigest(),
 'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
receipt.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(value,ensure_ascii=False))
'''
        code = (code.replace("__REMOTE__", repr(REMOTE))
                    .replace("__COMMIT__", repr(commit))
                    .replace("__PLAN_SHA__", repr(plan_sha))
                    .replace("__OLD_SHA__", repr(OLD_PLAN_SHA)))
        result = run_code(client, code)
    finally:
        client.close()
    receipt = {"experiment_id": "S1-02", "launched_utc": datetime.now(timezone.utc).isoformat(),
               "uploaded": uploads, "remote": result}
    output = LOCAL / "preflight/controller_recovery_launch.json"
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"receipt": str(output), "pid": result["pid"],
                      "plan_sha256": plan_sha}, ensure_ascii=False))


if __name__ == "__main__":
    main()
