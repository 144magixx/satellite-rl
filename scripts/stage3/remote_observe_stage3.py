"""Read-only dual-instance observation using workbench API, PID birth and source."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys


sys.path.insert(0, "D:/paper/outputs/verification/g11_remote_launch_20261001")
from remote_transport import connect, execute  # noqa: E402


BASE = "/home/u2024110224/third_stage_satellite_rl/s3_power_20261003"
DQN = "/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python"
PREFLIGHT = Path("D:/paper/outputs/experiments/stage3_power_20261003/preflight")
PORTS = {20418: 8772, 20424: 8773}


def observe(port: int, commit: str) -> dict:
    submitted = json.loads((PREFLIGHT / f"jan25_submission_{port}.json").read_text(encoding="utf-8"))
    launched = json.loads((PREFLIGHT / f"workbench_launch_{port}.json").read_text(encoding="utf-8"))
    if submitted.get("commit") != commit or launched.get("commit") != commit:
        raise RuntimeError("Submission or workbench launch receipt commit differs")
    run_id = submitted["job"]["run_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Unexpected run id")
    code = f'''
import datetime,hashlib,json,os,pathlib,subprocess,urllib.request
root=pathlib.Path({BASE!r})/{str(port)!r}
repo=root/'repo';archive=root/'archive';api_port={PORTS[port]}
run_id={run_id!r};commit={commit!r}
out={{'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'ssh_port':{port},'workbench_port':api_port,'run_id':run_id}}
def run(args):
 p=subprocess.run(args,capture_output=True,text=True)
 return p.stdout.strip() if p.returncode==0 else None
out['repo_head']=run(['git','-C',str(repo),'rev-parse','HEAD'])
out['repo_dirty']=run(['git','-C',str(repo),'status','--porcelain'])
def proc(pid):
 if not isinstance(pid,int) or pid<=0:return None
 path=pathlib.Path('/proc')/str(pid)/'stat'
 try:
  raw=path.read_text();parts=raw[raw.rfind(')')+2:].split()
  return {{'pid':pid,'state':parts[0],'start_ticks':int(parts[19])}}
 except (OSError,ValueError,IndexError):return None
out['server_process']=proc({launched['pid']})
out['server_birth_matches']=(out['server_process'] is not None and
 out['server_process']['start_ticks']=={launched['start_ticks']})
def api(path):
 request=urllib.request.Request(f'http://127.0.0.1:{{api_port}}/api'+path,
  headers={{'Cache-Control':'no-cache'}})
 with urllib.request.urlopen(request,timeout=8) as response:return json.load(response)
try:
 system=api('/system');detail=api('/runs/'+run_id)
 out['system']={{k:system.get(k) for k in ('source_root','archive_root','cuda','cuda_build','python','package_file','server_time')}}
 job=detail['job'];status=detail['status']
 out['job']={{k:job.get(k) for k in ('run_id','request_id','kind','git_commit','source_root','model_id','scene_id','evaluation_units')}}
 out['status']={{k:status.get(k) for k in ('stage','env_step','budget_env_steps','pid','process_alive','reported_stage','completed_at','metrics','optimizer_progress')}}
 out['worker_process']=proc(status.get('pid'))
 directory=archive/'outputs/workbench/jobs'/run_id
 launch=directory/'launch.json'
 if launch.is_file():
  launch_data=json.loads(launch.read_text())
  out['worker_launch']={{k:launch_data.get(k) for k in ('pid','process_start_time')}}
  if out['worker_process'] is not None and launch_data.get('process_start_time') is not None:
   boot=next(int(line.split()[1]) for line in pathlib.Path('/proc/stat').read_text().splitlines() if line.startswith('btime '))
   actual=boot+out['worker_process']['start_ticks']/os.sysconf('SC_CLK_TCK')
   out['worker_birth_matches']=abs(actual-launch_data['process_start_time'])<0.02
  else:out['worker_birth_matches']=None
 active=status.get('stage') in ('running','evaluating','initializing','starting')
 out['worker_birth_consistent']=(not active or out.get('worker_birth_matches') is True)
 source=pathlib.Path(job.get('source_root') or '')
 manifest=source/'source_manifest.json'
 if manifest.is_file():
  out['source_manifest_sha256']=hashlib.sha256(manifest.read_bytes()).hexdigest()
  out['source_manifest_commit']=json.loads(manifest.read_text()).get('git_commit')
 out['source_frozen']=(job.get('git_commit')==commit and out.get('source_manifest_commit')==commit)
 out['api_consistent']=(out['system']['source_root']==str(repo) and out['system']['archive_root']==str(archive)
  and out['system']['cuda'] is True and out['repo_head']==commit and out['repo_dirty']==''
  and out['server_birth_matches'] and out['source_frozen'] and out['worker_birth_consistent'])
except Exception as exc:
 out['api_error']=type(exc).__name__+': '+str(exc)[:300]
print(json.dumps(out,ensure_ascii=False))
'''
    encoded = base64.b64encode(code.encode()).decode()
    command = DQN + " -c 'exec(__import__(\"base64\").b64decode(\"" + encoded + "\"))'"
    transport = connect(port)
    try:
        response = execute(transport, command, timeout=60)
    finally:
        transport.close()
    if response["exit_code"]:
        raise RuntimeError(response["stderr"][-1200:])
    return json.loads(response["stdout"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    commit = args.commit.lower()
    if not re.fullmatch("[0-9a-f]{40}", commit):
        raise ValueError("Full frozen commit required")
    result = {"observed_utc": datetime.now(timezone.utc).isoformat(),
              "instances": {}}
    for port in PORTS:
        try:
            result["instances"][str(port)] = observe(port, commit)
        except Exception as exc:
            result["instances"][str(port)] = {"error": type(exc).__name__ + ": " + str(exc)[:400]}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
