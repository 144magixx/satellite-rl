"""Review or launch the approved persistent 20424 S2-P17 controller.

Default prints a read-only plan. ``--execute`` stages an exact user-confirmed
scope and starts the controller; that controller submits formal workbench jobs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ops_common import (ARCHIVE, BASE, DQN, LOCAL, OPS, REPO, REQUESTS,
                        WORKBENCH_PORT, load_scope, remote_file, remote_python,
                        require_commit, sha256, utc_now, write_json)


def plan(commit: str, scope: dict, scope_sha: str) -> dict:
    code = f'''
import datetime,hashlib,json,pathlib,socket,subprocess,urllib.request
repo=pathlib.Path({REPO!r});archive=pathlib.Path({ARCHIVE!r});ops=pathlib.Path({OPS!r})
head=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
dirty=subprocess.run(['git','-C',str(repo),'status','--porcelain'],capture_output=True,text=True,check=True).stdout.strip()
if head!={commit!r} or dirty:raise RuntimeError('Remote repo differs from clean scientific commit')
with socket.socket() as sock:
 sock.settimeout(2);open_=sock.connect_ex(('127.0.0.1',{WORKBENCH_PORT}))==0
if not open_:raise RuntimeError('20424:8781 formal workbench is not running')
def api(path):
 req=urllib.request.Request(f'http://127.0.0.1:{WORKBENCH_PORT}/api'+path,headers={{'Cache-Control':'no-cache'}})
 with urllib.request.urlopen(req,timeout=10) as response:return json.load(response)
system=api('/system')
if system.get('source_root')!=str(repo) or system.get('archive_root')!=str(archive) or system.get('cuda') is not True:
 raise RuntimeError('Live workbench source/archive/CUDA differs')
launch=ops/'workbench_8781_launch.json'
if not launch.is_file() or json.loads(launch.read_text()).get('max_workers')!=2:
 raise RuntimeError('Initial training workbench must have max_workers=2')
paths={{'train':pathlib.Path({scope['train_record_path']!r}),
 'validation':pathlib.Path({(BASE + '/data/prepared/validation_eligible12_v2/record.json')!r}),
 'test':pathlib.Path({(BASE + '/data/prepared/test/record.json')!r})}}
checks={{}}
for role,path in paths.items():
 if not path.is_file():raise RuntimeError(role+' prepared record missing')
 value=json.loads(path.read_text())
 checks[role]={{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
  'count':value.get('count'),'mother_count':len({{item['mother_scene_id'] for item in value.get('files',[])}})}}
t0=pathlib.Path('/home/u2024110224/second_stage_satellite_rl/g11/20418/archive/models/runs/workbench/train_sac_20261001_162126_33e89e95/final')
checks['h0']={{'path':str(t0),'policy_sha256':hashlib.sha256((t0/'policy.pt').read_bytes()).hexdigest()}}
existing=[{{k:item.get(k) for k in ('run_id','request_id','kind','stage','env_step')}} for item in api('/runs')['runs']
 if item.get('request_id') in {list(REQUESTS.values())!r}]
print(json.dumps({{'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'target':'u2024110224@10.160.17.102:20424','commit':head,'workbench_port':{WORKBENCH_PORT},
 'max_workers':2,'datasets':checks,'existing_s2p17_jobs':existing}},ensure_ascii=False))
'''
    remote = remote_python(code, timeout=120)
    if (remote["datasets"]["train"]["sha256"] != scope["train_record_sha256"]
            or remote["datasets"]["train"]["count"] != scope["derived_scene_count"]
            or remote["datasets"]["train"]["mother_count"] != scope["mother_count"]
            or remote["datasets"]["validation"]["sha256"] !=
            "fa049ce2a881d1c5da81d2d6e2338821f7dfb6052684a999d438805d9b5e6525"
            or remote["datasets"]["test"]["sha256"] !=
            "202d3972c6e668af0f814585604ed3201e464490fc4cbcfea89e79fe3426e404"
            or remote["datasets"]["h0"]["policy_sha256"] !=
            "ae095e8246d22b2716ba154315cee18584917d215580821cb53f9aeba78eb824"):
        raise RuntimeError("Remote approved scene count/SHA or H0 weight differs")
    return {"status": "ready_to_start_controller_no_jobs_created", "checked_utc": utc_now(),
            "scientific_commit": commit, "scope_sha256": scope_sha,
            "approved_decision": scope["decision"], "planned_request_ids": REQUESTS,
            "remote": remote}


def execute(commit: str, scope_path: Path, scope: dict, scope_sha: str) -> dict:
    checked = plan(commit, scope, scope_sha)
    pending = OPS + "/approved_scope.json.part"
    remote_file(scope_path, pending)
    remote = remote_python(f'''
import datetime,hashlib,json,os,pathlib,subprocess
ops=pathlib.Path({OPS!r});part=ops/'approved_scope.json.part';target=ops/'approved_scope.json'
actual=hashlib.sha256(part.read_bytes()).hexdigest()
if actual!={scope_sha!r}:raise RuntimeError('Uploaded user scope SHA differs')
if target.exists():
 if hashlib.sha256(target.read_bytes()).hexdigest()!=actual:
  raise RuntimeError('Existing controller scope differs; no replacement after launch')
 part.unlink()
else:part.rename(target)
repo=pathlib.Path({REPO!r});script=repo/'scripts/variable_beams/controller.py'
cmd=[{DQN!r},'-B',str(script),'--commit',{commit!r},'--scope-sha256',{scope_sha!r},'--plan']
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONPATH']=str(repo/'src')
probe=subprocess.run(cmd,capture_output=True,text=True,env=env,timeout=1800)
if probe.returncode:raise RuntimeError('Controller plan rejected: '+probe.stderr[-1000:])
approved=json.loads(probe.stdout.splitlines()[-1])
launch=ops/'controller_launch.json'
if launch.exists():
 old=json.loads(launch.read_text())
 proc=pathlib.Path('/proc')/str(old['pid'])
 if proc.is_dir():
  stat=(proc/'stat').read_text();birth=int(stat[stat.rfind(')')+2:].split()[19])
  if birth==old['start_ticks']:
   if old['commit']!={commit!r} or old['scope_sha256']!={scope_sha!r}:
    raise RuntimeError('Existing controller belongs to another scope/commit')
   print(json.dumps({{'status':'already_running','launch':old,'controller_plan':approved}},ensure_ascii=False));raise SystemExit
 raise RuntimeError('Prior controller exited; inspect its manifest/failure before a deliberate restart')
log=ops/'controller.log'
cmd[-1]='--daemon'
with log.open('ab') as output:
 process=subprocess.Popen(cmd,cwd=repo,env=env,stdin=subprocess.DEVNULL,
  stdout=output,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
stat=(pathlib.Path('/proc')/str(process.pid)/'stat').read_text()
ticks=int(stat[stat.rfind(')')+2:].split()[19])
receipt={{'status':'persistent_controller_started','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'target':'u2024110224@10.160.17.102:20424','pid':process.pid,'start_ticks':ticks,
 'commit':{commit!r},'scope_sha256':{scope_sha!r},'log':str(log),
 'manifest':str(ops/'controller_manifest.json'),'plan':approved}}
launch.write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
print(json.dumps(receipt,ensure_ascii=False))
''', timeout=2100)
    result = {"status": remote["status"], "checked_utc": utc_now(),
              "plan": checked, "remote": remote}
    path = LOCAL / f"controller_launch_{commit[:12]}.json"
    write_json(path, result)
    return {"receipt": str(path), **result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--approved-scope", type=Path, required=True)
    parser.add_argument("--scope-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    commit = require_commit(args.commit)
    scope_path = args.approved_scope.resolve()
    scope = load_scope(scope_path, args.scope_sha256)
    result = (execute(commit, scope_path, scope, args.scope_sha256)
              if args.execute else plan(commit, scope, args.scope_sha256))
    if not args.execute:
        path = LOCAL / f"submission_plan_{commit[:12]}.json"
        write_json(path, result)
        result["receipt"] = str(path)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
