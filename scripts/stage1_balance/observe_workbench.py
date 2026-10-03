"""Read current S1-02 workbench tasks on 20418 with PID/source evidence.

Every call saves a bounded, timestamped local snapshot. It never resumes,
submits, stops, or changes a remote task or the root run manifest.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from remote_sources import LOCAL, REMOTE, connect, run_code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-port", type=int, default=8892)
    args = parser.parse_args()
    if args.api_port != 8892:
        raise ValueError("S1-02 observes only its isolated 8892 workbench")
    code = r'''
import datetime,hashlib,json,os,pathlib,subprocess,time,urllib.request
root=pathlib.Path(__REMOTE__);ops=root/'ops';port=__PORT__
route_file=ops/'current_controller_route.json'
route=json.loads(route_file.read_text()) if route_file.is_file() else None
route_issue=None
if route:
 if (route.get('experiment_id')!='S1-02' or
     route.get('plan_relative')!='ops/controller_plan_recovery.json' or
     route.get('state_relative')!='ops/controller_state_recovery.json'):
  route_issue='Recovery route path/identity differs from isolated S1-02 contract'
 plan_file=root/'ops/controller_plan_recovery.json'
 state_file=root/'ops/controller_state_recovery.json'
else:
 plan_file=ops/'controller_plan.json';state_file=ops/'controller_state.json'
observed=datetime.datetime.now(datetime.timezone.utc).isoformat()
plan=json.loads(plan_file.read_text()) if plan_file.is_file() else None
state=json.loads(state_file.read_text()) if state_file.is_file() else None
issues=[];tasks=[]
if route_issue:issues.append(route_issue)
if route and plan_file.is_file():
 actual=hashlib.sha256(plan_file.read_bytes()).hexdigest()
 if (actual!=route.get('plan_sha256') or
     plan.get('source_commit')!=route.get('source_commit') or
     not state or state.get('plan_sha256')!=actual):
  issues.append('Recovery route plan SHA/source/state binding differs')
def api(path):
 with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/'+path,timeout=12) as response:
  return json.load(response)
system=None
try:system=api('system')
except Exception as error:
 if plan:issues.append('Workbench API unavailable: '+repr(error))
if plan and system:
 if (pathlib.Path(system.get('source_root','')).resolve()!=root/'repo' or
     pathlib.Path(system.get('archive_root','')).resolve()!=root or
     pathlib.Path(system.get('package_file','')).resolve()!=root/'repo/src/satellite_rl/__init__.py' or
     system.get('cuda') is not True or system.get('cuda_build')!='12.1'):
  issues.append('Live workbench source/archive/package/CUDA differs')
if state and system:
 identifiers=[v['run_id'] for v in state.get('jobs',{}).values()]
 if len(identifiers)!=len(set(identifiers)):
  issues.append('Controller task run IDs are not unique')
 for label,record in sorted(state.get('jobs',{}).items()):
  run_id=record['run_id'];item={'label':label,'run_id':run_id}
  try:
   detail=api('runs/'+run_id);job=detail['job'];status=detail['status'];frozen=detail['config']
   item.update(phase=status.get('stage'),env_step=status.get('env_step'),
    budget_env_steps=status.get('budget_env_steps',frozen.get('training',{}).get('total_steps') if job.get('kind')=='train' else None),
    evaluated_scenes=status.get('evaluated_scenes'),evaluation_total=status.get('evaluation_total'),
    updated_at=status.get('heartbeat_at'),pid=status.get('pid'),
    git_commit=job.get('git_commit'),source_root=job.get('source_root'),
    request_id=job.get('request_id'),kind=job.get('kind'))
   expected_source=root/'outputs/workbench/source_snapshots'/plan['source_commit'] if plan else None
   allow=(plan.get('recovery_context',{}).get('allowed_completed_prepare',{})
          .get('validation') if plan and label=='prepare_validation' else None)
   reused=bool(allow and run_id==allow.get('run_id') and
               status.get('stage')=='completed')
   if reused:expected_source=root/'outputs/workbench/source_snapshots'/allow['source_commit']
   expected_commit=allow['source_commit'] if reused else plan['source_commit'] if plan else None
   item['reused_completed_prepare']=reused
   if (not plan or job.get('request_id')!=record['request_id'] or
       job.get('git_commit')!=expected_commit or
       pathlib.Path(job.get('source_root','')).resolve()!=expected_source or
       (reused and job.get('request_id')!=allow['request_id'])):
    issues.append(label+': task ID/source/commit mismatch')
   preflight=status.get('preflight') or {}
   item['worker_preflight']={key:preflight.get(key) for key in
     ('python','python_resolved','torch','torch_cuda','gpu','package_file','git_commit','source_root')} if preflight else None
   if preflight and (pathlib.Path(preflight.get('package_file','')).resolve()!=
                     expected_source/'src/satellite_rl/__init__.py' or
                     preflight.get('git_commit')!=expected_commit):
    issues.append(label+': worker frozen package/commit mismatch')
   pid=status.get('pid')
   active=status.get('stage') in ('running','training','evaluating','validating',
                                  'preparing','warmup','initializing')
   item['active_stage']=active
   if pid and (pathlib.Path('/proc')/str(pid)).exists():
    proc=pathlib.Path('/proc')/str(pid)
    raw=(proc/'stat').read_text();ticks=int(raw[raw.rfind(')')+2:].split()[19])
    born=time.time()-time.monotonic()+ticks/os.sysconf('SC_CLK_TCK')
    argv=[part.decode('utf-8','replace') for part in
          (proc/'cmdline').read_bytes().split(bytes([0])) if part]
    command=' '.join(argv)
    exe=(proc/'exe').resolve();cwd=(proc/'cwd').resolve()
    item['process']={'pid':pid,'start_ticks':ticks,
      'created_utc':datetime.datetime.fromtimestamp(born,datetime.timezone.utc).isoformat(),
      'exe':str(exe),'cwd':str(cwd),'command':command[:500]}
    started=status.get('started_at')
    if started:
     start_at=datetime.datetime.fromisoformat(started.replace('Z','+00:00')).timestamp()
     if abs(start_at-born)>15:issues.append(label+': PID creation time differs from job started_at')
    dqn=pathlib.Path('/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python').resolve()
    argv_exe=pathlib.Path(argv[0]).resolve() if argv else None
    recorded_exe=pathlib.Path(preflight.get('python_resolved','')).resolve() if preflight else None
    if active and (expected_source is None or cwd!=expected_source or
      argv_exe!=dqn or exe!=dqn or recorded_exe!=exe or
      'satellite_rl.workbench.runner' not in command):
     issues.append(label+': active worker process/source/interpreter differs')
   elif active:
    issues.append(label+': active worker PID absent from /proc')
   else:item['process']=None
  except Exception as error:
   item['read_error']=repr(error);issues.append(label+': API/task read failed')
  tasks.append(item)
gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu',
 '--format=csv,noheader'],capture_output=True,text=True,timeout=12)
if gpu.returncode:issues.append('GPU query failed')
result={'experiment_id':'S1-02','observed_utc':observed,
 'target':'u2024110224@10.160.17.102:20418','status':'attention' if issues else
   'awaiting_workbench' if not plan else 'awaiting_controller' if not state else 'observed',
 'controller_phase':state.get('phase') if state else None,
 'controller_route':route,
 'source_commit':plan.get('source_commit') if plan else None,
 'api_port':port,'api_system':{key:system.get(key) for key in
   ('python','torch','cuda_build','cuda','package_file','source_root','archive_root')} if system else None,
 'gpu':gpu.stdout.strip(),'tasks':tasks,'task_count':len(tasks),'issues':issues}
print(json.dumps(result,ensure_ascii=False))
'''
    code = code.replace('__REMOTE__', repr(REMOTE)).replace('__PORT__', str(args.api_port))
    client = connect()
    try:
        result = run_code(client, code)
    finally:
        client.close()
    observed = datetime.fromisoformat(result["observed_utc"])
    output = LOCAL / "preflight" / ("observe_" + observed.strftime("%Y%m%dT%H%M%SZ") + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    output.write_text(encoded, encoding="utf-8")
    (LOCAL / "preflight/observe_latest.json").write_text(encoded, encoding="utf-8")
    print(json.dumps({"snapshot": str(output), "observed_utc": result["observed_utc"],
                      "status": result["status"], "controller_phase": result["controller_phase"],
                      "tasks": result["task_count"], "issues": result["issues"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
