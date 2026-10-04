"""One bounded API/SSH observation of S1-03; never schedules or changes jobs."""
from __future__ import annotations

import json
from pathlib import Path

from remote import LOCAL, PORTS, REMOTE_BASE, connect, run_code, save, now


def observe(port):
    root = f"{REMOTE_BASE}/{port}"
    api_port = {20418: 8894, 20424: 8895}[port]
    code = f"""
import pathlib,datetime,json,hashlib,subprocess,urllib.request,os
root=pathlib.Path({root!r});began=datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(path,default=None):
 p=pathlib.Path(path);return json.loads(p.read_text()) if p.is_file() else default
def sha(path):
 with pathlib.Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def api(endpoint):
 with urllib.request.urlopen('http://127.0.0.1:{api_port}/api/'+endpoint,timeout=15) as response:return json.load(response)
def process(pid):
 p=pathlib.Path('/proc')/str(pid)
 if not p.is_dir():return None
 stat=(p/'stat').read_text();ticks=int(stat[stat.rfind(')')+2:].split()[19])
 command=(p/'cmdline').read_bytes().decode('utf-8','replace').replace('\\x00',' ')
 env=dict(item.split('=',1) for item in (p/'environ').read_bytes().decode('utf-8','replace').split('\\x00') if '=' in item)
 return {{'pid':pid,'birth_ticks':ticks,'command':command,'pythonpath':env.get('PYTHONPATH'),
  'instance_port':env.get('SATELLITE_RL_SSH_PORT')}}
issues=[];registry=read(root/'ops/current_workbench_service.json',{{}})
server=process(registry.get('pid')) if registry.get('pid') else None
if not server or server['birth_ticks']!=registry.get('start_ticks') or str(root) not in server['command']:
 issues.append('current workbench service identity not verified')
system=api('system');listing=api('coverage/runs')['runs'];tasks=[];sources={{}}
active={{'starting','initializing','preparing','training','validating','evaluating','refining','warmup','running'}}
for listed in listing:
 detail=api('runs/'+listed['run_id']);job=detail['job'];status=detail['status']
 if job.get('research_stage')!='coverage':continue
 taskdir=root/'outputs/workbench/jobs'/job['run_id'];config=read(taskdir/'config.json',{{}})
 worker=process(status.get('pid')) if status.get('stage') in active and status.get('pid') else None
 source=pathlib.Path(job['source_root'])
 if str(source) not in sources:
  manifest=read(source/'source_manifest.json',{{}})
  mismatch=[name for name,value in manifest.get('files',{{}}).items()
    if not (source/name).is_file() or sha(source/name)!=value]
  sources[str(source)]={{'git_commit':manifest.get('git_commit'),'file_count':len(manifest.get('files',{{}})),
   'mismatch':mismatch,'manifest_sha256':sha(source/'source_manifest.json') if (source/'source_manifest.json').is_file() else None}}
  if mismatch or not manifest:issues.append(job['run_id']+': source snapshot mismatch')
 if sources[str(source)]['git_commit']!=job.get('git_commit'):
  issues.append(job['run_id']+': source commit binding mismatch')
 if worker:
  launch=read(taskdir/'launch.json',{{}})
  boot=next(int(line.split()[1]) for line in pathlib.Path('/proc/stat').read_text().splitlines() if line.startswith('btime '))
  actual_birth=boot+worker['birth_ticks']/os.sysconf('SC_CLK_TCK')
  expected_birth=launch.get('process_start_time')
  worker.update(actual_birth_time=actual_birth,expected_birth_time=expected_birth,
   launch_pid=launch.get('pid'),birth_verified=launch.get('pid')==worker['pid'] and
   expected_birth is not None and abs(actual_birth-float(expected_birth))<0.02)
  if not worker['birth_verified']:issues.append(job['run_id']+': active worker birth not verified')
 canonical_config_sha=hashlib.sha256(json.dumps(config,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
 if job.get('config_hash') and canonical_config_sha!=job['config_hash']:
  issues.append(job['run_id']+': config canonical hash differs from frozen job')
 if status.get('stage') in active and worker and (str(taskdir) not in worker['command'] or worker['pythonpath']!=str(source/'src')):
  issues.append(job['run_id']+': active worker command/source mismatch')
 tasks.append({{'run_id':job['run_id'],'request_id':job.get('request_id'),'kind':job['kind'],
  'route':job.get('reconstruction_route'),'stage':status.get('stage'),'env_step':status.get('env_step',0),
  'budget_env_steps':status.get('budget_env_steps'),'update_step':status.get('update_step'),
  'prepared_scenes':status.get('prepared_scenes'),'preparation_total':status.get('preparation_total'),
  'evaluated_scenes':status.get('evaluated_scenes'),'evaluation_total':status.get('evaluation_total'),
  'stop_reason':status.get('stop_reason'),'source_commit':job.get('git_commit'),
  'source_root':str(source),'execution_target':job.get('execution_target'),
  'config_sha256':sha(taskdir/'config.json'),'status_sha256':sha(taskdir/'status.json'),
  'canonical_config_sha256':canonical_config_sha,'expected_config_hash':job.get('config_hash'),
  'active_process':worker,'error':status.get('error')}})
gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=10).stdout.strip()
data=root/'data/test'
source_status={{name:read(data/name) for name in ('raw/download_status.json','preparation_status.json')}} if {port}==20418 else None
s2=read('/home/u2024110224/second_stage_satellite_rl/variable_beams_20261004/20424/ops/controller_manifest.json',{{}}) if {port}==20424 else None
s2summary={{'phase':s2.get('phase'),'jobs':{{k:{{'run_id':v.get('run_id'),'stage':v.get('stage'),'env_step':v.get('env_step')}} for k,v in s2.get('jobs',{{}}).items() if k.endswith('_sca')}}}} if s2 else None
print(json.dumps({{'target':'u2024110224@10.160.17.102:{port}','observed_utc':began,
 'observation_finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'registry':registry,'server_process':server,'system':system,'gpu':gpu,'tasks':tasks,
 'sources':sources,'source_acquisition':source_status,'s2_readonly_context':s2summary,'issues':issues}},ensure_ascii=False))
"""
    client = connect(port)
    try:
        return run_code(client, code, timeout=90)
    finally:
        client.close()


def main():
    previous_path = LOCAL / 'preflight/observe_latest.json'
    previous = json.loads(previous_path.read_text(encoding='utf-8')) if previous_path.is_file() else {}
    result = {'experiment_id':'S1-03','local_started_utc':now(),'hosts':{}}
    for port in PORTS:
        try:
            result['hosts'][str(port)] = observe(port)
        except Exception as error:
            last = previous.get('hosts',{}).get(str(port),{})
            result['hosts'][str(port)] = {'error':str(error),
                'last_valid_observed_utc':last.get('observed_utc',last.get('last_valid_observed_utc')),
                'missing_this_round':['API tasks','server/worker identity','source/config SHA','GPU/VRAM']}
    for label,path in [('controller',LOCAL/'controller_state.json'),
                       ('source_bridge',LOCAL/'preflight/source_bridge_state.json'),
                       ('archive',LOCAL/'archive_state.json')]:
        result[label] = json.loads(path.read_text(encoding='utf-8-sig')) if path.is_file() else None
    stamp=now().replace(':','').replace('-','').split('.')[0]
    path=LOCAL/'preflight'/('observe_'+stamp+'.json')
    save(path,result);save(previous_path,result)
    print(json.dumps({'snapshot':str(path),'hosts':{port:{
        'observed_utc':value.get('observed_utc'),'gpu':value.get('gpu'),
        'tasks':[{k:task.get(k) for k in ('run_id','route','stage','env_step','prepared_scenes','evaluated_scenes')}
                 for task in value.get('tasks',[])],'issues':value.get('issues'),'error':value.get('error')}
        for port,value in result['hosts'].items()},'controller_phase':(result['controller'] or {}).get('phase')},ensure_ascii=False))


if __name__=='__main__':main()
