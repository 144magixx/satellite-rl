"""Save a read-only, timestamped S1-02 preflight from instance 20418."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from remote_sources import LOCAL, connect, run_code


REMOTE_CHECK = r'''
import datetime,hashlib,json,pathlib,subprocess
root=pathlib.Path('/home/u2024110224/first_stage_satellite_rl')
def call(argv):
 p=subprocess.run(argv,capture_output=True,text=True,timeout=45)
 return {'exit_code':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr.strip()[-500:]}
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
records={}
for role in ('train','validation','test'):
 path=root/'data_inputs'/role/'record.json'
 if path.is_file():
  data=json.loads(path.read_text())
  details=list((root/'data_inputs'/role).glob('source_observations_*.jsonl.gz'))
  records[role]={'path':str(path),'sha256':sha(path),'count':data.get('count'),
    'source_dates':data.get('source',{}).get('source_dates'),
    'partition_role':data.get('source',{}).get('partition_role'),
    'detail_count':len(details)}
scenes={}
for path in sorted((root/'outputs/workbench/scenes').glob('coverage_scenes_*/record.json')):
 data=json.loads(path.read_text());source=data.get('source',{})
 scenes[path.parent.name]={'count':data.get('count'),'record_sha256':sha(path),
   'role':source.get('partition_role'),'source_dates':source.get('source_dates'),
   'initialization':source.get('initialization')}
jobs={}
for path in sorted((root/'outputs/workbench/jobs').glob('*/status.json')):
 data=json.loads(path.read_text());pid=data.get('pid')
 jobs[path.parent.name]={'stage':data.get('stage'),'created_at':data.get('created_at'),
   'started_at':data.get('started_at'),'pid':pid,'pid_exists_now':bool(pid and (pathlib.Path('/proc')/str(pid)).exists()),
   'env_step':data.get('env_step'),'budget_env_steps':data.get('budget_env_steps'),
   'evaluated_scenes':data.get('evaluated_scenes'),'evaluation_total':data.get('evaluation_total')}
dqn=str(root/'envs/DQN/bin/python')
probe=call([dqn,'-c',"import json,sys,torch,satellite_rl;print(json.dumps({'python':sys.executable,'torch':torch.__version__,'torch_cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'package_file':satellite_rl.__file__}))"])
if probe['exit_code']!=0:raise RuntimeError('DQN probe failed: '+probe['stderr'])
probe=json.loads(probe['stdout'])
git=call(['git','-C',str(root/'repo'),'rev-parse','HEAD'])
git_status=call(['git','-C',str(root/'repo'),'status','--short','--branch'])
gpu=call(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'])
gpu_process=call(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader'])
process=call(['ps','-eo','pid,lstart,etime,args','--sort=start_time'])
relevant=[line for line in process['stdout'].splitlines() if any(word in line for word in ('satellite_rl','workbench','download_source.py'))]
ports=call(['ss','-ltnp'])
listeners=[line for line in ports['stdout'].splitlines() if any(f':{n} ' in line for n in (8876,8770,8771,8892))]
disk=call(['df','-B1',str(root)])
print(json.dumps({'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'remote_target':'u2024110224@10.160.17.102:20418','host':call(['hostname'])['stdout'],
 'user':call(['id','-un'])['stdout'],'dqn':probe,'gpu':gpu,'gpu_compute_apps':gpu_process,
 'relevant_processes':relevant,'selected_port_listeners':listeners,'disk':disk['stdout'],
 'legacy_git_commit':git['stdout'],'legacy_git_status':git_status['stdout'],
 'legacy_data_records':records,'legacy_workbench_scenes':scenes,
 'legacy_job_statuses':jobs},ensure_ascii=False))
'''


def main() -> None:
    client = connect()
    try:
        evidence = run_code(client, REMOTE_CHECK)
    finally:
        client.close()
    evidence["captured_local_utc"] = datetime.now(timezone.utc).isoformat()
    evidence["note"] = ("Read-only snapshot. Existing job PID fields are historical; "
                        "pid_exists_now alone does not prove job identity. Formal jobs "
                        "must use the new isolated workbench and frozen source.")
    output = LOCAL / "preflight" / "remote_readonly_20418.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(evidence, ensure_ascii=False, indent=2) + "\n"
    observed = datetime.fromisoformat(evidence["observed_utc"])
    unique = output.parent / ("remote_readonly_20418_" + observed.strftime("%Y%m%d_%H%M%S") + ".json")
    unique.write_text(encoded, encoding="utf-8")
    output.write_text(encoded, encoding="utf-8")
    print(json.dumps({"output": str(output), "observed_utc": evidence["observed_utc"],
                      "gpu": evidence["gpu"]["stdout"],
                      "legacy_jobs": len(evidence["legacy_job_statuses"]),
                      "live_relevant_processes": len(evidence["relevant_processes"])},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
