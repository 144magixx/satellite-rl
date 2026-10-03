"""Read-only S3-P02 check of both pinned remote instances."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


sys.path.insert(0, "D:/paper/outputs/verification/g11_remote_launch_20261001")
from remote_transport import connect, execute  # noqa: E402


BASE = "/home/u2024110224/third_stage_satellite_rl/s3_all_routes_20261003"
DQN = "/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python"
OUTPUT = Path("D:/paper/outputs/experiments/stage3_all_routes_20261003/preflight")


def check(port: int) -> dict:
    script = r'''
import datetime,json,os,pathlib,shutil,socket,subprocess,urllib.request
base=pathlib.Path(__BASE__)
def run(args,timeout=35,env=None):
 p=subprocess.run(args,capture_output=True,text=True,timeout=timeout,env=env)
 return {{'exit_code':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr[-500:]}}
def birth(pid):
 try:
  proc=pathlib.Path('/proc')/str(pid)
  stat=(proc/'stat').read_text(); return int(stat[stat.rfind(')')+2:].split()[19])
 except (OSError,IndexError,ValueError):return None
result={{'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'ssh_port':__PORT__,'hostname':socket.gethostname(),
 'gpu':run(['nvidia-smi','--query-gpu=name,uuid,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits']),
 'gpu_processes':run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits']),
 'disk_free_bytes':shutil.disk_usage('/home').free,
 'dqn':None,
 'ports':{{}},'old_workbench':{{}},'paths':{{}}}}
thread_env={{name:'1' for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS',
 'OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')}}
result['planned_thread_environment']=thread_env
cpu_lines=pathlib.Path('/proc/cpuinfo').read_text().splitlines()
result['cpu']={{'model':next((line.split(':',1)[1].strip() for line in cpu_lines if line.startswith('model name')),None),
 'logical_count':os.cpu_count(),
 'affinity_count':len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None}}
result['dqn']=run([__DQN__,'-B','-c','import json,sys,torch,scipy,numpy,satellite_rl;print(json.dumps({{"python":sys.executable,"python_version":sys.version.split()[0],"torch":torch.__version__,"cuda_build":torch.version.cuda,"cuda":torch.cuda.is_available(),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"scipy":scipy.__version__,"numpy":numpy.__version__,"package":satellite_rl.__file__}}))'],60,
 env=dict(os.environ,**thread_env,PYTHONDONTWRITEBYTECODE='1'))
for p in (8774,8775):
 with socket.socket() as sock:
  sock.settimeout(1);result['ports'][str(p)]=sock.connect_ex(('127.0.0.1',p))==0
for p in (8772,8773):
 try:
  with urllib.request.urlopen(f'http://127.0.0.1:{{p}}/api/runs',timeout=3) as response:runs=json.load(response).get('runs',[])
  with urllib.request.urlopen(f'http://127.0.0.1:{{p}}/api/system',timeout=3) as response:system=json.load(response)
  result['old_workbench'][str(p)]={{'system':{{k:system.get(k) for k in ('source_root','archive_root','package_file','python','cuda','gpu_name')}},
   'runs':[{{k:r.get(k) for k in ('run_id','stage','env_step','request_id','worker_pid')} for r in runs if str(r.get('run_id','')).startswith('refine_sac_')]}}
 except Exception as exc:result['old_workbench'][str(p)]={{'error':type(exc).__name__+': '+str(exc)}}
prior=pathlib.Path('/home/u2024110224/third_stage_satellite_rl/s3_power_20261003')/__PORT_NAME__
prior_port=8772 if __PORT__==20418 else 8773
old=result['old_workbench'].get(str(prior_port),{{}})
launch=prior/'ops'/f'workbench_{{prior_port}}_launch.json'
if launch.is_file():
 launched=json.loads(launch.read_text())
 old['server_pid']=launched.get('pid')
 old['server_start_ticks_expected']=launched.get('start_ticks')
 old['server_start_ticks_live']=birth(launched.get('pid'))
 old['server_birth_matches']=old['server_start_ticks_expected']==old['server_start_ticks_live']
head=run(['git','-C',str(prior/'repo'),'rev-parse','HEAD'])
old['frozen_repo_head']=head['stdout'] if head['exit_code']==0 else None
for name,path in {{'base':base,'port_root':base/__PORT_NAME__,'shared':base/'shared_data',
 'prior_base':pathlib.Path('/home/u2024110224/third_stage_satellite_rl/s3_power_20261003'),
 'prior_jan25':pathlib.Path('/home/u2024110224/third_stage_satellite_rl/s3_power_20261003/shared_data')}}.items():
 result['paths'][name]={{'path':str(path),'exists':path.exists()}}
jan26=[]
for parent in (pathlib.Path('/home/u2024110224/third_stage_satellite_rl'),
               pathlib.Path('/home/u2024110224/second_stage_satellite_rl')):
 for path in parent.glob('*/shared_data/raw/v2024.01.26-planes-readsb-staging-0.tar'):
  if len(jan26)>=20:break
  jan26.append({{'path':str(path),'bytes':path.stat().st_size}})
result['jan26_candidates']=jan26
result['process_births']={{}}
for item in result['old_workbench'].values():
 for row in item.get('runs',[]):
  pid=row.get('worker_pid')
  if isinstance(pid,int):result['process_births'][str(pid)]=birth(pid)
print(json.dumps(result,ensure_ascii=False))
'''
    script = (script.replace("{{", "{").replace("}}", "}")
              .replace("__BASE__", repr(BASE)).replace("__DQN__", repr(DQN))
              .replace("__PORT_NAME__", repr(str(port))).replace("__PORT__", str(port)))
    payload = base64.b64encode(script.encode()).decode()
    transport = connect(port)
    try:
        result = execute(transport, "python3 -c 'exec(__import__(\"base64\").b64decode(\"" + payload + "\"))'", timeout=180)
    finally:
        transport.close()
    if result["exit_code"]:
        raise RuntimeError(result["stderr"][-1000:])
    return json.loads(result["stdout"])


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {"status": "read_only_preflight", "checked_utc": datetime.now(timezone.utc).isoformat(), "instances": {}}
    for port in (20418, 20424):
        report["instances"][str(port)] = check(port)
    path = OUTPUT / "remote_preflight.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"receipt": str(path), "summary": {key: {"gpu": value["gpu"]["stdout"], "ports": value["ports"], "jan26_candidates": value["jan26_candidates"]} for key, value in report["instances"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
