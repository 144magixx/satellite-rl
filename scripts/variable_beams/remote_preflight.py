"""Read-only 20424 preflight for the variable-beam experiment."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


TRANSPORT = Path("D:/paper/outputs/verification/g11_remote_launch_20261001")
sys.path.insert(0, str(TRANSPORT))
from remote_transport import connect, execute  # noqa: E402

OUTPUT = Path("D:/paper/outputs/experiments/variable_beams_20261004/preflight")


REMOTE = r'''
import datetime,json,os,pathlib,shutil,socket,subprocess,urllib.request
root=pathlib.Path('/home/u2024110224')
def run(args,timeout=45):
 try:
  p=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
  return {'exit_code':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr[-600:]}
 except Exception as e:return {'error':repr(e)}
def files(pattern):
 result=[]
 for p in root.glob(pattern):
  if len(result)>=300:break
  try:result.append({'path':str(p),'bytes':p.stat().st_size,'sha256':None})
  except OSError:pass
 return result
out={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'target':'u2024110224@10.160.17.102:20424','hostname':socket.gethostname(),
 'cpu':{'logical':os.cpu_count(),'affinity':len(os.sched_getaffinity(0))},
 'memory':run(['free','-b']),'disk_free_bytes':shutil.disk_usage('/home').free,
 'gpu':run(['nvidia-smi','--query-gpu=name,uuid,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits']),
 'gpu_processes':run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits']),
 'listeners':run(['ss','-ltnp']),
 'processes':run(['ps','-eo','pid,lstart,args','--sort=-start_time']),
 'dqn':run(['/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python','-B','-c',
   'import json,sys,torch,satellite_rl;print(json.dumps({"python":sys.executable,"torch":torch.__version__,"cuda_build":torch.version.cuda,"cuda":torch.cuda.is_available(),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"package":satellite_rl.__file__}))'],60),
 'paths':{},'workbenches':{},'data_candidates':{}}
out['known_data_roots']={}
for rel in ('second_stage_satellite_rl/archive/outputs/experiments/europe_adsb_20260928',
            'second_stage_satellite_rl/g11/20424/archive/outputs/workbench/scenes',
            'third_stage_satellite_rl/s3_si_priority_20261003/shared_data/raw'):
 p=root/rel
 out['known_data_roots'][rel]={'exists':p.exists(),'children':[x.name for x in p.iterdir()][:30] if p.is_dir() else []}
for name,path in {'g11':'second_stage_satellite_rl/g11/20424','stage3_priority':'third_stage_satellite_rl/s3_si_priority_20261003/20424','variable':'second_stage_satellite_rl/variable_beams_20261004/20424'}.items():
 p=root/path;out['paths'][name]={'path':str(p),'exists':p.exists()}
for port in (8771,8775,8777):
 try:
  with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/system',timeout=3) as r:system=json.load(r)
  with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/runs',timeout=3) as r:runs=json.load(r)
  out['workbenches'][str(port)]={'system':{k:system.get(k) for k in ('source_root','archive_root','package_file','python','cuda','gpu_name')},'runs':[{k:x.get(k) for k in ('run_id','stage','env_step','request_id','worker_pid')} for x in runs.get('runs',[])]}
 except Exception as e:out['workbenches'][str(port)]={'error':repr(e)}
for tag,patterns in {'jan08_09':['second_stage_satellite_rl/archive/outputs/experiments/europe_adsb_20260928/snapshots/train/2024-01-08/*.csv','second_stage_satellite_rl/archive/outputs/experiments/europe_adsb_20260928/snapshots/train/2024-01-09/*.csv'],
 'jan12':['second_stage_satellite_rl/archive/outputs/experiments/europe_adsb_20260928/snapshots/validation/2024-01-12/*.csv','second_stage_satellite_rl/g11/20424/archive/outputs/workbench/scenes/*/record.json'],
 'jan29':['third_stage_satellite_rl/*/shared_data/raw/v2024.01.29-planes-readsb-staging-0.tar','second_stage_satellite_rl/*/shared_data/raw/v2024.01.29-planes-readsb-staging-0.tar']}.items():
 out['data_candidates'][tag]=sum((files(x) for x in patterns),[])
out['g11_scene_records']=[]
for item in out['data_candidates']['jan12']:
 try:
  p=pathlib.Path(item['path']);r=json.loads(p.read_text())
  out['g11_scene_records'].append({'path':str(p),'count':r.get('count'),'role':r.get('source',{}).get('partition_role'),
   'source_dates':r.get('source',{}).get('source_dates'),'snapshot_dates':r.get('source',{}).get('snapshot_dates'),
   'input_examples':[{'path':x.get('path'),'exists':pathlib.Path(x.get('path','')).exists(),'sha256':x.get('sha256')} for x in r.get('source',{}).get('files',[])[:2]]})
 except Exception as e:out['g11_scene_records'].append({'path':item['path'],'error':repr(e)})
print(json.dumps(out,ensure_ascii=False))
'''


def main() -> None:
    encoded = base64.b64encode(REMOTE.encode()).decode()
    transport = connect(20424)
    try:
        result = execute(transport, "python3 -c 'exec(__import__(\"base64\").b64decode(\"" + encoded + "\"))'", timeout=180)
    finally:
        transport.close()
    if result["exit_code"]:
        raise RuntimeError(result["stderr"][-1000:])
    report = json.loads(result["stdout"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / ("remote_preflight_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"receipt": str(path), "dqn": report["dqn"], "gpu": report["gpu"],
                      "cpu": report["cpu"], "disk_free_bytes": report["disk_free_bytes"],
                      "workbenches": report["workbenches"], "known_data_roots":report['known_data_roots'],
                      "g11_scene_records":report['g11_scene_records'], "data_candidates": {
                          key: {"count": len(value), "first": value[:2]} for key, value in report["data_candidates"].items()}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
