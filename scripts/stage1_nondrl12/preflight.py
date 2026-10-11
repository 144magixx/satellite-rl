"""Bounded read-only four-instance readiness probe; never starts a job."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from . import ops

CODE = '''import os,sys,json,datetime,subprocess,shutil,socket,pathlib
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cpu_count':os.cpu_count(),'affinity':sorted(os.sched_getaffinity(0)),'disk':shutil.disk_usage('/home/u2024110224')._asdict(),'memory':pathlib.Path('/proc/meminfo').read_text().splitlines()[:3]}
out['gpu']=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10).stdout
out['gpu_processes']=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10).stdout
p='/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python'
r=subprocess.run([p,'-c',"import sys,torch,numpy,scipy,satellite_rl,json;print(json.dumps({'python':sys.executable,'torch':torch.__version__,'cuda':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'package':satellite_rl.__file__,'scipy':scipy.__version__}))"],capture_output=True,text=True,timeout=45)
out['dqn_exit']=r.returncode;out['dqn']=json.loads(r.stdout.splitlines()[-1]) if r.returncode==0 else {'error':r.stderr[-700:]}
out['ports']={}
for port in (8940,8941,8942,8943):
 s=socket.socket();s.settimeout(1);out['ports'][str(port)]=s.connect_ex(('127.0.0.1',port))==0;s.close()
out['group_exists']=pathlib.Path('/home/u2024110224/first_stage_satellite_rl/s1_nd12').exists()
print(json.dumps(out))
'''

def main():
    def one(port):
        try:
            result=ops.remote_python(CODE,port,timeout=80)
            return str(port),({'connected':True,'probe':json.loads(result['stdout'].splitlines()[-1])}
                             if result['exit_code']==0 else {'connected':True,'error':result['stderr'][-700:]})
        except Exception as exc:
            return str(port), {'connected':False,'error':type(exc).__name__+': '+str(exc)[:400]}
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=dict(pool.map(one,ops.SSH_PORTS))
    value={'observed_utc':datetime.now(timezone.utc).isoformat(),'lanes':rows}
    folder=ops.LOCAL/'preflight';folder.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    (folder/f'resources_{stamp}.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    for port,row in rows.items():
        probe=row.get('probe',{})
        print(json.dumps({'port':port,'connected':row['connected'],'error':row.get('error'),
                         'gpu':probe.get('gpu'),'gpu_processes':probe.get('gpu_processes'),
                         'dqn':probe.get('dqn'),'affinity':probe.get('affinity'),'ports':probe.get('ports')},ensure_ascii=False))

if __name__=='__main__':
    main()
