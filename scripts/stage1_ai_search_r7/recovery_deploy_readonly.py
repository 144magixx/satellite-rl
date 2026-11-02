"""One bounded SSH read of the original74 service; never mutates remote files."""
import argparse,hashlib,json,os,shlex
from pathlib import Path
import paramiko

REMOTE = r'''
import datetime,hashlib,json,os,subprocess,urllib.request
from pathlib import Path
root=Path('/home/u2024110224/first_stage_satellite_rl/s1_ai17/20474/round7/archive')
run='r7_labels_20261013_234807_312cdce7'
out={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ssh_port':20474,'api_port':9015,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'processes':[],'listen_inodes':[],'records':{},'api':{}}
for table in ('/proc/net/tcp','/proc/net/tcp6'):
 for line in Path(table).read_text().splitlines()[1:]:
  fields=line.split()
  if int(fields[1].split(':')[-1],16)==9015 and fields[3]=='0A':out['listen_inodes'].append(fields[9])
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if p.stat().st_uid!=os.getuid():continue
  raw=(p/'cmdline').read_bytes();tokens=[v.decode() for v in raw.split(b'\0') if v];cmd=' '.join(tokens)
  if str(root) not in cmd:continue
  listeners=[]
  for f in (p/'fd').iterdir():
   try:
    link=os.readlink(f)
    if link.startswith('socket:[') and link[8:-1] in out['listen_inodes']:listeners.append(link)
   except (FileNotFoundError,ProcessLookupError):pass
  if str(root) not in cmd and not listeners:continue
  stat=(p/'stat').read_text();parts=stat[stat.rfind(')')+2:].split()
  env=dict(v.split(b'=',1) for v in (p/'environ').read_bytes().split(b'\0') if b'=' in v)
  out['processes'].append({'pid':int(p.name),'birth_kind':'proc_start_ticks','birth':int(parts[19]),'boot_id':out['boot_id'],'state':parts[0],'cmdline':tokens,'cmdline_sha256':hashlib.sha256(raw).hexdigest(),'pythonpath':env.get(b'PYTHONPATH',b'').decode(),'listen_sockets':listeners})
 except (FileNotFoundError,ProcessLookupError):pass
jobs=root/'outputs/workbench/jobs'
out['jobs']=[]
for d in sorted(jobs.iterdir()):
 if not (d/'job.json').is_file():continue
 s=json.loads((d/'status.json').read_text())
 out['jobs'].append({'run_id':d.name,'status':s.get('status'),'phase':s.get('phase'),'worker_pid':s.get('worker_pid')})
d=jobs/run;job=json.loads((d/'job.json').read_text())
for name in ('job.json','config.json','case_index.json','status.json','launch.json'):
 p=d/name
 if p.is_file():out['records'][name]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
out['source_root']=job['source_root'];out['source_manifest_sha256']=job['source_manifest_sha256'];out['config_sha256']=job['config_sha256']
for path in ('/api/health','/api/ai-search-r7/tasks/'+run):
 try:
  with urllib.request.urlopen('http://127.0.0.1:9015'+path,timeout=10) as r:out['api'][path]={'status':r.status,'body':json.loads(r.read())}
 except Exception as e:out['api'][path]={'error':type(e).__name__+': '+str(e)}
out['resources']={'cpu_count':os.cpu_count(),'meminfo':Path('/proc/meminfo').read_text(),'disk_available_bytes':os.statvfs(root).f_bavail*os.statvfs(root).f_frsize}
r=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
out['resources']['gpu']={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
print(json.dumps(out,ensure_ascii=False))
'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-root',required=True);args=p.parse_args()
    out=Path(args.output_root);out.mkdir(parents=True,exist_ok=False)
    client=paramiko.SSHClient();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('10.160.17.102',port=20474,username='u2024110224',password=os.environ['BUPT_PASSWORD'],look_for_keys=False,allow_agent=False,timeout=20)
    try:
        stdin,stdout,stderr=client.exec_command('/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python -B -',timeout=60)
        stdin.write(REMOTE);stdin.channel.shutdown_write();raw=stdout.read();err=stderr.read();code=stdout.channel.recv_exit_status()
        (out/'stdout.bin').write_bytes(raw);(out/'stderr.bin').write_bytes(err)
        if code:raise RuntimeError('readonly remote check failed, inspect saved stderr')
        result=json.loads(raw);(out/'preflight.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print(json.dumps({'output_root':str(out),'preflight_sha256':hashlib.sha256((out/'preflight.json').read_bytes()).hexdigest(),'checked_utc':result['checked_utc'],'processes':result['processes'],'jobs':result['jobs']},ensure_ascii=False))
    finally:client.close()

if __name__=='__main__':main()
