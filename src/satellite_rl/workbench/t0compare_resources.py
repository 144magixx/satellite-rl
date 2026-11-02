"""Fresh local lane observation; never treats GPUidle as old-owner release."""
from pathlib import Path
import json,os,subprocess,urllib.request,datetime
from satellite_rl.workbench.runtime import process_start_time
OLD_API={20418:9006,20424:9007,20473:9008,20474:9009}
def observe(port,known_jobs):
 q={}
 for name,args in [('gpu',['--query-gpu=uuid,memory.free,temperature.gpu']),('compute',['--query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory'])]:
  p=subprocess.run(['nvidia-smi',*args,'--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15);q[name]={'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
 own=[];foreign=[];errors=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  except FileNotFoundError:continue
  except PermissionError:continue
  if 'workbench.runner' in command or 'r6_runtime_bootstrap' in command or 'ai_search_r6_stage' in command or 't0compare_runner' in command:
   row={'pid':int(p.name),'birth':process_start_time(int(p.name)),'command':command}
   if '/s2_t24/'+str(port)+'/' in command and 't0compare_runner' in command:own.append(row)
   else:foreign.append(row)
 old={}
 try:
  with urllib.request.urlopen('http://127.0.0.1:'+str(OLD_API[port])+'/api/runs',timeout=45) as r:raw=json.load(r)
  rows=raw.get('runs',raw.get('tasks',[])) if isinstance(raw,dict) else raw;old={'query_succeeded':True,'rows':rows}
 except Exception as e:old={'query_succeeded':False,'error_type':type(e).__name__,'error':str(e)}
 memory=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
 return {'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'port':port,'queries':q,'own_workers':own,'foreign_workers':foreign,'old_API':old,'CPU_affinity':sorted(os.sched_getaffinity(0)),'cgroup':Path('/proc/self/cgroup').read_text(),'memory_available_bytes':memory,'known_jobs':known_jobs,'probe_errors':errors}
def reasons(observation,proof,max_workers):
 out=[];queries=observation['queries']
 if observation['foreign_workers']:out.append('old_or_foreign_scientific_worker_present')
 if any(queries[k]['exit_code']!=0 for k in ('gpu','compute')):out.append('GPU_query_failed')
 gpu=queries['gpu']['stdout'].strip().split(',')
 if not gpu or gpu[0].strip()!=proof.get('gpu_uuid'):out.append('GPU_identity_mismatch')
 if len(gpu)!=3 or float(gpu[1])<2048 or float(gpu[2])>85:out.append('GPU_memory_temperature_guard')
 if observation['memory_available_bytes']<8*1024**3:out.append('host_memory_reserve')
 if len(observation['own_workers'])>=max_workers:out.append('lane_own_worker_slots_full')
 own_by_pid={row['pid']:row for row in observation['own_workers']}
 for row in observation['own_workers']:
  match=[j for j in observation['known_jobs'] if j.get('launch',{}).get('pid')==row['pid'] and abs(j['launch'].get('process_start_time',-1)-row['birth'])<.01]
  if len(match)!=1:out.append('own_worker_identity_unknown')
 rawapps=queries['compute']['stdout'].strip();apps=[]
 if rawapps:
  for line in rawapps.splitlines():
   fields=[x.strip() for x in line.split(',')]
   if len(fields)!=4:out.append('NVML_compute_schema_unknown');continue
   apps.append(int(fields[0]))
  # Actual engineering must establish namespace mapping; never guess host PID.
  mapping=proof.get('GPU_namespace_mapping',{})
  if mapping.get('actual_mapping_verified') is not True or sorted(apps)!=sorted(row['host_PID'] for row in mapping.get('rows',[])):out.append('NVML_actual_namespace_mapping_unverified')
  for row in mapping.get('rows',[]):
   actual=own_by_pid.get(row['namespace_PID'])
   if actual is None or abs(actual['birth']-row['process_start_time'])>.01:out.append('NVML_namespace_worker_birth_differs')
 old=observation['old_API']
 if not old.get('query_succeeded'):
  if proof.get('old_service_retirement_verified') is not True:out.append('old_workbench_API_unknown')
 else:
  for row in old['rows']:
   status=row.get('status');stage=status.get('stage') if isinstance(status,dict) else row.get('stage')
   if stage not in ('completed','failed','stopped','cancelled','migrated','attention_required'):out.append('old_workbench_task_nonterminal_or_unknown')
 return sorted(set(out))
