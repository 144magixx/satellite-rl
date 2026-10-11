"""Fresh resource observations and fail-closed admission for S2-F20 lanes."""
from datetime import datetime, timezone
import json
from . import ops

OLD_API_PORTS = dict(zip(ops.SSH_PORTS, (8940, 8941, 8942, 8943)))
TERMINAL = {'completed', 'failed', 'stopped', 'cancelled', 'invalidated'}


def process_candidate(pid, probe_pid, command):
    """The probe's own argv can contain the strings it searches for."""
    return int(pid) != int(probe_pid) and ('satellite_rl' in command or 'nondrl12' in command)


def probe(port):
    info = ops.instance(port)
    import inspect
    from satellite_rl.workbench.fair20_guard import cgroup_inventory
    cgroup_source=inspect.getsource(cgroup_inventory)
    code = f'''
import datetime,json,os,pathlib,shutil,subprocess,urllib.request
from pathlib import Path
{cgroup_source}
p=pathlib.Path('/proc'); processes=[]
for d in p.iterdir():
 if not d.name.isdigit():continue
 if int(d.name)==os.getpid():continue
 try:
  cmd=(d/'cmdline').read_bytes().replace(b'\\0',b' ').decode('utf-8','replace')
  if 'satellite_rl' in cmd or 'nondrl12' in cmd:
   raw=(d/'stat').read_text();processes.append({{'pid':int(d.name),'birth_ticks':int(raw[raw.rfind(')')+2:].split()[19]),'command':cmd}})
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
try:
 with urllib.request.urlopen('http://127.0.0.1:{OLD_API_PORTS[port]}/api/coverage/nondrl/runs',timeout=10) as r:old=json.load(r)
except Exception as e:old={{'unavailable':type(e).__name__}}
q=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.used,memory.free,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
c=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
base=pathlib.Path({info['root']!r}); ancestor=base
while not ancestor.exists():ancestor=ancestor.parent
mem={{}}
for line in (p/'meminfo').read_text().splitlines():
 if line.startswith(('MemAvailable:','MemTotal:')):mem[line.split(':')[0]]=int(line.split()[1])*1024
cg=cgroup_inventory()
print(json.dumps({{'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'port':{port},'probe_pid':os.getpid(),'gpu':q.stdout.strip(),'gpu_query_exit':q.returncode,'compute_apps':c.stdout.strip(),'compute_query_exit':c.returncode,'processes':processes,'old_tasks':old,'cpu_affinity':sorted(os.sched_getaffinity(0)),'cgroup':cg,'memory':mem,'disk_free':shutil.disk_usage(ancestor).free}}))
'''
    result = ops.remote_python(code, port, timeout=60)
    if result['exit_code']:
        raise RuntimeError('Resource probe failed on '+str(port))
    return json.loads(result['stdout'].splitlines()[-1])


def admission(observation, release, commit, max_age=60):
    from satellite_rl.workbench.fair20_guard import admission_reasons
    from pathlib import Path
    receipt=None
    evidence=release.get('evidence_path')
    if evidence and Path(evidence).is_file():
        receipt=json.loads(Path(evidence).read_text(encoding='utf-8-sig'))
    reasons=admission_reasons(observation,release,commit,receipt=receipt,max_age=max_age)
    if not evidence:reasons.append('release evidence missing')
    return reasons
