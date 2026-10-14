"""Bounded transport on the two user-authorized W26 endpoints."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import base64
import json
import os
import shlex

PORTS = (20418, 20424)
HOST = '10.160.17.102'
USER = 'u2024110224'
DQN = '/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python'
BASE = '/home/u2024110224/second_stage_satellite_rl/s2_w26'
API_PORTS = {20418: 9060, 20424: 9061}


def connect(port):
    if port not in PORTS:
        raise ValueError('W26 may use only 20418/20424')
    import paramiko
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=port, username=USER, password=os.environ['BUPT_PASSWORD'],
                   timeout=20, auth_timeout=20, banner_timeout=20, look_for_keys=False, allow_agent=False)
    return client


def remote_python(port, code, timeout=60, interpreter='python3'):
    encoded = base64.b64encode(code.encode('utf-8')).decode('ascii')
    command = shlex.quote(interpreter) + ' -B -c ' + shlex.quote(
        'import base64;exec(base64.b64decode(' + repr(encoded) + '))')
    client = connect(port)
    try:
        _, stdout, stderr = client.exec_command(command, timeout=timeout)
        output, error = stdout.read().decode('utf-8', 'replace'), stderr.read().decode('utf-8', 'replace')
        return {'exit_code': stdout.channel.recv_exit_status(), 'stdout': output, 'stderr': error}
    finally:
        client.close()


def probe(port):
    code = '''
import datetime,json,os,pathlib,shutil,subprocess,urllib.request
root=pathlib.Path('/proc'); processes=[]
for entry in root.iterdir():
 if not entry.name.isdigit() or int(entry.name)==os.getpid(): continue
 try:
  argv=(entry/'cmdline').read_bytes().split(b'\\0')
  command=b' '.join(argv).decode('utf-8','replace')
  if any(x in command for x in ('satellite_rl','s2_t24','s2_w26','s1_ai17','workbench','train_sac')):
   # Omit transient python -c wrappers that embed this probe as argv text.
   if b'-c' in argv: continue
   stat=(entry/'stat').read_text(); fields=stat[stat.rfind(')')+2:].split()
   processes.append(dict(pid=int(entry.name),birth_ticks=int(fields[19]),state=fields[0],command=command))
 except (FileNotFoundError,PermissionError,ProcessLookupError): pass
def command(args):
 try:
  result=subprocess.run(args,text=True,capture_output=True,timeout=25)
  return dict(exit_code=result.returncode,stdout=result.stdout.strip(),stderr=result.stderr.strip())
 except Exception as error:return dict(error=type(error).__name__)
apis={}
for p in (9040,9041,9060,9061):
 try:
  with urllib.request.urlopen('http://127.0.0.1:%d/api/status'%p,timeout=2) as reply:apis[p]=dict(status=reply.status,body=reply.read(20000).decode())
 except Exception as error:apis[p]=dict(error=type(error).__name__)
print(json.dumps(dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 boot_id=(root/'sys/kernel/random/boot_id').read_text().strip(),processes=processes,
 gpu=command(['nvidia-smi','--query-gpu=uuid,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),
 compute=command(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits']),
 cpu_affinity=sorted(os.sched_getaffinity(0)),load=os.getloadavg(),disk_free=shutil.disk_usage('/home').free,
 memory=[line for line in (root/'meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:'))],
 cgroup=(root/'self/cgroup').read_text(),apis=apis,
 dqn=command([DQN,'-B','-c',"import sys,torch,importlib.util,json; s=importlib.util.find_spec('satellite_rl'); print(json.dumps(dict(python=sys.executable,torch=str(torch.__version__),cuda=torch.version.cuda,available=torch.cuda.is_available(),package=None if s is None else s.origin)))"]))))
'''.replace('DQN', repr(DQN))
    result = remote_python(port, code, timeout=60)
    result['ssh_port'] = port
    if result['exit_code'] == 0:
        result['observation'] = json.loads(result['stdout'].splitlines()[-1])
    return result


def upload(port, local, remote):
    allowed = PurePosixPath(BASE)
    target = PurePosixPath(remote)
    if not target.is_relative_to(allowed) or '..' in target.parts:
        raise ValueError('writes must stay inside isolated W26 root')
    client = connect(port)
    try:
        sftp = client.open_sftp()
        current = PurePosixPath('/')
        for part in target.parent.parts[1:]:
            current /= part
            try:sftp.stat(str(current))
            except FileNotFoundError:sftp.mkdir(str(current))
        sftp.put(str(local), str(target))
        sftp.close()
    finally:client.close()


def api(port, route, body=None, timeout=60):
    if not route.startswith('/api/') or any(c not in '/_-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789' for c in route):
        raise ValueError('invalid API route')
    code = '''
import json,urllib.request,urllib.error
data=BODY
req=urllib.request.Request(URL,data=None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json'})
try:
 with urllib.request.urlopen(req,timeout=TIMEOUT) as reply:print(json.dumps(dict(status=reply.status,body=json.load(reply))))
except urllib.error.HTTPError as error:print(json.dumps(dict(status=error.code,error=error.read().decode())))
'''.replace('BODY', repr(body)).replace('URL', repr(f'http://127.0.0.1:{API_PORTS[port]}{route}')).replace('TIMEOUT', str(timeout))
    result = remote_python(port, code, timeout=timeout+5)
    if result['exit_code']:
        raise RuntimeError(result['stderr'][-1500:])
    return json.loads(result['stdout'].splitlines()[-1])


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args=parser.parse_args(); output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise ValueError('new observation file required')
    def safe(port):
        try:return probe(port)
        except Exception as error:return dict(ssh_port=port,error=type(error).__name__,detail=str(error))
    with ThreadPoolExecutor(max_workers=2) as pool: results=list(pool.map(safe,PORTS))
    output.write_text(json.dumps(dict(checked_utc=datetime.now(timezone.utc).isoformat(),results=results),ensure_ascii=False,indent=2),encoding='utf-8')
    for result in results:
        observed=result.get('observation',{})
        print(json.dumps(dict(port=result['ssh_port'],error=result.get('error'),exit_code=result.get('exit_code'),
            gpu=observed.get('gpu'),compute=observed.get('compute'),dqn=observed.get('dqn'),
            processes=len(observed.get('processes',[])),disk_free=observed.get('disk_free')),ensure_ascii=False))


if __name__=='__main__':main()
