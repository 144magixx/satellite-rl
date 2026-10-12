"""Transport only. Formal science always enters the experiment workbench."""
from pathlib import Path, PurePosixPath
import json
import re
from scripts.s2p19 import ops as transport

SSH_PORTS = (20424, 20473)
API_PORTS = dict(zip(SSH_PORTS, (8960, 8961)))
BASE = '/home/u2024110224/first_stage_satellite_rl/s1_g16'
LOCAL = Path('D:/paper/outputs/experiments/first_stage_geometry_load_20261012')
DQN = '/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python'

def instance(port):
    if port not in SSH_PORTS:
        raise ValueError('Only the two authorized instances are allowed')
    root = f'{BASE}/{port}'
    return {'ssh_port': port, 'api_port': API_PORTS[port], 'root': root,
            'repo': root+'/repo', 'archive': root+'/archive',
            'ops': root+'/ops', 'data': root+'/data'}

def connect(port):
    instance(port)
    return transport.connect(port)

def remote_python(code, port, timeout=120):
    instance(port)
    return transport.remote_python(code, port, timeout=timeout)

def upload(path, remote_path, port):
    instance(port)
    target = PurePosixPath(remote_path)
    if (not target.is_relative_to(PurePosixPath(BASE)) or '..' in target.parts or
            not re.fullmatch(r'[A-Za-z0-9_./-]+', remote_path)):
        raise ValueError('G16 uploads must stay inside the isolated group')
    result = remote_python(
        f"import pathlib;pathlib.Path({remote_path!r}).parent.mkdir(parents=True,exist_ok=True)", port)
    if result['exit_code']:
        raise RuntimeError('Could not create isolated G16 upload parent: ' + result['stderr'][-500:])
    return transport.upload(Path(path), remote_path, port)

def api(port, path, body=None, method=None, timeout=60):
    instance(port)
    if not re.fullmatch(r'/api/[A-Za-z0-9_/-]+', path):
        raise ValueError('Invalid workbench path')
    method = method or ('POST' if body is not None else 'GET')
    if method not in ('GET', 'POST') or ((method == 'GET') != (body is None)):
        raise ValueError('Invalid workbench method/body')
    payload = None if body is None else json.dumps(body, ensure_ascii=False)
    code = f'''import json,urllib.request,urllib.error
data={payload!r}
r=urllib.request.Request('http://127.0.0.1:{API_PORTS[port]}{path}',data=None if data is None else data.encode(),method={method!r},headers={{'Content-Type':'application/json','Cache-Control':'no-cache'}})
try:
 with urllib.request.urlopen(r,timeout={timeout}) as response: print(json.dumps({{'status':response.status,'body':json.load(response)}}))
except urllib.error.HTTPError as e: print(json.dumps({{'status':e.code,'error':e.read().decode('utf-8','replace')[:2000]}}))
'''
    result = remote_python(code, port, timeout=timeout+30)
    if result['exit_code']:
        raise RuntimeError(f'{port} workbench transport: '+result['stderr'][-600:])
    value = json.loads(result['stdout'].splitlines()[-1])
    if value['status'] >= 400:
        raise RuntimeError(f'{port} HTTP {value["status"]}: '+value.get('error',''))
    return value['body']

