"""Pinned transport; only this group's workbench may create science tasks."""
from pathlib import Path, PurePosixPath
import base64
import json
import re
from scripts.s2p19 import ops as transport

SSH_PORTS = (20418, 20424, 20473, 20474)
API_PORTS = dict(zip(SSH_PORTS, (8970, 8971, 8972, 8973)))
BASE = '/home/u2024110224/first_stage_satellite_rl/s1_ai17'
LOCAL = Path('D:/paper/outputs/experiments/first_stage_ai_search_20261013')
DQN = '/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python'


def instance(port):
    if port not in SSH_PORTS: raise ValueError('Only the four authorized AI17 lanes')
    root = f'{BASE}/{port}'
    return {'ssh_port': port, 'api_port': API_PORTS[port], 'root': root,
            'repo': root+'/repo', 'archive': root+'/archive', 'ops': root+'/ops', 'data': root+'/data'}


def connect(port):
    instance(port)
    return transport.connect(port)


def remote_python(code, port, timeout=90):
    instance(port)
    return transport.remote_python('import sys;sys.dont_write_bytecode=True\n'+code, port, timeout=timeout)


def upload(path, remote_path, port):
    instance(port); target = PurePosixPath(remote_path)
    if (not target.is_relative_to(PurePosixPath(BASE)) or '..' in target.parts or
            not re.fullmatch(r'[A-Za-z0-9_./-]+', remote_path)):
        raise ValueError('AI17 upload must remain inside the group')
    result = remote_python(f'import pathlib;pathlib.Path({remote_path!r}).parent.mkdir(parents=True,exist_ok=True)', port)
    if result['exit_code']: raise RuntimeError(result['stderr'][-500:])
    result = transport.upload(Path(path), remote_path, port)
    if result['local_sha256'] != result['remote']: raise RuntimeError('uploaded bytes differ')
    return result


def api(port, path, body=None, method=None, timeout=60):
    instance(port)
    if not re.fullmatch(r'/api/ai-search/[A-Za-z0-9_/-]+', path):
        raise ValueError('AI17 API cannot alter another experiment')
    method = method or ('POST' if body is not None else 'GET')
    if method not in ('GET', 'POST') or ((method == 'GET') != (body is None)):
        raise ValueError('invalid API method/body')
    payload = None if body is None else json.dumps(body, ensure_ascii=False)
    code = f'''import json,urllib.request,urllib.error,sys
data=sys.stdin.read() if {payload is not None!r} else None
r=urllib.request.Request('http://127.0.0.1:{API_PORTS[port]}{path}',data=None if data is None else data.encode(),method={method!r},headers={{'Content-Type':'application/json','Cache-Control':'no-cache'}})
try:
 with urllib.request.urlopen(r,timeout={timeout}) as response: print(json.dumps({{'status':response.status,'body':json.load(response)}}))
except urllib.error.HTTPError as e: print(json.dumps({{'status':e.code,'error':e.read().decode('utf-8','replace')[:2000]}}))
'''
    encoded = base64.b64encode(code.encode()).decode('ascii')
    command = f'{DQN} -B -c "import base64;exec(base64.b64decode(\'{encoded}\'))"'
    client = connect(port)
    try:
        stdin, stdout, stderr = client.exec_command(command, timeout=timeout+20)
        if payload is not None: stdin.write(payload.encode('utf-8')); stdin.flush()
        stdin.channel.shutdown_write()
        out = stdout.read().decode(); err = stderr.read().decode(); exit_code = stdout.channel.recv_exit_status()
    finally: client.close()
    if exit_code: raise RuntimeError(f'{port} workbench transport: '+err[-700:])
    value = json.loads(out.splitlines()[-1])
    if value['status'] >= 400: raise RuntimeError(f'{port} HTTP {value["status"]}: '+value.get('error', ''))
    return value['body']
