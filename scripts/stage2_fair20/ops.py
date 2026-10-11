"""Transport only. Formal science always enters the experiment workbench."""
from pathlib import Path, PurePosixPath
import json
import os
import base64
import re
from scripts.s2p19 import ops as transport

SSH_PORTS = (20418, 20424, 20473, 20474)
API_PORTS = dict(zip(SSH_PORTS, (8950, 8951, 8952, 8953)))
BASE = '/home/u2024110224/second_stage_satellite_rl/s2_f20'
LOCAL = Path(os.environ.get('SATELLITE_RL_F20_CONTROL_ROOT', 'D:/paper/outputs/experiments/s2_fair20_20261012'))
DQN = '/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python'

def instance(port):
    if port not in SSH_PORTS:
        raise ValueError('Only the four authorized instances are allowed')
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
    if port != 20474:raise ValueError('latest user allocation permits writes only on 20474')
    target = PurePosixPath(remote_path)
    if (not target.is_relative_to(PurePosixPath(instance(port)['root'])) or '..' in target.parts or
            not re.fullmatch(r'[A-Za-z0-9_./-]+', remote_path)):
        raise ValueError('F20 uploads must stay inside the isolated group')
    return transport.upload(Path(path), remote_path, port)

def api(port, path, body=None, method=None, timeout=60):
    instance(port)
    if not re.fullmatch(r'/api/[A-Za-z0-9_/-]+', path):
        raise ValueError('Invalid workbench path')
    method = method or ('POST' if body is not None else 'GET')
    if method not in ('GET', 'POST') or ((method == 'GET') != (body is None)):
        raise ValueError('Invalid workbench method/body')
    if method == 'POST' and port != 20474:raise ValueError('latest user allocation permits scientific tasks only on 20474')
    payload = None if body is None else json.dumps(body, ensure_ascii=False)
    use_stdin = payload is not None and len(payload.encode('utf-8')) > 64 * 1024
    data_expression = "__import__('sys').stdin.read()" if use_stdin else repr(payload)
    code = f'''import json,urllib.request,urllib.error
data={data_expression}
r=urllib.request.Request('http://127.0.0.1:{API_PORTS[port]}{path}',data=None if data is None else data.encode(),method={method!r},headers={{'Content-Type':'application/json','Cache-Control':'no-cache'}})
try:
 with urllib.request.urlopen(r,timeout={timeout}) as response: print(json.dumps({{'status':response.status,'body':json.load(response)}}))
except urllib.error.HTTPError as e: print(json.dumps({{'status':e.code,'error':e.read().decode('utf-8','replace')[:2000]}}))
'''
    if use_stdin:
        encoded = base64.b64encode(code.encode()).decode('ascii')
        command = f'{DQN} -c "import base64;exec(base64.b64decode(\'{encoded}\'))"'
        client = connect(port)
        try:
            stdin, stdout, stderr = client.exec_command(command, timeout=timeout+30)
            stdin.write(payload.encode('utf-8'))
            stdin.flush()
            stdin.channel.shutdown_write()
            result = {'stdout': stdout.read().decode('utf-8'), 'stderr': stderr.read().decode('utf-8'), 'exit_code': stdout.channel.recv_exit_status()}
        finally:
            client.close()
    else:
        result = remote_python(code, port, timeout=timeout+30)
    if result['exit_code']:
        raise RuntimeError(f'{port} workbench transport: '+result['stderr'][-600:])
    value = json.loads(result['stdout'].splitlines()[-1])
    if value['status'] >= 500:
        raise ConnectionError(f'{port} workbench temporary HTTP {value["status"]}')
    if value['status'] >= 400:
        raise RuntimeError(f'{port} HTTP {value["status"]}: '+value.get('error',''))
    return value['body']
