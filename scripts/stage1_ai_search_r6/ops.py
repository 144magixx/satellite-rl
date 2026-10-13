"""Reuse pinned SSH transport, with independent R6 roots and API boundaries."""
from pathlib import Path,PurePosixPath
import json,re

GROUP='S1-AI17-R6'
SSH_PORTS=(20418,20424,20473,20474)
API_PORTS=dict(zip(SSH_PORTS,(9006,9007,9008,9009)))
BASE='/home/u2024110224/first_stage_satellite_rl/s1_ai17'
LOCAL=Path('D:/paper/outputs/experiments/first_stage_ai_search_20261013/round6')
DQN='/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python'


def instance(port):
    if port not in SSH_PORTS:raise ValueError('R6 permits only four approved candidate lanes')
    root=f'{BASE}/{port}/round6'
    return {'ssh_port':port,'api_port':API_PORTS[port],'root':root,'repo':root+'/repo',
            'archive':root+'/archive','ops':root+'/ops','data':root+'/data'}


def transport():
    from scripts.s2p19 import ops as pinned
    return pinned


def connect(port):instance(port);return transport().connect(port)


def remote_python(code,port,timeout=90):
    # Script via stdin keeps a 120-source plan/POST out of Linux argv limits.
    instance(port);client=connect(port)
    try:
        stdin,stdout,stderr=client.exec_command(DQN+' -B -',timeout=timeout)
        stdin.write('import sys;sys.dont_write_bytecode=True\n'+code);stdin.flush();stdin.channel.shutdown_write()
        output=stdout.read().decode('utf-8','replace');error=stderr.read().decode('utf-8','replace')
        return {'exit_code':stdout.channel.recv_exit_status(),'stdout':output,'stderr':error}
    finally:client.close()


def upload(path,target,port):
    instance(port);p=PurePosixPath(target)
    if not p.is_relative_to(PurePosixPath(BASE)) or '..' in p.parts or not re.fullmatch(r'[A-Za-z0-9_./-]+',target):
        raise ValueError('R6 upload cannot write R1 or another experiment')
    result=remote_python(f'import pathlib;pathlib.Path({target!r}).parent.mkdir(parents=True,exist_ok=True)',port)
    if result['exit_code']:raise RuntimeError(result['stderr'][-600:])
    receipt=transport().upload(Path(path),target,port)
    if receipt['local_sha256']!=receipt['remote']:raise ValueError('uploaded bytes differ')
    return receipt


def api(port,path,body=None,timeout=45):
    instance(port)
    if not re.fullmatch(r'/api/ai-search-r6/[A-Za-z0-9_/-]+',path):raise ValueError('R6 API cannot operate R1')
    # Use the same pinned DQN transport without mutating R1's global port map.
    payload=None if body is None else json.dumps(body,ensure_ascii=False)
    code=f'''import urllib.request,urllib.error,json
data={payload!r};method='GET' if data is None else 'POST'
req=urllib.request.Request('http://127.0.0.1:{API_PORTS[port]}{path}',data=None if data is None else data.encode(),method=method,headers={{'Content-Type':'application/json','Cache-Control':'no-cache'}})
try:
 with urllib.request.urlopen(req,timeout={timeout}) as response:print(json.dumps({{'status':response.status,'body':json.load(response)}}))
except urllib.error.HTTPError as e:print(json.dumps({{'status':e.code,'error':e.read().decode('utf-8','replace')[:2000]}}))
'''
    result=remote_python(code,port,timeout=timeout+20)
    if result['exit_code']:raise RuntimeError(result['stderr'][-800:])
    parsed=json.loads(result['stdout'].splitlines()[-1])
    if parsed['status']>=400:raise RuntimeError(f'{port} HTTP{parsed["status"]}: '+parsed.get('error',''))
    return parsed['body']
