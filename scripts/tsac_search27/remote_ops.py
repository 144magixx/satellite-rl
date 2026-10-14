"""Explicit per-account W27 SSH transport. Never logs credentials or submits jobs."""
import base64,hashlib,json,os,shlex
from pathlib import Path,PurePosixPath
import paramiko

HOST='10.160.17.102'
LOCAL=Path('D:/paper/outputs/experiments/s2_tsac_search27_20261015')
REPO=Path(__file__).resolve().parents[2]
NODES={
 20418:dict(user='u2024110224',base='/home/u2024110224/second_stage_satellite_rl/s2_w27',python='/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python',api=9070),
 20424:dict(user='u2024110224',base='/home/u2024110224/second_stage_satellite_rl/s2_w27',python='/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python',api=9071),
 20627:dict(user='u2025140002',base='/home/u2025140002/second_stage_satellite_rl/s2_w27',python='/home/u2025140002/second_stage_satellite_rl/envs/DQN/bin/python',api=9072)}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as stream:
  for block in iter(lambda:stream.read(1048576),b''):h.update(block)
 return h.hexdigest()

def write_once(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 raw=(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
 if path.exists():
  if path.read_bytes()!=raw:raise ValueError('existing immutable local record differs: '+str(path))
 else:path.write_bytes(raw)
 return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())

def connect(port):
 node=NODES[port];client=paramiko.SSHClient();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
 client.connect(HOST,port=port,username=node['user'],password=os.environ['BUPT_PASSWORD'],look_for_keys=False,allow_agent=False,timeout=20,banner_timeout=20,auth_timeout=20)
 return client

def remote_python(port,code,*,timeout=60,interpreter=None):
 client=connect(port)
 try:
  command=shlex.quote(interpreter or NODES[port]['python'])+' -B -'
  stdin,stdout,stderr=client.exec_command(command,timeout=timeout)
  stdin.write(code.encode('utf8'));stdin.flush();stdin.channel.shutdown_write()
  out=stdout.read().decode('utf8',errors='replace');err=stderr.read().decode('utf8',errors='replace')
  return dict(exit_code=stdout.channel.recv_exit_status(),stdout=out,stderr=err)
 finally:client.close()

def read_remote(port,path):
 client=connect(port)
 try:
  with client.open_sftp()as sftp:
   with sftp.open(path,'rb')as stream:return stream.read()
 finally:client.close()

def publish(port,file,path):
 file=Path(file);expected=sha(file);dest=PurePosixPath(path);base=PurePosixPath(NODES[port]['base'])
 if not dest.is_relative_to(base)or'..'in dest.parts:raise ValueError('write outside isolated W27 account root')
 client=connect(port)
 try:
  with client.open_sftp()as sftp:
   try:
    with sftp.open(path,'rb')as f:old=f.read()
   except FileNotFoundError:old=None
   if old is not None:
    if hashlib.sha256(old).hexdigest()!=expected:raise ValueError('immutable remote file differs')
   else:
    parent=PurePosixPath('/')
    for piece in dest.parent.parts[1:]:
     parent/=piece
     try:sftp.stat(str(parent))
     except FileNotFoundError:sftp.mkdir(str(parent))
    sftp.put(str(file),path)
   with sftp.open(path,'rb')as f:
    h=hashlib.sha256()
    for block in iter(lambda:f.read(1048576),b''):h.update(block)
   if h.hexdigest()!=expected:raise ValueError('remote upload SHA mismatch')
 finally:client.close()
 return dict(path=path,sha256=expected)

def api(port,route,body=None,timeout=30):
 if not route.startswith('/')or any(c not in '/_-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'for c in route):raise ValueError('invalid API route')
 request=dict(url='http://127.0.0.1:'+str(NODES[port]['api'])+route,body=body,timeout=timeout)
 code='''
import urllib.request,urllib.error,json
r=REQUEST
req=urllib.request.Request(r['url'],data=None if r['body']is None else json.dumps(r['body']).encode(),headers={'Content-Type':'application/json'})
try:
 with urllib.request.urlopen(req,timeout=r['timeout'])as stream:print(json.dumps(dict(status=stream.status,body=json.load(stream))))
except urllib.error.HTTPError as e:print(json.dumps(dict(status=e.code,error=e.read().decode())))
'''.replace('REQUEST',repr(request))
 raw=remote_python(port,code,timeout=timeout+5)
 if raw['exit_code']:raise RuntimeError(raw['stderr'][-1000:])
 return json.loads(raw['stdout'])
