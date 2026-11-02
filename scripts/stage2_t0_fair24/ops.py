"""S2-T24 transport; explicit four user-authorized lanes, own-root writes only."""
from pathlib import Path,PurePosixPath
import json,base64,os,sys
PORTS={20418:9040,20424:9041,20473:9042,20474:9043};BASE='/home/u2024110224/second_stage_satellite_rl/s2_t24';DQN='/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python'
def connect(port):
 if port not in PORTS:raise ValueError('unauthorized lane')
 import paramiko
 password=os.environ['BUPT_PASSWORD'];client=paramiko.SSHClient();client.set_missing_host_key_policy(paramiko.AutoAddPolicy());client.connect('10.160.17.102',port=port,username='u2024110224',password=password,timeout=20,look_for_keys=False,allow_agent=False);return client
def api(port,path,body=None,timeout=60):
 if not path.startswith('/api/t0compare/') or '..' in path:raise ValueError('only own typed workbench API')
 code='import json,sys,urllib.request;p=sys.stdin.read();r=urllib.request.Request('+repr('http://127.0.0.1:'+str(PORTS[port])+path)+',data=p.encode() if p else None,headers={"Content-Type":"application/json"});print(json.dumps(json.load(urllib.request.urlopen(r,timeout='+str(timeout)+'))))';encoded=base64.b64encode(code.encode()).decode();client=connect(port)
 try:
  stdin,out,err=client.exec_command(DQN+' -B -c "import base64;exec(base64.b64decode(\''+encoded+'\'))"',timeout=timeout+30)
  if body is not None:stdin.write(json.dumps(body).encode());stdin.flush()
  stdin.channel.shutdown_write();text=out.read().decode();errors=err.read().decode();exitcode=out.channel.recv_exit_status()
 finally:client.close()
 if exitcode:raise RuntimeError('workbench transport '+str(exitcode)+': '+errors[-2000:])
 return json.loads(text)
def own_path(port,path):
 target=PurePosixPath(path);root=PurePosixPath(BASE)/str(port)
 if port not in PORTS or '..' in target.parts or not target.is_relative_to(root) or target==root:raise ValueError('write outside isolated own lane')
 return str(target)
