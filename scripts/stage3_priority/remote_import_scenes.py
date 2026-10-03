"""Import one frozen Jan27 scene set into an isolated stage 3 workbench.

Run once per SSH instance after its source commit and workbench are verified.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import re
import sys


sys.path.insert(0, "D:/paper/outputs/verification/g11_remote_launch_20261001")
from remote_transport import connect, execute  # noqa: E402


BASE = "/home/u2024110224/third_stage_satellite_rl/s3_si_priority_20261003"
DQN = "/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python"
PORTS = {20418: 8776, 20424: 8777}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssh-port", type=int, choices=PORTS, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    commit = args.commit.lower()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("A full frozen commit is required")
    code = f'''
import datetime,hashlib,json,pathlib,subprocess,sys,urllib.request
from urllib.error import URLError
import numpy as np
base=pathlib.Path({BASE!r})
root=base/{str(args.ssh_port)!r}
repo=root/'repo';archive=root/'archive';ops=root/'ops';shared=base/'shared_data'
port={PORTS[args.ssh_port]};commit={commit!r}
head=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
if head!=commit:raise RuntimeError('Stage3 repo commit differs from requested frozen commit')
sys.path.insert(0,str(repo/'src'))
import satellite_rl
if pathlib.Path(satellite_rl.__file__).resolve()!=repo/'src/satellite_rl/__init__.py':
 raise RuntimeError('Stage3 package import is not from frozen repo')
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as source:
  for block in iter(lambda:source.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def api(method,path,body=None,timeout=120):
 data=None if body is None else json.dumps(body,ensure_ascii=False).encode()
 request=urllib.request.Request(f'http://127.0.0.1:{{port}}/api'+path,data=data,method=method,
  headers={{'Content-Type':'application/json','Cache-Control':'no-cache'}})
 with urllib.request.urlopen(request,timeout=timeout) as response:return json.load(response)
system=api('GET','/system')
if system.get('source_root')!=str(repo) or system.get('archive_root')!=str(archive) or system.get('cuda') is not True:
 raise RuntimeError('Stage3 workbench source/archive/CUDA mismatch')
verification=json.loads((shared/'independent_test_verification.json').read_text())
if verification.get('status')!='verified_input_scenes_only_not_model_evaluation' or verification.get('source_date')!='2024-01-27' or verification.get('scene_count')!=100:
 raise RuntimeError('Jan27 100-scene input verification missing')
record_path=shared/'staged_test/record.json';record=json.loads(record_path.read_text())
if sha(record_path)!=verification['record_sha256'] or record.get('count')!=100 or record['source']['source_dates']!=['2024-01-27']:
 raise RuntimeError('Jan27 frozen scene record changed')
expected=[item['sha256'] for item in record['files']]
for item in record['files']:
 if sha(pathlib.Path(item['path']))!=item['sha256']:raise RuntimeError('Jan27 source scene SHA changed')
def matches():
 return [item for item in api('GET','/scenes')['scenes'] if item.get('source',{{}}).get('reimport',{{}}).get('record_id')==record['id']]
found=matches()
if len(found)>1:raise RuntimeError('Duplicate Jan27 imports')
if found:imported=found[0]
else:
 try:imported=api('POST','/scenes/import',{{'path':str(shared/'staged_test'),'name':'S3_Jan27_independent_100','profile':'europe_v1'}},timeout=900)
 except (URLError,OSError,TimeoutError):
  found=matches()
  if len(found)!=1:raise RuntimeError('Scene import outcome unknown')
  imported=found[0]
if imported['count']!=100 or imported['source']['source_dates']!=['2024-01-27'] or imported['profile']!='europe_v1':
 raise RuntimeError('Imported count/date/profile mismatch')
if [item['sha256'] for item in imported['source']['reimport']['files']]!=expected:
 raise RuntimeError('Imported original scene order SHA mismatch')
for index,(left,right) in enumerate(zip(record['files'],imported['files'])):
 with np.load(left['path'],allow_pickle=False) as a,np.load(right['path'],allow_pickle=False) as b:
  if set(a.files)!=set(b.files) or any(a[k].dtype!=b[k].dtype or not np.array_equal(a[k],b[k]) for k in a.files):
   raise RuntimeError(f'Imported array differs at scene {{index}}')
target=archive/'outputs/workbench/scenes'/imported['id']/'record.json'
receipt={{'status':'imported_no_model_evaluation','checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'ssh_port':{args.ssh_port},'workbench_port':port,'commit':commit,'scene_id':imported['id'],
 'count':100,'date_utc':'2024-01-27','original_record_sha256':verification['record_sha256'],
 'ordered_source_scene_sha256':hashlib.sha256(''.join(expected).encode()).hexdigest(),
 'imported_record_sha256':sha(target),'all_arrays_equal':True}}
(ops/'jan27_scene_import_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
print(json.dumps(receipt,ensure_ascii=False))
'''
    encoded = base64.b64encode(code.encode()).decode()
    command = DQN + " -B -c 'exec(__import__(\"base64\").b64decode(\"" + encoded + "\"))'"
    transport = connect(args.ssh_port)
    try:
        result = execute(transport, command, timeout=1800)
    finally:
        transport.close()
    if result["exit_code"]:
        raise RuntimeError(result["stderr"][-1500:])
    print(result["stdout"])


if __name__ == "__main__":
    main()
