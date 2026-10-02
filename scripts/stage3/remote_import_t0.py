"""Register exact G11 T0 150k terminal policy in one stage 3 workbench."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import re
import sys


sys.path.insert(0, "D:/paper/outputs/verification/g11_remote_launch_20261001")
from remote_transport import connect, execute  # noqa: E402


BASE = "/home/u2024110224/third_stage_satellite_rl/s3_power_20261003"
DQN = "/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python"
T0 = "/home/u2024110224/second_stage_satellite_rl/g11/20418/archive/models/runs/workbench/train_sac_20261001_162126_33e89e95/final"
RUN_ID = "train_sac_20261001_162126_33e89e95"
POLICY_SHA = "ae095e8246d22b2716ba154315cee18584917d215580821cb53f9aeba78eb824"
METADATA_SHA = "1d4055e40ba6314ad8753ae64c1321a30fff7d7c3b25429b86ffca92c5388dde"
PORTS = {20418: 8772, 20424: 8773}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssh-port", type=int, choices=PORTS, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    commit = args.commit.lower()
    if not re.fullmatch("[0-9a-f]{40}", commit):
        raise ValueError("Full frozen source commit required")
    code = f'''
import datetime,hashlib,json,pathlib,subprocess,sys,urllib.request
from urllib.error import URLError
root=pathlib.Path({BASE!r})/{str(args.ssh_port)!r}
repo=root/'repo';archive=root/'archive';ops=root/'ops'
if subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()!={commit!r}:
 raise RuntimeError('Stage3 source commit differs')
sys.path.insert(0,str(repo/'src'))
import satellite_rl
from satellite_rl.workbench.checkpoints import load_policy
if pathlib.Path(satellite_rl.__file__).resolve()!=repo/'src/satellite_rl/__init__.py':
 raise RuntimeError('Model import package source differs')
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def api(method,path,body=None,timeout=120):
 data=None if body is None else json.dumps(body,ensure_ascii=False).encode()
 req=urllib.request.Request(f'http://127.0.0.1:{PORTS[args.ssh_port]}/api'+path,data=data,method=method,
  headers={{'Content-Type':'application/json','Cache-Control':'no-cache'}})
 with urllib.request.urlopen(req,timeout=timeout) as response:return json.load(response)
system=api('GET','/system')
if system.get('source_root')!=str(repo) or system.get('archive_root')!=str(archive):
 raise RuntimeError('Stage3 workbench source/archive differs')
path=pathlib.Path({T0!r})
if sha(path/'policy.pt')!={POLICY_SHA!r} or sha(path/'model.json')!={METADATA_SHA!r}:
 raise RuntimeError('Original G11 T0 final asset SHA changed')
policy=load_policy(path,device='cpu')
provenance=policy.metadata.get('provenance') or {{}}
if (provenance.get('run_id')!={RUN_ID!r} or provenance.get('env_step')!=150000
 or provenance.get('git_commit')!='85f667cd5731e284c19b2590b5dcf5308a58f234'):
 raise RuntimeError('Strict T0 load returned unexpected provenance')
def matches():
 return [item for item in api('GET','/models')['models']
  if item.get('sha256')=={POLICY_SHA!r} and (item.get('provenance') or {{}}).get('run_id')=={RUN_ID!r}]
found=matches()
if len(found)>1:raise RuntimeError('Duplicate G11 T0 imports')
if found:model=found[0]
else:
 try:model=api('POST','/models/import',{{'path':str(path),'name':'S3_G11_T0_fixed_150k'}},timeout=900)
 except (URLError,OSError,TimeoutError):
  found=matches()
  if len(found)!=1:raise RuntimeError('T0 import outcome unknown')
  model=found[0]
eligible=api('GET','/power-refinement/models')['models']
if model.get('sha256')!={POLICY_SHA!r} or model['id'] not in [item['id'] for item in eligible]:
 raise RuntimeError('T0 imported but absent from stage3 eligible model list')
receipt={{'status':'g11_t0_imported_strictly_no_evaluation',
 'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'ssh_port':{args.ssh_port},'workbench_port':{PORTS[args.ssh_port]},'commit':{commit!r},
 'model_id':model['id'],'model_path':model['path'],'policy_sha256':{POLICY_SHA!r},
 'original_metadata_sha256':{METADATA_SHA!r},'source_run_id':{RUN_ID!r},'env_step':150000}}
(ops/'t0_import_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
print(json.dumps(receipt,ensure_ascii=False))
'''
    encoded = base64.b64encode(code.encode()).decode()
    command = DQN + " -c 'exec(__import__(\"base64\").b64decode(\"" + encoded + "\"))'"
    transport = connect(args.ssh_port)
    try:
        result = execute(transport, command, timeout=1800)
    finally:
        transport.close()
    if result["exit_code"]:
        raise RuntimeError(result["stderr"][-1500:])
    receipt = json.loads(result["stdout"])
    destination = Path("D:/paper/outputs/experiments/stage3_power_20261003/preflight")
    path = destination / f"t0_import_{args.ssh_port}.json"
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"receipt": str(path), "model_id": receipt["model_id"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
