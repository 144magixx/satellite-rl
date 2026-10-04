"""Freeze, launch and observe Jan08/09 half-grid mother preparation on 20424."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ops_common import BASE, DQN, LOCAL, remote_file, remote_python, sha256, write_json


HERE = Path(__file__).resolve().parent
ROOT = BASE + "/dev/replenishment_source_20261004T0500Z"
DATA_SOURCE = BASE + "/dev/data_source_20261003T191000Z"
DATA = BASE + "/data"
AUTH = Path("D:/paper/outputs/experiments/variable_beams_20261004/replenishment/authorization.json")
OUTPUT = Path("D:/paper/outputs/experiments/variable_beams_20261004/replenishment")
G11 = "/home/u2024110224/second_stage_satellite_rl/g11/20424/archive/outputs/workbench/scenes/scenes_20261001_161621_17ceaa35/record.json"
FILES = {
    "prepare_replenishment.py": HERE / "prepare_replenishment.py",
    "prepare_snapshots_reference.py": Path("D:/paper/outputs/experiments/europe_adsb_20260928/prepare_snapshots.py"),
}


def freeze() -> dict:
    expected = {name: sha256(path) for name, path in FILES.items()}
    authorization_sha = sha256(AUTH)
    remote_python(f'''
import json,pathlib
root=pathlib.Path({ROOT!r});auth=pathlib.Path({(DATA + '/replenishment/authorization.json')!r})
if root.exists():raise RuntimeError('Replenishment source snapshot already exists')
root.mkdir(parents=True);auth.parent.mkdir(parents=True,exist_ok=True)
print(json.dumps({{'ready':True,'root':str(root)}}))
''')
    for name, path in FILES.items():
        remote_file(path, ROOT + "/" + name)
    remote_file(AUTH, DATA + "/replenishment/authorization.json.part")
    report = remote_python(f'''
import datetime,hashlib,json,pathlib
root=pathlib.Path({ROOT!r});base=pathlib.Path({DATA_SOURCE!r});expected={expected!r}
auth=pathlib.Path({(DATA + '/replenishment/authorization.json')!r});part=pathlib.Path(str(auth)+'.part')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,digest in expected.items():
 if sha(root/name)!=digest:raise RuntimeError('Replenishment script SHA differs: '+name)
if sha(part)!={authorization_sha!r}:raise RuntimeError('Authorization SHA differs')
if auth.exists():
 if sha(auth)!={authorization_sha!r}:raise RuntimeError('Existing authorization differs')
 part.unlink()
else:part.rename(auth)
manifest={{'status':'frozen_input_preparation_source_only',
 'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'target':'u2024110224@10.160.17.102:20424','root':str(root),
 'base_data_source':str(base),'base_data_source_manifest_sha256':sha(base/'source_manifest.json'),
 'authorization_sha256':sha(auth),'files':expected}}
(root/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
for p in root.iterdir():p.chmod(0o444)
print(json.dumps({{'status':manifest['status'],'root':str(root),
 'manifest_sha256':sha(root/'source_manifest.json'),'authorization_sha256':sha(auth),
 'files':expected}},ensure_ascii=False))
''')
    path = OUTPUT / "source_freeze.json"
    write_json(path, report)
    return {"receipt": str(path), **report}


def start() -> dict:
    code = f'''
import datetime,hashlib,json,os,pathlib,subprocess
base=pathlib.Path({BASE!r});root=pathlib.Path({ROOT!r});data=pathlib.Path({DATA!r})
manifest=json.loads((root/'source_manifest.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as source:
  for chunk in iter(lambda:source.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
for name,digest in manifest['files'].items():
 if sha(root/name)!=digest:raise RuntimeError('Frozen replenishment source changed: '+name)
auth=data/'replenishment/authorization.json'
if sha(auth)!=manifest['authorization_sha256']:raise RuntimeError('Authorization changed')
if sha(pathlib.Path({DATA_SOURCE!r})/'source_manifest.json')!=manifest['base_data_source_manifest_sha256']:
 raise RuntimeError('Frozen aviation data source changed')
for name,size,expected in (('v2024.01.08-planes-readsb-prod-0.tar',1610444800,'a0b35aed430b8cd7b4de3fef3db79ce8efee958a2a1f2daa32056731db9074bd'),
                           ('v2024.01.09-planes-readsb-prod-0.tar',1443092480,'df67efea6c4502177480a08e46fe91dc59f9e4c54c31c88bd1301c91f964b12e')):
 path=data/'raw'/name
 if not path.is_file() or path.stat().st_size!=size or sha(path)!=expected:
  raise RuntimeError('Original Jan08/09 tar missing or SHA differs: '+name)
ops=data/'ops';ops.mkdir(parents=True,exist_ok=True)
launch=ops/'replenishment_launch.json'
if launch.exists():
 old=json.loads(launch.read_text());proc=pathlib.Path('/proc')/str(old['pid'])
 if proc.is_dir():
  stat=(proc/'stat').read_text();birth=int(stat[stat.rfind(')')+2:].split()[19])
  if birth==old['start_ticks']:
   print(json.dumps({{'status':'already_running','launch':old}}));raise SystemExit
 if (data/'prepared/train_mothers_replaced576/record.json').is_file():
  print(json.dumps({{'status':'already_completed','launch':old}}));raise SystemExit
 raise RuntimeError('Prior replenishment worker exited; inspect failure before restart')
env=os.environ.copy();env.update(PYTHONPATH={DATA_SOURCE!r}+'/src',PYTHONDONTWRITEBYTECODE='1',
 VARIABLE_BASE={BASE!r},VARIABLE_DATA_SOURCE={DATA_SOURCE!r},
 VARIABLE_G11_RECORD={G11!r},VARIABLE_AUTHORIZATION=str(auth),
 SATELLITE_RL_OUTPUT_ROOT=str(data/'scratch'),
 OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
probe=subprocess.run([{DQN!r},'-B','-c',
 'import json,sys,torch,satellite_rl;print(json.dumps({{"python":sys.executable,"torch":torch.__version__,"cuda_build":torch.version.cuda,"cuda":torch.cuda.is_available(),"package":satellite_rl.__file__}}))'],
 env=env,capture_output=True,text=True,timeout=90,check=True)
runtime=json.loads(probe.stdout.splitlines()[-1])
if runtime['cuda'] is not True or pathlib.Path(runtime['package']).resolve()!=pathlib.Path({DATA_SOURCE!r})/'src/satellite_rl/__init__.py':
 raise RuntimeError('DQN/CUDA or frozen data package import mismatch')
log=ops/'replenishment.log'
with log.open('ab') as output:
 process=subprocess.Popen([{DQN!r},'-B',str(root/'prepare_replenishment.py')],
  env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
  start_new_session=True,close_fds=True)
stat=(pathlib.Path('/proc')/str(process.pid)/'stat').read_text()
ticks=int(stat[stat.rfind(')')+2:].split()[19])
receipt={{'status':'preparing_56_mother_inputs_only',
 'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'target':'u2024110224@10.160.17.102:20424','pid':process.pid,'start_ticks':ticks,
 'source_manifest_sha256':sha(root/'source_manifest.json'),
 'authorization_sha256':sha(auth),'runtime':runtime,
 'status_path':str(data/'replenishment/preparation_status.json'),
 'selection_contract':str(data/'replenishment/selection_contract.json'),
 'output_record':str(data/'prepared/train_mothers_replaced576/record.json'),'log':str(log)}}
launch.write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
print(json.dumps(receipt,ensure_ascii=False))
'''
    result = remote_python(code, timeout=240)
    path = OUTPUT / "launch.json"
    write_json(path, result)
    return {"receipt": str(path), **result}


def status() -> dict:
    report = remote_python(f'''
import datetime,json,pathlib
data=pathlib.Path({DATA!r});ops=data/'ops';launch=ops/'replenishment_launch.json'
record=json.loads(launch.read_text()) if launch.is_file() else None
birth=None
if record:
 proc=pathlib.Path('/proc')/str(record['pid'])
 if proc.is_dir():
  stat=(proc/'stat').read_text();birth=int(stat[stat.rfind(')')+2:].split()[19])
state=data/'replenishment/preparation_status.json';log=ops/'replenishment.log'
print(json.dumps({{'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'launch':record,'process_birth_matches':birth==record['start_ticks'] if record else None,
 'status':json.loads(state.read_text()) if state.is_file() else None,
 'record_exists':(data/'prepared/train_mothers_replaced576/record.json').is_file(),
 'log_tail':log.read_text(errors='replace')[-700:] if log.is_file() else None}},ensure_ascii=False))
''')
    path = OUTPUT / "status.json"
    write_json(path, report)
    return {"receipt": str(path), **report}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "start", "status"))
    action = parser.parse_args().action
    print(json.dumps({"freeze": freeze, "start": start, "status": status}[action](), ensure_ascii=False))


if __name__ == "__main__":
    main()
