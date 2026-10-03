"""Launch and observe isolated Jan29 mother input preparation on 20424."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


TRANSPORT = Path("D:/paper/outputs/verification/g11_remote_launch_20261001")
sys.path.insert(0, str(TRANSPORT))
from remote_transport import connect, execute, paramiko  # noqa: E402

HERE = Path(__file__).resolve().parent
OUTPUT = Path("D:/paper/outputs/experiments/variable_beams_20261004/preflight")
ROOT = "/home/u2024110224/second_stage_satellite_rl/variable_beams_20261004/20424"
DATA = ROOT + "/data"
SOURCE = ROOT + "/dev/data_source_20261003T191000Z"
PREP = ROOT + "/dev/jan29_mother_source_20261003T191300Z"
G11 = "/home/u2024110224/second_stage_satellite_rl/g11/20424/archive/outputs/workbench/scenes/scenes_20261001_161621_17ceaa35/record.json"
DQN = "/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python"
FILES = {
    "prepare_jan29_mothers.py": HERE / "prepare_jan29_mothers.py",
    "prepare_snapshots_reference.py": Path("D:/paper/scripts/stage3_priority/prepare_snapshots_reference.py"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def call(transport, code: str, timeout=120) -> dict:
    encoded = base64.b64encode(code.encode()).decode()
    result = execute(transport, "python3 -B -c 'exec(__import__(\"base64\").b64decode(\"" + encoded + "\"))'", timeout=timeout)
    if result["exit_code"]:
        raise RuntimeError(result["stderr"][-1200:])
    return json.loads(result["stdout"])


def freeze() -> dict:
    local_sha = {name: sha256(path) for name, path in FILES.items()}
    transport = connect(20424)
    try:
        call(transport, f'''
import json,pathlib
target=pathlib.Path({PREP!r})
if target.exists():raise RuntimeError('Jan29 prep source already exists')
target.mkdir(parents=True)
print(json.dumps({{'status':'created','target':str(target)}}))
''')
        with paramiko.SFTPClient.from_transport(transport) as sftp:
            for name, path in FILES.items():
                sftp.put(str(path), PREP + "/" + name)
        report = call(transport, f'''
import datetime,hashlib,json,pathlib
root=pathlib.Path({PREP!r});source=pathlib.Path({SOURCE!r});expected={local_sha!r}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,digest in expected.items():
 if sha(root/name)!=digest:raise RuntimeError('Jan29 prep script SHA mismatch')
manifest={{'status':'frozen_jan29_mother_preparation_source','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'target':'u2024110224@10.160.17.102:20424','root':str(root),
 'base_data_source':str(source),'base_data_source_manifest_sha256':sha(source/'source_manifest.json'),
 'files':expected}}
(root/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
for p in root.iterdir():p.chmod(0o444)
print(json.dumps({{'status':manifest['status'],'root':str(root),'manifest_sha256':sha(root/'source_manifest.json'),
 'files':expected}},ensure_ascii=False))
''')
    finally:
        transport.close()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "jan29_mother_source_freeze.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"receipt": str(path), **report}


def start() -> dict:
    code = f'''
import datetime,hashlib,json,os,pathlib,subprocess
root=pathlib.Path({ROOT!r});data=pathlib.Path({DATA!r});source=pathlib.Path({SOURCE!r});prep=pathlib.Path({PREP!r})
manifest=prep/'source_manifest.json';frozen=json.loads(manifest.read_text())
for name,digest in frozen['files'].items():
 if hashlib.sha256((prep/name).read_bytes()).hexdigest()!=digest:
  raise RuntimeError('Frozen Jan29 preparation file changed: '+name)
if hashlib.sha256((source/'source_manifest.json').read_bytes()).hexdigest()!=frozen['base_data_source_manifest_sha256']:
 raise RuntimeError('Base data source manifest changed')
download=json.loads((data/'raw/download_status.json').read_text())
if download.get('status')!='completed' or download.get('date_utc')!='2024-01-29':
 raise RuntimeError('Official Jan29 archive download incomplete')
ops=data/'ops';ops.mkdir(parents=True,exist_ok=True)
launch_path=ops/'jan29_mother_launch.json'
if launch_path.is_file():
 old=json.loads(launch_path.read_text());proc=pathlib.Path('/proc')/str(old['pid'])
 if proc.is_dir():
  stat=(proc/'stat').read_text();birth=int(stat[stat.rfind(')')+2:].split()[19])
  if birth==old['start_ticks']:
   print(json.dumps({{'status':'already_running','launch':old}}));raise SystemExit
 if (data/'jan29_mother/record.json').is_file():
  print(json.dumps({{'status':'already_completed','launch':old}}));raise SystemExit
 raise RuntimeError('Previous Jan29 mother preparation failed; inspect status before any retry')
env=os.environ.copy();env.update(PYTHONPATH=str(source/'src'),PYTHONDONTWRITEBYTECODE='1',
 STAGE3_G07_FROZEN_SRC=str(source/'src'),VARIABLE_DATA_ROOT=str(data),
 VARIABLE_DATA_SOURCE=str(source),VARIABLE_G11_RECORD={G11!r},
 SATELLITE_RL_OUTPUT_ROOT=str(data/'scratch'),
 OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
log=ops/'jan29_mother.log'
with log.open('ab') as output:
 process=subprocess.Popen([{DQN!r},'-B',str(prep/'prepare_jan29_mothers.py')],
  env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
  start_new_session=True,close_fds=True)
stat=(pathlib.Path('/proc')/str(process.pid)/'stat').read_text()
ticks=int(stat[stat.rfind(')')+2:].split()[19])
receipt={{'status':'preparing_jan29_mother_inputs_only',
 'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'target':'u2024110224@10.160.17.102:20424','pid':process.pid,'start_ticks':ticks,
 'source_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
 'official_archive_sha256':download['sha256'],
 'status_path':str(data/'jan29_mother/preparation_status.json'),
 'output_record':str(data/'jan29_mother/record.json'),'log':str(log)}}
launch_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
print(json.dumps(receipt,ensure_ascii=False))
'''
    transport = connect(20424)
    try:
        report = call(transport, code)
    finally:
        transport.close()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "jan29_mother_launch.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"receipt": str(path), **report}


def status() -> dict:
    code = f'''
import datetime,json,pathlib
data=pathlib.Path({DATA!r});launch=data/'ops/jan29_mother_launch.json'
record=json.loads(launch.read_text()) if launch.is_file() else None
birth=None
if record:
 proc=pathlib.Path('/proc')/str(record['pid'])
 if proc.is_dir():
  stat=(proc/'stat').read_text();birth=int(stat[stat.rfind(')')+2:].split()[19])
state=data/'jan29_mother/preparation_status.json';log=data/'ops/jan29_mother.log'
print(json.dumps({{'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'launch':record,'process_birth_matches':birth==record['start_ticks'] if record else None,
 'status':json.loads(state.read_text()) if state.is_file() else None,
 'record_exists':(data/'jan29_mother/record.json').is_file(),
 'log_tail':log.read_text(errors='replace')[-700:] if log.is_file() else None}},ensure_ascii=False))
'''
    transport = connect(20424)
    try:
        report = call(transport, code)
    finally:
        transport.close()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "jan29_mother_status.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"receipt": str(path), **report}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "start", "status"))
    action = parser.parse_args().action
    print(json.dumps({"freeze": freeze, "start": start, "status": status}[action](), ensure_ascii=False))


if __name__ == "__main__":
    main()
