"""Freeze S2-P17 data source and run input generation on 20424 only."""
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
WORKTREE = HERE.parents[1]
OUTPUT = Path("D:/paper/outputs/experiments/variable_beams_20261004/preflight")
BASE = "/home/u2024110224/second_stage_satellite_rl/variable_beams_20261004/20424"
CORE_DEV = BASE + "/dev/20261003T185801Z"
DATA_SOURCE = BASE + "/dev/data_source_20261003T191000Z"
DQN = "/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python"
INPUT_ROOT = BASE + "/data/source_csv"
OUTPUT_ROOT = BASE + "/data/prepared"
OUTPUT_NAMES = {"train": "train", "validation": "validation_eligible12_v2", "test": "test"}
G11_RECORDS = {
    "train": BASE + "/data/prepared/train_mothers_replaced576/record.json",
    "validation": "/home/u2024110224/second_stage_satellite_rl/g11/20424/archive/outputs/workbench/scenes/scenes_20261001_161628_819160d7/record.json",
}
FILES = {
    "src/satellite_rl/workbench/aviation.py": WORKTREE / "src/satellite_rl/workbench/aviation.py",
    "src/satellite_rl/workbench/datasets.py": WORKTREE / "src/satellite_rl/workbench/datasets.py",
    "prepare_dataset.py": HERE / "prepare_dataset.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def call(transport, code: str, timeout: int = 120) -> dict:
    encoded = base64.b64encode(code.encode()).decode()
    result = execute(transport, "python3 -B -c 'exec(__import__(\"base64\").b64decode(\"" + encoded + "\"))'", timeout=timeout)
    if result["exit_code"]:
        raise RuntimeError(result["stderr"][-1400:])
    return json.loads(result["stdout"])


def freeze() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    local_sha = {name: sha256(path) for name, path in FILES.items()}
    transport = connect(20424)
    try:
        init = call(transport, f'''
import json,pathlib,shutil
core=pathlib.Path({CORE_DEV!r});target=pathlib.Path({DATA_SOURCE!r})
if not (core/'src/satellite_rl/__init__.py').is_file():raise RuntimeError('Core dev snapshot missing')
if target.exists():raise RuntimeError('Data source snapshot already exists; use a new name')
shutil.copytree(core/'src',target/'src')
print(json.dumps({{'status':'base_snapshot_copied','source':str(core),'target':str(target)}}))
''')
        with paramiko.SFTPClient.from_transport(transport) as sftp:
            for name, path in FILES.items():
                sftp.put(str(path), DATA_SOURCE + "/" + name)
        report = call(transport, f'''
import datetime,hashlib,json,pathlib,os
root=pathlib.Path({DATA_SOURCE!r});expected={local_sha!r}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as s:
  for chunk in iter(lambda:s.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
for name,digest in expected.items():
 if sha(root/name)!=digest:raise RuntimeError('Uploaded source SHA mismatch: '+name)
files={{str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()}}
manifest={{'status':'frozen_data_source_no_training','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'target':'u2024110224@10.160.17.102:20424','root':str(root),
 'base_commit':'70572f7','base_snapshot':{CORE_DEV!r},
 'base_snapshot_archive_sha256':'9d2f5a860594bda22dcb8017a34f51cd34f9fe75d67010d234231c56cc37ce79',
 'uncommitted_file_overrides':expected,'file_count':len(files),'files':files}}
(root/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
for p in root.rglob('*'):
 if p.is_file():p.chmod(0o444)
print(json.dumps({{'status':manifest['status'],'root':str(root),'file_count':len(files),
 'manifest_sha256':sha(root/'source_manifest.json'),'overrides':expected}},ensure_ascii=False))
''')
    finally:
        transport.close()
    path = OUTPUT / "data_source_freeze.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"receipt": str(path), **report}


def start(role: str) -> dict:
    if role not in ("train", "validation", "test"):
        raise ValueError("Unknown dataset role")
    source_record = G11_RECORDS.get(role)
    if source_record is None:
        source_record = BASE + "/data/jan29_mother/record.json"
    input_root = INPUT_ROOT if role != "test" else BASE + "/data/snapshots"
    output_name = OUTPUT_NAMES[role]
    code = f'''
import datetime,hashlib,json,os,pathlib,subprocess
base=pathlib.Path({BASE!r});source=pathlib.Path({DATA_SOURCE!r})
manifest=source/'source_manifest.json'
if not manifest.is_file():raise RuntimeError('Frozen data source missing')
frozen=json.loads(manifest.read_text())
for name,digest in frozen['files'].items():
 if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:
  raise RuntimeError('Frozen data source changed: '+name)
record=pathlib.Path({source_record!r})
if not record.is_file():raise RuntimeError('Mother record missing: '+str(record))
input_root=pathlib.Path({input_root!r})
output=pathlib.Path({OUTPUT_ROOT!r})/{output_name!r}
ops=base/'data/ops';ops.mkdir(parents=True,exist_ok=True)
launch_path=ops/('prepare_'+{role!r}+'_launch.json')
if launch_path.is_file():
 old=json.loads(launch_path.read_text())
 proc=pathlib.Path('/proc')/str(old['pid'])
 if proc.is_dir():
  stat=(proc/'stat').read_text();birth=int(stat[stat.rfind(')')+2:].split()[19])
  if birth==old['start_ticks']:
   print(json.dumps({{'status':'already_running','launch':old}}));raise SystemExit
 if (output/'record.json').is_file():
  print(json.dumps({{'status':'already_completed','launch':old}}));raise SystemExit
 resume_count=old.get('resume_count',0)+1
else:resume_count=0
env=os.environ.copy();env.update(PYTHONPATH=str(source/'src'),PYTHONDONTWRITEBYTECODE='1',
 OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
 SATELLITE_RL_OUTPUT_ROOT=str(base/'data/scratch'))
(base/'data/scratch').mkdir(parents=True,exist_ok=True)
log=ops/('prepare_'+{role!r}+'.log')
with log.open('ab') as handle:
 process=subprocess.Popen([{DQN!r},'-B',str(source/'prepare_dataset.py'),
  '--source-record',str(record),'--input-root',str(input_root),
  '--output-dir',str(output),'--role',{role!r}],env=env,
  stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
  start_new_session=True,close_fds=True)
stat=(pathlib.Path('/proc')/str(process.pid)/'stat').read_text()
ticks=int(stat[stat.rfind(')')+2:].split()[19])
receipt={{'status':'preparing_inputs_only','role':{role!r},
 'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'target':'u2024110224@10.160.17.102:20424','pid':process.pid,'start_ticks':ticks,
 'source_root':str(source),'source_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
 'mother_record':str(record),'mother_record_sha256':hashlib.sha256(record.read_bytes()).hexdigest(),
 'input_root':str(input_root),'output_dir':str(output),'log':str(log),
 'resume_count':resume_count}}
launch_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
print(json.dumps(receipt,ensure_ascii=False))
'''
    transport = connect(20424)
    try:
        report = call(transport, code)
    finally:
        transport.close()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"prepare_{role}_launch.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"receipt": str(path), **report}


def status(role: str) -> dict:
    output_name = OUTPUT_NAMES[role]
    code = f'''
import datetime,json,pathlib
base=pathlib.Path({BASE!r});ops=base/'data/ops';out=pathlib.Path({OUTPUT_ROOT!r})/{output_name!r}
launch=ops/('prepare_'+{role!r}+'_launch.json')
record=json.loads(launch.read_text()) if launch.is_file() else None
birth=None
if record:
 proc=pathlib.Path('/proc')/str(record['pid'])
 if proc.is_dir():
  stat=(proc/'stat').read_text();birth=int(stat[stat.rfind(')')+2:].split()[19])
state=out/'preparation_status.json'
print(json.dumps({{'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'role':{role!r},'launch':record,'process_birth_matches':birth==record['start_ticks'] if record else None,
 'status':json.loads(state.read_text()) if state.is_file() else None,
 'record_exists':(out/'record.json').is_file(),
 'log_tail':(ops/('prepare_'+{role!r}+'.log')).read_text(errors='replace')[-700:] if (ops/('prepare_'+{role!r}+'.log')).is_file() else None}},ensure_ascii=False))
'''
    transport = connect(20424)
    try:
        report = call(transport, code)
    finally:
        transport.close()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"prepare_{role}_status.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"receipt": str(path), **report}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "start", "status"))
    parser.add_argument("--role", choices=("train", "validation", "test"))
    args = parser.parse_args()
    if args.action != "freeze" and args.role is None:
        parser.error("--role is required for start/status")
    report = freeze() if args.action == "freeze" else start(args.role) if args.action == "start" else status(args.role)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
