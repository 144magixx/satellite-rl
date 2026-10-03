"""Stage frozen S1-02 date contracts and manage isolated remote downloads.

Only 20418 is contacted, using remote_stage1's pinned SSH key and the existing
BUPT_PASSWORD environment variable. Downloads persist after this client exits.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import remote_stage1  # noqa: E402


LOCAL = Path("D:/paper/outputs/experiments/first_stage_s1_02_20261004")
REMOTE = "/home/u2024110224/first_stage_satellite_rl/s1_02/20418"
SOURCE_FILES = ("release_api.json", "PREFERRED_RELEASES.txt",
                "LICENSE-ODbL-official.txt", "selection_contract.json",
                "source_preflight.json")


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def connect():
    from types import SimpleNamespace

    return remote_stage1.connect(SimpleNamespace(
        host="10.160.17.102", port=20418, user="u2024110224",
        host_key="a6ym3EfC35pPdkqWO8lg+4As2nLDI1eKu8cur4P7WDc"))


def run_code(client, code: str) -> dict:
    encoded = base64.b64encode(code.encode()).decode()
    command = "python3 -B -c 'exec(__import__(\"base64\").b64decode(\"" + encoded + "\"))'"
    return json.loads(remote_stage1.execute(client, command))


def upload_once(sftp, local: Path, remote: str) -> dict:
    payload = local.read_bytes()
    try:
        with sftp.open(remote, "rb") as stream:
            previous = stream.read()
    except FileNotFoundError:
        previous = None
    if previous is not None and previous != payload:
        raise RuntimeError("Existing remote frozen source differs: " + remote)
    if previous is None:
        sftp.put(str(local), remote)
    with sftp.open(remote, "rb") as stream:
        if digest(stream.read()) != digest(payload):
            raise RuntimeError("Remote upload SHA mismatch: " + remote)
    return {"path": remote, "bytes": len(payload), "sha256": digest(payload),
            "action": "verified_existing" if previous is not None else "uploaded_and_verified"}


def stage(client, stage_name: str) -> dict:
    local_data = LOCAL / stage_name / "data"
    pin = json.loads((local_data / "source_preflight.json").read_text(encoding="utf-8"))
    if pin["group"] != "S1-02" or pin["stage"] != stage_name:
        raise RuntimeError("Local frozen source stage mismatch")
    remote_data = f"{REMOTE}/data/{stage_name}"
    remote_stage1.execute(client, f"mkdir -p {remote_data}/raw {REMOTE}/ops")
    # The scene builder reads the stable top-level status path while the
    # downloader writes atomically inside raw/. Keep it a relative symlink.
    remote_stage1.execute(client,
                          f"test -L {remote_data}/download_status.json || "
                          f"ln -s raw/download_status.json {remote_data}/download_status.json")
    uploaded = []
    with client.open_sftp() as sftp:
        for name in SOURCE_FILES:
            uploaded.append(upload_once(sftp, local_data / name, remote_data + "/" + name))
        helper = Path(__file__).with_name("download_source.py")
        uploaded.append(upload_once(sftp, helper, REMOTE + "/ops/download_source.py"))
        for name in ("prepare_aviation_scenes.py", "prepare_snapshots_reference.py"):
            uploaded.append(upload_once(sftp, Path(__file__).with_name(name), REMOTE + "/ops/" + name))
    return {"stage": stage_name, "remote_data": remote_data, "files": uploaded}


def launch(client, stage_name: str) -> dict:
    data = f"{REMOTE}/data/{stage_name}"
    helper = REMOTE + "/ops/download_source.py"
    code = f'''
import datetime,json,os,pathlib,subprocess
data=pathlib.Path({data!r});helper=pathlib.Path({helper!r})
pin=json.loads((data/'source_preflight.json').read_text())
if pin['group']!='S1-02' or pin['stage']!={stage_name!r} or not helper.is_file():
 raise RuntimeError('Pinned source or helper missing')
receipt=data/'raw'/'launch_receipt.json'
if receipt.exists():
 old=json.loads(receipt.read_text())
 status=data/'raw'/'download_status.json'
 if status.exists() and json.loads(status.read_text()).get('status')=='completed':
  print(json.dumps({{'status':'already_completed','previous':old}}));raise SystemExit
 pid=old.get('pid');proc=pathlib.Path('/proc')/str(pid)
 if proc.exists():
  stat=(proc/'stat').read_text();ticks=int(stat[stat.rfind(')')+2:].split()[19])
  command=(proc/'cmdline').read_bytes().decode('utf-8','replace')
  if ticks==old.get('start_ticks') and str(helper) in command and str(data) in command:
   print(json.dumps({{'status':'already_running','previous':old}}));raise SystemExit
log=data/'raw'/'download.log'
with log.open('ab') as stream:
 process=subprocess.Popen(['python3','-B',str(helper),'--data-dir',str(data)],
  stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
  start_new_session=True,close_fds=True)
stat=(pathlib.Path('/proc')/str(process.pid)/'stat').read_text()
ticks=int(stat[stat.rfind(')')+2:].split()[19])
value={{'status':'launched','stage':{stage_name!r},'pid':process.pid,
 'start_ticks':ticks,'helper':str(helper),'data_dir':str(data),'log':str(log),
 'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}}
temporary=receipt.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\\n');temporary.replace(receipt)
print(json.dumps(value))
'''
    return run_code(client, code)


def status(client, stage_name: str) -> dict:
    data = f"{REMOTE}/data/{stage_name}"
    code = f'''
import json,pathlib,datetime
data=pathlib.Path({data!r});raw=data/'raw'
def read(name):
 p=raw/name
 return json.loads(p.read_text()) if p.is_file() else None
launch=read('launch_receipt.json');download=read('download_status.json')
pid=launch.get('pid') if launch else None;live=False
if pid and (pathlib.Path('/proc')/str(pid)).exists():
 p=pathlib.Path('/proc')/str(pid);stat=(p/'stat').read_text();ticks=int(stat[stat.rfind(')')+2:].split()[19])
 command=(p/'cmdline').read_bytes().decode('utf-8','replace')
 live=(ticks==launch['start_ticks'] and str(data) in command and 'download_source.py' in command)
pin=json.loads((data/'source_preflight.json').read_text()) if (data/'source_preflight.json').is_file() else None
parts={{p.name:p.stat().st_size for p in raw.glob('*.part')}} if raw.is_dir() else {{}}
print(json.dumps({{'stage':{stage_name!r},'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'pin':pin,'launch':launch,'live':live,'download':download,'partial_files':parts}}))
'''
    return run_code(client, code)


def build(client, stage_name: str) -> dict:
    data = f"{REMOTE}/data/{stage_name}"
    helper = REMOTE + "/ops/prepare_aviation_scenes.py"
    reference = REMOTE + "/ops/prepare_snapshots_reference.py"
    frozen = "/home/u2024110224/first_stage_satellite_rl/outputs/workbench/source_snapshots/b3a11c51e9da6612a43cb005a79c9e89a37e448d"
    train_record = "/home/u2024110224/first_stage_satellite_rl/data_inputs/train/record.json"
    dqn = "/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python"
    code = f'''
import datetime,json,os,pathlib,subprocess
data=pathlib.Path({data!r});helper=pathlib.Path({helper!r});reference=pathlib.Path({reference!r})
pin=json.loads((data/'source_preflight.json').read_text())
download=json.loads((data/'raw'/'download_status.json').read_text())
if (pin['stage']!={stage_name!r} or download['status']!='completed'
 or download['bytes']!=pin['expected_bytes'] or not helper.is_file() or not reference.is_file()):
 raise RuntimeError('Pinned archive not complete or preparation helper absent')
receipt=data/'build_launch_receipt.json';state=data/'preparation_status.json'
if receipt.exists():
 old=json.loads(receipt.read_text())
 if state.exists() and json.loads(state.read_text()).get('status')=='completed_not_imported_or_evaluated':
  print(json.dumps({{'status':'already_completed','previous':old}}));raise SystemExit
 pid=old.get('pid');proc=pathlib.Path('/proc')/str(pid)
 if proc.exists():
  stat=(proc/'stat').read_text();ticks=int(stat[stat.rfind(')')+2:].split()[19])
  command=(proc/'cmdline').read_bytes().decode('utf-8','replace')
  if ticks==old.get('start_ticks') and str(helper) in command:
   print(json.dumps({{'status':'already_running','previous':old}}));raise SystemExit
env=os.environ.copy();env.update(STAGE1_DATA_ROOT=str(data),STAGE1_DATE=pin['date_utc'],
 STAGE1_SOURCE_ROOT={frozen!r},STAGE1_FROZEN_SRC={frozen + '/src'!r},
 STAGE1_TRAIN_RECORD={train_record!r})
log=data/'build.log'
with log.open('ab') as stream:
 process=subprocess.Popen([{dqn!r},'-B',str(helper)],env=env,
  stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
  start_new_session=True,close_fds=True)
stat=(pathlib.Path('/proc')/str(process.pid)/'stat').read_text()
ticks=int(stat[stat.rfind(')')+2:].split()[19])
value={{'status':'launched','stage':{stage_name!r},'pid':process.pid,
 'start_ticks':ticks,'helper':str(helper),'reference':str(reference),
 'data_dir':str(data),'log':str(log),
 'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}}
temporary=receipt.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\\n');temporary.replace(receipt)
print(json.dumps(value))
'''
    return run_code(client, code)


def build_status(client, stage_name: str) -> dict:
    data = f"{REMOTE}/data/{stage_name}"
    code = f'''
import json,pathlib,datetime
data=pathlib.Path({data!r})
def read(path):return json.loads(path.read_text()) if path.is_file() else None
launch=read(data/'build_launch_receipt.json');state=read(data/'preparation_status.json')
manifest=read(data/'independent_test_manifest.json');pid=launch.get('pid') if launch else None
live=False
if pid and (pathlib.Path('/proc')/str(pid)).exists():
 p=pathlib.Path('/proc')/str(pid);stat=(p/'stat').read_text();ticks=int(stat[stat.rfind(')')+2:].split()[19])
 command=(p/'cmdline').read_bytes().decode('utf-8','replace')
 live=(ticks==launch['start_ticks'] and 'prepare_aviation_scenes.py' in command)
log=data/'build.log'
print(json.dumps({{'stage':{stage_name!r},'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'launch':launch,'live':live,'state':state,'manifest':manifest,
 'log_tail':log.read_text(errors='replace')[-1200:] if log.is_file() else ''}}))
'''
    return run_code(client, code)


def save_receipt(stage_name: str, action: str, payload: dict) -> None:
    output = LOCAL / "preflight"
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"remote_{stage_name}_{action}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage", "launch", "status", "build", "build_status"))
    parser.add_argument("--stage", choices=("A", "B", "both"), default="both")
    args = parser.parse_args()
    stages = ("A", "B") if args.stage == "both" else (args.stage,)
    client = connect()
    try:
        for item in stages:
            actions = {"stage": stage, "launch": launch, "status": status,
                       "build": build, "build_status": build_status}
            result = {"checked_utc": datetime.now(timezone.utc).isoformat(),
                      "result": actions[args.action](client, item)}
            save_receipt(item, args.action, result)
            print(json.dumps(result, ensure_ascii=False))
    finally:
        client.close()


if __name__ == "__main__":
    main()
