"""Stage pinned Jan31 source material and launch durable acquisition on 20418."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from remote import LOCAL, REMOTE_BASE, DQN, connect, run_code, upload_once, save, now


ROOT = REMOTE_BASE + "/20418"
DATA = ROOT + "/data/test"


def stage(client):
    run_code(client, f"""
import pathlib,json
for path in ({DATA!r}, {ROOT + '/ops'!r}, {DATA + '/raw'!r}): pathlib.Path(path).mkdir(parents=True,exist_ok=True)
link=pathlib.Path({DATA!r})/'download_status.json'
if not link.exists() and not link.is_symlink(): link.symlink_to('raw/download_status.json')
print(json.dumps({{'status':'directories_ready'}}))
""")
    files = []
    with client.open_sftp() as sftp:
        for name in ("release_api.json", "PREFERRED_RELEASES.txt", "LICENSE-ODbL-official.txt",
                     "selection_contract.json", "source_preflight.json"):
            files.append(upload_once(sftp, LOCAL / "test/data" / name, DATA + "/" + name))
        for name in ("download_source.py", "prepare_aviation_scenes.py", "prepare_snapshots_reference.py"):
            files.append(upload_once(sftp, Path(__file__).with_name(name), ROOT + "/ops/" + name))
    return files


def launch(client, mode):
    helper = ROOT + "/ops/" + ("download_source.py" if mode == "download" else "prepare_aviation_scenes.py")
    code = f"""
import os,pathlib,json,subprocess,datetime
data=pathlib.Path({DATA!r});helper=pathlib.Path({helper!r})
receipt=data/{(mode + '_launch.json')!r}
if receipt.exists():
 old=json.loads(receipt.read_text());proc=pathlib.Path('/proc')/str(old['pid'])
 if proc.exists():
  raw=(proc/'stat').read_text();ticks=int(raw[raw.rfind(')')+2:].split()[19])
  cmd=(proc/'cmdline').read_bytes().decode().replace('\\x00',' ')
  if ticks==old['start_ticks'] and str(helper) in cmd:
   print(json.dumps({{'status':'already_running','receipt':old}}));raise SystemExit
status=data/({'raw/download_status.json' if mode == 'download' else 'preparation_status.json'!r})
if status.exists() and json.loads(status.read_text()).get('status','').startswith('completed'):
 print(json.dumps({{'status':'already_completed','record':json.loads(status.read_text())}}));raise SystemExit
env=os.environ.copy();env.update(STAGE1_DATA_ROOT=str(data),STAGE1_DATE='2024-01-31')
argv=[{DQN!r},'-B',str(helper)]
if {mode!r}=='download':argv+=['--data-dir',str(data)]
log=data/({mode!r}+'.log')
with log.open('ab') as stream:
 process=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
  env=env,start_new_session=True,close_fds=True)
proc=pathlib.Path('/proc')/str(process.pid);raw=(proc/'stat').read_text()
value={{'status':'launched','mode':{mode!r},'pid':process.pid,'start_ticks':int(raw[raw.rfind(')')+2:].split()[19]),
 'helper':str(helper),'data':str(data),'log':str(log),'python':{DQN!r},
 'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}}
receipt.write_text(json.dumps(value,indent=2)+'\\n');print(json.dumps(value))
"""
    return run_code(client, code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("download", "prepare"), required=True)
    args = parser.parse_args()
    client = connect(20418)
    try:
        files = stage(client)
        outcome = launch(client, args.mode)
    finally:
        client.close()
    receipt = {"recorded_utc": now(), "files": files, "outcome": outcome,
               "scope": "Raw source acquisition/cleaning, not PPO training or formal method evaluation"}
    save(LOCAL / "preflight" / ("source_" + args.mode + "_launch.json"), receipt)
    print(json.dumps(outcome))


if __name__ == "__main__":
    main()
