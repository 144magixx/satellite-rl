"""Initialize isolated S3-P02 storage and manage the shared Jan26 data download.

Requires BUPT_PASSWORD in the environment. Uses the existing pinned SSH host keys.
The supported actions never submit or resume training/evaluation jobs.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[2]
CONNECTION = Path("D:/paper/outputs/verification/g11_remote_launch_20261001")
sys.path.insert(0, str(CONNECTION))
from remote_transport import connect, execute, paramiko  # noqa: E402


BASE = "/home/u2024110224/third_stage_satellite_rl/s3_all_routes_20261003"
DQN = "/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python"
SCRIPT = Path(__file__).with_name("remote_data_download.py")
REMOTE_SCRIPT = BASE + "/shared_data/ops/remote_data_download.py"
DATE = "2024-01-26"
ASSET_ID = 148112104
EXPECTED_BYTES = 1679800320
URL = "https://github.com/adsblol/globe_history_2024/releases/download/v2024.01.26-planes-readsb-staging-0/v2024.01.26-planes-readsb-staging-0.tar"


def remote_python(transport, code: str) -> dict:
    encoded = base64.b64encode(code.encode()).decode()
    command = "python3 -c 'exec(__import__(\"base64\").b64decode(\"" + encoded + "\"))'"
    response = execute(transport, command, timeout=120)
    if response["exit_code"]:
        raise RuntimeError("Remote action failed: " + response["stderr"][-500:])
    return json.loads(response["stdout"])


def initialize(port: int) -> dict:
    code = f'''
import json, pathlib, subprocess, sys
base=pathlib.Path({BASE!r})
root=base/{str(port)!r}
for path in (root/'repo',root/'archive',root/'locks',root/'ops',
             base/'shared_data/raw',base/'shared_data/prepared',base/'shared_data/ops'):
    path.mkdir(parents=True,exist_ok=True)
receive=root/'receive.git'
if not (receive/'HEAD').is_file():
    bare=subprocess.run(['git','init','--bare',str(receive)],capture_output=True,text=True,check=True)
p=subprocess.run([{DQN!r},'-c','import json,sys,scipy,torch,satellite_rl;print(json.dumps({{"python":sys.executable,"scipy":scipy.__version__,"torch":torch.__version__,"cuda":torch.cuda.is_available(),"package":satellite_rl.__file__}}))'],capture_output=True,text=True,timeout=60)
print(json.dumps({{'port':{port},'root':str(root),'shared':str(base/'shared_data'),
                  'receive_git':str(receive),'receive_ready':(receive/'HEAD').is_file(),
                  'dqn_exit':p.returncode,'dqn':json.loads(p.stdout.splitlines()[-1]) if p.returncode==0 else None,
                  'dqn_stderr':p.stderr[-400:]}}))
'''
    transport = connect(port)
    try:
        return remote_python(transport, code)
    finally:
        transport.close()


def start_download() -> dict:
    transport = connect(20424)
    try:
        with paramiko.SFTPClient.from_transport(transport) as sftp:
            sftp.put(str(SCRIPT), REMOTE_SCRIPT)
        code = f'''
import json,os,pathlib,subprocess
base=pathlib.Path({BASE!r})
status=base/'shared_data/raw/download_status.json'
if status.exists():
    data=json.loads(status.read_text())
    if data.get('status')=='completed':
        print(json.dumps({{'status':'already_completed','download':data}}));raise SystemExit
    if data.get('status')=='running':
        print(json.dumps({{'status':'already_running','download':data}}));raise SystemExit
log=base/'shared_data/ops/jan26_download.log'
with log.open('ab') as output:
    process=subprocess.Popen([{DQN!r},{REMOTE_SCRIPT!r},'--output-dir',str(base/'shared_data/raw'),
        '--date',{DATE!r},'--expected-bytes',{str(EXPECTED_BYTES)!r},'--asset-id',{str(ASSET_ID)!r},'--url',{URL!r}],
        stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
print(json.dumps({{'status':'started','pid':process.pid,'log':str(log),'script':{REMOTE_SCRIPT!r},'status_path':str(status)}}))
'''
        return remote_python(transport, code)
    finally:
        transport.close()


def status(port: int) -> dict:
    code = f'''
import json,pathlib
base=pathlib.Path({BASE!r})
path=base/'shared_data/raw/download_status.json'
data=json.loads(path.read_text()) if path.is_file() else None
print(json.dumps({{'port':{port},'status_path':str(path),'download':data,
                  'root_exists':(base/{str(port)!r}).is_dir(),
                  'shared_inode':(base/'shared_data').stat().st_ino if (base/'shared_data').exists() else None}}))
'''
    transport = connect(port)
    try:
        return remote_python(transport, code)
    finally:
        transport.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("init", "start-download", "status"))
    args = parser.parse_args()
    if args.action == "init":
        result = {str(port): initialize(port) for port in (20418, 20424)}
    elif args.action == "start-download":
        result = start_download()
    else:
        result = {str(port): status(port) for port in (20418, 20424)}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
