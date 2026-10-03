"""Plan or start the isolated S2-P17 workbench on 20424:8781."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ops_common import (ARCHIVE, BASE, DQN, LOCAL, REPO, WORKBENCH_PORT,
                        require_commit, remote_python, utc_now, write_json)


def run(commit: str, mode: str, execute: bool) -> dict:
    receipt_path = LOCAL / f"deployment_{commit[:12]}.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("scientific_commit") != commit or receipt.get("status") not in (
            "deployed_clean_no_workbench_or_jobs", "already_deployed_clean"):
        raise RuntimeError("干净正式部署回执缺失")
    code = f'''
import datetime,json,pathlib,socket,subprocess,os
repo=pathlib.Path({REPO!r});archive=pathlib.Path({ARCHIVE!r});commit={commit!r};mode={mode!r}
head=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
dirty=subprocess.run(['git','-C',str(repo),'status','--porcelain'],capture_output=True,text=True,check=True).stdout.strip()
if head!=commit or dirty:raise RuntimeError('Frozen repo differs')
script=repo/'scripts/variable_beams/remote_server.py'
if not script.is_file():raise RuntimeError('Committed server controller missing')
with socket.socket() as sock:
 sock.settimeout(2);busy=sock.connect_ex(('127.0.0.1',{WORKBENCH_PORT}))==0
if {execute!r}:
 env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
 p=subprocess.run([{DQN!r},'-B',str(script),'--commit',commit,'--mode',mode],
  capture_output=True,text=True,env=env,timeout=180)
 if p.returncode:raise RuntimeError(p.stderr[-1200:])
 result=json.loads(p.stdout.splitlines()[-1])
else:
 result={{'status':'workbench_launch_plan_only','checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'commit':commit,'mode':mode,'max_workers':2 if mode=='training' else 1,
  'workbench_port':{WORKBENCH_PORT},'port_busy':busy,'repo':str(repo),'archive':str(archive)}}
print(json.dumps(result,ensure_ascii=False))
'''
    result = remote_python(code, timeout=240 if execute else 90)
    output = {"status": result["status"], "checked_utc": utc_now(),
              "target": "u2024110224@10.160.17.102:20424",
              "commit": commit, "mode": mode, "remote": result}
    path = LOCAL / f"workbench_{mode}_{commit[:12]}.json"
    write_json(path, output)
    return {"receipt": str(path), **output}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--mode", choices=("training", "evaluation"), required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(require_commit(args.commit), args.mode, args.execute), ensure_ascii=False))


if __name__ == "__main__":
    main()
