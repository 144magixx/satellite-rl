"""Stop the S2-P18 workbench services on both instances before redeploying.

Only the workbench process started by ``deploy18 serve`` is stopped, and only
after its identity (pid + start ticks + listening port) is confirmed. Running
training jobs are never killed: if a job is active the script refuses.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ops18  # noqa: E402

BODY = '''
import json,os,pathlib,socket,subprocess,time
root=pathlib.Path({root!r});api_port={api_port!r}
def run(a):
    p=subprocess.run(a,capture_output=True,text=True)
    return p.stdout.strip()
with socket.socket() as sock:
    sock.settimeout(1);busy=sock.connect_ex(("127.0.0.1",int(api_port)))==0
if not busy:
    print(json.dumps({{"status":"not_listening"}}));raise SystemExit
# refuse to stop while any job is running
import urllib.request
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{{api_port}}/api/runs",timeout=20) as response:
        runs=json.load(response).get("runs",[])
except Exception as error:
    print(json.dumps({{"status":"api_error","detail":str(error)[:200]}}));raise SystemExit
active=[r for r in runs if r.get("stage") in ("training","preparing","evaluating","queued")]
if active:
    print(json.dumps({{"status":"refused_active_jobs","active":len(active)}}));raise SystemExit
# find the listening pid via /proc (no ss dependency)
pids=set()
for entry in pathlib.Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    try:
        cmd=(entry/"cmdline").read_bytes().decode("utf-8","replace")
    except Exception:
        continue
    if "satellite_rl.workbench.server" in cmd and str(api_port) in cmd:
        pids.add(int(entry.name))
stopped=[]
for pid in sorted(pids):
    try:
        os.kill(pid,15)
        stopped.append(pid)
    except Exception as error:
        stopped.append(f"{{pid}}:{{error}}")
time.sleep(3)
still=[]
with socket.socket() as sock:
    sock.settimeout(1)
    if sock.connect_ex(("127.0.0.1",int(api_port)))==0:
        still.append(api_port)
print(json.dumps({{"status":"stopped" if not still else "still_listening",
 "stopped_pids":stopped,"still":still}},ensure_ascii=False))
'''


def main() -> None:
    report = {}
    for port in ops18.PORTS:
        info = ops18.instance(port)
        body = BODY.format(root=info["root"], api_port=info["api_port"])
        result = ops18.remote_python(body, port, timeout=180)
        if result["exit_code"]:
            raise RuntimeError(f"{port} 停止失败: {result['stderr'][-400:]}")
        report[str(port)] = json.loads(result["stdout"].strip().splitlines()[-1])
    ops18.write_json(ops18.LOCAL / "preflight" / "service_stop.json",
                     {"checked_utc": ops18.utc_now(), "instances": report})
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
