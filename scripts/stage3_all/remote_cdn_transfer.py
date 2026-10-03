"""Route the pinned official Jan26 archive through a temporary GitHub CDN URL."""
from __future__ import annotations

import base64
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlparse


sys.path.insert(0, "D:/paper/outputs/verification/g11_remote_launch_20261001")
from remote_transport import connect, execute  # noqa: E402


BASE = "/home/u2024110224/third_stage_satellite_rl/s3_all_routes_20261003"
PIN = Path("D:/paper/outputs/experiments/stage3_all_routes_20261003/data/source_preflight.json")
OPS = Path("D:/paper/outputs/experiments/stage3_all_routes_20261003/preflight")
OLD_PID = 44257

HELPER = '''from __future__ import annotations
import hashlib,json,os,pathlib,subprocess
from datetime import datetime,timezone
base=pathlib.Path(os.environ['S3_BASE'])
raw=base/'shared_data/raw'
status=raw/'download_status.json'
target=raw/os.environ['S3_ASSET_NAME']
partial=target.with_suffix('.tar.cdn.part')
length=int(os.environ['S3_ASSET_BYTES'])
p=subprocess.run(['curl','--location','--fail','--silent','--show-error',
 '--connect-timeout','20','--max-time','3600','--continue-at','-',
 '--output',str(partial),os.environ['S3_CDN_URL']],capture_output=True,text=True,timeout=3650)
data=json.loads(status.read_text())
if p.returncode or not partial.exists() or partial.stat().st_size!=length:
 data.update(status='failed',error='CDN transfer exit '+str(p.returncode)+': '+p.stderr[-300:],
             partial_bytes=partial.stat().st_size if partial.exists() else 0)
else:
 partial.replace(target)
 h=hashlib.sha256()
 with target.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 data.update(status='completed',completed_utc=datetime.now(timezone.utc).isoformat(),
             bytes=target.stat().st_size,sha256=h.hexdigest(),transfer='official_release_cdn_redirect')
status.write_text(json.dumps(data,ensure_ascii=False,indent=2))
'''


def main() -> None:
    pin = json.loads(PIN.read_text(encoding="utf-8"))
    for attempt in range(5):
        probe = subprocess.run(["curl.exe", "-I", "-L", "--silent", "--show-error",
                                "--connect-timeout", "20", "--max-time", "45",
                                "-o", "NUL", "-w", "%{url_effective}\n",
                                pin["archive_url"]], capture_output=True, text=True, timeout=50)
        if probe.returncode == 0:
            break
        time.sleep(attempt + 1)
    if probe.returncode:
        raise RuntimeError("Official redirect HEAD failed: " + probe.stderr[-300:])
    cdn = probe.stdout.splitlines()[0]
    length = pin["expected_bytes"]
    if urlparse(cdn).hostname != "release-assets.githubusercontent.com" or length != pin["expected_bytes"]:
        raise RuntimeError("Official redirect host or byte count differs")
    script = '''import datetime,json,os,pathlib,signal,subprocess,time,urllib.request
base=pathlib.Path(__BASE__)
raw=base/'shared_data/raw';ops=base/'shared_data/ops'
status=raw/'download_status.json'
old=json.loads(status.read_text())
if old.get('url')!=__OFFICIAL__ or old.get('expected_bytes')!=__LENGTH__:
 raise RuntimeError('Pinned old status differs')
pid=__OLD_PID__;proc=pathlib.Path('/proc')/str(pid)
if proc.exists():
 cmd=(proc/'cmdline').read_bytes().decode('utf-8','replace')
 if 'remote_data_download.py' not in cmd or str(base) not in cmd:
  raise RuntimeError('Old PID was reused by an unrelated process')
 os.kill(pid,signal.SIGTERM)
 for _ in range(30):
  if not proc.exists():break
  time.sleep(0.1)
target=raw/__ASSET__
if target.exists():raise RuntimeError('Final archive already exists')
request=urllib.request.Request(__CDN__,method='HEAD')
with urllib.request.urlopen(request,timeout=30) as response:
 if int(response.headers['Content-Length'])!=__LENGTH__:
  raise RuntimeError('Remote CDN content length differs')
helper=ops/'cdn_transfer.py'
helper.write_text(__HELPER__)
env=os.environ.copy()
env.update(S3_BASE=str(base),S3_ASSET_NAME=__ASSET__,S3_ASSET_BYTES=str(__LENGTH__),S3_CDN_URL=__CDN__)
log=ops/'jan26_cdn_transfer.log'
with log.open('ab') as output:
 p=subprocess.Popen(['/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python',str(helper)],
  env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
  start_new_session=True,close_fds=True)
print(json.dumps({'status':'started','pid':p.pid,'helper':str(helper),
 'status_file':str(status),'date_utc':'2024-01-26','expected_bytes':__LENGTH__,
 'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}))
'''
    for marker, value in {
        "__BASE__": repr(BASE), "__OFFICIAL__": repr(pin["archive_url"]),
        "__LENGTH__": str(length), "__OLD_PID__": str(OLD_PID),
        "__ASSET__": repr(pin["archive_name"]), "__CDN__": repr(cdn),
        "__HELPER__": repr(HELPER),
    }.items():
        script = script.replace(marker, value)
    payload = base64.b64encode(script.encode()).decode()
    transport = connect(20424)
    try:
        response = execute(transport, "python3 -c 'exec(__import__(\"base64\").b64decode(\"" + payload + "\"))'", timeout=90)
    finally:
        transport.close()
    if response["exit_code"]:
        raise RuntimeError(response["stderr"][-1000:])
    result = json.loads(response["stdout"])
    OPS.mkdir(parents=True, exist_ok=True)
    receipt = OPS / "cdn_transfer_launch.json"
    receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"receipt": str(receipt), **result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
