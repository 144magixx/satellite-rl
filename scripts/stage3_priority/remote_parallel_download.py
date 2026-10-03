"""Bounded parallel Range transfer of the pinned Jan27 official release."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlparse

sys.path.insert(0, "D:/paper/outputs/verification/g11_remote_launch_20261001")
from remote_transport import connect, execute, paramiko  # noqa: E402


BASE = "/home/u2024110224/third_stage_satellite_rl/s3_si_priority_20261003"
DQN = "/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python"
PIN = Path("D:/paper/outputs/experiments/stage3_si_priority_20261003/data/source_preflight.json")
OUTPUT = Path("D:/paper/outputs/experiments/stage3_si_priority_20261003/preflight")
HELPER = Path(__file__).with_name("parallel_range_helper.py")


def remote(code: str, timeout: int = 150) -> dict:
    payload = base64.b64encode(code.encode()).decode()
    transport = connect(20424)
    try:
        result = execute(transport, "python3 -B -c 'exec(__import__(\"base64\").b64decode(\"" + payload + "\"))'", timeout=timeout)
    finally:
        transport.close()
    if result["exit_code"]:
        raise RuntimeError(result["stderr"][-1200:])
    return json.loads(result["stdout"])


def probe() -> dict:
    pin = json.loads(PIN.read_text(encoding="utf-8"))
    head = subprocess.run(["curl.exe", "-I", "-L", "--silent", "--show-error",
                           "--connect-timeout", "20", "--max-time", "45",
                           "-o", "NUL", "-w", "%{url_effective}\n", pin["archive_url"]],
                          capture_output=True, text=True, timeout=50)
    if head.returncode:
        raise RuntimeError("Official CDN redirect unavailable")
    cdn = head.stdout.splitlines()[0]
    if urlparse(cdn).hostname != "release-assets.githubusercontent.com":
        raise RuntimeError("Official redirect host differs")
    code = f'''
import datetime,json,pathlib,subprocess,tempfile
url={cdn!r};size={pin['expected_bytes']}
with tempfile.TemporaryDirectory() as folder:
 out=pathlib.Path(folder)/'range.bin';headers=pathlib.Path(folder)/'headers.txt'
 p=subprocess.run(['curl','--location','--fail','--silent','--show-error','--connect-timeout','20',
  '--max-time','90','--range','0-65535','--dump-header',str(headers),'--output',str(out),url],
  capture_output=True,text=True,timeout=95)
 data=out.read_bytes() if out.is_file() else b''
 lines=headers.read_text(errors='replace').splitlines() if headers.is_file() else []
 content_range=next((line.split(':',1)[1].strip() for line in reversed(lines)
   if line.lower().startswith('content-range:')),None)
if p.returncode or content_range!=f'bytes 0-65535/{{size}}' or len(data)!=65536:
 raise RuntimeError('Official release does not honor exact Range request')
print(json.dumps({{'status':'official_range_supported','checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'bytes':len(data),'content_range':content_range}}))
'''
    result = remote(code)
    path = OUTPUT / "official_range_probe.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"receipt": str(path), **result}


def start() -> dict:
    pin = json.loads(PIN.read_text(encoding="utf-8"))
    checked = json.loads((OUTPUT / "official_range_probe.json").read_text(encoding="utf-8"))
    if checked["status"] != "official_range_supported":
        raise RuntimeError("Official Range probe is required")
    head = subprocess.run(["curl.exe", "-I", "-L", "--silent", "--show-error",
                           "--connect-timeout", "20", "--max-time", "45",
                           "-o", "NUL", "-w", "%{url_effective}\n", pin["archive_url"]],
                          capture_output=True, text=True, timeout=50)
    if head.returncode:
        raise RuntimeError("Official CDN redirect unavailable")
    cdn = head.stdout.splitlines()[0]
    if urlparse(cdn).hostname != "release-assets.githubusercontent.com":
        raise RuntimeError("Official redirect host differs")
    remote_helper = BASE + "/shared_data/ops/parallel_range_helper.py"
    transport = connect(20424)
    try:
        with paramiko.SFTPClient.from_transport(transport) as sftp:
            sftp.put(str(HELPER), remote_helper)
    finally:
        transport.close()
    code = f'''
import datetime,json,os,pathlib,signal,subprocess,time
base=pathlib.Path({BASE!r});raw=base/'shared_data/raw';ops=base/'shared_data/ops'
pin={pin!r};status_path=raw/'download_status.json';status=json.loads(status_path.read_text())
if (status.get('url')!=pin['archive_url'] or status.get('expected_bytes')!=pin['expected_bytes']
 or status.get('asset_id')!=pin['asset_id'] or status.get('status') not in ('running','failed')):
 raise RuntimeError('Pinned official download status differs')
old=pathlib.Path('/proc/1187');old_status='absent'
if old.is_dir():
 cmd=(old/'cmdline').read_bytes().decode('utf-8','replace')
 if 'cdn_transfer.py' not in cmd or str(base) not in cmd:
  raise RuntimeError('Old CDN PID belongs to another process')
 os.kill(1187,signal.SIGTERM);old_status='terminated_verified_cdn_transfer'
 time.sleep(1)
prior=pathlib.Path('/proc/2441')
if prior.is_dir():
 cmd=(prior/'cmdline').read_bytes().decode('utf-8','replace')
 stat=(prior/'stat').read_text();ticks=int(stat[stat.rfind(')')+2:].split()[19])
 if 'parallel_range_helper.py' not in cmd or ticks!=762822768:
  raise RuntimeError('Prior parallel downloader PID identity differs')
 if os.getpgid(2441)!=2441:raise RuntimeError('Prior downloader is not its isolated process group')
 os.killpg(2441,signal.SIGTERM)
 time.sleep(1)
if (raw/pin['archive_name']).is_file():
 raise RuntimeError('Final official archive already exists')
env=os.environ.copy();env.update(S3_BASE=str(base),S3_URL={cdn!r},S3_OFFICIAL_URL=pin['archive_url'],
 S3_ASSET=pin['archive_name'],S3_EXPECTED_BYTES=str(pin['expected_bytes']))
log=ops/'jan27_parallel_range.log'
with log.open('ab') as output:
 p=subprocess.Popen([{DQN!r},{remote_helper!r}],env=env,
  stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
  start_new_session=True,close_fds=True)
stat=(pathlib.Path('/proc')/str(p.pid)/'stat').read_text();ticks=int(stat[stat.rfind(')')+2:].split()[19])
print(json.dumps({{'status':'parallel_official_range_started','pid':p.pid,'start_ticks':ticks,
 'old_download':old_status,'helper':{remote_helper!r},'log':str(log),
 'status_path':str(raw/'parallel_status.json'),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}}))
'''
    result = remote(code)
    path = OUTPUT / "parallel_range_launch.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"receipt": str(path), **result}


def status() -> dict:
    code = f'''
import json,pathlib
base=pathlib.Path({BASE!r});raw=base/'shared_data/raw';p=raw/'parallel_status.json'
state=json.loads(p.read_text()) if p.is_file() else None
parts=raw/'parallel_parts'
files={{x.name:x.stat().st_size for x in parts.glob('chunk_*.part')}} if parts.is_dir() else {{}}
responses={{x.name:x.stat().st_size for x in parts.glob('chunk_*.response')}} if parts.is_dir() else {{}}
log=base/'shared_data/ops/jan27_parallel_range.log'
print(json.dumps({{'status':state,'chunks':files,'responses':responses,
 'total_partial_bytes':sum(files.values())+sum(responses.values()),
 'log_tail':log.read_text(errors='replace')[-800:] if log.is_file() else ''}}))
'''
    return remote(code)


def main() -> None:
    action = sys.argv[1] if len(sys.argv) > 1 else "probe"
    if action not in {"probe", "start", "status"}:
        raise ValueError("Choose probe, start or status")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"probe": probe, "start": start, "status": status}[action](), ensure_ascii=False))


if __name__ == "__main__":
    main()
