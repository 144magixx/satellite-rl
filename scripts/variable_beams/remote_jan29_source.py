"""Pin and optionally download the official Jan29 readsb archive on 20424."""
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

LOCAL_OUTPUT = Path("D:/paper/outputs/experiments/variable_beams_20261004/preflight")
PREFERRED = Path("D:/paper/outputs/experiments/stage3_si_priority_20261003/data/PREFERRED_RELEASES.txt")
DOWNLOADER = Path("D:/paper/scripts/stage3_priority/remote_data_download.py")
BASE = "/home/u2024110224/second_stage_satellite_rl/variable_beams_20261004/20424/data"
DATE = "2024-01-29"
TAG = "v2024.01.29-planes-readsb-staging-0"
URL = f"https://github.com/adsblol/globe_history_2024/releases/download/{TAG}/{TAG}.tar"
API = f"https://api.github.com/repos/adsblol/globe_history_2024/releases/tags/{TAG}"
DQN = "/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python"


def call(code: str, timeout: int = 180) -> dict:
    encoded = base64.b64encode(code.encode()).decode()
    transport = connect(20424)
    try:
        result = execute(transport, "python3 -B -c 'exec(__import__(\"base64\").b64decode(\"" + encoded + "\"))'", timeout=timeout)
    finally:
        transport.close()
    if result["exit_code"]:
        raise RuntimeError(result["stderr"][-1200:])
    return json.loads(result["stdout"])


def preflight() -> dict:
    preferred_sha = hashlib.sha256(PREFERRED.read_bytes()).hexdigest()
    if URL not in PREFERRED.read_text(encoding="utf-8"):
        raise RuntimeError("Jan29 URL missing from previously pinned preferred release list")
    code = f'''
import datetime,json,pathlib,urllib.request
api={API!r};name={TAG!r}+'.tar';url={URL!r}
req=urllib.request.Request(api,headers={{'User-Agent':'S2-P17-data-preflight','Accept':'application/vnd.github+json'}})
with urllib.request.urlopen(req,timeout=40) as response:
 release=json.load(response)
assets=[x for x in release.get('assets',[]) if x.get('name')==name and x.get('browser_download_url')==url]
if len(assets)!=1:raise RuntimeError('Exactly one official Jan29 archive asset required')
asset=assets[0]
out={{'status':'official_source_metadata_verified_no_download',
 'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'target':'u2024110224@10.160.17.102:20424','release_tag':release.get('tag_name'),
 'release_url':release.get('html_url'),'asset_id':asset['id'],'asset_name':asset['name'],
 'asset_url':asset['browser_download_url'],'asset_size':asset['size'],
 'asset_digest':asset.get('digest'),'asset_created_at':asset.get('created_at'),
 'asset_updated_at':asset.get('updated_at')}}
print(json.dumps(out,ensure_ascii=False))
'''
    report = call(code)
    if report["release_tag"] != TAG or report["asset_size"] < 1_000_000:
        raise RuntimeError("Official Jan29 release identity or size invalid")
    report["preferred_release_list_sha256"] = preferred_sha
    LOCAL_OUTPUT.mkdir(parents=True, exist_ok=True)
    path = LOCAL_OUTPUT / "jan29_official_source.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"receipt": str(path), **report}


def start_download() -> dict:
    source = json.loads((LOCAL_OUTPUT / "jan29_official_source.json").read_text(encoding="utf-8"))
    if (source.get("status") != "official_source_metadata_verified_no_download"
            or source.get("asset_url") != URL or source.get("release_tag") != TAG):
        raise RuntimeError("Jan29 official source preflight missing or altered")
    script_remote = BASE + "/ops/remote_data_download.py"
    transport = connect(20424)
    try:
        init_code = f'''
import json,pathlib
base=pathlib.Path({BASE!r})
for part in ('raw','ops'): (base/part).mkdir(parents=True,exist_ok=True)
print(json.dumps({{'data_root':str(base),'ready':True}}))
'''
        encoded = base64.b64encode(init_code.encode()).decode()
        init = execute(transport, "python3 -B -c 'exec(__import__(\"base64\").b64decode(\"" + encoded + "\"))'", timeout=30)
        if init["exit_code"]:
            raise RuntimeError(init["stderr"][-500:])
        with paramiko.SFTPClient.from_transport(transport) as sftp:
            sftp.put(str(DOWNLOADER), script_remote)
    finally:
        transport.close()
    code = f'''
import datetime,json,os,pathlib,subprocess
base=pathlib.Path({BASE!r});raw=base/'raw';ops=base/'ops'
status=raw/'download_status.json';asset={source['asset_id']};size={source['asset_size']}
if status.exists():
 current=json.loads(status.read_text())
 if current.get('date_utc')!={DATE!r} or current.get('asset_id')!=asset or current.get('expected_bytes')!=size:
  raise RuntimeError('Existing raw download belongs to another asset')
 if current.get('status') in ('running','completed'):
  print(json.dumps({{'status':'already_'+current['status'],'download':current}}));raise SystemExit
log=ops/'jan29_download.log'
with log.open('ab') as output:
 process=subprocess.Popen([{DQN!r},'-B',{script_remote!r},'--output-dir',str(raw),
  '--date',{DATE!r},'--expected-bytes',str(size),'--asset-id',str(asset),'--url',{URL!r}],
  stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
stat=(pathlib.Path('/proc')/str(process.pid)/'stat').read_text()
ticks=int(stat[stat.rfind(')')+2:].split()[19])
receipt={{'status':'download_started','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'target':'u2024110224@10.160.17.102:20424','pid':process.pid,'start_ticks':ticks,
 'asset_id':asset,'expected_bytes':size,'status_path':str(status),'log':str(log)}}
(ops/'jan29_download_launch.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
print(json.dumps(receipt,ensure_ascii=False))
'''
    receipt = call(code)
    path = LOCAL_OUTPUT / "jan29_download_launch.json"
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"receipt": str(path), **receipt}


def status() -> dict:
    code = f'''
import datetime,json,pathlib
base=pathlib.Path({BASE!r});raw=base/'raw';ops=base/'ops'
path=raw/'download_status.json'
data=json.loads(path.read_text()) if path.is_file() else None
launch=ops/'jan29_download_launch.json'
launched=json.loads(launch.read_text()) if launch.is_file() else None
birth=None
if launched:
 proc=pathlib.Path('/proc')/str(launched['pid'])
 if proc.is_dir():
  stat=(proc/'stat').read_text();birth=int(stat[stat.rfind(')')+2:].split()[19])
print(json.dumps({{'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'download':data,'process_birth_matches':birth==launched['start_ticks'] if launched else None,
 'partial_bytes':(raw/({TAG!r}+'.tar.part')).stat().st_size if (raw/({TAG!r}+'.tar.part')).is_file() else None,
 'log_tail':(ops/'jan29_download.log').read_text(errors='replace')[-500:] if (ops/'jan29_download.log').is_file() else None}},ensure_ascii=False))
'''
    report = call(code)
    path = LOCAL_OUTPUT / "jan29_download_status.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"receipt": str(path), **report}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "start-download", "status"))
    action = parser.parse_args().action
    print(json.dumps({"preflight": preflight, "start-download": start_download,
                      "status": status}[action](), ensure_ascii=False))


if __name__ == "__main__":
    main()
