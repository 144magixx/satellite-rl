"""Stage pinned Jan25 conversion inputs and run scene preparation on shared /home.

This controls data preparation only. It never starts model evaluation workers.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import tarfile


HERE = Path(__file__).resolve().parent
LOCAL_DATA = Path("D:/paper/outputs/experiments/stage3_power_20261003/data")
LOCAL_PREFLIGHT = LOCAL_DATA.parent / "preflight"
G07_COMMIT = "01457f245bf0ab0e23873616e44975693014e3d0"
G07_SOURCE = Path("D:/paper/outputs/workbench/source_snapshots") / G07_COMMIT
G07_TEST = Path("D:/paper/outputs/workbench/scenes/scenes_20260928_142054_9bc86790/record.json")
BASE = "/home/u2024110224/third_stage_satellite_rl/s3_power_20261003"
SHARED = BASE + "/shared_data"
OPS = SHARED + "/ops"
DQN = "/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python"
sys.path.insert(0, "D:/paper/outputs/verification/g11_remote_launch_20261001")
from remote_transport import connect, execute, paramiko  # noqa: E402


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def remote_python(transport, code: str) -> dict:
    payload = base64.b64encode(code.encode()).decode()
    command = "python3 -c 'exec(__import__(\"base64\").b64decode(\"" + payload + "\"))'"
    result = execute(transport, command, timeout=180)
    if result["exit_code"]:
        raise RuntimeError(result["stderr"][-1000:])
    return json.loads(result["stdout"])


def package_frozen() -> Path:
    target = LOCAL_PREFLIGHT / f"g07_frozen_source_{G07_COMMIT}.tar.gz"
    if target.exists():
        return target
    LOCAL_PREFLIGHT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(target, "w:gz") as archive:
        manifest = G07_SOURCE / "source_manifest.json"
        archive.add(manifest, arcname=f"{G07_COMMIT}/source_manifest.json")
        for file in sorted((G07_SOURCE / "src").rglob("*")):
            if file.is_file() and "__pycache__" not in file.parts and file.suffix != ".pyc":
                archive.add(file, arcname=f"{G07_COMMIT}/src/{file.relative_to(G07_SOURCE / 'src').as_posix()}")
    return target


def stage() -> dict:
    frozen = package_frozen()
    local_files = {
        "source_preflight.json": LOCAL_DATA / "source_preflight.json",
        "selection_contract.json": LOCAL_DATA / "selection_contract.json",
        "release_api.json": LOCAL_DATA / "release_api.json",
        "PREFERRED_RELEASES.txt": LOCAL_DATA / "PREFERRED_RELEASES.txt",
        "LICENSE-ODbL-official.txt": LOCAL_DATA / "LICENSE-ODbL-official.txt",
    }
    ops_files = {
        "prepare_aviation_scenes.py": HERE / "prepare_aviation_scenes.py",
        "prepare_snapshots_reference.py": HERE / "prepare_snapshots_reference.py",
        "verify_aviation_scenes.py": HERE / "verify_aviation_scenes.py",
        "g07_test_record.json": G07_TEST,
        frozen.name: frozen,
    }
    transport = connect(20424)
    try:
        with paramiko.SFTPClient.from_transport(transport) as sftp:
            for name, path in local_files.items():
                sftp.put(str(path), SHARED + "/" + name)
            for name, path in ops_files.items():
                sftp.put(str(path), OPS + "/" + name)
        expected = {SHARED + "/" + name: sha(path) for name, path in local_files.items()}
        expected.update({OPS + "/" + name: sha(path) for name, path in ops_files.items()})
        code = f'''
import hashlib,json,pathlib,tarfile
expected={expected!r}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
for name,digest in expected.items():
 if sha(pathlib.Path(name))!=digest:raise RuntimeError('Uploaded SHA mismatch: '+name)
ops=pathlib.Path({OPS!r})
source=ops/{G07_COMMIT!r}
if not source.exists():
 with tarfile.open(ops/{frozen.name!r},'r:gz') as archive:
  for member in archive.getmembers():
   if not member.name.startswith({(G07_COMMIT + '/')!r}) or '..' in pathlib.PurePosixPath(member.name).parts:
    raise RuntimeError('Unexpected source tar path')
  archive.extractall(path=ops)
raw=json.loads((pathlib.Path({SHARED!r})/'raw/download_status.json').read_text())
pin=json.loads((pathlib.Path({SHARED!r})/'source_preflight.json').read_text())
if raw.get('status')!='completed' or raw.get('sha256') is None or raw.get('bytes')!=pin['expected_bytes'] or raw.get('url')!=pin['archive_url']:
 raise RuntimeError('Official archive download not complete or pin mismatch')
down={{'status':'completed','source':'remote_shared_raw_download_status.json','assets':{{'2024.01.25':{{'url':raw['url'],'expected_bytes':raw['expected_bytes'],'bytes':raw['bytes'],'sha256':raw['sha256']}}}}}}
(pathlib.Path({SHARED!r})/'download_status.json').write_text(json.dumps(down,indent=2))
print(json.dumps({{'staged_files':len(expected),'frozen_source':str(source),'archive_sha256':raw['sha256'],'shared_data':{SHARED!r}}}))
'''
        return remote_python(transport, code)
    finally:
        transport.close()


def start() -> dict:
    transport = connect(20424)
    try:
        code = f'''
import json,os,pathlib,subprocess
shared=pathlib.Path({SHARED!r})
status=shared/'preparation_status.json'
if status.exists():
 data=json.loads(status.read_text())
 if data.get('status') not in ('failed',):
  print(json.dumps({{'status':'already_'+data.get('status','unknown'),'preparation':data}}));raise SystemExit
environment=os.environ.copy()
environment.update(STAGE3_DATA_ROOT=str(shared),STAGE3_DATE='2024-01-25',
 STAGE3_G07_FROZEN_ROOT=str(shared/'ops'/{G07_COMMIT!r}),
 STAGE3_G07_FROZEN_SRC=str(shared/'ops'/{G07_COMMIT!r}/'src'),
 STAGE3_G07_TEST_RECORD=str(shared/'ops/g07_test_record.json'))
log=shared/'ops/jan25_scene_preparation.log'
with log.open('ab') as output:
 process=subprocess.Popen([{DQN!r},str(shared/'ops/prepare_aviation_scenes.py')],env=environment,
  stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
print(json.dumps({{'status':'started','pid':process.pid,'log':str(log),'preparation_status':str(status)}}))
'''
        return remote_python(transport, code)
    finally:
        transport.close()


def status() -> dict:
    transport = connect(20424)
    try:
        code = f'''
import json,pathlib
shared=pathlib.Path({SHARED!r})
result={{}}
for name in ('preparation_status.json','independent_test_manifest.json'):
 p=shared/name
 if p.is_file():
  data=json.loads(p.read_text())
  result[name]={{k:data.get(k) for k in ('status','date_utc','count','built_count','ended_utc','error','source_archive_sha256')}}
print(json.dumps(result))
'''
        return remote_python(transport, code)
    finally:
        transport.close()


def verify() -> dict:
    transport = connect(20424)
    try:
        code = f'''
import json,os,subprocess
environment=os.environ.copy()
environment.update(STAGE3_DATA_ROOT={SHARED!r},STAGE3_DATE='2024-01-25',
 STAGE3_G07_FROZEN_SRC={ (OPS + '/' + G07_COMMIT + '/src')!r})
p=subprocess.run([{DQN!r},{(OPS + '/verify_aviation_scenes.py')!r}],env=environment,
 capture_output=True,text=True,timeout=300)
print(json.dumps({{'exit_code':p.returncode,'stdout':p.stdout[-1500:],'stderr':p.stderr[-1500:]}}))
'''
        return remote_python(transport, code)
    finally:
        transport.close()


def collect() -> dict:
    targets = ("raw/download_status.json", "selection_contract.json", "source_preflight.json",
               "preparation_status.json", "independent_test_manifest.json",
               "independent_test_verification.json", "staged_test/record.json")
    destination = LOCAL_PREFLIGHT / "remote_jan25_receipts"
    destination.mkdir(parents=True, exist_ok=True)
    transport = connect(20424)
    try:
        with paramiko.SFTPClient.from_transport(transport) as sftp:
            for relative in targets:
                remote = SHARED + "/" + relative
                if sftp.stat(remote).st_size > 20 * 1024 * 1024:
                    raise RuntimeError("Receipt exceeds expected size: " + relative)
                local = destination / relative.replace("/", "__")
                sftp.get(remote, str(local))
    finally:
        transport.close()
    checks = {}
    for port in (20418, 20424):
        transport = connect(port)
        try:
            code = f'''
import hashlib,json,pathlib
shared=pathlib.Path({SHARED!r})
names=['staged_test/record.json','independent_test_verification.json','raw/download_status.json']
result={{}}
for name in names:
 path=shared/name
 result[name]={{'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}
print(json.dumps(result))
'''
            checks[str(port)] = remote_python(transport, code)
        finally:
            transport.close()
    local = {relative: {"path": str(destination / relative.replace("/", "__")),
                        "sha256": sha(destination / relative.replace("/", "__"))} for relative in targets}
    if any(checks["20418"][name]["sha256"] != checks["20424"][name]["sha256"]
           for name in checks["20418"]):
        raise RuntimeError("Shared Jan25 receipts differ between SSH instances")
    for name in checks["20424"]:
        if local[name]["sha256"] != checks["20424"][name]["sha256"]:
            raise RuntimeError("Downloaded local receipt SHA differs: " + name)
    output = {"status": "jan25_inputs_verified_on_both_instances_no_evaluation",
              "local_receipts": local, "remote_checks": checks,
              "shared_data": SHARED,
              "source_archive_sha256": json.loads((destination / "raw__download_status.json").read_text())["sha256"]}
    path = LOCAL_PREFLIGHT / "remote_jan25_data_receipt.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"receipt": str(path), "status": output["status"],
            "scene_record_sha256": checks["20424"]["staged_test/record.json"]["sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("stage", "start", "status", "verify", "collect"))
    action = parser.parse_args().action
    print(json.dumps({"stage": stage, "start": start, "status": status, "verify": verify,
                      "collect": collect}[action](), ensure_ascii=False))


if __name__ == "__main__":
    main()
