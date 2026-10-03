"""Launch and inspect the bounded Jan12 A/B/C development check on 20418."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, "D:/paper/outputs/verification/g11_remote_launch_20261001")
from remote_transport import connect, execute, paramiko  # noqa: E402


HERE = Path(__file__).resolve().parent
ROOT = Path("D:/paper/outputs/verification/stage3_si_priority_20261003/jan12_dev")
BASE = "/home/u2024110224/third_stage_satellite_rl/s3_si_priority_20261003"
DQN = "/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python"
SOURCE = Path("D:/paper/outputs/verification/stage3_power_20261003/real_t0_development_probe/probe_same_initial.json")
SCENES = Path("D:/paper/outputs/experiments/europe_tmg_20261001/remote_archive/20424/archive/outputs/workbench/scenes/scenes_20261001_161628_819160d7")
REMOTE_INPUT = BASE + "/shared_data/dev_jan12_probe"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare() -> tuple[Path, dict]:
    ROOT.mkdir(parents=True, exist_ok=True)
    old = json.loads(SOURCE.read_text(encoding="utf-8"))
    model = old["model"]
    paths = {
        "policy.pt": Path(model["policy"]["path"]),
        "model.json": Path(model["metadata"]["path"]),
        "T0.json": Path(model["config"]["path"]),
        "probe_jan12_remote.py": HERE / "probe_jan12_remote.py",
    }
    if (sha(paths["policy.pt"]) != "ae095e8246d22b2716ba154315cee18584917d215580821cb53f9aeba78eb824"
            or sha(paths["model.json"]) != model["metadata"]["sha256"]
            or sha(paths["T0.json"]) != model["config"]["sha256"]):
        raise RuntimeError("T0 model endpoint asset SHA mismatch")
    for old_scene in old["scenes"][:2]:
        idx = old_scene["scene_index"]
        if idx not in (0, 1):
            raise RuntimeError("Expected preexisting Jan12 scene indices 0 and 1")
        paths[f"scene_{idx:05d}.npz"] = SCENES / f"scene_{idx:05d}.npz"
        paths[f"initial_scene_{idx:05d}.json"] = Path(old_scene["initial_path"])
        if (sha(paths[f"scene_{idx:05d}.npz"]) != old_scene["scene_sha256"] or
                sha(paths[f"initial_scene_{idx:05d}.json"]) != old_scene["initial_sha256"]):
            raise RuntimeError("Jan12 saved scene or T0 initial SHA mismatch")
    manifest = {"status": "frozen_development_inputs_only", "checked_utc": datetime.now(timezone.utc).isoformat(),
                "source_probe": str(SOURCE), "source_probe_sha256": sha(SOURCE),
                "date_utc": "2024-01-12", "scene_indices": [0, 1],
                "policy_inference_calls": 0, "training_interactions": 0,
                "files": [{"name": name, "bytes": path.stat().st_size, "sha256": sha(path)}
                          for name, path in sorted(paths.items())]}
    archive = ROOT / "jan12_inputs.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for name, path in sorted(paths.items()):
            output.write(path, name)
        output.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    manifest["package_sha256"] = sha(archive)
    (ROOT / "input_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return archive, manifest


def call(code: str, timeout: int = 120) -> dict:
    payload = base64.b64encode(code.encode()).decode()
    transport = connect(20418)
    try:
        result = execute(transport, "python3 -B -c 'exec(__import__(\"base64\").b64decode(\"" + payload + "\"))'", timeout=timeout)
    finally:
        transport.close()
    if result["exit_code"]:
        raise RuntimeError(result["stderr"][-1300:])
    return json.loads(result["stdout"])


def start() -> dict:
    archive, manifest = prepare()
    source = json.loads((Path("D:/paper/outputs/verification/stage3_si_priority_20261003/remote_dev") /
                         "dev_test_20418_test_all_1456a49aa5c3c9bb.json").read_text(encoding="utf-8"))
    if source["exit_code"] != 0:
        raise RuntimeError("Full development suite must pass before Jan12 numerical check")
    transport = connect(20418)
    try:
        mkdir = execute(transport, "mkdir -p " + REMOTE_INPUT, timeout=30)
        if mkdir["exit_code"]:
            raise RuntimeError(mkdir["stderr"][-500:])
        with paramiko.SFTPClient.from_transport(transport) as sftp:
            sftp.put(str(archive), REMOTE_INPUT + "/jan12_inputs.zip.part")
    finally:
        transport.close()
    code = f'''
import datetime,hashlib,json,os,pathlib,subprocess,zipfile
root=pathlib.Path({REMOTE_INPUT!r});part=root/'jan12_inputs.zip.part';archive=root/'jan12_inputs.zip'
if hashlib.sha256(part.read_bytes()).hexdigest()!={manifest['package_sha256']!r}:
 raise RuntimeError('Jan12 input upload SHA mismatch')
part.replace(archive)
with zipfile.ZipFile(archive) as z:
 names=z.namelist()
 if any(pathlib.PurePosixPath(name).is_absolute() or '..' in pathlib.PurePosixPath(name).parts for name in names):
  raise RuntimeError('Unexpected Jan12 input path')
 z.extractall(root)
source=pathlib.Path({source['snapshot_root']!r})
deps=pathlib.Path({(BASE + '/shared_data/deps/cvxpy-1.6.5-clarabel-0.11.1')!r})
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(source/'src')+os.pathsep+str(deps),
 S3P03_JAN12_ROOT=str(root),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
log=root/'probe.log'
with log.open('ab') as output:
 p=subprocess.Popen([{DQN!r},'-B',str(root/'probe_jan12_remote.py')],cwd=source,env=env,
  stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
stat=(pathlib.Path('/proc')/str(p.pid)/'stat').read_text();ticks=int(stat[stat.rfind(')')+2:].split()[19])
print(json.dumps({{'status':'started_development_only','pid':p.pid,'start_ticks':ticks,
 'source_snapshot':str(source),'input_sha256':{manifest['package_sha256']!r},
 'log':str(log),'status_file':str(root/'status.json'),
 'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}}))
'''
    result = call(code)
    path = ROOT / "launch_receipt.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"receipt": str(path), **result}


def status() -> dict:
    launch = json.loads((ROOT / "launch_receipt.json").read_text(encoding="utf-8"))
    code = f'''
import json,pathlib
root=pathlib.Path({REMOTE_INPUT!r});pid={launch['pid']};expected={launch['start_ticks']}
proc=pathlib.Path('/proc')/str(pid)
if proc.is_dir():
 stat=(proc/'stat').read_text();live=int(stat[stat.rfind(')')+2:].split()[19])
else:live=None
state=json.loads((root/'status.json').read_text()) if (root/'status.json').is_file() else None
print(json.dumps({{'process_birth_matches':live==expected,'pid':pid,'status':state,
 'receipt_exists':(root/'receipt.json').is_file(),
 'log_tail':(root/'probe.log').read_text(errors='replace')[-1000:] if (root/'probe.log').is_file() else ''}}))
'''
    return call(code)


def collect() -> dict:
    state = status()
    if not state["status"] or state["status"]["status"] != "completed":
        raise RuntimeError("Jan12 development check is not complete")
    checks = {}
    transport = connect(20418)
    try:
        with paramiko.SFTPClient.from_transport(transport) as sftp:
            for name in ("receipt.json", "status.json", "probe.log"):
                sftp.get(REMOTE_INPUT + "/" + name, str(ROOT / name))
                checks[name] = sha(ROOT / name)
    finally:
        transport.close()
    receipt = json.loads((ROOT / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("status") != "completed_development_only" or len(receipt.get("cases", [])) != 12:
        raise RuntimeError("Jan12 receipt case grid incomplete")
    result = {"status": "jan12_2_scenes_2_budgets_ABC_development_verified",
              "collected_utc": datetime.now(timezone.utc).isoformat(),
              "files_sha256": checks, "case_count": 12,
              "scene_indices": sorted({row["scene_index"] for row in receipt["cases"]}),
              "methods": sorted({row["method"] for row in receipt["cases"]})}
    path = ROOT / "collection_receipt.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"receipt": str(path), **result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "status", "collect"))
    action = parser.parse_args().action
    print(json.dumps({"start": start, "status": status, "collect": collect}[action](), ensure_ascii=False))


if __name__ == "__main__":
    main()
