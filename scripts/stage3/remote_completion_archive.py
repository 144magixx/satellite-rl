"""Archive both completed stage 3 jobs and verify every transferred file SHA.

Reads only the two current stage 3 workbench jobs and their frozen inputs.
The official 1.56 GB aviation tar remains in shared remote storage with its SHA.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys


sys.path.insert(0, "D:/paper/outputs/verification/g11_remote_launch_20261001")
from remote_transport import connect, execute, paramiko  # noqa: E402
from remote_observe_stage3 import observe  # noqa: E402


BASE = "/home/u2024110224/third_stage_satellite_rl/s3_power_20261003"
DQN = "/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python"
PREFLIGHT = Path("D:/paper/outputs/experiments/stage3_power_20261003/preflight")
ARCHIVE = Path("D:/paper/outputs/experiments/stage3_power_20261003/remote_archive")
PORTS = (20418, 20424)
MAX_FILES_PER_PORT = 10000
MAX_BYTES_PER_PORT = 5 * 1024**3
RESULT_GATE_SOURCE = '''
def completed_result_ok(status, result):
    return (status.get('stage') == 'completed'
            and result.get('research_stage') == 'power_refinement'
            and result.get('summary') is not None
            and len(result.get('scenes') or []) == 900)
'''


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def remote_json(transport, code: str, timeout: int = 600) -> dict:
    encoded = base64.b64encode(code.encode()).decode()
    command = DQN + " -c 'exec(__import__(\"base64\").b64decode(\"" + encoded + "\"))'"
    result = execute(transport, command, timeout=timeout)
    if result["exit_code"]:
        raise RuntimeError(result["stderr"][-1500:])
    return json.loads(result["stdout"])


def get_plan(transport, port: int, commit: str) -> dict:
    submitted = json.loads((PREFLIGHT / f"jan25_submission_{port}.json").read_text(encoding="utf-8"))
    model = json.loads((PREFLIGHT / f"t0_import_{port}.json").read_text(encoding="utf-8"))
    scene = json.loads((PREFLIGHT / f"scene_import_{port}.json").read_text(encoding="utf-8"))
    run_id = submitted["job"]["run_id"]
    if (submitted.get("commit") != commit or model.get("commit") != commit
            or scene.get("commit") != commit):
        raise RuntimeError("Frozen submission/import commit differs")
    code = f'''
import datetime,hashlib,json,pathlib,urllib.request
root=pathlib.Path({BASE!r})/{str(port)!r}
archive=root/'archive';run_id={run_id!r};commit={commit!r}
jobdir=archive/'outputs/workbench/jobs'/run_id
job=json.loads((jobdir/'job.json').read_text())
status=json.loads((jobdir/'status.json').read_text())
if (job.get('kind')!='refine' or job.get('git_commit')!=commit
 or status.get('stage')!='completed' or status.get('env_step')!=900
 or status.get('budget_env_steps')!=900):
 raise RuntimeError('Stage3 job not complete at fixed 900 evaluation units')
request=urllib.request.Request('http://127.0.0.1:{8772 if port==20418 else 8773}/api/runs/'+run_id+'/results',
 headers={{'Cache-Control':'no-cache'}})
with urllib.request.urlopen(request,timeout=15) as response:api_result=json.load(response)
# Completed workbench results.json carries research_stage/summary/scenes,
# while the completed stage is recorded in status.json and /runs/<id>.
{RESULT_GATE_SOURCE}
if not completed_result_ok(status,api_result):
 raise RuntimeError('Workbench API has no completed result')
model=pathlib.Path({model['model_path']!r})
scene=archive/'outputs/workbench/scenes'/{scene['scene_id']!r}
source=pathlib.Path(job.get('source_root') or '')
roots={{'job':jobdir,'model':model,'scene':scene,'source':source}}
files=[];total=0
for label,base in roots.items():
 if not base.is_dir() or not base.resolve().is_relative_to(archive.resolve()):
  raise RuntimeError('Frozen artifact root missing or outside isolated stage3 archive: '+label)
 for path in sorted(base.rglob('*')):
  if not path.is_file():continue
  relative=path.relative_to(base).as_posix()
  if path.is_symlink():raise RuntimeError('Unexpected symlink in frozen task artifact')
  size=path.stat().st_size;total+=size
  if len(files)>=10000 or total>5*1024**3:raise RuntimeError('Stage3 archive bound exceeded')
  h=hashlib.sha256()
  with path.open('rb') as stream:
   for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
  files.append({{'label':label,'relative':relative,'remote_path':str(path),
                'bytes':size,'sha256':h.hexdigest()}})
ops=root/'ops'
for name in ('workbench_{8772 if port==20418 else 8773}_launch.json',
             't0_import_receipt.json','jan25_scene_import_receipt.json',
             'jan25_{"total" if port==20418 else "initial"}_submission.json'):
 path=ops/name
 if path.is_file():
  content=path.read_bytes()
  files.append({{'label':'ops','relative':name,'remote_path':str(path),
                'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}})
print(json.dumps({{'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'ssh_port':{port},'run_id':run_id,'job_status':status['stage'],'env_step':status['env_step'],
 'source_root':str(source),'source_manifest_sha256':hashlib.sha256((source/'source_manifest.json').read_bytes()).hexdigest(),
 'model_path':str(model),'scene_path':str(scene),'api_summary':api_result['summary'],
 'files':files,'file_count':len(files),'total_bytes':sum(x['bytes'] for x in files)}},ensure_ascii=False))
'''
    return remote_json(transport, code, timeout=1200)


def transfer(transport, port: int, plan: dict) -> dict:
    destination = ARCHIVE / str(port)
    destination.mkdir(parents=True, exist_ok=True)
    copied = skipped = 0
    with paramiko.SFTPClient.from_transport(transport) as sftp:
        for item in plan["files"]:
            if item["bytes"] < 0 or item["bytes"] > MAX_BYTES_PER_PORT:
                raise RuntimeError("Invalid file size in remote manifest")
            relative = Path(item["relative"])
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError("Unsafe relative path in remote manifest")
            target = destination / item["label"] / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file() and target.stat().st_size == item["bytes"] and sha(target) == item["sha256"]:
                skipped += 1
                continue
            partial = target.with_name(target.name + ".part")
            sftp.get(item["remote_path"], str(partial))
            if partial.stat().st_size != item["bytes"] or sha(partial) != item["sha256"]:
                raise RuntimeError("Transferred file SHA/size differs: " + item["remote_path"])
            partial.replace(target)
            copied += 1
    manifest = {"port": port, "run_id": plan["run_id"],
                "source_manifest_sha256": plan["source_manifest_sha256"],
                "api_summary": plan["api_summary"],
                "files": [{**item, "local_path": str(destination / item["label"] / item["relative"])}
                          for item in plan["files"]],
                "file_count": plan["file_count"], "total_bytes": plan["total_bytes"],
                "copied": copied, "skipped_verified": skipped,
                "completed_utc": datetime.now(timezone.utc).isoformat()}
    manifest_path = destination / "archive_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"manifest": str(manifest_path), "file_count": plan["file_count"],
            "copied": copied, "skipped_verified": skipped,
            "total_bytes": plan["total_bytes"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    commit = args.commit.lower()
    if not re.fullmatch("[0-9a-f]{40}", commit):
        raise ValueError("Full frozen commit required")
    observations = {str(port): observe(port, commit) for port in PORTS}
    for port, item in observations.items():
        if (item.get("api_consistent") is not True or item.get("source_frozen") is not True
                or item.get("status", {}).get("stage") != "completed"
                or item.get("status", {}).get("env_step") != 900):
            raise RuntimeError(f"Port {port} not verifiably completed through API/PID/source")
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    (ARCHIVE / "completion_api_observation.json").write_text(
        json.dumps(observations, ensure_ascii=False, indent=2), encoding="utf-8")
    receipts = {}
    for port in PORTS:
        transport = connect(port)
        try:
            plan = get_plan(transport, port, commit)
            if plan["file_count"] > MAX_FILES_PER_PORT or plan["total_bytes"] > MAX_BYTES_PER_PORT:
                raise RuntimeError("Remote stage3 artifact bound exceeded")
            receipts[str(port)] = transfer(transport, port, plan)
        finally:
            transport.close()
    result = {"status": "both_stage3_jobs_completed_and_sha_archived",
              "completed_utc": datetime.now(timezone.utc).isoformat(),
              "commit": commit, "observations": observations,
              "ports": receipts,
              "raw_aviation_archive_remote": BASE + "/shared_data/raw/v2024.01.25-planes-readsb-staging-0.tar",
              "raw_aviation_archive_sha256": "9c1038f6b1e44c0dbbdd4817d74465d08c7375069ae4b1681feba9ded9aeca44"}
    path = ARCHIVE / "completion_archive_receipt.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"receipt": str(path), "status": result["status"],
                      "ports": receipts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
