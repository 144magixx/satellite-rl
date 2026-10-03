"""Submit the authorized Jan27 SI-priority comparison through workbench APIs.

Port 20418 runs the 3000 W total-budget arm; 20424 runs the initial-power arm.
Both run A/B/C on identical frozen scenes and policy samples.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys


sys.path.insert(0, "D:/paper/outputs/verification/g11_remote_launch_20261001")
from remote_transport import connect, execute  # noqa: E402


DQN = "/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python"
BASE = "/home/u2024110224/third_stage_satellite_rl/s3_si_priority_20261003"
PREFLIGHT = Path("D:/paper/outputs/experiments/stage3_si_priority_20261003/preflight")
PORTS = {20418: (8776, "total"), 20424: (8777, "initial")}
REQUESTS = {20418: "s3p03-20261003-jan27-total", 20424: "s3p03-20261003-jan27-initial"}
METHODS = ["sca", "sca_si_lp", "sca_si_lb"]
POLICY_SHA = "ae095e8246d22b2716ba154315cee18584917d215580821cb53f9aeba78eb824"


def run_remote(port: int, code: str, timeout: int = 120) -> dict:
    encoded = base64.b64encode(code.encode()).decode()
    command = DQN + " -B -c 'exec(__import__(\"base64\").b64decode(\"" + encoded + "\"))'"
    transport = connect(port)
    try:
        result = execute(transport, command, timeout=timeout)
    finally:
        transport.close()
    if result["exit_code"]:
        raise RuntimeError(f"SSH {port} request failed: " + result["stderr"][-1400:])
    return json.loads(result["stdout"])


def code(port: int, commit: str, submit: bool) -> str:
    wb_port, mode = PORTS[port]
    t0 = json.loads((PREFLIGHT / f"t0_import_{port}.json").read_text(encoding="utf-8"))
    scenes = json.loads((PREFLIGHT / f"scene_import_{port}.json").read_text(encoding="utf-8"))
    data = json.loads((PREFLIGHT / "remote_jan27_data_receipt.json").read_text(encoding="utf-8"))
    scene_record_sha = data["remote_checks"]["20424"]["staged_test/record.json"]["sha256"]
    if (t0.get("commit") != commit or scenes.get("commit") != commit
            or t0.get("policy_sha256") != POLICY_SHA
            or scenes.get("original_record_sha256") != scene_record_sha):
        raise RuntimeError(f"Frozen source/model/scene import receipt differs on {port}")
    return f'''
import datetime,json,pathlib,subprocess,urllib.request
root=pathlib.Path({BASE!r})/{str(port)!r}
repo=root/'repo';archive=root/'archive';ops=root/'ops';port={wb_port}
head=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
if head!={commit!r}:raise RuntimeError('Frozen checkout commit differs')
def api(method,path,body=None,timeout=120):
 data=None if body is None else json.dumps(body,ensure_ascii=False).encode()
 req=urllib.request.Request(f'http://127.0.0.1:{{port}}/api'+path,data=data,method=method,
  headers={{'Content-Type':'application/json','Cache-Control':'no-cache'}})
 with urllib.request.urlopen(req,timeout=timeout) as response:return json.load(response)
system=api('GET','/system')
if system.get('source_root')!=str(repo) or system.get('archive_root')!=str(archive) or system.get('cuda') is not True:
 raise RuntimeError('Stage3 live workbench differs')
model_id={t0['model_id']!r};scene_id={scenes['scene_id']!r};mode={mode!r}
models=api('GET','/power-refinement/models')['models']
scenes=api('GET','/power-refinement/scenes')['scenes']
if model_id not in [m['id'] for m in models] or scene_id not in [s['id'] for s in scenes]:
 raise RuntimeError('Imported T0/Jan27 missing from stage3 eligible lists')
defaults=api('GET','/power-refinement/defaults')['defaults']
if (defaults.get('repeats')!=3
 or defaults.get('action_mode')!='stochastic' or defaults.get('action_seed')!=53001
 or defaults.get('optimizer_seed')!=63001
 or defaults.get('protect_basic_service') is not True or defaults.get('maxiter')!=300
 or defaults.get('maxeval')!=2000 or defaults.get('max_time_seconds')!=30.0
 or defaults.get('si_stall_window')!=200 or defaults.get('si_stall_tolerance')!=1e-6
 or defaults.get('si_min_evaluations')!=200 or defaults.get('si_max_fraction')!=0.85
 or defaults.get('service_margin_ratio')!=5e-9):
 raise RuntimeError('Stage3 defaults differ from approved common protocol')
config=dict(defaults,methods={METHODS!r},power_budget_mode=mode)
validated=api('POST','/power-refinement/validate',config)['config']
if validated!=config:raise RuntimeError('Validated configuration differs')
body={{'model_id':model_id,'scene_id':scene_id,'config':config,'request_id':{REQUESTS[port]!r}}}
if {submit!r}:
 job=api('POST','/power-refinement/runs',body,timeout=180)
else:job=None
result={{'status':'submitted' if {submit!r} else 'ready_to_submit',
 'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'ssh_port':{port},'workbench_port':port,'commit':head,'mode':mode,
 'request_id':body['request_id'],'model_id':model_id,'scene_id':scene_id,
 'config':config,'evaluation_units_expected':100*3*3,'job':job}}
if {submit!r}:(ops/f'jan27_{{mode}}_submission.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False))
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--submit", action="store_true", help="Actually create both workbench jobs")
    args = parser.parse_args()
    commit = args.commit.lower()
    if not re.fullmatch("[0-9a-f]{40}", commit):
        raise ValueError("Full frozen commit required")
    preflight = {str(port): run_remote(port, code(port, commit, False)) for port in PORTS}
    if any(item["status"] != "ready_to_submit" for item in preflight.values()):
        raise RuntimeError("At least one workbench failed submission preflight")
    path = PREFLIGHT / "jan27_submission_preflight.json"
    path.write_text(json.dumps(preflight, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.submit:
        print(json.dumps({"status": "ready_to_submit_no_jobs_created", "receipt": str(path)}, ensure_ascii=False))
        return
    submitted = {}
    for port in PORTS:
        result = run_remote(port, code(port, commit, True), timeout=300)
        submitted[str(port)] = result
        (PREFLIGHT / f"jan27_submission_{port}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "both_submitted", "run_ids": {
        port: value["job"]["run_id"] for port, value in submitted.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
