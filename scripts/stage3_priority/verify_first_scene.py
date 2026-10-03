"""Verify the first nine A/B/C cases and both live workbench processes."""
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
from remote_observe_stage3 import observe  # noqa: E402


ROOT = Path("D:/paper/outputs/experiments/stage3_si_priority_20261003")
PRE = ROOT / "preflight"
BASE = "/home/u2024110224/third_stage_satellite_rl/s3_si_priority_20261003"
POLICY_SHA = "ae095e8246d22b2716ba154315cee18584917d215580821cb53f9aeba78eb824"
METHODS = ("sca", "sca_si_lp", "sca_si_lb")
FIELDS = ("initialization_seconds", "physics_setup_seconds", "optimizer_wall_seconds",
          "optimizer_callback_io_seconds", "optimizer_excluding_callback_seconds",
          "verification_seconds", "total_pipeline_seconds")


def check(port: int, commit: str) -> dict:
    submitted = json.loads((PRE / f"jan27_submission_{port}.json").read_text(encoding="utf-8"))
    run_id = submitted["job"]["run_id"]
    code = f'''
import datetime,hashlib,json,math,pathlib
root=pathlib.Path({BASE!r})/{str(port)!r}
job=root/'archive/outputs/workbench/jobs'/{run_id!r}
status=json.loads((job/'status.json').read_text())
cases=[];methods={METHODS!r};fields={FIELDS!r};policy={POLICY_SHA!r}
for repeat in range(3):
 common=[]
 a_path=job/'power_cases'/f'scene_00000_r{{repeat}}_sca.json'
 if not a_path.is_file():raise RuntimeError('First-scene A case incomplete')
 a=json.loads(a_path.read_text())
 a_case_sha=hashlib.sha256(a_path.read_bytes()).hexdigest()
 for method in methods:
  path=job/'power_cases'/f'scene_00000_r{{repeat}}_{{method}}.json'
  if not path.is_file():raise RuntimeError('First scene not yet complete: '+str(path))
  row=json.loads(path.read_text());timing=row.get('timing') or {{}}
  if (row.get('source_model_sha256')!=policy or len(row.get('beams') or [])!=100
      or row.get('optimizer_seed') is not None or row.get('method')!=method):
   raise RuntimeError('First-scene model, method, seed or beam identity differs')
  if any(key not in timing or not math.isfinite(timing[key]) or timing[key]<0 for key in fields):
   raise RuntimeError('Missing/nonfinite timing: '+str(path))
  if not math.isclose(timing['optimizer_callback_io_seconds']+timing['optimizer_excluding_callback_seconds'],
                      timing['optimizer_wall_seconds'],rel_tol=1e-8,abs_tol=1e-6):
   raise RuntimeError('Optimizer timing decomposition differs')
  parts=sum(timing[k] for k in ('initialization_seconds','physics_setup_seconds',
                              'optimizer_wall_seconds','verification_seconds'))
  if method=='sca':
   if ('reference_total_seconds' in timing or
       row.get('reference_allocation_sha256')!=a.get('reference_allocation_sha256')):
    raise RuntimeError('A case unexpectedly charges reference or SHA differs')
  else:
   for key in ('reference_optimizer_wall_seconds','reference_verification_seconds','reference_total_seconds'):
    if key not in timing or not math.isfinite(timing[key]) or timing[key]<0:
     raise RuntimeError('B/C shared A timing field missing')
   if not math.isclose(timing['reference_total_seconds'],
       a['timing']['optimizer_wall_seconds']+a['timing']['verification_seconds'],rel_tol=1e-8,abs_tol=1e-6):
    raise RuntimeError('Shared A timing differs')
   parts+=timing['reference_total_seconds']
   if (row.get('reference_allocation_sha256')!=a.get('reference_allocation_sha256')
       or row.get('reference_case_sha256')!=a_case_sha
       or row.get('reference_solver_evaluations')!=a['solver']['evaluations']
       or row.get('complete_route_solver_evaluations')!=a['solver']['evaluations']+row['solver']['evaluations']
       or row.get('complete_route_solver_evaluations')>4000
       or row.get('reference_metrics')!=a['metrics']):
    raise RuntimeError('B/C common A source, metrics or full budget differs')
  if not math.isclose(parts,timing['total_pipeline_seconds'],rel_tol=1e-8,abs_tol=1e-6):
   raise RuntimeError('Total timing decomposition differs')
  common.append(row)
  cases.append({{'scene_sha':row['source_scene_sha256'],'repeat':repeat,'method':method,
    'initial_sha':row['initial_allocation_sha256'],'reference_sha':row['reference_allocation_sha256'],
    'reference_case_sha':row.get('reference_case_sha256'),'policy_sha':row['source_model_sha256'],
    'timing':timing,'metrics':{{k:row['metrics'][k] for k in ('final_si_mean','final_sgm_mean','power_w')}}}})
 if len({{row['initial_allocation_sha256'] for row in common}})!=1:
  raise RuntimeError('First scene T0 initialization differs across methods')
print(json.dumps({{'port':{port},'run_id':{run_id!r},
 'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'status':status.get('stage'),'env_step':status.get('env_step'),
 'first_scene_cases':cases}},ensure_ascii=False))
'''
    payload = base64.b64encode(code.encode()).decode()
    transport = connect(port)
    try:
        response = execute(transport, "python3 -B -c 'exec(__import__(\"base64\").b64decode(\"" + payload + "\"))'", timeout=120)
    finally:
        transport.close()
    if response["exit_code"]:
        raise RuntimeError(response["stderr"][-1200:])
    result = json.loads(response["stdout"])
    observed = observe(port, commit)
    if (observed.get("api_consistent") is not True or observed.get("source_frozen") is not True or
            observed.get("server_birth_matches") is not True or
            observed.get("worker_birth_matches") is not True):
        raise RuntimeError(f"Live process/source acceptance failed on {port}")
    result["live_observation"] = observed
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    commit = args.commit.lower()
    if not re.fullmatch("[0-9a-f]{40}", commit):
        raise ValueError("Full frozen commit required")
    results = {str(port): check(port, commit) for port in (20418, 20424)}
    left, right = results["20418"], results["20424"]
    if len(left["first_scene_cases"]) != 9 or len(right["first_scene_cases"]) != 9:
        raise RuntimeError("First-scene A/B/C case grid is incomplete")
    for a, b in zip(left["first_scene_cases"], right["first_scene_cases"]):
        if (a["scene_sha"], a["repeat"], a["method"], a["initial_sha"]) != (
                b["scene_sha"], b["repeat"], b["method"], b["initial_sha"]):
            raise RuntimeError("Both budgets differ in frozen scene/action/initial SHA")
    result = {"status": "first_scene_9_cases_per_budget_verified",
              "checked_utc": datetime.now(timezone.utc).isoformat(),
              "science_commit": commit, "ports": results,
              "same_initial_allocation_sha_across_budgets_and_methods": True,
              "shared_A_reference_sha_and_timing_verified": True,
              "policy_sha256": POLICY_SHA}
    path = PRE / "first_scene_verified.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"receipt": str(path), "status": result["status"],
                      "env_steps": {p: r["env_step"] for p, r in results.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
