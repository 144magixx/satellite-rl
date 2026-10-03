"""Exercise the SI-priority worker preflight without evaluating a scenario."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, "D:/paper/outputs/verification/g11_remote_launch_20261001")
from remote_transport import connect, execute  # noqa: E402


BASE = "/home/u2024110224/third_stage_satellite_rl/s3_si_priority_20261003"
DQN = "/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python"
SNAPSHOT = BASE + "/shared_data/dev_snapshots/43ab4e81ecd88547"
OUTPUT = Path("D:/paper/outputs/verification/stage3_si_priority_20261003/remote_dev")


def check(port: int) -> dict:
    code = f'''
import datetime,hashlib,json,os,pathlib,shutil,subprocess
base=pathlib.Path({BASE!r});snapshot=pathlib.Path({SNAPSHOT!r});port={port}
source=base/'shared_data/dev_worker_source/43ab4e81ecd88547'
jobdir=base/str(port)/'dev_worker_preflight'
deps=base/'shared_data/deps/cvxpy-1.6.5-clarabel-0.11.1'
if not source.is_dir():
 source.mkdir(parents=True)
 shutil.copytree(snapshot/'src',source/'src')
files={{str(path.relative_to(source)):hashlib.sha256(path.read_bytes()).hexdigest()
 for path in sorted((source/'src').rglob('*')) if path.is_file()}}
identity='development-snapshot-43ab4e81ecd88547'
(source/'source_manifest.json').write_text(json.dumps({{'git_commit':identity,'files':files}},indent=2))
jobdir.mkdir(parents=True,exist_ok=True)
dependency_files=[{{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}
 for package,version in (('cvxpy','1.6.5'),('clarabel','0.11.1'))
 for name in ('METADATA','RECORD')
 for path in (deps/f'{{package}}-{{version}}.dist-info'/name,)]
if len(dependency_files)!=4:raise RuntimeError('Solver distribution files missing')
job={{'kind':'refine','run_id':'dev_worker_preflight_{port}',
 'source_root':str(source),'git_commit':identity,
 'priority_solver_deps_path':str(deps),'priority_solver_dependency_files':dependency_files}}
config={{'research_stage':'power_refinement','power_refinement':{{'methods':['sca','sca_si_lp','sca_si_lb']}}}}
(jobdir/'job.json').write_text(json.dumps(job,indent=2))
(jobdir/'config.json').write_text(json.dumps(config,indent=2))
script="""from pathlib import Path
import json
import satellite_rl.power_refinement.runner as runner
runner.evaluate_refinement=lambda *args,**kwargs: {{"development_preflight_only":True}}
result=runner.run_refinement_job(Path(__JOBDIR__))
status=json.loads((Path(__JOBDIR__)/'status.json').read_text())
print(json.dumps({{"result":result,"status":status}}))""".replace('__JOBDIR__',repr(str(jobdir)))
env=os.environ.copy();env.update(PYTHONPATH=str(source/'src')+os.pathsep+str(deps),
 PYTHONDONTWRITEBYTECODE='1',SATELLITE_RL_PRIORITY_DEPS=str(deps),
 OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
check=subprocess.run([{DQN!r},'-B','-c',script],cwd=source,env=env,
 capture_output=True,text=True,timeout=120)
if check.returncode:raise RuntimeError('Worker preflight failed: '+check.stderr[-1400:])
response=json.loads(check.stdout.splitlines()[-1])
preflight=response['status']['preflight']
solver=preflight['priority_solver']
if (response['result']!={{'development_preflight_only':True}}
 or response['status']['stage']!='initializing'
 or preflight['source_root']!=str(source)
 or not preflight['package_file'].startswith(str(source/'src'))
 or solver['dependency_root']!=str(deps)
 or not solver['cvxpy_file'].startswith(str(deps))
 or not solver['clarabel_file'].startswith(str(deps))
 or solver['cvxpy_version']!='1.6.5' or solver['clarabel_version']!='0.11.1'
 or solver['solver']!='CLARABEL'):
 raise RuntimeError('Actual DQN worker dependency/source preflight differs')
print(json.dumps({{'status':'real_worker_preflight_only_verified','checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'port':port,'source_root':str(source),'source_manifest_sha256':hashlib.sha256((source/'source_manifest.json').read_bytes()).hexdigest(),
 'job_dir':str(jobdir),'preflight':preflight,'dependency_files':dependency_files,
 'evaluation_calls':0,'training_interactions':0}}))
'''
    payload = base64.b64encode(code.encode()).decode()
    transport = connect(port)
    try:
        result = execute(transport, "python3 -B -c 'exec(__import__(\"base64\").b64decode(\"" + payload + "\"))'", timeout=180)
    finally:
        transport.close()
    if result["exit_code"]:
        raise RuntimeError(result["stderr"][-1800:])
    return json.loads(result["stdout"])


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    receipt = {"checked_utc": datetime.now(timezone.utc).isoformat(),
               "instances": {str(port): check(port) for port in (20418, 20424)}}
    path = OUTPUT / "worker_preflight_both_ports.json"
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"receipt": str(path), "summary": {p: r["status"] for p, r in receipt["instances"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
