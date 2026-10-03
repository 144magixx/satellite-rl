"""Resume same e22 S1-02 controller state after audited analysis-only fix."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, "C:/Users/Admin/.codex/worktrees/stage1-balance-actions/paper/scripts/stage1_balance")
from remote_sources import LOCAL, REMOTE, connect, run_code  # noqa: E402


PLAN_SHA = "a9cebefc5b06eaf25f7459c7670b9bb11f9aaf218dfd90d5ecb0f987872f0bf7"
SCIENCE = "e22f449ccee11d525bce1c8011f297dc87bc86e1"
ANALYSIS = "57ceffac9162578e0c924f551466d408a80fc527"


def main() -> None:
    code = r'''
import datetime,hashlib,json,os,pathlib,subprocess,urllib.request
root=pathlib.Path(__REMOTE__);ops=root/'ops';repo=root/'repo'
plan=json.loads((ops/'controller_plan_recovery.json').read_text())
state=json.loads((ops/'controller_state_recovery.json').read_text())
route=json.loads((ops/'current_controller_route.json').read_text())
override=ops/'analysis_override'/__ANALYSIS__
contract=json.loads((override/'contract.json').read_text())
if (hashlib.sha256((ops/'controller_plan_recovery.json').read_bytes()).hexdigest()!=__PLAN_SHA__ or
    plan['source_commit']!=__SCIENCE__ or state['plan_sha256']!=__PLAN_SHA__ or
    state['phase']!='needs_attention' or route['plan_sha256']!=__PLAN_SHA__ or
    route['source_commit']!=__SCIENCE__ or contract['science_commit']!=__SCIENCE__ or
    contract['analysis_commit']!=__ANALYSIS__):
 raise RuntimeError('Frozen recovery plan/state/route/analysis source differs')
old=json.loads((ops/'controller_recovery_launch.json').read_text())
proc=pathlib.Path('/proc')/str(old['pid'])
if proc.exists():
 stat=(proc/'stat').read_text();ticks=int(stat[stat.rfind(')')+2:].split()[19])
 cmd=(proc/'cmdline').read_bytes().decode('utf-8','replace')
 if ticks==old['start_ticks'] and 'controller_recovery.py' in cmd:
  raise RuntimeError('Previous recovery controller still running')
expected={'A_dev_A0','A_dev_A1','A_dev_A2','A_dev_A3'}
if not expected.issubset(state['jobs']):raise RuntimeError('Four recovery A tasks missing')
for label in expected:
 job=state['jobs'][label];directory=root/'outputs/workbench/jobs'/job['run_id']
 status=json.loads((directory/'status.json').read_text())
 if (status['stage']!='completed' or status.get('evaluated_scenes')!=20 or
     not (directory/'results.json').is_file()):
  raise RuntimeError(label+' completed 20-scene result differs')
active={'queued','starting','initializing','warmup','training','validating',
        'running','evaluating','preparing'}
for path in (root/'outputs/workbench/jobs').glob('*/status.json'):
 status=json.loads(path.read_text())
 if status.get('stage') in active:raise RuntimeError('Active/queued work exists before continuation')
for name,meta in contract['files'].items():
 file=override/name
 if hashlib.sha256(file.read_bytes()).hexdigest()!=meta['sha256']:
  raise RuntimeError('Analysis override file SHA differs: '+name)
post=ops/'postprocess_override.py';helper=ops/'controller_recovery_attempt2.py'
if not post.is_file() or not helper.is_file():raise RuntimeError('Audited ops helpers missing')
if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()!=__SCIENCE__:
 raise RuntimeError('Scientific repo moved after four A methods')
with urllib.request.urlopen('http://127.0.0.1:8892/api/system',timeout=10) as response:system=json.load(response)
if pathlib.Path(system['source_root']).resolve()!=repo or system.get('cuda') is not True:
 raise RuntimeError('Scientific workbench moved or CUDA unavailable')
receipt=ops/'controller_recovery_attempt2_launch.json'
if receipt.exists():raise RuntimeError('Second controller attempt already launched')
python='/home/u2024110224/first_stage_satellite_rl/envs/DQN/bin/python'
env=dict(os.environ,PYTHONPATH=str(repo/'src'),PYTHONDONTWRITEBYTECODE='1',
 OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
check=subprocess.run([python,'-B',str(helper),'--plan',str(ops/'controller_plan_recovery.json'),
 '--state',str(ops/'controller_state_recovery.json'),'--check'],cwd=repo,env=env,
 capture_output=True,text=True,timeout=60)
if check.returncode:raise RuntimeError('Second controller precheck failed: '+check.stderr[-1000:])
prior_state_sha=hashlib.sha256((ops/'controller_state_recovery.json').read_bytes()).hexdigest()
log=ops/'controller_recovery_attempt2.log'
with log.open('ab',buffering=0) as stream:
 p=subprocess.Popen([python,'-B',str(helper),'--plan',str(ops/'controller_plan_recovery.json'),
  '--state',str(ops/'controller_state_recovery.json')],cwd=repo,env=env,
  stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
  start_new_session=True,close_fds=True)
stat=(pathlib.Path('/proc')/str(p.pid)/'stat').read_text();ticks=int(stat[stat.rfind(')')+2:].split()[19])
value={'status':'recovery_controller_attempt2_launched','pid':p.pid,'start_ticks':ticks,
 'source_commit':__SCIENCE__,'plan_sha256':__PLAN_SHA__,
 'prior_state_sha256':prior_state_sha,'analysis_commit':contract['analysis_commit'],
 'analysis_contract_sha256':hashlib.sha256((override/'contract.json').read_bytes()).hexdigest(),
 'controller_script_sha256':hashlib.sha256(helper.read_bytes()).hexdigest(),
 'postprocess_script_sha256':hashlib.sha256(post.read_bytes()).hexdigest(),
 'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
receipt.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(value,ensure_ascii=False))
'''
    code = (code.replace("__REMOTE__", repr(REMOTE))
                .replace("__PLAN_SHA__", repr(PLAN_SHA))
                .replace("__SCIENCE__", repr(SCIENCE))
                .replace("__ANALYSIS__", repr(ANALYSIS)))
    client = connect()
    try:
        result = run_code(client, code)
    finally:
        client.close()
    output = LOCAL / "preflight/controller_recovery_attempt2_launch.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"receipt": str(output), "pid": result["pid"],
                      "analysis_commit": result["analysis_commit"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
