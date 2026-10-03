"""Install and verify S3-P03 convex solver wheels outside the shared DQN environment."""
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
OUTPUT = Path("D:/paper/outputs/experiments/stage3_si_priority_20261003/preflight")


def remote(port: int, action: str) -> dict:
    code = f'''
import datetime,importlib.util,json,os,pathlib,subprocess,sys
root=pathlib.Path({BASE!r});deps=root/'shared_data/deps/cvxpy-1.6.5-clarabel-0.11.1'
dqn={DQN!r};action={action!r}
thread={{name:'1' for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')}}
env=dict(os.environ,**thread,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(deps))
old=subprocess.run([dqn,'-B','-c','import json,numpy,scipy,torch;print(json.dumps({{"numpy":numpy.__version__,"scipy":scipy.__version__,"torch":torch.__version__}}))'],capture_output=True,text=True,check=True,env=dict(os.environ,**thread,PYTHONDONTWRITEBYTECODE='1'))
before=json.loads(old.stdout)
available={{name:(spec.origin if (spec:=importlib.util.find_spec(name)) else None) for name in ('cvxpy','clarabel','osqp','scs')}}
distinfo={{name:[{{'directory':str(path),'metadata':(path/'METADATA').is_file(),
  'record':(path/'RECORD').is_file()}} for path in deps.glob(name+'-*.dist-info')]
  for name in ('cvxpy','clarabel')}}
install=None
if action=='install':
 deps.mkdir(parents=True,exist_ok=True)
 if not (deps/'cvxpy').is_dir() or not (deps/'clarabel').is_dir():
  p=subprocess.run([dqn,'-m','pip','install','--disable-pip-version-check','--no-input','--no-deps',
    '--target',str(deps),'cvxpy==1.6.5','clarabel==0.11.1'],capture_output=True,text=True,timeout=600)
  install={{'returncode':p.returncode,'stdout_tail':p.stdout[-1000:],'stderr_tail':p.stderr[-1000:]}}
  if p.returncode:raise RuntimeError('Isolated solver install failed: '+p.stderr[-600:])
  (deps/'solver_requirements.txt').write_text('cvxpy==1.6.5\\nclarabel==0.11.1\\n--no-deps\\n')
 else:install={{'status':'existing_target'}}
probe='import json,numpy,scipy,torch,cvxpy,clarabel;print(json.dumps({{"numpy":numpy.__version__,"scipy":scipy.__version__,"torch":torch.__version__,"cvxpy":cvxpy.__version__,"clarabel":clarabel.__version__,"solvers":cvxpy.installed_solvers(),"cvxpy_file":cvxpy.__file__,"clarabel_file":clarabel.__file__}}))'
check=subprocess.run([dqn,'-B','-c',probe],capture_output=True,text=True,env=env,timeout=90)
after=json.loads(check.stdout.splitlines()[-1]) if check.returncode==0 else None
if action=='install' and (check.returncode or after is None or before['numpy']!=after['numpy'] or before['scipy']!=after['scipy'] or before['torch']!=after['torch'] or after['cvxpy']!='1.6.5' or after['clarabel']!='0.11.1' or 'CLARABEL' not in after['solvers']):
 raise RuntimeError('Isolated solver verification failed: '+check.stderr[-1000:])
print(json.dumps({{'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'port':{port},'action':action,
 'deps_root':str(deps),'available_before':available,'distinfo':distinfo,
 'dqn_before':before,'install':install,
 'probe_returncode':check.returncode,'probe_stderr_tail':check.stderr[-1000:],'after':after}}))
'''
    payload = base64.b64encode(code.encode()).decode()
    transport = connect(port)
    try:
        result = execute(transport, "python3 -B -c 'exec(__import__(\"base64\").b64decode(\"" + payload + "\"))'", timeout=720)
    finally:
        transport.close()
    if result["exit_code"]:
        raise RuntimeError(result["stderr"][-1400:])
    return json.loads(result["stdout"])


def main() -> None:
    action = sys.argv[1] if len(sys.argv) > 1 else "probe"
    if action not in ("probe", "install"):
        raise ValueError("action must be probe or install")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = {str(port): remote(port, action) for port in (20418, 20424)}
    path = OUTPUT / f"remote_dependencies_{action}.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"receipt": str(path), "summary": {k: v["after"] for k, v in results.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
