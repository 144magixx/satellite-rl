"""Launch a persistent coordinator with a kernel ownership lock and durable logs."""
from pathlib import Path
import json
import os
import subprocess
import sys
from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.storage import now
from .controller import validate_plan

def launch(plan_path,state_path,python=None):
    plan_path=Path(plan_path).resolve();state_path=Path(state_path).resolve()
    validate_plan(json.loads(plan_path.read_text(encoding='utf-8-sig')))
    state_path.parent.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).resolve().parents[2]
    env=dict(os.environ);env['PYTHONPATH']=str(source/'src')+os.pathsep+str(source)
    command=[python or sys.executable,'-u','-m','scripts.stage1_geoload.controller','--plan',str(plan_path),'--state',str(state_path)]
    flags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0
    with state_path.with_suffix('.log').open('ab',buffering=0) as log:
        child=subprocess.Popen(command,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            creationflags=flags,start_new_session=os.name!='nt',close_fds=True)
    from satellite_rl.workbench.runtime import process_start_time
    receipt={'pid':child.pid,'process_start_time':process_start_time(child.pid),'command':command,'started_at':now()}
    atomic_json(state_path.with_suffix('.launch.json'),receipt)
    return receipt
