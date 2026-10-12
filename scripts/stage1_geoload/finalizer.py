"""Final completion audit; delivery and scientific result are separate facts."""
import json
from pathlib import Path
from . import ops

def audit(plan,state):
    if plan.get('group')!='S1-G16' or state.get('group')!='S1-G16':raise ValueError('wrong finalizer group')
    tasks=state.get('tasks',{})
    if not tasks:raise ValueError('no authorized task ledger')
    identities=[(x['port'],x['run_id']) for x in tasks.values()]
    if len(set(identities))!=len(identities):raise ValueError('duplicate task ownership')
    rows=[]
    for key,task in tasks.items():
        info=ops.api(task['port'],'/api/runs/'+task['run_id'])
        job,status=info['job'],info['status']
        if job.get('experiment_id')!='S1-G16' or job.get('scientific_commit')!=plan['scientific_commit']:
            raise ValueError('task scientific source differs')
        rows.append({'key':key,'port':task['port'],'run_id':task['run_id'],'stage':status['stage'],
                     'completed':status['stage']=='completed','artifacts':info['artifacts']})
    value={'group':'S1-G16','tasks':rows,'all_tasks_completed':all(x['completed'] for x in rows),
           'delivery_verified':False,'scientific_commit':plan['scientific_commit']}
    return value
