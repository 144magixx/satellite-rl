"""Read completed original preparation IDs and sign their canonical manifests."""
import argparse,json
from pathlib import Path
from satellite_rl.training.run_monitor import atomic_json
from . import ops


def collect(state_path,output):
    state=json.loads(Path(state_path).read_text(encoding='utf-8'))
    if state.get('group')!=ops.GROUP or state.get('phase')!='completed' or len(state['intents'])!=3:
        raise ValueError('three original common preparations must have completed')
    refs={};receipts={}
    for intent in state['intents'].values():
        port=intent['port'];run=intent['run_id'];detail=ops.api(port,'/api/ai-search-r2/tasks/'+run)
        job=detail['job']
        if job['mode']!='prepare_common' or detail['status']['stage']!='completed':raise ValueError('preparation not completed')
        reference=detail['artifacts']['common_manifest']
        code=f'''import sys,json,pathlib,hashlib
sys.path.insert(0,{job['source_root']!r}+'/src')
from satellite_rl.workbench.storage import read_json,digest
from satellite_rl.workbench.runtime import pid_alive
from satellite_rl.workbench.coverage_ai_search import pin
directory=pathlib.Path({detail['path']!r});job=read_json(directory/'job.json',{{}});launch=read_json(directory/'launch.json',{{}});status=read_json(directory/'status.json',{{}})
if pid_alive(status.get('pid') or launch.get('pid'),launch.get('process_start_time')):raise ValueError('original common worker remains alive')
manifest=read_json(pin({reference!r}),{{}})
if manifest.get('completed')is not True or manifest['scientific_commit']!=job['scientific_commit'] or set(manifest['cases'])!={{str(i) for i in job['case_indices']}}:raise ValueError('canonical common manifest not closed')
for row in manifest['cases'].values():pin(row['common'])
source=pathlib.Path(job['source_root']);source_manifest=read_json(source/'source_manifest.json',{{}})
if digest(source/'source_manifest.json')!=job['source_manifest_sha256']:raise ValueError('original source manifest changed')
for relative,sha in source_manifest['files'].items():
 target=(source/relative).resolve()
 if not target.is_relative_to(source.resolve()) or digest(target)!=sha:raise ValueError('original science source member changed')
print(json.dumps({{'run_id':job['run_id'],'worker_exit_verified':True,'case_count':len(manifest['cases']),'source_commit':job['scientific_commit'],'common_manifest':{reference!r}}}))
'''
        actual=ops.remote_python(code,port,timeout=50)
        if actual['exit_code']:raise ValueError(actual['stderr'][-1000:])
        receipt=json.loads(actual['stdout'].splitlines()[-1]);refs[str(port)]=reference;receipts[str(port)]=receipt
    spec=json.loads(Path(state_path).with_name('spec.json').read_text(encoding='utf-8'))
    expected=sum(len(task['body']['case_indices']) for task in spec['tasks'])
    if set(refs)!={str(p) for p in ops.SSH_PORTS} or sum(r['case_count'] for r in receipts.values())!=expected:
        raise ValueError('three-lane exact plan common closure differs')
    atomic_json(output,refs);atomic_json(str(output)+'.receipt.json',{'group':ops.GROUP,'original_IDs_only':True,'science_calls':0,'lanes':receipts})
    return refs


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--state',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(collect(args.state,args.output)))
