"""Publish only frozen small source metadata/plan; create a stable controller spec."""
import argparse
import hashlib
import json
from pathlib import Path
from satellite_rl.training.run_monitor import atomic_json
from . import ops


def publish(plan_path,state_root,port=20418):
    plan_path=Path(plan_path).resolve();plan=json.loads(plan_path.read_text(encoding='utf-8'))
    if plan.get('group')!='S1-AI17' or plan.get('authorized') is not True or plan.get('status')=='uncommitted_candidate_not_for_submission':
        raise ValueError('publish requires root-reviewed frozen authorized source plan')
    commit=plan['scientific_commit']
    from .deploy import git
    if git('rev-parse','HEAD')!=commit or git('status','--porcelain','--','src','scripts','tests'):
        raise ValueError('frozen plan must match clean committed source')
    source=ops.LOCAL/'source';files=[source/'PREFERRED_RELEASES.txt',source/'LICENSE-ODbL.txt',source/'preparation_reference.py']
    files.extend(source/date/name for date in ('2024-02-17','2024-02-18','2024-02-19','2024-02-20')
                 for name in ('release_api.json','source_preflight.json'))
    design=Path(__file__).resolve().parent/'round1_design.json'
    receipts=[]
    for file in files:
        if file.is_file():receipts.append(ops.upload(file,ops.BASE+'/source/'+file.relative_to(source).as_posix(),port))
    if hashlib.sha256(design.read_bytes()).hexdigest()!=plan['design']['sha256']:
        raise ValueError('actual committed machine design differs from plan')
    receipts.append(ops.upload(design,ops.BASE+'/source/round1_design.json',port))
    sha=hashlib.sha256(plan_path.read_bytes()).hexdigest()
    target=ops.BASE+'/source/plans/'+commit+'_'+sha+'.json';receipts.append(ops.upload(plan_path,target,port))
    tasks=[]
    for key,contract in plan['tasks'].items():
        request='ai17_'+plan.get('round','round')+'_'+commit[:12]+'_'+key
        tasks.append({'port':contract['executor_port'],'body':{**contract,'task_key':key,'request_id':request,
                     'plan':{'path':target,'sha256':sha}}})
    spec={'group':'S1-AI17','scientific_commit':commit,'tasks':tasks,'resume_interrupted':True,
          'plan':{'path':target,'sha256':sha},'science_submission_policy':'workbench_only_original_request_id'}
    spec['watch_training_requests']=[item['body']['request_id'] for item in tasks if item['body']['mode']=='train']
    state_root=Path(state_root).resolve();state_root.mkdir(parents=True,exist_ok=True)
    spec_path=state_root/'published_spec.json';atomic_json(spec_path,spec)
    receipt={'plan_sha256':sha,'remote_plan':target,'source_uploads':receipts,'spec_path':str(spec_path),
             'science_jobs_submitted':0}
    atomic_json(state_root/'publish_receipt.json',receipt)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True);parser.add_argument('--state-root',required=True)
    args=parser.parse_args();print(json.dumps(publish(args.plan,args.state_root),ensure_ascii=False))
