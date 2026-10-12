"""Publish a root-reviewed finite R2 plan and source audits, without POST."""
import argparse,json
from pathlib import Path
from satellite_rl.training.run_monitor import atomic_json
from . import ops,plan_builder
from .deploy import git


def publish(plan_path,inventory_path,review_path,state_root,qualification_path=None):
    plan_path=Path(plan_path).resolve();inventory=json.loads(Path(inventory_path).read_text(encoding='utf-8'))
    plan=json.loads(plan_path.read_text(encoding='utf-8'));review_path=Path(review_path).resolve()
    review=json.loads(review_path.read_text(encoding='utf-8'));sha=plan_builder.digest(plan_path);commit=plan['scientific_commit']
    if (plan.get('authorized') is not True or plan.get('group')!=ops.GROUP or review.get('passed') is not True
            or review.get('group')!=ops.GROUP or review.get('scientific_commit')!=commit or review.get('plan_sha256')!=sha
            or review.get('design_sha256')!=plan['design']['sha256']):
        raise ValueError('exact root-reviewed plan/source publication receipt required')
    if git('rev-parse','HEAD')!=commit or git('status','--porcelain','--','src','scripts','tests','pyproject.toml'):
        raise ValueError('publish requires clean frozen R2 scientific source')
    expanded=plan_builder.audit(plan,inventory)
    if plan.get('phase')!='pilot' or plan.get('mother_order')!=inventory['mother_order'] or len(plan['mother_order'])!=20:
        raise ValueError('exact full20-mother UTC order and pilot phase required')
    if any(t['mode']=='evaluate_closed_pool' for t in plan['tasks'].values()) and set(plan.get('qualification_refs',{}))!={'validator_equivalence','charge_before_budget'}:
        raise ValueError('closed-pool publication requires both source-bound engineering qualification refs')
    design=Path(__file__).with_name('round2_design.json')
    if plan['design']!={'path':ops.BASE+'/source/round2_design.json','sha256':plan_builder.digest(design)}:
        raise ValueError('design remote path/committed bytes differ')
    receipts=[ops.upload(design,plan['design']['path'],20418)]
    if plan.get('qualification_refs'):
        available=json.loads(Path(qualification_path).read_text(encoding='utf-8')) if qualification_path else {}
        for key,reference in plan['qualification_refs'].items():
            local=Path(available[key]['local_path'])
            if {k:available[key][k] for k in ('path','sha256')}!=reference or plan_builder.digest(local)!=reference['sha256']:
                raise ValueError('actual engineering qualification bytes/path differ')
            receipts.append(ops.upload(local,reference['path'],20418))
    if {t['mode'] for t in plan['tasks'].values()}=={'prepare_common'}:
        required={(r['audit_path'],r['audit_sha256']) for t in plan['tasks'].values() for refs in t['source_refs'].values() for r in refs}
        available={(r['audit_path'],r['audit_sha256']):r['audit_local_canonical'] for refs in inventory['source_refs'].values() for r in refs}
        for key in sorted(required):
            local=Path(available[key])
            if plan_builder.digest(local)!=key[1]:raise ValueError('original accepted case audit bytes changed')
            receipts.append(ops.upload(local,key[0],20418))
    target=ops.BASE+'/source/plans/'+commit+'_'+sha+'.json'
    receipts.append(ops.upload(plan_path,target,20418))
    review_target=ops.BASE+'/source/admissions/'+plan_builder.digest(review_path)+'.json'
    receipts.append(ops.upload(review_path,review_target,20418))
    root=Path(state_root).resolve();root.mkdir(parents=True,exist_ok=True)
    tasks=[{'port':t['executor_port'],'body':{**t,'task_key':key,
        'request_id':'ai17r2_'+commit[:12]+'_'+sha[:12]+'_'+key,'plan':{'path':target,'sha256':sha},
        'root_review':{'path':review_target,'sha256':plan_builder.digest(review_path)}}}
        for key,t in plan['tasks'].items()]
    spec={'group':ops.GROUP,'scientific_commit':commit,'automatic_extension':False,'tasks':tasks,
        'plan':{'path':target,'sha256':sha},'root_review':{'path':str(review_path),'sha256':plan_builder.digest(review_path)},
        'training_updates':0,'validation_test_algorithms':0,'submission':'workbench_only_same_request_id_unknown_outcome'}
    spec_path=root/'published_spec.json'
    if spec_path.exists() and json.loads(spec_path.read_text(encoding='utf-8'))!=spec:raise ValueError('state root already owns another immutable spec')
    atomic_json(spec_path,spec)
    receipt={'group':ops.GROUP,'plan_sha256':sha,'expanded_audit':expanded,'remote_plan':target,
        'uploads':receipts,'spec_path':str(spec_path),'spec_sha256':plan_builder.digest(spec_path),'science_jobs_submitted':0}
    atomic_json(root/'publish_receipt.json',receipt);return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','inventory','review','state-root'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--qualification-refs')
    args=parser.parse_args();print(json.dumps(publish(args.plan,args.inventory,args.review,args.state_root,args.qualification_refs)))
