"""Publish the exact root-reviewed source/plan; task POST belongs to controller."""
import argparse,json
from pathlib import Path
from satellite_rl.training.run_monitor import atomic_json
from . import ops,plan_builder,batch_metadata
from .deploy import git

def publish(plan_path,inputs_path,review_path,state_root):
 path=Path(plan_path);plan=json.loads(path.read_text('utf-8'));inputs=json.loads(Path(inputs_path).read_text('utf-8'));review=json.loads(Path(review_path).read_text('utf-8'));sha=plan_builder.digest(path);commit=plan['scientific_commit'];d=Path(__file__).with_name('round5_design.json')
 if (plan.get('group')!=ops.GROUP or plan.get('authorized')is not True or review.get('passed')is not True or review.get('group')!=ops.GROUP or review.get('scientific_commit')!=commit or review.get('plan_sha256')!=sha or review.get('design_sha256')!=plan['design']['sha256']):raise ValueError('exact root review required')
 if git('rev-parse','HEAD')!=commit or git('status','--porcelain','--','src','scripts','tests','pyproject.toml','.gitattributes'):raise ValueError('root-frozen clean source required')
 expanded=plan_builder.audit(plan,inputs['identities'])
 if plan['design']!={'path':ops.BASE+'/source/round5_design.json','sha256':plan_builder.digest(d)}:raise ValueError('committed protocol bytes/remote ref differ')
 uploads=[ops.upload(d,plan['design']['path'],20418),batch_metadata.upload(inputs['metadata'],ops.LOCAL/'publication_metadata',20418)]
 target=ops.BASE+'/source/plans/'+commit+'_'+sha+'.json';review_target=ops.BASE+'/source/admissions/'+plan_builder.digest(review_path)+'.json';uploads.extend([ops.upload(path,target,20418),ops.upload(review_path,review_target,20418)])
 root=Path(state_root).resolve();root.mkdir(parents=True,exist_ok=True)
 tasks=[{'port':t['executor_port'],'body':{**t,'task_key':key,'request_id':'ai17r5_'+commit[:12]+'_'+sha[:12]+'_'+key,'plan':{'path':target,'sha256':sha},'root_review':{'path':review_target,'sha256':plan_builder.digest(review_path)}}}for key,t in plan['tasks'].items()]
 spec={'group':ops.GROUP,'scientific_commit':commit,'automatic_extension':False,'tasks':tasks,'plan':{'path':target,'sha256':sha},'science_updates':0,'submission':'workbench_only_same_request_unknown_outcome'};specpath=root/'published_spec.json'
 if specpath.exists()and json.loads(specpath.read_text('utf-8'))!=spec:raise ValueError('another immutable spec owns state')
 atomic_json(specpath,spec);receipt={'group':ops.GROUP,'plan_sha256':sha,'expanded':expanded,'uploads':uploads,'spec_path':str(specpath),'spec_sha256':plan_builder.digest(specpath),'scientific_tasks_submitted':0};atomic_json(root/'publish_receipt.json',receipt);return receipt
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['plan','inputs','review','state-root']:p.add_argument('--'+name,required=True)
 a=p.parse_args();print(json.dumps(publish(a.plan,a.inputs,a.review,a.state_root)))
