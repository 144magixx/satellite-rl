"""W28 independent, fail-closed plan admission. No server or implicit dispatch.

The runtime starts only through the root-reviewed deployment adapter; this module
can validate finite metadata plans and bind each injected episode before then.
"""
from pathlib import Path
from satellite_rl.t0_width.data import file_sha,read_json_ref
from satellite_rl.t0_width.logging import content_sha
from .plan import matrix
from .proposals import SEED_CONTRACT

def validate_frozen_source(plan):
    source=Path(plan['source_root']).resolve()
    if not Path(__file__).resolve().is_relative_to(source):raise ValueError('runtime import outside frozen source')
    for rel,sha in plan['source_manifest'].items():
        path=(source/rel).resolve()
        if not path.is_relative_to(source) or file_sha(path)!=sha:raise ValueError('immutable source member differs')
    needed={p.relative_to(source).as_posix() for folder in ('src','scripts') for p in (source/folder).rglob('*.py')}
    if not needed.issubset(plan['source_manifest']):raise ValueError('complete Python closure required')

def validate_plan(plan,metadata,gate,*,runtime=False,dispatch_epoch=None):
    if (plan.get('schema')!='W28FinitePlanV1' or plan.get('experiment_id')!='S2-W28'
            or plan.get('root_approved') is not True or plan.get('max_workers')!=1
            or plan.get('new_training')!=0 or plan.get('new_parameter_updates')!=0
            or plan.get('new_data')!=0 or plan.get('S3_calls')!=0
            or plan.get('proposal_seed_contract')!=SEED_CONTRACT):
        raise ValueError('root exact zero-training W28 plan required')
    actual=matrix(metadata,plan['phase'])
    if plan.get('matrix')!=actual:raise ValueError('exact N/scene/method finite budgets differ')
    gate.validate(dispatch_epoch or plan['rootpauseepoch'])
    closure=read_json_ref(plan['w27_closure_ref'])
    if closure.get('root_approved') is not True or closure.get('formal_report_closed') is not True:
        raise ValueError('W27 formal closure required before W28 runtime')
    if plan['phase']=='known':
        decision=read_json_ref(plan['development_decision_ref'])
        if (decision.get('schema')!='W28DevelopmentDecisionV1' or decision.get('root_approved') is not True
                or decision.get('experiment_id')!='S2-W28' or decision.get('completed_development_episodes')!=240
                or decision.get('passed') is not True or decision.get('known_episodes')!=560
                or decision.get('development_plan_SHA')!=plan.get('development_plan_SHA')
                or decision.get('policy_SHA')!=plan['policy_ref']['sha256']
                or decision.get('source_commit')!=plan['source_commit']
                or decision.get('development_manifest_SHA')!=plan['development_manifest_SHA']):
            raise ValueError('locked full development gate required')
    if runtime:
        validate_frozen_source(plan)
        qualification=read_json_ref(plan['qualification_ref'])
        if (qualification.get('root_approved') is not True or qualification.get('passed') is not True
                or qualification.get('physical_calls_attempted',321)>320
                or qualification.get('real_commit_attempts',65)>64
                or qualification.get('candidate_restore_attempts',257)>256
                or qualification.get('model_forward_attempts',65)>64
                or qualification.get('directory_fsync_available') is not True
                or qualification.get('parameter_updates')!=0):raise ValueError('bounded engineering qualification required')
        policy=plan['policy_ref']
        if file_sha(policy['path'])!=policy['sha256'] or plan['policy_node']!=150000 or plan['policy_group']!='S10':raise ValueError('exact W26 S10/150k policy required')
    return dict(plan_SHA=content_sha(plan),matrix=actual,validated=True,runtime=runtime)

def episode_identity(plan,index):
    job=plan['matrix']['jobs'][index]
    return dict(job,experiment_id='S2-W28',plan_SHA=content_sha(plan),matrix_cursor=index,
        source_commit=plan['source_commit'],policy_SHA=plan['policy_ref']['sha256'],data_SHA=plan['data_manifest_ref']['sha256'])
