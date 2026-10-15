"""Finite root-frozen engineering checks; never invoked by import or tests."""
from pathlib import Path
import pickle
from time import perf_counter
from satellite_rl.t0_width.data import read_json_ref,file_sha
from satellite_rl.t0_width.logging import atomic_json,content_sha
from .runtime import save_pickle
from .loading import load_policy,load_environment
from .transaction import Journal,UnknownCost,close_transaction,snapshot,state_sha,restore_memory,fsync_directory
from .worker import one_decision,candidate_log
from .episode import publish_committed
from .proposals import frozen_s10_probabilities
from .binding import pack_environment,unpack_environment

def validate_engineering(plan,gate,dispatch_epoch=None):
    if (plan.get('schema')!='W28EngineeringPlanV1' or plan.get('root_approved') is not True
            or plan.get('runtime_approved') is not True or plan.get('experiment_id')!='S2-W28'
            or plan.get('parameter_updates')!=0 or not 1<=len(plan.get('checks',[]))<=64
            or plan.get('limits')!=dict(query=256,commit=64,forward=64,proposal=64)):
        raise ValueError('bounded root W28 engineering contract required')
    from .workbench import validate_frozen_source
    validate_frozen_source(plan)
    if plan.get('check_contract_version')=='W28QualificationChecksV1':
        from .checks import validate_catalog
        bounds=validate_catalog(plan['checks'],read_json_ref(plan['fixture_manifest_ref']))
        if bounds!=plan['catalog_bounds']:raise ValueError('catalog finite budget differs')
        from .migration import validate_source_requalification
        validate_source_requalification(plan)
    gate.validate(dispatch_epoch or plan['rootpauseepoch'])
    closure=read_json_ref(plan['w27_closure_ref'])
    if closure.get('root_approved') is not True or closure.get('formal_report_closed') is not True:raise ValueError('W27 report must close before engineering')
    for index,check in enumerate(plan['checks']):
        if check.get('method') not in ('P1','P4','R1','R4') or check.get('check_index')!=index:raise ValueError('exact ordered engineering check list required')

def run_engineering(plan,root,stop_requested,progress=lambda value:None,*,policy_loader=load_policy,environment_loader=load_environment):
    if plan.get('runtime_approved') is not True:raise ValueError('draft engineering runtime forbidden')
    if plan.get('check_contract_version')=='W28QualificationChecksV1':
        from .qualification import run_checks
        return run_checks(plan,root,stop_requested,progress)
    root=Path(root);root.mkdir(parents=True,exist_ok=True);path=root/'engineering_cursor.pkl'
    if path.with_suffix('.tmp').exists():raise UnknownCost('unconfirmed engineering cursor write; preserve and audit')
    if not path.exists() and ((root/'current.pkl').exists() or (root/'engineering_steps.jsonl').exists()):raise UnknownCost('engineering cursor missing beside prior attempts')
    fsync_directory(root)
    cursor=pickle.loads(path.read_bytes()) if path.exists() else dict(plan_SHA=content_sha(plan),cursor=0,bindings={},unknown=[],timing=[])
    started=perf_counter();elapsed_before=cursor.get('elapsed_active_seconds',0.)
    def checkpoint():
        cursor['elapsed_active_seconds']=elapsed_before+perf_counter()-started;save_pickle(path,cursor)
    if cursor['plan_SHA']!=content_sha(plan):raise ValueError('immutable engineering plan differs')
    if cursor['unknown']:raise UnknownCost('unknown engineering setup must be accounted, never replayed')
    rows=read_json_ref(plan['data_manifest_ref'])['rows'];journal=Journal(root/'current.pkl',dict(plan_SHA=content_sha(plan),task_id=plan['run_id']),plan['limits'])
    agent=None
    for index in range(cursor['cursor'],len(plan['checks'])):
        if stop_requested():
            checkpoint();return dict(stage='safe_stopped',checks=cursor['cursor'],costs=journal.value['attempts'])
        check=plan['checks'][index];case_id=check['case_id'];binding=root/('binding_'+str(check['mother'])+'.pkl')
        row=rows[check['mother']]
        if row['case_id']!=case_id:raise ValueError('engineering mother mapping differs')
        if case_id not in cursor['bindings']:
            cursor['unknown']=[dict(kind='initial_kernel_preparation',check=index)];checkpoint()
            env,timing=environment_loader(dict(plan,phase='development'),row,check['N']);observation=env._observation()
            save_pickle(binding,dict(environment=pack_environment(env),observation=observation))
            cursor['bindings'][case_id]=dict(path=str(binding),sha256=file_sha(binding));cursor['timing'].append(timing);cursor['unknown']=[];checkpoint()
        else:
            ref=cursor['bindings'][case_id]
            if file_sha(ref['path'])!=ref['sha256']:raise ValueError('engineering saved kernel/input binding differs')
            saved=pickle.loads(Path(ref['path']).read_bytes());env=unpack_environment(saved['environment']);observation=saved['observation']
        tx=journal.value['transaction']
        resumed_commit=False
        if tx and tx['context'].get('repeat')==index and tx['context'].get('case_id')==case_id:
            if tx.get('pending'):raise UnknownCost('engineering query/commit unknown; no retry')
            restore_memory(env,tx['committed_state'] if tx['stage'] in ('committed','closed') else tx['base'])
            if tx['stage'] in ('committed','closed'):
                resumed_commit=True;result=tx['commit_result']
                if tx['stage']=='committed':publish_committed(journal,root/'engineering_steps.jsonl')
        elif env.beam_idx!=check['step']:raise ValueError('engineering cannot skip/replay prefix actions')
        if check['method'].startswith('P') and agent is None:
            cursor['unknown']=[dict(kind='actor_load',physical_calls=0)];checkpoint()
            agent,timing=policy_loader(plan);cursor['timing'].append(timing);cursor['unknown']=[];checkpoint()
        # one_decision's context is exact method/case/repeat/step; engineering
        # uses repeat=index, so no check can consume another check's proposals.
        if not resumed_commit:
            result=one_decision(env,observation,check['method'],case_id,index,journal,
                (lambda obs,mask:frozen_s10_probabilities(agent,obs,mask)) if agent else None,stop_requested)
            if result is None:
                checkpoint();return dict(stage='safe_stopped',checks=index,costs=journal.value['attempts'])
            publish_committed(journal,root/'engineering_steps.jsonl')
        # Save next prefix without reconstructing/replaying any PHY. Cursor and
        # binding must close atomically by SHA; unknown replacement is blocked.
        nextbinding=root/('binding_'+str(check['mother'])+'_'+str(index)+'.pkl')
        if nextbinding.exists():
            candidate=pickle.loads(nextbinding.read_bytes())
            if state_sha(snapshot(unpack_environment(candidate['environment'])))!=state_sha(snapshot(env)):raise UnknownCost('orphan engineering binding differs; preserve and audit')
        else:save_pickle(nextbinding,dict(environment=pack_environment(env),observation=result[0]))
        previous_binding=Path(cursor['bindings'][case_id]['path'])
        cursor['bindings'][case_id]=dict(path=str(nextbinding),sha256=file_sha(nextbinding));cursor['cursor']=index+1;checkpoint()
        if previous_binding!=nextbinding and previous_binding.resolve().is_relative_to(root.resolve()) and previous_binding.exists():previous_binding.unlink();fsync_directory(root)
        progress(dict(stage='engineering',checks=index+1,costs=journal.value['attempts']))
    costs=journal.value['attempts']
    result=dict(schema='W28EngineeringReceiptV1',stage='completed',root_approved=False,passed=False,
        exact_step_transaction_checks=True,checks=cursor['cursor'],real_commit_attempts=costs['commit'],
        candidate_restore_attempts=costs['query'],physical_calls_attempted=costs['commit']+costs['query'],
        model_forward_attempts=costs['forward'],parameter_updates=0,
        directory_fsync_available=__import__('os').name=='posix',kernel_preparation_calls=len(cursor['bindings']),restore_PHY_calls=0,
        remaining_root_qualification=['N1/N200 real engineering fixtures','original probability law comparison','all interruption points','source/resource/watchdog independent audit'])
    atomic_json(root/'qualification_candidate.json',result)
    fsync_directory(root)
    checkpoint()
    return result
