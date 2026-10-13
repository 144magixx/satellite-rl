"""Fresh same-ID worker loading original523 science and an additive audited parent adapter."""
import argparse,contextvars,json,os,sys,time
from pathlib import Path
from .contracts import RUN_ID,SCIENCE,CORE_SHA,atomic,digest,read,validate,process_identity,fresh_resource
from .compat import audit_baseline,preserve_unknown_cost

def install_node_hooks(contract,contract_ref,operations_ref,ops_directory):
    """Only parent audit/timing hooks; child algorithm modules remain original523."""
    from satellite_rl.workbench import r7_node,r7_node_audit
    from satellite_rl.workbench.storage import digest as original_digest
    original_row=r7_node.audit_row;legacy_proposal=r7_node_audit.audit_proposal
    original_cost=r7_node.audit_costs;original_guard=r7_node.guarded;original_supply=r7_node.supply_case;original_node=r7_node.node
    request_context=contextvars.ContextVar('R7StrictOriginalBaselineRequest')
    def checked_proposal(scene,proposal,identity,prepared,teacher,method,producer):
        request=request_context.get()
        if request['identity']!=identity or request['method']!=method or request['producer_sha256']!=producer:
            raise ValueError('baseline row/request identity before compatibility')
        return audit_baseline(legacy_proposal,scene,proposal,identity,prepared,method,producer,request['prepared_ref'])
    def checked_row(scene,prepared,value,request):
        if request['method']=='AI':return original_row(scene,prepared,value,request)
        if request['producer_sha256']!=contract['original_source_manifest_ref']['sha256'] or request.get('b7_decoder_source_SHA')!=CORE_SHA or prepared.get('source_sha256')!=CORE_SHA:
            raise ValueError('actual baseline original523 producer/unchanged decoder source')
        token=request_context.set(request)
        try:return original_row(scene,prepared,value,request)
        finally:request_context.reset(token)
    def durable_supply(job,config,i,target):
        began=time.monotonic();value=original_supply(job,config,i,target)
        atomic(Path(target)/'recovery_input_timer.json',dict(schema='R7RecoveryObservedParentTimingV1',
            operations_source_ref=operations_ref,contract_ref=contract_ref,case_index=i,
            observed_seconds_before_this_receipt_publication=time.monotonic()-began,
            original_lost_input_time_reconstructed=False,scientific_new_calls=0))
        return value
    def durable_guard(target,request,*args,**kwargs):
        began=time.monotonic();result=original_guard(target,request,*args,**kwargs)
        reused=result[0].get('reused_completed_call')is True
        atomic(Path(target)/'recovery_parent_timer.json',dict(schema='R7RecoveryObservedParentTimingV1',
            contract_ref=contract_ref,operations_source_ref=operations_ref,request_ref=dict(path=str(Path(target)/'request.json'),sha256=original_digest(Path(target)/'request.json')),
            complete_call_returned=True,reused_original_call=reused,new_scientific_calls=0 if reused else 1,
            observed_seconds_before_this_receipt_publication=time.monotonic()-began,
            original_complete_parent_seconds=None if reused else 'enclosing original-node caller timer retained in node_row',
            receipt_IO_is_additional_recovery_overhead=True))
        return result
    def cost(row,stage_results):
        if row['case_index']:
            result=original_cost(row,stage_results)
            row['operations_recovery_ref']=dict(contract_ref=contract_ref,operations_source_ref=operations_ref)
            return result
        from satellite_rl.workbench.r7_node_guard_audit import audit_method_proof
        if not all(value.get('closed')is True for value in stage_results.values()):raise ValueError('case0 original/new W/T guard closure')
        for method in ('AI','RULE','RANDOM'):
            if audit_method_proof(method,row['method_rows'][method],row).get('closed')is not True:raise ValueError('case0 full original method/WAL/D5 guard closure')
        return preserve_unknown_cost(row,contract_ref,operations_ref,contract['original_evidence_refs'])
    def node_with_additive_call_inventory(job,config,directory):
        result=original_node(job,config,directory)
        observed=dict(prepare=0,T=0,D5=0);reused=[]
        for i in job['case_indices']:
            raw_row=json.loads(Path(result['cases'][str(i)]['path']).read_bytes())
            for stage,key in [('preparation','prepare'),('T','T'),('AI','D5'),('RULE','D5'),('RANDOM','D5')]:
                timer=Path(directory)/'bundles'/str(i)/stage/'recovery_parent_timer.json'
                if not timer.exists():
                    if stage!='preparation' and raw_row['status']=='preparation_unavailable':continue
                    raise ValueError('attempted node stage has no durable recovery timer')
                item=json.loads(timer.read_bytes());observed[key]+=item['new_scientific_calls']
                if item['reused_original_call']:reused.append([i,stage])
        spent=dict(prepare=1,T=0,D5=2)
        if reused!=[[0,'preparation'],[0,'AI'],[0,'RULE']] or any(observed[key]>contract['remaining_calls_max'][key] for key in observed):raise ValueError('original reused slots/new epoch science count differs')
        if result['new_calls']!={key:spent[key]+observed[key] for key in spent}:raise ValueError('unchanged frozen manifest logical whole-node totals differ')
        atomic(ops_directory/'recovery_call_accounting.json',dict(schema='R7Node2000RecoveryScientificCallAccountingV1',
            contract_ref=contract_ref,operations_source_ref=operations_ref,original_spent_calls=dict(prepare=1,T=0,D5=2),
            recovery_new_calls=observed,whole_node_calls=result['new_calls'],original_manifest_new_calls_field_means_logical_whole_phase=True,
            reused_slots=reused,model_updates=0,original_case0_complete_cold_costs=None,strict_speed_claims=False))
        return result
    r7_node_audit.audit_proposal=checked_proposal
    r7_node.audit_row=checked_row;r7_node.guarded=durable_guard;r7_node.supply_case=durable_supply;r7_node.audit_costs=cost;r7_node.node=node_with_additive_call_inventory
    return dict(schema='R7NodeRecoveryActualParentHookReceiptV1',strict8_before_legacy6=True,
        AI_audit_path_unchanged=True,original_algorithm_children_unchanged=True,
        timers_written_before_fallible_row_audit=True,case0_full_cold_cost_remains_null=True,
        source_commit=SCIENCE,operations_source_ref=operations_ref,contract_ref=contract_ref)

def main():
    p=argparse.ArgumentParser();p.add_argument('--contract',required=True);p.add_argument('--contract-sha256',required=True);p.add_argument('--root-review',required=True);p.add_argument('--root-review-sha256',required=True);a=p.parse_args()
    contract_ref=dict(path=a.contract,sha256=a.contract_sha256);review_ref=dict(path=a.root_review,sha256=a.root_review_sha256)
    early=read(contract_ref);handoff=Path(early['operations_directory'])/'parent_locks_released.json';until=time.monotonic()+30
    while not handoff.exists():
        if time.monotonic()>=until:raise RuntimeError('parent ownership handoff unknown; no science or automatic retry')
        time.sleep(.05)
    transferred=json.loads(handoff.read_bytes())
    current=process_identity(os.getpid());expected=transferred.get('actual_worker') or {}
    if transferred.get('locks_released')is not True or transferred.get('contract_ref')!=contract_ref or current is None or any(expected.get(key)!=current[key] for key in ('pid','birth_ticks','birth_kind','boot_id')):raise ValueError('exact fresh worker PID/birth ownership handoff required')
    contract,job,config=validate(contract_ref,review_ref);fresh_resource(contract)
    directory=Path(contract['original_job_ref']['path']).parent;ops=Path(contract['operations_directory']);operations_ref=contract['operations_source_files']['bootstrap.py']
    # Oldjob/source remain byte identical. Original science must win import resolution.
    source=Path(job['source_root']);sys.path[:0]=[str(source/'src'),str(source)]
    from satellite_rl.workbench import r7_runner,r7_node
    from satellite_rl.workbench.r7_deadline import stop_boundary
    stop_boundary(job,directory)
    receipt=install_node_hooks(contract,contract_ref,operations_ref,ops)
    receipt.update(actual_worker=process_identity(os.getpid()),actual_scientific_node_module=str(Path(r7_node.__file__).resolve()),original_job_ref=contract['original_job_ref'],original_config_ref=contract['original_config_ref'])
    if Path(r7_node.__file__).resolve()!=source/'src/satellite_rl/workbench/r7_node.py':raise ValueError('actual science import must originate in original frozen523')
    atomic(ops/'actual_worker_hooks.json',receipt)
    original_atomic=r7_runner.atomic_json
    protected={'failure.json','runtime_v3_identity.json','worker_deadline_preflight.json'}
    def preserve_original_epoch(path,value):
        path=Path(path)
        if path.parent==directory and path.name in protected:
            if path.name=='runtime_v3_identity.json' and json.loads(path.read_bytes())!=value:raise ValueError('original runtime identity changed')
            return original_atomic(ops/('recovery_'+path.name),value)
        return original_atomic(path,value)
    r7_runner.atomic_json=preserve_original_epoch
    began=time.monotonic();worker_identity=process_identity(os.getpid());ticks=os.sysconf('SC_CLK_TCK');birth_lower=(worker_identity['birth_ticks']-1)/ticks
    atomic(ops/'worker_epoch_started.json',dict(actual_worker=worker_identity,monotonic_started=began,worker_birth_lower_CLOCK_BOOTTIME_seconds=birth_lower,SC_CLK_TCK=ticks,contract_ref=contract_ref,original_job_unchanged=True))
    try:r7_runner.run_job(directory,resume=False)
    finally:
        atomic(ops/'worker_epoch_exit.json',dict(observed_parent_epoch_seconds_after_hooks=time.monotonic()-began,worker_enclosing_spend_upper_bound_seconds=time.clock_gettime(time.CLOCK_BOOTTIME)-birth_lower,
            enclosing_bound_includes_module_startup_and_hook_preflight=True,upper_bound_is_not_exact_method_latency=True,original_failure_epoch_seconds=None,
            additive_new_epoch_only=True,contract_ref=contract_ref,original_job_sha256=digest(directory/'job.json'),original_config_sha256=digest(directory/'config.json')))
    if digest(directory/'job.json')!=contract['original_job_ref']['sha256'] or digest(directory/'config.json')!=contract['original_config_ref']['sha256']:raise ValueError('original job/config bytes mutated')

if __name__=='__main__':main()
