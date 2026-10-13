"""Fresh20 development scenes; serial four-method cyclic cold comparison."""
import time,shutil
from pathlib import Path
from .storage import read_json,digest
from .coverage_ai_search import pin
from .r6_io import read_asset
from .r7_labels import guarded,supply_case,CORE_SHA
from .r7_model_io import signed_json,verify_model_source_equivalence
from .r7_node_audit import audit_row
from .r7_node_guard_audit import stage_proof,audit_preparation_teacher,audit_costs
from satellite_rl.training.run_monitor import atomic_json

def method_order(i):
    methods=['AI','RULE','RANDOM','T'];shift=i%4
    return methods[shift:]+methods[:shift]

def node(job,config,directory):
    from scripts.stage1_ai_search_r4.audit import audit_layout,audit_ledger
    directory=Path(directory);refs=job['model_inputs'];initial=signed_json(refs['initialization_receipt'])
    actual=verify_model_source_equivalence(refs['source_equivalence'],refs['initialization_receipt'],initial)
    provenance={k:actual[k] for k in ('model_graph_source_SHA','feature_graph_source_SHA','adapter_source_SHA','b7_decoder_source_SHA')}
    provenance.update(model_initial_state_SHA=initial['model_initial_state_SHA'],source_epoch=job['scientific_commit'])
    index=read_json(directory/'case_index.json',{});cases={}
    def stop():
        if shutil.disk_usage(directory).free<8589934592:raise OSError('8GiB reserve')
        return (directory/'stop.request').exists()
    for i in job['case_indices']:
        target=directory/'bundles'/str(i)
        if str(i) in index:
            old=read_json(pin(index[str(i)]),{})
            if old['source_commit']!=job['scientific_commit'] or old['checkpoint_ref']!=refs['checkpoint']:raise ValueError('original node source/checkpoint changed')
            for ref in old['artifact_refs']:pin(ref)
            cases[str(i)]=old;continue
        start=time.monotonic();record,scene,request,source=supply_case(job,config,i,target);input_seconds=time.monotonic()-start
        begin=time.monotonic();pg,prepared,pref=guarded(target/'preparation',{**request,'stage':'preparation'},20,stop);prepare_seconds=time.monotonic()-begin
        plain=dict(positions=scene.positions.tolist(),demands=scene.demands.tolist(),aircraft_ids=list(scene.aircraft_ids),profile=scene.profile)
        unavailable=prepared['result']['status']!='available'
        proofs={'preparation':stage_proof(target/'preparation')}
        cost_components=dict(input=input_seconds,preparation_parent=prepare_seconds+(pg['actual_stage_wall_seconds'] if pg.get('reused_completed_call') else 0.))
        methods={};results={m:None for m in ('AI','RULE','RANDOM','T')};costs={m:None for m in results};method_rows={};artifact_refs=[pref];teacher_closed=False;teacher_metrics=None
        if unavailable:
            missing=dict(valid=True,published=False,nonfallback=False,status='unavailable',metrics=None,identity_closed=True,budget_closed=True)
            methods={m:dict(missing) for m in ('AI','RULE','RANDOM')}
            method_rows={m:dict(status='unexecuted',reason='witness_unavailable',request=None,result_ref=None,guard_proof=None,audit=dict(missing)) for m in methods}
        else:
            for method in method_order(i):
                begin=time.monotonic()
                if method=='T':
                    tg,teacher,tref=guarded(target/'T',{**request,'stage':'teacher','prepared_ref':pref},45,stop)
                    proofs['teacher']=stage_proof(target/'T');artifact_refs.append(tref);results['T']=tref
                    if teacher['result']['status']=='available':
                        independent=audit_layout(plain,teacher['result']['layout'],prepared['result']['Ncap'],prepared['result']['Omega_cap'])
                        teacher_metrics=independent['metrics'] if independent['valid'] else None
                        teacher_closed=bool(independent['valid'] and audit_ledger(teacher['result']['cost'],len(scene.demands),'TEACHER')['valid'])
                    cost_components['T_parent']=time.monotonic()-begin+(tg['actual_stage_wall_seconds'] if tg.get('reused_completed_call') else 0.)
                    continue
                req=dict(method=method,identity=request['identity'],scene_ref=request['scene_ref'],prepared_ref=pref,
                         producer_sha256=job['source_manifest_sha256'],model_identity=job['model_identity'],b7_decoder_source_SHA=CORE_SHA)
                if method=='AI':req.update(checkpoint_ref=refs['checkpoint'],normalization_ref=refs['normalization'],AI_provenance=provenance,runtime_profile_SHA=initial['runtime_profile_SHA'])
                guard,value,result_ref=guarded(target/method,req,90 if method=='AI' else 15,stop,module='satellite_rl.workbench.r7_model_eval_stage',rss=4294967296 if method=='AI' else 2147483648)
                checked=audit_row(plain,prepared['result'],value,req);methods[method]=checked;results[method]=result_ref;artifact_refs.append(result_ref)
                cost_components[method+'_parent']=time.monotonic()-begin+(guard['actual_stage_wall_seconds'] if guard.get('reused_completed_call') else 0.)
                method_rows[method]=dict(request=req,result_ref=result_ref,guard_proof=stage_proof(target/method),audit=checked)
            costs={m:cost_components['input']+cost_components['preparation_parent']+cost_components[m+'_parent'] for m in costs}
        row=dict(case_index=i,identity=request['identity'],methods=methods,method_rows=method_rows,
                 result_refs=results,artifact_refs=artifact_refs,scene_ref=request['scene_ref'],prepared_ref=pref,
                 teacher_metrics=teacher_metrics,teacher_identity_and_cost_closed=teacher_closed,
                 costs=costs,cost_components=cost_components,total_request=int(record['total_request_bps_int64']),
                 source_commit=job['scientific_commit'],checkpoint_ref=refs['checkpoint'],model_identity=job['model_identity'],
                 status='preparation_unavailable' if unavailable else 'complete',stage_proofs=proofs,
                 core_sha256=CORE_SHA,producer_sha256=job['source_manifest_sha256'],source_dataset_sha256=config['datasets']['development']['sha256'],
                 method_order=method_order(i),new_calls=dict(prepare=1,T=0 if unavailable else 1,D5=0 if unavailable else 3),publication_seconds=0.)
        stage_results=audit_preparation_teacher(row)
        row['preparation_identity_and_cost_closed']=stage_results['preparation']['closed']
        row['teacher_identity_and_cost_closed']=bool(teacher_closed and not unavailable and stage_results['preparation']['closed'] and stage_results['teacher']['closed'])
        audit_costs(row,stage_results)
        publish=time.monotonic();path=target/'node_row.json';atomic_json(path,row);index[str(i)]=dict(path=str(path),sha256=digest(path));atomic_json(directory/'case_index.json',index)
        # Publication is paid once physically and included in all complete
        # method estimates, outside the immutable row through an endpoint.
        endpoint=target/'cost_endpoint.json';atomic_json(endpoint,dict(raw_ref=index[str(i)],identity=request['identity'],common_row_index_publication_seconds=time.monotonic()-publish,costs_before_publication=costs,requires_separate_isolated_epoch_for_strict_speed=True))
        index[str(i)]['cost_endpoint_ref']=dict(path=str(endpoint),sha256=digest(endpoint));atomic_json(directory/'case_index.json',index);cases[str(i)]=row
    manifest=dict(group=job['experiment_id'],phase='node2000',scientific_commit=job['scientific_commit'],cases=index,
                  checkpoint_ref=refs['checkpoint'],model_identity=job['model_identity'],new_calls={k:sum(read_json(pin(index[str(i)]),{})['new_calls'][k] for i in job['case_indices']) for k in ('prepare','T','D5')},future_dispatch_allowed=False)
    atomic_json(directory/'result_manifest.json',manifest);return manifest
