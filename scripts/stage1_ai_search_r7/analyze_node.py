"""Independent fixed2000/node20 gate; no inference, search, or tensor updates."""
from pathlib import Path
import math,time
import numpy as np
from satellite_rl.workbench.coverage_ai_search import pin
from satellite_rl.workbench.storage import read_json,digest
from satellite_rl.workbench.r6_io import read_asset,pin_tree
from satellite_rl.workbench.r7_uniform_data import load_algorithm_case
from satellite_rl.workbench.r7_node_audit import audit_row,node_gate
from satellite_rl.training.run_monitor import atomic_json
from scripts.stage1_ai_search_r4.audit import audit_layout,audit_ledger
from .contract import GROUP
from satellite_rl.workbench.r7_node_guard_audit import audit_preparation_teacher,audit_method_proof,audit_costs
from satellite_rl.workbench.r7_case_binding import validate_binding
from satellite_rl.workbench.r7_publish import publish_terminal
from satellite_rl.workbench.r7_postprocess_source import execution_source,new_output

def analyze(manifest_refs,dataset_ref,checkpoint_ref,model_identity,output,phase='node2000',heldout_data_gate_ref=None,statistics_ref=None,training_refs=None):
    output=new_output(output);postsource=execution_source()
    dataset=read_json(pin(dataset_ref),{});cases={};commit=None
    from satellite_rl.workbench.r7_successor_binding import EVALUATIONS,evaluation_binding
    if phase not in EVALUATIONS:raise ValueError('original fixed model evaluation phase')
    role,count,updates=EVALUATIONS[phase]
    if phase=='test':
        if not statistics_ref:raise ValueError('pre-frozen statistics required before test case reads')
        statistics=read_json(pin(statistics_ref),{})
        from . import test_statistics
        if statistics.get('statistical_source_sha256')!=digest(test_statistics.__file__) or statistics.get('matrix_generated_once_before_test_reads')is not True:raise ValueError('actual frozen pre-test statistical source')
    if phase!='node2000' and not training_refs:raise ValueError('exact fixed6000 model endpoint/setup/normalization refs')
    if training_refs:
        if training_refs['checkpoint']!=checkpoint_ref:raise ValueError('analyzer fixed checkpoint differs')
        evaluation_binding(phase,training_refs,model_identity)
    if dataset.get('schema')!='R7UniformDatasetV1' or dataset.get('role')!=role or dataset.get('count')!=count:raise ValueError('root frozen true-role20/40 dataset')
    for ref in manifest_refs:
        manifest=read_json(pin(ref),{})
        if manifest.get('group')!=GROUP or manifest.get('phase')!=phase or manifest.get('checkpoint_ref')!=checkpoint_ref or manifest.get('model_identity')!=model_identity or manifest.get('checkpoint_updates')!=updates or manifest.get('role')!=role:raise ValueError('exact fixed model/source/role chain')
        if training_refs and any(manifest.get(k+'_ref')!=training_refs[k] for k in ('training_endpoint','train_setup_endpoint','normalization')):raise ValueError('same actual fixed training endpoint/setup/norm')
        if phase=='test' and manifest.get('test_statistical_plan_ref')!=statistics_ref:raise ValueError('original actual test statistical freeze')
        if commit and commit!=manifest['scientific_commit']:raise ValueError('two source commits')
        commit=manifest['scientific_commit']
        for i,row in manifest['cases'].items():
            if i in cases:raise ValueError('duplicate host owner')
            cases[i]=row
    if sorted(map(int,cases))!=list(range(count)):raise ValueError('exact original20/40 denominator; no missing cases')
    independent=[];typed_layouts={};typed_directory=Path(output).with_name(Path(output).stem+'_typed_layouts')
    for i in range(count):
        ref=cases[str(i)];row=read_json(pin(ref),{});ident=row['identity'];source=dataset['cases'][i]
        common_audit_started=time.monotonic()
        record,scene=load_algorithm_case(source['case_ref'],dataset['authority_ref'],heldout_data_gate_ref,role)
        if row['case_index']!=i or row['source_commit']!=commit or row['model_identity']!=model_identity or row['checkpoint_ref']!=checkpoint_ref or ident['shared_case_content_sha256']!=record['content_sha256'] or scene.input_sha256!=ident['input_sha256']:raise ValueError('independent root scene identity')
        plain=dict(positions=scene.positions.tolist(),demands=scene.demands.tolist(),aircraft_ids=list(scene.aircraft_ids),profile=scene.profile)
        validate_binding(record,role,i,source)
        if ident['role']!=role or row['source_dataset_sha256']!=dataset_ref['sha256'] or row['core_sha256']!='9f187e4cd8122a6fce415b590d0ab0a385724b7d840cc88dd874e34ea936aea5':raise ValueError('original true role/source/core binding')
        prepared=read_asset(row['prepared_ref'])['result'];stage_results=audit_preparation_teacher(row)
        if prepared['identity']!=ident or row.get('preparation_identity_and_cost_closed')is not stage_results['preparation']['closed'] or not audit_ledger(prepared['ledger'],len(scene.demands),'prepare')['valid']:raise ValueError('independent original prepare identity/ledger/closure')
        costs=audit_costs(row,stage_results);unavailable=row['status']=='preparation_unavailable'
        methods={};teacher_closed=False;publish_candidates={};method_audit_seconds={m:0. for m in ('AI','RULE','RANDOM','T')}
        if unavailable:
            if prepared['status']=='available' or set(row['result_refs'])!=set(('AI','RULE','RANDOM','T')) or any(v is not None for v in row['result_refs'].values()) or row['new_calls']!=dict(prepare=1,T=0,D5=0):raise ValueError('unavailable node prep must not spend T or D5')
            missing=dict(valid=True,published=False,nonfallback=False,status='unavailable',metrics=None,identity_closed=True,budget_closed=True)
            methods={m:dict(missing) for m in ('AI','RULE','RANDOM')}
            if row['methods']!=methods or any(row['method_rows'][m].get('status')!='unexecuted' or row['method_rows'][m].get('result_ref')is not None or row['method_rows'][m].get('guard_proof')is not None for m in methods):raise ValueError('typed unavailable methods and full20 denominator')
        else:
            if prepared['status']!='available' or row['new_calls']!=dict(prepare=1,T=1,D5=3):raise ValueError('available node exact4 method call counts')
            teacher=read_asset(row['result_refs']['T'])['result']
            if any(teacher.get(k)!=v for k,v in ident.items()):raise ValueError('original T identity')
            teacher_audit_started=time.monotonic()
            teacher_audit=audit_layout(plain,teacher['layout'],prepared['Ncap'],prepared['Omega_cap']) if teacher['status']=='available' else None
            teacher_closed=bool(stage_results['preparation']['closed'] and stage_results['teacher']['closed'] and teacher_audit and teacher_audit['valid'] and audit_ledger(teacher['cost'],len(scene.demands),'TEACHER')['valid'])
            method_audit_seconds['T']=time.monotonic()-teacher_audit_started
            if teacher_closed:
                from scripts.stage1_ai_search_r4.audit import canonical_sha
                publish_candidates['T']=(teacher['layout'],{**teacher_audit,'parent_case_SHA':record['content_sha256'],'layout_sha256':canonical_sha(teacher['layout'])},row['result_refs']['T'])
            for method in ('AI','RULE','RANDOM'):
                method_audit_started=time.monotonic()
                actual=row['method_rows'][method];req=actual['request'];guard_closure=audit_method_proof(method,actual,row)
                if method=='AI' and (req['checkpoint_ref']!=checkpoint_ref or req.get('checkpoint_updates')!=updates or req['model_identity']!=model_identity):raise ValueError('actual AI fixed update/checkpoint/source')
                if row['cost_components'][method+'_parent']<guard_closure['outer_wall_seconds']:raise ValueError('actual outer cold method wall not fully paid')
                actual_value=read_asset(actual['result_ref']);checked=audit_row(plain,prepared,actual_value,req)
                if checked!=actual['audit'] or checked!=row['methods'][method]:raise ValueError('producer metrics differ from independent raw audit')
                methods[method]={**checked,'budget_closed':bool(checked['budget_closed'] and guard_closure['closed'])}
                method_audit_seconds[method]=time.monotonic()-method_audit_started
                if checked['published'] and methods[method]['budget_closed'] and checked['identity_closed']:
                    publish_candidates[method]=(actual_value['result']['layout'],checked['geometry_audit'],actual['result_ref'])
        if row.get('teacher_identity_and_cost_closed')is not teacher_closed:raise ValueError('producer teacher closure differs from original guarded source')
        endpoint=read_json(pin(ref['cost_endpoint_ref']),{})
        if endpoint['raw_ref']['sha256']!=ref['sha256'] or endpoint['identity']!=ident:raise ValueError('raw publication cost endpoint')
        extra=endpoint['common_row_index_publication_seconds'];endpoint_costs=endpoint['costs_before_publication']
        if endpoint_costs!=costs or not isinstance(extra,(float,int)) or not math.isfinite(extra) or extra<0:raise ValueError('closed publication cost')
        common_audit_seconds=max(0.,time.monotonic()-common_audit_started-sum(method_audit_seconds.values()))
        typed_layouts[str(i)]={m:None for m in ('AI','RULE','RANDOM','T')};publication_seconds={m:0. for m in typed_layouts[str(i)]}
        for method,(legacy_layout,geometry_audit,terminal_ref) in publish_candidates.items():
            publication_began=time.monotonic()
            publication=publish_terminal(record,source['case_ref'],dataset['authority_ref'],legacy_layout,prepared,geometry_audit,
                                         dict(original_terminal_asset_SHA=terminal_ref,original_node_row_SHA=ref,original_node_cost_endpoint_SHA=ref['cost_endpoint_ref'],original_prepared_asset_SHA=row['prepared_ref']),
                                         typed_directory/str(i)/method,method,postprocess_source=postsource,original_algorithm_source_epoch=commit)
            typed_layouts[str(i)][method]=publication;publication_seconds[method]=time.monotonic()-publication_began
        complete_costs={m:(v+extra+common_audit_seconds+method_audit_seconds[m]+publication_seconds[m] if v is not None else None) for m,v in costs.items()}
        atomic_json(typed_directory/str(i)/'complete_method_costs.json',dict(case_index=i,original_node_cost_endpoint_ref=ref['cost_endpoint_ref'],original_complete_costs=costs,
                    common_row_publication_seconds=extra,common_independent_read_audit_seconds_each_method=common_audit_seconds,method_audit_seconds=method_audit_seconds,
                    typed_layout_conversion_validation_publication_seconds=publication_seconds,complete_method_costs=complete_costs,same_rule_all_four_methods=True,new_geometry_search_or_PHY_calls=0,
                    caller_whole_publisher_elapsed_includes_internal_final_IO_hash_read=True,final_this_endpoint_IO_separate_metadata=True,
                    original_algorithm_source_epoch=commit,**postsource))
        am=methods['AI'];metrics=am.get('metrics');widths=metrics.get('widths',[]) if metrics else []
        witness_N=len(prepared.get('Wenv',{}).get('centers',[]))
        joint=bool(am['nonfallback'] and metrics and metrics['N']!=witness_N and len(widths)>1 and np.ptp(widths)>1e-12 and any(w>.3+1e-12 for w in widths))
        independent.append(dict(case_index=i,mother_id=record['original_mother_id'],methods=methods,costs=complete_costs,actual_AI_both_N_width_capability=joint,
                                teacher_identity_and_cost_closed=teacher_closed,total_request=record['total_request_bps_int64']))
    gate=node_gate(independent,phase,statistics_ref,Path(output).with_name('test_confirmation.json') if phase=='test' else None);gate.update(group=GROUP,model_identity=model_identity,checkpoint_sha256=checkpoint_ref['sha256'],
                                         manifest_refs=manifest_refs,dataset_ref=dataset_ref,scientific_commit=commit,science_updates=0,
                                         typed_layouts=typed_layouts,typed_publication_is_postprocessing=True,original_layouts_used=True,new_decoder_or_PHY_calls=0)
    gate.update(original_algorithm_source_epoch=commit,**postsource)
    atomic_json(output,gate);return gate
