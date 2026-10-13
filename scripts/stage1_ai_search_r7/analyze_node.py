"""Independent fixed2000/node20 gate; no inference, search, or tensor updates."""
from pathlib import Path
import math
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

def analyze(manifest_refs,dataset_ref,checkpoint_ref,model_identity,output):
    dataset=read_json(pin(dataset_ref),{});cases={};commit=None
    if dataset.get('schema')!='R7UniformDatasetV1' or dataset.get('role')!='development' or dataset.get('count')!=20:raise ValueError('root frozen development20 dataset')
    for ref in manifest_refs:
        manifest=read_json(pin(ref),{})
        if manifest.get('group')!=GROUP or manifest.get('phase')!='node2000' or manifest.get('checkpoint_ref')!=checkpoint_ref or manifest.get('model_identity')!=model_identity:raise ValueError('exact2000 model/source chain')
        if commit and commit!=manifest['scientific_commit']:raise ValueError('two source commits')
        commit=manifest['scientific_commit']
        for i,row in manifest['cases'].items():
            if i in cases:raise ValueError('duplicate host owner')
            cases[i]=row
    if sorted(map(int,cases))!=list(range(20)):raise ValueError('exact20 original denominator; no missing cases')
    independent=[]
    for i in range(20):
        ref=cases[str(i)];row=read_json(pin(ref),{});ident=row['identity'];source=dataset['cases'][i]
        record,scene=load_algorithm_case(source['case_ref'],dataset['authority_ref'])
        if row['case_index']!=i or row['source_commit']!=commit or row['model_identity']!=model_identity or row['checkpoint_ref']!=checkpoint_ref or ident['shared_case_content_sha256']!=record['content_sha256'] or scene.input_sha256!=ident['input_sha256']:raise ValueError('independent root scene identity')
        plain=dict(positions=scene.positions.tolist(),demands=scene.demands.tolist(),aircraft_ids=list(scene.aircraft_ids),profile=scene.profile)
        validate_binding(record,'development',i,source)
        if ident['role']!='development' or row['source_dataset_sha256']!=dataset_ref['sha256'] or row['core_sha256']!='9f187e4cd8122a6fce415b590d0ab0a385724b7d840cc88dd874e34ea936aea5':raise ValueError('original development/source/core binding')
        prepared=read_asset(row['prepared_ref'])['result'];stage_results=audit_preparation_teacher(row)
        if prepared['identity']!=ident or row.get('preparation_identity_and_cost_closed')is not stage_results['preparation']['closed'] or not audit_ledger(prepared['ledger'],len(scene.demands),'prepare')['valid']:raise ValueError('independent original prepare identity/ledger/closure')
        costs=audit_costs(row,stage_results);unavailable=row['status']=='preparation_unavailable'
        methods={};teacher_closed=False
        if unavailable:
            if prepared['status']=='available' or set(row['result_refs'])!=set(('AI','RULE','RANDOM','T')) or any(v is not None for v in row['result_refs'].values()) or row['new_calls']!=dict(prepare=1,T=0,D5=0):raise ValueError('unavailable node prep must not spend T or D5')
            missing=dict(valid=True,published=False,nonfallback=False,status='unavailable',metrics=None,identity_closed=True,budget_closed=True)
            methods={m:dict(missing) for m in ('AI','RULE','RANDOM')}
            if row['methods']!=methods or any(row['method_rows'][m].get('status')!='unexecuted' or row['method_rows'][m].get('result_ref')is not None or row['method_rows'][m].get('guard_proof')is not None for m in methods):raise ValueError('typed unavailable methods and full20 denominator')
        else:
            if prepared['status']!='available' or row['new_calls']!=dict(prepare=1,T=1,D5=3):raise ValueError('available node exact4 method call counts')
            teacher=read_asset(row['result_refs']['T'])['result']
            if any(teacher.get(k)!=v for k,v in ident.items()):raise ValueError('original T identity')
            teacher_audit=audit_layout(plain,teacher['layout'],prepared['Ncap'],prepared['Omega_cap']) if teacher['status']=='available' else None
            teacher_closed=bool(stage_results['preparation']['closed'] and stage_results['teacher']['closed'] and teacher_audit and teacher_audit['valid'] and audit_ledger(teacher['cost'],len(scene.demands),'TEACHER')['valid'])
            for method in ('AI','RULE','RANDOM'):
                actual=row['method_rows'][method];req=actual['request'];guard_closure=audit_method_proof(method,actual,row)
                if method=='AI' and (req['checkpoint_ref']!=checkpoint_ref or req['model_identity']!=model_identity):raise ValueError('actual AI checkpoint/source')
                if row['cost_components'][method+'_parent']<guard_closure['outer_wall_seconds']:raise ValueError('actual outer cold method wall not fully paid')
                checked=audit_row(plain,prepared,read_asset(actual['result_ref']),req)
                if checked!=actual['audit'] or checked!=row['methods'][method]:raise ValueError('producer metrics differ from independent raw audit')
                methods[method]={**checked,'budget_closed':bool(checked['budget_closed'] and guard_closure['closed'])}
        if row.get('teacher_identity_and_cost_closed')is not teacher_closed:raise ValueError('producer teacher closure differs from original guarded source')
        endpoint=read_json(pin(ref['cost_endpoint_ref']),{})
        if endpoint['raw_ref']['sha256']!=ref['sha256'] or endpoint['identity']!=ident:raise ValueError('raw publication cost endpoint')
        extra=endpoint['common_row_index_publication_seconds'];endpoint_costs=endpoint['costs_before_publication']
        if endpoint_costs!=costs or not isinstance(extra,(float,int)) or not math.isfinite(extra) or extra<0:raise ValueError('closed publication cost')
        independent.append(dict(case_index=i,methods=methods,costs={m:(v+extra if v is not None else None) for m,v in costs.items()},
                                teacher_identity_and_cost_closed=teacher_closed,total_request=record['total_request_bps_int64']))
    gate=node_gate(independent);gate.update(group=GROUP,model_identity=model_identity,checkpoint_sha256=checkpoint_ref['sha256'],
                                         manifest_refs=manifest_refs,dataset_ref=dataset_ref,scientific_commit=commit,science_updates=0)
    atomic_json(output,gate);return gate
