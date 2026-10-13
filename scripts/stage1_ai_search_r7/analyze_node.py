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
        prepared=read_asset(row['prepared_ref'])['result'];teacher=read_asset(row['result_refs']['T'])['result']
        if prepared['identity']!=ident or any(teacher.get(k)!=v for k,v in ident.items()):raise ValueError('original T identity')
        teacher_audit=audit_layout(plain,teacher['layout'],prepared['Ncap'],prepared['Omega_cap']) if teacher['status']=='available' else None
        teacher_closed=bool(teacher_audit and teacher_audit['valid'] and audit_ledger(teacher['cost'],len(scene.demands),'TEACHER')['valid'] and row['teacher_identity_and_cost_closed'])
        methods={}
        for method in ('AI','RULE','RANDOM'):
            actual=row['method_rows'][method];req=actual['request'];guard=read_json(pin(actual['guard_ref']),{});pin_tree(guard)
            if guard.get('status')!='completed' or guard.get('unknown_algorithm_spend') or req['identity']!=ident or req['method']!=method:raise ValueError('actual method identity/guard budget')
            if method=='AI' and (req['checkpoint_ref']!=checkpoint_ref or req['model_identity']!=model_identity):raise ValueError('actual AI checkpoint/source')
            checked=audit_row(plain,prepared,read_asset(actual['result_ref']),req)
            if checked!=actual['audit'] or checked!=row['methods'][method]:raise ValueError('producer metrics differ from independent raw audit')
            methods[method]=checked
        endpoint=read_json(pin(ref['cost_endpoint_ref']),{})
        if endpoint['raw_ref']['sha256']!=ref['sha256'] or endpoint['identity']!=ident:raise ValueError('raw publication cost endpoint')
        extra=endpoint['common_row_index_publication_seconds'];costs=endpoint['costs_before_publication']
        if costs!=row['costs'] or not isinstance(extra,(float,int)) or not math.isfinite(extra) or extra<0:raise ValueError('closed publication cost')
        independent.append(dict(case_index=i,methods=methods,costs={m:v+extra for m,v in costs.items()},
                                teacher_identity_and_cost_closed=teacher_closed,total_request=record['total_request_bps_int64']))
    gate=node_gate(independent);gate.update(group=GROUP,model_identity=model_identity,checkpoint_sha256=checkpoint_ref['sha256'],
                                         manifest_refs=manifest_refs,dataset_ref=dataset_ref,scientific_commit=commit,science_updates=0)
    atomic_json(output,gate);return gate
