"""Independent raw label/cost/layout re-audit; no search, inference or updates."""
from pathlib import Path
import math
import numpy as np
from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.storage import read_json,digest
from satellite_rl.workbench.coverage_ai_search import pin
from satellite_rl.workbench.r6_io import read_asset,pin_tree
from .label_audit import audit_bundle,audit_exact_proposal,gate
from .contract import GROUP

def analyze(manifest_refs,dataset_ref,model_identity,output):
    from satellite_rl.workbench.r7_uniform_data import load_algorithm_case
    dataset=read_json(pin(dataset_ref),{})
    if dataset.get('schema')!='R7UniformDatasetV1' or dataset.get('role')!='train' or dataset.get('count')!=576:raise ValueError('new train576 dataset')
    cases={};source_commit=None
    for ref in manifest_refs:
        manifest=read_json(pin(ref),{})
        if manifest.get('group')!=GROUP or manifest.get('phase')!='labels' or manifest['source_dataset_sha256']!=dataset_ref['sha256']:raise ValueError('actual new uniform label manifest')
        if source_commit is not None and source_commit!=manifest['scientific_commit']:raise ValueError('two scientific sources')
        source_commit=manifest['scientific_commit']
        for i,row in manifest['cases'].items():
            if i in cases:raise ValueError('duplicate case owner')
            cases[i]=row
    if sorted(map(int,cases))!=list(range(576)):raise ValueError('all576 labels accounted including failures')
    results=[];qualified=[]
    for i in range(576):
        ref=cases[str(i)];pin_tree(ref);row=read_json(pin(ref),{});ident=row['identity']
        if row['source_commit']!=source_commit or row['input_case_index']!=i or ident['role']!='train' or ident['profile']!='uniform21600000':raise ValueError('original train identity')
        for guard in row['guard_refs']:pin_tree(guard)
        endpoint=read_json(pin(ref['cost_endpoint_ref']),{})
        values=endpoint['components'].values()
        if endpoint['raw_ref']['sha256']!=ref['sha256'] or endpoint['identity']!=ident or any(not isinstance(x,(int,float)) or not math.isfinite(x) or x<0 for x in values) or not math.isclose(sum(endpoint['components'].values()),endpoint['complete_offline_label_seconds'],rel_tol=1e-12,abs_tol=1e-9):raise ValueError('independent exact cost endpoint')
        assets=[read_asset(row[k]) for k in ('prepared_ref','original_T_terminal_ref','D5_qualification_ref')]
        source=dataset['cases'][i]
        if source['case_ref']!=row['shared_case_ref']:raise ValueError('original root case source changed')
        record,scene=load_algorithm_case(source['case_ref'],dataset['authority_ref'])
        if ident['shared_case_content_sha256']!=record['content_sha256'] or ident['input_sha256']!=scene.input_sha256:raise ValueError('uniform shared/S1 source identity')
        with np.load(pin(row['scene_ref']),allow_pickle=False) as z:
            plain=dict(positions=z['positions'].tolist(),demands=z['demands'].tolist(),aircraft_ids=list(map(str,z['aircraft_ids'])),profile=str(z['profile']))
        if not plain['demands'] or any(x!=21600000 for x in plain['demands']):raise ValueError('uniform input all-plane request')
        if plain['positions']!=scene.positions.tolist() or plain['aircraft_ids']!=list(scene.aircraft_ids) or plain['profile']!=scene.profile:raise ValueError('all original planes and positions required')
        checked=audit_bundle(plain,*assets,ident,assets[0]['source_sha256'],bool(row['cost_closed'] and endpoint['cost_closed']))
        exact=audit_exact_proposal(read_asset(assets[2]['proposal_ref']),assets[1]['result'],ident,assets[0]['source_sha256'],row['producer_sha256'])
        if checked!=row['label_audit'] or exact!=row['proposal_audit']:raise ValueError('producer audit differs from independent replay')
        results.append(checked)
        if checked['qualified']:
            qualified.append(dict(case_index=i,case_id=ident['case_id'],input_sha=ident['input_sha256'],role='train',identity=ident,
                                  scene_ref=row['scene_ref'],shared_case_ref=row['shared_case_ref'],label_ref_sha=row['original_T_terminal_ref']['sha256'],
                                  original_T_terminal_ref=row['original_T_terminal_ref'],label_receipt_ref=ref))
    result=gate(results);result.update(group=GROUP,phase='labels',model_identity=model_identity,
                                     scientific_commit=source_commit,source_dataset_sha256=dataset_ref['sha256'],manifest_refs=manifest_refs)
    atomic_json(output,result)
    table=Path(output).with_name('qualified_train.json')
    atomic_json(table,dict(schema='R7QualifiedTrainManifestV1',role='train',source_dataset_sha256=dataset_ref['sha256'],
                           labels_gate_sha256=digest(output),qualified_cases=qualified,only_original_T_labels=True))
    return result
