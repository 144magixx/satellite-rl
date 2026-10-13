"""Independent raw label/cost/layout re-audit; no search, inference or updates."""
from pathlib import Path
import math,time
import numpy as np
from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.storage import read_json,digest
from satellite_rl.workbench.coverage_ai_search import pin
from satellite_rl.workbench.r6_io import read_asset,pin_tree
from .label_audit import audit_bundle,audit_exact_proposal,gate
from .contract import GROUP
from satellite_rl.workbench.r7_label_guard_audit import audit_chain
from satellite_rl.workbench.r7_publish import publish_terminal
from satellite_rl.workbench.r7_postprocess_source import execution_source,new_output

def analyze(manifest_refs,dataset_ref,model_identity,output):
    output=new_output(output);postsource=execution_source()
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
    results=[];qualified=[];typed_teachers={};typed_directory=Path(output).with_name(Path(output).stem+'_typed_teachers')
    for i in range(576):
        audit_began=time.monotonic();ref=cases[str(i)];pin_tree(ref);row=read_json(pin(ref),{});ident=row['identity'];layout_audits={}
        if row['source_commit']!=source_commit or row['input_case_index']!=i or ident['role']!='train' or ident['profile']!='uniform21600000' or row.get('source_dataset_sha256')!=dataset_ref['sha256'] or row.get('core_sha256')!='9f187e4cd8122a6fce415b590d0ab0a385724b7d840cc88dd874e34ea936aea5':raise ValueError('original train identity and frozen core')
        for guard in row['guard_refs']:pin_tree(guard)
        endpoint=read_json(pin(ref['cost_endpoint_ref']),{})
        values=endpoint['components'].values()
        if endpoint['raw_ref']['sha256']!=ref['sha256'] or endpoint['identity']!=ident or any(not isinstance(x,(int,float)) or not math.isfinite(x) or x<0 for x in values) or not math.isclose(sum(endpoint['components'].values()),endpoint['complete_offline_label_seconds'],rel_tol=1e-12,abs_tol=1e-9):raise ValueError('independent exact cost endpoint')
        closure=audit_chain(row)
        if row.get('cost_closed')is not closure['closed'] or endpoint.get('cost_closed')is not closure['closed'] or row.get('actual_calls')!=closure['actual_calls']:raise ValueError('producer cost/call closure differs from original guard/request/caps proof')
        source=dataset['cases'][i]
        if source['case_ref']!=row['shared_case_ref']:raise ValueError('original root case source changed')
        record,scene=load_algorithm_case(source['case_ref'],dataset['authority_ref'])
        from satellite_rl.workbench.r7_case_binding import validate_binding
        validate_binding(record,'train',i,source)
        if ident['shared_case_content_sha256']!=record['content_sha256'] or ident['input_sha256']!=scene.input_sha256:raise ValueError('uniform shared/S1 source identity')
        with np.load(pin(row['scene_ref']),allow_pickle=False) as z:
            plain=dict(positions=z['positions'].tolist(),demands=z['demands'].tolist(),aircraft_ids=list(map(str,z['aircraft_ids'])),profile=str(z['profile']))
        if not plain['demands'] or any(x!=21600000 for x in plain['demands']):raise ValueError('uniform input all-plane request')
        if plain['positions']!=scene.positions.tolist() or plain['aircraft_ids']!=list(scene.aircraft_ids) or plain['profile']!=scene.profile:raise ValueError('all original planes and positions required')
        if row.get('status')=='preparation_unavailable':
            from scripts.stage1_ai_search_r4.audit import audit_ledger
            prepared=read_asset(row['prepared_ref'])
            if prepared['identity']!=ident or prepared['source_sha256']!=row['core_sha256'] or prepared['result']['status']=='available' or row.get('original_T_terminal_ref')is not None or row.get('D5_qualification_ref')is not None or not audit_ledger(prepared['result']['ledger'],len(plain['demands']),'prepare')['valid']:raise ValueError('typed unavailable witness cannot spend teacher/decoder')
            checked=dict(qualified=False,nontrivial=False,capability=False,mother_id=ident['mother_id'],label_origin='no_T_witness_unavailable',typed_unavailable=True);exact=None
        else:
            assets=[read_asset(row[k]) for k in ('prepared_ref','original_T_terminal_ref','D5_qualification_ref')]
            def capture(method,audit):layout_audits[method]=audit
            checked=audit_bundle(plain,*assets,ident,row['core_sha256'],closure['closed'],on_layout_audit=capture)
            exact=audit_exact_proposal(read_asset(assets[2]['proposal_ref']),assets[1]['result'],ident,row['core_sha256'],row['producer_sha256'])
        if checked!=row['label_audit'] or exact!=row['proposal_audit']:raise ValueError('producer audit differs from independent replay')
        independent_audit_seconds=time.monotonic()-audit_began;typed_teachers[str(i)]=None;publication_seconds=0.
        if closure['closed'] and 'T' in layout_audits:
            publication_began=time.monotonic()
            published=publish_terminal(record,source['case_ref'],dataset['authority_ref'],assets[1]['result']['layout'],assets[0]['result'],layout_audits['T'],
                                       dict(original_T_terminal_asset_SHA=row['original_T_terminal_ref'],original_label_row_SHA=ref,original_label_cost_endpoint_SHA=ref['cost_endpoint_ref'],original_prepared_asset_SHA=row['prepared_ref']),
                                       typed_directory/str(i)/'T','T',postprocess_source=postsource,original_algorithm_source_epoch=source_commit)
            typed_teachers[str(i)]=published;publication_seconds=time.monotonic()-publication_began
        atomic_json(typed_directory/str(i)/'complete_offline_label_cost.json',dict(original_label_cost_endpoint_ref=ref['cost_endpoint_ref'],
                    original_complete_offline_label_seconds=endpoint['complete_offline_label_seconds'],independent_postprocess_audit_seconds=independent_audit_seconds,
                    typed_T_conversion_validation_publication_seconds=publication_seconds,complete_offline_label_with_actual_typed_T_seconds=endpoint['complete_offline_label_seconds']+independent_audit_seconds+publication_seconds,
                    label_origin_remains_original_T=True,new_prepare_T_D5_or_PHY_calls=0,
                    caller_whole_publisher_elapsed_includes_internal_final_IO_hash_read=True,final_this_endpoint_IO_separate_metadata=True,
                    original_algorithm_source_epoch=source_commit,**postsource))
        results.append(checked)
        if checked['qualified']:
            qualified.append(dict(case_index=i,case_id=ident['case_id'],input_sha=ident['input_sha256'],role='train',identity=ident,
                                  scene_ref=row['scene_ref'],shared_case_ref=row['shared_case_ref'],label_ref_sha=row['original_T_terminal_ref']['sha256'],
                                  original_T_terminal_ref=row['original_T_terminal_ref'],label_receipt_ref=ref))
    result=gate(results);result.update(group=GROUP,phase='labels',model_identity=model_identity,
                                     scientific_commit=source_commit,source_dataset_sha256=dataset_ref['sha256'],manifest_refs=manifest_refs,
                                     typed_teacher_layouts=typed_teachers,original_T_labels_unchanged=True,typed_layout_publication_is_separate_postprocessing=True,new_prepare_T_D5_PHY_calls=0)
    result.update(original_algorithm_source_epoch=source_commit,**postsource)
    atomic_json(output,result)
    table=Path(output).with_name('qualified_train.json')
    atomic_json(table,dict(schema='R7QualifiedTrainManifestV1',role='train',source_dataset_sha256=dataset_ref['sha256'],
                           labels_gate_sha256=digest(output),qualified_cases=qualified,only_original_T_labels=True,original_algorithm_source_epoch=source_commit,**postsource))
    return result
