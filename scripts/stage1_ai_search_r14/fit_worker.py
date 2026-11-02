"""The single admitted96-mother fit or single24-mother union veto, never a search."""
import json
import time
from pathlib import Path
from .lane import pinned
from .runtime import require_execution,ObjectStore,atomic_bytes,json_bytes
from .pipeline import OnceIntent,fit_gate
from .bindings import read_json,bind_fit_labels,good_exit,same_ref,verify_loaded_source,validate_cold_pairing
from .stop_control import child_stop
from .dispatcher import verify_receipt


def execute(request):
    require_execution(request)
    from satellite_rl.coverage.ai_search_r14.tree import fit_tree,select_baseline
    from satellite_rl.coverage.ai_search_r14.policies import apply_calibration_veto
    verify_loaded_source(request['source_root'],request['source_refs'])
    out=Path(request['output_root']);out.mkdir(parents=True,exist_ok=False)
    store=ObjectStore(request['shared_object_root'])
    stopped=child_stop(request,time.monotonic())
    if request['phase']=='fit_baseline':
        manifest=json.loads(pinned(request['exact140_metadata_ref']))
        B_members=read_json(request['dependency_receipts']['B_all140'])['member_receipts']
        train_members=read_json(request['dependency_receipts']['teacher_all96'])['member_receipts']
        B={r['recipe_index']:r for r in map(read_json,B_members)}
        train={r['recipe_index']:r for r in map(read_json,train_members)}
        for receipt in list(B.values())+list(train.values()):good_exit(receipt['actual_exit'])
        fit_gate(manifest,B,train)
        expected=[r['original_recipe_row']['case_id'] for r in manifest['rows'] if r['r14_partition']=='train']
        ordered=[train[r['recipe_index']] for r in manifest['rows'] if r['r14_partition']=='train']
        label_refs=[r['label_ref'] for r in ordered]
        if request.get('qualified_train_label_refs',label_refs)!=label_refs:raise ValueError('no alternate label ref for same case')
        labels=[read_json(ref) for ref in label_refs]
        if len(labels)!=96 or [r['case_id'] for r in labels]!=expected:
            raise ValueError('actual fixed96 training rows in original order')
        bind_fit_labels(label_refs,labels,ordered,expected)
        OnceIntent(request['shared_fit_intent_root']).claim('sole_model_fit',request,request)
        X=[r['features'] for r in labels];safe=[r['safe'] for r in labels];paid=[r['paid'] for r in labels]
        if stopped():raise RuntimeError('pause before sole fit')
        tree=fit_tree(X,safe,paid,expected,seed=51001)
        atomic_bytes(out/'fit_tree_completed_ref.json',json_bytes(store.put_json(tree)))
        if stopped():raise RuntimeError('pause before strong baseline selection')
        baseline=select_baseline(X,safe,paid,expected)
        result={'tree':tree,'baseline':baseline,'actual_train_cases':expected,'actual_fits':1,
                'complete_split_leaf_decisions_retained':True,'input_label_refs':label_refs}
    elif request['phase']=='calibration_veto':
        OnceIntent(request['shared_fit_intent_root']).claim('sole_calibration_veto',request,request)
        original=read_json(request['initial_policy_bundle_ref']);records=[]
        manifest=read_json(request['exact140_metadata_ref'])
        expected_calibration=[r['original_recipe_row']['case_id'] for r in manifest['rows'] if r['r14_partition']=='calibration']
        if request['calibration_mother_ids']!=expected_calibration:raise ValueError('original fixed24 calibration identities')
        offline_refs=request['calibration_offline_receipt_refs']
        declared_offline=read_json(request['dependency_receipts']['cal_offline24'])['member_receipts']
        declared_cold=read_json(request['dependency_receipts']['cal_all24x3'])['member_receipts']
        if offline_refs!=declared_offline:raise ValueError('exact24 actual aggregate offline refs')
        used_cold=[]
        if len(offline_refs)!=24:raise ValueError('all24 actual offline calibration pairings')
        for ref,case_id in zip(offline_refs,request['calibration_mother_ids']):
            sealed=verify_receipt(ref);pair=read_json(sealed['output_ref'])
            if pair['case_id']!=case_id or pair['online_calls_reused'] is not False or pair['actual_offline_layout_calls']!=6:
                raise ValueError('actual distinct offline calibration source')
            for policy in ('AI','baseline'):
                if not same_ref(pair['initial_policy_refs'][policy],original['policy_refs'][policy]):raise ValueError('immutable initial fitted policy reference')
            validate_cold_pairing([verify_receipt(coldref) for coldref in pair['cold_receipt_refs']],pair)
            used_cold.extend(pair['cold_receipt_refs'])
            records.extend(pair['calibration_records'])
        identity=lambda r:(r['sha256'],r['bytes'])
        if len(used_cold)!=72 or set(map(identity,used_cold))!=set(map(identity,declared_cold)):raise ValueError('actual72 calibration source refs, no replacement teachers')
        if stopped():raise RuntimeError('pause before sole aggregate veto')
        tree,baseline,receipt=apply_calibration_veto(original['tree'],original['baseline'],records,request['calibration_mother_ids'])
        result={'tree':tree,'baseline':baseline,'receipt':receipt,'initial_policy_bundle_ref':request['initial_policy_bundle_ref']}
    else:raise ValueError('only single fit or union veto')
    result['policy_refs']={'AI':store.put_json(result['tree']),'baseline':store.put_json(result['baseline'])}
    ref=store.put_json(result);atomic_bytes(out/'published_bundle_ref.json',json_bytes(ref))

if __name__=='__main__':
    from .worker_entry import main
    main(execute)
