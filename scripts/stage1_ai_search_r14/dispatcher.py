"""Fixed worker routing and derived evidence collection, never Boolean qualification."""
import json
import posixpath
from pathlib import Path
from .runtime import json_bytes,atomic_bytes,ObjectStore
from .bindings import read_json,good_exit,same_ref,bind_task

WORKERS={'data':'data_worker','B':'B_worker','teacher':'cold_worker','fit':'fit_worker','veto':'fit_worker',
         'cal':'cold_worker','dev':'cold_worker','offline':'offline_worker','engineering':'engineering_worker',
         'data_registry':'evidence_worker','aggregate':'evidence_worker'}
def worker_module(kind):
    if kind not in WORKERS:raise ValueError('fixed admitted worker kind')
    return 'scripts.stage1_ai_search_r14.'+WORKERS[kind]

def command_for_task(plan,task):
    ref=task['request_ref']
    return [plan['interpreter'],'-B','-u','-m',worker_module(task['kind']),
            '--request',ref['path'],'--sha256',ref['sha256'],'--bytes',str(ref['bytes'])]

def make_request(plan,task,*,inputs,dependencies):
    """Fixed admitted plan inheritance plus actual collector refs, never invented receipts."""
    copied=('scientific_execution_authorized','root_exact_source_input_admitted','source_specific_pure_checks_accepted',
        'actual_runtime_owner_stop_space_admitted','runtime_epoch_ref','port','retry','source_root','source_refs','source_manifest_ref',
        'private_dependency_paths','private_dependency_manifest_ref','shared_object_root','shared_audit_ledger',
        'shared_input_ledger_root','shared_fit_intent_root','own_stop_marker','global_stop_marker')
    request={k:plan[k] for k in copied}
    request['root_admission_ref']=plan['child_stage_admission_ref']
    identity={k:task[k] for k in ('task_id','kind','phase','recipe_index','arm','repeat','fixture_index','purpose') if k in task}
    if any(k in inputs for k in (*identity,*copied)):raise ValueError('inputs cannot replace task/admission/source identity')
    task_id=task['task_id'].replace(':','_')
    join=posixpath.join if plan['control_root'].startswith('/') else lambda *p:str(Path(*p))
    request.update(inputs);request.update(identity)
    request.update(output_root=join(plan['output_root'],task_id),
        own_child_stop_marker=join(plan['control_root'],task_id,'stop.request'),dependency_receipts=dependencies)
    bind_task(task,request);check_dependencies(request)
    return request

def check_dependencies(request):
    required={'data':('engineering_all8',), 'engineering':(), 'B':('data','data_authority'), 'teacher':('B','engineering_all8'),
       'fit':('B_all140','teacher_all96'), 'cal':('initial_fit',), 'veto':('initial_fit','cal_all24x3','cal_offline24'),
       'dev':('veto',), 'offline':('cold_complete',),'aggregate':(),'data_registry':('data_all140',)}[request['kind']]
    proof=request.get('dependency_receipts',{})
    for name in required:
        if name not in proof:raise ValueError('missing actual stage predecessor '+name)
        receipt=verify_receipt(proof[name])
        if receipt.get('evidence_role')!=name:raise ValueError('wrong derived predecessor role')
        if name=='data' and not same_ref(receipt['output_ref']['case_ref'],request['case_ref']):raise ValueError('actual typed case predecessor SHA')
        if name=='B' and not same_ref(proof[name],request['sealed_B_receipt_ref']):raise ValueError('same sealed actual B predecessor')
        if name=='data_authority' and not same_ref(read_json(receipt['output_ref'])['authority_envelope_ref'],request['authority_envelope_ref']):raise ValueError('actual full140 source registry')
        if name in ('initial_fit','veto'):
            expected_bundle=request.get('policy_bundle_ref',request.get('initial_policy_bundle_ref'))
            if not same_ref(receipt['output_ref'],expected_bundle):raise ValueError('frozen actual policy bundle predecessor')
            if request.get('arm') in ('AI','strong_causal_baseline'):
                key='AI' if request['arm']=='AI' else 'baseline'
                if not same_ref(read_json(receipt['output_ref'])['policy_refs'][key],request['policy_ref']):raise ValueError('actual branch policy source')
    return True

def verify_receipt(ref,depth=0):
    if depth>3:raise ValueError('finite derived evidence depth')
    receipt=read_json(ref)
    if receipt.get('actual_exit'):
        facts=receipt['actual_exit'];good_exit(facts)
        if facts.get('task_id')!=receipt['task_id'] or not same_ref(facts.get('request_ref'),receipt['request_ref']):raise ValueError('sealed leaf actual exit/request source join')
        original=read_json(receipt['request_ref'])
        for key in ('task_id','kind','recipe_index','arm','repeat','fixture_index'):
            if original.get(key)!=receipt.get(key):raise ValueError('sealed original request '+key)
        if original.get('source_refs')!=receipt.get('source_refs'):raise ValueError('sealed exact source closure references')
    if receipt.get('member_receipts'):
        members=receipt.get('member_receipts',[])
        caps={'data_all140':140,'B_all140':140,'teacher_all96':96,'cal_all24x3':72,'cal_offline24':24,'engineering_all8':8}
        if len(members)!=caps.get(receipt.get('evidence_role')):raise ValueError('fixed actual predecessor denominator')
        values=[verify_receipt(r,depth+1) for r in members]
        if len({v['task_id'] for v in values})!=len(values):raise ValueError('unique actual member tasks required')
        if [v['task_id'] for v in values]!=receipt['member_task_ids']:raise ValueError('member actual task refs differ')
        from .data_manifest import PARTITIONS
        role=receipt['evidence_role']
        if role in ('data_all140','B_all140','teacher_all96'):
            indices=[i for a in PARTITIONS.values() for i in a] if role!='teacher_all96' else list(PARTITIONS['train'])
            kind={'data_all140':'data','B_all140':'B','teacher_all96':'teacher'}[role]
            if sorted(v['recipe_index'] for v in values)!=sorted(indices) or any(v['kind']!=kind for v in values):raise ValueError('fixed source mother role set')
        if role=='cal_all24x3':
            expected={(i,a) for i in PARTITIONS['calibration'] for a in ('AI','strong_causal_baseline','full_SHC')}
            if {(v['recipe_index'],v['arm']) for v in values}!=expected:raise ValueError('24x3 actual calibration arms')
        if role=='cal_offline24' and {v['recipe_index'] for v in values}!=set(PARTITIONS['calibration']):raise ValueError('24 actual offline calibration mothers')
        if role=='engineering_all8' and {v['fixture_index'] for v in values}!=set(range(8)):raise ValueError('original8 actual engineering fixtures')
    elif not receipt.get('actual_exit'):raise ValueError('actual leaf exit or complete derived members required')
    return receipt

def collect_task(task,request,exit_facts):
    good_exit(exit_facts)
    if (exit_facts.get('task_id')!=task['task_id'] or exit_facts.get('kind')!=task['kind']
        or not same_ref(exit_facts.get('request_ref'),task['request_ref'])):raise ValueError('direct child exact task/request exit source')
    from .bindings import bind_task
    bind_task(task,request)
    out=Path(request['output_root']);kind=task['kind'];store=ObjectStore(request['shared_object_root'])
    result={'schema':'R14DerivedActualTaskReceiptV1','kind':kind,'task_id':task['task_id'],
        'source_task_id':task['task_id'],'actual_exit':exit_facts,'request_ref':task['request_ref'],
        'recipe_index':request.get('recipe_index'),'arm':request.get('arm'),'repeat':request.get('repeat'),
        'fixture_index':request.get('fixture_index'),'source_refs':request['source_refs']}
    if kind=='B':
        sequence=json.loads((out/'objects.json').read_bytes())
        refs=[x['ref'] for x in sequence if x['label']=='qualified_B_before_parent_exit']
        if len(refs)!=1:raise ValueError('one actual success B publication')
        B=read_json(refs[0])
        if B['status']!='success' or B.get('qualified') is not True:raise ValueError('actual B qualification')
        result.update(B_ref=refs[0],case_ref=request['case_ref'],case_id=B['case_id'],
            cache_ref=B['cache_manifest']['cache_ref'],inner_cache_SHA=B['cache_manifest']['inner_cache_SHA'],
            status='success',proposal_success=True,qualified=True,complete_cost_exit=True,
            nested_independent_audit_elapsed=B.get('nested_independent_audit_elapsed',0.),
            independent_layouts=B['independent_layouts'],evidence_role='B')
    elif kind in ('teacher','cal','dev'):
        result_ref=json.loads((out/'result.json').read_bytes());record=read_json(result_ref)
        if record.get('task_id')!=task['task_id'] or record.get('arm')!=request['arm'] or record.get('complete') is not True:
            raise ValueError('actual cold output task/arm/complete join')
        result.update(result_ref=result_ref,case_id=record['case_id'],features=record['features'],
            selected_policy=record['selected_policy'],terminal_metrics=record['own_terminal_audit']['metrics'],
            nested_independent_audit_elapsed=record.get('nested_independent_audit_elapsed',0.),
            B_ref=request['B_ref'],case_ref=request['case_ref'],cache_ref=request['cache_ref'],
            policy_ref=request.get('policy_ref'),policy_bundle_ref=request.get('policy_bundle_ref'),evidence_role='cold_complete')
        if kind=='teacher':
            label_ref=json.loads((out/'training_label_row.json').read_bytes());label=read_json(label_ref)
            if label['source_task_id']!=task['task_id']:raise ValueError('label actual teacher source')
            result.update(label_ref=label_ref,all_positions_audited=True,teacher_action='cap32',
                B_and_all_accepted_positions_independently_audited=True,qualified=True,complete_cost_exit=True,evidence_role='teacher')
    else:
        filename={'data':'typed_case_derivation.json','fit':'published_bundle_ref.json','veto':'published_bundle_ref.json',
                  'data_registry':'evidence_receipt_ref.json','aggregate':'evidence_receipt_ref.json',
                  'offline':'offline_receipt_ref.json','engineering':'engineering_receipt_ref.json'}[kind]
        raw=json.loads((out/filename).read_bytes())
        result.update(output_ref=raw,evidence_role={'data':'data','fit':'initial_fit','veto':'veto',
                     'data_registry':'data_authority','aggregate':request.get('evidence_role'),
                     'offline':'offline','engineering':'engineering'}[kind])
        if kind=='aggregate':
            group=read_json(raw)
            result.update(member_receipts=group['member_receipts'],member_task_ids=group['member_task_ids'])
    ref=store.put_json(result);atomic_bytes(out/'sealed_actual_receipt_ref.json',json_bytes(ref));return result,ref

def aggregate(role,member_refs,*,expected_ids,store):
    receipts=[verify_receipt(r) for r in member_refs]
    if len(receipts)!=len(expected_ids) or [r['task_id'] for r in receipts]!=expected_ids:raise ValueError('fixed complete predecessor denominator/order')
    for receipt in receipts:good_exit(receipt['actual_exit'])
    ref=store.put_json({'schema':'R14DerivedStageReceiptV1','evidence_role':role,'member_receipts':member_refs,
        'member_task_ids':expected_ids})
    verify_receipt(ref);return ref

def indexed_receipts(member_refs,*,kind,expected_indices):
    values=[verify_receipt(ref) for ref in member_refs]
    if [v['recipe_index'] for v in values]!=list(expected_indices) or any(v['kind']!=kind for v in values):
        raise ValueError('fixed source-qualified indexed receipts')
    return {v['recipe_index']:v for v in values}

def publish_data_authority(manifest,member_refs,*,store):
    """Independent collector derives the full registry from actual140 data exits.

    Pure identity/numerical-schema qualification, no geometry or new source
    decode. Root still pins this exact published envelope for the next stage.
    """
    from satellite_rl.shared_interface.content import read_wire,content_sha
    from satellite_rl.shared_interface.contracts import Authority,validate_case
    from .identity_adapter import task_binding
    from .lane import pinned
    expected=manifest['rows'];indices=[r['recipe_index'] for r in expected]
    receipts=indexed_receipts(member_refs,kind='data',expected_indices=indices)
    cases=[];bindings=[]
    for selected in expected:
        receipt=receipts[selected['recipe_index']];derived=receipt['output_ref']
        case=read_wire(pinned(derived['case_ref']))
        binding=task_binding(selected,case)
        if (case['source']['raw_perplane_SHA']!=selected['original_recipe_row']['source_asset_SHA']
            or case['source']['metadata_job_SHA']!=manifest['source_job_ref']['sha256']):raise ValueError('actual typed case original source chain')
        cases.append(case);bindings.append(binding)
    registry={'schema':'R7PublishedParentRegistryV1','case_sha_by_id':{c['case_id']:c['content_sha256'] for c in cases},
              'layout_shas':[],'allocation_shas':[]}
    if len(registry['case_sha_by_id'])!=140:raise ValueError('all fixed140 unique typed mothers')
    registry_sha=content_sha(registry);authority=Authority(registry,registry_sha)
    for case in cases:validate_case(case,authority,geometry_check=False)
    envelope={'schema':'R14PublishedParentAuthorityV1','group':'S1-AI17-R14','independently_qualified':True,
        'shared_registry':registry,'task_bindings':bindings,'actual_data_receipt_refs':member_refs,
        'source_recipe_ref':manifest['recipe_ref'],'qualification':'separate collector source/identity/schema; no geometry claim'}
    ref=store.put_json(envelope)
    return {'authority_envelope_ref':ref,'authority_envelope_content_sha256':ref['sha256'],
            'shared_registry_content_sha256':registry_sha,'root_exact_next_stage_pin_required':True}
