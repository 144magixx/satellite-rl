"""Read-only, root-bound accounting of the two original labels74 failures."""
from pathlib import Path
from .coverage_ai_search import pin
from .storage import read_json
from .r6_io import read_asset
from .r7_label_guard_audit import audit_stage

RUN='r7_labels_20261013_234807_312cdce7'
SCIENCE='13702ff5d4c2aac8c559ddf79bdd2d8202dec45f'
CORE='9f187e4cd8122a6fce415b590d0ab0a385724b7d840cc88dd874e34ea936aea5'
ADDENDUM='0f0b0492c861be9361a4afda12ff130ea6e0c1140d19a87760b6f0da3c7700be'
FAILED=(295,297)
PRESERVED=list(range(1,295,2))
REMAINING=list(range(299,576,2))

def json_ref(ref):return read_json(pin(ref),{})

def load_recovery(refs,cases,dataset_sha,source_commit):
    """Final root acceptance, original archived147 map and exact fixed partition."""
    if set(refs)!= {'contract_ref','root_review_ref','accounting_acceptance_ref','original_index_archive_ref'}:
        raise ValueError('exact recovery contract/review/accounting/original archived index refs required')
    c=json_ref(refs['contract_ref']);r=json_ref(refs['root_review_ref']);a=json_ref(refs['accounting_acceptance_ref'])
    if c.get('schema')!='R7Labels74OperationalRecoveryV1' or c.get('run_id')!=RUN or c.get('scientific_commit')!=SCIENCE or source_commit!=SCIENCE:
        raise ValueError('one original137 labels74 recovery only')
    if r.get('passed')is not True or r.get('run_id')!=RUN or r.get('contract_sha256')!=refs['contract_ref']['sha256'] or r.get('operational_commit')!=c.get('operational_commit'):
        raise ValueError('actual root exact recovery acceptance required')
    if c['accounting_addendum_ref']['sha256']!=ADDENDUM or a.get('passed')is not True or a.get('addendum_ref',{}).get('sha256')!=ADDENDUM or a.get('source_commit')!=SCIENCE or a.get('run_id')!=RUN:
        raise ValueError('root original accounting addendum acceptance')
    add=json_ref(c['accounting_addendum_ref']);part=add['fixed_original_partition']
    if c.get('preserved_case_indices')!=PRESERVED or c.get('failed_case_indices')!=list(FAILED) or c.get('remaining_case_indices')!=REMAINING or set(c['failed_cases'])!={'295','297'}:
        raise ValueError('exact147+2+139 recovery partition')
    if part['published_indices']!=PRESERVED or part['partial_unavailable_indices']!=list(FAILED) or part['never_spawned_remaining_indices']!=REMAINING:
        raise ValueError('partition differs from accepted addendum')
    if c.get('new_stage_attempts_max')!=dict(prepare=139,T=139,D5=139) or c.get('repeat_original_attempts')is not False or c.get('label_denominator')!=576:
        raise ValueError('unchanged no-repeat budget and denominator')
    job=json_ref(c['original_job_ref'])
    if job!=c['original_job'] or job['run_id']!=RUN or job['case_indices']!=list(range(1,576,2)) or job['original_task_contract']['config']['datasets']['train']['sha256']!=dataset_sha:
        raise ValueError('original sameID odd288 job/dataset')
    for ref in c['operations_source_refs'].values():pin(ref)
    owner=json_ref(c['original_owner_ref'])
    if owner.get('run_id')!=RUN or owner.get('request_hash')!=job['request_hash'] or owner.get('job_sha256')!=c['original_job_ref']['sha256']:
        raise ValueError('original sameID owner')
    if refs['original_index_archive_ref']['sha256']!=c['original_case_index_ref']['sha256']:
        raise ValueError('original147 archive SHA differs; current mutable index is not original evidence')
    old=json_ref(refs['original_index_archive_ref'])
    if sorted(map(int,old))!=PRESERVED:raise ValueError('original index must contain exactly147')
    for i in PRESERVED:
        case=json_ref(old[str(i)])
        if case.get('case_index')!=i or case.get('mode')!='labels' or case.get('parent_job_sha256')!=c['original_job_ref']['sha256'] or case.get('result_ref')!=cases[str(i)]:
            raise ValueError('published147 original case/label ref changed')
    return dict(refs=refs,contract=c,addendum=add)

def audit_failed(row,index,source,dataset,context):
    """No normal label/guard fabricated; independently bind requests and saved work."""
    c=context['contract'];entry=c['failed_cases'].get(str(index))
    if index not in FAILED or entry is None:raise ValueError('only two frozen original failures allowed')
    expected=dict(schema='R7OperationalFailedLabelDispositionV1',run_id=RUN,input_case_index=index,
                  identity=entry['identity'],scientific_commit=SCIENCE,qualified=False,nontrivial=False,capability=False,
                  cost_closed=False,status='infrastructure_interrupted',original_attempts_retried=False,
                  known_unknown_accounting=entry['accounting'],original_evidence_refs=entry['evidence_refs'],
                  root_contract_ref=context['refs']['contract_ref'],root_review_ref=context['refs']['root_review_ref'],operational_commit=c['operational_commit'])
    if row!=expected:raise ValueError('failed disposition differs from exact root contract; no normal payload permitted')
    evidence=entry['evidence_refs']
    for ref in evidence.values():pin(ref)
    def ref(stage,name):
        return evidence[f'case_workers__{index}__bundles__{index}__{stage}__{name}']
    ident=entry['identity'];accepted=context['addendum']['cases'][str(index)]
    if ident!=accepted['original_identity'] or ident['role']!='train' or ident['profile']!='uniform21600000' or ident['case_id']!=source['case_id'] or ident['mother_id']!=source['original_mother_id'] or ident['shared_case_content_sha256']!=source['case_content_sha256']:
        raise ValueError('failure original all-plane role/case identity')
    from satellite_rl.shared_interface.content import read_wire
    from .r7_case_binding import validate_binding
    record=read_wire(pin(source['case_ref']).read_bytes());validate_binding(record,'train',index,source)
    if record['content_sha256']!=ident['shared_case_content_sha256'] or record['source_asset_SHA']!=ident['source_asset_SHA']:
        raise ValueError('failure identity differs from original shared wire')
    accounting=entry['accounting'];n=len(record['aircraft_ids'])
    expected_calls=dict(prepare=1,T=int(index==297),D5=0)
    if accounting['n']!=n or accounting['attempted']!=expected_calls or accounting.get('actual_complete_label_cost_seconds')is not None or accounting.get('unknown_cost_preserved')is not True:
        raise ValueError('failure call count/unknown actual wall cannot be imputed')
    output=json_ref(ref('preparation','stage_output.json'));asset=read_asset(output['asset_ref'])
    if output['identity']!=ident or output['stage']!='preparation' or asset['identity']!=ident or asset['source_sha256']!=CORE or asset['result']['ledger']!=accounting['original_preparation_ledger']:
        raise ValueError('saved original preparation asset/ledger differs')
    ledger=asset['result']['ledger']
    for key,value in accepted['original_preparation_ledger'].items():
        if ledger.get(key)!=value:raise ValueError('preparation ledger differs from accepted accounting')
    reserve=dict(prepare_raypairs=1024*n if index==295 else ledger['raypairs'],T_raypairs=4096*n if index==297 else 0,
                 T_proposals=2048 if index==297 else 0,T_refits=16 if index==297 else 0,D5_raypairs=0,D5_proposals=0,D5_refits=0)
    if accounting['conservative_budget_charged']!=reserve or accounting['preparation_guard_closed']is not (index==297):
        raise ValueError('unknown work must reserve original full authorized cap')
    for stage in ('preparation','teacher') if index==297 else ('preparation',):
        request_ref=ref(stage,'request.json');request=json_ref(request_ref);started=json_ref(ref(stage,'started.json'));wal=json_ref(ref(stage,'stage.wal.json'))
        phase=request['phase_manifest'];scene_ref=evidence[f'case_workers__{index}__bundles__{index}__scene.npz']
        if request['identity']!=ident or request['scene_ref']!=scene_ref or request['source_sha256']!=CORE or request['producer_sha256']!=c['original_job']['source_manifest_sha256'] or request['stage']!=stage:
            raise ValueError('failed original request/case/core/producer')
        if phase.get('stage')!='labels' or phase.get('role')!='train' or phase.get('admitted')is not True or phase.get('dataset_sha256')!=dataset['sha256'] or phase.get('identity_sha256')!=ident['identity_sha256'] or phase.get('cases',{}).get(ident['case_id'])!=ident:
            raise ValueError('failed original phase capability')
        command=wal.get('command',[])
        if started.get('science_call_reserved')is not True or started.get('request_sha256')!=request_ref['sha256'] or wal.get('status')!='child_started' or wal.get('hard_seconds')!=(20 if stage=='preparation' else 45) or wal.get('owned_parent_children_rss_bytes')!=2147483648 or type(wal.get('pid'))is not int or type(wal.get('birth'))is not int:
            raise ValueError('failed original reservation/WAL/caps')
        if command!=[asset['python'],'-B','-u','-m','satellite_rl.workbench.r7_stage','--request',request_ref['path'],'--directory',str(Path(request_ref['path']).parent)]:
            raise ValueError('original failed stage command')
        if stage=='teacher' and request.get('prepared_ref')!=output['asset_ref']:raise ValueError('failed teacher original prepared asset')
    if index==297:
        closure=audit_stage(guard_ref=ref('preparation','guard.json'),request_ref=ref('preparation','request.json'),started_ref=ref('preparation','started.json'),wal_ref=ref('preparation','stage.wal.json'),guard_receipt_ref=ref('preparation','guard_receipt.json'),
                            stage='preparation',identity=ident,scene_ref=evidence[f'case_workers__{index}__bundles__{index}__scene.npz'],source_sha=CORE,producer_sha=c['original_job']['source_manifest_sha256'],dataset_sha=dataset['sha256'],asset_ref=output['asset_ref'])
        if not closure['closed'] or closure['wall_seconds']!=accepted['preparation_stage_wall_seconds']:raise ValueError('known preparation guard closure differs')
    checked=dict(qualified=False,nontrivial=False,capability=False,mother_id=ident['mother_id'],label_origin='no_T_terminal_infrastructure_interruption',failed_original_disposition=True)
    return checked,dict(actual_calls=expected_calls,conservative_budget_charged=reserve,actual_complete_label_seconds=None,
                        unknown_stage=accounting['unknown_stage'],accepted_partial_accounting=accepted,strict_full_cost_or_speed_claim_allowed=False)

def phase_accounting(calls,failed_indices):
    if failed_indices!=list(FAILED):raise ValueError('exact two failed dispositions required')
    if any(type(calls[k])is not int or not 0<=calls[k]<=v for k,v in dict(prepare=576,T=575,D5=574).items()):
        raise ValueError('original576/575/574 call caps')
    return dict(actual_started_calls=calls,failed_indices=failed_indices,whole_label_phase_exact_cost_closed=False,
                actual_complete_label_phase_seconds=None,strict_speed_comparison_allowed=False,exact_amortization_allowed=False,
                unknown_work_is_reserved_not_actual_measurement=True,guard_caps_are_not_empirical_wall_bounds=True,
                missing_two_not_proven_unbiased=True,qualified_labels_require_individual_cost_closure=True)
