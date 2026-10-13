"""Strict full R7 binding before an in-memory legacy identity projection."""
from copy import deepcopy

LEGACY_FIELDS=('case_id','mother_id','profile','input_sha256','identity_sha256','role')
FULL_FIELDS=(*LEGACY_FIELDS,'shared_case_content_sha256','source_asset_SHA')

def baseline_view(proposal,identity,prepared,method,producer_sha,prepared_ref):
    if method not in ('RULE','RANDOM') or set(identity)!=set(FULL_FIELDS):
        raise ValueError('exact R7 baseline/full eight-field identity required')
    if identity['role']!='development' or identity['profile']!='uniform21600000':
        raise ValueError('original node2000 uniform development identity')
    if any(not isinstance(v,str) or not v for v in identity.values()):
        raise ValueError('typed nonempty R7 identity values')
    if proposal.get('case_identity')!=identity or prepared.get('identity')!=identity:
        raise ValueError('complete sharedcase/source identity differs before projection')
    if prepared.get('input_sha256')!=identity['input_sha256']:
        raise ValueError('actual prepared input SHA differs')
    provenance=proposal.get('provenance',{})
    if provenance.get('case_identity')!=identity:
        raise ValueError('full proposal provenance identity differs')
    for key,value in dict(input_sha=identity['input_sha256'],role_sha=identity['identity_sha256']).items():
        if proposal.get(key)!=value or provenance.get(key)!=value:
            raise ValueError('proposal/provenance input or role differs')
    if provenance.get('source_method')!=method or provenance.get('producer_sha256')!=producer_sha or provenance.get('prepared_source_sha256')!=prepared.get('source_sha256'):
        raise ValueError('actual method/producer/prepared source differs')
    if not prepared_ref or provenance.get('evidence_refs')!=[prepared_ref]:
        raise ValueError('proposal original prepared evidence ref differs')
    view=deepcopy(proposal)
    view['case_identity']={key:identity[key] for key in LEGACY_FIELDS}
    return view

def audit_baseline(legacy_audit,scene,proposal,identity,prepared,method,producer_sha,prepared_ref):
    view=baseline_view(proposal,identity,prepared,method,producer_sha,prepared_ref)
    result=legacy_audit(scene,view,identity,prepared,None,method,producer_sha)
    return {**result,'R7_full_identity_checked_before_legacy_projection':True,
            'original_proposal_modified':False}

def preserve_unknown_cost(row,contract_ref,operations_source_ref,original_stage_refs):
    """Lost original parent timers are never replaced by reused-child or reread time."""
    if row['case_index']!=0:raise ValueError('only original failed case0 has this cost disposition')
    row['recovery_observed_timing_components']=deepcopy(row['cost_components'])
    row['recovery_observed_timing_components_scope']='mixed caller measurements plus reused original outer guard wall; NOT additive new epoch cost or reconstructed original cold cost; use durable recovery_parent_timer and independent worker epoch envelope for recovery-only accounting'
    row['cost_components']={key:None for key in row['cost_components']}
    # The unchanged523 caller retains this same object for its cost endpoint.
    # Mutate in place so every publication outlet preserves the unknown value.
    row['costs'].clear()
    row['costs'].update({method:None for method in ('AI','RULE','RANDOM','T')})
    row['recovery_accounting']=dict(schema='R7NodeOriginalUnknownColdCostDispositionV1',
        contract_ref=contract_ref,operations_source_ref=operations_source_ref,
        original_stage_refs=original_stage_refs,original_case0_complete_cold_costs=None,
        original_exact_cost_gate=False,unknown_original_parent_timers=True,
        reread_audit_or_reused_child_wall_does_not_reconstruct_original_cold_cost=True,
        original_spent_calls=dict(prepare=1,T=0,D5=2),new_epoch_calls=dict(prepare=0,T=1,D5=1),
        strict_speed_claims=False)
    return row['costs']
