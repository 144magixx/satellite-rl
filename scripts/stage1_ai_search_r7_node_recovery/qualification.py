"""Readback separates unchanged original exact-cost gate from named future recovery facts."""
def original_gate_and_recovery_facts(cases,original_node_gate,*,amendment_ref,bound_ref):
    if len(cases)!=20 or [row['case_index'] for row in cases]!=list(range(20)):
        raise ValueError('original twenty case denominator and order')
    if cases[0]['costs']!={method:None for method in ('AI','RULE','RANDOM','T')}:
        raise ValueError('original failed case0 complete cold costs stay null')
    original=original_node_gate(cases,phase='node2000')
    if original['passed']is not False or original['hard_identity_budget_cost_closed']is not False:
        raise ValueError('original exact-cost qualification can never be presented as recovered')
    return dict(schema='R7Node2000RecoveryQualificationResultV2',
        qualification_name='node2000_engineering_recovery_quality_and_bounded_spend_v2',
        original_gate=original,original_gate_passed=False,original_hard_identity_budget_cost_closed=False,
        original_case0_exact_complete_cold_costs=None,root_amendment_ref=amendment_ref,
        failure_epoch_spend_upper_bound_ref=bound_ref,denominator=20,
        nonfallback=original['nonfallback'],quality_successes=original['quality_successes'],
        unchanged_quality_thresholds_met=original['nonfallback']>=15 and original['quality_successes']>=5,
        recovery_qualification_passed=False,scientific_call_budget_closed=None,
        root_result_acceptance_ref=None,
        required_independent_closure=['all80 original method slots and geometry/source identities',
            'original1W/2D5 plus resumed19W/20T/58D5 request-WAL-guard audit',
            'audited original enclosing bound and persisted new epoch accounting',
            'root independently accepted complete named qualification result'],
        speed_claims_allowed=False,future_dispatch_allowed=False,
        no_generic_passed_field=True)
