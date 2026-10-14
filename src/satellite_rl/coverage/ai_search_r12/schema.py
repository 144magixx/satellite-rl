"""Versioned producer/reader vocabulary; missing/unknown always fails closed."""
from enum import Enum
SCHEMA = 'R12NodeV2'
class NodeState(str, Enum):
    KNOWN_UNAVAILABLE = 'known_unavailable'
    GRAPH_INFEASIBLE = 'graph_infeasible'
    REJECTED = 'rejected'
    NO_IMPROVEMENT = 'no_improvement'
    ADMISSIBLE = 'admissible'
    UNKNOWN = 'unknown'
COMPLETED = frozenset(s.value for s in NodeState if s is not NodeState.UNKNOWN)
COMPLETION_PAIRS = {
    NodeState.KNOWN_UNAVAILABLE.value: frozenset({('known_unavailable','analytic_structure_bound')}),
    NodeState.GRAPH_INFEASIBLE.value: frozenset({('fixed_anchor_capacity_infeasible','complete_pinned_graph_hi_m_infeasible')}),
    NodeState.REJECTED.value: frozenset({
        ('degenerate_mean_ray','observed_degenerate_mean'),
        ('envelope_over1_conservative_kernel','observed_outward_envelope'),
        ('whole_Omega_cap','actual_whole_Omega'),
        ('dual_current_B_quality_protection','complete_raw_metrics')}),
    NodeState.NO_IMPROVEMENT.value: frozenset({('exact_current_tie_or_worse','complete_raw_metrics')}),
    NodeState.ADMISSIBLE.value: frozenset({('strict_raw_objective_improvement','complete_hard_legal_dual_protected')}),
}
def qualified_complete(node):
    pair=(node.get('raw_producer_label'),node.get('proof_qualification'))
    return (node.get('schema') == SCHEMA and node.get('complete') is True
            and pair in COMPLETION_PAIRS.get(node.get('state'),())
            and all(k in node for k in ('semantic_SHA','terminal','metrics','ledger','exact_evidence')))
