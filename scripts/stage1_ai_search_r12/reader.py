"""Independent checkpoint reader shell; numerical adapter needs separate root review.

No import of producer core, SceneAudit, solver, geometry or model is permitted.
The supplied auditor must independently verify raw arrays, never producer flags.
"""
import time
from satellite_rl.coverage.ai_search_r12.schema import SCHEMA, NodeState, COMPLETION_PAIRS, qualified_complete
from .persistence import content_sha, identity

class Incomplete(RuntimeError): pass

LIMITS = {'rays': 4715, 'raypairs': 1000000, 'layouts': 84}

def proof_semantics(node):
    """Raw label and proof qualification remain separate from semantic state."""
    pair=node.get('raw_producer_label'),node.get('proof_qualification')
    if pair not in COMPLETION_PAIRS.get(node.get('state'),()):
        raise Incomplete('contradictory or unqualified raw/proof/semantic state')
    return node['state']

def historical_infeasible(record):
    """Explicit historical vocabulary bridge; never upgrades proof strength.

    Not a production-node parser: R12 nodes still require their exact V1 labels.
    """
    label=record.get('raw_producer_label');proof=record.get('proof_qualification')
    if label not in ('infeasible','graph_infeasible') or not isinstance(proof,str) or not proof or 'unknown' in proof:
        raise Incomplete('historical unknown/missing label or proof cannot normalize')
    return dict(state=NodeState.GRAPH_INFEASIBLE.value,raw_producer_label=label,proof_qualification=proof)

class Reader:
    def __init__(self, journal, *, clock=time.monotonic, prior_seconds=0.,stop=None):
        self.journal, self.clock = journal, clock
        self.stop=stop
        if not isinstance(prior_seconds,(int,float)) or not 0<=prior_seconds<120:raise Incomplete('original reader prior cost not zeroed/reset')
        self.started = clock()
        if journal.has('reader_budget'):
            self.budget = journal.read('reader_budget')
            if self.budget.get('prior_seconds',0.)!=prior_seconds:raise Incomplete('reader prior debit changed')
            if self.budget.get('inflight'):
                raise Incomplete('unknown inflight audit; never automatically repeat GEO')
        else:
            self.budget = dict(rays=0, raypairs=0, layouts=0, seconds=prior_seconds, prior_seconds=prior_seconds,inflight=None)
        self.base_seconds = self.budget['seconds']

    def boundary(self):
        if self.stop is not None and self.stop():raise Incomplete('cooperative reader pause before new node/GEO')

    def _persist_budget(self):
        self.budget['seconds'] = self.base_seconds + self.clock() - self.started
        self.journal.write('reader_budget', self.budget)
        if self.budget['seconds'] > 120:
            raise Incomplete('reader total120s exhausted; committed evidence retained')

    def audit(self, node, *, source_ref, input_ref, parent_cache_SHA,
              static_allowance, independent_auditor):
        self.boundary()
        if node.get('schema') != SCHEMA or not qualified_complete(node):
            raise Incomplete('missing/shared-schema unknown node')
        proof_semantics(node)
        if node.get('source_content_id')!=content_sha({k:v for k,v in node.items() if k!='source_content_id'}):
            raise Incomplete('canonical complete raw node content identity before GEO')
        if not node.get('semantic_SHA') or not parent_cache_SHA:
            raise Incomplete('missing semantic/parent cache identity')
        if set(static_allowance) != set(LIMITS) or any(type(v) is not int or v < 0 for v in static_allowance.values()):
            raise Incomplete('prospective exact geometry allowance required')
        binding = dict(node_SHA=content_sha(node), source=identity(source_ref),
                       input=identity(input_ref), parent_cache_SHA=parent_cache_SHA)
        key = 'audit_' + content_sha(binding)
        if self.journal.has(key):
            receipt=self.journal.read(key)
            if receipt.get('schema')!='R12IndependentAuditCheckpointV2':raise Incomplete('versioned cached audit receipt')
            if content_sha(receipt.get('binding'))!=content_sha(binding):raise Incomplete('cached audit binding changed')
            result=receipt.get('result',{})
            if any(result.get(k) is not True for k in ('passed','geometry_verified','identity_verified')):raise Incomplete('cached audit not validated')
            parent_key='geometry_'+parent_cache_SHA
            if not self.journal.has(parent_key):raise Incomplete('cached audit geometry checkpoint missing')
            latest=self.journal.read(parent_key)
            if latest.get('parent_cache_SHA')!=parent_cache_SHA or latest.get('cache_content_SHA')!=content_sha({k:v for k,v in latest.items() if k!='cache_content_SHA'}):
                raise Incomplete('latest superset geometry cache corrupted')
            refs=result.get('validated_layout_SHA_metrics')
            if not isinstance(refs,dict) or any(latest.get('layouts',{}).get(k)!=v for k,v in refs.items()):
                raise Incomplete('cached layout result references missing/changed')
            exact_refs=result.get('validated_layout_SHA_exact')
            if not isinstance(exact_refs,dict) or any(latest.get('exact',{}).get(k)!=v for k,v in exact_refs.items()):
                raise Incomplete('cached exact layout references missing/changed')
            self._persist_budget()
            return receipt  # durable validated superset cache, zero repeated GEO
        if self.budget['inflight']:
            raise Incomplete('unresolved in-flight audit')
        for name, limit in LIMITS.items():
            if self.budget[name] + static_allowance[name] > limit:
                raise Incomplete('reader prospective ' + name + ' ceiling')
        self._persist_budget()
        # Charge before entering geometry. Failure cannot refund or replay it.
        for name in LIMITS:
            self.budget[name] += static_allowance[name]
        self.budget['inflight'] = key
        self._persist_budget()
        self.boundary()
        parent_key = 'geometry_' + parent_cache_SHA
        cache = self.journal.read(parent_key) if self.journal.has(parent_key) else None
        if cache is not None and cache.get('cache_content_SHA')!=content_sha({k:v for k,v in cache.items() if k!='cache_content_SHA'}):
            raise Incomplete('geometry cache content identity changed')
        result, updated_cache = independent_auditor(node, cache, dict(static_allowance))
        if not isinstance(result, dict) or result.get('passed') is not True:
            raise Incomplete('independent audit failed; raw producer node preserved')
        if result.get('geometry_verified') is not True or result.get('identity_verified') is not True:
            raise Incomplete('independent full geometry and identity required')
        if not isinstance(updated_cache, dict) or updated_cache.get('parent_cache_SHA') != parent_cache_SHA:
            raise Incomplete('same-parent SHA-bound reusable geometry cache required')
        actual = result.get('actual_cost', {})
        if set(actual) != set(LIMITS) or any(type(actual[k]) is not int or not 0 <= actual[k] <= static_allowance[k] for k in LIMITS):
            raise Incomplete('actual geometry cost missing or exceeded')
        receipt = dict(schema='R12IndependentAuditCheckpointV2', binding=binding,
                       result=result, reserved_cost=static_allowance)
        # Geometry and layout audit durable BEFORE cursor progression.
        updated_cache['cache_content_SHA']=content_sha({k:v for k,v in updated_cache.items() if k!='cache_content_SHA'})
        receipt['geometry_cache_content_SHA']=updated_cache['cache_content_SHA']
        self.journal.write(parent_key, updated_cache)
        self.journal.write(key, receipt)
        self.budget['inflight'] = None
        self._persist_budget()
        self.journal.write('reader_cursor', dict(last_completed=key))
        return receipt

def gate(rows):
    """Read only frozen small table; counts supplied only after full reader closure."""
    if len(rows) != 12 or len({r.get('original_index') for r in rows}) != 12:
        raise Incomplete('full12 denominator required')
    from .admission import INDICES
    if set(r['original_index'] for r in rows) != set(INDICES) or any(r.get('complete') is not True for r in rows):
        raise Incomplete('exact full12 closure required')
    names = ('O2_hit_S', 'O2_hit_Q', 'O2_hit_O1', 'F00_hit_Q', 'F01_hit_Q', 'F10_hit_Q', 'F11_hit_Q')
    if any(type(r.get(k)) is not bool for r in rows for k in names):
        raise Incomplete('unknown is not zero')
    counts = {k: sum(r[k] for r in rows) for k in names}
    fixed = max(counts['F' + seq + '_hit_Q'] for seq in ('00', '01', '10', '11'))
    best_fixed=min(('00','01','10','11'),key=lambda seq:(-counts['F'+seq+'_hit_Q'],seq))
    conditions = (counts['O2_hit_S'] >= 3, counts['O2_hit_Q'] >= 3,
                  counts['O2_hit_Q'] - fixed >= 3, counts['O2_hit_O1'] >= 2)
    return dict(counts=counts, best_fixed_hits=fixed, best_fixed_sequence=best_fixed,conditions=conditions,
                passed=all(conditions), denominator=12)
