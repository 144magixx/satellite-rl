"""Only the prospectively specified R11 Gh arm; saved R10 common is mandatory."""
import copy
import time
from ..ai_search_r10.flow import Ledger
from ..ai_search_r10.costs import edge_cost
from ..ai_search_r10.core import _common_valid, _layout_global
from ..ai_search_r9.core import require, audit, protected, objective
from .flow import solve

DESIGN_SHA='6a4fd42f6510c86ef9bd5db881374a8a426e5493078ee8d7d159a0f73e67a2f1'
METHODS=('Gh',)

def cost_matrix(distances,eligible,bids,ownerids,refs,family,ledger):
    """Original Gr scalar formula; per-entry construction charged before work."""
    n,b=len(ownerids),len(bids)
    ledger.scan('cost_full_table_entries',n*b)
    result=[]
    for i in range(n):
        row=[]
        for j in range(b):
            if eligible[i][j]:
                ledger.tick('bigint_multiplications',2)
                ledger.tick('bigint_additions_subtractions',2)
                row.append(edge_cost(distances[i][j],refs[j],bids[j]!=ownerids[i],n,'Gr'))
            else:row.append(0)
        result.append(row)
    return result


def _global(scene,common,method,ledger,observations=None):
    origin=common['origin']
    results=[]
    best=dict(terminal=copy.deepcopy(origin['layout']),metrics=copy.deepcopy(origin['metrics']),position=-1,status='incumbent')
    if observations is not None:observations.update(results=results,selected=best)
    b=len(origin['layout']['beam_ids'])
    ownerids=[origin['layout']['beam_ids'][i] for i in origin['layout']['assignment']]
    for slot in common['N_slots']:
        record=copy.deepcopy(slot)
        results.append(record)
        ledger.tick('global_structure_slots')
        if slot['status']!='available':
            ledger.tick('global_'+slot['status'])
            continue
        record['status']='entered_unknown'
        target=slot['target_N']
        prefix=common['added_center_prefix'][:target-b]
        bids=list(origin['layout']['beam_ids'])+[x['beam_id'] for x in prefix]
        refs=list(origin['layout']['widths_deg'])+[.3]*len(prefix)
        ledger.scan('global_structure_slice',common['n']*target)
        distances=[row[:target] for row in common['distances_deg']]
        adjacency=[row[:target] for row in common['adjacency']]
        costs=cost_matrix(distances,adjacency,bids,ownerids,refs,'Gr',ledger)
        flow=solve(tuple(scene.aircraft_ids),bids,adjacency,costs,common['q'],ledger,ncap=origin['Ncap'])
        record['flow']=flow
        if not flow['feasible']:
            record['status']='graph_infeasible'
            continue
        layout,reason=_layout_global(scene,common,slot,flow['assignment'],ledger)
        record['raw_layout']=layout
        if reason:
            record.update(status='rejected',reason=reason)
            continue
        metrics=audit(scene,layout,origin['Ncap'],origin['Omega_cap'],ledger)
        record['raw_metrics']=metrics
        if not protected(metrics,origin['metrics']):
            record.update(status='rejected',reason='common_B_quality_protection')
            continue
        record.update(status='protected',terminal=layout,metrics=metrics)
        if objective(metrics)+(slot['position'],)<objective(best['metrics'])+(best['position'],):
            best=record
            if observations is not None:observations['selected']=best
    return dict(results=results,selected=best,global_matching_slots=len(results))


def run_method(scene,common,method='Gh',*,stop=None):
    """Saved-common single arm. Native caller guards input/IO plus 300s/2GiB.

    stop must be a lightweight in-memory/time predicate, never per-arc disk IO.
    Unknown exceptions retain observed partial work; no retry or free fallback.
    """
    require(method=='Gh','only frozen Gh arm')
    _common_valid(scene,common)
    began=time.monotonic()
    ledger=Ledger(common['n'],stop=stop)
    observations=dict(complete=False,partial_status='unknown',method='Gh',
        selected=dict(terminal=copy.deepcopy(common['origin']['layout']),
        metrics=copy.deepcopy(common['origin']['metrics']),position=-1,status='incumbent'))
    try:
        result=_global(scene,common,method,ledger,observations)
    except Exception as error:
        error.partial_ledger=ledger.record()
        error.partial_result=copy.deepcopy(observations)
        raise
    result.update(schema='R11GhMethodV1',method='Gh',design_sha256=DESIGN_SHA,
        complete=True,identity=copy.deepcopy(common['identity']),
        original_index=common['original_index'],common_SHA=common['common_SHA'],
        origin_SHA=common['origin_SHA'],q=common['q'],partial_status=None,
        ledger=ledger.record(),historical_common_ledger=copy.deepcopy(common['common_ledger']),
        historical_common_reexecuted=False,observed_method_seconds=time.monotonic()-began,
        new_RULE_D5_T_W_model_calls=0)
    return result
