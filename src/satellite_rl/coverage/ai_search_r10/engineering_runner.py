"""Single-pass runner for separately admitted R10 exact eight fixtures.

No main entry, no file/network/dispatch side effects. Merely importing this
module or inspecting its plan performs no geometry or flow. Caller must pin
the exact specs/plan/source and obtain root engineering admission first.
"""
import copy
import numpy as np
from . import core
from .costs import matrix as cost_matrix
from .flow import Ledger, ResourceUnknown, solve
from .engineering_fixtures import exact_specs,execution_plan


def _expected_residual_cap_stop(error,ledger):
    return (str(error)=='residual_edge_inspections_cap' and ledger.data['flow_queries']==1 and
            ledger.data['graph_arc_initialization_writes']>0 and ledger.data['heap_pushes_plus_pops']>0)


def _create(spec):
    from ..ai_search.core import Scene
    scene=Scene(np.asarray(spec['positions'],np.float64),np.asarray(spec['demands'],np.float64),
                tuple(spec['aircraft_ids']),spec['profile'])
    identity=dict(case_id=spec['case_id'],mother_id=spec['case_id'],profile='uniform21600000',role='train',
        input_sha256=scene.input_sha256,identity_sha256=core.digest([spec['case_id'],'synthetic_identity']),
        shared_case_content_sha256=core.digest(spec),source_asset_SHA=core.digest([spec['case_id'],'synthetic_source']))
    layout=copy.deepcopy(spec['layout'])
    counts=np.bincount(layout['assignment'],minlength=len(layout['beam_ids']))
    origin=dict(identity=identity,layout=layout,metrics=core.quality(counts,layout['widths_deg']),
        Ncap=spec['Ncap'],Omega_cap=spec['Omega_cap'],layout_SHA=core.digest(layout),
        source_refs=[dict(role='frozen_synthetic_literal_spec',sha256=core.digest(spec))])
    return scene,origin


def run_one(spec,*,admitted_spec_sha256,stop=None):
    """Exactly the frozen mode once, with completed/failed observations retained."""
    number=spec['fixture_id']
    expected=exact_specs()[number-1]
    if spec!=expected or core.digest(spec)!=admitted_spec_sha256:
        raise ValueError('literal synthetic spec differs from pre-result admission')
    limits=execution_plan()['fixture_limits'][number-1]
    observations=dict(fixture_id=number,spec=copy.deepcopy(spec),limits=limits,complete=False,
        scene_constructions=0,common_calls=0,global_flow_entries=0,local_kernel_entries=0,inner_R9_flow_entries=0)
    try:
        scene,origin=_create(spec)
        observations['scene_constructions']=1
        observations['origin']=origin
        observations['common_calls']=1
        common=core.prepare_common(scene,origin,stop=stop)
        observations['common']=common
        outputs=[]
        observations['method_results']=outputs
        for method in limits['arms']:
            observations['active_method']=method
            result=core.run_method(scene,common,method,stop=stop)
            outputs.append(result)
            flows=result['ledger']['flow_queries']
            if method in ('Gd','Gr','Gw'):observations['global_flow_entries']+=flows
            else:
                observations['local_kernel_entries']+=result['kernel_attempts']
                observations['inner_R9_flow_entries']+=flows
            observations.pop('active_method',None)
        if number in (6,7):
            outer=Ledger(common['n'],stop=stop)
            current=dict(layout=copy.deepcopy(origin['layout']),metrics=copy.deepcopy(origin['metrics']))
            pool=core._pool(scene,common,current,'SH-C',outer)
            action=core.ACTIONS[3 if number==6 else 1][0]
            selected=next(d for d in pool if d['action']==action and d['structurally_eligible'])
            observations.update(initial_pool=pool,local_kernel_entries=1)
            trial=core._kernel(scene,common,current,selected,outer)
            observations['kernel_result']=trial
            observations['inner_R9_flow_entries']=outer.data['flow_queries']
            if number==6:
                if trial['reason']!='whole_Omega_cap':raise AssertionError('fixed Omega rejection')
                required=2*.5
                outward=float(np.nextafter(required,np.inf))
                observations['literal_envelope_probe']=dict(distance_deg=.5,required_fullwidth=required,
                    outward_fullwidth=outward,rejected=outward>1.,geometry_calls=0,solver_calls=0)
                if not outward>1.:raise AssertionError('nextafter boundary must not clip')
            else:
                if trial['status']!='admissible':raise AssertionError('frozen single2to2 improvement')
                directions=[d for d in pool if d['action']==core.ACTIONS[1][0]]
                if len(directions)!=2 or len({d['semantic_SHA'] for d in directions})!=1:
                    raise AssertionError('reverse semantic deduplication')
                current=dict(layout=trial['terminal'],metrics=trial['metrics'])
                observations['committed_pool']=core._pool(scene,common,current,'SH-C',outer)
                if core.physical_state(scene,current['layout'],common['identity']['case_id'])==selected['state_SHA']:
                    raise AssertionError('new input state must differ')
                for bid in trial['new_beam_lineage']['new_ids']:
                    if selected['state_SHA'] not in bid or selected['semantic_SHA'] not in bid:
                        raise AssertionError('new ID binds state and semantic descriptor')
            observations['direct_ledger']=outer.record()
        if number==8:
            slot=next(s for s in common['N_slots'] if s['status']=='available')
            b=slot['target_N']
            layout=origin['layout']
            bids=layout['beam_ids']
            ownerids=[bids[o] for o in layout['assignment']]
            ledger=Ledger(common['n'],caps={'residual_edge_inspections':0},stop=stop)
            adjacency=[row[:b] for row in common['adjacency']]
            distance=[row[:b] for row in common['distances_deg']]
            costs=cost_matrix(distance,adjacency,bids,ownerids,layout['widths_deg'],'Gd',ledger)
            observations['global_flow_entries']=1
            try:solve(scene.aircraft_ids,bids,adjacency,costs,common['q'],ledger)
            except ResourceUnknown as error:
                if not _expected_residual_cap_stop(error,ledger):
                    raise
                observations['expected_unknown']=dict(reason=str(error),ledger=ledger.record(),
                    terminal=copy.deepcopy(origin['layout']),metrics=copy.deepcopy(origin['metrics']),complete=False)
                if ledger.data['flow_results']:raise AssertionError('unknown cannot be graph infeasible')
            else:raise AssertionError('residual0 must stop')
        if number==1:
            if outputs[0]['results'][0]['flow']['counts']!=[2,2]:raise AssertionError('lower one and q occupancy')
        if number==2:
            flow=outputs[0]['results'][0]['flow']
            assignment=flow['assignment']
            changes=sum(origin['layout']['beam_ids'][owner]!=origin['layout']['beam_ids'][old]
                        for owner,old in zip(assignment,origin['layout']['assignment']))
            if changes!=1:raise AssertionError('Gr lex primary changes')
        if number==3:
            if outputs[0]['results'][0]['status']!='count_infeasible' or outputs[1]['kernel_attempts']!=1:
                raise AssertionError('fixed count bound cannot prune S2')
        if number==4 and observations['global_flow_entries']!=0:raise AssertionError('original Ncap no replacement')
        if number==5 and outputs[0]['results'][0]['status']!='graph_infeasible':raise AssertionError('finite Hall infeasibility')
        if (observations['global_flow_entries']>limits['global_flows'] or
            observations['local_kernel_entries']>limits['local_kernels'] or
            observations['inner_R9_flow_entries']>limits['inner_R9_flows']):
            raise AssertionError('frozen engineering call cap exceeded; no retry')
        observations.update(complete=True,passed=True,new_RULE_D5_T_W_model_calls=0)
        return observations
    except Exception as error:
        observations['entry_counts_status']='observed_lower_bounds_only_on_interruption; missing entry counts are not evidence of zero'
        observations['interrupted_core']={k:copy.deepcopy(getattr(error,k))
            for k in ('partial_result','partial_ledger','partial_kernel') if hasattr(error,k)}
        error.engineering_observations=observations
        raise
