"""Eight exact handwritten inputs. No geometry occurs when freezing specs.

Only a separately admitted engineering worker may call create_fixture/run_one.
Each fixture selects one descriptor, total eight candidate slots, no RULE/D5.
"""
import copy
from types import SimpleNamespace
from .core import ACTIONS,REQUEST,digest

def _spec(number,groups,action,source='region-a',ncap=4,cap=.001,expected='admissible'):
    positions=[];ids=[];owners=[];beamids=[];anchors=[]
    for j,(bid,points) in enumerate(groups):
        beamids.append(bid)
        for point in points:
            aid='f%02d-a%03d'%(number,len(ids));ids.append(aid);positions.append(point);owners.append(j)
            if len(anchors)==j:anchors.append(aid)
    return dict(fixture_id=number,case_id='r9-engineering-%02d'%number,profile='europe_v1',aircraft_ids=ids,positions=positions,demands=[REQUEST]*len(ids),layout=dict(beam_ids=beamids,anchor_aircraft_ids=anchors,centers=[positions[ids.index(a)] for a in anchors],widths_deg=[.3]*len(groups),assignment=owners),Ncap=ncap,Omega_cap=cap,select_source_beam_id=source,select_action=action,expected_status=expected)

def exact_specs():
    close=[[50.,13.+i*.01] for i in range(9)];outside=[[52.,20.]]*10
    return [
        _spec(1,[('region-a',close[:6])],ACTIONS[0][0]),
        _spec(2,[('region-a',close[:4]),('region-b',close[4:6]),('outside',outside[:2])],ACTIONS[1][0]),
        _spec(3,[('region-a',close[:6]),('region-b',close[6:9])],ACTIONS[2][0]),
        _spec(4,[('region-a',close[:2]),('region-b',close[2:4]),('outside',outside)],ACTIONS[3][0]),
        _spec(5,[('region-a',[[50.,5.]]*2),('region-b',[[50.,21.]]*2)],ACTIONS[3][0],expected='rejected'),
        _spec(6,[('region-a',[[50.,11.5]]*2),('region-b',[[50.,14.5]]*2)],ACTIONS[3][0],cap=.000045,expected='rejected'),
        _spec(7,[('region-a',close[:4]),('region-b',close[4:6]),('outside',outside)],ACTIONS[1][0]),
        _spec(8,[('region-a',[[50.,13.]]*6),('region-b',[[50.,13.]]*3)],ACTIONS[2][0]),
    ]

def create_fixture(spec):
    from ..ai_search.core import Scene,Layout
    import numpy as np
    scene=Scene(np.asarray(spec['positions']),np.asarray(spec['demands']),tuple(spec['aircraft_ids']),spec['profile'])
    identity=dict(case_id=spec['case_id'],mother_id=spec['case_id'],profile='uniform21600000',role='train',input_sha256=scene.input_sha256,identity_sha256=digest([spec['case_id'],'identity']),shared_case_content_sha256=digest(spec),source_asset_SHA=digest([spec['case_id'],'synthetic-source']))
    layout=copy.deepcopy(spec['layout']);prepared=SimpleNamespace(identity=identity,status='available',Wenv=Layout.from_record(layout),anchors=tuple(layout['anchor_aircraft_ids']),Ncap=spec['Ncap'],Omega_cap=spec['Omega_cap'])
    rule=dict(status='success',layout=copy.deepcopy(layout),**identity)
    return scene,prepared,rule

def run_one(spec):
    from .core import select_origin,build_pool,evaluate_candidate
    scene,prepared,rule=create_fixture(spec);origin=select_origin(scene,prepared,rule);pool=build_pool(scene,origin)
    selected=[d for d in pool['descriptors'] if d['source_beam_id']==spec['select_source_beam_id'] and d['action']==spec['select_action']]
    if len(selected)!=1:raise AssertionError('one exact descriptor')
    result=evaluate_candidate(scene,origin,selected[0])
    if result['status']!=spec['expected_status']:raise AssertionError((spec['fixture_id'],result['status'],result['reason']))
    if spec['fixture_id']==5 and result['reason']!='fixed_anchor_capacity_infeasible':raise AssertionError('declared infeasible fixed adjacency')
    if spec['fixture_id']==6 and result['reason']!='whole_Omega_cap':raise AssertionError('declared Omega rejection')
    if spec['fixture_id']==7 and selected[0]['global_peak_lower_count']!=10:raise AssertionError('outside global lower bound')
    if spec['fixture_id']==8 and result['selected_anchor_ids']!=spec['aircraft_ids'][:3]:raise AssertionError('exact coincident stable-ID anchor ties')
    return dict(fixture_id=spec['fixture_id'],origin=origin,pool=pool,result=result,new_RULE_D5_W_T_model_calls=0,actual_candidate_slots=1)
