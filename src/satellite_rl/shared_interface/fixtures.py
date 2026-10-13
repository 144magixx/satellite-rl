"""Prepare exact R7 v2 hand-written recipe objects, WITHOUT running qualification."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from .content import seal,content_sha,wire_bytes
from .contracts import Authority,CONSTRAINT_VERSION
from .geometry import GEOMETRY_SHA
from .spherical_gaussian import PHYSICS,PHYSICS_SHA
from .stages import make_case,make_layout

DESIGN_SHA='603e0a1e2e075bf714702ba4b1b68adef78639c2cc25529633da5d133996a916'
ACCEPTANCE_SHA='eaede816edfd2c3163bfa4567381a313fa9cd5be9ad73a759c813e5988875e6a'
COUNTS=dict(roundtrips=36,coefficient_builds=12,capacity_evaluations=12,pattern_points=60,normalization_integrals=3)


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_recipe(spec):
    """Static array construction and specified Omega expressions only.

    No visible/coverage check, scalar pattern, coefficient/capacity, optimizer,
    dataset/teacher/decoder invocation occurs in this preparation function.
    """
    ident=spec['id'];N=spec['N'];ids=[f'{ident}:a{i:03d}' for i in range(N)];beams=[f'{ident}:b{i:03d}' for i in range(N)]
    positions=spec.get('positions',[[52.,8.]]*N)
    widths=spec.get('widths',[.3,.5,1.]*((N+2)//3) if spec.get('widths_rule')=='[.3,.5,1.0][i%3]' else [.5]*N)[:N]
    case=make_case(dict(case_id=ident,actual_role='synthetic_engineering',source_role='synthetic_engineering',
        original_mother_id=ident,UTC='2026-10-13T00:00:00Z',source_asset_SHA=DESIGN_SHA,
        source=dict(synthetic=True,recipe_id=ident,recipe_contract_sha=DESIGN_SHA)),ids,positions)
    cap=float(N*2*np.pi*(1-np.cos(np.deg2rad(.5))))
    omega=float(np.sum(2*np.pi*(1-np.cos(np.deg2rad(widths)/2))))
    layout=make_layout(case,beam_ids=beams,source_slot_ids=beams,anchor_aircraft_ids=ids,centres_lat_lon_deg=positions,
        full_width_deg=widths,owner_aircraft_to_beam_id=[[a,b] for a,b in zip(ids,beams)],Ncap=N,
        geometry_Omega_cap_sr=cap,geometry_Omega_sr=omega,geometry_audit=dict(status='not_executed_recipe'),
        source_and_cost=dict(synthetic=True,recipe_contract_sha=DESIGN_SHA,planner_calls=0,teacher_calls=0,decoder_calls=0))
    starts=np.zeros(N,np.int64);counts=np.array([1 if i<8 else 0 for i in range(N)],np.int64);powers=np.array([10. if i<8 else 0. for i in range(N)])
    if 'allocation' in spec:
        for row in spec['allocation']:
            i=row['beam_index'];starts[i]=row['start'];counts[i]=row['count'];powers[i]=row['power_w']
    if spec.get('allocation_rule')=='onlyb000_b008_active_start0_count1_power10;others_empty;group_i_mod8':
        counts[:]=0;powers[:]=0;counts[[0,8]]=1;powers[[0,8]]=10.
    if spec.get('allocation_rule')=='beam_i_start_i_count1_power10':starts=np.arange(N,dtype=np.int64);counts[:]=1;powers[:]=10.
    allocation=seal(dict(schema='DiscreteAllocationV1',parent_case_SHA=case['content_sha256'],parent_layout_SHA=layout['content_sha256'],
        physics_config_SHA=PHYSICS_SHA,beam_ids=beams.copy(),processing_index_to_beam_id=beams.copy(),
        group_by_beam_id={bid:i%8 for i,bid in enumerate(beams)},actual_start_slot=starts,actual_slot_count=counts,
        actual_whole_beam_power_w=powers,requested_actions_separate=dict(status='hand_specified_actual_fixture_no_requested_policy'),
        constraint_version=CONSTRAINT_VERSION,units=dict(slot='integer25MHz_slot',power='whole_beam_watt')))
    original=copy.deepcopy(dict(case=case,layout=layout,allocation=allocation))
    if 'allocation_row_permutation' in spec:
        permutation=spec['allocation_row_permutation'];allocation['beam_ids']=[beams[i] for i in permutation]
        for key in ('actual_start_slot','actual_slot_count','actual_whole_beam_power_w'):allocation[key]=allocation[key][permutation].copy()
        allocation=seal(allocation)
    mutation=spec.get('mutation')
    if mutation=='delete_owner_pair_for_a002_parentcase_unchanged':layout['owner_aircraft_to_beam_id'].pop(2);layout=seal(layout)
    elif mutation=='third_owner_pair_aircraftID_a002_to_a000;duplicate_a000_missing_a002':layout['owner_aircraft_to_beam_id'][2][0]=ids[0];layout=seal(layout)
    elif mutation=='case_request0=21599999;update_sum_and_selfhash;fixed_request_bps_still21600000':case['request_bps_int64'][0]=21599999;case['total_request_bps_int64']=21599999;case=seal(case)
    elif mutation=='delete_layout_full_width_deg':layout.pop('full_width_deg');layout=seal(layout)
    registry=dict(schema='R7PublishedParentRegistryV1',case_sha_by_id={ident:original['case']['content_sha256']},
        layout_shas=[original['layout']['content_sha256']],allocation_shas=sorted(set([original['allocation']['content_sha256'],allocation['content_sha256']])),source=dict(recipe_contract_sha=DESIGN_SHA))
    return dict(case=case,layout=layout,allocation=allocation,original_authoritative_objects=original,
        authority_registry=registry,authority_registry_sha=content_sha(registry),expected=spec['expected'],recipe=spec)


def write_plan(design_path,acceptance_path,directory):
    if file_sha(design_path)!=DESIGN_SHA or file_sha(acceptance_path)!=ACCEPTANCE_SHA:raise ValueError('root accepted exact design required')
    root=Path(directory)
    if root.exists():raise ValueError('new fixed qualification plan directory only')
    root.mkdir(parents=True);design=json.loads(Path(design_path).read_text(encoding='utf8'))
    recipes=design['interface_qualification']['fixed_fixtures']
    if len(recipes)!=12 or design['interface_qualification']['planned_counts']!=COUNTS:raise ValueError('frozen exact12/counts')
    members={}
    for spec in recipes:
        objects=object_recipe(spec);folder=root/spec['id'];folder.mkdir()
        for name in ('case','layout','allocation','authority_registry'):
            path=folder/(name+'.json');path.write_bytes(wire_bytes(seal(objects[name]) if name=='authority_registry' else objects[name]));members[path.relative_to(root).as_posix()]=file_sha(path)
        canonical_path=folder/'allocation_canonical.json';canonical_path.write_bytes(wire_bytes(objects['original_authoritative_objects']['allocation']));members[canonical_path.relative_to(root).as_posix()]=file_sha(canonical_path)
        (folder/'expected.json').write_text(json.dumps(dict(expected=objects['expected'],recipe=spec,
            authority_registry_sha=objects['authority_registry_sha']),sort_keys=True,indent=2),encoding='utf8')
        members[(folder/'expected.json').relative_to(root).as_posix()]=file_sha(folder/'expected.json')
    source_root=Path(__file__).resolve().parent
    chain={name:file_sha(source_root/name) for name in ('__init__.py','content.py','contracts.py','geometry.py','spherical_gaussian.py','link.py','stages.py','fixtures.py','qualification.py')}
    plan=dict(schema='R7InterfaceQualificationPlanV1',design_sha=DESIGN_SHA,root_acceptance_sha=ACCEPTANCE_SHA,
        fixture_count=12,planned_counts=COUNTS,fixture_members=members,caller_chain_source_sha256=chain,
        actual_callers=['satellite_rl.shared_interface.stages.s1_publish_import',
            'satellite_rl.shared_interface.stages.UniformStage2Input.from_wire->export_actual_allocation->canonical_bytes_compare->terminal_fixed_evaluation',
            'frozen legal allocation permutation of actual S2 exported record->UniformStage3FixedInput.from_wire->fixed_initial_evaluation'],
        count_definition='36 roundtrips=12fixedfixtures*3 stage object-graph serialization/reception checks; actual object read_wire calls separately reported; S2 export is part of its stage object graph not a new case',
        setup_operations='hand-specified arrays/hash and12 cap/actualOmega expressions; no visibility/coverage/pattern/PHY call',
        execution_counts=dict(roundtrips=0,coefficient_builds=0,capacity_evaluations=0,pattern_points=0,normalization_integrals=0),
        no_new_data_labels_planning_optimizer_model_solver=True)
    (root/'plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True),encoding='utf8');return plan
