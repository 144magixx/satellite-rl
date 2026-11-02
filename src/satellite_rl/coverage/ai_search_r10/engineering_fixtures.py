"""Prospective eight synthetic input specifications. No geometry or execution.

These are proposals until separately pinned by the root. Import and exact_specs
only construct literal arrays. No runner is exposed, preventing accidental use
as an implicit engineering/scientific execution admission.
"""
from .core import REQUEST


def _spec(number,positions,owners,anchors,ncap,cap,mode,checks):
    ids=['r10-f%02d-a%02d'%(number,i) for i in range(len(positions))]
    b=max(owners)+1
    return dict(fixture_id=number,case_id='r10-engineering-%02d'%number,
        profile='europe_v1',aircraft_ids=ids,positions=positions,
        demands=[REQUEST]*len(ids),layout=dict(beam_ids=['r10-f%02d-b%02d'%(number,j) for j in range(b)],
        anchor_aircraft_ids=[ids[i] for i in anchors],centers=[positions[i] for i in anchors],
        widths_deg=[.3]*b,assignment=owners),Ncap=ncap,Omega_cap=cap,
        mode=mode,expected_checks=checks,status='prospective_not_executed')


def exact_specs():
    return [
        _spec(1,[[50.,13.],[50.,13.],[50.,13.],[50.,13.]],[0,0,0,1],[0,0],2,.001,
            'common_then_Gd', ['shared anchor ID is permitted; second anchor aircraft is owned elsewhere',
            'q=2; fixed N=2 full flow counts are exactly [2,2]',
            'both beams nonempty; all four stable IDs and integer 21600000 requests conserved',
            'no own-anchor pin; protected strict peak improvement']),
        _spec(2,[[50.,13.],[50.,13.01],[50.,13.02],[50.,13.03],[50.,13.04],[50.,13.05]],
            [0,0,0,0,1,1],[0,4],2,.001,'common_then_Gr',
            ['q=3; counts [3,3]; original stable beam IDs define changed cost',
             'exact integer objective equals one ownership change times Wd plus total quantized D',
             'raw matching and full actual layout retained; all widths floor0.3']),
        _spec(3,[[50.,13.],[50.,13.01],[50.,13.02],[50.,13.03],[50.,13.04]],
            [0,0,0,1,1],[0,3],3,.001,'common_then_Gd_and_S2',
            ['fixed N=2 count_infeasible, no fixed-N flow call',
             'N=3 available; exactly one maximum-source prefix center, no extra source',
             'variable-N finite graph full flow feasible; protection may reject actual [2,2,1] CV',
             'S2 is executed despite fixed-N q count bound, attempts>=1 and every candidate N remains2']),
        _spec(4,[[50.,13.],[50.,13.01],[50.,13.02],[50.,13.03],[50.,13.04]],
            [0,0,0,1,1],[0,3],2,.001,'common_then_Gd',
            ['all target Ns equal2; slots1/2 duplicate slot0, never substitute N=3',
             'count_infeasible without flow entry; exact original B retained; original Ncap remains2']),
        _spec(5,[[50.,5.],[50.,5.],[50.,5.],[50.,21.]],
            [0,0,0,1],[0,3],2,.001,'common_then_Gd',
            ['n=4,N=2,q=2 passes count bound but left three planes only reach left beam',
             'finite frozen graph infeasible; reason is Hall capacity, not resource_unknown',
             'original B and all IDs remain; absence of extra peak-owned distinct coordinate recorded']),
        _spec(6,[[50.,11.5],[50.,11.5],[50.,14.5],[50.,14.5]],
            [0,0,1,1],[0,2],2,.000045,'common_then_one_merge_kernel_and_literal_envelope_boundary',
            ['select one source-nearest 2to1 descriptor, execute exactly one local kernel',
             'matching exists but actual merged width Omega exceeds original .000045 cap',
             'save raw local assignment/selected anchor before width or Omega rejection',
             'same fixture arithmetic probe uses literal maxdistance=.5deg: nextafter(1,+inf)>1 must reject, never clip',
             'arithmetic boundary probe is not a separate geometry or solver call']),
        _spec(7,[[50.,13.],[50.,13.01],[50.,13.02],[50.,13.03]],
            [0,0,0,1],[0,3],3,.001,'two_pool_snapshots_one_2to2_kernel',
            ['generate all4 action types from every current source; reverse 2to2 semantic hashes equal',
             'execute exactly one chosen2to2 kernel, expected admissible [2,2]',
             'rebuild one pool on committed actual layout; same region action has a different physical state semantic key',
             'new beam IDs include original case/stateSHA/semanticSHA/selectedanchorID',
             'pure mock tests separately establish32 commit opportunities and no-progress next-unseen behavior']),
        _spec(8,[[50.,13.],[50.,13.01],[50.,13.02],[50.,13.03]],
            [0,0,0,1],[0,3],2,.001,'common_then_one_Gd_solve_residual_cap0',
            ['use common pinned geometry table, integer Gd costs, and a Ledger tightening residual inspections to0',
             'one solver entry raises ResourceUnknown at first arc inspection',
             'no graph_infeasible result is fabricated; actual graph writes/heap operations before stop retained',
             'exact persisted common B remains sole legal fallback; no retry or second solve']),
    ]


def execution_plan():
    """Prospective exact call limits, no engineering call on construction."""
    return dict(schema='R10EightSyntheticEngineeringPlanV2',fixture_count=8,
        common_calls=8,scene_constructions=8,global_solver_entries_max=5,
        local_kernel_entries_expected=3,inner_R9_flow_entries_expected_max=7,
        total_flow_entries_expected_max=12,model_RULE_D5_T_W_calls=0,
        generic_S2_kernel_safety_guard=32,
        expectation_note='f3 has one physical2beam2to2 region; minimal counts3/2 and floor widths retain exact raw objective, so no commit and only one unique eligible attempt. Production S2 retains its generic32 safety ceiling; that ceiling is not observed work or an expected32 calls.',
        fixture_limits=[
            dict(fixture_id=1,common=1,arms=['Gd'],global_flows=1,local_kernels=0,inner_R9_flows=0),
            dict(fixture_id=2,common=1,arms=['Gr'],global_flows=1,local_kernels=0,inner_R9_flows=0),
            dict(fixture_id=3,common=1,arms=['Gd','S2'],global_flows=1,local_kernels=1,inner_R9_flows=3,generic_S2_kernel_safety_guard=32),
            dict(fixture_id=4,common=1,arms=['Gd'],global_flows=0,local_kernels=0,inner_R9_flows=0),
            dict(fixture_id=5,common=1,arms=['Gd'],global_flows=1,local_kernels=0,inner_R9_flows=0),
            dict(fixture_id=6,common=1,arms=[],global_flows=0,local_kernels=1,inner_R9_flows=1,literal_algebra_probes=1,pool_builds=1),
            dict(fixture_id=7,common=1,arms=[],global_flows=0,local_kernels=1,inner_R9_flows=3,pool_builds=2),
            dict(fixture_id=8,common=1,arms=[],global_flows=1,local_kernels=0,inner_R9_flows=0,residual_cap=0),
        ],
        interpretation='Literal plan only. Separate root source/spec admission is required to call runner. No retry or input/expected-outcome edits after results.')
