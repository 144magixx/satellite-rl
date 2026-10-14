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
