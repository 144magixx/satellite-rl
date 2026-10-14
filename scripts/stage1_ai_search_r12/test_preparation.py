"""Pure hand-written records only: zero geometry/flow/native/remote calls."""
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timezone
from .persistence import Journal, identity, content_sha
from .reader import Reader, Incomplete, gate, proof_semantics, historical_infeasible
from .admission import check, INDICES
from satellite_rl.coverage.ai_search_r12.schema import SCHEMA

def math_binding():
    from .admission import EXECUTION_DESIGN_VERSION,DESIGN_SHA,DESIGN_ACCEPTANCE_SHA
    return dict(mathematical_design_version=EXECUTION_DESIGN_VERSION,design_ref=dict(sha256=DESIGN_SHA),root_design_acceptance_ref=dict(sha256=DESIGN_ACCEPTANCE_SHA))

class PreparationTests(unittest.TestCase):
    def test_existing_null_or_malformed_pause_marker_never_absent(self):
        from .worker import _markers
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'stop.request';plan=dict(watched_pause_paths=[str(path)])
            self.assertIsNone(_markers(plan)[str(path)])
            path.write_text('null',encoding='utf8')
            self.assertIsNotNone(_markers(plan)[str(path)])
            path.write_text('{',encoding='utf8')
            with self.assertRaises(ValueError):_markers(plan)

    def test_reader_marker_stops_before_new_callback_without_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            journal=Journal(temp);reader=Reader(journal,prior_seconds=10,stop=lambda:True)
            with self.assertRaisesRegex(Incomplete,'pause'):
                reader.audit({},source_ref={},input_ref={},parent_cache_SHA='x',static_allowance={},
                             independent_auditor=lambda *args:self.fail('new GEO callback after stop'))
            self.assertFalse(journal.has('reader_cursor'));journal.close()

    def test_v2_contact_domain_and_not_pruning_diagnostic(self):
        from .region_v2 import select,domain
        counts=[5,5,1,3];ids=['a','b','c','d']
        result=select(counts,ids,{2:[(2.,2.),(2.,2.)],3:[(.1,.1),(.2,.2)]},1.)
        self.assertEqual(result['region'],[0,1,3]);self.assertEqual(result['support'],3)
        self.assertFalse(3<=5-2-1)  # diagnostic false does not prune chosen support
        self.assertEqual(domain([2]*31+[1])[1],[])
        self.assertEqual(select([2,2],['a','b'],{},1.)['reason'],'unsupported_no_slack_support')

    def test_v2_exact_permutations_and_float_display_noise(self):
        from .exact_order import components,key
        a=components([3,2,1],[1.,1e-16,1e-16]);b=components([1,3,2],[1e-16,1.,1e-16])
        self.assertEqual(a,b);self.assertEqual(key(a),key(b))
        # CV and float Omega display perturbations do not appear in this key.
        display_a=dict(CV=.2,actual_Omega=1.);display_b=dict(CV=.2-1e-16,actual_Omega=1.-1e-16)
        self.assertNotEqual(display_a,display_b);self.assertFalse(key(b)<key(a))

    def test_v2_rational_tail_and_cross_N_NU(self):
        from .exact_order import components,key
        a=components([4,3]+[1]*9,[.1]*11);b=components([4,4]+[1]*8,[.1]*10)
        self.assertEqual(a['n'],b['n']);self.assertGreater(a['T'],b['T']);self.assertLess(key(a),key(b))
        c=components([4,4],[.1]*2);d=components([3,3,2],[.1]*3)
        self.assertLess(d['U'],c['U']);self.assertGreater(d['NU'],c['NU'])

    def test_v2_static_contact_and_member_cache_bound(self):
        from .region_v2 import domain,reader_pair_reservation
        p,c=domain([2]*30+[1]*170);self.assertEqual(len(p)*len(c),5100)
        for N in range(1,201):
            for s in range(1,min(30,N)+1):self.assertLessEqual(s*(N-s),5100)
        self.assertEqual(36*128**2+36*5100+72*128+12*4715,839220)
        self.assertLess(839220,1000000)
        first=reader_pair_reservation(293,61,new_contact=True,new_region=True,eligible=True)
        reused=reader_pair_reservation(293,61,new_contact=False,new_region=False,eligible=True)
        self.assertEqual(first-reused,5100+61*61)

    def test_explicit_9537_old804_transport_alias_no_recursive_open(self):
        from .historical_s import audited_table
        sha='804dd2e1cbaa264ceda230d153bb2e5b29fb10ab0703c7745aeef1cac1ec49e5'
        value=dict(schema='R12OldStrongSAuditedCountsV1',complete=True,old_control_ref=dict(path='D:/historical/old804.json',sha256=sha,bytes=3132796),
            source_plan_ref=dict(path='D:/historical/unavailable-source-plan.json'),
            reader_budget=dict(original_seconds=120,IO_subcap_seconds=10,conservative_IO_debit_seconds=10,remaining_reader_hard_seconds=110),
            new_geometry_calls=0,new_flow_calls=0,new_model_calls=0,
            rows=[dict(original_index=i,aircraft_count=1 if p<11 else 4704) for p,i in enumerate(INDICES)])
        current=dict(path='/owned/new73/config/old_controls_804.json',sha256=sha,bytes=3132796)
        self.assertEqual(len(audited_table(value,10,current)),12)
        with self.assertRaises(ValueError):audited_table(value,0,current)
        with self.assertRaises(ValueError):audited_table(value,10,dict(current,bytes=1))

    def test_v2_AST_project_import_closure(self):
        from .dependency_closure import project_closure
        closure=project_closure(Path(__file__).resolve().parents[2])
        self.assertTrue({'src/satellite_rl/coverage/scenes.py','src/satellite_rl/coverage/environment.py','src/satellite_rl/coverage/baselines.py',
                         'scripts/stage1_ai_search_r12/exact_order.py','scripts/stage1_ai_search_r12/historical_s.py'}<=set(closure))
        self.assertFalse(any('ai_search_r11' in name for name in closure))
    def test_IO_adapter_never_generates_or_copies_rays(self):
        from .normalize import normalize_one,CONTROL_SHA
        ids=['a','b','c'];positions=[[0.,0.],[1.,0.],[2.,0.]]
        layout=dict(beam_ids=['x','y'],anchor_aircraft_ids=['a','c'],centers=[positions[0],positions[2]],widths_deg=[.3,.3],assignment=[0,0,1])
        ident=dict(case_id='pure',mother_id='pure',profile='uniform21600000',role='train',input_sha256='a'*64,
            identity_sha256='b'*64,shared_case_content_sha256='c'*64,source_asset_SHA='d'*64)
        metric=dict(N=2,maxload=43200000,top10_mean=43200000.,CV=.3,actual_Omega=.0001,total_request=64800000)
        origin=dict(identity=ident,layout=layout,layout_SHA=content_sha(layout),metrics=metric,Ncap=2,Omega_cap=1.)
        original=dict(schema='R10ExistingSceneAndBestOriginInputV1',original_index=0,origin=origin,
            scene=dict(aircraft_ids=ids,positions=positions,demands=[21600000]*3,rays=[[1,0,0]]*3,input_sha256='a'*64,profile='europe_v1'))
        arm=dict(state='complete',whole_method_unknown=False,metrics=metric,layout_SHA='e'*64)
        row=dict(original_index=0,original_aircraft_count=3,common_state='complete',caps=dict(Ncap=2,Omega_cap=1.),origin_metrics=metric,arms={'S2':arm,'SH-C':arm})
        spec=dict(original_index=0,aircraft_count=3,identity=ident,B_layout_sha256=origin['layout_SHA'],Ncap=2,Omega_cap=1.,control_row_index=0,input_ref=dict(path='original',sha256='f'*64,bytes=1))
        adapter=normalize_one(original,row,spec,dict(path='old804',sha256=CONTROL_SHA,bytes=1),physical_parameters=dict(earth_radius_km=6371,orbit_radius_km=42164,satellite_longitude_deg=13.),geometry_profile='europe_v1')
        self.assertNotIn('scene',adapter);self.assertNotIn('origin',adapter)
        self.assertEqual(adapter['expected_B_layout_SHA'],origin['layout_SHA'])
        self.assertFalse(adapter['historical_rays_source']['used_as_new_production_rays'])
        self.assertEqual(original['scene']['rays'],[[1,0,0]]*3)

    def test_engineering_review_cannot_use_science_admission(self):
        from .engineering import validate
        with tempfile.TemporaryDirectory() as temp:
            from .admission import GLOBAL_PAUSE,PORT_PAUSES
            plan=dict(**math_binding(),schema='R12EngineeringPlanV2',group='S1-AI17-R12',executor_port=20473,
                pause_deadline_utc='2026-10-14T23:00:00Z',watched_pause_paths=[str(Path(temp)/'left'),str(Path(temp)/'right')])
            plan['own_stop_marker']=str(Path(temp)/'own_stop');plan['watched_pause_paths']=[GLOBAL_PAUSE,*PORT_PAUSES,plan['own_stop_marker']]
            with self.assertRaises(ValueError):validate(plan,dict(schema='RootR12ExecutionAdmissionV2'),datetime(2026,10,14,19,tzinfo=timezone.utc))

    def test_single_workbench_intent_never_replayed(self):
        from .server import Endpoint
        with tempfile.TemporaryDirectory() as temp:
            endpoint=Endpoint.__new__(Endpoint);endpoint.journal=Journal(temp)
            endpoint.plan=dict(request_id='fixed',content_SHA='a'*64);endpoint.path='/exact'
            endpoint.journal.write('submission_intent',dict(unknown=True))
            with self.assertRaises(ValueError):endpoint.post('/exact/start',dict(request_id='fixed',plan_content_SHA='a'*64),popen=lambda *a,**kw:self.fail('must never launch'))
            endpoint.journal.close()

    def test_historical_semantics_preserve_proof_not_unknown(self):
        a=historical_infeasible(dict(raw_producer_label='infeasible',proof_qualification='reported_without_cut'))
        b=historical_infeasible(dict(raw_producer_label='graph_infeasible',proof_qualification='reported_without_cut'))
        self.assertEqual(a['state'],b['state'])
        self.assertNotEqual(a['raw_producer_label'],b['raw_producer_label'])
        self.assertEqual(a['proof_qualification'],'reported_without_cut')
        for raw in ({},dict(raw_producer_label='unknown',proof_qualification='reported_without_cut')):
            with self.assertRaises(Incomplete):historical_infeasible(raw)
        with self.assertRaises(Incomplete):proof_semantics(dict(state='admissible',raw_producer_label='known_unavailable',proof_qualification='analytic_structure_bound'))

    def test_pure_cut_certificate_without_flow(self):
        from .geometry_reader import verify_flow_proofs
        members=['a','b','c','d'];anchors=['a','b']
        adjacency=[[True,False],[False,True],[True,False],[True,False]]
        assignment=[0,1,0,0]
        queries=[dict(q=4,feasible=True,assignment=assignment),dict(q=3,feasible=True,assignment=assignment),dict(q=2,feasible=False,
            proof=dict(kind='source_sink_cut_v1',reachable_free_member_ids=['c','d'],
                       reachable_anchor_ids=['a'],crossing_free_anchor_arcs=0,cut_capacity=1,required_free=2))]
        self.assertTrue(verify_flow_proofs(members,anchors,adjacency,queries,3,assignment))
        queries[2]['proof']['cut_capacity']=2
        with self.assertRaises(Incomplete):verify_flow_proofs(members,anchors,adjacency,queries,3,assignment)

    def test_transport_identity(self):
        self.assertEqual(identity(dict(path='left',sha256='a'*64,bytes=8)),
                         identity(dict(path='right',sha256='a'*64,bytes=8)))

    def test_complete_checkpoint_cached(self):
        node=dict(schema=SCHEMA,state='graph_infeasible',complete=True,
            semantic_SHA='b'*64,raw_producer_label='fixed_anchor_capacity_infeasible',
            proof_qualification='complete_pinned_graph_hi_m_infeasible',terminal={},metrics={},ledger={},exact_evidence={})
        node['source_content_id']=content_sha(node)
        with tempfile.TemporaryDirectory() as temp:
            journal=Journal(temp);reader=Reader(journal);calls=[]
            def mock(node,cache,allowance):
                calls.append(1)
                return dict(passed=True,geometry_verified=True,identity_verified=True,
                            validated_layout_SHA_metrics={},validated_layout_SHA_exact={},actual_cost=dict(rays=0,raypairs=0,layouts=0)),dict(parent_cache_SHA='c'*64,layouts={},exact={})
            kwargs=dict(source_ref=dict(sha256='a'*64,bytes=1),input_ref=dict(sha256='d'*64,bytes=2),
                        parent_cache_SHA='c'*64,static_allowance=dict(rays=0,raypairs=0,layouts=0),independent_auditor=mock)
            reader.audit(node,**kwargs);reader.audit(node,**kwargs)
            self.assertEqual(len(calls),1)
            self.assertTrue(journal.has('reader_cursor'))
            cache=journal.read('geometry_'+'c'*64);cache['layouts']['later_superset']={'N':1}
            cache['cache_content_SHA']=content_sha({k:v for k,v in cache.items() if k!='cache_content_SHA'})
            journal.write('geometry_'+'c'*64,cache)
            reader.audit(node,**kwargs)
            self.assertEqual(len(calls),1)
            (Path(temp)/('geometry_'+'c'*64+'.json')).unlink()
            with self.assertRaises(Incomplete):reader.audit(node,**kwargs)
            node['state']='unknown'
            with self.assertRaises(Incomplete):reader.audit(node,**kwargs)
            journal.close()

    def test_unknown_inflight_never_replayed(self):
        with tempfile.TemporaryDirectory() as temp:
            journal=Journal(temp)
            journal.write('reader_budget',dict(rays=0,raypairs=0,layouts=0,seconds=0,inflight='unknown'))
            with self.assertRaises(Incomplete):Reader(journal)
            journal.close()

    def test_cursor_after_completed_node(self):
        with tempfile.TemporaryDirectory() as temp:
            journal=Journal(temp)
            payload=dict(path='0',semantic_SHA='a'*64)
            journal.callback('entered',payload)
            with self.assertRaises(RuntimeError):journal.callback('entered',payload)
            journal.callback('completed',dict(path='0',node=dict(semantic_SHA='a'*64,complete=True)))
            journal.callback('alias',dict(path='00',semantic_SHA='a'*64,source_content_id='b'*64))
            self.assertEqual(journal.read('node_'+'a'*64)['event'],'completed')
            journal.close()

    def test_allAND_full_denominator(self):
        rows=[dict(original_index=i,complete=True, O2_hit_S=True,O2_hit_Q=True,O2_hit_O1=True,
                   F00_hit_Q=False,F01_hit_Q=False,F10_hit_Q=False,F11_hit_Q=True) for i in INDICES]
        self.assertFalse(gate(rows)['passed'])
        with self.assertRaises(Incomplete):gate(rows[:-1])

    def test_pause_blocks_without_admission(self):
        with self.assertRaisesRegex(ValueError,'hard pause'):check(math_binding(), {},now=datetime(2026,10,14,23,tzinfo=timezone.utc),marker_values={})

if __name__=='__main__':unittest.main()
