"""Identity/control mocks and integer costs only; no real ray geometry."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from satellite_rl.coverage.ai_search_r10 import core
from satellite_rl.coverage.ai_search_r10.costs import edge_cost
from satellite_rl.coverage.ai_search_r10.flow import Ledger
from satellite_rl.coverage.ai_search_r10.local_flow import minimum_capacity
from satellite_rl.coverage.ai_search_r9.flow import minimum_capacity as frozen_minimum
from satellite_rl.coverage.ai_search_r9.flow import Ledger as FrozenLedger
from satellite_rl.coverage.ai_search_r9.core import choose_anchors as frozen_anchors
from satellite_rl.coverage.ai_search_r10.anchors import choose_anchors as observed_anchors


class CoreMockTests(unittest.TestCase):
    def identity(self):
        x={k:'x' for k in core.IDENTITY_KEYS}
        x.update(case_id='case',role='train',profile='uniform21600000',input_sha256='input')
        return x

    def test_exact_quantization_and_lex_dominance(self):
        self.assertEqual(edge_cost(.5,.3,False,4,'Gd'),250000)
        weight=4*250001+1
        self.assertEqual(edge_cost(.5,.3,True,4,'Gr'),weight+250000)
        self.assertEqual(edge_cost(.5,.3,False,4,'Gw'),910*weight+250000)
        self.assertGreater(weight,4*250001)
        with self.assertRaises(ValueError):edge_cost(.6,.3,False,4,'Gd')

    def test_SHC_action_tail_is_original_string_not_menu_ordinal(self):
        actions=[core.ACTIONS[i][0] for i in (2,1,3)]
        descriptors=[dict(action=a,rank=[3,-1,10,.001,'source','partner',a]) for a in actions]
        ordered=core._ordered_descriptors(descriptors,'SH-C',{},Ledger(1))
        self.assertEqual([d['action'] for d in ordered],[core.ACTIONS[i][0] for i in (3,1,2)])

    def test_metric_qualification_preserves_source_and_allows_root_numeric_protocol(self):
        source=dict(N=2,maxload=43200000,total_request=86400000,top10_mean=43200000.,CV=.1,actual_Omega=.00004)
        initial=copy.deepcopy(source)
        recomputed=copy.deepcopy(source)
        recomputed['CV']=float(np.nextafter(source['CV'],np.inf))
        recomputed['top10_mean']=float(np.nextafter(source['top10_mean'],np.inf))
        # Cos cancellation can exceed four Omega ULPs while well below 1e-12sr.
        recomputed['actual_Omega']+=5e-13
        core.qualify_metrics(recomputed,source)
        self.assertEqual(source,initial)
        for key,value in [('CV',.1000001),('top10_mean',43200001.),('actual_Omega',.000040000002),('N',3)]:
            bad=copy.deepcopy(recomputed)
            bad[key]=value
            with self.assertRaises(core.InterfaceError if hasattr(core,'InterfaceError') else ValueError):
                core.qualify_metrics(bad,source)

    def test_physical_state_ordinal_permutation_invariant(self):
        scene=SimpleNamespace(aircraft_ids=('d','a','c','b'))
        a=dict(beam_ids=['z','a'],anchor_aircraft_ids=['a','b'],centers=[[1.,2.],[3.,4.]],widths_deg=[.3,.4],assignment=[0,0,1,1])
        b=dict(beam_ids=['a','z'],anchor_aircraft_ids=['b','a'],centers=[[3.,4.],[1.,2.]],widths_deg=[.4,.3],assignment=[1,1,0,0])
        self.assertEqual(core.physical_state(scene,a,'case'),core.physical_state(scene,b,'case'))
        b['widths_deg'][0]=np.nextafter(.4,1.)
        self.assertNotEqual(core.physical_state(scene,a,'case'),core.physical_state(scene,b,'case'))

    def test_common_mock_keeps_peak_owned_member_prefix_and_duplicate_N(self):
        scene=SimpleNamespace(aircraft_ids=('a','b','c','d'),positions=np.array([[0.,0.],[1.,0.],[2.,0.],[3.,0.]]),
            rays=np.array([[1.,0.,0.]]*4),demands=np.full(4,core.REQUEST),input_sha256='input')
        layout=dict(beam_ids=['x','y'],anchor_aircraft_ids=['a','c'],centers=[[0.,0.],[2.,0.]],widths_deg=[.3,.3],assignment=[0,0,1,1])
        metrics=dict(maxload=2*core.REQUEST,top10_mean=2*core.REQUEST,CV=0.,N=2,actual_Omega=1.,total_request=4*core.REQUEST)
        origin=dict(identity=self.identity(),layout=layout,metrics=metrics,Ncap=3,Omega_cap=2.,layout_SHA=core.digest(layout),source_refs=[{'saved':'ref'}])
        with patch.object(core,'audit',return_value=metrics),patch.object(core,'angles',side_effect=lambda a,b,l:np.zeros(np.broadcast_shapes(np.shape(a)[:-1],np.shape(b)[:-1]))):
            common=core.prepare_common(scene,origin,original_index=0)
        self.assertEqual([x['anchor_aircraft_id'] for x in common['added_center_prefix']],['b','d'])
        self.assertEqual([s['status'] for s in common['N_slots']],['count_infeasible','count_infeasible','duplicate_N'])
        self.assertEqual(common['origin']['layout'],layout)
        self.assertEqual(common['common_ledger']['flow_queries'],0)

    def test_frozen_local_numeric_recipe_on_hand_supplied_graphs(self):
        graphs=[(['a','b','c'],['a','b'],[[True,True],[True,True],[True,True]]),
                (['a','b','c','d'],['a','d'],[[True,False],[True,False],[False,True],[False,True]]),
                (['a','b','c'],['a'],[[True],[False],[True]])]
        for members,anchors,adjacency in graphs:
            inner=core._KernelLedger(len(members),Ledger(len(members)))
            actual=minimum_capacity(members,anchors,adjacency,inner)
            expected=frozen_minimum(members,anchors,adjacency,FrozenLedger(len(members)))
            self.assertEqual(actual,expected)
            self.assertGreater(inner.outer.data['graph_arc_initialization_writes'],0)
            self.assertEqual(inner.outer.data['scan_input_count'],inner.data['scan_input_count'])

    def test_anchor_recipe_equivalence_with_abstract_distance_mock(self):
        scene=SimpleNamespace(aircraft_ids=('d','c','b','a'),rays=np.array([[1.,0.,0.],[.99,.01,0.],[.98,.02,0.],[.97,.03,0.]]))
        # Substitute the distance service in both versions: this is an abstract
        # arithmetic/ordering comparison, never the true ray primitive.
        def fake_angles(a,b,ledger):return np.sum((a-b)**2,axis=-1)
        with patch('satellite_rl.coverage.ai_search_r9.core.angles',side_effect=fake_angles),patch('satellite_rl.coverage.ai_search_r10.anchors.angles',side_effect=fake_angles):
            for k in (1,2,3):
                expected=frozen_anchors(scene,list(scene.aircraft_ids),k,FrozenLedger(4))
                actual=observed_anchors(scene,list(scene.aircraft_ids),k,core._KernelLedger(4,Ledger(4)))
                self.assertEqual(actual,expected)

    def test_no_progress_tries_next_semantic_and_skips_reverse_duplicate(self):
        common=dict(origin=dict(layout={'version':0},metrics={'score':0}))
        pool=[dict(semantic_SHA='one',structurally_eligible=True),dict(semantic_SHA='one',structurally_eligible=True),
              dict(semantic_SHA='two',structurally_eligible=False),dict(semantic_SHA='three',structurally_eligible=True)]
        with patch.object(core,'_pool',return_value=pool),patch.object(core,'_kernel',return_value={'status':'no_improvement'}) as kernel:
            result=core._sequential(None,common,'SH-C',Ledger(1))
        self.assertEqual(kernel.call_count,2)
        self.assertEqual(result['termination'],'no_progress_pool_exhausted')
        self.assertEqual([s['screening'] for s in result['screenings']],['kernel_attempted','duplicate_semantics','known_unavailable','kernel_attempted'])

    def test_32_commits_have_32_opportunities_and_no_q_early_stop(self):
        common=dict(origin=dict(layout={'version':0},metrics={'score':0}))
        def pool(scene,common,current,method,ledger):
            return [dict(semantic_SHA=str(current['layout']['version']),structurally_eligible=True)]
        def kernel(scene,common,current,descriptor,ledger):
            return dict(status='admissible',terminal={'version':current['layout']['version']+1},metrics={'score':0})
        with patch.object(core,'_pool',side_effect=pool),patch.object(core,'_kernel',side_effect=kernel):
            result=core._sequential(None,common,'SH-C',Ledger(1))
        self.assertEqual(result['kernel_attempts'],32)
        self.assertEqual(result['selected']['terminal']['version'],32)

    def test_unknown_method_preserves_last_observed_legal_point_and_cost(self):
        common=dict(n=1,origin=dict(layout={'version':0},metrics={'score':0}))
        def interrupted(scene,common,method,ledger,observations):
            ledger.tick('flow_queries')
            observations['selected']={'terminal':{'version':1},'metrics':{'score':-1}}
            from satellite_rl.coverage.ai_search_r10.flow import ResourceUnknown
            raise ResourceUnknown('test_stop',ledger)
        with patch.object(core,'_common_valid'),patch.object(core,'_global',side_effect=interrupted):
            with self.assertRaises(core.ResourceUnknown) as caught:
                core.run_method(None,common,'Gd')
        self.assertEqual(caught.exception.partial_result['selected']['terminal'],{'version':1})
        self.assertEqual(caught.exception.partial_ledger['flow_queries'],1)
        self.assertFalse(caught.exception.partial_result['complete'])

    def test_engineering_unknown_reason_predicate_never_accepts_pause(self):
        from satellite_rl.coverage.ai_search_r10.engineering_runner import _expected_residual_cap_stop
        ledger=Ledger(1)
        ledger.data.update(flow_queries=1,graph_arc_initialization_writes=10,heap_pushes_plus_pops=2)
        self.assertTrue(_expected_residual_cap_stop(core.ResourceUnknown('residual_edge_inspections_cap'),ledger))
        self.assertFalse(_expected_residual_cap_stop(core.ResourceUnknown('external_guard_or_pause'),ledger))
        ledger.data['heap_pushes_plus_pops']=0
        self.assertFalse(_expected_residual_cap_stop(core.ResourceUnknown('residual_edge_inspections_cap'),ledger))

    def test_rejected_kernel_keeps_raw_local_matching(self):
        scene=SimpleNamespace(aircraft_ids=('a','b'),rays=np.array([[1.,0.,0.]]*2),positions=np.array([[50.,13.],[50.,13.]]))
        layout=dict(beam_ids=['x','y'],anchor_aircraft_ids=['a','b'],centers=scene.positions.tolist(),widths_deg=[.3,.3],assignment=[0,1])
        common=dict(n=2,identity={'case_id':'case'},origin={'Ncap':2,'Omega_cap':0.})
        current=dict(layout=layout,metrics={})
        descriptor=dict(member_ids=['a','b'],target_k=1,region_beam_ids=['x','y'],semantic_SHA='semantic',state_SHA='state')
        with patch.object(core,'choose_anchors',return_value=['a']),patch.object(core,'minimum_capacity',return_value={'capacity':2,'assignment':[0,0]}):
            result=core._kernel(scene,common,current,descriptor,Ledger(2))
        self.assertEqual(result['reason'],'whole_Omega_cap')
        self.assertEqual(result['raw_local_matching']['assignment'],[0,0])
        self.assertEqual(result['raw_layout']['assignment'],[0,0])


if __name__=='__main__':unittest.main()
