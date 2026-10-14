"""Pure integer/arithmetic/metadata tests; never run geometry fixtures/model."""
import ast,copy,unittest
from pathlib import Path
from satellite_rl.coverage.ai_search_r9.flow import Ledger,ResourceUnknown,minimum_capacity
from satellite_rl.coverage.ai_search_r9.core import protected,objective,method_order,best_terminal,CommonLedger,select_origin,InterfaceError
from satellite_rl.coverage.ai_search_r9.value_model import FEATURE_NAMES,ARCHITECTURE,strict_pairs,PairSampler
from satellite_rl.coverage.ai_search_r9.engineering_fixtures import exact_specs

class PureR9(unittest.TestCase):
    def test_pinned_nonempty_capacity_and_deterministic_ties(self):
        ids=tuple('a%d'%i for i in range(9));anchors=ids[:3];adj=[[True]*3 for _ in ids];l=Ledger(9);r=minimum_capacity(ids,anchors,adj,l)
        self.assertEqual(r,dict(capacity=3,assignment=[0,1,2,0,0,1,1,2,2]));self.assertLessEqual(l.data['flow_queries'],8)
    def test_impossible_adjacency_not_unknown(self):
        l=Ledger(3);self.assertIsNone(minimum_capacity(('a','b','c'),('a',),[[True],[False],[True]],l));self.assertEqual(l.data['flow_queries'],1)
    def test_capacity_search_full128_bound(self):
        ids=tuple('a%03d'%i for i in range(128));l=Ledger(128);r=minimum_capacity(ids,ids[:3],[[True]*3 for _ in ids],l)
        self.assertEqual(r['capacity'],43);self.assertLessEqual(l.data['flow_queries'],8);self.assertLessEqual(l.data['residual_edge_inspections'],1048576)
    def test_counter_unknown_not_infeasibility(self):
        l=Ledger(3);l.data['residual_edge_inspections']=1048576
        with self.assertRaises(ResourceUnknown):minimum_capacity(('a','b','c'),('a',),[[True],[True],[True]],l)
    def test_common_ledger_cumulative_not_reset(self):
        first=CommonLedger(3);first.pairs(7);second=CommonLedger(3,first.record());second.pairs(2);self.assertEqual(second.data['raypairs'],9)
        second.data['raypairs']=3*3+19900
        with self.assertRaises(ResourceUnknown):second.pairs(1)
    def test_RULE_fallback_rejected_before_geometry(self):
        from types import SimpleNamespace
        identity={k:'x' for k in ('case_id','mother_id','profile','role','input_sha256','identity_sha256','shared_case_content_sha256','source_asset_SHA')};identity.update(profile='uniform21600000',role='train')
        scene=SimpleNamespace(aircraft_ids=('a',),input_sha256='x');p=SimpleNamespace(identity=identity,status='available',Ncap=1,Omega_cap=.1)
        with self.assertRaisesRegex(InterfaceError,'successful RULE floor'):select_origin(scene,p,{'status':'fallback'})
    def test_no_numeric_or_coerced_IDs(self):
        with self.assertRaises(ValueError):minimum_capacity((1,2),(1,),[[True],[True]],Ledger(2))
    def test_dual_quality_protection_rejects_lex_only(self):
        a=dict(maxload=99,top10_mean=102,CV=.5);b=dict(maxload=100,top10_mean=100,CV=.5);self.assertFalse(protected(a,b))
    def test_exact_tie_keeps_origin(self):
        m=dict(maxload=100,top10_mean=90,CV=.5,N=3,actual_Omega=.01);origin=dict(layout={},metrics=m)
        self.assertEqual(best_terminal(origin,[dict(position=0,status='admissible',metrics=copy.deepcopy(m),terminal={})])['position'],-1)
    def test_common_structural_mask_and_deterministic_orders(self):
        ds=[dict(position=i,structurally_eligible=i!=0,global_peak_lower_count=10-i,Omega_lower_bound=.1*i,region_beam_ids=['b%d'%i],action='split') for i in range(4)];p=dict(case_id='x',descriptors=ds)
        for method in ('Hq','HOmega','R','A'):
            args={'scores':[100,1,2,3]} if method=='A' else {};order=method_order(p,method,**args);self.assertEqual(order[-1],0);self.assertEqual(order,method_order(p,method,**args))
    def test_pairs_omit_identical_rejections_and_internal(self):
        a=dict(maxload=100,top10_mean=90,CV=.5,N=3,actual_Omega=.01);b={**a,'maxload':99};pairs=strict_pairs([a,a,b]);self.assertEqual(pairs,[(2,0),(2,1)])
        s=PairSampler({'fit':pairs,'internal':[(0,1)]},['fit']);self.assertEqual({x[0] for x in s.sample()},{'fit'});state=copy.deepcopy(s.state());expected=s.sample();s.restore(state);self.assertEqual(s.sample(),expected)
    def test_model_and_eight_geometry_inputs_frozen_not_executed(self):
        self.assertEqual(ARCHITECTURE,(24,64,32,1));self.assertEqual(len(FEATURE_NAMES),24);specs=exact_specs();self.assertEqual([s['fixture_id'] for s in specs],list(range(1,9)))
        self.assertTrue(all(len(s['aircraft_ids'])==len(s['positions'])==len(s['demands']) for s in specs));self.assertTrue(all(all(isinstance(x,str) for x in s['aircraft_ids']+s['layout']['beam_ids']) for s in specs))

if __name__=='__main__':unittest.main()
