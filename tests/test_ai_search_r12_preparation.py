"""Pure preparation checks: no scene, ray, angle, flow, geometry or model calls."""
import ast
from pathlib import Path
import unittest
from satellite_rl.coverage.ai_search_r12.core import region_indices, mechanism_gate
from satellite_rl.coverage.ai_search_r12.ledger import Ledger, MotherLedger
from satellite_rl.coverage.ai_search_r12.schema import NodeState, qualified_complete

class Preparation(unittest.TestCase):
    def test_support_all_peaks_and_contact_required(self):
        from satellite_rl.coverage.ai_search_r12.core import support_region
        pairs={(0,2):(0.,.1),(1,2):(0.,.1)}
        x=support_region([3,3,1],['z','b','c'],pairs)
        self.assertEqual(x['region'],[1,2,0])
        self.assertFalse(x['immediate_peak_reduction_necessary'])
        self.assertEqual(support_region([3,3],['z','b'],{})['unsupported_reason'],'unsupported_no_slack_support')
        pairs={(0,2):(1.,1.),(1,2):(1.,1.)}
        self.assertEqual(support_region([3,3,1],['z','b','c'],pairs)['unsupported_reason'],'unsupported_no_contacting_support')
    def test_support_rank_and_permutation(self):
        from satellite_rl.coverage.ai_search_r12.core import support_region
        counts=[5,2,1];ids=['p','b','a'];pairs={(0,1):(0.,.2),(0,2):(0.,.3)}
        self.assertEqual(support_region(counts,ids,pairs)['support'],2)
        self.assertEqual(support_region([1,5,2],['a','p','b'],{(1,0):(0.,.3),(1,2):(0.,.2)})['support'],0)
    def test_exact_permutation_and_crossN(self):
        from satellite_rl.coverage.ai_search_r12.exact_order import exact_evidence,exact_key
        a=exact_evidence([1,4,2],[.3,.5,.8]);b=exact_evidence([2,1,4],[.8,.3,.5])
        self.assertEqual(exact_key(a),exact_key(b))
        self.assertEqual(a['NU'],3*21)
        # U alone decreases while NU increases: unequal N needs NU.
        c=exact_evidence([3,3],[.3,.3]);d=exact_evidence([3,2,1],[.3]*3)
        self.assertLess(d['U'],c['U']);self.assertGreater(d['NU'],c['NU'])
    def test_rational_tail(self):
        from satellite_rl.coverage.ai_search_r12.exact_order import exact_evidence,exact_key
        from fractions import Fraction
        e=exact_evidence([4,3]+[1]*9,[.3]*11)
        self.assertEqual(exact_key(e)[1],Fraction(7,2))
    def test_graph_shape_bound_all_allowed_m_k(self):
        peak=0
        for m in range(1,129):
            for k in range(1,min(m,32)+1):
                free=m-k;nodes=free+k+2;arcs=2*(free+free*k+k)
                self.assertLessEqual(nodes,130);self.assertLessEqual(arcs,6400)
                self.assertLessEqual(8*arcs,65536)
                # Each augment finds a simple path: <=2k+2 arcs.
                inspections=8*((free+1)*arcs+free*(2*k+2)+free*k)
                self.assertLessEqual(inspections,8388608)
                self.assertLessEqual(8*free*(2*k+2)*2,524288)
                self.assertLessEqual(8*(free+2)*nodes,1048576)
                peak=max(peak,arcs)
        self.assertEqual(peak,6400)
    def test_static_pair_bound_matches_incremental_anchor_loops(self):
        for m in range(1,129):
            for k in range(1,min(m,32)+1):
                anchor=m
                if k>=2:anchor+=m+(m-1)
                if k>=3:anchor+=2*(m-2)
                if k>=4:anchor+=(k-3)*m
                self.assertLessEqual(anchor,(k+2)*m)
                self.assertLessEqual(anchor+k*m,8448)
        self.assertEqual(max(s*(200-s) for s in range(1,31)),5100)
        self.assertEqual(6*8500+15300,66300)
        self.assertEqual(12*66300+28*4715,927620)
    def test_capacities_and_anchor_pass_bound(self):
        for m in range(1,129):
            for k in range(1,min(m,32)+1):
                lo,hi=(m+k-1)//k,m;calls=1
                # Worst branch, choose larger remaining interval each time.
                span=hi-lo
                while span:span//=2;calls+=1
                self.assertLessEqual(calls,8)
                self.assertLessEqual(k+1 if k>=4 else k+1,34)
        self.assertEqual(Ledger(8).caps['flow_queries'],8)
        self.assertEqual(MotherLedger(8).caps['kernels'],6)
    def test_semantic_unknown_and_missing_fail_closed(self):
        self.assertFalse(qualified_complete({}))
        self.assertFalse(qualified_complete(dict(schema='R12NodeV2',complete=True,state='unknown')))
        self.assertEqual(len(set(x.value for x in NodeState)),6)
        node=dict(schema='R12NodeV2',complete=True,state='admissible',semantic_SHA='s',terminal={},metrics={},ledger={},exact_evidence={},
                  raw_producer_label='strict_raw_objective_improvement',proof_qualification='unknown_not_a_negative')
        self.assertFalse(qualified_complete(node))
        node['proof_qualification']='complete_hard_legal_dual_protected'
        self.assertTrue(qualified_complete(node))
        node['state']='graph_infeasible'
        self.assertFalse(qualified_complete(node))
    def test_new_namespace_does_not_depend_on_dirty_R11(self):
        folder=Path(__file__).parents[1]/'src/satellite_rl/coverage/ai_search_r12'
        for file in folder.glob('*.py'):
            ast.parse(file.read_text(encoding='utf8'))
            self.assertNotIn('ai_search_r11',file.read_text(encoding='utf8'))
    def test_fixture_short_arc_literal_distance_expectation(self):
        # Pure coordinate-distance spec, never calls real angular geometry.
        x=[0,1,2,4,8,14,21,32];selected=[7,0]
        for expected in (5,6,4):
            remaining=[j for j in range(8) if j not in selected]
            chosen=min(remaining,key=lambda j:(-min(abs(x[j]-x[k]) for k in selected),j))
            self.assertEqual(chosen,expected);selected.append(chosen)
        self.assertEqual(sorted(selected[:4]),[0,5,6,7])
        self.assertEqual(sorted(selected),[0,4,5,6,7])
    def test_gate_requires_full_independent_rows(self):
        with self.assertRaises(ValueError):mechanism_gate([])

if __name__=='__main__':unittest.main()
