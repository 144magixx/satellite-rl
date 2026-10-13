"""Static/synthetic replay tests: never invoke a native solve or CUDA update."""
import tempfile
import unittest
import numpy as np
import torch
from unittest.mock import patch
from satellite_rl.coverage.ai_search.core import Scene
from satellite_rl.coverage.ai_search_r3.pool import prepare_case,PreparedCase
from satellite_rl.coverage.ai_search_r3.model import CandidateCountNetwork,features_graph,select_slots
from satellite_rl.coverage.ai_search_r3.integer import build_problem,decode,validate_vector
from satellite_rl.coverage.ai_search_r3.postprocess import polish_envelope

def synthetic():return Scene([[45,10],[45.05,10.05],[50,15],[50.01,15.01]],[3e6,0,8e6,1e6],('d','a','c','b'))

class CoreTests(unittest.TestCase):
    def setUp(self):self.p=prepare_case(synthetic(),'synthetic_core')
    def test_cache_and_witness(self):
        self.assertTrue((self.p.witness.assignment>=0).all())
        with tempfile.TemporaryDirectory() as d:
            self.p.save(d+'/p.npz');other=PreparedCase.load(d+'/p.npz')
            self.assertEqual(self.p.content_sha,other.content_sha)
    def test_full_seed_and_stage2_replay(self):
        problem=build_problem(self.p,self.p.available)
        self.assertEqual(problem.fingerprint(),build_problem(self.p,self.p.available).fingerprint())
        self.assertTrue(validate_vector(problem,problem.seed)[0])
        layout,_=decode(self.p,problem,problem.seed)
        self.assertEqual(len(layout.assignment),4)
        second=build_problem(self.p,self.p.available,2,problem.seed)
        self.assertTrue(validate_vector(second,second.seed)[0])
        bad=problem.seed.copy();bad[0]=.5
        self.assertFalse(validate_vector(problem,bad)[0])
        final,e=polish_envelope(self.p,layout);self.assertTrue(e['audit']['valid'])
        self.assertLessEqual(e['actual_Omega'],e['menu_Omega']+1e-12)
    def test_selection_copy_prefix_and_no_oracle_substitution(self):
        for method in ('RULE','RANDOM'):
            slots,_=select_slots(self.p,method)
            self.assertTrue(set(self.p.mandatory).issubset(slots))
            for s in slots:
                if s%2:self.assertIn(s-1,slots)
        slots,meta=select_slots(self.p,'ORACLE_RESTRICT');self.assertIsNone(slots)
    def test_crosswidth_copies_and_prepare_guard(self):
        from satellite_rl.coverage.ai_search_r3.pool import PreparationIncomplete
        with self.assertRaises(PreparationIncomplete):prepare_case(synthetic(),'guard',seconds=0)
        p=self.p;problem=build_problem(p,p.available)
        # No center-wide exclusion: all three menu widths and both copies exist.
        for c in np.flatnonzero(p.origins[:,0]):
            self.assertTrue(set(range(int(c)*6,int(c)*6+6)).issubset(problem.slots))
        slots,_=select_slots(p,'ORACLE_RESTRICT',oracle_label={'active_slot_ids':p.mandatory[:2].tolist()})
        self.assertIsNotNone(slots)
        with self.assertRaisesRegex(ValueError,'legal_edge_variable_guard'):build_problem(p,p.available,max_edges=0)
        # Explicit coexisting same-center different-width beams, including zero-request owner.
        centers=np.flatnonzero(p.origins[:,0]);first=int(centers[0]);second=int(centers[1])
        active=[first*6,second*6,second*6+2]
        for case_active in (active,[first*6,second*6,second*6+1]):
            problem=build_problem(p,p.available);x=np.zeros_like(problem.seed)
            mapping={int(s):j for j,s in enumerate(problem.slots)}
            for s in case_active:x[mapping[s]]=1
            assignment={0:case_active[0],1:case_active[0],2:case_active[1],3:case_active[2]}
            for e,(i,b) in enumerate(zip(problem.edge_aircraft,problem.edge_slot)):
                if int(problem.slots[b])==assignment[int(i)]:x[len(problem.slots)+e]=1
            x[problem.zindex]=8/12
            self.assertTrue(validate_vector(problem,x)[0])
            layout,_=decode(p,problem,x);self.assertEqual(len(layout.centers),3)
    def test_network_parameter_and_equivariance(self):
        torch.manual_seed(51001);net=CandidateCountNetwork();self.assertEqual(sum(p.numel() for p in net.parameters()),170249)
        f,g=features_graph(self.p);perm=np.arange(len(f))[::-1].copy();inverse=np.argsort(perm)
        mapped=g[perm].copy();valid=mapped>=0;mapped[valid]=inverse[mapped[valid]]
        first=net(torch.tensor(f),torch.tensor(g));second=net(torch.tensor(f[perm]),torch.tensor(mapped))
        torch.testing.assert_close(first[perm],second,atol=2e-6,rtol=2e-6)
    def test_monitor_owned_memory_and_birth_cleanup(self):
        from satellite_rl.coverage.ai_search_r3.solver import case_memory,cleanup_owned
        with patch('satellite_rl.coverage.ai_search_r3.solver.own_rss',return_value=100),patch('satellite_rl.coverage.ai_search_r3.solver.process_identity',return_value={'rss':200,'birth':'b','children':[]}):
            self.assertEqual(case_memory(42,parent_pid=40)['case_tree_RSS'],300)
        with patch('satellite_rl.coverage.ai_search_r3.solver.process_identity',side_effect=[{'birth':'different'},{'birth':'same'}]),patch('satellite_rl.coverage.ai_search_r3.solver.os.kill') as kill:
            cleanup_owned({42:'same',43:'original'})
            kill.assert_called_once()
            self.assertEqual(kill.call_args.args[0],42)
    def test_zero_coverage_finite_logits_gradients(self):
        net=CandidateCountNetwork();f,g=features_graph(self.p);f[:,[7,13,19]]=0
        logits=net(torch.tensor(f),torch.tensor(g));self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue((logits[:,:,1:]==-1e9).all());logits[:,:,0].sum().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()))
    def test_pipeline_guard_and_invalid_problem_not_scale(self):
        from satellite_rl.coverage.ai_search_r3.evaluate import evaluate_method
        with tempfile.TemporaryDirectory() as d:
            result=evaluate_method(self.p,'WITNESS',d,public_input_seconds=31)
            self.assertEqual(result['status'],'pipeline_time_guard');self.assertIsNone(result['layout'])
            with patch('satellite_rl.coverage.ai_search_r3.evaluate.build_problem',side_effect=ValueError('invalid_seed')),patch('satellite_rl.coverage.ai_search_r3.evaluate.run_native') as native:
                result=evaluate_method(self.p,'RULE',d,vendor_dir='unused')
                self.assertEqual(result['status'],'invalid_integer_problem');self.assertNotIn('scale_guard',result);native.assert_not_called()
    def test_reference_target_bound_to_actual_chosen_source(self):
        from satellite_rl.coverage.ai_search_r3.evaluate import reference_target_evidence
        native=dict(value_valid=True,native_start_feasibility_accepted=True,decode={'valid':True},
            status='HighsModelStatus.kTimeLimit',reference_type='optimized_native',mip_dual_bound=.70)
        p=self.p
        # A valid new native teacher label cannot qualify an uncertified selected external W.
        self.assertFalse(reference_target_evidence(p,p.witness,'external_W',[native])['valid'])
        self.assertTrue(reference_target_evidence(p,p.witness,'native_stage1',[native])['valid'])
        native['mip_dual_bound']=.75
        self.assertTrue(reference_target_evidence(p,p.witness,'external_W',[native])['valid'])
        native['native_start_feasibility_accepted']=False
        self.assertFalse(reference_target_evidence(p,p.witness,'external_W',[native])['valid'])

if __name__=='__main__':unittest.main()
