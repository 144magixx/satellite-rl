"""Synthetic arrays and mocked geometry only; zero actual microfixture ray calls."""
import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from satellite_rl.coverage.ai_search_r8.core import (DESIGN_SHA,CHECKPOINT_SHA,IDENTITY_FIELDS,REF_FIELDS,local_transfer,integer_key,prospective_key,random_key,canonical,softmax_owner,affinity,array_sha,digest)
from satellite_rl.coverage.ai_search_r8.ledger import Ledger,Limit

class MockGeometry:
    def __init__(self,n,B,reject=False):self.n=n;self.B=B;self.reject=reject;self.real_geometry_calls=0
    def audit(self,centers,widths,owners,ids,ledger):ledger.geometry(self.n,'mock_pair_count');return {'valid':True,'mock_only':True}
    def distances(self,centers,ledger):
        ledger.charge('distance_table_elements',self.n*self.B);ledger.geometry(self.n*self.B,'mock_pair_count')
        return np.tile(np.arange(self.B,dtype=np.float64)/100,(self.n,1))
    def widths_for_move(self,owners,s,d,widths,distances,ledger):
        ledger.charge('cached_member_distance_scan',int(np.count_nonzero((owners==s)|(owners==d))))
        return None if self.reject else np.array(widths,dtype=np.float64,copy=True)

def fixture(arm='G',owners=(0,0,0,0,1,2)):
    n=len(owners);B=max(owners)+1;ids=tuple('plane:%02d'%i for i in range(n))
    scene=SimpleNamespace(aircraft_ids=ids,demands=np.full(n,21600000.,np.float64),profile='europe_v1',input_sha256='input')
    beam_ids=tuple('beam:%02d'%b for b in range(B));centers=np.zeros((B,2),np.float64)
    w=SimpleNamespace(assignment=np.asarray(owners,np.int64),widths=np.full(B,.3),beam_ids=beam_ids,centers=centers)
    w.to_record=lambda:dict(centers=centers.tolist(),widths_deg=w.widths.tolist(),assignment=w.assignment.tolist(),beam_ids=list(beam_ids))
    identity={key:'identity:'+key for key in IDENTITY_FIELDS};identity.update(role='development',profile='uniform21600000',input_sha256='input')
    prepared=SimpleNamespace(scene=scene,Wenv=w,identity=identity,Ncap=B,Omega_cap=1.,status='available',anchors=ids[:B],source_sha256='oldcore')
    logits=np.zeros((n,200),np.float32)
    provenance=dict(design_sha256=DESIGN_SHA,checkpoint_sha256=CHECKPOINT_SHA,case_identity=identity,actual_arm=arm,source_epoch='frozennew',adapter_source_SHA='newcore',source_refs={k:dict(path='original/'+k,sha256='1'*64) for k in REF_FIELDS},owner_array_sha256=array_sha(logits))
    return prepared,logits,provenance

class PureLocalTests(unittest.TestCase):
    def run_arm(self,arm='G',reject=False,stop=lambda:False):
        prepared,owner,provenance=fixture(arm)
        with tempfile.TemporaryDirectory() as directory:
            result=local_transfer(prepared,owner if arm=='A' else None,arm=arm,provenance=provenance,state_path=Path(directory)/'state.json',stop_callback=stop,geometry_backend=MockGeometry(6,3,reject),clock=lambda:0.)
            return result
    def test_integer_improvement_and_whole_requests(self):
        result=self.run_arm();self.assertLess(tuple(result['local_result']['final_integer_key']),tuple(result['local_result']['initial_integer_key']))
        self.assertEqual(sum(result['local_result']['beam_counts']),6);self.assertTrue(all(x>0 for x in result['local_result']['beam_counts']))
        self.assertEqual(sum(result['local_result']['beam_request_bps_int64']),6*21600000);self.assertEqual(result['local_result']['initial_layout']['centers'],result['local_result']['final_layout']['centers'])
    def test_rejected_mock_geometry_keeps_original_point(self):
        result=self.run_arm(reject=True);self.assertEqual(result['local_result']['initial_layout'],result['local_result']['final_layout'])
        self.assertTrue(all(t['status']=='rejected' for t in result['trace']))
    def test_three_arms_have_same_fixed_original_pool_and_no_teacher_proposer(self):
        pools=[]
        for arm in ('G','A','R'):
            p,l,v=fixture(arm)
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'state.json';r=local_transfer(p,l if arm=='A' else None,arm=arm,provenance=v,state_path=path,stop_callback=lambda:False,geometry_backend=MockGeometry(6,3),clock=lambda:0.)
                pools.append(json.loads(path.read_bytes())['pool']);self.assertTrue(r['proposal']['provenance']['EXACT_compatibility_only_not_teacher']);self.assertEqual(r['proposal']['provenance']['actual_outer_arm'],arm)
                self.assertNotIn('offset',r['proposal']);self.assertEqual(r['proposal']['slot_ids'],sorted(r['proposal']['slot_ids']))
                self.assertLessEqual(r['ledger']['candidate_extractions'],128);self.assertLessEqual(r['ledger']['scalar_rank_items'],65536)
        self.assertEqual(pools[0],pools[1]);self.assertEqual(pools[1],pools[2])
    def test_uniform_affinity_G_A_identical_integer_ties(self):
        g=self.run_arm('G');a=self.run_arm('A');self.assertEqual(g['trace'],a['trace']);self.assertEqual(g['local_result'],a['local_result'])
        self.assertEqual(a['ledger']['softmax_exp'],1200);self.assertLessEqual(a['ledger']['accepted_member_sum_update_elements'],51200)
    def test_singleton_sources_have_no_pool(self):
        p,l,v=fixture(owners=(0,1,2))
        with tempfile.TemporaryDirectory() as directory:
            r=local_transfer(p,None,arm='G',provenance=v,state_path=Path(directory)/'state.json',stop_callback=lambda:False,geometry_backend=MockGeometry(3,3),clock=lambda:0.)
            self.assertEqual(r['trace'],[]);self.assertEqual(r['ledger'].get('candidate_extractions',0),0)
    def test_random_hash_exact_recipe(self):
        candidate=['飞行器','s','d'];expected=__import__('hashlib').sha256(canonical([51001,'case','r8_transfer_order',*candidate])).hexdigest()
        self.assertEqual(random_key('case',candidate),expected)
    def test_one_scan_integer_rank_exactly_equals_full_count_key(self):
        for counts in (np.array([4,1,1]),np.arange(1,21),np.arange(1,201)):
            ledger=Ledger(int(counts.sum()),clock=lambda:0.);old=integer_key(counts)
            for source,destination in ((0,1),(1,0),(len(counts)-1,0)):
                future=counts.copy();future[source]-=1;future[destination]+=1
                self.assertEqual(prospective_key(counts,source,destination,old,ledger),integer_key(future))
            self.assertEqual(ledger.data['integer_scan_items'],3*len(counts))
    def test_affinity_excludes_self_and_query_permutation_is_only_approximate_fixture(self):
        rng=np.random.default_rng(81);logits=rng.normal(size=(6,200)).astype(np.float32);ledger=Ledger(6,clock=lambda:0.);prob=softmax_owner(logits,ledger);counts=np.array([4,1,1]);sums=np.vstack([prob[:4].sum(axis=0),prob[4],prob[5]])
        a=affinity(prob,sums,counts,0,0,1,ledger);expected=np.dot(prob[0],prob[4])-np.dot(prob[0],prob[1:4].mean(axis=0));self.assertEqual(a,float(expected))
        permutation=rng.permutation(200);b=affinity(prob[:,permutation],sums[:,permutation],counts,0,0,1,ledger)
        self.assertLessEqual(abs(a-b),1e-12+1e-12*max(abs(a),abs(b)))
    def test_source_and_cache_SHA_mismatch_fail(self):
        p,l,v=fixture('A');v['owner_array_sha256']='wrong'
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):local_transfer(p,l,arm='A',provenance=v,state_path=Path(directory)/'state.json',stop_callback=lambda:False,geometry_backend=MockGeometry(6,3),clock=lambda:0.)
    def test_pause_resumes_same_pool_clock_and_softmax_quota(self):
        p,l,v=fixture('A')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'state.json';paused=local_transfer(p,l,arm='A',provenance=v,state_path=path,stop_callback=lambda:True,geometry_backend=MockGeometry(6,3),clock=lambda:0.)
            self.assertEqual(paused['status'],'paused');self.assertIsNone(paused['proposal']);old=json.loads(path.read_bytes())
            root=dict(root_accepted=True,prior_state_sha256=digest(path),input_provenance=v,verified_charged_active_seconds=.1)
            resumed=local_transfer(p,None,arm='A',provenance=v,state_path=path,stop_callback=lambda:False,resume_admission=root,geometry_backend=MockGeometry(6,3),clock=lambda:0.)
            self.assertEqual(resumed['ledger']['softmax_exp'],1200);self.assertEqual(old['pool'],json.loads(path.read_bytes())['pool']);self.assertGreaterEqual(resumed['ledger']['active_seconds'],.1)
    def test_pending_extraction_and_unapproved_resume_never_replay(self):
        p,l,v=fixture()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'state.json';local_transfer(p,None,arm='G',provenance=v,state_path=path,stop_callback=lambda:True,geometry_backend=MockGeometry(6,3),clock=lambda:0.)
            state=json.loads(path.read_bytes());state['pending']={'already_extracted':True};path.write_bytes(canonical(state))
            root=dict(root_accepted=True,prior_state_sha256=digest(path),input_provenance=v,verified_charged_active_seconds=.1)
            with self.assertRaises(ValueError):local_transfer(p,None,arm='G',provenance=v,state_path=path,stop_callback=lambda:False,resume_admission=root,geometry_backend=MockGeometry(6,3),clock=lambda:0.)
    def test_reserved_terminal_pairs_and_exact_quotas(self):
        ledger=Ledger(6,clock=lambda:0.);ledger.charge('raypairs',509*6)
        with self.assertRaises(Limit):ledger.charge('raypairs',1)
        ledger.terminal=True;ledger.charge('raypairs',3*6);self.assertEqual(ledger.record()['raypairs'],512*6)
        ledger.data['candidate_extractions']=128
        with self.assertRaises(Limit):ledger.charge('candidate_extractions',1)

if __name__=='__main__':unittest.main()
