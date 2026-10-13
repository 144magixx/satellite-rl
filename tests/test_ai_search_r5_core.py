"""Static/mock checks only. No whole D5 or preparation/teacher searches."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from satellite_rl.coverage.ai_search_r5 import core as c
from satellite_rl.coverage.ai_search.core import Layout

class R5CoreTests(unittest.TestCase):
    def setUp(self):
        self.identity=dict(case_id='c',mother_id='m',profile='base',input_sha256='input',identity_sha256='id',role='development')
        self.scene=SimpleNamespace(demands=np.array([2.,1.]),aircraft_ids=('a','b'),input_sha256='input',positions=np.array([[48.,8.],[48.,9.]]),rays=np.array([[1.,0.,0.],[1.,0.,0.]]),profile='europe_v1')
        self.p=SimpleNamespace(scene=self.scene,identity=self.identity,source_sha256='old',status='available',Ncap=2,Omega_cap=1.)
        self.prop=dict(schema='ProposalV2',status='ready',case_identity=self.identity,input_sha='input',role_sha='id',provenance=c._provenance(self.p,'EXACT','new',[]),slot_ids=['x','y'],anchor_aircraft_ids=['a','a'],scores_n_by_K=[[1.,0.],[0.,1.]],mask_n_by_K=[[True,True],[True,True]],construction_cost={'seconds':0.})
    def test_union_identity_offset_and_shape(self):
        self.assertIsNone(c.validate_proposal(self.p,self.prop)[0])
        for change in ({'offset':[]},{'case_identity':{}},{'slot_ids':['x','x']},{'mask_n_by_K':[[1,1],[1,1]]},{'scores_n_by_K':[[np.nan,0],[0,1]]},{'anchor_aircraft_ids':['z','a']}):
            with self.subTest(change=change),self.assertRaises(c.InterfaceError):c.validate_proposal(self.p,{**self.prop,**change})
    def test_unavailable_tag_does_not_access_layout(self):
        u=dict(schema='ProposalV2',status='unavailable',reason='no_teacher',provenance=self.prop['provenance'],construction_cost={'seconds':0.})
        self.assertEqual(c.validate_proposal(self.p,u)[0],'no_teacher')
        for key in ('slot_ids','scores_n_by_K','mask_n_by_K','layout'):
            with self.assertRaises(c.InterfaceError):c.validate_proposal(self.p,{**u,key:[]})
    def test_typed_limits(self):
        self.p.Ncap=1;self.assertEqual(c.validate_proposal(self.p,self.prop)[0],'typed_proposal_count_exceeds_cap')
        self.p.Ncap=2;self.assertEqual(c.validate_proposal(self.p,{**self.prop,'mask_n_by_K':[[False,False],[False,False]]})[0],'typed_all_scores_unavailable')
    def test_role_manifest_true_role_required(self):
        manifest=dict(stage='P0',role='development',admitted=True,identity_sha256='id',dataset_sha256='a'*64,cases={'c':self.identity})
        c.validate_phase(self.scene,self.identity,manifest)
        for key,val in [('role','train'),('admitted',False),('dataset_sha256','x'),('cases',{})]:
            with self.assertRaises(c.InterfaceError):c.validate_phase(self.scene,self.identity,{**manifest,key:val})
        train={**self.identity,'role':'train'}
        c.validate_phase(self.scene,train,{**manifest,'stage':'labels','role':'train','cases':{'c':train}})
    def test_q0_absolute_tolerances_and_immutable_key(self):
        q=(10.,5.,.1,2,.2)
        self.assertFalse(c.q0_protected((9.,5.,.2,2,.2),q))
        self.assertTrue(c.q0_protected((9.,5.+1e-7,.1+1e-10,2,.2),q))
        self.assertEqual(q,(10.,5.,.1,2,.2))
    def test_ledger_precharge_and_terminal_reserve(self):
        l=c.Ledger(2,10,5,2,1,reserve=3);l.geometry(14);l.proposal(refit=True)
        with self.assertRaises(c.Exhausted):l.geometry(1)
        self.assertEqual(l.data['raypairs'],14);l.terminal=True;l.geometry(6)
        self.assertEqual(l.record()['terminal_reserve_raypairs'],6)
        with self.assertRaises(c.Exhausted):l.proposal(refit=True)
    def test_insert_masked_scores_not_accessed(self):
        class Scores:
            def __getitem__(self,key):raise AssertionError('masked score read')
        state=dict(ids=['x'],anchors=['a'],centers=np.array([[48.,8.]]),widths=np.array([.3]),owners=np.array([0,-1]))
        l=c.Ledger(2,2048,5,512,8,reserve=3)
        with patch.object(c,'ray_angle_numpy',return_value=.1),patch.object(c,'_fit',return_value=True),patch.object(c,'_legal',return_value=True),patch.object(c,'_accepted'):
            new=c._insert(self.p,state,Scores(),np.zeros((2,1),bool),['x'],l)
        self.assertEqual(new['owners'].tolist(),[0,0]);self.assertEqual(l.data['proposals'],1)
    def test_eviction_equal_max_multiplicity_zero_gain(self):
        state=dict(ids=['x'],anchors=['a'],centers=np.array([[48.,8.]]),widths=np.array([float(np.nextafter(.8,np.inf))]),owners=np.array([0,0]))
        self.p.Omega_cap=0.;l=c.Ledger(2,2048,5,512,8,reserve=3)
        with patch.object(c,'ray_angle_numpy',return_value=np.array([.4,.4])):
            new=c._evict(self.p,state,np.ones((2,1)),np.ones((2,1),bool),['x'],l)
        self.assertEqual(l.data['proposals'],0);self.assertEqual(new['owners'].tolist(),[0,0])
        self.assertEqual(l.data['truncation_reason'],'eviction_no_positive_legal_gain')
    def test_eviction_rejected_candidates_not_retried_same_state(self):
        state=dict(ids=['x'],anchors=['a'],centers=np.array([[48.,8.]]),widths=np.array([.8]),owners=np.array([0,0]))
        self.p.Omega_cap=0.;l=c.Ledger(2,2048,5,512,8,reserve=3)
        with patch.object(c,'ray_angle_numpy',return_value=np.array([.4,.1])),patch.object(c,'_fit',return_value=True),patch.object(c,'_legal',return_value=False):
            c._evict(self.p,state,np.ones((2,1)),np.ones((2,1),bool),['x'],l)
        self.assertEqual(l.data['proposals'],1);self.assertEqual(len(l.data['evictions']),1)
    def test_d4_conversion_does_not_call_old_proposer(self):
        self.prop['provenance']['source_method']='OWNER10'
        with patch('satellite_rl.coverage.ai_search_r4.core.build_proposal',side_effect=AssertionError('old proposer')),patch('satellite_rl.coverage.ai_search_r4.core.decode_groups',return_value={'status':'success'}) as decoder:
            out=c.decode_d4(self.p,self.prop,contract_sha256='contract',adapter_source_sha256='adapter')
        inner=decoder.call_args.args[1]
        self.assertEqual(inner['source_sha256'],'old');self.assertEqual(inner['source_method'],'TEACHER_EXACT')
        self.assertEqual(out['derivation']['diagnostic_method'],'OWNER10');self.assertEqual(self.p.source_sha256,'old')
        self.assertEqual(out['derivation']['proposal_before'],self.prop)
    def test_zero_request_internal_key_is_finite_no_label(self):
        self.scene.demands=np.zeros(2)
        state=dict(ids=['x','y'],owners=np.array([0,1]),widths=np.array([.3,.3]))
        self.assertEqual(c._key(self.scene,state)[:3],(0.,0.,0.))
    def test_strict_envelope_one_degree_not_clipped(self):
        l=c.Ledger(2,2048,5)
        with patch.object(c,'_angles_and_chord',return_value=(np.array([.5000001]),np.array([0.]))):
            self.assertIsNone(c.envelope(self.scene,[0],self.scene.positions[0],l))
        with patch.object(c,'_angles_and_chord',return_value=(np.array([.5]),np.array([0.]))),patch.object(l,'center_rays',return_value=np.array([[1.,0.,0.]])):
            self.assertEqual(c.envelope(self.scene,[0],self.scene.positions[0],l),1.)

if __name__=='__main__':unittest.main()
