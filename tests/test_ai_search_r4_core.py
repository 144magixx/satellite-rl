"""Static/hand-computed union and accounting tests; no whole optimization.

Geometry qualification is an explicit, one-shot separate entry point. Saved
qualification records are inspected only; unittest never reruns those cases.
"""
import hashlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from satellite_rl.coverage.ai_search_r4.core import (
    InterfaceError, Ledger, digest, lex_better, validate_proposal, decode_groups,
    _teacher_checked, build_proposal, physical_equivalent, envelope, _phase, _key)
from satellite_rl.coverage.ai_search.core import Layout


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.identity=dict(case_id='case:Ω',mother_id='mother',profile='base',
            input_sha256='input',identity_sha256='identity',role='development')
        self.scene=SimpleNamespace(demands=np.array([2.,0.]),aircraft_ids=('plane:α','plane:β'))
        self.p=SimpleNamespace(scene=self.scene,identity=self.identity,source_sha256='source',Ncap=2,status='available')
        self.unavailable_provenance={**self.identity,'source_sha256':'source','source_method':'TEACHER_EXACT'}
        self.proposal=dict(status='ready',**self.identity,source_sha256='source',source_method='RULE',
            slot_ids=['b0','b1'],anchor_aircraft_ids=['plane:α','plane:α'],
            scores_nxK_float32_finite=[[1.,0.],[0.,1.]],score_mask_nxK_bool=[[True,True],[True,True]],
            direction_offset_Kx3=[[0.,0.,0.],[0.,0.,0.]])

    def test_hash_canonical_utf8(self):
        expected=hashlib.sha256('[51001,"case:Ω","anchor","plane:α"]'.encode('utf8')).digest()
        self.assertEqual(digest('case:Ω','anchor','plane:α'),expected)

    def test_duplicate_anchors_allowed(self):
        reason,parsed=validate_proposal(self.p,self.proposal)
        self.assertIsNone(reason);self.assertEqual(parsed[1],['plane:α','plane:α'])

    def test_missing_payload_and_unavailable_never_read_arrays(self):
        self.assertEqual(validate_proposal(self.p,None)[0],'typed_unavailable_missing_payload')
        self.assertEqual(validate_proposal(self.p,{'status':'unavailable','reason':'unavailable_teacher',
            'provenance':self.unavailable_provenance})[0],'unavailable_teacher')
        result=decode_groups(self.p,None)
        self.assertEqual(result['status'],'unavailable');self.assertEqual(result['cost']['raypairs'],0)
        self.assertEqual(result['cost']['proposals'],0)

    def test_invalid_status_hard_error(self):
        for proposal in ({},{'status':'valid'},{'status':'unavailable_teacher'}):
            with self.subTest(proposal=proposal),self.assertRaises(InterfaceError):validate_proposal(self.p,proposal)

    def test_unavailable_arrays_hard_error(self):
        with self.assertRaises(InterfaceError):validate_proposal(self.p,dict(status='unavailable',slot_ids=[]))

    def test_teacher_union_negative(self):
        self.assertFalse(_teacher_checked(self.p,None))
        self.assertFalse(_teacher_checked(self.p,dict(status='unavailable',provenance=self.unavailable_provenance)))
        for teacher in ({},dict(status='valid'),dict(status='unavailable'),dict(status='unavailable',layout={}),dict(status='available')):
            with self.subTest(teacher=teacher),self.assertRaises(InterfaceError):_teacher_checked(self.p,teacher)

    def test_missing_teacher_two_offline_slots(self):
        for method in ('TEACHER_EXACT','TEACHER_PERTURBED'):
            result=build_proposal(self.p,method,teacher=None)
            self.assertEqual((result['status'],result['reason']),('unavailable','unavailable_teacher'))
            self.assertNotIn('scores_nxK_float32_finite',result)

    def test_teacher_forbidden_rule_random(self):
        for method in ('RULE','RANDOM'):
            with self.assertRaises(InterfaceError):build_proposal(self.p,method,teacher={'status':'unavailable'})

    def test_identity_source_role_hard_errors(self):
        for field in ('case_id','mother_id','input_sha256','identity_sha256','role','source_sha256'):
            proposal={**self.proposal,field:'wrong'}
            with self.subTest(field=field),self.assertRaises(InterfaceError):validate_proposal(self.p,proposal)

    def test_shape_nonfinite_ids_hard_errors(self):
        changes=[dict(slot_ids=['b0','b0']),dict(anchor_aircraft_ids=['missing','plane:α']),
                 dict(scores_nxK_float32_finite=[[float('nan'),0.],[0.,1.]]),
                 dict(score_mask_nxK_bool=[[1,1],[1,1]]),dict(direction_offset_Kx3=[[0.,0.]])]
        for change in changes:
            with self.subTest(change=change),self.assertRaises(InterfaceError):validate_proposal(self.p,{**self.proposal,**change})

    def test_count_and_all_masked_typed_reasons(self):
        self.p.Ncap=1
        self.assertEqual(validate_proposal(self.p,self.proposal)[0],'typed_proposal_count_exceeds_cap')
        self.p.Ncap=2
        proposal={**self.proposal,'score_mask_nxK_bool':[[False,False],[False,False]]}
        self.assertEqual(validate_proposal(self.p,proposal)[0],'typed_all_scores_unavailable')

    def test_precharge_rejection_does_not_reset(self):
        ledger=Ledger(2,3,100,proposals=1,refits=1,reserve=1)
        ledger.proposal(refit=True);ledger.geometry(4)
        with self.assertRaises(RuntimeError):ledger.geometry(1)
        self.assertEqual(ledger.data['raypairs'],4)
        self.assertEqual(ledger.data['proposals'],1)
        ledger.terminal=True;ledger.geometry(2)
        self.assertEqual(ledger.record()['raypairs'],6)

    def test_cache_actual_coordinate_bytes(self):
        ledger=Ledger(1,10,100)
        with patch('satellite_rl.coverage.ai_search_r4.core.satellite_rays_numpy',return_value=np.array([[1.,0.,0.]])) as primitive:
            ledger.center_rays([[48.,8.]],'europe_v1');ledger.center_rays([[48.,8.]],'europe_v1')
            ledger.center_rays([[48.001,8.]],'europe_v1')
        self.assertEqual(primitive.call_count,2)

    def test_strict_lex_with_raw_load_tolerances(self):
        self.assertFalse(lex_better((1.,2.,3.,4,5.),(1.,2.,3.,4,5.)))
        self.assertTrue(lex_better((.9,100.,100.,100,100.),(1.,2.,3.,4,5.)))
        self.assertFalse(lex_better((1.-1e-7,2.,3.,4,5.),(1.,2.,3.,4,5.)))

    def test_physical_equivalence_beam_and_anchor_ids_irrelevant(self):
        scene=SimpleNamespace(aircraft_ids=('a','b'))
        left=Layout([[48,8],[49,9]],[.3,.4],[0,1],('old0','old1'))
        right=Layout([[49,9],[48,8]],[.4,.3],[1,0],('renamed0','renamed1'))
        self.assertTrue(physical_equivalent(scene,left,right))

    def test_continuous_envelope_outward_rounding_and_no_clipping(self):
        # Hand geometry primitives, not another synthetic scene optimization.
        scene=SimpleNamespace(profile='europe_v1',rays=np.array([[1.,0.,0.]]))
        ledger=Ledger(1,10,100)
        with patch('satellite_rl.coverage.ai_search_r4.core.visible_numpy',return_value=np.array([True])), \
             patch.object(ledger,'center_rays',return_value=np.array([[1.,0.,0.]])), \
             patch('satellite_rl.coverage.ai_search_r4.core.ray_angle_numpy',return_value=np.array([.214])):
            self.assertEqual(envelope(scene,[0],[48.,8.],ledger),np.nextafter(.428,np.inf))
        with patch('satellite_rl.coverage.ai_search_r4.core.visible_numpy',return_value=np.array([True])), \
             patch.object(ledger,'center_rays',return_value=np.array([[1.,0.,0.]])), \
             patch('satellite_rl.coverage.ai_search_r4.core.ray_angle_numpy',return_value=np.array([.5000000000001])):
            self.assertIsNone(envelope(scene,[0],[48.,8.],ledger))
        self.assertEqual(ledger.data['raypairs'],2)

    def test_no_phase_snapshot_work_after_nominal_deadline(self):
        ledger=Ledger(2,100,0,proposals=20)
        sentinel=object()
        # p deliberately has no scene: deadline is inspected before accessing
        # input or constructing any snapshot/sort/load state.
        self.assertIs(_phase(SimpleNamespace(),sentinel,'MOVE',10,ledger),sentinel)
        self.assertEqual(ledger.data['proposals'],0)
        self.assertEqual(ledger.data['sort_input_count'],0)
        self.assertEqual(ledger.data['truncation_reason'],'nominal_time_limit')

    def test_unavailable_source_and_role_are_hard_errors(self):
        good=dict(status='unavailable',reason='unavailable_teacher',provenance=self.unavailable_provenance)
        for key in ('role','case_id','source_sha256'):
            bad={**good,'provenance':{**self.unavailable_provenance,key:'wrong'}}
            with self.subTest(key=key),self.assertRaises(InterfaceError):validate_proposal(self.p,bad)
            with self.subTest(teacher_key=key),self.assertRaises(InterfaceError):_teacher_checked(self.p,bad)
        for key in ('score_mask_nxK_bool','direction_offset_Kx3'):
            with self.subTest(key=key),self.assertRaises(InterfaceError):validate_proposal(self.p,{**good,key:[]})

    def test_metric_key_records_actual_sort_without_changing_value(self):
        ledger=Ledger(2,10,100)
        state={'owners':np.array([0,1]),'ids':['b0','b1'],'widths':np.array([.3,.4])}
        expected=_key(self.scene,state)
        self.assertEqual(_key(self.scene,state,ledger),expected)
        self.assertEqual(ledger.data['sort_input_count'],2)


if __name__=='__main__':unittest.main()
