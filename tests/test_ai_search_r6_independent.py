"""Independent R6 admission checks: pure mocks and CPU state only.

Never calls prepare, T, D5, CUDA or an optimizer and never reads datasets.
"""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from satellite_rl.workbench.r6_io import write_asset, read_asset
from satellite_rl.workbench.ai_search_r6_runner import guarded
from scripts.stage1_ai_search_r6.label_audit import audit_bundle
from scripts.stage1_ai_search_r4.audit import scene_sha


class IndependentAdmissionTests(unittest.TestCase):
    def bundle(self):
        scene=dict(positions=[[0.,0.],[0.,0.]], demands=[1.,1.], aircraft_ids=['a','b'], profile='europe_v1')
        identity=dict(role='train',case_id='c',mother_id='m',profile='p',identity_sha256='role',input_sha256=scene_sha(scene))
        metrics=dict(maxload=1.,top10_mean=1.,CV=0.,N=2)
        layout=dict(widths_deg=[.3,.4])
        p=dict(status='available',identity=identity,source_sha256='source',ledger={},raw_w={},Wenv={},Ncap=2,Omega_cap=1.)
        t=dict(status='available',identity=identity,provenance=identity,source_sha256='source',cost={},layout=layout,metrics=metrics,audit=dict(valid=True))
        d=dict(status='success',identity=identity,provenance=identity,cost={},layout=layout,metrics=metrics)
        wrap=lambda x:dict(identity=identity,source_sha256='source',result=x)
        return scene,wrap(p),wrap(t),wrap(d),identity

    def mocked_audit(self,bundle):
        scene,p,t,d,identity=bundle
        checked=dict(valid=True,metrics=dict(maxload=1.,top10_mean=1.,CV=0.,N=2,actual_Omega=1.))
        with patch('scripts.stage1_ai_search_r4.audit.audit_layout',return_value=checked),patch('scripts.stage1_ai_search_r4.audit.audit_ledger',return_value={'valid':True}),patch('scripts.stage1_ai_search_r5.audit.audit_decoder_ledger',return_value={'valid':True}):
            return audit_bundle(scene,p,t,d,identity,'source',True)

    def test_available_schema_positive_ignores_producer_auditflag(self):
        bundle=list(self.bundle());bundle[2]['result']['audit']['valid']=False
        result=self.mocked_audit(bundle)
        self.assertTrue(result['qualified'])
        self.assertEqual(result['label_origin'],'original_T_terminal')

    def test_typed_unavailable_positive_preserves_candidate(self):
        scene,p,t,d,identity=self.bundle()
        p['result'].update(status='unavailable',raw_w=None,Wenv=None)
        t['result']={k:v for k,v in t['result'].items() if k not in ('layout','metrics','audit')}
        t['result']['status']='unavailable'
        d['result']={k:v for k,v in d['result'].items() if k not in ('layout','metrics')}
        d['result']['status']='unavailable'
        with patch('scripts.stage1_ai_search_r4.audit.audit_layout') as geometry,patch('scripts.stage1_ai_search_r4.audit.audit_ledger',return_value={'valid':True}),patch('scripts.stage1_ai_search_r5.audit.audit_decoder_ledger',return_value={'valid':True}):
            result=audit_bundle(scene,p,t,d,identity,'source',True)
            geometry.assert_not_called()
        self.assertFalse(result['qualified']);self.assertTrue(result['typed_unavailable'])
        self.assertEqual(result['mother_id'],identity['mother_id'])

    def test_scene_identity_mismatch_rejected_before_geometry(self):
        bundle=list(self.bundle());bundle[0]['demands'][0]=99.
        with self.assertRaises(ValueError):self.mocked_audit(bundle)

    def test_terminal_source_mismatch_rejected(self):
        bundle=list(self.bundle());bundle[2]['result']['source_sha256']='other'
        with self.assertRaises(ValueError):self.mocked_audit(bundle)

    def test_unavailable_decoder_still_audits_consumed_budget(self):
        bundle=list(self.bundle());bundle[3]['result']['status']='unavailable'
        scene,p,t,d,identity=bundle
        with patch('scripts.stage1_ai_search_r4.audit.audit_layout',return_value={'valid':True,'metrics':{'N':2,'actual_Omega':1.}}),patch('scripts.stage1_ai_search_r4.audit.audit_ledger',return_value={'valid':True}),patch('scripts.stage1_ai_search_r5.audit.audit_decoder_ledger',return_value={'valid':False}):
            with self.assertRaisesRegex(ValueError,'ledger'):audit_bundle(scene,p,t,d,identity,'source',True)

    def test_compressed_array_values_lossless_and_sha_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            original={'floats':[0.,-0.,1.0000000000000002,1e-200], 'ids':['飞机','z'], 'role':'train'}
            ref=write_asset(Path(tmp)/'data.gz',original)
            self.assertEqual(read_asset(ref),original)
            changed=copy.deepcopy(ref);changed['canonical_json_sha256']='0'*64
            with self.assertRaises(ValueError):read_asset(changed)

    def test_completed_wal_changed_request_not_replayed(self):
        with tempfile.TemporaryDirectory() as tmp:
            from satellite_rl.workbench.storage import digest
            path=Path(tmp);(path/'request.json').write_text('{"identity":"old"}')
            (path/'started.json').write_text('{"request_sha256":"'+digest(path/'request.json')+'"}')
            (path/'guard.json').write_text('{"status":"completed"}')
            with patch('satellite_rl.workbench.ai_search_r6_runner.run_stage') as run:
                with self.assertRaises(ValueError):guarded(path,{'identity':'new'},1,lambda:False)
                run.assert_not_called()

    def test_sampler_rejects_negative_pending_index(self):
        from satellite_rl.coverage.ai_search_r6.learning import CaseSampler
        table=[dict(case_index=i,case_id=str(i)) for i in range(19)]
        sampler=CaseSampler(table);sampler.assemble();state=sampler.state_dict()
        state['pending'][0]=-1;state['pending_case_ids'][0]='18'
        with self.assertRaises(ValueError):CaseSampler(table).load_state_dict(state)

    def test_sampler_cross_epoch_resume_same_next_batch(self):
        from satellite_rl.coverage.ai_search_r6.learning import CaseSampler
        table=[dict(case_index=i,case_id=str(i)) for i in range(19)]
        a=CaseSampler(table);a.assemble();a.commit();pending=a.assemble()
        b=CaseSampler(table);b.load_state_dict(a.state_dict())
        self.assertEqual(b.assemble(),pending)
        a.commit();b.commit();self.assertEqual(a.assemble(),b.assemble())

    def test_original_T_exact_arrays_independently_checked(self):
        from scripts.stage1_ai_search_r6.label_audit import audit_exact_proposal
        identity=self.bundle()[-1]
        teacher={'status':'available','layout':{'beam_ids':['s0','s1'],'anchor_aircraft_ids':['a','b'],'assignment':[0,1]}}
        proposal={'status':'ready','provenance':{'source_method':'EXACT','case_identity':identity,'input_sha':identity['input_sha256'],'role_sha':identity['identity_sha256'],'prepared_source_sha256':'prepared','producer_sha256':'producer'},'slot_ids':['s0','s1'],'anchor_aircraft_ids':['a','b'],'scores_n_by_K':[[1.,0.],[0.,1.]],'mask_n_by_K':[[True,True],[True,True]]}
        proposal.update(schema='ProposalV2',case_identity=identity,input_sha=identity['input_sha256'],role_sha=identity['identity_sha256'])
        self.assertTrue(audit_exact_proposal(proposal,teacher,identity,'prepared','producer')['passed'])
        wrong=copy.deepcopy(proposal);wrong['scores_n_by_K'][0]=[0.,1.]
        with self.assertRaises(ValueError):audit_exact_proposal(wrong,teacher,identity,'prepared','producer')
        wrong=copy.deepcopy(proposal);wrong['anchor_aircraft_ids'].reverse()
        with self.assertRaises(ValueError):audit_exact_proposal(wrong,teacher,identity,'prepared','producer')

    def test_teacher_target_fractional_owners_rejected(self):
        import numpy as np
        from satellite_rl.coverage.ai_search_r6.learning import TeacherTarget
        target=TeacherTarget(('a','b'),np.array([.5,1.]),('a','b'),('s0','s1'))
        with self.assertRaises(ValueError):target.ordered()

    def test_engineering_step_failure_keeps_write_ahead_spend(self):
        from satellite_rl.coverage.ai_search_r6.model_qualification import _counted_optimizer_step
        receipt={'updates':[]};saved=[]
        def persist():saved.append(copy.deepcopy(receipt))
        def fake_step(*args,**kwargs):
            self.assertEqual(saved[-1]['updates'][-1]['status'],'started_unknown_consumed')
            raise RuntimeError('mock partial step')
        with self.assertRaises(RuntimeError):_counted_optimizer_step(fake_step,None,(),{},receipt,persist)
        self.assertEqual(receipt['updates'][0]['status'],'unknown_consumed')
        receipt['updates']*=4
        with self.assertRaises(RuntimeError):_counted_optimizer_step(lambda *a:None,None,(),{},receipt,persist)

    def test_engineering_budget_global_claim_cannot_replay(self):
        from satellite_rl.coverage.ai_search_r6.model_qualification import _consume_budget,_source_binding
        from satellite_rl.coverage.ai_search_r6.features import record_sha
        with tempfile.TemporaryDirectory() as tmp:
            source=_source_binding('a'*40);output=Path(tmp)/'output'
            budget=dict(source,source_manifest_sha=record_sha(source),output_path=str(output),global_claim_path=str(Path(tmp)/'claim.json'),reservation_id='mock-only')
            _consume_budget(budget,'spec',output)
            with self.assertRaises(FileExistsError):_consume_budget(copy.deepcopy(budget),'spec',output)
            changed=copy.deepcopy(budget);changed['output_path']=str(Path(tmp)/'other')
            with self.assertRaises(FileExistsError):_consume_budget(changed,'spec',Path(tmp)/'other')


if __name__=='__main__':unittest.main()
