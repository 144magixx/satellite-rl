"""Pure metadata fixtures only: no scenes/geometry/stages/model executions."""
import copy,unittest
from pathlib import Path
from unittest.mock import patch
from satellite_rl.workbench import r7_failed_label_audit as f
from scripts.stage1_ai_search_r7.label_audit import gate

def context_fixture():
    refs={k:dict(path=k,sha256=(str(i)*64)) for i,k in enumerate(('contract_ref','root_review_ref','accounting_acceptance_ref','original_index_archive_ref'),1)}
    c=dict(schema='R7Labels74OperationalRecoveryV1',run_id=f.RUN,scientific_commit=f.SCIENCE,operational_commit='a'*40,
           preserved_case_indices=f.PRESERVED,failed_case_indices=list(f.FAILED),remaining_case_indices=f.REMAINING,
           failed_cases={'295':{},'297':{}},new_stage_attempts_max=dict(prepare=139,T=139,D5=139),repeat_original_attempts=False,label_denominator=576,
           accounting_addendum_ref=dict(path='addendum',sha256=f.ADDENDUM),original_job_ref=dict(path='job',sha256='j'*64),
           original_owner_ref=dict(path='owner',sha256='o'*64),operations_source_refs={},original_case_index_ref=dict(path='mutable_index',sha256=refs['original_index_archive_ref']['sha256']))
    job=dict(run_id=f.RUN,case_indices=list(range(1,576,2)),request_hash='r'*64,original_task_contract=dict(config=dict(datasets=dict(train=dict(sha256='d'*64)))))
    c['original_job']=job
    cases={str(i):dict(path='label'+str(i),sha256='l'*64) for i in range(576)}
    old={str(i):dict(path='case'+str(i),sha256='p'*64) for i in f.PRESERVED}
    objects={'contract_ref':c,'root_review_ref':dict(passed=True,run_id=f.RUN,contract_sha256=refs['contract_ref']['sha256'],operational_commit='a'*40),
             'accounting_acceptance_ref':dict(passed=True,run_id=f.RUN,source_commit=f.SCIENCE,addendum_ref=dict(sha256=f.ADDENDUM)),
             'addendum':dict(fixed_original_partition=dict(published_indices=f.PRESERVED,partial_unavailable_indices=list(f.FAILED),never_spawned_remaining_indices=f.REMAINING)),
             'job':job,'owner':dict(run_id=f.RUN,request_hash='r'*64,job_sha256='j'*64),'original_index_archive_ref':old}
    objects.update({v['path']:dict(case_index=i,mode='labels',parent_job_sha256='j'*64,result_ref=cases[str(i)]) for i,v in ((i,old[str(i)]) for i in f.PRESERVED)})
    return refs,cases,objects

class FailureAccounting(unittest.TestCase):
    def test_archived147_exact_refs_partition_and_root_acceptance(self):
        refs,cases,objects=context_fixture()
        with patch.object(f,'json_ref',side_effect=lambda r:objects[r['path']]),patch.object(f,'pin'):
            context=f.load_recovery(refs,cases,'d'*64,f.SCIENCE)
            self.assertEqual(context['contract']['failed_case_indices'],[295,297])
            cases['293']=dict(path='replacement',sha256='x'*64)
            with self.assertRaisesRegex(ValueError,'published147'):f.load_recovery(refs,cases,'d'*64,f.SCIENCE)

    def test_unknown_disposition_never_grants_label_or_cost_credit(self):
        entry=dict(identity={},accounting={},evidence_refs={})
        context=dict(contract=dict(failed_cases={'295':entry},operational_commit='a'*40),refs={'contract_ref':{},'root_review_ref':{}})
        for row in ({'schema':'NormalLabel'},{'schema':'R7OperationalFailedLabelDispositionV1','qualified':True}):
            with self.assertRaises(ValueError):f.audit_failed(row,295,{}, {},context)
        self.assertEqual(f.phase_accounting(dict(prepare=576,T=575,D5=574),[295,297])['actual_complete_label_phase_seconds'],None)
        with self.assertRaises(ValueError):f.phase_accounting(dict(prepare=576,T=576,D5=574),[295,297])

    def test_576_denominator_thresholds_keep_failed2_and_qualified_only_T(self):
        rows=[dict(qualified=True,nontrivial=i<116,capability=i<6,mother_id=str(i),label_origin='original_T_terminal') for i in range(576)]
        for i in f.FAILED:rows[i]=dict(qualified=False,nontrivial=False,capability=False,mother_id=str(i),label_origin='no_T_terminal_infrastructure_interruption')
        result=gate(rows);self.assertTrue(result['passed']);self.assertEqual(result['qualified'],574);self.assertEqual(result['denominator'],576)
        for i in range(56):rows[i]['qualified']=False
        self.assertFalse(gate(rows)['passed'])
        with self.assertRaises(ValueError):gate(rows[:-2])

    def test_failure_evidence_hash_is_checked_before_any_payload_access(self):
        refs={'contract_ref':{},'root_review_ref':{}}
        entry=dict(identity={},accounting={},evidence_refs={'original':dict(path='changed',sha256='x'*64)})
        c=dict(failed_cases={'295':entry},operational_commit='a'*40)
        row=dict(schema='R7OperationalFailedLabelDispositionV1',run_id=f.RUN,input_case_index=295,identity={},scientific_commit=f.SCIENCE,
                 qualified=False,nontrivial=False,capability=False,cost_closed=False,status='infrastructure_interrupted',original_attempts_retried=False,
                 known_unknown_accounting={},original_evidence_refs=entry['evidence_refs'],root_contract_ref={},root_review_ref={},operational_commit='a'*40)
        with patch.object(f,'pin',side_effect=ValueError('SHA differs')) as check:
            with self.assertRaisesRegex(ValueError,'SHA differs'):f.audit_failed(row,295,{}, {},dict(contract=c,refs=refs))
            check.assert_called_once_with(entry['evidence_refs']['original'])

    def test_saved_partial295_consumes_reserve_without_T_or_wall_estimate(self):
        from unittest.mock import MagicMock
        ident=dict(role='train',profile='uniform21600000',case_id='mock295',mother_id='mock_mother',shared_case_content_sha256='c'*64,source_asset_SHA='s'*64,identity_sha256='i'*64)
        evidence={f'case_workers__295__bundles__295__preparation__{name}':dict(path='preparation/'+name,sha256='e'*64) for name in ('stage_output.json','request.json','started.json','stage.wal.json')}
        evidence['case_workers__295__bundles__295__scene.npz']=dict(path='scene.npz',sha256='n'*64)
        ledger=dict(raypairs=12,raypairs_max=2048,seconds=.01)
        reserve=dict(prepare_raypairs=2048,T_raypairs=0,T_proposals=0,T_refits=0,D5_raypairs=0,D5_proposals=0,D5_refits=0)
        accounting=dict(n=2,attempted=dict(prepare=1,T=0,D5=0),actual_complete_label_cost_seconds=None,unknown_cost_preserved=True,original_preparation_ledger=ledger,
                        conservative_budget_charged=reserve,preparation_guard_closed=False,unknown_stage='preparation_guard')
        entry=dict(identity=ident,accounting=accounting,evidence_refs=evidence);refs=dict(contract_ref={},root_review_ref={})
        c=dict(failed_cases={'295':entry},operational_commit='a'*40,original_job=dict(source_manifest_sha256='p'*64))
        row=dict(schema='R7OperationalFailedLabelDispositionV1',run_id=f.RUN,input_case_index=295,identity=ident,scientific_commit=f.SCIENCE,
                 qualified=False,nontrivial=False,capability=False,cost_closed=False,status='infrastructure_interrupted',original_attempts_retried=False,
                 known_unknown_accounting=accounting,original_evidence_refs=evidence,root_contract_ref={},root_review_ref={},operational_commit='a'*40)
        source=dict(case_id='mock295',original_mother_id='mock_mother',case_content_sha256='c'*64,case_ref={})
        phase=dict(stage='labels',role='train',admitted=True,dataset_sha256='d'*64,identity_sha256='i'*64,cases={'mock295':ident})
        objects={'preparation/stage_output.json':dict(identity=ident,stage='preparation',asset_ref={}),
                 'preparation/request.json':dict(identity=ident,scene_ref=evidence['case_workers__295__bundles__295__scene.npz'],source_sha256=f.CORE,producer_sha256='p'*64,stage='preparation',phase_manifest=phase),
                 'preparation/started.json':dict(science_call_reserved=True,request_sha256='e'*64),
                 'preparation/stage.wal.json':dict(status='child_started',hard_seconds=20,owned_parent_children_rss_bytes=2147483648,pid=1,birth=2,
                       command=['mockPython','-B','-u','-m','satellite_rl.workbench.r7_stage','--request','preparation/request.json','--directory','preparation'])}
        context=dict(contract=c,refs=refs,addendum=dict(cases={'295':dict(original_identity=ident,original_preparation_ledger=ledger)}))
        record=dict(content_sha256='c'*64,source_asset_SHA='s'*64,aircraft_ids=['mockID1','mockID2'])
        with patch.object(f,'pin',return_value=MagicMock()),patch.object(f,'json_ref',side_effect=lambda r:objects[r['path']]),patch.object(f,'read_asset',return_value=dict(identity=ident,source_sha256=f.CORE,python='mockPython',result=dict(ledger=ledger))),patch('satellite_rl.shared_interface.content.read_wire',return_value=record),patch('satellite_rl.workbench.r7_case_binding.validate_binding'):
            checked,cost=f.audit_failed(row,295,source,dict(sha256='d'*64),context)
            self.assertFalse(checked['qualified']);self.assertEqual(cost['actual_calls'],dict(prepare=1,T=0,D5=0));self.assertIsNone(cost['actual_complete_label_seconds']);self.assertEqual(cost['conservative_budget_charged']['prepare_raypairs'],2048)
            accounting['conservative_budget_charged']['prepare_raypairs']=12
            with self.assertRaisesRegex(ValueError,'full authorized cap'):f.audit_failed(row,295,source,dict(sha256='d'*64),context)

if __name__=='__main__':unittest.main()
