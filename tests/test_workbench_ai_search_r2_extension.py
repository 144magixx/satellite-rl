"""Finite operation-layer extension checks; no solver or remote execution."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from satellite_rl.coverage.ai_search.config import GeometryConfig
from satellite_rl.coverage.ai_search_r2.config import TaskConfig
from satellite_rl.workbench import coverage_ai_search_r2 as policy,ai_search_r2_extension as ext
from satellite_rl.workbench.storage import digest
from scripts.stage1_ai_search_r2 import plan_builder,admission_bundle,batch_metadata,upgrade
from satellite_rl.workbench.ai_search_r2_runner import production_provenance_metadata

BASE=Path('D:/paper/outputs/experiments/first_stage_ai_search_20261013')


@unittest.skipUnless((BASE/'round2/analysis/pilot/root_gate_acceptance.json').is_file(),'requires original signed pilot evidence in D:/paper outputs')
class ExtensionAdmission(unittest.TestCase):
    """Read-only integration with root-accepted signed evidence; no solve."""
    @classmethod
    def setUpClass(cls):
        cls.original_bundle,cls.original_uploads=admission_bundle.build(BASE/'round2/analysis/pilot/root_gate_acceptance.json',BASE/'round2/engineering/root_source_freeze_acceptance.json')

    def setUp(self):
        self.design=Path(__file__).resolve().parents[1]/'scripts/stage1_ai_search_r2/round2_design.json'
        bundle,uploads=deepcopy(self.original_bundle),self.original_uploads
        old_refs={k:bundle[k] for k in ('root_gate','audit','gate','source_freeze','interpretation')}
        for key,ref in old_refs.items():bundle[key]={'path':uploads[ref['path']]['local_path'],'sha256':ref['sha256']}
        bundle['aliases']={old:next(bundle[k] for k in old_refs if old_refs[k]==ref) for old,ref in bundle['aliases'].items()}
        self.bundle=bundle
        mothers=[f'm{i}' for i in range(20)];indices=[i for i in range(80) if i//4 not in (0,4,8,11,15,19)]
        self.inventory={'phase':'extension','mother_order':mothers,'selected_mothers':[mothers[i] for i in range(20) if i not in (0,4,8,11,15,19)],
            'selected_case_indices':indices,'development_identities':{'mother_ids':[mothers[i//4] for i in range(80)]},
            'source_refs':{str(i):[{'method':m,'path':'/engineering/source','sha256':'d'*64,'audit_path':'/engineering/audit','audit_sha256':'e'*64} for m in ('07-CV','07-LEX','JOINT-CV','JOINT-LEX','POST_SHRINK')] for i in indices}}
        self.prepared=plan_builder.prepare_plan('c'*40,self.design,self.inventory,{'path':'/engineering/packed.npz','sha256':'a'*64,'role':'development','count':80},bundle)
        self.plan=plan_builder.evaluation_plan(self.prepared,{str(p):{'path':'/engineering/common.json','sha256':'b'*64} for p in (20418,20424,20473)})
        self.plan['design']={'path':str(self.design),'sha256':digest(self.design)}
        self.task=next(iter(self.plan['tasks'].values()))
        self.core=patch.object(ext,'actual_core_sha256',return_value=bundle['scientific_core_sha256']);self.core.start();self.addCleanup(self.core.stop)
        self.dependencies=patch.object(ext,'actual_dependency_sha256',return_value=bundle['scientific_dependency_sha256']);self.dependencies.start();self.addCleanup(self.dependencies.stop)

    def test_exact_remaining56_896slots_and_original_g3_failure_allowed(self):
        self.assertEqual(plan_builder.audit(self.prepared,self.inventory)['cases'],56)
        self.assertEqual(plan_builder.audit(self.plan,self.inventory)['decoder_task_calls'],896)
        ext.validate(self.plan)
        self.assertEqual(sorted(len(t['case_indices']) for t in self.plan['tasks'].values()),[16,20,20])

    def test_no_or_wrong_gate_original_sha_rejected(self):
        changed=deepcopy(self.plan);changed.pop('extension_admission')
        with self.assertRaisesRegex(ValueError,'root gate'):ext.validate(changed)
        for key in ('audit','gate','source_freeze'):
            changed=deepcopy(self.plan);changed['extension_admission'][key]['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'exact root-accepted'):ext.validate(changed)

    def test_alias_cannot_change_expected_sha(self):
        changed=deepcopy(self.plan);path=next(iter(changed['extension_admission']['aliases']))
        changed['extension_admission']['aliases'][path]['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'SHA'):ext.validate(changed)

    def test_core_or_other_scientific_dependency_change_rejected(self):
        with patch.object(ext,'actual_core_sha256',return_value={'different':'0'*64}):
            with self.assertRaisesRegex(ValueError,'core bytes'):ext.validate(self.plan)
        with patch.object(ext,'actual_dependency_sha256',return_value={'different':'0'*64}):
            with self.assertRaisesRegex(ValueError,'dependency'):ext.validate(self.plan)
        changed=deepcopy(self.plan);changed['scientific_dependency_sha256'].pop(next(iter(changed['scientific_dependency_sha256'])))
        with self.assertRaisesRegex(ValueError,'dependency'):ext.validate(changed)

    def test_remaining_scope_rejects_pilot_replay_and_split_mother(self):
        def fixture_pin(ref):
            return Path(ref['path']) if Path(ref['path']).exists() else self.design
        with patch.object(policy,'pin',side_effect=fixture_pin):
            policy.validate_task(self.task,self.plan,'20418')
            changed=deepcopy(self.plan);changed['selected_case_indices'][0]=0
            with self.assertRaisesRegex(ValueError,'scope'):policy.validate_task(self.task,changed,'20418')
            changed=deepcopy(self.plan);tasks=list(changed['tasks'].values());tasks[0]['case_indices'][0],tasks[1]['case_indices'][0]=tasks[1]['case_indices'][0],tasks[0]['case_indices'][0]
            with self.assertRaisesRegex(ValueError,'split'):policy.validate_task(tasks[0],changed,'20418')

    def test_original_f02_qualification_reuse_without_relabeling(self):
        root=BASE/'round2/engineering'
        self.plan['qualification_refs']={name:{'path':str(root/filename),'sha256':ext.PILOT_QUALIFICATION_SHA[name]} for name,filename in
            [('validator_equivalence','qualification_validator_equivalence.json'),('charge_before_budget','qualification_charge_before_budget.json')]}
        with TemporaryDirectory() as temp:
            path=Path(temp)/'review.json';plan_ref={'sha256':'a'*64}
            path.write_text(json.dumps({'group':policy.GROUP,'passed':True,'scientific_commit':'c'*40,'plan_sha256':'a'*64,'design_sha256':ext.DESIGN_SHA}),encoding='utf-8')
            policy.validate_admission(self.plan,plan_ref,{'path':str(path),'sha256':digest(path)})
            with patch.object(ext,'actual_core_sha256',return_value={'changed':'f'*64}):
                with self.assertRaisesRegex(ValueError,'core bytes'):policy.validate_admission(self.plan,plan_ref,{'path':str(path),'sha256':digest(path)})

    def test_new_epoch_cannot_submit_old_pilot(self):
        with TemporaryDirectory() as temp:
            p=Path(temp)/'review.json';p.write_text(json.dumps({'group':policy.GROUP,'passed':True,'scientific_commit':'c'*40,'plan_sha256':'a'*64,'design_sha256':ext.DESIGN_SHA}),encoding='utf-8')
            changed=deepcopy(self.plan);changed['phase']='pilot'
            with self.assertRaisesRegex(ValueError,'cannot be repeated'):policy.validate_admission(changed,{'sha256':'a'*64},{'path':str(p),'sha256':digest(p)})

    @unittest.skipUnless((BASE/'round2/analysis/extension_source/contracts.py').is_file(),'requires independent sealed extension reader')
    def test_actual_producer_metadata_to_independent_contract_reader(self):
        import importlib.util
        path=BASE/'round2/analysis/extension_source/contracts.py';spec=importlib.util.spec_from_file_location('independent_extension_contracts',path)
        reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
        mapping=json.loads((BASE/'round2/analysis/pilot/path_map.json').read_text(encoding='utf-8'))
        resolver=reader.base.Resolver(mapping)
        job={'scientific_commit':'c'*40,'case_indices':self.plan['selected_case_indices'],
            'references':{'extension_'+key:self.bundle[key] for key in ('root_gate','audit','gate','source_freeze','interpretation')}}
        wire=production_provenance_metadata(job,self.plan)
        context=reader.load_context(wire['admission_ref'],wire['admission_evidence_refs'],resolver,self.plan['design'])
        self.assertEqual(wire['phase'],'extension');self.assertEqual(wire['algorithm_core_commit'],reader.CORE)
        self.assertEqual(wire['execution_commit'],'c'*40);self.assertEqual(context['pilot']['actual_cases'],24)
        self.assertEqual(set(wire['admission_evidence_refs']),{'root_gate','audit','gate','source_freeze','interpretation'})
        self.assertEqual(wire['scientific_dependency_sha256'],self.plan['scientific_dependency_sha256'])

    def test_upgrade_generated_code_and_batch_member_guard(self):
        for port in (20418,20424,20473):
            for execute in (False,True):compile(upgrade.lane_code(port,execute),'upgrade-engineering-only','exec')
        with TemporaryDirectory() as temp:
            path=Path(temp)/'fixture.json';path.write_text('{}')
            good={'local':str(path),'remote':'/home/u2024110224/first_stage_satellite_rl/s1_ai18/source/fixture.json','sha256':digest(path)}
            batch_metadata.validate_members([good])
            bad={**good,'remote':good['remote'].replace('/s1_ai18/','/s1_ai17/')}
            with self.assertRaisesRegex(ValueError,'outside'):batch_metadata.validate_members([bad])
            with self.assertRaisesRegex(ValueError,'duplicate'):batch_metadata.validate_members([good,good])


class StandaloneOperationChecks(unittest.TestCase):
    def test_remaining56_generation_needs_no_signed_external_assets(self):
        mothers=[f'm{i}' for i in range(20)];selected=[i for i in range(80) if i//4 not in (0,4,8,11,15,19)]
        inventory={'phase':'extension','mother_order':mothers,'selected_mothers':[mothers[i] for i in range(20) if i not in (0,4,8,11,15,19)],
            'selected_case_indices':selected,'development_identities':{'mother_ids':[mothers[i//4] for i in range(80)]},
            'source_refs':{str(i):[{'method':m} for m in ('07-CV','07-LEX','JOINT-CV','JOINT-LEX','POST_SHRINK')] for i in selected}}
        bundle={'root_gate':{'path':'/engineering/admission.json','sha256':'a'*64},'scientific_core_sha256':{'fixture':'b'*64},'scientific_dependency_sha256':{'fixture':'b'*64}}
        design=Path(__file__).resolve().parents[1]/'scripts/stage1_ai_search_r2/round2_design.json'
        plan=plan_builder.prepare_plan('c'*40,design,inventory,{'path':'/engineering/input.npz','sha256':'d'*64,'role':'development','count':80},bundle)
        self.assertEqual(plan_builder.audit(plan,inventory)['cases'],56)
        evaluated=plan_builder.evaluation_plan(plan,{str(p):{'path':'/engineering/common.json','sha256':'e'*64} for p in (20418,20424,20473)})
        self.assertEqual(plan_builder.audit(evaluated,inventory)['decoder_task_calls'],896)
        self.assertFalse(set(selected)&{4*m+p for m in (0,4,8,11,15,19) for p in range(4)})

    def test_upgrade_dryrun_syntax_and_no_fourth_lane(self):
        for port in (20418,20424,20473):
            compile(upgrade.lane_code(port,False),'readonly-upgrade-check','exec')
            compile(upgrade.lane_code(port,True,{'old_registry_sha256':'a'*64}),'bounded-service-stop','exec')
        with self.assertRaises(ValueError):upgrade.lane_code(20474,False)

    def test_metadata_batch_preserves_original_sha_and_rejects_escape(self):
        with TemporaryDirectory() as temp:
            path=Path(temp)/'fixture.json';path.write_bytes(b'{"fixture":true}\r\n')
            record={'local':str(path),'remote':'/home/u2024110224/first_stage_satellite_rl/s1_ai18/source/fixture.json','sha256':digest(path)}
            batch_metadata.validate_members([record])
            with self.assertRaises(ValueError):batch_metadata.validate_members([{**record,'sha256':'0'*64}])
            with self.assertRaises(ValueError):batch_metadata.validate_members([{**record,'remote':'/tmp/other.json'}])

    def test_unknown_new_startup_reply_recovers_original_new_registry(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);review=root/'review.json';review.write_text(json.dumps({'passed':True,'scientific_commit':'c'*40}))
            for mode in ('prepare_common','closed_pool'):
                d=root/'control'/mode;d.mkdir(parents=True)
                (d/'launch.json').write_text(json.dumps({'pid':999999,'process_start_time':1}))
                (d/'state.json').write_text(json.dumps({'phase':'completed'}))
            output=root/'upgrade.json';journal=Path(str(output)+'.journal.json')
            journal.write_text(json.dumps({'new_commit':'c'*40,'checks':{},'stop_intents':{},
                'stopped':{'20418':{'old_registry_sha256':'a'*64}},'started':{'20424':{},'20473':{}}}))
            captured=[]
            def remote(port,code):
                compile(code,'unknown-new-service-readonly-probe','exec');captured.append((port,code))
                self.assertIn('recovered_new_service',code)
                return {'recovered_new_service':True,'service':{'registry':{'source_commit':'c'*40}}}
            def git(*args):return 'c'*40 if args==('rev-parse','HEAD') else ''
            with patch.object(upgrade.ops,'LOCAL',root),patch.object(upgrade,'pid_alive',return_value=False),patch.object(upgrade.deploy,'git',side_effect=git),patch.object(upgrade.deploy,'remote',side_effect=remote),patch.object(upgrade.deploy,'lane') as lane:
                result=upgrade.upgrade('c'*40,review,output,execute=True)
                lane.assert_not_called()
            self.assertEqual(len(captured),1);self.assertEqual(set(result['new_services']),{'20418','20424','20473'})
            self.assertTrue(json.loads(journal.read_text())['started']['20418']['recovered_new_service'])


if __name__=='__main__':unittest.main()
