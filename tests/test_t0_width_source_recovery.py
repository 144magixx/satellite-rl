import hashlib
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import torch
from satellite_rl.t0_width.logging import atomic_json,content_sha
from satellite_rl.t0_width.workbench import Workbench,validate_engineering_source_recovery,sha,S11_RECOVERY_RUN_ID,read

class SourceRecoveryTests(unittest.TestCase):
    def fixture(self,root):
        directory=root/'runs'/S11_RECOVERY_RUN_ID;directory.mkdir(parents=True)
        old=root/'old';new=root/'new';manifests=[]
        for source in (old,new):
            manifest={}
            for relative in ('src/fixture.py','scripts/fixture.py'):
                p=source/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture=True\n')
                manifest[relative]=sha(p)
            manifests.append(manifest)
        old_plan=root/'oldplan.json';new_plan=root/'newplan.json'
        atomic_json(old_plan,{'source_root':str(old),'source_manifest':manifests[0]})
        atomic_json(new_plan,{'authorized':True,'source_commit':'b'*40,'source_root':str(new),'source_manifest':manifests[1]})
        config={'phase':'engineering','group':'S11','training':{'total_steps':2048,'seed':73026},'model_dir':str(root/'models')}
        configsha=content_sha(config);atomic_json(directory/'config.json',config)
        status={'run_id':S11_RECOVERY_RUN_ID,'task_key':'engineering_S11','mode':'engineering','request_sha256':S11_RECOVERY_RUN_ID+'0'*40,
                'plan_sha256':sha(old_plan),'source_commit':'a'*40,'status':'needs_root_recovery_accounting'}
        atomic_json(directory/'status.json',status)
        atomic_json(directory/'job.json',{'status':status,'source_root':str(old),'source_manifest':manifests[0],'config_sha256':configsha})
        model=Path(config['model_dir']);model.mkdir()
        logrefs={};offsets={}
        for name in ('steps','episodes','evaluations'):
            p=directory/'logs'/(name+'.jsonl.gz');p.parent.mkdir(exist_ok=True);p.write_bytes(b'prefix')
            logrefs[name]={'path':str(p),'sha256':sha(p),'bytes':6};offsets[name]=6
        current=model/'current.pt'
        torch.save({'version':'W26ResumeV1','config_sha':configsha,'agent':{},'progress':{'env_step':962},'scene_rng':None,
                    'rng':{},'current_case':None,'environment':None,'observation':None,'log_offsets':offsets},current)
        wal=directory/'training_decision_wal.pt';torch.save({'stage':'committed','env_step':962,'config_sha':configsha,'log_offsets':offsets},wal)
        newref={'path':str(new_plan),'sha256':sha(new_plan)}
        receipt={'schema':'W26EngineeringSourceRecoveryV1','root_approved':True,'passed':True,'experiment_id':'S2-W26',
            'run_id':S11_RECOVERY_RUN_ID,'executor_port':20424,'env_step':962,'original_job_SHA':sha(directory/'job.json'),
            'original_config_file_SHA':sha(directory/'config.json'),'original_config_content_SHA':configsha,
            'original_plan_ref':{'path':str(old_plan),'sha256':sha(old_plan)},'original_source_commit':'a'*40,
            'new_plan_ref':newref,'new_source_root':str(new),'new_source_commit':'b'*40,'new_source_manifest':manifests[1],
            'current_ref':{'path':str(current),'sha256':sha(current)},'decision_wal_ref':{'path':str(wal),'sha256':sha(wal)},'log_refs':logrefs}
        rpath=root/'receipt.json';atomic_json(rpath,receipt)
        return directory,receipt,rpath,newref,current

    def run_validation(self,directory,receipt_path,newref,current):
        with patch('satellite_rl.t0_width.workbench.S11_RECOVERY_CURRENT_SHA',sha(current)):
            return validate_engineering_source_recovery(directory,{'path':str(receipt_path),'sha256':sha(receipt_path)},newref,20424)

    def test_exact962_source_repair_passes_without_mutating_old_assets(self):
        with tempfile.TemporaryDirectory()as f:
            d,r,p,n,c=self.fixture(Path(f));before={name:sha(d/name)for name in ('job.json','config.json','status.json')}
            self.run_validation(d,p,n,c)
            self.assertEqual(before,{name:sha(d/name)for name in before})

    def test_unique_log_suffix_rejected(self):
        with tempfile.TemporaryDirectory()as f:
            d,r,p,n,c=self.fixture(Path(f))
            with (d/'logs/steps.jsonl.gz').open('ab')as stream:stream.write(b'new_spend')
            with self.assertRaisesRegex(ValueError,'suffix'):
                self.run_validation(d,p,n,c)

    def test_intent_or_newer_wal_spend_rejected(self):
        with tempfile.TemporaryDirectory()as f:
            d,r,p,n,c=self.fixture(Path(f));wal=Path(r['decision_wal_ref']['path'])
            value=torch.load(wal,weights_only=False);value.update(stage='intent',env_step=963);torch.save(value,wal)
            r['decision_wal_ref']['sha256']=sha(wal);atomic_json(p,r)
            with self.assertRaisesRegex(ValueError,'committed962'):
                self.run_validation(d,p,n,c)

    def test_other_group_or_missing_root_approval_rejected(self):
        with tempfile.TemporaryDirectory()as f:
            d,r,p,n,c=self.fixture(Path(f));r['root_approved']=False;atomic_json(p,r)
            with self.assertRaisesRegex(ValueError,'only exact S11'):
                self.run_validation(d,p,n,c)

    def test_controller_same_id_sidecar_is_once_only_and_preserves_job(self):
        with tempfile.TemporaryDirectory()as f:
            d,r,p,n,c=self.fixture(Path(f));oldjob=sha(d/'job.json');oldconfig=sha(d/'config.json')
            controller=Workbench.__new__(Workbench);controller.lock=threading.RLock();controller.plan_path=Path(n['path'])
            controller.plan_sha=n['sha256'];controller.executor_port=20424;controller.max_workers=2
            controller.status_path=lambda run:d/'status.json';controller.runs=lambda:[]
            controller.validate_admission=lambda mode:None
            launches=[]
            def launch(run,**kwargs):launches.append((run,kwargs));return read(d/'status.json')
            controller.launch=launch
            body={'plan_sha256':n['sha256'],'recovery_admission_ref':{'path':str(p),'sha256':sha(p)}}
            with patch('satellite_rl.t0_width.workbench.S11_RECOVERY_CURRENT_SHA',sha(c)):
                result=controller.recover_engineering_source(S11_RECOVERY_RUN_ID,body)
                with self.assertRaisesRegex(ValueError,'already dispatched'):
                    controller.recover_engineering_source(S11_RECOVERY_RUN_ID,body)
            self.assertEqual(len(launches),1);self.assertEqual(launches[0][0],S11_RECOVERY_RUN_ID)
            self.assertEqual(launches[0][1]['plan_path'],r['original_plan_ref']['path'])
            self.assertEqual(sha(d/'job.json'),oldjob);self.assertEqual(sha(d/'config.json'),oldconfig)
            self.assertTrue(Path(result['source_recovery_ref']['path']).exists())

if __name__=='__main__':unittest.main()
