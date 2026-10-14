import json
from pathlib import Path
import tempfile
import subprocess
import sys
import time
import unittest
from unittest.mock import patch
from satellite_rl.t0_width.workbench import Workbench,sha,now,boot_id,alive,birth
from satellite_rl.t0_width.logging import atomic_json

SOURCE=Path(__file__).resolve().parents[1]

class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.gpu_patch=patch('satellite_rl.t0_width.workbench.gpu_uuids',return_value=['GPU-fixture']);self.gpu_patch.start()
        refs={}
        for name in ('source','resources','data','engineering'):
            path=self.root/(name+'.json');atomic_json(path,dict(root_approved=True,passed=True,experiment_id='S2-W26',source_commit='a'*40,executor_port=20418,checked_UTC=now(),boot_id=boot_id(),gpu_uuid='GPU-fixture'));refs[name]=dict(path=str(path),sha256=sha(path))
        manifest={p.relative_to(SOURCE).as_posix():sha(p) for folder in ('src','scripts') for p in (SOURCE/folder).rglob('*.py')}
        self.plan=self.root/'plan.json';atomic_json(self.plan,dict(schema='W26AuthorizedPlanV1',group='S2-W26',authorized=True,source_commit='a'*40,source_root=str(SOURCE),source_manifest=manifest,admission_refs=refs,tasks=dict(A=dict(executor_port=20418,mode='engineering',config=dict(group='S00',phase='engineering',training=dict(total_steps=2048,seed=73026,scene_order_seed=73026))),B=dict(executor_port=20424,mode='download',config={}))))
        self.workbench=Workbench(self.root/'lane',self.plan,20418)
    def tearDown(self):self.workbench.close();self.gpu_patch.stop();self.temp.cleanup()
    def request(self):return dict(request_id='one',task_key='A',plan_sha256=self.workbench.plan_sha)
    def fake_launch(self,run_id):return json.loads(self.workbench.status_path(run_id).read_text())
    def test_fixed_plan_idempotence_and_task_owner(self):
        with patch.object(self.workbench,'launch',side_effect=self.fake_launch):
            first=self.workbench.submit(self.request());second=self.workbench.submit(self.request())
            self.assertEqual(first['run_id'],second['run_id'])
            request=self.request();request['request_id']='two'
            with self.assertRaisesRegex(ValueError,'owner'):self.workbench.submit(request)
        request=self.request();request['task_key']='B'
        with self.assertRaisesRegex(ValueError,'another executor'):self.workbench.submit(request)
        request=self.request();request['config']={}
        with self.assertRaises(ValueError):self.workbench.submit(request)
    def test_source_and_admission_change_rejected(self):
        plan=json.loads(self.plan.read_text());plan['authorized']=False;atomic_json(self.plan,plan)
        with self.assertRaisesRegex(ValueError,'plan changed'):self.workbench.submit(self.request())
    def test_admission_sha(self):
        path=self.root/'data.json';atomic_json(path,dict(root_approved=False))
        with self.assertRaisesRegex(ValueError,'admission SHA'):self.workbench.submit(self.request())
    def test_restart_reconciliation_and_sameid_checkpoint(self):
        with patch.object(self.workbench,'launch',side_effect=self.fake_launch):status=self.workbench.submit(self.request())
        path=self.workbench.status_path(status['run_id']);status.update(status='running',pid=99999999,process_birth=1);atomic_json(path,status)
        self.workbench.reconcile();self.assertEqual(json.loads(path.read_text())['status'],'interrupted')
        with self.assertRaisesRegex(ValueError,'checkpoint'):self.workbench.control(status['run_id'],'resume',dict(plan_sha256=self.workbench.plan_sha))
        model=Path(status['model_dir']);model.mkdir(parents=True);(model/'current.pt').write_bytes(b'fixture only')
        with patch.object(self.workbench,'launch',side_effect=self.fake_launch):
            recovered=self.workbench.control(status['run_id'],'resume',dict(plan_sha256=self.workbench.plan_sha))
        self.assertEqual(recovered['run_id'],status['run_id'])
    def test_stop_and_live_lane_exclusion(self):
        with self.assertRaisesRegex(RuntimeError,'already owns'):Workbench(self.root/'lane',self.plan,20418)
        with patch.object(self.workbench,'launch',side_effect=self.fake_launch):status=self.workbench.submit(self.request())
        self.workbench.control(status['run_id'],'stop',dict(plan_sha256=self.workbench.plan_sha))
        self.assertTrue((self.workbench.status_path(status['run_id']).parent/'stop.request').exists())

    def test_real_bounded_child_exit_reaped(self):
        with patch.object(self.workbench,'launch',side_effect=self.fake_launch):status=self.workbench.submit(self.request())
        child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(0.15)'])
        try:
            status.update(status='completed',pid=child.pid,process_birth=birth(child.pid),boot_id=boot_id())
            atomic_json(self.workbench.status_path(status['run_id']),status)
            self.workbench.children[status['run_id']]=child
            self.assertTrue(alive(status))
            time.sleep(.3)
            self.workbench.reconcile()
            self.assertNotIn(status['run_id'],self.workbench.children)
            self.assertIsNotNone(child.returncode)
            self.assertFalse(alive(status))
        finally:
            if child.poll() is None:child.terminate()
            child.wait(timeout=5)

    def test_resource_receipt_expired_or_wrong_boot(self):
        path=self.root/'resources.json';record=json.loads(path.read_text());record['checked_UTC']='2000-01-01T00:00:00+00:00';atomic_json(path,record)
        self.workbench.plan['admission_refs']['resources']['sha256']=sha(path)
        with self.assertRaisesRegex(ValueError,'expired'):self.workbench.validate_admission('prepare')
        record['checked_UTC']=now();record['boot_id']='another boot';atomic_json(path,record)
        self.workbench.plan['admission_refs']['resources']['sha256']=sha(path)
        with self.assertRaisesRegex(ValueError,'boot/GPU'):self.workbench.validate_admission('prepare')

    def test_prepare_shards_complete_and_disjoint(self):
        tasks=self.workbench.plan['tasks']
        manifest=self.root/'manifest.json';atomic_json(manifest,dict(rows=[dict(role='train') for _ in range(636)]))
        for port,indices in [(20418,list(range(318))),(20424,list(range(318,636)))]:
            admission=self.root/('prepare'+str(port)+'.json');atomic_json(admission,dict(root_approved=True,manifest_SHA=sha(manifest),remote_port=port,case_indices=indices))
            tasks['P'+str(port)]=dict(mode='prepare',executor_port=port,config=dict(case_indices=indices,mixed_allowed=True,manifest_ref=dict(path=str(manifest),sha256=sha(manifest)),execution_admission_ref=dict(path=str(admission),sha256=sha(admission))))
        self.workbench.validate_plan()
        tasks['P20424']['config']['case_indices'][0]=317
        with self.assertRaises(ValueError):self.workbench.validate_plan()

    def test_mode_cannot_bypass_scientific_gate(self):
        config=self.workbench.plan['tasks']['A']['config']
        config['phase']='scientific';config['training']['total_steps']=150000
        with self.assertRaisesRegex(ValueError,'actual runner phase'):self.workbench.validate_plan()
        config['phase']='engineering'
        with self.assertRaisesRegex(ValueError,'step budget'):self.workbench.validate_plan()
        config['training']['total_steps']=2048;config['training']['seed']=51001
        with self.assertRaisesRegex(ValueError,'seed contract'):self.workbench.validate_plan()

    def test_data_mode_cannot_select_runner(self):
        self.workbench.plan['tasks']['B']['config']['task_kind']='confirmation'
        with self.assertRaisesRegex(ValueError,'runner selectors'):self.workbench.validate_plan()

    def test_mode_remote_executor_exact(self):
        ref_path=self.root/'download.json';atomic_json(ref_path,dict(remote_port=20424))
        self.workbench.plan['tasks']['D']=dict(mode='download',executor_port=20418,config=dict(download_admission_ref=dict(path=str(ref_path),sha256=sha(ref_path))))
        body=dict(request_id='wrongport',task_key='D',plan_sha256=self.workbench.plan_sha)
        with self.assertRaisesRegex(ValueError,'remote executor'):self.workbench.submit(body)

if __name__=='__main__':unittest.main()
