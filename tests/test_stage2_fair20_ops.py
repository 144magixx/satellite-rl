"""Resource ownership and stable request regression checks; no remote compute."""
import tempfile
import unittest
import io
import json
from unittest.mock import patch
from datetime import datetime,timezone
from pathlib import Path
from scripts.stage2_fair20 import ops,resources,plan_builder,controller
from satellite_rl.workbench.fair20_data import select_mothers,partition_assets

class Fair20OpsTests(unittest.TestCase):
    def test_canonical_control_root_and_large_post_uses_stdin(self):
        self.assertEqual(ops.LOCAL,Path('D:/paper/outputs/experiments/s2_fair20_20261012'))
        class Channel:
            def shutdown_write(self):pass
            def recv_exit_status(self):return 0
        class Stream(io.BytesIO):
            channel=Channel()
        class Client:
            def __init__(self):self.input=Stream();self.command=None
            def exec_command(self,command,timeout):
                self.command=command
                return self.input,Stream(b'{"status":200,"body":{"ok":true}}'),Stream()
            def close(self):pass
        client=Client();body={'request_id':'s2f20-sourceprep','metadata':'x'*200000}
        with patch.object(ops,'connect',return_value=client):
            self.assertEqual(ops.api(20474,'/api/fair20/sourceprep',body),{'ok':True})
        self.assertLess(len(client.command.encode()),4096)
        self.assertEqual(json.loads(client.input.getvalue()),body)
    def observation(self):
        return {'port':20474,'observed_utc':datetime.now(timezone.utc).isoformat(),'gpu':'GPU-pin, NVIDIA 4090 D, 4, 24572, 0, 42','gpu_query_exit':0,'compute_query_exit':0,'compute_apps':'','disk_free':30*1024**3,'memory':{'MemAvailable':20*1024**3},'cpu_affinity':[0,1],'cgroup':{'cpu.max':'200000 100000'},'processes':[],'old_tasks':{'tasks':[]}}
    def release(self):
        return {'scientific_commit':'a'*40,'port':20474,'old_lane_all_work_terminal':True,'old_future_training_reserved':False,'evidence_path':'frozen-proof.json','gpu_uuid':'GPU-pin'}
    def test_idle_gpu_does_not_release_old_scheduler(self):
        r=self.release();r['old_future_training_reserved']=True
        self.assertIn('old scheduler still owns lane',resources.admission(self.observation(),r,'a'*40))
    def test_busy_gpu_blocks(self):
        o=self.observation();o['compute_apps']='100, GPU-pin, 800'
        self.assertTrue(resources.admission(o,self.release(),'a'*40))
    def test_pending_old_job_blocks(self):
        o=self.observation();o['old_tasks']={'tasks':[{'job':{'run_id':'x'},'status':{'stage':'queued'}}]}
        self.assertIn('old API has pending or active work',resources.admission(o,self.release(),'a'*40))
    def test_empty_release_evidence_blocks(self):
        r=self.release();r.pop('evidence_path')
        self.assertTrue(resources.admission(self.observation(),r,'a'*40))
    def test_clean_admission(self):self.assertEqual(resources.admission(self.observation(),self.release(),'a'*40),[])
    def test_probe_ignores_itself_but_real_old_worker_blocks(self):
        command='python -c code with nondrl12 and satellite_rl literals'
        self.assertFalse(resources.process_candidate(123,123,command))
        self.assertTrue(resources.process_candidate(456,123,command))
        observation=self.observation();observation['probe_pid']=123
        observation['processes']=[{'pid':123,'birth_ticks':1,'command':command}]
        self.assertEqual(resources.admission(observation,self.release(),'a'*40),[])
        observation['processes'].append({'pid':456,'birth_ticks':2,'command':'python -m satellite_rl.workbench.worker /s1_nd12/archive'})
        self.assertIn('old lane worker or coordinator remains: 456',resources.admission(observation,self.release(),'a'*40))
    def test_exact_single_lane_plan(self):
        p=plan_builder.build('a'*40,{'train_record_path':'/input/record.json'})
        self.assertEqual({t['port'] for t in p['tasks']},{20474})
        self.assertEqual(len([t for t in p['tasks'] if t['kind']=='shorttune']),6)
        self.assertEqual(len([t for t in p['tasks'] if t['kind']=='train']),3)
        self.assertEqual(p['budgets']['complete_B_calls'],8328)
        self.assertEqual(len([t for t in p['tasks'] if t['kind']=='matrix']),1)
        prep=next(t for t in p['tasks'] if t['kind']=='prepare')
        self.assertNotIn('datasets',prep['body'])
        self.assertIn('sourceprep_run_id',prep['body'])
    def test_typed_reference_resolution(self):
        v=controller.resolve({'a':plan_builder.ref('x','run_id')},{'tasks':{'x':{'run_id':'stable'}}})
        self.assertEqual(v,{'a':'stable'})
    def test_task_identity_is_stable(self):
        p=plan_builder.build('a'*40,{})
        p['tasks'][0]['body']['request_id']='new-random-id'
        with self.assertRaises(ValueError):controller.validate_plan(p)
    def test_registration_idempotent_and_occupied_never_admitted(self):
        jobs=[];posts=[]
        def api(port,path,body=None,method=None):
            if path.endswith('/tasks'):return {'tasks':jobs}
            if path.endswith('/sourceprep'):
                posts.append(path);job={'run_id':'r1','request_id':body['request_id'],'git_commit':'a'*40}
                jobs.append({'job':job,'status':{'stage':'queued_resources'}});return {'job':job}
            if path.endswith('/tasks/r1'):return jobs[0]
            raise AssertionError('unexpected admission API')
        plan={'experiment_id':'S2-F20','scientific_commit':'a'*40,'allowed_ports':[20474],'tasks':[plan_builder.task('sourceprep',20474,'sourceprep',{},[])]}
        with tempfile.TemporaryDirectory() as temp:
            c=controller.Controller(plan,temp,api=api,probe=lambda _:self.observation())
            self.assertTrue(c.tick());self.assertTrue(c.tick())
            self.assertEqual(posts,['/api/fair20/sourceprep'])
            self.assertEqual(c.state['tasks']['sourceprep']['stage'],'queued_resources')
    def test_actual_running_and_attention_worker_update_aggregate_heartbeat(self):
        job={'run_id':'r1','request_id':'s2f20-sourceprep','git_commit':'a'*40}
        status={'stage':'queued_resources'}
        def api(port,path,body=None,method=None):
            if path.endswith('/tasks'):return {'tasks':[{'job':job,'status':status}]}
            if path.endswith('/tasks/r1'):return {'job':job,'status':status}
            raise AssertionError('unexpected API submission')
        plan={'experiment_id':'S2-F20','scientific_commit':'a'*40,'allowed_ports':[20474],'tasks':[plan_builder.task('sourceprep',20474,'sourceprep',{},[])]}
        with tempfile.TemporaryDirectory() as temp:
            c=controller.Controller(plan,temp,api=api)
            self.assertTrue(c.tick());self.assertEqual(c.state['stage'],'queued_resources')
            status['stage']='preparing'
            self.assertTrue(c.tick());self.assertEqual(c.state['stage'],'running')
            manifest=json.loads((Path(temp)/'process.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['heartbeat_at'],c.state['heartbeat_at'])
            self.assertEqual(manifest['stage'],'running')
            self.assertIn('observed_utc',c.state)
            status['stage']='attention_required'
            self.assertFalse(c.tick());self.assertEqual(c.state['stage'],'attention_required')
    def test_upload_parent_traversal_blocked_before_connect(self):
        with self.assertRaises(ValueError):ops.upload(Path('missing'),ops.BASE+'/20474/../20418/file',20474)
    def test_source_selection_spacing_excludes_before_scores(self):
        c=[{'selected_time_utc':f'2024-02-13T00:{i:02d}:00+00:00','eligible_200':True} for i in range(0,60,5)]
        self.assertEqual(select_mothers(c,2,{c[0]['selected_time_utc']}),[1,5])
    def test_published_partition_sources_do_not_disclose_test_to_validation(self):
        receipts=[{'date':date,'path':'/raw/'+date+'.tar','sha256':'a'*64} for date in ('2024-02-13','2024-02-14','2024-02-15')]
        validation=partition_assets('validation',receipts)
        test=partition_assets('test',receipts)
        self.assertEqual([r['date'] for r in validation],['2024-02-13'])
        self.assertEqual([r['date'] for r in test],['2024-02-14','2024-02-15'])
        self.assertNotIn('/raw/2024-02-14.tar',str(validation))
        self.assertEqual(len(receipts),3)

if __name__=='__main__':unittest.main()
