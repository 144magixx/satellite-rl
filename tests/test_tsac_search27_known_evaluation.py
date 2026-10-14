import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from satellite_rl.tsac_search27.known_evaluation import jobs_for,run_known_task
from satellite_rl.t0_width.runner import RecoveryAccountingRequired

class KnownMatrixTests(unittest.TestCase):
    def test_dynamic_exact_common_budget(self):
        self.assertEqual(len(jobs_for([],50000))*40,680)
        self.assertEqual(len(jobs_for(['E'],150000))*40,560)
        self.assertEqual(len(jobs_for(['LP'],150000))*40,560)
        self.assertEqual(len(jobs_for(['E','LP'],150000))*40,680)
        for passed,node in [([],150000),(['E'],50000),(['E','E'],150000)]:
            with self.assertRaises(ValueError):jobs_for(passed,node)
    def test_stop_inside_repeat_then_cursor_resume_never_repeats_committed_rows(self):
        jobs=jobs_for(['E'],150000);calls=[];stop={'value':False}
        config=dict(run_id='matrix',node=150000,known_manifest_ref={'path':'manifest','sha256':'m'},
                    w26_original_unlock={'manifest_SHA':'m','manifest_ref':{'path':'manifest','sha256':'m'},'receipt_ref':{'path':'unlock'}},budget_episodes=560)
        manifest=dict(schema='W26PublishedDatasetV1',root_approved=True,role='test',rows=[{'case_id':str(i)} for i in range(40)])
        def read(ref):
            if ref['path']=='manifest':return manifest
            if ref['path']=='unlock':return dict(schema='W26ConfirmationUnlockV1',root_approved=True,manifest_SHA='m')
            import json
            return json.loads(Path(ref['path']).read_text())
        def evaluate(agent,rows,group,**kw):
            self.assertIs(kw['stop_requested'],request_stop)
            self.assertEqual(kw['seed_base'],275000);self.assertEqual(group,'S10')
            self.assertTrue(kw['phase'].startswith('known_'))
            cursor=kw['wal_context']['matrix_cursor']
            if cursor==2 and not stop['value']:stop['value']=True;return [],False
            calls.append(cursor);kw['row_committed'](1);return [],True
        def request_stop():return stop['value']
        with tempfile.TemporaryDirectory() as temp,patch('satellite_rl.tsac_search27.known_evaluation.validate_lock',return_value=({},jobs)),patch('satellite_rl.tsac_search27.known_evaluation.read_json_ref',side_effect=read),patch('satellite_rl.tsac_search27.known_evaluation.load_agents',return_value={m:object() for m,r in jobs}),patch('satellite_rl.tsac_search27.known_evaluation.evaluate_cases',side_effect=evaluate):
            first=run_known_task(config,temp,stop_requested=request_stop)
            self.assertEqual(first['stage'],'stopped_during_evaluation');self.assertEqual(first['completed_episodes'],2)
            stop['value']=False
            # Disable mock's one-time stop now, retain identical immutable task.
            original=evaluate
            def finish(agent,rows,group,**kw):
                calls.append(kw['wal_context']['matrix_cursor']);kw['row_committed'](1);return [],True
            with patch('satellite_rl.tsac_search27.known_evaluation.evaluate_cases',side_effect=finish):last=run_known_task(config,temp,stop_requested=request_stop)
            self.assertEqual(last['completed_episodes'],560);self.assertEqual(calls,list(range(560)))
    def test_shared_evaluator_stops_mid_repeat_and_keeps_exact_wal(self):
        import torch
        from satellite_rl.t0_width.runner import BackgroundPolicy,evaluate_cases
        from satellite_rl.t0_width.environment import T0WidthEnv
        from test_t0_width_runner import fixture
        class CountPolicy(BackgroundPolicy):
            def __init__(self):super().__init__('random','cpu');self.actions=0
            def act(self,*a,**kw):self.actions+=1;return super().act(*a,**kw)
        agent=CountPolicy()
        with tempfile.TemporaryDirectory() as temp:
            wal=Path(temp)/'wal.pt';record={'case_id':'fixture','role':'test'}
            result,done=evaluate_cases(agent,[record],'S10',loader=fixture,env_factory=T0WidthEnv,
                                      wal_path=wal,repeats=1,stop_requested=lambda:agent.actions>=2)
            self.assertFalse(done);self.assertEqual(agent.actions,2)
            self.assertEqual(torch.load(wal,weights_only=False)['stage'],'committed')
            result,done=evaluate_cases(agent,[record],'S10',loader=fixture,env_factory=T0WidthEnv,wal_path=wal,repeats=1)
            self.assertTrue(done);self.assertEqual(agent.actions,5);self.assertFalse(wal.exists())

    def test_geometry_source_and_git_commits_are_separately_locked(self):
        from satellite_rl.tsac_search27.known_evaluation import validate_lock
        provenance=dict(policy_ref={'path':'policy','sha256':'p'},schema={'variant':'B'},config_SHA='c',source_SHA='core9f',data_manifest_SHA='logical',env_step=150000)
        policies={'B':dict(provenance),'E':dict(provenance),'HIST':dict(provenance,data_manifest_SHA='historical5d')}
        c=dict(task_kind='known_evaluation',phase='known_scenes',run_id='r',node=150000,known_manifest_ref={'path':'old40'},
               terminal_lock_ref={'path':'lock'},policies=policies,source_sha='core9f',training_source_sha='core9f',
               data_manifest_sha='logical',training_source_commit='trainGit',evaluation_source_commit='evalGit',budget_episodes=560,historical_training_plan_ref={'path':'historyplan'})
        lock=dict(schema='W27KnownMatrixLockV1',root_approved=True,experiment_id='S2-W27',run_id='r',node=150000,
                  manifest_ref=c['known_manifest_ref'],policies=policies,source_SHA='core9f',data_manifest_SHA='logical',
                  training_source_commit='trainGit',evaluation_source_commit='evalGit',not_independent_test=True,
                  all_continuation_decisions_locked=True,decision_ref={'path':'decision'},passed_variants=['E'],historical_training_plan_ref={'path':'historyplan'})
        from satellite_rl.t0_width.logging import content_sha
        historical_config=dict(group='S10',source_sha='core9f',data_manifest_sha='historical5d')
        policies['HIST']['config_SHA']=content_sha(historical_config)
        def read(ref):
            if ref['path']=='lock':return lock
            if ref['path']=='historyplan':return dict(schema='W26AuthorizedPlanV1',authorized=True,tasks={'history':dict(mode='train',config=historical_config)})
            return dict(root_approved=True,passed_variants=['E'])
        with patch('satellite_rl.tsac_search27.known_evaluation.read_json_ref',side_effect=read):
            validate_lock(c)
            for bad in [dict(c,source_sha='evalGit'),dict(c,training_source_sha='trainGit'),dict(c,evaluation_source_commit='other')]:
                with self.assertRaises(ValueError):validate_lock(bad)
    def test_training_config_must_bind_logical_dataset_identity(self):
        from satellite_rl.tsac_search27.control import validate_training_admission
        gate=dict(schema='W27TrainingDataAdmissionV1',root_approved=True,experiment_id='S2-W27',source_SHA='core',
                  qualified_counts={'train':576,'development':20},logical_dataset_identity='logical')
        c=dict(data_admission_ref={'path':'gate'},source_sha='core',data_manifest_sha='perPortFileSHA')
        with patch('satellite_rl.tsac_search27.control.read_json_ref',return_value=gate):
            with self.assertRaisesRegex(ValueError,'logical dataset'):validate_training_admission(c)

    def test_root_lock_checked_before_known_manifest_open(self):
        with tempfile.TemporaryDirectory() as temp,patch('satellite_rl.tsac_search27.known_evaluation.validate_lock',side_effect=ValueError('closed')),patch('satellite_rl.tsac_search27.known_evaluation.read_json_ref') as read:
            with self.assertRaises(ValueError):run_known_task({},temp)
            read.assert_not_called()

if __name__=='__main__':unittest.main()
