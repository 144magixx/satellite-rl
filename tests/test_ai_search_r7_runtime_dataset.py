"""TN04/TN05 pure startup/source mismatch checks; actual CUDA/model calls zero."""
import copy,json,tempfile,unittest,sys
from pathlib import Path,PurePosixPath
from types import SimpleNamespace
from contextlib import nullcontext
from unittest.mock import patch
from satellite_rl.training.run_monitor import atomic_json
from satellite_rl.workbench.r7_training_dataset import validate_training_dataset

def ref(name,sha='a'*64):return dict(path='/pure_mock/'+name,sha256=sha)

class ActualTrainingDatasetBindings(unittest.TestCase):
    def fixture(self):
        refs=dict(labels_gate=ref('gate'),qualified=ref('qualified'),authority=ref('authority'))
        dataset=ref('dataset');dataset.update(role='train',count=576)
        docs=dict(dataset=dict(schema='R7UniformDatasetV1',role='train',count=576,fixed_request_bps=21600000,authority_ref=refs['authority']),
                  gate=dict(group='S1-AI17-R7',phase='labels',passed=True,model_identity='one_r7',source_dataset_sha256=dataset['sha256']),
                  qualified=dict(schema='R7QualifiedTrainManifestV1',role='train',labels_gate_sha256=refs['labels_gate']['sha256'],source_dataset_sha256=dataset['sha256']))
        return refs,dataset,docs
    def test_actual_config_gate_and_qualified_must_use_same_source(self):
        refs,dataset,docs=self.fixture()
        with patch('satellite_rl.workbench.r7_training_dataset.pin',side_effect=lambda r:Path(r['path'])),patch('satellite_rl.workbench.r7_training_dataset.read_json',side_effect=lambda p,*a:docs[p.name]):
            self.assertEqual(validate_training_dataset(refs,dataset,'one_r7')['source_dataset_sha256'],dataset['sha256'])
            changed=dict(dataset,sha256='b'*64)
            with self.assertRaisesRegex(ValueError,'config training dataset'):validate_training_dataset(refs,changed,'one_r7')
    def test_independently_changed_qualified_or_gate_source_rejected(self):
        for key in ('qualified','gate'):
            refs,dataset,docs=self.fixture();docs[key]['source_dataset_sha256']='b'*64
            with self.subTest(key=key),patch('satellite_rl.workbench.r7_training_dataset.pin',side_effect=lambda r:Path(r['path'])),patch('satellite_rl.workbench.r7_training_dataset.read_json',side_effect=lambda p,*a:docs[p.name]):
                with self.assertRaises(ValueError):validate_training_dataset(refs,dataset,'one_r7')

class FreshActualWorkerOrder(unittest.TestCase):
    def run_fixture(self,initialized):
        import satellite_rl.workbench.r7_runner as subject
        import satellite_rl.workbench.r7_model_train as trainer
        from satellite_rl.coverage.ai_search_r6 import runtime_v3
        events=[];state={'initialized':initialized}
        fake_torch=SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:state['initialized']))
        identity={'pure_mock':'one_configured_identity'}
        def configure():
            runtime_v3.pre_cuda_environment();events.append('configure_before_cuda');state['initialized']=True;return identity
        def remote_dqn():
            events.append('deployment_cuda_query');self.assertTrue(state['initialized'])
        def verify(store,job,policy):policy.deployment(store)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);job=dict(run_id='pure_mock_runtime',scientific_commit='a'*40,mode='train2000',model_inputs={},archive_root='/home/u2024110224/first_stage_satellite_rl/s1_ai17/20474/round7/archive',source_root=str(root))
            atomic_json(root/'job.json',job);atomic_json(root/'config.json',dict(schema_version=16,research_stage='coverage'));atomic_json(root/'status.json',{})
            with patch.dict(sys.modules,{'torch':fake_torch}),patch('satellite_rl.workbench.r7_deadline.stop_boundary',return_value=True),patch.dict('os.environ',{'SATELLITE_RL_SSH_PORT':'20474','CUBLAS_WORKSPACE_CONFIG':':4096:8'}),patch.object(runtime_v3,'configure_v3_runtime',side_effect=configure) as configured,patch.object(subject,'Store',side_effect=lambda archive,**kwargs:SimpleNamespace(archive_root=SimpleNamespace(resolve=lambda:PurePosixPath(archive)))),patch.object(subject.policy,'Path',PurePosixPath),patch.object(subject,'exclusive_lock',side_effect=lambda *a,**k:nullcontext()),patch.object(subject.frozen_round_tasks,'verify',side_effect=verify) as verified,patch('satellite_rl.workbench.coverage._require_remote_dqn',side_effect=remote_dqn),patch.object(trainer,'train',return_value={'updates':0}) as train:
                if initialized:
                    with self.assertRaisesRegex(RuntimeError,'must precede any CUDA'):subject.run_job(root)
                    verified.assert_not_called();train.assert_not_called()
                else:
                    subject.run_job(root);configured.assert_called_once();train.assert_called_once()
                    self.assertEqual(json.loads((root/'runtime_v3_identity.json').read_text()),identity)
            return events
    def test_configure_precedes_actual_policy_deployment_cuda_query_once(self):self.assertEqual(self.run_fixture(False),['configure_before_cuda','deployment_cuda_query'])
    def test_already_initialized_cuda_rejected_before_verification_or_training(self):self.assertEqual(self.run_fixture(True),[])

if __name__=='__main__':unittest.main()
