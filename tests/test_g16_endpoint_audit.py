import copy
import hashlib
import json
import random
import unittest
import numpy as np
import torch
from scripts.stage1_geoload.endpoint_audit import SCIENCE,check_readiness,validate_states,remote_code


class EndpointAuditTests(unittest.TestCase):
    def fixture(self):
        protocol={'scientific_commit':SCIENCE,'seed':51001,'updates':10000,'batch':32,'microbatch':4,
                  'learning_rate':3e-4,'network':{},'teacher_source_sha256':'t'*64,'validation_sha256':'v'*64}
        common={'format':'s1_g16_geometry_mlm_v1','update':10000,'seed':51001,'from_scratch':True,
                'protocol_sha256':hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest(),
                'dataset_sha256':'t'*64,'model':{'weight':torch.tensor([1.,2.])},
                'ledger':{'retained_updates':10000,'unknown_attempts':0,'known_failed_attempts':0,'replayed_attempts':0}}
        cp=copy.deepcopy(common);final=copy.deepcopy(common);final['network']={}
        cp.update(protocol=protocol,nodes=[2500,5000,7500,10000],
            optimizer={'param_groups':[{'params':[0],'lr':3e-4}],
                       'state':{0:{'step':torch.tensor(10000.),'exp_avg':torch.zeros(2),'exp_avg_sq':torch.ones(2)}}},
            python_rng=random.Random(51001).getstate(),numpy_rng=np.random.RandomState(51001).get_state(),
            sampler_rng=np.random.default_rng(51001).bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=[torch.zeros(16,dtype=torch.uint8)])
        loss=[{'update':i,'loss':.5,'sample_ids':list(range(32)),'attempt_id':i} for i in range(1,10001)]
        attempts=[{'update':i,'status':'committed','attempt_id':i} for i in range(1,10001)]
        wal={'update':10000,'status':'committed','attempt_id':10000}
        return cp,final,protocol,loss,attempts,wal

    def validate(self,data):
        cp,final,protocol,loss,attempts,wal=data
        return validate_states(cp,final,protocol,'t'*64,'v'*64,loss,attempts,wal,sample_count=32)

    def test_complete_schema_passes_without_changing_global_rng(self):
        data=self.fixture();before=torch.get_rng_state().clone();numpy_before=np.random.get_state();python_before=random.getstate()
        result=self.validate(data)
        self.assertEqual(result['updates'],10000);self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertEqual(python_before,random.getstate());self.assertTrue(np.array_equal(numpy_before[1],np.random.get_state()[1]))

    def test_parameter_or_dtype_drift_rejected(self):
        for changed in (torch.tensor([1.,3.]),torch.tensor([1.,2.],dtype=torch.float64),torch.tensor([1.,float('nan')])):
            data=self.fixture();data[1]['model']['weight']=changed
            with self.assertRaisesRegex(ValueError,'model tensor'):self.validate(data)

    def test_previous_checkpoint_not_endpoint(self):
        data=self.fixture();data[0]['update']=2500
        with self.assertRaisesRegex(ValueError,'endpoint'):self.validate(data)

    def test_loss_gap_wal_pending_and_sampler_error_rejected(self):
        data=self.fixture();data[3].pop(6)
        with self.assertRaisesRegex(ValueError,'loss updates'):self.validate(data)
        data=self.fixture();data[5]['status']='pending'
        with self.assertRaisesRegex(ValueError,'final WAL'):self.validate(data)
        data=self.fixture();data[3][7]['sample_ids'][0]=99
        with self.assertRaisesRegex(ValueError,'sampler record'):self.validate(data)

    def test_optimizer_nonfinite_or_wrong_step_rejected(self):
        data=self.fixture();data[0]['optimizer']['state'][0]['step']=torch.tensor(9999.)
        with self.assertRaisesRegex(ValueError,'Adam retained step'):self.validate(data)
        data=self.fixture();data[0]['optimizer']['state'][0]['exp_avg'][0]=float('inf')
        with self.assertRaisesRegex(ValueError,'Adam nonfinite'):self.validate(data)

    def test_node_and_teacher_provenance_rejected(self):
        data=self.fixture();data[0]['nodes']=[2500,5000,7500]
        with self.assertRaisesRegex(ValueError,'validation nodes'):self.validate(data)
        data=self.fixture();data[2]['teacher_source_sha256']='other'
        with self.assertRaisesRegex(ValueError,'teacher/validation'):self.validate(data)

    def test_terminal_gating_blocks_active_other_tasks_and_missing_zip(self):
        plan={'group':'S1-G16','scientific_commit':SCIENCE};state={'group':'S1-G16','all_computation_completed':True,
            'tasks':{'train_mlm':{'port':20473,'stage':'completed'},'common_test':{'stage':'running'}},'archives':{'train_mlm':{}}}
        with self.assertRaisesRegex(ValueError,'tasks incomplete'):check_readiness(plan,state)
        state['tasks']['common_test']['stage']='completed';self.assertEqual(check_readiness(plan,state)['port'],20473)
        state['archives']={}
        with self.assertRaisesRegex(ValueError,'checkpoint task ZIP'):check_readiness(plan,state)

    def test_remote_command_contains_pinned_path_not_member_inventory(self):
        import base64
        code=remote_code('/frozen/source','/postops/audit.py','/jobs/train','/postops/manifest_'+('a'*64)+'.json','a'*64,'train')
        self.assertLess(len(base64.b64encode(code.encode())),4096)
        self.assertNotIn('members',code)
        self.assertIn('/postops/manifest_',code)
        compile(code,'remote_command','exec')


if __name__=='__main__':unittest.main()
