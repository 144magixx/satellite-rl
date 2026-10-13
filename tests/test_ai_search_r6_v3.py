"""Pure mocks/reference arithmetic only; NEVER runs qualification A or B."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from satellite_rl.coverage.ai_search_r6.runtime_v3 import PROFILE,pre_cuda_environment,validate_flags,json_sha
from satellite_rl.coverage.ai_search_r6.qualification_v3 import ContinuationLedger,_analytic_reference,_array_comparison,compare_tree


class QualificationV3Mocks(unittest.TestCase):
    def test_runtime_environment_and_drift_fail_closed(self):
        with patch.dict(os.environ,{},clear=True),patch('torch.cuda.is_initialized',return_value=False):
            pre_cuda_environment();self.assertEqual(os.environ['CUBLAS_WORKSPACE_CONFIG'],':4096:8')
        with patch.dict(os.environ,{'CUBLAS_WORKSPACE_CONFIG':':16:8'},clear=True):
            with self.assertRaises(RuntimeError):pre_cuda_environment()
        with patch.dict(os.environ,{},clear=True),patch('torch.cuda.is_initialized',return_value=True):
            with self.assertRaises(RuntimeError):pre_cuda_environment()
        flags={key:PROFILE[key] for key in ('float32_matmul_precision','cuda_matmul_allow_tf32','cudnn_allow_tf32','cudnn_benchmark','cudnn_deterministic','deterministic_algorithms','warn_only','CUBLAS_WORKSPACE_CONFIG')}
        flags.update(autocast_cuda=False,autocast_cpu=False,default_dtype='torch.float32');validate_flags(flags)
        for key in ('cuda_matmul_allow_tf32','warn_only','autocast_cuda'):
            changed=copy.deepcopy(flags);changed[key]=True
            with self.assertRaises(RuntimeError):validate_flags(changed)

    def test_continuation_exclusive_and_attempts_preserve_original(self):
        original=dict(updates=[dict(ordinal=1,status='completed'),dict(ordinal=2,status='completed')]);before=copy.deepcopy(original)
        with tempfile.TemporaryDirectory() as tmp:
            budget=dict(continuation_claim_path=str(Path(tmp)/'claim.json'),source_commit='new',source_manifest_sha='source',
                original_claim_sha='old_sha',initial_file_sha='init',fixture_sha='fixture',runtime_profile_sha=json_sha(PROFILE),
                output_path=str(Path(tmp)/'out'),reservation_id='root_once')
            ledger=ContinuationLedger(budget,original)
            with self.assertRaises(FileExistsError):ContinuationLedger(budget,original)
            first=ledger.reserve('baseline');self.assertEqual(first['global_ordinal'],3)
            self.assertEqual(json.loads(ledger.path.read_text())['new_updates'][0]['status'],'started_unknown_consumed')
            ledger.finish(first,error=RuntimeError('pure mock unknown'));second=ledger.reserve('resumed');ledger.finish(second)
            self.assertEqual(second['global_ordinal'],4)
            with self.assertRaises(RuntimeError):ledger.reserve('forbidden')
            self.assertEqual(original,before);self.assertEqual(ledger.value['historical_updates'],before['updates'])

    def test_reference_is_independent_closed_uniform_example(self):
        # Pure analytic two-aircraft arithmetic, no model/torch backward/SciPy.
        arrays=dict(enable=np.zeros((1,200),np.float32),owner=np.zeros((1,3,200),np.float32),
            pointer=np.zeros((1,200,3),np.float32))
        target=SimpleNamespace(aircraft_ids=('a','b'),anchor_ids=('a','b'),owners=np.array([0,1]))
        losses,grads=_analytic_reference(arrays,[target])
        self.assertEqual(losses['A'],2);self.assertEqual(losses['G'],2)
        self.assertAlmostEqual(losses['E'],(2+198*.1)*np.log(2))
        self.assertAlmostEqual(losses['O']/2,np.log(200));self.assertAlmostEqual(losses['P']/2,np.log(2))
        self.assertEqual(grads['enable'][0,0],-.5/3200)
        self.assertEqual(grads['owner'][0,0,0],(1/200-1)/2)
        self.assertTrue((grads['owner'][0,2:]==0).all());self.assertTrue((grads['pointer'][0,2:]==0).all())

    def test_all_named_comparisons_do_not_abort_first_failure(self):
        first=dict(a=torch.tensor([1.]),b=torch.tensor([2.]),rng=dict(cpu=torch.tensor([0],dtype=torch.uint8)))
        second=dict(a=torch.tensor([3.]),b=torch.tensor([4.]),rng=dict(cpu=torch.tensor([1],dtype=torch.uint8)))
        result=compare_tree(first,second)
        self.assertEqual(set(result),{'root.a','root.b','root.rng.cpu'});self.assertTrue(all(not r['passed'] for r in result.values()))
        self.assertFalse(_array_comparison(np.array([np.nan]),np.array([np.nan]))['passed'])
        zero=_array_comparison(np.array([1.]),np.array([0.]))
        self.assertTrue(zero['zero_reference_norm']);self.assertIsNone(zero['relative_L2'])
        self.assertFalse(compare_tree(dict(step=torch.tensor(2.)),dict(step=torch.tensor(2.)))['root.step']['passed'])


if __name__=='__main__':unittest.main()
