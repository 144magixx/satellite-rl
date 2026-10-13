"""Caller typed-status adapter; no solver, CUDA, scene or real-data computation."""
import copy,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
from satellite_rl.workbench.ai_search_r3_runner import oracle_label_argument
from satellite_rl.coverage.ai_search_r3.evaluate import evaluate_method
class CallerLabelTests(unittest.TestCase):
 def test_unavailable_becomes_none(self):
  descriptor={'status':'label_unavailable','reason':'no_native_reference_or_certified_start'};before=copy.deepcopy(descriptor);self.assertIsNone(oracle_label_argument(descriptor));self.assertEqual(descriptor,before)
 def test_available_passed_same_object(self):
  descriptor={'status':'label_available','active_slot_ids':[2,8]};self.assertIs(oracle_label_argument(descriptor),descriptor)
 def test_no_label_native_zero(self):
  p=SimpleNamespace(case_id='synthetic_interface_only',scene=SimpleNamespace(input_sha256='dummy'),content_sha='dummy',witness=None,ledger={'seconds':0.})
  descriptor={'status':'label_unavailable','reason':'bounded_reference_missing'}
  def consumer(prepared,method,model,label):self.assertIsNone(label);return None,{'reason':'oracle_label_unavailable'}
  native=Mock(side_effect=AssertionError('native solver forbidden in adapter fixture'))
  with tempfile.TemporaryDirectory()as root,patch('satellite_rl.coverage.ai_search_r3.evaluate.select_slots',side_effect=consumer),patch('satellite_rl.coverage.ai_search_r3.evaluate.run_native',native):
   row=evaluate_method(p,'ORACLE_RESTRICT',Path(root),oracle_label=oracle_label_argument(descriptor));self.assertEqual(row['status'],'label_unavailable');self.assertEqual(row['solver_calls'],[]);native.assert_not_called()
 def test_other_typed_status_not_truth(self):self.assertIsNone(oracle_label_argument({'status':'unknown','reason':'no_accepted_label'}))
if __name__=='__main__':unittest.main()