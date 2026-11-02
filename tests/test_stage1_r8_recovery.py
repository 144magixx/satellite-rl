"""Pure corrective routing/accounting and compressed-reader checks; no algorithm."""
import tempfile,pathlib,json,gzip,hashlib,unittest,copy
from unittest.mock import patch
from scripts.stage1_ai_search_r8.child import read_qualified_envelope
from scripts.stage1_ai_search_r8.recovery import validate_recovery

class Recovery(unittest.TestCase):
 def test_minimal_gzip_ref_and_original_content_hashes(self):
  canon=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
  value=dict(inner={'original':True});value['inner_JSON_SHA']=canon(value['inner']);value['outer_JSON_SHA']=canon(value)
  with tempfile.TemporaryDirectory() as tmp:
   p=pathlib.Path(tmp)/'original.json.gz';p.write_bytes(gzip.compress(json.dumps(value).encode()));ref=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest());self.assertEqual(read_qualified_envelope(ref),value)
   wrong={**ref,'sha256':'0'*64}
   with self.assertRaises(ValueError):read_qualified_envelope(wrong)
 def fixture(self,tmp):
  root=pathlib.Path(tmp)/'round8/archive';epoch=root/'operations_recovery/attempt1'
  p=dict(run_id='r8_pilot_fixed',request_id='original',archive_root=str(root),scientific_commit='original',design_ref={},cases=[],case_indices=list(range(20)),arms=['G','A','R'],checkpoint_ref={},original_source_manifest_ref={},scientific_new_calls=dict(local=60,D5=60))
  p['source_files']={'src/satellite_rl/coverage/ai_search_r8/'+name:dict(sha256=name) for name in ('__init__.py','core.py','geometry.py','ledger.py')}
  recovery=dict(contract_ref={'path':'contract'},root_review_ref={'path':'review'},epoch_root=str(epoch),remaining_actual_calls_max=dict(local=59,D5=59),original_slot_index_ref={'path':'index'},original_failed_status_ref={'path':'status'});p['operations_recovery']=recovery
  c=dict(schema='R8SameIDBeforeLocalReaderRecoveryContractV1',run_id=p['run_id'],archive_root=p['archive_root'],epoch_root=str(epoch),original_plan_ref={'path':'oldplan'},actual_original_calls=dict(local=1,D5=1),failed_A_actual_local_calls=0,remaining_actual_calls_max=dict(local=59,D5=59),original_slot_index_ref={'path':'index'},original_failed_status_ref={'path':'status'},protected_original_refs=[],original_processes=[])
  r=dict(contract_ref=recovery['contract_ref'],execution_authorized=True,passed=True);data={'contract':c,'review':r,'oldplan':copy.deepcopy(p),'index':{'0:G':{'path':'G'}}}
  return p,c,data
 def test_sameID_budget59_and_only_originalG_reuse(self):
  with tempfile.TemporaryDirectory() as tmp:
   p,c,data=self.fixture(tmp)
   with patch('scripts.stage1_ai_search_r8.recovery.read',side_effect=lambda ref:data[ref['path']]),patch('scripts.stage1_ai_search_r8.recovery.pin'):
    self.assertEqual(validate_recovery(p),c);c['remaining_actual_calls_max']['local']=60
    with self.assertRaises(ValueError):validate_recovery(p)
 def test_enteredA_or_different_cases_block_recovery(self):
  with tempfile.TemporaryDirectory() as tmp:
   p,c,data=self.fixture(tmp);path=pathlib.Path(p['archive_root'])/'slots/0/A/local_state/state.json';path.parent.mkdir(parents=True);path.write_text('{}')
   with patch('scripts.stage1_ai_search_r8.recovery.read',side_effect=lambda ref:data[ref['path']]),patch('scripts.stage1_ai_search_r8.recovery.pin'):
    with self.assertRaises(ValueError):validate_recovery(p)
if __name__=='__main__':unittest.main()
