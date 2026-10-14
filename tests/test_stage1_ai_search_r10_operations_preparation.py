"""Operations-only synthetic checks; no source model, optimizer or geometry imports."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from scripts.stage1_ai_search_r10.contracts import ARMS,MOTHERS,IDENTITY_FIELDS,AUTHORITY_SHA,publish,check_identity,validate_execution
from scripts.stage1_ai_search_r10.schedule import partition,stage_order,Reservations
from scripts.stage1_ai_search_r10.lifecycle import adjudicate
from scripts.stage1_ai_search_r10.control import boundary,Paused,FINGERPRINT
from scripts.stage1_ai_search_r10.costs import measured_scope

def identity(i=0):
 value={k:'original'+str(i) for k in IDENTITY_FIELDS};value['role']='train';return value

class OperationsPreparation(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def test_identity_no_string_coercion(self):
  value=identity();value['case_id']=2
  with self.assertRaises(ValueError):check_identity(value)
 def test_fixed_six_mother_partition_and_serial_methods(self):
  rows=[dict(original_index=i,identity=identity(i)) for i in MOTHERS];assignment={str(i):20473 if j<6 else 20474 for j,i in enumerate(MOTHERS)};local=partition(rows,assignment,20473);order=stage_order(local)
  self.assertEqual(len(local),6);self.assertEqual(len(order),36);self.assertEqual([s for i,s in order[:6]],['common',*ARMS])
 def test_wrong_host_partition_rejected(self):
  rows=[dict(original_index=i,identity=identity(i)) for i in MOTHERS]
  with self.assertRaises(ValueError):partition(rows,{str(i):20473 for i in MOTHERS},20473)
 def test_permanent_unknown_reservation_never_replays(self):
  ledger=Reservations(self.root,'run','epoch',{0:identity()});ledger.reserve(0,'common',{})
  with self.assertRaises(FileExistsError):ledger.reserve(0,'common',{})
 def test_arm_requires_original_same_common(self):
  ledger=Reservations(self.root,'run','epoch',{0:identity()})
  with self.assertRaises(ValueError):ledger.reserve(0,'Gd',{})
  ref=publish(self.root/'common.json',dict(run_id='run',original_index=0,identity=identity()))
  ledger.reserve(0,'Gd',{},ref)
 def test_completed_stage_immutable(self):
  ledger=Reservations(self.root,'run','epoch',{0:identity()});ledger.reserve(0,'common',{});value=dict(run_id='run',identity=identity(),original_index=0,stage='common');ledger.finish(0,'common',value)
  with self.assertRaises(FileExistsError):ledger.finish(0,'common',value)
 def test_unknown_third_process_blocks_signalling(self):
  owner=dict(pid=1,birth_ticks=10,boot_id='b',uid=7);third=dict(pid=3,birth_ticks=12,boot_id='b',uid=7,state='S');result=adjudicate([owner],[dict(owner,state='S'),third],'b');self.assertFalse(result['execution_ended']);self.assertEqual(result['signal_eligible'],[])
 def test_reused_pid_blocks_signalling(self):
  owner=dict(pid=1,birth_ticks=10,boot_id='b',uid=7);result=adjudicate([owner],[dict(owner,birth_ticks=11,state='S')],'b');self.assertFalse(result['execution_ended']);self.assertEqual(result['signal_eligible'],[])
 def test_zombie_explicit_not_absent(self):
  owner=dict(pid=1,birth_ticks=10,boot_id='b',uid=7);result=adjudicate([owner],[dict(owner,state='Z')],'b');self.assertTrue(result['execution_ended']);self.assertEqual(len(result['zombies']),1)
 def epoch(self):return dict(group='S1-AI17-R10',run_id='run',root_accepted=True,authority_ref=dict(path='mock',sha256=AUTHORITY_SHA),watched_pause_paths=['historical'],historical_hold_exemptions=[dict(path='historical',reason='historical0700_pause',archived_evidence_ref=dict(path='mock',sha256='x'),**{k:('x' if k=='sha256' else 1) for k in FINGERPRINT})])
 def test_exact_historical_marker_only_exempt(self):
  epoch=self.epoch();old=epoch['historical_hold_exemptions'][0]
  with patch('scripts.stage1_ai_search_r10.control.pin',return_value=None):boundary(epoch,'run',self.root,'mock',snapshot=lambda p:old if p=='historical' else None)
 def test_same_bytes_new_generation_blocks(self):
  epoch=self.epoch();old=dict(epoch['historical_hold_exemptions'][0],st_ctime_ns=2)
  with patch('scripts.stage1_ai_search_r10.control.pin',return_value=None):
   with self.assertRaises(Paused):boundary(epoch,'run',self.root,'mock',snapshot=lambda p:old if p=='historical' else None)
 def test_new_user_pause_blocks(self):
  epoch=self.epoch();epoch['watched_pause_paths'].append('new')
  with patch('scripts.stage1_ai_search_r10.control.pin',return_value=None):
   with self.assertRaises(Paused):boundary(epoch,'run',self.root,'mock',snapshot=lambda p:epoch['historical_hold_exemptions'][0] if p=='historical' else dict(value=dict(user_pause=True)) if p=='new' else None)
 def test_failure_scope_keeps_measured_elapsed(self):
  ticks=iter([1.,4.]);p=self.root/'cost.json'
  with self.assertRaises(RuntimeError):
   with measured_scope(p,'input',clock=lambda:next(ticks)):raise RuntimeError('source error')
  value=json.loads(p.read_bytes());self.assertEqual(value['observed_seconds_to_finally'],3.);self.assertIsNone(value['complete_worker_exit_cost_seconds'])
 def test_preparation_can_never_execute(self):
  p=publish(self.root/'plan.json',dict(group='S1-AI17-R10',source_input_budget_frozen=False));r=publish(self.root/'review.json',dict(passed=True,execution_authorized=True,exact_plan_ref=p))
  with self.assertRaisesRegex(ValueError,'preparation'):validate_execution(p,r)

if __name__=='__main__':unittest.main()
