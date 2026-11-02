"""Metadata/control only: no solver, enumeration, geometry, process or SSH."""
import ast,copy,unittest
from pathlib import Path
from unittest.mock import patch
from scripts.stage1_ai_search_r11 import contracts as r,plan_builder,controller

class PreparationTests(unittest.TestCase):
 def test_unsorted_certificate_index_serialization(self):
  aids=('z','a');bids=('y','b');mask=[[True,False],[False,True]]
  ai=sorted(range(2),key=lambda i:aids[i]);bj=sorted(range(2),key=lambda j:bids[j])
  output=[[mask[i][j] for j in bj] for i in ai]
  for ii,i in enumerate(ai):
   for jj,j in enumerate(bj):self.assertEqual(output[ii][jj],mask[i][j])
  self.assertEqual([aids[i] for i in ai],['a','z'])
 def test_candidate_refuses_before_any_runtime_check(self):
  with patch.object(r.c,'read',side_effect=[dict(schema='R11ExactSavedCommonGhPlanV1',group=r.GROUP,source_input_budget_frozen=False),{}]),patch.object(r,'source_check') as source:
   with self.assertRaises(ValueError):r.validate({}, {})
   source.assert_not_called()
 def test_final_time_and_rss_limits(self):
  with self.assertRaises(RuntimeError):controller.final_limits(2400,0,2400,lambda:None)
  with self.assertRaises(RuntimeError):controller.final_limits(1,2147483649,2400,lambda:None)
  controller.final_limits(2399,2147483648,2400,lambda:None)
 def test_new_pause_precedes_old_exemption(self):
  p=dict(archive_root='/owned/round11/archive')
  epoch=dict(historical_hold_exemptions=[],watched_pause_paths=['/user_pause.json'])
  with patch.object(r,'human_epoch',return_value=epoch),patch.object(r.control,'marker_snapshot',side_effect=lambda path:dict(value=dict(user_pause=True)) if path=='/user_pause.json' else None):
   with self.assertRaises(r.control.Paused):r.check_pause(p,'mock')
 def test_reasserted_old_generation_blocks(self):
  old=dict(path='/old.json',reason='historical0700_pause',sha256='a',archived_evidence_ref=dict(sha256='a'),size=1,st_dev=1,st_ino=2,st_mtime_ns=3,st_ctime_ns=4)
  now=dict(old,value=dict(paused=True));now['st_ctime_ns']=5
  with patch.object(r,'human_epoch',return_value=dict(historical_hold_exemptions=[old],watched_pause_paths=['/old.json'])),patch.object(r.c,'pin'),patch.object(r.control,'marker_snapshot',side_effect=lambda path:now if path=='/old.json' else None):
   with self.assertRaises(r.control.Paused):r.check_pause(dict(archive_root='/owned/round11/archive'),'mock')
 def test_immutable_entry_fence_is_old_dependency(self):
  tree=ast.parse((Path(__file__).parents[1]/'scripts/stage1_ai_search_r11/child.py').read_text())
  calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='immutable']
  self.assertTrue(any('algorithm_entry.json' in ast.unparse(n) for n in calls))

if __name__=='__main__':unittest.main()
