"""Actual accepted legacy helper against fake /proc; no SSH, signals or child."""
import importlib.util,types,unittest
from pathlib import Path
from unittest.mock import patch

LEGACY=Path('D:/paper/outputs/experiments/first_stage_ai_search_20261013/round10/runtime_preparation_01/engineering_persistent_launcher_once.py')
MEMBER=dict(pid=888,birth_ticks=20,boot_id='boot',uid=500902,state='R')
class FakePath:
 def __init__(self,path):self.value=str(path)
 @property
 def name(self):return self.value.rsplit('/',1)[-1]
 def __truediv__(self,name):return FakePath(self.value+'/'+str(name))
 def iterdir(self):return [FakePath('/proc/888')]
 def stat(self):return types.SimpleNamespace(st_uid=500902)
 def read_text(self):
  if self.value=='/proc/sys/kernel/random/boot_id':return 'boot'
  raise AssertionError('unexpected fake read '+self.value)
 def read_bytes(self):
  if self.value=='/proc/888/cmdline':return b'unique_payload.py\0--unique-evidence\0/owned/native/noop_evidence.json\0'
  raise AssertionError('unexpected fake read '+self.value)

def load():
 stub=types.SimpleNamespace(**{k:None for k in ('DQN','read','sha','validate','bootstrap','boundary')})
 with patch.dict('sys.modules',{'engineering_driver':stub}):
  spec=importlib.util.spec_from_file_location('unchanged_accepted_legacy_registry_helper',LEGACY);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 return m
def observe(registry,member=MEMBER):
 m=load();fakec=types.SimpleNamespace(identity=lambda pid:member,same_process=lambda a,b:bool(a and b and all(a[k]==b[k] for k in ('pid','birth_ticks','boot_id','uid'))))
 with patch.object(m,'Path',FakePath),patch.object(m.os,'getuid',return_value=500902,create=True):return m.final_tree_observation(fakec,dict(helper_refs={'engineering_child.py':dict(path='unique_payload.py')},output_root='/owned/native'),registry)

from scripts.stage1_ai_search_r11_prospective import reaper

class RegistryInterfaceChecks(unittest.TestCase):
 def test_actual_production_key_with_actual_legacy_helper(self):
  result=observe({reaper.key(MEMBER):MEMBER});self.assertEqual(result['unknown_live'],[]);self.assertEqual(result['registered_live'],[MEMBER])
 def test_actual_key_reused_birth_still_unknown(self):
  current=dict(MEMBER,birth_ticks=21);result=observe({reaper.key(MEMBER):MEMBER},current);self.assertEqual(result['unknown_live'],[current]);self.assertEqual(result['registered_live'],[])

 def test_existing_4key_falsely_labels_same_registered_child_unknown(self):
  result=observe({(888,20,'boot',500902):MEMBER});self.assertEqual(result['registered_live'],[MEMBER]);self.assertEqual(result['unknown_live'],[MEMBER])
 def test_legacy_3key_preserves_full_UID_identity_without_false_unknown(self):
  result=observe({(888,20,'boot'):MEMBER});self.assertEqual(result['registered_live'],[MEMBER]);self.assertEqual(result['unknown_live'],[]);self.assertEqual(result['registered_observations'][0]['registered']['uid'],500902)
 def test_unregistered_reused_generation_stays_unknown(self):
  current=dict(MEMBER,birth_ticks=21);result=observe({(888,20,'boot'):MEMBER},current);self.assertEqual(result['registered_live'],[]);self.assertEqual(result['unknown_live'],[current])
if __name__=='__main__':unittest.main()
