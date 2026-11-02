"""Pure authority/marker-generation control checks; no science or live signals."""
import tempfile,pathlib,json,hashlib,unittest,os
from unittest.mock import patch
from scripts.stage1_ai_search_r8 import deadline,contracts

class HumanResume(unittest.TestCase):
 def fixture(self,root):
  epoch=root/'epoch1';epoch.mkdir();marker=root/'old_hold.json';marker.write_text('{"user_pause":true}');backup=root/'archived_hold.json';backup.write_bytes(marker.read_bytes());snapshot=deadline.marker_snapshot(str(marker));snapshot.pop('value');snapshot.update(reason='historical0700_pause',archived_evidence_ref=dict(path=str(backup),sha256=contracts.digest(backup)))
  e=dict(epoch_root=str(epoch),new_deadline_unix_seconds=None,historical_hold_exemptions=[snapshot],watched_pause_paths=[str(marker)])
  registration=dict(root_accepted=True,archive_root=str(root),group=contracts.GROUP,deadline_unix_seconds=contracts.DEADLINE,shared_pause_paths=[str(marker)])
  plan=dict(archive_root=str(root),deadline_registration_ref={},human_resume_authority_ref={'sha256':'only_mock'})
  return plan,e,registration,marker
 def test_exact_old_marker_and_past_cutoff_only_are_exempt(self):
  with tempfile.TemporaryDirectory() as tmp:
   plan,e,r,marker=self.fixture(pathlib.Path(tmp))
   with patch.object(deadline,'resume_epoch',return_value=e),patch.object(deadline,'read',return_value=r),patch.object(deadline,'identity',return_value=None),patch.object(deadline.time,'time',return_value=contracts.DEADLINE+100),patch.object(deadline,'storage_boundary',return_value={}):
    self.assertTrue(deadline.boundary(plan,'only_signed_epoch')['passed']);self.assertTrue(deadline.before_start(plan,'no_invented_deadline')['passed'])
   self.assertEqual(marker.read_text(),'{"user_pause":true}')
 def test_same_content_replacement_blocks_and_preserves_old_marker(self):
  with tempfile.TemporaryDirectory() as tmp:
   plan,e,r,marker=self.fixture(pathlib.Path(tmp));replacement=marker.with_name('replacement');replacement.write_bytes(marker.read_bytes());os.replace(replacement,marker)
   with patch.object(deadline,'resume_epoch',return_value=e),patch.object(deadline,'read',return_value=r),patch.object(deadline,'identity',return_value=None):
    with self.assertRaises(deadline.Paused):deadline.boundary(plan,'new_generation')
   self.assertTrue((pathlib.Path(e['epoch_root'])/'pause_state.json').exists());self.assertFalse((pathlib.Path(plan['archive_root'])/'pause_state.json').exists())
 def test_new_epoch_stop_is_never_exempt(self):
  with tempfile.TemporaryDirectory() as tmp:
   plan,e,r,marker=self.fixture(pathlib.Path(tmp));(pathlib.Path(e['epoch_root'])/'stop.request').write_bytes(b'')
   with patch.object(deadline,'resume_epoch',return_value=e),patch.object(deadline,'read',return_value=r),patch.object(deadline,'identity',return_value=None):
    with self.assertRaises(deadline.Paused):deadline.boundary(plan,'new_stop')
 def test_partial_authority_never_opens_deadline(self):
  with self.assertRaises(ValueError):contracts.resume_epoch(dict(human_resume_authority_ref={'sha256':'mock'}))
if __name__=='__main__':unittest.main()
