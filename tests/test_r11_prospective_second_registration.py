import copy,json,unittest
from pathlib import Path
from unittest.mock import patch
from scripts.stage1_ai_search_r11_prospective import noop
REG=Path('D:/paper/outputs/experiments/first_stage_ai_search_20261013/round11/engineering_qualification_design_01/root_second_noop_registration.json')
class RegistrationChecks(unittest.TestCase):
 def values(self):
  reg=json.loads(REG.read_bytes());rr=dict(path=str(REG),sha256='2dfb7184c47a8467c3f8a35b24f8444cf0d38c38f7253dc5fde0fa72b542f0dc');p=dict(schema='R11ProspectiveSingleNoopPlanV1',ssh_port=20473,run_id=reg['run_id'],output_root=reg['archive_root'],maximum_attempts=1,maximum_children=1,hard_total_seconds=10,rss_bytes=134217728,storage_max_bytes=1048576,registration_ref=rr,prospective_lifecycle_acceptance_ref=dict(sha256='a26e12fcc51c5af9b042548187fe1947a74b75e03acd77e8fda2606651089d57'),payload_ref=dict(sha256='payload'),pause_plan=dict(run_id=reg['run_id'],archive_root=reg['archive_root'],source_files={'scripts/stage1_ai_search_r11_prospective/noop_payload.py':dict(sha256='payload')}));a=dict(schema='RootR11ProspectiveSingleNoopAdmissionV1',exact_plan_ref={'p':1},passed=True,engineering_noop_execution_authorized=True,source_accepted=True,runtime_accepted=True,scientific_execution_authorized=False,automatic_retry=False,registration_ref=rr);return p,a,reg
 def check(self,p,a,reg):
  with patch.object(noop.c,'read',side_effect=lambda r:p if r=={'p':1} else a if r=={'r':1} else reg if r==p['registration_ref'] else {}),patch.object(noop.c,'pin',return_value=Path(noop.__file__).with_name('noop_payload.py')),patch.object(noop.r,'source_check'),patch.object(noop.r,'human_epoch'),patch.object(noop.r,'storage_check'):return noop.check({'p':1},{'r':1})
 def test_explicit_second_registration_positive(self):
  p,a,r=self.values();self.assertEqual(self.check(p,a,r)['run_id'],r['run_id'])
 def test_old_registration_rejected(self):
  p,a,r=self.values();p['registration_ref']=dict(path='old',sha256='old')
  with self.assertRaises(ValueError):self.check(p,a,r)
 def test_wrong_run_rejected(self):
  p,a,r=self.values();p['run_id']='old01'
  with self.assertRaises(ValueError):self.check(p,a,r)
 def test_unbound_review_rejected(self):
  p,a,r=self.values();a.pop('registration_ref')
  with self.assertRaises(ValueError):self.check(p,a,r)
if __name__=='__main__':unittest.main()
