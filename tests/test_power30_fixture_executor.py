"""Pure owned-entry/cap/accounting mocks; never create an environment/model."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from satellite_rl.power30.fixture_executor import FIXTURE_LIMITS,validate_fixture_config,run_owned_fixture_task
from satellite_rl.power30.ledger import initialize


class OwnedFixtureTests(unittest.TestCase):
    def config(self,root):
        case={'role':'train','case_id':'fixed'}
        return dict(phase='engineering',task_kind='physical_fixture',fixture_limits=FIXTURE_LIMITS.copy(),
                    case_record=case,train_cases=[case],source_sha='mock',run_id='mock-owned',ledger_root=str(Path(root)/'ledger'))

    def test_exact_nonlearning_caps_and_root_feature_limit(self):
        c=self.config('mock')
        with patch('satellite_rl.power30.fixture_executor.fixture_admission',return_value={'max_feature':1000}):
            validate_fixture_config(c)
            c['fixture_limits']['update']=1
            with self.assertRaises(ValueError):validate_fixture_config(c)
            c['fixture_limits']=FIXTURE_LIMITS.copy();c['training']={}
            with self.assertRaises(ValueError):validate_fixture_config(c)

    def test_owner_rejection_precedes_permanent_ledger_or_fixture_entry(self):
        with patch('satellite_rl.power30.fixture_executor.validate_fixture_config'),patch('satellite_rl.power30.fixture_executor.RuntimeControl') as ctl,patch('satellite_rl.power30.fixture_executor.CallLedger') as ledger,patch('satellite_rl.power30.fixture_executor.run_fixtures') as fixture:
            ctl.return_value.validate_owner.side_effect=ValueError('wrong actual owner')
            with self.assertRaises(ValueError):run_owned_fixture_task(self.config('mock'),'mock')
            ledger.assert_not_called();fixture.assert_not_called()

    def test_owned_mock_call_double_debit_and_failure_keeps_open(self):
        for fail in (False,True):
            with self.subTest(fail=fail),tempfile.TemporaryDirectory() as root:
                c=self.config(root);directory=Path(root)/'run';directory.mkdir()
                initialize(c['ledger_root'],{})
                def fixture(config,path,*,call_hook,stop_requested,owner_check):
                    owner_check();call_hook('kernel');call_hook('coeff');call_hook('phy');call_hook('feature')
                    if fail:raise RuntimeError('mock unknown after entry')
                    return {'status':'passed','calls':{'phy':1}}
                with patch('satellite_rl.power30.fixture_executor.validate_fixture_config'),patch('satellite_rl.power30.fixture_executor.validate_data_admission'),patch('satellite_rl.power30.fixture_executor.RuntimeControl') as ctl,patch('satellite_rl.power30.fixture_executor.run_fixtures',side_effect=fixture):
                    ctl.return_value.stop_requested.return_value=False
                    if fail:
                        with self.assertRaises(RuntimeError):run_owned_fixture_task(c,directory)
                    else:
                        result=run_owned_fixture_task(c,directory)
                        self.assertEqual(result['stage'],'completed')
                        self.assertNotIn('env_step',result)
                import json
                value=json.loads((Path(c['ledger_root'])/'workers'/'mock-owned.json').read_text())
                self.assertEqual(value['actual']['phy'],1)
                self.assertEqual(value['actual']['fixture_phy'],1)
                self.assertNotIn('engineering_update',value['actual'])
                self.assertEqual(value['open'],fail)


if __name__=='__main__':unittest.main()
