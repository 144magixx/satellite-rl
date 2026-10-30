"""Pure storage/identity boundary checks: no du, remote, environment or model."""
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from satellite_rl.power30.storage_guard import StorageGuard,validate_contract
from satellite_rl.power30.control import validate_engineering_data_admission


class StorageGuardTests(unittest.TestCase):
    def config(self,root):
        marker=root/'control'/'STORAGE_STOP'
        return dict(pause_markers=[str(marker)],reserve_free_bytes=16*1024**3,
            storage_control=dict(owned_storage_root=str(root),baseline_allocated_bytes=100,
                max_growth_bytes=1000,guard_interval_seconds=60,
                storage_probe_timeout_seconds=15,storage_stop_marker=str(marker)))

    def test_growth_boundary_cadence_and_only_owned_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'s2_w30';root.mkdir();run=root/'run';run.mkdir()
            other=Path(temp)/'other_chat';other.mkdir()
            c=self.config(root);clock=[0.];guard=StorageGuard(c,run,clock=lambda:clock[0])
            with patch('satellite_rl.power30.storage_guard.subprocess.run',return_value=SimpleNamespace(stdout='1100\t'+str(root)+'\n')) as du:
                self.assertFalse(guard.check());clock[0]=59.;self.assertFalse(guard.check());self.assertEqual(du.call_count,1)
                args=du.call_args.args[0];self.assertIn('-P',args);self.assertIn('-x',args)
                clock[0]=60.;du.return_value.stdout='1101\t'+str(root)+'\n'
                self.assertTrue(guard.check())
            self.assertTrue(Path(c['storage_control']['storage_stop_marker']).exists())
            self.assertFalse(any(other.iterdir()))

    def test_unverifiable_measurement_stops_and_scope_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'s2_w30';root.mkdir();run=root/'run';run.mkdir();c=self.config(root)
            guard=StorageGuard(c,run)
            with patch('satellite_rl.power30.storage_guard.subprocess.run',side_effect=subprocess.TimeoutExpired('du',15)):
                self.assertTrue(guard.check())
            self.assertTrue(Path(c['storage_control']['storage_stop_marker']).exists())
            c['storage_control']['storage_stop_marker']=str(Path(temp)/'other_chat_STOP')
            with self.assertRaises(ValueError):validate_contract(c,run)
            self.assertFalse((Path(temp)/'other_chat_STOP').exists())

    def test_engineering_exact576_train_and20_identity_without_eval(self):
        train=[dict(role='train',case_id='train'+str(i)) for i in range(576)]
        dev=[dict(role='development',case_id='dev'+str(i)) for i in range(20)]
        config=dict(train_cases=train,dev_cases=[],development_identity_rows=dev,
            data_manifest_sha='same',data_manifest_ref={'path':'manifest','sha256':'same'},data_admission_ref={'path':'receipt'})
        receipt=dict(experiment_id='S2-W30',root_approved=True,passed=True,data_manifest_sha='same',
            uniform_request_bps=21600000,train_mothers=576,dev_mothers=20)
        with patch('satellite_rl.power30.control.read_ref',side_effect=lambda ref:receipt if ref['path']=='receipt' else {'rows':train+dev}):
            validate_engineering_data_admission(config)
            config['train_cases']=train[:-1]
            with self.assertRaises(ValueError):validate_engineering_data_admission(config)
            config['train_cases']=train;config['dev_cases']=dev
            with self.assertRaises(ValueError):validate_engineering_data_admission(config)


if __name__=='__main__':unittest.main()
