import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from satellite_rl.workbench.coverage_geoload import default_config, validate_config, pin
from satellite_rl.workbench.geoload_runner import load_cases, _run_mother, exclusive_lock, result_key
from satellite_rl.workbench.storage import digest

class GeometryRuntimeTests(unittest.TestCase):
    def test_frozen_budget_rejects_extra_fields(self):
        c=default_config();c['search']['physical_calls']=12
        with self.assertRaises(ValueError):validate_config(c)
    def test_seed_immutable(self):
        c=default_config();c['training']['seed']=51002
        with self.assertRaises(ValueError):validate_config(c)
    def test_unknown_metric_config_rejected(self):
        c=default_config();c['coverage']['capacity']=600
        with self.assertRaises(ValueError):validate_config(c)
    def test_sha_pin(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a';p.write_text('first');r={'path':str(p),'sha256':digest(p)}
            self.assertEqual(pin(r),p.resolve());p.write_text('second')
            with self.assertRaises(ValueError):pin(r)
    def test_packed_dataset_groups_mothers(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'inputs.npz'
            np.savez(p,positions=np.array([[45.,5.],[45.1,5.],[45.,5.],[45.1,5.]]),
                demands=np.array([1.,2.,2.,1.]),offsets=np.array([0,2,4]),
                case_ids=np.array(['a_base','a_hot']),mother_ids=np.array(['a','a']),demand_profiles=np.array(['base','one_hotspot']))
            groups=load_cases({'path':str(p),'sha256':digest(p),'count':2})
            self.assertEqual(len(groups),1);self.assertEqual(len(groups[0]),2)
            rows=_run_mother(groups[0],['BH'],{'seconds':0.,'max_attempts':0},'evaluate')
            self.assertTrue(all(x['metrics']['evidence']['valid'] for x in rows))
            self.assertEqual(rows[0]['initial_sha256'],rows[1]['initial_sha256'])
    def test_kernel_exclusive_lock(self):
        with tempfile.TemporaryDirectory() as d:
            with exclusive_lock(Path(d)/'lock'):
                with self.assertRaises(OSError):
                    with exclusive_lock(Path(d)/'lock'):pass
    def test_keys_do_not_alias(self):
        self.assertNotEqual(result_key('a','BC'),result_key('aB','C'))

if __name__=='__main__':unittest.main()
