"""Evaluation wiring checks with no scene solve, policy rollout, or SCA."""
import tempfile
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from satellite_rl.s2fix.experiments import evaluate_scene


class EvaluationWiringTests(unittest.TestCase):
    def test_source_cases_loads_content_addressed_objects(self):
        from satellite_rl.s2fix.data import source_cases
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'objects').mkdir()
            rows = []
            for case_id in ('first', 'second'):
                payload = json.dumps({'case_id': case_id}).encode('utf-8')
                digest = hashlib.sha256(payload).hexdigest()
                (root / 'objects' / digest).write_bytes(payload)
                rows.append({'case_ref': {'path': 'case.json', 'sha256': digest}})
            (root / 'development_published.json').write_text(
                json.dumps({'rows': rows}), encoding='utf-8')
            # A basename-based lookup must not silently load a shared wrong case.
            (root / 'objects' / 'case.json').write_text(
                json.dumps({'case_id': 'wrong'}), encoding='utf-8')
            cases = source_cases('development', inputs_dir=root)
        self.assertEqual([case['case_id'] for case in cases], ['first', 'second'])

    def test_manifest_set_name_and_list_power_result(self):
        stage=SimpleNamespace(case={'UTC':'2024-02-24T00:00:00Z'},layout={
            'beam_ids':['a','b'],'request_bps_by_beam_int64':np.array([100,200]),
            'aircraft_count_by_beam':np.array([1,2])})
        allocation=(np.array([0,1]),np.array([0,0]),np.array([1,1]),np.array([40.,50.]))
        summary=dict(W=.5,SI=.5,SGM=.4,capacity_bps=np.array([50.,100.]))
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'beams').mkdir()
            with patch('satellite_rl.s2fix.experiments.load_input',return_value=stage), \
                 patch('satellite_rl.s2fix.experiments.greedy_allocation',return_value=allocation), \
                 patch('satellite_rl.s2fix.experiments.metrics',return_value=summary), \
                 patch('satellite_rl.s2fix.experiments.run_sca',return_value={'power_w':[40.,50.],'sca_seconds':.01}):
                rows=evaluate_scene((0,{'set_name':'dev','case_id':'case'},['G1'],{},directory,'cpu',1))
            self.assertEqual(rows[0]['set'],'dev')
            self.assertTrue(rows[0]['constraints_ok'])
            with np.load(rows[0]['beams']) as saved:
                np.testing.assert_array_equal(saved['power_after'],[40.,50.])

    def test_cached_i0_loader_and_plain_action_schema(self):
        from satellite_rl.s2fix.i0 import load_policy
        from satellite_rl.power30.schema import schema_identity
        load_policy.cache_clear()
        payload={'config':{'device':'cpu'},'agent':{}}
        with patch('satellite_rl.s2fix.i0.torch.load',return_value=payload) as read, \
             patch('satellite_rl.s2fix.i0.ImprovementSAC') as construct:
            first=load_policy('model.pt','cpu')
            self.assertIs(first,load_policy('model.pt','cpu'))
            read.assert_called_once()
            construct.assert_called_once()
        load_policy.cache_clear()
        with patch('satellite_rl.power30.menus.hashlib.sha256',side_effect=AssertionError('SHA called')):
            schema=schema_identity('P01','M1',plain=True)
        self.assertNotIn('sha256',schema)
        self.assertNotIn('sha256',schema['action'])


if __name__=='__main__':unittest.main()
