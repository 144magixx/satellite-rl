import unittest
from unittest.mock import patch
from pathlib import Path
import importlib.util
import pandas as pd
from test_s2fix_physics import scene
from satellite_rl.litcmp.pipeline import METHODS, evaluate_scene
from satellite_rl.litcmp.data import shard_indices


class PipelineTest(unittest.TestCase):
    def test_all_methods_and_budget_columns(self):
        stage = scene()
        stage.case['UTC'] = '2024-01-12T00:00:00Z'
        with patch('satellite_rl.litcmp.pipeline.records', return_value=[dict(set_name='dev', case_id='small')]), \
             patch('satellite_rl.litcmp.pipeline.load_input', return_value=stage):
            for method in METHODS:
                with self.subTest(method=method):
                    row = evaluate_scene(dict(set_dir='unused', index=0, method=method, seed=51001))
                    budget = (int(method.split('_')[1]) if '_' in method else 320 if method == 'C2' else 0)
                    self.assertEqual(row['probes'], budget - int(method.startswith('B2')))
                    self.assertTrue(row['constraints_ok'])
                    for key in ('W', 'SI', 'SGM', 'aircraft_worst10', 'total_power_w'):
                        self.assertIn('before_' + key, row)
                        self.assertIn('after_' + key, row)
                    for key in ('t_init', 't_search', 't_refine', 't_total'):
                        self.assertGreaterEqual(row[key], 0)

    def test_shards_and_paired_summary(self):
        a, b = (set(i for i, _ in shard_indices(range(20), f'{i}/2')) for i in range(2))
        self.assertFalse(a & b)
        self.assertEqual(a | b, set(range(20)))
        path = Path(__file__).resolve().parents[1] / 'scripts/litcmp/summarize_stage1.py'
        spec = importlib.util.spec_from_file_location('lit_summary', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        common = dict(scene_set='DEV24', date='2024-01-12', frame_time='t', N_budget=100,
            N_used=10, top10=2, CV=.1, Omega_sr=.01, width_mean=.3, width_min=.3,
            width_max=.3, spacing_violations=0, t_total=1, feasible=True)
        lit = pd.DataFrame([dict(common, arm='A2', effort=seed, peak=peak)
                            for seed, peak in ((51001, 4), (51002, 8))])
        ref = pd.DataFrame([dict(common, arm=arm, effort=1, peak=peak)
                            for arm in ('FLOW-AO', 'FLOW-AO-no-spacing', 'INIT') for peak in (2, 4)])
        result = module.summarize(lit, ref).iloc[0]
        self.assertEqual(result['peak'], 6)
        self.assertEqual(result['FLOW-AO_peak_ratio_lit_over_ref'], 2)
        self.assertEqual(result['FLOW-AO_paired_scenes'], 1)
        self.assertEqual(result['FLOW-AO-no-spacing_not_higher_fraction'], 1)


if __name__ == '__main__':
    unittest.main()
