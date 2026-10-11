"""Short timeout/reference recovery proofs; no formal teacher jobs."""
import copy
import gzip
import json
import pickle
from pathlib import Path
import tempfile
import unittest
import numpy as np
from unittest.mock import patch

from satellite_rl.coverage.nondrl.core import BudgetStop,Problem,layout_sha256
from satellite_rl.coverage.nondrl.runner import FrozenInitialPhysics
from satellite_rl.coverage.nondrl.teachers import generate_teacher,trial_label,cv_value
from test_nondrl_learning import fixture


class TeacherBudgetTests(unittest.TestCase):
    def test_frozen_I0_then_normal_candidate_has_identical_geometry_and_physics_charges(self):
        problem = fixture(); scene = copy.deepcopy(problem.scene); scene.demands *= 100000000.
        regular = Problem(scene, problem.config, initial=problem.initial)
        cached = Problem(scene, problem.config, initial=problem.initial,
                         physical_evaluator=FrozenInitialPhysics(scene, regular.initial,
                                                                regular.initial_physical))
        cached._physical_evaluator = None
        proposed = type(problem.initial).from_record(problem.initial.to_record())
        proposed.assignment = np.array([0, 0, 1, 1])
        deltas = []
        outcomes = []
        for current in (regular, cached):
            before = copy.deepcopy(current.calls)
            outcomes.append(current.consider(proposed, 'real-geometry-budget-proof'))
            deltas.append({k: current.calls[k] - before[k] for k in before})
        self.assertEqual(outcomes[0]['metrics_all'], outcomes[1]['metrics_all'])
        def numerical(value):
            if isinstance(value, dict):
                return {k: numerical(v) for k, v in value.items() if k not in ('timing', 'timings') and not k.endswith('_seconds')}
            if isinstance(value, list):return [numerical(v) for v in value]
            return value
        self.assertEqual(numerical(outcomes[0]['physical']), numerical(outcomes[1]['physical']))
        self.assertEqual(deltas[0], deltas[1])
        self.assertGreater(deltas[1]['geometry_calls'], 1)
        self.assertEqual(deltas[1]['physical_calls'], 1)
        self.assertGreater(deltas[1]['aircraft_link_checks'], 0)

    def receipts(self,result):
        base=Path(result['path']).parent
        return [json.loads(gzip.decompress((base/item['path']).read_bytes())) for item in result['problem_receipts']]

    def test_MLM_pool_timeout_keeps_complete_reference_not_failure(self):
        with tempfile.TemporaryDirectory() as f:
            problem=fixture();initial=layout_sha256(problem.initial)
            with patch('satellite_rl.coverage.nondrl.classical.region_pool',side_effect=BudgetStop('online_time_limit')):
                result=generate_teacher([problem],'01',f,{'workers':1,'max_seconds':1.,'max_trials':12})
            self.assertEqual(result['problems'],1);self.assertEqual(result['trials'],0)
            self.assertEqual(len(result['problem_receipts']),1)
            receipt=self.receipts(result)[0]
            self.assertEqual(receipt['initial_sha256'],initial)
            self.assertEqual(layout_sha256(type(problem.initial).from_record(receipt['final_layout'])),initial)
            self.assertIn('unknown',receipt['completion_reason'])
            self.assertFalse(trial_label(problem,{'reason':'budget_unknown'},cv_value(problem))['known'])

    def test_menu_generation_timeout_keeps02and03_references(self):
        for route in ('02','03'):
            with self.subTest(route=route),tempfile.TemporaryDirectory() as f:
                problem=fixture()
                with patch.object(problem,'menu',side_effect=BudgetStop('geometry_call_limit')):
                    result=generate_teacher([problem],route,f,{'workers':1,'max_seconds':2.})
                self.assertEqual(result['problems'],1);self.assertEqual(len(result['problem_receipts']),1)
                self.assertEqual(result['trials'],0)

    def test_serial_global_trial_cap_finalizes_remaining_I0_without_more_trials(self):
        with tempfile.TemporaryDirectory() as f:
            problems=[fixture() for _ in range(4)]
            result=generate_teacher(problems,'02',f,{'workers':1,'max_trials':1,'max_seconds':10.,'max_steps':1})
            self.assertEqual(result['trials'],1);self.assertEqual(result['problems'],4)
            self.assertEqual(len(result['problem_receipts']),4)
            self.assertEqual(result['reference_only_problems'],3)
            self.assertTrue(all('reference_only' in r['completion_reason'] for r in self.receipts(result)[1:]))

    def test_parallel_exhausted_reservation_keeps_all_approved_references(self):
        with tempfile.TemporaryDirectory() as f:
            result=generate_teacher([fixture() for _ in range(4)],'02',f,
                {'workers':2,'max_trials':1,'max_seconds':10.,'max_steps':1})
            self.assertEqual(result['trials'],1);self.assertEqual(result['problems'],4)
            self.assertEqual(len(result['problem_receipts']),4)
            self.assertFalse(result['reservations'])
            self.assertGreaterEqual(result['reference_only_problems'],3)

    def test_timeout_after_committed_MLM_samples_keeps_rows_rng_and_resume(self):
        with tempfile.TemporaryDirectory() as f:
            problem=fixture();problem.config=type(problem.config)(online_seconds=20.,max_rounds=2)
            # Two identical region instances let the first commit real samples,
            # then exercise the exact failing preparation path from source105.
            region=problem.regions()[0]
            import satellite_rl.coverage.nondrl.classical as classical
            original=classical.region_pool;count=[0]
            def pool(p,r):
                count[0]+=1
                if count[0]>1:raise BudgetStop('online_time_limit')
                return original(p,r)
            with patch.object(problem,'regions',return_value=[region,region]),patch.object(classical,'region_pool',side_effect=pool):
                first=generate_teacher([problem],'01',f,{'workers':1,'max_seconds':20.,'max_trials':12})
            self.assertEqual(first['rows'],4);self.assertEqual(first['trials'],1)
            shards=copy.deepcopy(first['shards']);rng=copy.deepcopy(first['rng'])
            again=generate_teacher([fixture()],'01',f,{'workers':1,'max_seconds':20.,'max_trials':12},resume=True)
            self.assertEqual(again['shards'],shards);self.assertEqual(again['rng'],rng)
            self.assertEqual(again['rows'],first['rows']);self.assertEqual(again['trials'],first['trials'])

    def test_initial_PHY_cache_is_scene_bound_picklable_and_never_new_call(self):
        problem=fixture();cache=FrozenInitialPhysics(problem.scene,problem.initial,problem.initial_physical)
        restored=pickle.loads(pickle.dumps(cache))
        with patch('satellite_rl.coverage.variable_downstream.evaluate_layout',side_effect=AssertionError('new PHY prohibited for shared I0')):
            self.assertEqual(restored(problem.scene,problem.initial),problem.initial_physical)
        changed=copy.deepcopy(problem.scene);changed.demands*=1.25
        with self.assertRaisesRegex(ValueError,'scene identity'):restored(changed,problem.initial)


if __name__=='__main__':unittest.main()
