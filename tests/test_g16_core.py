import ast
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from satellite_rl.coverage.geoload.core import Scene, Layout, metrics, validate_layout, initial_layout, evaluate_case
from satellite_rl.coverage.geoload.classical import run_classical
from satellite_rl.coverage.geoload.config import SearchConfig

class G16CoreTests(unittest.TestCase):
    def setUp(self):
        self.scene = Scene([[45, 5], [45, 5.1], [45, 5.2], [45, 5.3]], [10, 9, 2, 1])
        self.initial = Layout([[45, 5.05], [45, 5.25]], [0, 0, 1, 1])

    def test_conservation_and_scaling(self):
        a = metrics(self.scene, self.initial)
        b = metrics(Scene(self.scene.positions, self.scene.demands * 3), self.initial)
        self.assertAlmostEqual(sum(a['beam_loads']), 22)
        self.assertEqual(a['P'], b['P'])
        self.assertEqual(a['Q'], b['Q'])
        self.assertAlmostEqual(a['T'], 19 / 11)

    def test_all_zero_still_covered_and_undefined(self):
        s = Scene(self.scene.positions, np.zeros(4))
        self.assertIsNone(metrics(s, self.initial)['P'])
        self.assertIsNone(metrics(s, self.initial)['CV'])
        self.assertFalse(validate_layout(s, Layout([[45, 5]], [0, 0, 0, -1]))['valid'])

    def test_edge_cannot_be_deleted(self):
        s = Scene([[45, 5], [45, 40]], [100, 0])
        initial = initial_layout(s)
        result = run_classical(s, initial, 'VARIABLE', SearchConfig(seconds=.05, allow_variable_n=True, target_n=1))
        self.assertTrue(validate_layout(s, result.layout)['valid'])
        self.assertEqual(result.metrics['N'], 2)

    def test_budget_atomic_and_input_immutable(self):
        result = run_classical(self.scene, self.initial, 'JOINT', SearchConfig(seconds=0))
        self.assertEqual(result.budgets['attempts'], 0)
        np.testing.assert_array_equal(result.layout.assignment, self.initial.assignment)
        with self.assertRaises(ValueError): self.initial.assignment[0] = 1

    def test_candidate_limit_and_improvement(self):
        result = run_classical(self.scene, self.initial, '09+SWAP', SearchConfig(seconds=1, max_attempts=20))
        self.assertLessEqual(result.budgets['attempts'], 20)
        self.assertLess(result.metrics['P'], metrics(self.scene, self.initial)['P'])
        self.assertTrue(validate_layout(self.scene, result.layout)['valid'])

    def test_whole_region_escapes_single_transfer_and_keeps_all_users(self):
        positions = [[45, 5 + i * .01] for i in range(6)]
        scene = Scene(positions, [8, 7, 6, 5, 0, 0])
        initial = Layout([[45, 5.01], [45, 5.04]], [0, 0, 1, 1, 0, 1])
        single = run_classical(scene, initial, '09-LEX', SearchConfig(seconds=1))
        joint = run_classical(scene, initial, 'JOINT', SearchConfig(seconds=1))
        self.assertEqual(single.metrics['P'], metrics(scene, initial)['P'])
        self.assertLess(joint.metrics['P'], single.metrics['P'])
        accepted = [row for row in joint.journal if row['reason'] == 'accepted' and row['kind'].startswith('whole_region')]
        self.assertTrue(accepted)
        self.assertEqual(accepted[0]['details']['adjustable_users'], 6)
        self.assertEqual(len(joint.layout.assignment), 6)
        self.assertTrue(validate_layout(scene, joint.layout)['valid'])

    def test_geometric_counterexample_and_n_cap(self):
        s = Scene([[45, 5], [45, 40]], [1, 1])
        self.assertFalse(validate_layout(s, Layout([[45, 5]], [0, 0]))['valid'])
        self.assertFalse(validate_layout(self.scene, Layout(np.tile([45, 5], (201, 1)), [0, 0, 0, 0]))['valid'])

    def test_downstream_import_boundary(self):
        root = Path(__file__).resolve().parents[1] / 'src/satellite_rl/coverage/geoload'
        for path in (root / name for name in ('core.py', 'classical.py', 'config.py')):
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse(any(s in (node.module or '') for s in ('variable_downstream', 'nondrl', 'physical', 'tdma', 'allocation')))

    def test_adapter(self):
        result = evaluate_case({'positions': self.scene.positions, 'demands': self.scene.demands,
                               'initial': self.initial.to_record()}, 'BH', {'seconds': 0})
        self.assertTrue(result['metrics']['evidence']['valid'])

    def test_reference_nodes_are_nested_and_shared_pool_permutation(self):
        cfg = SearchConfig(seconds=1, max_attempts=20, checkpoint_attempts=(1, 4, 20), checkpoint_seconds=(.01, .1))
        result = run_classical(self.scene, self.initial, 'JOINT', cfg, candidate_order=lambda s, l, pool: list(reversed(pool)))
        for node in result.to_record()['reference_nodes']:
            self.assertTrue(validate_layout(self.scene, Layout.from_record(node['layout']))['valid'])
            if node['kind'] == 'attempts': self.assertLessEqual(node['state_attempts'], node['limit'])
            if node['kind'] == 'paired':
                self.assertLessEqual(node['state_attempts'], node['candidate_limit'])
                self.assertLessEqual(node['state_seconds'], node['seconds_limit'])
        with self.assertRaises(ValueError):
            run_classical(self.scene, self.initial, 'JOINT', cfg, candidate_order=lambda s, l, pool: pool[:-1])

    def test_first_64_rejects_do_not_hide_candidate_65(self):
        good=Layout(self.initial.centers,[0,1,1,0])
        def stream(scene,current,method,config,stats=None):
            for index in range(64): yield 'same',self.initial,{'index':index}
            yield 'improve',good,{'index':64}
        config=SearchConfig(seconds=1,max_attempts=65,pool_size=64)
        with patch('satellite_rl.coverage.geoload.classical.candidates',stream):
            identity=run_classical(self.scene,self.initial,'JOINT',config)
            hook=run_classical(self.scene,self.initial,'JOINT',config,candidate_order=lambda s,l,p:p)
        self.assertEqual(identity.budgets['attempts'],65)
        self.assertEqual(hook.budgets['attempts'],65)
        self.assertEqual(identity.budgets['candidate_pools'],2)
        self.assertLess(hook.metrics['P'],metrics(self.scene,self.initial)['P'])
        np.testing.assert_array_equal(hook.layout.assignment,identity.layout.assignment)

    def test_late_validator_is_charged_but_cannot_publish(self):
        good=Layout(self.initial.centers,[0,1,1,0]);clock=[0.]
        def stream(*args):yield 'improve',good,{}
        original=validate_layout
        def slow(scene,layout,geometry):
            result=original(scene,layout,geometry);clock[0]+=.1;return result
        with patch('satellite_rl.coverage.geoload.classical.candidates',stream), \
             patch('satellite_rl.coverage.geoload.classical.perf_counter',lambda:clock[0]), \
             patch('satellite_rl.coverage.geoload.classical.validate_layout',slow):
            result=run_classical(self.scene,self.initial,'JOINT',SearchConfig(seconds=.05,max_attempts=4,
                checkpoint_seconds=(.05,),checkpoint_attempts=(1,)))
        np.testing.assert_array_equal(result.layout.assignment,self.initial.assignment)
        self.assertEqual(result.budgets['attempts'],1)
        self.assertEqual(result.budgets['late_candidates'],1)
        self.assertEqual(result.journal[0]['reason'],'late_budget_reject')
        self.assertIn('candidate_layout',result.journal[0])
        self.assertEqual(result.to_record()['reference_nodes'][0]['state_seconds'],0.)
        self.assertGreater(result.budgets['deadline_overrun_seconds'],0.)

    def test_reference_time_node_excludes_later_success(self):
        good=Layout(self.initial.centers,[0,1,1,0]);clock=[0.]
        def stream(scene,current,*args):
            if np.array_equal(current.assignment,self.initial.assignment):yield 'improve',good,{}
        original=validate_layout
        def slow(scene,layout,geometry):
            result=original(scene,layout,geometry);clock[0]+=.1;return result
        with patch('satellite_rl.coverage.geoload.classical.candidates',stream), \
             patch('satellite_rl.coverage.geoload.classical.perf_counter',lambda:clock[0]), \
             patch('satellite_rl.coverage.geoload.classical.validate_layout',slow):
            result=run_classical(self.scene,self.initial,'JOINT',SearchConfig(seconds=.2,max_attempts=4,
                checkpoint_seconds=(.05,.2),checkpoint_attempts=(1,4)))
        nodes=result.to_record()['reference_nodes']
        np.testing.assert_array_equal(nodes[0]['layout']['assignment'],self.initial.assignment)
        self.assertEqual(nodes[0]['state_seconds'],0.)
        self.assertLess(nodes[1]['metrics']['P'],nodes[0]['metrics']['P'])

if __name__ == '__main__': unittest.main()
