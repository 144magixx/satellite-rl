import ast
from pathlib import Path
import unittest
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
        with self.assertRaises(ValueError):
            run_classical(self.scene, self.initial, 'JOINT', cfg, candidate_order=lambda s, l, pool: pool[:-1])

if __name__ == '__main__': unittest.main()
