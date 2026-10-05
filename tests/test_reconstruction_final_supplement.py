"""Failure checks for the independent diagnostic-best audit."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / 'scripts/stage1_reconstruction/audit_final_supplement.py'
if not PATH.is_file():
    PATH = Path(__file__).with_name('audit_final_supplement.py')
spec = importlib.util.spec_from_file_location('supplement', PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SupplementTests(unittest.TestCase):
    def test_rejects_best_quality_mismatch(self):
        row = {'best_record_diagnostic': {'centers': [], 'assignment': [], 'quality': 1},
               'best_quality_diagnostic': 1, 'metrics': {'quality': 0}}
        with patch.object(module, 'metrics', return_value=({'quality': 0},None,None,None)):
            with self.assertRaisesRegex(ValueError, 'best physical quality'):
                module.audit_best(None, {'state': {'edge_beam_ids': []}}, row, {})

    def test_rejects_best_below_terminal(self):
        row = {'best_record_diagnostic': {'centers': [], 'assignment': [], 'quality': 1},
               'best_quality_diagnostic': 1, 'metrics': {'quality': 2}}
        with patch.object(module, 'metrics', return_value=({'quality': 1},None,None,None)):
            with self.assertRaisesRegex(ValueError, 'best below terminal'):
                module.audit_best(None, {'state': {'edge_beam_ids': []}}, row, {})


if __name__ == '__main__':
    unittest.main()
