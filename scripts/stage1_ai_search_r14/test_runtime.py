"""Pure IO/control checks only; no fixture geometry, solver, data or fit."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from .runtime import (BudgetLedger, ObjectStore, PauseGuard, development_schedule,
                      require_execution, produce_B)
from .identity_adapter import task_binding
from .pipeline import stage_plan, fit_gate
from .input_adapter import InputLedger, read_source


class RuntimeTests(unittest.TestCase):
    def test_false_blocks_before_backend(self):
        class Bomb:
            def prepare_case(self, *a, **k):
                raise AssertionError('backend must not be entered')
        with self.assertRaises(ValueError):
            produce_B(None, None, config={}, source_sha256='x', producer_sha256='y',
                      phase_manifest={}, checkpoint=lambda *a: None, backend=Bomb())
        with self.assertRaises(ValueError):
            read_source({}, {}, config={}, ledger=None, checkpoint=None)

    def test_input_budget_one_time_and_no_expansion(self):
        with tempfile.TemporaryDirectory() as d:
            ledger = InputLedger(d)
            ledger.reserve(0, 100, 10)
            with self.assertRaises(ValueError):
                ledger.reserve(0, 100, 10)
            with self.assertRaises(ValueError):
                ledger.reserve(3, 100, 2001)
            ledger.finish(0, 50)
            with self.assertRaises(ValueError):
                ledger.finish(0, 50)

    def test_latin_balanced_serial(self):
        schedule = development_schedule()
        self.assertEqual(len(schedule), 60)
        for m in range(20):
            rows = [x for x in schedule if x['mother'] == m]
            self.assertEqual(len({x['arms_serial'][0] for x in rows}), 3)
            self.assertTrue(all(len(set(x['arms_serial'])) == 3 for x in rows))

    def test_finite_stage_plan_and_missing_fit_inputs(self):
        plan = stage_plan()
        self.assertEqual([len(plan[k]) for k in ('B', 'training', 'calibration', 'development')], [140, 96, 72, 180])
        self.assertFalse(plan['dispatch_allowed'])
        with self.assertRaises(ValueError):
            fit_gate({'rows': []}, {}, {})

    def test_object_dedup_and_unresolved_intent(self):
        with tempfile.TemporaryDirectory() as d:
            store = ObjectStore(d)
            self.assertEqual(store.put(b'small'), store.put(b'small'))
            (Path(d) / 'mutation.intent.json').write_text('{}')
            with self.assertRaises(ValueError):
                store.put(b'other')

    def test_duplicate_and_unknown_reservations(self):
        with tempfile.TemporaryDirectory() as d:
            ledger = BudgetLedger(d)
            ledger.reserve('one', 'B', 120)
            with self.assertRaises(ValueError):
                ledger.reserve('one', 'B', 120)
            with self.assertRaises(ValueError):
                ledger.reserve('two', 'B', 3500)
            with self.assertRaises(ValueError):
                ledger.finish('one', 5, {'raw_returncode': 0})
            ledger.finish('one', 7, {'pid': 1, 'process_start_identity': 'synthetic-control-only',
                 'raw_returncode': 0, 'reaped': True, 'descendants_live': 0})
            ledger.reserve('two', 'B', 3500)

    def test_history_byte_exception_and_new_pause(self):
        with tempfile.TemporaryDirectory() as d:
            old, own, new = [Path(d) / x for x in ('old', 'own', 'new')]
            old.write_bytes(b'old')
            refs = [{'path': str(old), 'bytes': 3, 'sha256': hashlib.sha256(b'old').hexdigest()}]
            guard = PauseGuard([own], [old, new], refs, seconds=120)
            guard.check()
            new.write_bytes(b'')
            with self.assertRaises(RuntimeError):
                guard.check()
            new.unlink()
            old.write_bytes(b'new')
            with self.assertRaises(RuntimeError):
                guard.check()
            old.unlink()
            with self.assertRaises(RuntimeError):
                guard.check()

    def test_save_reserve(self):
        ticks = [0.]
        with tempfile.TemporaryDirectory() as d:
            guard = PauseGuard([Path(d)/'own'], [Path(d)/'global'], seconds=120, clock=lambda: ticks[0])
            ticks[0] = 108.
            with self.assertRaises(RuntimeError):
                guard.check()
            guard.check(new_work=False)

    def test_calibration_cannot_rewrite_source(self):
        row = {'case_id': 'original', 'actual_role': 'train', 'source_role': 'train',
               'original_mother_id': 'train:original', 'UTC': 'original', 'source_asset_SHA': 'a'*64,
               'source_aircraft_count': 1}
        case = {**row, 'source_asset_SHA': 'a'*64, 'aircraft_ids': ['a'],
                'fixed_request_bps': 21600000, 'total_request_bps_int64': 21600000, 'content_sha256': 'b'*64}
        entry = {'recipe_index': 288, 'r14_partition': 'calibration', 'original_recipe_row': row}
        self.assertEqual(task_binding(entry, case)['actual_role'], 'train')
        with self.assertRaises(ValueError):
            task_binding(entry, {**case, 'actual_role': 'calibration'})


if __name__ == '__main__':
    unittest.main()
