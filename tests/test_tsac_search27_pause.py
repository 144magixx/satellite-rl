from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

from satellite_rl.tsac_search27.pause import Gate, GateClosed, initialize, watchdog_tick


def instant(value):
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


class PauseGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='w27_gate_test_')
        self.directory = Path(self.tmp.name)
        self.now = instant('2070-01-01T22:57:59')
        self.spec = dict(experiment_id='S2-W27', node_id='20418', scope_root=str(self.directory / 'scope'),
                         uid=1000, soft_UTC='2070-01-01T22:58:00Z', hard_UTC='2070-01-01T23:00:00Z',
                         contract_SHA='contract')
        self.epoch = initialize(self.directory / 'control', self.spec)
        self.gate = Gate(self.directory / 'control', clock=lambda: self.now)
        self.worker = dict(experiment_id='S2-W27', task_id='task1', pid=321, birth_ticks=123,
                           boot_id='boot1', argv=['python', str(self.directory / 'scope' / 'task.json')],
                           cwd=str(self.directory / 'scope'), uid=1000,
                           stop_url='http://127.0.0.1:9070/api/runs/task1/stop')

    def tearDown(self):
        self.tmp.cleanup()

    def register(self):
        self.gate.register_worker(self.epoch, self.worker, inspector=lambda pid: self.worker)

    def test_planned_generation_preserves_old_owner_and_cannot_open_hold(self):
        from unittest.mock import patch
        self.register()
        replacement = dict(self.worker, pid=322, birth_ticks=124)
        receipt = dict(run_id='task1', rootpauseepoch=self.epoch, contract_SHA='contract', previous_worker=self.worker)
        continuation = dict(run_dir='fixture', config={}, receipt_ref={})
        inspect = lambda pid: replacement if pid == 322 else None
        with self.assertRaises(ValueError):
            self.gate.register_worker(self.epoch, replacement, inspector=inspect)
        with patch('satellite_rl.tsac_search27.control.validate_continuation', return_value=receipt):
            self.gate.register_worker(self.epoch, replacement, inspector=inspect, continuation=continuation)
        self.assertEqual(len(self.gate.snapshot()['workers']), 2)
        self.assertEqual(self.gate.snapshot()['events'][-1]['kind'], 'planned_pilot_continuation')
        self.now = instant('2070-01-01T22:58:00')
        with self.assertRaises(GateClosed), patch('satellite_rl.tsac_search27.control.validate_continuation', return_value=receipt):
            self.gate.register_worker(self.epoch, dict(replacement, pid=323), inspector=lambda pid:dict(replacement,pid=323), continuation=continuation)

    def test_unchanged_checks_do_not_write_but_deadline_hold_does(self):
        from unittest.mock import patch
        import satellite_rl.tsac_search27.pause as pause
        with patch.object(pause,'atomic_json',wraps=pause.atomic_json) as save:
            for _ in range(10): self.assertFalse(self.gate.stop_requested(self.epoch))
            self.assertEqual(save.call_count,0)
            self.now=instant('2070-01-01T22:58:00')
            self.assertTrue(self.gate.stop_requested(self.epoch))
            self.assertEqual(save.call_count,1)
            self.assertTrue(self.gate.stop_requested(self.epoch))
            self.assertEqual(save.call_count,1)

    def test_soft_hold_persists_across_new_gate_date_and_stale_epoch(self):
        self.assertFalse(self.gate.stop_requested(self.epoch))
        self.now = instant('2070-01-01T22:58:00')
        self.assertTrue(self.gate.stop_requested(self.epoch))
        future = Gate(self.gate.directory, clock=lambda: instant('2070-01-02T08:00:00'))
        with self.assertRaises(GateClosed):
            future.validate(self.epoch)
        self.assertTrue(future.snapshot()['hold'])

    def test_only_explicit_bound_human_receipt_resumes(self):
        self.now = instant('2070-01-01T23:01:00')
        self.gate.refresh()
        receipt = dict(receipt_id='human1', user_message_reference='conversation/user/continue',
                       issued_UTC='2070-01-01T23:01:00Z', hold_epoch=self.gate.snapshot()['hold_epoch'],
                       contract_SHA='contract', root_attested_user_continue=True)
        for patch in (dict(root_attested_user_continue=False), dict(contract_SHA='other'),
                      dict(hold_epoch='old'), dict(issued_UTC='2070-01-01T22:59:59Z')):
            with self.assertRaises(GateClosed):
                self.gate.resume(dict(receipt, **patch))
        epoch2 = self.gate.resume(receipt)
        self.assertNotEqual(epoch2, self.epoch)
        self.assertEqual(self.gate.validate(epoch2), epoch2)
        self.assertTrue(self.gate.stop_requested(self.epoch))
        with self.assertRaises(GateClosed):
            self.gate.resume(receipt)

    def test_worker_other_group_or_scope_rejected(self):
        for patch in (dict(experiment_id='S1-AI17'), dict(cwd='/outside'), dict(argv=['python','other.py']), dict(uid=1001),
                      dict(stop_url='http://example.org/api/runs/task1/stop')):
            bad = dict(self.worker, **patch)
            with self.assertRaises(ValueError):
                self.gate.register_worker(self.epoch, bad, inspector=lambda pid: bad)

    def test_pid_reuse_and_reboot_never_signal_or_api_other_process(self):
        self.register()
        self.now = instant('2070-01-01T23:00:00')
        calls = []
        for patch in (dict(birth_ticks=999), dict(boot_id='newboot'), dict(argv=['other-job'])):
            other = dict(self.worker, **patch)
            result = watchdog_tick(self.gate, inspector=lambda pid: other,
                                   api_stop=lambda w: calls.append('api'), terminate=lambda w: calls.append('term'))
            self.assertEqual(result['live'], 0)
        self.assertEqual(calls, [])

    def test_soft_api_then_hard_exact_term_unknown_spend_preserved(self):
        self.register()
        self.now = instant('2070-01-01T22:58:00')
        calls = []
        api = lambda w: calls.append('api') or dict(status=200)
        term = lambda w: calls.append('term') or dict(sent=True)
        watchdog_tick(self.gate, inspector=lambda pid: self.worker, api_stop=api, terminate=term)
        self.assertEqual(calls, ['api'])
        self.now = instant('2070-01-01T23:00:00')
        outcome = watchdog_tick(self.gate, inspector=lambda pid: self.worker, api_stop=api, terminate=term)
        self.assertEqual(calls, ['api', 'api', 'term'])
        self.assertEqual(outcome['unresolved'], 1)
        record = self.gate.snapshot()['enforcement']['task1']
        self.assertTrue(record['unknown_interaction_spend_not_zero'])
        self.assertFalse(record['free_replay_allowed'])
        watchdog_tick(self.gate, inspector=lambda pid: self.worker, api_stop=api, terminate=term)
        self.assertEqual(calls, ['api', 'api', 'term'])

    def test_empty_registry_holds_without_touching_any_process(self):
        self.now = instant('2070-01-01T23:00:00')
        def forbidden(*args):
            self.fail('empty registry must not inspect or stop unrelated processes')
        outcome = watchdog_tick(self.gate, inspector=forbidden, api_stop=forbidden, terminate=forbidden)
        self.assertTrue(outcome['hold'])
        self.assertEqual(outcome['live'], 0)

    def test_cutoff_blocks_new_worker_registration(self):
        self.now = instant('2070-01-01T22:58:00')
        with self.assertRaises(GateClosed):
            self.register()
        self.assertEqual(self.gate.snapshot()['workers'], [])


if __name__ == '__main__':
    unittest.main()
