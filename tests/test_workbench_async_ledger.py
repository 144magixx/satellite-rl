import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from satellite_rl.workbench.config import default_config, validate_config
from satellite_rl.workbench.checkpoints import config_hash
from satellite_rl.workbench.runner import Ledger, _jsonl, _reconcile, _restore_logging


class Writer:
    def __init__(self, **kwargs):
        self.rows = []
        self.kwargs = kwargs

    def add_scalar(self, tag, value, step):
        self.rows.append((tag, value, step))

    def flush(self):
        pass

    def close(self):
        pass


class AsyncLedgerTests(unittest.TestCase):
    def test_telemetry_is_operational_but_validated(self):
        old = default_config()
        old.pop('telemetry')
        new = validate_config(old)
        self.assertEqual(config_hash(old), config_hash(new))
        new['telemetry']['mode'] = 'async'
        self.assertEqual(config_hash(old), config_hash(new))
        new['telemetry']['queue_capacity'] = 0
        with self.assertRaises(ValueError):
            validate_config(new)

    def test_sync_async_preserve_all_values_and_boundary(self):
        configs = []
        recorded = []
        with tempfile.TemporaryDirectory() as temporary:
            for mode in ('sync', 'async'):
                directory = Path(temporary) / mode
                directory.mkdir()
                config = default_config()
                config['telemetry'].update(mode=mode, queue_capacity=16)
                configs.append(config)
                with patch('torch.utils.tensorboard.SummaryWriter', Writer):
                    ledger = Ledger(directory, {'run_id': mode}, config, {'gpu': 'CPU test'})
                    try:
                        for step in range(1, 201):
                            row = {'env_step': step, 'values': {'loss': step / 10}}
                            _jsonl(directory / 'updates.jsonl', row)
                            row['values']['loss'] = -999
                            if step % 50 == 0:
                                ledger.record('learning', step, {'loss': step / 10})
                                ledger.save_status(env_step=step)
                        ledger.flush()
                        rows = [json.loads(s) for s in (directory / 'updates.jsonl').read_text().splitlines()]
                        self.assertEqual([x['env_step'] for x in rows], list(range(1, 201)))
                        self.assertEqual(rows[-1]['values']['loss'], 20)
                        metrics = [json.loads(s) for s in (directory / 'metrics.jsonl').read_text().splitlines()]
                        recorded.append((rows, [(x['group'], x['step'], x['values']) for x in metrics],
                                         copy.deepcopy(ledger.writer.rows)))
                        self.assertEqual(json.loads((directory / 'status.json').read_text())['env_step'], 200)
                    finally:
                        ledger.close()
                    self.assertTrue(ledger.closed)
                    # Closed sink registration must not capture a later synchronous append.
                    _jsonl(directory / 'updates.jsonl', {'env_step': 201})
            self.assertEqual(recorded[0], recorded[1])

    def test_resume_discards_only_uncommitted_partial_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary) / 'updates.jsonl'
            p.write_text('{"env_step":1}\n{"env_step":2}\n{"env_step":3', encoding='utf-8')
            _reconcile(temporary, 1)
            self.assertEqual([json.loads(s) for s in p.read_text().splitlines()], [{'env_step': 1}])
            p.write_text('{"env_step":1}\ninvalid\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                _reconcile(temporary, 1)
            self.assertIn('invalid', p.read_text())

    def test_async_resume_opens_tensorboard_after_trim(self):
        with tempfile.TemporaryDirectory() as temporary, patch('torch.utils.tensorboard.SummaryWriter', Writer):
            directory = Path(temporary)
            config = default_config()
            config['telemetry']['mode'] = 'async'
            first = Ledger(directory, {'run_id': 'resume'}, config, {})
            _jsonl(directory / 'updates.jsonl', {'env_step': 1})
            _jsonl(directory / 'updates.jsonl', {'env_step': 2})
            first.close()
            resumed = Ledger(directory, {'run_id': 'resume'}, config, {}, resume=True)
            try:
                self.assertIsNone(resumed.writer)
                _restore_logging(resumed, directory, 1)
                self.assertEqual(resumed.writer.kwargs['purge_step'], 2)
                _jsonl(directory / 'updates.jsonl', {'env_step': 2, 'resumed': True})
            finally:
                resumed.close()
            self.assertEqual([json.loads(s) for s in (directory/'updates.jsonl').read_text().splitlines()],
                             [{'env_step': 1}, {'env_step': 2, 'resumed': True}])


if __name__ == '__main__':
    unittest.main()
