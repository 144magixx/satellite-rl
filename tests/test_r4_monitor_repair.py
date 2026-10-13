"""Operations-only tests: mocked /proc and stages, zero geometry/native/CUDA."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from satellite_rl.workbench import r4_monitor_repair as repair
from satellite_rl.workbench.storage import digest, read_json


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')
        return {'path': str(path), 'sha256': digest(path)}

    def proc(self, status='Name: mock\n', statm='20 3 0 0 0 0 0'):
        root = self.root / 'proc' / '10'
        (root / 'task' / '10').mkdir(parents=True)
        (root / 'status').write_text(status)
        (root / 'statm').write_text(statm)
        (root / 'task' / '10' / 'children').write_text('')
        return self.root / 'proc'

    def test_live_missing_vmrss_uses_actual_statm(self):
        samples = []
        self.assertEqual(repair.rss_tree(10, self.proc(), 4096, samples), 12288)
        self.assertEqual(samples[0]['source'], 'statm.resident')

    def test_malformed_fails_closed(self):
        for status, statm in [('VmRSS: broken kB', '20 3 0 0 0 0 0'),
                              ('Name: mock', 'broken')]:
            with self.subTest(status=status):
                proc = self.root / 'proc'
                if proc.exists():
                    import shutil
                    shutil.rmtree(proc)
                with self.assertRaises(ValueError):
                    repair.rss_tree(10, self.proc(status, statm), 4096)

    def test_exit_enoent_and_valid_status(self):
        self.assertEqual(repair.rss_tree(99, self.root / 'empty', 4096), 0)
        self.assertEqual(repair.rss_tree(10, self.proc('VmRSS: 7 kB'), 4096), 7168)

    def test_input_reread_ledger_classifies_and_preserves_original(self):
        import hashlib
        case = 'original-case'
        receipt = self.root / 'bundles' / hashlib.sha256(case.encode()).hexdigest() / 'input_read_receipt.json'
        self.write(receipt, {'actual_seconds': .1})
        original_sha = digest(receipt)
        wrapped = repair.input_read_ledger(self.root, lambda ref, ids: [{'case_id': case}])
        self.assertEqual(wrapped({}, [0]), [{'case_id': case}])
        repair.input_read_ledger(self.root, lambda ref, ids: [{'case_id': 'new-case'}])({}, [1])
        records = [json.loads(line) for line in (self.root / 'operations_input_reads.jsonl').read_text().splitlines()]
        self.assertEqual([r['extra_operations_reread'] for r in records], [True, False])
        self.assertEqual(digest(receipt), original_sha)

    def test_failed_input_read_is_recorded_in_finally(self):
        def fail(ref, ids):
            raise OSError('mock input failure')
        with self.assertRaises(OSError):
            repair.input_read_ledger(self.root, fail)({}, [0])
        record = json.loads((self.root / 'operations_input_reads.jsonl').read_text())
        self.assertEqual(record['classification'], 'failed_read_unknown')
        self.assertIn('OSError', record['error'])
        self.assertGreaterEqual(record['actual_read_seconds'], 0)

    def fixture(self):
        run = sorted(repair.RUNS)[0]
        directory = self.root / run
        original = directory / 'bundles' / 'casehash' / 'preparation'
        command = ['python', '-m', 'stage', '--directory', str(original)]
        output = self.write(original / 'stage_output.json', {'result': {'status': 'available'}})
        guard = self.write(original / 'guard_receipt.json', {
            'status': 'hard_unpublished', 'reason': 'monitor_error', 'exit_code': 0,
            'output_ref': output, 'unknown_algorithm_spend': True, 'actual_stage_wall_seconds': .4})
        refs = {'guard': guard, 'output': output,
                'wal': self.write(original / 'stage.wal.json', {'command': command}),
                'request': self.write(original / 'request.json', {'stage': 'preparation'})}
        job = {'run_id': run, 'request_id': 'original_request', 'request_hash': 'hash',
               'scientific_commit': repair.SCIENCE, 'source_root': '/original',
               'source_manifest_sha256': 'sha', 'config_sha256': 'config',
               'references': {'plan': {}}, 'original_task_contract': {'budget': 0}}
        sources = {name: {'path': repair.__file__, 'sha256': digest(repair.__file__)}
                   for name in ('module', 'bootstrap', 'runtime', 'api')}
        entry = {'original_job': dict(job), 'evidence_refs': {'audit': output},
                 'prepare_retry': {'original_directory': str(original), 'approved': True,
                     'original_processes_gone': True, 'source_identity_ledger_verified': True,
                     'references': refs}}
        contract = {'schema': 'R4OperationsRepairV1', 'repair_id': 'repair1',
                    'scientific_commit': repair.SCIENCE, 'extra_prepare_attempts': 3,
                    'total_prepare_attempt_cap': 27, 'preserve_original_unknown_cost': True,
                    'operations_source_refs': sources, 'tasks': {r: entry for r in repair.RUNS}}
        contract_ref = self.write(self.root / 'contract.json', contract)
        review_ref = self.write(self.root / 'review.json', {'passed': True, 'repair_id': 'repair1',
                                                         'contract_sha256': contract_ref['sha256']})
        metadata = {'repair_id': 'repair1', 'contract_ref': contract_ref, 'root_review_ref': review_ref}
        job['operations_repair'] = metadata
        return job, directory, original, command, refs

    def test_retry_routes_once_and_preserves_unknown_original(self):
        job, directory, original, command, refs = self.fixture()
        calls = []
        def guarded(cmd, target, *args, **kwargs):
            calls.append((cmd, target))
            return {'status': 'completed', 'output_ref': {'path': str(target / 'stage_output.json')}}
        value = repair.repaired_guard(job, guarded)(command, original, 20, 2147483648)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], directory / 'operations_repair/repair1/prepare_retry/casehash')
        self.assertEqual(calls[0][0][-1], str(calls[0][1]))
        self.assertIn('ops_execution', value)
        self.assertTrue(read_json(original / 'guard_receipt.json')['unknown_algorithm_spend'])
        self.assertEqual(digest(original / 'guard_receipt.json'), refs['guard']['sha256'])

    def test_bootstrap_command_preserves_owned_runner_visibility(self):
        job, directory, _, _, _ = self.fixture()
        command = repair.execution_command(job, directory, 'python')
        self.assertEqual(command[-2:], ['--original-runner', 'satellite_rl.workbench.runner'])
        self.assertEqual(command[command.index('--job-dir') + 1], str(directory))

    def test_wrong_id_sha_or_root_review_rejected(self):
        job, _, _, _, refs = self.fixture()
        for field, wrong in [('run_id', 'foreign'), ('source_manifest_sha256', 'wrong')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                repair.validate_repair(dict(job, **{field: wrong}), job['operations_repair'])
        Path(refs['output']['path']).write_text('{}')
        with self.assertRaises(ValueError):
            repair.validate_repair(job, job['operations_repair'])

    def test_unapproved_retry_never_calls_child(self):
        job, _, original, command, _ = self.fixture()
        contract, entry = repair.validate_repair(job, job['operations_repair'])
        entry['prepare_retry']['approved'] = False
        from unittest.mock import Mock
        child = Mock()
        with patch.object(repair, 'validate_repair', return_value=(contract, entry)):
            with self.assertRaises(ValueError):
                repair.repaired_guard(job, child)(command, original, 20, 2147483648)
        child.assert_not_called()

    def test_unknown_retry_wal_never_spawns_again(self):
        job, directory, original, command, _ = self.fixture()
        target = directory / 'operations_repair/repair1/prepare_retry/casehash'
        self.write(target / 'stage.wal.json', {'status': 'child_started'})
        from satellite_rl.workbench.r4_stage_guard import run_stage
        from unittest.mock import Mock
        child = Mock()
        with self.assertRaises(ValueError):
            repair.repaired_guard(job, run_stage)(command, original, 20, 2147483648, popen=child)
        child.assert_not_called()

    def test_starting_and_completed_api_retry_is_idempotent(self):
        job, directory, _, _, _ = self.fixture()
        self.write(directory / 'job.json', job)
        scheduler = SimpleNamespace(lock=threading.RLock(),
                                    store=SimpleNamespace(job_dir=lambda run: directory))
        from satellite_rl.workbench import frozen_round_tasks
        for stage in ('queued', 'starting', 'running', 'completed'):
            self.write(directory / 'status.json', {'stage': stage})
            with patch.object(frozen_round_tasks, 'verify'), patch.object(frozen_round_tasks, 'resume') as resum:
                result = repair.repair_resume(scheduler, job['run_id'], None, job['operations_repair'])
                self.assertEqual(result['stage'], stage)
                resum.assert_not_called()

    def test_resume_same_identity_history_and_idempotence(self):
        job, directory, _, _, _ = self.fixture()
        job['owner_path'] = str(directory / 'owner.json')
        self.write(directory / 'owner.json', {'run_id': job['run_id']})
        self.write(directory / 'job.json', job)
        self.write(directory / 'status.json', {'stage': 'failed'})
        scheduler = SimpleNamespace(lock=threading.RLock(),
                                    store=SimpleNamespace(job_dir=lambda run: directory))
        def resume(s, run, policy):
            self.write(directory / 'recovery_history/1/job.json', job)
            self.write(directory / 'status.json', {'stage': 'queued'})
            return {'stage': 'queued'}
        from satellite_rl.workbench import frozen_round_tasks
        with patch.object(frozen_round_tasks, 'verify'), patch.object(frozen_round_tasks, 'resume', side_effect=resume) as resum:
            repair.repair_resume(scheduler, job['run_id'], None, job['operations_repair'])
            repair.repair_resume(scheduler, job['run_id'], None, job['operations_repair'])
            self.assertEqual(resum.call_count, 1)
        final = read_json(directory / 'job.json')
        self.assertEqual(final['request_id'], job['request_id'])
        self.assertEqual(final['original_task_contract'], job['original_task_contract'])
        self.assertTrue((directory / 'recovery_history/1/job.json').exists())


if __name__ == '__main__':
    unittest.main()
