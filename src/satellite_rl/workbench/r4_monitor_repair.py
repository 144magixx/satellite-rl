"""Source-pinned operations epoch; never replace the original scientific snapshot."""
import os
import json
import time
import hashlib
from pathlib import Path
from .storage import read_json, digest, now, validate_id
from .coverage_ai_search import pin
from satellite_rl.training.run_monitor import atomic_json

SCIENCE = '69e83fd4a2998c539540c282e2a0906505b03d97'
RUNS = {'b0_ai_search_r4_20261013_150920_a0ebefd3',
        'b0_ai_search_r4_20261013_150925_aa879954',
        'b0_ai_search_r4_20261013_150931_5fdf8e1f'}


def rss_tree(pid, proc=Path('/proc'), page_size=None, samples=None):
    """Actual kernel RSS; ESRCH ignored only after gone/Z/X confirmation."""
    if os.name == 'nt' and proc == Path('/proc'):
        from .r4_stage_guard import rss_tree as original
        return original(pid)
    page_size = page_size or os.sysconf('SC_PAGE_SIZE')
    queue, seen, total = [pid], set(), 0
    while queue:
        current = queue.pop()
        if current in seen:
            continue
        seen.add(current)
        root = Path(proc) / str(current)
        try:
            lines = (root / 'status').read_text().splitlines()
            rows = [line.split() for line in lines if line.startswith('VmRSS:')]
            if rows:
                if len(rows) != 1 or len(rows[0]) != 3 or rows[0][2] != 'kB':
                    raise ValueError('malformed VmRSS')
                resident = int(rows[0][1]) * 1024
                source = 'status.VmRSS'
            else:
                fields = (root / 'statm').read_text().split()
                if len(fields) != 7 or any(not x.isdecimal() for x in fields):
                    raise ValueError('malformed statm')
                resident = int(fields[1]) * page_size
                source = 'statm.resident'
            if resident < 0:
                raise ValueError('negative RSS')
            total += resident
            if samples is not None and source == 'statm.resident':
                samples.append({'pid': current, 'source': source, 'rss_bytes': resident, 'at': now()})
            queue.extend(int(x) for x in (root / 'task' / str(current) / 'children').read_text().split())
        except FileNotFoundError:
            continue
        except ProcessLookupError:
            try:
                raw=(root/'stat').read_text();state=raw[raw.rfind(')')+2:].split()[0]
                gone=state in ('Z','X')
            except (FileNotFoundError,ProcessLookupError):
                if root.exists():raise
                gone=True
            if not gone:raise
            if samples is not None:samples.append(dict(pid=current,source='confirmed_proc_exit_ESRCH',at=now()))
            continue
    return total


def validate_repair(job, metadata):
    """Root-signed references and immutable original identity are required at each launch."""
    if job['run_id'] not in RUNS or job['scientific_commit'] != SCIENCE:
        raise ValueError('repair restricted to the three original B0 IDs')
    contract = read_json(pin(metadata['contract_ref']), {})
    review = read_json(pin(metadata['root_review_ref']), {})
    validate_id(metadata['repair_id'])
    if (contract.get('schema') != 'R4OperationsRepairV1' or
            contract.get('repair_id') != metadata.get('repair_id') or
            review.get('passed') is not True or
            review.get('repair_id') != contract['repair_id'] or
            review.get('contract_sha256') != metadata['contract_ref']['sha256'] or
            contract.get('scientific_commit') != SCIENCE or
            set(contract.get('tasks', {})) != RUNS or
            contract.get('extra_prepare_attempts') != 3 or
            contract.get('total_prepare_attempt_cap') != 27 or
            contract.get('preserve_original_unknown_cost') is not True):
        raise ValueError('exact root repair approval missing')
    entry = contract['tasks'][job['run_id']]
    if set(contract['operations_source_refs']) != {'module', 'bootstrap', 'runtime', 'api'}:
        raise ValueError('exact operations source members required')
    if not entry.get('evidence_refs'):
        raise ValueError('independent root evidence required')
    for key in ('run_id', 'request_id', 'request_hash', 'scientific_commit', 'source_root',
                'source_manifest_sha256', 'config_sha256', 'references', 'original_task_contract'):
        if entry['original_job'][key] != job[key]:
            raise ValueError('original repair task identity changed: ' + key)
    for ref in contract['operations_source_refs'].values():
        pin(ref)
    if digest(__file__) != contract['operations_source_refs']['module']['sha256']:
        raise ValueError('executing repair module differs')
    for ref in entry['evidence_refs'].values():
        pin(ref)
    retry = entry['prepare_retry']
    for ref in retry['references'].values():
        pin(ref)
    if set(retry['references']) != {'guard', 'wal', 'request', 'output'}:
        raise ValueError('exact original failed stage refs required')
    return contract, entry


def repair_resume(scheduler, run_id, policy, body):
    from . import frozen_round_tasks
    with scheduler.lock:
        directory = scheduler.store.job_dir(run_id)
        job = read_json(directory / 'job.json', {})
        frozen_round_tasks.verify(scheduler.store, job, policy)
        metadata = {key: body[key] for key in ('repair_id', 'contract_ref', 'root_review_ref')}
        validate_repair(job, metadata)
        status = read_json(directory / 'status.json', {})
        if job.get('operations_repair'):
            if job['operations_repair'] != metadata:
                raise ValueError('cannot switch repair contract on original ID')
            if status['stage'] in ('queued', 'starting', 'running', 'completed'):
                return status
        result = frozen_round_tasks.resume(scheduler, run_id, policy)
        job = read_json(directory / 'job.json', {})
        job['operations_repair'] = metadata
        atomic_json(directory / 'job.json', job)
        owner = read_json(job['owner_path'], {})
        owner['job_sha256'] = digest(directory / 'job.json')
        atomic_json(job['owner_path'], owner)
        return result


def execution_command(job, directory, python):
    contract, _ = validate_repair(job, job['operations_repair'])
    return [python, '-B', '-u', str(pin(contract['operations_source_refs']['bootstrap'])),
            '--job-dir', str(directory), '--resume',
            '--original-runner', 'satellite_rl.workbench.runner']


def input_read_ledger(directory, original):
    """Operational rereads remain separate from original scientific cost receipts."""
    directory = Path(directory)
    path = directory / 'operations_input_reads.jsonl'
    def load(reference, indices):
        began = time.perf_counter()
        rows, error = None, None
        try:
            rows = original(reference, indices)
            return rows
        except BaseException as exc:
            error = type(exc).__name__ + ': ' + str(exc)
            raise
        finally:
            elapsed = time.perf_counter() - began
            previous = None if rows is None else [
                (directory / 'bundles' / hashlib.sha256(row['case_id'].encode()).hexdigest() /
                 'input_read_receipt.json').exists() for row in rows]
            record = {'case_indices': list(indices), 'actual_read_seconds': elapsed,
                      'previous_receipt_exists': previous,
                      'extra_operations_reread': None if previous is None else any(previous),
                      'classification': 'failed_read_unknown' if previous is None else
                          'operations_reread' if any(previous) else 'original_first_read',
                      'error': error, 'at': now(), 'scientific_cost_receipts_unchanged': True}
            with path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
    return load


def repaired_guard(job, original):
    contract, entry = validate_repair(job, job['operations_repair'])
    execution = {'repair_id': contract['repair_id'], 'scientific_commit': SCIENCE,
                 'operations_source_refs': contract['operations_source_refs'],
                 'contract_ref': job['operations_repair']['contract_ref'],
                 'root_review_ref': job['operations_repair']['root_review_ref']}

    def run(command, directory, limit_seconds, memory_bytes, stop=lambda: False, **kwargs):
        directory = Path(directory).resolve()
        retry = entry['prepare_retry']
        if directory == Path(retry['original_directory']).resolve():
            refs = retry['references']
            for ref in refs.values():
                pin(ref)
            old = read_json(pin(refs['guard']), {})
            wal = read_json(pin(refs['wal']), {})
            request = read_json(pin(refs['request']), {})
            if (retry.get('approved') is not True or request.get('stage') != 'preparation' or
                    old.get('status') != 'hard_unpublished' or old.get('reason') != 'monitor_error' or
                    old.get('exit_code') != 0 or old.get('output_ref') != refs['output'] or
                    wal.get('command') != command or limit_seconds != 20 or
                    retry.get('original_processes_gone') is not True or
                    retry.get('source_identity_ledger_verified') is not True or
                    Path(refs['guard']['path']).resolve() != directory / 'guard_receipt.json' or
                    Path(refs['wal']['path']).resolve() != directory / 'stage.wal.json' or
                    Path(refs['request']['path']).resolve() != directory / 'request.json'):
                raise ValueError('prepare retry not exactly authorized')
            job_directory = directory.parents[2]
            target = job_directory / 'operations_repair' / contract['repair_id'] / 'prepare_retry' / directory.parent.name
            target.mkdir(parents=True, exist_ok=True)
            # Exactly one fresh WAL is permitted. Original guard/WAL/output stay unchanged.
            rewritten = list(command)
            if rewritten.count('--directory') != 1 or rewritten[-2:] != ['--directory', str(directory)]:
                raise ValueError('unexpected original prepare command')
            rewritten[-1] = str(target)
            samples = []
            kwargs['memory'] = lambda pid: rss_tree(pid, samples=samples)
            value = original(rewritten, target, limit_seconds, memory_bytes, stop, **kwargs)
            value = dict(value, ops_execution=execution)
            atomic_json(target / 'operations_guard_receipt.json',
                        {'guard': value, 'statm_fallback_samples': samples,
                         'original_failed_stage_refs': refs, 'original_unknown_cost_preserved': True,
                         'extra_prepare_attempt': 1})
            return value
        samples = []
        kwargs['memory'] = lambda pid: rss_tree(pid, samples=samples)
        value = original(command, directory, limit_seconds, memory_bytes, stop, **kwargs)
        value = dict(value, ops_execution=execution)
        atomic_json(directory / 'operations_guard_receipt.json',
                    {'guard': value, 'statm_fallback_samples': samples})
        return value
    return run


def main(argv=None):
    import argparse
    from . import r4_stage_guard
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true', required=True)
    parser.add_argument('--original-runner', required=True, choices=['satellite_rl.workbench.runner'])
    args = parser.parse_args(argv)
    job = read_json(args.job_dir / 'job.json', {})
    contract, _ = validate_repair(job, job['operations_repair'])
    original = r4_stage_guard.run_stage
    r4_stage_guard.run_stage = repaired_guard(job, original)
    from . import ai_search_runner
    ai_search_runner.load_cases = input_read_ledger(args.job_dir, ai_search_runner.load_cases)
    # Import after injection: original runner's `from ... import run_stage` binds the shim.
    from .runner import run_job
    atomic_json(args.job_dir / 'operations_execution_receipt.json',
                {'repair': job['operations_repair'], 'operations_source_refs': contract['operations_source_refs'],
                 'scientific_commit': SCIENCE, 'original_source_root': job['source_root'], 'started_at': now(),
                 'operations_input_reads_path': str(args.job_dir / 'operations_input_reads.jsonl'),
                 'operations_input_reads_append_only': True})
    run_job(args.job_dir, resume=True)
