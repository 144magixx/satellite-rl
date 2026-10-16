"""Fixture-qualified dispatch kernel; no transport, daemon, or implicit authority.

Deployment must supply separately reviewed adapters and a root-pinned delegation.
The controller never creates root engineering acceptance or reconciles unknown POSTs.

Operational boundary (not yet deployed): root first freezes the controller source,
full source/plan/data and exact current-stage whitelist, after actual identity and
storage qualification. The deployment adapter authenticates that root document,
checks raw plan bytes, constructs this library and invokes tick at low cost. It
must verify live service birth/boot and each run/config, saved probe checkpoint,
closed ledger and new pause state from original artifacts. No adapter is provided
here. SSH, probe collection, uploading receipts, workbench HTTP mapping, singleton
controller ownership and real deployment qualification remain separate work.
Only the injected receipt writer may materialize a delegated fresh resources
receipt; it cannot write engineering/science root acceptance. Model review keeps
its separate 8/4/2-hour cadence. An unresolved intent requires root reconciliation
and a new explicit controller generation, never deletion/replay by this library.
The caller's root_approved boolean and hash are consistency checks only, not
authentication. This library is not a deployable authorization boundary without
an independently trusted root-pin/authentication adapter reviewed separately.
"""
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone


MENUS = {'P00', 'P10', 'P01', 'P11'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf8') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class Controller:
    """One caller only. All reads/probes/POSTs are injected, never network defaults.

    `delegation` is a root-authenticated, SHA-pinned document supplied by the
    deployment adapter; its hash must be an independently pinned root argument.
    `snapshot` returns task-keyed workbench status, with live identity checked by
    the adapter. It must check actual config.json after workbench-added run_id
    and model_dir against job/config hash and the plan; then return the original
    plan config's digest in config_sha256. These normalized verified fields are
    adapter output, not fields already supplied by the current workbench API.
    `probe` returns measured facts, never inferred from old PID.
    `write_receipt` durably writes fresh ops evidence and returns {path, sha256}.
    `post(action, run_id, body)` performs one request and returns a verified
    envelope {outcome:'accepted', evidence_verified:true, run_id, task_key,
    plan_sha256, status}. Adapter verifies actual server response and identities;
    null, unknown, error and unverified responses remain unresolved durable intent.
    """
    def __init__(self, directory, plan, delegation, delegation_sha, *, snapshot,
                 probe, write_receipt, post, clock=None):
        if digest(delegation) != delegation_sha:
            raise ValueError('Delegation bytes differ')
        if (delegation.get('root_approved') is not True
                or delegation.get('schema') != 'W30OpsDelegationV1'
                or delegation.get('authority', {}).get('schema') != 'W30HumanImplementationAuthorityV1'
                or delegation.get('authority', {}).get('group_window_hours') != 168
                or delegation.get('authority', {}).get('new_human_pause_overrides') is not True
                or not delegation.get('deadline_utc')
                or delegation.get('deadline_utc') != delegation['authority'].get('group_deadline_UTC')
                or not delegation.get('plan_sha256')
                or delegation.get('plan_content_sha256') != digest(plan)):
            raise ValueError('Explicit root delegation required')
        if delegation.get('controller_source_sha256') != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            raise ValueError('Controller source is not root-pinned')
        self.directory = Path(directory)
        self.plan = plan
        self.delegation = delegation
        self.delegation_sha = delegation_sha
        self.deadline = delegation['deadline_utc']
        if datetime.fromisoformat(self.deadline.replace('Z', '+00:00')).tzinfo is None:
            raise ValueError('Authority deadline must carry timezone')
        self.snapshot, self.probe = snapshot, probe
        self.write_receipt, self.post = write_receipt, post
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.validate_matrix()

    def validate_matrix(self):
        if digest(self.delegation) != self.delegation_sha:
            raise ValueError('Delegation changed')
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != self.delegation['controller_source_sha256']:
            raise ValueError('Controller source changed')
        if digest(self.plan) != self.delegation['plan_content_sha256']:
            raise ValueError('Plan changed')
        tasks = self.plan['tasks']
        allowed = self.delegation['task_keys']
        if len(allowed) != 8 or set(allowed) != set(tasks):
            raise ValueError('Exactly current-stage eight root-whitelisted tasks')
        cells = set()
        for key, task in tasks.items():
            if task['executor_port'] not in (20655, 20654, 20653, 20627):
                raise ValueError('Unauthorized executor port')
            config = task['config']
            mode = task['mode']
            if mode != self.delegation['mode'] or mode not in ('engineering', 'train'):
                raise ValueError('No extension or confirmation task')
            expected = (2048, 73030) if mode == 'engineering' else (150000, 51001)
            if ((config['training']['total_steps'], config['training']['seed']) != expected
                    or config.get('first_round', True) is not True
                    or config['source_sha'] != self.plan['source_commit']
                    or config['deadline_utc'] != self.deadline):
                raise ValueError('Frozen first-stage budget/source/deadline differs')
            cell = (config['menu'], config['group'])
            if cell in cells:
                raise ValueError('Duplicate cell')
            cells.add(cell)
            if digest(config) != self.delegation['config_sha256'][key]:
                raise ValueError('Config differs from root binding')
        if cells != {(m, g) for m in MENUS for g in ('M0', 'M1')}:
            raise ValueError('Complete fixed eight-cell matrix required')
        for menu in MENUS:
            pair = [t for t in tasks.values() if t['config']['menu'] == menu]
            if pair[0]['executor_port'] != pair[1]['executor_port']:
                raise ValueError('Menu M0/M1 must retain same device lane')

    def tick(self):
        """One bounded read/dispatch pass. Unresolved durable intents only observe.

        There is deliberately no sleep loop; deployment owns low-cost cadence.
        Model 8/4/2h review is independent from this health/queue pass.
        """
        self.validate_matrix()
        statuses = self.snapshot()
        if self.clock() >= datetime.fromisoformat(self.deadline.replace('Z', '+00:00')):
            return 'deadline'
        # All M0 finish before the M1 wave; fixed ordering never uses scores.
        tasks = self.plan['tasks']
        # Adapter must verify status against live service birth/boot, run/config,
        # checkpoint and durable ledger. Raw API status words are insufficient.
        for key, status in statuses.items():
            if key not in tasks or status.get('evidence_verified') is not True:
                raise ValueError('Unverified workbench snapshot')
            if status.get('config_sha256') != digest(tasks[key]['config']):
                raise ValueError('Snapshot run/config identity differs')
        unresolved = [p for p in self.directory.glob('*.json')
                      if json.loads(p.read_text(encoding='utf8')).get('state') == 'intent']
        if unresolved:
            return 'unknown_observe_only'
        m0_done = all(self.completed(statuses.get(k), t) for k, t in tasks.items()
                      if t['config']['group'] == 'M0')
        for key in self.delegation['task_keys']:
            task = tasks[key]
            lane_statuses = [s for k, s in statuses.items()
                             if tasks[k]['executor_port'] == task['executor_port']]
            if any(s.get('status') in ('failed', 'interrupted', 'needs_root_recovery_accounting')
                   for s in lane_statuses):
                continue
            if any(s.get('status') == 'stopped_during_evaluation'
                   or (s.get('status') == 'stopped'
                       and s.get('stop_reason') != 'engineering_2016_recovery_probe')
                   for s in lane_statuses):
                continue
            if any(s.get('alive') is True or s.get('status') in ('created', 'starting', 'running', 'stop_requested')
                   for s in lane_statuses):
                continue
            if task['config']['group'] == 'M1' and not m0_done:
                continue
            status = statuses.get(key)
            if self.completed(status, task):
                continue
            action, run_id = 'submit', None
            if status:
                if (task['mode'] == 'engineering' and status.get('status') == 'stopped'
                        and status.get('progress', {}).get('env_step') == 2016
                        and status.get('progress', {}).get('update_step') == 17
                        and status.get('alive') is False
                        and status.get('stop_reason') == 'engineering_2016_recovery_probe'
                        and status.get('checkpoint_saved') is True
                        and status.get('ledger_closed') is True):
                    action, run_id = 'resume', status['run_id']
                else:
                    continue  # failed/unknown/user stop never gets automatic recovery
            intent_path = self.directory / (key + '-' + action + '.json')
            if intent_path.exists():
                continue  # includes accepted and unknown; never re-POST automatically
            facts = self.probe(key)
            self.validate_facts(key, facts)
            receipt = dict(facts, schema='W30DelegatedFreshOpsV1', experiment_id='S2-W30',
                           root_approved=True, passed=True, delegated=True,
                           delegation_sha256=self.delegation_sha,
                           plan_sha256=self.delegation['plan_sha256'], source_commit=self.plan['source_commit'],
                           config_sha256=digest(task['config']), task_key=key,
                           max_age_seconds=60)
            ops_ref = self.write_receipt(receipt)
            body = dict(plan_sha256=self.delegation['plan_sha256'], ops_admission_ref=ops_ref)
            if action == 'submit':
                body.update(task_key=key, request_id=digest(dict(
                    delegation=self.delegation_sha, task=key))[:32])
            intent = dict(action=action, run_id=run_id, body=body, state='intent')
            save(intent_path, intent)
            try:
                answer = self.post(action, run_id, body)
            except Exception as error:
                # Intent remains durable even if transport accepted but reply lost.
                intent.update(transport_error=repr(error))
                save(intent_path, intent)
                return 'unknown_observe_only'
            expected_run_id = run_id if action == 'resume' else digest({
                name:body[name] for name in ('request_id', 'task_key', 'plan_sha256')})[:24]
            if (not isinstance(answer, dict) or answer.get('outcome') != 'accepted'
                    or answer.get('evidence_verified') is not True
                    or answer.get('run_id') != expected_run_id
                    or answer.get('task_key') != key
                    or answer.get('plan_sha256') != self.delegation['plan_sha256']
                    or answer.get('status') not in ('created', 'starting', 'running', 'completed', 'stopped')):
                intent.update(answer=answer, response_class='unresolved')
                save(intent_path, intent)
                return 'unknown_observe_only'
            intent.update(state='answered', answer=answer)
            save(intent_path, intent)
            return 'dispatched'
        return 'observe'

    @staticmethod
    def completed(status, task):
        total = task['config']['training']['total_steps']
        return bool(status and status.get('status') == 'completed'
                    and status.get('alive') is False
                    and status.get('progress', {}).get('env_step') == total
                    and status.get('progress', {}).get('update_step') == total - 1999)

    def validate_facts(self, key, facts):
        expected = self.delegation['identities'][str(self.plan['tasks'][key]['executor_port'])]
        for field in ('username', 'uid', 'boot_id', 'gpu_uuid', 'interpreter',
                      'owner', 'claim_path', 'workbench_pid', 'process_birth'):
            if facts.get(field) != expected[field]:
                raise ValueError('Changed identity requires root: ' + field)
        if facts['username'] != 'u2025140002':
            raise ValueError('Unauthorized username')
        for field in ('source_sha256', 'input_sha256', 'pause_contract_sha256',
                      'budget_contract_sha256'):
            if facts.get(field) != self.delegation[field]:
                raise ValueError('Contract bytes changed: ' + field)
        if self.delegation.get('new_human_pause_overrides') is not True:
            raise ValueError('New human pause always overrides delegation')
        age = (self.clock() - datetime.fromisoformat(facts['checked_UTC'].replace('Z', '+00:00'))).total_seconds()
        if not 0 <= age <= 60 or facts.get('deadline_utc') != self.deadline:
            raise ValueError('Fresh same-deadline ops evidence required')
        for field in ('source_verified', 'input_verified', 'exclusive_claim_verified',
                      'storage_peak_verified', 'shared_budget_verified',
                      'pause_clear', 'no_other_owner_conflict', 'lane_idle'):
            if facts.get(field) is not True:
                raise ValueError('Ops fact not passed: ' + field)
        storage = {}
        for field in ('storage_required_bytes_excluding_reserve', 'storage_reserve_bytes'):
            value = self.delegation.get(field)
            if type(value) is not int or value <= 0 or facts.get(field) != value:
                raise ValueError('Root-pinned positive storage bytes required: ' + field)
            storage[field] = value
        if storage['storage_reserve_bytes'] < 16 * 1024**3:
            raise ValueError('Root storage reserve below 16GiB')
        free_bytes = facts.get('free_bytes')
        if type(free_bytes) is not int or free_bytes < sum(storage.values()):
            raise ValueError('Insufficient peak plus reserve space')
        if self.plan['tasks'][key]['mode'] == 'train':
            gate = facts.get('root_engineering_acceptance', {})
            if (gate != self.delegation.get('root_engineering_acceptance')
                    or not gate.get('sha256') or gate.get('accepted') is not True
                    or gate.get('actual_gpu_engineering') is not True
                    or gate.get('throughput_accepted') is not True):
                raise ValueError('Separate root actual engineering/throughput gate required')
            for name in ('source_sha256', 'plan_sha256', 'data_sha256',
                         'numeric_validation_sha256', 'update_validation_sha256',
                         'resume_validation_sha256', 'full_loop_throughput_sha256'):
                if not gate.get(name):
                    raise ValueError('Missing root engineering evidence: ' + name)
            if (gate['source_sha256'] != self.delegation['source_sha256']
                    or gate['data_sha256'] != self.delegation['input_sha256']
                    or not self.delegation.get('engineering_plan_sha256')
                    or gate['plan_sha256'] != self.delegation['engineering_plan_sha256']):
                raise ValueError('Root engineering source/data/engineering plan differs')
