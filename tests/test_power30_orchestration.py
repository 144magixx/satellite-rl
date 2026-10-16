"""Pure fixtures: no agent, environment, PHY, remote call or optimizer."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from datetime import datetime, timezone
from satellite_rl.power30 import orchestration as module


DEADLINE = '2026-10-23T01:24:16.154153+00:00'


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.now = datetime(2026, 10, 16, 3, tzinfo=timezone.utc)
        self.tasks = {}
        for index, menu in enumerate(sorted(module.MENUS)):
            for group in ('M0', 'M1'):
                self.tasks[menu + group] = dict(mode='engineering', executor_port=(20655,20654,20653,20627)[index],
                    config=dict(menu=menu, group=group, source_sha='source',
                                deadline_utc=DEADLINE,
                                training=dict(total_steps=2048, seed=73030)))
        self.plan = dict(tasks=self.tasks, source_commit='source')
        self.identity = dict(username='u2025140002', uid=12, boot_id='boot',
                             gpu_uuid='gpu', interpreter='/DQN/python', owner='owner',
                             claim_path='/claim', workbench_pid=12, process_birth='birth')
        self.delegation = dict(schema='W30OpsDelegationV1', root_approved=True,
            controller_source_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
            new_human_pause_overrides=True, deadline_utc=DEADLINE,
            authority=dict(schema='W30HumanImplementationAuthorityV1', group_window_hours=168,
                           new_human_pause_overrides=True, group_deadline_UTC=DEADLINE),
            plan_sha256='raw-file-sha', plan_content_sha256=module.digest(self.plan), task_keys=list(self.tasks), mode='engineering',
            config_sha256={k:module.digest(t['config']) for k,t in self.tasks.items()},
            identities={str(t['executor_port']):self.identity for t in self.tasks.values()},
            source_sha256='source', input_sha256='input', pause_contract_sha256='pause',
            budget_contract_sha256='budget', storage_required_bytes_excluding_reserve=90*1024**3,
            storage_reserve_bytes=16*1024**3)
        self.facts = dict(self.identity, checked_UTC=self.now.isoformat(),
            deadline_utc=DEADLINE, source_sha256='source', input_sha256='input',
            pause_contract_sha256='pause', budget_contract_sha256='budget',
            free_bytes=106*1024**3, storage_required_bytes_excluding_reserve=90*1024**3,
            storage_reserve_bytes=16*1024**3)
        for field in ('source_verified','input_verified','exclusive_claim_verified',
                      'storage_peak_verified','shared_budget_verified','pause_clear',
                      'no_other_owner_conflict','lane_idle'):
            self.facts[field] = True
        self.statuses, self.calls, self.receipts = {}, [], []

    def controller(self, post=None):
        def write(receipt):
            self.receipts.append(receipt)
            return dict(path='/fixture/receipt', sha256=module.digest(receipt))
        def default_post(*args):
            self.calls.append(args)
            action, run_id, body = args
            task_key = body.get('task_key') or next(k for k,s in self.statuses.items() if s['run_id']==run_id)
            expected = run_id if action == 'resume' else module.digest({
                name:body[name] for name in ('request_id','task_key','plan_sha256')})[:24]
            return dict(status='starting', outcome='accepted', evidence_verified=True,
                        run_id=expected, task_key=task_key, plan_sha256=body['plan_sha256'])
        return module.Controller(self.temp.name, self.plan, self.delegation,
            module.digest(self.delegation), snapshot=lambda:self.statuses,
            probe=lambda key:copy.deepcopy(self.facts), write_receipt=write,
            post=post or default_post, clock=lambda:self.now)

    def status(self, key, state='completed', steps=2048, updates=49):
        return dict(status=state, alive=False, evidence_verified=True, run_id='same-id',
                    config_sha256=module.digest(self.tasks[key]['config']),
                    progress=dict(env_step=steps, update_step=updates))

    def test_fixed_m0_then_m1_and_fresh_receipts(self):
        controller = self.controller()
        for _ in range(4):
            self.assertEqual(controller.tick(), 'dispatched')
            key = self.calls[-1][2]['task_key']
            self.assertEqual(self.tasks[key]['config']['group'], 'M0')
            self.statuses[key] = self.status(key)
        self.assertEqual(controller.tick(), 'dispatched')
        self.assertEqual(self.tasks[self.calls[-1][2]['task_key']]['config']['group'], 'M1')
        self.assertEqual(len(self.receipts), 5)
        self.assertTrue(all(r['delegation_sha256'] == module.digest(self.delegation) for r in self.receipts))

    def test_two_gpu_busy_lane_skipped_then_complete_m0_wave(self):
        mapping = {'P00':20627, 'P11':20627, 'P10':20653, 'P01':20653}
        for task in self.tasks.values():
            task['executor_port'] = mapping[task['config']['menu']]
        self.delegation.update(plan_content_sha256=module.digest(self.plan),
            identities={str(port):self.identity for port in (20627,20653)})
        controller = self.controller()
        self.assertEqual(controller.tick(), 'dispatched')
        first = self.calls[-1][2]['task_key']
        self.statuses[first] = dict(self.status(first, 'running', 10, 0), alive=True)
        self.assertEqual(controller.tick(), 'dispatched')
        second = self.calls[-1][2]['task_key']
        self.assertNotEqual(mapping[self.tasks[first]['config']['menu']],
                            mapping[self.tasks[second]['config']['menu']])
        self.statuses[second] = dict(self.status(second, 'running', 10, 0), alive=True)
        self.assertEqual(controller.tick(), 'observe')
        self.statuses[first] = self.status(first)
        self.assertEqual(controller.tick(), 'dispatched')
        self.assertEqual(self.tasks[self.calls[-1][2]['task_key']]['config']['group'], 'M0')

    def test_unknown_post_never_repeated_even_if_snapshot_changes(self):
        def lost(*args):
            self.calls.append(args)
            raise TimeoutError()
        controller = self.controller(lost)
        self.assertEqual(controller.tick(), 'unknown_observe_only')
        key = self.calls[0][2]['task_key']
        self.statuses[key] = self.status(key, 'stopped', 2016, 17)
        self.assertEqual(controller.tick(), 'unknown_observe_only')
        self.assertEqual(len(self.calls), 1)

    def test_only_verified_engineering_stop_resumes_same_id_once(self):
        key = next(iter(self.tasks))
        self.statuses[key] = dict(self.status(key, 'stopped', 2016, 17),
            stop_reason='engineering_2016_recovery_probe', checkpoint_saved=True, ledger_closed=True)
        controller = self.controller()
        self.assertEqual(controller.tick(), 'dispatched')
        self.assertEqual(self.calls[0][0:2], ('resume', 'same-id'))
        controller.tick()
        self.assertEqual(sum(c[0]=='resume' for c in self.calls), 1)

    def test_changed_identity_pause_stale_space_budget_fail_closed(self):
        for field, value in [('uid',99), ('pause_clear',False), ('shared_budget_verified',False),
                             ('checked_UTC','2026-10-15T00:00:00Z'), ('free_bytes',0)]:
            with self.subTest(field=field):
                old = self.facts[field]
                self.facts[field] = value
                with self.assertRaises(ValueError):
                    self.controller().tick()
                self.facts[field] = old
        self.assertFalse(self.calls)

    def test_mutated_plan_and_unverified_snapshot_rejected(self):
        controller = self.controller()
        self.statuses[next(iter(self.tasks))] = dict(status='completed')
        with self.assertRaises(ValueError):
            controller.tick()
        self.statuses.clear()
        self.tasks[next(iter(self.tasks))]['config']['training']['seed'] = 51002
        with self.assertRaises(ValueError):
            controller.tick()

    def test_generic_clean_stop_is_not_engineering_probe(self):
        key = next(iter(self.tasks))
        self.statuses[key] = self.status(key, 'stopped', 2016, 17)
        controller = self.controller()
        controller.tick()
        self.assertTrue(all(call[0] != 'resume' for call in self.calls))

    def test_expired_authority_dispatches_nothing(self):
        self.now = datetime(2026, 10, 24, tzinfo=timezone.utc)
        self.assertEqual(self.controller().tick(), 'deadline')
        self.assertFalse(self.calls)

    def test_science_requires_pinned_complete_root_engineering_evidence(self):
        for task in self.tasks.values():
            task['mode']='train'
            task['config']['training']=dict(total_steps=150000, seed=51001)
        self.delegation.update(mode='train', plan_content_sha256=module.digest(self.plan),
            config_sha256={k:module.digest(t['config']) for k,t in self.tasks.items()})
        with self.assertRaises(ValueError):
            self.controller().tick()
        self.assertFalse(self.calls)

    def qualify_science_fixture(self):
        for task in self.tasks.values():
            task['mode'] = 'train'
            task['config']['training'] = dict(total_steps=150000, seed=51001)
        gate = dict(sha256='root-receipt', accepted=True, actual_gpu_engineering=True,
                    throughput_accepted=True, source_sha256='source', data_sha256='input',
                    plan_sha256='engineering-plan', numeric_validation_sha256='numeric',
                    update_validation_sha256='update', resume_validation_sha256='resume',
                    full_loop_throughput_sha256='throughput')
        self.delegation.update(mode='train', plan_content_sha256=module.digest(self.plan),
            config_sha256={k:module.digest(t['config']) for k,t in self.tasks.items()},
            engineering_plan_sha256='engineering-plan', root_engineering_acceptance=copy.deepcopy(gate))
        self.facts['root_engineering_acceptance'] = copy.deepcopy(gate)

    def test_science_valid_gate_and_wrong_source_data_engineering_plan(self):
        self.qualify_science_fixture()
        for field in ('source_sha256', 'data_sha256', 'plan_sha256'):
            with self.subTest(field=field):
                original = self.delegation['root_engineering_acceptance'][field]
                # Even if both pinned gate and observed gate agree, they must
                # bind this source/data and the separate engineering plan.
                self.delegation['root_engineering_acceptance'][field] = 'stale'
                self.facts['root_engineering_acceptance'][field] = 'stale'
                with self.assertRaises(ValueError):
                    self.controller().tick()
                self.delegation['root_engineering_acceptance'][field] = original
                self.facts['root_engineering_acceptance'][field] = original
        self.assertFalse(self.calls)
        self.assertEqual(self.controller().tick(), 'dispatched')

    def test_unknown_normal_post_responses_stay_durable_unresolved(self):
        for index, answer in enumerate((None, {}, {'status':'unknown'},
                                        {'status':500}, {'status':'starting'})):
            with self.subTest(answer=answer):
                def post(*args):
                    self.calls.append(args)
                    return answer
                controller = self.controller(post)
                controller.directory = Path(self.temp.name) / str(index)
                before = len(self.calls)
                self.assertEqual(controller.tick(), 'unknown_observe_only')
                self.assertEqual(controller.tick(), 'unknown_observe_only')
                self.assertEqual(len(self.calls), before+1)

    def test_authority_deadline_precision_and_chinese_digest(self):
        from satellite_rl.power30.logging import content_sha
        self.assertEqual(module.digest({'中文':'配置'}), content_sha({'中文':'配置'}))
        controller = self.controller()
        self.assertEqual(controller.deadline, DEADLINE)
        self.assertEqual(controller.tick(), 'dispatched')
        self.assertEqual(self.receipts[0]['deadline_utc'], DEADLINE)
        # Time-equivalent text cannot override authority's exact bytes.
        self.delegation['deadline_utc'] = '2026-10-23T01:24:16.154153Z'
        with self.assertRaises(ValueError):
            self.controller()

    def test_storage_root_bound_exact_boundary_missing_and_insufficient(self):
        controller = self.controller()
        key = next(iter(self.tasks))
        # Frozen 106GiB is 90GiB workload plus 16GiB reserve, counted once.
        controller.validate_facts(key, self.facts)
        for field in ('storage_required_bytes_excluding_reserve', 'storage_reserve_bytes'):
            for value in (None, 0, -1, True):
                with self.subTest(field=field, value=value):
                    facts = copy.deepcopy(self.facts)
                    if value is None:
                        facts.pop(field)
                    else:
                        facts[field] = value
                    with self.assertRaises(ValueError):
                        controller.validate_facts(key, facts)
            original = self.delegation.pop(field)
            with self.assertRaises(ValueError):
                self.controller().validate_facts(key, self.facts)
            self.delegation[field] = original
        facts = dict(self.facts, free_bytes=106*1024**3-1)
        with self.assertRaises(ValueError):
            controller.validate_facts(key, facts)


if __name__ == '__main__':
    unittest.main()
