"""Adapter qualification with file fixtures and fake transport only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from satellite_rl.power30 import orchestration_adapter as adapter
from satellite_rl.power30.orchestration import digest, save


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def ref(self, name, value):
        path = self.root / name
        save(path, value)
        return dict(path=str(path), sha256=adapter.file_sha(path))

    def local_adapter(self):
        self.plan = dict(source_commit='source', tasks={'arm':dict(executor_port=20627)})
        plan_ref = self.ref('plan.json', self.plan)
        authority = dict(group_deadline_UTC='2026-10-23T01:24:16.154153+00:00')
        budget_ref = self.ref('budget-contract.json', {})
        delegation = dict(authority=authority, plan_sha256=plan_ref['sha256'],
                          budget_contract_sha256=budget_ref['sha256'])
        pins = dict(adapter_source_sha256=adapter.file_sha(adapter.__file__),
            controller_directory=str(self.root/'controller'),
            plan_ref=plan_ref, authority_ref=self.ref('authority.json', authority),
            budget_contract_ref=budget_ref,
            delegation_ref=self.ref('delegation.json', delegation),
            endpoints={'20627':dict(host='10.160.17.102', port=20627)})
        class FakeBridge:
            def __init__(bridge, endpoint):
                bridge.answer = None
                bridge.requests = []
            def close(bridge): pass
            def http(bridge, *args):
                bridge.requests.append(args)
                return bridge.answer
            def inspect(bridge, operation):
                return {}
        return adapter.Adapter(pins, bridge_factory=FakeBridge)

    def test_http10_body_survives_channel_until_consumed(self):
        import io
        payload=b'{"group":"S2-W30","runs":[]}'
        class Channel:
            def __init__(self):
                self.stream=io.BytesIO(b'HTTP/1.0 200 OK\r\nContent-Length: '+str(len(payload)).encode()+b'\r\n\r\n'+payload)
                self.sent=b'';self.closed=False
            def settimeout(self,value):pass
            def sendall(self,value):self.sent+=value
            def makefile(self,*args):return self.stream
            def close(self):self.closed=True;self.stream.close()
        for method,body in [('GET',None),('POST',{'key':'value'})]:
            channel=Channel()
            bridge=object.__new__(adapter.SSHBridge)
            bridge.endpoint=dict(api_port=9120)
            bridge.transport=type('Transport',(),dict(open_channel=lambda *a,**kw:channel))()
            self.assertEqual(bridge.http(method,'/api/status',body),{'group':'S2-W30','runs':[]})
            self.assertTrue(channel.closed)
            self.assertTrue(channel.sent.startswith((method+' /api/status HTTP/1.0').encode()))
            self.assertIn(b'Content-Type: application/json\r\n',channel.sent)
            if body is not None:self.assertTrue(channel.sent.endswith(b'{"key": "value"}'))

    def test_raw_pin_change_and_authority_mismatch_rejected(self):
        ref = self.ref('pinned.json', {'a':1})
        self.assertEqual(adapter.read_ref(ref), {'a':1})
        save(ref['path'], {'a':2})
        with self.assertRaises(ValueError):
            adapter.read_ref(ref)
        instance = self.local_adapter()
        pins = copy.deepcopy(instance.pins)
        pins['authority_ref'] = self.ref('changed-authority.json', {})
        with self.assertRaises(ValueError):
            adapter.Adapter(pins)

    def test_post_ack_normalization_once_and_unknown_response(self):
        instance = self.local_adapter()
        bridge = instance.bridge(20627)
        body = dict(request_id='request', task_key='arm', plan_sha256=instance.pins['plan_ref']['sha256'],
                    ops_admission_ref=dict(path='/ops', sha256='ops'))
        run_id = digest({key:body[key] for key in ('request_id','task_key','plan_sha256')})[:24]
        bridge.answer = dict(run_id=run_id, task_key='arm', source_commit='source',
                             plan_sha256=body['plan_sha256'], status='starting')
        self.assertEqual(instance.post('submit', None, body)['outcome'], 'accepted')
        self.assertEqual(len(bridge.requests), 1)
        for answer in (None, {}, {'http_status':500}, dict(bridge.answer, source_commit='stale')):
            bridge.answer = answer
            self.assertEqual(instance.post('submit', None, body)['outcome'], 'unknown')
        instance.last_statuses = {'arm':dict(run_id=run_id)}
        bridge.answer = dict(run_id=run_id, task_key='arm', source_commit='source',
                             plan_sha256=body['plan_sha256'], status='starting')
        resume = dict(plan_sha256=body['plan_sha256'], ops_admission_ref=body['ops_admission_ref'])
        self.assertEqual(instance.post('resume', run_id, resume)['run_id'], run_id)
        self.assertEqual(bridge.requests[-1][1], '/api/runs/' + run_id + '/resume')

    def test_singleton_is_exclusive_and_never_steals_stale_directory(self):
        first = adapter.Singleton(self.root / 'controller')
        with self.assertRaises(FileExistsError):
            adapter.Singleton(self.root / 'controller')
        first.close()
        stale = self.root / 'controller' / 'singleton'
        stale.mkdir()
        save(stale / 'owner.json', dict(pid=-1, boot_id='old', process_birth=0))
        with self.assertRaises(FileExistsError):
            adapter.Singleton(self.root / 'controller')

    def test_completed_engineering_cap_does_not_block_scientific_budget(self):
        limits = dict(engineering_interaction=16384, engineering_update=392,
                      scientific_interaction=3600000, scientific_update=3552024)
        contract = dict(limits=limits, first_round_limits=dict(scientific_interaction=1200000),
                        authority_ref={'path':'authority','sha256':'sha'}, minimum_charged={},
                        minimum_charged_seconds={}, time_limits=dict(engineering=28800, scientific=576000))
        budget = dict(experiment_id='S2-W30', first_round=True, authority_ref=contract['authority_ref'],
                      limits=limits, charged=dict(engineering_interaction=16384, engineering_update=392))
        adapter.verify_budget(budget, contract, 'scientific')
        with self.assertRaises(ValueError):
            adapter.verify_budget(budget, contract, 'engineering')
        budget['charged']['engineering_update'] += 1
        with self.assertRaises(ValueError):
            adapter.verify_budget(budget, contract, 'scientific')

    def status_fixture(self):
        run_id = 'a'*24
        workbench = self.root / 'workbench'
        directory = workbench / 'runs' / run_id
        model_dir = workbench / 'models' / run_id
        config = dict(ledger_root=str(self.root / 'ledger'), pause_markers=[])
        actual = dict(config, run_id=run_id, model_dir=str(model_dir))
        plan = dict(source_commit='source', tasks={'arm':dict(mode='engineering', executor_port=20627, config=config)})
        status = dict(task_key='arm', run_id=run_id, status='stopped', plan_sha256='plan',
                      pid=123,process_birth=456,boot_id='boot',
                      source_commit='source', request_sha256='request', progress=dict(env_step=2016, update_step=17))
        save(directory / 'status.json', status)
        save(directory / 'config.json', actual)
        save(directory / 'job.json', dict(config_sha256=digest(actual)))
        save(workbench / 'task_owners' / 'arm.json', dict(run_id=run_id, request_sha256='request'))
        model_dir.mkdir(parents=True)
        (model_dir / 'current.pt').write_bytes(b'fixture only; never torch loaded')
        save(model_dir / 'resume_index.json', dict(env_step=2016,current_SHA=adapter.file_sha(model_dir / 'current.pt')))
        save(Path(config['ledger_root']) / 'workers' / (run_id+'.json'), dict(run_id=run_id,open=False,
              actual=dict(engineering_interaction=2016, engineering_update=17)))
        pins = dict(workbench_root=str(workbench), plan_ref=dict(sha256='plan'), executor_port=20627)
        return pins, plan, directory

    def test_original_config_probe_checkpoint_closed_ledger_and_no_stop(self):
        pins, plan, directory = self.status_fixture()
        with patch.object(adapter,'verify_service'), patch.object(adapter,'linux_alive',return_value=False), \
                patch.object(adapter,'linux_boot',return_value='boot'), \
                patch.object(adapter,'linux_process_identity',return_value=dict(pid=123,state='Z',process_birth='456')):
            result = adapter.inspect_statuses(pins, plan)['arm']
            self.assertTrue(result['checkpoint_saved'])
            self.assertEqual(result['stop_reason'], 'engineering_2016_recovery_probe')
            self.assertEqual(result['closed_ledger']['actual']['engineering_update'],17)
            self.assertEqual(result['resume_index']['env_step'],2016)
            self.assertEqual(result['process_exit_evidence']['observed_process']['state'],'Z')
            (directory / 'stop.request').write_text('human stop')
            self.assertFalse(adapter.inspect_statuses(pins,plan)['arm']['checkpoint_saved'])
            save(directory / 'config.json', {})
            with self.assertRaises(ValueError):
                adapter.inspect_statuses(pins,plan)

    def test_verified_snapshot_durable_before_resume_and_survives_changed_worker(self):
        instance=self.local_adapter()
        record=dict(status='stopped',ledger_closed=True,resume_index=dict(env_step=2016,current_SHA='checkpoint'),
                    closed_ledger=dict(open=False,actual=dict(engineering_update=17,engineering_interaction=2016)),
                    process_exit_evidence=dict(pid=123,recorded_process_birth=456,recorded_boot_id='boot',recorded_owner_live=False))
        bridge=instance.bridge(20627)
        bridge.inspect=lambda operation:{'arm':record}
        self.assertEqual(instance.snapshot()['arm']['status'],'stopped')
        paths=list((self.root/'controller'/'snapshots').glob('*.json'))
        self.assertEqual(len(paths),1)
        saved=json.loads(paths[0].read_text(encoding='utf8'))
        self.assertEqual(saved['schema'],'W30VerifiedWorkbenchSnapshotV1')
        self.assertEqual(saved['plan_sha256'],instance.pins['plan_ref']['sha256'])
        record['closed_ledger']['open']=True
        record['resume_index']['env_step']=2048
        self.assertFalse(saved['statuses']['arm']['closed_ledger']['open'])
        self.assertEqual(saved['statuses']['arm']['resume_index']['env_step'],2016)

    def test_health_poll_default_and_root_bounded_science_cadence(self):
        self.assertEqual(adapter.health_poll_seconds({}),60)
        self.assertEqual(adapter.health_poll_seconds(dict(poll_seconds=3600)),3600)
        for value in (4,3601,True,0,-1):
            with self.assertRaises(ValueError):adapter.health_poll_seconds(dict(poll_seconds=value))

    def test_same_birth_live_owner_cannot_be_called_exited(self):
        pins,plan,_=self.status_fixture()
        with patch.object(adapter,'verify_service'), patch.object(adapter,'linux_alive',return_value=False), \
                patch.object(adapter,'linux_boot',return_value='boot'), \
                patch.object(adapter,'linux_process_identity',return_value=dict(pid=123,state='R',process_birth='456')):
            with self.assertRaises(ValueError):adapter.inspect_statuses(pins,plan)

    def test_ops_clock_root_precharge_real_active_and_unknown_reservation(self):
        directory = str(self.root / 'clock')
        contract = dict(controller_ops_active_seconds=1800, prior_CPU_engineering_seconds=10.454249300004449,
                        minimum_charged_seconds=dict(engineering=1810.454249300004449))
        budget_ref = self.ref('clock-budget.json', contract)
        receipt = dict(root_approved=True, passed=True, controller_ops_active_seconds=1800,
                       controller_directory=directory, budget_contract_sha256=budget_ref['sha256'])
        pins = dict(controller_directory=directory, controller_ops_active_seconds=1800,
                    max_pass_seconds=120, budget_contract_ref=budget_ref,
                    ops_precharge_ref=self.ref('precharge.json', receipt))
        times = iter((10., 12.5, 20.))
        clock = adapter.OpsClock(pins, clock=lambda:next(times))
        self.assertTrue(clock.begin())
        clock.finish()
        self.assertEqual(clock.value['charged_seconds'], 2.5)
        self.assertEqual(clock.value['actual_seconds'], 2.5)
        self.assertTrue(clock.begin())
        self.assertEqual(clock.value['charged_seconds'], 132.5)
        with self.assertRaises(ValueError):
            adapter.OpsClock(pins)
        pins['controller_ops_active_seconds'] = 1900
        with self.assertRaises(ValueError):
            adapter.OpsClock(pins)

    def test_final_root_launch_admission_cannot_be_omitted_or_repurposed(self):
        delegation = dict(mode='engineering', controller_source_sha256='controller')
        pins = dict(delegation_ref=self.ref('launch-delegation.json',delegation),
                    adapter_source_sha256='adapter', launcher_source_sha256='launcher',
                    plan_ref=dict(path='plan',sha256='plan-sha'),
                    ops_precharge_ref=dict(path='precharge',sha256='precharge-sha'))
        accepted = dict(root_approved=True, passed=True, phase='engineering',
                        plan_sha256='plan-sha',adapter_source_sha256='adapter',
                        controller_source_sha256='controller',launcher_source_sha256='launcher',
                        ops_precharge_ref=pins['ops_precharge_ref'])
        for index, change in enumerate((dict(root_approved=False),dict(phase='train'),
                                       dict(plan_sha256='other'),dict(ops_precharge_ref={}))) :
            pins['controller_launch_admission_ref'] = self.ref('launch-denied-%s.json'%index,dict(accepted,**change))
            with self.assertRaises(ValueError):
                adapter.verify_launch_admission(pins)
        pins['controller_launch_admission_ref'] = self.ref('launch-accepted.json',accepted)
        adapter.verify_launch_admission(pins)

    def test_hidden_launcher_embedded_inspection_is_valid_python(self):
        import ast
        source = (Path(__file__).parents[1]/'scripts'/'power30'/'start_controller.ps1').read_text(encoding='utf8')
        line = next(line for line in source.splitlines() if 'p=psutil.Process' in line)
        ast.parse(line.split("'")[1])
        self.assertIn('-WindowStyle Hidden',source)
        self.assertNotIn('BUPT_PASSWORD=',source)


if __name__ == '__main__':
    unittest.main()
