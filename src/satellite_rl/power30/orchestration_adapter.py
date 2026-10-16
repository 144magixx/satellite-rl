"""Root-pinned W30 SSH/SFTP/loopback adapter and bounded persistent controller.

No connection or scientific call occurs on import. Root creates a separate pins
file and passes its raw SHA as a trusted launch argument after independent review.
Engineering and scientific generations have distinct pins/delegations/plans.
Never use an unpinned file's root_approved field as authentication.
"""
import argparse
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from .orchestration import Controller, digest, save


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_ref(ref):
    if file_sha(ref['path']) != ref['sha256']:
        raise ValueError('Pinned original bytes changed: ' + ref['path'])
    return json.loads(Path(ref['path']).read_text(encoding='utf8'))


def health_poll_seconds(pins):
    value = pins.get('poll_seconds',60)
    if type(value) is not int or not 5 <= value <= 3600:
        raise ValueError('Root-pinned 5..3600 second health cadence required')
    return value


def next_health_delay_seconds(result,poll_seconds):
    return min(poll_seconds,10) if result=='dispatched' else poll_seconds


def local_boot_matches(recorded, observed):
    # Windows psutil boot_time is estimated independently in each process.
    # PID and process create_time remain exact; remote Linux UUID is untouched.
    if os.name != 'nt':
        return recorded == observed
    try:
        if not recorded.startswith('boot-time:') or not observed.startswith('boot-time:'):
            return False
        old, new = float(recorded[10:]), float(observed[10:])
        return math.isfinite(old) and math.isfinite(new) and abs(old-new) <= 5
    except (ValueError,TypeError,AttributeError):
        return False


def verify_single_pass_parent(pins):
    import psutil
    owner = json.loads((Path(pins['controller_directory']) / 'singleton' / 'owner.json').read_text())
    pending = json.loads((Path(pins['controller_directory']) / 'ops_clock.json').read_text()).get('pending')
    if (not pending or owner['pid'] != os.getppid()
            or owner['process_birth'] != psutil.Process(os.getppid()).create_time()
            or not local_boot_matches(owner['boot_id'],'boot-time:' + str(psutil.boot_time()))):
        raise ValueError('Single pass requires live singleton parent and reserved ops time')


def persist_child_outputs(directory, stamp, stdout, stderr):
    refs = {}
    for name, value in (('stdout',stdout),('stderr',stderr)):
        path = Path(directory) / 'pass_outputs' / (stamp + '.' + name + '.txt')
        path.parent.mkdir(parents=True,exist_ok=True)
        payload = value if isinstance(value,bytes) else (value or '').encode('utf8')
        with path.open('wb') as stream:
            stream.write(payload);stream.flush();os.fsync(stream.fileno())
        refs[name] = dict(path=str(path),sha256=hashlib.sha256(payload).hexdigest())
    return refs


def recursive_refs(value):
    if isinstance(value, dict):
        if set(value) == {'path', 'sha256'}:
            yield value
        else:
            for child in value.values():
                yield from recursive_refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from recursive_refs(child)


def linux_birth(pid):
    try:
        text = Path('/proc/%s/stat' % int(pid)).read_text()
        tail = text[text.rfind(')') + 2:].split()
        return None if tail[0] in ('Z', 'X') else tail[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def linux_boot():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def linux_process_identity(pid):
    try:
        text = Path('/proc/%s/stat' % int(pid)).read_text()
        tail = text[text.rfind(')') + 2:].split()
        return dict(pid=int(pid), state=tail[0], process_birth=tail[19])
    except (FileNotFoundError, ProcessLookupError):
        return None


def linux_alive(status):
    return (status.get('boot_id') == linux_boot()
            and status.get('process_birth') is not None
            and str(status['process_birth']) == linux_birth(status.get('pid', -1)))


def run_output(arguments):
    return subprocess.run(arguments, check=True, capture_output=True, text=True,
                          timeout=15).stdout.strip()


def verify_service(pins):
    identity = pins['identity']
    service = json.loads((Path(pins['workbench_root']) / 'server_owner.json').read_text())
    claim = json.loads(Path(identity['claim_path']).read_text())
    if (not linux_alive(service) or service['pid'] != identity['workbench_pid']
            or str(service['process_birth']) != str(identity['process_birth'])
            or service['boot_id'] != identity['boot_id']):
        raise ValueError('Root-pinned workbench birth/boot changed')
    for field, expected in dict(workbench_pid=service['pid'], process_birth=service['process_birth'],
                                boot_id=service['boot_id'], gpu_uuid=identity['gpu_uuid']).items():
        if str(claim.get(field)) != str(expected):
            raise ValueError('GPU claim ownership changed: ' + field)
    # Read actual proc executable and command line rather than trusting JSON.
    pid = service['pid']
    if os.path.realpath('/proc/%s/exe' % pid) != os.path.realpath(identity['interpreter']):
        raise ValueError('Workbench actual interpreter changed')
    command = Path('/proc/%s/cmdline' % pid).read_bytes().split(b'\0')
    if (b'satellite_rl.power30.workbench' not in command
            or os.fsencode(pins['plan_ref']['path']) not in command):
        raise ValueError('Workbench command/plan differs')
    if Path('/proc/%s' % pid).stat().st_uid != identity['uid']:
        raise ValueError('Workbench UID differs')
    return service


def verify_budget(budget, contract, phase):
    if (budget.get('experiment_id') != 'S2-W30' or budget.get('limits') != contract['limits']
            or budget.get('first_round') is not True
            or budget.get('authority_ref') != contract['authority_ref']):
        raise ValueError('Shared permanent budget contract differs')
    for kind, count in budget['charged'].items():
        cap = contract['first_round_limits'].get(kind, contract['limits'][kind])
        if type(count) is not int or not 0 <= count <= cap:
            raise ValueError('Shared call budget exhausted/invalid')
    for kind, floor in contract['minimum_charged'].items():
        if budget['charged'].get(kind, 0) < floor:
            raise ValueError('Prior CPU/unknown debit missing')
    for name, count in budget.get('charged_seconds', {}).items():
        if not 0 <= count <= contract['time_limits'][name]:
            raise ValueError('Shared time budget exhausted/invalid')
    for name, floor in contract['minimum_charged_seconds'].items():
        if budget.get('charged_seconds', {}).get(name, 0) < floor:
            raise ValueError('Prior CPU time debit missing')
    needed = ('engineering_interaction', 'engineering_update') if phase == 'engineering' else ('scientific_interaction', 'scientific_update')
    for kind in needed:
        cap = contract['first_round_limits'].get(kind, contract['limits'][kind])
        if budget['charged'].get(kind, 0) >= cap:
            raise ValueError('Intended phase call budget exhausted')
    if budget.get('charged_seconds', {}).get(phase, 0) >= contract['time_limits'][phase]:
        raise ValueError('Intended phase time budget exhausted')


def inspect_statuses(pins, plan):
    """Read original job/config/status/checkpoint-index/ledger; no torch load."""
    verify_service(pins)
    root = Path(pins['workbench_root'])
    statuses = {}
    for path in sorted((root / 'runs').glob('*/status.json')):
        status = json.loads(path.read_text())
        key = status['task_key']
        if key not in plan['tasks'] or key in statuses:
            raise ValueError('Unexpected or duplicate task owner')
        task = plan['tasks'][key]
        if task['executor_port'] != pins['executor_port']:
            raise ValueError('Foreign lane run')
        run_id = status['run_id']
        if path.parent.name != run_id or len(run_id) != 24:
            raise ValueError('Run directory identity differs')
        config = json.loads((path.parent / 'config.json').read_text())
        expected = dict(task['config'], run_id=run_id, model_dir=str(root / 'models' / run_id))
        job = json.loads((path.parent / 'job.json').read_text())
        if config != expected or digest(config) != job['config_sha256']:
            raise ValueError('Actual run config/job/plan differs')
        if (status['plan_sha256'] != pins['plan_ref']['sha256']
                or status['source_commit'] != plan['source_commit']):
            raise ValueError('Run source/plan binding differs')
        owner = json.loads((root / 'task_owners' / (key + '.json')).read_text())
        if owner['run_id'] != run_id or owner['request_sha256'] != status['request_sha256']:
            raise ValueError('Task owner differs')
        status = dict(status, evidence_verified=True, alive=linux_alive(status),
                      config_sha256=digest(task['config']))
        if status['status'] in ('completed', 'stopped') and not status['alive']:
            progress = status['progress']
            ledger_path = Path(config['ledger_root']) / 'workers' / (run_id + '.json')
            ledger_bytes = ledger_path.read_bytes()
            ledger = json.loads(ledger_bytes)
            status['ledger_closed'] = ledger['open'] is False and ledger['run_id'] == run_id
            if not status['ledger_closed']:
                raise ValueError('Exited clean status has open ledger')
            status['closed_ledger'] = ledger
            status['closed_ledger_ref'] = dict(path=str(ledger_path), sha256=hashlib.sha256(ledger_bytes).hexdigest())
            observed = linux_process_identity(status['pid'])
            owner_live = bool(observed and str(observed['process_birth']) == str(status['process_birth'])
                              and observed['state'] not in ('Z','X'))
            if status['boot_id'] != linux_boot() or owner_live:
                raise ValueError('Original owner exit/boot not verified at evidence barrier')
            status['process_exit_evidence'] = dict(pid=status['pid'],
                recorded_process_birth=status['process_birth'], recorded_boot_id=status['boot_id'],
                actual_boot_id=linux_boot(), observed_process=observed, recorded_owner_live=False,
                checked_UTC=datetime.now(timezone.utc).isoformat())
            if task['mode'] == 'engineering' and status['status'] == 'stopped' and progress.get('env_step') == 2016:
                index_path = Path(config['model_dir']) / 'resume_index.json'
                index_bytes = index_path.read_bytes()
                index = json.loads(index_bytes)
                status['resume_index'] = index
                status['resume_index_ref'] = dict(path=str(index_path),sha256=hashlib.sha256(index_bytes).hexdigest())
                checkpoint = Path(config['model_dir']) / 'current.pt'
                # Frozen runner checks this exact branch before user stop and
                # saves engineering_recovery_probe_done in the hashed checkpoint.
                # Check index, closed ledger and original stop controls too.
                saved = (index['env_step'] == 2016 and index['current_SHA'] == file_sha(checkpoint)
                         and progress.get('update_step') == 17
                         and ledger['actual'].get('engineering_interaction') == 2016
                         and ledger['actual'].get('engineering_update') == 17
                         and not (path.parent / 'stop.request').exists()
                         and not any(Path(p).exists() for p in config['pause_markers']))
                status['checkpoint_saved'] = saved
                if saved:
                    status['checkpoint_ref'] = dict(path=str(checkpoint),sha256=index['current_SHA'])
                    status['stop_reason'] = 'engineering_2016_recovery_probe'
        statuses[key] = status
    return statuses


def inspect_probe(pins, plan, task_key):
    """Fresh full dispatch qualification. Immutable originals are SHA checked."""
    import pwd
    import shutil
    identity = pins['identity']
    if (os.getuid() != identity['uid'] or pwd.getpwuid(os.getuid()).pw_name != 'u2025140002'
            or linux_boot() != identity['boot_id']
            or os.path.realpath(sys.executable) != os.path.realpath(identity['interpreter'])):
        raise ValueError('Actual authenticated helper identity differs')
    verify_service(pins)
    source = Path(plan['source_root']).resolve()
    actual_members = {p.relative_to(source).as_posix() for p in (source / 'src').rglob('*.py')}
    if not actual_members.issubset(plan['source_manifest']):
        raise ValueError('Unfrozen source member')
    for relative, expected in plan['source_manifest'].items():
        path = (source / relative).resolve()
        if not path.is_relative_to(source) or file_sha(path) != expected:
            raise ValueError('Frozen source changed')
    config = plan['tasks'][task_key]['config']
    manifest = read_ref(config['data_manifest_ref'])
    if config['data_manifest_ref']['sha256'] != config['data_manifest_sha']:
        raise ValueError('Data manifest identity differs')
    refs = {ref['path']:ref for ref in recursive_refs(manifest)}
    for ref in refs.values():
        read_ref(ref)
    contracts = {name:read_ref(ref) for name, ref in pins['contract_refs'].items()}
    for name in ('source', 'data', 'execution'):
        receipt = contracts[name]
        if (receipt.get('experiment_id') != 'S2-W30' or receipt.get('root_approved') is not True
                or receipt.get('passed') is not True):
            raise ValueError('Actual root admission missing: ' + name)
    d = pins['delegation']
    if (pins['contract_refs']['source']['sha256'] != d['source_sha256']
            or contracts['source'].get('source_commit') != plan['source_commit']
            or contracts['source'].get('source_manifest_sha256') != digest(plan['source_manifest'])
            or contracts['data'].get('data_manifest_sha') != config['data_manifest_sha']
            or contracts['execution'].get('phase') != config['phase']
            or contracts['execution'].get('source_sha') != plan['source_commit']
            or contracts['authority'] != d['authority']
            or config['deadline_utc'] != d['authority']['group_deadline_UTC']):
        raise ValueError('Original source/data/execution/authority binding differs')
    statuses = inspect_statuses(pins, plan)
    if any(s['alive'] or s['status'] in ('created', 'starting', 'running', 'stop_requested')
           for s in statuses.values()):
        raise ValueError('GPU lane is busy')
    uuids = run_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader']).splitlines()
    if uuids != [identity['gpu_uuid']]:
        raise ValueError('Expected one visible root-pinned GPU')
    processes = run_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'])
    if processes.strip():
        raise ValueError('Unowned GPU compute process conflict')
    budget_contract = read_ref(pins['contract_refs']['budget'])
    budget = json.loads((Path(config['ledger_root']) / 'budget.json').read_text())
    verify_budget(budget, budget_contract, config['phase'])
    if any(Path(p).exists() for p in config['pause_markers']):
        raise ValueError('New pause overrides controller')
    if digest(config['pause_markers']) != d['pause_contract_sha256']:
        raise ValueError('Pause paths differ')
    facts = dict(identity, checked_UTC=datetime.now(timezone.utc).isoformat(),
                 executor_port=pins['executor_port'], deadline_utc=config['deadline_utc'],
                 source_sha256=d['source_sha256'], input_sha256=config['data_manifest_sha'],
                 pause_contract_sha256=digest(config['pause_markers']),
                 budget_contract_sha256=pins['contract_refs']['budget']['sha256'],
                 free_bytes=shutil.disk_usage(pins['workbench_root']).free)
    for field in ('storage_required_bytes_excluding_reserve', 'storage_reserve_bytes'):
        facts[field] = d[field]
    for field in ('source_verified', 'input_verified', 'exclusive_claim_verified',
                  'storage_peak_verified', 'shared_budget_verified', 'pause_clear',
                  'no_other_owner_conflict', 'lane_idle'):
        facts[field] = True
    if d['mode'] == 'train':
        gate = read_ref(pins['contract_refs']['engineering'])
        if gate != d['root_engineering_acceptance']:
            raise ValueError('Actual root engineering acceptance differs')
        facts['root_engineering_acceptance'] = gate
    return facts


class SSHBridge:
    """Short authenticated sessions, host fingerprint pinned BEFORE password auth."""
    def __init__(self, endpoint):
        import paramiko
        self.endpoint = endpoint
        transport = paramiko.Transport(socket.create_connection((endpoint['host'], endpoint['port']), 15))
        try:
            transport.start_client(timeout=15)
            fingerprint = hashlib.sha256(transport.get_remote_server_key().asbytes()).hexdigest()
            if fingerprint != endpoint['host_key_sha256']:
                raise ValueError('SSH server identity changed')
            transport.auth_password('u2025140002', os.environ['BUPT_PASSWORD'])
            if not transport.is_authenticated():
                raise ValueError('SSH authentication failed')
            self.transport = transport
            self.sftp = paramiko.SFTPClient.from_transport(transport)
        except BaseException:
            transport.close()
            raise

    def close(self):
        self.sftp.close()
        self.transport.close()

    def read(self, path):
        with self.sftp.open(path, 'rb') as stream:
            return stream.read()

    def inspect(self, operation, task_key=None):
        endpoint = self.endpoint
        # Before executing even the helper, verify its actual bytes by SFTP.
        if hashlib.sha256(self.read(endpoint['helper_path'])).hexdigest() != endpoint['helper_sha256']:
            raise ValueError('Remote helper source changed')
        arguments = [endpoint['interpreter'], '-m', 'satellite_rl.power30.orchestration_adapter',
                     '--remote-pins', endpoint['remote_pins_ref']['path'],
                     '--remote-pins-sha', endpoint['remote_pins_ref']['sha256'], '--operation', operation]
        if task_key is not None:
            arguments += ['--task-key', task_key]
        command = 'env PYTHONPATH=' + shlex.quote(endpoint['source_root'] + '/src') + ' ' + shlex.join(arguments)
        channel = self.transport.open_session(timeout=15)
        channel.settimeout(60)
        try:
            channel.exec_command(command)
            output = channel.makefile('rb').read(4 * 1024 * 1024)
            error = channel.makefile_stderr('rb').read(16384)
            if channel.recv_exit_status() != 0:
                raise ValueError('Read-only helper rejected: ' + error.decode('utf8', 'replace')[-1500:])
            return json.loads(output)
        finally:
            channel.close()

    def upload_receipt(self, receipt):
        payload = json.dumps(receipt, ensure_ascii=False, sort_keys=True, allow_nan=False).encode('utf8')
        checksum = hashlib.sha256(payload).hexdigest()
        path = self.endpoint['receipt_directory'] + '/' + checksum + '.json'
        temporary = path + '.upload'
        with self.sftp.open(temporary, 'wx') as stream:
            stream.write(payload)
            stream.flush()
        if hashlib.sha256(self.read(temporary)).hexdigest() != checksum:
            raise ValueError('Uploaded ops receipt differs')
        self.sftp.rename(temporary, path)
        return dict(path=path, sha256=checksum)

    def http(self, method, path, body=None):
        port = self.endpoint['api_port']
        channel = self.transport.open_channel('direct-tcpip', ('127.0.0.1', port), ('127.0.0.1', 0), timeout=15)
        channel.settimeout(25)
        response = None
        try:
            payload = b'' if body is None else json.dumps(body, ensure_ascii=False).encode('utf8')
            # HTTPConnection closes its socket early for HTTP/1.0 responses.
            # Paramiko makefile has no real-socket ownership reference counting:
            # retain the channel ourselves until HTTPResponse consumed its body.
            header = ('%s %s HTTP/1.0\r\nHost: 127.0.0.1:%s\r\n'
                      'Content-Type: application/json\r\nContent-Length: %s\r\n'
                      'Connection: close\r\n\r\n') % (method,path,port,len(payload))
            channel.sendall(header.encode('ascii') + payload)
            response = http.client.HTTPResponse(channel,method=method)
            response.begin()
            data = response.read(4 * 1024 * 1024)
            if response.status != 200:
                return dict(http_status=response.status, raw=data.decode('utf8', 'replace')[:2000])
            return json.loads(data)
        finally:
            if response is not None:
                response.close()
            channel.close()


class Adapter:
    def __init__(self, pins, *, bridge_factory=SSHBridge):
        self.pins = pins
        self.plan = read_ref(pins['plan_ref'])
        self.delegation = read_ref(pins['delegation_ref'])
        self.bridge_factory = bridge_factory
        for text_port, endpoint in pins['endpoints'].items():
            if (int(text_port) not in (20655,20654,20653,20627)
                    or endpoint['port'] != int(text_port) or endpoint['host'] != '10.160.17.102'):
                raise ValueError('Only explicitly authorized W30 host/ports')
        if pins['adapter_source_sha256'] != file_sha(__file__):
            raise ValueError('Local adapter source changed')
        if self.delegation['plan_sha256'] != pins['plan_ref']['sha256']:
            raise ValueError('Raw plan SHA differs')
        if pins['budget_contract_ref']['sha256'] != self.delegation['budget_contract_sha256']:
            raise ValueError('Controller/shared budget contract differs')
        authority = read_ref(pins['authority_ref'])
        if self.delegation['authority'] != authority:
            raise ValueError('Pinned human authority differs')
        self.bridges = {}

    def bridge(self, port):
        if port not in self.bridges:
            self.bridges[port] = self.bridge_factory(self.pins['endpoints'][str(port)])
        return self.bridges[port]

    def close(self):
        for bridge in self.bridges.values():
            bridge.close()
        self.bridges.clear()

    def snapshot(self):
        all_statuses = {}
        for text_port in self.pins['endpoints']:
            result = self.bridge(int(text_port)).inspect('snapshot')
            if set(result) & set(all_statuses):
                raise ValueError('Duplicate tasks across lanes')
            all_statuses.update(result)
        stamp = datetime.now(timezone.utc)
        snapshot = dict(schema='W30VerifiedWorkbenchSnapshotV1',checked_UTC=stamp.isoformat(),
                        plan_sha256=self.pins['plan_ref']['sha256'],statuses=all_statuses)
        # This durable barrier precedes every candidate submit/resume in tick.
        name = stamp.strftime('%Y%m%dT%H%M%S.%fZ') + '-' + digest(snapshot)[:12] + '.json'
        save(Path(self.pins['controller_directory']) / 'snapshots' / name, snapshot)
        self.last_statuses = all_statuses
        return all_statuses

    def probe(self, key):
        return self.bridge(self.plan['tasks'][key]['executor_port']).inspect('probe', key)

    def receipt(self, receipt):
        key = receipt['task_key']
        return self.bridge(self.plan['tasks'][key]['executor_port']).upload_receipt(receipt)

    def post(self, action, run_id, body):
        key = body.get('task_key')
        if action == 'resume':
            candidates = [k for k,s in self.last_statuses.items() if s['run_id'] == run_id]
            if len(candidates) != 1:
                raise ValueError('Ambiguous same-ID resume')
            key = candidates[0]
        port = self.plan['tasks'][key]['executor_port']
        path = '/api/runs' if action == 'submit' else '/api/runs/' + run_id + '/resume'
        answer = self.bridge(port).http('POST', path, body)
        expected_id = run_id or digest({name:body[name] for name in ('request_id','task_key','plan_sha256')})[:24]
        if (not isinstance(answer, dict) or answer.get('run_id') != expected_id
                or answer.get('task_key') != key or answer.get('plan_sha256') != body['plan_sha256']
                or answer.get('source_commit') != self.plan['source_commit']):
            return dict(outcome='unknown', answer=answer)
        return dict(outcome='accepted', evidence_verified=True, run_id=expected_id,
                    task_key=key, plan_sha256=body['plan_sha256'], status=answer['status'])


class Singleton:
    """Exclusive generation; stale owner is never stolen automatically."""
    def __init__(self, directory):
        import psutil
        self.directory = Path(directory)
        self.lock = self.directory / 'singleton'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock.mkdir()  # atomic; any old lock requires root reconciliation
        save(self.lock / 'owner.json', dict(pid=os.getpid(), process_birth=psutil.Process().create_time(),
             boot_id='boot-time:' + str(psutil.boot_time()), started_UTC=datetime.now(timezone.utc).isoformat()))

    def close(self):
        (self.lock / 'owner.json').unlink()
        self.lock.rmdir()


class OpsClock:
    """Local permanent subaccount of a root-precharged shared engineering sum.

    Before any pass, reserve its hard subprocess timeout plus 10s parent margin.
    A killed/unknown pass never refunds. Pending state blocks a fresh generation
    until root reconciliation. Sleep is excluded; the 168h deadline still applies.
    """
    def __init__(self, pins, *, clock=time.monotonic):
        self.path = Path(pins['controller_directory']) / 'ops_clock.json'
        self.clock = clock
        self.limit = pins['controller_ops_active_seconds']
        self.timeout = pins['max_pass_seconds']
        if (type(self.limit) is not int or self.limit <= 0 or type(self.timeout) is not int
                or not 1 <= self.timeout <= 300 or self.timeout + 10 > self.limit):
            raise ValueError('Finite root controller allocation/pass timeout required')
        receipt = read_ref(pins['ops_precharge_ref'])
        contract = read_ref(pins['budget_contract_ref'])
        if (receipt.get('root_approved') is not True
                or receipt.get('passed') is not True
                or receipt.get('controller_ops_active_seconds') != self.limit
                or receipt.get('controller_directory') != pins['controller_directory']
                or receipt.get('budget_contract_sha256') != pins['budget_contract_ref']['sha256']
                or contract.get('controller_ops_active_seconds') != self.limit
                or contract['minimum_charged_seconds'].get('engineering', 0)
                   < self.limit + contract['prior_CPU_engineering_seconds']):
            raise ValueError('Root shared engineering precharge missing')
        self.identity = pins['ops_precharge_ref']['sha256']
        self.value = json.loads(self.path.read_text()) if self.path.exists() else dict(
            precharge_sha256=self.identity, charged_seconds=0., actual_seconds=0., passes=0)
        if self.value['precharge_sha256'] != self.identity or self.value.get('pending'):
            raise ValueError('Unknown/mismatched controller time needs root reconciliation')

    def begin(self):
        reserve = self.timeout + 10
        if self.value['charged_seconds'] + reserve > self.limit:
            return False
        self.value['charged_seconds'] += reserve
        self.value['pending'] = dict(reserved_seconds=reserve,
                                    started_UTC=datetime.now(timezone.utc).isoformat())
        save(self.path, self.value)
        self.started = self.clock()
        return True

    def finish(self):
        actual = self.clock() - self.started
        reserved = self.value.pop('pending')['reserved_seconds']
        self.value['charged_seconds'] += actual - reserved
        self.value['actual_seconds'] += actual
        self.value['passes'] += 1
        save(self.path, self.value)
        if actual > reserved or self.value['charged_seconds'] > self.limit:
            raise ValueError('Controller pass exceeded finite reserved time; stop')


def run_single_pass(pins):
    adapter = Adapter(pins)
    try:
        controller = Controller(Path(pins['controller_directory']) / 'intents',
            adapter.plan, adapter.delegation, digest(adapter.delegation),
            snapshot=adapter.snapshot, probe=adapter.probe,
            write_receipt=adapter.receipt, post=adapter.post)
        result = controller.tick()
        if result == 'observe' and all(controller.completed(adapter.last_statuses.get(k),t)
                                       for k,t in adapter.plan['tasks'].items()):
            result = 'stage_completed'
        save(Path(pins['controller_directory']) / 'controller_status.json',
             dict(result=result, checked_UTC=datetime.now(timezone.utc).isoformat()))
        return result
    finally:
        adapter.close()


def verify_launch_admission(pins):
    receipt = read_ref(pins['controller_launch_admission_ref'])
    delegation = read_ref(pins['delegation_ref'])
    if (receipt.get('root_approved') is not True or receipt.get('passed') is not True
            or receipt.get('phase') != delegation['mode']
            or receipt.get('plan_sha256') != pins['plan_ref']['sha256']
            or receipt.get('adapter_source_sha256') != pins['adapter_source_sha256']
            or receipt.get('controller_source_sha256') != delegation['controller_source_sha256']
            or receipt.get('launcher_source_sha256') != pins['launcher_source_sha256']
            or receipt.get('ops_precharge_ref') != pins['ops_precharge_ref']):
        raise ValueError('Separate actual root controller launch admission required')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pins'); parser.add_argument('--pins-sha')
    parser.add_argument('--single-pass', action='store_true')
    parser.add_argument('--remote-pins'); parser.add_argument('--remote-pins-sha')
    parser.add_argument('--operation', choices=('snapshot','probe'))
    parser.add_argument('--task-key')
    args = parser.parse_args()
    if args.remote_pins:
        pins = read_ref(dict(path=args.remote_pins, sha256=args.remote_pins_sha))
        plan = read_ref(pins['plan_ref'])
        if pins['adapter_source_sha256'] != file_sha(__file__):
            raise ValueError('Remote helper source changed')
        result = inspect_statuses(pins, plan) if args.operation == 'snapshot' else inspect_probe(pins, plan, args.task_key)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return
    pins_ref = dict(path=args.pins, sha256=args.pins_sha)
    pins = read_ref(pins_ref)
    verify_launch_admission(pins)
    if not isinstance(pins.get('local_pause_markers'), list) or not pins['local_pause_markers']:
        raise ValueError('Explicit future local pause markers required')
    poll_seconds = health_poll_seconds(pins)
    if args.single_pass:
        verify_single_pass_parent(pins)
        print(json.dumps(dict(result=run_single_pass(pins))))
        return
    group_singleton = Singleton(pins['group_controller_directory'])
    singleton = None
    try:
        singleton = Singleton(pins['controller_directory'])
        ops_clock = OpsClock(pins)
        while True:
            # Immutable pins, delegation, authority and plan reread every pass.
            current = read_ref(pins_ref)
            authority = read_ref(current['authority_ref'])
            if datetime.now(timezone.utc) >= datetime.fromisoformat(authority['group_deadline_UTC'].replace('Z','+00:00')):
                return
            if any(Path(p).exists() for p in pins['local_pause_markers']):
                return
            if not ops_clock.begin():
                save(Path(pins['controller_directory']) / 'controller_status.json', dict(result='controller_active_budget_exhausted'))
                return
            output_refs = None
            try:
                completed = subprocess.run([sys.executable, '-m', 'satellite_rl.power30.orchestration_adapter',
                    '--pins', args.pins, '--pins-sha', args.pins_sha, '--single-pass'],
                    check=False, capture_output=True, text=True, timeout=ops_clock.timeout)
                output_refs = persist_child_outputs(pins['controller_directory'],
                    datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'),completed.stdout,completed.stderr)
                completed.check_returncode()
                result = json.loads(completed.stdout)['result']
            except Exception as error:
                # Reservation stays pending/charged. No automatic process retry.
                if output_refs is None:
                    output_refs = persist_child_outputs(pins['controller_directory'],
                        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'-error',
                        getattr(error,'stdout',None),getattr(error,'stderr',None))
                save(Path(pins['controller_directory']) / 'controller_status.json',
                     dict(result='unknown_pass_needs_root', error=repr(error),child_output_refs=output_refs))
                return
            ops_clock.finish()
            if result in ('deadline', 'unknown_observe_only', 'stage_completed'):
                return
            time.sleep(next_health_delay_seconds(result,poll_seconds))
    finally:
        if singleton is not None:
            singleton.close()
        group_singleton.close()


if __name__ == '__main__':
    main()
