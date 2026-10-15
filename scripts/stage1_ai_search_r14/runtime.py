"""Fail-closed R14 preparation components; no service, SSH or implicit dispatch.

Single shared-domain lock serializes ledger/store mutations across both lanes.
Stale locks/intents are evidence requiring reconciliation, never auto-retries.
"""
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import time
import contextvars

CONTROL = contextvars.ContextVar('r14_control',default=None)

@contextmanager
def control_scope(check):
    token=CONTROL.set(check)
    try:yield
    finally:CONTROL.reset(token)

GIB = 1024 ** 3
STAGE_SECONDS = {'B': 3600., 'trajectory': 3600., 'fit_baseline': 60., 'audit': 900., 'ops': 2640.}
D5_CAPS = {'EVICT': 128, 'INSERT': 256, 'REFIT': 8, 'SPLIT': 64, 'MOVE': 56}


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf8')


@contextmanager
def exclusive(path,timeout=30.,clock=time.monotonic,sleeper=time.sleep):
    path = Path(path)
    deadline=clock()+timeout
    while True:
        control=CONTROL.get()
        if control is not None:control()
        try:
            fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);break
        except FileExistsError:
            if clock()>=deadline:raise TimeoutError('bounded shared lock wait; unknown lock preserved')
            sleeper(min(.01,max(0,deadline-clock())))
    try:
        os.write(fd, json_bytes({'pid': os.getpid(), 'created_ns': time.time_ns()}))
        os.fsync(fd)
        yield
    finally:
        os.close(fd)
        path.unlink()


def atomic_bytes(path, raw):
    path = Path(path)
    temporary = path.with_name(path.name + '.part')
    with temporary.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class ObjectStore:
    """One indexed canonical domain, no recursive inventory during writes.

    Interrupted data/index transaction leaves an intent that blocks all new
    mutations. Existing objects are read/hash checked before deduplication.
    """
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, raw):
        if not isinstance(raw, bytes) or len(raw) > 32 * 1024 ** 2:
            raise ValueError('bounded object bytes required')
        digest = hashlib.sha256(raw).hexdigest()
        with exclusive(self.root / '.domain.lock'):
            intent = self.root / 'mutation.intent.json'
            if intent.exists():
                raise ValueError('unresolved atomic mutation; no automatic reconciliation')
            index_path = self.root / 'index.json'
            index = json.loads(index_path.read_bytes()) if index_path.exists() else {}
            target = self.root / digest
            if digest in index:
                if target.read_bytes() != raw:
                    raise ValueError('canonical object bytes changed')
                return {'sha256': digest, 'bytes': len(raw), 'path': str(target)}
            updated = {**index, digest: len(raw)}
            encoded = json_bytes(updated)
            # Reserve128MiB of the same2GiB for bounded control/index/log files;
            # this is not extra space outside the canonical-domain cap.
            if len(updated) > 8192 or sum(updated.values()) + len(encoded) + 128*1024**2 > 2 * GIB:
                raise ValueError('shared canonical object/index quota')
            if target.exists():
                raise ValueError('unindexed object: preserve and reconcile')
            atomic_bytes(intent, json_bytes({'sha256': digest, 'bytes': len(raw)}))
            atomic_bytes(target, raw)
            atomic_bytes(index_path, encoded)
            intent.unlink()
        return {'sha256': digest, 'bytes': len(raw), 'path': str(target)}

    def put_json(self,value):
        raw=json_bytes(value)
        if len(raw)<=32*1024**2:return self.put(raw)
        refs=[self.put(raw[i:i+8*1024**2]) for i in range(0,len(raw),8*1024**2)]
        return self.put(json_bytes({'schema':'R14ChunkedJSONV1','bytes':len(raw),
            'sha256':hashlib.sha256(raw).hexdigest(),'chunks':refs}))


class BudgetLedger:
    """Reservations prevent two lanes overspending the shared outer-time cap.

    Full outer tasks charge their B/trajectory ceiling. Nested offline audit
    additionally charges overlapping audit900; group total adds outer only.
    Unknown task costs stay reserved; stages never transfer unused quota.
    """
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _load(self):
        p = self.root / 'budget.json'
        return json.loads(p.read_bytes()) if p.exists() else {'tasks': {}}

    def available(self,stage):
        with exclusive(self.root/'.budget.lock'):
            tasks=list(self._load()['tasks'].values())
            total=sum(t.get('elapsed_seconds',t['reserved_seconds']) for t in tasks)
            used=sum(t.get('stage_charges',t.get('stage_reservations',{t['stage']:t.get('elapsed_seconds',t['reserved_seconds'])})).get(stage,0) for t in tasks)
            return max(0.,min(10800-total,STAGE_SECONDS[stage]-used))

    def reserve(self, task_id, stage, seconds,additional_stage_reservations=None):
        if stage not in STAGE_SECONDS or type(seconds) not in (float, int) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('bounded stage reservation')
        with exclusive(self.root / '.budget.lock'):
            ledger = self._load()
            tasks = ledger['tasks']
            if task_id in tasks:
                raise ValueError('one dispatch only; completed/failed/unknown IDs cannot repeat')
            charged = lambda t: t.get('elapsed_seconds', t['reserved_seconds'])
            if sum(charged(t) for t in tasks.values()) + seconds > 10800:
                raise ValueError('whole-group aggregate outer duration cap')
            reservations={stage:float(seconds),**(additional_stage_reservations or {})}
            if stage in (additional_stage_reservations or {}):raise ValueError('nested reservation cannot replace full outer')
            for name,amount in reservations.items():
                if name not in STAGE_SECONDS or type(amount) not in (float,int) or not 0<amount<=seconds:raise ValueError('bounded nested stage reservation')
                if name!=stage and (name!='audit' or amount!=seconds):raise ValueError('nested audit must reserve full hard outer, never unsafe smaller allowance')
                existing=sum(t.get('stage_charges',t.get('stage_reservations',{t['stage']:charged(t)})).get(name,0) for t in tasks.values())
                if existing+amount>STAGE_SECONDS[name]:raise ValueError('stage aggregate cap; no quota transfer')
            tasks[task_id] = {'stage': stage, 'reserved_seconds': float(seconds), 'state': 'reserved',
                              'stage_reservations':reservations,
                              'wall_reserved_ns': time.time_ns()}
            atomic_bytes(self.root / 'budget.json', json_bytes(ledger))

    def finish(self, task_id, elapsed, exit_facts,stage_charges=None):
        if type(elapsed) not in (float, int) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('actual outer elapsed required')
        required = {'pid', 'process_start_identity', 'raw_returncode', 'reaped', 'descendants_live'}
        if not required <= set(exit_facts) or exit_facts['reaped'] is not True or exit_facts['descendants_live'] != 0:
            raise ValueError('direct actual exit and whole owned tree closure required')
        with exclusive(self.root / '.budget.lock'):
            ledger = self._load()
            task = ledger['tasks'][task_id]
            if task['state'] != 'reserved':
                raise ValueError('terminal task cannot finish twice')
            charges=stage_charges or {task['stage']:elapsed}
            if (any(k not in task['stage_reservations'] or type(v) not in (float,int) or not 0<=v<=elapsed for k,v in charges.items())
                or charges.get(task['stage'])!=elapsed):raise ValueError('full outer stage plus overlapping bounded nested audit')
            if any(charges[k]>task['stage_reservations'][k] for k in charges):
                task.update(state='failed',elapsed_seconds=elapsed,stage_charges=charges,exit_facts=exit_facts,
                    failure='actual stage reservation exceeded; costs preserved')
                atomic_bytes(self.root/'budget.json',json_bytes(ledger));raise ValueError('actual overlapping reservation overrun')
            task.update(elapsed_seconds=elapsed,stage_charges=charges, exit_facts=exit_facts,
                        state='complete' if exit_facts['raw_returncode'] == 0 and elapsed <= task['reserved_seconds'] else 'failed')
            atomic_bytes(self.root / 'budget.json', json_bytes(ledger))

    def finish_inline_ops(self,task_id,elapsed):
        with exclusive(self.root/'.budget.lock'):
            ledger=self._load();task=ledger['tasks'][task_id]
            if task['stage']!='ops' or task['state']!='reserved' or elapsed<0:raise ValueError('actual inline ops only')
            task.update(state='complete',elapsed_seconds=elapsed,stage_charges={'ops':elapsed},scope='inline control operation, no child exit claim')
            atomic_bytes(self.root/'budget.json',json_bytes(ledger))


class PauseGuard:
    def __init__(self, own_paths, global_paths, historical_exact_refs=(), *, seconds, clock=time.monotonic):
        if not own_paths or not global_paths or not 0 < seconds <= 10800:
            raise ValueError('own/global pause and finite bound mandatory')
        self.paths = tuple(map(Path, (*own_paths, *global_paths)))
        self.history = {str(Path(r['path'])): r for r in historical_exact_refs}
        if not set(self.history) <= set(map(str, self.paths)):
            raise ValueError('historical exceptions must remain watched')
        self.clock, self.started, self.seconds = clock, clock(), seconds

    def check(self, new_work=True):
        for path in self.paths:
            if path.is_symlink():
                raise RuntimeError('linked pause marker fails closed')
            try:
                with path.open('rb') as stream:
                    raw = stream.read(131073)
            except FileNotFoundError:
                if str(path) in self.history:
                    raise RuntimeError('historical pause evidence disappeared')
                continue
            ref = self.history.get(str(path))
            if ref is None or len(raw) != ref['bytes'] or hashlib.sha256(raw).hexdigest() != ref['sha256']:
                raise RuntimeError('new/changed/malformed own or global pause')
        if self.clock() - self.started >= self.seconds * (.9 if new_work else 1.):
            raise RuntimeError('save/reap reserve or hard outer deadline reached')


def development_schedule():
    arms = ('AI', 'strong_causal_baseline', 'full_SHC')
    return [{'mother': m, 'repeat': r, 'arms_serial': [arms[(m+r+k) % 3] for k in range(3)]}
            for m in range(20) for r in range(3)]


def require_execution(config):
    """Preparation defaults can never accidentally execute a fixture/payload."""
    mandatory = ('scientific_execution_authorized', 'root_exact_source_input_admitted',
                 'source_specific_pure_checks_accepted', 'actual_runtime_owner_stop_space_admitted')
    if any(config.get(k) is not True for k in mandatory):
        raise ValueError('preparation only; independent source/input/runtime admission missing')
    if config.get('port') not in (20473, 20474) or config.get('retry') != 0:
        raise ValueError('only73/74 and no retry')
    if not config.get('root_admission_ref') or not config.get('runtime_epoch_ref'):
        raise ValueError('exact independently pinned admission and new epoch required')


def produce_B(scene, identity, *, config, source_sha256, producer_sha256,
              phase_manifest, checkpoint, backend=None):
    """Native unchanged W→RULE→D5; caller guards120s,108s save boundary.

    checkpoint persists each full unique object/cost and accepted state. No T,
    accepted-history independent audit, fallback promotion or substitution.
    The backend is injected only after frozen-source admission.
    """
    require_execution(config)
    if backend is None:
        raise ValueError('explicit frozen original R5 backend required')
    checkpoint('before_W', None)
    prepared = backend.prepare_case(scene, identity, source_sha256,
                                    phase_manifest=phase_manifest, time_limit=10.)
    checkpoint('W', prepared.to_dict())
    if prepared.status != 'available':
        return {'status': 'unavailable', 'qualified': False, 'reason': prepared.reason}
    checkpoint('before_RULE', None)
    proposal = backend.build_proposal(prepared, 'RULE', producer_sha256=producer_sha256)
    checkpoint('RULE', proposal)
    checkpoint('before_D5', None)
    result = backend.decode_groups(prepared, proposal, time_limit=5.,
                                   on_accepted=lambda *args: checkpoint('D5_accepted', args))
    checkpoint('D5', result)
    return {'prepared': prepared, 'proposal':proposal, 'result': result, 'qualified': False,
            'status': result['status'], 'requires_independent_three_state_audit': True}
