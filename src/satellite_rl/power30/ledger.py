"""Permanent cross-worker bounded accounting; reservations survive unknown exits.

The shared filesystem must support atomic mkdir and rename (qualified by ops).
Calls reserve blocks BEFORE entry. A failed/unknown process keeps its unspent
reservation charged; only a normal close releases provably unentered calls.
Actual counters are checkpointed at each evidence barrier, independently of ML
checkpoints. Thus restoring a policy cannot restore or refund the call budget.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
from .logging import atomic_json

LIMITS = dict(phy=5060480, kernel=5100000, coeff=5100000,
              feature=10200000, netforward=67000000,
              engineering_interaction=16384, fixture_phy=4096,
              extra_forward=8192, scientific_interaction=3600000,
              evaluation_episode=7200, layout_attempt=80,
              update=3552416, engineering_update=392, scientific_update=3552024)
TIME_LIMITS = dict(engineering=8*3600, data=4*3600, evaluation=24*3600,
                   report=4*3600, scientific=480*3600)


class BudgetExceeded(RuntimeError):
    pass


@contextmanager
def directory_lock(root, timeout=15):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / '.ledger_lock'
    until = time.monotonic() + timeout
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            if time.monotonic() >= until:
                raise RuntimeError('Shared ledger lock unavailable; never steal unknown lock')
            time.sleep(.02)
    try:
        yield
    finally:
        lock.rmdir()


class CallLedger:
    def __init__(self, root, run_id, phase, block=128, allowances=None):
        self.root = Path(root)
        self.run_id = run_id
        self.phase = 'engineering' if phase == 'correctness' else phase
        self.block = block
        self.allowances = dict(allowances or {})
        self.path = self.root / 'workers' / (run_id + '.json')
        self.global_path = self.root / 'budget.json'
        self.last_clock = time.monotonic()
        self.last_flush = self.last_clock
        with directory_lock(root):
            if not self.global_path.exists():
                raise ValueError('Root must initialize permanent W30 budget before entry')
            budget = json.loads(self.global_path.read_text(encoding='utf8'))
            if budget.get('experiment_id') != 'S2-W30' or budget.get('limits') != LIMITS:
                raise ValueError('W30 immutable call limits mismatch')
            if self.path.exists():
                self.value = json.loads(self.path.read_text(encoding='utf8'))
                if self.value.get('open'):
                    # Old reservation is not silently reused or refunded. Root
                    # must classify an interrupted worker before same-ID resume.
                    raise RuntimeError('Open permanent ledger requires root reconciliation')
            else:
                self.value = dict(run_id=run_id, actual={}, burned_unknown={}, reserved={}, seconds={}, reserved_seconds={},
                                  batches={}, failures=[], open=False)
            self.value['open'] = True
            atomic_json(self.path, self.value)
        self._time()

    def _time(self):
        if not self.value.get('open'):
            return
        stamp = time.monotonic()
        elapsed = stamp - self.last_clock
        self.last_clock = stamp
        self.value['seconds'][self.phase] = self.value['seconds'].get(self.phase, 0.) + elapsed
        actual = self.value['seconds'][self.phase]+self.value.get('burned_unknown_seconds',{}).get(self.phase,0.)
        reserved = self.value.setdefault('reserved_seconds', {}).get(self.phase, 0.)
        if self.value.get('open') and actual + 5 >= reserved:
            with directory_lock(self.root):
                budget = json.loads(self.global_path.read_text(encoding='utf8'))
                cap = TIME_LIMITS[self.phase]
                if self.phase == 'scientific' and budget.get('first_round', True):
                    cap = 160*3600
                charged = budget.setdefault('charged_seconds', {}).get(self.phase, 0.)
                amount = 30.
                if charged + amount > cap or (self.phase == 'scientific' and reserved + amount > 20*3600):
                    raise BudgetExceeded(self.phase + ' permanent time budget exhausted')
                budget['charged_seconds'][self.phase] = charged + amount
                atomic_json(self.global_path, budget)
                self.value['reserved_seconds'][self.phase] = reserved + amount
                atomic_json(self.path, self.value)

    def set_phase(self, phase):
        self.flush()
        self.phase = 'engineering' if phase == 'correctness' else phase

    def charge(self, kind, **metadata):
        if kind not in LIMITS:
            raise ValueError('Unregistered actual call: ' + kind)
        self._time()
        actual = self.value['actual'].get(kind, 0)
        reserved = self.value['reserved'].get(kind, 0)
        burned=self.value.get('burned_unknown',{}).get(kind,0)
        if actual+burned >= reserved:
            with directory_lock(self.root):
                budget = json.loads(self.global_path.read_text(encoding='utf8'))
                charged = budget.setdefault('charged', {}).get(kind, 0)
                limit = LIMITS[kind]
                if budget.get('first_round',True):
                    limit = dict(scientific_interaction=1200000,
                                 evaluation_episode=1440,update=1184400,
                                 scientific_update=1184008).get(kind,limit)
                amount = min(self.block, limit - charged)
                if kind in self.allowances:
                    amount = min(amount, self.allowances[kind]-reserved)
                if amount <= 0:
                    raise BudgetExceeded(kind + ' permanent budget exhausted')
                # Persist the global reservation first: any crash in the two
                # writes overcharges rather than making unknown spend free.
                budget['charged'][kind] = charged + amount
                atomic_json(self.global_path, budget)
                self.value['reserved'][kind] = reserved + amount
                atomic_json(self.path, self.value)
        self.value['actual'][kind] = actual + 1
        if kind == 'netforward':
            key = str(metadata.get('network', 'unknown')) + ':' + str(metadata.get('batch_size', 'unknown'))
            self.value['batches'][key] = self.value['batches'].get(key, 0) + 1
        if time.monotonic() - self.last_flush >= 5:
            self.flush()

    def flush(self):
        self._time()
        atomic_json(self.path, self.value)
        self.last_flush = time.monotonic()

    def snapshot(self):
        return json.loads(json.dumps(self.value))

    def record_io(self,path,size):
        io=self.value.setdefault('logical_atomic_io',dict(writes=0,bytes=0))
        io['writes']+=1;io['bytes']+=int(size)

    def close(self):
        if not self.value.get('open'):
            return
        self.flush()
        with directory_lock(self.root):
            budget = json.loads(self.global_path.read_text(encoding='utf8'))
            for kind, reserved in self.value['reserved'].items():
                unused = reserved - self.value['actual'].get(kind, 0) - self.value.get('burned_unknown',{}).get(kind,0)
                budget['charged'][kind] -= unused
                self.value['reserved'][kind] -= unused
            for phase, reserved in self.value['reserved_seconds'].items():
                unused = max(0., reserved - self.value['seconds'].get(phase, 0.)-self.value.get('burned_unknown_seconds',{}).get(phase,0.))
                budget['charged_seconds'][phase] -= unused
                self.value['reserved_seconds'][phase] -= unused
            # Worker closed first. A crash before global write conservatively
            # retains excess reservation and cannot refund twice on resume.
            self.value['open'] = False
            atomic_json(self.path, self.value)
            atomic_json(self.global_path, budget)


def initialize(root, authority_ref, *, first_round=True):
    """Root preparation only; existing accounting is immutable to this helper."""
    root = Path(root)
    with directory_lock(root):
        path = root / 'budget.json'
        if path.exists():
            raise FileExistsError('Never reinitialize permanent experiment budget')
        atomic_json(path, dict(experiment_id='S2-W30', limits=LIMITS,
                    authority_ref=authority_ref, charged={}, first_round=first_round))


def reconcile_unknown(root,run_id,receipt_ref):
    """Root-reviewed loss accounting only; no rollback or automatic retry.

    Every unproven reserved call is burned and permanently distinguished from
    observed calls. Time must be classified from actual outer process evidence,
    never presumed to equal the 30-second running reservation.
    """
    from .control import read_ref,sha
    receipt=read_ref(receipt_ref);root=Path(root);path=root/'workers'/(run_id+'.json')
    with directory_lock(root):
        if (receipt.get('schema')!='W30UnknownSpendReconciliationV1'
                or receipt.get('experiment_id')!='S2-W30' or receipt.get('run_id')!=run_id
                or receipt.get('root_approved') is not True or receipt.get('independent_review_passed') is not True
                or receipt.get('old_process_exit_verified') is not True or receipt.get('worker_ledger_SHA')!=sha(path)):
            raise ValueError('Explicit root-reviewed actual unknown-exit classification required')
        value=json.loads(path.read_text(encoding='utf8'));budget=json.loads((root/'budget.json').read_text(encoding='utf8'))
        if not value.get('open'):raise ValueError('Only original unknown open ledger')
        for kind,reserved in value['reserved'].items():
            value.setdefault('burned_unknown',{})[kind]=reserved-value['actual'].get(kind,0)
        seconds=receipt['outer_process_seconds_by_phase']
        if set(seconds)!=set(value['reserved_seconds']):raise ValueError('Every phase requires actual outer-time classification')
        for phase,outer in seconds.items():
            if outer<value['seconds'].get(phase,0.):
                raise ValueError('Outer classification cannot undercut observed running time')
            bound=max(outer,value['reserved_seconds'][phase])
            budget['charged_seconds'][phase]+=bound-value['reserved_seconds'][phase]
            value['reserved_seconds'][phase]=bound
            value.setdefault('burned_unknown_seconds',{})[phase]=bound-value['seconds'].get(phase,0.)
        value['root_classified_outer_seconds']=seconds
        value['open']=False;value.setdefault('reconciliation_refs',[]).append(receipt_ref)
        # Preserve old bytes; accounting writes can only increase consumption.
        history=root/'reconciliations'/receipt_ref['sha256'];history.mkdir(parents=True,exist_ok=False)
        (history/'old_worker.json').write_bytes(path.read_bytes())
        atomic_json(root/'budget.json',budget);atomic_json(path,value)


def unlock_conditional(root,receipt_ref):
    """Expand only the preregistered second/third-seed matrix, without reset."""
    from .control import read_ref
    receipt=read_ref(receipt_ref)
    if (receipt.get('experiment_id')!='S2-W30' or receipt.get('root_approved') is not True
            or receipt.get('independent_review_passed') is not True
            or receipt.get('first_round_complete') is not True
            or receipt.get('development_gate_passed') is not True or receipt.get('new40_sealed') is not True):
        raise ValueError('Three preregistered AND gates and independent root review required')
    root=Path(root)
    with directory_lock(root):
        path=root/'budget.json';budget=json.loads(path.read_text(encoding='utf8'))
        if budget.get('first_round') is not True:
            raise ValueError('Conditional gate already consumed; no duplicate expansion')
        budget['first_round']=False;budget['conditional_unlock_ref']=receipt_ref
        atomic_json(path,budget)
