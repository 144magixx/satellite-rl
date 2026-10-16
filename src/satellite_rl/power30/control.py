"""Fail-closed W30 root admission, pause, storage and owned-process controls."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from datetime import datetime, timezone
from satellite_rl.t0_width.workbench import birth, boot_id


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_ref(ref):
    if set(ref) != {'path', 'sha256'} or sha(ref['path']) != ref['sha256']:
        raise ValueError('Admission reference SHA differs')
    return json.loads(Path(ref['path']).read_text(encoding='utf8'))


def validate_data_admission(config):
    receipt = read_ref(config['data_admission_ref'])
    if (receipt.get('experiment_id') != 'S2-W30' or receipt.get('root_approved') is not True
            or receipt.get('passed') is not True or receipt.get('data_manifest_sha') != config['data_manifest_sha']):
        raise ValueError('Root-qualified W30 data required')
    if receipt.get('uniform_request_bps') != 21600000 or receipt.get('train_mothers') != 576 or receipt.get('dev_mothers') != 20:
        raise ValueError('W30 complete uniform input contract')
    manifest_ref=config['data_manifest_ref']
    manifest=read_ref(manifest_ref)
    if manifest_ref['sha256']!=config['data_manifest_sha']:
        raise ValueError('Runtime pinned manifest bytes differ from source admission')
    rows=manifest['rows']
    if (len(rows)!=596 or any(r['role'] not in ('train','development') for r in rows)
            or len({r['case_id'] for r in rows})!=596
            or [r for r in rows if r['role']=='train']!=config['train_cases']
            or [r for r in rows if r['role']=='development']!=config['dev_cases']):
        raise ValueError('All arms require exact ordered same576train+20dev rows; test sealed')


def validate_engineering_data_admission(config):
    if config.get('dev_cases')!=[]:
        raise ValueError('Engineering cannot execute development evaluation')
    # Preserve the exact20 development identities while keeping them outside
    # the engineering evaluation list. No case/layout/PHY is opened here.
    from copy import copy
    check=copy(config);check['dev_cases']=config['development_identity_rows']
    validate_data_admission(check)


class RuntimeControl:
    def __init__(self, config, directory):
        self.config = config
        self.directory = Path(directory)
        self.last_check = 0.
        self.reason = None
        self.started = time.monotonic()
        self.storage_guard=None

    def arm_storage_guard(self):
        from .storage_guard import StorageGuard
        self.storage_guard=StorageGuard(self.config,self.directory)

    def validate_owner(self):
        owner = json.loads((self.directory / 'status.json').read_text(encoding='utf8'))
        if (owner.get('pid') != os.getpid() or owner.get('process_birth') != birth(os.getpid())
                or owner.get('boot_id') != boot_id() or owner.get('run_id') != self.config['run_id']):
            raise ValueError('W30 may only enter via persistent workbench owned worker')
        receipt = read_ref(self.config['execution_admission_ref'])
        if (receipt.get('experiment_id') != 'S2-W30' or receipt.get('root_approved') is not True
                or receipt.get('passed') is not True or receipt.get('source_sha') != self.config['source_sha']
                or receipt.get('phase') != self.config['phase']):
            raise ValueError('Root phase/source execution admission required')
        if self.config['phase'] == 'scientific' and receipt.get('engineering_accepted') is not True:
            raise ValueError('Scientific work requires separate actual engineering acceptance')
        authority=read_ref(receipt['authority_ref'])
        if (authority.get('schema')!='W30HumanImplementationAuthorityV1'
                or authority.get('new_human_pause_overrides') is not True
                or self.config['deadline_utc']!=authority.get('group_deadline_UTC')
                or authority.get('group_window_hours')!=168):
            raise ValueError('Exact authority-origin 168h window required')
        if receipt.get('pause_markers') != self.config.get('pause_markers'):
            raise ValueError('Frozen future-pause contract differs from root receipt')
        from .storage_guard import validate_contract
        validate_contract(self.config,self.directory)
        if receipt.get('storage_control')!=self.config.get('storage_control'):
            raise ValueError('Root fixed storage control contract differs')
        if not self.config.get('pause_markers'):
            raise ValueError('Future pause marker contract required')
        if float(self.config['reserve_free_bytes']) < 16*1024**3:
            raise ValueError('Storage safety reserve below frozen 16GiB')
        if self.stop_requested(force=True):
            raise RuntimeError('Admission stopped: ' + str(self.reason))

    def stop_requested(self, force=False):
        stamp = time.monotonic()
        if not force and stamp - self.last_check < .5:
            return self.reason is not None
        self.last_check = stamp
        if self.storage_guard is not None and self.storage_guard.check():
            self.reason='owned_storage_STOP'
        if any(Path(p).exists() for p in self.config['pause_markers']):
            self.reason = 'user_or_root_pause'
        deadline = datetime.fromisoformat(self.config['deadline_utc'].replace('Z', '+00:00'))
        if datetime.now(timezone.utc) >= deadline:
            self.reason = '168h_group_window_expired'
        if shutil.disk_usage(self.directory).free < self.config['reserve_free_bytes']:
            self.reason = 'storage_safety_reserve'
        # Counter files are permanent across restart; elapsed active work is
        # never inferred from the restored policy checkpoint.
        from .ledger import TIME_LIMITS
        totals = {}
        own_seconds = 0.
        for path in (Path(self.config['ledger_root']) / 'workers').glob('*.json'):
            item = json.loads(path.read_text(encoding='utf8'))
            for phase, seconds in item['seconds'].items():
                totals[phase] = totals.get(phase, 0.) + seconds + item.get('burned_unknown_seconds',{}).get(phase,0.)
            if item['run_id'] == self.config['run_id']:
                own_seconds = item['seconds'].get('scientific', 0.) + item.get('burned_unknown_seconds',{}).get('scientific',0.)
        if self.config['phase'] == 'scientific' and own_seconds >= 20*3600-10:
            self.reason = 'per_arm_20_gpu_hours'
        for phase, seconds in totals.items():
            cap = (160*3600 if self.config.get('first_round', True) else TIME_LIMITS['scientific']) if phase == 'scientific' else TIME_LIMITS[phase]
            if seconds >= cap-40:
                self.reason = phase + '_group_time_limit'
        return self.reason is not None

    def check_or_raise(self):
        # Boundary stopping is handled by runner so that the exact transition
        # and update finish and are saved. Never stop a half-mutated optimizer.
        if self.reason is not None:
            raise RuntimeError('Already stopped runtime cannot start actual call')
