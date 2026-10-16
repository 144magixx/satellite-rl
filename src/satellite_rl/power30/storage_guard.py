"""Periodic server-reported allocated-byte guard for this W30 tree only."""
import os
from pathlib import Path
import subprocess
import time
from .logging import atomic_json


def validate_contract(config,directory):
    c=config['storage_control'];root=Path(c['owned_storage_root'])
    if (not root.is_absolute() or root.name!='s2_w30' or not root.is_dir()
            or root.resolve()!=root.absolute() or not Path(directory).resolve().is_relative_to(root)):
        raise ValueError('Actual canonical owned s2_w30 storage scope required')
    marker=Path(c['storage_stop_marker'])
    if (not marker.is_absolute() or not marker.parent.resolve().is_relative_to(root)
            or marker.resolve()!=marker.absolute() or str(marker) not in config['pause_markers']):
        raise ValueError('Storage STOP must stay in this owned root and all pause contracts')
    if (type(c['baseline_allocated_bytes']) is not int or c['baseline_allocated_bytes']<0
            or type(c['max_growth_bytes']) is not int or not 0<c['max_growth_bytes']<=90*1024**3
            or type(c['guard_interval_seconds']) is not int or not 1<=c['guard_interval_seconds']<=60
            or type(c['storage_probe_timeout_seconds']) is not int or not 1<=c['storage_probe_timeout_seconds']<=15
            or config['reserve_free_bytes']<16*1024**3):
        raise ValueError('Root-fixed finite storage growth, cadence and reserve required')
    for key in ('model_dir','ledger_root'):
        if key in config and not Path(config[key]).resolve().is_relative_to(root):
            raise ValueError('Worker storage destination leaves owned scope: '+key)
    return c


class StorageGuard:
    def __init__(self,config,directory,*,clock=time.monotonic):
        self.contract=validate_contract(config,directory)
        self.directory=Path(directory);self.clock=clock;self.last=None;self.stopped=False

    def check(self):
        if self.stopped:return True
        stamp=self.clock()
        if self.last is not None and stamp-self.last<self.contract['guard_interval_seconds']:
            return False
        self.last=stamp;c=self.contract
        try:
            # GNU du reports st_blocks allocation from the server; this is not
            # a personal quota reading. -P never follows child symlinks, -x
            # stays on the owned filesystem. No cross-scope tree is traversed.
            result=subprocess.run(['du','-s','-B1','-x','-P','--',c['owned_storage_root']],
                check=True,capture_output=True,text=True,timeout=c['storage_probe_timeout_seconds'])
            fields=result.stdout.rstrip('\n').split('\t')
            if len(fields)!=2 or fields[1]!=c['owned_storage_root']:
                raise ValueError('Cannot reliably bind allocated-byte report to owned root')
            allocated=int(fields[0])
            if allocated<0:raise ValueError('Negative allocation report')
            growth=max(0,allocated-c['baseline_allocated_bytes'])
            receipt=dict(server_reported_allocated_bytes=allocated,growth_bytes=growth,
                max_growth_bytes=c['max_growth_bytes'],personal_quota='unknown',
                measurement_scope=c['owned_storage_root'],elapsed_seconds=self.clock()-stamp)
            if growth>c['max_growth_bytes']:
                receipt['stop_reason']='owned_storage_growth_limit'
                self.stopped=True
        except Exception as exc:
            self.stopped=True
            receipt=dict(stop_reason='owned_storage_measurement_unverifiable',error=repr(exc),
                         personal_quota='unknown',elapsed_seconds=self.clock()-stamp)
        if self.stopped:
            marker=Path(c['storage_stop_marker']);marker.parent.mkdir(parents=True,exist_ok=True)
            try:
                fd=os.open(marker,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
                with os.fdopen(fd,'w',encoding='utf8') as stream:
                    stream.write(receipt['stop_reason']+'\n');stream.flush();os.fsync(stream.fileno())
            except FileExistsError:
                pass # Another owned worker already stopped this group; retain it.
        atomic_json(self.directory/'storage_probe.json',receipt)
        return self.stopped
