"""W26 owned-process guards; disappearing /proc entries are exit races only."""
import errno
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from datetime import datetime,timezone
from .logging import atomic_json

EXIT_ERRORS={errno.ENOENT,errno.ESRCH}
def now():return datetime.now(timezone.utc).isoformat()
def proc_stat(pid):
    try:
        text=(Path('/proc')/str(pid)/'stat').read_text()
    except OSError as exc:
        if exc.errno in EXIT_ERRORS:return None
        raise
    fields=text.rsplit(') ',1)[1].split()
    return dict(state=fields[0],ppid=int(fields[1]),birth=int(fields[19]))

def birth(pid):
    if os.name=='nt':
        import psutil
        try:return psutil.Process(pid).create_time()
        except psutil.NoSuchProcess:return None
    stat=proc_stat(pid)
    return None if stat is None else stat['birth']

def rss_tree(pid):
    if os.name=='nt':
        import psutil
        try:parent=psutil.Process(pid);processes=[parent]+parent.children(recursive=True)
        except psutil.NoSuchProcess:return 0
        total=0
        for process in processes:
            try:total+=process.memory_info().rss
            except psutil.NoSuchProcess:pass
        return total
    stats={}
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            stat=proc_stat(int(entry.name))
            if stat is not None:stats[int(entry.name)]=stat
    descendants={pid};changed=True
    while changed:
        new={i for i,s in stats.items() if s['ppid'] in descendants}-descendants
        descendants.update(new);changed=bool(new)
    total=0;page=os.sysconf('SC_PAGE_SIZE')
    for owner in descendants:
        try:resident=int((Path('/proc')/str(owner)/'statm').read_text().split()[1])
        except OSError as exc:
            if exc.errno in EXIT_ERRORS:continue
            raise
        total+=resident*page
    return total

def terminate_owned(child,expected_birth):
    if child.poll() is not None:child.wait();return
    identity=birth(child.pid)
    if identity is None:
        child.wait(timeout=3);return
    if identity!=expected_birth:raise RuntimeError('owned process birth changed; refuse kill')
    kill_signal=getattr(signal,'SIGKILL',9)
    for sig in (signal.SIGTERM,kill_signal):
        if child.poll() is not None:child.wait();return
        if birth(child.pid)!=expected_birth:
            child.wait(timeout=3);return
        try:
            if os.name!='nt':
                if os.getpgid(child.pid)!=child.pid:raise RuntimeError('owned process group changed; refuse kill')
                os.killpg(child.pid,sig)
            elif sig==signal.SIGTERM:child.terminate()
            else:child.kill()
        except OSError as exc:
            if exc.errno not in EXIT_ERRORS:raise
            child.wait(timeout=3);return
        try:child.wait(timeout=2 if sig==signal.SIGTERM else 3);return
        except subprocess.TimeoutExpired:
            if sig==kill_signal:raise

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pin(ref):
    if digest(ref['path'])!=ref['sha256']:raise ValueError('guard output SHA changed')

def run_stage(command,directory,limit_seconds,memory_bytes,stop=lambda:False,popen=subprocess.Popen,clock=time.monotonic,memory=rss_tree,sleep=time.sleep):
    if not 0<limit_seconds<=20 or not 0<memory_bytes<=4*1024**3:raise ValueError('W26 hard limit exceeds20s/4GiB')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    wal=directory/'stage.wal.json';output=directory/'stage_output.json';receipt=directory/'guard_receipt.json'
    if receipt.exists():
        old=json.loads(wal.read_text())
        if old['command']!=command or old['hard_seconds']!=limit_seconds or old['owned_parent_children_rss_bytes']!=memory_bytes:raise ValueError('original guarded stage contract changed')
        value=json.loads(receipt.read_text())
        for name in ('output_ref','accepted_snapshot_index_ref'):
            if value.get(name):pin(value[name])
        return value
    if wal.exists():raise ValueError('unknown interrupted stage; no automatic replay')
    if output.exists():raise ValueError('orphan stage output without WAL/receipt')
    began=clock();record=dict(status='intent_before_spawn',command=command,started_utc=now(),hard_seconds=limit_seconds,owned_parent_children_rss_bytes=memory_bytes,unknown_spend_preserved=True)
    atomic_json(wal,record);peak=0;reason=None;monitor_error=None
    with (directory/'stage.log').open('ab',buffering=0) as log:
        child=popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=os.name!='nt',creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        identity=birth(child.pid)
        if identity is None and child.poll() is None:raise RuntimeError('live child identity unavailable; preserve unknown WAL')
        record.update(status='child_started',pid=child.pid,birth=identity);atomic_json(wal,record)
        try:
            while child.poll() is None:
                peak=max(peak,memory(os.getpid()))
                if stop():reason='requested_stop'
                elif clock()-began>=limit_seconds:reason='hard_time_guard'
                elif peak>memory_bytes:reason='owned_tree_memory_guard'
                if reason:terminate_owned(child,identity);break
                sleep(.025)
        except Exception as exc:
            reason='monitor_error';monitor_error=repr(exc)
            try:terminate_owned(child,identity)
            except Exception as terminate_error:monitor_error+='; terminate='+repr(terminate_error)
        if child.poll() is not None:child.wait()
    success=not reason and child.returncode==0 and output.is_file()
    result=dict(status='completed' if success else 'hard_unpublished',reason=None if success else reason or ('child_error' if child.returncode!=0 else 'missing_output'),actual_stage_wall_seconds=clock()-began,peak_parent_children_rss_bytes=peak,pid=child.pid,birth=identity,exit_code=child.returncode,unknown_algorithm_spend=reason is not None,output_ref=None,hard_interface_error=reason=='monitor_error' or (not reason and child.returncode!=0),monitor_error=monitor_error)
    for name,path in [('output_ref',output),('accepted_snapshot_index_ref',directory/'accepted_snapshot_index.json')]:
        if path.exists():result[name]=dict(path=str(path),sha256=digest(path))
    atomic_json(receipt,result);return result
