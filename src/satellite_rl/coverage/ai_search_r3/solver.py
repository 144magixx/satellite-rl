"""Owned native subprocess with write-ahead accounting and hard RSS/watchdog."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import signal

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,allow_nan=False,sort_keys=True),encoding='utf8');os.replace(tmp,path)

def own_rss(pid):
    if sys.platform!='win32':
        try:return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[21])*os.sysconf('SC_PAGE_SIZE')
        except FileNotFoundError:return 0
    import psutil
    try:return psutil.Process(pid).memory_info().rss
    except psutil.NoSuchProcess:return 0

def process_identity(pid):
    if sys.platform!='win32':
        try:
            fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
            rss=int(fields[21])*os.sysconf('SC_PAGE_SIZE');children=Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
            return dict(pid=pid,birth=fields[19],own_rss=rss,rss=rss+sum(process_identity(int(c))['rss'] for c in children),children=children)
        except FileNotFoundError:return dict(pid=pid,birth=None,own_rss=0,rss=0,children=[])
    import psutil
    try:
        p=psutil.Process(pid);children=p.children(recursive=True)
        own=p.memory_info().rss
        return dict(pid=pid,birth=p.create_time(),own_rss=own,rss=own+sum(c.memory_info().rss for c in children),children=[c.pid for c in children])
    except psutil.NoSuchProcess:return dict(pid=pid,birth=None,own_rss=0,rss=0,children=[])

def case_memory(native_pid,parent_pid=None):
    parent_rss=own_rss(os.getpid() if parent_pid is None else parent_pid)
    native=process_identity(native_pid)
    # The case worker's unrelated children are excluded; only its own RSS and this native tree belong here.
    return dict(parent_own_RSS=parent_rss,native_tree_RSS=native['rss'],case_tree_RSS=parent_rss+native['rss'],native=native)

def cleanup_owned(owned):
    for pid,birth in reversed(list(owned.items())):
        current=process_identity(pid)
        if current['birth'] is None or current['birth']!=birth:continue
        try:os.kill(pid,signal.SIGTERM if sys.platform=='win32' else signal.SIGKILL)
        except (ProcessLookupError,PermissionError):pass

def run_native(problem,directory,vendor_dir,seconds,nodes,watchdog,stop=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);token=uuid.uuid4().hex
    input_path=directory/(token+'.npz');output=directory/(token+'.json');wal=directory/(token+'.wal.json');log=directory/(token+'.log')
    problem.save(input_path)
    command=[sys.executable,'-m','satellite_rl.coverage.ai_search_r3.native_worker','--input',str(input_path),'--output',str(output),
        '--vendor',str(vendor_dir),'--seconds',str(seconds),'--nodes',str(nodes)]
    record=dict(call_id=token,state='started_unknown_spend',time_limit=seconds,node_limit=nodes,watchdog_limit=watchdog,
        complete_seed_independent_audit=True,start_wall=time.time(),command=command,problem_fingerprint=problem.fingerprint(),
        selected_slot_ids=problem.slots.tolist(),stage=problem.stage,temporary_CSR_transport=True)
    atomic_json(wal,record);started=time.monotonic();peak=0;case_peak=0;reason=None;owned={}
    with log.open('w',encoding='utf8') as stream:
        proc=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy())
        try:
            identity=process_identity(proc.pid);owned[proc.pid]=identity['birth'];record['owned_process']=identity;atomic_json(wal,record)
            while proc.poll() is None:
                memory=case_memory(proc.pid);current=memory['native'];peak=max(peak,current['rss']);case_peak=max(case_peak,memory['case_tree_RSS'])
                for child in current['children']:
                    info=process_identity(int(child))
                    if info['birth'] is not None:owned[int(child)]=info['birth']
                if current['birth'] is not None and current['birth']!=identity['birth']:reason='owned_pid_birth_changed'
                if case_peak>2147483648:reason='case_parent_plus_native_tree_RSS_guard'
                if time.monotonic()-started>watchdog:reason='watchdog'
                if stop is not None and stop():reason='requested_stop'
                if reason:break
                time.sleep(.025)
        except BaseException as exc:
            record.update(monitor_exception=type(exc).__name__,owned_process_births=owned,exceptional_CSR_retained=str(input_path))
            atomic_json(wal,record);raise
        finally:
            cleanup_owned(owned)
            if proc.poll() is None:proc.kill()
            proc.wait()
    elapsed=time.monotonic()-started
    result=json.loads(output.read_text(encoding='utf8')) if output.exists() else dict(status='native_unavailable',vector=None,
        value_valid=False,message=reason or 'native_failed',native_elapsed=None,mip_gap=None,mip_dual_bound=None,mip_node_count=None)
    result.update(wall_elapsed=elapsed,watchdog=reason=='watchdog',fallback_reason=reason,peak_tree_RSS=peak,
        native_log_ref=str(log),call_wal_ref=str(wal),exit_code=proc.returncode,problem_fingerprint=problem.fingerprint(),
        native_peak_tree_RSS=peak,case_parent_plus_native_peak_tree_RSS=case_peak,owned_process_births=owned,
        selected_slot_ids=problem.slots.tolist(),stage=problem.stage)
    result['native_output_ref']=str(output) if output.exists() else None
    result['native_output_sha256']=hashlib.sha256(output.read_bytes()).hexdigest() if output.exists() else None
    cleanup_owned(owned)
    record.update(state='completed',wall_elapsed=elapsed,peak_tree_RSS=peak,case_parent_plus_native_peak_tree_RSS=case_peak,
        owned_process_births=owned,result_ref=str(output),exit_code=proc.returncode,reason=reason)
    atomic_json(wal,record)
    # Sparse matrix transport is transient, original input builder+solver vector are retained.
    if proc.returncode==0 and not reason:
        input_path.unlink();record['temporary_CSR_deleted_after_original_output_preserved']=True
    else:record['exceptional_CSR_retained']=str(input_path)
    atomic_json(wal,record)
    return result
