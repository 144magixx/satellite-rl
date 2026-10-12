"""Opt-in G16 b520 client bridge: replace only pipeline.export_job.

Never deploys, stops processes or releases compute workers. Root supplies an
exact handoff lease after old client exit and kernel-lock release verification.
Unknown archive/packer errors persist needs_attention: retrying the original
20-second pipeline loop never repeats remote packing/download automatically.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

SCIENCE='495bff976c59c40ddb33de6b5b37b20a894a1347'
CONTROLLER='b520e576ce5f8952ef89a87136575df7c1f1b478'
PLAN_SHA='0b520943dec9e82f78ba44e0579cc3c50f25f2518761b519c51414bf5096e806'
SCIENCE_PLAN_SHA='1c41e7446665b11c5b9b7f6adc9c5f5fa70885e9c71989cb2104c0ef8a03620e'
ORIGINAL={
 'pipeline.py':'38711fbb4ca6d29811cce0c9829cabddd6a4b4de81f35fde0eba5d0597e7070b',
 'archive.py':'8ffe82a7c2658c2acba1ffaaa3f3997434b7886a0b58949a0d85c7bf4d5ed279',
 'ops.py':'4c310d139ce4f7eefdc68368fbc0a744fe1549a5cd0a41c4c528037db26d04a4',
 'controller.py':'4afc22dc211e19d23745542988dabcff0ff22dddbd7d585f1b18fb693642c35c'}
TRANSPORT_SHA='394761637607a61810d163984a72221fc95ae59673a72f19f06c686576363a84'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.partial');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)


def guard(plan_path,state_path,lease_path,*,directory=None,old_alive=None):
    directory=Path(directory or Path(__file__).parent)
    for name,expected in ORIGINAL.items():
        if sha(directory/name)!=expected:raise ValueError('original b520 source SHA differs: '+name)
    if sha(directory/'archive_transport.py')!=TRANSPORT_SHA:raise ValueError('candidate transport source SHA differs')
    if sha(plan_path)!=PLAN_SHA:raise ValueError('frozen run plan SHA differs')
    plan=read(plan_path);state=read(state_path);lease=read(lease_path)
    science_plan=Path(plan_path).parent/'source'/'science_plan.json'
    if sha(science_plan)!=SCIENCE_PLAN_SHA or any(plan.get('science_plan_refs',{}).get(str(port),{}).get('sha256')!=SCIENCE_PLAN_SHA for port in (20424,20473)):
        raise ValueError('frozen scientific plan SHA differs')
    if plan.get('group')!='S1-G16' or plan.get('scientific_commit')!=SCIENCE or plan.get('authorized') is not True:raise ValueError('scientific plan/source differs')
    if state.get('group')!='S1-G16' or state.get('plan_sha256')!=PLAN_SHA or state.get('scientific_commit')!=SCIENCE:raise ValueError('state frozen plan/source differs')
    if state.get('archives') or state.get('all_computation_completed') or state.get('compute_and_archives_complete'):raise ValueError('handoff must precede archival phase')
    active=[row for row in state.get('tasks',{}).values() if row.get('stage')!='completed']
    if not active or any(row.get('stage')!='running' for row in active):raise ValueError('all active scientific workers must still be running at handoff')
    if lease.get('schema')!='g16_transport_handoff_v1' or lease.get('status')!='handoff_verified' or lease.get('group')!='S1-G16':raise ValueError('root verified lease required')
    if lease.get('old_controller_pid')!=24220 or abs(lease.get('old_controller_birth',0)-1791790345.992857)>.01:raise ValueError('old client exact identity differs')
    if lease.get('old_controller_exit_verified') is not True or lease.get('state_lock_release_verified') is not True or lease.get('legacy_export_excluded') is not True:raise ValueError('client exit/OS lock/export exclusion not verified')
    if lease.get('controller_commit')!=CONTROLLER or lease.get('scientific_commit')!=SCIENCE or lease.get('plan_sha256')!=PLAN_SHA:raise ValueError('lease source/plan differs')
    if lease.get('state_sha256')!=sha(state_path) or Path(lease.get('state_path','')).resolve()!=Path(state_path).resolve():raise ValueError('lease state snapshot/path differs')
    if lease.get('bridge_sha256')!=sha(__file__) or lease.get('transport_sha256')!=TRANSPORT_SHA:raise ValueError('lease bridge/transport SHA differs')
    if old_alive is None:
        import psutil
        def old_alive(pid,birth):
            try:return abs(psutil.Process(pid).create_time()-birth)<.01
            except psutil.NoSuchProcess:return False
    if old_alive(lease['old_controller_pid'],lease['old_controller_birth']):raise ValueError('exact old controller still alive')
    return dict(group='S1-G16',controller_commit=CONTROLLER,scientific_commit=SCIENCE,
        plan_sha256=PLAN_SHA,science_plan_sha256=SCIENCE_PLAN_SHA,state_sha256=sha(state_path),lease_sha256=sha(lease_path),
        original_sources=ORIGINAL,bridge_sha256=sha(__file__),transport_sha256=TRANSPORT_SHA,
        only_replacement='pipeline.export_job',remote_workers_stopped=False)


def adapter(transport,journal_dir):
    journal_dir=Path(journal_dir)
    def export_job(port,run_id,destination):
        if port not in (20424,20473):raise ValueError('unauthorized export endpoint')
        # Use transport binding validation before run_id becomes a filename.
        import re
        if not re.fullmatch('[A-Za-z0-9_-]+',run_id):raise ValueError('unsafe export task')
        entry=journal_dir/(str(port)+'_'+run_id+'.json')
        if entry.exists() and read(entry).get('status')=='needs_attention':
            raise RuntimeError('archive transport needs_attention; root audit required; no blind retry')
        destination=Path(destination)
        remote_pin=destination.with_suffix('.remote.receipt.json')
        old_pin=destination.with_suffix('.receipt.json')
        pin=remote_pin if remote_pin.exists() else old_pin if old_pin.exists() else None
        try:
            if destination.exists() and pin is None:raise ValueError('existing local ZIP has no exact receipt; refusing repack')
            save(entry,dict(status='started',port=port,run_id=run_id,
                existing_receipt=str(pin) if pin else None,client_pid=os.getpid(),
                bridge_sha256=sha(__file__),transport_sha256=TRANSPORT_SHA,observed_unix=time.time()))
            result=transport.one_export(port,run_id,destination,existing_receipt=pin,
                ownership_confirmed=True,window=4 << 20)
            save(entry,dict(status='verified',port=port,run_id=run_id,receipt=result,
                bridge_sha256=sha(__file__),transport_sha256=TRANSPORT_SHA,
                scope='engineering_transport_only',observed_unix=time.time()))
            return result
        except Exception as error:
            save(entry,dict(status='needs_attention',port=port,run_id=run_id,error=str(error),
                scientific_failure=False,remote_packer_retry_allowed=False,
                remote_workers_stopped=False,observed_unix=time.time()))
            raise RuntimeError('archive transport needs_attention: '+str(error)) from error
    return export_job


def install(pipeline,transport,journal_dir):
    """The only assignment to the imported original pipeline module."""
    if getattr(pipeline.export_job,'_g16_transport_bridge',False):raise ValueError('bridge already installed')
    before=dict(vars(pipeline));replacement=adapter(transport,journal_dir)
    replacement._g16_transport_bridge=True
    pipeline.export_job=replacement
    if any(getattr(pipeline,name) is not value for name,value in before.items() if name!='export_job'):
        raise ValueError('unexpected pipeline identity mutation')
    return before['export_job']


def run(plan_path,state_path,lease_path,journal_dir):
    provenance=guard(plan_path,state_path,lease_path)
    from . import pipeline, archive_transport
    # Acquire and release exactly the same OS lock. Original run owns it for its
    # whole loop; another client racing in later is refused by original run.
    with pipeline.exclusive_lock(Path(state_path).with_suffix('.lock')):pass
    journal=Path(journal_dir);journal.mkdir(parents=True,exist_ok=True)
    save(journal/'bridge_start.json',{**provenance,'pid':os.getpid(),'started_unix':time.time()})
    install(pipeline,archive_transport,journal/'exports')
    return pipeline.run(plan_path,state_path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--state',required=True)
    parser.add_argument('--lease',required=True);parser.add_argument('--journal',required=True)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if args.check_only:print(json.dumps(guard(args.plan,args.state,args.lease),indent=2))
    else:run(args.plan,args.state,args.lease,args.journal)


if __name__=='__main__':main()
