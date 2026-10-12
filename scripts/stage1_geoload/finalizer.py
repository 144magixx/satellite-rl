"""Bounded offline post-finalizer; no SSH/workbench or science execution.

Frozen inputs -> accepted closure -> actual external model receipts -> strict
report -> figures. Interrupted/failed stages require attention, never retries.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
import time
from pathlib import Path
from .strict_audit import AuditError, JobZip, digest


def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(path)


def identity(pid=None):
    import psutil
    process=psutil.Process(pid or os.getpid())
    return dict(pid=process.pid,process_start_time=process.create_time(),actual_command=process.cmdline(),
                executable=process.exe(),cwd=process.cwd(),observed_at=now())


@contextmanager
def exclusive(path):
    """Kernel lock is authoritative; a stale PID/lease never grants ownership."""
    with Path(path).open('a+b') as f:
        if os.name=='nt':
            import msvcrt
            if Path(path).stat().st_size==0:f.write(b'0');f.flush()
            f.seek(0)
            try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            except OSError as e:raise AuditError('post-finalizer kernel lock already held') from e
        else:
            import fcntl
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except OSError as e:raise AuditError('post-finalizer kernel lock already held') from e
        try:yield
        finally:
            if os.name=='nt':f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(f,fcntl.LOCK_UN)


def controller_observation(receipt):
    import psutil
    if not isinstance(receipt.get('pid'),int) or not isinstance(receipt.get('process_start_time'),(int,float)):
        raise AuditError('controller actual PID/birth absent')
    try:actual=identity(receipt['pid'])
    except psutil.NoSuchProcess:return dict(status='original_controller_exited',expected=receipt,observed_at=now())
    if abs(actual['process_start_time']-receipt['process_start_time'])>1e-3:
        return dict(status='pid_reused_original_controller_exited',expected=receipt,actual=actual)
    expected=receipt.get('actual_command',receipt.get('command'))
    if not expected or actual['actual_command']!=expected:raise AuditError('controller PID/birth actual command differs')
    return dict(status='original_controller_alive',expected=receipt,actual=actual)


def freeze(inputs,out):
    folder=out/'inputs';folder.mkdir(exist_ok=True);manifest={}
    for label,source in inputs.items():
        if source is None:continue
        source=Path(source).resolve()
        if not source.is_file():raise AuditError('snapshot missing '+label)
        sha=digest(source);data=source.read_bytes()
        if digest(source)!=sha:raise AuditError('snapshot source changed '+label)
        target=folder/(label+'.json')
        if target.exists():raise AuditError('uncommitted input snapshot exists')
        target.write_bytes(data)
        if digest(target)!=sha:raise AuditError('snapshot digest differs')
        manifest[label]=dict(source_path=str(source),path=str(target),sha256=sha,bytes=len(data))
    save(out/'input_manifest.json',manifest);return manifest


def outputs(out,paths):
    return {str(p.relative_to(out)):dict(sha256=digest(p),bytes=p.stat().st_size) for p in paths if p.is_file()}


def verify_model(plan,state,gpu,export):
    task=state['tasks']['train_mlm'];receipt=state['archives']['train_mlm']
    archive=JobZip(receipt,plan['scientific_commit'])
    try:
        final=archive.manifest['members']['model/final.pt']['sha256'];checkpoint=archive.manifest['members']['model/checkpoint.pt']['sha256']
        for value in (gpu,export):
            if value.get('group')!='S1-G16' or value.get('scientific_commit')!=plan['scientific_commit'] or value.get('run_id')!=task['run_id'] or value.get('verified') is not True or value.get('status')!='passed':raise AuditError('actual model receipt source/identity differs')
        if gpu.get('final_sha256')!=final or gpu.get('checkpoint_sha256')!=checkpoint or export.get('model_sha256')!=final or export.get('checkpoint_sha256')!=checkpoint or export.get('train_zip_sha256')!=receipt['sha256']:
            raise AuditError('actual model receipt SHA differs')
        if gpu.get('requested_interpreter')!=plan['worker_interpreter'] or gpu.get('requested_interpreter_realpath')!=gpu.get('interpreter') or gpu.get('interpreter_samefile') is not True or gpu.get('port')!=task['port'] or gpu.get('device')!='cuda' or not all(gpu.get(k) is True for k in ('cuda_available','strict_load','strict_model_load','strict_state_dict','parameters_match_checkpoint','worker_exited','member_hashes_verified')):
            raise AuditError('actual DQN CUDA/strict endpoint receipt absent')
        if gpu.get('seed')!=51001 or gpu.get('updates',gpu.get('update'))!=10000 or gpu.get('from_scratch') is not True or gpu.get('parameter_updates_performed')!=0 or gpu.get('new_evaluations_performed')!=0:
            raise AuditError('actual model seed/update/bounded audit differs')
        if digest(export['model_path'])!=final or digest(export['metadata_path'])!=export['metadata_sha256']:raise AuditError('actual exported small model/metadata SHA differs')
        metadata=read(export['metadata_path'])
        if metadata.get('sha256')!=final or metadata.get('checkpoint_sha256')!=checkpoint or metadata.get('run_id')!=task['run_id'] or metadata.get('scientific_commit')!=plan['scientific_commit']:raise AuditError('actual model export metadata identity differs')
        if export.get('checkpoint_zip_path')!=receipt['local_path'] or export.get('checkpoint_member')!='model/checkpoint.pt':raise AuditError('actual checkpoint canonical reference differs')
        return dict(status='accepted',verified=True,gpu_receipt=gpu,export_receipt=export)
    finally:archive.close()


def run(inputs,output,*,hooks=None):
    """One bounded five-stage pass. Hooks are for synthetic tests only."""
    hooks=hooks or {};out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True);path=out/'postfinalizer_state.json'
    with exclusive(out/'postfinalizer.lock'):
        actual=identity();save(out/'lease.json',dict(**actual,lock='held_by_current_process'))
        state=read(path) if path.exists() else dict(group='S1-G16',status='running',steps={},history=[],created_at=now())
        state['process']=actual;active=None
        def persist():state['updated_at']=now();save(path,state)
        try:
            if state['status'] in ('needs_attention','blocked'):
                state['history'].append(dict(at=now(),event='automatic_retry_refused'));persist();return state
            if any(s.get('status')=='running' for s in state['steps'].values()):raise AuditError('interrupted partial stage; no automatic rerun')
            if state['steps'].get('snapshot',{}).get('status')=='completed':
                manifest=read(out/'input_manifest.json')
                if {k:str(Path(v).resolve()) for k,v in inputs.items() if v is not None}!={k:r['source_path'] for k,r in manifest.items()}:raise AuditError('resume input paths differ')
                for label,r in manifest.items():
                    if digest(r['path'])!=r['sha256'] or digest(r['source_path'])!=r['sha256']:raise AuditError('frozen/source input changed; new output required')
                for step in state['steps'].values():
                    for rel,r in step.get('outputs',{}).items():
                        if digest(out/rel)!=r['sha256']:raise AuditError('committed output changed '+rel)
            else:
                active='snapshot';state['steps'][active]=dict(status='running',started_at=now());persist();manifest=freeze(inputs,out)
                if not {'plan','state','registry','controller'}.issubset(manifest):raise AuditError('mandatory snapshot input absent')
                plan=read(manifest['plan']['path']);science=read(manifest['state']['path'])
                if plan.get('group')!='S1-G16' or science.get('group')!='S1-G16' or science.get('plan_sha256')!=manifest['plan']['sha256']:raise AuditError('plan/state binding differs')
                if not science.get('compute_and_archives_complete') or not science.get('tasks') or set(science.get('archives',{}))!=set(science['tasks']) or any(t.get('stage')!='completed' for t in science['tasks'].values()):raise AuditError('science tasks/ZIP exports not terminal and closed')
                observed=hooks.get('controller',controller_observation)(read(manifest['controller']['path']));save(out/'controller_observation.json',observed)
                if observed['status']=='original_controller_alive':raise AuditError('original controller remains alive; fresh terminal snapshot required')
                state['steps'][active].update(status='completed',completed_at=now(),outputs=outputs(out,[out/'input_manifest.json',out/'controller_observation.json']));persist()
            plan=read(manifest['plan']['path']);science=read(manifest['state']['path']);gate=science.get('gate',{}).get('passed')
            if type(gate) is not bool:raise AuditError('teacher gate unknown')
            from .archive_acceptance import audit
            from .report import build as report
            from .publication import build as publication
            for active in ('closure','model','report','publication'):
                if state['steps'].get(active,{}).get('status')=='completed':continue
                state['steps'][active]=dict(status='running',started_at=now());persist()
                if active=='closure':
                    value=hooks.get('closure',audit)(plan,science,read(manifest['registry']['path']));save(out/'closure.json',value)
                    if value.get('status')!='accepted' or value.get('issues'):raise AuditError('archive closure not accepted; see closure.json')
                    paths=[out/'closure.json']
                elif active=='model':
                    if gate:
                        if not {'gpu','export'}.issubset(manifest):raise AuditError('actual GPU/model export receipts missing')
                        value=hooks.get('model',verify_model)(plan,science,read(manifest['gpu']['path']),read(manifest['export']['path']))
                        if value.get('verified') is not True:raise AuditError('actual model receipt not accepted')
                    else:
                        if any(k=='train_mlm' or k.startswith('teacher_') for k in science['tasks']):raise AuditError('gatefail branch unexpectedly trained')
                        value=dict(status='science_stopped',verified=False,reason='teacher_gate_failed; no_model_expected')
                    save(out/'model_validation.json',value);paths=[out/'model_validation.json']
                elif active=='report':
                    closure=read(out/'closure.json')
                    if gate:closure['model_gpu_validation']=read(out/'model_validation.json')
                    value=hooks.get('report',report)(manifest['plan']['path'],manifest['state']['path'],out/'report',closure)
                    if value.get('formal_ready') is not True or value.get('status')!='formal_ready' or value.get('issues'):raise AuditError('strict report not formal_ready')
                    paths=list((out/'report').rglob('*'))
                else:
                    value=hooks.get('publication',publication)(out/'report',out/'figures')
                    if value.get('synthetic') or not (out/'figures'/'manifest.json').is_file():raise AuditError('formal publication manifest absent')
                    paths=list((out/'figures').rglob('*'))
                state['steps'][active].update(status='completed',completed_at=now(),outputs=outputs(out,paths));persist()
            state.update(status='complete',analysis_verified=True,delivery_verified=False,requires_human_publication_and_delivery=True,
                         scientific_commit=plan['scientific_commit'],learning_branch='trained_endpoint_audited' if gate else 'science_stopped_teacher_gate_failed',completed_at=now());persist();return state
        except Exception as error:
            if active and state['steps'].get(active,{}).get('status')=='running':state['steps'][active].update(status='needs_attention',error=str(error),failed_at=now())
            state.update(status='needs_attention',delivery_verified=False,error=str(error));state['history'].append(dict(at=now(),step=active,error=str(error)));persist();return state
        finally:save(out/'lease.json',dict(**actual,lock='released',finished_at=now()))


def wait_until_terminal(inputs,output,timeout=21600,poll_seconds=60,*,probe=None,sleeper=time.sleep):
    """Optional bounded read-only wait before the one-shot immutable snapshot.

    This may wait for root's actual endpoint receipts, but never creates them.
    Neither missing evidence nor controller errors trigger science recovery.
    """
    if timeout<=0 or not 1<=poll_seconds<=60:raise AuditError('wait bounds require positive timeout and poll 1..60s')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    with exclusive(out/'postfinalizer_wait.lock'):
        path=out/'wait_state.json'
        if path.exists() and read(path).get('status')=='needs_attention':return False
        actual=identity();started=time.monotonic();checks=0
        probe=probe or controller_observation
        while True:
            checks+=1;reasons=[];gate=None
            try:
                science=read(inputs['state']);gate=science.get('gate',{}).get('passed')
                if not science.get('compute_and_archives_complete'):reasons.append('computation/archive_receipts_not_terminal')
                if science.get('controller_error'):reasons.append('controller_error_requires_root_attention')
                if gate is True:
                    for label in ('gpu','export'):
                        if not inputs.get(label) or not Path(inputs[label]).is_file():reasons.append('awaiting_actual_'+label+'_receipt')
                elif gate is not False:reasons.append('teacher_gate_unknown')
                for label in ('plan','registry','controller'):
                    if not inputs.get(label) or not Path(inputs[label]).is_file():reasons.append('missing_'+label+'_input')
                if not reasons and probe(read(inputs['controller']))['status']=='original_controller_alive':reasons.append('terminal_controller_still_alive')
            except Exception as error:reasons.append('read_unknown: '+str(error))
            elapsed=time.monotonic()-started
            value=dict(group='S1-G16',status='ready' if not reasons else 'waiting',process=actual,lock='held_by_current_process',
                       checks=checks,elapsed_seconds=elapsed,timeout_seconds=timeout,reasons=reasons,observed_at=now())
            if elapsed>=timeout and reasons:value['status']='needs_attention';value['automatic_retry']=False
            save(path,value)
            if value['status'] in ('ready','needs_attention'):return value['status']=='ready'
            sleeper(min(poll_seconds,max(.001,timeout-elapsed)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for label in ('plan','state','registry','controller'):parser.add_argument('--'+label,required=True)
    for label in ('gpu','export'):parser.add_argument('--'+label)
    parser.add_argument('--wait-until-terminal',action='store_true');parser.add_argument('--wait-timeout',type=float,default=21600)
    parser.add_argument('--poll-seconds',type=float,default=60)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    inputs={label:getattr(args,label) for label in ('plan','state','registry','controller','gpu','export')}
    if args.wait_until_terminal and not wait_until_terminal(inputs,args.output,args.wait_timeout,args.poll_seconds):return 2
    # Gatefail does not demand nonexistent model receipts provided for a potential gatepass.
    if read(inputs['state']).get('gate',{}).get('passed') is False:
        inputs['gpu']=None;inputs['export']=None
    result=run(inputs,args.output)
    print(json.dumps(dict(status=result['status'],error=result.get('error'),output=args.output)))
    return 0 if result['status']=='complete' else 2


if __name__=='__main__':raise SystemExit(main())
