"""Narrow read-only-RSS operational epoch and same-ID labels74 recovery."""
import errno,os,hashlib,json,shutil,time,sys
from pathlib import Path
from .storage import read_json,digest,now
from .coverage_ai_search import pin
from .geoload_runner import exclusive_lock
from satellite_rl.training.run_monitor import atomic_json

RUN='r7_labels_20261013_234807_312cdce7'
SCIENCE='13702ff5d4c2aac8c559ddf79bdd2d8202dec45f'
FAILED=[295,297]
REMAINING=list(range(299,576,2))

def gone_or_zombie(root):
    try:
        raw=(root/'stat').read_text();return raw[raw.rfind(')')+2:].split()[0] in ('Z','X')
    except (FileNotFoundError,ProcessLookupError):return True

def original_process_alive(observation,proc=Path('/proc')):
    kind=observation.get('birth_kind')
    if kind not in ('proc_start_ticks','unix_boot_seconds') or not observation.get('boot_id'):raise ValueError('explicit original PID birth units and boot_id required')
    current_boot=(Path(proc)/'sys/kernel/random/boot_id').read_text().strip()
    if current_boot!=observation['boot_id']:return False
    if kind=='unix_boot_seconds':
        from .runtime import pid_alive
        return pid_alive(observation['pid'],observation['birth'])
    try:
        raw=(Path(proc)/str(observation['pid'])/'stat').read_text();parts=raw[raw.rfind(')')+2:].split()
        return int(parts[19])==observation['birth'] and parts[0] not in ('Z','X')
    except (FileNotFoundError,ProcessLookupError):return False

def rss_tree(pid,proc=Path('/proc'),page_size=None,samples=None):
    """Same kernel RSS meaning/caps; only confirmed proc exit ESRCH is ignored."""
    page_size=page_size or os.sysconf('SC_PAGE_SIZE');queue=[pid];seen=set();total=0
    while queue:
        current=queue.pop()
        if current in seen:continue
        seen.add(current);root=Path(proc)/str(current)
        try:
            lines=(root/'status').read_text().splitlines();rows=[line.split() for line in lines if line.startswith('VmRSS:')]
            if rows:
                if len(rows)!=1 or len(rows[0])!=3 or rows[0][2]!='kB':raise ValueError('malformed VmRSS')
                resident=int(rows[0][1])*1024;source='status.VmRSS'
            else:
                fields=(root/'statm').read_text().split()
                if len(fields)!=7 or any(not x.isdecimal() for x in fields):raise ValueError('malformed statm')
                resident=int(fields[1])*page_size;source='statm.resident'
            if resident<0:raise ValueError('negative RSS')
            total+=resident
            if samples is not None and source=='statm.resident':samples.append(dict(pid=current,source=source,rss_bytes=resident,at=now()))
            queue.extend(int(x) for x in (root/'task'/str(current)/'children').read_text().split())
        except FileNotFoundError:continue
        except ProcessLookupError:
            if not gone_or_zombie(root):raise
            if samples is not None:samples.append(dict(pid=current,source='confirmed_proc_exit_ESRCH',at=now()))
            continue
    return total

def validate(job,metadata):
    contract=read_json(pin(metadata['contract_ref']),{});review=read_json(pin(metadata['root_review_ref']),{})
    if job.get('run_id')!=RUN or job.get('scientific_commit')!=SCIENCE or job.get('request_id')!='r7_labels137_20474_attempt1':raise ValueError('one original74 labels ID only')
    if contract.get('schema')!='R7Labels74OperationalRecoveryV1' or contract.get('run_id')!=RUN or contract.get('scientific_commit')!=SCIENCE or contract.get('failed_case_indices')!=FAILED or contract.get('remaining_case_indices')!=REMAINING or contract.get('preserved_case_indices')!=list(range(1,295,2)):raise ValueError('exact147+2+139 original partition')
    if review.get('passed')is not True or review.get('run_id')!=RUN or review.get('contract_sha256')!=metadata['contract_ref']['sha256'] or review.get('operational_commit')!=contract.get('operational_commit'):raise ValueError('root exact operational source/contract acceptance')
    original=contract['original_job']
    for key,value in original.items():
        if job.get(key)!=value:raise ValueError('original scientific job field changed: '+key)
    if contract.get('new_stage_attempts_max')!=dict(prepare=139,T=139,D5=139) or contract.get('repeat_original_attempts')is not False or contract.get('label_denominator')!=576:raise ValueError('remaining original budget and unchanged denominator')
    if not contract.get('accounting_addendum_ref'):raise ValueError('root frozen conservative accounting addendum required')
    pin(contract['accounting_addendum_ref'])
    if set(contract['operations_source_refs'])!=set(('module','bootstrap','server')):raise ValueError('exact three operational source members')
    for ref in contract['operations_source_refs'].values():pin(ref)
    if digest(__file__)!=contract['operations_source_refs']['module']['sha256']:raise ValueError('actual narrow operations module differs')
    source=Path(job['source_root']);manifest=read_json(source/'source_manifest.json',{})
    if digest(source/'source_manifest.json')!=job['source_manifest_sha256'] or manifest.get('git_commit')!=SCIENCE or len(manifest['files'])!=698:raise ValueError('original698 immutable scientific snapshot')
    if any(digest(source/name)!=sha for name,sha in manifest['files'].items()):raise ValueError('original science member changed')
    owner=read_json(pin(contract['original_owner_ref']),{})
    if owner.get('run_id')!=RUN or owner.get('request_hash')!=job['request_hash'] or owner.get('job_sha256')!=contract['original_job_ref']['sha256']:raise ValueError('original unique semantic owner differs')
    return contract

def resume(app,body):
    metadata={k:body[k] for k in ('contract_ref','root_review_ref')};directory=app.store.job_dir(RUN)
    with app.scheduler.lock,exclusive_lock(directory/'operations_recovery.lock'):
        job=read_json(directory/'job.json',{});contract=validate(job,metadata);status=read_json(directory/'status.json',{})
        sidecar=directory/'operations_recovery.json'
        if sidecar.exists():
            if read_json(sidecar,{})!=metadata:raise ValueError('another operations recovery already owns this original ID')
            return dict(run_id=RUN,status=status,idempotent=True,new_science_dispatch=False)
        from .runtime import pid_alive
        launch=read_json(directory/'launch.json',{})
        if status.get('stage')!='failed' or pid_alive(status.get('pid'),launch.get('process_start_time')):raise ValueError('actual original worker must be failed and gone')
        if digest(directory/'job.json')!=contract['original_job_ref']['sha256'] or digest(directory/'case_index.json')!=contract['original_case_index_ref']['sha256']:raise ValueError('original job/index no longer matches read-only diagnosis')
        from .r7_resources import observe
        actual=observe(app.store.archive_root)
        if actual['foreign_workers']:raise ValueError('foreign science owner present')
        for observation in contract['original_pid_births']:
            if original_process_alive(observation):raise ValueError('original owned child still alive')
        history=directory/'operations_recovery_history'/contract['repair_id']
        if history.exists():raise ValueError('original operation attempt already exists without published metadata; no blind replay')
        history.mkdir(parents=True)
        for name in ('job.json','status.json','failure.json','launch.json','case_index.json','parallel_ledger.json'):
            if (directory/name).is_file():shutil.copyfile(directory/name,history/name)
        # Preserve exact original job/config/request/owner bytes. The only new
        # contract lives in a separate sidecar and launch/bootstrap metadata.
        atomic_json(sidecar,metadata)
        status.update(stage='queued',pid=None,error=None,heartbeat_at=now(),operations_recovery=metadata,scientific_source_unchanged=True);atomic_json(directory/'status.json',status)
        return dict(run_id=RUN,status=status,idempotent=False,new_science_dispatch=False)

def install_worker(job,metadata,directory):
    """Memory-only operational hooks. Frozen137 files and algorithm functions stay."""
    contract=validate(job,metadata);directory=Path(directory)
    from . import r4_monitor_repair,r7_parallel
    r4_monitor_repair.rss_tree=rss_tree
    index=read_json(directory/'case_index.json',{})
    existing={int(i) for i in index}
    if not set(contract['preserved_case_indices'])<=existing or existing-set(contract['preserved_case_indices'])-set(FAILED)-set(REMAINING):raise ValueError('original case partition changed')
    for i in contract['preserved_case_indices']:
        old=read_json(pin(index[str(i)]),{})
        if old['case_index']!=i or old['mode']!='labels' or old['parent_job_sha256']!=contract['original_job_ref']['sha256']:raise ValueError('original complete case SHA/identity not reusable')
        pin(old['result_ref']);pin(old['result_ref']['cost_endpoint_ref'])
    for i in FAILED:
        if str(i) in index:continue
        entry=contract['failed_cases'][str(i)]
        for reference in entry['evidence_refs'].values():pin(reference)
        target=directory/'operations_recovery_history'/contract['repair_id']/'failed_cases'/str(i);target.mkdir(parents=True,exist_ok=True)
        disposition=dict(schema='R7OperationalFailedLabelDispositionV1',run_id=RUN,input_case_index=i,identity=entry['identity'],
                         scientific_commit=SCIENCE,qualified=False,nontrivial=False,capability=False,cost_closed=False,status='infrastructure_interrupted',
                         original_attempts_retried=False,known_unknown_accounting=entry['accounting'],original_evidence_refs=entry['evidence_refs'],
                         root_contract_ref=metadata['contract_ref'],root_review_ref=metadata['root_review_ref'],operational_commit=contract['operational_commit'])
        failed=target/'failed_label.json';atomic_json(failed,disposition);result=target/'case_result.json'
        atomic_json(result,dict(schema='R7OperationalFailedCaseResultV1',case_index=i,mode='labels',parent_job_sha256=contract['original_job_ref']['sha256'],result_ref=dict(path=str(failed),sha256=digest(failed))))
        index[str(i)]=dict(path=str(result),sha256=digest(result));atomic_json(directory/'case_index.json',index)
    for i in REMAINING:
        if str(i) not in index and (directory/'case_workers'/str(i)/'started.json').exists():raise ValueError('remaining case already has an attempted original call; no replay')
    # Existing original parallel implementation launches only queue indices.
    # Rewrite only case-worker invocation to install the same RSS observer in
    # that new process. Scientific W/T/D argv and guards remain unchanged.
    original_popen=r7_parallel.subprocess.Popen
    def owned_popen(command,*args,**kwargs):
        tokens=list(command)
        if 'satellite_rl.workbench.r7_case_worker' in tokens:
            parent=tokens[tokens.index('--parent-dir')+1]
            if Path(parent).resolve()!=directory.resolve():raise ValueError('case bootstrap outside original job')
            replacement=[tokens[0],'-B','-u',str(pin(contract['operations_source_refs']['bootstrap'])),'--mode','case','--contract',metadata['contract_ref']['path'],'--contract-sha256',metadata['contract_ref']['sha256'],'--review',metadata['root_review_ref']['path'],'--review-sha256',metadata['root_review_ref']['sha256'],*tokens[tokens.index('--parent-dir'):]]
            command[:]=replacement
        return original_popen(command,*args,**kwargs)
    r7_parallel.subprocess.Popen=owned_popen
    return contract
