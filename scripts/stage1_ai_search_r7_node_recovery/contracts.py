"""One root-pinned operational contract; original science job remains immutable."""
import datetime,hashlib,json,math,os,time
from pathlib import Path

RUN_ID='r7_node2000_20261014_042707_19e62edd'
SCIENCE='523fbc2b1e2480dcc563d302ddcc85317e271aa1'
CORE_SHA='9f187e4cd8122a6fce415b590d0ab0a385724b7d840cc88dd874e34ea936aea5'
PACKAGE_FILES=('__init__.py','compat.py','contracts.py','bootstrap.py','server.py','qualification.py')
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pin(reference):
    path=Path(reference['path'])
    if digest(path)!=reference['sha256']:raise ValueError('pinned recovery byte differs: '+str(path))
    return path
def read(reference):return json.loads(pin(reference).read_bytes())
def require_logical_copy(logical,actual,contract):
    """Transport location remains external to original frozen proof content."""
    pin(actual)
    if logical==actual:return
    mapping=read(contract['logical_reference_transport_map_ref'])
    if mapping.get('schema')!='R7NodeRecoveryLogicalRefTransportMapV1' or logical.get('sha256')!=actual.get('sha256') or mapping.get('remote_ref_by_original_path',{}).get(logical.get('path'))!=actual:
        raise ValueError('original logical metadata ref needs exact root-pinned actual byte transport mapping')
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w',encoding='utf8') as stream:
        json.dump(value,stream,sort_keys=True,ensure_ascii=False,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
def process_identity(pid):
    root=Path('/proc')/str(pid)
    if not root.exists():return None
    try:raw=(root/'stat').read_text()
    except (FileNotFoundError,ProcessLookupError):
        if not root.exists():return None
        raise
    fields=raw[raw.rfind(')')+2:].split()
    return dict(pid=pid,state=fields[0],birth_ticks=int(fields[19]),birth_kind='proc_start_ticks',
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
def require_gone(observation):
    if observation.get('birth_kind')!='proc_start_ticks' or not isinstance(observation.get('birth_ticks'),int) or not observation.get('boot_id'):
        raise ValueError('explicit ticks and boot identity required')
    current=process_identity(observation['pid'])
    if current and current['state'] not in ('Z','X') and current['boot_id']==observation['boot_id'] and current['birth_ticks']==observation['birth_ticks']:
        raise ValueError('original owned process still live')
    return current
def validate(contract_ref,review_ref,*,gone=True):
    contract=read(contract_ref);review=read(review_ref)
    if contract.get('schema')!='R7Node2000SameIDOperationsRecoveryContractV1' or contract.get('run_id')!=RUN_ID or contract.get('scientific_commit')!=SCIENCE:
        raise ValueError('sole original run/source recovery contract')
    if review.get('passed')is not True or review.get('execution_authorized')is not True or review.get('contract_ref')!=contract_ref or review.get('run_id')!=RUN_ID or review.get('operations_only')is not True:
        raise ValueError('root exact operations admission')
    expected=contract['operations_source_files']
    if set(expected)!=set(PACKAGE_FILES):raise ValueError('complete additive operations package source closure')
    for name in PACKAGE_FILES:
        reference=expected[name];pin(reference)
        if Path(reference['path']).resolve()!=Path(__file__).with_name(name).resolve():raise ValueError('actual loaded operations file path differs')
    job=read(contract['original_job_ref']);config=read(contract['original_config_ref'])
    if job.get('run_id')!=RUN_ID or job.get('mode')!='node2000' or job.get('executor_port')!=20474 or job.get('scientific_commit')!=SCIENCE or job.get('case_indices')!=list(range(20)):
        raise ValueError('original one20 task changed')
    if job.get('source_manifest_sha256')!=contract['original_source_manifest_ref']['sha256']:raise ValueError('original714 snapshot differs')
    manifest=read(contract['original_source_manifest_ref'])
    if manifest.get('git_commit')!=SCIENCE or Path(contract['original_source_manifest_ref']['path']).parent.resolve()!=Path(job['source_root']).resolve():raise ValueError('original scientific source epoch/location')
    owner=read(contract['original_owner_ref'])
    if contract['original_owner_ref']['path']!=job['owner_path'] or owner.get('run_id')!=RUN_ID or owner.get('job_sha256')!=contract['original_job_ref']['sha256'] or owner.get('request_hash')!=job['request_hash']:raise ValueError('original unique scientific owner unchanged')
    evidence={str(pin(reference)):reference['sha256'] for reference in contract['original_evidence_refs']}
    if not contract.get('original_evidence_refs') or contract.get('remaining_calls_max')!=dict(prepare=19,T=20,D5=58) or contract.get('spent_calls')!=dict(prepare=1,T=0,D5=2):
        raise ValueError('original W/AI/RULE reuse and remaining cap required')
    if contract.get('original_case0_complete_cold_costs')is not None or contract.get('original_exact_cost_gate')is not False:
        raise ValueError('unknown original complete cost must remain unknown')
    if contract.get('old9016_shutdown_required')is not True:raise ValueError('old scheduler must be gone before worker')
    if gone:
        for observation in [*contract['original_processes'],contract['old9016_process']]:require_gone(observation)
    directory=Path(contract['original_job_ref']['path']).parent
    if directory/'job.json'!=Path(contract['original_job_ref']['path']) or directory/'config.json'!=Path(contract['original_config_ref']['path']):raise ValueError('exact original job/config location')
    if (directory/'case_index.json').exists():raise ValueError('original failure contract expects no completed case index; no replay contract')
    protected=[directory/name for name in ('failure.json','launch.json','runtime_v3_identity.json','worker_deadline_preflight.json')]
    protected += [path for path in (directory/'bundles'/'0').rglob('*') if path.is_file()]
    if any(str(path) not in evidence for path in protected):raise ValueError('complete original failed case and epoch evidence must be pinned')
    amendment=read(contract['amendment_ref']);acceptance=read(contract['root_amendment_acceptance_ref'])
    require_logical_copy(acceptance['amendment_ref'],contract['amendment_ref'],contract)
    if amendment.get('qualification_name')!='node2000_engineering_recovery_quality_and_bounded_spend_v2' or acceptance.get('schema')!='R7RootProspectiveRecoveryAmendmentAcceptanceV2' or acceptance.get('passed')is not True or acceptance.get('prospective_attestation',{}).get('remaining_Random_T_case0_and_all_case1to19_not_started')is not True:
        raise ValueError('explicit prospective named v2 root amendment required before recovery')
    bound=read(contract['failure_epoch_spend_upper_bound_ref'])
    require_logical_copy(acceptance['failure_envelope_ref'],contract['failure_epoch_spend_upper_bound_ref'],contract)
    upper=bound.get('conservative_duration_upper_seconds')
    if acceptance.get('run_id')!=RUN_ID or not isinstance(upper,(int,float)) or isinstance(upper,bool) or not math.isfinite(upper) or upper<0 or acceptance.get('failure_epoch_upper_bound_seconds')!=upper or bound.get('exact_original_parent_cost_seconds')is not None:
        raise ValueError('root audited enclosing failed epoch bound; not exact method cost')
    for stage in ('preparation','AI','RULE'):
        root=directory/'bundles'/'0'/stage
        guard=json.loads((root/'guard.json').read_bytes())
        if guard.get('status')!='completed' or guard.get('exit_code')!=0 or guard.get('unknown_algorithm_spend')is not False:
            raise ValueError('original guarded slot must be closed and reused')
    for i in range(20):
        stages=('RANDOM','T') if i==0 else ('preparation','AI','RULE','RANDOM','T')
        if any((directory/'bundles'/str(i)/stage/'started.json').exists() for stage in stages):raise ValueError('remaining original slot already attempted; no automatic replay')
    return contract,job,config

def fresh_resource(contract):
    resource=read(contract['actual_resource_route_ref'])
    observed=resource.get('observed_unix_seconds')
    if observed is None:observed=datetime.datetime.fromisoformat(resource['observed_utc']).timestamp()
    age=time.time()-observed
    if not -5<=age<=120 or resource.get('ready')is not True or resource.get('root_accepted')is not True or resource.get('ssh_port')!=20474 or resource.get('scientific_commit')!=SCIENCE:
        raise ValueError('fresh root approved actual74 route required')
    return resource
