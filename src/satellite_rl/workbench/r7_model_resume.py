"""Root-reconciled known checkpoint same-phase-ID recovery; no job mutation."""
from pathlib import Path
from .storage import read_json,digest,now
from .coverage_ai_search import pin
from .geoload_runner import exclusive_lock
from satellite_rl.training.run_monitor import atomic_json

def validate_recovery(job,contract,review,index,completion,owner):
    if job.get('mode') not in ('train2000','train6000') or contract.get('schema')!='R7KnownCheckpointSameIDRecoveryV1' or contract.get('run_id')!=job['run_id'] or contract.get('model_identity')!=job['model_identity'] or contract.get('scientific_commit')!=job['scientific_commit']:raise ValueError('exact known learner same-ID recovery only')
    if review.get('passed')is not True or review.get('contract_sha256')!=contract['self_external_sha256'] or review.get('run_id')!=job['run_id'] or review.get('scientific_commit')!=job['scientific_commit']:raise ValueError('root exact recovery admission')
    if contract.get('original_checkpoint_index_sha256')!=index['_actual_sha256'] or contract.get('checkpoint_ref')!=index.get('latest') or contract.get('optimizer_completed_update')!=completion.get('completed_optimizer_update') or index.get('updates')!=completion.get('completed_optimizer_update') or completion.get('recoverable_state_published')is not True:raise ValueError('known completed optimizer state must be durable; unknown replay forbidden')
    if owner.get('last_completed_update')!=index['updates'] or owner.get('last_completed_checkpoint_ref',{}).get('sha256')!=index['latest']['sha256'] or owner.get('segment_runs',{}).get(job['mode'])!=job['run_id'] or owner.get('model_identity')!=job['model_identity'] or owner.get('model_dir')!=job['model_dir']:raise ValueError('same global learner owner/model directory/completed steps')
    if not index['updates']<contract['end_update']<=6000 or not index['updates']<=owner['science_updates_reserved']<=6000 or contract.get('repeat_completed_optimizer_steps')is not False:raise ValueError('unchanged cumulative6000 reserved budget')
    evidence=read_json(pin(contract['clock_evidence_ref']),{})
    if evidence.get('verified')is not True or evidence.get('unknown_tail_accounted')is not True or evidence.get('checkpoint_file_sha')!=index['latest']['sha256'] or not index.get('charged_active_seconds',0)<=evidence.get('charged_active_seconds',-1)<28800:raise ValueError('root hashed clock interval/debit including full IO')
    return True

def resume(app,run_id,body):
    if set(body)!=set(('recovery_contract_ref','root_review_ref')):raise ValueError('only exact root recovery contract/review refs')
    directory=app.store.job_dir(run_id);job=read_json(directory/'job.json',{})
    from .r7_deadline import require
    require(job['model_inputs'],job['archive_root'],job['experiment_id'],job['mode'],job['references']['plan'],dict(path=str(directory/'job.json'),sha256=digest(directory/'job.json')))
    with app.scheduler.lock,exclusive_lock(directory/'model_recovery.lock'):
        contract=read_json(pin(body['recovery_contract_ref']),{});review=read_json(pin(body['root_review_ref']),{})
        # External raw file SHA is supplied in memory; no impossible self-hash field.
        contract={**contract,'self_external_sha256':body['recovery_contract_ref']['sha256']}
        if contract.get('original_job_sha256')!=digest(directory/'job.json') or contract.get('original_config_sha256')!=digest(directory/'config.json'):raise ValueError('original job/config bytes must remain unchanged')
        status=read_json(directory/'status.json',{});launch=read_json(directory/'launch.json',{})
        from .runtime import pid_alive
        if status.get('stage') not in ('failed','interrupted','stopped') or pid_alive(status.get('pid'),launch.get('process_start_time')):raise ValueError('original phase must be stopped and actual worker gone')
        index=read_json(directory/'checkpoint_index.json',{});index['_actual_sha256']=digest(directory/'checkpoint_index.json');pin(index['latest'])
        completion=read_json(directory/'optimizer_completion.json',{})
        from .r7_model_train import MODEL_OWNER_ROOT
        from satellite_rl.coverage.ai_search_r6.features import record_sha
        owner=read_json(MODEL_OWNER_ROOT/(record_sha(job['model_identity'])+'.json'),{})
        validate_recovery(job,contract,review,index,completion,owner)
        config=read_json(directory/'config.json',{})
        if contract['end_update']!=config['training']['end_update']:raise ValueError('recovery cannot extend original phase cap')
        from .storage import validate_id
        attempt=validate_id(contract['recovery_id']);history=directory/'model_recovery_history'/attempt
        if history.exists():raise ValueError('original recovery intent exists; GET only, no blind replay')
        history.mkdir(parents=True)
        for name in ('status.json','failure.json','launch.json','checkpoint_index.json','optimizer_completion.json','activity_clock.json','stop.request'):
            p=directory/name
            if p.is_file():(history/name).write_bytes(p.read_bytes())
        receipt=dict(run_id=run_id,model_identity=job['model_identity'],checkpoint_ref=contract['checkpoint_ref'],original_checkpoint_index_sha256=contract['original_checkpoint_index_sha256'],
                     recovery_clock=dict(evidence_path=contract['clock_evidence_ref']['path'],evidence_sha=contract['clock_evidence_ref']['sha256'],clock_path=str(directory/'activity_clock.json')))
        atomic_json(directory/'root_recovery.json',receipt);atomic_json(directory/'root_recovery_pin.json',dict(sha256=digest(directory/'root_recovery.json')))
        atomic_json(history/'root_recovery_contract.json',dict(body=body,original_job_sha256=contract['original_job_sha256'],original_config_sha256=contract['original_config_sha256'],no_model_or_optimizer_call_in_reconciliation=True))
        (directory/'stop.request').unlink(missing_ok=True) # only after root explicit continue/deadline admission
        status.update(stage='queued',pid=None,error=None,heartbeat_at=now(),root_model_recovery=body);atomic_json(directory/'status.json',status)
        return dict(run_id=run_id,status=status,original_job_config_owner_unchanged=True)
