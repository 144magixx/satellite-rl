"""Locked known-scene Direct matrix; no optimizer updates or independent-test claim."""
from pathlib import Path
import json
import torch
from satellite_rl.t0_width.data import read_json_ref, file_sha, load_published_layout
from satellite_rl.t0_width.logging import EvidenceLogs, atomic_json, content_sha
from satellite_rl.t0_width.runner import evaluate_cases, BackgroundPolicy, RecoveryAccountingRequired, seed_all


def jobs_for(passed, node):
    if not isinstance(passed,list) or len(set(passed))!=len(passed) or set(passed)-{'E','LP'}:
        raise ValueError('locked passed candidate list invalid')
    if node != (150000 if passed else 50000):
        raise ValueError('known matrix must match gate-selected common budget')
    models=['B']+passed+['HIST'] if passed else ['B','E','LP','HIST']
    return [(m,r) for m in models for r in range(3)]+[('random',r) for r in range(3)]+[('fixed',0),('greedy',0)]


def validate_lock(config):
    if config.get('task_kind')!='known_evaluation' or config.get('phase')!='known_scenes' or 'training' in config:
        raise ValueError('known evaluation selector only')
    # source_sha is the original geometry/scientific CORE identity, never Git.
    # Git ancestry lives in the two explicit commit fields and is root locked.
    lock=read_json_ref(config['terminal_lock_ref'])
    if (lock.get('schema')!='W27KnownMatrixLockV1' or lock.get('root_approved') is not True
            or lock.get('experiment_id')!='S2-W27' or lock.get('run_id')!=config['run_id']
            or lock.get('node')!=config['node'] or lock.get('manifest_ref')!=config['known_manifest_ref']
            or lock.get('policies')!=config['policies'] or lock.get('source_SHA')!=config['source_sha']
            or lock.get('data_manifest_SHA')!=config['data_manifest_sha']
            or config.get('training_source_sha')!=config['source_sha']
            or not config.get('training_source_commit') or not config.get('evaluation_source_commit')
            or lock.get('training_source_commit')!=config['training_source_commit']
            or lock.get('evaluation_source_commit')!=config['evaluation_source_commit']
            or lock.get('not_independent_test') is not True or lock.get('all_continuation_decisions_locked') is not True):
        raise ValueError('root final known-matrix lock required')
    decision=read_json_ref(lock['decision_ref'])
    if decision.get('root_approved') is not True or decision.get('passed_variants')!=lock['passed_variants']:
        raise ValueError('locked all-candidate gate decision mismatch')
    jobs=jobs_for(lock['passed_variants'],config['node'])
    methods={m for m,r in jobs}-{'random','fixed','greedy'}
    if set(config['policies'])!=methods or config.get('budget_episodes')!=40*len(jobs):
        raise ValueError('exact dynamic model set/episode budget required')
    for method,p in config['policies'].items():
        if set(p)!={'policy_ref','schema','config_SHA','source_SHA','data_manifest_SHA','env_step'} or p['env_step']!=config['node']:
            raise ValueError('policy provenance contract required')
        if method!='HIST' and (p['source_SHA']!=config['training_source_sha'] or p['data_manifest_SHA']!=config['data_manifest_sha']):
            raise ValueError('new model original frozen training provenance mismatch')
    history_ref=config.get('historical_training_plan_ref')
    if not history_ref or lock.get('historical_training_plan_ref')!=history_ref:
        raise ValueError('original W26 historical training plan SHA reference required')
    old=read_json_ref(history_ref)
    if old.get('schema')!='W26AuthorizedPlanV1' or old.get('authorized') is not True:raise ValueError('original authorized W26 plan required')
    configs=[task['config'] for task in old['tasks'].values() if task.get('mode')=='train' and task['config'].get('group')=='S10']
    if len(configs)!=1:raise ValueError('original exact history S10 config missing')
    original=configs[0];hist=config['policies']['HIST']
    if (hist['config_SHA']!=content_sha(original) or hist['source_SHA']!=original['source_sha']
            or hist['data_manifest_SHA']!=original['data_manifest_sha']):raise ValueError('history must bind original actual W26 config/source/data')
    return lock,jobs


def load_agents(config):
    from .agent import SearchAgent,build_config
    from satellite_rl.t0_width.agent import WidthAgent,build_config as old_config
    device=config.get('device','cuda'); seed_all(51001); agents={}
    for method,p in config['policies'].items():
        ref=p['policy_ref']
        if file_sha(ref['path'])!=ref['sha256']: raise ValueError('locked model file SHA differs')
        policy=torch.load(ref['path'],map_location=device,weights_only=False)
        agent=WidthAgent(old_config('S10',device),device) if method=='HIST' else SearchAgent(build_config(method,device),device)
        if (policy.get('schema')!=p['schema'] or p['schema']!=agent.schema
                or policy.get('env_step')!=p['env_step'] or policy.get('config_sha')!=p['config_SHA']
                or policy.get('source_sha')!=p['source_SHA'] or policy.get('data_manifest_sha')!=p['data_manifest_SHA']):
            raise ValueError('strict policy node/schema/config/source/data binding differs')
        agent.actor.load_state_dict(policy['actor'],strict=True); agents[method]=agent
    for name in ('random','fixed','greedy'): agents[name]=BackgroundPolicy(name,device)
    return agents


def close_final_committed_wal(wal,task_sha,budget,cursor):
    if not wal.exists():return True
    saved=torch.load(wal,map_location='cpu',weights_only=False);context=saved['identity'].get('context',{})
    if (cursor==budget and context.get('task_sha')==task_sha and context.get('matrix_cursor')==budget-1
            and saved.get('stage')=='committed' and saved.get('done') is True):
        wal.unlink();return True
    return False


def run_known_task(config,run_dir,progress_callback=None,stop_requested=None):
    lock,jobs=validate_lock(config) # No known arrays opened before all-model root lock.
    manifest=read_json_ref(config['known_manifest_ref']); rows=manifest.get('rows',[])
    if (manifest.get('schema')!='W26PublishedDatasetV1' or manifest.get('root_approved') is not True
            or manifest.get('role')!='test' or len(rows)!=40 or len({r['case_id'] for r in rows})!=40):
        raise ValueError('exact old W26 40 known mothers required')
    unlock=config['w26_original_unlock']
    if unlock.get('manifest_SHA')!=config['known_manifest_ref']['sha256'] or unlock.get('manifest_ref')!=config['known_manifest_ref']:
        raise ValueError('original W26 known-data unlock partition differs')
    original=read_json_ref(unlock['receipt_ref'])
    if original.get('schema')!='W26ConfirmationUnlockV1' or original.get('root_approved') is not True or original.get('manifest_SHA')!=unlock['manifest_SHA']:
        raise ValueError('historical W26 unlock required alongside independent new W27 matrix lock')
    d=Path(run_dir);d.mkdir(parents=True,exist_ok=True);task_sha=content_sha(config)
    cursor_path=d/'known_cursor.json';wal=d/'known_wal.pt'
    progress=read_json_ref({'path':str(cursor_path),'sha256':file_sha(cursor_path)}) if cursor_path.exists() else dict(task_sha=task_sha,cursor=0,log_offsets={name:0 for name in EvidenceLogs.NAMES})
    if progress['task_sha']!=task_sha or not 0<=progress['cursor']<=config['budget_episodes']: raise ValueError('matrix resume identity/cursor differs')
    def accounting(reason):
        result=dict(stage='needs_root_recovery_accounting',task_sha=task_sha,completed_episodes=progress['cursor'],unknown_spend_not_zero=True,preserved_original_files=True,reason=reason)
        atomic_json(d/'recovery_accounting_required.json',result);return result
    for name,offset in progress['log_offsets'].items():
        path=d/'logs'/(name+'.jsonl.gz')
        if (path.stat().st_size if path.exists() else 0)!=offset:return accounting('uncommitted unique evidence suffix: '+name)
    logs=EvidenceLogs(d/'logs',progress['log_offsets'])
    def notify(stage,**fields):
        value=dict(stage=stage,phase='known_scenes',completed_episodes=progress['cursor'],budget=config['budget_episodes'],**fields)
        if progress_callback:progress_callback(value)
        return value
    def loader(row):
        from satellite_rl.shared_interface.stages import UniformStage2Input
        from satellite_rl.shared_interface.spherical_gaussian import PHYSICS
        case,layout,authority=load_published_layout(row,role='test',confirmation_unlock=unlock)
        return UniformStage2Input(case,layout,authority,PHYSICS)
    try:
        agents=load_agents(config)
        for mother,row in enumerate(rows):
            rotate=mother%len(jobs);ordered=jobs[rotate:]+jobs[:rotate]
            for slot,(method,repeat) in enumerate(ordered):
                cursor=mother*len(jobs)+slot
                if cursor<progress['cursor']:continue
                if stop_requested is not None and stop_requested():return notify('stopped')
                if wal.exists():
                    prior=torch.load(wal,map_location='cpu',weights_only=False)
                    context=prior['identity'].get('context',{})
                    if context.get('task_sha')!=task_sha:return accounting('WAL task identity differs')
                    if context.get('matrix_cursor',cursor)<cursor:
                        if prior.get('stage')!='committed' or prior.get('done') is not True:return accounting('prior intent unknown spend; preserve WAL')
                        wal.unlink()
                def commit(_):
                    progress['cursor']=cursor+1;progress['log_offsets']=logs.barrier();atomic_json(cursor_path,progress)
                    notify('known_evaluation',mother=mother,method=method,repeat=repeat)
                _,finished=evaluate_cases(agents[method],[row],'S10',loader=loader,seed_base=275000,repeats=1,
                    mother_offset=mother,repeat_offset=repeat,node=config['node'],logs=logs,phase='known_'+method,
                    stop_requested=stop_requested,wal_path=wal,row_committed=commit,
                    wal_context=dict(task_sha=task_sha,matrix_cursor=cursor))
                if not finished:return notify('stopped_during_evaluation')
        if not close_final_committed_wal(wal,task_sha,config['budget_episodes'],progress['cursor']):
            return accounting('final committed cursor has an unclosed mismatched WAL')
        return notify('completed')
    except RecoveryAccountingRequired as exc:return accounting(str(exc))
    finally:logs.close()
