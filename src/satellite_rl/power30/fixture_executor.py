"""Owned W30 physical-fixture workbench entry, never a learner trajectory."""
import os
from pathlib import Path
import time
from .control import RuntimeControl, validate_data_admission
from .engineering_fixtures import fixture_admission, run_fixtures
from .ledger import CallLedger
from .logging import atomic_json
from satellite_rl.t0_width.workbench import birth, boot_id, now


FIXTURE_LIMITS=dict(phy=1800,kernel=9,coeff=1800,feature=1000,
                    netforward=0,update=0)


def validate_fixture_config(config):
    if (config.get('phase')!='engineering' or config.get('task_kind')!='physical_fixture'
            or config.get('fixture_limits')!=FIXTURE_LIMITS
            or any(k in config for k in ('training','menu','group','terminal_policies'))):
        raise ValueError('Physical fixture cannot carry learner or alternative task selectors')
    receipt=fixture_admission(config)
    if receipt.get('max_feature')!=1000:
        raise ValueError('Root fixture feature cap must equal1000')
    return receipt


def run_owned_fixture_task(config,directory,progress_callback=None,stop_requested=None):
    started=time.monotonic();directory=Path(directory)
    validate_fixture_config(config)
    control=RuntimeControl(config,directory)
    control.validate_owner()
    ledger=CallLedger(config['ledger_root'],config['run_id'],'engineering',
        allowances=dict(FIXTURE_LIMITS,fixture_phy=1800,
                        engineering_interaction=0,engineering_update=0,
                        scientific_interaction=0,scientific_update=0,extra_forward=0))
    control.arm_storage_guard()
    identity=dict(pid=os.getpid(),process_birth=birth(os.getpid()),boot_id=boot_id(),
        entered_UTC=now(),source_sha=config['source_sha'],run_id=config['run_id'])
    atomic_json(directory/'fixture_process_record.json',identity)
    def stop():return bool((stop_requested and stop_requested()) or control.stop_requested())
    def hook(kind,**metadata):
        if kind not in ('phy','coeff','kernel','feature'):
            raise ValueError('Physical fixture cannot enter a model/update/interaction path')
        if stop():raise RuntimeError('Fixture pause/deadline/storage/time stop; no automatic continuation')
        if kind=='phy':ledger.charge('fixture_phy',**metadata)
        ledger.charge(kind,**metadata)
        ledger.flush()
    def notify(stage,**fields):
        value=dict(stage=stage,task_kind='physical_fixture',calls=ledger.snapshot(),**fields)
        if progress_callback:progress_callback(value)
        return value
    try:
        validate_data_admission(config)
        if config['case_record'] not in config['train_cases']:
            raise ValueError('Fixture mother must be an exact qualified frozen train row')
        notify('running')
        result=run_fixtures(config,directory/'fixtures',call_hook=hook,
                            stop_requested=stop,owner_check=control.validate_owner)
        if result.get('status')!='passed':
            raise RuntimeError('No fixture pass; preserve original accounting')
        ledger.close()
        identity.update(returned_after_ledger_close_UTC=now(),
            wrapper_seconds_including_close=time.monotonic()-started,
            excludes_final_receipt_and_process_exit_tail=True,
            outer_actual_exit_observation_required=True,ledger_closed=True)
        atomic_json(directory/'fixture_process_record.json',identity)
        value=notify('completed',fixture_result=result,fixture_passed=True,
                     process_record=identity,root_actual_acceptance_required=True)
        atomic_json(directory/'fixture_executor_result.json',value)
        return value
    except BaseException as exc:
        ledger.flush() # Failed/unknown entry is never normally closed/refunded.
        identity.update(failed_UTC=now(),wrapper_seconds_to_failure=time.monotonic()-started,
                        ledger_closed=not ledger.snapshot()['open'],
                        unknown_spend_not_zero=ledger.snapshot()['open'],error=repr(exc))
        atomic_json(directory/'fixture_process_record.json',identity)
        notify('failed',fixture_passed=False,unknown_spend_not_zero=True)
        raise
