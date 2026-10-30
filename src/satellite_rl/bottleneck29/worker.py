"""One admitted lane. Persistent boundaries and original-trajectory-only replay."""
import hashlib
import pickle
import shutil
from pathlib import Path

from . import core
from .ledger import Halt, Ledger, atomic_pickle, digest

SCIENCE_LIMITS = dict(phy=18825, phy_prefix=3000, phy_minimum5=14325,
                      phy_fine_power=1500, kernel=15, seconds=900)
ENGINEERING_LIMITS = dict(phy=256, phy_prefix=256, phy_minimum5=256,
                          phy_fine_power=256, kernel=4, seconds=900)


def run_lane(config, load_case, active_started=None):
    """load_case(mother_record) -> (validated UniformStage2Input, saved indices).

    No policy or layout generation entry exists. Deployment supplies the audited
    input adapter and binds its source alongside core and this module.
    """
    lane = config['lane']
    if type(lane) is not int or not 0 <= lane < 4:
        raise Halt('invalid lane')
    root = Path(config['root'])
    ledger = Ledger(root, SCIENCE_LIMITS, config['binding'],active_started=active_started)
    ledger.storage_limit=min(config['max_incremental_bytes'],512*1024**2)
    stop_paths = [Path(p) for p in config['stop_paths']]
    def boundary():
        ledger.boundary()
        if any(p.exists() for p in stop_paths):
            raise Halt('user pause or lane stop')
        # Constant number of stat calls, never recursive scans.
        if shutil.disk_usage(root).free < config['minimum_free_bytes']:
            raise Halt('storage recovery margin')
        if ledger.storage_bytes()>=ledger.storage_limit:raise Halt('registered lane storage allowance')
    def meter(category):
        def wrapped(kind, call_id, operation):
            boundary()
            return ledger.meter('phy_'+category if kind == 'phy' else kind, call_id, operation)
        return wrapped
    ledger.begin()
    try:
        mothers = config['mothers']
        if len(mothers) != 20 or len({m['case_id'] for m in mothers}) != 20:
            raise Halt('exact ordered twenty distinct mothers required')
        for index, mother in enumerate(mothers):
            if index % 4 != lane:
                continue
            boundary()
            stage_input, actions = load_case(mother)
            case = mother['case_id']
            context = core.prepare_context(stage_input, meter('prefix'), case+'/kernel')
            if len(actions) != context.N:
                raise Halt('complete original trajectory required')
            snapshot = core.initial_snapshot(context)
            for position in core.prefix_positions(context.N):
                while snapshot['beam_idx'] < position:
                    k = snapshot['beam_idx']
                    snapshot = core.replay_original_action(context, snapshot, actions[k],
                        meter('prefix'), case+'/prefix/'+str(k))['next_snapshot']
                state_id = case+'/'+str(position)
                snapshot_path = root/'snapshots'/(hashlib.sha256(state_id.encode()).hexdigest()+'.pkl')
                saved_snapshot=ledger.artifact(state_id+'/snapshot',snapshot_path,snapshot)
                core.restore_snapshot(context,saved_snapshot)
                snapshot_sha = digest(snapshot_path)
                tables = core.candidate_specs(context, snapshot, actions[position])
                physical_cache = {}
                for table, specs in tables.items():
                    for sequence, spec in enumerate(specs):
                        boundary()
                        row_id = state_id+'/'+table+'/'+str(sequence)
                        result_path = root/'rows'/(hashlib.sha256(row_id.encode()).hexdigest()+'.pkl')
                        saved=ledger.artifact(row_id,result_path)
                        if saved is not None:
                            # A committed row contains all numbers and binds snapshot/source.
                            if saved['snapshot_sha'] != snapshot_sha or saved['spec'] != spec or saved['binding'] != config['binding']:
                                raise Halt('durable candidate binding mismatch')
                            physical_cache[(spec['start'],spec['count'],spec['power_w'])] = row_id
                            continue
                        key = (spec['start'],spec['count'],spec['power_w'])
                        if spec['valid'] and key in physical_cache:
                            result = dict(status='cache', reference=physical_cache[key], physical_calls=0)
                        else:
                            result = core.evaluate_candidate(context, snapshot, spec, meter(table), row_id)
                            # Full N numeric arrays remain; remove duplicate prefix allocation.
                            result.pop('next_snapshot', None)
                            if spec['valid']:
                                physical_cache[key] = row_id
                        ledger.artifact(row_id,result_path,dict(row_id=row_id,snapshot_sha=snapshot_sha, spec=spec,
                            binding=config['binding'], result=result))
                        if digest(snapshot_path) != snapshot_sha:
                            raise Halt('immutable snapshot changed')
                ledger.artifact('cursor',root/'cursor.pkl', dict(case_id=case, prefix=position, status='state_complete'),replace=True)
        ledger.artifact('completed',root/'completed.pkl', dict(status='completed', counts=ledger.counts()))
    except BaseException as error:
        ledger.artifact('stopped',root/'stopped.pkl', dict(status='stopped', reason=repr(error),
            counts=ledger.counts(), unresolved=ledger.unknown()),replace=True)
        raise
    finally:
        ledger.finish()
