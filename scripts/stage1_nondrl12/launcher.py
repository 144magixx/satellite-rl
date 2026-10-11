"""Single persistent client: pinned preparation -> registration -> workbench DAG.

The local process survives browser/model exit. Workbench workers own science and
their durable checkpoints; this client never starts a scientific script itself.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

from satellite_rl.coverage.nondrl.runner import worker_lock
from satellite_rl.workbench.runtime import process_start_time
from scripts.stage1_nondrl12 import data_controller as data
from scripts.stage1_nondrl12 import ops
from scripts.stage1_nondrl12.controller import Controller


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_inputs(plan_path, commit):
    plan = data.read(plan_path)
    if (plan.get('experiment_id') != 'S1-ND12' or
            plan.get('scientific_commit') != commit or
            plan.get('freeze_status') != 'frozen_for_execution'):
        raise ValueError('explicit frozen ND12 plan/source required')
    return plan, data.sha(plan_path)


def preparation_recovery(state):
    """At most two same-ID API recoveries; the API rejects a live worker."""
    for row in state['tasks'].values():
        if row['stage'] in ('failed', 'interrupted'):
            if row.get('resume_attempts', 0) >= 2:
                raise RuntimeError('official preparation recovery cap reached: ' + row['run_id'])
            # Save the attempt before publication so an unknown network outcome
            # never grants an unrecorded extra recovery.
            row['resume_attempts'] = row.get('resume_attempts', 0) + 1
            row.setdefault('history', []).append({'at': now(), 'status': row['last_status']})
            data.save(data.STATE, state)
            ops.api(row['port'], '/api/runs/' + row['run_id'] + '/resume', {})
            row['stage'] = 'queued'
            data.save(data.STATE, state)
        elif row['stage'] in ('stopped', 'cancelled'):
            raise RuntimeError('explicitly stopped preparation requires diagnosis: ' + row['run_id'])


def run(plan_path, commit, preparation_hours=12):
    plan, plan_sha = validate_inputs(plan_path, commit)
    directory = ops.LOCAL / 'controller'
    directory.mkdir(parents=True, exist_ok=True)
    launch_dir = directory / 'launcher_lock'
    launch_dir.mkdir(exist_ok=True)
    modules = [Path(__file__), Path(data.__file__), Path(ops.__file__),
               Path(__file__).with_name('controller.py')]
    guards = {str(p): data.sha(p) for p in modules}
    state = {'experiment_id': 'S1-ND12', 'pid': os.getpid(),
             'process_start_time': process_start_time(os.getpid()),
             'scientific_commit': commit, 'plan_sha256': plan_sha,
             'module_sha256': guards, 'started_utc': now(), 'phase': 'starting'}

    def save(phase, **fields):
        state.update(phase=phase, observed_utc=now(), **fields)
        data.save(directory / 'launcher.json', state)
        print(json.dumps({'at': state['observed_utc'], 'phase': phase, **fields}, ensure_ascii=False), flush=True)

    def guard():
        if data.sha(plan_path) != plan_sha or any(data.sha(p) != sha for p, sha in guards.items()):
            raise RuntimeError('persistent client source/plan changed; preserve state and diagnose')
        if (directory / 'launcher.stop.request').exists():
            raise RuntimeError('explicit launcher stop requested')

    with worker_lock(launch_dir):
        try:
            save('uploading_pinned_metadata')
            data.upload_sources()
            expires = time.monotonic() + preparation_hours * 3600
            while True:
                guard()
                prepared = data.prepare_once(commit)
                save('preparing', tasks={key: {k: row.get(k) for k in ('port', 'run_id', 'stage', 'resume_attempts')}
                                          for key, row in prepared['tasks'].items()})
                if prepared['all_completed']:
                    break
                preparation_recovery(prepared)
                if time.monotonic() >= expires:
                    raise TimeoutError('bounded preparation wait expired; do not kill live remote jobs')
                time.sleep(30)
            guard()
            save('registering_canonical_datasets')
            registry = data.register(commit)
            # A separate lock prevents an accidental direct controller invocation
            # from competing with the same DAG.
            with worker_lock(directory):
                controller = Controller(directory, plan, registry)
                save('running_workbench_dag', registry_sha256=data.sha(data.REGISTRY))
                while True:
                    guard()
                    current = controller.tick()
                    save('running_workbench_dag', controller_phase=current['phase'], jobs=len(current['jobs']))
                    if current['phase'] in ('report_pending', 'attention_required', 'stopped'):
                        save(current['phase'], controller_attention=current.get('attention'))
                        return current['phase']
                    time.sleep(20)
        except Exception as error:
            save('attention_required', error=repr(error), traceback=traceback.format_exc())
            return 'attention_required'


def main():
    if (ops.LOCAL / 'controller' / 'execution_placement.json').exists():
        raise RuntimeError('executor placement is active; use scripts.stage1_nondrl12.launcher_pipeline; legacy prepare/register is disabled')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--preparation-hours', type=float, default=12)
    args = parser.parse_args()
    if not 0 < args.preparation_hours <= 24:
        parser.error('preparation hours must be bounded within 24')
    phase = run(args.plan, args.commit, args.preparation_hours)
    raise SystemExit(0 if phase == 'report_pending' else 2)


if __name__ == '__main__':
    main()
