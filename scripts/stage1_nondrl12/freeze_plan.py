"""Freeze the reviewed ND12 plan against a clean, tested source revision."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from . import data_controller as data
from . import ops
from satellite_rl.workbench.coverage_nondrl import default_nondrl_config


def freeze():
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    if git('status', '--porcelain'):
        raise ValueError('refusing to freeze a dirty implementation')
    commit = git('rev-parse', 'HEAD')
    path = ops.LOCAL / 'run_plan.json'
    plan = data.read(path)
    if plan.get('freeze_status') == 'frozen_for_execution':
        if plan['scientific_commit'] != commit:
            raise ValueError('existing plan belongs to another scientific source')
        return {'already_frozen': True, 'scientific_commit': commit, 'plan_sha256': data.sha(path)}
    if plan.get('freeze_status') != 'implementation_draft_not_launchable':
        raise ValueError('unexpected plan state')
    receipt = data.read(ops.LOCAL / 'preflight' / 'final_code_verification.json')
    if not receipt.get('passed') or receipt.get('source_commit') != commit:
        raise ValueError('matching completed source verification required')
    old = ops.LOCAL / 'preflight' / 'plan_before_freeze.json'
    if not old.exists():
        old.write_bytes(path.read_bytes())
    plan.update(freeze_status='frozen_for_execution', scientific_commit=commit,
                frozen_utc=datetime.now(timezone.utc).isoformat(),
                configuration_by_route={f'{i:02d}': default_nondrl_config(f'{i:02d}') for i in range(1, 13)},
                partition_manifest_sha256=data.sha(data.PARTITION),
                code_verification={'path': str(ops.LOCAL / 'preflight/final_code_verification.json'),
                                   'sha256': data.sha(ops.LOCAL / 'preflight/final_code_verification.json')},
                execution_notes=[
                    'Formal tasks use remote DQN/CUDA for learned updates; local client only uses base Python for SSH/API orchestration.',
                    'Eight bounded CPU workers supply labels/common initial layouts per job; scientific batch/precision are not hot-modified.',
                    'Offline developer probes and fixed ranker validation labels are sub-budgets, not extra teacher budgets.',
                    'Report actual source, CPU affinity overlap, concurrent jobs and noninterruptible solver overshoots; do not rank uncontrolled wall time as pure algorithm speed.',
                    'Full test layouts are hidden from training/selection; creating common test I0 is deterministic preparation only.',
                ])
    data.save(path, plan)
    manifest_path = ops.LOCAL / 'run_manifest.json'
    manifest = data.read(manifest_path)
    manifest.update(status='source_frozen_pending_deployment', scientific_commit=commit,
                    plan_path=str(path), plan_sha256=data.sha(path),
                    partition_manifest_sha256=plan['partition_manifest_sha256'],
                    last_observed_utc=plan['frozen_utc'])
    data.save(manifest_path, manifest)
    return {'scientific_commit': commit, 'plan_sha256': data.sha(path), 'frozen': True}


if __name__ == '__main__':
    print(json.dumps(freeze(), ensure_ascii=False, indent=2))
