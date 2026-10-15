"""Finite lane/stage planning, with no network or child-process launch API.

This preparation is deliberately not a root controller. A final accepted owned
backend must supply exact source pins, runtime epoch, hard kill/reap and direct
exit facts before any plan can execute.
"""
from pathlib import Path
from .data_manifest import PARTITIONS, require_all_B
from .runtime import exclusive, atomic_bytes, json_bytes, require_execution, development_schedule


def stage_plan():
    B = [{'task_id': f'B:{i}', 'recipe_index': i, 'stage': 'B',
          'kind':'B','phase':'B',
          'additional_stage_reservations':{'audit':120},
          'outer_seconds': 120, 'stop_new_work_seconds': 108,
          'W_calls': 1, 'RULE_calls': 1, 'D5_calls': 1, 'T_calls': 0}
         for indices in PARTITIONS.values() for i in indices]
    train = [{'task_id': f'train:{i}:full', 'recipe_index': i, 'arm': 'full_SHC',
              'kind':'teacher','phase':'train_labels',
              'additional_stage_reservations':{'audit':120},
              'stage': 'trajectory', 'outer_seconds': 120}
             for i in PARTITIONS['train']]
    calibration = [{'task_id': f'cal:{i}:{arm}', 'recipe_index': i, 'arm': arm,
                    'kind':'cal','phase':'calibration',
                    'stage': 'trajectory', 'outer_seconds': 120, 'serial_group': f'cal:{i}'}
                   for i in PARTITIONS['calibration']
                   for arm in ('AI', 'strong_causal_baseline', 'full_SHC')]
    development = [{'task_id': f"dev:{row['mother']}:{row['repeat']}:{arm}",
                    'kind':'dev','phase':'development',
                    'recipe_index': 576 + row['mother'], 'arm': arm, 'stage': 'trajectory',
                    'outer_seconds': 120, 'serial_group': f"dev:{row['mother']}",
                    'repeat': row['repeat'], 'serial_position': pos}
                   for row in development_schedule() for pos, arm in enumerate(row['arms_serial'])]
    data=[{'task_id':f'data:{t["recipe_index"]}','recipe_index':t['recipe_index'],'kind':'data',
           'phase':'data_qualification','stage':'B','outer_seconds':120} for t in B]
    engineering=[{'task_id':f'engineering:{j}','kind':'engineering','phase':'engineering',
       'fixture_index':j,'stage':'trajectory','outer_seconds':120,'additional_stage_reservations':{'audit':120}}
        for j in range(8)]
    offline=[{'task_id':'offline:'+t['task_id'],'recipe_index':t['recipe_index'],'kind':'offline',
        'phase':'offline_pairing','repeat':t.get('repeat'),'stage':'audit','outer_seconds':120,
        'purpose':'calibration' if t['kind']=='cal' else 'development'}
        for t in calibration+development if t['arm']=='full_SHC']
    return {'schema': 'R14FiniteStagePlanV1', 'dispatch_allowed': False,'data':data,
            'engineering_tasks':engineering,'offline_tasks':offline,
            'fit_task':{'task_id':'sole_fit','kind':'fit','phase':'fit_baseline','stage':'fit_baseline','outer_seconds':60},
            'veto_task':{'task_id':'sole_veto','kind':'veto','phase':'calibration_veto','stage':'fit_baseline','outer_seconds':60},
            'B': B, 'training': train, 'calibration': calibration, 'development': development,
            'fits': 1, 'fit_prerequisite': 'all140 independently qualified B plus96 qualified traces',
            'calibration_prerequisite': 'single immutable fitted tree and train baseline',
            'development_prerequisite': 'all24 calibration complete then one aggregate veto freeze',
            'trajectory_total': 348, 'SHC_kernels_max': 11136, 'SHC_internal_flows_max': 89088,
            'engineering': {'fixtures': 8, 'kernels': 16, 'flows': 128, 'W_RULE_D5_T_fit': 0},
            'one_owned_persistent_backend_per_lane': True,
            'each_measured_request_fresh_parent': True,
            'shared_time_ledger_and_storage_domain_required': True}


class OnceIntent:
    """Exclusive durable dispatch/fit intent; never clears failed/unknown work."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def claim(self, task_id, exact_request, config):
        require_execution(config)
        if not task_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.' for c in task_id):
            raise ValueError('safe exact once task ID')
        target = self.directory / (task_id + '.intent.json')
        with exclusive(self.directory / '.once.lock'):
            if target.exists():
                raise ValueError('already reserved; unknown outcome is not retry authority')
            atomic_bytes(target, json_bytes({'schema': 'R14DispatchIntentV1', 'task_id': task_id,
                         'request': exact_request, 'root_admission_ref': config['root_admission_ref'],
                         'runtime_epoch_ref': config['runtime_epoch_ref'], 'retry': 0}))
        return target


def fit_gate(manifest, B_receipts, train_receipts):
    require_all_B(manifest, B_receipts)
    if set(train_receipts) != set(PARTITIONS['train']):
        raise ValueError('all fixed96 train trajectories required before sole fit')
    for i, receipt in train_receipts.items():
        if (receipt.get('qualified') is not True or receipt.get('complete_cost_exit') is not True
                or receipt.get('recipe_index') != i or receipt.get('teacher_action') != 'cap32'
                or receipt.get('B_and_all_accepted_positions_independently_audited') is not True):
            raise ValueError('incomplete real label source; no fit')
    return True
