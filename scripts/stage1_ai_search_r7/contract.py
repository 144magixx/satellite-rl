"""The v2 R7 contract; metadata-only, never opens aircraft/heldout arrays."""
from copy import deepcopy
import re

GROUP = 'S1-AI17-R7'
DESIGN_SHA = '603e0a1e2e075bf714702ba4b1b68adef78639c2cc25529633da5d133996a916'
ROOT_DESIGN_SHA = 'eaede816edfd2c3163bfa4567381a313fa9cd5be9ad73a759c813e5988875e6a'
PORTS = (20473, 20474)
TRAIN_PORT = 20474
REQUEST_BPS = 21600000
TRAIN_INDICES = list(range(576))
DEV_INDICES = list(range(20))
PHASES = {
    'data_train_dev': dict(role='train_development', cases=596, builds=596, prepare=0, T=0, D5=0, updates=0),
    'labels': dict(role='train', cases=576, builds=0, prepare=576, T=576, D5=576, updates=0),
    'train_setup': dict(role='train', cases=None, builds=0, prepare=0, T=0, D5=0, updates=0),
    'train2000': dict(role='train', cases=None, builds=0, prepare=0, T=0, D5=0, updates=2000),
    'node2000': dict(role='development', cases=20, builds=0, prepare=20, T=20, D5=60, updates=0),
    'train6000': dict(role='train', cases=None, builds=0, prepare=0, T=0, D5=0, updates=4000),
    'fixed6000_development': dict(role='development', cases=20, builds=0, prepare=20, T=20, D5=60, updates=0),
    'validation': dict(role='validation', cases=20, builds=20, prepare=20, T=20, D5=60, updates=0),
    'test': dict(role='test', cases=40, builds=40, prepare=40, T=40, D5=120, updates=0),
}
FIRST_PHASES = ('data_train_dev', 'labels', 'train_setup', 'train2000', 'node2000')
SEALED_PHASES = tuple(p for p in PHASES if p not in FIRST_PHASES)
PRIOR = {'data_train_dev': 'shared_interface', 'labels': 'data_train_dev',
         'train_setup': 'labels', 'train2000': 'train_setup', 'node2000': 'train2000',
         'train6000': 'node2000', 'fixed6000_development': 'train6000',
         'validation': 'fixed6000_development', 'test': 'validation'}

def require_ref(ref):
    if not isinstance(ref, dict) or not isinstance(ref.get('path'), str) or not ref['path'] or not re.fullmatch('[0-9a-f]{64}', ref.get('sha256', '')):
        raise ValueError('actual externally pinned reference required')
    return deepcopy(ref)

def qualified_indices(indices):
    if not isinstance(indices, list) or not 519 <= len(indices) <= 576 or indices != sorted(set(indices)) or any(type(i) is not int or i not in TRAIN_INDICES for i in indices):
        raise ValueError('519..576 original ordered train indices required')
    return list(indices)

def sealed_phase_interface(phase):
    if phase not in SEALED_PHASES: raise ValueError('not a sealed successor')
    return dict(phase=phase, **deepcopy(PHASES[phase]), dispatch_allowed=False,
                prior_gate=PRIOR[phase], requires_separate_root_exact_plan=True)

def total_budget():
    return {key: sum(p[key] for p in PHASES.values()) for key in ('builds','prepare','T','D5','updates')}
