"""Frozen T0 scientific fields; explicit bounded engineering exceptions."""
from copy import deepcopy

T0_FIELDS = {'schema_version': 1,
 'network': {'kind': 'transformer',
             'transformer_layout': 'tsac_natural_context',
             'hidden_sizes': [256, 256],
             'activation': 'relu',
             'dropout': 0.0,
             'cnn_channels': [16, 32],
             'kernel_size': 3,
             'embedding_dim': 128,
             'num_heads': 8,
             'num_layers': 2,
             'feedforward_dim': 512,
             'sac_policy_structure': 'factorized',
             'sac_critic_structure': 'additive',
             'temporal_mode': 'none',
             'temporal_window': 16,
             'graph_mode': 'none'},
 'state': {'features': ['beam',
                        'power',
                        'interference',
                        'occupancy',
                        'natural_global_context'],
           'demand_visibility': 'known',
           'context_level': 'compact'},
 'reward': {'preset': 'legacy_repaired',
            'terms': [{'id': 'si_delta',
                       'weight': 1,
                       'scale': 1,
                       'timing': 'step'}]},
 'environment': {'profile': 'europe_v1',
                 'power_budget_w': 3000,
                 'basic_fraction': 0.1,
                 'action_mode': 'legacy_clip',
                 'action_mask': 'joint_feasible',
                 'power_levels_w': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]},
 'training': {'seed': 51001,
              'scene_order': 'shuffle_each_cycle',
              'scene_order_seed': 51001,
              'total_steps': 150000,
              'batch_size': 256,
              'actor_lr': 1e-05,
              'critic_lr': 0.0001,
              'alpha_lr': 0.0001,
              'gamma': 0.99,
              'tau': 0.005,
              'replay_capacity': 20000,
              'warmup_steps': 2000,
              'update_every': 1,
              'gradient_steps': 1,
              'initial_alpha': 0.01,
              'target_entropy': 6.9,
              'alpha_mode': 'fixed',
              'fixed_alpha': 0.01,
              'gradient_clip': 10,
              'rollout_steps': 256,
              'ppo_epochs': 4,
              'gae_lambda': 0.95,
              'clip_ratio': 0.2,
              'entropy_coef': 0.01,
              'value_coef': 0.5,
              'eps_start': 1,
              'eps_end': 0.05,
              'eps_decay_steps': 5000,
              'target_update_interval': 500,
              'checkpoint_interval': 10000,
              'log_interval': 50,
              'device': 'cuda'}}

ENGINEERING_LIMITS = {'batch_size':256, 'warmup_steps':2000,
    'target_update_interval':500, 'rollout_steps':256, 'ppo_epochs':4,
    'total_steps':2048}

def canonical_config(config, purpose='science'):
    if purpose not in ('science', 'engineering'):
        raise ValueError('Explicit science or engineering purpose required')
    if config.get('algorithm') not in ('sac', 'ppo', 'dqn'):
        raise ValueError('Unknown T0 comparison algorithm')
    if set(config) - (set(T0_FIELDS) | {'name', 'algorithm', 'evaluation', 'telemetry'}):
        raise ValueError('Unknown fields outside the Direct T0 contract')
    for section, expected in T0_FIELDS.items():
        actual = config.get(section)
        if section != 'training':
            if actual != expected:
                raise ValueError('Frozen T0 field differs: '+section)
            continue
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError('Training keys differ from frozen T0')
        for key, value in expected.items():
            candidate = actual[key]
            if isinstance(candidate, bool) and not isinstance(value, bool):
                raise ValueError('Boolean is not a numeric training parameter: '+key)
            if purpose == 'engineering' and key in ENGINEERING_LIMITS:
                if isinstance(candidate, bool) or not isinstance(candidate, int):
                    raise ValueError('Engineering override must be an integer: '+key)
                lower = 0 if key == 'warmup_steps' else 1
                if not lower <= candidate <= ENGINEERING_LIMITS[key]:
                    raise ValueError('Engineering override exceeds bounded limit: '+key)
            elif candidate != value:
                raise ValueError('Frozen training field differs: '+key)
    if purpose == 'science':
        evaluation = config.get('evaluation', {})
        expected_eval = {'validation_count':20,'test_count':40,'validation_seed':274000,
            'test_seed':275000,'interval_steps':10000,'action_mode':'stochastic',
            'repeats':3,'extra_steps':[0,2000,5000]}
        if evaluation != expected_eval:
            raise ValueError('Frozen Direct evaluation fields differ')
    return deepcopy(config)
