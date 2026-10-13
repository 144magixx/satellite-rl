"""Original T0 environment through an explicit algorithm-independent view."""
from copy import deepcopy
from satellite_rl.workbench.config import config_summary
from satellite_rl.workbench.environment import WorkbenchEnv
from .algorithms import validate_contract


def environment_config(config):
    """Original 85f validator only accepts masked SAC; physical semantics identical.

    This view changes algorithm metadata for constructing the original wrapper,
    never its state, reward, action, scene or physics fields. External task and
    model records must retain the actual comparison algorithm configuration.
    """
    validate_contract(config, 207, (100, 10, 10))
    view = deepcopy(config)
    view['algorithm'] = 'sac'
    view['training'].update(alpha_mode='fixed', fixed_alpha=.01)
    return view


def make_env(config):
    return WorkbenchEnv(environment_config(config))


def summary(config):
    return config_summary(environment_config(config))
