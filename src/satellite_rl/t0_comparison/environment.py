"""Original T0 environment through an explicit algorithm-independent view."""
from copy import deepcopy
from satellite_rl.workbench.config import config_summary
from satellite_rl.workbench.environment import WorkbenchEnv
from .algorithms import validate_contract


def environment_config(config, purpose='science'):
    """Original 85f validator only accepts masked SAC; physical semantics identical.

    This view changes algorithm metadata for constructing the original wrapper,
    never its state, reward, action, scene or physics fields. External task and
    model records must retain the actual comparison algorithm configuration.
    """
    validate_contract(config, 207, (100, 10, 10), purpose)
    view = deepcopy(config)
    view['algorithm'] = 'sac'
    view['training'].update(alpha_mode='fixed', fixed_alpha=.01)
    if purpose == 'engineering':
        # Engineering fixtures have their own finite step budget. They must not
        # inherit formal validation nodes or silently execute formal evaluation.
        view['evaluation'].update(validation_count=0, test_count=0, extra_steps=[],
                                  interval_steps=max(1, view['training']['total_steps']))
    return view


def make_env(config, purpose='science'):
    return WorkbenchEnv(environment_config(config, purpose))


def summary(config, purpose='science'):
    return config_summary(environment_config(config, purpose))
