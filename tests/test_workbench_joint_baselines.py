"""Learned head choices must not invalidate or change original rule baselines."""
from copy import deepcopy
import unittest

from satellite_rl.workbench.baselines import BaselinePolicy
from satellite_rl.workbench.config import default_config, validate_config
from satellite_rl.workbench.environment import WorkbenchEnv
from satellite_rl.workbench.runner import _baseline_environment_config


class JointBaselineTests(unittest.TestCase):
    def test_all_four_structures_keep_rule_baseline_actions_and_physics(self):
        reference = {}
        for policy, critic in (("factorized", "additive"), ("factorized", "joint"),
                               ("conditional", "additive"), ("conditional", "joint")):
            config = default_config()
            config["network"].update(sac_policy_structure=policy, sac_critic_structure=critic)
            config["environment"].update(profile="europe_v1", action_mask="joint_feasible")
            config = validate_config(config)
            original = deepcopy(config)
            baseline_config = _baseline_environment_config(config)
            self.assertEqual(config, original)
            self.assertEqual(baseline_config["environment"]["profile"], "europe_v1")
            self.assertEqual(baseline_config["reward"], config["reward"])
            for name in ("fixed", "greedy"):
                with self.subTest(policy=policy, critic=critic, baseline=name):
                    env = WorkbenchEnv(baseline_config)
                    state = env.reset(seed=51001)
                    rule = BaselinePolicy(name, env)
                    actions = []
                    for _ in range(100):
                        action = rule.act(state)
                        actions.append(list(action))
                        state, _reward, done, _info = env.step(action)
                        if done:
                            break
                    result = (actions, env.metrics())
                    if name not in reference:
                        reference[name] = result
                    self.assertEqual(result, reference[name])


if __name__ == "__main__":
    unittest.main()
