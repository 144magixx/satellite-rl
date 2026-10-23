"""报告21的 X2（组号=处理顺序%8）与 X4（直出T-SAC 自己选组）：分组、按组状态、保留槽数、整回合与 SAC 更新。"""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from satellite_rl.s2fix.data import read_records, load_input
from satellite_rl.s2fix.tsac import config_for, make_env, make_agent, save_policy, load_policy
from satellite_rl.s2fix.tsac_choose import GROUPS, STATE_WIDTH, INTERFERENCE, OCCUPANCY, MARGIN, GroupChoiceEnv

DATA = Path("D:/paper/outputs/experiments/s2_pure_tabu_20261021/data")


def scene(index=0):
    return load_input(read_records(DATA, "dev_d432_n150")[index])


def random_legal(mask, rng):
    f, l = np.argwhere(mask['frequency_slots'])[rng.integers(mask['frequency_slots'].sum())]
    return int(f), int(l), int(rng.choice(np.flatnonzero(mask['power'])))


class GroupingTests(unittest.TestCase):
    def test_order8_groups_are_the_spread_rule_groups(self):
        from satellite_rl.s2fix.greedy import spread_allocation
        stage = scene()
        env = make_env(stage, 'order8')
        np.testing.assert_array_equal(env.groups, spread_allocation(scene())[0])
        self.assertEqual(len(set(env.groups.tolist())), 8)

    def test_choose_state_slices_equal_the_fixed_group_state(self):
        """同一串放置下，X4 状态里"当前束按顺序%8该在的那个组"的三条通道，与 X2 环境的状态一致。"""
        fixed, choose, rng = make_env(scene(), 'order8'), GroupChoiceEnv(scene()), np.random.default_rng(3)
        fixed_state, choose_state = fixed.reset(), choose.reset()
        for _ in range(60):
            c = fixed.indices[fixed.beam_idx]
            g = int(fixed.groups[c])
            for channel, start in ((slice(4, 104), INTERFERENCE), (slice(104, 204), OCCUPANCY), (slice(208, 308), MARGIN)):
                np.testing.assert_allclose(choose_state[start + 100 * g:start + 100 * g + 100], fixed_state[channel],
                                           atol=2e-6)
            np.testing.assert_array_equal(choose_state[:4], fixed_state[:4])
            np.testing.assert_array_equal(choose_state[MARGIN - 4:MARGIN], fixed_state[204:208])
            f, l, p = random_legal(fixed.action_mask(), rng)
            self.assertTrue(choose.action_mask()['frequency_slots'][g * 100 + f, l])
            fixed_state, fixed_reward, _, _ = fixed.step((f, l, p))
            choose_state, choose_reward, _, _ = choose.step((g * 100 + f, l, p))
            self.assertAlmostEqual(fixed_reward, choose_reward, places=9)
            np.testing.assert_allclose(choose.rates, fixed.rates, rtol=1e-12)
        self.assertEqual(len(choose_state), STATE_WIDTH)

    def test_choose_mask_keeps_one_free_slot_per_remaining_beam(self):
        env = GroupChoiceEnv(scene())
        env.occupied[:] = True
        env.occupied[3, 10:15] = False
        env.occupied[6, 50:52] = False
        env.beam_idx = env.N - 3                      # 本束之后还剩 2 束
        mask = env.action_mask()['frequency_slots']
        self.assertTrue(mask[310, 4] and mask[650, 1])
        self.assertFalse(mask[311, 4] or mask[650, 2] or mask[310, 5])
        env.beam_idx = env.N - 5                      # 还剩 4 束：7 个空槽最多给本束 3 个
        mask = env.action_mask()['frequency_slots']
        self.assertTrue(mask[310, 2] and mask[312, 2])
        self.assertFalse(mask[310, 3])

    def test_full_episodes_match_the_common_evaluation(self):
        from satellite_rl.s2fix.evaluate import capacities
        rng = np.random.default_rng(5)
        for grouping in ('order8', 'choose'):
            stage = scene(1)
            env = make_env(stage, grouping)
            env.reset()
            done = False
            while not done:
                _, _, done, info = env.step(random_legal(env.action_mask(), rng))
            self.assertEqual(info['allocation']['actual_slot_count'].min(), 1)
            np.testing.assert_allclose(capacities(scene(1), env.groups, env.starts, env.counts, env.powers),
                                       env.rates, rtol=1e-9)
            for g in range(GROUPS):
                taken = np.zeros(100, int)
                for i in np.flatnonzero(env.groups == g):
                    taken[env.starts[i]:env.starts[i] + env.counts[i]] += 1
                self.assertLessEqual(taken.max(), 1)


class ChooseAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_update_act_and_reload(self):
        agent = make_agent(config_for(grouping='choose'))
        self.assertEqual(agent.action_dims, (800, 10, 11))
        mask = dict(frequency_slots=np.zeros((800, 10), bool), power=np.ones(11, bool))
        mask['frequency_slots'][[5, 333, 799], 0] = True
        mask['frequency_slots'][333, 3] = True
        rng = np.random.default_rng(0)
        for _ in range(256):
            state = rng.random(STATE_WIDTH).astype(np.float32)
            agent.observe(state, (333, 3, 2), 1., state, True, action_mask=mask)
        before = agent.actor.start_head.weight.detach().clone()
        result = agent.update()
        self.assertTrue(np.isfinite(result['actor_loss']))
        self.assertFalse(torch.equal(before, agent.actor.start_head.weight))
        for _ in range(20):
            f, l, p = agent.act(rng.random(STATE_WIDTH).astype(np.float32), action_mask=mask)
            self.assertTrue(mask['frequency_slots'][f, l])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            save_policy(agent, agent.config, path, 0)
            loaded = load_policy(path)
        self.assertEqual(type(loaded).__name__, 'GroupChoiceAgent')
        for name in ('actor', 'critic1', 'critic2'):
            for a, b in zip(getattr(agent, name).state_dict().values(), getattr(loaded, name).state_dict().values()):
                self.assertTrue(torch.equal(a, b))

    def test_default_config_is_unchanged_l2(self):
        self.assertNotIn('s2fix_grouping', config_for())
        self.assertEqual(make_agent(config_for()).parameter_counts()['actor'], 419449)
        self.assertEqual(config_for(grouping='order8')['s2fix_grouping'], 'order8')


if __name__ == "__main__":
    unittest.main()
