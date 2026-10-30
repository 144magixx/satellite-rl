"""S2-LAB 另外三条规则：旧贪心、原版论文贪心、分散摆放。动作合法、回合跑完、功率不超预算，以及开头两束的已知动作。"""
import tempfile
import unittest

import numpy as np
import torch

from satellite_rl.s2lab.scenes import DatasetConfig
from satellite_rl.s2lab.env import BatchEnv, EnvOptions, ItemSampler, SceneBank
from satellite_rl.s2lab.rules import make_rule
from tests.test_s2lab_core import synthetic_scene, write_dataset

RULES = ('greedy_old', 'greedy_paper', 'dispersed')


class RulesExtraTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tmp = tempfile.TemporaryDirectory()
        self.config = DatasetConfig(name='test', power_budget_w=1500.)
        self.scenes = [synthetic_scene(f's{i}', 20 + 7 * i, i) for i in range(3)]
        write_dataset(self.tmp.name, self.scenes, self.config)
        self.bank = SceneBank(self.tmp.name, [s.scene_id for s in self.scenes], self.config.physics_obj(), self.device)

    def tearDown(self):
        self.tmp.cleanup()

    def _env(self, **kw):
        items = [dict(key=s.scene_id, per_aircraft_bps=50e6) for s in self.scenes]
        return BatchEnv(self.bank, self.config, EnvOptions(batch_size=3, **kw), self.device,
                        ItemSampler(self.bank, self.config, items, False, 0))

    def test_legal_and_complete(self):
        for group_mode in ('free', 'mod8', 'greedy'):
            for name in RULES:
                label = f'{name}/{group_mode}'
                env = self._env(group_mode=group_mode)
                rule = make_rule(name, env, 0)
                self.assertEqual(rule.name, name)
                while env.active.any():
                    obs = env.observe()
                    a = rule.act(obs)
                    active = env.active
                    self.assertTrue(obs['mask_gs'].gather(1, a[:, :1])[active].all(), label)
                    self.assertTrue(obs['mask_pw'].gather(1, a[:, 2:3])[active].all(), label)
                    env.step(a)
                self.assertEqual(len(env.finished), 3, label)
                for m in env.finished:
                    self.assertFalse(m['terminated'], label)
                    self.assertLessEqual(m['power_w'], self.config.power_budget_w + 1e-3, label)
                    self.assertGreater(m['W'], 0., label)

    def test_old_greedies_first_two_beams(self):
        """空频谱上第一束：组 0、起点 0、5 槽；第二束：组 1、起点 5（全部组合计占用最少的第一个槽）。功率按需求分档。"""
        for name in ('greedy_old', 'greedy_paper'):
            env = self._env()
            rule = make_rule(name, env, 0)
            for step in range(2):
                a = rule.act(env.observe())
                d = env.demand[env.arange_b, env.cur].cpu().numpy()
                watts = np.where(d >= 8e8, 30., np.where(d >= 4e8, 25., 20.))
                self.assertTrue((a[:, 0] == step * env.T + 5 * step).all(), name)
                self.assertTrue((a[:, 1] == 4).all(), name)
                np.testing.assert_allclose(env.levels[a[:, 2]].cpu().numpy(), watts)
                env.step(a)

    def test_dispersed_groups_follow_demand_rank(self):
        env = self._env()
        rule = make_rule('dispersed', env, 0)
        while env.active.any():
            env.step(rule.act(env.observe()))
        for b in range(3):
            n = int(env.bank.n[env.scene[b]])
            demand = env.demand[b, :n].cpu().numpy()
            expected = np.empty(n, int)
            expected[np.argsort(-demand, kind='stable')] = np.arange(n) % 8
            np.testing.assert_array_equal(env.grp[b, :n].cpu().numpy(), expected)


if __name__ == '__main__':
    unittest.main()
