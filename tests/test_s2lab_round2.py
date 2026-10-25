"""第二轮修复的回归测试：入池状态是动作前的、PPO 缓存不随环境变、每束令牌耦合不饱和；新选项（14 令牌、起点候选、
delta_intf、无掩码惩罚、joint 双 Q、自动温度、LayerNorm、诊断包装）能跑通。"""
import tempfile
import unittest

import torch

from satellite_rl.s2lab.agents import make_agent, to_store
from satellite_rl.s2lab.diagnostics import DiagnosticPolicy
from satellite_rl.s2lab.env import BatchEnv, EnvOptions, ItemSampler, SceneBank
from satellite_rl.s2lab.evaluate import run_policy
from satellite_rl.s2lab.scenes import DatasetConfig
from tests.test_s2lab_core import synthetic_scene, write_dataset


class Round2Tests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tmp = tempfile.TemporaryDirectory()
        self.config = DatasetConfig(name='test', power_budget_w=1500.)
        self.scenes = [synthetic_scene(f's{i}', 20 + 7 * i, i) for i in range(3)]
        write_dataset(self.tmp.name, self.scenes, self.config)
        self.bank = SceneBank(self.tmp.name, [s.scene_id for s in self.scenes], self.config.physics_obj(), self.device)

    def tearDown(self):
        self.tmp.cleanup()

    def _env(self, train=True, **kw):
        items = [dict(key=s.scene_id, per_aircraft_bps=50e6) for s in self.scenes]
        return BatchEnv(self.bank, self.config, EnvOptions(batch_size=3, **kw), self.device, ItemSampler(self.bank, self.config, items, train, 0))

    def test_replay_stores_pre_action_occupancy(self):
        env = self._env()
        agent = make_agent('sac', dict(encoder='tf_blocks', size=dict(d=32, layers=1, heads=4), warmup=1000, batch=8, replay=100, batch_envs=3),
                           13, self.device)
        obs = env.observe()
        for _ in range(4):                                     # 先放几束，让占用图非空
            a = agent.act(obs); env.step(a); obs = env.observe()
        before = env.occ.clone()
        active = env.active.clone()
        a = agent.act(obs)
        r, d = env.step(a)
        nxt = env.observe()
        agent.observe(obs, a, r, d, nxt, active)
        stored = agent.replay.obs['occ'][agent.replay.pos - 3:agent.replay.pos]
        self.assertTrue(torch.equal(stored, before[active]))
        self.assertFalse(torch.equal(before, env.occ))         # 这一步确实改了占用

    def test_ppo_buffer_not_aliased(self):
        env = self._env()
        obs = env.observe()
        stored = to_store(obs)
        snapshot = stored['occ'].clone()
        for _ in range(5):
            env.step(torch.stack([torch.zeros(3, dtype=torch.long, device=self.device) + 3,
                                  torch.full((3,), 2, device=self.device), torch.zeros(3, dtype=torch.long, device=self.device)], 1))
        self.assertTrue(torch.equal(stored['occ'], snapshot))

    def test_beam_coupling_not_saturated(self):
        env = self._env(beam_tokens=True)
        beam, valid = env._beam_tokens(env.cur)
        cols = beam[..., 8:10][valid]
        self.assertGreater(len(torch.unique(cols.round(decimals=3))), 5)
        self.assertLess(float(cols.max()), 1.0 + 1e-6)

    def test_new_options_run(self):
        cases = [
            ('sac', 'tsac14', dict(group_mode='greedy', legacy_obs=True), dict(size=dict(d=32, layers=1, heads=4, ff=64, intf_mode='dbw'))),
            ('sac', 'tf_legacy', dict(group_mode='greedy', start_mode='rule'), {}),
            ('sac', 'tf_legacy', dict(group_mode='greedy', start_mode='topk', reward_mode='delta_intf'), {}),
            ('sac', 'tf_blocks', dict(mask_mode='none', illegal_penalty=2.), dict(double_q='per_head')),
            ('sac', 'cnn', {}, dict(auto_alpha=True, size=dict(layernorm=True))),
            ('sac', 'mlp', {}, dict(size=dict(layernorm=True))),
        ]
        for algo, enc, envkw, agentkw in cases:
            cfg = dict(encoder=enc, size=dict(d=32, layers=1, heads=4), warmup=10, batch=16, replay=200, batch_envs=3)
            cfg.update(agentkw)
            env = self._env(**envkw)
            agent = make_agent(algo, cfg, 13, self.device, 20)
            obs = env.observe()
            if envkw.get('start_mode') == 'rule':
                self.assertTrue((obs['mask_gs'].sum(1) == 1).all())
            if envkw.get('start_mode') == 'topk':
                self.assertTrue((obs['mask_gs'].sum(1) <= 8).all())
            for _ in range(30):
                active = env.active.clone()
                a = agent.act(obs); r, d = env.step(a); nxt = env.observe()
                agent.observe(obs, a, r, d, nxt, active); info = agent.update(0); obs = nxt
                self.assertTrue(torch.isfinite(r).all())
            self.assertIsNotNone(info, enc)
            state = agent.state_dict()
            agent2 = make_agent(algo, cfg, 13, self.device, 20)
            agent2.load_state_dict(state)
            self.assertTrue(torch.equal(agent.act(obs, explore=False), agent2.act(obs, explore=False)), enc)

    def test_diagnostics_run(self):
        agent = make_agent('sac', dict(encoder='tf_blocks', size=dict(d=32, layers=1, heads=4)), 13, self.device)
        items = [dict(key=s.scene_id, per_aircraft_bps=50e6) for s in self.scenes]
        for mode in ('random_start', 'intf_zero', 'intf_shuffle'):
            res = run_policy(lambda env, m=mode: DiagnosticPolicy(agent, env, m), self.bank, self.config, items, EnvOptions(batch_size=3), self.device)
            self.assertEqual(len(res), 3)


if __name__ == '__main__':
    unittest.main()
