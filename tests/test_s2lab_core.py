"""S2-LAB 核心测试：方向图连续性、环境容量与 numpy 参考一致、掩码合法性、规则合法性、断点续跑一致。"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from satellite_rl.s2lab.physics import PhysicsV3, s672_gain_dbi, peak_gain_dbi, SceneTensors, reference_capacities, sgm_values, path_gain
from satellite_rl.s2lab.scenes import DatasetConfig
from satellite_rl.s2lab.env import BatchEnv, EnvOptions, SceneBank
from satellite_rl.s2lab.rules import make_rule
from satellite_rl.s2lab.agents import make_agent


def synthetic_scene(key, n, seed):
    """合成场景：欧洲上空随机中心、0.3—0.6° 宽、1—26 架飞机。"""
    rng = np.random.default_rng(seed)
    physics = PhysicsV3()
    lat = rng.uniform(40, 55, n)
    lon = rng.uniform(0, 20, n)
    centres = np.column_stack([lat, lon])
    widths = rng.uniform(.3, .6, n)
    members = rng.integers(1, 27, n)
    from satellite_rl.shared_interface.geometry import directions_and_distance
    from satellite_rl.s2lab.physics import angles_deg, gain_linear
    rays, distance = directions_and_distance(centres, validate=False)
    rx = 10 ** (physics.rx_gain_dBi / 10) * 10 ** (-physics.other_loss_dB / 10)
    sig = gain_linear(np.zeros(n), widths, physics) * rx * .9
    theta = angles_deg(rays, rays)
    coupling = gain_linear(theta, widths[None, :], physics) * rx
    np.fill_diagonal(coupling, 0.)
    adjacent = theta < physics.adjacent_deg
    np.fill_diagonal(adjacent, False)
    return SceneTensors(key, '2024-01-01', members, widths.astype(np.float32), centres.astype(np.float32),
                        distance.astype(np.float32), sig.astype(np.float32), coupling.astype(np.float32), adjacent, int(members.sum()))


def write_dataset(root, scenes, config):
    from dataclasses import asdict
    root = Path(root)
    (root / 'scenes').mkdir(parents=True)
    for s in scenes:
        np.savez_compressed(root / 'scenes' / f'{s.scene_id}.npz', members=s.members, width_deg=s.width_deg, lat_lon=s.lat_lon,
                            distance_m=s.distance_m, sig=s.sig, coupling=s.coupling, adjacent=s.adjacent,
                            aircraft_count=s.aircraft_count, date=s.date, scene_id=s.scene_id, source='synthetic')
    (root / 'config.json').write_text(json.dumps(asdict(config)), encoding='utf-8')
    items = [dict(key=s.scene_id, per_aircraft_bps=50e6, stratum='marginal') for s in scenes]
    (root / 'splits.json').write_text(json.dumps(dict(train=[dict(key=s.scene_id) for s in scenes], dev=items, confirm=items)), encoding='utf-8')


class PhysicsTests(unittest.TestCase):
    def test_pattern_continuous_and_monotone(self):
        p = PhysicsV3(sidelobe_level_dB=-25.)
        theta = np.linspace(0, 10, 4001)
        g = s672_gain_dbi(theta, .5, p)
        self.assertAlmostEqual(g[0], peak_gain_dbi(.5, p))
        self.assertAlmostEqual(s672_gain_dbi(.25, .5, p), peak_gain_dbi(.5, p) - 3, places=6)
        self.assertTrue((np.diff(g) <= 1e-9).all())
        self.assertLess(np.abs(np.diff(g)).max(), .2)       # 无跳变
        self.assertAlmostEqual(s672_gain_dbi(1., .5, p), peak_gain_dbi(.5, p) - 25, places=6)   # 旁瓣底
        self.assertEqual(s672_gain_dbi(60., .5, p), 0.)

    def test_sgm_shape(self):
        d = np.array([1e8, 1e8, 1e8])
        self.assertAlmostEqual(sgm_values(np.array([1e8, 0., 5e7]), d)[0], 1.)
        self.assertEqual(sgm_values(np.array([1e8, 0., 5e7]), d)[1], 0.)
        self.assertLess(sgm_values(np.array([1e8, 0., 5e7]), d)[2], .9)


class EnvTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tmp = tempfile.TemporaryDirectory()
        self.config = DatasetConfig(name='test', power_budget_w=1500.)
        self.scenes = [synthetic_scene(f's{i}', 20 + 7 * i, i) for i in range(3)]
        write_dataset(self.tmp.name, self.scenes, self.config)
        self.physics = self.config.physics_obj()
        self.bank = SceneBank(self.tmp.name, [s.scene_id for s in self.scenes], self.physics, self.device)

    def tearDown(self):
        self.tmp.cleanup()

    def _env(self, **kw):
        from satellite_rl.s2lab.env import ItemSampler
        items = [dict(key=s.scene_id, per_aircraft_bps=50e6) for s in self.scenes]
        opts = EnvOptions(batch_size=3, **kw)
        return BatchEnv(self.bank, self.config, opts, self.device, ItemSampler(self.bank, self.config, items, False, 0))

    def test_capacity_matches_reference_and_masks_legal(self):
        for rule_name in ('greedy', 'fixed', 'random'):
            env = self._env()
            rule = make_rule(rule_name, env, 0)
            while env.active.any():
                obs = env.observe()
                a = rule.act(obs)
                self.assertTrue(obs['mask_gs'].gather(1, a[:, :1]).all(), rule_name)
                self.assertTrue(obs['mask_pw'].gather(1, a[:, 2:3]).all(), rule_name)
                env.step(a)
            self.assertEqual(len(env.finished), 3)
            for m in env.finished:
                self.assertLessEqual(m['power_w'], self.config.power_budget_w + 1e-3)
                self.assertFalse(m['terminated'])
                self.assertGreater(m['W'], 0.)
            # 每一步的全体容量与 numpy 参考一致
            env4 = self._env()
            rule = make_rule(rule_name, env4, 0)
            while env4.active.any():
                before = env4.capacities()
                groups = env4.grp.clone(); starts = env4.start.clone(); counts = env4.cnt.clone(); powers = env4.pw.clone()
                for b in range(3):
                    if not bool(env4.alive[b]):
                        continue
                    n = int(env4.n[b])
                    ref = reference_capacities(self.scenes[int(env4.scene[b])], groups[b, :n].cpu().numpy(), starts[b, :n].cpu().numpy(),
                                               counts[b, :n].cpu().numpy(), powers[b, :n].cpu().numpy(), self.physics)
                    np.testing.assert_allclose(before[b, :n].cpu().numpy(), ref, rtol=2e-4, atol=1.)
                env4.step(rule.act(env4.observe()))

    def test_no_mask_overlap_unserved(self):
        env = self._env(mask_mode='none')
        obs = env.observe()
        a = torch.tensor([[5, 3, 2]] * 3, device=self.device)        # 同组同起点两次 → 第二束不服务
        env.step(a)
        env.step(a)
        self.assertTrue((env.cnt[torch.arange(3), env.order[:, 1]] == 0).all())

    def test_agents_step_and_checkpoint(self):
        from satellite_rl.s2lab.env import ItemSampler
        for algo, enc in (('sac', 'tf_blocks'), ('sac', 'cnn'), ('sac', 'mlp'), ('dqn', 'tf_blocks'), ('ppo', 'tf_blocks'), ('sac', 'tf_beams')):
            cfg = dict(encoder=enc, size=dict(d=32, layers=1, heads=4), pointer=(enc == 'tf_blocks'), warmup=10, batch=16, replay=200,
                       rollout=4, minibatch=16, batch_envs=3)
            agent = make_agent(algo, cfg, 13, self.device, 20)
            items = [dict(key=s.scene_id) for s in self.scenes]
            opts = EnvOptions(batch_size=3, beam_tokens=(enc == 'tf_beams'))
            env = BatchEnv(self.bank, self.config, opts, self.device, ItemSampler(self.bank, self.config, items, True, 0))
            obs = env.observe()
            for _ in range(30):
                active = env.active.clone()
                a = agent.act(obs)
                r, d = env.step(a)
                nxt = env.observe()
                agent.observe(obs, a, r, d, nxt, active)
                agent.update(0)
                obs = nxt
            state = agent.state_dict()
            agent2 = make_agent(algo, cfg, 13, self.device, 20)
            agent2.load_state_dict(state)
            a1 = agent.act(obs, explore=False)
            a2 = agent2.act(obs, explore=False)
            self.assertTrue(torch.equal(a1, a2), algo + enc)


if __name__ == '__main__':
    unittest.main()
