"""第三轮测试：SAC3 的 R0 与第二轮 SAC 逐数相同；历史取小 ≤ 求期望后取小；合法落位不截断；按槽数筛选的窗口数；
连续功率在区间内且演员能经 Q 对功率求梯度；补充特征有限、在范围内、能进编码器；入池观测与动作前观测一致；
PPO/DQN/MLP-SAC 的 14 令牌版本能跑。"""
import copy
import tempfile
import unittest

import torch

from satellite_rl.s2lab.agents import make_agent
from satellite_rl.s2lab.env import BatchEnv, EnvOptions, ItemSampler, SceneBank
from satellite_rl.s2lab.features import FEATURE_NAMES
from satellite_rl.s2lab.masks import conditional_masks, free_run
from satellite_rl.s2lab.matrix import MAIN_SIZE, SAC_CORE
from satellite_rl.s2lab.scenes import DatasetConfig
from tests.test_s2lab_core import synthetic_scene, write_dataset

BASE = dict(SAC_CORE, encoder='tsac14', size=dict(MAIN_SIZE, d=32, layers=1, heads=4, ff=64, intf_mode='in'), warmup=40, batch=32,
            replay=400, batch_envs=4, double_q='joint')


class Round3Tests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tmp = tempfile.TemporaryDirectory()
        self.config = DatasetConfig(name='test', power_budget_w=1500.)
        self.scenes = [synthetic_scene(f's{i}', 25 + 6 * i, i) for i in range(4)]
        write_dataset(self.tmp.name, self.scenes, self.config)
        self.bank = SceneBank(self.tmp.name, [s.scene_id for s in self.scenes], self.config.physics_obj(), self.device)

    def tearDown(self):
        self.tmp.cleanup()

    def _env(self, train=True, **kw):
        items = [dict(key=s.scene_id, per_aircraft_bps=60e6) for s in self.scenes]
        opts = EnvOptions(batch_size=4, group_mode='greedy', reward_mode='delta', legacy_obs=True, **kw)
        return BatchEnv(self.bank, self.config, opts, self.device, ItemSampler(self.bank, self.config, items, train, 0))

    def _fill(self, agent, env, steps=60):
        obs = env.observe()
        for _ in range(steps):
            active = env.active.clone()
            out = agent.act(obs)
            action, power = out if isinstance(out, tuple) else (out, None)
            r, d = env.step(action, power)
            nxt = env.observe()
            agent.observe(obs, action, r, d, nxt, active)
            obs = nxt
        return obs

    def test_r0_matches_round2_sac(self):
        torch.manual_seed(0)
        old = make_agent('sac', dict(BASE), 13, self.device)
        new = make_agent('sac3', dict(BASE), 13, self.device)
        for a, b in ((new.actor.base, old.actor), (new.q1.base, old.q1), (new.q2.base, old.q2), (new.q1_t.base, old.q1_t), (new.q2_t.base, old.q2_t)):
            a.load_state_dict(b.state_dict())
        new.opt_actor = torch.optim.Adam(new.actor.parameters(), lr=BASE['actor_lr'])
        new.opt_critic = torch.optim.Adam(list(new.q1.parameters()) + list(new.q2.parameters()), lr=BASE['critic_lr'])
        self._fill(old, self._env())
        new.replay = old.replay
        for _ in range(3):
            torch.manual_seed(7)
            io = old._update_once()
            torch.manual_seed(7)
            inn = new._update_once()
            for k in ('critic_loss', 'actor_loss', 'entropy', 'q'):
                self.assertAlmostEqual(io[k], inn[k], places=4, msg=k)
        for a, b in ((new.actor.base, old.actor), (new.q1.base, old.q1)):
            for pa, pb in zip(a.parameters(), b.parameters()):
                # 损失逐数相同；参数只差浮点求和顺序带来的噪声经 Adam 放大（远小于一步学习率 1e-4）
                self.assertTrue(torch.allclose(pa, pb, atol=5e-5))

    def test_hist_not_above_joint(self):
        torch.manual_seed(1)
        agent = make_agent('sac3', dict(BASE, double_q='hist'), 13, self.device)
        self._fill(agent, self._env())
        obs, action, reward, done, nxt, power = agent._sample()
        with torch.no_grad():
            out = agent.actor(nxt)
            lc, lx, lp, _ = agent._dist(out, nxt)
            w, _ = agent._weights(lc, lx, lp)
            t1, t2 = agent._critic_outs(nxt, True)
            Q1, Q2 = agent._q_full(t1, lc, lx, lp), agent._q_full(t2, lc, lx, lp)
            hist = (w * torch.minimum(Q1, Q2)).sum((1, 2, 3))
            joint = torch.minimum((w * Q1).sum((1, 2, 3)), (w * Q2).sum((1, 2, 3)))
            self.assertTrue((hist <= joint + 1e-5).all())
            self.assertTrue(torch.allclose(w.sum((1, 2, 3)), torch.ones_like(hist), atol=1e-4))

    def test_legal_policy_never_truncates(self):
        for policy, topk in (('legal', 0), ('cond', 0), ('cond', 8)):
            torch.manual_seed(2)
            env = self._env(train=False, intf_rel=topk > 0)
            agent = make_agent('sac3', dict(BASE, policy=policy, topk=topk, warmup=10 ** 9), 13, self.device)
            obs = env.observe()
            while env.alive.any():
                mc, mx = conditional_masks(obs['occ'], obs['allow'], obs['cmax'], topk, obs.get('intf_rel'))
                if topk:
                    self.assertTrue((mx.sum(-1) <= 8).all())
                g = obs['allow'].float().argmax(1)
                fr = free_run(obs['occ'])[torch.arange(env.B, device=self.device), g]
                c = torch.arange(1, 11, device=self.device)
                self.assertTrue((~mx | (fr[:, None, :] >= c[None, :, None])).all())
                env.step(agent.act(obs))
                obs = env.observe()
            self.assertTrue(all(m['truncated_beams'] == 0 for m in env.finished), policy)

    def test_continuous_power(self):
        for pip, qp in (('count', 'count'), ('count_start', 'count_start')):
            torch.manual_seed(3)
            cfg = dict(BASE, policy='cond', power='continuous', pi_power=pip, q_power=qp, warmup=20)
            agent = make_agent('sac3', cfg, 13, self.device)
            env = self._env()
            obs = env.observe()
            action, power = agent.act(obs)
            self.assertTrue(((power >= .5 - 1e-6) & (power <= obs['pmax'].clamp(min=.5) + 1e-5)).all())
            self._fill(agent, env)
            info = agent._update_once()
            self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in info.values()))
            grads = [p.grad for p in agent.actor.p_mlp.parameters()]
            self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in grads), pip)

    def test_features_and_replay_audit(self):
        env = self._env(features=FEATURE_NAMES)
        size = dict(BASE['size'], feats=list(FEATURE_NAMES))
        agent = make_agent('sac3', dict(BASE, size=size), 13, self.device)
        obs = env.observe()
        snapshots = []
        for _ in range(40):
            active = env.active.clone()
            snap = {k: v.clone() for k, v in obs.items()}
            action = agent.act(obs)
            r, d = env.step(action)
            nxt = env.observe()
            agent.observe(obs, action, r, d, nxt, active)
            snapshots.append((snap, active))
            obs = nxt
        for k in FEATURE_NAMES:
            v = obs['f_' + k]
            self.assertTrue(torch.isfinite(v).all() and v.min() >= 0 and v.max() <= 1.2 + 1e-6, k)
        # 入池样本＝动作前观测（float16 误差内）
        rep = agent.replay
        pos = 0
        for snap, active in snapshots:
            idx = torch.nonzero(active).flatten()
            for j, i in enumerate(idx.tolist()):
                for k, v in rep.obs.items():
                    a, b = v[pos + j].float(), snap[k][i].float()
                    self.assertTrue(torch.allclose(a, b, rtol=2e-3, atol=2e-3), k)
            pos += len(idx)
        agent._update_once()

    def test_structural_variants_run(self):
        variants = [('sac3', dict(BASE, size=dict(BASE['size'], no_global=True))),
                    ('sac3', dict(BASE, size=dict(BASE['size'], intf_zero=True))),
                    ('sac3', dict(BASE, encoder='mlp14', size=dict(d=32, hidden=64))),
                    ('sac3', dict(BASE, policy='cond', q_power='count')),
                    ('sac3', dict(BASE, auto_alpha=True)),
                    ('ppo', dict(encoder='tsac14', size=BASE['size'], rollout=4, epochs=1, minibatch=16, batch_envs=4)),
                    ('dqn', dict(encoder='tsac14', size=BASE['size'], warmup=20, batch=16, replay=200, batch_envs=4))]
        for algo, cfg in variants:
            torch.manual_seed(4)
            agent = make_agent(algo, cfg, 13, self.device)
            env = self._env()
            obs = env.observe()
            for _ in range(30):
                active = env.active.clone()
                out = agent.act(obs)
                action, power = out if isinstance(out, tuple) else (out, None)
                r, d = env.step(action, power)
                nxt = env.observe()
                agent.observe(obs, action, r, d, nxt, active)
                info = agent.update(0)
                obs = nxt
            state = agent.state_dict()
            agent2 = make_agent(algo, copy.deepcopy(cfg), 13, self.device)
            agent2.load_state_dict(state)
            out1, out2 = agent.act(obs, explore=False), agent2.act(obs, explore=False)
            a1 = out1[0] if isinstance(out1, tuple) else out1
            a2 = out2[0] if isinstance(out2, tuple) else out2
            self.assertTrue(torch.equal(a1, a2), algo)


if __name__ == '__main__':
    unittest.main()
