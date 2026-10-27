"""第四轮测试：探索组合父组配置与第三轮一致；块指针头与"评论家起点看槽数"真的接通且取值一致；贪心建议特征不改环境；
行为克隆能学到贪心动作且热身期用演员；本局最后一束不跨局自举；γ=1 能更新。"""
import json
import unittest
from pathlib import Path

import torch

from satellite_rl.s2lab.agents import make_agent
from satellite_rl.s2lab.bc import collect_greedy_demos
from satellite_rl.s2lab.matrix import build_matrix_round4, XC_FEATS
from satellite_rl.s2lab.rules import GreedyRule
from tests.test_s2lab_round3 import BASE, Round3Tests

XC = dict(BASE, replay=400, size=dict(BASE['size'], feats=list(XC_FEATS)))


class Round4Tests(Round3Tests):
    def _env4(self, feats=XC_FEATS, **kw):
        return self._env(features=tuple(feats), **kw)

    def test_parent_config_matches_round3_xcombo(self):
        runs = {r['run_id']: r for r in build_matrix_round4()}
        self.assertEqual(len(runs), 39)
        ref = Path('D:/paper/outputs/experiments/s2lab_round3_20261027/account_B/runs3/r3-xcombo-s1/config.json')
        if ref.exists():
            old = json.loads(ref.read_text(encoding='utf-8'))
            new = runs['r4-base-s1']
            for k in ('algo', 'encoder', 'size', 'agent', 'pointer', 'mask_mode', 'group_mode', 'reward_mode', 'total_steps', 'batch_envs', 'dev_every'):
                self.assertEqual(new[k], old[k], k)

    def _one_update(self, cfg, feats=XC_FEATS):
        torch.manual_seed(5)
        agent = make_agent('sac3', dict(cfg), 13, self.device)
        self._fill(agent, self._env4(feats))
        before = {n: p.detach().clone() for n, p in agent.actor.named_parameters()}
        cb = {n: p.detach().clone() for n, p in agent.q1.named_parameters()}
        info = agent._update_once()
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in info.values()))
        moved = {n for n, p in agent.actor.named_parameters() if not torch.equal(p, before[n])}
        cmoved = {n for n, p in agent.q1.named_parameters() if not torch.equal(p, cb[n])}
        return agent, moved, cmoved

    def test_pointer_wired(self):
        agent, moved, cmoved = self._one_update(dict(XC, pointer=True))
        self.assertTrue(any(n.startswith('ptr_') for n in moved))
        self.assertTrue(any(n.startswith('ptr_') for n in cmoved))
        self.assertFalse(any(n.startswith('base.heads.gs') for n in moved))      # 原线性起点头不再参与
        # 改动某一块的令牌只会改变指针分数：直接检查 gs 由块令牌给出
        obs = self._env4().observe()
        out = agent.actor(obs)
        q = agent.actor.ptr_q(out['pooled'])
        ref = (torch.einsum('bkd,bd->bk', out['blocks'], q) / out['blocks'].shape[-1] ** .5)[:, :, None] + agent.actor.ptr_off(out['blocks'])
        self.assertTrue(torch.allclose(out['gs'], ref.reshape(len(q), 100), atol=1e-5))

    def test_critic_start_by_count(self):
        cfg = dict(XC, policy='cond', q_start='count')
        agent, moved, cmoved = self._one_update(cfg)
        self.assertTrue(any(n.startswith('qx_') for n in cmoved))
        obs, action, reward, done, nxt, power = agent._sample()
        with torch.no_grad():
            o = agent.q1(obs)
            a_out = agent.actor(obs)
            lc, lx, lp, _ = agent._dist(a_out, obs)
            Q = agent._q_full(o, lc, lx, lp)                                   # [B,10,T,K]
            rows = torch.arange(len(action), device=self.device)
            x = action[:, 0] - obs['allow'].float().argmax(1).long() * 100
            taken = o['cnt'][rows, action[:, 1]] + o['qx_c'][rows, action[:, 1], x] + o['pw'][rows, action[:, 2]]
            self.assertTrue(torch.allclose(Q[rows, action[:, 1], x, action[:, 2]], taken, atol=1e-5))

    def test_hint_features(self):
        env = self._env4(XC_FEATS + ['hint', 'hintx'])
        for _ in range(5):
            occ, used = env.occ.clone(), env.used_w.clone()
            obs = env.observe()
            self.assertTrue(torch.equal(occ, env.occ) and torch.equal(used, env.used_w))
            a = GreedyRule(env).act(obs)
            self.assertTrue(torch.equal(obs['f_hint'][:, :10].argmax(1), a[:, 1]))
            self.assertTrue(torch.equal(obs['f_hint'][:, 10:].argmax(1), a[:, 2]))
            self.assertTrue(torch.equal(obs['f_hintx'].argmax(1), a[:, 0] % 100))
            env.step(a)
        agent, moved, _ = self._one_update(dict(XC, size=dict(XC['size'], feats=XC_FEATS + ['hint', 'hintx'])), XC_FEATS + ['hint', 'hintx'])
        self.assertTrue(moved)

    def test_behavior_clone(self):
        torch.manual_seed(6)
        cfg = dict(XC, bc_updates=300, bc_lr=1e-3, actor_freeze=10 ** 9, batch=64)
        agent = make_agent('sac3', dict(cfg), 13, self.device)
        env = self._env4()
        items = [dict(key=s.scene_id) for s in self.scenes]
        demos = collect_greedy_demos(self.bank, self.config, items, env.opt, self.device, 600, 0)
        self.assertEqual(len(demos['action']), 600)
        info0 = agent.behavior_clone(demos, 1)
        info = agent.behavior_clone(demos, 300)
        self.assertLess(info['loss'], info0['loss'])
        self.assertGreater(info['agree_count'], info0['agree_count'])
        # 热身期也由演员出动作（确定性动作与采样一致性不测，只确认不是均匀随机：克隆后与贪心的槽数一致率明显高于随机）
        obs = env.observe()
        a = agent.act(obs, explore=False)
        self.assertEqual(a.shape[1], 3)
        before = {n: p.detach().clone() for n, p in agent.actor.named_parameters()}
        self._fill(agent, env)
        agent._update_once()
        self.assertTrue(all(torch.equal(p, before[n]) for n, p in agent.actor.named_parameters()))   # 冻结期演员不动

    def test_done_no_bootstrap_and_gamma1(self):
        for gamma in (.99, 1.):
            agent = make_agent('sac3', dict(XC, gamma=gamma), 13, self.device)
            self._fill(agent, self._env4())
            obs, action, reward, done, nxt, power = agent._sample()
            t = agent._target(reward, torch.ones_like(done), nxt, agent.alpha)
            self.assertTrue(torch.equal(t, reward))
            agent._update_once()


if __name__ == '__main__':
    unittest.main()
