"""学习方法：因子化离散 SAC（T-SAC/CNN-SAC/MLP-SAC 只差编码器）、DQN、PPO。

统一接口：act(obs, explore) → LongTensor[B,3]；observe(obs, action, reward, done, next_obs)；
update(step) → dict 或 None；state_dict()/load_state_dict() 含优化器与回放池。
"""
import copy
import math

import torch
import torch.nn.functional as F

from .nets import PolicyNet, masked, head_masks, count_parameters

OBS_KEYS = ('scal', 'intf', 'occ', 'allow', 'mask_gs', 'mask_cnt', 'mask_pw', 'beam', 'beam_valid', 'glob', 'intf_abs',
            'cmax', 'pmax', 'intf_rel', 'f_pneed', 'f_share', 'f_outint', 'f_snr', 'f_sens', 'f_future', 'f_gspec', 'f_hint', 'f_hintx')
STORE_DTYPE = dict(scal=torch.float16, intf=torch.float16, occ=torch.bool, allow=torch.bool, mask_gs=torch.bool,
                   mask_cnt=torch.bool, mask_pw=torch.bool, beam=torch.float16, beam_valid=torch.bool,
                   glob=torch.float16, intf_abs=torch.float16,
                   cmax=torch.long, pmax=torch.float32, intf_rel=torch.float32, f_pneed=torch.float16, f_share=torch.float16,
                   f_outint=torch.float16, f_snr=torch.float16, f_sens=torch.float16, f_future=torch.float16, f_gspec=torch.float16,
                   f_hint=torch.float16, f_hintx=torch.float16)


def to_store(obs):
    """必须是副本：.to(同类型) 不复制，PPO 缓存会跟着环境变（第一轮的实现错误之一）。"""
    return {k: obs[k].to(STORE_DTYPE[k]).clone() for k in OBS_KEYS if k in obs}


def to_model(obs):
    return {k: (v.float() if v.dtype == torch.float16 else v) for k, v in obs.items()}


class Replay:
    def __init__(self, capacity, device):
        self.capacity, self.device = capacity, device
        self.obs, self.next_obs = None, None
        self.action = torch.zeros(capacity, 3, dtype=torch.long, device=device)
        self.reward = torch.zeros(capacity, device=device)
        self.done = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.size, self.pos = 0, 0

    def _alloc(self, sample):
        return {k: torch.zeros((self.capacity,) + tuple(v.shape[1:]), dtype=v.dtype, device=self.device) for k, v in sample.items()}

    def add(self, obs, action, reward, done, next_obs, valid):
        obs, next_obs = to_store(obs), to_store(next_obs)
        if self.obs is None:
            self.obs, self.next_obs = self._alloc(obs), self._alloc(next_obs)
        idx = torch.nonzero(valid).flatten()
        n = len(idx)
        if n == 0:
            return
        slots = (self.pos + torch.arange(n, device=self.device)) % self.capacity
        for k in obs:
            self.obs[k][slots] = obs[k][idx]
            self.next_obs[k][slots] = next_obs[k][idx]
        self.action[slots] = action[idx]
        self.reward[slots] = reward[idx]
        self.done[slots] = done[idx]
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch):
        idx = torch.randint(0, self.size, (batch,), device=self.device)
        obs = to_model({k: v[idx] for k, v in self.obs.items()})
        nxt = to_model({k: v[idx] for k, v in self.next_obs.items()})
        return obs, self.action[idx], self.reward[idx], self.done[idx].float(), nxt

    def state_dict(self):
        return dict(obs=self.obs, next_obs=self.next_obs, action=self.action, reward=self.reward, done=self.done,
                    size=self.size, pos=self.pos)

    def load_state_dict(self, state):
        self.obs, self.next_obs = state['obs'], state['next_obs']
        self.action, self.reward, self.done = state['action'], state['reward'], state['done']
        self.size, self.pos = state['size'], state['pos']


def sample_heads(logits, masks, explore):
    actions, logps = [], []
    for lg, m in zip(logits, masks):
        lg = masked(lg, m)
        if explore:
            a = torch.distributions.Categorical(logits=lg).sample()
        else:
            a = lg.argmax(-1)
        actions.append(a)
        logps.append(F.log_softmax(lg, -1).gather(1, a[:, None]).squeeze(1))
    return torch.stack(actions, 1), torch.stack(logps, 1).sum(1)


def random_valid(masks):
    out = []
    for m in masks:
        probs = m.float() / m.float().sum(1, keepdim=True)
        out.append(torch.multinomial(probs, 1).squeeze(1))
    return torch.stack(out, 1)


class AgentBase:
    off_policy = True

    def __init__(self, cfg, K, device, beam_dim=20):
        self.cfg, self.K, self.device, self.beam_dim = cfg, K, device, beam_dim
        self.updates = 0

    def net(self, value=False):
        return PolicyNet(self.cfg['encoder'], self.cfg.get('size'), self.K, self.cfg.get('pointer', False), value, self.beam_dim).to(self.device)

    def parameter_count(self):
        return count_parameters(self.main_module())


class SAC(AgentBase):
    def __init__(self, cfg, K, device, beam_dim=20):
        super().__init__(cfg, K, device, beam_dim)
        self.actor = self.net()
        self.q1, self.q2 = self.net(), self.net()
        self.q1_t, self.q2_t = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        for p in list(self.q1_t.parameters()) + list(self.q2_t.parameters()):
            p.requires_grad_(False)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.get('actor_lr', cfg.get('lr', 3e-4)))
        self.opt_critic = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.get('critic_lr', cfg.get('lr', 3e-4)))
        self.alpha = float(cfg.get('alpha', .01))
        self.auto_alpha = bool(cfg.get('auto_alpha', False))
        self.log_alpha = torch.tensor(math.log(self.alpha), device=device, requires_grad=True)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=cfg.get('alpha_lr', 3e-4))
        self.gamma, self.tau = cfg.get('gamma', .99), cfg.get('tau', .005)
        self.batch, self.warmup = cfg.get('batch', 256), cfg.get('warmup', 2000)
        self.updates_per_step = cfg.get('utd', .5)
        self.replay = Replay(cfg.get('replay', 100000), device)
        self.env_steps = 0
        self._pending = 0.

    def main_module(self):
        return self.actor

    @torch.no_grad()
    def act(self, obs, explore=True):
        masks = head_masks(obs)
        if explore and self.env_steps < self.warmup:
            return random_valid(masks)
        (gs, cnt, pw), _ = self.actor(obs)
        return sample_heads((gs, cnt, pw), masks, explore)[0]

    def observe(self, obs, action, reward, done, next_obs, valid):
        self.replay.add(obs, action, reward, done, next_obs, valid)
        self.env_steps += int(valid.sum())

    def _policy(self, net, obs):
        (gs, cnt, pw), _ = net(obs)
        out = []
        for lg, m in zip((gs, cnt, pw), head_masks(obs)):
            lg = masked(lg, m)
            logp = F.log_softmax(lg, -1)
            out.append((logp.exp() * m, logp))
        return out

    def update(self, step):
        if self.env_steps < self.warmup or self.replay.size < self.batch:
            return None
        self._pending += self.updates_per_step * self.cfg.get('batch_envs', 64)
        n = int(self._pending)
        self._pending -= n
        info = None
        for _ in range(n):
            info = self._update_once()
        return info

    def _update_once(self):
        """double_q='joint'（第二轮默认，"求期望后取小"）：两个评论家各自把三头的期望值加总，再取较小的一个；
        注意这不是历史代码（agents/sac_math.py、power30）的"在每个完整动作上取小、再按策略求期望"，两者不等价；
        'per_head'（第一轮做法，"逐头取小"）：每头先取两评论家较小值再相加。
        auto_alpha：温度自适应，目标熵 = kappa × Σ_h log(合法动作数)。"""
        joint = self.cfg.get('double_q', 'joint') == 'joint'
        alpha = self.log_alpha.exp().detach() if self.auto_alpha else self.alpha
        obs, action, reward, done, nxt = self.replay.sample(self.batch)
        with torch.no_grad():
            pi_next = self._policy(self.actor, nxt)
            (q1n, _), (q2n, _) = self.q1_t(nxt), self.q2_t(nxt)
            ent_next = sum(-(pi * logp).sum(1) for pi, logp in pi_next)
            if joint:
                v1 = sum((pi * a1).sum(1) for (pi, _), a1 in zip(pi_next, q1n))
                v2 = sum((pi * a2).sum(1) for (pi, _), a2 in zip(pi_next, q2n))
                v_next = torch.minimum(v1, v2) + alpha * ent_next
            else:
                v_next = sum((pi * torch.minimum(a1, a2)).sum(1) for (pi, _), a1, a2 in zip(pi_next, q1n, q2n)) + alpha * ent_next
            target = reward + self.gamma * (1 - done) * v_next
        (q1, _), (q2, _) = self.q1(obs), self.q2(obs)
        q1_sa = sum(h.gather(1, action[:, i:i + 1]).squeeze(1) for i, h in enumerate(q1))
        q2_sa = sum(h.gather(1, action[:, i:i + 1]).squeeze(1) for i, h in enumerate(q2))
        critic_loss = F.mse_loss(q1_sa, target) + F.mse_loss(q2_sa, target)
        self.opt_critic.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 10.)
        self.opt_critic.step()
        pi_obs = self._policy(self.actor, obs)
        q1d, q2d = [h.detach() for h in q1], [h.detach() for h in q2]      # 复用本步评论家前向（已分离），省两次前向
        if joint:
            with torch.no_grad():
                v1 = sum((pi.detach() * a1).sum(1) for (pi, _), a1 in zip(pi_obs, q1d))
                v2 = sum((pi.detach() * a2).sum(1) for (pi, _), a2 in zip(pi_obs, q2d))
                pick1 = (v1 <= v2)[:, None]
            qsel = [torch.where(pick1, a1, a2) for a1, a2 in zip(q1d, q2d)]
        else:
            qsel = [torch.minimum(a1, a2) for a1, a2 in zip(q1d, q2d)]
        actor_loss, entropy = 0., 0.
        for (pi, logp), q in zip(pi_obs, qsel):
            actor_loss = actor_loss + (pi * (alpha * logp - q)).sum(1).mean()
            entropy = entropy - (pi * logp).sum(1).mean()
        self.opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.)
        self.opt_actor.step()
        if self.auto_alpha:
            n_legal = sum(torch.log(m.float().sum(1).clamp(min=1.)) for m in head_masks(obs))
            target_ent = self.cfg.get('alpha_kappa', .3) * n_legal
            ent_now = sum(-(pi * logp).sum(1) for pi, logp in pi_obs).detach()
            alpha_loss = -(self.log_alpha * (target_ent - ent_now)).mean()
            self.opt_alpha.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.opt_alpha.step()
        with torch.no_grad():
            for net, tgt in ((self.q1, self.q1_t), (self.q2, self.q2_t)):
                for p, pt in zip(net.parameters(), tgt.parameters()):
                    pt.mul_(1 - self.tau).add_(p, alpha=self.tau)
        self.updates += 1
        return dict(critic_loss=float(critic_loss), actor_loss=float(actor_loss), entropy=float(entropy), q=float(q1_sa.mean()),
                    alpha=float(alpha))

    def state_dict(self):
        return dict(actor=self.actor.state_dict(), q1=self.q1.state_dict(), q2=self.q2.state_dict(),
                    q1_t=self.q1_t.state_dict(), q2_t=self.q2_t.state_dict(), opt_actor=self.opt_actor.state_dict(),
                    opt_critic=self.opt_critic.state_dict(), env_steps=self.env_steps, updates=self.updates, pending=self._pending,
                    log_alpha=self.log_alpha.detach().clone(), opt_alpha=self.opt_alpha.state_dict())

    def load_state_dict(self, state):
        for name in ('actor', 'q1', 'q2', 'q1_t', 'q2_t', 'opt_actor', 'opt_critic'):
            getattr(self, name).load_state_dict(state[name])
        self.env_steps, self.updates, self._pending = state['env_steps'], state['updates'], state['pending']
        if 'log_alpha' in state:
            with torch.no_grad():
                self.log_alpha.copy_(state['log_alpha'])
            self.opt_alpha.load_state_dict(state['opt_alpha'])

    def policy_state(self):
        return dict(kind='sac', cfg=self.cfg, K=self.K, beam_dim=self.beam_dim, actor=self.actor.state_dict())


class DQN(AgentBase):
    def __init__(self, cfg, K, device, beam_dim=20):
        super().__init__(cfg, K, device, beam_dim)
        self.q = self.net()
        self.q_t = copy.deepcopy(self.q)
        for p in self.q_t.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.get('lr', 3e-4))
        self.gamma = cfg.get('gamma', .99)
        self.batch, self.warmup = cfg.get('batch', 256), cfg.get('warmup', 2000)
        self.eps_start, self.eps_end, self.eps_steps = cfg.get('eps_start', 1.), cfg.get('eps_end', .05), cfg.get('eps_steps', 100000)
        self.target_every = cfg.get('target_every', 1000)
        self.updates_per_step = cfg.get('utd', .5)
        self.replay = Replay(cfg.get('replay', 100000), device)
        self.env_steps = 0
        self._pending = 0.

    def main_module(self):
        return self.q

    def epsilon(self):
        f = min(1., self.env_steps / max(1, self.eps_steps))
        return self.eps_start + (self.eps_end - self.eps_start) * f

    @torch.no_grad()
    def act(self, obs, explore=True):
        masks = head_masks(obs)
        (gs, cnt, pw), _ = self.q(obs)
        greedy = torch.stack([masked(h, m).argmax(-1) for h, m in zip((gs, cnt, pw), masks)], 1)
        if not explore:
            return greedy
        rand = random_valid(masks)
        pick = torch.rand(greedy.shape[0], device=self.device) < self.epsilon()
        return torch.where(pick[:, None], rand, greedy)

    def observe(self, obs, action, reward, done, next_obs, valid):
        self.replay.add(obs, action, reward, done, next_obs, valid)
        self.env_steps += int(valid.sum())

    def update(self, step):
        if self.env_steps < self.warmup or self.replay.size < self.batch:
            return None
        self._pending += self.updates_per_step * self.cfg.get('batch_envs', 64)
        n = int(self._pending)
        self._pending -= n
        info = None
        for _ in range(n):
            info = self._update_once()
        return info

    def _update_once(self):
        obs, action, reward, done, nxt = self.replay.sample(self.batch)
        with torch.no_grad():
            (on, _) = self.q(nxt)
            (tg, _) = self.q_t(nxt)
            masks = head_masks(nxt)
            v_next = 0.
            for h_on, h_tg, m in zip(on, tg, masks):
                best = masked(h_on, m).argmax(-1, keepdim=True)
                v_next = v_next + h_tg.gather(1, best).squeeze(1)
            target = reward + self.gamma * (1 - done) * v_next
        (q, _) = self.q(obs)
        q_sa = sum(h.gather(1, action[:, i:i + 1]).squeeze(1) for i, h in enumerate(q))
        loss = F.smooth_l1_loss(q_sa, target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 10.)
        self.opt.step()
        self.updates += 1
        if self.updates % self.target_every == 0:
            self.q_t.load_state_dict(self.q.state_dict())
        return dict(loss=float(loss), q=float(q_sa.mean()), eps=self.epsilon())

    def state_dict(self):
        return dict(q=self.q.state_dict(), q_t=self.q_t.state_dict(), opt=self.opt.state_dict(),
                    env_steps=self.env_steps, updates=self.updates, pending=self._pending)

    def load_state_dict(self, state):
        self.q.load_state_dict(state['q']); self.q_t.load_state_dict(state['q_t']); self.opt.load_state_dict(state['opt'])
        self.env_steps, self.updates, self._pending = state['env_steps'], state['updates'], state['pending']

    def policy_state(self):
        return dict(kind='dqn', cfg=self.cfg, K=self.K, beam_dim=self.beam_dim, actor=self.q.state_dict())


class PPO(AgentBase):
    off_policy = False

    def __init__(self, cfg, K, device, beam_dim=20):
        super().__init__(cfg, K, device, beam_dim)
        self.net_ac = self.net(value=True)
        self.opt = torch.optim.Adam(self.net_ac.parameters(), lr=cfg.get('lr', 3e-4), eps=1e-5)
        self.gamma, self.lam = cfg.get('gamma', .99), cfg.get('lam', .95)
        self.rollout, self.epochs, self.minibatch = cfg.get('rollout', 32), cfg.get('epochs', 4), cfg.get('minibatch', 512)
        self.clip, self.vf_coef, self.ent_coef = cfg.get('clip', .2), cfg.get('vf_coef', .5), cfg.get('ent_coef', .01)
        self.buffer = []
        self.env_steps = 0
        self._last_logp = None
        self._last_value = None

    def main_module(self):
        return self.net_ac

    @torch.no_grad()
    def act(self, obs, explore=True):
        (gs, cnt, pw), value = self.net_ac(obs)
        action, logp = sample_heads((gs, cnt, pw), head_masks(obs), explore)
        self._last_logp, self._last_value = logp, value
        return action

    def observe(self, obs, action, reward, done, next_obs, valid):
        self.buffer.append(dict(obs=to_store(obs), action=action, reward=reward, done=done, valid=valid,
                                logp=self._last_logp, value=self._last_value))
        self.env_steps += int(valid.sum())
        self._next_obs = next_obs

    def update(self, step):
        if len(self.buffer) < self.rollout:
            return None
        with torch.no_grad():
            _, last_value = self.net_ac(self._next_obs)
        T = len(self.buffer)
        adv = [None] * T
        gae = torch.zeros_like(last_value)
        next_value = last_value
        for t in reversed(range(T)):
            b = self.buffer[t]
            nonterminal = 1 - b['done'].float()
            delta = b['reward'] + self.gamma * next_value * nonterminal - b['value']
            gae = delta + self.gamma * self.lam * nonterminal * gae
            adv[t] = gae
            next_value = b['value']
        valid = torch.cat([b['valid'] for b in self.buffer])
        obs = {k: torch.cat([b['obs'][k] for b in self.buffer])[valid] for k in self.buffer[0]['obs']}
        action = torch.cat([b['action'] for b in self.buffer])[valid]
        old_logp = torch.cat([b['logp'] for b in self.buffer])[valid]
        value_old = torch.cat([b['value'] for b in self.buffer])[valid]
        advantage = torch.cat(adv)[valid]
        returns = advantage + value_old
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        n = len(action)
        info = {}
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, self.minibatch):
                idx = perm[i:i + self.minibatch]
                mb = to_model({k: v[idx] for k, v in obs.items()})
                (gs, cnt, pw), value = self.net_ac(mb)
                logp, entropy = 0., 0.
                for j, (lg, m) in enumerate(zip((gs, cnt, pw), head_masks(mb))):
                    lp = F.log_softmax(masked(lg, m), -1)
                    logp = logp + lp.gather(1, action[idx, j:j + 1]).squeeze(1)
                    entropy = entropy - (lp.exp() * m * lp).sum(1)
                ratio = torch.exp(logp - old_logp[idx])
                a = advantage[idx]
                policy_loss = -torch.minimum(ratio * a, ratio.clamp(1 - self.clip, 1 + self.clip) * a).mean()
                value_loss = F.mse_loss(value, returns[idx])
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy.mean()
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net_ac.parameters(), .5)
                self.opt.step()
                info = dict(policy_loss=float(policy_loss), value_loss=float(value_loss), entropy=float(entropy.mean()))
        self.updates += 1
        self.buffer = []
        return info

    def state_dict(self):
        return dict(net=self.net_ac.state_dict(), opt=self.opt.state_dict(), env_steps=self.env_steps, updates=self.updates)

    def load_state_dict(self, state):
        self.net_ac.load_state_dict(state['net']); self.opt.load_state_dict(state['opt'])
        self.env_steps, self.updates = state['env_steps'], state['updates']

    def policy_state(self):
        return dict(kind='ppo', cfg=self.cfg, K=self.K, beam_dim=self.beam_dim, actor=self.net_ac.state_dict())


def make_agent(algo, cfg, K, device, beam_dim=20):
    if algo == 'sac3':
        from .sac3 import SAC3
        return SAC3(cfg, K, device, beam_dim)
    cls = dict(sac=SAC, dqn=DQN, ppo=PPO)[algo]
    return cls(cfg, K, device, beam_dim)


def load_policy(path, device):
    saved = torch.load(path, map_location=device, weights_only=False)
    agent = make_agent(saved['kind'], saved['cfg'], saved['K'], device, saved['beam_dim'])
    agent.main_module().load_state_dict(saved['actor'])
    agent.main_module().eval()
    return agent
