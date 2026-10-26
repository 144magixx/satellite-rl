"""第三轮 SAC（SAC3）：在 14 令牌、规则定组（每步只有一个允许组）的设定下，把三个决定写成完整动作上的分布。

记号：o 观测，c 槽数（1—10），x 本组内起点（0—T-1），p 功率。完整动作 a=(c,x,p)。
策略（cfg['policy']）：
  indep  R0：三个头各自独立，π(c)·π(x)·π(p)；起点掩码＝空闲起点，执行时槽数可被截短（与第二轮相同）。
  legal  L ：先抽 c，起点只允许"本组从这里起 c 个槽全空"的完整窗口：π(c)·π(x|c)，π(x|c) 用同一组起点 logits 加 c 的掩码；功率独立。
  cond   A ：起点头和功率头看得到 c：π(c)·π(x|o,c)·π(p|o,c)。cfg['topk']>0 时只留每个 c 下干扰最低的 topk 个窗口（B′）。
功率（cfg['power']）：discrete 13 档；continuous 连续，p＝0.5 W＋u·(pmax−0.5 W)，u＝sigmoid(z)，z~N(μ,σ)。
  cfg['pi_power']：count π(p|o,c)；count_start π(p|o,c,x)。
评论家：Q(o,a)＝q_c(o,c)＋q_x(o,x)＋q_p，q_p 按 cfg['q_power']：plain q_p(o,p)；count q_p(o,c,p)；count_start q_p(o,c,x,p)
  （连续功率时 q_p 是以瓦数为输入的小网络）。
双 Q（cfg['double_q']）：joint＝"求期望后取小" min(E_w Q1, E_w Q2)（第二轮做法）；hist＝"取小后求期望" E_w min(Q1,Q2)（历史做法）。
期望、熵都在完整动作的联合分布 w(a)=π(c)π(x|c)π(p|c[,x]) 上精确求和（连续功率每个 (c[,x]) 采一个样本）。
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .agents import AgentBase, Replay, to_model
from .masks import conditional_masks, group_index
from .nets import PolicyNet, masked, count_parameters, NEG

C = 10
P_MIN = .5


def _gather_group(t, idx):
    """t[..., G*T] 取允许组的 T 个位置：idx[B,T]。t 形状 [B,G*T] 或 [B,10,G*T]。"""
    if t.dim() == 2:
        return t.gather(1, idx)
    return t.gather(2, idx[:, None, :].expand(-1, t.shape[1], -1))


class Net3(nn.Module):
    def __init__(self, cfg, K, role):
        super().__init__()
        self.cfg, self.role, self.K = cfg, role, K
        self.base = PolicyNet(cfg['encoder'], cfg.get('size'), K, False, False, 20)
        d = self.base.encoder.out_dim
        G, T = 8, 100
        self.T = T
        pol, power = cfg.get('policy', 'indep'), cfg.get('power', 'discrete')
        qp, pip = cfg.get('q_power', 'plain'), cfg.get('pi_power', 'count')
        if role == 'actor' and pol == 'cond':
            self.c_emb = nn.Embedding(C, 16)
            self.cond = nn.Sequential(nn.Linear(d + 16, d), nn.ReLU())
            self.gs_c = nn.Linear(d, G * T)
            if power == 'discrete':
                self.pw_c = nn.Linear(d, K)
        if role == 'actor' and power == 'continuous':
            self.pc_emb = nn.Embedding(C, 16)
            self.p_h = nn.Linear(d, 64)
            extra = 0
            if pip == 'count_start':
                self.p_x, self.p_off = nn.Linear(d, 32), nn.Embedding(10, 32)
                extra = 32
            self.p_mlp = nn.Sequential(nn.Linear(64 + 16 + extra, 64), nn.ReLU(), nn.Linear(64, 2))
        if role == 'critic' and power == 'discrete' and qp == 'count':
            self.qc_emb = nn.Embedding(C, 16)
            self.q_cond = nn.Sequential(nn.Linear(d + 16, d), nn.ReLU())
            self.qp_c = nn.Linear(d, K)
        if role == 'critic' and power == 'continuous':
            self.qc_emb = nn.Embedding(C, 16)
            self.q_h = nn.Linear(d, 64)
            extra = 0
            if qp == 'count_start':
                self.q_x, self.q_off = nn.Linear(d, 32), nn.Embedding(10, 32)
                extra = 32
            self.q_mlp = nn.Sequential(nn.Linear(64 + 16 + extra + 1, 64), nn.ReLU(), nn.Linear(64, 1))

    def _slot_feats(self, blocks, lin, off):
        """块令牌输出 [B,10,d] → 每个起点的特征 [B,T,32]：所在块的投影＋块内偏移嵌入。"""
        z = lin(blocks)                                                  # [B,10,32]
        s = torch.arange(self.T, device=blocks.device)
        return z[:, s // 10] + off(s % 10)[None]

    def forward(self, obs):
        pooled, blocks = self.base.encoder(obs)
        gs, cnt, pw = self.base.heads(pooled, blocks)
        g = group_index(obs['allow'])
        idx = g[:, None] * self.T + torch.arange(self.T, device=g.device)[None, :]
        out = dict(pooled=pooled, blocks=blocks, idx=idx, cnt=cnt, gs=_gather_group(gs, idx), pw=pw)
        cfg = self.cfg
        if self.role == 'actor' and cfg.get('policy') == 'cond':
            h = self.cond(torch.cat([pooled[:, None].expand(-1, C, -1), self.c_emb.weight[None].expand(pooled.shape[0], -1, -1)], -1))
            out['gs_c'] = _gather_group(self.gs_c(h), idx)                 # [B,10,T]
            if cfg.get('power', 'discrete') == 'discrete':
                out['pw_c'] = self.pw_c(h)                                  # [B,10,K]
        if self.role == 'actor' and cfg.get('power') == 'continuous':
            hp = self.p_h(pooled)                                           # [B,64]
            B = pooled.shape[0]
            if cfg.get('pi_power', 'count') == 'count':
                inp = torch.cat([hp[:, None].expand(-1, C, -1), self.pc_emb.weight[None].expand(B, -1, -1)], -1)   # [B,10,80]
            else:
                zx = self._slot_feats(blocks, self.p_x, self.p_off)       # [B,T,32]
                inp = torch.cat([hp[:, None, None].expand(-1, C, self.T, -1), self.pc_emb.weight[None, :, None].expand(B, -1, self.T, -1),
                                 zx[:, None].expand(-1, C, -1, -1)], -1)     # [B,10,T,112]
            ms = self.p_mlp(inp)
            out['mu'], out['logstd'] = ms[..., 0], ms[..., 1].clamp(-5., 1.)
        if self.role == 'critic' and cfg.get('power', 'discrete') == 'discrete' and cfg.get('q_power', 'plain') == 'count':
            h = self.q_cond(torch.cat([pooled[:, None].expand(-1, C, -1), self.qc_emb.weight[None].expand(pooled.shape[0], -1, -1)], -1))
            out['pw'] = self.qp_c(h)                                        # [B,10,K]
        if self.role == 'critic' and cfg.get('power') == 'continuous':
            out['hq'] = self.q_h(pooled)
            if cfg.get('q_power') == 'count_start':
                out['zq'] = self._slot_feats(blocks, self.q_x, self.q_off)
        return out

    def q_power(self, out, p_w, c=None, x=None):
        """连续功率的 q_p。p_w 为瓦数：形状 [B,10]（count）或 [B,10,T]（count_start）时对全部 c[,x] 求值；
        给出 c[B]（0—9）、x[B] 时 p_w 为 [B]，只对所选动作求值。"""
        hq = out['hq']
        B = hq.shape[0]
        start = self.cfg.get('q_power') == 'count_start'
        if c is not None:
            parts = [hq, self.qc_emb(c)]
            if start:
                parts.append(out['zq'][torch.arange(B, device=hq.device), x])
            return self.q_mlp(torch.cat(parts + [p_w[:, None] / 50.], -1)).squeeze(-1)
        if not start:
            inp = torch.cat([hq[:, None].expand(-1, C, -1), self.qc_emb.weight[None].expand(B, -1, -1), p_w[..., None] / 50.], -1)
        else:
            inp = torch.cat([hq[:, None, None].expand(-1, C, self.T, -1), self.qc_emb.weight[None, :, None].expand(B, -1, self.T, -1),
                             out['zq'][:, None].expand(-1, C, -1, -1), p_w[..., None] / 50.], -1)
        return self.q_mlp(inp).squeeze(-1)


class SAC3(AgentBase):
    def __init__(self, cfg, K, device, beam_dim=20):
        super().__init__(cfg, K, device, beam_dim)
        cfg.setdefault('policy', 'indep')
        cfg.setdefault('power', 'discrete')
        cfg.setdefault('q_power', 'plain')
        cfg.setdefault('pi_power', 'count')
        cfg.setdefault('double_q', 'joint')
        self.actor = Net3(cfg, K, 'actor').to(device)
        self.q1, self.q2 = Net3(cfg, K, 'critic').to(device), Net3(cfg, K, 'critic').to(device)
        self.q1_t, self.q2_t = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        for p in list(self.q1_t.parameters()) + list(self.q2_t.parameters()):
            p.requires_grad_(False)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.get('actor_lr', 1e-5))
        self.opt_critic = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.get('critic_lr', 1e-4))
        self.alpha = float(cfg.get('alpha', .01))
        self.alpha_p = float(cfg.get('alpha_p', .01))
        self.auto_alpha = bool(cfg.get('auto_alpha', False))
        self.log_alpha = torch.tensor(math.log(self.alpha), device=device, requires_grad=True)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=cfg.get('alpha_lr', 3e-4))
        self.gamma, self.tau = cfg.get('gamma', .99), cfg.get('tau', .005)
        self.batch, self.warmup = cfg.get('batch', 256), cfg.get('warmup', 2000)
        self.updates_per_step = cfg.get('utd', 1.)
        self.replay = Replay(cfg.get('replay', 20000), device)
        self.power_buf = torch.zeros(cfg.get('replay', 20000), device=device)
        self.env_steps = 0
        self._pending = 0.
        self._last_power = None
        self.continuous = cfg['power'] == 'continuous'

    def main_module(self):
        return self.actor

    # ------------------------------------------------------------ 分布
    def _dist(self, out, obs, uniform=False):
        """返回 logp_c[B,10]、logp_x[B,10,T]、离散功率 logp_p[B,10,K]（连续时为 None）及各掩码。"""
        cfg = self.cfg
        B = out['cnt'].shape[0]
        if cfg['policy'] == 'indep':
            mc = obs['mask_cnt']
            mx = obs['mask_gs'].gather(1, out['idx'])[:, None, :].expand(-1, C, -1)
        else:
            mc, mx = conditional_masks(obs['occ'], obs['allow'], obs['cmax'], cfg.get('topk', 0), obs.get('intf_rel'))
        lc = out['cnt']
        lx = out['gs_c'] if cfg['policy'] == 'cond' else out['gs'][:, None, :].expand(-1, C, -1)
        if uniform:
            lc, lx = torch.zeros_like(lc), torch.zeros_like(lx)
        logp_c = F.log_softmax(masked(lc, mc), -1)
        logp_x = F.log_softmax(masked(lx, mx), -1)
        logp_p, mp = None, None
        if not self.continuous:
            mp = obs['mask_pw'][:, None, :].expand(-1, C, -1)
            lp = out['pw_c'] if cfg['policy'] == 'cond' else out['pw'][:, None, :].expand(-1, C, -1)
            if uniform:
                lp = torch.zeros_like(lp)
            logp_p = F.log_softmax(masked(lp, mp), -1)
        return logp_c, logp_x, logp_p, (mc, mx, mp)

    def _sample_power(self, out, obs, explore=True):
        """连续功率：对每个 (c) 或 (c,x) 采一个样本。返回瓦数、u 空间对数密度（退化区间为 0）。"""
        mu, logstd = out['mu'], out['logstd']
        z = mu + logstd.exp() * torch.randn_like(mu) if explore else mu
        logp = torch.distributions.Normal(mu, logstd.exp()).log_prob(z) - (F.logsigmoid(z) + F.logsigmoid(-z))
        u = torch.sigmoid(z)
        span = (obs['pmax'] - P_MIN).clamp(min=0)
        span = span.view(-1, *([1] * (mu.dim() - 1)))
        p = P_MIN + u * span
        logp = torch.where(span > 1e-3, logp, torch.zeros_like(logp))
        return p, logp

    # ------------------------------------------------------------ 行动
    @torch.no_grad()
    def act(self, obs, explore=True):
        uniform = explore and self.env_steps < self.warmup
        out = self.actor(obs)
        logp_c, logp_x, logp_p, _ = self._dist(out, obs, uniform)
        B = logp_c.shape[0]
        rows = torch.arange(B, device=self.device)
        pick = (lambda lg: torch.distributions.Categorical(logits=lg).sample()) if explore else (lambda lg: lg.argmax(-1))
        c = pick(logp_c)
        x = pick(logp_x[rows, c])
        g = group_index(obs['allow'])
        if self.continuous:
            if self.cfg['pi_power'] == 'count':
                mu, logstd = out['mu'][rows, c], out['logstd'][rows, c]
            else:
                mu, logstd = out['mu'][rows, c, x], out['logstd'][rows, c, x]
            if uniform:
                u = torch.rand(B, device=self.device)
            else:
                z = mu + logstd.exp() * torch.randn_like(mu) if explore else mu
                u = torch.sigmoid(z)
            power = P_MIN + u * (obs['pmax'] - P_MIN).clamp(min=0)
            self._last_power = power
            action = torch.stack([g * 100 + x, c, torch.zeros_like(c)], 1)
            return action, power
        p = pick(logp_p[rows, c])
        return torch.stack([g * 100 + x, c, p], 1)

    def observe(self, obs, action, reward, done, next_obs, valid):
        if self.continuous:
            idx = torch.nonzero(valid).flatten()
            slots = (self.replay.pos + torch.arange(len(idx), device=self.device)) % self.replay.capacity
            self.power_buf[slots] = self._last_power[idx]
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

    # ------------------------------------------------------------ 期望
    def _q_full(self, out, logp_c, logp_x, logp_p, p_w=None):
        """全部完整动作上的 Q：离散 [B,10,T,K]；连续 [B,10,T]（功率为每个 (c[,x]) 的采样值 p_w）。"""
        qc, qx = out['cnt'], out['gs']
        if not self.continuous:
            qp = out['pw']
            qp = qp[:, None, None, :] if qp.dim() == 2 else qp[:, :, None, :]
            return qc[:, :, None, None] + qx[:, None, :, None] + qp
        net = (self.q1, self.q2, self.q1_t, self.q2_t)[out['net'] - 1]
        qp = net.q_power(out, p_w)
        if qp.dim() == 2:
            qp = qp[:, :, None]
        return qc[:, :, None] + qx[:, None, :] + qp

    def _weights(self, logp_c, logp_x, logp_p):
        if self.continuous:
            logw = logp_c[:, :, None] + logp_x                                           # [B,10,T]
        else:
            logw = logp_c[:, :, None, None] + logp_x[:, :, :, None] + logp_p[:, :, None, :]   # [B,10,T,K]
        return logw.exp(), logw

    def _critic_outs(self, obs, target):
        nets = (self.q1_t, self.q2_t) if target else (self.q1, self.q2)
        outs = []
        for i, net in enumerate(nets):
            o = net(obs)
            o['net'] = (3 + i) if target else (1 + i)
            outs.append(o)
        return outs

    def _update_once(self):
        cfg = self.cfg
        hist = cfg['double_q'] == 'hist'
        alpha = self.log_alpha.exp().detach() if self.auto_alpha else self.alpha
        obs, action, reward, done, nxt, power = self._sample()
        dims = (1, 2) if self.continuous else (1, 2, 3)
        # ---- 目标值
        with torch.no_grad():
            a_out = self.actor(nxt)
            lc, lx, lp, _ = self._dist(a_out, nxt)
            w, logw = self._weights(lc, lx, lp)
            t1, t2 = self._critic_outs(nxt, True)
            if self.continuous:
                p_w, logpu = self._sample_power(a_out, nxt)
                Q1, Q2 = self._q_full(t1, lc, lx, lp, p_w), self._q_full(t2, lc, lx, lp, p_w)
                logpu = logpu[:, :, None].expand_as(w) if logpu.dim() == 2 else logpu
                ent = -(w * logw).sum(dims) * alpha - (w * logpu).sum(dims) * self.alpha_p
            else:
                Q1, Q2 = self._q_full(t1, lc, lx, lp), self._q_full(t2, lc, lx, lp)
                ent = -(w * logw).sum(dims) * alpha
            if hist:
                v = (w * torch.minimum(Q1, Q2)).sum(dims)
            else:
                v = torch.minimum((w * Q1).sum(dims), (w * Q2).sum(dims))
            target = reward + self.gamma * (1 - done) * (v + ent)
        # ---- 评论家
        o1, o2 = self._critic_outs(obs, False)
        rows = torch.arange(len(reward), device=self.device)
        g = group_index(obs['allow'])
        x = action[:, 0] - g * 100
        c = action[:, 1]

        def q_taken(o, net):
            q = o['cnt'][rows, c] + o['gs'][rows, x]
            if self.continuous:
                return q + net.q_power(o, power, c, x)
            pw = o['pw']
            return q + (pw[rows, action[:, 2]] if pw.dim() == 2 else pw[rows, c, action[:, 2]])
        q1_sa, q2_sa = q_taken(o1, self.q1), q_taken(o2, self.q2)
        critic_loss = F.mse_loss(q1_sa, target) + F.mse_loss(q2_sa, target)
        self.opt_critic.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 10.)
        self.opt_critic.step()
        # ---- 演员（评论家输出已分离；连续功率保留 Q 对功率的导数）
        a_out = self.actor(obs)
        lc, lx, lp, (mc, mx, mp) = self._dist(a_out, obs)
        w, logw = self._weights(lc, lx, lp)
        d1 = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in o1.items()}
        d2 = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in o2.items()}
        if self.continuous:
            p_w, logpu = self._sample_power(a_out, obs)
            Q1, Q2 = self._q_full(d1, lc, lx, lp, p_w), self._q_full(d2, lc, lx, lp, p_w)
            logpu = logpu[:, :, None].expand_as(w) if logpu.dim() == 2 else logpu
        else:
            Q1, Q2 = self._q_full(d1, lc, lx, lp), self._q_full(d2, lc, lx, lp)
            logpu = torch.zeros_like(w)
        if hist:
            q_sel = torch.minimum(Q1, Q2)
        else:
            with torch.no_grad():
                pick1 = ((w * Q1).sum(dims) <= (w * Q2).sum(dims)).view(-1, *([1] * (w.dim() - 1)))
            q_sel = torch.where(pick1, Q1, Q2)
        if not self.continuous:
            q_sel = q_sel.detach()
        actor_loss = (w * (alpha * logw + self.alpha_p * logpu - q_sel)).sum(dims).mean()
        self.opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.)
        self.opt_actor.step()
        entropy = float(-(w * logw).sum(dims).mean())
        if self.auto_alpha:
            if self.continuous:
                legal = (mc[:, :, None] & mx).float().sum((1, 2))
            else:
                legal = (mc[:, :, None, None] & mx[:, :, :, None] & mp[:, :, None, :]).float().sum((1, 2, 3))
            target_ent = cfg.get('alpha_kappa', .3) * torch.log(legal.clamp(min=1.))
            ent_now = -(w * logw).sum(dims).detach()
            alpha_loss = -(self.log_alpha * (target_ent - ent_now)).mean()
            self.opt_alpha.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.opt_alpha.step()
        with torch.no_grad():
            for net, tgt in ((self.q1, self.q1_t), (self.q2, self.q2_t)):
                for p, pt in zip(net.parameters(), tgt.parameters()):
                    pt.mul_(1 - self.tau).add_(p, alpha=self.tau)
        self.updates += 1
        return dict(critic_loss=float(critic_loss), actor_loss=float(actor_loss), entropy=entropy, q=float(q1_sa.mean()),
                    alpha=float(alpha))

    def _sample(self):
        idx = torch.randint(0, self.replay.size, (self.batch,), device=self.device)
        r = self.replay
        obs = to_model({k: v[idx] for k, v in r.obs.items()})
        nxt = to_model({k: v[idx] for k, v in r.next_obs.items()})
        return obs, r.action[idx], r.reward[idx], r.done[idx].float(), nxt, self.power_buf[idx]

    # ------------------------------------------------------------ 存取
    def state_dict(self):
        return dict(actor=self.actor.state_dict(), q1=self.q1.state_dict(), q2=self.q2.state_dict(),
                    q1_t=self.q1_t.state_dict(), q2_t=self.q2_t.state_dict(), opt_actor=self.opt_actor.state_dict(),
                    opt_critic=self.opt_critic.state_dict(), env_steps=self.env_steps, updates=self.updates, pending=self._pending,
                    log_alpha=self.log_alpha.detach().clone(), opt_alpha=self.opt_alpha.state_dict(), power_buf=self.power_buf)

    def load_state_dict(self, state):
        for name in ('actor', 'q1', 'q2', 'q1_t', 'q2_t', 'opt_actor', 'opt_critic', 'opt_alpha'):
            getattr(self, name).load_state_dict(state[name])
        self.env_steps, self.updates, self._pending = state['env_steps'], state['updates'], state['pending']
        with torch.no_grad():
            self.log_alpha.copy_(state['log_alpha'])
        self.power_buf = state['power_buf']

    def policy_state(self):
        return dict(kind='sac3', cfg=self.cfg, K=self.K, beam_dim=self.beam_dim, actor=self.actor.state_dict())

    def parameter_count(self):
        return count_parameters(self.actor)
