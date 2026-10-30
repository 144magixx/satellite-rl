"""GPU 批量环境：B 个场景并行，每步给当前波束定（组×起点、槽数、功率档）。

状态张量全部驻留 device；一步的代价是几次 [B,N,100] 级别的张量运算。
掩码模式 full：组×起点只能选允许组内的空闲槽，槽数超过空闲连续段时截断，功率不超预算减预留。
掩码模式 none：不做任何限制；与同组已占槽重叠的波束得 0 槽（不服务），功率超预算则回合终止、其余波束不服务。
奖励 paper：±1（当前束是否满足）+ 当前束 SGM − 超 50 Mbps 偏差惩罚（已发表口径，只看先分配波束对本束的干扰）。
奖励 delta：场景目标 J=Σ(SGM_i+min(SI_i,1)) 在本步前后的增量（含本束对已分配波束的干扰影响）。
"""
from dataclasses import dataclass
import math

import numpy as np
import torch

from . import features as feature_mod
from .masks import free_run as _free_run_of
from .physics import PhysicsV3, path_gain
from .scenes import DatasetConfig, load_ratio, stratum_of, STRATA, load_scene

DEFAULT_POWER_LEVELS = (0.5, 1., 2., 5., 10., 15., 20., 25., 30., 35., 40., 45., 50.)


class SceneBank:
    """一个集合的全部场景，填充到 Nmax 后放到 device。"""

    def __init__(self, dataset_dir, keys, physics, device, nmax=None):
        scenes = [load_scene(dataset_dir, k) for k in keys]
        self.keys = list(keys)
        self.n = torch.tensor([s.n for s in scenes], dtype=torch.long, device=device)
        nmax = nmax or int(self.n.max())
        self.nmax = nmax
        S = len(scenes)
        self.members = torch.zeros(S, nmax, dtype=torch.float32, device=device)
        self.width = torch.zeros(S, nmax, device=device)
        self.latlon = torch.zeros(S, nmax, 2, device=device)
        self.sig = torch.zeros(S, nmax, device=device)
        self.path = torch.zeros(S, nmax, physics.slots, device=device)
        self.coupling = torch.zeros(S, nmax, nmax, device=device)
        self.adjacent = torch.zeros(S, nmax, nmax, dtype=torch.bool, device=device)
        self.aircraft = torch.tensor([s.aircraft_count for s in scenes], dtype=torch.float32, device=device)
        for i, s in enumerate(scenes):
            n = s.n
            self.members[i, :n] = torch.from_numpy(s.members.astype(np.float32))
            self.width[i, :n] = torch.from_numpy(s.width_deg)
            self.latlon[i, :n] = torch.from_numpy(s.lat_lon)
            self.sig[i, :n] = torch.from_numpy(s.sig)
            self.path[i, :n] = torch.from_numpy(path_gain(s.distance_m, physics).astype(np.float32))
            self.coupling[i, :n, :n] = torch.from_numpy(s.coupling)
            self.adjacent[i, :n, :n] = torch.from_numpy(s.adjacent)
        self.valid = torch.arange(nmax, device=device)[None, :] < self.n[:, None]
        self.members_total = self.members.sum(1)

    def __len__(self):
        return len(self.keys)


@dataclass
class EnvOptions:
    batch_size: int = 64
    mask_mode: str = 'full'       # full | none
    group_mode: str = 'free'      # free | mod8 | greedy
    reward_mode: str = 'paper'    # paper | delta | delta_intf（delta 再罚本步给已放波束新增的干扰损失）
    order: str = 'demand'         # demand | random
    power_levels: tuple = DEFAULT_POWER_LEVELS
    beam_tokens: bool = False     # 是否构造每束一令牌观测
    legacy_obs: bool = False      # 额外给 glob[B,3]（平均需求、进度、剩余总需求）与 intf_abs（绝对 dBW 映射），供 14 令牌编码器
    start_mode: str = 'free'      # free | rule（起点只许干扰最小的窗口）| topk（只许干扰最小的前 k 个窗口）
    start_topk: int = 8
    start_window: int = 4         # 起点打分用的窗口长度（槽）
    illegal_penalty: float = 0.   # 无掩码时：撞已占槽/超预算的额外罚分；超预算终止时其余每束再罚 1（>0 时生效）
    intf_lambda: float = 1.       # delta_intf 的干扰项权重
    features: tuple = ()          # 第三轮状态补充特征（features.FEATURE_NAMES 的子集）
    intf_rel: bool = False        # 观测里给线性 I/N（按槽数筛低干扰窗口时用）


class ItemSampler:
    """训练：按分层比例重采样 (场景, 单机需求)；评价：按固定清单顺序出队。"""

    def __init__(self, bank, config, items, train, seed):
        self.bank, self.config, self.train = bank, config, train
        self.rng = np.random.default_rng(seed)
        self.index = {k: i for i, k in enumerate(bank.keys)}
        self.items = [(self.index[it['key']], it.get('per_aircraft_bps')) for it in items if it['key'] in self.index]
        self.cursor = 0
        self.exhausted = False
        if train:
            lo, hi = config.per_aircraft_bps_range
            totals = bank.members_total.cpu().numpy()
            draws = self.rng.uniform(lo, hi, (4000,))
            scenes = self.rng.integers(0, len(self.items), 4000)
            natural = np.zeros(3)
            for sc, v in zip(scenes, draws):
                natural[STRATA.index(stratum_of(load_ratio(totals[self.items[sc][0]], v, config), config))] += 1
            natural = np.maximum(natural / natural.sum(), 1e-3)
            weight = np.asarray(config.train_strata_share) / natural
            self.accept = weight / weight.max()
            self.totals = totals

    def next(self):
        if self.train:
            lo, hi = self.config.per_aircraft_bps_range
            while True:
                sc, _ = self.items[int(self.rng.integers(0, len(self.items)))]
                v = float(self.rng.uniform(lo, hi))
                s = STRATA.index(stratum_of(load_ratio(self.totals[sc], v, self.config), self.config))
                if self.rng.uniform() <= self.accept[s]:
                    return sc, v
        if self.cursor >= len(self.items):
            self.exhausted = True
            return None
        item = self.items[self.cursor]
        self.cursor += 1
        return item


class BatchEnv:
    def __init__(self, bank, config, options, device, sampler, auto_reset=True):
        self.bank, self.config, self.opt, self.device, self.sampler = bank, config, options, device, sampler
        self.auto_reset = auto_reset      # False：场景结束后保留分配供读取（元启发式评价用）
        self.physics = config.physics_obj()
        p = self.physics
        self.B, self.N, self.T, self.G = options.batch_size, bank.nmax, p.slots, p.groups
        self.levels = torch.tensor(options.power_levels, dtype=torch.float32, device=device)
        self.K = len(options.power_levels)
        self.noise = p.noise_w
        self.budget = float(config.power_budget_w)
        self.reserve = float(config.reserve_w_per_beam)
        self.cap_w = float(config.beam_power_cap_w)
        self.arange_t = torch.arange(self.T, device=device)
        self.arange_b = torch.arange(self.B, device=device)
        self.arange_n = torch.arange(self.N, device=device)
        B, N, T, G = self.B, self.N, self.T, self.G
        self.scene = torch.zeros(B, dtype=torch.long, device=device)
        self.alive = torch.zeros(B, dtype=torch.bool, device=device)   # 槽位是否装着场景（评价集耗尽后为 False）
        self.n = torch.zeros(B, dtype=torch.long, device=device)
        self.demand = torch.zeros(B, N, device=device)
        self.per_aircraft = torch.zeros(B, device=device)
        self.order = torch.zeros(B, N, dtype=torch.long, device=device)
        self.t = torch.zeros(B, dtype=torch.long, device=device)
        self.grp = torch.full((B, N), -1, dtype=torch.long, device=device)
        self.start = torch.zeros(B, N, dtype=torch.long, device=device)
        self.cnt = torch.zeros(B, N, dtype=torch.long, device=device)
        self.req = torch.zeros(B, N, dtype=torch.long, device=device)     # 申请的槽数（执行时可能被截短）
        self.pw = torch.zeros(B, N, device=device)
        self.A = torch.zeros(B, 2, N, T, device=device)
        self.M = torch.zeros(B, 2, N, T, device=device)
        self.occ = torch.zeros(B, G, T, dtype=torch.bool, device=device)
        self.used_w = torch.zeros(B, device=device)
        self.terminated = torch.zeros(B, dtype=torch.bool, device=device)
        self.objective = torch.zeros(B, device=device)
        self.episode_items = [None] * B
        self.finished = []     # 刚结束场景的指标字典列表（由 step 填充，调用方取走）
        for b in range(B):
            self._load(b)

    # ------------------------------------------------------------ 场景装载
    def _load(self, b):
        item = self.sampler.next()
        if item is None:
            self.alive[b] = False
            self.n[b] = 0
            return
        sc, v = item
        self.episode_items[b] = (self.bank.keys[sc], v)
        self.alive[b] = True
        self.scene[b] = sc
        n = int(self.bank.n[sc])
        self.n[b] = n
        self.per_aircraft[b] = v
        self.demand[b] = self.bank.members[sc] * v
        if self.opt.order == 'demand':
            key = -self.demand[b] + (~self.bank.valid[sc]).float() * 1e30
            self.order[b] = torch.argsort(key)
        else:
            perm = torch.randperm(n, device=self.device)
            self.order[b, :n] = perm
            self.order[b, n:] = torch.arange(n, self.N, device=self.device)
        self.t[b] = 0
        self.grp[b] = -1
        self.start[b] = 0
        self.cnt[b] = 0
        self.req[b] = 0
        self.pw[b] = 0.
        self.A[b] = 0.
        self.M[b] = 0.
        self.occ[b] = False
        self.used_w[b] = 0.
        self.terminated[b] = False
        self.objective[b] = 0.

    @property
    def cur(self):
        return self.order[self.arange_b, self.t.clamp(max=self.N - 1)]

    @property
    def active(self):
        return self.alive & ~self.terminated & (self.t < self.n)

    # ------------------------------------------------------------ 物理量
    def _interference(self, receivers):
        """当前接收波束在两种极化上每槽受到的干扰功率 [B,2,T]（含交叉极化泄漏）。"""
        m = self.M[self.arange_b, :, receivers]                      # [B,2,T]
        path = self.bank.path[self.scene, receivers]                  # [B,T]
        xpd = self.physics.xpd_linear
        same = m
        other = m.flip(1)
        return path[:, None, :] * (same + xpd * other)

    def capacities(self, interference=True):
        """全部波束当前容量 [B,N]（bps）。"""
        scene = self.scene
        parity = (self.grp % 2).clamp(min=0)                          # 未分配按 0，对应 A 为 0
        own = torch.gather(self.A, 1, parity[:, None, :, None].expand(-1, 1, -1, self.T))[:, 0]   # [B,N,T]
        path = self.bank.path[scene]                                  # [B,N,T]
        desired = self.bank.sig[scene][:, :, None] * path * own
        if interference:
            same = torch.gather(self.M, 1, parity[:, None, :, None].expand(-1, 1, -1, self.T))[:, 0]
            other = torch.gather(self.M, 1, (1 - parity)[:, None, :, None].expand(-1, 1, -1, self.T))[:, 0]
            inter = path * (same + self.physics.xpd_linear * other)
        else:
            inter = torch.zeros_like(desired)
        sinr = desired / (self.noise + inter) * self.physics.impl_linear
        eff = torch.log2(1 + sinr).clamp(max=self.physics.spectral_efficiency_cap) * (own > 0)
        return self.physics.slot_Hz * eff.sum(-1)

    def _sgm(self, capacity):
        demand = self.demand
        beta = demand.sum(1, keepdim=True) / self.n.clamp(min=1)[:, None]
        ratio = capacity / demand.clamp(min=1.)
        real = torch.where(ratio >= 1, ratio - 1, 1 - 1 / ratio.clamp(min=1e-9))
        imag = (capacity - demand) / beta
        sgm = 1 - (1 - torch.exp(-torch.hypot(real, imag))) ** 3
        return torch.where(capacity > 0, sgm, torch.zeros_like(sgm)) * self.bank.valid[self.scene]

    def _intf_si_loss(self, capacity):
        """各束因干扰损失的满足率之和（无干扰 SI − 有干扰 SI），[B]。"""
        valid = self.bank.valid[self.scene]
        d = self.demand.clamp(min=1.)
        cap0 = self.capacities(interference=False)
        return (((cap0 / d).clamp(max=1.) - (capacity / d).clamp(max=1.)) * valid).sum(1)

    def _objective(self, capacity):
        si = (capacity / self.demand.clamp(min=1.)).clamp(max=1.) * self.bank.valid[self.scene]
        return (self._sgm(capacity) + si).sum(1)

    # ------------------------------------------------------------ 掩码与观测
    def _free_run(self):
        """free_run[b,g,s] = 从 s 起连续空闲槽数（s 已占为 0）。"""
        return _free_run_of(self.occ)

    def count_cap(self, allow, free_run=None):
        """每束槽数上限：允许组内最长空闲段，且给后面每个波束至少留 1 个槽（full 模式）。"""
        free_run = self._free_run() if free_run is None else free_run
        maxrun = torch.where(allow[:, :, None], free_run, torch.zeros_like(free_run)).amax(-1).amax(-1)
        if self.opt.mask_mode != 'full':
            return maxrun.clamp(min=1, max=self.physics.max_slots_per_beam)
        free_total = (~self.occ).sum(-1).sum(-1)                      # 全部组的空闲槽，后续波束可用任一组
        reserve = (self.n - self.t - 1).clamp(min=0)
        return torch.minimum(maxrun, free_total - reserve).clamp(min=1, max=self.physics.max_slots_per_beam)

    def allowed_groups(self):
        B, G = self.B, self.G
        if self.opt.group_mode == 'free':
            return torch.ones(B, G, dtype=torch.bool, device=self.device)
        if self.opt.group_mode == 'mod8':
            allow = torch.zeros(B, G, dtype=torch.bool, device=self.device)
            allow[self.arange_b, self.t % G] = True
            return allow
        # greedy：取对当前束干扰最小且仍有空闲槽的组
        intf = self._interference(self.cur)                           # [B,2,T]
        score = intf.mean(-1)                                         # [B,2]
        free = (~self.occ).sum(-1)                                    # [B,G]
        parity = torch.arange(G, device=self.device) % 2
        total = score[:, parity] / self.noise + (free == 0).float() * 1e9 - free.float() * 1e-6
        allow = torch.zeros(B, G, dtype=torch.bool, device=self.device)
        allow[self.arange_b, total.argmin(1)] = True
        return allow

    def observe(self):
        """返回 dict：scal[B,10] intf[B,2,T] occ[B,G,T] allow[B,G] mask_gs[B,G*T] mask_cnt[B,10] mask_pw[B,K]
        可选 beam[B,N,F] beam_valid[B,N]。全部 device 张量。"""
        B, G, T = self.B, self.G, self.T
        cur = self.cur
        scene = self.scene
        intf = self._interference(cur)
        intf_db = 10 * torch.log10(intf / self.noise + 1e-12)
        intf_norm = ((intf_db + 20) / 40).clamp(0, 1)
        allow = self.allowed_groups()
        free_run = self._free_run()
        remaining = (self.n - self.t).clamp(min=1).float()
        reserve = self.reserve * (remaining - 1)
        budget_left = (self.budget - self.used_w - reserve).clamp(min=0)
        if self.opt.mask_mode == 'full':
            mask_gs = (allow[:, :, None] & (free_run >= 1)).reshape(B, G * T)
            if self.opt.start_mode != 'free':
                mask_gs = self._start_candidates(intf, allow, free_run, mask_gs)
            cmax = self.count_cap(allow, free_run)
            mask_cnt = (torch.arange(1, 11, device=self.device)[None, :] <= cmax[:, None])
            mask_pw = self.levels[None, :] <= budget_left[:, None] + 1e-6
            mask_pw[:, 0] |= ~mask_pw.any(1)       # 预算真不够时至少允许最低档（随后由预算检查处理）
            mask_cnt[:, 0] |= ~mask_cnt.any(1)
            mask_gs[self.arange_b, 0] |= ~mask_gs.any(1)
        else:
            mask_gs = torch.ones(B, G * T, dtype=torch.bool, device=self.device)
            mask_cnt = torch.ones(B, 10, dtype=torch.bool, device=self.device)
            mask_pw = torch.ones(B, self.K, dtype=torch.bool, device=self.device)
        demand = self.demand[self.arange_b, cur]
        single_cap = self.physics.max_slots_per_beam * self.physics.slot_Hz * self.physics.spectral_efficiency_cap
        scal = torch.stack([
            demand / 1e9, self.bank.width[scene, cur], self.bank.latlon[scene, cur, 0] / 90, self.bank.latlon[scene, cur, 1] / 180,
            self.bank.members[scene, cur] / 30, self.t.float() / self.n.clamp(min=1), (self.budget - self.used_w) / self.budget,
            remaining / self.N, self.n.float() / self.N, demand / single_cap], 1)
        # occ 必须复制：环境的占用图在 step() 里原地修改，引用会让入池的"动作前状态"变成动作后（第一轮的实现错误）
        obs = dict(scal=scal, intf=intf_norm, occ=self.occ.clone(), allow=allow, mask_gs=mask_gs, mask_cnt=mask_cnt, mask_pw=mask_pw)
        if self.opt.beam_tokens:
            obs['beam'], obs['beam_valid'] = self._beam_tokens(cur)
        if self.opt.legacy_obs:
            ordered = self.demand.gather(1, self.order)
            pos = self.arange_n[None, :]
            remaining_demand = (ordered * ((pos >= self.t[:, None]) & (pos < self.n[:, None]))).sum(1)
            mean_demand = self.demand.sum(1) / self.n.clamp(min=1)
            obs['glob'] = torch.stack([mean_demand / 1e9, self.t.float() / self.n.clamp(min=1), remaining_demand / (self.N * 1e9)], 1)
            obs['intf_abs'] = ((10 * torch.log10(intf + 1e-30) + 300) / 200).clamp(0, 1)   # 历史做法：绝对 dBW 映射
        # 第三轮：按槽数条件化掩码与连续功率需要的量
        obs['cmax'] = cmax if self.opt.mask_mode == 'full' else torch.full((B,), 10, dtype=torch.long, device=self.device)
        obs['pmax'] = torch.minimum(budget_left, torch.full_like(budget_left, self.cap_w))
        if self.opt.intf_rel:
            obs['intf_rel'] = intf / self.noise
        if self.opt.features:
            g = allow.float().argmax(1)
            obs.update(feature_mod.compute(self, cur, g, intf, budget_left, self.opt.features))
        return obs

    def _start_candidates(self, intf, allow, free_run, mask_gs):
        """起点候选：在允许组的空闲起点里，按"从该起点起 window 个槽（不超过空闲段）的平均干扰"打分，
        rule 只留最小的 1 个，topk 留最小的 k 个。"""
        B, G, T = self.B, self.G, self.T
        parity = torch.arange(G, device=self.device) % 2
        x = intf[:, parity]                                                      # [B,G,T]
        cs = torch.cat([torch.zeros(B, G, 1, device=self.device), x.cumsum(-1)], -1)
        w = free_run.clamp(min=1, max=self.opt.start_window)                     # [B,G,T]
        end = (self.arange_t[None, None, :] + w).clamp(max=T)
        cost = (cs.gather(2, end) - cs[:, :, :T]) / w.float()
        cost = torch.where(mask_gs.reshape(B, G, T), cost, torch.full_like(cost, float('inf'))).reshape(B, G * T)
        k = 1 if self.opt.start_mode == 'rule' else self.opt.start_topk
        k = min(k, G * T)
        idx = cost.topk(k, largest=False).indices
        keep = torch.zeros_like(mask_gs)
        keep.scatter_(1, idx, True)
        out = keep & mask_gs
        out[self.arange_b, 0] |= ~out.any(1)
        return out

    def _beam_tokens(self, cur):
        scene = self.scene
        B, N = self.B, self.N
        allocated = (self.grp >= 0).float()
        onehot = torch.nn.functional.one_hot(self.grp.clamp(min=0), self.G).float() * allocated[:, :, None]
        # 耦合系数不含路径损耗；折算成"每瓦每槽的干扰÷噪声"再取 dB（第一轮直接取 dB 导致全部饱和成 0/1）
        path = self.bank.path[scene].mean(-1)                         # [B,N] 各束中心的平均路径增益
        c_in = self.bank.coupling[scene, cur] * path[self.arange_b, cur][:, None] / self.noise      # cur ← j
        c_out = self.bank.coupling[scene][self.arange_b, :, cur] * path / self.noise                # j ← cur
        def db(x):
            return ((10 * torch.log10(x + 1e-30) + 40) / 60).clamp(0, 1)   # I/N（每瓦每槽）−40..+20 dB 映射
        feats = [self.demand / 1e9, self.bank.width[scene], self.bank.latlon[scene, :, 0] / 90, self.bank.latlon[scene, :, 1] / 180,
                 allocated, self.start.float() / self.T, self.cnt.float() / 10, self.pw / 50, db(c_in), db(c_out),
                 self.bank.adjacent[scene][self.arange_b, cur].float(), (self.arange_n[None, :] == cur[:, None]).float()]
        beam = torch.cat([torch.stack(feats, -1), onehot], -1)
        return beam, self.bank.valid[scene]

    # ------------------------------------------------------------ 一步
    def step(self, action, power_w=None):
        """action: LongTensor [B,3] = (组×起点 索引, 槽数-1, 功率档索引)；power_w[B]（瓦）给出时用连续功率代替档位。
        返回 (reward[B], done[B])。非活动槽位的动作被忽略。结束的场景指标追加到 self.finished 并自动装入下一景。"""
        B, G, T = self.B, self.G, self.T
        active = self.active
        cur = self.cur
        g = action[:, 0] // T
        s = action[:, 0] % T
        c = action[:, 1] + 1
        p = self.levels[action[:, 2]] if power_w is None else power_w.float()
        self.req[self.arange_b, cur] = torch.where(active, c, self.req[self.arange_b, cur])
        free_all = self._free_run()
        free_run = free_all[self.arange_b, g, s]
        c = torch.minimum(c, T - s)
        if self.opt.mask_mode == 'full':
            allow = torch.zeros(B, G, dtype=torch.bool, device=self.device)
            allow[self.arange_b, g] = True
            c = torch.minimum(torch.minimum(c, free_run), self.count_cap(allow, free_all))
            over = torch.zeros(B, dtype=torch.bool, device=self.device)
        else:
            collided = free_run < c
            c = torch.where(free_run >= c, c, torch.zeros_like(c))          # 与同组已占槽重叠 → 不服务
            over = self.used_w + p > self.budget + 1e-6
        p = torch.where(c > 0, p, torch.zeros_like(p))
        p = torch.minimum(p, (self.budget - self.used_w).clamp(min=0))
        c = torch.where(active & ~over, c, torch.zeros_like(c))
        p = torch.where(active & ~over, p, torch.zeros_like(p))
        if self.opt.reward_mode in ('delta', 'delta_intf'):
            cap_before = self.capacities()
            before = self._objective(cap_before)
            if self.opt.reward_mode == 'delta_intf':
                loss_before = self._intf_si_loss(cap_before)
        # 写入分配
        slot_mask = (self.arange_t[None, :] >= s[:, None]) & (self.arange_t[None, :] < (s + c)[:, None])   # [B,T]
        per_slot = torch.where(c > 0, p / c.clamp(min=1).float(), torch.zeros_like(p))
        parity = g % 2
        add = slot_mask.float() * per_slot[:, None]                                        # [B,T]
        self.A[self.arange_b, parity, cur] += add
        self.M[self.arange_b, parity] += self.bank.coupling[self.scene][self.arange_b, :, cur][:, :, None] * add[:, None, :]
        self.occ[self.arange_b, g] |= slot_mask & (c > 0)[:, None]
        self.grp[self.arange_b, cur] = torch.where(active, g, self.grp[self.arange_b, cur])
        self.start[self.arange_b, cur] = torch.where(active, s, self.start[self.arange_b, cur])
        self.cnt[self.arange_b, cur] = torch.where(active, c, self.cnt[self.arange_b, cur])
        self.pw[self.arange_b, cur] = torch.where(active, p, self.pw[self.arange_b, cur])
        self.used_w += p
        # 奖励
        capacity = self.capacities()
        if self.opt.reward_mode in ('delta', 'delta_intf'):
            after = self._objective(capacity)
            reward = after - before
            if self.opt.reward_mode == 'delta_intf':
                reward = reward - self.opt.intf_lambda * (self._intf_si_loss(capacity) - loss_before)
        else:
            cap_cur = capacity[self.arange_b, cur]
            dem_cur = self.demand[self.arange_b, cur]
            sgm_cur = self._sgm(capacity)[self.arange_b, cur]
            satisfied = torch.where(cap_cur >= dem_cur, torch.ones_like(cap_cur), -torch.ones_like(cap_cur))
            reward = satisfied + sgm_cur - ((cap_cur - dem_cur).abs() - 5e7).clamp(min=0) / 1e9
        if self.opt.mask_mode != 'full' and self.opt.illegal_penalty > 0:
            illegal = (collided | over) & active
            remaining_after = (self.n - self.t - 1).clamp(min=0).float()
            reward = reward - self.opt.illegal_penalty * illegal.float() - (over & active).float() * remaining_after
        reward = reward * active.float()
        self.terminated |= over & active
        self.t += active.long()
        done = active & (self.terminated | (self.t >= self.n))
        if done.any():
            self._finish(done, capacity)
        return reward, done

    # ------------------------------------------------------------ 终态指标
    def _finish(self, done, capacity):
        idx = torch.nonzero(done).flatten()
        cap_noint = self.capacities(interference=False)
        valid = self.bank.valid[self.scene]
        demand = self.demand
        si = (capacity / demand.clamp(min=1.)).clamp(max=1.) * valid
        sgm = self._sgm(capacity)
        served = torch.minimum(capacity, demand) * valid
        served_noint = torch.minimum(cap_noint, demand) * valid
        occ_beam = self.A.sum(1) > 0                                                   # [B,N,T]
        overlap = torch.matmul(occ_beam.float(), occ_beam.float().transpose(1, 2))      # [B,N,N] 共用槽数
        parity = self.grp % 2
        copol = (parity[:, :, None] == parity[:, None, :]) & (self.grp[:, :, None] != self.grp[:, None, :]) & (self.grp[:, :, None] >= 0) & (self.grp[:, None, :] >= 0)
        upper = torch.triu(torch.ones(self.N, self.N, dtype=torch.bool, device=self.device), 1)[None]
        adj = self.bank.adjacent[self.scene] & copol & upper
        expected = self.cnt[:, :, None].float() * self.cnt[:, None, :].float() / self.T
        intf = self._all_interference()                                                # [B,N,T] 占用槽上的 I
        objective = self._objective(capacity)
        for b in idx.tolist():
            n = int(self.n[b])
            key, v = self.episode_items[b]
            d = demand[b, :n]
            total_d = float(d.sum())
            cop = copol[b] & upper[0]
            metrics = dict(
                key=key, slot=b, per_aircraft_bps=float(v), beams=n, aircraft=float(self.bank.aircraft[self.scene[b]]),
                load_ratio=float(load_ratio(float(self.bank.members_total[self.scene[b]]), float(v), self.config)),
                W=float(served[b, :n].sum() / total_d), W_noint=float(served_noint[b, :n].sum() / total_d),
                SI=float(si[b, :n].mean()), SGM=float(sgm[b, :n].mean()),
                worst10=float(self._worst10(si[b, :n], self.bank.members[self.scene[b], :n])),
                power_w=float(self.pw[b, :n].sum()), slots=int(self.cnt[b, :n].sum()),
                copol_overlap=int((overlap[b] * cop.float()).sum()),
                unserved=int((self.cnt[b, :n] == 0).sum()), terminated=bool(self.terminated[b]),
                adj_overlap=float((overlap[b] * adj[b].float()).sum()), adj_expected=float((expected[b] * adj[b].float()).sum()),
                mean_in_db=float(self._mean_in_db(intf[b, :n], occ_beam[b, :n])),
                objective=float(objective[b]),
                requested_slots=int(self.req[b, :n].sum()), truncated_beams=int((self.req[b, :n] != self.cnt[b, :n]).sum()))
            metrics['interference_loss'] = 1 - metrics['W'] / max(metrics['W_noint'], 1e-9)
            metrics['adj_ratio'] = metrics['adj_overlap'] / metrics['adj_expected'] if metrics['adj_expected'] > 0 else float('nan')
            self.finished.append(metrics)
            if self.auto_reset:
                self._load(b)
            else:
                self.alive[b] = False

    def _all_interference(self):
        parity = (self.grp % 2).clamp(min=0)
        same = torch.gather(self.M, 1, parity[:, None, :, None].expand(-1, 1, -1, self.T))[:, 0]
        other = torch.gather(self.M, 1, (1 - parity)[:, None, :, None].expand(-1, 1, -1, self.T))[:, 0]
        return self.bank.path[self.scene] * (same + self.physics.xpd_linear * other)

    def _mean_in_db(self, intf, occ):
        if not occ.any():
            return float('nan')
        return float((10 * torch.log10(intf[occ] / self.noise + 1e-12)).mean())

    @staticmethod
    def _worst10(si, members):
        service = torch.repeat_interleave(si, members.long())
        k = max(1, int(math.ceil(.1 * len(service))))
        return torch.topk(service, k, largest=False).values.mean()

    def allocation(self, b):
        n = int(self.n[b])
        return (self.grp[b, :n].cpu().numpy(), self.start[b, :n].cpu().numpy(), self.cnt[b, :n].cpu().numpy(), self.pw[b, :n].cpu().numpy())
