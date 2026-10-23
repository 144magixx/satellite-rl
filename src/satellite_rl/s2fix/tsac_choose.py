"""X4：直出T-SAC 自己选组。

每束一次决定"组×起点、槽数、功率"，组×起点共 8×100=800 项（编号 = 组×100 + 起点）。
其余与 L2（P01-M1）相同：奖励、功率菜单、SAC 设定、Transformer 宽度与层数。

状态：L2 的三条逐槽通道（本束在各槽收到的干扰、占用、本束会伤到已放波束的余量摘要）
按 8 个组各算一份（8×100×3），地理、需求、功率余量、全局量、束宽不变，共 2408 维。
保留槽数：剩下的每束至少要能在某个组里找到 1 个空槽。
网络：频率槽令牌由 L2 的 10 个扩成 8 组×10 段（加组位置编码），共 84 个令牌；
800 个组×起点打分由各频率槽令牌读出（每个令牌给自己那 10 个起点打分），槽数与功率仍由第一个令牌读出。
"""
from copy import deepcopy
from time import perf_counter

import numpy as np
import torch
from torch import nn

from satellite_rl.agents.action_masking import entropy_ratio
from satellite_rl.agents.workbench_algorithms import WorkbenchAgent
from satellite_rl.power30.agent import Power30Agent
from satellite_rl.power30.environment import Power30Env
from satellite_rl.power30.menus import legal_power_mask

GROUPS = 8
STATE_WIDTH = 4 + 3 * GROUPS * 100 + 4
INTERFERENCE, OCCUPANCY, CONTEXT, WIDTH, MARGIN = 4, 4 + GROUPS * 100, 4 + 2 * GROUPS * 100, 7 + 2 * GROUPS * 100, 8 + 2 * GROUPS * 100
# 只有同极化（组号奇偶相同）、不同组的波束之间有干扰：ELIGIBLE[g, h]
ELIGIBLE = np.array([[float(h % 2 == g % 2 and h != g) for h in range(GROUPS)] for g in range(GROUPS)])


class GroupChoiceEnv(Power30Env):
    def __init__(self, stage_input):
        self.rx = 10 ** (stage_input.physics['rx_gain_dBi'] / 10)
        super().__init__(stage_input, 'P01', 'M1')

    def reset(self):
        # 未处理波束的组号只是占位（它们没有槽，不参与任何干扰计算），处理时改成策略选的组
        order = self.stage_input.processing_index_to_beam_id
        self.stage_input.group_by_beam_id = {bid: k % GROUPS for k, bid in enumerate(order)}
        self.groups = np.asarray([self.stage_input.group_by_beam_id[b] for b in self.ids], np.int64)
        return super().reset()

    def action_mask(self):
        starts = np.zeros((GROUPS, 100, 10), bool)
        power = np.zeros(len(self.power_table), bool)
        if self.beam_idx == self.N:
            return dict(frequency_slots=starts.reshape(GROUPS * 100, 10), power=power)
        free = ~self.occupied
        longest = min(10, int(free.sum()) - (self.N - self.beam_idx - 1))
        window = free
        for count in range(1, longest + 1):
            if count > 1:
                window = window[:, :-1] & free[:, count - 1:]
            starts[:, :101 - count, count - 1] = window
        power[:] = legal_power_mask(self.menu, float(self.powers.sum()), self.N - self.beam_idx - 1)
        return dict(frequency_slots=starts.reshape(GROUPS * 100, 10), power=power)

    def step(self, action):
        step_started = perf_counter()
        choice, l, p = map(int, action)
        mask = self.action_mask()
        if not (mask['frequency_slots'][choice, l] and mask['power'][p]):
            raise ValueError('illegal action')
        group, f = divmod(choice, 100)
        c = int(self.indices[self.beam_idx])
        self.groups[c] = group
        self.stage_input.group_by_beam_id[self.ids[c]] = group
        return self._place(f, l, p, action, step_started)

    def _observation(self):
        if self.beam_idx == self.N:
            return np.zeros(STATE_WIDTH, np.float32)
        c = self.indices[self.beam_idx]
        lat, lon = self.layout['centres_lat_lon_deg'][c]
        received = np.zeros((GROUPS, 100))
        for j in np.flatnonzero(self.counts):
            received[self.groups[j], self.starts[j]:self.starts[j] + self.counts[j]] += (
                self.kernel['gain'][c, j] * self.powers[j] / self.counts[j])
        incoming = (ELIGIBLE @ received) * self.rx * self.kernel['path'][c]
        normalized = np.clip((10 * np.log10(incoming + 1e-30) + 300) / 200, 0, 1)
        remaining = self.indices[self.beam_idx:]
        return np.concatenate([
            [2 * (lat - 52.5) / 55, 2 * (lon - 5) / 90, self.demands[c] / 550e6, (3000 - self.powers.sum()) / 3000],
            normalized.ravel(), self.occupied.ravel().astype(float),
            [self.beta / 6e8, self.beam_idx / self.N, self.demands[remaining].sum() / (self.N * 6e8)],
            [self.layout['full_width_deg'][c]], self._margin_summary(c).ravel()]).astype(np.float32)

    def _margin_summary(self, current):
        """L2 余量摘要按候选组各算一份：本束若放进组 g，会在各槽伤到哪些已放波束（按其满足程度加权）。"""
        result = np.zeros((GROUPS, 100))
        coeff = self.last_coefficients
        if coeff is None:
            return result
        owner, slots = coeff.slot_owner, coeff.occupied_slots
        denominator = coeff.noise + coeff.interference @ self.powers
        weight = np.exp(-np.abs(self.rates / self.demands - 1.0))
        harm = (weight[owner] * self.kernel['gain'][owner, current] * self.rx
                * self.kernel['path'][owner, slots] / denominator)
        np.add.at(result, (self.groups[owner], slots), harm)
        return np.log1p(ELIGIBLE @ result)


class GroupChoiceNetwork(nn.Module):
    """L2 的令牌编码器（同一组权重初始化），频率槽令牌扩成 8 组×10 段，组×起点打分逐令牌读出。"""

    def __init__(self, base, seed):
        super().__init__()
        encoder = base.encoder
        for name in ('geo_embed', 'rate_embed', 'power_embed', 'slot_embed', 'net', 'global_embed'):
            setattr(self, name, getattr(encoder, name))
        width = encoder.pos_embed.shape[-1]
        self.scalar_pos = nn.Parameter(encoder.pos_embed.detach()[:, :3].clone())
        self.slot_pos = nn.Parameter(encoder.pos_embed.detach()[:, 3:].clone().unsqueeze(1))
        self.group_pos = nn.Parameter(torch.zeros(1, GROUPS, 1, width))
        self.global_pos = nn.Parameter(encoder.extra_pos_embed.detach().clone())
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 80021)
            self.start_head = nn.Linear(width, 10)
        self.count_head, self.power_head = base.heads[1], base.heads[2]

    def forward(self, states):
        batch = states.shape[0]
        geo = torch.cat((states[:, :2], states[:, WIDTH:WIDTH + 1]), -1)
        channels = [states[:, a:a + GROUPS * 100].reshape(batch, GROUPS, 10, 10)
                    for a in (INTERFERENCE, OCCUPANCY, MARGIN)]
        slots = self.slot_embed(torch.cat(channels, -1)) + self.slot_pos + self.group_pos
        scalars = torch.stack((self.geo_embed(geo), self.rate_embed(states[:, 2:3]),
                               self.power_embed(states[:, 3:4])), 1) + self.scalar_pos
        context = self.global_embed(states[:, CONTEXT:CONTEXT + 3]).unsqueeze(1) + self.global_pos
        sequence = torch.cat((scalars, slots.reshape(batch, GROUPS * 10, -1), context), 1).permute(1, 0, 2)
        output = self.net(sequence).permute(1, 0, 2)
        first = output[:, 0]
        return (self.start_head(output[:, 3:3 + GROUPS * 10]).reshape(batch, GROUPS * 100),
                self.count_head(first), self.power_head(first))


class GroupChoiceAgent(Power30Agent):
    """SAC 算法与 L2 相同（分解式掩码策略、加性双Q、固定温度），只把"起点"动作扩成"组×起点"。"""

    def __init__(self, config, device=None, call_hook=None):
        super().__init__(config, device=device, call_hook=call_hook)
        self.action_dims = (GROUPS * 100, 10, len(self.power_table))
        self.target_entropy_ratio = entropy_ratio(self.train['target_entropy'], self.action_dims)
        for index, name in enumerate(('actor', 'critic1', 'critic2')):
            network = GroupChoiceNetwork(getattr(self, name), self.train['seed'] + 100003 * (index + 1))
            setattr(self, name, network.to(self.device))
        self.target1, self.target2 = deepcopy(self.critic1).eval(), deepcopy(self.critic2).eval()
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.train['actor_lr'])
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=self.train['critic_lr'])
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=self.train['critic_lr'])
        self.schema.update(version='s2fix_tsac_choose_v1', dimension=STATE_WIDTH, action_dims=list(self.action_dims))

    def observe(self, state, action, reward, next_state, done, action_mask=None, next_action_mask=None, **kwargs):
        return WorkbenchAgent.observe(self, state, action, reward, next_state, done,
                                      action_mask, next_action_mask, **kwargs)
