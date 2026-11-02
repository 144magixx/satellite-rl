"""原版环境上的场景、整套分配评价器，以及按原版物理移植的容量感知贪心与随机合法初解。

评价器不改原环境：把整套 (组, 起点, 槽数, 功率) 按原版处理顺序逐束交给 Env.step，
取回合结束时的逐束速率（含后来者干扰对先前波束的扣减）。
"""
import importlib
import os
from types import SimpleNamespace

import numpy as np

# 环境版本（环境变量 ORIG_ENV）：
#   original  45427b4 原版原样（默认）
#   fixed     c4963ba 修正版：功率按槽分摊、干扰记账修正、方向图在 0.25° 处连续（尾部仍用 log10 算宽度）
#   fixed_ln  fixed 再把方向图尾部宽度改用自然对数，即真正的 0.5° 半功率高斯
ENV_VARIANT = os.environ.get("ORIG_ENV", "original")
ENV = importlib.import_module("satellite_rl.original_compare.env_45427b4" if ENV_VARIANT == "original"
                              else "satellite_rl.original_compare.env_c4963ba")
if ENV_VARIANT == "fixed_ln":
    def _ln_tail(self, G_max, phi):
        beamwidth = 0.5
        sigma = beamwidth / (2 * np.sqrt(2 * np.log(2)))
        phi = abs(phi)
        if phi <= beamwidth / 2:
            return G_max - 3 * (phi / (beamwidth / 2)) ** 2
        edge = beamwidth / 2
        return G_max - 3 - 10 / np.log(10) * (phi ** 2 - edge ** 2) / (2 * sigma ** 2)
    ENV.AntennaPattern.transmitting_antenna_pattern = _ln_tail
POWER_SPLIT = ENV_VARIANT != "original"
# 场景几何（环境变量 ORIG_GEOMETRY）：cities 原版 100 城（默认）；jitter 每城加 1° 随机偏移；uniform 原版代码自带的中国范围内均匀随机
GEOMETRY = os.environ.get("ORIG_GEOMETRY", "cities")

AntennaPattern, BeamInfo, CurrentBeamState, Env = ENV.AntennaPattern, ENV.BeamInfo, ENV.CurrentBeamState, ENV.Env
SatelliteFreqState, R, beam_num, c = ENV.SatelliteFreqState, ENV.R, ENV.beam_num, ENV.c
geo_to_cartesian, BOLTZMANN, power_threshold, satellite_lon = ENV.geo_to_cartesian, ENV.k, ENV.power_threshold, ENV.satellite_lon

SLOT_HZ = 2.5e7
F0_HZ = 1.77e10 + 50 * SLOT_HZ          # 100 槽的中心频率，只用于贪心估算槽数
TX_GAIN_DBI, RX_GAIN_DBI, LOSS_DB = 50., 40., 5.
NOISE_K = 290.
# 原版超过 3000 W 立即结束回合；dB 往返有末位误差，预算留一点余量。
POWER_BUDGET = power_threshold * (1 - 1e-9)
PATTERN = AntennaPattern()


def make_scene(seed, geometry=None):
    """原版的场景生成：100 个城市（城市表正好 100 个，全部抽中）、随机需求，按需求从大到小排序。

    geometry 不是 cities 时，需求与顺序照旧，只把波束位置换掉（另用一个随机数流，不影响需求）。
    """
    geometry = geometry or GEOMETRY
    np.random.seed(seed)
    beams = BeamInfo()
    if geometry != "cities":
        rng = np.random.default_rng(seed + 7919)
        n = len(beams.beam_location)
        if geometry == "jitter":
            beams.beam_location = beams.beam_location + rng.normal(0, 1.0, (n, 2))
        else:
            beams.beam_location = np.column_stack((rng.uniform(ENV.LAT_MIN, ENV.LAT_MAX, n),
                                                   rng.uniform(ENV.LON_MIN, ENV.LON_MAX, n)))
        beams.cal_distance()
        beams.beam_info['beam_location'] = beams.beam_location
        beams.beam_info['distances'] = beams.distances
    return beams


def demand_of(beams):
    return beams.beam_info['rate'].ravel().astype(float)


def new_env(beams):
    env = Env(SatelliteFreqState(), beams, CurrentBeamState(beams.beam_info))
    state = env.reset(beams)
    return env, state


def final_rates(env):
    rate = np.zeros(beam_num)
    rate[:len(env.beam_bps)] = [float(np.ravel(x)[0]) for x in env.beam_bps]
    return rate


def metrics(env, beams):
    demand, rate = demand_of(beams), final_rates(env)
    si = np.minimum(rate / demand, 1.)
    sgm = np.array([float(np.ravel(env.SGM_rate(r, d)[0])[0]) for r, d in zip(rate, demand)])
    worst = np.sort(si)[:beam_num // 10]
    return dict(W=float(np.minimum(rate, demand).sum() / demand.sum()), SI=float(si.mean()),
                SGM=float(sgm.mean()), worst10=float(worst.mean()),
                power_w=float(env.total_power), beams_done=len(env.beam_bps),
                overload=bool(env.overload), slots=int(sum(h[3] for h in env.history_beam)))


def replay(beams, groups, starts, slots, power_w):
    """一次"真实评价"：整套分配在原版环境里从头走一遍。"""
    env, _ = new_env(beams)
    done, i = False, 0
    while not done:
        _, _, done, _ = env.step([int(groups[i]), int(starts[i]), int(slots[i]),
                                  10 * np.log10(power_w[i])])
        i += 1
    return env


def evaluate(beams, groups, starts, slots, power_w):
    env = replay(beams, groups, starts, slots, power_w)
    demand, rate = demand_of(beams), final_rates(env)
    si = np.minimum(rate / demand, 1.)
    sgm = np.array([float(np.ravel(env.SGM_rate(r, d)[0])[0]) for r, d in zip(rate, demand)])
    return SimpleNamespace(capacity_bps=rate, beam_si=si, si=float(si.mean()), sgm=float(sgm.mean()),
                           W=float(np.minimum(rate, demand).sum() / demand.sum()), env=env)


# ---------------- 原版物理下的链路量 ----------------

def beam_angles(locations):
    """卫星看两波束中心的夹角（度），与原版 calculate_beam_angle 相同。"""
    sat = np.deg2rad(satellite_lon)
    vectors = np.stack([geo_to_cartesian(lat, lon) for lat, lon in locations]) - [R * np.cos(sat), R * np.sin(sat), 0.]
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.rad2deg(np.arccos(np.clip(unit @ unit.T, -1., 1.)))


def coupling_gain(locations):
    """波束 j 的发射方向图在波束 i 中心处的线性增益；对角为主瓣增益。"""
    angles = beam_angles(locations)
    gain = np.vectorize(lambda phi: 10 ** (PATTERN.transmitting_antenna_pattern(TX_GAIN_DBI, phi) / 10))(angles)
    return gain


def slot_capacity(beams, power_w):
    """单槽容量。原版每个槽都按整束功率计算（功率不按槽分摊）；修正版传入的是每槽功率。"""
    distance = beams.beam_info['distances'].ravel() * 1e3
    received = power_w * 10 ** ((TX_GAIN_DBI + RX_GAIN_DBI) / 10) / (4 * np.pi * distance * F0_HZ / c) ** 2
    cinr = received / (BOLTZMANN * NOISE_K * SLOT_HZ) * 10 ** (-LOSS_DB / 10)
    return SLOT_HZ * np.log2(1 + cinr)


def estimated_slots(beams):
    demand = demand_of(beams)
    estimated_power = np.minimum(power_threshold * demand / demand.sum(), 50.)
    if not POWER_SPLIT:
        return np.clip(np.ceil(demand / slot_capacity(beams, estimated_power)).astype(int), 1, 10)
    slots = np.full(len(demand), 10, int)
    for k in range(10, 0, -1):     # 与 S2-FIX 相同：取能满足需求的最少槽数，都不够则 10
        enough = k * slot_capacity(beams, estimated_power / k) >= demand
        slots[enough] = k
    return slots


def proportional_power(demand, budget=POWER_BUDGET, cap=50.):
    demand = np.asarray(demand, float)
    power = np.zeros(len(demand))
    free = np.ones(len(demand), bool)
    remaining = min(budget, cap * len(demand))
    while free.any():
        quota = remaining * demand[free] / demand[free].sum()
        indices = np.flatnonzero(free)
        hit = quota >= cap
        if not hit.any():
            power[indices] = quota
            break
        power[indices[hit]] = cap
        free[indices[hit]] = False
        remaining -= cap * hit.sum()
    power *= min(1., budget / np.nextafter(power.sum(), np.inf))
    return power


def shrink_to_group(slots, members, demand):
    """某组超过 100 槽时按需求比例缩减到合法（与 S2-FIX 相同）。"""
    if slots[members].sum() > 100:
        quota = (100 - len(members)) * demand[members] / demand[members].sum()
        count = np.minimum(slots[members], 1 + np.floor(quota).astype(int))
        while count.sum() < 100 and np.any(count < slots[members]):
            score = np.where(count < slots[members], quota - (count - 1), -np.inf)
            count[np.argmax(score)] += 1
        slots[members] = count


def greedy_allocation(beams, fill_groups=False, fixed_groups=False, free_starts=False):
    """容量感知贪心 G1，逐行对应 S2-FIX greedy.py，链路量换成原版物理。

    三个开关只用于诊断，正式对比不用：
    fill_groups  分组后把每组剩余的槽按需求/槽数比逐个补给组内波束（每束≤10），
                 检验"原版功率不按槽分摊、多给槽几乎不花代价"的影响；
    fixed_groups 组号改为原版 SAC 的固定规则（处理顺序 %8），8 组都用上；
    free_starts  起点可放在组内任意空位（原规则是组内首尾相接排）。
    """
    demand = demand_of(beams)
    n = len(demand)
    gain = coupling_gain(beams.beam_info['beam_location'])
    power = proportional_power(demand)
    slots = estimated_slots(beams)
    order = np.argsort(-demand, kind='stable')
    groups = np.full(n, -1, int)
    used = np.zeros(8, int)
    penalty = gain.max() * n
    for i in order:
        costs = [gain[i, (groups >= 0) & (groups % 2 == g % 2) & (groups != g)].sum() +
                 penalty * max(0, used[g] + slots[i] - 100) for g in range(8)]
        groups[i] = g = int(np.argmin(costs))
        used[g] += slots[i]
    if fixed_groups:
        groups = np.arange(n) % 8
    for g in range(8):
        shrink_to_group(slots, np.flatnonzero(groups == g), demand)
    if fill_groups:
        for g in range(8):
            members = np.flatnonzero(groups == g)
            while len(members) and slots[members].sum() < 100 and np.any(slots[members] < 10):
                score = np.where(slots[members] < 10, demand[members] / slots[members], -np.inf)
                slots[members[np.argmax(score)]] += 1
    starts = np.zeros(n, int)
    cursor = np.zeros(8, int)
    remaining = np.bincount(groups, weights=slots, minlength=8).astype(int)
    assigned = np.zeros(n, bool)
    occupied = np.zeros((8, 100), bool)
    for i in order:
        g = groups[i]
        other = np.flatnonzero(assigned & (groups % 2 == g % 2) & (groups != g))
        if free_starts:   # 诊断：组内任意空位都可选，不必首尾相接
            choices = np.array([s for s in range(101 - slots[i]) if not occupied[g, s:s + slots[i]].any()])
            if len(choices) == 0:   # 空位被切碎、放不下整块时缩到最长空段（组较满时偶尔发生）
                free = np.flatnonzero(~occupied[g])
                longest = max(np.split(free, np.flatnonzero(np.diff(free) > 1) + 1), key=len)
                slots[i] = len(longest)
                choices = np.array([longest[0]])
        else:
            choices = np.arange(cursor[g], 101 - remaining[g])
        overlap = np.maximum(0, np.minimum(choices[:, None] + slots[i], starts[other] + slots[other]) -
                             np.maximum(choices[:, None], starts[other]))
        starts[i] = choices[np.argmin(overlap @ gain[i, other])]
        occupied[g, starts[i]:starts[i] + slots[i]] = True
        cursor[g] = starts[i] + slots[i]
        remaining[g] -= slots[i]
        assigned[i] = True
    return groups, starts, slots, power


def random_allocation(beams, seed):
    """纯禁忌的随机合法初解：随机顺序循环分 8 组，组内按该顺序连续排槽（与 S2-FIX 相同）。"""
    demand = demand_of(beams)
    slots = estimated_slots(beams)
    order = np.random.default_rng(seed).permutation(len(demand))
    groups, starts = np.empty(len(demand), int), np.zeros(len(demand), int)
    groups[order] = np.arange(len(order)) % 8
    for group in range(8):
        members = order[groups[order] == group]
        shrink_to_group(slots, members, demand)
        starts[members] = np.cumsum(slots[members]) - slots[members]
    return groups, starts, slots, proportional_power(demand)
