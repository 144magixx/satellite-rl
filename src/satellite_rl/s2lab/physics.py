"""物理v3：ITU-R S.672 旁瓣方向图、交叉极化泄漏、可配频谱效率封顶。

与链路v2的差别：
- 发射方向图由无旁瓣的球面高斯换成 ITU-R S.672-4 单馈源圆波束参考方向图
  （主瓣抛物线 → 近旁瓣底 LN → −25·log 衰减 → 0 dBi 地板）。
- 异极化（组号奇偶不同）不再完全隔离，按 XPD 泄漏。
- 5 dB“其它损耗”同时作用于有用信号和干扰（v2 只扣有用信号）。
- 频谱效率封顶默认 5.5 bit/s/Hz（v2 为 4.5）。
其余沿用 v2：峰值增益 50 dBi@0.5° 按立体角缩放、接收 40 dBi、噪声温度 631 K、
每槽 25 MHz、f0 17.7 GHz、每束功率按槽均摊、1 dB 实现损耗、逐飞机离轴增益的成员平均。
"""
from dataclasses import dataclass, asdict
import numpy as np

from satellite_rl.shared_interface.geometry import directions_and_distance

C_M_PER_S = 299792458.
K_BOLTZMANN = 1.380649e-23


@dataclass(frozen=True)
class PhysicsV3:
    version: str = 's2lab_physics_v3'
    sidelobe_level_dB: float = -20.      # S.672 LN，可取 -20/-25/-30
    xpd_dB: float = 20.                  # 交叉极化隔离度
    spectral_efficiency_cap: float = 5.5
    implementation_loss_dB: float = 1.
    other_loss_dB: float = 5.            # 作用于全部接收信号
    reference_full_width_deg: float = .5
    reference_peak_gain_dBi: float = 50.
    rx_gain_dBi: float = 40.
    noise_temperature_K: float = 631.
    slot_Hz: float = 25e6
    f0_Hz: float = 17.7e9
    slots: int = 100
    groups: int = 8
    max_slots_per_beam: int = 10
    adjacent_deg: float = 1.             # “相邻波束”判定角，用于躲干扰指标

    def to_dict(self):
        return asdict(self)

    @property
    def noise_w(self):
        return K_BOLTZMANN * self.noise_temperature_K * self.slot_Hz

    @property
    def xpd_linear(self):
        return 10 ** (-self.xpd_dB / 10)

    @property
    def impl_linear(self):
        return 10 ** (-self.implementation_loss_dB / 10)

    def slot_frequency_hz(self):
        return self.f0_Hz + (np.arange(self.slots) + .5) * self.slot_Hz


def peak_gain_dbi(width_deg, physics):
    """峰值增益随半功率全宽缩放：50 dBi@0.5°，窄一倍高 6 dB。"""
    width = np.asarray(width_deg, np.float64)
    return physics.reference_peak_gain_dBi + 20 * np.log10(physics.reference_full_width_deg / width)


def s672_gain_dbi(theta_deg, width_deg, physics):
    """ITU-R S.672-4 单馈源圆波束参考方向图（dBi）。theta 为离轴角，width 为半功率全宽。"""
    theta = np.asarray(theta_deg, np.float64)
    width = np.asarray(width_deg, np.float64)
    psi0 = width / 2.
    ln = physics.sidelobe_level_dB
    a = np.sqrt(-ln / 3.)
    b = 6.32
    gm = peak_gain_dbi(width, physics)
    ratio = theta / psi0
    main = gm - 3. * ratio ** 2
    floor = gm + ln
    with np.errstate(divide='ignore'):
        far = np.where(ratio > 0, gm + ln + 20. - 25. * np.log10(np.maximum(ratio, 1e-12)), gm)
    gain = np.where(ratio <= a, main, np.where(ratio <= b, floor, np.maximum(far, 0.)))
    return gain


def gain_linear(theta_deg, width_deg, physics):
    return 10 ** (s672_gain_dbi(theta_deg, width_deg, physics) / 10)


def angles_deg(rays_a, rays_b):
    """两组单位方向向量之间的夹角矩阵（度），rays_a[n,3] × rays_b[m,3] → [n,m]。"""
    cos = np.clip(rays_a @ rays_b.T, -1., 1.)
    return np.rad2deg(np.arccos(cos))


@dataclass
class SceneTensors:
    """一个场景的全部与分配无关的物理量（numpy，float32）。"""
    scene_id: str
    date: str
    members: np.ndarray        # [N] 每束飞机数
    width_deg: np.ndarray      # [N]
    lat_lon: np.ndarray        # [N,2]
    distance_m: np.ndarray     # [N] 卫星到束中心距离
    sig: np.ndarray            # [N] 有用信号系数（成员平均增益×接收增益×损耗），乘 path×P/c 得每槽接收功率
    coupling: np.ndarray       # [N,N] 接收 i ← 发射 j 的耦合系数（j 的方向图在 i 中心处×接收增益×损耗），不含路径
    adjacent: np.ndarray       # [N,N] bool 中心夹角 < adjacent_deg 且 i≠j
    aircraft_count: int

    @property
    def n(self):
        return int(len(self.members))


def build_scene_tensors(case, layout, physics, scene_id=None):
    """从 s2fix 格式的 case/layout 字典构造场景张量。"""
    centres = np.asarray(layout['centres_lat_lon_deg'], np.float64)
    widths = np.asarray(layout['full_width_deg'], np.float64)
    ids = list(layout['beam_ids'])
    n = len(ids)
    rays, distance = directions_and_distance(centres, validate=False)
    aircraft_rays, _ = directions_and_distance(np.asarray(case['lat_lon_deg'], np.float64), validate=False)
    by_id = {aid: i for i, aid in enumerate(case['aircraft_ids'])}
    members = [[] for _ in ids]
    index = {bid: i for i, bid in enumerate(ids)}
    for aid, bid in layout['owner_aircraft_to_beam_id']:
        members[index[bid]].append(by_id[aid])
    rx = 10 ** (physics.rx_gain_dBi / 10) * 10 ** (-physics.other_loss_dB / 10)
    sig = np.zeros(n, np.float64)
    for i, group in enumerate(members):
        theta = angles_deg(aircraft_rays[group], rays[i:i + 1])[:, 0]
        sig[i] = gain_linear(theta, widths[i], physics).mean() * rx
    theta_ij = angles_deg(rays, rays)                      # [i,j] i 中心相对 j 的离轴角（对称）
    coupling = gain_linear(theta_ij, widths[None, :], physics) * rx   # j 的方向图宽度
    np.fill_diagonal(coupling, 0.)
    adjacent = (theta_ij < physics.adjacent_deg)
    np.fill_diagonal(adjacent, False)
    counts = np.asarray([len(m) for m in members], np.int64)
    return SceneTensors(scene_id or case['case_id'], case['UTC'][:10], counts, widths.astype(np.float32),
                        centres.astype(np.float32), distance.astype(np.float32), sig.astype(np.float32),
                        coupling.astype(np.float32), adjacent, int(len(case['aircraft_ids'])))


def path_gain(distance_m, physics):
    """[N,100] 自由空间路径增益（线性，<1）。"""
    f = physics.slot_frequency_hz()
    return (C_M_PER_S / (4 * np.pi * np.asarray(distance_m, np.float64)[:, None] * f[None, :])) ** 2


def sgm_values(capacity_bps, demand_bps):
    """论文 SGM：每束一个 0—1 的值；容量 0 记 0。"""
    capacity = np.asarray(capacity_bps, np.float64)
    demand = np.asarray(demand_bps, np.float64)
    beta = float(demand.mean())
    result = np.zeros_like(capacity)
    positive = capacity > 0
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        c, d = capacity[positive], demand[positive]
        ratio = c / d
        real = np.where(ratio >= 1, ratio - 1, 1 - 1 / ratio)
        imag = (c - d) / beta
        result[positive] = 1 - (1 - np.exp(-np.hypot(real, imag))) ** 3
    return np.nan_to_num(result)


def reference_capacities(scene, groups, starts, counts, powers, physics, interference=True):
    """numpy 参考实现：给定完整分配算每束容量（bps）。用于测试和逐景核对。"""
    n = scene.n
    path = path_gain(scene.distance_m, physics)
    per_slot = np.zeros(n)
    active = counts > 0
    per_slot[active] = powers[active] / counts[active]
    occupancy = np.zeros((n, physics.slots))
    for j in range(n):
        if counts[j] > 0:
            occupancy[j, starts[j]:starts[j] + counts[j]] = per_slot[j]
    parity = np.asarray(groups) % 2
    capacity = np.zeros(n)
    for i in range(n):
        if counts[i] == 0:
            continue
        slots = np.arange(starts[i], starts[i] + counts[i])
        desired = scene.sig[i] * path[i, slots] * per_slot[i]
        if interference:
            same = (parity == parity[i]) & (np.asarray(groups) != groups[i])
            other = parity != parity[i]
            inter = (scene.coupling[i][same] @ occupancy[same][:, slots] +
                     physics.xpd_linear * (scene.coupling[i][other] @ occupancy[other][:, slots]))
            inter = path[i, slots] * inter
        else:
            inter = 0.
        sinr = desired / (physics.noise_w + inter) * physics.impl_linear
        eff = np.minimum(np.log2(1 + sinr), physics.spectral_efficiency_cap)
        capacity[i] = physics.slot_Hz * eff.sum()
    return capacity
