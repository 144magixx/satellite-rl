"""鐗╃悊v3锛欼TU-R S.672 鏃佺摚鏂瑰悜鍥俱€佷氦鍙夋瀬鍖栨硠婕忋€佸彲閰嶉璋辨晥鐜囧皝椤躲€?
涓庨摼璺痸2鐨勫樊鍒細
- 鍙戝皠鏂瑰悜鍥剧敱鏃犳梺鐡ｇ殑鐞冮潰楂樻柉鎹㈡垚 ITU-R S.672-4 鍗曢婧愬渾娉㈡潫鍙傝€冩柟鍚戝浘
  锛堜富鐡ｆ姏鐗╃嚎 鈫?杩戞梺鐡ｅ簳 LN 鈫?鈭?5路log 琛板噺 鈫?0 dBi 鍦版澘锛夈€?- 寮傛瀬鍖栵紙缁勫彿濂囧伓涓嶅悓锛変笉鍐嶅畬鍏ㄩ殧绂伙紝鎸?XPD 娉勬紡銆?- 5 dB鈥滃叾瀹冩崯鑰椻€濆悓鏃朵綔鐢ㄤ簬鏈夌敤淇″彿鍜屽共鎵帮紙v2 鍙墸鏈夌敤淇″彿锛夈€?- 棰戣氨鏁堢巼灏侀《榛樿 5.5 bit/s/Hz锛坴2 涓?4.5锛夈€?鍏朵綑娌跨敤 v2锛氬嘲鍊煎鐩?50 dBi@0.5掳 鎸夌珛浣撹缂╂斁銆佹帴鏀?40 dBi銆佸櫔澹版俯搴?631 K銆?姣忔Ы 25 MHz銆乫0 17.7 GHz銆佹瘡鏉熷姛鐜囨寜妲藉潎鎽娿€? dB 瀹炵幇鎹熻€椼€侀€愰鏈虹杞村鐩婄殑鎴愬憳骞冲潎銆?"""
from dataclasses import dataclass, asdict
import numpy as np

from satellite_rl.shared_interface.geometry import directions_and_distance

C_M_PER_S = 299792458.
K_BOLTZMANN = 1.380649e-23


@dataclass(frozen=True)
class PhysicsV3:
    version: str = 's2lab_physics_v3'
    sidelobe_level_dB: float = -20.      # S.672 LN锛屽彲鍙?-20/-25/-30
    xpd_dB: float = 20.                  # 浜ゅ弶鏋佸寲闅旂搴?    spectral_efficiency_cap: float = 5.5
    implementation_loss_dB: float = 1.
    other_loss_dB: float = 5.            # 浣滅敤浜庡叏閮ㄦ帴鏀朵俊鍙?    reference_full_width_deg: float = .5
    reference_peak_gain_dBi: float = 50.
    rx_gain_dBi: float = 40.
    noise_temperature_K: float = 631.
    slot_Hz: float = 25e6
    f0_Hz: float = 17.7e9
    slots: int = 100
    groups: int = 8
    max_slots_per_beam: int = 10
    adjacent_deg: float = 1.             # 鈥滅浉閭绘尝鏉熲€濆垽瀹氳锛岀敤浜庤翰骞叉壈鎸囨爣

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
    """宄板€煎鐩婇殢鍗婂姛鐜囧叏瀹界缉鏀撅細50 dBi@0.5掳锛岀獎涓€鍊嶉珮 6 dB銆?""
    width = np.asarray(width_deg, np.float64)
    return physics.reference_peak_gain_dBi + 20 * np.log10(physics.reference_full_width_deg / width)


def s672_gain_dbi(theta_deg, width_deg, physics):
    """ITU-R S.672-4 鍗曢婧愬渾娉㈡潫鍙傝€冩柟鍚戝浘锛坉Bi锛夈€倀heta 涓虹杞磋锛寃idth 涓哄崐鍔熺巼鍏ㄥ銆?""
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
    """涓ょ粍鍗曚綅鏂瑰悜鍚戦噺涔嬮棿鐨勫す瑙掔煩闃碉紙搴︼級锛宺ays_a[n,3] 脳 rays_b[m,3] 鈫?[n,m]銆?""
    cos = np.clip(rays_a @ rays_b.T, -1., 1.)
    return np.rad2deg(np.arccos(cos))


@dataclass
class SceneTensors:
    """涓€涓満鏅殑鍏ㄩ儴涓庡垎閰嶆棤鍏崇殑鐗╃悊閲忥紙numpy锛宖loat32锛夈€?""
    scene_id: str
    date: str
    members: np.ndarray        # [N] 姣忔潫椋炴満鏁?    width_deg: np.ndarray      # [N]
    lat_lon: np.ndarray        # [N,2]
    distance_m: np.ndarray     # [N] 鍗槦鍒版潫涓績璺濈
    sig: np.ndarray            # [N] 鏈夌敤淇″彿绯绘暟锛堟垚鍛樺钩鍧囧鐩娒楁帴鏀跺鐩娒楁崯鑰楋級锛屼箻 path脳P/c 寰楁瘡妲芥帴鏀跺姛鐜?    coupling: np.ndarray       # [N,N] 鎺ユ敹 i 鈫?鍙戝皠 j 鐨勮€﹀悎绯绘暟锛坖 鐨勬柟鍚戝浘鍦?i 涓績澶劽楁帴鏀跺鐩娒楁崯鑰楋級锛屼笉鍚矾寰?    adjacent: np.ndarray       # [N,N] bool 涓績澶硅 < adjacent_deg 涓?i鈮爅
    aircraft_count: int

    @property
    def n(self):
        return int(len(self.members))


def build_scene_tensors(case, layout, physics, scene_id=None):
    """浠?s2fix 鏍煎紡鐨?case/layout 瀛楀吀鏋勯€犲満鏅紶閲忋€?""
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
    theta_ij = angles_deg(rays, rays)                      # [i,j] i 涓績鐩稿 j 鐨勭杞磋锛堝绉帮級
    coupling = gain_linear(theta_ij, widths[None, :], physics) * rx   # j 鐨勬柟鍚戝浘瀹藉害
    np.fill_diagonal(coupling, 0.)
    adjacent = (theta_ij < physics.adjacent_deg)
    np.fill_diagonal(adjacent, False)
    counts = np.asarray([len(m) for m in members], np.int64)
    return SceneTensors(scene_id or case['case_id'], case['UTC'][:10], counts, widths.astype(np.float32),
                        centres.astype(np.float32), distance.astype(np.float32), sig.astype(np.float32),
                        coupling.astype(np.float32), adjacent, int(len(case['aircraft_ids'])))


def path_gain(distance_m, physics):
    """[N,100] 鑷敱绌洪棿璺緞澧炵泭锛堢嚎鎬э紝<1锛夈€?""
    f = physics.slot_frequency_hz()
    return (C_M_PER_S / (4 * np.pi * np.asarray(distance_m, np.float64)[:, None] * f[None, :])) ** 2


def sgm_values(capacity_bps, demand_bps):
    """璁烘枃 SGM锛氭瘡鏉熶竴涓?0鈥? 鐨勫€硷紱瀹归噺 0 璁?0銆?""
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
    """numpy 鍙傝€冨疄鐜帮細缁欏畾瀹屾暣鍒嗛厤绠楁瘡鏉熷閲忥紙bps锛夈€傜敤浜庢祴璇曞拰閫愭櫙鏍稿銆?""
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
