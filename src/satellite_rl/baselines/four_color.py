from satellite_rl.paths import project_path
import os, math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from satellite_rl.envs.satellite import BeamInfo, SatelliteFreqState, CurrentBeamState, Env

import csv

# -------- 可调参数（与 greedy/fixed 风格一致） --------
EPISODES     = 1
MA_WINDOW    = 200
BANDS        = [(0, 25), (25, 50), (50, 75), (75, 100)]  # 4色→4个不重叠频段
# -----------------------------------------------------

def moving_average(x, w=200):
    x = np.asarray(x, float).reshape(-1)
    if len(x) < w:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="valid")

def _color_from_geo(lat, lon, lat_min=4.0, lat_max=54.0, lon_min=73.0, lon_max=135.0, grid=10):
    """把经纬度映射到棋盘格颜色：color ∈ {0,1,2,3}"""
    # 防越界
    lat = float(np.clip(lat, lat_min, lat_max - 1e-6))
    lon = float(np.clip(lon, lon_min, lon_max - 1e-6))
    # 网格索引
    i = int((lat - lat_min) / (lat_max - lat_min) * grid)
    j = int((lon - lon_min) / (lon_max - lon_min) * grid)
    i = max(0, min(grid - 1, i))
    j = max(0, min(grid - 1, j))
    return (i + j) % 4

def _choose_power_w(required_bps):
    """与 greedy 同风格的按需分档功率（单位：W）"""
    if required_bps >= 800e6:
        return 30
    elif required_bps >= 400e6:
        return 25
    else:
        return 20

def four_color_policy(env):
    # 1) 颜色 & 组（原样）
    lat, lon = env.beam_info['beam_location'][env.beam_idx]
    color = _color_from_geo(lat, lon)
    base_group = env.beam_idx % 8

    occ_rows = env.satellite_freq_state.freq_pool_real  # shape (8, 100)

    # 一个小工具：在指定 group 的指定 band 内找 “空位>0 的最空闲列”
    def pick_in_band(group, band_l, band_r):
        row = np.asarray(occ_rows[group], dtype=int)
        band = row[band_l:band_r]
        free = 10 - band                      # 每列剩余可用槽数（按 10 槽容量假设）
        if np.max(free) <= 0:
            return None, 0                    # 这段 band 没有空位
        local = int(np.argmax(free))          # 选空位最多的列（最小负载）
        freq_idx = band_l + local
        return freq_idx, int(free[local])

    # 颜色→频段（保持原来的 BANDS）
    band_l, band_r = BANDS[color]

    # 回退顺序：① 本组，同色 band → ② 同色另一极化组 → ③ 其它颜色的 band（避免饿死）
    candidates = []
    # ① base_group + 本色
    candidates.append((base_group, band_l, band_r))
    # ② 同色另一极化组
    alt_group = base_group ^ 1
    candidates.append((alt_group, band_l, band_r))
    # ③ 其它颜色（轮询 3 个）
    for dc in (1, 2, 3):
        oc = (color + dc) % 4
        l, r = BANDS[oc]
        candidates.append((base_group, l, r))
        candidates.append((alt_group, l, r))

    chosen = None
    free_slots_at_col = 0
    for g, l, r in candidates:
        f, free = pick_in_band(g, l, r)
        if f is not None and free > 0:
            chosen = (g, f)
            free_slots_at_col = free
            break

    # 兜底：极端情况下所有列都满了，仍然给一个（原本的）本色最空闲列，slots=1
    if chosen is None:
        row = np.asarray(occ_rows[base_group], dtype=int)
        band = row[band_l:band_r]
        local = int(np.argmin(band))
        freq = band_l + local
        group = base_group
        free_slots_at_col = 1
    else:
        group, freq = chosen

    # 需求驱动的“期望槽数”（仍然会被 free 裁剪）
    required = float(env.current_beam_state.beam_info['rate'][env.beam_idx])
    if required >= 800e6:
        want_slots = 10
        pw_w = 30
    elif required >= 400e6:
        want_slots = 6
        pw_w = 25
    else:
        want_slots = 3
        pw_w = 20

    slots = max(1, min(free_slots_at_col, want_slots))
    power_db = 10 * math.log10(pw_w)

    return [group, freq, slots, power_db]


def run_four_color(num_epochs=EPISODES, ma_window=MA_WINDOW):
    # 环境对象（与 fixed/greedy 一致）
    beam_info = BeamInfo()
    sat = SatelliteFreqState()
    cur = CurrentBeamState(beam_info.beam_info)
    env = Env(sat, beam_info, cur)
    # --- 初始化日志 ---
    log_dir = project_path("logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"four_color_log_{datetime.now().strftime('%Y%m%d%H%M')}.csv")
    log_fields = [
        'episode','beam_id','satisfaction','satisfaction_increment','total_satisfaction',
        'success_slots','spectral_efficiency_avg','power','is_interfere','out','occupy',
        'required_rate','location','group','freq','origin_slots','sgm_rate',
        'interfere_in_range','power_gap'
    ]
    with open(log_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()


    returns, satisfactions = [], []

    for ep in range(num_epochs):
        state = env.reset(BeamInfo())
        done, ep_ret, ep_satis = False, 0.0, 0.0

        while not done:
            action = four_color_policy(env)
            print(action)
            next_state, reward, done, info = env.step(action)

            # 写单步日志（仿 sac_full.py 风格）
            log_data = {
                'episode': ep,
                'beam_id': info['beam_id'],
                'satisfaction': float(f"{info['current_satisfaction']:.2f}"),
                'satisfaction_increment': float(f"{info['satisfaction_increment']:.2f}"),
                'total_satisfaction': float(f"{info['total_satisfaction']:.2f}"),
                'success_slots': info['success_slots'],
                "spectral_efficiency_avg": float(f"{info['spectral_efficiency_avg']:.2f}"),
                "power": float(f"{info['power']:.2f}"),
                "is_interfere": info['is_interfere'],
                "out": info['out'],
                "occupy": info['occupy'],
                "required_rate": info['required_rate'],
                'location': info['location'],
                'group': info['group'],
                'freq': info['freq'],
                'origin_slots': info['origin_slots'],
                'sgm_rate': float(f"{info['sgm_rate'].item():.2f}"),
                'interfere_in_range': info['interfere_in_range'],
                'power_gap': info['power_gap']
            }
            with open(log_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=log_fields)
                writer.writerow(log_data)


            ep_ret += float(reward)
            ep_satis = float(info['total_satisfaction'])
            state = next_state

        print(f"[FourColor] Episode {ep+1}: Total Return={ep_ret:.2f}, Satisfaction={ep_satis:.3f}")
        returns.append(ep_ret)
        satisfactions.append(ep_satis)

        if (ep + 1) % 200 == 0:
            print(f"[FourColor] Ep {ep+1}/{num_epochs} | Return={ep_ret:.2f} | Satis={ep_satis:.3f}")

    # 指标落盘
    os.makedirs(project_path("metrics"), exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d%H%M")
    out_npz = project_path(f"metrics/four_color_metrics_{stamp}.npz")
    np.savez(out_npz, returns=np.array(returns), satisfactions=np.array(satisfactions))
    print(f"[Saved] {out_npz}")

    # 画图
    os.makedirs(project_path("figs"), exist_ok=True)

    # Return 曲线
    ret_s = moving_average(returns, ma_window)
    x_ret = np.arange(1, len(ret_s) + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(x_ret, ret_s, label="Four-Color Reuse", linestyle="-.", marker="s",
             markevery=max(1, len(x_ret)//20))
    plt.xlabel("Episode")
    plt.ylabel(f"Episode Return (Moving Avg, window={ma_window})")
    plt.title("Episode Return Curve (Four-Color)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    ret_png = project_path(f"figs/four_color_return_{stamp}.png")
    plt.tight_layout(); plt.savefig(ret_png, dpi=300); print(f"[Saved] {ret_png}")

    # Satisfaction 曲线
    sat_s = moving_average(satisfactions, ma_window)
    x_sat = np.arange(1, len(sat_s) + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(x_sat, sat_s, label="Four-Color Reuse", linestyle="--", marker="o",
             markevery=max(1, len(x_sat)//20))
    plt.xlabel("Episode")
    plt.ylabel(f"Episode Satisfaction (Moving Avg, window={ma_window})")
    plt.title("Episode Satisfaction Curve (Four-Color)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    sat_png = project_path(f"figs/four_color_satisfaction_{stamp}.png")
    plt.tight_layout(); plt.savefig(sat_png, dpi=300); print(f"[Saved] {sat_png}")

if __name__ == "__main__":
    run_four_color()
