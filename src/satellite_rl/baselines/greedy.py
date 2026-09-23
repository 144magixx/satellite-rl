from satellite_rl.paths import project_path
# file: run_greedy_4000.py
import os, math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from satellite_rl.envs.satellite import BeamInfo, SatelliteFreqState, CurrentBeamState, Env



def moving_average(x, w=200):
    x = np.asarray(x, float).reshape(-1)
    if len(x) < w: return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="valid")

def greedy_policy(env):
    """
    Greedy++:
      1) 在当前 group 内选择占用最少的起始频点；
      2) 时隙=该频点的可用空位数（至少1）；
      3) 功率用中档（减少干扰惩罚）。
    """
    group = env.beam_idx % 8

    # 频谱占用矩阵：shape(10,10)，行=slots，列=freq 起点（你环境里是这样组织的）
    occ = np.asarray(env.satellite_freq_state.freq_pool_real, dtype=int)  # 0/1
    # 统计每个频点列的总占用量，越小越空闲
    col_load = occ.sum(axis=0)  # (10,)
    #print(env.satellite_freq_state.freq_pool_real)
    freq = int(np.argmin(col_load))  # 选当前最空闲的频点

    # 该频点这一列还有多少空位（10-已占用）
    free_slots = int(10 - occ[:, freq].sum())
    slots = max(1, min(5, free_slots))  # 至少1个，最多10个


    # 中等功率（10~20W），比50W更少冲突惩罚；可按需求粗调
    required = float(env.current_beam_state.beam_info['rate'][env.beam_idx])
    # 简单分档：需求高给 20W，需求中等给 15W，需求低给 10W
    if required >= 800e6:
        pw_w = 30
    elif required >= 400e6:
        pw_w = 25
    else:
        pw_w = 20

    power_db = 10 * math.log10(pw_w)

    return [group, freq, slots, power_db]

def run_greedy(num_epochs=1, ma_window=200):
    # 环境
    beam_info = BeamInfo()
    sat = SatelliteFreqState()
    cur = CurrentBeamState(beam_info.beam_info)
    env = Env(sat, beam_info, cur)

    returns, satisfactions = [], []

    for ep in range(num_epochs):
        state = env.reset(BeamInfo())
        done, ep_ret, ep_satis = False, 0.0, 0.0
        while not done:
            action = greedy_policy(env)
            next_state, reward, done, info = env.step(action)


            ep_ret += float(reward)
            ep_satis = float(info['total_satisfaction'])
            state = next_state
        print(f"[Greedy] Episode {ep + 1}: Total Return={ep_ret:.2f}, Satisfaction={ep_satis:.2f}")
        returns.append(ep_ret)
        satisfactions.append(ep_satis)
        if (ep + 1) % 200 == 0:
            print(f"[Greedy] Ep {ep+1}/{num_epochs} | Return={ep_ret:.2f} | Satis={ep_satis:.2f}")

    # 保存指标
    os.makedirs(project_path("metrics"), exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d%H%M")
    out_npz = project_path(f"metrics/greedy_metrics_{stamp}.npz")
    np.savez(out_npz, returns=np.array(returns), satisfactions=np.array(satisfactions))
    print(f"[Saved] {out_npz}")

    # 画图
    os.makedirs(project_path("figs"), exist_ok=True)
    # Return 曲线
    ret_s = moving_average(returns, ma_window)
    x_ret = np.arange(1, len(ret_s) + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(x_ret, ret_s, label="Greedy", linestyle="--", marker="o", markevery=max(1, len(x_ret)//20))
    plt.xlabel("Episode")
    plt.ylabel(f"Episode Return (Moving Avg, window={ma_window})")
    plt.title("Episode Return Curve (Greedy)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    ret_png = project_path(f"figs/greedy_return_{stamp}.png")
    plt.tight_layout(); plt.savefig(ret_png, dpi=300); print(f"[Saved] {ret_png}")

    # Satisfaction 曲线
    sat_s = moving_average(satisfactions, ma_window)
    x_sat = np.arange(1, len(sat_s) + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(x_sat, sat_s, label="Greedy", linestyle="-.", marker="s", markevery=max(1, len(x_sat)//20))
    plt.xlabel("Episode")
    plt.ylabel(f"Episode Satisfaction (Moving Avg, window={ma_window})")
    plt.title("Episode Satisfaction Curve (Greedy)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    sat_png = project_path(f"figs/greedy_satisfaction_{stamp}.png")
    plt.tight_layout(); plt.savefig(sat_png, dpi=300); print(f"[Saved] {sat_png}")

if __name__ == "__main__":
    run_greedy(num_epochs=10, ma_window=200)
