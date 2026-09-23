from satellite_rl.paths import project_path
# file: run_fixed_4000.py
import os, math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from satellite_rl.envs.satellite import BeamInfo, SatelliteFreqState, CurrentBeamState, Env

# -------- 可调固定方案参数（按需改） --------
FIXED_FREQ   = 50      # 固定起始频点 0~99
FIXED_SLOTS  = 8       # 固定时隙数 1~10
FIXED_POWERW = 30      # 固定功率（单位：W），常用 10/15/20
MA_WINDOW    = 200     # 曲线滑动平均窗口
EPISODES     = 1    # 训练（评估）回合数
# ---------------------------------------

def moving_average(x, w=200):
    x = np.asarray(x, float).reshape(-1)
    if len(x) < w: return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="valid")

# 顶部若尚未导入，请确保：
import numpy as np
import math

def fixed_policy(env):
    """
    Fixed（顺序分配版）：
    - 固定 slots 和 power；
    - 起始频点 freq 按从左到右顺序扫描，选第一个未满的列。
    """
    group = env.beam_idx % 8

    # 占用矩阵：行=slot(10)，列=freq起点(100)；值为0/1
    occ = np.asarray(env.satellite_freq_state.freq_pool_real, dtype=int)  # shape (10, 100)
    num_freq = occ.shape[1]

    # 从左到右扫描，找到第一个未占满的频率列
    freq = env.beam_idx // 8 * FIXED_SLOTS


    # 固定时隙与功率
    slots = FIXED_SLOTS                      # 例如 5
    power_db = 10 * math.log10(FIXED_POWERW) # 例如 10/15/20 W

    return [group, freq, slots, power_db]



def run_fixed(num_epochs=EPISODES, ma_window=MA_WINDOW):
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
            action = fixed_policy(env)
            print(action)
            next_state, reward, done, info = env.step(action)
            ep_ret += float(reward)
            ep_satis = float(info['total_satisfaction'])
            state = next_state

        # 控制台打印每回合结果
        print(f"[Fixed] Episode {ep+1}: Total Return={ep_ret:.2f}, Satisfaction={ep_satis:.2f}")

        returns.append(ep_ret)
        satisfactions.append(ep_satis)

    # 保存指标
    os.makedirs(project_path("metrics"), exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d%H%M")
    out_npz = project_path(f"metrics/fixed_metrics_{stamp}.npz")
    np.savez(out_npz, returns=np.array(returns), satisfactions=np.array(satisfactions))
    print(f"[Saved] {out_npz}")

    # 画图并保存
    os.makedirs(project_path("figs"), exist_ok=True)

    ret_s = moving_average(returns, ma_window)
    x_ret = np.arange(1, len(ret_s) + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(x_ret, ret_s, label=f"Fixed (f={FIXED_FREQ}, s={FIXED_SLOTS}, P={FIXED_POWERW}W)",
             linestyle="-.", marker="s", markevery=max(1, len(x_ret)//20))
    plt.xlabel("Episode")
    plt.ylabel(f"Episode Return (Moving Avg, window={ma_window})")
    plt.title("Episode Return Curve (Fixed)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    ret_png = project_path(f"figs/fixed_return_{stamp}.png")
    plt.tight_layout(); plt.savefig(ret_png, dpi=300); print(f"[Saved] {ret_png}")

    sat_s = moving_average(satisfactions, ma_window)
    x_sat = np.arange(1, len(sat_s) + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(x_sat, sat_s, label=f"Fixed (f={FIXED_FREQ}, s={FIXED_SLOTS}, P={FIXED_POWERW}W)",
             linestyle="--", marker="o", markevery=max(1, len(x_sat)//20))
    plt.xlabel("Episode")
    plt.ylabel(f"Episode Satisfaction (Moving Avg, window={ma_window})")
    plt.title("Episode Satisfaction Curve (Fixed)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    sat_png = project_path(f"figs/fixed_satisfaction_{stamp}.png")
    plt.tight_layout(); plt.savefig(sat_png, dpi=300); print(f"[Saved] {sat_png}")

if __name__ == "__main__":
    run_fixed()
