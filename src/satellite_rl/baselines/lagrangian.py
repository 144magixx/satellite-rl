import numpy as np
import math
from satellite_rl.envs.satellite import BeamInfo, SatelliteFreqState, CurrentBeamState, Env


def lagrangian_relaxation(env, max_iter=50, alpha=1.0):
    """
    使用Lagrangian松弛+次梯度法进行资源分配。
    松弛频率重用约束，逐波束优化slot/power选择。
    """
    group = env.beam_idx % 8
    occ = np.asarray(env.satellite_freq_state.freq_pool_real, dtype=int)
    required_rate = float(env.current_beam_state.beam_info['rate'][env.beam_idx])

    bandwidth = 2.5e7
    noise_power = 1e-10
    power_levels = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    max_slots = 10
    freq_dim = occ.shape[1]

    # 初始化对偶变量：对每个频段引入λ
    lambd = np.zeros(freq_dim)
    best_obj = float('inf')
    best_action = [group, 0, 1, 10]

    for it in range(max_iter):
        print(best_action)
        obj_val = 0
        best_freq = None
        best_slot = None
        best_power = None

        for f in range(freq_dim):
            if occ[:, f].sum() >= 10:
                continue  # 跳过已满频段

            slot_cap = 10 - occ[:, f].sum()
            for s in range(1, slot_cap + 1):
                for p in power_levels:
                    snr = p / noise_power
                    se = np.log2(1 + snr)
                    rate = se * bandwidth * s
                    if rate >= required_rate:
                        cost = p + lambd[f]  # Lagrangian目标
                        if cost < best_obj:
                            best_obj = cost
                            best_freq = f
                            best_slot = s
                            best_power = p

        # 次梯度更新λ
        subgrad = np.zeros(freq_dim)
        if best_freq is not None:
            subgrad[best_freq] = 1
        lambd = lambd + alpha * (subgrad - 1)
        lambd = np.maximum(lambd, 0)  # 投影到正数域

        if best_freq is not None:
            best_action = [group, best_freq, best_slot, 10 * math.log10(best_power)]

    return best_action


if __name__ == "__main__":
    beam_info = BeamInfo()
    sat = SatelliteFreqState()
    cur = CurrentBeamState(beam_info.beam_info)
    env = Env(sat, beam_info, cur)

    state = env.reset(BeamInfo())
    done = False
    total_reward = 0
    while not done:
        action = lagrangian_relaxation(env, max_iter=20, alpha=1.0)

        next_state, reward, done, info = env.step(action)
        total_reward += reward
        state = next_state

    print("[Lagrangian] Done. Total Reward:", total_reward)
    print("Final satisfaction:", info['total_satisfaction'])