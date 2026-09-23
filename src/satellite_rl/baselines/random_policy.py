from satellite_rl.paths import project_path
# random_baseline.py
import os
import numpy as np
from datetime import datetime

from satellite_rl.envs.satellite import SatelliteFreqState, BeamInfo, Env, CurrentBeamState, beam_num  # 环境复用  :contentReference[oaicite:3]{index=3}

def run_random_baseline(num_epochs: int = 4000, seed: int = 42):
    # 可复现实验
    np.random.seed(seed)

    # 环境初始化（与训练脚本一致的构造方式）
    beam_info = BeamInfo()
    satellite_freq_state = SatelliteFreqState()
    current_beam_state = CurrentBeamState(beam_info.beam_info)
    env = Env(satellite_freq_state, beam_info, current_beam_state)

    returns = []
    satisfactions = []

    for ep in range(num_epochs):
        # 每回合重置一个新的随机 beam 序列（与训练脚本保持一致做法） :contentReference[oaicite:4]{index=4}
        beam_info = BeamInfo()
        state = env.reset(beam_info)

        ep_return = 0.0
        done = False
        ep_satisfaction = 0.0

        while not done:
            # —— 纯随机动作（对比基线）——
            # 频组：取当前束所在组 (0~7)，与训练时的 env_action 对齐  :contentReference[oaicite:5]{index=5}
            group = env.beam_idx % 8
            freq  = np.random.randint(0, 100)     # 起始频率槽 0~99
            slots = np.random.randint(1, 11)      # 槽数 1~10
            pidx  = np.random.randint(0, 10)      # 功率索引 0~9

            # 与训练脚本一致的功率映射：10*log10((pidx+1)*5) dB  :contentReference[oaicite:6]{index=6}
            power_db = 10 * np.log10((pidx + 1) * 5)

            env_action = [group, freq, slots, power_db]

            # 与环境交互
            next_state, reward, done, info = env.step(env_action)

            ep_return += float(reward)
            # 记录最新总满足率（环境里已累计） :contentReference[oaicite:7]{index=7}
            ep_satisfaction = info.get('total_satisfaction', ep_satisfaction)

            state = next_state

        returns.append(ep_return)
        satisfactions.append(ep_satisfaction)

        util = np.count_nonzero(env.satellite_freq_state.freq_pool_real)
        print(f"[RAND] Episode {ep:4d} | Return: {ep_return:.2f} | "
              f"Satisfaction: {ep_satisfaction:.2f} | util={util}")

    # 保存 npz（文件名带时间戳；与训练脚本存放目录保持一致） :contentReference[oaicite:8]{index=8}
    os.makedirs(project_path('random_data'), exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d%H%M')
    path = project_path(f'random_data/random_metrics_{stamp}.npz')
    np.savez(path, returns=np.array(returns, dtype=np.float32),
                   satisfactions=np.array(satisfactions, dtype=np.float32))
    print(f"[RAND] Saved baseline metrics to {path}")

if __name__ == "__main__":
    run_random_baseline(num_epochs=4000, seed=42)
