from satellite_rl.paths import project_path
import torch
import numpy as np
import matplotlib.pyplot as plt
from satellite_rl.agents.sac_transformer import SAC
from satellite_rl.envs.satellite import BeamInfo, SatelliteFreqState, CurrentBeamState, Env

# ========== 载入模型 ==========


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = project_path("saved_models/sac-t/actor.pth")  # 模型路径

    # 定义环境（与训练一致）
    beam_info = BeamInfo()
    satellite_freq_state = SatelliteFreqState()
    current_beam_state = CurrentBeamState(beam_info.beam_info)
    env = Env(satellite_freq_state, beam_info, current_beam_state)

    # SAC参数匹配（必须与训练时一致）
    n_states = 204
    n_hiddens = 128
    freq_dim = 100
    slots_dim = 10
    power_dim = 10

    agent = SAC(n_states, n_hiddens, freq_dim, slots_dim, power_dim,
                actor_lr=1e-5, critic_lr=1e-4, alpha_lr=1e-4,
                target_entropy=-6.9, tau=0.005, gamma=0.99, device=device)

    # 只加载策略网络
    agent.actor.load_state_dict(torch.load(model_path, map_location=device))
    agent.actor.eval()

    # ========== 生成样本数据 ==========
    beam_info = BeamInfo()
    state = env.reset(beam_info)
    allocate_rates = []
    required_rates = []

    done = False
    while not done:
        action = agent.take_action(state)
        env_action = [
            env.beam_idx % 8,
            action[0],
            action[1] + 1,
            10 * np.log10((action[2] + 1) * 5)
        ]
        next_state, reward, done, info = env.step(env_action)

        # 转成标量，避免塞入 ndarray
        req = float(np.ravel(info["required_rate"])[0])

        # 实际分配速率更稳妥的取法：env 会把当前波束的实际速率 push 到 env.beam_bps
        # 如果你坚持用 current_satisfaction * required_rate 也行，同样转成 float
        if len(env.beam_bps) > 0:
            alloc = float(env.beam_bps[-1])
        else:
            alloc = float(info["current_satisfaction"]) * req

        required_rates.append(req)
        allocate_rates.append(alloc)

        state = next_state


    # ========== 绘图 ==========
    req_arr = np.asarray(required_rates, dtype=float).ravel() / 1e6
    alloc_arr = np.asarray(allocate_rates, dtype=float).ravel() / 1e6

    indices = np.arange(len(req_arr))
    bar_width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(indices - bar_width/2, req_arr,   bar_width, label='Required Rate (Mbps)',  alpha=0.7)
    plt.bar(indices + bar_width/2, alloc_arr, bar_width, label='Allocated Rate (Mbps)', alpha=0.7)



    plt.xlabel("Beam Index")
    plt.ylabel("Rate (Mbps)")
    plt.title("Comparison of Required vs Allocated Rates per Beam")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(project_path("rate_comparison_bar.png"), dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
