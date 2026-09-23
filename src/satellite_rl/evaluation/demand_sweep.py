from satellite_rl.evaluation.checkpoints import load_actor_only, load_network_checkpoint
from satellite_rl.action_config import DEFAULT_POWER_STEP_W, PPO_DIRECT_POWER_STEP_W
from satellite_rl.paths import project_path
# test_compare_vs_demand.py
import os, math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from satellite_rl.envs.satellite import SatelliteFreqState, BeamInfo, Env, CurrentBeamState, beam_num

# === 算法实现（保持与你工程文件名一致；没有的可以先注释掉对应 import + METHODS）===
import torch
from satellite_rl.agents.sac_transformer import SAC as SAC_Transformer          # Transformer 版 SAC
from satellite_rl.agents.sac_cnn import SAC as SAC_CNN              # CNN 版 SAC
from satellite_rl.agents.sac_mlp import SAC as SAC_FC               # FC 版 SAC
from satellite_rl.agents.dqn import DQNAgent                       # DQN
from satellite_rl.agents.ppo_direct import PPO                             # PPO

# ======= 字体设置（论文风格，与你现有脚本一致）=======
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['axes.labelsize'] = 16
matplotlib.rcParams['axes.titlesize'] = 16
matplotlib.rcParams['legend.fontsize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = '#111111'
matplotlib.rcParams['axes.labelcolor'] = '#111111'
matplotlib.rcParams['xtick.color'] = '#111111'
matplotlib.rcParams['ytick.color'] = '#111111'


# ---------------- 简单基线策略（与你 metrics_generate.py 一致风格） ----------------
def fixed_policy(env, fixed_slots=8, fixed_power_w=30):
    """从左往右依次选择频率槽起始位置；时隙和功率固定"""
    freq = env.beam_idx // 8 * fixed_slots
    group = env.beam_idx % 8
    power_db = 10.0 * math.log10(float(fixed_power_w))
    return [group, int(freq), int(fixed_slots), float(power_db)]


def greedy_policy(env):
    """选最空闲频点；时隙=该列空位数(≥1)；功率按需求粗分档"""
    occ = np.asarray(env.satellite_freq_state.freq_pool_real, dtype=int)  # (8,100)
    freq = int(np.argmin(occ.sum(axis=0)))
    free_slots = int(10 - occ[:, freq].sum())
    slots = max(1, min(5, free_slots))

    req = float(env.current_beam_state.beam_info['rate'][env.beam_idx])
    pw = 30 if req >= 8e8 else 25 if req >= 4e8 else 20
    return [env.beam_idx % 8, freq, slots, 10.0 * math.log10(float(pw))]


# ---------------- 工具函数：加载 actor / 构造环境 / 动作映射 ----------------


def to_env_action(env, action_triplet, power_step_w=DEFAULT_POWER_STEP_W):
    """把 (freq_idx, slot_idx, pwr_idx) -> [group, freq, slots, power_dB]"""
    g = env.beam_idx % 8
    f = int(action_triplet[0])               # 0..99
    s = int(action_triplet[1]) + 1           # 1..10
    p_w = (int(action_triplet[2]) + 1) * power_step_w # power_step_w controls the training-compatible power grid
    p_db = 10.0 * math.log10(p_w)
    return [g, f, s, p_db]


def build_beams(seed=2025):
    """固定一个场景：固定 seed 生成一套 beam_location + rate（同一场景）"""
    np.random.seed(seed)
    beam_info = BeamInfo()   # 注意：内部会生成 rate 并排序/乱序应用
    return beam_info


def apply_avg_demand(beam_info, demand_mbps_per_beam):
    """
    将当前 beam_info.rate 做整体缩放，使得平均需求= demand_mbps_per_beam (Mbps/beam)
    - 保持相对分布形状不变（同一场景）
    """
    target_mean = float(demand_mbps_per_beam) * 1e6
    cur_mean = float(np.mean(beam_info.rate))
    scale = target_mean / max(cur_mean, 1e-12)

    beam_info.rate = beam_info.rate * scale
    beam_info.beam_info['rate'] = beam_info.rate  # 关键：Env/CurrentBeamState 用这里的 rate :contentReference[oaicite:2]{index=2}
    return beam_info


def build_env_from_beams(beam_info):
    """用给定 beam_info 构造一个干净环境"""
    sat = SatelliteFreqState()
    cur = CurrentBeamState(beam_info.beam_info)
    env = Env(sat, beam_info, cur)
    return env


def run_episode_per_beam(beam_info, agent_kind):
    """
    在同一套 beam_info 上跑一局，返回 per-beam 的 si 序列（即 satisfaction）
    """
    # 每个方法都用同一套 beam_info（公平）；环境状态独立
    env = build_env_from_beams(beam_info)
    state = env.reset(beam_info)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = None
    if agent_kind == 'sac_trans':
        agent = SAC_Transformer(n_states=204, n_hiddens=128,
                                freq_dim=100, slots_dim=10, power_dim=10,
                                actor_lr=1e-5, critic_lr=1e-4, alpha_lr=1e-4,
                                target_entropy=-6.9, tau=0.005, gamma=0.99,
                                device=device)
        load_actor_only(agent, project_path(r'saved_models/sac-t'), algorithm=agent_kind)

    elif agent_kind == 'sac_cnn':
        agent = SAC_CNN(n_states=204, n_hiddens=128,
                        freq_dim=100, slots_dim=10, power_dim=10,
                        actor_lr=1e-5, critic_lr=1e-4, alpha_lr=1e-4,
                        target_entropy=-6.9, tau=0.005, gamma=0.99,
                        device=device)
        load_actor_only(agent, project_path(r'saved_models/sac-cnn'), algorithm=agent_kind)

    elif agent_kind == 'sac_fc':
        agent = SAC_FC(n_states=204, n_hiddens=128,
                       freq_dim=100, slots_dim=10, power_dim=10,
                       actor_lr=1e-5, critic_lr=1e-4, alpha_lr=1e-4,
                       target_entropy=-6.9, tau=0.005, gamma=0.99,
                       device=device)
        load_actor_only(agent, project_path(r'saved_models/sac-fc'), algorithm=agent_kind)

    elif agent_kind == 'dqn':
        agent = DQNAgent()
        load_network_checkpoint(
            agent.online, project_path("saved_models/dqn/dqn_online.pth"),
            algorithm=agent_kind, state_dict_key="online", evaluation_mode=True,
        )

    elif agent_kind == 'ppo':
        agent = PPO(state_dim=204,
                    actor_lr=1e-4, critic_lr=1e-3,
                    lmbda=0.95, epochs=4, eps=0.2, gamma=0.99,
                    device=device)
        load_actor_only(agent, project_path(r'saved_models/ppo'), algorithm=agent_kind)

    si = []
    done = False
    while not done:
        if agent_kind.startswith('sac'):
            a = agent.take_action(state)  # [freq,slot,pwr]（索引）
            env_action = to_env_action(env, a)

        elif agent_kind == 'dqn':
            a = agent.choose_action(state, env, eps=0.0)  # 贪心
            env_action = to_env_action(env, a)

        elif agent_kind == 'ppo':
            a = agent.take_action(state)[0]
            env_action = to_env_action(env, a, power_step_w=PPO_DIRECT_POWER_STEP_W)

        elif agent_kind == 'greedy':
            env_action = greedy_policy(env)

        elif agent_kind == 'fixed':
            env_action = fixed_policy(env)

        else:
            # random
            a = [np.random.randint(0, 100), np.random.randint(0, 10), np.random.randint(0, 10)]
            env_action = to_env_action(env, a)

        next_state, reward, done, info = env.step(env_action)

        # satisfaction(si) 就是 current_satisfaction :contentReference[oaicite:3]{index=3}
        si.append(float(info.get('sgm_rate', np.nan)))
        state = next_state

    return np.asarray(si, dtype=float)


def plot_si_vs_demand(demands, curves, labels, out_png):
    """
    demands: list[float] x轴
    curves: dict(method -> list[y])
    """
    markers = ['s', '<', 'v', '>', '*', '^', '+', 'x', 'o', 'D']
    linestyles = ['-', '-', '-', '-', '-', '-', '-', '--', '--', '--']

    plt.figure(figsize=(7.0, 5.3))
    xs = np.array(demands, dtype=float)

    for i, m in enumerate(labels.keys()):
        ys = np.array(curves[m], dtype=float)
        plt.plot(xs, ys,
                 label=labels[m],
                 linestyle=linestyles[i % len(linestyles)],
                 marker=markers[i % len(markers)],
                 markersize=8,
                 linewidth=1.8,
                 markerfacecolor='none')

    plt.xlabel("Average capacity demand per beam (Mbps)")
    plt.ylabel("Average satisfaction (SI)")
    plt.ylim(0.5, 1.0)
    plt.xticks(xs)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best', frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print("saved:", out_png)

def save_curves_npz(demands, curves, labels, out_path):
    np.savez(
        out_path,
        demands=np.array(demands),
        **{f"{k}_si": np.array(v) for k, v in curves.items()},
        labels=labels
    )
    print("saved curves to:", out_path)


def main():
    # === 你要对比的方法（没有权重/实现的先从 METHODS 里删掉即可）===
    METHODS = ['random', 'greedy', 'fixed', 'dqn', 'ppo', 'sac_fc', 'sac_cnn', 'sac_trans']
    LABELS = {
        'random':   'Random',
        'greedy':   'Greedy',
        'fixed':    'Fixed Scheme',
        'dqn':      'MLP-DQN',
        'ppo':      'MLP-PPO',
        'sac_fc':   'MLP-SAC',
        'sac_cnn':  'CNN-SAC',
        'sac_trans':'Proposed T-SAC',
    }

    # === 同一场景（固定 seed）===
    SCENE_SEED = 2025

    # === 扫描流量需求（Mbps/beam）===
    DEMANDS = [300, 400, 500, 600, 700]

    # 输出目录
    os.makedirs(project_path("eval_fig/si"), exist_ok=True)
    out_png = project_path("eval_fig/si/si_mean_vs_demand.png")

    curves = {m: [] for m in METHODS}

    # 固定一套“场景基准 beams”（位置/排序固定），然后对 rate 缩放
    base_beams = build_beams(seed=SCENE_SEED)

    for d in DEMANDS:
        print(f"\n=== Demand = {d} Mbps/beam ===")
        # 为避免方法间互相污染：每个 demand 下，复制一份 beam_info 再缩放
        beams_d = build_beams(seed=SCENE_SEED)
        beams_d = apply_avg_demand(beams_d, d)

        for m in METHODS:
            # 为保证每个算法在该 demand 下都用同一套 beams：再复制一份（避免 env 修改内部引用）
            beams_m = build_beams(seed=SCENE_SEED)
            beams_m = apply_avg_demand(beams_m, d)

            si = run_episode_per_beam(beams_m, agent_kind=m)
            si_mean = float(np.nanmean(si))
            curves[m].append(si_mean)
            print(f"[{m:8s}] SI_mean = {si_mean:.4f}")

    plot_si_vs_demand(DEMANDS, curves, LABELS, out_png)
    save_curves_npz(DEMANDS, curves, LABELS, project_path("sgm_vs_demand_curves.npz"))


if __name__ == "__main__":
    main()
