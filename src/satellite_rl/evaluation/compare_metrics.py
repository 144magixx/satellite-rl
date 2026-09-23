from satellite_rl.evaluation.terminal_metrics import terminal_beam_arrays
from satellite_rl.evaluation.checkpoints import load_actor_only, load_network_checkpoint
from satellite_rl.action_config import DEFAULT_POWER_STEP_W, PPO_DIRECT_POWER_STEP_W
from satellite_rl.paths import project_path
# test_compare_per_beam.py
import os, math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime

# === 环境 ===
from satellite_rl.envs.satellite import SatelliteFreqState, BeamInfo, Env, CurrentBeamState, beam_num

# === 算法实现（与你的工程中文件名保持一致）===
from satellite_rl.agents.sac_transformer import SAC as SAC_Transformer          # Transformer 版 SAC
from satellite_rl.agents.sac_cnn import SAC as SAC_CNN              # CNN 版 SAC
from satellite_rl.agents.sac_mlp import SAC as SAC_FC               # FC 版 SAC
from satellite_rl.agents.dqn import DQNAgent                    # DQN
from satellite_rl.agents.ppo_direct import PPO
# --- 只加载 Actor 的通用函数（SAC 三种架构通用） ---
import torch
# ======= 字体与配色全局设置 =======
matplotlib.rcParams['font.family'] = 'Times New Roman'     # 字体
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['axes.unicode_minus'] = False

# 字体大小（适合论文）
matplotlib.rcParams['font.size'] = 14           # 全局字体大小
matplotlib.rcParams['axes.labelsize'] = 16      # 坐标轴标题
matplotlib.rcParams['axes.titlesize'] = 16      # 图标题
matplotlib.rcParams['legend.fontsize'] = 12     # 图例
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12

# 字体颜色加深（默认黑色基础上略深灰）
matplotlib.rcParams['text.color'] = '#111111'
matplotlib.rcParams['axes.labelcolor'] = '#111111'
matplotlib.rcParams['xtick.color'] = '#111111'
matplotlib.rcParams['ytick.color'] = '#111111'
import math

def fixed_policy(env, fixed_slots=8, fixed_power_w=30):
    """从左往右依次选择频率槽起始位置；时隙和功率固定"""
    # 频谱占用矩阵 shape=(10, 100)
    occ = np.asarray(env.satellite_freq_state.freq_pool_real, dtype=int)

    # 获取频率总数
    num_freq = occ.shape[1]

    # 从左到右扫描，找到第一个未满的频率槽
    freq = env.beam_idx // 8 * fixed_slots

    # 固定参数
    group = env.beam_idx % 8
    power_db = 10 * math.log10(float(fixed_power_w))

    return [group, freq, int(fixed_slots), power_db]


def greedy_policy(env):
    # 选最空闲频点；时隙=该列空位数(≥1)；功率按需求粗分档
    occ = np.asarray(env.satellite_freq_state.freq_pool_real, dtype=int)
    freq = int(np.argmin(occ.sum(axis=0)))
    free_slots = int(10 - occ[:, freq].sum())
    slots = max(1, min(5, free_slots))
    req = float(env.current_beam_state.beam_info['rate'][env.beam_idx])
    pw = 30 if req >= 8e8 else 25 if req >= 4e8 else 20

    return [env.beam_idx % 8, freq, slots, 10*math.log10(float(pw))]




# ---------------- 工具函数 ----------------
def build_env_with_fixed_beams(seed=42):
    np.random.seed(seed)
    beam_info = BeamInfo()  # 生成一套固定顺序的波束序列
    sat = SatelliteFreqState()
    cur = CurrentBeamState(beam_info.beam_info)
    env = Env(sat, beam_info, cur)
    return env, beam_info

def to_env_action(env, action_triplet, power_step_w=DEFAULT_POWER_STEP_W):
    """把 (freq_idx, slot_idx, pwr_idx) -> [group, freq, slots, power_dB]"""
    g = env.beam_idx % 8
    f = int(action_triplet[0])                  # 0..99
    s = int(action_triplet[1]) + 1              # 1..10
    p_w = (int(action_triplet[2]) + 1) * power_step_w    # power_step_w controls the training-compatible power grid
    p_db = 10.0 * math.log10(p_w)
    return [g, f, s, p_db]

def cinr_db_from_eta(eta):
    """eta = log2(1+SINR)  =>  SINR = 2^eta - 1"""
    if eta is None or not np.isfinite(float(eta)):
        return np.nan
    sinr = max(2.0**float(eta) - 1.0, 1e-12)
    return 10.0 * math.log10(sinr)

def run_episode_per_beam(env_ctor, agent_kind, seed=42, load=False):
    """
    env_ctor: callable() -> (env, beam_info)   —— 每个方法都用同一批 beam 顺序
    agent_kind: {'sac_trans','sac_cnn','sac_fc','dqn','random'}
    返回: dict(cinr, si, sgm)
    """
    # 固定波束顺序
    base_env, base_beams = env_ctor()
    # 为公平：每个算法用同一套 beam 顺序独立跑一遍
    sat = SatelliteFreqState()
    cur = CurrentBeamState(base_beams.beam_info)
    env = Env(sat, base_beams, cur)
    state = env.reset(base_beams)

    # 构建 agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        # DQN 评测同理只需“在线网络”/policy：
        load_network_checkpoint(
            agent.online, project_path("saved_models/dqn/dqn_online.pth"),
            algorithm=agent_kind, state_dict_key="online", evaluation_mode=True,
        )

    elif agent_kind == 'ppo':
        agent = PPO(state_dim=204,
            actor_lr=1e-4,
            critic_lr=1e-3,
            lmbda=0.95,
            epochs=4,
            eps=0.2,
            gamma=0.99,
            device=device)
        load_actor_only(agent, project_path(r'saved_models/ppo'), algorithm=agent_kind)



    cinr, si, sgm = [], [], []
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
            env_action = greedy_policy(env)  # 已是 [group,freq,slots,p_dB]

        elif agent_kind == 'fixed':
            env_action = fixed_policy(env)  # 已是 [group,freq,slots,p_dB]

        else:
            a = [np.random.randint(0, 100), np.random.randint(0, 10), np.random.randint(0, 10)]
            env_action = to_env_action(env, a)

        next_state, reward, done, info = env.step(env_action)

        # 逐波束指标
        eta = info.get('spectral_efficiency_avg', np.nan)
        cinr.append(cinr_db_from_eta(eta))
        si.append(float(info.get('current_satisfaction', np.nan)))
        sgm_val = info.get('sgm_rate', np.nan)
        if hasattr(sgm_val, 'item'):
            try:
                sgm_val = float(sgm_val.item())
            except Exception:
                pass
        sgm.append(float(sgm_val))
        state = next_state

    final = terminal_beam_arrays(env)
    # CINR 从全部更新后的历史槽计算；未服务波束保留缺失标记。
    final_cinr = np.full(len(final['rates_bps']), np.nan)
    for beam in range(len(final_cinr)):
        values = [float(np.asarray(row[2]).reshape(-1)[0]) for row in env.history_cinr if row[4] == beam]
        if values:
            final_cinr[beam] = 10 * np.log10(np.mean(10 ** (np.asarray(values) / 10)))
    return {'cinr': final_cinr, 'si': final['satisfaction'], 'sgm': final['sgm']}

def pad_fixed(arr, L, fill):
    """将一维数组截断/补齐到长度 L；不足用 fill 补"""
    a = np.asarray(arr, dtype=float).ravel()
    if len(a) >= L:
        return a[:L]
    out = np.full(L, fill, dtype=float)
    out[:len(a)] = a
    return out

# ---------------- 主流程：跑 5 个方法并作一张图 ----------------
def main():
    METHODS = ['sac_trans', 'sac_cnn', 'sac_fc', 'dqn', 'ppo', 'greedy', 'fixed', 'random']
    LABELS = {
        'sac_trans': 'Proposed T-SAC',
        'sac_fc': 'MLP-SAC',
        'sac_cnn': 'CNN-SAC',
        'dqn': 'MLP-DQN',
        'ppo': 'MLP-PPO',
        'greedy': 'Greedy',
        'fixed': 'Fixed Scheme',
        'random': 'Random'
    }

    # ===== 跑 10 个场景（不同随机种子）=====
    NUM_SCENES = 10
    SEEDS = [2025 + i for i in range(NUM_SCENES)]

    # 收集统计：method -> 场景索引 -> 指标均值/方差
    stats = {m: {'cinr_mean':[], 'cinr_var':[],
                 'si_mean':[],   'si_var':[],
                 'sgm_mean':[],  'sgm_var':[]} for m in METHODS}

    for si_idx, seed in enumerate(SEEDS):
        print(f'== Scenario {si_idx} (seed={seed}) ==')
        # 用固定 beams 构造器（保留你原来的环境/动作映射与加载逻辑）
        env_ctor = lambda: build_env_with_fixed_beams(seed=seed)
        for m in METHODS:
            res = run_episode_per_beam(env_ctor, m, seed=seed)

            # --- 预处理：替换/补齐 ---
            cinr = np.asarray(res.get('cinr', []), dtype=float).ravel()
            cinr = np.where(cinr<-10.0, -10.0, cinr)  # -120 → -10
            cinr = pad_fixed(cinr, beam_num, fill=-10.0)
            si   = pad_fixed(res.get('si',  []), beam_num, fill=0.0)
            sgm  = pad_fixed(res.get('sgm', []), beam_num, fill=0.0)

            # --- 按波束求均值/方差（本场景）---
            stats[m]['cinr_mean'].append(float(np.mean(cinr)))
            stats[m]['cinr_var' ].append(float(np.var(cinr)))
            stats[m]['si_mean'  ].append(float(np.mean(si)))
            stats[m]['si_var'   ].append(float(np.var(si)))
            stats[m]['sgm_mean' ].append(float(np.mean(sgm)))
            stats[m]['sgm_var'  ].append(float(np.var(sgm)))

    # 转为数组
    for m in METHODS:
        for k in stats[m]:
            stats[m][k] = np.asarray(stats[m][k], dtype=float)

    def plot_series(metric_key_mean, metric_key_var, ylabel, out_png):
        """在同一张图上绘制均值（上）与方差（下）"""
        xs = np.arange(NUM_SCENES)
        markers = ['o', 's', '^', 'v', '>', 'D', 'P']
        linestyles = ['-', '--', '-.', ':', '-', '--', '-.']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#7f7f7f', '#17becf']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True)


        # ---- 上：均值 ----
        for i, m in enumerate(METHODS):
            ax1.plot(xs, stats[m][metric_key_mean],
                     label=LABELS[m],
                     linestyle=linestyles[i % len(linestyles)],
                     marker=markers[i % len(markers)],
                     markevery=1,
                     markersize=5,
                     linewidth=1.8)
        ax1.set_ylabel(f"{ylabel} Mean")
        ax1.grid(True, linestyle='--', alpha=0.5)
        # 图例放在上图下方
        # ——放在画完上半图所有曲线之后——
        # 图例放画布底部靠上，避免压到 x 轴标签
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels,
                   loc='lower center', bbox_to_anchor=(0.5, 0.02),
                   ncol=4, frameon=False)

        # 保证图例不被裁剪（把原来的 subplots_adjust 替换或合并）
        plt.subplots_adjust(top=0.97, bottom=0.20, left=0.08, right=0.98, hspace=0.06)
        # ---- 下：方差 ----
        for i, m in enumerate(METHODS):
            ax2.plot(xs, stats[m][metric_key_var],
                     linestyle=linestyles[i % len(linestyles)],
                     marker=markers[i % len(markers)],
                     markevery=1,
                     markersize=4,
                     linewidth=1.5)
        ax2.set_xlabel("Scenario Index", loc="right", labelpad=8)

        ax2.set_ylabel(f"{ylabel} Var")
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.set_xticks(np.arange(NUM_SCENES))

        plt.subplots_adjust(top=0.96, bottom=0.10, left=0.08, right=0.98, hspace=0.05)

        plt.savefig(out_png, dpi=300)
        print('saved:', out_png)

    def plot_single_metric(metric_key, ylabel, out_png, stats, METHODS, LABELS):
        """绘制某个指标（均值或方差）的单图折线图"""
        xs = np.arange(len(next(iter(stats.values()))[metric_key]))
        markers = ['o', 's', '^', 'v', '>', 'D', 'P']
        linestyles = ['-', '--', '-.', ':', '-', '--', '-.']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#7f7f7f', '#17becf']

        plt.figure(figsize=(5, 5))
        for i, m in enumerate(METHODS):
            plt.plot(xs, stats[m][metric_key],
                     label=LABELS[m],
                     linestyle=linestyles[i % len(linestyles)],
                     marker=markers[i % len(markers)],
                     markersize=5,
                     linewidth=1.8)
        plt.xlabel("Scenario Index", loc="right")
        plt.ylabel(ylabel)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='best', fontsize=10)
        plt.tight_layout()
        plt.savefig(out_png, dpi=300)
        print('saved:', out_png)

    os.makedirs(project_path('eval_fig/cinr'), exist_ok=True)
    os.makedirs(project_path('eval_fig/si'), exist_ok=True)
    os.makedirs(project_path('eval_fig/sgm'), exist_ok=True)
    # ===== 输出 6 张图：每个指标的“均值折线”和“方差折线”=====
    # plot_series('cinr_mean', 'cinr_var', 'CINR (dB)', f'eval_fig/cinr/scenes_cinr_mean_var.png')
    # plot_series('si_mean', 'si_var', 'Satisfaction', f'eval_fig/si/scenes_si_mean_var.png')
    # plot_series('sgm_mean', 'sgm_var', 'SGM', f'eval_fig/sgm/scenes_sgm_mean_var.png')
    #
    # plot_series('cinr_mean', 'cinr_var', 'CINR (dB)', f'eval_fig/cinr/scenes_cinr_mean_var.eps')
    # plot_series('si_mean', 'si_var', 'Satisfaction', f'eval_fig/si/scenes_si_mean_var.eps')
    # plot_series('sgm_mean', 'sgm_var', 'SGM', f'eval_fig/sgm/scenes_sgm_mean_var.eps')
    plot_single_metric('cinr_mean', 'SINR (dB) Mean', project_path('eval_fig/cinr/scenes_cinr_mean.png'), stats, METHODS, LABELS)
    plot_single_metric('cinr_var', 'SINR (dB) Var', project_path('eval_fig/cinr/scenes_cinr_var.png'), stats, METHODS, LABELS)

    plot_single_metric('si_mean', 'Satisfaction Mean', project_path('eval_fig/si/scenes_si_mean.png'), stats, METHODS, LABELS)
    plot_single_metric('si_var', 'Satisfaction Var', project_path('eval_fig/si/scenes_si_var.png'), stats, METHODS, LABELS)

    plot_single_metric('sgm_mean', 'SGM Mean', project_path('eval_fig/sgm/scenes_sgm_mean.png'), stats, METHODS, LABELS)
    plot_single_metric('sgm_var', 'SGM Var', project_path('eval_fig/sgm/scenes_sgm_var.png'), stats, METHODS, LABELS)

    plot_single_metric('cinr_mean', 'SINR (dB) Mean', project_path('eval_fig/cinr/scenes_cinr_mean.eps'), stats, METHODS, LABELS)
    plot_single_metric('cinr_var', 'SINR (dB) Var', project_path('eval_fig/cinr/scenes_cinr_var.eps'), stats, METHODS, LABELS)

    plot_single_metric('si_mean', 'Satisfaction Mean', project_path('eval_fig/si/scenes_si_mean.eps'), stats, METHODS, LABELS)
    plot_single_metric('si_var', 'Satisfaction Var', project_path('eval_fig/si/scenes_si_var.eps'), stats, METHODS, LABELS)

    plot_single_metric('sgm_mean', 'SGM Mean', project_path('eval_fig/sgm/scenes_sgm_mean.eps'), stats, METHODS, LABELS)
    plot_single_metric('sgm_var', 'SGM Var', project_path('eval_fig/sgm/scenes_sgm_var.eps'), stats, METHODS, LABELS)


if __name__ == '__main__':
    main()
