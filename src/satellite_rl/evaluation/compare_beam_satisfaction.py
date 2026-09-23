from satellite_rl.evaluation.checkpoints import load_network_checkpoint
from satellite_rl.action_config import DEFAULT_POWER_STEP_W, PPO_DIRECT_POWER_STEP_W
from satellite_rl.paths import project_path
import os, math
import numpy as np
import matplotlib.pyplot as plt
import torch

# ==== 环境与各模型 ====
from satellite_rl.envs.satellite import BeamInfo, SatelliteFreqState, CurrentBeamState, Env  # :contentReference[oaicite:6]{index=6}
from satellite_rl.agents.sac_transformer import SAC                      # Transformer-SAC (T-SAC)         :contentReference[oaicite:7]{index=7}
from satellite_rl.agents.sac_mlp import SAC as SAC_FC         # MLP-SAC                          :contentReference[oaicite:8]{index=8}
from satellite_rl.agents.sac_cnn import SAC as SAC_CNN       # CNN-SAC                          :contentReference[oaicite:9]{index=9}
from satellite_rl.agents.dqn import DQNAgent                 # DQN (factored 3-head)            :contentReference[oaicite:10]{index=10}
from satellite_rl.agents.ppo_direct import PPO                       # PPO                              :contentReference[oaicite:11]{index=11}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _as_triplet(act):
    """
    把 take_action 的返回统一成 [f, s, p] 三元动作索引（全是 int）。
    支持几种常见形式：
      - [f, s, p] 或 np.ndarray/torch.Tensor(3,)
      - ( [f,s,p], logp ) 二元组
    """
    # 若是 (action, logp) 这样的二元组
    if isinstance(act, (tuple, list)) and len(act) == 2 and not np.isscalar(act[0]):
        act = act[0]

    # 转成一维
    if isinstance(act, torch.Tensor):
        act = act.detach().cpu().numpy()
    act = np.asarray(act).reshape(-1)

    if act.size != 3:
        raise ValueError(f"expect 3-dim action, got shape {act.shape} / value {act}")

    return [int(act[0]), int(act[1]), int(act[2])]

# === 通用：将各智能体的离散动作转为环境 action ===
def to_env_action(env, a_triplet, power_step_w=DEFAULT_POWER_STEP_W):
    """
    a_triplet: [freq_idx(0-99), slots_idx(0-9), power_idx(0-9)]
    环境需要: [group(0-7), freq(0-99), slots(1-10), power_dB]
    功率档位默认每档 5 W；PPO direct 显式使用每档 3 W。
    """
    group = env.beam_idx % 8
    freq  = int(a_triplet[0])
    slots = int(a_triplet[1]) + 1
    p_w   = (int(a_triplet[2]) + 1) * power_step_w
    power_db = 10.0 * math.log10(p_w)
    return [group, freq, slots, power_db]

# === 基线策略 ===
def greedy_policy(env):
    """
    Greedy：列负载最小的起始频点 + 能放多少时隙就放多少 + 需求分档功率
    参考你之前的 Greedy 版本。:contentReference[oaicite:12]{index=12}
    """
    occ = np.asarray(env.satellite_freq_state.freq_pool_real, dtype=int)  # (10,100)
    col_load = occ.sum(axis=0)
    freq = int(np.argmin(col_load))
    free_slots = int(10 - occ[:, freq].sum())
    slots = max(1, min(10, free_slots))
    req = float(env.current_beam_state.beam_info['rate'][env.beam_idx])
    if req >= 8e8: pw = 30
    elif req >= 4e8: pw = 25
    else: pw = 20
    return [env.beam_idx % 8, freq, slots, 10*math.log10(pw)]

def fixed_policy(env, fixed_slots=5, fixed_pw=30):
    """
    Fixed*：固定时隙与功率；频点选“最空闲列”（避免硬固定起点更公平）。
    参考你的 Fixed 版本思路。:contentReference[oaicite:13]{index=13}
    """
    occ = np.asarray(env.satellite_freq_state.freq_pool_real, dtype=int)
    col_load = occ.sum(axis=0)
    freq = int(np.argmin(col_load))
    return [env.beam_idx % 8, freq, int(fixed_slots), 10*math.log10(float(fixed_pw))]

# === 单回合评测，返回每个波束的满意度（容量匹配率） ===
@torch.no_grad()
def eval_once(agent_kind):
    """
    agent_kind ∈ {'t-sac','mlp-sac','cnn-sac','dqn','ppo','greedy','fixed'}
    返回: satisfaction_per_beam (len=100, [0..1])
    """
    # 统一环境（100 个波束，城市位置 & 需求已在环境内部随机/排序） :contentReference[oaicite:14]{index=14}
    beam_info = BeamInfo()
    sat = SatelliteFreqState()
    cur = CurrentBeamState(beam_info.beam_info)
    env = Env(sat, beam_info, cur)

    # ---- 初始化各智能体并加载权重 ----
    if agent_kind == 't-sac':
        agent = SAC(204, 128, 100, 10, 10, 1e-5, 1e-4, 1e-4,
                    target_entropy=-6.9, tau=0.005, gamma=0.99, device=device)
        load_network_checkpoint(agent.actor, project_path("saved_models/sac-t/actor.pth"), algorithm=agent_kind, map_location=device)
        policy = lambda s: to_env_action(env, _as_triplet(agent.take_action(s)))  # take_action 返回 ([f,s,p], logp)
    elif agent_kind == 'mlp-sac':
        agent = SAC_FC(204, 128, 100, 10, 10, 1e-5, 1e-4, 1e-4,
                       target_entropy=-6.9, tau=0.005, gamma=0.99, device=device)
        load_network_checkpoint(agent.actor, project_path("saved_models/sac-fc/actor.pth"), algorithm=agent_kind, map_location=device)
        policy = lambda s: to_env_action(env, _as_triplet(agent.take_action(s)))
    elif agent_kind == 'cnn-sac':
        agent = SAC_CNN(204, 128, 100, 10, 10, 1e-5, 1e-4, 1e-4,
                        target_entropy=-6.9, tau=0.005, gamma=0.99, device=device)
        load_network_checkpoint(agent.actor, project_path("saved_models/sac-cnn/actor.pth"), algorithm=agent_kind, map_location=device)
        policy = lambda s: to_env_action(env, _as_triplet(agent.take_action(s)))
    elif agent_kind == 'dqn':
        agent = DQNAgent()
        load_network_checkpoint(agent.online, project_path("saved_models/dqn/dqn_online.pth"), algorithm=agent_kind, map_location=device)
        def dqn_policy(s):
            s_t = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            qf, qs, qp = agent.online(s_t)               # 三个 head 的 logits/Q 值  :contentReference[oaicite:15]{index=15}
            a = [qf.argmax(-1).item(), qs.argmax(-1).item(), qp.argmax(-1).item()]
            return to_env_action(env, a)
        policy = dqn_policy
    elif agent_kind == 'ppo':
        # PPO_204 中策略网络输出 (freq, slots, power) 三个分类分布  :contentReference[oaicite:16]{index=16}
        ppo = PPO(state_dim=204, actor_lr=1e-4, critic_lr=1e-4,
                  lmbda=0.95, epochs=4, eps=0.2, gamma=0.99,
                  device=device, mini_batch_size=64)
        load_network_checkpoint(ppo.actor, project_path("saved_models/ppo/actor.pth"), algorithm=agent_kind, map_location=device)
        def ppo_policy(s):
            s_t = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            f_logits, sl_logits, pw_logits = ppo.actor(s_t)  # 概率分布
            a = [f_logits.argmax(-1).item(), sl_logits.argmax(-1).item(), pw_logits.argmax(-1).item()]
            return to_env_action(env, a, power_step_w=PPO_DIRECT_POWER_STEP_W)
        policy = ppo_policy
    elif agent_kind == 'greedy':
        policy = lambda s: greedy_policy(env)
    elif agent_kind == 'fixed':
        policy = lambda s: fixed_policy(env)
    else:
        raise ValueError("unknown agent_kind")

    # ---- 开始单回合评测 ----
    s = env.reset(BeamInfo())
    done = False
    satis_per_beam = []

    while not done:
        a = policy(s)
        s, r, done, info = env.step(a)
        # current_satisfaction ∈ [0,1]，就是“容量匹配率”；逐波束记录  :contentReference[oaicite:17]{index=17}
        satis_per_beam.append(float(info["current_satisfaction"]))

    return np.array(satis_per_beam, dtype=float)

# === 主函数：跑一轮并画对比曲线 ===
if __name__ == "__main__":
    algs = [
        ("T-SAC",     "t-sac"),
        # ("MLP-SAC",   "mlp-sac"),
        # ("CNN-SAC",   "cnn-sac"),
        # ("MLP-DQN",   "dqn"),
        # ("MLP-PPO",   "ppo"),
        # ("Greedy",    "greedy"),
        # ("Fixed",     "fixed"),
    ]

    results = {}
    for name, key in algs:
        print(f"[Eval] {name} ...")
        results[name] = eval_once(key)   # shape = (100,)

    # ---- 新增：长度不够 100 的用 0 填充 ----
    target_len = 100
    for k, v in results.items():
        arr = np.asarray(v, float)
        if len(arr) < target_len:
            arr = np.pad(arr, (0, target_len - len(arr)), constant_values=0)
        else:
            arr = arr[:target_len]
        results[k] = arr

    # --- 绘图：100 个波束的容量匹配率（不透明，便于 EPS 导出）---

    x = np.arange(1, 101)
    plt.figure(figsize=(12, 6))
    markers = ['o','s','^','v','>','D','P']
    linestyles = ['-','--','-.',':','-','--','-.']
    for i, (name, _) in enumerate(algs):
        y = results[name] * 100.0  # 转为 %
        plt.plot(x, y, label=name, linestyle=linestyles[i%len(linestyles)],
                 marker=markers[i%len(markers)], markevery=5, linewidth=1.6)

    plt.xlabel("Beam Index")
    plt.ylabel("Capacity Matching Rate (%)")
    plt.title("Per-Beam Satisfaction (One Episode, N=100 Beams)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    os.makedirs(project_path("figs"), exist_ok=True)
    plt.tight_layout()
    plt.savefig(project_path("figs/satisfaction_compare_100beams.pdf"), dpi=600)  # 也可改 .eps
    plt.show()

    # 也保存原始数据，便于论文统计
    os.makedirs(project_path("metrics"), exist_ok=True)
    np.savez(project_path("metrics/satisfaction_compare_100beams_once.npz"), **results)
    print("[Saved] figs/satisfaction_compare_100beams.pdf & metrics/satisfaction_compare_100beams_once.npz")
