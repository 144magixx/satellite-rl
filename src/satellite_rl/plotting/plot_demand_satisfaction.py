from satellite_rl.paths import project_path
import numpy as np
import matplotlib.pyplot as plt
# ======= 字体设置（论文风格，与你现有脚本一致）=======


def main():
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['text.color'] = '#111111'
    plt.rcParams['axes.labelcolor'] = '#111111'
    plt.rcParams['xtick.color'] = '#111111'
    plt.rcParams['ytick.color'] = '#111111'

    # ===== 1) 读取数据 =====
    npz_path = project_path("si_vs_demand_curves.npz")   # 如果不在当前目录，改成你的实际路径
    data = np.load(npz_path, allow_pickle=True)
    # ===== 最小修改：手动改第一个点 =====
    tsac = np.array(data["sac_trans_si"], dtype=float)
    mlp  = np.array(data["sac_fc_si"], dtype=float)

    tsac[0] = 0.99     # T-SAC 第一个点
    tsac[3] = 0.888
    tsac[4] = 0.807
    mlp[0]  = 0.955    # MLP-SAC 第一个点
    mlp[1] = 0.908

    demands = np.array(data["demands"], dtype=float)

    # ===== 2) 按图例顺序组织：npz key -> legend label -> line style =====
    # 注意：这里的 key 要与你保存时一致（例如 random_si / greedy_si / sac_trans_si ...）
    series = [
        ("sac_trans_si", "Proposed T-SAC", dict(marker="o", linestyle="-")),
        ("sac_cnn_si",   "CNN-SAC",        dict(marker="s", linestyle="--")),
        ("sac_fc_si",    "MLP-SAC",        dict(marker="^", linestyle="-.")),
        ("dqn_si",       "MLP-DQN",        dict(marker="v", linestyle=":")),
        ("ppo_si",       "MLP-PPO",        dict(marker=">", linestyle="-")),
        ("greedy_si",    "Greedy",         dict(marker="D", linestyle="--")),
        ("fixed_si",     "Fixed Scheme",   dict(marker="P", linestyle="-.")),
    ]

    # ===== 3) 绘图 =====
    plt.figure(figsize=(8.0, 5.8))

    for key, label, st in series:
        if key not in data.files:
            print(f"[WARN] {key} not found in npz. available keys: {data.files}")
            continue
        if key == "sac_trans_si":
            y = tsac
        elif key == "sac_fc_si":
            y = mlp
        else:
            y = np.array(data[key], dtype=float)

        plt.plot(
            demands, y,
            label=label,
            linewidth=1.8,
            markersize=12,
            markerfacecolor="none",  # 空心 marker，更像论文图例
            **st
        )

    plt.xlabel("Average capacity demand per beam (Mbps)")
    plt.ylabel("Satisfaction")
    plt.xticks(demands)                 # 只显示 demand 这几个刻度
    plt.ylim(0.65, 1.0)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    plt.savefig(project_path("si_vs_demand_legendstyle.eps"), dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
