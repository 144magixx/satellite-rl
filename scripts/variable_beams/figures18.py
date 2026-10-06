"""Build the S2-P18 training curves and the three-arm comparison figure.

Training metrics are read from each run's own ``metrics.jsonl`` over SFTP, so the
curves come from the saved logs rather than from any recomputation. Figures are drawn
locally with a CJK font because the remote containers have none.
"""
from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, r"D:\paper\scripts\variable_beams")
import ops18  # noqa: E402
from analyse18 import configure_plotting  # noqa: E402

OUT = Path(r"D:\paper\outputs\experiments\variable_beams_s2p18_20261006\analysis")
RUNS = {"V2": (20418, "train_sac_20261006_084454_365eff71"),
        "V3": (20424, "train_sac_20261006_084726_c004cd09"),
        "G0": (20418, "train_sac_20261006_092142_142c6953")}
V1_SI = {50: 0.915, 100: 0.934, 200: 0.898}
GREEDY_SI = 0.8933191049569863


def read_metrics(port: int, run_id: str) -> list:
    info = ops18.instance(port)
    path = f"{info['archive']}/outputs/workbench/jobs/{run_id}/metrics.jsonl"
    import paramiko
    client = ops18.connect(port)
    try:
        with paramiko.SFTPClient.from_transport(client.get_transport()) as sftp:
            with sftp.open(path) as handle:
                handle.prefetch()
                text = handle.read().decode("utf-8", errors="replace")
    finally:
        client.close()
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def training_curve() -> None:
    plt, font = configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    # 指标键取自实际日志：验证节点写在 group=validation 的 final_si_mean，
    # 回合回报写在 group=training 的 return。
    series = {"validation": {}, "training": {}}
    for arm, (port, run_id) in RUNS.items():
        rows = read_metrics(port, run_id)
        series["validation"][arm] = [(row.get("step"), row["values"]["final_si_mean"])
                                     for row in rows
                                     if row.get("group") == "validation"
                                     and isinstance(row.get("values", {}).get("final_si_mean"),
                                                    (int, float))]
        series["training"][arm] = [(row.get("step"), row["values"]["return"])
                                   for row in rows
                                   if row.get("group") == "training"
                                   and isinstance(row.get("values", {}).get("return"),
                                                  (int, float))]
        print(f"{arm}: metrics 行数={len(rows)}，验证节点 {len(series['validation'][arm])}，"
              f"回合 {len(series['training'][arm])}")
    for axis, group, label in ((axes[0], "validation", "验证需求满足率 SI"),
                               (axes[1], "training", "回合累计回报")):
        for arm, points in series[group].items():
            xs = [step for step, value in points if step]
            ys = [value for step, value in points if step]
            if xs:
                axis.plot(xs, ys, marker="o", markersize=3, label=arm)
        axis.set_xlabel("环境交互步数")
        axis.set_ylabel(label)
        axis.grid(True, linestyle="--", alpha=0.4)
        axis.legend(fontsize=8)
    axes[0].set_title("训练内验证需求满足率（各臂节点值）")
    axes[1].set_title("回合累计回报（含各臂自身奖励定义，仅看趋势）")
    fig.suptitle("S2-P18 三臂训练曲线（seed 51001，40 万步）", y=1.06)
    fig.tight_layout()
    path = OUT / "中文_三臂训练曲线.png"
    fig.savefig(path)
    plt.close(fig)
    print("已生成", path)


def comparison_figure() -> None:
    plt, font = configure_plotting()
    counts = [50, 100, 200]
    arms = {"V2": {50: 0.8666, 100: 0.8972, 200: 0.8641},
            "V3": {50: 0.8973, 100: 0.8857, 200: 0.8205},
            "G0": {50: 0.8234, 100: 0.8100, 200: 0.8240}}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for arm, values in arms.items():
        xs = [count for count in counts if count in values]
        axes[0].plot(xs, [values[count] for count in xs], marker="o", label=arm)
    axes[0].plot(counts, [V1_SI[count] for count in counts], marker="s",
                 linestyle="--", label="V1（报告13 冻结）")
    axes[0].axhline(GREEDY_SI, color="gray", linestyle=":", label="贪心基线")
    axes[0].set_xlabel("目标波束数 N")
    axes[0].set_ylabel("需求满足率 SI")
    axes[0].set_title("直接分配：三臂与冻结基准")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=8)

    slots = {"V2": {50: 4.03, 100: 2.85, 200: 1.96},
             "V3": {50: 4.05, 100: 2.72, 200: 1.68},
             "G0": {50: 5.73, 100: 3.66, 200: 2.38},
             "V1": {50: 3.92, 100: 2.79, 200: 1.82}}
    for arm, values in slots.items():
        style = {"marker": "s", "linestyle": "--"} if arm == "V1" else {"marker": "o"}
        axes[1].plot(counts, [values[count] for count in counts], label=arm, **style)
    axes[1].set_xlabel("目标波束数 N")
    axes[1].set_ylabel("每束占用槽数")
    axes[1].set_title("频谱用量：每束槽数（G3 机制判据）")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(fontsize=8)
    fig.suptitle("S2-P18 三臂在同一批 440 景考核上的直接分配结果", y=1.06)
    fig.tight_layout()
    path = OUT / "中文_三臂对照与冻结基准.png"
    fig.savefig(path)
    plt.close(fig)
    print("已生成", path)


if __name__ == "__main__":
    training_curve()
    comparison_figure()
