"""汇总 results.jsonl：方法均值、相对贪心的逐景配对差（自助法区间）、三张图。

    python -m satellite_rl.original_compare.summarize RUN_DIR
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ORDER = ["T-SAC", "MLP-SAC", "改进式SAC(I0)", "容量感知贪心", "纯禁忌32",
         "贪心+禁忌32", "贪心+禁忌320", "贪心+禁忌96", "原版旧贪心"]
LABEL = {"T-SAC": "直出T-SAC", "MLP-SAC": "去Transformer的MLP-SAC", "改进式SAC(I0)": "改进式SAC（I0）",
         "容量感知贪心": "容量感知贪心", "纯禁忌32": "纯禁忌搜索32次", "贪心+禁忌32": "贪心＋禁忌搜索32次",
         "贪心+禁忌320": "贪心＋禁忌搜索320次", "贪心+禁忌96": "同耗时贪心＋禁忌搜索96次",
         "原版旧贪心": "原版自带贪心（参照）"}
KEYS = ("W", "SI", "worst10", "SGM", "power_w", "slots", "seconds")


def load(run_dir):
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").open(encoding="utf8")]
    rows.sort(key=lambda r: r["seed"])
    table = {m: {k: np.array([np.mean([x[k] for x in r["methods"][m]]) for r in rows]) for k in KEYS}
             for m in ORDER}
    for m in ORDER:
        table[m]["overload"] = np.array([np.mean([x["overload"] for x in r["methods"][m]]) for r in rows])
    return [r["seed"] for r in rows], table


def bootstrap(diff, rng, count=10000):
    means = diff[rng.integers(0, len(diff), (count, len(diff)))].mean(1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main(run_dir):
    run_dir = Path(run_dir)
    seeds, table = load(run_dir)
    rng = np.random.default_rng(0)
    summary = dict(scenes=len(seeds), methods={})
    for m in ORDER:
        t = table[m]
        entry = {k: float(t[k].mean()) for k in KEYS}
        entry["overload_rate"] = float(t["overload"].mean())
        for ref in ("容量感知贪心", "贪心+禁忌32"):
            for k in ("W", "worst10", "SGM"):
                entry[f"{k}_vs_{ref}"] = bootstrap(t[k] - table[ref][k], rng)
        entry["W_win_rate_vs_容量感知贪心"] = float((t["W"] > table["容量感知贪心"]["W"] + 1e-12).mean())
        summary["methods"][m] = entry
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf8")

    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    names = [LABEL[m] for m in ORDER]
    colors = ["#2a9d8f" if "SAC" in m or "I0" in m else "#e76f51" if "禁忌" in m else "#8d99ae" for m in ORDER]

    # 图1：两项主指标
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, key, title in zip(axes, ("W", "worst10"), ("业务量加权满足率", "最差10%波束平均满足率")):
        values = [table[m][key] * 100 for m in ORDER]
        means = [v.mean() for v in values]
        errors = np.array([[v.mean() - np.percentile(rng.choice(v, (4000, len(v))).mean(1), 2.5),
                            np.percentile(rng.choice(v, (4000, len(v))).mean(1), 97.5) - v.mean()] for v in values]).T
        y = np.arange(len(ORDER))[::-1]
        ax.barh(y, means, xerr=errors, color=colors, alpha=.85, capsize=3)
        for yi, mean in zip(y, means):
            ax.text(mean + 0.5, yi, f"{mean:.1f}%", va="center", fontsize=9)
        ax.set_title(title)
        ax.set_xlabel("%")
        ax.grid(axis="x", alpha=.3)
    axes[0].set_yticks(np.arange(len(ORDER))[::-1], names)
    axes[0].set_xlim(40, 105)
    axes[1].set_xlim(0, 105)
    fig.suptitle(f"原版框架，{len(seeds)}个随机场景（每景100束），误差线为95%区间")
    fig.tight_layout()
    fig.savefig(figures / "f1_main_metrics.png", dpi=150)
    plt.close(fig)

    # 图2：逐景相对容量感知贪心
    fig, ax = plt.subplots(figsize=(11, 5))
    base = table["容量感知贪心"]["W"]
    shown = [m for m in ORDER if m != "容量感知贪心"]
    for x, m in enumerate(shown):
        diff = (table[m]["W"] - base) * 100
        ax.scatter(x + rng.uniform(-.2, .2, len(diff)), diff, s=10, alpha=.5, color=colors[ORDER.index(m)])
        ax.plot([x - .3, x + .3], [diff.mean()] * 2, color="black", lw=2)
    ax.axhline(0, color="gray", lw=1)
    ax.set_xticks(range(len(shown)), [LABEL[m] for m in shown], rotation=20, ha="right")
    ax.set_ylabel("比容量感知贪心高多少（百分点）")
    ax.set_title("逐景业务量加权满足率差（每点一景，黑线为平均）")
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(figures / "f2_per_scene_vs_greedy.png", dpi=150)
    plt.close(fig)

    # 图3：质量—耗时
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for m in ORDER:
        ax.scatter(table[m]["seconds"].mean(), table[m]["W"].mean() * 100, s=60, color=colors[ORDER.index(m)])
        ax.annotate(LABEL[m], (table[m]["seconds"].mean(), table[m]["W"].mean() * 100),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("单景耗时（秒，对数轴；含初解构造与全部试算）")
    ax.set_ylabel("业务量加权满足率（%）")
    ax.set_title("质量与耗时")
    ax.grid(alpha=.3, which="both")
    fig.tight_layout()
    fig.savefig(figures / "f3_quality_vs_time.png", dpi=150)
    plt.close(fig)

    for m in ORDER:
        e = summary["methods"][m]
        print(f"{LABEL[m]:24s} W={e['W']*100:5.1f} SI={e['SI']*100:5.1f} worst10={e['worst10']*100:5.1f} "
              f"SGM={e['SGM']:.3f} P={e['power_w']:.0f} slots={e['slots']:.0f} t={e['seconds']:.2f}s "
              f"overload={e['overload_rate']:.2f} dW(G1)={e['W_vs_容量感知贪心'][0]*100:+.2f} "
              f"[{e['W_vs_容量感知贪心'][1]*100:+.2f},{e['W_vs_容量感知贪心'][2]*100:+.2f}] "
              f"dW(S1)={e['W_vs_贪心+禁忌32'][0]*100:+.2f} "
              f"[{e['W_vs_贪心+禁忌32'][1]*100:+.2f},{e['W_vs_贪心+禁忌32'][2]*100:+.2f}]")


if __name__ == "__main__":
    main(sys.argv[1])
