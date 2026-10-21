"""Render advisor figures with the saved pure-tabu 32-probe result."""
import ast
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "outputs/experiments/s2_pure_tabu_20261021"
OUT = EXPERIMENT / "presentation_figures"
OUT.mkdir(exist_ok=True)
SUMMARY = json.loads((EXPERIMENT / "presentation_summary.json").read_text(encoding="utf-8"))
PURE = next(row for row in SUMMARY["rows"] if row["set"] == "conf_d432_n150" and row["method"] == "PTS32")
OLD_CSV = ROOT / "docs/Claude维护/导师汇报/codex版/数据与图/第二阶段_确认88景.csv"
DSH_SOURCE = ROOT / "outputs/advisor_brief_20261018/dsh版/build/make_figs_v2.py"
CC_SOURCE = ROOT / "outputs/advisor_brief_20261018/cc版/make_figs_v2.py"
with OLD_CSV.open(encoding="utf-8-sig", newline="") as stream:
    OLD = {row["method"]: row for row in csv.DictReader(stream)}


def source_data(path, name, *, function=None, first_label=None, symbols=None):
    """Read the historical plotting literals without executing its output code."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    scope = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function) if function else tree
    for node in ast.walk(scope):
        if not isinstance(node, ast.Assign) or not any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            continue
        value = eval(compile(ast.Expression(node.value), str(path), "eval"), {"__builtins__": {}}, symbols or {})
        if first_label is None or value[0][0] == first_label:
            return value
    raise ValueError(f"{name} not found in {path}")

plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei"],
                     "axes.unicode_minus": False, "figure.dpi": 180,
                     "savefig.dpi": 180, "axes.spines.top": False,
                     "axes.spines.right": False})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white", pad_inches=.14)
    plt.close(fig)


def bars(ax, labels, values, colors, fmt, bounds):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colors, height=.62)
    ax.set_yticks(y, labels)
    ax.set_xlim(*bounds)
    ax.grid(axis="x", alpha=.16)
    ax.set_axisbelow(True)
    for yi, value in zip(y, values):
        ax.text(value + (bounds[1]-bounds[0])*.012, yi, fmt(value), va="center", fontsize=9)


def dsh_figures():
    blue, light, gray, orange, brown = "#1f5fa8", "#6d9fd8", "#9aa5b1", "#e08a2e", "#c97b63"
    names = ["容量感知\n贪心", "纯禁忌\n32次", "贪心＋禁忌\n32次", "贪心＋禁忌\n320次", "SAC改进式\n(I0, 32次)", "T-SAC直出"]
    original_before = source_data(DSH_SOURCE, "before", function="fig_sgm_sca")
    original_after = source_data(DSH_SOURCE, "after", function="fig_sgm_sca")
    before = [original_before[0], PURE["before_SGM"], *original_before[1:]]
    after = [original_after[0], PURE["after_SGM"], *original_after[1:]]
    colors = [gray, light, light, blue, orange, brown]

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.1), gridspec_kw={"width_ratios": [.7, 1.3]})
    published = source_data(DSH_SOURCE, "vals1", function="fig_sgm_protocols")
    left.bar(["旧贪心", "T-SAC"], published, color=[gray, orange], width=.55)
    left.set_ylim(0, 1.02)
    left.set_title("已发表协议：SGM", fontsize=12)
    left.text(0, -.29, "21.6 Mbps，旧贪心，开发20景", transform=left.transAxes, fontsize=9)
    for i, v in enumerate(published): left.text(i, v+.018, f"{v:.3f}", ha="center")
    x = np.arange(len(names))
    right.bar(x, before, color=colors, width=.62)
    right.set_xticks(x, names, fontsize=9)
    right.set_ylim(.6, 1.02)
    right.set_title("压力协议：第三阶段之前的SGM", fontsize=12)
    right.text(0, -.29, "43.2 Mbps，确认88景；纯禁忌从随机合法初解起步", transform=right.transAxes, fontsize=9)
    for i, v in enumerate(before): right.text(i, v+.014, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout(w_pad=2)
    save(fig, "dsh_G1_SGM两种协议.png")

    fig, ax = plt.subplots(figsize=(11, 4.1))
    ax.bar(x-.19, before, width=.37, color="white", edgecolor=colors, linewidth=1.7,
           hatch="///", label="第三阶段之前")
    ax.bar(x+.19, after, width=.37, color=colors, label="SCA功率细调之后")
    ax.set_xticks(x, names, fontsize=9)
    ax.set_ylim(.7, 1.04)
    ax.set_ylabel("供需匹配程度 SGM")
    ax.legend(loc="upper left", ncol=2, frameon=False)
    for i, (a, b) in enumerate(zip(before, after)):
        ax.text(i-.19, a+.01, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(i+.19, b+.01, f"{b:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    save(fig, "dsh_G2_SGM第三阶段前后.png")

    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    original_points = source_data(DSH_SOURCE, "points", function="fig_all_methods_probes")
    points = []
    for name, xp, yp, family, *_ in original_points:
        label = name.replace("禁忌", "贪心＋禁忌") if name.startswith("禁忌") else name
        color = blue if family == "ours" and xp > 1 else orange if family == "learn" else gray
        points.append((label, xp, yp, color, "D" if family == "learn" else "o"))
    points.insert(2, ("纯禁忌32", 32, PURE["after_W"]*100, light, "o"))
    for name, xp, yp, color, marker in points:
        ax.scatter(xp, yp, s=70, color=color, marker=marker, zorder=3)
        offset = (5, -12) if name == "纯禁忌32" else (5, 5)
        ax.annotate(name, (xp, yp), xytext=offset, textcoords="offset points", fontsize=8, color=color)
    original_unlabeled = source_data(DSH_SOURCE, "unlabeled", function="fig_all_methods_probes")
    original_goal = [original_unlabeled[i] for i in (1, 3, 4)]
    anneal_goal = [original_unlabeled[i] for i in (0, 2)] + [next((x, y) for label, x, y, *_ in original_points if label == "退火 3200")]
    for line, label, marker in ((original_goal, "遗传：原文供需差目标", "o"),
                                (anneal_goal, "退火：原文供需差目标", "s")):
        ax.plot([p[0] for p in line], [p[1] for p in line], linestyle=":", color=gray,
                marker=marker, markerfacecolor="white", markeredgecolor=gray, label=label)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(.6, 7000)
    ax.set_ylim(87.5, 96.3)
    ax.set_xlabel("真实评价次数（对数轴；不搜索的方法放在1次附近示意）")
    ax.set_ylabel("需求满足率（%）")
    ax.axhline(92.5, color="#888888", linestyle=":")
    ax.grid(alpha=.15)
    fig.tight_layout()
    save(fig, "dsh_G5_评价次数与满足率.png")


def cc_figures():
    blue, aqua, gray, light = "#2a78d6", "#1baf7a", "#b8b6ae", "#dcdad3"
    source_colors = {"BLUE": blue, "AQUA": aqua, "GRAY": gray, "LIGHT": light}
    original_rows = source_data(CC_SOURCE, "rows", first_label="改进式学习（SAC，32次评价）", symbols=source_colors)
    rows = [(name.replace("禁忌搜索", "贪心＋禁忌搜索"), sg, w, color)
            for name, sg, w, color in original_rows]
    rows.append(("纯禁忌32次", PURE["after_SGM"], PURE["after_W"]*100, blue))
    rows.sort(key=lambda row: row[1], reverse=True)
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    bars(a, [r[0] for r in rows], [r[1] for r in rows], [r[3] for r in rows],
         lambda v:f"{v:.3f}", (.88, 1.02))
    bars(b, [r[0] for r in rows], [r[2] for r in rows], [r[3] for r in rows],
         lambda v:f"{v:.1f}%", (85, 98))
    b.tick_params(axis="y", left=False, labelleft=False)
    a.set_title("供需匹配程度 SGM")
    b.set_title("需求满足率")
    fig.suptitle("43.2 Mbps/架，确认88景，SCA之后")
    fig.tight_layout()
    save(fig, "cc_v2_11_高需求全方法.png")

    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8))
    series = source_data(CC_SOURCE, "series", first_label="禁忌搜索（本研究）", symbols=source_colors)
    for name, xs, sg, w, color, style, marker in series:
        name = name.replace("禁忌搜索", "贪心＋禁忌搜索")
        a.plot(xs, sg, linestyle=style, marker=marker, color=color, label=name)
        b.plot(xs, w, linestyle=style, marker=marker, color=color)
    a.scatter([32], [PURE["after_SGM"]], color=blue, marker="^", s=90, label="纯禁忌32次")
    b.scatter([32], [PURE["after_W"]*100], color=blue, marker="^", s=90)
    a.scatter([32], [.990], color=aqua, marker="*", s=180, label="改进式学习32次")
    b.scatter([32], [94.9], color=aqua, marker="*", s=180)
    for ax, level in ((a, .973), (b, 92.5)):
        ax.axhline(level, color="#52514e", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_xticks([32, 96, 320, 3200], ["32", "96", "320", "3200"])
        ax.set_xlabel("每场景真实评价次数")
        ax.grid(alpha=.14)
    a.set_title("SGM（SCA后）")
    b.set_title("需求满足率（%，SCA后）")
    fig.legend(loc="lower center", ncol=4, bbox_to_anchor=(.5, -.04), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, 1))
    save(fig, "cc_v2_13_搜索效率.png")


def codex_figures():
    teal, navy, gray, gold = "#40999A", "#264B63", "#93A9B7", "#C4A052"
    same_machine = {row["method"]: row for row in SUMMARY["rows"]
                    if row["set"] == "conf_d432_n150" and row["seed"] == 51001}
    fields = [name for name in next(iter(OLD.values())) if name not in ("verify_seconds", "timing_source")]
    fields += ["verify_seconds", "timing_source"]
    with (OUT / "codex_第二阶段_确认88景.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key in ("L2", "G1", "PTS32", "GTS32", "S1", "GTS96", "S96", "GTS320", "S2", "L1", "L3"):
            if key in same_machine:
                source = same_machine[key]
                row = {name: source.get(name, "") for name in fields}
                row["method"] = key
                row["set"] = "conf_d432_n150"
                row["scene_seconds"] = ""
                row["model_loading_seconds"] = "0"
                row["probes"] = str(int(source["probes_mean"]))
                row["timing_source"] = "same_machine_pure_tabu_20261021"
            else:
                row = dict(OLD[key], timing_source="historical_s2fix_20261018")
            writer.writerow(row)

    methods = ["L2", "G1", "PTS32", "S1", "S2", "L1"]
    labels = ["直接生成SAC", "按需求贪心", "纯禁忌32次", "贪心＋禁忌32次", "贪心＋禁忌320次", "学习改进32次"]
    colors = ["#C27A79", gray, teal, teal, teal, "#1B827E"]
    def metric(m, key): return float(PURE[key] if m == "PTS32" else OLD[m][key])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    for ax, key, title in zip(axes, ("after_W", "after_aircraft_worst10"),
                              ("平均需求满足率", "最差一成飞机的满足率")):
        bars(ax, labels, [metric(m, key)*100 for m in methods], colors,
             lambda v:f"{v:.1f}%", (0, 110))
        ax.set_title(title)
    axes[1].set_yticklabels([])
    fig.tight_layout()
    save(fig, "codex_02_第二阶段服务效果.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    for m, label, color, marker in [("PTS32", "纯禁忌32次", teal, "^"),
                                     ("GTS32", "贪心＋禁忌32次", gray, "o"),
                                     ("GTS96", "贪心＋禁忌96次", navy, "o"),
                                     ("GTS320", "贪心＋禁忌320次", gold, "o"),
                                     ("L1", "学习改进32次（历史）", teal, "D")]:
        row = OLD[m] if m == "L1" else same_machine[m]
        x, y = float(row["search_seconds"]), float(row["after_W"])*100
        ax.scatter(x, y, color=color, marker=marker, s=90,
                   facecolor="white" if m == "L1" else color, linewidth=2)
        if m in ("L1", "GTS32"):
            target = (1.05, 95.55) if m == "L1" else (1.2, 94.12)
            ax.annotate(label, (x, y), xytext=target, textcoords="data",
                        arrowprops=dict(arrowstyle="-", color=color, linewidth=.8),
                        ha="left", color=color, fontsize=9)
        else:
            offset = {"GTS96": (8, 10), "GTS320": (-8, 9)}.get(m, (5, 5))
            ax.annotate(label, (x, y), xytext=offset, textcoords="offset points",
                        ha="right" if m == "GTS320" else "left", color=color, fontsize=9)
    ax.set_xlim(0, 10)
    ax.set_ylim(89, 96)
    ax.set_xlabel("第二阶段局部改进耗时（秒，不含共同功率细调）")
    ax.set_ylabel("最终需求满足率（%）")
    ax.grid(alpha=.16)
    fig.text(.5, .01, "搜索方法为本次同机重评；空心I0为历史计时。严格速度比较见第二阶段报告25。",
             ha="center", fontsize=9, color="#52514e")
    fig.tight_layout(rect=(0, .05, 1, 1))
    save(fig, "codex_03_学习与搜索耗时.png")


if __name__ == "__main__":
    dsh_figures()
    cc_figures()
    codex_figures()
