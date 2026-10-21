"""Build one publication summary, figures, and report from the saved evaluation."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sync_markdown import name_results, rel


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/experiments/s2_pure_tabu_20261021"
EVAL = OUT / "eval"
FIG = OUT / "figures"
CONF = "conf_d432_n150"
METHODS = ["Random0", "PTS32", "PTS96", "PTS320", "G1", "GTS32", "GTS96", "GTS320"]
LABELS = {"Random0": "随机合法初解", "G1": "容量感知贪心"}
LABELS.update({f"PTS{b}": f"纯禁忌搜索{b}次" for b in (32, 96, 320)})
LABELS.update({f"GTS{b}": f"贪心＋禁忌搜索{b}次" for b in (32, 96, 320)})
SET_LABELS = {CONF: "确认88景 · 43.2 Mbps · 150束预算", "dev_d432_n150": "开发20景 · 43.2 Mbps",
              "dev_d216_n150": "开发20景 · 21.6 Mbps", "dev_d864_n150": "开发20景 · 86.4 Mbps",
              "dev_d432_n200": "开发20景 · 43.2 Mbps · 200束预算"}


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def percent(x):
    return f"{x*100:.1f}%"


def interval(item, scale=1, digits=3):
    lo, hi = item["ci95"]
    return f'{item["mean"]*scale:+.{digits}f} [{lo*scale:+.{digits}f}, {hi*scale:+.{digits}f}]'


def find_group(data, dataset, method, seed=51001):
    return next(r for r in data["groups"] if (r["set"], r["method"], r["seed"]) == (dataset, method, seed))


def presentation(data):
    rows = [dict(find_group(data, CONF, method), label=LABELS[method]) for method in METHODS]
    by_method = {r["method"]: r for r in rows}
    p, g = by_method["PTS320"], by_method["GTS320"]
    gap = 100*(g["after_W"] - p["after_W"])
    direction = f"低{gap:.1f}个百分点" if gap >= 0 else f"高{-gap:.1f}个百分点"
    conclusion = (f'纯禁忌搜索320次的需求满足率为{percent(p["after_W"])}，'
                  f'贪心＋禁忌搜索320次为{percent(g["after_W"])}，前者{direction}；'
                  f'SGM分别为{p["after_SGM"]:.3f}和{g["after_SGM"]:.3f}。')
    payload = dict(source="codex", date="2026-10-21", scope=SET_LABELS[CONF], scene_count=88,
                   stage="SCA后", conclusion=conclusion, rows=rows,
                   algorithm_definition={"pure": "随机合法初解＋相同禁忌搜索；不经贪心组/槽位置择优",
                                         "hybrid": "容量感知贪心初解＋相同禁忌搜索"},
                   timing="同一20627，4路场景并行、每景同核单线程；初解＋搜索＋核验＋SCA，不含第一阶段和数据文件读写",
                   data_source=str(EVAL / "analysis.json"), report=str(ROOT / "docs/第二阶段/实验报告/25_纯禁忌搜索与贪心加禁忌搜索配对评价.md"))
    (OUT / "presentation_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def figures(data, include_pressure=True):
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 11, "axes.titlesize": 12,
        "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white",
        "figure.facecolor": "white", "svg.fonttype": "path", "pdf.fonttype": 42})
    groups = {r["method"]: r for r in data["groups"] if r["set"] == CONF and r["seed"] == 51001}
    families = [("PTS", "Random0", "纯禁忌搜索", "#0072B2", "o", "-"),
                ("GTS", "G1", "贪心＋禁忌搜索", "#D55E00", "s", "--")]
    budgets = [0, 32, 96, 320]

    def save(fig, name):
        fig.savefig(FIG / f"{name}.png", dpi=300)
        fig.savefig(FIG / f"{name}.svg")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.2), layout="constrained")
    for ax, (field, title, scale) in zip(axes.flat, [("before_SGM", "第二阶段输出：供需匹配", 1),
            ("before_W", "第二阶段输出：需求满足率", 100), ("after_SGM", "SCA后：供需匹配", 1), ("after_W", "SCA后：需求满足率", 100)]):
        for prefix, initial, label, color, marker, style in families:
            values = [groups[initial][field]] + [groups[f"{prefix}{b}"][field] for b in budgets[1:]]
            ax.plot(budgets, np.array(values)*scale, label=label, color=color, marker=marker, linestyle=style, linewidth=2)
        ax.set_title(title); ax.set_xlabel("候选真实评价上限（次；0为初解）")
        ax.set_ylabel("SGM" if scale == 1 else "满足率（%）")
        ax.set_xticks(budgets); ax.grid(alpha=.18)
    axes[0, 0].legend(frameon=False, fontsize=10)
    fig.suptitle("相同搜索规则，比较不同初解｜43.2 Mbps，原确认88景均值", fontsize=14)
    save(fig, "F1_搜索预算与质量")

    raw = list(csv.DictReader((EVAL / CONF / "results.csv").open(encoding="utf-8")))
    by_method = {m: {r["scene_id"]: r for r in raw if r["method"] == m} for m in METHODS}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), layout="constrained")
    for ax, budget in zip(axes, (32, 96, 320)):
        lhs, rhs = by_method[f"PTS{budget}"], by_method[f"GTS{budget}"]
        ids = sorted(lhs)
        x = [100*float(rhs[k]["after_W"]) for k in ids]
        y = [100*float(lhs[k]["after_W"]) for k in ids]
        ax.scatter(x, y, color="#0072B2", alpha=.65, s=22, edgecolors="white", linewidth=.35)
        ax.plot([0, 100], [0, 100], color="#777777", linestyle="--", linewidth=1)
        ax.set(xlim=(0, 102), ylim=(0, 102), title=f"{budget}次评价", xlabel="贪心＋禁忌搜索满足率（%）", ylabel="纯禁忌搜索满足率（%）")
        ax.set_aspect("equal"); ax.grid(alpha=.15)
    fig.suptitle("每个点是一对相同场景｜SCA后，点在虚线下表示纯禁忌较低", fontsize=14)
    save(fig, "F2_逐场景配对满足率")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), layout="constrained")
    for ax, field, scale, ylabel in [(axes[0], "after_SGM", 1, "SCA后SGM"), (axes[1], "after_W", 100, "SCA后满足率（%）")]:
        for prefix, initial, label, color, marker, style in families:
            values = [groups[initial]] + [groups[f"{prefix}{b}"] for b in budgets[1:]]
            ax.plot([v["total_seconds"] for v in values], [v[field]*scale for v in values], color=color, marker=marker, linestyle=style, label=label)
            for b, v in zip(budgets, values):
                ax.annotate(str(b), (v["total_seconds"], v[field]*scale), xytext=(4, 6 if prefix == "GTS" else -13), textcoords="offset points", fontsize=9, color=color)
        ax.set_xlabel("初解＋搜索＋核验＋SCA平均总时间（秒）"); ax.set_ylabel(ylabel)
        ax.margins(x=.10, y=.12)
        ax.grid(alpha=.18)
    axes[0].legend(frameon=False, fontsize=10)
    fig.suptitle("质量与完整计算时间｜数字为探测上限，4路场景并行、每景单线程", fontsize=14)
    save(fig, "F3_质量与总耗时")

    if not include_pressure:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), layout="constrained")
    demands = [21.6, 43.2, 86.4]
    sets = ["dev_d216_n150", "dev_d432_n150", "dev_d864_n150"]
    for method, color, marker, style in [("Random0", "#777777", "x", ":"), ("G1", "#009E73", "^", "-."), ("PTS320", "#0072B2", "o", "-"), ("GTS320", "#D55E00", "s", "--")]:
        rows = [find_group(data, dataset, method) for dataset in sets]
        axes[0].plot(demands, [r["after_W"]*100 for r in rows], color=color, marker=marker, linestyle=style, label=LABELS[method])
        axes[1].plot(demands, [r["after_SGM"] for r in rows], color=color, marker=marker, linestyle=style, label=LABELS[method])
    for ax in axes:
        ax.set_xticks(demands); ax.set_xlabel("单机请求（Mbps）"); ax.grid(alpha=.18)
    axes[0].set_ylabel("SCA后满足率（%）"); axes[1].set_ylabel("SCA后SGM")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("需求增加后，初始化的影响如何变化｜原开发20景，150束预算", fontsize=14)
    save(fig, "F4_需求压力对照")


def report(data, public):
    r = {row["method"]: row for row in public["rows"]}
    pairs = [next(p for p in data["paired"] if p["set"] == CONF and p["lhs"] == f"PTS{b}" and p["rhs"] == f"GTS{b}") for b in (32, 96, 320)]
    main = table(["方法", "SCA前SGM", "SCA前满足率", "SCA后SGM", "SCA后满足率", "SCA后最差10%飞机", "总时间"],
        [[LABELS[m], f'{r[m]["before_SGM"]:.3f}', percent(r[m]["before_W"]), f'{r[m]["after_SGM"]:.3f}', percent(r[m]["after_W"]), percent(r[m]["after_aircraft_worst10"]), f'{r[m]["total_seconds"]:.2f}秒'] for m in METHODS])
    timing = table(["方法", "初解", "搜索", "核验", "SCA", "总计均值", "总计中位数", "总计95分位"],
        [[LABELS[m]] + [f'{r[m][key]:.2f}' for key in ("initial_seconds", "search_seconds", "verify_seconds", "sca_seconds", "total_seconds", "total_seconds_median", "total_seconds_p95")] for m in METHODS])
    delta = table(["预算", "SCA后满足率差（百分点）", "SCA后SGM差", "纯禁忌更高/持平/更低场景数"],
        [[b, interval(p["after_W"], 100, 1), interval(p["after_SGM"]), f'{p["after_W"]["better"]}/{p["after_W"]["same"]}/{p["after_W"]["worse"]}'] for b, p in zip((32, 96, 320), pairs)])
    resource = table(["方法", "全束平均满足率（SCA前 / 后）", "总功率（W，SCA后）", "总槽数", "实际探测均值"],
        [[LABELS[m], f'{percent(r[m]["before_SI"])} / {percent(r[m]["after_SI"])}', f'{r[m]["after_total_power_w"]:.0f}', f'{r[m]["after_total_slots"]:.1f}', f'{r[m]["probes_mean"]:.1f}'] for m in METHODS])
    stress = []
    for dataset in SET_LABELS:
        p, g = find_group(data, dataset, "PTS320"), find_group(data, dataset, "GTS320")
        stress.append([SET_LABELS[dataset], f'{p["after_SGM"]:.3f} / {percent(p["after_W"])}', f'{g["after_SGM"]:.3f} / {percent(g["after_W"])}', f'{100*(p["after_W"]-g["after_W"]):+.1f}'])
    stress_table = table(["集合", "纯禁忌320：SGM / 满足率", "贪心＋禁忌320：SGM / 满足率", "满足率差（百分点）"], stress)
    sensitivity = table(["初解种子", "随机初解", "纯禁忌32", "纯禁忌96", "纯禁忌320"],
        [[seed] + [f'{find_group(data, "dev_d432_n150", m, seed)["after_SGM"]:.3f} / {percent(find_group(data, "dev_d432_n150", m, seed)["after_W"])}' for m in ("Random0", "PTS32", "PTS96", "PTS320")] for seed in (51001, 51002, 51003)])
    pure_gain = 100*(r["PTS320"]["after_W"]-r["Random0"]["after_W"])
    hybrid_gain = 100*(r["GTS320"]["after_W"]-r["G1"]["after_W"])
    init_gap = 100*(r["G1"]["after_W"]-r["Random0"]["after_W"])
    time_gap = r["PTS320"]["total_seconds"]-r["GTS320"]["total_seconds"]
    init_time_gap = r["G1"]["initial_seconds"]-r["Random0"]["initial_seconds"]
    overlap_pct = 100*(r["Random0"]["before_overlap_slots"]/r["G1"]["before_overlap_slots"]-1)
    timing_sentence = f'纯禁忌320次平均总时间比组合方法{"多" if time_gap>=0 else "少"}{abs(time_gap):.2f}秒。'
    figures_rel = "../../../outputs/experiments/s2_pure_tabu_20261021/figures/"
    text = f'''# 纯禁忌搜索与贪心＋禁忌搜索配对评价
来源：codex

日期：2026-10-21。场景：原确认88景为主，另有80景压力与预算比较；主档开发集补两种随机初解，共1504个场景—方法结果。

**{public["conclusion"]}** 因此，项目原有32、96、320次禁忌成绩必须明确标为“贪心＋禁忌搜索”。纯禁忌的成绩单列，不能沿用组合方法的数字。

纯禁忌从随机合法分配开始，用320次搜索把满足率提高{pure_gain:.1f}个百分点；组合方法在贪心基础上提高{hybrid_gain:.1f}个百分点。初解本身的满足率相差{init_gap:.1f}个百分点，说明应把初始化质量与后续搜索的贡献一起看。{timing_sentence}

## 1. 两种方法到底差在哪里

**纯禁忌搜索**：把波束随机排列、循环分入8个频率组，再按随机顺序连续排槽。不依据候选容量或干扰逐个挑最优位置。槽数仍按单束容量估算，参考功率按需求比例分配；某组超过100槽时按比例缩减到合法范围。

**贪心＋禁忌搜索**：先运行原容量感知贪心，选择频率组和槽位置，再运行同一禁忌搜索。历史代码中的S1、S96、S2分别对应32、96、320次组合方法。

两者共用六类邻域、候选优先级、长度8的完整布局禁忌表、每轮最多4次真实评价、保存最好解和相对初解的基本服务保护规则。预算32/96/320对应最多8/24/80轮。搜索过程中可以暂时变差，输出保留的最好合格方案。全部搜索结束后才接SCA连续功率优化。

复用原S2-FIX实际场景与链路模型v2；主档每架飞机43.2 Mbps、最多150束、8组各100槽、每束最多10槽、总功率3000 W、每束50 W。两种初解使用同一批飞机、波束中心、宽度和归属。详细设计见[方案33](../方案/33_纯禁忌搜索与贪心加禁忌搜索配对评价.md)。

## 2. 主结果：原确认88景

下面每一行都来自本次同机重评，主初解种子51001。SGM和全束SI按波束平均；需求满足率按业务量加权。最差10%飞机按束内均分估计，其具体计算沿用原S2-FIX。全部场景等权平均。

{main}

![两种初解下的预算与SGM、需求满足率曲线]({figures_rel}F1_搜索预算与质量.png)

图1：0次代表不搜索的初解，32、96、320代表候选真实评价上限；没有平滑或插值出的额外观测。上排为SCA前，下排为SCA后。

同场景“纯禁忌减贪心＋禁忌”的差值如下。方括号为按日期、再按日期内场景重采样的95%区间；正值表示纯禁忌更高，负值表示组合方法更高。

{delta}

![每个真实场景的两种初始化满足率比较]({figures_rel}F2_逐场景配对满足率.png)

图2：88个点分别对应88个场景。虚线为两种方法相同，点在线下表示纯禁忌较低；三个预算采用相同坐标范围。

## 3. 性能：把初解和SCA都计入

在远端20627同一软件环境下，固定4个不同物理CPU核并行处理场景；同一场景内各方法在同一核上单线程串行运行，方法执行顺序随场景循环轮换。全部正式计时采用这一共同并发条件，前期单进程试跑单列为诊断。下面单位均为秒。总时间为初解、搜索、核验及SCA四段实测之和，不含第一阶段布局生成及输入输出文件读写；没有把其他平台的历史耗时拼进来。

{timing}

![同机计算总时间与结果质量]({figures_rel}F3_质量与总耗时.png)

图3：横轴使用完整的第二、三阶段计算时间，点旁数字是探测上限。随机初解或贪心初解都真实计时，0次也包含SCA。

贪心构造本身平均只需{r["G1"]["initial_seconds"]:.2f}秒，比随机构造多约{init_time_gap:.2f}秒。纯禁忌320次的总时间为{r["PTS320"]["total_seconds"]:.1f}秒，组合方法为{r["GTS320"]["total_seconds"]:.1f}秒：省去贪心的这点构造时间，没有变成整个流程的速度收益。

原始记录也解释了初解为何重要：随机初解平均使用{r["Random0"]["before_total_slots"]:.0f}槽，贪心使用{r["G1"]["before_total_slots"]:.0f}槽，用量接近；随机初解的同极化跨组重叠槽对计数约{r["Random0"]["before_overlap_slots"]:.0f}，贪心约{r["G1"]["before_overlap_slots"]:.0f}，前者多约{overlap_pct:.0f}%。现有邻域主要做交换、边界调整和小幅移动，320次搜索仍没补回频率安排的起点差距。这里的重叠计数是几何机会计数，不是干扰功率的百分比。

## 4. 需求压力、波束预算和初解种子

先固定320次搜索比较各集合，下面均为SCA后结果。150束和200束是上限，实际非空波束数随场景变化。

{stress_table}

低需求和200束预算下，纯禁忌320次的满足率也低于组合方法。高需求86.4 Mbps档，两者满足率为67.1%和68.0%，差距缩小到0.9个百分点，配对差的95%区间包含0；SGM分别为0.799和0.798，接近持平。主档的优势不能直接推广为每档、每项指标都同样明显。

![不同需求下两种初始化的搜索效果]({figures_rel}F4_需求压力对照.png)

图4：三档请求使用原开发20景对应的布局。不同需求档的布局也承接原S2-FIX设定，比较每一档内的方法差，不把它当作仅改变业务量的孤立物理消融。

86.4 Mbps档保留了原输入中一景第一阶段负载超限的场景，详见末尾限制；没有删除该景或重新生成布局。主档确认88景均满足原每束26架的负载上限。

三种初解种子在主档开发20景的结果如下，单元格为“SGM / 需求满足率”，均为SCA后。三次起点的成绩分别保留，不挑最好种子。

{sensitivity}

纯禁忌320次的三种起点得到92.3%—92.9%的满足率，均低于同批开发场景上的贪心94.4%和贪心＋禁忌320次96.9%。这两种贪心对照只使用主种子记录；额外两种种子仅改变随机初解。

## 5. 资源与正确性

下表的全束平均满足率（SI）对每个波束等权；前表的需求满足率按业务量加权。飞机多的波束在后者中权重更大，两项指标分别列出。

{resource}

全部1504个结果通过槽边界、组内不重叠、单束与总功率约束检查；失败数量和原数组复算结果见[核对结果](../../../outputs/experiments/s2_pure_tabu_20261021/eval/verification.json)。保存的逐束容量重新计算了满足率、SGM、最差10%、功率和用槽，与结果表一致。

新评价器在最小与最大飞机数的真实开发场景上，与原贪心＋禁忌32次的输出逐数组相同。纯初始化路径在禁止调用贪心函数时仍能完成，固定种子可复现；[冻结源码复核日志](../../../outputs/experiments/s2_pure_tabu_20261021/check.log)保留了这次检查。

## 6. 结果如何引用

报告、PPT、Word、网页和问答中，写完整名称“纯禁忌搜索（随机合法初解）”或“贪心＋禁忌搜索（容量感知贪心初解）”。32、96、320是搜索探测上限，不包括构造初解的计算；比较总耗时必须同时包括初解。

历史S2-FIX和文献对照中的94.5%—94.9%来自贪心＋禁忌搜索。原表保留其日期和平台；本次两种初始化的直接比较统一引用本报告。教学小例的80%、100%等数字继续标为演示值。

本次对照限定于现有邻域、保护规则和搜索预算，不代表所有可能的纯禁忌实现。原确认集已被先前研究使用；随机起点的三种子检查只覆盖一天的20个开发场景。高需求档的2024-01-12 12:37场景共有1678架、150束，其中86束装14架，超过原13架上限；本次原样保留，因此高需求结果包含这一历史布局缺口。SCA保存了最终分配和资源可行性，未另存每次求解器收敛状态。

## 数据与图

本次算法、评价、数据核对、统计与四幅图均由codex生成。场景及原物理/搜索代码来自S2-FIX冻结版本`aa9186a`，原S2-FIX报告由cc撰写；本次不更改原实验成绩。

| 内容 | 入口 |
| --- | --- |
| 逐场景结果、分段耗时、逐束数组、候选轨迹 | [eval目录](../../../outputs/experiments/s2_pure_tabu_20261021/eval/) |
| 168景实际输入 | [data目录](../../../outputs/experiments/s2_pure_tabu_20261021/data/) |
| 全集合与全部种子汇总 | [summary.csv](../../../outputs/experiments/s2_pure_tabu_20261021/eval/summary.csv) |
| 配对差与区间 | [analysis.json](../../../outputs/experiments/s2_pure_tabu_20261021/eval/analysis.json) |
| 按日期分别汇总 | [by_date.csv](../../../outputs/experiments/s2_pure_tabu_20261021/eval/by_date.csv) |
| 环境与运行条件 | [run_manifest.json](../../../outputs/experiments/s2_pure_tabu_20261021/eval/run_manifest.json) |
| 报告、PPT、网页共用摘要 | [presentation_summary.json](../../../outputs/experiments/s2_pure_tabu_20261021/presentation_summary.json) |
| 发布与绘图脚本 | [publish_results.py](../../../scripts/pure_tabu_docs/publish_results.py) |
| 文档同步清单 | [文档同步清单](../../../outputs/experiments/s2_pure_tabu_20261021/document_sync.md) |

运行清单记录CPU、解释器、NumPy/SciPy版本与线程数。正式评价算法冻结为`3aa1928`，四核调度为`12868d5`，后处理脚本保存在本次源码归档；原件在本地与指定远端各保存一套。没有新增模型训练。
'''
    path = ROOT / "docs/第二阶段/实验报告/25_纯禁忌搜索与贪心加禁忌搜索配对评价.md"
    path.write_text(text, encoding="utf-8")


def navigation(public):
    import re
    report_path = Path(public["report"])
    question = ROOT / "Q&A/评估与指标/纯禁忌搜索与贪心加禁忌搜索的区别和实测结果.md"
    r = {row["method"]: row for row in public["rows"]}
    rows = [[LABELS[m], f'{r[m]["after_SGM"]:.3f}', percent(r[m]["after_W"]), f'{r[m]["total_seconds"]:.2f}秒']
            for m in ("PTS32", "GTS32", "PTS96", "GTS96", "PTS320", "GTS320")]
    content = f'''# 纯禁忌搜索与贪心＋禁忌搜索的区别和实测结果
来源：codex

问题：评估纯禁忌搜索的指标和性能，并在项目报告、PPT、网页等材料中明确区分两种算法。

回答日期：2026-10-21。

**{public["conclusion"]}** 项目原来写的94.5%—94.9%属于贪心＋禁忌搜索，不能称作纯禁忌的结果。

纯禁忌先随机生成合法完整分配，再按禁忌规则搜索。贪心＋禁忌先按容量和干扰构造一个较好的初解，再运行相同搜索。两者都需要初解；本次去掉的是贪心的候选择优过程，公共的槽数估算和需求比例功率仍保留。

同一远端、每景同核单线程、4路场景并行，原确认88景、每机43.2 Mbps；下面SGM与满足率均为SCA后，时间含初解、搜索、核验与SCA。

{table(["方法", "SGM", "需求满足率", "平均总时间"], rows)}

32、96、320是候选真实评价上限；两种搜索共用邻域、禁忌表和输出规则。三种初解种子在开发20景分别报告，不挑最好种子。完整SCA前后指标、压力档、资源与计时分位数见[报告25]({rel(report_path, question)})，相关报告/PPT/Word/网页均从[同一摘要]({rel(OUT / 'presentation_summary.json', question)})同步。

## 追问：主对比表为什么仍只有贪心＋禁忌？我要纯禁忌32次

此前只给旧行补了初始化名称，并在附页加入纯禁忌对照，漏了主表。本次把“纯禁忌搜索32次（随机初解）”直接补入主要算法对比：需求满足率{percent(r['PTS32']['after_W'])}、最差10%飞机{percent(r['PTS32']['after_aircraft_worst10'])}、SGM {r['PTS32']['after_SGM']:.3f}，均为SCA后。原94.5%的行仍是贪心＋禁忌32次，两者分别展示。

## 追问：格式与其他行一致，并同步dsh、cc和中期材料

2026-10-22：先前的绿色是额外添加的强调，现已去掉。纯禁忌行沿用原表的字体、字号、字重、边框及交替底色。dsh、cc、codex三套PPT、Word、网页及对应PDF同步主比较，中期材料在第二阶段正文单列本次43.2 Mbps、确认88景的补充实验；相关展示图和公开CSV也加入纯禁忌32次，保留历史实验的条件和原成绩。具体入口见[文档同步清单]({rel(OUT / 'document_sync.md', question)})。
'''
    question.write_text(content, encoding="utf-8")
    names = ["EXPERIMENT_STATUS.md", "docs/00_目录.md", "docs/第二阶段/00_目录.md", "docs/Claude维护/00_说明.md", "Q&A/问答索引.md", "docs/Claude维护/导师汇报/codex版/数据与图/来源说明.md"]
    for name in names:
        path = ROOT / name
        original = path.read_text(encoding="utf-8-sig")
        start, end = "<!-- PURE_TABU_NAV_START -->", "<!-- PURE_TABU_NAV_END -->"
        text = re.sub(re.escape(start) + r".*?" + re.escape(end) + r"\n*", "", original, flags=re.S)
        if name == "EXPERIMENT_STATUS.md":
            for marker in ("S2_FIX", "LITCMP"):
                pattern = rf"(<!-- {marker}_START -->)(.*?)(<!-- {marker}_END -->)"
                text = re.sub(pattern, lambda match: match[1] + name_results(match[2]) + match[3], text, flags=re.S)
        if name == "Q&A/问答索引.md":
            text = name_results(text)
        block = (f'{start}\n**2026-10-21 纯禁忌与贪心＋禁忌已完成同机比较（来源：codex）。** '
                 f'{public["conclusion"]} [完整报告25]({rel(report_path, path)})，'
                 f'[方案33]({rel(ROOT / "docs/第二阶段/方案/33_纯禁忌搜索与贪心加禁忌搜索配对评价.md", path)})，'
                 f'[算法区别与结果问答]({rel(question, path)})。'
                 '原S1/S96/S2统一标明贪心＋禁忌搜索；纯禁忌单列，报告、PPT、Word与网页使用同一实测摘要。\n'
                 f'{end}\n\n')
        source = re.search(r"^来源：[^\n]*\n", text, flags=re.M) if name.endswith("来源说明.md") else None
        pos = source.end() if source else text.find("\n") + 1
        text = text[:pos] + "\n" + block + text[pos:].lstrip("\n")
        backup = OUT / "document_backups" / name
        if not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_bytes(path.read_bytes())
        path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--presentation-only", action="store_true")
    args = parser.parse_args()
    data = json.loads((EVAL / "analysis.json").read_text(encoding="utf-8"))
    for method in METHODS:
        if find_group(data, CONF, method)["n"] != 88:
            raise ValueError("The confirmation set is incomplete")
    public = presentation(data)
    if args.presentation_only:
        figures(data, include_pressure=False)
    else:
        verified = json.loads((EVAL / "verification.json").read_text(encoding="utf-8"))
        if data["rows"] != 1504 or data["constraints_failed"] or verified["rows"] != 1504 or verified["failures"]:
            raise ValueError("Evaluation or verification is incomplete")
        figures(data)
        report(data, public)
        navigation(public)
    print(json.dumps({"rows": data["rows"], "presentation": str(OUT / "presentation_summary.json")}, ensure_ascii=True))


if __name__ == "__main__":
    main()
