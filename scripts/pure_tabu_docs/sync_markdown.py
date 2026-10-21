"""Synchronize initialization names and measured results in maintained Markdown."""
import argparse
import json
import os
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/experiments/s2_pure_tabu_20261021"
REPORT = ROOT / "docs/第二阶段/实验报告/25_纯禁忌搜索与贪心加禁忌搜索配对评价.md"
BEGIN = "<!-- PURE_TABU_COMPARISON_20261021_START -->"
END = "<!-- PURE_TABU_COMPARISON_20261021_END -->"
MAIN_ROW = "<!-- PURE_TABU_MAIN_ROW -->"
MAIN_BEGIN = "<!-- PURE_TABU_MAIN_START -->"
MAIN_END = "<!-- PURE_TABU_MAIN_END -->"

RESULT_FILES = [
    "docs/Claude维护/S2-FIX第二阶段需求压力/05_第二阶段方案_需求压力下的频率槽学习改进与三阶段连通.md",
    "docs/Claude维护/S2-FIX第二阶段需求压力/06_S2-FIX给Codex的实现说明.md",
    "docs/Claude维护/S2-FIX第二阶段需求压力/12_S2-FIX第二阶段需求压力实验报告.md",
    "docs/Claude维护/文献对照/08_三阶段文献对照方法清单.md",
    "docs/Claude维护/文献对照/09_文献对照实验方案.md",
    "docs/Claude维护/文献对照/11_文献对照实验报告.md",
    "docs/Claude维护/导师汇报/cc版/14_近期工作导师汇报材料.md",
    "docs/Claude维护/导师汇报/cc版/15_导师汇报讲解提纲.md",
    "docs/Claude维护/导师汇报/dsh版/16_近期工作导师汇报材料.md",
    "docs/Claude维护/导师汇报/dsh版/17_导师汇报讲解提纲.md",
    "docs/Claude维护/导师汇报/dsh版/口径清单.md",
    "docs/Claude维护/导师汇报/dsh版/README.md",
    "docs/Claude维护/导师汇报/codex版/01_导师汇报材料.md",
    "docs/Claude维护/导师汇报/codex版/03_讲解提纲.md",
    "docs/Claude维护/导师汇报/codex版/05_中期研究进展同步稿.md",
    "docs/Claude维护/中期报告铺垫/02_第二阶段频率功率联合分配研究现状与后续方向.md",
    "docs/Claude维护/中期报告铺垫/13_中期报告可写性判断与组织方案.md",
    "Q&A/算法与训练/T-SAC在第二阶段的定位与毕设主线.md",
    "Q&A/算法与训练/I0是什么与T-SAC的关系.md",
    "Q&A/评估与指标/S2-FIX报告里的SGM结果.md",
    "Q&A/评估与指标/评价环境是否过于宽松.md",
    "Q&A/评估与指标/中期报告能否在S2-FIX后收尾.md",
]
HISTORICAL_FILES = [
    "docs/资料/04_三阶段研究瓶颈与关键问题分析.md",
    "docs/资料/05_第二三阶段中期进展报告草稿.md",
    "docs/资料/06_中期研究进展与路线调整_导师讨论版.md",
    "docs/资料/07_HTS资源分配中期报告_正式版.md",
    "docs/Claude维护/路线与评估/01_路线评估与方向建议.md",
    "docs/Claude维护/中期报告铺垫/03_第三阶段功率精细优化与三阶段衔接研究现状与后续方向.md",
    "docs/第二阶段/实验报告/15_SAC三路线超越贪心的独立场景对照.md",
    "Q&A/算法与训练/三阶段研究瓶颈与关键问题.md",
]
TEACHING_FILES = [
    "docs/Claude维护/导师汇报/codex版/00_阅读顺序.md",
    "Q&A/算法与训练/各阶段算法逐步交互演示.md",
]
MECHANISM_FILES = ["Q&A/算法与训练/禁忌搜索为什么有效以及随机性来自哪里.md"]
BODY_LOCATIONS = {
    "docs/资料/04_三阶段研究瓶颈与关键问题分析.md": "## 4 第二阶段 频率槽与功率联合分配",
    "docs/资料/05_第二三阶段中期进展报告草稿.md": "## 三 第二阶段离散结构学习的研究进展",
    "docs/资料/06_中期研究进展与路线调整_导师讨论版.md": "## 二 对当前最好证据的判断",
    "docs/资料/07_HTS资源分配中期报告_正式版.md": "### 3.4 阶段研究结果与分析",
    "docs/Claude维护/中期报告铺垫/02_第二阶段频率功率联合分配研究现状与后续方向.md": "## 三、已完成的研究工作与主要结果",
    "docs/Claude维护/中期报告铺垫/03_第三阶段功率精细优化与三阶段衔接研究现状与后续方向.md": "### 3.2 与第二阶段连起来的已有证据",
    "docs/Claude维护/中期报告铺垫/13_中期报告可写性判断与组织方案.md": "## 3. 对照开题承诺",
    "docs/Claude维护/导师汇报/cc版/15_导师汇报讲解提纲.md": "## 第 20 页 第二阶段：高需求压力测试",
    "docs/Claude维护/导师汇报/dsh版/17_导师汇报讲解提纲.md": "### 第 15 页　方法全家福（横轴）",
    "docs/Claude维护/导师汇报/codex版/03_讲解提纲.md": "## 15｜43.2 Mbps压力下，T-SAC主线面临更强基线",
    "docs/Claude维护/S2-FIX第二阶段需求压力/12_S2-FIX第二阶段需求压力实验报告.md": "## 3. 主结果：43.2 Mbps",
    "docs/Claude维护/文献对照/11_文献对照实验报告.md": "## 4. 第二阶段：频率槽与功率",
}
HISTORICAL_CHARTS = (
    "v2_11_高需求全方法.png", "v2_13_搜索效率.png", "v2_14_为什么需要第二阶段.png", "v2_16_端到端.png",
    "G1_SGM两种协议.png", "G2_SGM第三阶段前后.png", "G5_评价次数与满足率.png", "G3_阶段差距.png", "F7_端到端.png",
)
CHART_NOTE = "图中“禁忌”指**贪心＋禁忌搜索（容量感知贪心初解）**；纯禁忌的实测结果见本篇开头的对照表。"


def rel(target, source):
    return os.path.relpath(target, source.parent).replace("\\", "/")


def name_results(text):
    """Only used for S2-FIX result documents, never generic tabu literature."""
    text = text.replace("容量感知贪心给可行解＋禁忌搜索局部改进", "贪心＋禁忌搜索（容量感知贪心给可行解，再局部改进）")
    text = text.replace("贪心起点上的禁忌搜索", "贪心＋禁忌搜索")
    text = text.replace("从 G1 出发的禁忌搜索", "从 G1 出发的贪心＋禁忌搜索")
    # Preserve already explicit names and Markdown link destinations.
    fragments = re.split(r"(\]\([^\n)]*\))", text)
    for i in range(0, len(fragments), 2):
        part = fragments[i].replace("贪心+禁忌搜索", "贪心＋禁忌搜索")
        part = re.sub(r"(?<!贪心＋)(?<!纯)禁忌搜索", "贪心＋禁忌搜索", part)
        part = re.sub(r"(?<!贪心＋)(?<!纯)禁忌(?=\s*(?:32|96|320|94\.5|0\.858|94\.9))", "贪心＋禁忌搜索", part)
        fragments[i] = part
    return "".join(fragments)


def block(path, category, data):
    lines = [BEGIN, "**2026-10-21 禁忌搜索初始化对照更新（本次修订来源：codex）。**", ""]
    if category == "historical":
        lines.append("本篇保留原实验时点的结果与计划；文中“尚未比较禁忌搜索”只指当时范围。后续S2-FIX已比较贪心起点上的搜索，本次进一步补上随机合法初解的纯禁忌搜索。")
    elif category in ("teaching", "mechanism"):
        lines.append("教学小例的分数用于解释搜索过程。真实实验分为**纯禁忌搜索（随机合法初解）**与**贪心＋禁忌搜索（容量感知贪心初解）**；两者使用相同搜索规则，不能把贪心起点的成绩称作纯禁忌。")
    else:
        lines.append("原有S1、S96、S2分别为**贪心＋禁忌搜索32、96、320次**，历史表和历史图中的同名/简写均按此理解。**纯禁忌搜索**从随机合法初解出发，不调用贪心的候选择优构造；槽数估算与参考功率规则共用。")
    lines.append("")
    if data:
        lines.extend([data["conclusion"], "", "同一远端、43.2 Mbps、原确认88景；下表SGM和需求满足率均为SCA后，总时间包含初解、搜索、核验和SCA，不含第一阶段布局。", "", "| 探测上限 | 纯禁忌：SGM / 满足率 / 总时间 | 贪心＋禁忌：SGM / 满足率 / 总时间 |", "| ---: | --- | --- |"])
        by_method = {row["method"]: row for row in data["rows"]}
        for budget in (32, 96, 320):
            p, g = by_method[f"PTS{budget}"], by_method[f"GTS{budget}"]
            cell = lambda r: f'{r["after_SGM"]:.3f} / {100*r["after_W"]:.1f}% / {r["total_seconds"]:.2f}秒'
            lines.append(f"| {budget} | {cell(p)} | {cell(g)} |")
        lines.append("")
        lines.append("原实验表继续保留原平台的历史数值；新旧初始化的结论与速度比较统一采用上表同机重评。")
    else:
        lines.append("本次配对评价正在进行，纯禁忌的指标尚未填写。已有94.5%—94.9%等成绩属于贪心＋禁忌搜索。")
    lines.extend(["", f"完整的SCA前后指标、压力档、种子敏感性与耗时分位数见[第二阶段报告25]({rel(REPORT, path)})；实验设计见[方案33]({rel(ROOT / 'docs/第二阶段/方案/33_纯禁忌搜索与贪心加禁忌搜索配对评价.md', path)})。", END, ""])
    return "\n".join(lines)


def main_comparison(path, text, data):
    """Put the pure 32-probe result beside the algorithms in the main narrative."""
    name = path.relative_to(ROOT).as_posix()
    prefix = "docs/Claude维护/导师汇报/"
    formats = {
        prefix + "codex版/01_导师汇报材料.md": ("| 方法 | 飞机平均W | 服务最低10% | SGM |", "compact"),
        prefix + "cc版/14_近期工作导师汇报材料.md": ("| 方法 | SGM | 满足率 | 最差 10% 飞机 | 每景评价次数 | 单景耗时 | 训练时间 |", "timing"),
    }
    r = next(row for row in data["rows"] if row["method"] == "PTS32")
    text = re.sub(r"^.*" + re.escape(MAIN_ROW) + r"\n?", "", text, flags=re.M)
    text = re.sub(re.escape(MAIN_BEGIN) + r".*?" + re.escape(MAIN_END) + r"\n*", "", text, flags=re.S)
    if name in formats:
        header, kind = formats[name]
        pattern = re.escape(header) + r"\n(?:\|[^\n]*\n)+"
        match = re.search(pattern, text)
        rows = match[0].splitlines()
        values = ["纯禁忌搜索32次（随机初解）", f'{r["after_W"]*100:.1f}%',
                  f'{r["after_aircraft_worst10"]*100:.1f}%', f'{r["after_SGM"]:.3f}']
        if kind == "timing":
            values = [values[0], values[3], values[1], values[2], "32", f'{r["total_seconds"]:.2f} 秒', "无需训练"]
        at = next(i for i, row in enumerate(rows) if "贪心＋禁忌搜索" in row)
        rows.insert(at, "| " + " | ".join(values) + " | " + MAIN_ROW)
        note = (f"{MAIN_BEGIN}\n新增纯禁忌32次行采用随机合法初解，来源为本次确认88景配对评价；"
                "表内旧贪心＋禁忌行仍是另一种方法，保留原成绩。"
                + ("纯禁忌耗时包含初解、搜索、核验和SCA；各旧行保留历史计时。" if kind == "timing" else "")
                + f"\n{MAIN_END}\n")
        text = text[:match.start()] + "\n".join(rows) + "\n\n" + note + text[match.end():].lstrip("\n")
    elif name in (prefix + "dsh版/16_近期工作导师汇报材料.md", prefix + "codex版/05_中期研究进展同步稿.md"):
        anchor = r"^\*\*学习类算法的完整对照[^\n]*\n" if "dsh版" in name else r"^当前43\.2[^\n]*\n"
        match = re.search(anchor, text, flags=re.M)
        note = (f"{MAIN_BEGIN}\n纯禁忌搜索32次（随机合法初解）：同一确认88景、SCA后，"
                f'需求满足率{r["after_W"]*100:.1f}%，最差10%飞机{r["after_aircraft_worst10"]*100:.1f}%，'
                f'SGM {r["after_SGM"]:.3f}，平均总时间{r["total_seconds"]:.2f}秒。'
                f"这是新增的纯禁忌对照，原贪心＋禁忌成绩单列。\n{MAIN_END}\n")
        text = text[:match.end()] + "\n" + note + text[match.end():].lstrip("\n")
    elif name in BODY_LOCATIONS:
        anchor = BODY_LOCATIONS[name]
        pos = text.index(anchor) + len(anchor)
        heading = anchor.split(" ", 1)[0] + "# 补充实验：纯禁忌搜索32次"
        by_method = {row["method"]: row for row in data["rows"]}
        g = by_method["GTS32"]
        intro = (f'随机合法初解再做32次禁忌搜索，SCA后满足率为{r["after_W"]*100:.1f}%，'
                 f'最差10%飞机为{r["after_aircraft_worst10"]*100:.1f}%，SGM为{r["after_SGM"]:.3f}；'
                 f'贪心＋禁忌32次的满足率为{g["after_W"]*100:.1f}%。'
                 '两种算法分别列出，初解方式是本次对照的区别。')
        lines = [MAIN_BEGIN, heading, "来源：codex（2026-10-21实验，2026-10-22同步）", "", intro]
        if "讲解提纲" not in name:
            lines.extend(["", "本次采用43.2 Mbps/机、同一确认88景及相同SCA；下面三行来自同机配对评价，均为SCA后结果。", "",
                          "| 方法 | 需求满足率 | 最差10%飞机 | SGM | 平均总耗时 |", "| --- | --- | --- | --- | --- |"])
            for method in ("G1", "PTS32", "GTS32"):
                row = by_method[method]
                lines.append(f'| {row["label"]} | {row["after_W"]*100:.1f}% | {row["after_aircraft_worst10"]*100:.1f}% | {row["after_SGM"]:.3f} | {row["total_seconds"]:.2f}秒 |')
            lines.extend(["", "总耗时包含初解、搜索、核验和SCA。此前不同需求、布局和链路设定下的表格按原实验解释。"])
        lines.extend(["", f"完整结果与三档搜索预算见[第二阶段报告25]({rel(REPORT, path)})。", MAIN_END, ""])
        text = text[:pos] + "\n\n" + "\n".join(lines) + "\n" + text[pos:].lstrip("\n")
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text(encoding="utf-8")) if args.summary else None
    changed, synchronized = [], []
    for category, names in (("results", RESULT_FILES), ("historical", HISTORICAL_FILES), ("teaching", TEACHING_FILES), ("mechanism", MECHANISM_FILES)):
        for name in names:
            path = ROOT / name
            if not path.exists():
                continue
            synchronized.append(name)
            original = path.read_text(encoding="utf-8-sig")
            text = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n*", "", original, flags=re.S)
            if category == "results":
                text = name_results(text)
                for figure in HISTORICAL_CHARTS:
                    pattern = rf"(!\[[^\n]*\]\([^\n)]*/{re.escape(figure)}\))"
                    text = re.sub(pattern + rf"(?:\n\n{re.escape(CHART_NOTE)})?", lambda m: m[1] + "\n\n" + CHART_NOTE, text)
            if data:
                text = main_comparison(path, text, data)
            # Keep the original authorship line immediately under the title.
            source = re.match(r"\A#[^\n]*\n(?:[ \t]*\n)*来源：[^\n]*\n", text)
            pos = source.end() if source else text.find("\n") + 1
            text = text[:pos] + "\n" + block(path, category, data) + "\n" + text[pos:].lstrip("\n")
            if text != original:
                backup = OUT / "document_backups" / name
                if not backup.exists():
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    backup.write_bytes(path.read_bytes())
                path.write_text(text, encoding="utf-8")
                changed.append(name)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "markdown_updated.json").write_text(json.dumps(synchronized, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"changed": len(changed), "files": changed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
