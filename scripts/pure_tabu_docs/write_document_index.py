"""List the report and synchronized deliverables with working relative links."""

import os
from pathlib import Path

from sync_markdown import BODY_LOCATIONS, HISTORICAL_FILES, MECHANISM_FILES, RESULT_FILES, TEACHING_FILES


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/experiments/s2_pure_tabu_20261021"
ADVISOR = ROOT / "docs/Claude维护/导师汇报"


def main():
    categories = {
        "方案、报告与问答": [
            ROOT / "docs/第二阶段/方案/33_纯禁忌搜索与贪心加禁忌搜索配对评价.md",
            ROOT / "docs/第二阶段/实验报告/25_纯禁忌搜索与贪心加禁忌搜索配对评价.md",
            ROOT / "Q&A/评估与指标/纯禁忌搜索与贪心加禁忌搜索的区别和实测结果.md",
            *[ROOT / name for name in RESULT_FILES + HISTORICAL_FILES + TEACHING_FILES + MECHANISM_FILES],
        ],
        "目录与状态入口": [ROOT / name for name in (
            "EXPERIMENT_STATUS.md", "docs/00_目录.md", "docs/第二阶段/00_目录.md",
            "docs/Claude维护/00_说明.md", "Q&A/问答索引.md",
            "docs/Claude维护/导师汇报/codex版/数据与图/来源说明.md",
        )],
        "PPT": sorted(ADVISOR.glob("*/*.pptx")),
        "Word": sorted(ADVISOR.glob("*/*.docx")) + sorted((ROOT / "docs/资料").glob("0[567]_*.docx")),
        "PDF": sorted((ADVISOR / "codex版").glob("*.pdf")),
        "网页与教学演示": sorted(ADVISOR.glob("*/*.html")) + [ROOT / "docs/资料/08_三阶段近期实验导读.html"],
        "公开数据与展示图": [ADVISOR / "codex版/数据与图" / name for name in
                          ("第二阶段_确认88景.csv", "02_第二阶段服务效果.png", "03_学习与搜索耗时.png")],
    }
    lines = [
        "# 纯禁忌搜索评价：文档同步清单", "来源：codex", "",
        "实验日期：2026-10-21；文档同步：2026-10-22。正式评价共1504条，报告、三套导师汇报和问答共用同一份实测摘要。",
        "主档确认88景：纯禁忌搜索32次满足率90.0%，贪心＋禁忌搜索32次94.5%；"
        "原始历史表保留原成绩，初始化名称与新增对照已区分。", "",
        "纯禁忌32次已直接加入主对比表：Codex版PPT第15页、cc版PPT第16页，以及对应Word、Markdown和网页主比较。"
        "该行的最差10%飞机满足率为56.3%，SGM为0.963。"
        "[查看第15页主表](office_preview_codex_slide15_final.png)。", "",
        "[完整数据摘要](presentation_summary.json) · [原始结果](eval/) · [四幅比较图](figures/) · [七张汇报展示图](presentation_figures/) · [冻结源码检查](check.log)", "",
        "本轮把纯禁忌32次放进主表、正文和现有比较图；表格使用原有普通行样式，没有单独的绿色底色或额外粗体。"
        "dsh和cc的主结果图已补纯禁忌系列，中期资料保留旧实验条件并在第二阶段正文单列本次补充实验。", "",
        "| 类型 | 文件 |", "| --- | --- |",
    ]
    counts = {}
    for category, paths in categories.items():
        paths = [path for path in paths if not path.name.startswith("~$")
                 and not path.stem.endswith(("_纯禁忌对照版", "_纯禁忌主表更新版"))]
        counts[category] = len(paths)
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(path)
            target = os.path.relpath(path, OUT).replace("\\", "/")
            lines.append(f"| {category} | [{path.relative_to(ROOT).as_posix()}]({target}) |")
    lines.extend([
        "", "核查：逐束数组与统计一致；既有文字材料保留历史数字；"
        "新增文档链接可达；网页完成桌面与窄屏检查；Office原件和对应PDF完成名称与实测表检查。", "",
        "## 后续重新生成材料时", "",
        "原材料生成器保留原实验版本。若重建了材料，按下列顺序补入本次对照：", "",
        "```powershell",
        "python -X utf8 scripts/pure_tabu_docs/publish_results.py",
        "python -X utf8 scripts/pure_tabu_docs/update_advisor_figures.py",
        "python -X utf8 scripts/pure_tabu_docs/sync_markdown.py --summary outputs/experiments/s2_pure_tabu_20261021/presentation_summary.json",
        "python -X utf8 scripts/pure_tabu_docs/sync_presentation_assets.py",
        "python -X utf8 scripts/pure_tabu_docs/sync_web.py",
        "python -X utf8 scripts/pure_tabu_docs/sync_office.py",
        "python -X utf8 scripts/pure_tabu_docs/export_codex_pdfs.py",
        "```", "",
        "加入纯禁忌系列的七张展示图另存到 presentation_figures，历史PNG标签修订在 historical_figures；"
        "原图和冻结实验资产保留，教学动画标明教学值。", "",
        "## 中期与讲解正文的对应位置", "",
        "| 文档 | 正文位置 |", "| --- | --- |",
    ])
    for name, heading in BODY_LOCATIONS.items():
        path = ROOT / name
        lines.append(f"| [{path.name}]({os.path.relpath(path, OUT).replace(chr(92), '/')}) | {heading.lstrip('# ')} |")
    (OUT / "document_sync.md").write_text("\n".join(lines), encoding="utf-8")
    print(counts)


if __name__ == "__main__":
    main()
