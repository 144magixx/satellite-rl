"""Synchronize the pure-tabu comparison into the existing Office materials."""

import json
import os
import re
from pathlib import Path

from docx import Document
from docx.shared import Inches as DocInches, Pt as DocPt
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "outputs/experiments/s2_pure_tabu_20261021/presentation_summary.json"
REPORT = "docs/第二阶段/实验报告/25_纯禁忌搜索与贪心加禁忌搜索配对评价.md"
MARKER = "2026-10-21 纯禁忌搜索与贪心初解对照"
FIGURE_NOTE = "图中禁忌搜索采用贪心初解；纯禁忌见新增对照页。"
OLD_FIGURE_NOTE = "图中贪心＋禁忌搜索采用贪心初解；纯禁忌见新增对照页。"
OFFICE = ROOT / "docs/Claude维护/导师汇报"
DOCS = [
    *sorted(OFFICE.rglob("*.docx")),
    *sorted((ROOT / "docs/资料").glob("0[567]_*.docx")),
]
PPTS = [p for p in sorted(OFFICE.rglob("*.pptx"))
        if not p.name.startswith("~$") and not p.stem.endswith("_纯禁忌对照版")]


def comparison_rows():
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    rows = {r["method"]: r for r in summary["rows"] if r["set"] == "conf_d432_n150"}
    return [(budget, rows[f"PTS{budget}"], rows[f"GTS{budget}"]) for budget in (32, 96, 320)]


def percent(row, key):
    return f"{row[key] * 100:.1f}%"


METHOD = re.compile(r"(?<!纯)(?<!贪心＋)(?<!贪心加)禁忌(?:搜索)?(?=\s*(?:32|96|320)\s*次)")
BASELINE = re.compile(r"(?<!纯)(?<!贪心＋)(?<!贪心加)禁忌搜索")
SHORTHAND = re.compile(r"(?<!纯)(?<!贪心＋)禁忌(?=\s+94\.5)")


def relabel_paragraph(paragraph):
    if FIGURE_NOTE in paragraph.text or OLD_FIGURE_NOTE in paragraph.text:
        return False
    changed = False
    for run in paragraph.runs:
        updated = METHOD.sub("贪心＋禁忌搜索", run.text)
        updated = SHORTHAND.sub("贪心＋禁忌搜索", updated)
        if any(word in paragraph.text for word in ("SGM", "满足率", "评价", "最好", "最强", "基线", "对照", "耗时")):
            updated = BASELINE.sub("贪心＋禁忌搜索", updated)
        if updated != run.text:
            run.text = updated
            changed = True
    return changed


def word_table(document, rows):
    table = document.add_table(rows=1, cols=5)
    table.style = "Table Grid"
    for cell, value in zip(table.rows[0].cells, ["搜索次数", "方法", "SCA前满足率", "SCA后满足率 / SGM", "总时间"]):
        cell.text = value
    for budget, pure, hybrid in rows:
        for initial, row in (("纯禁忌搜索（随机初解）", pure), ("贪心＋禁忌搜索", hybrid)):
            cells = table.add_row().cells
            values = [str(budget), initial, percent(row, "before_W"),
                      f"{percent(row, 'after_W')} / {row['after_SGM']:.3f}",
                      f"{row['total_seconds']:.2f} 秒"]
            for cell, value in zip(cells, values):
                cell.text = value
    for cell in table.rows[0].cells:
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.bold = True
                run.font.size = DocPt(9)
    for row in table.rows[1:]:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = DocPt(9)


def update_docx(path, rows):
    document = Document(path)
    changed = False
    for paragraph in document.paragraphs:
        changed = relabel_paragraph(paragraph) or changed
    caption_numbers = {
        "cc版": {"图15", "图17", "图18", "图20"},
        "dsh版": {"图 4", "图 5", "图 6", "图 8", "图 9"},
    }
    if path.parent.name in caption_numbers and ("汇报材料" in path.stem):
        for paragraph in document.paragraphs:
            if any(paragraph.text.startswith(number) for number in caption_numbers[path.parent.name]):
                for run in paragraph.runs:
                    if OLD_FIGURE_NOTE in run.text:
                        run.text = run.text.replace(" " + OLD_FIGURE_NOTE, "").replace(OLD_FIGURE_NOTE, "")
                        changed = True
                if FIGURE_NOTE not in paragraph.text:
                    paragraph.add_run(" " + FIGURE_NOTE)
                    changed = True
    for table in document.tables:
        if table.rows[0].cells[0].text == "搜索次数" and table.rows[0].cells[1].text == "初解":
            table.rows[0].cells[1].text = "方法"
            for row in table.rows[1:]:
                method_cell = row.cells[1]
                method_cell.text = {
                    "随机合法初解": "纯禁忌搜索（随机初解）",
                    "容量感知贪心初解": "贪心＋禁忌搜索",
                }[method_cell.text]
            changed = True
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    changed = relabel_paragraph(paragraph) or changed
    if any(MARKER in p.text for p in document.paragraphs):
        if changed:
            document.save(path)
        return changed
    document.add_page_break()
    document.add_heading(MARKER, level=1)
    document.add_paragraph("本次对照更新来源：codex；原文内容与原作者来源标注保留。实验数据与表格由 codex 根据确认集评价摘要生成。")
    document.add_paragraph(
        "结论：同样搜索320次，纯禁忌搜索满足91.1%的需求，贪心＋禁忌搜索满足94.9%；"
        "后者高3.9个百分点。纯禁忌搜索从随机合法分配开始，不调用贪心候选择优；"
        "两组使用相同的禁忌搜索与SCA。"
    )
    document.add_paragraph(
        "历史材料中的S1、S96、S2依次指贪心＋禁忌搜索32、96、320次；"
        "历史图中的“禁忌搜索”也采用贪心初解。此处新增的“纯禁忌搜索”是单独对照。"
    )
    document.add_paragraph("43.2 Mbps/机、150束预算、同一确认88景。满足率为逐机平均需求满足率；下表将SCA前后分开。")
    word_table(document, rows)
    document.add_paragraph(
        "时间为同一20627节点、四路场景并行且每景单线程测得的每景平均墙钟时间，"
        "包含初解、搜索、核验和SCA，不含第一阶段布局及数据读写。历史实验时间采用原口径，速度比较请用本表。"
    )
    document.add_paragraph(f"完整指标、压力档和敏感性结果见 {REPORT}。")
    document.save(path)
    return True


def add_text(slide, left, top, width, height, text, size, bold=False, color=(30, 44, 55)):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    frame = box.text_frame
    frame.word_wrap = True
    paragraph = frame.paragraphs[0]
    paragraph.text = text
    paragraph.font.name = "Microsoft YaHei"
    paragraph.font.size = Pt(size)
    paragraph.font.bold = bold
    paragraph.font.color.rgb = RGBColor(*color)
    return box


def update_pptx(path, rows):
    presentation = Presentation(path)
    changed = False
    for slide in presentation.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in list(shape.text_frame.paragraphs):
                    if paragraph.text == OLD_FIGURE_NOTE:
                        paragraph._p.getparent().remove(paragraph._p)
                        changed = True
                        continue
                    changed = relabel_paragraph(paragraph) or changed
            if shape.has_table:
                for row in shape.table.rows:
                    for cell in row.cells:
                        for paragraph in cell.text_frame.paragraphs:
                            changed = relabel_paragraph(paragraph) or changed
        for shape in list(slide.shapes):
            if shape.has_text_frame and shape.text == OLD_FIGURE_NOTE:
                slide.shapes._spTree.remove(shape._element)
                changed = True
    figure_pages = {
        "cc版": {15: 1.56, 18: 1.56, 20: 1.56},
        "dsh版": {10: 6.38, 14: 6.31, 16: 5.83, 19: 6.72},
    }
    for number, top in figure_pages.get(path.parent.name, {}).items():
        slide = presentation.slides[number - 1]
        if not any(FIGURE_NOTE in shape.text for shape in slide.shapes if shape.has_text_frame):
            add_text(slide, .61, top, 11.6, .23, FIGURE_NOTE, 10, color=(70, 85, 95))
            changed = True
    if path.parent.name == "dsh版":
        slide = presentation.slides[10]
        if not any(FIGURE_NOTE in shape.text for shape in slide.shapes if shape.has_text_frame):
            frame = max((shape.text_frame for shape in slide.shapes if shape.has_text_frame and shape.top / 914400 > 5 and shape.top / 914400 < 7),
                        key=lambda frame: len(frame.text))
            paragraph = frame.add_paragraph()
            paragraph.text = FIGURE_NOTE
            paragraph.font.name = "Microsoft YaHei"
            paragraph.font.size = Pt(10)
            paragraph.font.color.rgb = RGBColor(70, 85, 95)
            changed = True
    if any(MARKER in shape.text for slide in presentation.slides for shape in slide.shapes if shape.has_text_frame):
        if not changed:
            return False
        pending = path.with_suffix(".syncing")
        presentation.save(pending)
        os.replace(pending, path)
        return path
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    add_text(slide, .55, .3, 12.2, .5, MARKER, 26, True)
    add_text(slide, .57, .85, 12, .42,
             "纯禁忌从随机合法初解出发；历史S1 / S96 / S2均为贪心＋禁忌搜索32 / 96 / 320次", 15)
    headers = ["搜索次数", "纯禁忌：SCA后满足率 / SGM / 总时间", "贪心＋禁忌：SCA后满足率 / SGM / 总时间"]
    table = slide.shapes.add_table(4, 3, Inches(.55), Inches(1.53), Inches(12.2), Inches(3.1)).table
    for col, width in enumerate((1.5, 5.35, 5.35)):
        table.columns[col].width = Inches(width)
    for col, heading in enumerate(headers):
        table.cell(0, col).text = heading
    for index, (budget, pure, hybrid) in enumerate(rows, start=1):
        table.cell(index, 0).text = str(budget)
        for col, row in ((1, pure), (2, hybrid)):
            table.cell(index, col).text = (
                f"{percent(row, 'after_W')} / {row['after_SGM']:.3f} / {row['total_seconds']:.2f}秒"
            )
    for row_index in range(4):
        for col in range(3):
            cell = table.cell(row_index, col)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(31, 61, 74) if row_index == 0 else RGBColor(242, 247, 248)
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.name = "Microsoft YaHei"
                paragraph.font.size = Pt(13 if row_index == 0 else 16)
                paragraph.font.bold = row_index == 0
                paragraph.font.color.rgb = RGBColor(255, 255, 255) if row_index == 0 else RGBColor(30, 44, 55)
    add_text(slide, .6, 4.95, 12, .52,
             "320次：贪心初解高3.9个百分点；纯禁忌提高搜索次数后仍未追平。", 20, True)
    add_text(slide, .6, 5.6, 12, .62,
             "同一确认88景、43.2 Mbps/机、150束；时间含初解＋搜索＋核验＋SCA，不含第一阶段与文件读写。", 13)
    add_text(slide, .6, 6.38, 12, .43,
             "本次修订/数据与图：codex；原幻灯片来源保留。完整结果：第二阶段报告25。", 11, color=(85, 101, 110))
    pending = path.with_suffix(".syncing")
    presentation.save(pending)
    try:
        os.replace(pending, path)
        return path
    except PermissionError:
        alternate = path.with_name(path.stem + "_纯禁忌对照版.pptx")
        os.replace(pending, alternate)
        return alternate


def main():
    rows = comparison_rows()
    changed = []
    for path in DOCS:
        if update_docx(path, rows):
            changed.append(path.relative_to(ROOT).as_posix())
    for path in PPTS:
        result = update_pptx(path, rows)
        if result:
            changed.append(result.relative_to(ROOT).as_posix())
    print(json.dumps({"changed": changed, "checked": len(DOCS) + len(PPTS)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
