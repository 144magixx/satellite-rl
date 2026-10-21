"""Synchronize the pure-tabu comparison into the existing Office materials."""

import json
import os
import re
from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.text.paragraph import Paragraph
from docx.shared import Inches as DocInches, Pt as DocPt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "outputs/experiments/s2_pure_tabu_20261021/presentation_summary.json"
REPORT = "docs/第二阶段/实验报告/25_纯禁忌搜索与贪心加禁忌搜索配对评价.md"
MARKER = "2026-10-21 纯禁忌搜索与贪心初解对照"
FIGURE_NOTE = "图中禁忌搜索采用贪心初解；纯禁忌见新增对照页。"
OLD_FIGURE_NOTE = "图中贪心＋禁忌搜索采用贪心初解；纯禁忌见新增对照页。"
UPDATED_FIGURE_NOTE = "图中贪心＋禁忌搜索用贪心初解；纯禁忌32次用随机合法初解。"
OFFICE = ROOT / "docs/Claude维护/导师汇报"
DOCS = [
    *sorted(OFFICE.rglob("*.docx")),
    *sorted((ROOT / "docs/资料").glob("0[567]_*.docx")),
]
PPTS = [p for p in sorted(OFFICE.rglob("*.pptx"))
        if not p.name.startswith("~$") and not p.stem.endswith(("_纯禁忌对照版", "_纯禁忌主表更新版"))]
FIGURES = ROOT / "outputs/experiments/s2_pure_tabu_20261021/presentation_figures"
PPT_FIGURES = {
    "cc版": {15: "cc_v2_11_高需求全方法.png", 17: "cc_v2_13_搜索效率.png"},
    "dsh版": {11: "dsh_G1_SGM两种协议.png", 14: "dsh_G2_SGM第三阶段前后.png",
              15: "dsh_G5_评价次数与满足率.png"},
}
WORD_FIGURES = {
    "cc版": {"图15": "cc_v2_11_高需求全方法.png", "图17": "cc_v2_13_搜索效率.png"},
    "dsh版": {"图 4": "dsh_G1_SGM两种协议.png", "图 5": "dsh_G2_SGM第三阶段前后.png",
              "图 6": "dsh_G5_评价次数与满足率.png"},
}


def replace_word_figures(document, variant):
    changed = False
    paragraphs = document.paragraphs
    for caption, filename in WORD_FIGURES.get(variant, {}).items():
        index = next((i for i, paragraph in enumerate(paragraphs) if paragraph.text.startswith(caption)), None)
        if index is None:
            continue
        image_paragraph = next(paragraph for paragraph in reversed(paragraphs[:index])
                               if paragraph._p.xpath(".//a:blip"))
        relation_id = image_paragraph._p.xpath(".//a:blip")[0].get(qn("r:embed"))
        part = document.part.related_parts[relation_id]
        new = (FIGURES / filename).read_bytes()
        if part.blob != new:
            part._blob = new
            changed = True
    return changed


def replace_ppt_figures(presentation, variant):
    changed = False
    for slide_number, filename in PPT_FIGURES.get(variant, {}).items():
        slide = presentation.slides[slide_number - 1]
        shape = next(shape for shape in slide.shapes if shape.shape_type == 13)
        part = slide.part.related_part(shape._element.blipFill.blip.rEmbed)
        new = (FIGURES / filename).read_bytes()
        if part.blob != new:
            part._blob = new
            changed = True
    return changed


def comparison_rows():
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    rows = {r["method"]: r for r in summary["rows"] if r["set"] == "conf_d432_n150"}
    return [(budget, rows[f"PTS{budget}"], rows[f"GTS{budget}"]) for budget in (32, 96, 320)]


def percent(row, key):
    return f"{row[key] * 100:.1f}%"


def pure_32():
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    return next(row for row in summary["rows"] if row["set"] == "conf_d432_n150" and row["method"] == "PTS32")


def insert_word_result(document, header, anchor, values):
    for table in document.tables:
        if [cell.text for cell in table.rows[0].cells][:len(header)] != header:
            continue
        if any("纯禁忌搜索32次" in cell.text or "纯禁忌搜索 32 次" in cell.text for row in table.rows for cell in row.cells):
            return False
        anchor_row = next(row for row in table.rows if anchor in row.cells[0].text)
        new_tr = deepcopy(anchor_row._tr)
        anchor_row._tr.addnext(new_tr)
        new_cells = [cell for row in table.rows if row._tr is new_tr for cell in row.cells]
        for cell, value in zip(new_cells, values):
            cell.paragraphs[0].runs[0].text = value
        return True
    return False


def normalize_word_result(document, header, template_label):
    for table in document.tables:
        if [cell.text for cell in table.rows[0].cells][:len(header)] != header:
            continue
        pure_row = next((row for row in table.rows if "纯禁忌搜索" in row.cells[0].text), None)
        if pure_row is None:
            return False
        has_green = any(cell._tc.get_or_add_tcPr().find(qn("w:shd")) is not None and
                        cell._tc.get_or_add_tcPr().find(qn("w:shd")).get(qn("w:fill")) == "E1F4EF"
                        for cell in pure_row.cells)
        has_bold = any(run.bold is True for cell in pure_row.cells for p in cell.paragraphs for run in p.runs)
        if not has_green and not has_bold:
            return False
        values = [cell.text for cell in pure_row.cells]
        template = next(row for row in table.rows if template_label in row.cells[0].text)
        fresh = deepcopy(template._tr)
        pure_row._tr.addnext(fresh)
        pure_row._tr.getparent().remove(pure_row._tr)
        new_row = next(row for row in table.rows if row._tr is fresh)
        for cell, value in zip(new_row.cells, values):
            cell.paragraphs[0].runs[0].text = value
        return True
    return False


def insert_word_main_sentence(document, anchor, pure):
    sentence = (
        "纯禁忌搜索32次从随机合法初解出发：同一88景SCA后满足率"
        f"{percent(pure, 'after_W')}、最低10%飞机{percent(pure, 'after_aircraft_worst10')}、"
        f"SGM {pure['after_SGM']:.3f}；单景总时间{pure['total_seconds']:.2f}秒"
        "（初解＋搜索＋核验＋SCA）。"
    )
    existing = next((paragraph for paragraph in document.paragraphs if sentence in paragraph.text), None)
    if existing is not None:
        if existing.style.name.startswith("Heading"):
            existing.style = document.styles["Normal"]
            return True
        return False
    source = next(paragraph for paragraph in document.paragraphs if anchor in paragraph.text)
    element = OxmlElement("w:p")
    source._p.addnext(element)
    paragraph = Paragraph(element, source._parent)
    paragraph.style = document.styles["Normal"] if source.style.name.startswith("Heading") else source.style
    paragraph.add_run(sentence)
    return True


def insert_midterm_section(document, anchor, pure, greedy, hybrid):
    title = "2026-10-21 需求压力补充实验"
    if any(title in paragraph.text for paragraph in document.paragraphs):
        return False
    source = next(paragraph for paragraph in document.paragraphs if anchor in paragraph.text)

    def after(node, value, style=None):
        element = OxmlElement("w:p")
        node.addnext(element)
        paragraph = Paragraph(element, source._parent)
        if style:
            paragraph.style = style
        paragraph.add_run(value)
        return paragraph

    level = "Heading 3" if source.style.name == "Heading 2" else "Heading 2"
    heading = after(source._p, title, level)
    origin = after(heading._p, "来源：codex")
    intro = after(origin._p,
                  "主档每机43.2 Mbps、150束、同一确认88景；下表为相同SCA后的逐机平均结果。"
                  "纯禁忌搜索从随机合法初解起步，贪心＋禁忌搜索从容量感知贪心初解起步。")
    table = document.add_table(rows=1, cols=5)
    table.style = "Table Grid"
    headers = ["方法", "满足率", "最低10%飞机", "SGM", "总时间/秒"]
    for cell, value in zip(table.rows[0].cells, headers):
        cell.text = value
    for name, row in (("纯禁忌搜索32次", pure), ("容量感知贪心", greedy),
                      ("贪心＋禁忌搜索32次", hybrid)):
        for cell, value in zip(table.add_row().cells,
                               [name, percent(row, "after_W"), percent(row, "after_aircraft_worst10"),
                                f"{row['after_SGM']:.3f}", f"{row['total_seconds']:.2f}"]):
            cell.text = value
    intro._p.addnext(table._tbl)
    after(table._tbl,
          f"纯禁忌搜索32次在SCA前满足率{percent(pure, 'before_W')}、SGM {pure['before_SGM']:.3f}；"
          "表中时间包括初解、搜索、核验与SCA，不含第一阶段布局和文件读写。"
          "本组是后续需求压力实验，旧表沿用原批次条件。")
    return True


def normalize_midterm_table(document):
    for table in document.tables:
        headers = [cell.text for cell in table.rows[0].cells]
        if headers != ["方法", "满足率", "最低10%飞机", "SGM", "总时间/秒"]:
            continue
        widths = [1.9, .9, 1.3, 1.1, .9]
        changed = table.autofit or any(abs(cell.width.inches - width) >= .02
                                      for cell, width in zip(table.rows[0].cells, widths))
        table.autofit = False
        for column, width in zip(table.columns, widths):
            column.width = DocInches(width)
        for row in table.rows:
            for cell, width in zip(row.cells, widths):
                cell.width = DocInches(width)
                for paragraph in cell.paragraphs:
                    fmt = paragraph.paragraph_format
                    changed = changed or fmt.first_line_indent != DocPt(0) or fmt.left_indent != DocPt(0)
                    fmt.first_line_indent = DocPt(0)
                    fmt.left_indent = DocPt(0)
        return changed
    return False


def insert_ppt_result(presentation, pure):
    table = next(shape.table for shape in presentation.slides[14].shapes
                 if shape.has_table and shape.table.cell(0, 0).text == "方法"
                 and any("直出T-SAC" in row.cells[0].text for row in shape.table.rows))
    if any("纯禁忌搜索32次" in row.cells[0].text for row in table.rows):
        return normalize_ppt_result(table, "改进式SAC（I0）", "纯禁忌搜索32次")
    anchor = next(row for row in table.rows if row.cells[0].text == "容量感知贪心")
    new_tr = deepcopy(anchor._tr)
    anchor._tr.addnext(new_tr)
    row = next(row for row in table.rows if row._tr is new_tr)
    values = ["纯禁忌搜索32次", percent(pure, "after_W"),
              percent(pure, "after_aircraft_worst10"), f"{pure['after_SGM']:.3f}"]
    for cell, value in zip(row.cells, values):
        cell.text_frame.paragraphs[0].runs[0].text = value
    for index, existing in enumerate(table.rows):
        existing.height = Inches(.49 if index == 0 else .5)
    return True


def normalize_ppt_result(table, template_label, pure_label):
    pure_row = next(row for row in table.rows if pure_label in row.cells[0].text)
    green = RGBColor(225, 244, 239)
    needs_style = any(cell.fill.type is not None and cell.fill.fore_color.rgb == green for cell in pure_row.cells)
    if needs_style:
        values = [cell.text for cell in pure_row.cells]
        if len(values) == 4:
            values[0] = "纯禁忌搜索32次"
        template = next(row for row in table.rows if template_label in row.cells[0].text)
        fresh = deepcopy(template._tr)
        pure_row._tr.addnext(fresh)
        pure_row._tr.getparent().remove(pure_row._tr)
        new_row = next(row for row in table.rows if row._tr is fresh)
        for cell, value in zip(new_row.cells, values):
            cell.text_frame.paragraphs[0].runs[0].text = value
    first = table.rows[1].cells[0].fill.fore_color.rgb
    second = table.rows[2].cells[0].fill.fore_color.rgb
    changed = needs_style
    for index, row in enumerate(table.rows):
        if index == 0:
            continue
        expected = first if index % 2 == 1 else second
        for cell in row.cells:
            if cell.fill.fore_color.rgb != expected:
                cell.fill.solid()
                cell.fill.fore_color.rgb = expected
                changed = True
    return changed


def insert_cc_ppt_result(presentation, pure):
    slide = presentation.slides[15]
    table = next(shape.table for shape in slide.shapes
                 if shape.has_table and shape.table.cell(0, 0).text == "方法"
                 and shape.table.cell(0, 1).text == "SGM")
    if any("纯禁忌搜索32次" in row.cells[0].text for row in table.rows):
        return normalize_ppt_result(table, "直出T-SAC", "纯禁忌搜索32次")
    anchor = next(row for row in table.rows if row.cells[0].text == "直出多层感知机")
    new_tr = deepcopy(anchor._tr)
    anchor._tr.addnext(new_tr)
    row = next(row for row in table.rows if row._tr is new_tr)
    values = ["纯禁忌搜索32次", f"{pure['after_SGM']:.3f}", percent(pure, "after_W"),
              percent(pure, "after_aircraft_worst10"), "32", f"{pure['total_seconds']:.2f}秒*", "无需训练"]
    for cell, value in zip(row.cells, values):
        cell.text_frame.paragraphs[0].runs[0].text = value
    for existing in table.rows:
        existing.height = Inches(.4)
    add_text(slide, .61, 5.04, 11.8, .18,
             "*纯禁忌以随机合法解起步，3.65秒为同机新评；旧行耗时沿用历史口径。", 9,
             color=(70, 85, 95))
    return True


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
    changed = replace_word_figures(document, path.parent.name)
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
                if FIGURE_NOTE not in paragraph.text and UPDATED_FIGURE_NOTE not in paragraph.text:
                    paragraph.add_run(" " + FIGURE_NOTE)
                    changed = True
    updated_captions = {
        "cc版": ("图15", "图17"),
        "dsh版": ("图 4", "图 5", "图 6"),
    }
    for paragraph in document.paragraphs:
        if any(paragraph.text.startswith(number) for number in updated_captions.get(path.parent.name, ())):
            for run in paragraph.runs:
                if FIGURE_NOTE in run.text:
                    run.text = run.text.replace(FIGURE_NOTE, UPDATED_FIGURE_NOTE)
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
    pure = pure_32()
    if path.parent.name == "codex版" and path.stem == "01_导师汇报材料":
        changed = insert_word_result(
            document, ["方法", "飞机平均W", "服务最低10%", "SGM"], "容量感知贪心",
            ["纯禁忌搜索32次（随机初解）", percent(pure, "after_W"),
             percent(pure, "after_aircraft_worst10"), f"{pure['after_SGM']:.3f}"]
        ) or changed
        changed = normalize_word_result(document, ["方法", "飞机平均W", "服务最低10%", "SGM"],
                                        "改进式SAC（I0）") or changed
    if path.parent.name == "cc版" and path.stem == "14_近期工作导师汇报材料":
        changed = insert_word_result(
            document, ["方法", "SGM", "满足率", "最差 10% 飞机"], "直出多层感知机版",
            ["纯禁忌搜索 32 次（随机初解）", f"{pure['after_SGM']:.3f}",
             percent(pure, "after_W"), percent(pure, "after_aircraft_worst10"),
             "32", f"{pure['total_seconds']:.2f} 秒（新评）", "无需训练"]
        ) or changed
        changed = normalize_word_result(document, ["方法", "SGM", "满足率", "最差 10% 飞机"],
                                        "直出 T-SAC") or changed
    if path.parent.name == "dsh版" and path.stem == "16_近期工作导师汇报材料":
        changed = insert_word_main_sentence(document, "学习类算法的完整对照", pure) or changed
    if path.parent.name == "codex版" and path.stem == "05_中期研究进展同步稿":
        changed = insert_word_main_sentence(document, "当前43.2 Mbps、确认88景", pure) or changed
    anchors = {
        ("cc版", "14_近期工作导师汇报材料"): "每架 43.2 Mbps、88 景、SCA 后",
        ("cc版", "15_导师汇报讲解提纲"): "第 20 页 第二阶段：高需求压力测试",
        ("codex版", "01_导师汇报材料"): "全部为同一确认88景",
        ("codex版", "03_讲解提纲"): "15｜43.2 Mbps压力下",
        ("dsh版", "17_导师汇报讲解提纲"): "第 15 页　方法全家福",
    }
    key = (path.parent.name, path.stem)
    if key in anchors:
        changed = insert_word_main_sentence(document, anchors[key], pure) or changed
    midterm_anchors = {
        "05_第二三阶段中期进展报告草稿": "第二阶段离散结构学习的研究进展",
        "06_中期研究进展与路线调整_导师讨论版": "对当前最好证据的判断",
        "07_HTS资源分配中期报告_正式版": "3.4 阶段研究结果与分析",
    }
    if path.stem in midterm_anchors:
        summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
        cohort = {r["method"]: r for r in summary["rows"] if r["set"] == "conf_d432_n150"}
        changed = insert_midterm_section(document, midterm_anchors[path.stem], pure,
                                         cohort["G1"], cohort["GTS32"]) or changed
        changed = normalize_midterm_table(document) or changed
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
    changed = replace_ppt_figures(presentation, path.parent.name)
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
        if not any(FIGURE_NOTE in shape.text or UPDATED_FIGURE_NOTE in shape.text
                   for shape in slide.shapes if shape.has_text_frame):
            add_text(slide, .61, top, 11.6, .23, FIGURE_NOTE, 10, color=(70, 85, 95))
            changed = True
    if path.parent.name == "dsh版":
        slide = presentation.slides[10]
        if not any(FIGURE_NOTE in shape.text or UPDATED_FIGURE_NOTE in shape.text
                   for shape in slide.shapes if shape.has_text_frame):
            frame = max((shape.text_frame for shape in slide.shapes if shape.has_text_frame and shape.top / 914400 > 5 and shape.top / 914400 < 7),
                        key=lambda frame: len(frame.text))
            paragraph = frame.add_paragraph()
            paragraph.text = FIGURE_NOTE
            paragraph.font.name = "Microsoft YaHei"
            paragraph.font.size = Pt(10)
            paragraph.font.color.rgb = RGBColor(70, 85, 95)
            changed = True
        slide = presentation.slides[14]
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text.startswith("32 次评价：") and "纯禁忌搜索90.0" not in shape.text:
                shape.text_frame.paragraphs[0].runs[0].text = (
                    "32 次评价：纯禁忌搜索90.0、贪心＋禁忌搜索94.5、SAC改进式94.9；"
                    "退火/遗传92.4/92.6。经典方法到3200次最高94.1；直出式约88。"
                )
                changed = True
    updated_note_slides = {"cc版": (15,), "dsh版": (11, 14)}
    for number in updated_note_slides.get(path.parent.name, ()):
        for shape in presentation.slides[number - 1].shapes:
            if shape.has_text_frame and FIGURE_NOTE in shape.text:
                for paragraph in shape.text_frame.paragraphs:
                    for run in paragraph.runs:
                        if FIGURE_NOTE in run.text:
                            run.text = run.text.replace(FIGURE_NOTE, UPDATED_FIGURE_NOTE)
                            changed = True
    if path.parent.name == "codex版" and path.stem == "02_导师汇报":
        changed = insert_ppt_result(presentation, pure_32()) or changed
    if path.parent.name == "cc版" and path.stem == "14_近期工作导师汇报":
        changed = insert_cc_ppt_result(presentation, pure_32()) or changed
    if any(MARKER in shape.text for slide in presentation.slides for shape in slide.shapes if shape.has_text_frame):
        if not changed:
            return False
        pending = path.with_suffix(".syncing")
        presentation.save(pending)
        try:
            os.replace(pending, path)
            return path
        except PermissionError:
            alternate = path.with_name(path.stem + "_纯禁忌主表更新版.pptx")
            os.replace(pending, alternate)
            return alternate
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
