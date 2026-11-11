"""生成周报模板与示例（Word）。"""
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

OUT = Path(__file__).resolve().parent
GRAY = RGBColor(0x80, 0x80, 0x80)


def set_font(run, name="宋体", size=10.5, bold=False, color=None, italic=False):
    run.font.name = name
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color is not None:
        run.font.color.rgb = color


def shade(cell, fill):
    tc_pr = cell._element.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def fix_widths(t, widths):
    t.autofit = False
    for i, w in enumerate(widths):
        t.columns[i].width = Cm(w)
    for row in t.rows:
        for i, w in enumerate(widths):
            row.cells[i].width = Cm(w)


def cell_text(cell, text, bold=False, hint=False, align=None):
    cell.text = ""
    p = cell.paragraphs[0]
    if align is not None:
        p.alignment = align
    run = p.add_run(text)
    set_font(run, bold=bold, color=GRAY if hint else None, italic=hint)


def new_document():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.5)
    sec.top_margin = sec.bottom_margin = Cm(2.2)
    normal = doc.styles["Normal"]
    normal.font.name = "宋体"
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(10.5)
    normal.paragraph_format.line_spacing = 1.3
    normal.paragraph_format.space_after = Pt(3)
    return doc


def heading(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_after = Pt(4)
    set_font(p.add_run(text), name="黑体", size=12, bold=True)


def para(doc, text, hint=False, indent=True):
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.first_line_indent = Pt(21)
    set_font(p.add_run(text), color=GRAY if hint else None, italic=hint)
    return p


def bullet(doc, n, label, text, hint=False):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Pt(21)
    p.paragraph_format.first_line_indent = Pt(-16)
    set_font(p.add_run(f"{n}. "))
    if label:
        set_font(p.add_run(label), bold=True, color=GRAY if hint else None)
    set_font(p.add_run(text), color=GRAY if hint else None, italic=hint)


def table(doc, header, rows, widths, hint=False):
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        cell_text(t.rows[0].cells[i], h, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
        shade(t.rows[0].cells[i], "E8EEF7")
    for r, row in enumerate(rows, start=1):
        for i, v in enumerate(row):
            cell_text(t.rows[r].cells[i], v, hint=hint,
                      align=WD_ALIGN_PARAGRAPH.CENTER if i == 0 else None)
    fix_widths(t, widths)
    return t


def title_block(doc, info, hint=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(8)
    set_font(p.add_run("研究生科研工作周报"), name="黑体", size=18, bold=True)
    t = doc.add_table(rows=3, cols=4)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    cells = [
        ("姓    名", info["name"], "周    次", info["week"]),
        ("起止日期", info["dates"], "工作日数", info["days"]),
        ("研究课题", info["topic"], None, None),
    ]
    for r, (a, b, c, d) in enumerate(cells):
        row = t.rows[r].cells
        cell_text(row[0], a, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
        shade(row[0], "E8EEF7")
        if c is None:
            merged = row[1].merge(row[3])
            cell_text(merged, b, hint=hint)
        else:
            cell_text(row[1], b, hint=hint)
            cell_text(row[2], c, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
            shade(row[2], "E8EEF7")
            cell_text(row[3], d, hint=hint)
    fix_widths(t, (2.4, 5.6, 2.4, 5.6))


def build(doc, w, hint=False):
    title_block(doc, w["info"], hint)
    heading(doc, "一、本周工作概述")
    para(doc, w["summary"], hint)
    heading(doc, "二、本周完成工作")
    for n, (label, text) in enumerate(w["work"], start=1):
        bullet(doc, n, label, text, hint)
    heading(doc, "三、主要结果")
    para(doc, w["result_lead"], hint)
    if w.get("table"):
        header, rows, widths = w["table"]
        table(doc, header, rows, widths, hint)
        if w.get("table_note"):
            p = para(doc, w["table_note"], hint=hint, indent=False)
            p.runs[0].font.size = Pt(9)
    heading(doc, "四、存在问题")
    for n, (label, text) in enumerate(w["issues"], start=1):
        bullet(doc, n, label, text, hint)
    heading(doc, "五、下周计划")
    for n, (label, text) in enumerate(w["plan"], start=1):
        bullet(doc, n, label, text, hint)


TEMPLATE = {
    "info": {"name": "姓名", "week": "第 N 周", "dates": "2026年X月X日—X月X日",
             "days": "X 天（注明节假日/调休）", "topic": "HTS智能波束覆盖及频率-功率资源分配算法研究"},
    "summary": "【一句话写清本周做了什么、得到什么结论。例：完成××实验，××方法比××高×个百分点，下一步转向××。】",
    "work": [
        ("工作1：", "【做了什么＋用了什么数据/规模＋结果怎样，1～3句。】"),
        ("工作2：", "【同上。方案设计、代码实现、训练、评价、文献调研、文档整理各写一条即可。】"),
        ("工作3：", "【可选。】"),
    ],
    "result_lead": "【先写结论和它意味着什么，再给数字。数字保留到能看出大小，并给出对比对象。】",
    "table": (["方法", "指标1", "指标2", "说明"],
              [["【方法A】", "", "", ""], ["【方法B】", "", "", ""], ["【基线】", "", "", ""]],
              (3.6, 3.0, 3.0, 6.4)),
    "table_note": "【可选：表格数据来源、场景数量。没有对比实验的周可删去表格。】",
    "issues": [("", "【遇到的问题或尚未解决的地方，1～3条；没有可写“无”。】")],
    "plan": [("", "【下周要做的事，1～3条，写清具体对象。】")],
}

EXAMPLE = {
    "info": {"name": "闵昱为", "week": "第 18 周", "dates": "2026年8月31日—9月4日",
             "days": "5 天", "topic": "HTS智能波束覆盖及频率-功率资源分配算法研究"},
    "summary": "本周把第二阶段三种SAC改进方案训练完，在880个测试场景上和贪心等方法做了统一对比。"
               "从贪心解出发做局部调整的方案效果最好，需求满足率比贪心高3个百分点左右，"
               "后面准备主要沿这个方向做。",
    "work": [
        ("", "前两轮的SAC都是从空分配开始一个波束一个波束往上加，结果一直比不过贪心。"
             "这周换了三种做法：第一种把奖励改成直接看最终的分配结果；第二种在第一种的基础上，"
             "往回放池里混一半贪心的样本；第三种直接拿贪心的结果当起点，SAC只负责调整频率槽。"),
        ("", "三个模型都用1月8日、9日的数据训练，共576个母场景，波束数在50到200之间，每个模型跑了40万步左右。"),
        ("", "测试用2月1日、2日新取的40个时刻，每个时刻按11种波束数和2种负载扩成22个场景，一共880个。"
             "为了比较公平，第三种方案和贪心、规则搜索、随机搜索用同一个起点，搜索次数也一样，"
             "功率细调前后的结果都做了统计。"),
        ("", "把实验方案和结果整理成了报告。"),
    ],
    "result_lead": "第三种方案在功率细调前后都比贪心好，比搜索次数相同的随机搜索也高一些，"
                   "可以看出模型学到的调整比随机去试要有效。前两种从空开始构造的方案还是不如贪心，这条路暂时先不往下做了。",
    "table": (["方法", "细调前满足率", "细调后满足率", "细调后比贪心"],
              [["贪心", "90.8%", "94.3%", "—"],
               ["随机搜索", "93.2%", "96.6%", "高2.4个百分点"],
               ["从空构造＋贪心样本", "88.5%", "92.6%", "低1.7个百分点"],
               ["贪心起点＋SAC调整", "94.1%", "97.3%", "高3.1个百分点"]],
              (4.4, 3.2, 3.2, 5.2)),
    "table_note": "注：880个场景的平均值。细调后指频率槽定下来以后再优化一遍功率。",
    "issues": [
        ("", "第三种方案细调前的SGM只比贪心高0.03左右。看了逐个波束的情况，六成多的波束分到的容量超过需求两倍，"
             "过供比较严重，细调后的匹配度主要是靠功率优化拉上来的，模型本身对匹配有多大作用还没分清楚。"),
        ("", "目前只训了一个种子，结果稳不稳还不好说；880个场景里有2个比贪心稍差。"),
        ("", "测试期间服务器重启过一次，中间还换过机器，各方法的耗时暂时不能直接比。"),
    ],
    "plan": [
        ("", "回到第一阶段，看一下不用强化学习的覆盖布局方法，先把调研和方案写出来。"),
        ("", "考虑怎么改奖励让模型少过供，比如把SGM加进奖励里试一下。"),
    ],
}

if __name__ == "__main__":
    for name, data, hint in (("周报模板.docx", TEMPLATE, True), ("周报示例_第18周_v2.docx", EXAMPLE, False)):
        doc = new_document()
        build(doc, data, hint)
        doc.save(OUT / name)
        print(OUT / name)
