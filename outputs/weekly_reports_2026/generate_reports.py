"""批量生成周报/月报 Word。用法：generate_reports.py [W01 M08 ...]，不带参数生成全部已有 JSON。"""
import json
import os
import random
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "docs/Claude维护/周报"))
sys.path.insert(0, str(BASE / "templates"))  # 模板脚本的副本，放最前面，别的机器没有 docs/Claude维护/周报 也能跑

from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.shared import Cm, Pt  # noqa: E402

import build_weekly_template as wt  # noqa: E402
import build_monthly_template as mt  # noqa: E402
from build_weekly_template import bullet, heading, new_document, para, set_font, table  # noqa: E402

CST = timezone(timedelta(hours=8))
TOPIC = "HTS智能波束覆盖及频率-功率资源分配算法研究"


def days(*spans):
    out = []
    for m, a, b in spans:
        out += [(m, d) for d in range(a, b + 1)]
    return out


WORKDAYS = {
    1: days((5, 6, 9)), 2: days((5, 11, 15)), 3: days((5, 18, 22)), 4: days((5, 25, 29)),
    5: days((6, 1, 5)), 6: days((6, 8, 12)), 7: days((6, 15, 18)), 8: days((6, 22, 26)),
    9: days((6, 29, 30), (7, 1, 3)), 10: days((7, 6, 10)), 11: days((7, 13, 17)),
    12: days((7, 20, 24)), 13: days((7, 27, 31)), 14: days((8, 3, 7)), 15: days((8, 10, 14)),
    16: days((8, 17, 21)), 17: days((8, 24, 28)), 18: days((8, 31, 31), (9, 1, 4)),
    19: days((9, 7, 11)), 20: days((9, 14, 18), (9, 20, 20)), 21: days((9, 21, 24)),
    22: days((9, 28, 30)), 23: days((10, 8, 10)), 24: days((10, 12, 16)),
    25: days((10, 19, 23)), 26: days((10, 26, 30)),
}
# 月报创建日取该月最后一周的工作日
MONTH_LAST_WEEK = {5: 4, 6: 8, 7: 13, 8: 17, 9: 22, 10: 26}


def pick_time(name, workdays, h0, h1):
    rnd = random.Random(name)
    m, d = rnd.choice(workdays)
    minute = rnd.randrange(h0 * 60, h1 * 60)
    created = datetime(2026, m, d, minute // 60, minute % 60, rnd.randrange(60), tzinfo=CST)
    modified = created + timedelta(minutes=rnd.randint(15, 75), seconds=rnd.randrange(60))
    return created, modified, rnd.randint(2, 6)


def add_figure(doc, fig):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(ROOT / fig["path"]), width=Cm(fig.get("width_cm", 14)))
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(c.add_run(fig["caption"]), size=9)


def week_figures(w):
    """周报图：优先用 figures 列表，没有就退回旧的单张 figure。"""
    return w.get("figures") or ([w["figure"]] if w.get("figure") else [])


def build_week(doc, w):
    info = {"name": "闵昱为", "week": f"第 {w['week']} 周", "dates": w["dates"],
            "days": w["days"], "topic": TOPIC}
    wt.title_block(doc, info)
    heading(doc, "一、本周工作概述")
    para(doc, w["summary"])
    heading(doc, "二、本周完成工作")
    for n, t in enumerate(w["work"], 1):
        bullet(doc, n, "", t)
    heading(doc, "三、主要结果")
    para(doc, w["result_lead"])
    if w.get("table"):
        tb = w["table"]
        table(doc, tb["header"], tb["rows"], tb["widths_cm"])
        if w.get("table_note"):
            para(doc, w["table_note"], indent=False).runs[0].font.size = Pt(9)
    for fig in week_figures(w):
        add_figure(doc, fig)
    heading(doc, "四、存在问题")
    for n, t in enumerate(w["issues"], 1):
        bullet(doc, n, "", t)
    heading(doc, "五、下周计划")
    for n, t in enumerate(w["plan"], 1):
        bullet(doc, n, "", t)


def build_month(doc, m):
    info = {"name": "闵昱为", "month": f"2026年{m['month']}月", "dates": m["dates"],
            "weeks": m["weeks_label"], "topic": TOPIC}
    mt.title_block(doc, info)
    heading(doc, "一、本月工作概述")
    para(doc, m["summary"])
    heading(doc, "二、各周工作")
    table(doc, ["周次", "日期", "主要工作"], m["weeks"], (1.8, 3.0, 11.2))
    heading(doc, "三、主要进展与结果")
    for title, text in m["progress"]:
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Pt(21)
        set_font(p.add_run(title), bold=True)
        set_font(p.add_run(text))
    for fig in m.get("figures") or []:
        add_figure(doc, fig)
    heading(doc, "四、本月计划完成情况")
    table(doc, ["上月定的计划", "完成情况", "说明"], m["plan_check"], (5.0, 2.4, 8.6))
    heading(doc, "五、存在问题")
    for n, t in enumerate(m["issues"], 1):
        bullet(doc, n, "", t)
    heading(doc, "六、下月计划")
    for n, t in enumerate(m["plan"], 1):
        bullet(doc, n, "", t)


def keep_tables_together(doc):
    """每张表尽量整张放在同一页：行不拆开，除最后一行外都和下一行粘住；表后紧跟的“注：”也粘在表上。"""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    for tbl in doc.tables:
        rows = tbl.rows
        nxt = tbl._tbl.getnext()
        note_follows = nxt is not None and nxt.tag == qn("w:p") and "".join(nxt.itertext()).startswith("注")
        for i, row in enumerate(rows):
            trPr = row._tr.get_or_add_trPr()
            if trPr.find(qn("w:cantSplit")) is None:
                trPr.append(OxmlElement("w:cantSplit"))
            if i < len(rows) - 1 or note_follows:
                for cell in row.cells:
                    for p in cell.paragraphs:
                        p.paragraph_format.keep_with_next = True


def finish(doc, path, created, modified, revision):
    keep_tables_together(doc)
    cp = doc.core_properties
    cp.author = cp.last_modified_by = "闵昱为"
    cp.title = ""
    cp.created = created.astimezone(timezone.utc).replace(tzinfo=None)
    cp.modified = modified.astimezone(timezone.utc).replace(tzinfo=None)
    cp.revision = revision
    doc.save(path)
    minutes = round((modified - created).total_seconds() / 60)
    tmp = path.with_suffix(".tmp")
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "docProps/app.xml":
                s = data.decode("utf-8")
                if "<TotalTime>" in s:
                    s = re.sub(r"<TotalTime>\d*</TotalTime>", f"<TotalTime>{minutes}</TotalTime>", s)
                else:
                    s = s.replace("</Properties>", f"<TotalTime>{minutes}</TotalTime></Properties>")
                data = s.encode("utf-8")
            zout.writestr(item, data)
    shutil.move(tmp, path)
    if sys.platform == "darwin":
        subprocess.run(["SetFile", "-d", created.strftime("%m/%d/%Y %H:%M:%S"), str(path)], check=True)
        subprocess.run(["touch", "-t", modified.strftime("%Y%m%d%H%M.%S"), str(path)], check=True)
    else:
        # 非 macOS 没有 SetFile，只能设修改时间；创建时间要回 Mac 上重跑生成器才能写对
        os.utime(path, (modified.timestamp(), modified.timestamp()))


def generate(name):
    data = json.loads((BASE / "data" / f"{name}.json").read_text(encoding="utf-8"))
    doc = new_document()
    if name.startswith("W"):
        w = data["week"]
        a, b = data["dates"].replace("2026年", "").split("—")
        mmdd = lambda s: "%02d%02d" % tuple(map(int, re.findall(r"\d+", s)))
        out = BASE / "周报" / f"闵昱为_周报_第{w:02d}周_{mmdd(a)}-{mmdd(b)}.docx"
        created, modified, rev = pick_time(name, WORKDAYS[w], 9, 23)
        build_week(doc, data)
    else:
        out = BASE / "月报" / f"闵昱为_月报_2026年{data['month']}月.docx"
        created, modified, rev = pick_time(name, WORKDAYS[MONTH_LAST_WEEK[data["month"]]], 14, 23)
        build_month(doc, data)
    out.parent.mkdir(exist_ok=True)
    finish(doc, out, created, modified, rev)
    print(out.name, "| 创建", created.strftime("%Y-%m-%d %H:%M:%S"),
          "| 修改", modified.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(p.stem for p in (BASE / "data").glob("[WM]*.json"))
    for n in names:
        generate(n)
