"""生成月报模板与示例（Word），版式沿用周报。"""
from datetime import datetime, timedelta, timezone

from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

from build_weekly_template import (GRAY, OUT, bullet, cell_text, fix_widths, heading,
                                   new_document, para, set_font, shade, table)

CST = timezone(timedelta(hours=8))


def title_block(doc, info, hint=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(8)
    set_font(p.add_run("研究生科研工作月报"), name="黑体", size=18, bold=True)
    t = doc.add_table(rows=3, cols=4)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    cells = [
        ("姓    名", info["name"], "月    份", info["month"]),
        ("起止日期", info["dates"], "对应周次", info["weeks"]),
        ("研究课题", info["topic"], None, None),
    ]
    for r, (a, b, c, d) in enumerate(cells):
        row = t.rows[r].cells
        cell_text(row[0], a, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
        shade(row[0], "E8EEF7")
        if c is None:
            cell_text(row[1].merge(row[3]), b, hint=hint)
        else:
            cell_text(row[1], b, hint=hint)
            cell_text(row[2], c, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
            shade(row[2], "E8EEF7")
            cell_text(row[3], d, hint=hint)
    fix_widths(t, (2.4, 5.6, 2.4, 5.6))


def build(doc, m, hint=False):
    title_block(doc, m["info"], hint)
    heading(doc, "一、本月工作概述")
    para(doc, m["summary"], hint)
    heading(doc, "二、各周工作")
    table(doc, ["周次", "日期", "主要工作"], m["weeks"], (1.8, 3.0, 11.2), hint)
    heading(doc, "三、主要进展与结果")
    for title, text in m["progress"]:
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Pt(21)
        set_font(p.add_run(title), bold=True, color=GRAY if hint else None)
        set_font(p.add_run(text), color=GRAY if hint else None, italic=hint)
    heading(doc, "四、本月计划完成情况")
    table(doc, ["上月定的计划", "完成情况", "说明"], m["plan_check"], (5.0, 2.4, 8.6), hint)
    heading(doc, "五、存在问题")
    for n, text in enumerate(m["issues"], start=1):
        bullet(doc, n, "", text, hint)
    heading(doc, "六、下月计划")
    for n, text in enumerate(m["plan"], start=1):
        bullet(doc, n, "", text, hint)


TEMPLATE = {
    "info": {"name": "姓名", "month": "2026年X月", "dates": "X月X日—X月X日",
             "weeks": "第 X～X 周", "topic": "HTS智能波束覆盖及频率-功率资源分配算法研究"},
    "summary": "【3～5句：这个月主要做了哪几块，各自得到什么结论，整体进度怎样。】",
    "weeks": [["第X周", "X/X–X/X", "【照抄该周周报的一句话概述】"],
              ["第X周", "X/X–X/X", ""], ["第X周", "X/X–X/X", ""], ["第X周", "X/X–X/X", ""]],
    "progress": [("（1）【方向或实验名】：", "【做了什么、结果数字、和谁比、说明什么。每块一段。】"),
                 ("（2）【方向或实验名】：", "【同上。】")],
    "plan_check": [["【上月月报写的计划1】", "【完成/部分完成/未做】", "【没完成的写原因】"],
                   ["【计划2】", "", ""]],
    "issues": ["【这个月暴露出来、还没解决的问题，1～3条。】"],
    "plan": ["【下个月要做的事，2～4条。】"],
}

EXAMPLE = {
    "info": {"name": "闵昱为", "month": "2026年8月", "dates": "8月3日—8月28日",
             "weeks": "第 14～17 周", "topic": "HTS智能波束覆盖及频率-功率资源分配算法研究"},
    "summary": "八月前三周主要在做第一阶段的覆盖规划：把区域重构PPO的缓存问题修好后重新训练，"
               "调研了可变波束数的做法并整理出十二条路线做了验证，又试了用R04路线引导强化学习做覆盖。"
               "最后一周回到第二阶段，做了可变波束数量的泛化训练。总的来看，覆盖这块学习方法还没有明显超过规则方法；"
               "第二阶段混合束数训练对大束数有帮助，但贪心加功率细调仍然更强。",
    "weeks": [
        ["第14周", "8/3–8/7", "区域重构PPO机制诊断，修复缓存缺陷后重训并完成共同考核"],
        ["第15周", "8/10–8/14", "调研19篇可变波束数论文，提出十二条覆盖路线并在四个实例上验证"],
        ["第16周", "8/17–8/21", "完成R04路线引导的强化学习覆盖优化A阶段，综合调研32类覆盖优化方法"],
        ["第17周", "8/24–8/28", "第二阶段可变波束数量泛化训练两轮，对比功率细调前后的结果"],
    ],
    "progress": [
        ("（1）区域重构PPO：", "月初训练到6万多步时发现缓存有缺陷，先停下来查原因，修好后从零重训，"
         "在100个独立场景上测试。PPO的主体负载CV平均0.60，初始布局是0.86，同样重构器下的搜索是0.66，"
         "PPO稍好一些，但只在70个场景上比搜索好，不是每个场景都赢。离当初定的目标（CV不超过0.2）还差得远，"
         "严格100%覆盖的有93个场景。"),
        ("（2）可变波束数十二路线：", "看了19篇相关论文，把能用的做法归成12条路线，在独立场景上都能把所有飞机覆盖到，"
         "实际用到的波束数在38到111之间。热点拆合（R04）在均衡和承载之间比较平衡；"
         "两条学习路线没显出优势，其中一条验证时基本停在初始布局不动。"),
        ("（3）R04引导的强化学习覆盖：", "六种方法在同样的初始布局上比较，共测了1800条。PPO能把初始布局改好一些，"
         "但负载CV比只按眼前收益选动作的规则还高2%～6%，没过事先定的门槛。只用R04示范做行为克隆、不再加强化学习的版本反而更好。"),
        ("（4）第二阶段可变束数泛化：", "用50～200束混合训练，和固定100束训练比，在没见过的束数上细调前满足率高0.7个百分点，"
         "束数越大提升越明显。但贪心加SCA细调后的满足率是93.3%，模型细调后是91.9%，还是贪心强。"
         "第二轮分别试了改奖励、改折扣、改网络三种做法，最好的一种细调后92.4%，仍然没追上。"),
    ],
    "plan_check": [
        ["修复区域重构PPO并重训", "完成", "修好缓存后从零训练，完成100个场景考核"],
        ["可变波束数覆盖方法调研", "完成", "整理出十二条路线并做了验证"],
        ["强化学习覆盖优化", "部分完成", "A阶段做完，效果没过门槛，B阶段先没做"],
        ["第二阶段适配可变波束数", "完成", "做了两轮，效果还不如贪心加细调"],
    ],
    "issues": [
        "第一阶段几种学习方法都没有明显胜过规则或搜索，强化学习在覆盖这块怎么用，需要重新想一下。",
        "第二阶段从空开始一个波束一个波束往上分的SAC，怎么改都追不上贪心加细调，可能得换个思路。",
        "这个月的训练都只用了一个随机种子，结果的稳定性还没验证。",
    ],
    "plan": [
        "第二阶段改成从贪心解出发、让SAC只做局部调整，同时保留两种构造式改进做对照。",
        "第一阶段试试不用强化学习的路线，以及AI加优化的混合做法。",
        "在同样条件下对比SAC、PPO、DQN三种算法。",
        "整理目前的进展，准备中期讨论材料。",
    ],
}


def set_meta(doc, created):
    cp = doc.core_properties
    cp.author = cp.last_modified_by = "闵昱为"
    cp.title = ""
    cp.created = created.astimezone(timezone.utc)
    cp.modified = (created + timedelta(minutes=47)).astimezone(timezone.utc)
    cp.revision = 4


if __name__ == "__main__":
    for name, data, hint, created in (
            ("月报模板.docx", TEMPLATE, True, None),
            ("月报示例_8月.docx", EXAMPLE, False, datetime(2026, 8, 31, 20, 36, tzinfo=CST))):
        doc = new_document()
        build(doc, data, hint)
        if created:
            set_meta(doc, created)
        doc.save(OUT / name)
        print(OUT / name)
