"""Refresh the three presentation pages from the published paired comparison."""

import html
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "outputs/experiments/s2_pure_tabu_20261021/presentation_summary.json"
PAGES = [
    ROOT / "docs/Claude维护/导师汇报/cc版/19_近期工作导师汇报交互网页.html",
    ROOT / "docs/Claude维护/导师汇报/codex版/04_近期工作交互汇报.html",
    ROOT / "docs/Claude维护/导师汇报/dsh版/18_近期工作导师汇报网页.html",
]
START = "<!-- TABU-INITIALIZATION-COMPARISON -->"
END = "<!-- /TABU-INITIALIZATION-COMPARISON -->"
ORDER = ("Random0", "PTS32", "PTS96", "PTS320", "G1", "GTS32", "GTS96", "GTS320")
MAIN_START = "/* PURE_TABU_MAIN_20261021_START */"
MAIN_END = "/* PURE_TABU_MAIN_20261021_END */"
STYLE = """<style id="tabu-sync-style">
.tabu-sync{padding:64px 24px;color:inherit;border-top:1px solid #64748b55;border-bottom:1px solid #64748b55;background:#0e233533}.tabu-sync-inner{max-width:1180px;margin:auto}.tabu-sync-kicker{font-size:12px;letter-spacing:.1em;color:#8ba8be}.tabu-sync h2{font-size:clamp(24px,3vw,36px);line-height:1.3;margin:14px 0 20px}.tabu-sync p{font-size:15px;line-height:1.85;margin:12px 0}.tabu-sync-defs{display:grid;grid-template-columns:1fr 1fr;gap:20px}.tabu-sync-defs p{padding:18px 22px;border:1px solid #64748b66;border-radius:12px}.tabu-sync-source{font-size:12px!important;color:#8ba8be}.tabu-sync-status{padding:14px 18px;background:#64748b22;border-radius:8px}.tabu-sync a{color:inherit;text-decoration:underline}.tabu-sync-scroll{overflow-x:auto;margin:22px 0}.tabu-sync table{border-collapse:collapse;width:100%;min-width:760px;font-size:14px}.tabu-sync th,.tabu-sync td{border-bottom:1px solid #64748b66;padding:12px;text-align:left;white-space:nowrap}.tabu-sync th{color:#8ba8be}.tabu-sync details{margin:16px 0}.tabu-sync summary{cursor:pointer}@media(max-width:650px){.tabu-sync{padding:40px 18px}.tabu-sync-defs{grid-template-columns:1fr;gap:0}.tabu-sync p{font-size:14px}}
</style>"""


def pct(value):
    return f"{value * 100:.1f}%"


def metric_rows(rows):
    return "".join(
        f'<tr data-method="{key}"><th scope="row">{html.escape(rows[key]["label"])}</th>'
        f'<td>{rows[key]["before_SGM"]:.3f}</td><td>{pct(rows[key]["before_W"])}</td>'
        f'<td>{rows[key]["after_SGM"]:.3f}</td><td>{pct(rows[key]["after_W"])}</td>'
        f'<td>{rows[key]["total_seconds"]:.2f}</td></tr>'
        for key in ORDER
    )


def timing_rows(rows):
    return "".join(
        f'<tr><th scope="row">{html.escape(rows[key]["label"])}</th>'
        + "".join(f'<td>{rows[key][field]:.2f}</td>' for field in ("initial_seconds", "search_seconds", "verify_seconds", "sca_seconds"))
        + "</tr>"
        for key in ORDER
    )


def comparison(data):
    rows = {row["method"]: row for row in data["rows"]}
    pure, hybrid, greedy = rows["PTS320"], rows["GTS320"], rows["G1"]
    gap = (hybrid["after_W"] - pure["after_W"]) * 100
    return "\n".join([
        START,
        '<section class="tabu-sync" id="tabu-initialization-comparison" aria-labelledby="tabu-sync-title">',
        f'<div class="tabu-sync-inner"><div class="tabu-sync-kicker">算法名称与初始化 · {data["date"]}</div>',
        '<h2 id="tabu-sync-title">纯禁忌搜索已完成：贪心初解仍是关键</h2>',
        '<p>历史 S1、S96、S2 分别是<strong>贪心＋禁忌搜索 32、96、320 次</strong>；原图中的成绩保留其当时实验条件。本表是同机重新配对评价，不覆盖历史数据。</p>',
        '<div class="tabu-sync-defs"><p><strong>纯禁忌搜索</strong><br>从随机合法完整分配出发，不按容量或干扰挑优；再运行禁忌搜索。</p><p><strong>贪心＋禁忌搜索</strong><br>从容量感知贪心初解出发，运行相同禁忌搜索。</p></div>',
        f'<p class="tabu-sync-status"><strong>搜索 320 次后，纯禁忌的需求满足率为 {pct(pure["after_W"])}，贪心＋禁忌为 {pct(hybrid["after_W"])}，差 {gap:.1f} 个百分点。</strong>纯禁忌仍低于只用贪心的 {pct(greedy["after_W"])}。这组条件下，优先保留容量感知初解。</p>',
        f'<p>{html.escape(data["scope"])}，同一批场景、同一连续功率优化（SCA）。32、96、320 次是候选方案的真实评价预算。</p>',
        '<p>同条件实测：SCA 前后及每景平均总耗时</p>',
        '<div class="tabu-sync-scroll" tabindex="0" role="region" aria-label="初始化与禁忌搜索实测表，可横向滚动"><table><thead><tr><th scope="col">方法</th><th scope="col">SCA 前 SGM</th><th scope="col">SCA 前满足率</th><th scope="col">SCA 后 SGM</th><th scope="col">SCA 后满足率</th><th scope="col">总耗时 / 秒</th></tr></thead><tbody>' + metric_rows(rows) + '</tbody></table></div>',
        f'<p class="tabu-sync-timing">计时：{html.escape(data["timing"])}。下表是各环节平均时间；本表总耗时不能与历史图中只计搜索的时间直接比较。</p>',
        '<details><summary>展开初解、搜索、核验和 SCA 耗时</summary><div class="tabu-sync-scroll" tabindex="0" role="region" aria-label="分项耗时表，可横向滚动"><table><thead><tr><th scope="col">方法</th><th scope="col">初解 / 秒</th><th scope="col">搜索 / 秒</th><th scope="col">核验 / 秒</th><th scope="col">SCA / 秒</th></tr></thead><tbody>' + timing_rows(rows) + '</tbody></table></div></details>',
        '<p><a href="../../../../outputs/experiments/s2_pure_tabu_20261021/presentation_summary.json">实测摘要</a> · <a href="../../../第二阶段/实验报告/25_纯禁忌搜索与贪心加禁忌搜索配对评价.md">完整报告与日期分层结果</a></p>',
        '<p class="tabu-sync-source">原作者和历史实验来源保留；本次对照数据、同步脚本与网页对照区：codex。</p></div></section>',
        END,
    ])


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def sync_page(original, block):
    updated, count = re.subn(re.escape(START) + r".*?" + re.escape(END), lambda _: block, original, flags=re.S)
    assert count <= 1
    if count == 0:
        assert "</body>" in updated
        updated = updated.replace("</body>", block + "\n</body>", 1)
    if 'id="tabu-sync-style"' not in updated:
        assert "</head>" in updated
        updated = updated.replace("</head>", STYLE + "\n</head>", 1)
    return updated


def sync_main_page(text, edition, row):
    name = "纯禁忌搜索32次（随机初解）"
    item = {
        "name": name,
        "sgm": round(row["after_SGM"], 6),
        "w": round(row["after_W"] * 100, 6),
        "worst": round(row["after_aircraft_worst10"] * 100, 6),
        "probes": 32,
        "t": round(row["total_seconds"], 6),
    }
    if edition == "codex版":
        anchor = "const C={blue:"
        remote = {"after_W": row["after_W"], "after_SGM": row["after_SGM"],
                  "after_aircraft_worst10": row["after_aircraft_worst10"],
                  "before_W": row["before_W"], "before_SGM": row["before_SGM"],
                  "total_seconds": row["total_seconds"]}
        injection = f"D.remote.PTS32={json.dumps(remote, ensure_ascii=False)};"
        old_button = '<button data-search="S1" aria-pressed="false">贪心+禁忌搜索32次</button>'
        new_button = '<button data-search="PTS32" aria-pressed="false">纯禁忌搜索32次（随机初解）</button>' + old_button
        if 'data-search="PTS32"' not in text:
            text = replace_once(text, old_button, new_button)
        old_arms = "const arms=['L2','L3','L1','G1','S1','S2'],names=['直出T-SAC','MLP-SAC','改进式SAC','贪心','贪心+禁忌搜索32次','贪心+禁忌搜索320次'];"
        new_arms = "const arms=['L2','L3','L1','G1','PTS32','S1','S2'],names=['直出T-SAC','MLP-SAC','改进式SAC','贪心','纯禁忌搜索32次（随机初解）','贪心+禁忌搜索32次','贪心+禁忌搜索320次'];"
        if old_arms in text:
            text = replace_once(text, old_arms, new_arms)
        old_insight = ":arm==='S1'?'贪心+禁忌搜索32次相对贪心使平均满足率高约2点"
        pure_insight = (f":arm==='PTS32'?'随机合法初解搜索32次：SCA后满足率{pct(row['after_W'])}、"
                        f"最差一成{pct(row['after_aircraft_worst10'])}、SGM {row['after_SGM']:.3f}；"
                        f"含SCA总耗时{row['total_seconds']:.2f}秒。'")
        new_insight = pure_insight + old_insight
        if "arm==='PTS32'?'随机合法初解搜索32次" not in text and old_insight in text:
            text = replace_once(text, old_insight, new_insight)
        text = re.sub(f"(?:{re.escape(pure_insight)}){{2,}}", pure_insight, text)
        text = text.replace("'比贪心高 '+one((r.after_W-base.after_W)*100)+' 个百分点'", "(r.after_W>=base.after_W?'比贪心高 ':'比贪心低 ')+one(Math.abs((r.after_W-base.after_W)*100))+' 个百分点'")
        text = text.replace("'比贪心高 '+one((r.after_aircraft_worst10-base.after_aircraft_worst10)*100)+' 个百分点'", "(r.after_aircraft_worst10>=base.after_aircraft_worst10?'比贪心高 ':'比贪心低 ')+one(Math.abs((r.after_aircraft_worst10-base.after_aircraft_worst10)*100))+' 个百分点'")
        text = text.replace("color:a==='G1'?C.gray:a==='S2'?C.deep:C.blue", "color:a==='PTS32'?C.orange:a==='G1'?C.gray:a==='S2'?C.deep:C.blue")
    elif edition == "dsh版":
        anchor = "/* ---------- ranking ---------- */"
        record = {"id": "PTS32", "name": name, "kind": "base", "W": item["w"],
                  "w10": item["worst"], "SGM": item["sgm"], "probes": 32, "t": item["t"]}
        injection = ("DATA.stage2.confirm.rows.splice(1,0," + json.dumps(record, ensure_ascii=False) + ");\n"
                     "DATA.stage2.same32.rows.splice(2,0," + json.dumps({"id": "PTS32", "name": name, "W": item["w"], "kind": "base"}, ensure_ascii=False) + ");")
        text = text.replace('id="rankChart" style="height:clamp(330px,34vw,520px)"', 'id="rankChart" style="height:clamp(420px,40vw,620px)"')
        text = text.replace('id="same32Chart" style="height:clamp(250px,26vw,360px)"', 'id="same32Chart" style="height:clamp(310px,30vw,430px)"')
        text = text.replace("hbars($('#rankChart'),{w:980,h:460", "hbars($('#rankChart'),{w:980,h:550")
        text = text.replace("hbars($('#same32Chart'),{w:820,h:310", "hbars($('#same32Chart'),{w:820,h:370")
        if "['含SCA总耗时',FMT(r.meta.t)+' 秒']" not in text:
            text = text.replace("['真实评价',r.meta.probes?r.meta.probes+' 次':'0 次']", "['真实评价',r.meta.probes?r.meta.probes+' 次':'0 次'],['含SCA总耗时',FMT(r.meta.t)+' 秒']")
        pure_card = (f'<div class="card sidecard" id="pure32-main-result"><div class="k">纯禁忌搜索32次 · 随机初解</div>'
                     f'<div class="v">SCA后满足率 <b>{pct(row["after_W"])}</b>，最差一成 <b>{pct(row["after_aircraft_worst10"])}</b>，'
                     f'SGM <b>{row["after_SGM"]:.3f}</b>；含SCA总耗时 <b>{row["total_seconds"]:.2f} 秒</b>。</div></div>')
        if 'id="pure32-main-result"' not in text:
            text = replace_once(text, '<div class="card sidecard"><div class="k">搜索对贪心</div>', pure_card + '<div class="card sidecard"><div class="k">搜索对贪心</div>')
        time_tip = ",[\'含SCA总耗时\',FMT(r.meta.t)+\' 秒\']"
        text = re.sub(f"(?:{re.escape(time_tip)}){{2,}}", time_tip, text)
    else:
        anchor = "/* ---------- 第二阶段：高需求全部方法 ---------- */"
        hd = {"name": name, "sgm": item["sgm"], "w": item["w"], "cat": "search", "n": "随机初解 · 32 次评价"}
        table = {"name": name, "sgm": item["sgm"], "w": item["w"], "worst": item["worst"],
                 "probes": 32, "t": item["t"], "train": 0, "trainS": "无需训练", "color": "#e5a36e"}
        injection = ("D.hd.splice(4,0," + json.dumps(hd, ensure_ascii=False) + ");\n"
                     "D.s2d.splice(3,0," + json.dumps(table, ensure_ascii=False) + ");")
        text = text.replace("E('td', 'n', f1(r.t) + ' 秒', tr)", "E('td', 'n', (r.name.startsWith('纯禁忌搜索32次') ? r.t.toFixed(2) : f1(r.t)) + ' 秒', tr)")
    marked = r"\n?" + re.escape(MAIN_START) + r".*?" + re.escape(MAIN_END) + r"\n?"
    text = re.sub(marked, "", text, flags=re.S)
    assert text.count(anchor) == 1, edition
    return text.replace(anchor, "\n" + MAIN_START + "\n" + injection + "\n" + MAIN_END + "\n" + anchor, 1)


def main():
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    block = comparison(data)
    pure32 = next(row for row in data["rows"] if row["method"] == "PTS32")
    changed = []
    for path in PAGES:
        original = path.read_text(encoding="utf-8-sig")
        updated = sync_main_page(sync_page(original, block), path.parent.name, pure32)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed.append(str(path.relative_to(ROOT)))

    edits = {
        "docs/Claude维护/导师汇报/codex版/06_算法交互演示.html": [
            ("'禁忌搜索（教学）','最近布局记忆'", "'禁忌搜索规则（教学）','最近布局记忆'"),
        ],
        "docs/Claude维护/导师汇报/codex版/09_第二阶段算法逐步演示.html": [
            ("tabu:'禁忌搜索 · 完整8轮演示'", "tabu:'禁忌搜索规则 · 8轮教学演示'"),
        ],
        "docs/Claude维护/导师汇报/dsh版/19_三阶段算法实验台.html": [
            ('"id":"tabu","name":"禁忌搜索"', '"id":"tabu","name":"贪心＋禁忌搜索（教学）"'),
            ('"recipe":["从贪心解出发"', '"recipe":["从容量感知贪心解出发"'),
        ],
    }
    for name, changes in edits.items():
        path = ROOT / name
        original = path.read_text(encoding="utf-8-sig")
        updated = original
        for old, new in changes:
            if old not in updated and new in updated:
                continue
            updated = replace_once(updated, old, new)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed.append(name)
    print(json.dumps(changed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
