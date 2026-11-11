"""按周报里已经做好的图，给月报挑图并写进 data/M??.json 的 figures。
选法写在下面的 SELECT：月份 -> [(周号, 该周 figures 里的第几张)]，图题沿用周报的图题，只重新编号。
用法：python outputs/weekly_reports_2026/figure_scripts/_set_month_figures.py 5 9 10   （不带参数跑全部已写好的月份）
"""
import json
import re
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"

SELECT = {
    5: [(1, 1), (1, 5), (2, 7), (2, 2), (3, 8), (3, 3), (4, 4), (4, 3)],
    6: [(5, 1), (5, 4), (6, 1), (6, 3), (7, 1), (7, 4), (8, 1), (8, 3), (8, 4)],
    7: [(9, 1), (10, 1), (10, 2), (11, 1), (12, 1), (12, 5), (13, 1), (13, 6)],
    8: [(14, 1), (14, 5), (14, 7), (15, 1), (15, 2), (16, 1), (16, 3), (17, 5), (17, 4), (17, 6)],
    9: [(18, 1), (18, 2), (19, 3), (20, 4), (21, 4), (21, 6), (22, 1), (22, 2), (22, 5)],
    10: [(23, 1), (23, 5), (23, 7), (24, 4), (24, 7), (24, 8), (25, 1), (25, 5), (26, 1), (26, 2), (26, 8)],
}


def build(month):
    figs = []
    for k, (w, idx) in enumerate(SELECT[month], 1):
        src = json.loads((DATA / f"W{w:02d}.json").read_text(encoding="utf-8"))["figures"][idx - 1]
        fig = dict(src)
        fig["caption"] = re.sub(r"^图\d+", f"图{k}", src["caption"])
        figs.append(fig)
    return figs


if __name__ == "__main__":
    months = [int(a) for a in sys.argv[1:]] or sorted(SELECT)
    for m in months:
        p = DATA / f"M{m:02d}.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        d["figures"] = build(m)
        p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"M{m:02d}: {len(d['figures'])} 张图")
