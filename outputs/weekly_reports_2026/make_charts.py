"""按周报表格里的数字画简单条形图，写回各周 JSON 的 figure 字段。
用法：python3 make_charts.py（用自带Python，只用Pillow）
"""
import json, os, re
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = f"{ROOT}/outputs/weekly_reports_2026/data"
OUT = f"{ROOT}/outputs/weekly_reports_2026/figures"
os.makedirs(OUT, exist_ok=True)
FONT = "/System/Library/Fonts/STHeiti Medium.ttc"

# 周 -> (取表里哪几列, 图题)
CFG = {
    1: ([1], "图1 五种算法和贪心在旧权重重评里的总满意度"),
    2: ([1, 2], "图1 四种分配策略在20个随机场景上的需求满足率和供需匹配度"),
    3: ([1, 2], "图1 欧洲航班数据上首轮训练的模型和基线对比"),
    4: ([1, 2], "图1 训练提速前后每500步的耗时（秒）"),
    5: ([1, 2], "图1 七个模型的需求满足率和供需匹配程度"),
    6: ([1, 2], "图1 五档状态维数的需求满足率和供需匹配程度"),
    7: ([1, 2], "图1 联合动作四组和贪心的需求满足率和供需匹配程度"),
    8: ([1, 2], "图1 九组模型在100个独立场景上的需求满足率和供需匹配程度"),
    9: ([1, 2], "图1 功率细调前后的需求满足率和供需匹配程度"),
    10: ([1, 4], "图1 七种功率优化路线的需求满足率和单次耗时"),
    11: ([4], "图1 三种细调设置的单次总耗时（秒）"),
    12: ([2, 3], "图1 七组覆盖规划的负载不均程度（越小越好）和达标场景数"),
    13: ([1, 2], "图1 覆盖和负载均衡几步改进后的结果"),
    14: ([1, 3], "图1 几种区域重构方法的主体负载CV和每景耗时"),
    15: ([1, 2], "图1 六条覆盖路线的需求CV和逐机承载"),
    16: ([1, 3], "图1 几种方法的负载CV和每景耗时"),
    18: ([1, 2], "图1 四种方法细调前后的需求满足率"),
    19: ([1, 2], "图1 几种负载重分组方法的最重波束负载和负载CV"),
    20: ([1, 2], "图1 修复器改进前后达标的场景数（门槛18个）"),
    21: ([1, 2], "图1 四种重构方式的峰值负载和负载CV"),
    22: ([1, 2], "图1 六种方法的需求满足率和供需匹配程度"),
    23: ([1, 3], "图1 不同功率档位下的SGM和总功率"),
}


def num(s):
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None


def draw(w, cols, rows, header):
    f12, f11 = ImageFont.truetype(FONT, 22), ImageFont.truetype(FONT, 20)
    names = [r[0] for r in rows]
    W, rh = 1500, 46
    left = 330 if len(cols) == 2 else 330
    n = len(cols)
    H = 70 + rh * len(rows) + 20
    img = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(img)
    pw = (W - left - 30) // n
    for i, nm in enumerate(names):
        dr.text((left - 12, 70 + i * rh + rh // 2), nm, font=f11, fill="black", anchor="rm")
    for k, c in enumerate(cols):
        x0 = left + k * pw
        vals = [num(r[c]) or 0 for r in rows]
        top = max(vals) or 1
        bw = pw - 160
        dr.text((x0 + bw // 2, 30), header[c], font=f12, fill="black", anchor="mm")
        dr.line([(x0, 60), (x0, H - 15)], fill="#888888", width=2)
        for i, (v, r) in enumerate(zip(vals, rows)):
            y = 70 + i * rh
            L = int(bw * v / top)
            dr.rectangle([x0 + 1, y + 7, x0 + 1 + L, y + rh - 7], fill="#4C78A8")
            dr.text((x0 + L + 10, y + rh // 2), r[c].replace(" ", ""), font=f11, fill="black", anchor="lm")
    p = f"{OUT}/W{w:02d}.png"
    img.save(p)
    return p


for w, (cols, cap) in CFG.items():
    f = f"{DATA}/W{w:02d}.json"
    d = json.load(open(f, encoding="utf-8"))
    t = d["table"]
    p = draw(w, cols, t["rows"], t["header"])
    d["figure"] = {"path": os.path.relpath(p, ROOT), "caption": cap}
    json.dump(d, open(f, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(w, p)

# 月报：挑本月里两周的图
MONTH = {5: [3], 6: [8]}
for m, ws in MONTH.items():
    f = f"{DATA}/M{m:02d}.json"
    d = json.load(open(f, encoding="utf-8"))
    figs = []
    for i, w in enumerate(ws, 1):
        cap = CFG[w][1].replace("图1", f"图{i}")
        figs.append({"path": os.path.relpath(f"{OUT}/W{w:02d}.png", ROOT), "caption": cap})
    d["figures"] = figs
    json.dump(d, open(f, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("M", m)
