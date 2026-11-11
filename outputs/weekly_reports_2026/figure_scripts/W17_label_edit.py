"""W17 三张“报告现成图”的中文改标签版本（原始逐点数据不在本机，只有报告里的图）。

做法：绘图区一个像素都不动、不缩放，只把绘图区以外带代号或英文的标题、轴标签、图例用白底盖掉，
再写同字号的中文。
  1. 第一轮实验（可变波束数量训练与功率细调前后对照）报告里的
     variable_beams_20261004/analysis/publication_figures_*/terminal_si_by_n_load.png
     -> figures/W17_第一轮分负载分阶段终态满足率.png
  3. 同目录 paired_v1_minus_v0_si_distribution_by_n_load.png（混合训练减固定100束训练的需求满足率差，20个母场景逐点）
     -> figures/W17_第一轮混合训练逐母场景差值.png
  2. 第二轮实验里的 variable_beams_s2p18_20261006/analysis/中文_三臂训练曲线.png 的左半幅（训练内验证需求满足率）
     -> figures/W17_第二轮训练内验证满足率.png
"""
import glob
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.normpath(os.path.join(HERE, "..", "figures"))
FONT = r"C:\Windows\Fonts\msyh.ttc"
SRC1 = glob.glob(r"D:\paper\outputs\experiments\variable_beams_20261004\analysis\publication_figures_*\terminal_si_by_n_load.png")[0]
SRC3 = glob.glob(r"D:\paper\outputs\experiments\variable_beams_20261004\analysis\publication_figures_*\paired_v1_minus_v0_si_distribution_by_n_load.png")[0]
SRC2 = r"D:\paper\outputs\experiments\variable_beams_s2p18_20261006\analysis\中文_三臂训练曲线.png"


def font(size):
    return ImageFont.truetype(FONT, size)


def cover(d, box):
    d.rectangle(box, fill="white")


def text_c(d, xy, s, size, fill="black"):
    d.text(xy, s, font=font(size), fill=fill, anchor="mm")


def text_rot(im, center, s, size, fill="black"):
    f = font(size)
    l, t, r, b = f.getbbox(s)
    tmp = Image.new("RGBA", (r - l + 8, b - t + 8), (255, 255, 255, 0))
    ImageDraw.Draw(tmp).text((4 - l, 4 - t), s, font=f, fill=fill)
    tmp = tmp.rotate(90, expand=True)
    im.paste(tmp, (int(center[0] - tmp.width / 2), int(center[1] - tmp.height / 2)), tmp)


def legend_item(d, x, y, color, label, size):
    d.line([(x, y), (x + 44, y)], fill=color, width=4)
    d.ellipse([x + 22 - 6, y - 6, x + 22 + 6, y + 6], fill=color)
    d.text((x + 58, y), label, font=font(size), fill="black", anchor="lm")


def fig1():
    im = Image.open(SRC1).convert("RGB")
    d = ImageDraw.Draw(im)
    # 总标题
    cover(d, (0, 0, im.width, 125))
    text_c(d, (1224, 62), "需求满足率：每个束数、每档负载各20个母场景，对全部目标波束取平均", 38)
    # 子图标题
    for cx, cy, s in [(704, 203, "负载×0.75 · 直接分配"), (1892, 203, "负载×0.75 · 功率细调后"),
                      (704, 766, "负载×1.25 · 直接分配"), (1892, 766, "负载×1.25 · 功率细调后")]:
        cover(d, (cx - 330, cy - 24, cx + 330, cy + 19))
        text_c(d, (cx, cy), s, 32)
    # y 轴标签
    for cx, cy in [(105, 437), (105, 1000), (1295, 437), (1295, 1000)]:
        cover(d, (cx - 22, cy - 150, cx + 20, cy + 150))
        text_rot(im, (cx - 3, cy), "需求满足率", 28)
    # x 轴标签
    for cx in (703, 1890):
        cover(d, (cx - 230, 1250, cx + 230, 1292))
        text_c(d, (cx, 1271), "目标波束数 N", 31)
    # 图例
    cover(d, (790, 1372, 1660, 1495))
    d.rectangle([810, 1386, 1638, 1479], outline=(200, 200, 200), width=2)
    c = {"v0": (56, 108, 176), "v1": (224, 123, 40), "h0": (117, 80, 162), "b0": (61, 150, 112), "b1": (173, 88, 106)}
    legend_item(d, 830, 1403, c["v0"], "固定100束训练的模型", 24)
    legend_item(d, 830, 1432, c["v1"], "混合波束数训练的模型", 24)
    legend_item(d, 830, 1461, c["h0"], "第三阶段早期的冻结模型", 24)
    legend_item(d, 1260, 1403, c["b0"], "固定规则", 24)
    legend_item(d, 1260, 1432, c["b1"], "贪心规则", 24)
    im = im.crop((20, 0, 2420, im.height))      # 只裁掉两侧空白，宽度压到 2400 px 以内，绘图区不缩放
    out = os.path.join(FIG, "W17_第一轮分负载分阶段终态满足率.png")
    im.save(out, format="PNG", optimize=True)
    print(out, im.size, os.path.getsize(out) / 1e6)


def fig2():
    im = Image.open(SRC2).convert("RGB")
    im = im.crop((0, 40, 1175, 792))          # 只留左半幅
    pad = 90
    canvas = Image.new("RGB", (im.width, im.height + pad), "white")
    canvas.paste(im, (0, 0))
    d = ImageDraw.Draw(canvas)
    # y 轴标签（原来是“验证需求满足率 SI”）
    cx, cy = 28, 360
    cover(d, (4, cy - 200, 47, cy + 200))
    text_rot(canvas, (cx, cy), "验证需求满足率", 26)
    # 图内图例：白底盖掉（图例框下方没有数据线）
    cover(d, (136, 80, 232, 168))
    # 底部图例
    cx0, y = 190, im.height + 40
    legend_item(d, cx0, y, (31, 119, 180), "改奖励", 28)
    legend_item(d, cx0 + 250, y, (255, 127, 14), "改折扣和整形", 28)
    legend_item(d, cx0 + 620, y, (44, 160, 44), "改网络结构", 28)
    out = os.path.join(FIG, "W17_第二轮训练内验证满足率.png")
    canvas.save(out, format="PNG", optimize=True)
    print(out, canvas.size, os.path.getsize(out) / 1e6)


def fig3():
    im = Image.open(SRC3).convert("RGB")
    d = ImageDraw.Draw(im)
    cover(d, (0, 0, im.width, 120))
    text_c(d, (1224, 48), "混合波束数训练减固定100束训练的需求满足率差", 38)
    text_c(d, (1224, 92), "每个束数、每档负载的20个母场景全部保留，同一母场景会在不同束数和负载里重复出现", 28)
    for cx, cy, s in [(727, 218, "负载×0.75 · 直接分配"), (1892, 218, "负载×0.75 · 功率细调后"),
                      (727, 798, "负载×1.25 · 直接分配"), (1892, 798, "负载×1.25 · 功率细调后")]:
        cover(d, (cx - 330, cy - 24, cx + 330, cy + 19))
        text_c(d, (cx, cy), s, 32)
    for cx, cy in [(108, 460), (108, 1040), (1352, 460), (1352, 1040)]:
        cover(d, (cx - 22, cy - 190, cx + 20, cy + 190))
        text_rot(im, (cx - 2, cy), "需求满足率差", 28)
    for cx in (727, 1892):
        cover(d, (cx - 230, 1295, cx + 230, 1336))
        text_c(d, (cx, 1315), "目标波束数 N", 31)
    cover(d, (850, 1415, 1600, 1480))
    d.rectangle([868, 1429, 1580, 1465], outline=(200, 200, 200), width=2)
    d.ellipse([890, 1441, 904, 1455], fill=(120, 150, 200))
    d.text((932, 1447), "每个点是一个母场景", font=font(24), fill="black", anchor="lm")
    d.line([(1215, 1447), (1255, 1447)], fill=(224, 123, 40), width=4)
    d.polygon([(1235, 1438), (1244, 1447), (1235, 1456), (1226, 1447)], fill=(224, 123, 40))
    d.text((1272, 1447), "20个母场景的平均", font=font(24), fill="black", anchor="lm")
    im = im.crop((20, 0, 2420, im.height))      # 只裁掉两侧空白，宽度压到 2400 px 以内，绘图区不缩放
    out = os.path.join(FIG, "W17_第一轮混合训练逐母场景差值.png")
    im.save(out, format="PNG", optimize=True)
    print(out, im.size, os.path.getsize(out) / 1e6)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()




