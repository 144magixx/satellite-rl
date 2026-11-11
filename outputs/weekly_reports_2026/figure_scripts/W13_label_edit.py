"""W13 三张“报告现成图”改中文标签版本（原始逐点数据不在本机，只有报告里的图）。

做法：绘图区一个像素都不动、不缩放，只把绘图区以外带代号、英文的标题、轴标签，
以及叠在空白处的图例用白底盖掉，再写中文。
原图（D:\\paper\\outputs\\experiments\\first_stage_s1_02_20261004\\analysis 下）：
  1. A_report/figures/all_and_subject_cv.png          -> figures/W13_逐场景负载CV.png
     改了：图例（原来带 A0~A3、C1 代号）、横轴标签
  2. B_delivery/figures/paired_menu_differences.png   -> figures/W13_PPO与搜索的逐场景差值.png
     改了：总标题（带 A2）、图例（带 X0/X1/X4）
  3. B_report/figures/training_X0.png（上半幅，训练回报散点）
     + B_report/figures/validation_X0.png（下半幅，验证场景主体CV）
                                                     -> figures/W13_四向PPO训练与验证曲线.png
     改了：两幅图的标题（带 X0、seed）、y 轴标签（原来是英文 subject_cv）
这些原图都是 webp 内容却用 .png 后缀，这里用 Pillow 读入后重新存成真 PNG。
"""
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.normpath(os.path.join(HERE, "..", "figures"))
EXP = r"D:\paper\outputs\experiments\first_stage_s1_02_20261004\analysis"
FONT = r"C:\Windows\Fonts\msyh.ttc"
BLUE, ORANGE, GREEN, RED = (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40)


def font(size):
    return ImageFont.truetype(FONT, size)


def cover(d, box):
    d.rectangle(box, fill="white")


def text_rot(im, center, s, size, fill="black"):
    f = font(size)
    l, t, r, b = f.getbbox(s)
    tmp = Image.new("RGBA", (r - l + 8, b - t + 8), (255, 255, 255, 0))
    ImageDraw.Draw(tmp).text((4 - l, 4 - t), s, font=f, fill=fill)
    tmp = tmp.rotate(90, expand=True)
    im.paste(tmp, (int(center[0] - tmp.width / 2), int(center[1] - tmp.height / 2)), tmp)


def save(im, name):
    out = os.path.join(FIG, name)
    im.convert("RGB").save(out, format="PNG", optimize=True)
    print(name, im.size, f"{os.path.getsize(out) / 1e6:.2f} MB")


def fig_cv():
    im = Image.open(os.path.join(EXP, "A_report", "figures", "all_and_subject_cv.png")).convert("RGB")
    d = ImageDraw.Draw(im)
    # 图例：两列，圆点 + 文字
    cover(d, (98, 28, 604, 94))
    d.rectangle([101, 31, 600, 91], outline=(205, 205, 205), width=2)
    items = [(122, 47, BLUE, "聚类基础布局"), (122, 74, ORANGE, "＋覆盖修补"),
             (396, 47, GREEN, "＋负载感知归属"), (396, 74, RED, "＋四向调位")]
    for x, y, c, s in items:
        d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=c)
        d.text((x + 24, y), s, font=font(21), fill="black", anchor="lm")
    # 横轴标签
    cover(d, (560, 1048, 1100, 1090))
    d.text((830, 1066), "100个场景的序号（每点一个场景，虚线为主体CV目标0.2）", font=font(22), fill="black", anchor="mm")
    save(im, "W13_逐场景负载CV.png")


def fig_pair():
    im = Image.open(os.path.join(EXP, "B_delivery", "figures", "paired_menu_differences.png")).convert("RGB")
    d = ImageDraw.Draw(im)
    cover(d, (0, 0, im.width, 46))
    d.text((960, 22), "同样动作下 PPO 减 搜索：每个场景的主体CV差和耗时差", font=font(32), fill="black", anchor="mm")
    cols = [im.getpixel(p) for p in [(147, 78), (393, 79), (579, 79)]]
    cover(d, (118, 58, 735, 100))
    d.rectangle([119, 59, 733, 98], outline=(205, 205, 205), width=2)
    for x, c, s in zip([147, 363, 579], cols, ["四向动作", "八方向动作", "相邻双束动作"]):
        d.ellipse([x - 7, 78 - 7, x + 7, 78 + 7], fill=c)
        d.text((x + 20, 78), s, font=font(23), fill="black", anchor="lm")
    save(im, "W13_PPO与搜索的逐场景差值.png")


def stack_train_valid():
    a = Image.open(os.path.join(EXP, "B_report", "figures", "training_X0.png")).convert("RGB")
    b = Image.open(os.path.join(EXP, "B_report", "figures", "validation_X0.png")).convert("RGB")
    ta = a.crop((0, 100, a.width, 570))            # 训练回报散点（上半幅）
    tb = b.crop((0, 470, b.width, 962))           # 验证场景主体CV（下半幅）
    W = max(ta.width, tb.width)
    band = 62
    H = band + ta.height + band + tb.height
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    canvas.paste(ta, (0, band))
    canvas.paste(tb, (0, band + ta.height + band))
    d.text((W // 2 + 40, band // 2 + 4), "训练过程：每个完整回合的累计奖励（每点一个回合，未做平滑）", font=font(31), fill="black", anchor="mm")
    y2 = band + ta.height
    d.text((W // 2 + 40, y2 + band // 2 + 4), "固定20个验证场景的主体负载CV（每点一次验证）", font=font(31), fill="black", anchor="mm")
    # 下幅 y 轴标签：原来是英文 subject_cv
    off = band + ta.height + band
    cover(d, (0, off + 40, 44, off + 440))
    text_rot(canvas, (22, off + 240), "主体负载CV", 26)
    # 上幅 x 轴没有标签，补一个
    cover(d, (500, band + ta.height - 6, 1100, band + ta.height))
    save(canvas, "W13_四向PPO训练与验证曲线.png")


if __name__ == "__main__":
    fig_cv()
    fig_pair()
    stack_train_valid()
