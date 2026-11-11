"""W14 三张“报告现成图”裁剪并改中文标签（原始逐回合、逐场景记录不在本机，只有报告里的图）。

做法：绘图区一个像素都不动、不缩放，只裁掉带实验代号的标题条、把绘图区以外带代号的标签换成中文，
叠在空白处的图例用白底盖掉。
原图（D:\\paper\\outputs\\experiments\\first_stage_s1_03_r1_20261005\\analysis 下，修复缓存缺陷后的重训）：
  1. action_curve_*/中文训练非STOP动作接受率.png 的上半幅（接受率曲线）+ 下半幅的横轴刻度
        -> figures/W14_修复后动作接受率曲线.png
     改了：裁掉标题（带 S1-03-R1 和 seed），盖掉图内图例（带 P1 代号）并改成图上方的中文标题
  2. final_report_*/中文共同考核覆盖均衡分布.png 的右上幅（主体CV逐景分布箱线图）
        -> figures/W14_主体负载CV的逐场景分布.png
     改了：横轴方法名（原来带 R0~R3、P1 代号）换成中文，补了纵轴标签
  3. final_report_*/中文训练验证与考核曲线.png 的上面两幅（训练回报、更新熵）
        -> figures/W14_修复后训练回报与策略熵.png
     改了：裁掉标题（带 seed），重写中文标题
原图都是 webp 内容却用 .png 后缀，这里用 Pillow 读入后重新存成真 PNG。
"""
import glob
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.normpath(os.path.join(HERE, "..", "figures"))
R1 = r"D:\paper\outputs\experiments\first_stage_s1_03_r1_20261005\analysis"
FONT = r"C:\Windows\Fonts\msyh.ttc"


def font(size):
    return ImageFont.truetype(FONT, size)


def pick(pattern):
    return glob.glob(os.path.join(R1, "**", pattern), recursive=True)[0]


def save(im, name):
    out = os.path.join(FIG, name)
    im.convert("RGB").save(out, format="PNG", optimize=True)
    print(name, im.size, f"{os.path.getsize(out) / 1e6:.2f} MB")


def text_rot(im, center, s, size, angle=90, fill="black", anchor_end=None):
    f = font(size)
    l, t, r, b = f.getbbox(s)
    tmp = Image.new("RGBA", (r - l + 10, b - t + 10), (255, 255, 255, 0))
    ImageDraw.Draw(tmp).text((5 - l, 5 - t), s, font=f, fill=fill)
    tmp = tmp.rotate(angle, expand=True, resample=Image.BICUBIC)
    if anchor_end is not None:       # 文字末端（右上角）对准 anchor_end
        pos = (int(anchor_end[0] - tmp.width), int(anchor_end[1]))
    else:
        pos = (int(center[0] - tmp.width / 2), int(center[1] - tmp.height / 2))
    im.paste(tmp, pos, tmp)


def fig_accept():
    im = Image.open(pick("中文训练非STOP动作接受率.png")).convert("RGB")
    d0 = ImageDraw.Draw(im)
    d0.rectangle([1176, 66, 1732, 114], fill="white")      # 图内图例（带 P1 代号）
    top = im.crop((0, 55, im.width, 530))
    strip = im.crop((0, 1022, im.width, 1100))              # 下幅的横轴刻度和标签（与上幅横轴一致）
    band = 60
    canvas = Image.new("RGB", (im.width, band + top.height + strip.height), "white")
    canvas.paste(top, (0, band))
    canvas.paste(strip, (0, band + top.height))
    d = ImageDraw.Draw(canvas)
    d.text((im.width // 2 + 40, band // 2 + 6), "修复后训练过程中，每16384步里被接受的动作占比（不含停止动作）", font=font(31), fill="black", anchor="mm")
    save(canvas, "W14_修复后动作接受率曲线.png")


def fig_box():
    src = Image.open(pick("中文共同考核覆盖均衡分布.png")).convert("RGB")
    x0 = 1120
    crop = src.crop((x0, 0, 2240, 705))
    pad = 44
    im = Image.new("RGB", (crop.width + pad, crop.height), "white")
    im.paste(crop, (pad, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([pad, 609, im.width, 705], fill="white")     # 盖掉带代号的刻度标签（刻度线保留）
    names = ["共同初始布局", "规则重构", "随机重构", "搜索重构", "PPO重构"]
    ticks = [1286, 1493, 1699, 1906, 2113]
    for x, s in zip(ticks, names):
        text_rot(im, None, s, 25, angle=25, anchor_end=(x - x0 + pad + 22, 614))
    text_rot(im, (22, 335), "主体负载CV", 25, angle=90)
    save(im, "W14_主体负载CV的逐场景分布.png")


def fig_train():
    src = Image.open(pick("中文训练验证与考核曲线.png")).convert("RGB")
    top = src.crop((0, 54, src.width, 596))
    band = 60
    canvas = Image.new("RGB", (src.width, band + top.height), "white")
    canvas.paste(top, (0, band))
    d = ImageDraw.Draw(canvas)
    d.text((540, band // 2 + 6), "训练回报（每个回合一条淡线）", font=font(32), fill="black", anchor="mm")
    d.text((1580, band // 2 + 6), "PPO策略更新熵", font=font(32), fill="black", anchor="mm")
    save(canvas, "W14_修复后训练回报与策略熵.png")


if __name__ == "__main__":
    fig_accept()
    fig_box()
    fig_train()
