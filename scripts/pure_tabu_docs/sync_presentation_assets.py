"""Update the maintained figure links and public data exports."""

from pathlib import Path
import re
import shutil

from sync_markdown import CHART_NOTE, OUT, ROOT, rel


FIGURES = {
    "outputs/advisor_brief_20261018/dsh版/figures/G1_SGM两种协议.png": "dsh_G1_SGM两种协议.png",
    "outputs/advisor_brief_20261018/dsh版/figures/G2_SGM第三阶段前后.png": "dsh_G2_SGM第三阶段前后.png",
    "outputs/advisor_brief_20261018/dsh版/figures/G5_评价次数与满足率.png": "dsh_G5_评价次数与满足率.png",
    "outputs/advisor_brief_20261018/cc版/figures_v2/v2_11_高需求全方法.png": "cc_v2_11_高需求全方法.png",
    "outputs/advisor_brief_20261018/cc版/figures_v2/v2_13_搜索效率.png": "cc_v2_13_搜索效率.png",
}
CAPTION = "图中分别列出纯禁忌与贪心＋禁忌；前者使用随机合法初解，后者使用容量感知贪心初解。"


def backup(path):
    target = OUT / "document_backups" / path.relative_to(ROOT)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)


def main():
    changed = []
    for path in (ROOT / "docs").rglob("*.md"):
        original = path.read_text(encoding="utf-8-sig")
        text = original
        for old, new in FIGURES.items():
            target = OUT / "presentation_figures" / new
            text = text.replace(rel(ROOT / old, path), rel(target, path))
            pattern = rf"(!\[[^\n]*\]\([^\n)]*/{re.escape(new)}\))\n\n{re.escape(CHART_NOTE)}"
            text = re.sub(pattern, lambda m: m[1] + "\n\n" + CAPTION, text)
        if text != original:
            backup(path)
            path.write_text(text, encoding="utf-8")
            changed.append(str(path.relative_to(ROOT)))
    public = ROOT / "docs/Claude维护/导师汇报/codex版/数据与图"
    for name in ("02_第二阶段服务效果.png", "03_学习与搜索耗时.png", "第二阶段_确认88景.csv"):
        path = public / name
        source = OUT / "presentation_figures" / ("codex_" + name)
        if path.read_bytes() != source.read_bytes():
            backup(path)
            shutil.copyfile(source, path)
            changed.append(str(path.relative_to(ROOT)))
    print("\n".join(changed) if changed else "unchanged")


if __name__ == "__main__":
    main()
