"""Republish historical plots with explicit hybrid names, preserving raw evidence."""
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/experiments/s2_pure_tabu_20261021/historical_figures"


def s2fix():
    source = ROOT / "outputs/experiments/s2_fix_20261018/report/analyze.py"
    code = source.read_text(encoding="utf-8")
    code = code.replace("禁忌搜索", "贪心＋禁忌搜索")
    code = code.replace('LABEL[m].split(" ")[0]', 'SHORT_LABEL[m]')
    code = code.replace("figsize=(6.8, 4.4)", "figsize=(8.2, 4.7)")
    ns = {"__file__": str(source), "__name__": "historical_s2fix_display"}
    exec(compile(code, str(source), "exec"), ns)
    folder = OUT / "s2fix"
    folder.mkdir(parents=True, exist_ok=True)
    ns["FIG"] = folder
    ns["SHORT_LABEL"] = {"G1": "贪心", "R": "随机\n搜索32", "S1": "贪心＋\n禁忌32",
        "S2": "贪心＋\n禁忌320", "S96": "贪心＋\n禁忌96", "L1": "I0\n改进SAC", "L2": "直出\nT-SAC", "L3": "直出\nMLP-SAC"}
    frame = ns["load"]()
    _, scenes = ns["scene_means"](frame)
    summary = scenes.groupby(["set", "method"])[ns["METRICS"]].mean().reset_index()
    present = [m for m in ns["METHODS"] if m in set(frame.method)]
    ns["figures"](summary, scenes, present)


def litcmp():
    source = ROOT / "outputs/experiments/litcmp_20261018/report/make_figs.py"
    code = source.read_text(encoding="utf-8")
    code = code.replace("S1 禁忌 32次", "S1 贪心＋禁忌32次").replace("S2 禁忌 320次", "S2 贪心＋禁忌320次")
    code = code.replace("FLOW-AO v2＋S2＋SCA", "FLOW-AO v2＋贪心＋禁忌320次＋SCA")
    code = code.replace("ncol=4, fontsize=8.5, bbox_to_anchor=(.5, -.02)", "ncol=2, fontsize=8.5, bbox_to_anchor=(.5, .005)")
    code = code.replace("rect=(0, .08, 1, 1)", "rect=(0, .16, 1, 1)")
    ns = {"__file__": str(source), "__name__": "historical_litcmp_display"}
    exec(compile(code, str(source), "exec"), ns)
    folder = OUT / "litcmp"
    folder.mkdir(parents=True, exist_ok=True)
    ns["FIG"] = folder
    ns["F3"]()
    ns["F5"]()


def update_links():
    for path in (ROOT / "docs").rglob("*.md"):
        text = path.read_text(encoding="utf-8-sig")
        original = text
        for experiment, folder in [("s2_fix_20261018", "s2fix"), ("litcmp_20261018", "litcmp")]:
            pattern = rf"outputs/experiments/{experiment}/report/figures/([^\n)]+\.png)"
            def replace(match):
                name = match.group(1)
                if (OUT / folder / name).exists():
                    return f"outputs/experiments/s2_pure_tabu_20261021/historical_figures/{folder}/{name}"
                return match.group(0)
            text = re.sub(pattern, replace, text)
        if text != original:
            backup = OUT.parent / "document_backups" / path.relative_to(ROOT)
            if not backup.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                backup.write_bytes(path.read_bytes())
            path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    s2fix()
    litcmp()
    update_links()
    print("Republished 8 historical figures; updated Markdown links; original evidence unchanged.")
