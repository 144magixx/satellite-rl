"""来源：codex。只导出codex版PDF与PPT核图。"""
from pathlib import Path
import runpy

root = Path(__file__).resolve().parents[4]
runpy.run_path(str(root / "outputs/advisor_brief_20261018/codex版/revision_20261019/render_office.py"), run_name="__main__")
