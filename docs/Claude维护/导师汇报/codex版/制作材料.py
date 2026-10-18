"""来源：codex。按共用证据重建codex版，不写其它版本。"""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[4]
work = root / "outputs/advisor_brief_20261018/codex版"
for script in [work / "revision_20261019/collect_evidence.py", work / "revision_20261019/content.py",
               work / "revision_20261019/build_documents.py", work / "revision_20261019/revise_h5.py", work / "web/build.py"]:
    subprocess.run([sys.executable, "-X", "utf8", str(script)], cwd=root, check=True)
