"""在远端某端口执行一个 shell 脚本文件（避开 PowerShell 引号问题）。
  python scripts/s2lab/rsh.py --port 20802 --file cmd.sh [--timeout 600]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import remote2  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--port', type=int, required=True)
ap.add_argument('--file', required=True)
ap.add_argument('--timeout', type=int, default=600)
a = ap.parse_args()
with remote2.connect(a.port) as c:
    print(remote2.run(c, Path(a.file).read_text(encoding='utf-8'), timeout=a.timeout, check=False))
