"""在 B 账号某张卡上后台执行一条命令（在仓库目录、带 PYTHONPATH），日志写到 ~/s2lab/logs/<name>.log。
  python scripts/s2lab/rbg.py --port 20878 --name eval_c2_rules --cmd "<python 参数...>"
命令里的 PY 会替换成 B 账号的 Python。
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('S2LAB_ROUND', '4')
sys.path.insert(0, str(Path(__file__).parent))
import remote2  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--port', type=int, required=True)
ap.add_argument('--name', required=True)
ap.add_argument('--cmd', required=True)
a = ap.parse_args()
p = remote2.paths('B')
with remote2.connect(a.port) as c:
    pid = remote2.background(c, p, a.cmd.replace('PY ', p['py'] + ' ', 1), f"{p['root']}/logs/{a.name}.log")
print('started', pid)
