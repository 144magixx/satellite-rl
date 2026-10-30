"""B 账号与本机之间传文件（base 环境运行，DQN 环境没有 paramiko）。
  python scripts/s2lab/xfer.py get <远端路径> <本机路径>
  python scripts/s2lab/xfer.py put <本机路径> <远端路径>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import remote2  # noqa: E402

cmd, src, dst = sys.argv[1:4]
with remote2.connect(remote2.first_port('B')) as c, c.open_sftp() as sftp:
    (sftp.get if cmd == 'get' else sftp.put)(src, dst)
print(cmd, src, '->', dst)
